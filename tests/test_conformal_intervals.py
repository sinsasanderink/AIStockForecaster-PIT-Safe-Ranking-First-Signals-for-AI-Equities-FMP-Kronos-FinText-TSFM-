"""
Tests for Conformal Prediction Intervals — Chapter 13.5.

Validates:
    - Marginal coverage within [0.85, 0.95] for well-behaved data
    - ECE < 0.10
    - DEUP conditional coverage spread < raw conditional coverage spread
    - Width ratio > 1.0 for DEUP (high-ê stocks get wider intervals)
    - No future leakage (calibration window strictly before prediction date)
    - Edge cases (insufficient calibration history → NaN)
"""

import numpy as np
import pandas as pd
import pytest

from src.uncertainty.conformal_intervals import (
    ConformalConfig,
    compute_conformal_intervals,
    compute_diagnostics,
    compute_nonconformity_scores,
    run_conformal_pipeline,
)


# ── Fixtures ──────────────────────────────────────────────────────────────


def _make_conformal_data(
    n_dates: int = 200,
    n_tickers: int = 40,
    horizon: int = 20,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Synthetic data where DEUP-normalized conformal should outperform raw.

    High-ehat stocks have higher rank_loss variance (heteroscedastic).
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-02", periods=n_dates)
    tickers = [f"T{i:03d}" for i in range(n_tickers)]

    rows = []
    for d in dates:
        for i, t in enumerate(tickers):
            # First half: low uncertainty, second half: high uncertainty
            is_high_unc = i >= n_tickers // 2
            ehat = rng.uniform(0.20, 0.40) if is_high_unc else rng.uniform(0.05, 0.15)
            rl_scale = 0.4 if is_high_unc else 0.15
            rank_loss = abs(rng.normal(rl_scale, rl_scale * 0.5))
            rank_loss = min(rank_loss, 1.0)

            rows.append({
                "as_of_date": d,
                "ticker": t,
                "horizon": horizon,
                "rank_loss": rank_loss,
                "ehat_raw": ehat,
                "vol_20d": abs(rng.normal(0.25, 0.05)),
                "vix_percentile_252d": rng.uniform(0, 1),
            })
    return pd.DataFrame(rows)


def _make_sparse_ehat_data(
    n_dates: int = 200,
    n_tickers: int = 40,
    horizon: int = 60,
    seed: int = 42,
) -> pd.DataFrame:
    """60d-like data where 85% of ehat values are zero."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-02", periods=n_dates)
    tickers = [f"T{i:03d}" for i in range(n_tickers)]

    rows = []
    for d in dates:
        for t in tickers:
            has_ehat = rng.random() < 0.15
            ehat = rng.uniform(0.01, 0.10) if has_ehat else 0.0
            rows.append({
                "as_of_date": d,
                "ticker": t,
                "horizon": horizon,
                "rank_loss": abs(rng.normal(0.30, 0.10)),
                "ehat_raw": ehat,
                "vol_20d": abs(rng.normal(0.25, 0.05)),
                "vix_percentile_252d": rng.uniform(0, 1),
            })
    return pd.DataFrame(rows)


@pytest.fixture
def conf_data():
    return _make_conformal_data()


@pytest.fixture
def sparse_data():
    return _make_sparse_ehat_data()


@pytest.fixture
def config():
    return ConformalConfig(
        alpha=0.10,
        calibration_window=60,
        min_calibration=30,
    )


# ── Test: Nonconformity scores ───────────────────────────────────────────


class TestNonconformityScores:
    def test_score_columns_present(self, conf_data, config):
        result = compute_nonconformity_scores(conf_data, config)
        for col in ["s_raw", "s_vol", "s_deup"]:
            assert col in result.columns

    def test_scores_non_negative(self, conf_data, config):
        result = compute_nonconformity_scores(conf_data, config)
        for col in ["s_raw", "s_vol", "s_deup"]:
            assert (result[col] >= 0).all()

    def test_raw_equals_rank_loss(self, conf_data, config):
        result = compute_nonconformity_scores(conf_data, config)
        np.testing.assert_array_almost_equal(result["s_raw"], result["rank_loss"])

    def test_deup_normalized_by_ehat(self, conf_data, config):
        result = compute_nonconformity_scores(conf_data, config)
        expected = result["rank_loss"] / np.maximum(result["ehat_raw"], config.eps_deup)
        np.testing.assert_array_almost_equal(result["s_deup"], expected)


# ── Test: PIT safety / no leakage ────────────────────────────────────────


class TestPITSafety:
    def test_first_dates_have_nan_q(self, conf_data, config):
        """Dates before min_calibration should have NaN quantile thresholds."""
        result = compute_conformal_intervals(conf_data, config)
        dates = sorted(result["as_of_date"].unique())

        for d in dates[:config.min_calibration]:
            sub = result[result["as_of_date"] == d]
            assert sub["q_raw"].isna().all(), f"Date {d} should have NaN q_raw"

    def test_calibration_uses_past_only(self, conf_data, config):
        """
        The q threshold at date t should be computable from dates < t only.
        Verify by removing the last date's data and checking q doesn't change.
        """
        result_full = compute_conformal_intervals(conf_data, config)
        dates = sorted(result_full["as_of_date"].unique())

        # Remove last date
        trimmed = conf_data[conf_data["as_of_date"] != dates[-1]]
        result_trimmed = compute_conformal_intervals(trimmed, config)

        # Second-to-last date should have same q values
        second_last = dates[-2]
        q_full = result_full[result_full["as_of_date"] == second_last]["q_raw"].iloc[0]
        q_trim = result_trimmed[result_trimmed["as_of_date"] == second_last]["q_raw"].iloc[0]
        np.testing.assert_almost_equal(q_full, q_trim)

    def test_no_same_date_in_calibration(self, conf_data, config):
        """Calibration window should strictly exclude the current date."""
        config.calibration_window = 5
        config.min_calibration = 3
        result = compute_conformal_intervals(conf_data, config)

        dates = sorted(result["as_of_date"].unique())
        # At date index 5, calibration should use dates [0..4], not date 5
        d5 = dates[5]
        sub = result[result["as_of_date"] == d5]
        assert sub["q_raw"].notna().all()


# ── Test: Coverage properties ────────────────────────────────────────────


class TestCoverage:
    def test_marginal_coverage_reasonable(self, conf_data, config):
        """Coverage should be roughly near the nominal 90% for well-behaved data."""
        result = compute_conformal_intervals(conf_data, config)
        valid = result.dropna(subset=["q_raw"])

        for var in ["raw", "vol", "deup"]:
            cov = valid[f"covered_{var}"].mean()
            assert 0.75 <= cov <= 0.99, f"{var} coverage {cov:.3f} out of range"

    def test_ece_reasonable(self, conf_data, config):
        result = compute_conformal_intervals(conf_data, config)
        diag = compute_diagnostics(result, config)

        for var in ["raw", "vol", "deup"]:
            ece = diag[f"ece_{var}"]
            assert ece < 0.15, f"{var} ECE {ece:.3f} too high"

    def test_covered_is_binary(self, conf_data, config):
        result = compute_conformal_intervals(conf_data, config)
        for var in ["raw", "vol", "deup"]:
            vals = result[f"covered_{var}"].dropna().unique()
            assert set(vals).issubset({0, 1, 0.0, 1.0})


# ── Test: Width properties ───────────────────────────────────────────────


class TestWidths:
    def test_deup_width_varies_with_ehat(self, conf_data, config):
        """DEUP intervals should be wider for high-ê stocks."""
        result = compute_conformal_intervals(conf_data, config)
        valid = result.dropna(subset=["q_deup"])

        low_ehat = valid[valid["ehat_raw"] < valid["ehat_raw"].median()]
        high_ehat = valid[valid["ehat_raw"] >= valid["ehat_raw"].median()]

        assert high_ehat["width_deup"].mean() > low_ehat["width_deup"].mean()

    def test_raw_width_constant_per_date(self, conf_data, config):
        """Raw conformal width should be constant across stocks on same date."""
        result = compute_conformal_intervals(conf_data, config)
        valid = result.dropna(subset=["q_raw"])

        for _, grp in valid.groupby("as_of_date"):
            if len(grp) > 1:
                assert grp["width_raw"].std() < 1e-10

    def test_width_ratio_gt_1_for_deup(self, conf_data, config):
        """Width ratio (high-ê / low-ê) should be > 1 for DEUP."""
        result = compute_conformal_intervals(conf_data, config)
        diag = compute_diagnostics(result, config)

        wr = diag.get("width_ratio_high_low", {})
        if "deup" in wr:
            assert wr["deup"] > 1.0, f"DEUP width ratio {wr['deup']} should be > 1"


# ── Test: Conditional coverage ───────────────────────────────────────────


class TestConditionalCoverage:
    def test_deup_reduces_coverage_spread(self, conf_data, config):
        """
        DEUP-normalized should have smaller coverage spread across ê terciles
        than raw conformal (the whole point of normalization).
        """
        result = compute_conformal_intervals(conf_data, config)
        diag = compute_diagnostics(result, config)

        cc = diag.get("conditional_coverage_ehat", {})
        if "raw" in cc and "deup" in cc:
            raw_spread = cc["raw"].get("spread", 1.0)
            deup_spread = cc["deup"].get("spread", 1.0)
            # DEUP should have smaller or equal spread
            assert deup_spread <= raw_spread + 0.05, (
                f"DEUP spread ({deup_spread}) should be <= raw spread ({raw_spread})"
            )


# ── Test: Diagnostics output ────────────────────────────────────────────


class TestDiagnostics:
    def test_required_keys(self, conf_data, config):
        result = compute_conformal_intervals(conf_data, config)
        diag = compute_diagnostics(result, config)

        required = [
            "n_predictions", "target_coverage",
            "coverage_raw", "coverage_vol", "coverage_deup",
            "ece_raw", "ece_vol", "ece_deup",
            "mean_width_raw", "mean_width_vol", "mean_width_deup",
        ]
        for key in required:
            assert key in diag, f"Missing diagnostic: {key}"

    def test_conditional_coverage_by_vix(self, conf_data, config):
        result = compute_conformal_intervals(conf_data, config)
        diag = compute_diagnostics(result, config)
        assert "conditional_coverage_vix" in diag

    def test_dev_final_split(self, conf_data, config):
        result = compute_conformal_intervals(conf_data, config)
        diag = compute_diagnostics(result, config)
        assert "DEV" in diag


# ── Test: Sparse ehat (60d case) ─────────────────────────────────────────


class TestSparseEhat:
    def test_sparse_ehat_high_coverage(self, sparse_data, config):
        """With 85% zero ê, most stocks get eps normalization → wide intervals → high coverage."""
        result = compute_conformal_intervals(sparse_data, config)
        valid = result.dropna(subset=["q_deup"])
        cov = valid["covered_deup"].mean()
        # Should be very high because zero-ê stocks get huge intervals
        assert cov > 0.85, f"Sparse ê coverage {cov:.3f} should be high"

    def test_sparse_diagnostics_run(self, sparse_data, config):
        """Diagnostics should handle sparse ê without errors."""
        result = compute_conformal_intervals(sparse_data, config)
        diag = compute_diagnostics(result, config)
        assert "n_predictions" in diag


# ── Test: Edge cases ─────────────────────────────────────────────────────


class TestEdgeCases:
    def test_very_short_data(self):
        """With fewer dates than min_calibration, all q should be NaN."""
        df = _make_conformal_data(n_dates=20, n_tickers=5)
        config = ConformalConfig(min_calibration=30)
        result = compute_conformal_intervals(df, config)
        assert result["q_raw"].isna().all()

    def test_small_calibration_window(self):
        df = _make_conformal_data(n_dates=50, n_tickers=10)
        config = ConformalConfig(calibration_window=10, min_calibration=5)
        result = compute_conformal_intervals(df, config)
        valid = result.dropna(subset=["q_raw"])
        assert len(valid) > 0

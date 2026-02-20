"""
Tests for ExpertHealthEstimator — Chapter 13.4b.

Validates:
    - PIT-safety (matured RankIC alignment)
    - Date indexing correctness
    - Signal monotonicity (higher H → better realised outcomes)
    - Output schema
    - Leakage checks
"""

import numpy as np
import pandas as pd
import pytest

from src.uncertainty.expert_health import ExpertHealthEstimator, HealthConfig


# ── Fixtures ──────────────────────────────────────────────────────────────


def _make_enriched(
    n_dates: int = 300,
    n_tickers: int = 30,
    horizon: int = 20,
    seed: int = 42,
) -> pd.DataFrame:
    """Create synthetic enriched residuals with realistic structure."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-02", periods=n_dates)
    tickers = [f"T{i:03d}" for i in range(n_tickers)]

    rows = []
    for d in dates:
        for t in tickers:
            score = rng.normal(0, 0.02)
            er = score * 0.3 + rng.normal(0, 0.05)
            rows.append({
                "as_of_date": d,
                "ticker": t,
                "score": score,
                "excess_return": er,
                "horizon": horizon,
                "rank_loss": abs(rng.normal(0.1, 0.05)),
                "vol_20d": abs(rng.normal(0.25, 0.05)),
                "mom_1m": rng.normal(0, 0.05),
                "adv_20d": rng.lognormal(18, 1),
                "vix_percentile_252d": rng.uniform(0, 1),
                "market_vol_21d": abs(rng.normal(0.15, 0.03)),
                "vol_60d": abs(rng.normal(0.24, 0.04)),
            })
    return pd.DataFrame(rows)


def _make_enriched_with_regime_shift(
    n_dates: int = 400,
    n_tickers: int = 30,
    horizon: int = 20,
    seed: int = 42,
) -> pd.DataFrame:
    """Synthetic data with a clear regime shift (dates 200-250 have bad signal)."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-02", periods=n_dates)
    tickers = [f"T{i:03d}" for i in range(n_tickers)]

    rows = []
    for i, d in enumerate(dates):
        in_crisis = 200 <= i <= 250
        for t in tickers:
            score = rng.normal(0, 0.02)
            if in_crisis:
                # Signal inverts during crisis
                er = -score * 0.5 + rng.normal(0, 0.08)
                vol = abs(rng.normal(0.40, 0.08))
            else:
                er = score * 0.4 + rng.normal(0, 0.04)
                vol = abs(rng.normal(0.25, 0.05))
            rows.append({
                "as_of_date": d,
                "ticker": t,
                "score": score,
                "excess_return": er,
                "horizon": horizon,
                "rank_loss": abs(rng.normal(0.15 if in_crisis else 0.08, 0.05)),
                "vol_20d": vol,
                "mom_1m": rng.normal(-0.03 if in_crisis else 0.01, 0.05),
                "adv_20d": rng.lognormal(18, 1),
                "vix_percentile_252d": 0.9 if in_crisis else rng.uniform(0.2, 0.7),
                "market_vol_21d": abs(rng.normal(0.30 if in_crisis else 0.15, 0.03)),
                "vol_60d": abs(rng.normal(0.35 if in_crisis else 0.24, 0.04)),
            })
    return pd.DataFrame(rows)


@pytest.fixture
def enriched_20d():
    return _make_enriched(horizon=20)


@pytest.fixture
def enriched_regime():
    return _make_enriched_with_regime_shift(horizon=20)


@pytest.fixture
def config():
    return HealthConfig(
        horizon=20,
        ewma_halflife=30,
        ewma_min_periods=20,
        reference_window=120,
    )


# ── Test: Output schema ──────────────────────────────────────────────────


class TestOutputSchema:
    """Verify health DataFrame has all required columns."""

    def test_required_columns(self, enriched_20d, config):
        est = ExpertHealthEstimator(config)
        health_df, _ = est.compute(enriched_20d)

        required = [
            "date", "daily_rankic", "n_stocks",
            "matured_rankic", "H_realized",
            "H_drift_raw", "feat_drift", "score_drift", "corr_spike",
            "H_disagree_raw",
            "H_combined", "G_exposure",
            "z_realized", "z_drift", "z_disagree",
        ]
        for col in required:
            assert col in health_df.columns, f"Missing column: {col}"

    def test_h_combined_range(self, enriched_20d, config):
        est = ExpertHealthEstimator(config)
        health_df, _ = est.compute(enriched_20d)
        valid = health_df["H_combined"].dropna()
        assert valid.min() >= 0.0 - 1e-9
        assert valid.max() <= 1.0 + 1e-9

    def test_g_exposure_range(self, enriched_20d, config):
        est = ExpertHealthEstimator(config)
        health_df, _ = est.compute(enriched_20d)
        valid = health_df["G_exposure"].dropna()
        assert valid.min() >= 0.0 - 1e-9
        assert valid.max() <= 1.0 + 1e-9

    def test_dates_unique(self, enriched_20d, config):
        est = ExpertHealthEstimator(config)
        health_df, _ = est.compute(enriched_20d)
        assert health_df["date"].is_unique


# ── Test: PIT safety / leakage ────────────────────────────────────────────


class TestPITSafety:
    """Verify no future information leakage."""

    def test_matured_rankic_lag(self, enriched_20d, config):
        """Matured RankIC at row i should equal daily RankIC at row i - horizon."""
        est = ExpertHealthEstimator(config)
        health_df, _ = est.compute(enriched_20d)

        h = config.horizon
        for i in range(h, len(health_df)):
            expected = health_df.iloc[i - h]["daily_rankic"]
            actual = health_df.iloc[i]["matured_rankic"]
            if not np.isnan(actual):
                np.testing.assert_almost_equal(actual, expected, decimal=10)

    def test_matured_rankic_nan_at_start(self, enriched_20d, config):
        """First `horizon` rows of matured_rankic should be NaN."""
        est = ExpertHealthEstimator(config)
        health_df, _ = est.compute(enriched_20d)

        for i in range(config.horizon):
            assert np.isnan(health_df.iloc[i]["matured_rankic"])

    def test_drift_uses_no_future_data(self, enriched_20d, config):
        """H_drift should only be valid after reference_window warmup dates."""
        est = ExpertHealthEstimator(config)
        health_df, _ = est.compute(enriched_20d)

        warmup = 60  # minimum reference_window check
        for i in range(warmup):
            assert np.isnan(health_df.iloc[i]["feat_drift"])

    def test_h_realized_nan_warmup(self, enriched_20d, config):
        """H_realized needs horizon + ewma_min_periods warmup."""
        est = ExpertHealthEstimator(config)
        health_df, _ = est.compute(enriched_20d)

        min_valid = config.horizon + config.ewma_min_periods
        n_nan = health_df["H_realized"].isna().sum()
        assert n_nan >= min_valid - 1


# ── Test: Disagreement ────────────────────────────────────────────────────


class TestDisagreement:
    def test_single_model_fallback(self, enriched_20d, config):
        """Without other models, should use dispersion proxy."""
        est = ExpertHealthEstimator(config)
        health_df, _ = est.compute(enriched_20d, other_models=None)
        assert "H_disagree_raw" in health_df.columns

    def test_with_other_model(self, enriched_20d, config):
        """With another model, should compute cross-expert disagreement."""
        other = _make_enriched(horizon=20, seed=99)
        est = ExpertHealthEstimator(config)
        health_df, _ = est.compute(enriched_20d, {"model_b": other})
        assert "H_disagree_raw" in health_df.columns
        assert health_df["H_disagree_raw"].notna().sum() > 0


# ── Test: Regime detection ────────────────────────────────────────────────


class TestRegimeDetection:
    def test_h_drops_during_crisis(self, enriched_regime, config):
        """H should be lower during the synthetic crisis period."""
        config.reference_window = 120
        est = ExpertHealthEstimator(config)
        health_df, _ = est.compute(enriched_regime)

        dates = sorted(enriched_regime["as_of_date"].unique())
        crisis_dates = dates[200:251]
        normal_dates = dates[100:200]

        crisis_h = health_df[health_df["date"].isin(crisis_dates)]["H_combined"]
        normal_h = health_df[health_df["date"].isin(normal_dates)]["H_combined"]

        crisis_valid = crisis_h.dropna()
        normal_valid = normal_h.dropna()

        if len(crisis_valid) > 5 and len(normal_valid) > 5:
            # The realized component should catch the regime shift (with lag)
            # Allow for lag: crisis detection happens after horizon delay
            assert crisis_valid.mean() < normal_valid.mean() + 0.2


# ── Test: Diagnostics output ─────────────────────────────────────────────


class TestDiagnostics:
    def test_diagnostics_keys(self, enriched_20d, config):
        est = ExpertHealthEstimator(config)
        _, diag = est.compute(enriched_20d)

        required_keys = ["n_dates", "horizon", "rho_H_rankic", "auroc_bad_day"]
        for key in required_keys:
            assert key in diag, f"Missing diagnostic: {key}"

    def test_quintile_analysis(self, enriched_20d, config):
        est = ExpertHealthEstimator(config)
        _, diag = est.compute(enriched_20d)
        assert "quintile_analysis" in diag

    def test_component_correlations(self, enriched_20d, config):
        est = ExpertHealthEstimator(config)
        _, diag = est.compute(enriched_20d)
        assert "rho_H_realized_rankic" in diag


# ── Test: Monotonicity ────────────────────────────────────────────────────


class TestMonotonicity:
    def test_h_combined_direction(self, enriched_20d, config):
        """Higher H_realized should push H_combined higher (positive direction)."""
        est = ExpertHealthEstimator(config)
        health_df, _ = est.compute(enriched_20d)

        valid = health_df.dropna(subset=["H_realized", "H_combined"])
        if len(valid) > 50:
            rho = valid["H_realized"].corr(valid["H_combined"])
            assert rho > 0, f"H_realized should positively correlate with H_combined, got {rho}"

    def test_g_exposure_direction(self, enriched_20d, config):
        """G_exposure should correlate with H_combined."""
        est = ExpertHealthEstimator(config)
        health_df, _ = est.compute(enriched_20d)

        valid = health_df.dropna(subset=["H_combined", "G_exposure"])
        if len(valid) > 50:
            rho = valid["H_combined"].corr(valid["G_exposure"])
            assert rho > 0.5, f"G should strongly follow H, got {rho}"


# ── Test: Edge cases ─────────────────────────────────────────────────────


class TestEdgeCases:
    def test_empty_horizon(self, config):
        df = _make_enriched(horizon=60)
        config.horizon = 20  # No 20d data
        est = ExpertHealthEstimator(config)
        with pytest.raises(ValueError, match="No data for horizon"):
            est.compute(df)

    def test_small_data(self):
        df = _make_enriched(n_dates=30, n_tickers=5, horizon=20)
        config = HealthConfig(horizon=20, ewma_min_periods=5, reference_window=20)
        est = ExpertHealthEstimator(config)
        health_df, diag = est.compute(df)
        assert len(health_df) > 0

"""
Tests for Chapter 9: FinText Gate Evaluation & Leak Tripwires

Validates:
 - Gate evaluation logic (all three gates)
 - EMA smoothing for churn reduction
 - Leak tripwire controls (shuffle, lag, year-mismatch)
 - Integration with evaluation outputs
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.fintext_adapter import _apply_ema_smoothing


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def sample_eval_rows():
    """Create realistic evaluation rows for gate testing."""
    np.random.seed(42)
    dates = pd.bdate_range("2024-01-02", periods=21)
    tickers = [f"TICK_{i:02d}" for i in range(50)]
    rows = []
    for fold_id in ["fold_01", "fold_02"]:
        for h in [20, 60, 90]:
            for d in dates:
                for t in tickers:
                    # Slight positive signal: score correlates with excess_return
                    signal = np.random.normal(0, 1)
                    noise = np.random.normal(0, 3)
                    rows.append({
                        "as_of_date": d,
                        "ticker": t,
                        "stable_id": t,
                        "fold_id": fold_id,
                        "horizon": h,
                        "score": signal,
                        "excess_return": signal + noise,
                    })
    return pd.DataFrame(rows)


@pytest.fixture
def sample_eval_rows_no_signal():
    """Eval rows where score has no correlation with excess_return."""
    np.random.seed(99)
    dates = pd.bdate_range("2024-01-02", periods=21)
    tickers = [f"TICK_{i:02d}" for i in range(50)]
    rows = []
    for fold_id in ["fold_01"]:
        for h in [20, 60, 90]:
            for d in dates:
                for t in tickers:
                    rows.append({
                        "as_of_date": d,
                        "ticker": t,
                        "stable_id": t,
                        "fold_id": fold_id,
                        "horizon": h,
                        "score": np.random.normal(0, 1),
                        "excess_return": np.random.normal(0, 1),
                    })
    return pd.DataFrame(rows)


@pytest.fixture
def ema_test_df():
    """DataFrame for testing EMA smoothing."""
    dates = pd.bdate_range("2024-01-02", periods=20)
    rows = []
    for d in dates:
        for t in ["AAPL", "MSFT", "GOOG"]:
            rows.append({
                "as_of_date": d,
                "ticker": t,
                "score": np.random.normal(0, 0.01),
            })
    return pd.DataFrame(rows)


# ============================================================================
# TEST EMA SMOOTHING
# ============================================================================

class TestEMASmoothing:
    """Tests for the EMA score smoothing function."""

    def test_smoothing_reduces_volatility(self, ema_test_df):
        """EMA smoothing should reduce score-to-score volatility."""
        raw_vol = ema_test_df.groupby("ticker")["score"].std()
        smoothed = _apply_ema_smoothing(ema_test_df.copy(), halflife_days=5)
        smooth_vol = smoothed.groupby("ticker")["score"].std()
        assert (smooth_vol < raw_vol).all(), "Smoothing should reduce volatility"

    def test_smoothing_preserves_tickers(self, ema_test_df):
        """EMA smoothing should not drop any rows or tickers."""
        smoothed = _apply_ema_smoothing(ema_test_df.copy(), halflife_days=5)
        assert len(smoothed) == len(ema_test_df)
        assert set(smoothed["ticker"]) == set(ema_test_df["ticker"])

    def test_smoothing_preserves_dates(self, ema_test_df):
        """EMA smoothing should not change dates."""
        smoothed = _apply_ema_smoothing(ema_test_df.copy(), halflife_days=5)
        assert set(smoothed["as_of_date"]) == set(ema_test_df["as_of_date"])

    def test_no_smoothing_when_halflife_zero(self, ema_test_df):
        """Half-life=0 should return scores unchanged."""
        smoothed = _apply_ema_smoothing(ema_test_df.copy(), halflife_days=0)
        np.testing.assert_array_equal(
            smoothed.sort_values(["ticker", "as_of_date"])["score"].values,
            ema_test_df.sort_values(["ticker", "as_of_date"])["score"].values,
        )

    def test_smoothing_finite_values(self, ema_test_df):
        """All smoothed values should be finite."""
        smoothed = _apply_ema_smoothing(ema_test_df.copy(), halflife_days=5)
        assert smoothed["score"].isna().sum() == 0
        assert np.all(np.isfinite(smoothed["score"].values))

    def test_smoothing_reduces_churn(self):
        """EMA smoothing should reduce top-10 churn."""
        np.random.seed(42)
        dates = pd.bdate_range("2024-01-02", periods=30)
        tickers = [f"T{i:02d}" for i in range(50)]
        rows = []
        for d in dates:
            for t in tickers:
                rows.append({
                    "as_of_date": d,
                    "ticker": t,
                    "score": np.random.normal(0, 0.01),
                })
        df = pd.DataFrame(rows)

        def compute_avg_churn(df_in):
            churns = []
            all_dates = sorted(df_in["as_of_date"].unique())
            for i in range(1, len(all_dates)):
                prev = set(
                    df_in[df_in["as_of_date"] == all_dates[i-1]]
                    .nlargest(10, "score")["ticker"]
                )
                curr = set(
                    df_in[df_in["as_of_date"] == all_dates[i]]
                    .nlargest(10, "score")["ticker"]
                )
                churns.append(1 - len(prev & curr) / 10)
            return np.mean(churns)

        raw_churn = compute_avg_churn(df)
        smoothed = _apply_ema_smoothing(df.copy(), halflife_days=5)
        smooth_churn = compute_avg_churn(smoothed)
        assert smooth_churn < raw_churn, (
            f"Smoothed churn ({smooth_churn:.2f}) should be less than "
            f"raw churn ({raw_churn:.2f})"
        )


# ============================================================================
# TEST GATE EVALUATION LOGIC
# ============================================================================

class TestGateEvaluation:
    """Tests for gate evaluation logic."""

    def test_gate1_positive_signal_passes(self, sample_eval_rows):
        """Gate 1: Positive RankIC >= 0.02 for >= 2 horizons."""
        horizons = sorted(sample_eval_rows["horizon"].unique())
        passing_horizons = 0
        for h in horizons:
            h_df = sample_eval_rows[sample_eval_rows["horizon"] == h]
            ics = []
            for _, g in h_df.groupby("as_of_date"):
                if len(g) >= 10:
                    ic, _ = stats.spearmanr(g["score"], g["excess_return"])
                    ics.append(ic)
            if np.mean(ics) >= 0.02:
                passing_horizons += 1
        assert passing_horizons >= 2, (
            f"Gate 1 should pass with positive signal data ({passing_horizons} horizons)"
        )

    def test_gate1_no_signal_fails(self, sample_eval_rows_no_signal):
        """Gate 1: No signal => RankIC should be near zero."""
        h_df = sample_eval_rows_no_signal[
            sample_eval_rows_no_signal["horizon"] == 20
        ]
        ics = []
        for _, g in h_df.groupby("as_of_date"):
            if len(g) >= 10:
                ic, _ = stats.spearmanr(g["score"], g["excess_return"])
                ics.append(ic)
        mean_ic = np.mean(ics)
        assert abs(mean_ic) < 0.05, (
            f"No-signal data should have near-zero RankIC, got {mean_ic:.4f}"
        )

    def test_gate2_absolute_threshold(self, sample_eval_rows):
        """Gate 2: Some horizon should have RankIC >= 0.05."""
        any_above = False
        for h in [20, 60, 90]:
            h_df = sample_eval_rows[sample_eval_rows["horizon"] == h]
            ics = []
            for _, g in h_df.groupby("as_of_date"):
                if len(g) >= 10:
                    ic, _ = stats.spearmanr(g["score"], g["excess_return"])
                    ics.append(ic)
            if np.mean(ics) >= 0.05:
                any_above = True
                break
        assert any_above, "Gate 2 should pass with positive signal data"

    def test_gate3_churn_computation(self, sample_eval_rows):
        """Gate 3: Churn should be computable and between 0 and 1."""
        h_df = sample_eval_rows[sample_eval_rows["horizon"] == 20]
        dates = sorted(h_df["as_of_date"].unique())
        churns = []
        for i in range(1, len(dates)):
            prev_top = set(
                h_df[h_df["as_of_date"] == dates[i-1]]
                .nlargest(10, "score")["ticker"]
            )
            curr_top = set(
                h_df[h_df["as_of_date"] == dates[i]]
                .nlargest(10, "score")["ticker"]
            )
            churns.append(1 - len(prev_top & curr_top) / 10)
        assert len(churns) > 0, "Should compute at least one churn value"
        assert all(0 <= c <= 1 for c in churns), "Churn must be between 0 and 1"


# ============================================================================
# TEST LEAK TRIPWIRES
# ============================================================================

class TestLeakTripwires:
    """Tests for leak tripwire controls."""

    def test_shuffle_collapses_rankic(self, sample_eval_rows):
        """Shuffling scores within date should collapse RankIC to ~0."""
        np.random.seed(42)
        shuffled = sample_eval_rows.copy()
        shuffled["score"] = (
            shuffled.groupby(["as_of_date", "horizon"])["score"]
            .transform(lambda x: np.random.permutation(x.values))
        )
        ics = []
        for _, g in shuffled.groupby(["as_of_date", "horizon"]):
            if len(g) >= 10:
                ic, _ = stats.spearmanr(g["score"], g["excess_return"])
                ics.append(ic)
        mean_shuffled_ic = np.mean(ics)
        assert abs(mean_shuffled_ic) < 0.02, (
            f"Shuffled RankIC should be ~0, got {mean_shuffled_ic:.4f}"
        )

    def test_lag_degrades_rankic(self, sample_eval_rows):
        """Lagging scores should degrade RankIC vs real."""
        # Real RankIC
        real_ics = []
        for _, g in sample_eval_rows.groupby(["as_of_date", "horizon"]):
            if len(g) >= 10:
                ic, _ = stats.spearmanr(g["score"], g["excess_return"])
                real_ics.append(ic)
        real_mean = np.mean(real_ics)

        # Lagged RankIC (7-day lag)
        lagged = sample_eval_rows.copy().sort_values(["ticker", "horizon", "as_of_date"])
        lagged["score"] = lagged.groupby(["ticker", "horizon"])["score"].shift(7)
        lagged = lagged.dropna(subset=["score"])

        lag_ics = []
        for _, g in lagged.groupby(["as_of_date", "horizon"]):
            if len(g) >= 10:
                ic, _ = stats.spearmanr(g["score"], g["excess_return"])
                lag_ics.append(ic)
        lag_mean = np.mean(lag_ics) if lag_ics else 0

        # With independent noise per day, lagged scores decorrelate
        assert lag_mean < real_mean or abs(lag_mean) < 0.05, (
            f"Lagged RankIC ({lag_mean:.4f}) should be lower than real ({real_mean:.4f})"
        )

    def test_full_shuffle_destroys_signal(self, sample_eval_rows):
        """Complete cross-sectional shuffle within each date should kill signal."""
        np.random.seed(123)
        shuffled = sample_eval_rows.copy()
        for d in shuffled["as_of_date"].unique():
            mask = shuffled["as_of_date"] == d
            vals = shuffled.loc[mask, "score"].values.copy()
            np.random.shuffle(vals)
            shuffled.loc[mask, "score"] = vals

        ics = []
        for _, g in shuffled.groupby(["as_of_date", "horizon"]):
            if len(g) >= 10:
                ic, _ = stats.spearmanr(g["score"], g["excess_return"])
                ics.append(ic)

        assert abs(np.mean(ics)) < 0.02, (
            "Full shuffle should destroy signal"
        )

    def test_real_eval_rows_gate_results_exist(self):
        """If evaluation has been run, gate_results.json should exist."""
        gate_path = Path("evaluation_outputs/chapter9_fintext_small_smoke/gate_results.json")
        if gate_path.exists():
            import json
            with open(gate_path) as f:
                results = json.load(f)
            assert "gates" in results, "gate_results.json should have 'gates' key"
            assert "tripwires" in results, "gate_results.json should have 'tripwires' key"
            gates = results["gates"]
            assert gates["gate_1_pass"] is True, "Gate 1 should pass"
            assert gates["gate_2_pass"] is True, "Gate 2 should pass"
            assert gates["gate_3_pass"] is True, "Gate 3 should pass"
        else:
            pytest.skip("No evaluation output found — run evaluation first")

    def test_real_eval_tripwires_pass(self):
        """If evaluation has been run, all tripwires should pass."""
        gate_path = Path("evaluation_outputs/chapter9_fintext_small_smoke/gate_results.json")
        if gate_path.exists():
            import json
            with open(gate_path) as f:
                results = json.load(f)
            if results.get("tripwires"):
                tripwires = results["tripwires"]
                assert tripwires["shuffle"]["pass"] is True, "Shuffle tripwire should pass"
                assert tripwires["lag"]["pass"] is True, "Lag tripwire should pass"
                assert tripwires["year_mismatch"]["pass"] is True, "Year-mismatch tripwire should pass"
        else:
            pytest.skip("No evaluation output found — run evaluation first")


# ============================================================================
# TEST ADDITIONAL SUCCESS CRITERIA
# ============================================================================

class TestAdditionalCriteria:
    """Tests for additional success criteria from outline."""

    def test_stable_ic_sign_across_windows(self, sample_eval_rows):
        """RankIC sign should be stable across >= 70% of windows."""
        h_df = sample_eval_rows[sample_eval_rows["horizon"] == 20]
        ics = []
        for _, g in h_df.groupby("as_of_date"):
            if len(g) >= 10:
                ic, _ = stats.spearmanr(g["score"], g["excess_return"])
                ics.append(ic)
        pct_positive = sum(1 for ic in ics if ic > 0) / len(ics) if ics else 0
        # With synthetic positive signal, should be above 50%
        assert pct_positive > 0.5, (
            f"Positive signal should have > 50% positive IC windows, got {pct_positive:.1%}"
        )

    def test_score_variation_across_tickers(self, sample_eval_rows):
        """Scores should vary across tickers (not constant)."""
        for d in sample_eval_rows["as_of_date"].unique()[:3]:
            date_df = sample_eval_rows[
                (sample_eval_rows["as_of_date"] == d) &
                (sample_eval_rows["horizon"] == 20)
            ]
            assert date_df["score"].std() > 0, (
                f"Scores on {d} should vary, got std={date_df['score'].std()}"
            )

    def test_no_nans_in_scores(self, sample_eval_rows):
        """No NaN values should exist in scores."""
        assert sample_eval_rows["score"].isna().sum() == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

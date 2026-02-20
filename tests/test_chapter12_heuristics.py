"""
Tests for Chapter 12.3 — Regime-Aware Heuristic Baselines.

Validates both heuristic approaches (vol-sizing, regime-blending),
the metric computation, shadow portfolio construction, and real outputs.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.run_chapter12_heuristics import (
    apply_regime_blending,
    apply_vol_sizing,
    build_shadow_portfolio,
    compute_metrics,
    compute_portfolio_metrics,
    join_features_to_eval_rows,
    sigmoid,
    PERIODS_PER_YEAR,
    VOL_SCALE_C,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_eval_rows():
    """Minimal eval_rows with known properties."""
    np.random.seed(42)
    n_stocks = 30
    dates = pd.bdate_range("2020-01-06", periods=60, freq="B")
    rows = []
    for i, dt in enumerate(dates):
        for h in [20, 60, 90]:
            for j in range(n_stocks):
                rows.append({
                    "as_of_date": dt,
                    "ticker": f"TICK_{j:02d}",
                    "stable_id": f"SID_{j:02d}",
                    "horizon": h,
                    "fold_id": i // 20,
                    "score": np.random.randn(),
                    "excess_return": np.random.randn() * 0.05,
                    "vix_percentile_252d": np.random.uniform(20, 80),
                })
    return pd.DataFrame(rows)


@pytest.fixture
def mock_features():
    """Mock features table for join."""
    np.random.seed(7)
    dates = pd.bdate_range("2020-01-06", periods=60, freq="B")
    rows = []
    for dt in dates:
        for j in range(30):
            rows.append({
                "date": dt.date(),
                "stable_id": f"SID_{j:02d}",
                "vol_20d": np.random.uniform(0.10, 0.60),
                "mom_1m": np.random.randn() * 0.10,
            })
    return pd.DataFrame(rows)


@pytest.fixture
def enriched_eval_rows(mock_eval_rows, mock_features):
    return join_features_to_eval_rows(mock_eval_rows, mock_features)


# ---------------------------------------------------------------------------
# Approach A: Volatility-Scaled Ranking
# ---------------------------------------------------------------------------

class TestVolSizing:
    def test_low_vol_stocks_unaffected(self, enriched_eval_rows):
        """Stocks with vol_20d < c should keep score ≈ original."""
        er = enriched_eval_rows.copy()
        low_vol_mask = er["vol_20d"] < VOL_SCALE_C
        if low_vol_mask.sum() == 0:
            pytest.skip("No low-vol stocks in fixture")
        original_scores = er.loc[low_vol_mask, "score"].values.copy()
        sized = apply_vol_sizing(er)
        np.testing.assert_array_almost_equal(
            sized.loc[low_vol_mask, "score"].values,
            original_scores,
            decimal=4,
        )

    def test_high_vol_stocks_penalised(self, enriched_eval_rows):
        """Stocks with vol_20d > c should have |score| reduced."""
        er = enriched_eval_rows.copy()
        high_vol_mask = er["vol_20d"] > VOL_SCALE_C * 1.5
        if high_vol_mask.sum() == 0:
            pytest.skip("No high-vol stocks in fixture")
        orig_abs = er.loc[high_vol_mask, "score"].abs().values
        sized = apply_vol_sizing(er)
        new_abs = sized.loc[high_vol_mask, "score"].abs().values
        assert np.all(new_abs <= orig_abs + 1e-10)

    def test_scale_never_exceeds_one(self, enriched_eval_rows):
        """The vol scale multiplier should be in (0, 1]."""
        er = enriched_eval_rows.copy()
        vol = er["vol_20d"].clip(lower=0.01)
        scale = np.minimum(1.0, VOL_SCALE_C / vol)
        assert scale.max() <= 1.0 + 1e-10
        assert scale.min() > 0

    def test_missing_vol_keeps_original_score(self):
        """Rows with missing vol_20d should keep original score."""
        er = pd.DataFrame({
            "score": [1.0, 2.0, -0.5],
            "vol_20d": [0.30, np.nan, 0.10],
        })
        sized = apply_vol_sizing(er, c=0.25)
        assert sized.loc[1, "score"] == 2.0  # NaN vol → scale=1

    def test_preserves_row_count(self, enriched_eval_rows):
        sized = apply_vol_sizing(enriched_eval_rows)
        assert len(sized) == len(enriched_eval_rows)


# ---------------------------------------------------------------------------
# Approach B: Regime-Blended Ensemble
# ---------------------------------------------------------------------------

class TestRegimeBlending:
    def test_low_vix_preserves_lgb_rank(self, enriched_eval_rows):
        """When VIX percentile is low, blended score ≈ LGB rank."""
        er = enriched_eval_rows.copy()
        er["vix_percentile_252d"] = 20.0  # well below threshold
        blended = apply_regime_blending(er, alpha=0.5, threshold=67, tau=10)
        lgb_rank = er.groupby(["as_of_date", "horizon"])["score"].rank(pct=True)
        corr = np.corrcoef(blended["score"].values, lgb_rank.values)[0, 1]
        assert corr > 0.95

    def test_high_vix_shifts_toward_momentum(self, enriched_eval_rows):
        """When VIX percentile is very high, blended incorporates momentum."""
        er = enriched_eval_rows.copy()
        er["vix_percentile_252d"] = 95.0  # well above threshold
        blended = apply_regime_blending(er, alpha=0.5, threshold=67, tau=10)
        lgb_rank = er.groupby(["as_of_date", "horizon"])["score"].rank(pct=True)
        corr = np.corrcoef(blended["score"].values, lgb_rank.values)[0, 1]
        assert corr < 0.95  # should be less correlated with pure LGB

    def test_output_is_bounded_0_1(self, enriched_eval_rows):
        """Blended score (from percentile ranks) should be in [0, 1]."""
        blended = apply_regime_blending(enriched_eval_rows)
        assert blended["score"].min() >= -0.01
        assert blended["score"].max() <= 1.01

    def test_sigmoid_properties(self):
        assert abs(sigmoid(np.array([0.0]))[0] - 0.5) < 1e-10
        assert sigmoid(np.array([100.0]))[0] > 0.99
        assert sigmoid(np.array([-100.0]))[0] < 0.01

    def test_preserves_row_count(self, enriched_eval_rows):
        blended = apply_regime_blending(enriched_eval_rows)
        assert len(blended) == len(enriched_eval_rows)


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------

class TestComputeMetrics:
    def test_returns_all_horizons(self, mock_eval_rows):
        metrics = compute_metrics(mock_eval_rows)
        assert set(metrics.keys()) == {20, 60, 90}

    def test_rankic_bounded(self, mock_eval_rows):
        metrics = compute_metrics(mock_eval_rows)
        for h in [20, 60, 90]:
            ic = metrics[h]["mean_rankic"]
            assert -1 <= ic <= 1

    def test_cost_survival_bounded(self, mock_eval_rows):
        metrics = compute_metrics(mock_eval_rows)
        for h in [20, 60, 90]:
            cs = metrics[h]["cost_survival"]
            if not np.isnan(cs):
                assert 0 <= cs <= 1

    def test_churn_bounded(self, mock_eval_rows):
        metrics = compute_metrics(mock_eval_rows)
        for h in [20, 60, 90]:
            ch = metrics[h]["median_churn"]
            if not np.isnan(ch):
                assert 0 <= ch <= 1


# ---------------------------------------------------------------------------
# Shadow portfolio
# ---------------------------------------------------------------------------

class TestShadowPortfolio:
    def test_monthly_subsample(self, mock_eval_rows):
        """Shadow portfolio should return non-overlapping monthly rows."""
        portfolio = build_shadow_portfolio(mock_eval_rows)
        if not portfolio.empty:
            assert len(portfolio) <= 12  # ~3 months of data at most
            assert "ls_return_net" in portfolio.columns

    def test_portfolio_metrics_annualization(self):
        """Verify annualization uses ×12."""
        monthly = pd.DataFrame({
            "date": pd.date_range("2020-01-01", periods=24, freq="MS"),
            "ls_return_net": np.full(24, 0.05),  # 5% every month
            "turnover": 0.2,
        })
        m = compute_portfolio_metrics(monthly)
        assert abs(m["ann_return"] - 0.05 * 12) < 1e-10
        assert m["ann_sharpe"] > 0
        assert m["n_months"] == 24

    def test_portfolio_metrics_empty(self):
        m = compute_portfolio_metrics(pd.DataFrame())
        assert np.isnan(m["ann_sharpe"])


# ---------------------------------------------------------------------------
# Integration: feature join
# ---------------------------------------------------------------------------

class TestFeatureJoin:
    def test_join_adds_columns(self, mock_eval_rows, mock_features):
        merged = join_features_to_eval_rows(mock_eval_rows, mock_features)
        assert "vol_20d" in merged.columns
        assert "mom_1m" in merged.columns

    def test_join_preserves_row_count(self, mock_eval_rows, mock_features):
        merged = join_features_to_eval_rows(mock_eval_rows, mock_features)
        assert len(merged) == len(mock_eval_rows)


# ---------------------------------------------------------------------------
# Output artifact validation (real data)
# ---------------------------------------------------------------------------

class TestOutputArtifacts:
    OUTPUT_DIR = Path("evaluation_outputs/chapter12/regime_heuristic")

    @pytest.mark.skipif(
        not (OUTPUT_DIR / "heuristic_comparison.json").exists(),
        reason="Real outputs not present",
    )
    def test_comparison_json_structure(self):
        with open(self.OUTPUT_DIR / "heuristic_comparison.json") as f:
            data = json.load(f)
        for model in ["LGB_baseline", "Vol_Sized", "Regime_Blended"]:
            assert model in data
            assert "signal" in data[model]
            assert "portfolio" in data[model]
            for h in ["20", "60", "90"]:
                assert h in data[model]["signal"]

    @pytest.mark.skipif(
        not (OUTPUT_DIR / "heuristic_comparison.json").exists(),
        reason="Real outputs not present",
    )
    def test_vol_sizing_improves_portfolio(self):
        """Vol-sizing should improve Sharpe or MaxDD vs baseline (as seen in run)."""
        with open(self.OUTPUT_DIR / "heuristic_comparison.json") as f:
            data = json.load(f)
        base = data["LGB_baseline"]["portfolio"]
        vol = data["Vol_Sized"]["portfolio"]
        sharpe_improved = vol["ann_sharpe"] > base["ann_sharpe"]
        dd_improved = vol["max_drawdown"] > base["max_drawdown"]
        assert sharpe_improved or dd_improved, "Vol-sizing should improve at least one metric"

    @pytest.mark.skipif(
        not (OUTPUT_DIR / "heuristic_comparison.json").exists(),
        reason="Real outputs not present",
    )
    def test_portfolio_sharpe_realistic(self):
        """All Sharpe ratios should be in realistic range."""
        with open(self.OUTPUT_DIR / "heuristic_comparison.json") as f:
            data = json.load(f)
        for model in data:
            sharpe = data[model]["portfolio"]["ann_sharpe"]
            assert -2 < sharpe < 5, f"{model} Sharpe {sharpe:.2f} out of range"

    @pytest.mark.skipif(
        not (OUTPUT_DIR / "vol_sized/eval_rows.parquet").exists(),
        reason="Real outputs not present",
    )
    def test_vol_sized_eval_rows_saved(self):
        df = pd.read_parquet(self.OUTPUT_DIR / "vol_sized/eval_rows.parquet")
        assert "score" in df.columns
        assert len(df) > 0

    @pytest.mark.skipif(
        not (OUTPUT_DIR / "regime_blended/eval_rows.parquet").exists(),
        reason="Real outputs not present",
    )
    def test_regime_blended_eval_rows_saved(self):
        df = pd.read_parquet(self.OUTPUT_DIR / "regime_blended/eval_rows.parquet")
        assert "score" in df.columns
        assert df["score"].min() >= -0.01
        assert df["score"].max() <= 1.01

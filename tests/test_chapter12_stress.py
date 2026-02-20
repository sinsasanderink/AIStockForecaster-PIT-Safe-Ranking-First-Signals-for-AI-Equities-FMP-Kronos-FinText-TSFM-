"""
Tests for Chapter 12.2 — Regime Stress Tests (Shadow Portfolio).

The stress tests operate on non-overlapping monthly returns (subsampled
from the raw overlapping 20-day forward return CSV).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.run_chapter12_stress_tests import (
    annotate_with_regime,
    compute_portfolio_metrics,
    compute_rolling_sharpe,
    find_worst_drawdowns,
    run_stress_tests,
    subsample_monthly,
    PERIODS_PER_YEAR,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def overlapping_returns_df():
    """
    Simulates the shadow_portfolio_returns.csv structure:
    one row per trading day with overlapping 20d forward returns.
    """
    np.random.seed(42)
    dates = pd.bdate_range("2020-01-02", periods=500)
    monthly_return = 0.02  # ~2% monthly = ~24% annual
    returns_20d = np.random.normal(monthly_return, 0.06, size=len(dates))
    df = pd.DataFrame({
        "date": dates,
        "long_return": returns_20d + 0.01,
        "short_return": -returns_20d * 0.3,
        "ls_return": returns_20d * 1.3 + 0.01,
        "ls_return_net": returns_20d * 1.3,
        "turnover": 0.2,
        "cost_drag": 0.0004,
        "n_long": 10,
        "n_short": 10,
    })
    df["date_key"] = df["date"].dt.date
    return df


@pytest.fixture
def monthly_returns_df(overlapping_returns_df):
    """Non-overlapping monthly subsample."""
    return subsample_monthly(overlapping_returns_df)


@pytest.fixture
def sample_regime_lookup():
    """Regime lookup covering 2020-2021."""
    dates = pd.bdate_range("2020-01-02", periods=500)
    np.random.seed(7)
    df = pd.DataFrame({
        "date": dates.date,
        "vix_percentile_252d": np.random.uniform(10, 90, len(dates)),
        "vix_level": np.random.uniform(12, 35, len(dates)),
        "market_regime": np.random.choice([-1, 0, 1], size=len(dates)),
    })
    df["market_regime_label"] = df["market_regime"].map(
        {-1: "bear", 0: "neutral", 1: "bull"}
    )
    df["vix_bucket"] = pd.cut(
        df["vix_percentile_252d"],
        bins=[-np.inf, 33, 67, np.inf],
        labels=["low", "mid", "high"],
    )
    return df.set_index("date")


# ---------------------------------------------------------------------------
# Subsampling tests
# ---------------------------------------------------------------------------

class TestSubsampleMonthly:
    def test_reduces_row_count(self, overlapping_returns_df):
        monthly = subsample_monthly(overlapping_returns_df)
        assert len(monthly) < len(overlapping_returns_df)
        n_months = overlapping_returns_df["date"].dt.to_period("M").nunique()
        assert len(monthly) == n_months

    def test_picks_first_day_per_month(self, overlapping_returns_df):
        monthly = subsample_monthly(overlapping_returns_df)
        for _, row in monthly.iterrows():
            month_mask = overlapping_returns_df["date"].dt.to_period("M") == row["date"].to_period("M")
            first_date = overlapping_returns_df.loc[month_mask, "date"].min()
            assert row["date"] == first_date

    def test_preserves_columns(self, overlapping_returns_df):
        monthly = subsample_monthly(overlapping_returns_df)
        for col in ["date", "ls_return_net", "turnover", "date_key"]:
            assert col in monthly.columns


# ---------------------------------------------------------------------------
# Metric computation tests
# ---------------------------------------------------------------------------

class TestComputePortfolioMetrics:
    def test_positive_returns_give_positive_sharpe(self):
        returns = pd.Series(np.full(24, 0.03))
        m = compute_portfolio_metrics(returns, "pos")
        assert m["ann_sharpe"] > 0
        assert m["ann_return"] > 0
        assert m["hit_rate"] == 1.0

    def test_negative_returns_give_negative_sharpe(self):
        returns = pd.Series(np.full(24, -0.02))
        m = compute_portfolio_metrics(returns, "neg")
        assert m["ann_sharpe"] < 0
        assert m["ann_return"] < 0
        assert m["hit_rate"] == 0.0

    def test_annualization_uses_12(self):
        np.random.seed(0)
        returns = pd.Series(np.random.normal(0.03, 0.05, 60))
        m = compute_portfolio_metrics(returns, "test")
        expected_ann_ret = returns.mean() * 12
        assert abs(m["ann_return"] - expected_ann_ret) < 1e-10

    def test_realistic_sharpe_range(self):
        """Monthly returns of ~3% ± 8% should give Sharpe in ~1-2 range."""
        np.random.seed(123)
        returns = pd.Series(np.random.normal(0.03, 0.08, 100))
        m = compute_portfolio_metrics(returns, "realistic")
        assert 0 < m["ann_sharpe"] < 5

    def test_max_drawdown_is_correct(self):
        returns = pd.Series([0.10, 0.05, -0.20, -0.10, 0.30])
        m = compute_portfolio_metrics(returns, "dd")
        cum = (1 + returns).cumprod()
        expected_dd = ((cum - cum.cummax()) / cum.cummax()).min()
        assert abs(m["max_drawdown"] - expected_dd) < 1e-8

    def test_short_series_returns_nan(self):
        returns = pd.Series([0.01, 0.02])
        m = compute_portfolio_metrics(returns, "short")
        assert np.isnan(m["ann_sharpe"])

    def test_zero_vol_returns_zero_sharpe(self):
        returns = pd.Series(np.zeros(20))
        m = compute_portfolio_metrics(returns, "flat")
        assert m["ann_sharpe"] == 0.0


class TestAnnotateWithRegime:
    def test_columns_added(self, monthly_returns_df, sample_regime_lookup):
        annotated = annotate_with_regime(monthly_returns_df, sample_regime_lookup)
        assert "vix_bucket" in annotated.columns
        assert "market_regime_label" in annotated.columns

    def test_all_dates_matched(self, monthly_returns_df, sample_regime_lookup):
        annotated = annotate_with_regime(monthly_returns_df, sample_regime_lookup)
        assert annotated["vix_bucket"].notna().all()


class TestFindWorstDrawdowns:
    def test_returns_n_drawdowns(self, monthly_returns_df):
        dd = find_worst_drawdowns(monthly_returns_df, n=3)
        assert len(dd) <= 3
        if not dd.empty:
            assert all(dd["max_dd"] <= 0)

    def test_drawdowns_sorted_ascending(self, monthly_returns_df):
        dd = find_worst_drawdowns(monthly_returns_df, n=5)
        if len(dd) > 1:
            assert dd["max_dd"].is_monotonic_increasing

    def test_duration_positive(self, monthly_returns_df):
        dd = find_worst_drawdowns(monthly_returns_df, n=5)
        if not dd.empty:
            assert (dd["duration_months"] > 0).all()


class TestComputeRollingSharpe:
    def test_output_length(self, monthly_returns_df):
        rs = compute_rolling_sharpe(monthly_returns_df["ls_return_net"], window=12)
        expected_len = max(0, len(monthly_returns_df) - 12 + 1)
        assert len(rs) == expected_len

    def test_positive_mean_returns_positive_sharpe(self):
        np.random.seed(0)
        returns = pd.Series(np.random.normal(0.03, 0.05, 36))
        rs = compute_rolling_sharpe(returns, window=12)
        assert rs.mean() > 0


# ---------------------------------------------------------------------------
# Integration test
# ---------------------------------------------------------------------------

class TestRunStressTests:
    def test_full_pipeline(self, overlapping_returns_df, sample_regime_lookup, tmp_path):
        csv_path = tmp_path / "returns.csv"
        overlapping_returns_df.drop(columns=["date_key"]).to_csv(csv_path, index=False)

        out_dir = tmp_path / "out"
        run_stress_tests(
            {"test_model": str(csv_path)},
            sample_regime_lookup,
            out_dir,
        )

        assert (out_dir / "regime_shadow_metrics.csv").exists()
        assert (out_dir / "rolling_sharpe.csv").exists()
        assert (out_dir / "worst_drawdowns.csv").exists()
        assert (out_dir / "regime_stress_report.json").exists()

        metrics = pd.read_csv(out_dir / "regime_shadow_metrics.csv")
        assert len(metrics) == 7  # overall + 3 vix + 3 market
        assert set(metrics["regime_axis"].unique()) == {"overall", "vix_bucket", "market_regime"}

        with open(out_dir / "regime_stress_report.json") as f:
            report = json.load(f)
        assert "methodology" in report
        assert "monthly" in report["methodology"]["return_frequency"].lower()
        assert "test_model" in report["rolling_sharpe_summary"]


# ---------------------------------------------------------------------------
# Output artifact validation (real data)
# ---------------------------------------------------------------------------

class TestOutputArtifacts:
    OUTPUT_DIR = Path("evaluation_outputs/chapter12")

    @pytest.mark.skipif(
        not (OUTPUT_DIR / "regime_shadow_metrics.csv").exists(),
        reason="Real outputs not present",
    )
    def test_metrics_csv_structure(self):
        df = pd.read_csv(self.OUTPUT_DIR / "regime_shadow_metrics.csv")
        required = {"model", "regime_axis", "bucket", "ann_sharpe", "ann_return", "max_drawdown"}
        assert required.issubset(set(df.columns))
        assert set(df["regime_axis"].unique()) == {"overall", "vix_bucket", "market_regime"}
        assert df["n_months"].min() > 0

    @pytest.mark.skipif(
        not (OUTPUT_DIR / "regime_shadow_metrics.csv").exists(),
        reason="Real outputs not present",
    )
    def test_sharpe_ratios_realistic(self):
        """Overall Sharpe ratios should be below 5 (sanity check).

        Regime-sliced buckets with <10 months can legitimately exceed
        this due to small-sample noise, so we only check overall.
        """
        df = pd.read_csv(self.OUTPUT_DIR / "regime_shadow_metrics.csv")
        overall = df[df["regime_axis"] == "overall"]["ann_sharpe"].dropna()
        assert overall.max() < 5.0, f"Overall Sharpe {overall.max():.2f} is unrealistically high"
        assert overall.min() > -5.0, f"Overall Sharpe {overall.min():.2f} is unrealistically low"

    @pytest.mark.skipif(
        not (OUTPUT_DIR / "worst_drawdowns.csv").exists(),
        reason="Real outputs not present",
    )
    def test_drawdowns_have_regime_context(self):
        df = pd.read_csv(self.OUTPUT_DIR / "worst_drawdowns.csv")
        assert "vix_bucket" in df.columns
        assert "market_regime_label" in df.columns
        assert df["max_dd"].max() <= 0

    @pytest.mark.skipif(
        not (OUTPUT_DIR / "regime_stress_report.json").exists(),
        reason="Real outputs not present",
    )
    def test_report_json_complete(self):
        with open(self.OUTPUT_DIR / "regime_stress_report.json") as f:
            report = json.load(f)
        assert "methodology" in report
        assert "regime_metrics" in report
        assert "rolling_sharpe_summary" in report
        assert "worst_drawdowns" in report
        for model in ["LGB_baseline", "Rank_Avg_2", "Learned_Stacking"]:
            assert model in report["rolling_sharpe_summary"]

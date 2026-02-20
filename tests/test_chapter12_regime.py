"""
Tests for Chapter 12.1 — Regime-Conditional Performance Diagnostics.
"""

import numpy as np
import pandas as pd
import pytest
from datetime import date

from scripts.run_chapter12_regime_diagnostics import (
    compute_per_date_rankic,
    compute_regime_metrics,
    load_regime_data,
    REGIME_AXES,
)


@pytest.fixture
def synthetic_eval_rows():
    """Create synthetic eval_rows with known RankIC patterns."""
    rng = np.random.RandomState(42)
    dates = pd.bdate_range("2020-01-01", periods=60, freq="B")
    tickers = [f"T{i:02d}" for i in range(20)]
    rows = []
    for d in dates:
        for t in tickers:
            score = rng.randn()
            noise = rng.randn() * 0.5
            rows.append({
                "as_of_date": d.date(),
                "ticker": t,
                "stable_id": t,
                "score": score,
                "excess_return": score * 0.3 + noise,
                "horizon": 20,
                "fold_id": 0,
            })
    return pd.DataFrame(rows)


class TestComputePerDateRankIC:
    def test_returns_dataframe(self, synthetic_eval_rows):
        result = compute_per_date_rankic(synthetic_eval_rows, 20)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 60

    def test_columns_present(self, synthetic_eval_rows):
        result = compute_per_date_rankic(synthetic_eval_rows, 20)
        for col in ["as_of_date", "horizon", "rankic", "n_stocks", "top10_mean_er"]:
            assert col in result.columns

    def test_positive_rankic_for_correlated_data(self, synthetic_eval_rows):
        result = compute_per_date_rankic(synthetic_eval_rows, 20)
        assert result["rankic"].mean() > 0

    def test_empty_for_missing_horizon(self, synthetic_eval_rows):
        result = compute_per_date_rankic(synthetic_eval_rows, 90)
        assert len(result) == 0


class TestComputeRegimeMetrics:
    def test_all_dates_selected(self, synthetic_eval_rows):
        per_date = compute_per_date_rankic(synthetic_eval_rows, 20)
        mask = pd.Series(True, index=per_date.index)
        metrics = compute_regime_metrics(per_date, mask, "ALL")
        assert metrics["n_dates"] == 60
        assert not np.isnan(metrics["mean_rankic"])

    def test_subset_fewer_dates(self, synthetic_eval_rows):
        per_date = compute_per_date_rankic(synthetic_eval_rows, 20)
        mask = pd.Series(False, index=per_date.index)
        mask.iloc[:10] = True
        metrics = compute_regime_metrics(per_date, mask, "subset")
        assert metrics["n_dates"] == 10

    def test_empty_subset_returns_nan(self, synthetic_eval_rows):
        per_date = compute_per_date_rankic(synthetic_eval_rows, 20)
        mask = pd.Series(False, index=per_date.index)
        metrics = compute_regime_metrics(per_date, mask, "empty")
        assert metrics["n_dates"] == 0
        assert np.isnan(metrics["mean_rankic"])

    def test_ic_stability_positive_for_good_signal(self, synthetic_eval_rows):
        per_date = compute_per_date_rankic(synthetic_eval_rows, 20)
        mask = pd.Series(True, index=per_date.index)
        metrics = compute_regime_metrics(per_date, mask, "ALL")
        assert metrics["ic_stability"] > 0

    def test_cost_survival_bounded(self, synthetic_eval_rows):
        per_date = compute_per_date_rankic(synthetic_eval_rows, 20)
        mask = pd.Series(True, index=per_date.index)
        metrics = compute_regime_metrics(per_date, mask, "ALL")
        assert 0 <= metrics["cost_survival"] <= 1

    def test_pct_positive_bounded(self, synthetic_eval_rows):
        per_date = compute_per_date_rankic(synthetic_eval_rows, 20)
        mask = pd.Series(True, index=per_date.index)
        metrics = compute_regime_metrics(per_date, mask, "ALL")
        assert 0 <= metrics["pct_positive"] <= 1


class TestRegimeAxes:
    def test_vix_percentile_buckets_exhaustive(self):
        vals = pd.Series([10, 33, 34, 50, 67, 68, 90])
        low = REGIME_AXES["vix_percentile_252d"]["low"](vals)
        mid = REGIME_AXES["vix_percentile_252d"]["mid"](vals)
        high = REGIME_AXES["vix_percentile_252d"]["high"](vals)
        assert (low | mid | high).all()

    def test_market_regime_buckets(self):
        vals = pd.Series(["bull", "bear", "neutral", "bull"])
        bull = REGIME_AXES["market_regime_label"]["bull"](vals)
        bear = REGIME_AXES["market_regime_label"]["bear"](vals)
        neutral = REGIME_AXES["market_regime_label"]["neutral"](vals)
        assert bull.sum() == 2
        assert bear.sum() == 1
        assert neutral.sum() == 1


class TestLoadRegimeData:
    def test_loads_and_decodes(self):
        df = load_regime_data("data/features.duckdb")
        assert "market_regime_label" in df.columns
        assert set(df["market_regime_label"].dropna().unique()) <= {"bull", "bear", "neutral"}
        assert len(df) > 2000

    def test_date_column_is_python_date(self):
        df = load_regime_data("data/features.duckdb")
        assert isinstance(df["date"].iloc[0], date)


class TestOutputArtifacts:
    """Verify the actual FULL-mode run produced valid output files."""

    def test_diagnostics_csv_exists(self):
        path = "evaluation_outputs/chapter12/regime_diagnostics.csv"
        df = pd.read_csv(path)
        assert len(df) > 0
        for col in ["model", "regime_axis", "bucket", "horizon", "mean_rankic"]:
            assert col in df.columns

    def test_all_models_present(self):
        df = pd.read_csv("evaluation_outputs/chapter12/regime_diagnostics.csv")
        assert set(df["model"].unique()) >= {"lgb_baseline"}

    def test_all_horizons_present(self):
        df = pd.read_csv("evaluation_outputs/chapter12/regime_diagnostics.csv")
        assert set(df["horizon"].unique()) == {20, 60, 90}

    def test_correlation_json_exists(self):
        import json
        with open("evaluation_outputs/chapter12/rankic_vs_vix_correlation.json") as f:
            data = json.load(f)
        assert "lgb_baseline" in data
        assert "90d_rankic_vs_vix_pctile" in data["lgb_baseline"]

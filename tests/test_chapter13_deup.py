"""Tests for Chapter 13.0 and 13.1 — residual archive + g(x) training."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_chapter13_deup import compute_rank_loss, enrich_with_regime_context
from src.uncertainty.deup_estimator import (
    G_FEATURES,
    prepare_g_features,
    train_g_walk_forward,
    _available_features,
)


# ── Fixtures ─────────────────────────────────────────────────────────────

@pytest.fixture
def sample_eval_rows():
    """Minimal eval_rows-like DataFrame."""
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=5, freq="B")
    tickers = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN"]
    rows = []
    for d in dates:
        for t in tickers:
            rows.append({
                "as_of_date": d,
                "ticker": t,
                "stable_id": f"STABLE_{t}",
                "horizon": 20,
                "fold_id": f"fold_{d.day:02d}",
                "score": np.random.randn() * 0.03,
                "excess_return": np.random.randn() * 0.10,
                "adv_20d": np.random.uniform(1e8, 1e10),
            })
    return pd.DataFrame(rows)


@pytest.fixture
def sample_regime_context():
    """Minimal regime_context-like DataFrame."""
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=5, freq="B")
    tickers = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN"]
    rows = []
    for d in dates:
        for t in tickers:
            rows.append({
                "date": d,
                "stable_id": f"STABLE_{t}",
                "ticker": t,
                "vol_20d": np.random.uniform(0.1, 0.5),
                "vol_60d": np.random.uniform(0.1, 0.5),
                "vol_of_vol": np.random.uniform(0.01, 0.1),
                "mom_1m": np.random.randn() * 0.05,
                "vix_percentile_252d": np.random.uniform(10, 90),
                "market_regime": np.random.choice([-1, 0, 1]),
                "market_vol_21d": np.random.uniform(0.1, 0.3),
                "market_return_21d": np.random.randn() * 0.03,
            })
    return pd.DataFrame(rows)


@pytest.fixture
def enriched_residuals(sample_eval_rows, sample_regime_context):
    """Full enriched residual DataFrame ready for g(x)."""
    np.random.seed(42)
    n_folds = 25
    dates_per_fold = 5
    tickers = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN"]
    all_rows = []
    for fold_i in range(n_folds):
        fold_id = f"fold_{fold_i + 1:03d}"
        base_date = pd.Timestamp("2018-01-01") + pd.Timedelta(days=fold_i * 30)
        for d_off in range(dates_per_fold):
            d = base_date + pd.Timedelta(days=d_off)
            for t in tickers:
                all_rows.append({
                    "as_of_date": d,
                    "ticker": t,
                    "stable_id": f"STABLE_{t}",
                    "horizon": 20,
                    "fold_id": fold_id,
                    "score": np.random.randn() * 0.03,
                    "excess_return": np.random.randn() * 0.10,
                    "adv_20d": np.random.uniform(1e8, 1e10),
                    "vol_20d": np.random.uniform(0.1, 0.5),
                    "vol_60d": np.random.uniform(0.1, 0.5),
                    "mom_1m": np.random.randn() * 0.05,
                    "vix_percentile_252d": np.random.uniform(10, 90),
                    "market_regime": np.random.choice([-1, 0, 1]),
                    "market_vol_21d": np.random.uniform(0.1, 0.3),
                    "market_return_21d": np.random.randn() * 0.03,
                })
    df = pd.DataFrame(all_rows)
    df = compute_rank_loss(df)
    return df


# ── 13.0 Tests ───────────────────────────────────────────────────────────

class TestComputeRankLoss:
    def test_adds_rank_loss_column(self, sample_eval_rows):
        result = compute_rank_loss(sample_eval_rows)
        assert "rank_loss" in result.columns
        assert "mae_loss" in result.columns

    def test_rank_loss_in_valid_range(self, sample_eval_rows):
        result = compute_rank_loss(sample_eval_rows)
        assert (result["rank_loss"] >= 0).all()
        assert (result["rank_loss"] <= 1).all()

    def test_mae_loss_non_negative(self, sample_eval_rows):
        result = compute_rank_loss(sample_eval_rows)
        assert (result["mae_loss"] >= 0).all()

    def test_preserves_row_count(self, sample_eval_rows):
        result = compute_rank_loss(sample_eval_rows)
        assert len(result) == len(sample_eval_rows)

    def test_rank_loss_is_zero_for_perfect_model(self):
        df = pd.DataFrame({
            "as_of_date": ["2020-01-01"] * 5,
            "ticker": ["A", "B", "C", "D", "E"],
            "horizon": [20] * 5,
            "score": [1.0, 2.0, 3.0, 4.0, 5.0],
            "excess_return": [0.01, 0.02, 0.03, 0.04, 0.05],
        })
        result = compute_rank_loss(df)
        assert result["rank_loss"].max() < 0.01

    def test_rank_loss_high_for_reversed_model(self):
        df = pd.DataFrame({
            "as_of_date": ["2020-01-01"] * 5,
            "ticker": ["A", "B", "C", "D", "E"],
            "horizon": [20] * 5,
            "score": [5.0, 4.0, 3.0, 2.0, 1.0],
            "excess_return": [0.01, 0.02, 0.03, 0.04, 0.05],
        })
        result = compute_rank_loss(df)
        assert result["rank_loss"].mean() > 0.3

    def test_per_date_computation(self):
        df = pd.DataFrame({
            "as_of_date": ["2020-01-01"] * 3 + ["2020-01-02"] * 3,
            "ticker": ["A", "B", "C"] * 2,
            "horizon": [20] * 6,
            "score": [1, 2, 3, 3, 2, 1],
            "excess_return": [0.01, 0.02, 0.03, 0.01, 0.02, 0.03],
        })
        result = compute_rank_loss(df)
        day1 = result[result["as_of_date"] == "2020-01-01"]
        day2 = result[result["as_of_date"] == "2020-01-02"]
        assert day1["rank_loss"].mean() < day2["rank_loss"].mean()


class TestEnrichWithRegimeContext:
    def test_adds_regime_columns(self, sample_eval_rows, sample_regime_context):
        result = enrich_with_regime_context(sample_eval_rows, sample_regime_context)
        assert "vol_20d" in result.columns
        assert "market_regime" in result.columns

    def test_preserves_row_count(self, sample_eval_rows, sample_regime_context):
        result = enrich_with_regime_context(sample_eval_rows, sample_regime_context)
        assert len(result) == len(sample_eval_rows)


# ── 13.1 Tests ───────────────────────────────────────────────────────────

class TestPrepareGFeatures:
    def test_adds_derived_features(self, sample_eval_rows):
        result = prepare_g_features(sample_eval_rows)
        assert "abs_score" in result.columns
        assert "cross_sectional_rank" in result.columns

    def test_abs_score_non_negative(self, sample_eval_rows):
        result = prepare_g_features(sample_eval_rows)
        assert (result["abs_score"] >= 0).all()

    def test_cross_sectional_rank_in_01(self, sample_eval_rows):
        result = prepare_g_features(sample_eval_rows)
        assert (result["cross_sectional_rank"] >= 0).all()
        assert (result["cross_sectional_rank"] <= 1).all()


class TestTrainGWalkForward:
    def test_produces_predictions(self, enriched_residuals):
        preds, diag = train_g_walk_forward(
            enriched_residuals,
            target_col="rank_loss",
            min_train_folds=20,
            horizons=[20],
        )
        assert len(preds) > 0
        assert "g_pred" in preds.columns
        assert "rank_loss" in preds.columns

    def test_g_pred_non_negative(self, enriched_residuals):
        preds, _ = train_g_walk_forward(
            enriched_residuals,
            target_col="rank_loss",
            min_train_folds=20,
            horizons=[20],
        )
        assert (preds["g_pred"] >= 0).all()

    def test_oos_only(self, enriched_residuals):
        """g(x) should not predict on training folds."""
        preds, _ = train_g_walk_forward(
            enriched_residuals,
            target_col="rank_loss",
            min_train_folds=20,
            horizons=[20],
        )
        all_folds = sorted(enriched_residuals["fold_id"].unique())
        train_only_folds = set(all_folds[:20])
        predicted_folds = set(preds["fold_id"].unique())
        assert predicted_folds.isdisjoint(train_only_folds), (
            "g(x) predicted on training-only folds!"
        )

    def test_diagnostics_contain_rho(self, enriched_residuals):
        _, diag = train_g_walk_forward(
            enriched_residuals,
            target_col="rank_loss",
            min_train_folds=20,
            horizons=[20],
        )
        assert 20 in diag
        assert "spearman_rho" in diag[20]

    def test_feature_importances_returned(self, enriched_residuals):
        _, diag = train_g_walk_forward(
            enriched_residuals,
            target_col="rank_loss",
            min_train_folds=20,
            horizons=[20],
        )
        assert "feature_importances" in diag
        assert len(diag["feature_importances"]) > 0

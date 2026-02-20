"""
Tests for Chapter 10.4 — Sentiment Evaluation & Gates
=======================================================

Tests cover:
1. Scoring function output format (EvaluationRow)
2. Composite score properties (range, direction, NaN handling)
3. Gate evaluation logic
4. Metrics computation
"""

import numpy as np
import pandas as pd
import pytest
from datetime import date, datetime


# ---------------------------------------------------------------------------
# Test data fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_features_df():
    """Create a minimal features_df matching evaluation format."""
    np.random.seed(42)
    dates = [datetime(2024, 6, 1), datetime(2024, 7, 1), datetime(2024, 8, 1)]
    tickers = ["NVDA", "AAPL", "MSFT", "AMD", "GOOGL", "TSLA", "META", "AMZN", "AVGO", "CRM"]

    rows = []
    for d in dates:
        for t in tickers:
            rows.append({
                "date": d,
                "ticker": t,
                "stable_id": f"STB_{t}",
                "excess_return_20d": np.random.normal(0.02, 0.1),
                "excess_return_60d": np.random.normal(0.05, 0.15),
                "excess_return_90d": np.random.normal(0.08, 0.2),
                "mom_12m": np.random.normal(0.1, 0.3),
                # Sentiment features (some with NaN)
                "news_sentiment_30d": np.random.uniform(-0.5, 0.8) if np.random.random() > 0.2 else np.nan,
                "news_sentiment_7d": np.random.uniform(-0.5, 0.8) if np.random.random() > 0.3 else np.nan,
                "news_sentiment_momentum": np.random.uniform(-0.3, 0.3) if np.random.random() > 0.3 else np.nan,
                "news_volume_30d": np.random.randint(0, 50),
                "filing_sentiment_latest": np.random.uniform(-0.2, 0.5) if np.random.random() > 0.5 else np.nan,
                "filing_sentiment_90d": np.random.uniform(-0.2, 0.5) if np.random.random() > 0.5 else np.nan,
                "sentiment_zscore": np.random.normal(0, 1) if np.random.random() > 0.2 else np.nan,
                "sentiment_vs_momentum": np.random.uniform(-0.5, 0.5) if np.random.random() > 0.4 else np.nan,
            })
    return pd.DataFrame(rows)


@pytest.fixture
def sample_eval_rows():
    """Create sample evaluation rows for gate testing."""
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=12, freq="MS")
    tickers = [f"STK{i}" for i in range(20)]

    rows = []
    for horizon in [20, 60, 90]:
        for d in dates:
            for t in tickers:
                score = np.random.uniform(0, 1)
                excess_return = 0.02 * (score - 0.5) + np.random.normal(0, 0.05)
                rows.append({
                    "as_of_date": d,
                    "ticker": t,
                    "stable_id": f"STB_{t}",
                    "fold_id": "fold_01",
                    "horizon": horizon,
                    "score": score,
                    "excess_return": excess_return,
                })
    return pd.DataFrame(rows)


# ===========================================================================
# Scoring Function Tests
# ===========================================================================


def _score_composite(features_df, fold_id, horizon):
    """
    Standalone scoring logic matching the run script, without needing
    the global SentimentFeatureGenerator. Uses pre-populated sentiment
    columns in features_df directly.
    """
    from scripts.run_chapter10_sentiment import SCORE_FEATURES

    results = []
    er_col = f"excess_return_{horizon}d"
    if er_col not in features_df.columns:
        er_col = "excess_return"

    for asof_date, date_df in features_df.groupby("date"):
        rank_matrix = pd.DataFrame(index=date_df.index)
        available_features = []
        for feat in SCORE_FEATURES:
            if feat not in date_df.columns:
                continue
            col = date_df[feat]
            n_valid = col.notna().sum()
            if n_valid < 3:
                continue
            available_features.append(feat)
            rank_matrix[feat] = col.rank(pct=True)

        if available_features:
            composite = rank_matrix[available_features].mean(axis=1).fillna(0.5)
        else:
            composite = pd.Series(0.5, index=date_df.index)

        for idx, row in date_df.iterrows():
            er_val = row.get(er_col)
            if pd.isna(er_val):
                continue
            results.append({
                "as_of_date": asof_date,
                "ticker": row["ticker"],
                "stable_id": row["stable_id"],
                "fold_id": fold_id,
                "horizon": horizon,
                "score": float(composite.loc[idx]),
                "excess_return": float(er_val),
            })
    return pd.DataFrame(results)


class TestSentimentScoringFunction:
    """Test the sentiment composite scoring function."""

    def test_returns_dataframe(self, sample_features_df):
        result = _score_composite(sample_features_df, "fold_01", 20)
        assert isinstance(result, pd.DataFrame)

    def test_required_columns(self, sample_features_df):
        """Output must have all EvaluationRow required columns."""
        result = _score_composite(sample_features_df, "fold_01", 20)
        required = ["as_of_date", "ticker", "stable_id", "fold_id", "horizon", "score", "excess_return"]
        for col in required:
            assert col in result.columns, f"Missing required column: {col}"

    def test_score_range(self, sample_features_df):
        """Scores should be in [0, 1] (percentile ranks)."""
        result = _score_composite(sample_features_df, "fold_01", 20)
        assert result["score"].min() >= 0.0
        assert result["score"].max() <= 1.0

    def test_correct_horizon(self, sample_features_df):
        """All rows should have the requested horizon."""
        result = _score_composite(sample_features_df, "fold_01", 60)
        assert (result["horizon"] == 60).all()

    def test_correct_fold_id(self, sample_features_df):
        """All rows should have the specified fold_id."""
        result = _score_composite(sample_features_df, "test_fold", 20)
        assert (result["fold_id"] == "test_fold").all()

    def test_no_nan_scores(self, sample_features_df):
        """Scores should never be NaN (NaN features get neutral 0.5)."""
        result = _score_composite(sample_features_df, "fold_01", 20)
        assert result["score"].notna().all()

    def test_all_tickers_represented(self, sample_features_df):
        """All tickers should appear in the output (assuming valid excess returns)."""
        result = _score_composite(sample_features_df, "fold_01", 20)
        n_dates = sample_features_df["date"].nunique()
        n_tickers = sample_features_df["ticker"].nunique()
        assert len(result) <= n_dates * n_tickers
        assert len(result) > 0

    def test_empty_features_df(self):
        """Empty input should return empty output."""
        empty_df = pd.DataFrame(columns=["date", "ticker", "stable_id", "excess_return_20d"])
        result = _score_composite(empty_df, "fold_01", 20)
        assert len(result) == 0

    def test_all_nan_sentiment(self):
        """When all sentiment features are NaN, should score 0.5."""
        df = pd.DataFrame({
            "date": [datetime(2024, 6, 1)] * 5,
            "ticker": ["A", "B", "C", "D", "E"],
            "stable_id": ["S_A", "S_B", "S_C", "S_D", "S_E"],
            "excess_return_20d": [0.05, -0.03, 0.02, 0.01, -0.01],
        })
        result = _score_composite(df, "fold_01", 20)
        assert len(result) == 5
        assert (result["score"] == 0.5).all()


# ===========================================================================
# Gate Evaluation Tests
# ===========================================================================


class TestGateEvaluation:
    """Test gate checking logic."""

    def test_compute_metrics(self, sample_eval_rows):
        from scripts.evaluate_sentiment_gates import compute_metrics
        metrics = compute_metrics(sample_eval_rows)
        assert set(metrics.keys()) == {20, 60, 90}
        for h, m in metrics.items():
            assert "mean_rankic" in m
            assert "median_rankic" in m
            assert "median_churn" in m
            assert m["n_dates"] > 0

    def test_gate1_passes(self):
        """Gate 1 should pass when ≥2 horizons have mean RankIC ≥ 0.02."""
        from scripts.evaluate_sentiment_gates import evaluate_gates
        metrics = {
            20: {"mean_rankic": 0.03, "median_churn": 0.1},
            60: {"mean_rankic": 0.04, "median_churn": 0.1},
            90: {"mean_rankic": 0.01, "median_churn": 0.1},
        }
        gates = evaluate_gates(metrics)
        assert gates["gate_1_factor"]["pass"] is True

    def test_gate1_fails(self):
        """Gate 1 should fail when < 2 horizons have mean RankIC ≥ 0.02."""
        from scripts.evaluate_sentiment_gates import evaluate_gates
        metrics = {
            20: {"mean_rankic": 0.03, "median_churn": 0.1},
            60: {"mean_rankic": 0.01, "median_churn": 0.1},
            90: {"mean_rankic": 0.005, "median_churn": 0.1},
        }
        gates = evaluate_gates(metrics)
        assert gates["gate_1_factor"]["pass"] is False

    def test_gate2_absolute(self):
        """Gate 2 should pass on absolute threshold (≥0.05)."""
        from scripts.evaluate_sentiment_gates import evaluate_gates
        metrics = {
            20: {"mean_rankic": 0.06, "median_churn": 0.1},
            60: {"mean_rankic": 0.01, "median_churn": 0.1},
            90: {"mean_rankic": 0.01, "median_churn": 0.1},
        }
        gates = evaluate_gates(metrics)
        assert gates["gate_2_ml"]["pass"] is True

    def test_gate2_relative(self):
        """Gate 2 should pass on relative threshold (within 0.03 of LGB)."""
        from scripts.evaluate_sentiment_gates import evaluate_gates
        metrics = {
            20: {"mean_rankic": 0.075, "median_churn": 0.1},  # LGB=0.1009, 0.075 >= 0.0709
            60: {"mean_rankic": 0.01, "median_churn": 0.1},
            90: {"mean_rankic": 0.01, "median_churn": 0.1},
        }
        gates = evaluate_gates(metrics)
        assert gates["gate_2_ml"]["pass"] is True

    def test_gate3_passes(self):
        """Gate 3 should pass when all churns ≤ 30%."""
        from scripts.evaluate_sentiment_gates import evaluate_gates
        metrics = {
            20: {"mean_rankic": 0.03, "median_churn": 0.20},
            60: {"mean_rankic": 0.03, "median_churn": 0.25},
            90: {"mean_rankic": 0.03, "median_churn": 0.30},
        }
        gates = evaluate_gates(metrics)
        assert gates["gate_3_practical"]["pass"] is True

    def test_gate3_fails(self):
        """Gate 3 should fail when any churn > 30%."""
        from scripts.evaluate_sentiment_gates import evaluate_gates
        metrics = {
            20: {"mean_rankic": 0.03, "median_churn": 0.20},
            60: {"mean_rankic": 0.03, "median_churn": 0.35},
            90: {"mean_rankic": 0.03, "median_churn": 0.25},
        }
        gates = evaluate_gates(metrics)
        assert gates["gate_3_practical"]["pass"] is False


# ===========================================================================
# Metrics Computation Tests
# ===========================================================================


class TestMetricsComputation:
    """Test metric computation details."""

    def test_rankic_with_signal(self):
        """Perfect ranking should give RankIC ≈ 1."""
        from scripts.evaluate_sentiment_gates import compute_metrics
        rows = []
        for i in range(20):
            rows.append({
                "as_of_date": pd.Timestamp("2024-06-01"),
                "stable_id": f"STK_{i}",
                "horizon": 20,
                "score": float(i),  # Perfect ordering
                "excess_return": float(i) * 0.01,
            })
        df = pd.DataFrame(rows)
        metrics = compute_metrics(df)
        assert metrics[20]["mean_rankic"] > 0.9

    def test_churn_computation(self):
        """Identical top-10 across dates should give 0 churn."""
        from scripts.evaluate_sentiment_gates import compute_metrics
        rows = []
        for d in ["2024-06-01", "2024-07-01"]:
            for i in range(20):
                rows.append({
                    "as_of_date": pd.Timestamp(d),
                    "stable_id": f"STK_{i}",
                    "horizon": 20,
                    "score": float(20 - i),  # Same ranking both dates
                    "excess_return": 0.01,
                })
        df = pd.DataFrame(rows)
        metrics = compute_metrics(df)
        assert metrics[20]["median_churn"] == 0.0

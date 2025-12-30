"""
Tests for ML Baselines (Chapter 7.3)

Tests the tabular_lgb baseline implementation:
1. Registration and listing
2. Training and prediction
3. Determinism (same inputs -> same outputs)
4. Integration with evaluation pipeline
5. Guardrails (train/val split enforcement)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import pandas as pd
import numpy as np
from datetime import date, timedelta

from src.evaluation.baselines import (
    BASELINE_REGISTRY,
    ML_BASELINES,
    list_baselines,
    generate_ml_baseline_scores,
    train_lgbm_ranking_model,
    predict_lgbm_scores,
    _compute_time_decay_weights,
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def synthetic_train_data():
    """Create synthetic training data for ML baseline tests."""
    np.random.seed(42)
    n_dates = 20
    n_tickers_per_date = 50
    
    rows = []
    for i in range(n_dates):
        d = date(2023, 1, 1) + timedelta(days=i*7)  # Weekly
        for j in range(n_tickers_per_date):
            ticker = f"TICK{j:03d}"
            stable_id = f"stable_{ticker}"
            
            # Generate correlated features and label
            mom_12m = np.random.randn()
            mom_6m = mom_12m * 0.7 + np.random.randn() * 0.3
            mom_3m = mom_6m * 0.7 + np.random.randn() * 0.3
            mom_1m = mom_3m * 0.7 + np.random.randn() * 0.3
            
            # Label is correlated with momentum
            excess_return = mom_12m * 0.1 + np.random.randn() * 0.05
            
            rows.append({
                "date": d,
                "ticker": ticker,
                "stable_id": stable_id,
                "mom_1m": mom_1m,
                "mom_3m": mom_3m,
                "mom_6m": mom_6m,
                "mom_12m": mom_12m,
                "vol_20d": abs(np.random.randn() * 0.2) + 0.1,
                "vol_60d": abs(np.random.randn() * 0.2) + 0.1,
                "max_drawdown_60d": -abs(np.random.randn() * 0.1),
                "adv_20d": abs(np.random.randn() * 1e6) + 1e5,
                "excess_return_20d": excess_return,
                "excess_return_60d": excess_return * 1.5,
                "excess_return_90d": excess_return * 2.0,
            })
    
    return pd.DataFrame(rows)


@pytest.fixture
def synthetic_val_data():
    """Create synthetic validation data (different dates from train)."""
    np.random.seed(123)  # Different seed for val
    n_dates = 5
    n_tickers_per_date = 50
    
    rows = []
    for i in range(n_dates):
        d = date(2023, 6, 1) + timedelta(days=i*7)  # Later dates
        for j in range(n_tickers_per_date):
            ticker = f"TICK{j:03d}"
            stable_id = f"stable_{ticker}"
            
            # Generate correlated features and label
            mom_12m = np.random.randn()
            mom_6m = mom_12m * 0.7 + np.random.randn() * 0.3
            mom_3m = mom_6m * 0.7 + np.random.randn() * 0.3
            mom_1m = mom_3m * 0.7 + np.random.randn() * 0.3
            
            # Label is correlated with momentum
            excess_return = mom_12m * 0.1 + np.random.randn() * 0.05
            
            rows.append({
                "date": d,
                "ticker": ticker,
                "stable_id": stable_id,
                "mom_1m": mom_1m,
                "mom_3m": mom_3m,
                "mom_6m": mom_6m,
                "mom_12m": mom_12m,
                "vol_20d": abs(np.random.randn() * 0.2) + 0.1,
                "vol_60d": abs(np.random.randn() * 0.2) + 0.1,
                "max_drawdown_60d": -abs(np.random.randn() * 0.1),
                "adv_20d": abs(np.random.randn() * 1e6) + 1e5,
                "excess_return_20d": excess_return,
                "excess_return_60d": excess_return * 1.5,
                "excess_return_90d": excess_return * 2.0,
            })
    
    return pd.DataFrame(rows)


# ============================================================================
# TESTS: Registration and Listing
# ============================================================================

def test_tabular_lgb_registered():
    """Test that tabular_lgb is registered in baseline registry."""
    assert "tabular_lgb" in BASELINE_REGISTRY
    assert "tabular_lgb" in ML_BASELINES
    assert "tabular_lgb" in list_baselines()


def test_tabular_lgb_definition():
    """Test that tabular_lgb has correct definition."""
    baseline = BASELINE_REGISTRY["tabular_lgb"]
    assert baseline.name == "tabular_lgb"
    assert "LightGBM" in baseline.description or "ranking" in baseline.description.lower()
    assert len(baseline.required_features) > 0  # Should have some required features


# ============================================================================
# TESTS: Time Decay Weights
# ============================================================================

def test_time_decay_weights_recent_higher():
    """Test that time decay gives higher weight to recent dates."""
    dates = pd.Series([
        date(2023, 1, 1),
        date(2023, 6, 1),
        date(2023, 12, 1),
    ])
    
    weights = _compute_time_decay_weights(dates, half_life_days=180)
    
    # Recent dates should have higher weight
    assert weights[2] > weights[1] > weights[0]
    
    # Weights should sum to len(dates) (normalized)
    assert abs(weights.sum() - len(dates)) < 1e-6


def test_time_decay_weights_deterministic():
    """Test that time decay weights are deterministic."""
    dates = pd.Series([date(2023, 1, 1) + timedelta(days=i*30) for i in range(12)])
    
    weights1 = _compute_time_decay_weights(dates)
    weights2 = _compute_time_decay_weights(dates)
    
    np.testing.assert_array_almost_equal(weights1, weights2)


# ============================================================================
# TESTS: Training and Prediction
# ============================================================================

def test_train_lgbm_model(synthetic_train_data):
    """Test that LightGBM model can be trained."""
    feature_cols = ["mom_1m", "mom_3m", "mom_6m", "mom_12m", "vol_20d", "adv_20d"]
    
    model = train_lgbm_ranking_model(
        train_df=synthetic_train_data,
        feature_cols=feature_cols,
        label_col="excess_return_20d",
        date_col="date",
        random_state=42,
        n_estimators=10,  # Small for speed
        verbose=-1
    )
    
    assert model is not None
    assert hasattr(model, 'predict')
    assert hasattr(model, 'feature_names_')
    assert len(model.feature_names_) == len(feature_cols)


def test_predict_lgbm_scores(synthetic_train_data, synthetic_val_data):
    """Test that trained model can generate predictions."""
    feature_cols = ["mom_1m", "mom_3m", "mom_6m", "mom_12m", "vol_20d", "adv_20d"]
    
    # Train
    model = train_lgbm_ranking_model(
        train_df=synthetic_train_data,
        feature_cols=feature_cols,
        label_col="excess_return_20d",
        date_col="date",
        random_state=42,
        n_estimators=10,
        verbose=-1
    )
    
    # Predict
    scores = predict_lgbm_scores(model, synthetic_val_data, feature_cols)
    
    assert len(scores) == len(synthetic_val_data)
    assert not np.isnan(scores).any()  # No NaN scores
    assert np.isfinite(scores).all()  # All finite


def test_lgbm_model_deterministic(synthetic_train_data, synthetic_val_data):
    """Test that LightGBM model is deterministic with fixed random_state."""
    feature_cols = ["mom_1m", "mom_3m", "mom_6m", "mom_12m", "vol_20d", "adv_20d"]
    
    # Train twice with same random_state
    model1 = train_lgbm_ranking_model(
        train_df=synthetic_train_data,
        feature_cols=feature_cols,
        label_col="excess_return_20d",
        random_state=42,
        n_estimators=10,
        verbose=-1
    )
    
    model2 = train_lgbm_ranking_model(
        train_df=synthetic_train_data,
        feature_cols=feature_cols,
        label_col="excess_return_20d",
        random_state=42,
        n_estimators=10,
        verbose=-1
    )
    
    # Predictions should be identical
    scores1 = predict_lgbm_scores(model1, synthetic_val_data, feature_cols)
    scores2 = predict_lgbm_scores(model2, synthetic_val_data, feature_cols)
    
    np.testing.assert_array_almost_equal(scores1, scores2, decimal=6)


# ============================================================================
# TESTS: Integration with Evaluation Pipeline
# ============================================================================

def test_generate_ml_baseline_scores(synthetic_train_data, synthetic_val_data):
    """Test that generate_ml_baseline_scores produces valid EvaluationRow format."""
    eval_rows = generate_ml_baseline_scores(
        train_df=synthetic_train_data,
        val_df=synthetic_val_data,
        baseline_name="tabular_lgb",
        fold_id="test_fold",
        horizon=20,
        excess_return_col="excess_return_20d"
    )
    
    # Check output format
    assert isinstance(eval_rows, pd.DataFrame)
    assert len(eval_rows) > 0
    
    # Check required columns
    required_cols = ["as_of_date", "ticker", "stable_id", "horizon", "fold_id", "score", "excess_return"]
    for col in required_cols:
        assert col in eval_rows.columns, f"Missing required column: {col}"
    
    # Check values
    assert (eval_rows["horizon"] == 20).all()
    assert (eval_rows["fold_id"] == "test_fold").all()
    assert not eval_rows["score"].isna().any()
    assert not eval_rows["excess_return"].isna().any()


def test_ml_baseline_no_duplicates(synthetic_train_data, synthetic_val_data):
    """Test that ML baseline doesn't produce duplicate (as_of_date, stable_id, horizon)."""
    eval_rows = generate_ml_baseline_scores(
        train_df=synthetic_train_data,
        val_df=synthetic_val_data,
        baseline_name="tabular_lgb",
        fold_id="test_fold",
        horizon=20,
        excess_return_col="excess_return_20d"
    )
    
    # Check for duplicates
    duplicate_check = eval_rows.groupby(["as_of_date", "stable_id", "horizon"]).size()
    duplicates = duplicate_check[duplicate_check > 1]
    
    assert len(duplicates) == 0, f"Found {len(duplicates)} duplicate keys"


def test_ml_baseline_deterministic_end_to_end(synthetic_train_data, synthetic_val_data):
    """Test that entire ML baseline pipeline is deterministic."""
    # Run 1
    eval_rows1 = generate_ml_baseline_scores(
        train_df=synthetic_train_data,
        val_df=synthetic_val_data,
        baseline_name="tabular_lgb",
        fold_id="test_fold",
        horizon=20,
        excess_return_col="excess_return_20d",
        lgbm_params={"random_state": 42, "n_estimators": 10, "verbose": -1}
    )
    
    # Run 2 (identical inputs)
    eval_rows2 = generate_ml_baseline_scores(
        train_df=synthetic_train_data,
        val_df=synthetic_val_data,
        baseline_name="tabular_lgb",
        fold_id="test_fold",
        horizon=20,
        excess_return_col="excess_return_20d",
        lgbm_params={"random_state": 42, "n_estimators": 10, "verbose": -1}
    )
    
    # Sort both by stable_id and date for comparison
    eval_rows1_sorted = eval_rows1.sort_values(["as_of_date", "stable_id"]).reset_index(drop=True)
    eval_rows2_sorted = eval_rows2.sort_values(["as_of_date", "stable_id"]).reset_index(drop=True)
    
    # Scores should be identical
    np.testing.assert_array_almost_equal(
        eval_rows1_sorted["score"].values,
        eval_rows2_sorted["score"].values,
        decimal=6
    )


# ============================================================================
# TESTS: Guardrails (Train/Val Split Enforcement)
# ============================================================================

def test_train_val_dates_non_overlapping(synthetic_train_data, synthetic_val_data):
    """Test that train and val data have non-overlapping dates."""
    train_dates = set(synthetic_train_data["date"].unique())
    val_dates = set(synthetic_val_data["date"].unique())
    
    overlap = train_dates & val_dates
    assert len(overlap) == 0, f"Train and val have overlapping dates: {overlap}"
    
    # Val dates should be after train dates
    max_train_date = synthetic_train_data["date"].max()
    min_val_date = synthetic_val_data["date"].min()
    assert min_val_date > max_train_date, "Val dates should be after train dates"


def test_ml_baseline_requires_train_data():
    """Test that ML baseline raises error if train_df is missing."""
    from src.evaluation.baselines import generate_baseline_scores
    
    val_df = pd.DataFrame({
        "date": [date(2023, 1, 1)],
        "ticker": ["TEST"],
        "stable_id": ["stable_TEST"],
        "mom_12m": [0.1],
        "excess_return_20d": [0.05]
    })
    
    with pytest.raises(ValueError, match="requires train_df"):
        generate_baseline_scores(
            features_df=val_df,
            baseline_name="tabular_lgb",
            fold_id="test",
            horizon=20,
            excess_return_col="excess_return_20d",
            train_df=None  # Missing train data
        )


# ============================================================================
# TESTS: Smoke Test (Integration with run_experiment)
# ============================================================================

def test_tabular_lgb_smoke_integration():
    """
    Smoke test: Run tabular_lgb with generate_baseline_scores.
    
    This tests integration with the baseline runner interface.
    """
    from src.evaluation.baselines import generate_baseline_scores
    
    # Create minimal synthetic train data
    np.random.seed(42)
    train_dates = pd.date_range("2023-01-01", periods=50, freq='D')
    
    train_rows = []
    for d in train_dates:
        for ticker in ["TICK001", "TICK002", "TICK003"]:
            train_rows.append({
                "date": d.date(),
                "ticker": ticker,
                "stable_id": f"stable_{ticker}",
                "mom_1m": np.random.randn(),
                "mom_3m": np.random.randn(),
                "mom_6m": np.random.randn(),
                "mom_12m": np.random.randn(),
                "vol_20d": abs(np.random.randn() * 0.2),
                "adv_20d": abs(np.random.randn() * 1e6),
                "excess_return_20d": np.random.randn() * 0.05,
            })
    
    train_df = pd.DataFrame(train_rows)
    
    # Create minimal synthetic val data (later dates)
    val_dates = pd.date_range("2023-03-01", periods=10, freq='D')
    
    val_rows = []
    for d in val_dates:
        for ticker in ["TICK001", "TICK002", "TICK003"]:
            val_rows.append({
                "date": d.date(),
                "ticker": ticker,
                "stable_id": f"stable_{ticker}",
                "mom_1m": np.random.randn(),
                "mom_3m": np.random.randn(),
                "mom_6m": np.random.randn(),
                "mom_12m": np.random.randn(),
                "vol_20d": abs(np.random.randn() * 0.2),
                "adv_20d": abs(np.random.randn() * 1e6),
                "excess_return_20d": np.random.randn() * 0.05,
            })
    
    val_df = pd.DataFrame(val_rows)
    
    # Test that we can generate scores using the baseline runner interface
    eval_rows = generate_baseline_scores(
        features_df=val_df,
        baseline_name="tabular_lgb",
        fold_id="smoke_test_fold",
        horizon=20,
        excess_return_col="excess_return_20d",
        train_df=train_df  # ML baselines require training data
    )
    
    assert len(eval_rows) > 0, "Should generate evaluation rows"
    assert (eval_rows["fold_id"] == "smoke_test_fold").all()
    assert (eval_rows["horizon"] == 20).all()
    assert not eval_rows["score"].isna().any()
    assert not eval_rows["excess_return"].isna().any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


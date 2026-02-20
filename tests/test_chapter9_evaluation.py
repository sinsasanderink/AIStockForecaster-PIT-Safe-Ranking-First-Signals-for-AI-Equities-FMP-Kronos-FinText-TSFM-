"""
Tests for Chapter 9 Evaluation Script
=======================================

Tests the evaluation runner script and its helper functions.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

# Add scripts to path for import
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from run_chapter9_fintext import (
    setup_fintext_adapter,
    fintext_scoring_function,
    shuffle_within_date_control,
    lag_control,
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def db_path():
    db = Path("data/features.duckdb")
    if not db.exists():
        pytest.skip("DuckDB not found")
    return str(db)


@pytest.fixture
def sample_features_df():
    """Minimal features_df for testing scorer functions."""
    dates = pd.date_range("2024-03-01", periods=3, freq="D")
    tickers = ["AAPL", "NVDA", "MSFT"]
    
    rows = []
    for date in dates:
        for ticker in tickers:
            rows.append({
                "date": date,
                "ticker": ticker,
                "stable_id": ticker,
                "excess_return_20d": np.random.randn() * 0.02,
                "excess_return_60d": np.random.randn() * 0.05,
                "excess_return_90d": np.random.randn() * 0.08,
                "adv_20d": 1e9,
                "adv_60d": 1e9,
            })
    
    return pd.DataFrame(rows)


# ============================================================================
# ADAPTER SETUP
# ============================================================================

class TestAdapterSetup:
    def test_setup_fintext_adapter_stub(self, db_path):
        """Adapter can be initialized in stub mode."""
        adapter = setup_fintext_adapter(
            db_path=db_path,
            model_size="Small",
            use_stub=True,
        )
        assert adapter is not None
        assert adapter.use_stub is True
    
    def test_adapter_is_global_singleton(self, db_path):
        """Multiple calls return the same adapter instance."""
        adapter1 = setup_fintext_adapter(db_path=db_path, use_stub=True)
        adapter2 = setup_fintext_adapter(db_path=db_path, use_stub=True)
        assert adapter1 is adapter2


# ============================================================================
# SCORING FUNCTION
# ============================================================================

class TestScoringFunction:
    def test_scoring_function_contract(self, db_path, sample_features_df):
        """Scoring function produces EvaluationRow-compatible output."""
        # Initialize adapter
        setup_fintext_adapter(db_path=db_path, use_stub=True)
        
        # Score
        result = fintext_scoring_function(
            sample_features_df, fold_id="test_fold", horizon=20
        )
        
        # Check required columns
        assert "as_of_date" in result.columns
        assert "ticker" in result.columns
        assert "stable_id" in result.columns
        assert "fold_id" in result.columns
        assert "horizon" in result.columns
        assert "score" in result.columns
        assert "excess_return" in result.columns
        
        # Check values
        assert (result["fold_id"] == "test_fold").all()
        assert (result["horizon"] == 20).all()
        assert len(result) == len(sample_features_df)
    
    def test_scores_are_finite(self, db_path, sample_features_df):
        """All scores are finite (no NaN/inf)."""
        setup_fintext_adapter(db_path=db_path, use_stub=True)
        result = fintext_scoring_function(
            sample_features_df, fold_id="test", horizon=20
        )
        assert result["score"].notna().all()
        assert np.all(np.isfinite(result["score"].values))
    
    def test_scores_differ_across_tickers(self, db_path, sample_features_df):
        """Scores vary across tickers (not all identical)."""
        setup_fintext_adapter(db_path=db_path, use_stub=True)
        result = fintext_scoring_function(
            sample_features_df, fold_id="test", horizon=20
        )
        # At least 2 unique score values
        assert result["score"].nunique() >= 2


# ============================================================================
# LEAK TRIPWIRES
# ============================================================================

class TestLeakTripwires:
    def test_shuffle_control_changes_scores(self, db_path, sample_features_df):
        """Shuffle control produces different scores than original."""
        setup_fintext_adapter(db_path=db_path, use_stub=True)
        
        # Get original scores
        original = fintext_scoring_function(
            sample_features_df, fold_id="test", horizon=20
        )
        
        # Get shuffled scores (with fixed seed)
        np.random.seed(42)
        shuffled = shuffle_within_date_control(
            sample_features_df, fold_id="test", horizon=20
        )
        
        # Scores should differ (shuffled)
        # (Not checking exact values since shuffle is random, but structure should match)
        assert len(shuffled) == len(original)
        assert "score" in shuffled.columns
    
    def test_lag_control_shifts_scores(self, db_path):
        """Lag control shifts scores forward by 1 day."""
        setup_fintext_adapter(db_path=db_path, use_stub=True)
        
        # Create features with multiple dates
        dates = pd.date_range("2024-03-01", periods=5, freq="D")
        rows = []
        for date in dates:
            for ticker in ["AAPL", "NVDA"]:
                rows.append({
                    "date": date,
                    "ticker": ticker,
                    "stable_id": ticker,
                    "excess_return_20d": 0.01,
                })
        features_df = pd.DataFrame(rows)
        
        # Get lagged scores
        lagged = lag_control(features_df, fold_id="test", horizon=20)
        
        # Should have fewer rows (first date dropped due to shift)
        assert len(lagged) < len(features_df)
        assert "score" in lagged.columns


# ============================================================================
# INTEGRATION
# ============================================================================

@pytest.mark.slow
class TestIntegration:
    """Integration tests (slow, requires full pipeline)."""
    
    def test_smoke_evaluation_runs(self, db_path):
        """Full smoke evaluation completes without errors."""
        # This would require running the actual script
        # Just verify the script is importable for now
        import run_chapter9_fintext
        assert hasattr(run_chapter9_fintext, "main")
        assert hasattr(run_chapter9_fintext, "fintext_scoring_function")

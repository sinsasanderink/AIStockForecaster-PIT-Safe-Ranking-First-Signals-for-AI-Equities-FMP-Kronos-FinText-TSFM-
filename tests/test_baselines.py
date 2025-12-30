"""
Tests for Phase 3 Baselines (Chapter 6)

CRITICAL: These tests ensure baselines:
1. Produce correct scoring direction (higher = better)
2. Prevent duplicates per (as_of_date, stable_id, horizon)
3. Are deterministic (shuffle → identical output)
4. Have monotone relationship with underlying features
"""

import pytest
import pandas as pd
import numpy as np
from datetime import date

from src.evaluation.baselines import (
    BASELINE_REGISTRY,
    BASELINE_MOM_12M,
    BASELINE_MOMENTUM_COMPOSITE,
    BASELINE_SHORT_TERM_STRENGTH,
    BASELINE_NAIVE_RANDOM,
    FACTOR_BASELINES,
    SANITY_BASELINES,
    compute_baseline_score,
    generate_baseline_scores,
    run_all_baselines,
    validate_baseline_monotonicity,
    get_baseline_description,
    list_baselines,
    _compute_naive_random_score,
)


# ============================================================================
# TEST FIXTURES
# ============================================================================

@pytest.fixture
def sample_features_df():
    """Create sample features DataFrame for testing."""
    np.random.seed(42)
    
    data = []
    dates = pd.date_range(start="2023-01-01", periods=3, freq="MS")
    tickers = ["AAPL", "MSFT", "GOOGL", "NVDA", "META"]
    
    for d in dates:
        for i, ticker in enumerate(tickers):
            data.append({
                "date": d,
                "ticker": ticker,
                "stable_id": f"STABLE_{ticker}",
                "mom_1m": np.random.randn() * 0.1,
                "mom_3m": np.random.randn() * 0.15,
                "mom_6m": np.random.randn() * 0.2,
                "mom_12m": np.random.randn() * 0.3,
                "excess_return": np.random.randn() * 0.05,
                "adv_20d": 1_000_000 * (i + 1),
                "sector": ["Technology", "Technology", "Technology", "Semiconductors", "Technology"][i],
                "vix_percentile_252d": 50 + np.random.randn() * 10
            })
    
    return pd.DataFrame(data)


@pytest.fixture
def deterministic_features_df():
    """Create features with known values for deterministic testing."""
    data = [
        {"date": date(2023, 1, 1), "ticker": "A", "stable_id": "S_A", 
         "mom_1m": 0.10, "mom_3m": 0.15, "mom_6m": 0.20, "mom_12m": 0.25, "excess_return": 0.05},
        {"date": date(2023, 1, 1), "ticker": "B", "stable_id": "S_B", 
         "mom_1m": 0.05, "mom_3m": 0.10, "mom_6m": 0.15, "mom_12m": 0.20, "excess_return": 0.03},
        {"date": date(2023, 1, 1), "ticker": "C", "stable_id": "S_C", 
         "mom_1m": -0.02, "mom_3m": 0.00, "mom_6m": 0.05, "mom_12m": 0.10, "excess_return": -0.01},
    ]
    return pd.DataFrame(data)


# ============================================================================
# TEST BASELINE DEFINITIONS
# ============================================================================

class TestBaselineDefinitions:
    """Test baseline definitions are correct and immutable."""
    
    def test_registry_has_expected_baselines(self):
        """Verify expected baselines are registered."""
        assert len(BASELINE_REGISTRY) == 4
        assert set(BASELINE_REGISTRY.keys()) == {
            "mom_12m", "momentum_composite", "short_term_strength", "naive_random"
        }
    
    def test_factor_baselines_list(self):
        """Verify factor baselines are correctly listed."""
        assert set(FACTOR_BASELINES) == {"mom_12m", "momentum_composite", "short_term_strength"}
    
    def test_sanity_baselines_list(self):
        """Verify sanity baselines are correctly listed."""
        assert set(SANITY_BASELINES) == {"naive_random"}
    
    def test_mom_12m_definition(self):
        """Verify mom_12m baseline definition."""
        b = BASELINE_MOM_12M
        assert b.name == "mom_12m"
        assert b.required_features == ("mom_12m",)
        assert "12-month" in b.description.lower() or "12 month" in b.description.lower()
    
    def test_momentum_composite_definition(self):
        """Verify momentum_composite baseline definition."""
        b = BASELINE_MOMENTUM_COMPOSITE
        assert b.name == "momentum_composite"
        assert set(b.required_features) == {"mom_1m", "mom_3m", "mom_6m", "mom_12m"}
    
    def test_short_term_strength_definition(self):
        """Verify short_term_strength baseline definition."""
        b = BASELINE_SHORT_TERM_STRENGTH
        assert b.name == "short_term_strength"
        assert b.required_features == ("mom_1m",)
    
    def test_naive_random_definition(self):
        """Verify naive_random baseline definition."""
        b = BASELINE_NAIVE_RANDOM
        assert b.name == "naive_random"
        assert b.required_features == ()  # No features required
        assert "random" in b.description.lower()
        assert "sanity" in b.description.lower() or "pipeline" in b.description.lower()
    
    def test_definitions_are_frozen(self):
        """Verify baseline definitions are immutable."""
        with pytest.raises(Exception):  # FrozenInstanceError
            BASELINE_MOM_12M.name = "something_else"


# ============================================================================
# TEST SCORE COMPUTATION
# ============================================================================

class TestScoreComputation:
    """Test baseline score computation."""
    
    def test_mom_12m_score_equals_feature(self, deterministic_features_df):
        """mom_12m score should equal the mom_12m feature value."""
        for _, row in deterministic_features_df.iterrows():
            score = compute_baseline_score(row, "mom_12m")
            assert score == row["mom_12m"]
    
    def test_short_term_strength_score_equals_feature(self, deterministic_features_df):
        """short_term_strength score should equal the mom_1m feature value."""
        for _, row in deterministic_features_df.iterrows():
            score = compute_baseline_score(row, "short_term_strength")
            assert score == row["mom_1m"]
    
    def test_momentum_composite_score(self, deterministic_features_df):
        """momentum_composite score should be average of 4 momentum features."""
        for _, row in deterministic_features_df.iterrows():
            score = compute_baseline_score(row, "momentum_composite")
            expected = (row["mom_1m"] + row["mom_3m"] + row["mom_6m"] + row["mom_12m"]) / 4
            assert abs(score - expected) < 1e-10
    
    def test_missing_feature_returns_none(self):
        """Score should be None if required feature is missing."""
        row = pd.Series({"mom_3m": 0.1, "mom_6m": 0.2})  # Missing mom_12m
        score = compute_baseline_score(row, "mom_12m")
        assert score is None
    
    def test_nan_feature_returns_none(self):
        """Score should be None if required feature is NaN."""
        row = pd.Series({"mom_12m": np.nan})
        score = compute_baseline_score(row, "mom_12m")
        assert score is None
    
    def test_unknown_baseline_raises(self):
        """Unknown baseline should raise ValueError."""
        row = pd.Series({"mom_12m": 0.1})
        with pytest.raises(ValueError, match="Unknown baseline"):
            compute_baseline_score(row, "unknown_baseline")


# ============================================================================
# TEST NAIVE RANDOM BASELINE
# ============================================================================

class TestNaiveRandomBaseline:
    """Test naive_random baseline - pipeline sanity check."""
    
    def test_naive_random_is_deterministic(self):
        """Same inputs must always produce same output."""
        as_of = date(2023, 1, 15)
        horizon = 20
        stable_id = "STABLE_AAPL"
        
        score1 = _compute_naive_random_score(as_of, horizon, stable_id)
        score2 = _compute_naive_random_score(as_of, horizon, stable_id)
        
        assert score1 == score2
    
    def test_naive_random_different_dates(self):
        """Different dates should (usually) produce different scores."""
        horizon = 20
        stable_id = "STABLE_AAPL"
        
        score1 = _compute_naive_random_score(date(2023, 1, 15), horizon, stable_id)
        score2 = _compute_naive_random_score(date(2023, 1, 16), horizon, stable_id)
        
        # Different dates should produce different scores
        # (theoretically could be same by chance, but astronomically unlikely)
        assert score1 != score2
    
    def test_naive_random_different_horizons(self):
        """Different horizons should produce different scores."""
        as_of = date(2023, 1, 15)
        stable_id = "STABLE_AAPL"
        
        score20 = _compute_naive_random_score(as_of, 20, stable_id)
        score60 = _compute_naive_random_score(as_of, 60, stable_id)
        score90 = _compute_naive_random_score(as_of, 90, stable_id)
        
        assert len({score20, score60, score90}) == 3
    
    def test_naive_random_different_stocks(self):
        """Different stocks should produce different scores."""
        as_of = date(2023, 1, 15)
        horizon = 20
        
        score1 = _compute_naive_random_score(as_of, horizon, "STABLE_AAPL")
        score2 = _compute_naive_random_score(as_of, horizon, "STABLE_MSFT")
        
        assert score1 != score2
    
    def test_naive_random_output_range(self):
        """Scores should be in [0, 1]."""
        for i in range(100):
            as_of = date(2023, 1, 1 + (i % 28))
            score = _compute_naive_random_score(as_of, 20, f"STABLE_{i}")
            assert 0 <= score <= 1
    
    def test_naive_random_generates_scores(self, sample_features_df):
        """naive_random should generate scores for all rows."""
        output = generate_baseline_scores(
            sample_features_df,
            baseline_name="naive_random",
            fold_id="fold_01",
            horizon=20
        )
        
        # Should produce output for all rows
        n_expected = len(sample_features_df.dropna(subset=["excess_return"]))
        assert len(output) == n_expected
        
        # All scores should be in [0, 1]
        assert (output["score"] >= 0).all()
        assert (output["score"] <= 1).all()
    
    def test_naive_random_determinism_in_batch(self, sample_features_df):
        """Batch generation should be deterministic."""
        output1 = generate_baseline_scores(
            sample_features_df,
            baseline_name="naive_random",
            fold_id="fold_01",
            horizon=20
        )
        
        # Shuffle and run again
        shuffled = sample_features_df.sample(frac=1.0, random_state=42)
        output2 = generate_baseline_scores(
            shuffled,
            baseline_name="naive_random",
            fold_id="fold_01",
            horizon=20
        )
        
        # Sort and compare
        output1_sorted = output1.sort_values(["as_of_date", "stable_id"]).reset_index(drop=True)
        output2_sorted = output2.sort_values(["as_of_date", "stable_id"]).reset_index(drop=True)
        
        pd.testing.assert_frame_equal(output1_sorted, output2_sorted)


# ============================================================================
# TEST EVALUATION ROW GENERATION
# ============================================================================

class TestEvaluationRowGeneration:
    """Test generation of evaluation rows in canonical format."""
    
    def test_output_columns(self, sample_features_df):
        """Verify output has required EvaluationRow columns."""
        output = generate_baseline_scores(
            sample_features_df,
            baseline_name="mom_12m",
            fold_id="fold_01",
            horizon=20
        )
        
        required_cols = ["as_of_date", "ticker", "stable_id", "horizon", "fold_id", "score", "excess_return"]
        for col in required_cols:
            assert col in output.columns
    
    def test_horizon_and_fold_set_correctly(self, sample_features_df):
        """Verify horizon and fold_id are set correctly."""
        output = generate_baseline_scores(
            sample_features_df,
            baseline_name="mom_12m",
            fold_id="test_fold",
            horizon=60
        )
        
        assert (output["horizon"] == 60).all()
        assert (output["fold_id"] == "test_fold").all()
    
    def test_no_duplicates(self, sample_features_df):
        """Verify no duplicates in output (as_of_date, stable_id, horizon)."""
        output = generate_baseline_scores(
            sample_features_df,
            baseline_name="mom_12m",
            fold_id="fold_01",
            horizon=20
        )
        
        # Check for duplicates
        duplicate_check = output.groupby(["as_of_date", "stable_id", "horizon"]).size()
        duplicates = duplicate_check[duplicate_check > 1]
        assert len(duplicates) == 0
    
    def test_duplicate_input_raises(self, sample_features_df):
        """Verify duplicate input rows raise an error."""
        # Create DataFrame with duplicates
        df = pd.concat([sample_features_df, sample_features_df.head(1)], ignore_index=True)
        
        with pytest.raises(ValueError, match="Duplicate entries"):
            generate_baseline_scores(
                df,
                baseline_name="mom_12m",
                fold_id="fold_01",
                horizon=20
            )
    
    def test_missing_required_column_raises(self, sample_features_df):
        """Verify missing required column raises error."""
        df = sample_features_df.drop(columns=["mom_12m"])
        
        with pytest.raises(ValueError, match="Missing required columns"):
            generate_baseline_scores(
                df,
                baseline_name="mom_12m",
                fold_id="fold_01",
                horizon=20
            )


# ============================================================================
# TEST SCORING DIRECTION (HIGHER = BETTER)
# ============================================================================

class TestScoringDirection:
    """Test that higher score = better (monotone with feature)."""
    
    def test_mom_12m_higher_is_better(self, deterministic_features_df):
        """Higher mom_12m should give higher score."""
        output = generate_baseline_scores(
            deterministic_features_df,
            baseline_name="mom_12m",
            fold_id="fold_01",
            horizon=20
        )
        
        # Stock A has highest mom_12m (0.25), should have highest score
        # Stock C has lowest mom_12m (0.10), should have lowest score
        output = output.sort_values("score", ascending=False)
        assert output.iloc[0]["stable_id"] == "S_A"
        assert output.iloc[-1]["stable_id"] == "S_C"
    
    def test_short_term_strength_higher_is_better(self, deterministic_features_df):
        """Higher mom_1m should give higher score."""
        output = generate_baseline_scores(
            deterministic_features_df,
            baseline_name="short_term_strength",
            fold_id="fold_01",
            horizon=20
        )
        
        # Stock A has highest mom_1m (0.10), should have highest score
        # Stock C has lowest mom_1m (-0.02), should have lowest score
        output = output.sort_values("score", ascending=False)
        assert output.iloc[0]["stable_id"] == "S_A"
        assert output.iloc[-1]["stable_id"] == "S_C"


# ============================================================================
# TEST DETERMINISM
# ============================================================================

class TestDeterminism:
    """Test that baseline scoring is deterministic."""
    
    def test_same_input_same_output(self, sample_features_df):
        """Same input should produce identical output."""
        output1 = generate_baseline_scores(
            sample_features_df,
            baseline_name="mom_12m",
            fold_id="fold_01",
            horizon=20
        )
        
        output2 = generate_baseline_scores(
            sample_features_df,
            baseline_name="mom_12m",
            fold_id="fold_01",
            horizon=20
        )
        
        pd.testing.assert_frame_equal(output1, output2)
    
    def test_shuffled_input_same_output(self, sample_features_df):
        """Shuffled input should produce identical output (after sorting)."""
        shuffled = sample_features_df.sample(frac=1.0, random_state=42)
        
        output_original = generate_baseline_scores(
            sample_features_df,
            baseline_name="mom_12m",
            fold_id="fold_01",
            horizon=20
        )
        
        output_shuffled = generate_baseline_scores(
            shuffled,
            baseline_name="mom_12m",
            fold_id="fold_01",
            horizon=20
        )
        
        # Sort both for comparison
        output_original = output_original.sort_values(
            ["as_of_date", "stable_id"]
        ).reset_index(drop=True)
        
        output_shuffled = output_shuffled.sort_values(
            ["as_of_date", "stable_id"]
        ).reset_index(drop=True)
        
        pd.testing.assert_frame_equal(output_original, output_shuffled)
    
    def test_all_baselines_deterministic(self, sample_features_df):
        """All baselines should be deterministic."""
        for baseline_name in list_baselines():
            output1 = generate_baseline_scores(
                sample_features_df,
                baseline_name=baseline_name,
                fold_id="fold_01",
                horizon=20
            )
            
            shuffled = sample_features_df.sample(frac=1.0, random_state=123)
            output2 = generate_baseline_scores(
                shuffled,
                baseline_name=baseline_name,
                fold_id="fold_01",
                horizon=20
            )
            
            # Sort and compare
            o1 = output1.sort_values(["as_of_date", "stable_id"]).reset_index(drop=True)
            o2 = output2.sort_values(["as_of_date", "stable_id"]).reset_index(drop=True)
            
            pd.testing.assert_frame_equal(o1, o2)


# ============================================================================
# TEST MONOTONICITY
# ============================================================================

class TestMonotonicity:
    """Test that baseline scores are monotone with underlying features."""
    
    def test_mom_12m_monotone(self, sample_features_df):
        """mom_12m score should be perfectly monotone with mom_12m feature."""
        assert validate_baseline_monotonicity(
            sample_features_df,
            baseline_name="mom_12m",
            feature_col="mom_12m"
        )
    
    def test_short_term_strength_monotone(self, sample_features_df):
        """short_term_strength score should be perfectly monotone with mom_1m feature."""
        assert validate_baseline_monotonicity(
            sample_features_df,
            baseline_name="short_term_strength",
            feature_col="mom_1m"
        )
    
    def test_momentum_composite_positive_correlation(self, sample_features_df):
        """momentum_composite should be positively correlated with mom_12m."""
        assert validate_baseline_monotonicity(
            sample_features_df,
            baseline_name="momentum_composite",
            feature_col="mom_12m"
        )


# ============================================================================
# TEST BATCH RUNNER
# ============================================================================

class TestBatchRunner:
    """Test running all baselines at once."""
    
    def test_run_all_baselines(self, sample_features_df):
        """run_all_baselines should return results for all 3 baselines."""
        results = run_all_baselines(
            sample_features_df,
            fold_id="fold_01",
            horizon=20
        )
        
        assert len(results) == 4
        assert set(results.keys()) == {"mom_12m", "momentum_composite", "short_term_strength", "naive_random"}
        
        for name, df in results.items():
            assert len(df) > 0
            assert "score" in df.columns
    
    def test_run_subset_of_baselines(self, sample_features_df):
        """Can run a subset of baselines."""
        results = run_all_baselines(
            sample_features_df,
            fold_id="fold_01",
            horizon=20,
            baselines=["mom_12m", "short_term_strength"]
        )
        
        assert len(results) == 2
        assert "mom_12m" in results
        assert "short_term_strength" in results
        assert "momentum_composite" not in results


# ============================================================================
# TEST HELPER FUNCTIONS
# ============================================================================

class TestHelpers:
    """Test helper functions."""
    
    def test_list_baselines(self):
        """list_baselines should return all baseline names."""
        baselines = list_baselines()
        assert len(baselines) == 4
        assert "mom_12m" in baselines
        assert "momentum_composite" in baselines
        assert "naive_random" in baselines
        assert "short_term_strength" in baselines
    
    def test_get_baseline_description(self):
        """get_baseline_description should return description."""
        desc = get_baseline_description("mom_12m")
        assert "mom_12m" in desc
        assert "12-month" in desc.lower() or "12 month" in desc.lower()
    
    def test_unknown_baseline_description_raises(self):
        """Unknown baseline should raise ValueError."""
        with pytest.raises(ValueError):
            get_baseline_description("unknown")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


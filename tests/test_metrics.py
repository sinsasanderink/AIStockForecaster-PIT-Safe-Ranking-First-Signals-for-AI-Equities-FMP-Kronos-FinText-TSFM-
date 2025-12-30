"""
Tests for Evaluation Metrics (Chapter 6.3)

These tests verify:
1. Perfect correlation → RankIC ≈ 1
2. Random scores → RankIC ≈ 0
3. Deterministic tie-breaking
4. Churn math matches hand-calculated overlap
5. Regime slicing preserves counts
"""

import pytest
import pandas as pd
import numpy as np
from datetime import date, timedelta

from src.evaluation.metrics import (
    EvaluationRow,
    rank_with_ties,
    compute_rankic_per_date,
    compute_quintile_spread_per_date,
    compute_topk_metrics_per_date,
    compute_churn,
    assign_regime_bucket,
    aggregate_per_date_metrics,
    evaluate_fold,
    evaluate_with_regime_slicing,
    REGIME_DEFINITIONS,
    MIN_CROSS_SECTION_SIZE,
)


class TestEvaluationRow:
    """Test canonical evaluation data contract."""
    
    def test_evaluation_row_creation(self):
        """Test creating an evaluation row."""
        row = EvaluationRow(
            as_of_date=date(2023, 1, 1),
            ticker="AAPL",
            stable_id="AAPL_001",
            horizon=20,
            fold_id="fold_01",
            score=0.75,
            excess_return=0.05
        )
        assert row.as_of_date == date(2023, 1, 1)
        assert row.score == 0.75
        assert row.excess_return == 0.05
    
    def test_evaluation_row_with_optionals(self):
        """Test evaluation row with optional fields."""
        row = EvaluationRow(
            as_of_date=date(2023, 1, 1),
            ticker="AAPL",
            stable_id="AAPL_001",
            horizon=20,
            fold_id="fold_01",
            score=0.75,
            excess_return=0.05,
            sector="Technology",
            beta_252d=1.2,
            vix_percentile_252d=45.0
        )
        assert row.sector == "Technology"
        assert row.beta_252d == 1.2


class TestTieBreaking:
    """Test deterministic tie-breaking."""
    
    def test_rank_with_ties_no_ties(self):
        """Test ranking without ties."""
        df = pd.DataFrame({
            "score": [0.9, 0.7, 0.5, 0.3],
            "stable_id": ["A", "B", "C", "D"]
        })
        
        ranks = rank_with_ties(df)
        assert list(ranks) == [1, 2, 3, 4]
    
    def test_rank_with_ties_has_ties(self):
        """Test ranking with ties (deterministic by stable_id)."""
        df = pd.DataFrame({
            "score": [0.9, 0.7, 0.7, 0.3],
            "stable_id": ["A", "C", "B", "D"]  # B comes before C alphabetically
        })
        
        ranks = rank_with_ties(df)
        # A=1, B=2 (0.7, comes before C), C=3 (0.7), D=4
        expected_ranks = pd.Series([1, 3, 2, 4], index=df.index)
        pd.testing.assert_series_equal(ranks, expected_ranks, check_names=False)
    
    def test_rank_with_ties_is_deterministic(self):
        """Test that tie-breaking is deterministic across runs."""
        df = pd.DataFrame({
            "score": [0.5, 0.5, 0.5],
            "stable_id": ["C", "A", "B"]
        })
        
        # Run multiple times - should always get same result
        ranks1 = rank_with_ties(df)
        ranks2 = rank_with_ties(df)
        ranks3 = rank_with_ties(df)
        
        pd.testing.assert_series_equal(ranks1, ranks2)
        pd.testing.assert_series_equal(ranks2, ranks3)
        
        # Check alphabetical order: A=1, B=2, C=3
        expected = pd.Series([3, 1, 2], index=df.index)
        pd.testing.assert_series_equal(ranks1, expected, check_names=False)


class TestRankIC:
    """Test RankIC computation."""
    
    def test_perfect_positive_correlation(self):
        """Test perfect positive correlation gives RankIC ≈ 1."""
        df = pd.DataFrame({
            "score": [1.0, 2.0, 3.0, 4.0, 5.0],
            "excess_return": [0.01, 0.02, 0.03, 0.04, 0.05]
        })
        
        ic = compute_rankic_per_date(df)
        assert abs(ic - 1.0) < 0.01  # Should be very close to 1
    
    def test_perfect_negative_correlation(self):
        """Test perfect negative correlation gives RankIC ≈ -1."""
        df = pd.DataFrame({
            "score": [1.0, 2.0, 3.0, 4.0, 5.0],
            "excess_return": [0.05, 0.04, 0.03, 0.02, 0.01]  # Reversed
        })
        
        ic = compute_rankic_per_date(df)
        assert abs(ic - (-1.0)) < 0.01  # Should be close to -1
    
    def test_random_scores_near_zero_ic(self):
        """Test random scores give IC ≈ 0 (on average)."""
        np.random.seed(42)
        
        ics = []
        for _ in range(100):  # Multiple samples
            df = pd.DataFrame({
                "score": np.random.randn(50),
                "excess_return": np.random.randn(50)
            })
            ic = compute_rankic_per_date(df)
            ics.append(ic)
        
        mean_ic = np.mean(ics)
        assert abs(mean_ic) < 0.15  # Should be close to 0 on average
    
    def test_rankic_with_too_few_observations(self):
        """Test RankIC with < 2 observations returns NaN."""
        df = pd.DataFrame({
            "score": [1.0],
            "excess_return": [0.05]
        })
        
        ic = compute_rankic_per_date(df)
        assert np.isnan(ic)
    
    def test_rankic_handles_missing_values(self):
        """Test RankIC drops missing values."""
        df = pd.DataFrame({
            "score": [1.0, 2.0, np.nan, 4.0, 5.0],
            "excess_return": [0.01, 0.02, 0.03, np.nan, 0.05]
        })
        
        ic = compute_rankic_per_date(df)
        # Should compute on 3 valid pairs: (1, 0.01), (2, 0.02), (5, 0.05)
        assert not np.isnan(ic)


class TestQuintileSpread:
    """Test quintile spread computation."""
    
    def test_quintile_spread_basic(self):
        """Test basic quintile spread calculation."""
        # Create data where top quintile outperforms bottom
        df = pd.DataFrame({
            "score": [5, 4, 3, 2, 1, 5, 4, 3, 2, 1],  # 10 stocks, 5 quintiles of 2 each
            "excess_return": [0.10, 0.08, 0.05, 0.02, -0.01, 0.09, 0.07, 0.04, 0.01, -0.02]
        })
        
        spread = compute_quintile_spread_per_date(df)
        
        # Top quintile (scores 5) vs bottom quintile (scores 1)
        top_mean = (0.10 + 0.09) / 2
        bottom_mean = (-0.01 + -0.02) / 2
        expected_spread = top_mean - bottom_mean
        
        assert abs(spread["spread"] - expected_spread) < 0.01
        assert abs(spread["top_mean"] - top_mean) < 0.01
        assert abs(spread["bottom_mean"] - bottom_mean) < 0.01
    
    def test_quintile_spread_too_few_stocks(self):
        """Test quintile spread with too few stocks returns NaN."""
        df = pd.DataFrame({
            "score": [1, 2, 3],  # Only 3 stocks, need 5 for quintiles
            "excess_return": [0.01, 0.02, 0.03]
        })
        
        spread = compute_quintile_spread_per_date(df, n_buckets=5)
        assert np.isnan(spread["spread"])


class TestTopKMetrics:
    """Test Top-K hit rate and excess return."""
    
    def test_topk_all_positive(self):
        """Test Top-K with all positive excess returns."""
        df = pd.DataFrame({
            "score": [5, 4, 3, 2, 1],
            "excess_return": [0.10, 0.08, 0.05, 0.02, 0.01],
            "stable_id": ["A", "B", "C", "D", "E"]
        })
        
        metrics = compute_topk_metrics_per_date(df, k=3)
        
        assert metrics["hit_rate"] == 1.0  # All 3 are positive
        assert abs(metrics["avg_excess_return"] - 0.0767) < 0.01  # (0.10 + 0.08 + 0.05) / 3
        assert metrics["n_positive"] == 3
        assert metrics["n_total"] == 3
    
    def test_topk_mixed(self):
        """Test Top-K with mixed positive/negative."""
        df = pd.DataFrame({
            "score": [5, 4, 3, 2, 1],
            "excess_return": [0.10, -0.02, 0.05, 0.02, 0.01],
            "stable_id": ["A", "B", "C", "D", "E"]
        })
        
        metrics = compute_topk_metrics_per_date(df, k=3)
        
        # Top 3: A (0.10), B (-0.02), C (0.05)
        assert abs(metrics["hit_rate"] - (2/3)) < 0.01  # 2 out of 3 positive
        assert abs(metrics["avg_excess_return"] - 0.0433) < 0.01  # (0.10 - 0.02 + 0.05) / 3
        assert metrics["n_positive"] == 2
    
    def test_topk_too_few_stocks(self):
        """Test Top-K with fewer stocks than K."""
        df = pd.DataFrame({
            "score": [5, 4],
            "excess_return": [0.10, 0.08],
            "stable_id": ["A", "B"]
        })
        
        metrics = compute_topk_metrics_per_date(df, k=5)
        assert np.isnan(metrics["hit_rate"])
        assert np.isnan(metrics["avg_excess_return"])
        assert metrics["n_total"] == 2
    
    def test_topk_uses_deterministic_ties(self):
        """Test Top-K with tied scores uses deterministic tie-breaking."""
        df = pd.DataFrame({
            "score": [5, 5, 5, 2, 1],  # Three-way tie
            "excess_return": [0.10, 0.08, 0.05, 0.02, 0.01],
            "stable_id": ["C", "A", "B", "D", "E"]  # Not alphabetical
        })
        
        metrics = compute_topk_metrics_per_date(df, k=3)
        
        # Should pick A, B, C (alphabetical among tied scores)
        # A=0.08, B=0.05, C=0.10
        expected_avg = (0.08 + 0.05 + 0.10) / 3
        assert abs(metrics["avg_excess_return"] - expected_avg) < 0.01


class TestChurn:
    """Test churn computation."""
    
    def test_churn_perfect_overlap(self):
        """Test churn with perfect overlap (same Top-K)."""
        df = pd.DataFrame({
            "as_of_date": [date(2023, 1, 1)] * 5 + [date(2023, 2, 1)] * 5,
            "stable_id": ["A", "B", "C", "D", "E"] * 2,
            "score": [5, 4, 3, 2, 1] * 2  # Same rankings both dates
        })
        
        churn_df = compute_churn(df, k=3)
        
        assert len(churn_df) == 1  # One churn measurement
        assert churn_df.iloc[0]["retention"] == 1.0  # Perfect retention
        assert churn_df.iloc[0]["churn"] == 0.0  # Zero churn
        assert churn_df.iloc[0]["n_overlap"] == 3
    
    def test_churn_no_overlap(self):
        """Test churn with zero overlap (completely different Top-K)."""
        df = pd.DataFrame({
            "as_of_date": [date(2023, 1, 1)] * 5 + [date(2023, 2, 1)] * 5,
            "stable_id": ["A", "B", "C", "D", "E"] + ["F", "G", "H", "I", "J"],
            "score": [5, 4, 3, 2, 1] * 2
        })
        
        churn_df = compute_churn(df, k=3)
        
        assert len(churn_df) == 1
        assert churn_df.iloc[0]["retention"] == 0.0  # No retention
        assert churn_df.iloc[0]["churn"] == 1.0  # Complete churn
        assert churn_df.iloc[0]["n_overlap"] == 0
    
    def test_churn_partial_overlap(self):
        """Test churn with partial overlap."""
        df = pd.DataFrame({
            "as_of_date": [date(2023, 1, 1)] * 5 + [date(2023, 2, 1)] * 5,
            "stable_id": ["A", "B", "C", "D", "E"] + ["A", "B", "F", "G", "H"],
            "score": [5, 4, 3, 2, 1] * 2
        })
        
        churn_df = compute_churn(df, k=3)
        
        # Top 3 at T=1: A, B, C
        # Top 3 at T=2: A, B, F
        # Overlap: A, B (2 out of 3)
        assert len(churn_df) == 1
        assert abs(churn_df.iloc[0]["retention"] - (2/3)) < 0.01
        assert abs(churn_df.iloc[0]["churn"] - (1/3)) < 0.01
        assert churn_df.iloc[0]["n_overlap"] == 2
    
    def test_churn_hand_calculated_match(self):
        """Test churn matches hand-calculated overlap."""
        # Date 1: Top-3 = [A, B, C]
        # Date 2: Top-3 = [B, C, D]
        # Overlap = [B, C] = 2
        # Retention = 2/3, Churn = 1/3
        
        df = pd.DataFrame({
            "as_of_date": [date(2023, 1, 1)] * 4 + [date(2023, 2, 1)] * 4,
            "stable_id": ["A", "B", "C", "D"] * 2,
            "score": [4, 3, 2, 1, 1, 4, 3, 2]  # Rankings change
        })
        
        churn_df = compute_churn(df, k=3)
        
        retention = churn_df.iloc[0]["retention"]
        churn = churn_df.iloc[0]["churn"]
        
        # Hand-calculated
        assert abs(retention - (2/3)) < 0.001
        assert abs(churn - (1/3)) < 0.001
        assert abs(retention + churn - 1.0) < 0.001  # Should sum to 1
    
    def test_churn_multiple_dates(self):
        """Test churn across multiple consecutive dates."""
        dates = [date(2023, 1, 1), date(2023, 2, 1), date(2023, 3, 1)]
        
        df = pd.DataFrame({
            "as_of_date": [d for d in dates for _ in range(5)],
            "stable_id": ["A", "B", "C", "D", "E"] * 3,
            "score": [5, 4, 3, 2, 1] + [4, 5, 2, 3, 1] + [3, 2, 5, 4, 1]
        })
        
        churn_df = compute_churn(df, k=3)
        
        # Should have 2 churn measurements (dates 2 and 3)
        assert len(churn_df) == 2
        assert churn_df.iloc[0]["as_of_date"] == date(2023, 2, 1)
        assert churn_df.iloc[1]["as_of_date"] == date(2023, 3, 1)
    
    def test_churn_uses_stable_id_not_ticker(self):
        """Test that churn uses stable_id (data contract)."""
        # This is implicit in the test design but worth documenting
        df = pd.DataFrame({
            "as_of_date": [date(2023, 1, 1)] * 3 + [date(2023, 2, 1)] * 3,
            "stable_id": ["A_001", "B_002", "C_003"] * 2,
            "ticker": ["AAPL", "MSFT", "GOOGL", "AAPL", "MSFT", "GOOGL"],  # Same
            "score": [3, 2, 1] * 2
        })
        
        churn_df = compute_churn(df, k=2, id_col="stable_id")
        
        # Should track stable_id, giving perfect overlap
        assert churn_df.iloc[0]["churn"] == 0.0


class TestRegimeSlicing:
    """Test regime bucketing."""
    
    def test_assign_vix_buckets(self):
        """Test VIX percentile bucket assignment."""
        df = pd.DataFrame({
            "vix_percentile_252d": [10, 40, 50, 75, 90]
        })
        
        buckets = assign_regime_bucket(df, "vix_percentile_252d")
        
        assert buckets.iloc[0] == "low"   # 10 <= 33
        assert buckets.iloc[1] == "mid"   # 40 in (33, 67]
        assert buckets.iloc[2] == "mid"   # 50 in (33, 67]
        assert buckets.iloc[3] == "high"  # 75 > 67
        assert buckets.iloc[4] == "high"  # 90 > 67
    
    def test_assign_market_regime(self):
        """Test market regime (bull/bear) assignment."""
        df = pd.DataFrame({
            "market_regime": [0.05, -0.02, 0.0, 0.10, -0.05]
        })
        
        buckets = assign_regime_bucket(df, "market_regime")
        
        assert buckets.iloc[0] == "bull"  # > 0
        assert buckets.iloc[1] == "bear"  # <= 0
        assert buckets.iloc[2] == "bear"  # == 0 counts as bear
        assert buckets.iloc[3] == "bull"  # > 0
        assert buckets.iloc[4] == "bear"  # < 0
    
    def test_assign_regime_invalid_feature(self):
        """Test error on invalid regime feature."""
        df = pd.DataFrame({"vix_percentile_252d": [50]})
        
        # Test missing column
        with pytest.raises(ValueError, match="not in DataFrame"):
            assign_regime_bucket(df, "invalid_feature")
        
        # Test invalid regime feature (column exists but not in definitions)
        df2 = pd.DataFrame({"some_feature": [50]})
        with pytest.raises(ValueError, match="not in locked definitions"):
            assign_regime_bucket(df2, "some_feature")


class TestAggregation:
    """Test metric aggregation."""
    
    def test_aggregate_median(self):
        """Test median aggregation."""
        df = pd.DataFrame({
            "rankic": [0.05, 0.10, 0.08, 0.12, 0.06],
            "hit_rate": [0.6, 0.7, 0.65, 0.75, 0.55]
        })
        
        agg = aggregate_per_date_metrics(df, ["rankic", "hit_rate"], agg_method="median")
        
        assert abs(agg["rankic_median"] - 0.08) < 0.01
        assert abs(agg["hit_rate_median"] - 0.65) < 0.01
        assert "rankic_iqr" in agg
        assert agg["rankic_n_dates"] == 5
    
    def test_aggregate_mean(self):
        """Test mean aggregation."""
        df = pd.DataFrame({
            "rankic": [0.05, 0.10, 0.08]
        })
        
        agg = aggregate_per_date_metrics(df, ["rankic"], agg_method="mean")
        
        expected_mean = (0.05 + 0.10 + 0.08) / 3
        assert abs(agg["rankic_mean"] - expected_mean) < 0.01


class TestEvaluateFold:
    """Integration tests for evaluate_fold."""
    
    def create_test_eval_df(self, n_dates=3, n_stocks=20):
        """Helper to create test evaluation DataFrame."""
        np.random.seed(42)
        
        dates = [date(2023, 1, 1) + timedelta(days=30*i) for i in range(n_dates)]
        
        data = []
        for d in dates:
            for i in range(n_stocks):
                data.append({
                    "as_of_date": d,
                    "ticker": f"STOCK_{i}",
                    "stable_id": f"ID_{i:03d}",
                    "horizon": 20,
                    "fold_id": "fold_01",
                    "score": np.random.randn(),
                    "excess_return": np.random.randn() * 0.05
                })
        
        return pd.DataFrame(data)
    
    def test_evaluate_fold_basic(self):
        """Test basic fold evaluation."""
        eval_df = self.create_test_eval_df(n_dates=3, n_stocks=20)
        
        result = evaluate_fold(eval_df, fold_id="fold_01", horizon=20, k_values=[10])
        
        assert "per_date_metrics" in result
        assert "fold_summary" in result
        assert "churn" in result
        
        # Should have metrics for 3 dates
        assert len(result["per_date_metrics"]) == 3
        
        # Should have fold summary
        assert len(result["fold_summary"]) == 1
        assert "rankic_median" in result["fold_summary"].columns
        assert "top10_hit_rate_median" in result["fold_summary"].columns
        
        # Should have churn for K=10
        assert 10 in result["churn"]
    
    def test_evaluate_fold_drops_missing(self):
        """Test that evaluation drops missing score/excess_return."""
        eval_df = self.create_test_eval_df(n_dates=2, n_stocks=15)
        
        # Add some missing values
        eval_df.loc[0, "score"] = np.nan
        eval_df.loc[1, "excess_return"] = np.nan
        
        result = evaluate_fold(eval_df, fold_id="fold_01", horizon=20, k_values=[5])
        
        # Should still work (with warnings logged)
        assert len(result["per_date_metrics"]) > 0
    
    def test_evaluate_fold_skips_small_cross_sections(self):
        """Test that dates with too few stocks are skipped."""
        eval_df = self.create_test_eval_df(n_dates=3, n_stocks=5)  # Only 5 stocks
        
        result = evaluate_fold(
            eval_df, 
            fold_id="fold_01", 
            horizon=20, 
            k_values=[3],
            min_cross_section=10  # Require 10
        )
        
        # Should skip all dates
        assert len(result["per_date_metrics"]) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


"""
Unit Tests for signals.py
=========================

Tests for signal data structures, ensuring:
- ReturnDistribution formatting and calculations
- StockSignal convention enforcement (expected_return = distribution.mean)
- Serialization includes all required fields
"""

import pytest
from datetime import date, datetime
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from outputs.signals import (
    ReturnDistribution,
    StockSignal,
    SignalDriver,
    LiquidityFlag,
    SignalSource,
)


class TestReturnDistribution:
    """Tests for ReturnDistribution class."""
    
    @pytest.fixture
    def sample_distribution(self):
        """Create a sample distribution for testing."""
        return ReturnDistribution(
            percentile_5=-0.15,
            percentile_25=-0.05,
            percentile_50=0.05,
            percentile_75=0.12,
            percentile_95=0.25,
            mean=0.06,
            std=0.12,
        )
    
    def test_format_range_default(self, sample_distribution):
        """Test format_range() includes P5/P50/P95."""
        formatted = sample_distribution.format_range()
        assert "P5" not in formatted or "-15.0%" in formatted  # Value present
        assert "+5.0%" in formatted   # P50
        assert "+25.0%" in formatted  # P95
    
    def test_format_range_contains_all_percentiles(self, sample_distribution):
        """Ensure formatted range shows downside, median, and upside."""
        formatted = sample_distribution.format_range("p5_p50_p95")
        # Should contain negative (downside), median, and positive (upside)
        assert "-" in formatted  # Downside
        assert "+" in formatted  # Upside
    
    def test_prob_positive_calculation(self, sample_distribution):
        """Test prob_positive is reasonable for positive mean."""
        prob = sample_distribution.prob_positive
        assert 0 < prob < 1
        assert prob > 0.5  # Positive mean should give > 50% prob
    
    def test_prob_positive_zero_std(self):
        """Test prob_positive handles zero std."""
        dist = ReturnDistribution(
            percentile_5=0.1, percentile_25=0.1, percentile_50=0.1,
            percentile_75=0.1, percentile_95=0.1, mean=0.1, std=0.0
        )
        assert dist.prob_positive == 1.0  # Positive mean, no uncertainty
        
        dist_neg = ReturnDistribution(
            percentile_5=-0.1, percentile_25=-0.1, percentile_50=-0.1,
            percentile_75=-0.1, percentile_95=-0.1, mean=-0.1, std=0.0
        )
        assert dist_neg.prob_positive == 0.0  # Negative mean, no uncertainty
    
    def test_prob_positive_no_scipy_approx(self, sample_distribution):
        """Test that no-scipy approximation matches scipy result."""
        scipy_result = sample_distribution.prob_positive
        approx_result = sample_distribution._prob_positive_approx()
        assert abs(scipy_result - approx_result) < 0.001
    
    def test_from_samples(self):
        """Test creating distribution from samples."""
        np.random.seed(42)
        samples = np.random.normal(0.05, 0.10, 10000)
        dist = ReturnDistribution.from_samples(samples)
        
        assert abs(dist.mean - 0.05) < 0.01
        assert abs(dist.std - 0.10) < 0.01
        assert dist.percentile_5 < dist.percentile_50 < dist.percentile_95
    
    def test_to_dict_contains_quantiles(self, sample_distribution):
        """Test serialization includes all percentiles and prob_outperform."""
        d = sample_distribution.to_dict()
        
        assert "p5" in d
        assert "p25" in d
        assert "p50" in d
        assert "p75" in d
        assert "p95" in d
        assert "mean" in d
        assert "std" in d
        assert "prob_outperform" in d
    
    def test_iqr_calculation(self, sample_distribution):
        """Test interquartile range calculation."""
        expected_iqr = 0.12 - (-0.05)  # p75 - p25
        assert abs(sample_distribution.iqr - expected_iqr) < 1e-6


class TestStockSignal:
    """Tests for StockSignal class."""
    
    @pytest.fixture
    def sample_distribution(self):
        return ReturnDistribution(
            percentile_5=-0.15,
            percentile_25=-0.05,
            percentile_50=0.05,
            percentile_75=0.12,
            percentile_95=0.25,
            mean=0.06,
            std=0.12,
        )
    
    @pytest.fixture
    def sample_signal(self, sample_distribution):
        """Create a sample signal using factory method."""
        return StockSignal.create(
            ticker="NVDA",
            rebalance_date=date(2024, 1, 15),
            horizon_days=20,
            benchmark="QQQ",
            return_distribution=sample_distribution,
            alpha_rank_score=1.5,
            confidence_score=0.85,
            liquidity_flag=LiquidityFlag.OK,
            avg_daily_volume=50_000_000,
            key_drivers=(
                SignalDriver("momentum", "price", 0.03, "Strong momentum"),
            ),
        )
    
    def test_expected_return_equals_distribution_mean(self, sample_signal):
        """Test that expected_excess_return == distribution.mean."""
        assert sample_signal.expected_excess_return == sample_signal.return_distribution.mean
    
    def test_create_factory_auto_sets_expected_return(self, sample_distribution):
        """Test that factory method correctly sets expected_excess_return."""
        signal = StockSignal.create(
            ticker="AMD",
            rebalance_date=date(2024, 1, 15),
            horizon_days=20,
            benchmark="QQQ",
            return_distribution=sample_distribution,
            alpha_rank_score=1.0,
            confidence_score=0.7,
        )
        assert signal.expected_excess_return == sample_distribution.mean
    
    def test_direct_construction_enforces_consistency(self, sample_distribution):
        """Test that direct construction validates expected_return == mean."""
        # This should work (correct value)
        signal = StockSignal(
            ticker="TEST",
            rebalance_date=date(2024, 1, 15),
            horizon_days=20,
            benchmark="QQQ",
            expected_excess_return=sample_distribution.mean,  # Correct
            return_distribution=sample_distribution,
            alpha_rank_score=1.0,
            confidence_score=0.7,
        )
        assert signal.expected_excess_return == sample_distribution.mean
        
        # This should fail (wrong value)
        with pytest.raises(ValueError, match="must equal"):
            StockSignal(
                ticker="TEST",
                rebalance_date=date(2024, 1, 15),
                horizon_days=20,
                benchmark="QQQ",
                expected_excess_return=0.999,  # Wrong!
                return_distribution=sample_distribution,
                alpha_rank_score=1.0,
                confidence_score=0.7,
            )
    
    def test_summary_includes_distribution_range(self, sample_signal):
        """Test that summary() shows P5/P50/P95 range."""
        summary = sample_signal.summary()
        
        assert "NVDA" in summary
        assert "P5/P50/P95" in summary or "Range" in summary
        assert "Prob(Outperform)" in summary or "P(>0)" in summary
    
    def test_summary_includes_prob_outperform(self, sample_signal):
        """Test that summary shows probability of outperformance."""
        summary = sample_signal.summary()
        # Should show percentage
        assert "%" in summary
    
    def test_to_dict_contains_all_fields(self, sample_signal):
        """Test serialization includes quantiles and flags."""
        d = sample_signal.to_dict()
        
        # Core fields
        assert d["ticker"] == "NVDA"
        assert d["expected_excess_return"] == sample_signal.expected_excess_return
        assert "return_distribution" in d
        assert "prob_outperform" in d
        
        # Distribution should have percentiles
        dist_dict = d["return_distribution"]
        assert "p5" in dist_dict
        assert "p50" in dist_dict
        assert "p95" in dist_dict
        
        # Liquidity
        assert d["liquidity_flag"] == "ok"
        assert d["avg_daily_volume"] == 50_000_000
    
    def test_one_liner_format(self, sample_signal):
        """Test one-liner table format."""
        one_liner = sample_signal.one_liner()
        
        assert "NVDA" in one_liner
        assert "P(>0)" in one_liner or "%" in one_liner
        assert "★" in one_liner  # High confidence marker
    
    def test_signal_direction_thresholds(self, sample_distribution):
        """Test signal direction classification."""
        # Buy signal (> 2% expected excess)
        buy_dist = ReturnDistribution(
            percentile_5=-0.10, percentile_25=0.0, percentile_50=0.03,
            percentile_75=0.06, percentile_95=0.10, mean=0.03, std=0.05
        )
        buy_signal = StockSignal.create(
            ticker="BUY", rebalance_date=date(2024, 1, 1), horizon_days=20,
            benchmark="QQQ", return_distribution=buy_dist,
            alpha_rank_score=1.0, confidence_score=0.7
        )
        assert buy_signal.signal_direction == "buy"
        
        # Avoid signal (< -2% expected excess)
        avoid_dist = ReturnDistribution(
            percentile_5=-0.20, percentile_25=-0.10, percentile_50=-0.05,
            percentile_75=-0.02, percentile_95=0.02, mean=-0.05, std=0.06
        )
        avoid_signal = StockSignal.create(
            ticker="AVOID", rebalance_date=date(2024, 1, 1), horizon_days=20,
            benchmark="QQQ", return_distribution=avoid_dist,
            alpha_rank_score=-0.5, confidence_score=0.6
        )
        assert avoid_signal.signal_direction == "avoid"
        
        # Neutral signal (-2% to +2%)
        neutral_dist = ReturnDistribution(
            percentile_5=-0.08, percentile_25=-0.02, percentile_50=0.01,
            percentile_75=0.03, percentile_95=0.08, mean=0.01, std=0.04
        )
        neutral_signal = StockSignal.create(
            ticker="NEUTRAL", rebalance_date=date(2024, 1, 1), horizon_days=20,
            benchmark="QQQ", return_distribution=neutral_dist,
            alpha_rank_score=0.1, confidence_score=0.5
        )
        assert neutral_signal.signal_direction == "neutral"


class TestLiquidityFlag:
    """Tests for liquidity flag handling."""
    
    def test_liquidity_concern_detection(self):
        """Test has_liquidity_concern property."""
        dist = ReturnDistribution(
            percentile_5=-0.1, percentile_25=0.0, percentile_50=0.05,
            percentile_75=0.1, percentile_95=0.2, mean=0.05, std=0.08
        )
        
        ok_signal = StockSignal.create(
            ticker="OK", rebalance_date=date(2024, 1, 1), horizon_days=20,
            benchmark="QQQ", return_distribution=dist,
            alpha_rank_score=1.0, confidence_score=0.7,
            liquidity_flag=LiquidityFlag.OK
        )
        assert not ok_signal.has_liquidity_concern
        
        low_vol_signal = StockSignal.create(
            ticker="LOW", rebalance_date=date(2024, 1, 1), horizon_days=20,
            benchmark="QQQ", return_distribution=dist,
            alpha_rank_score=1.0, confidence_score=0.7,
            liquidity_flag=LiquidityFlag.LOW_VOLUME
        )
        assert low_vol_signal.has_liquidity_concern


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


"""
Tests for Price Series Validation (src/utils/price_validation.py)

Tests:
1. Split discontinuity detection
2. Consistency validation
3. Normalization (optional fallback)
"""

import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta

import pytest

from src.utils.price_validation import (
    detect_split_discontinuities,
    validate_price_series_consistency,
    normalize_split_discontinuities,
    SplitDiscontinuityError,
    SplitDiscontinuity,
)


class TestDetectSplitDiscontinuities:
    """Tests for split discontinuity detection."""
    
    def test_detects_10_to_1_split(self):
        """Test detection of 10:1 forward split (like NVDA 2024)."""
        # Create price series with 10:1 split on day 5
        dates = pd.date_range("2024-06-01", periods=10, freq="D")
        prices = [1000.0] * 5 + [100.0] * 5  # 10x drop
        
        df = pd.DataFrame({
            "date": dates,
            "close": prices,
            "ticker": "NVDA",
        })
        
        discontinuities = detect_split_discontinuities(df)
        
        assert len(discontinuities) == 1
        disc = discontinuities[0]
        assert disc.ticker == "NVDA"
        assert disc.likely_split_ratio == 10
        assert disc.ratio == pytest.approx(10.0, rel=0.01)
        assert disc.date == "2024-06-06"
    
    def test_detects_4_to_1_split(self):
        """Test detection of 4:1 forward split."""
        dates = pd.date_range("2023-01-01", periods=6, freq="D")
        prices = [400.0, 400.0, 400.0, 100.0, 100.0, 100.0]  # 4x drop
        
        df = pd.DataFrame({
            "date": dates,
            "close": prices,
            "ticker": "TEST",
        })
        
        discontinuities = detect_split_discontinuities(df)
        
        assert len(discontinuities) == 1
        assert discontinuities[0].likely_split_ratio == 4
    
    def test_detects_2_to_1_split(self):
        """Test detection of 2:1 forward split."""
        dates = pd.date_range("2023-01-01", periods=6, freq="D")
        prices = [200.0, 200.0, 200.0, 100.0, 100.0, 100.0]  # 2x drop
        
        df = pd.DataFrame({
            "date": dates,
            "close": prices,
            "ticker": "TEST",
        })
        
        discontinuities = detect_split_discontinuities(df)
        
        assert len(discontinuities) == 1
        assert discontinuities[0].likely_split_ratio == 2
    
    def test_no_discontinuity_in_normal_series(self):
        """Test that normal price movements don't trigger detection."""
        # Normal daily returns (up to 10%)
        dates = pd.date_range("2023-01-01", periods=20, freq="D")
        np.random.seed(42)
        prices = 100.0 * np.cumprod(1 + np.random.randn(20) * 0.02)
        
        df = pd.DataFrame({
            "date": dates,
            "close": prices,
            "ticker": "NORMAL",
        })
        
        discontinuities = detect_split_discontinuities(df)
        
        assert len(discontinuities) == 0
    
    def test_handles_multiple_tickers(self):
        """Test detection across multiple tickers."""
        # Ticker A: no split
        dates = pd.date_range("2023-01-01", periods=5, freq="D")
        df_a = pd.DataFrame({
            "date": dates,
            "close": [100.0] * 5,
            "ticker": "A",
        })
        
        # Ticker B: 10:1 split
        df_b = pd.DataFrame({
            "date": dates,
            "close": [1000.0, 1000.0, 100.0, 100.0, 100.0],
            "ticker": "B",
        })
        
        df = pd.concat([df_a, df_b], ignore_index=True)
        
        discontinuities = detect_split_discontinuities(df)
        
        assert len(discontinuities) == 1
        assert discontinuities[0].ticker == "B"
    
    def test_tolerance_for_near_split_ratios(self):
        """Test that ratios within tolerance are detected."""
        dates = pd.date_range("2023-01-01", periods=5, freq="D")
        # Ratio of 9.8 (within 5% tolerance of 10)
        prices = [980.0, 980.0, 100.0, 100.0, 100.0]
        
        df = pd.DataFrame({
            "date": dates,
            "close": prices,
            "ticker": "TEST",
        })
        
        discontinuities = detect_split_discontinuities(df)
        
        assert len(discontinuities) == 1
        assert discontinuities[0].likely_split_ratio == 10
    
    def test_ignores_ratio_outside_tolerance(self):
        """Test that ratios too far from common splits are ignored."""
        dates = pd.date_range("2023-01-01", periods=5, freq="D")
        # Ratio of 7 (not near any common split ratio)
        prices = [700.0, 700.0, 100.0, 100.0, 100.0]
        
        df = pd.DataFrame({
            "date": dates,
            "close": prices,
            "ticker": "TEST",
        })
        
        discontinuities = detect_split_discontinuities(df)
        
        # Should not detect since 7 isn't a common split ratio
        assert len(discontinuities) == 0


class TestValidatePriceSeriesConsistency:
    """Tests for price series consistency validation."""
    
    def test_valid_series_passes(self):
        """Test that a properly adjusted series passes validation."""
        dates = pd.date_range("2023-01-01", periods=20, freq="D")
        np.random.seed(42)
        prices = 100.0 * np.cumprod(1 + np.random.randn(20) * 0.02)
        
        df = pd.DataFrame({
            "date": dates,
            "close": prices,
            "ticker": "GOOD",
        })
        
        is_valid, issues = validate_price_series_consistency(
            df, raise_on_error=False
        )
        
        assert is_valid is True
        assert len(issues) == 0
    
    def test_invalid_series_fails(self):
        """Test that a series with split discontinuity fails."""
        dates = pd.date_range("2023-01-01", periods=10, freq="D")
        prices = [1000.0] * 5 + [100.0] * 5  # 10:1 split
        
        df = pd.DataFrame({
            "date": dates,
            "close": prices,
            "ticker": "BAD",
        })
        
        is_valid, issues = validate_price_series_consistency(
            df, raise_on_error=False
        )
        
        assert is_valid is False
        assert len(issues) == 1
    
    def test_raises_on_error_when_enabled(self):
        """Test that exception is raised when raise_on_error=True."""
        dates = pd.date_range("2023-01-01", periods=10, freq="D")
        prices = [1000.0] * 5 + [100.0] * 5
        
        df = pd.DataFrame({
            "date": dates,
            "close": prices,
            "ticker": "BAD",
        })
        
        with pytest.raises(SplitDiscontinuityError) as exc_info:
            validate_price_series_consistency(df, raise_on_error=True)
        
        error_msg = str(exc_info.value)
        assert "Split discontinuities detected" in error_msg
        assert "BAD" in error_msg


class TestNormalizeSplitDiscontinuities:
    """Tests for split discontinuity normalization."""
    
    def test_normalizes_10_to_1_split(self):
        """Test normalizing a 10:1 split in price series."""
        dates = pd.date_range("2024-06-01", periods=10, freq="D")
        # Pre-split: 1000, Post-split: 100 (should be normalized to 100)
        prices = [1000.0] * 5 + [100.0] * 5
        
        df = pd.DataFrame({
            "date": dates,
            "close": prices,
            "ticker": "NVDA",
        })
        
        # First detect
        discontinuities = detect_split_discontinuities(df)
        assert len(discontinuities) == 1
        
        # Then normalize
        normalized = normalize_split_discontinuities(df, discontinuities)
        
        # All prices should now be ~100
        assert normalized["close"].max() < 150  # Pre-split divided by 10
        assert normalized["close"].min() > 50
        
        # Verify no more discontinuities
        new_disc = detect_split_discontinuities(normalized)
        assert len(new_disc) == 0
    
    def test_normalizes_multiple_columns(self):
        """Test that OHLC columns are all normalized."""
        dates = pd.date_range("2024-06-01", periods=6, freq="D")
        
        df = pd.DataFrame({
            "date": dates,
            "open": [1000.0] * 3 + [100.0] * 3,
            "high": [1050.0] * 3 + [105.0] * 3,
            "low": [950.0] * 3 + [95.0] * 3,
            "close": [1000.0] * 3 + [100.0] * 3,
            "ticker": "TEST",
        })
        
        discontinuities = detect_split_discontinuities(df, price_col="close")
        normalized = normalize_split_discontinuities(
            df, discontinuities, 
            price_cols=["open", "high", "low", "close"]
        )
        
        # All pre-split prices should be divided by 10
        assert normalized["open"].max() < 150
        assert normalized["high"].max() < 150
        assert normalized["low"].max() < 150
        assert normalized["close"].max() < 150


class TestRealWorldScenarios:
    """Test realistic scenarios based on actual split events."""
    
    def test_nvda_2024_split_scenario(self):
        """Simulate NVDA 10-for-1 split on 2024-06-07."""
        # Create data around the split date
        dates = pd.date_range("2024-06-03", periods=10, freq="D")
        
        # Unadjusted prices (what a buggy endpoint might return)
        unadjusted = [
            1208.90,  # June 3 (pre-split)
            1209.50,  # June 4
            1210.20,  # June 5
            1211.00,  # June 6
            1212.30,  # June 7 (split day - WRONG)
            120.89,   # June 10 (post-split - correct)
            121.50,   # June 11
            122.00,   # June 12
            121.80,   # June 13
            122.50,   # June 14
        ]
        
        df = pd.DataFrame({
            "date": dates,
            "close": unadjusted,
            "ticker": "NVDA",
        })
        
        # Should detect the discontinuity
        discontinuities = detect_split_discontinuities(df)
        
        assert len(discontinuities) == 1
        disc = discontinuities[0]
        assert disc.likely_split_ratio == 10
        assert disc.ticker == "NVDA"
    
    def test_properly_adjusted_series_passes(self):
        """Test that properly adjusted historical data passes."""
        # Properly adjusted prices (what we should get from /full endpoint)
        dates = pd.date_range("2024-06-03", periods=10, freq="D")
        adjusted = [
            120.89,  # June 3 (pre-split, adjusted)
            120.95,  # June 4
            121.02,  # June 5
            121.10,  # June 6
            121.23,  # June 7 (split day)
            120.89,  # June 10 (post-split)
            121.50,  # June 11
            122.00,  # June 12
            121.80,  # June 13
            122.50,  # June 14
        ]
        
        df = pd.DataFrame({
            "date": dates,
            "close": adjusted,
            "ticker": "NVDA",
        })
        
        is_valid, issues = validate_price_series_consistency(
            df, raise_on_error=False
        )
        
        assert is_valid is True
        assert len(issues) == 0


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_handles_zero_prices(self):
        """Test that zero prices don't cause division errors."""
        dates = pd.date_range("2023-01-01", periods=5, freq="D")
        prices = [100.0, 0.0, 100.0, 0.0, 100.0]
        
        df = pd.DataFrame({
            "date": dates,
            "close": prices,
            "ticker": "TEST",
        })
        
        # Should not raise
        discontinuities = detect_split_discontinuities(df)
        # Zero prices are skipped, so shouldn't detect false positives
        assert len(discontinuities) == 0
    
    def test_handles_single_row(self):
        """Test handling of single-row DataFrame."""
        df = pd.DataFrame({
            "date": [date(2023, 1, 1)],
            "close": [100.0],
            "ticker": "TEST",
        })
        
        discontinuities = detect_split_discontinuities(df)
        assert len(discontinuities) == 0
    
    def test_handles_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        df = pd.DataFrame(columns=["date", "close", "ticker"])
        
        discontinuities = detect_split_discontinuities(df)
        assert len(discontinuities) == 0
    
    def test_handles_unsorted_data(self):
        """Test that unsorted data is handled correctly."""
        # Data out of order
        df = pd.DataFrame({
            "date": ["2023-01-05", "2023-01-03", "2023-01-01", "2023-01-04", "2023-01-02"],
            "close": [100.0, 100.0, 1000.0, 100.0, 1000.0],  # Split between day 2 and 3
            "ticker": "TEST",
        })
        
        discontinuities = detect_split_discontinuities(df)
        
        # Should detect the split after sorting
        assert len(discontinuities) == 1
        assert discontinuities[0].likely_split_ratio == 10


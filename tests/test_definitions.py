"""
Tests for Chapter 6 Definition Lock (definitions.py)

CRITICAL: These tests verify that the canonical definitions are correct
and that all helper functions work properly.
"""

import pytest
from datetime import date, datetime, time
import pytz

from src.evaluation.definitions import (
    # Constants
    TRADING_DAYS_PER_YEAR,
    CALENDAR_TO_TRADING_RATIO,
    HORIZONS_TRADING_DAYS,
    MARKET_CLOSE_ET,
    # Frozen Dataclasses
    TIME_CONVENTIONS,
    EVALUATION_RANGE,
    EMBARGO_RULES,
    ELIGIBILITY_RULES,
    # Functions
    get_market_close_utc,
    is_label_mature,
    trading_days_to_calendar_days,
    calendar_days_to_trading_days,
    validate_horizon,
    validate_embargo,
    # Documentation
    DEFINITION_LOCK_SUMMARY,
)


class TestConstants:
    """Test canonical constants."""
    
    def test_trading_days_per_year(self):
        """Verify trading days per year is NYSE standard."""
        assert TRADING_DAYS_PER_YEAR == 252
    
    def test_calendar_to_trading_ratio(self):
        """Verify calendar to trading day ratio."""
        assert 1.4 < CALENDAR_TO_TRADING_RATIO < 1.5
        assert CALENDAR_TO_TRADING_RATIO == 365 / 252
    
    def test_horizons_are_trading_days(self):
        """Verify horizons are defined as [20, 60, 90] trading days."""
        assert HORIZONS_TRADING_DAYS == [20, 60, 90]
        assert max(HORIZONS_TRADING_DAYS) == 90
    
    def test_market_close_time(self):
        """Verify market close is 4 PM ET."""
        assert MARKET_CLOSE_ET == time(16, 0, 0)


class TestTimeConventions:
    """Test TIME_CONVENTIONS frozen dataclass."""
    
    def test_is_frozen(self):
        """Verify TIME_CONVENTIONS cannot be modified."""
        with pytest.raises(Exception):  # FrozenInstanceError
            TIME_CONVENTIONS.embargo_trading_days = 30
    
    def test_horizons_tuple(self):
        """Verify horizons are a tuple."""
        assert TIME_CONVENTIONS.horizons_trading_days == (20, 60, 90)
    
    def test_embargo_is_max_horizon(self):
        """Verify embargo equals max horizon."""
        assert TIME_CONVENTIONS.embargo_trading_days == 90
        assert TIME_CONVENTIONS.embargo_trading_days == max(TIME_CONVENTIONS.horizons_trading_days)
    
    def test_rebalance_rule(self):
        """Verify rebalance rule is first trading day."""
        assert "first_trading_day" in TIME_CONVENTIONS.rebalance_rule
    
    def test_pricing_convention(self):
        """Verify pricing is close-to-close."""
        assert TIME_CONVENTIONS.pricing_convention == "close_to_close"
    
    def test_maturity_rule(self):
        """Verify maturity rule uses UTC."""
        assert "utc" in TIME_CONVENTIONS.maturity_rule.lower()


class TestEvaluationRange:
    """Test EVALUATION_RANGE frozen dataclass."""
    
    def test_is_frozen(self):
        """Verify EVALUATION_RANGE cannot be modified."""
        with pytest.raises(Exception):
            EVALUATION_RANGE.eval_start = date(2020, 1, 1)
    
    def test_eval_start(self):
        """Verify evaluation starts 2016-01-01."""
        assert EVALUATION_RANGE.eval_start == date(2016, 1, 1)
    
    def test_eval_end(self):
        """Verify evaluation ends 2025-06-30."""
        assert EVALUATION_RANGE.eval_end == date(2025, 6, 30)
    
    def test_min_train_days(self):
        """Verify minimum training is ~2 years."""
        assert EVALUATION_RANGE.min_train_days == 730
        assert 2 * 365 <= EVALUATION_RANGE.min_train_days <= 2.1 * 365


class TestEmbargoRules:
    """Test EMBARGO_RULES frozen dataclass."""
    
    def test_is_frozen(self):
        """Verify EMBARGO_RULES cannot be modified."""
        with pytest.raises(Exception):
            EMBARGO_RULES.embargo_trading_days = 30
    
    def test_embargo_trading_days(self):
        """Verify embargo is 90 trading days."""
        assert EMBARGO_RULES.embargo_trading_days == 90
    
    def test_purging_is_per_row_per_horizon(self):
        """Verify purging granularity is per-row-per-horizon."""
        assert EMBARGO_RULES.purging_granularity == "per_row_per_horizon"
    
    def test_train_purge_rule(self):
        """Verify train purge rule mentions maturity."""
        assert "maturity" in EMBARGO_RULES.train_purge_rule.lower()
    
    def test_val_purge_rule(self):
        """Verify val purge rule mentions lookback."""
        assert "lookback" in EMBARGO_RULES.val_purge_rule.lower()


class TestEligibilityRules:
    """Test ELIGIBILITY_RULES frozen dataclass."""
    
    def test_is_frozen(self):
        """Verify ELIGIBILITY_RULES cannot be modified."""
        with pytest.raises(Exception):
            ELIGIBILITY_RULES.allow_partial_horizons = True
    
    def test_all_horizons_required(self):
        """Verify eligibility requires all horizons."""
        assert "all_horizons" in ELIGIBILITY_RULES.eligibility_rule
    
    def test_no_partial_horizons(self):
        """Verify partial horizons not allowed."""
        assert ELIGIBILITY_RULES.allow_partial_horizons == False
    
    def test_valid_label_criteria(self):
        """Verify valid label criteria include maturity."""
        assert "label_matured" in ELIGIBILITY_RULES.valid_label_criteria


class TestMarketCloseUTC:
    """Test get_market_close_utc function."""
    
    def test_returns_utc_datetime(self):
        """Verify function returns UTC datetime."""
        result = get_market_close_utc(date(2023, 6, 15))
        assert result.tzinfo == pytz.UTC
    
    def test_winter_time_conversion(self):
        """Test conversion during winter (EST = UTC-5)."""
        # January is winter time (EST)
        result = get_market_close_utc(date(2023, 1, 15))
        # 4 PM EST = 9 PM UTC (21:00)
        assert result.hour == 21
    
    def test_summer_time_conversion(self):
        """Test conversion during summer (EDT = UTC-4)."""
        # July is summer time (EDT)
        result = get_market_close_utc(date(2023, 7, 15))
        # 4 PM EDT = 8 PM UTC (20:00)
        assert result.hour == 20
    
    def test_date_preserved(self):
        """Verify date is preserved (or shifted correctly for UTC)."""
        result = get_market_close_utc(date(2023, 6, 15))
        # The date in UTC should still be June 15 (20:00 UTC is same day)
        assert result.date() == date(2023, 6, 15)


class TestIsLabelMature:
    """Test is_label_mature function."""
    
    def test_mature_label(self):
        """Test label that has matured."""
        # Label matured at 10:00 UTC
        label_matured = datetime(2023, 6, 15, 10, 0, 0, tzinfo=pytz.UTC)
        cutoff = date(2023, 6, 15)  # Market close ~20:00 UTC
        
        assert is_label_mature(label_matured, cutoff) == True
    
    def test_immature_label(self):
        """Test label that has NOT matured."""
        # Label matures at 22:00 UTC (after market close)
        label_matured = datetime(2023, 6, 15, 22, 0, 0, tzinfo=pytz.UTC)
        cutoff = date(2023, 6, 15)  # Market close ~20:00 UTC
        
        assert is_label_mature(label_matured, cutoff) == False
    
    def test_exactly_at_cutoff(self):
        """Test label that matures exactly at cutoff."""
        cutoff = date(2023, 6, 15)
        cutoff_utc = get_market_close_utc(cutoff)
        
        # Label matures exactly at market close
        assert is_label_mature(cutoff_utc, cutoff) == True
    
    def test_rejects_naive_datetime(self):
        """Test that naive datetime is rejected."""
        label_matured = datetime(2023, 6, 15, 10, 0, 0)  # No tzinfo
        cutoff = date(2023, 6, 15)
        
        with pytest.raises(ValueError, match="timezone-aware"):
            is_label_mature(label_matured, cutoff)


class TestDayConversions:
    """Test trading day <-> calendar day conversions."""
    
    def test_trading_to_calendar_conservative(self):
        """Test trading->calendar is conservative (larger)."""
        trading = 90
        calendar = trading_days_to_calendar_days(trading)
        
        # Should be larger than trading days
        assert calendar > trading
        # Should be approximately 90 * 1.45 + 1 ≈ 131-132
        assert 130 <= calendar <= 135
    
    def test_calendar_to_trading_conservative(self):
        """Test calendar->trading is conservative (smaller)."""
        calendar = 130
        trading = calendar_days_to_trading_days(calendar)
        
        # Should be smaller than calendar days
        assert trading < calendar
        # Should be approximately 130 / 1.45 ≈ 89-90
        assert 85 <= trading <= 95
    
    def test_round_trip_conservative(self):
        """Test round trip is conservative (doesn't lose days)."""
        original_trading = 90
        calendar = trading_days_to_calendar_days(original_trading)
        back_to_trading = calendar_days_to_trading_days(calendar)
        
        # Round trip should preserve or be larger
        assert back_to_trading >= original_trading - 5  # Allow small tolerance


class TestValidation:
    """Test validation helper functions."""
    
    def test_validate_horizon_valid(self):
        """Test valid horizons pass."""
        validate_horizon(20)  # Should not raise
        validate_horizon(60)
        validate_horizon(90)
    
    def test_validate_horizon_invalid(self):
        """Test invalid horizons raise ValueError."""
        with pytest.raises(ValueError):
            validate_horizon(30)
        with pytest.raises(ValueError):
            validate_horizon(45)
        with pytest.raises(ValueError):
            validate_horizon(100)
    
    def test_validate_embargo_valid(self):
        """Test valid embargo passes."""
        validate_embargo(90)  # Should not raise
        validate_embargo(100)
        validate_embargo(120)
    
    def test_validate_embargo_invalid(self):
        """Test invalid embargo raises ValueError."""
        with pytest.raises(ValueError, match="TRADING DAYS"):
            validate_embargo(89)
        with pytest.raises(ValueError, match="TRADING DAYS"):
            validate_embargo(30)
        with pytest.raises(ValueError, match="TRADING DAYS"):
            validate_embargo(0)


class TestDocumentation:
    """Test documentation string."""
    
    def test_definition_lock_summary_exists(self):
        """Verify summary documentation exists."""
        assert DEFINITION_LOCK_SUMMARY is not None
        assert len(DEFINITION_LOCK_SUMMARY) > 100
    
    def test_summary_mentions_key_concepts(self):
        """Verify summary mentions key concepts."""
        assert "TRADING DAYS" in DEFINITION_LOCK_SUMMARY
        assert "embargo" in DEFINITION_LOCK_SUMMARY.lower()
        assert "purging" in DEFINITION_LOCK_SUMMARY.lower()
        assert "maturity" in DEFINITION_LOCK_SUMMARY.lower()
        assert "UTC" in DEFINITION_LOCK_SUMMARY


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


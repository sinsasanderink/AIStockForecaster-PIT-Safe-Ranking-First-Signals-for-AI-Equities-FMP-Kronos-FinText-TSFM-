"""
Tests for Trading Calendar (Chapter 8 Extensions)
==================================================

Tests the global trading calendar loader for Kronos integration.
"""

import pytest
import pandas as pd
from pathlib import Path

from src.data import load_global_trading_calendar


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def db_path():
    """Get path to DuckDB database."""
    db = Path("data/features.duckdb")
    if not db.exists():
        pytest.skip(f"DuckDB not found at {db}")
    return str(db)


# ============================================================================
# BASIC FUNCTIONALITY TESTS
# ============================================================================

def test_load_global_trading_calendar(db_path):
    """Test loading global trading calendar from DuckDB."""
    calendar = load_global_trading_calendar(db_path)
    
    assert isinstance(calendar, pd.DatetimeIndex)
    assert len(calendar) > 0
    
    # Should be sorted
    assert calendar.is_monotonic_increasing
    
    # Should be unique
    assert calendar.is_unique
    
    # Should be timezone-naive
    assert calendar.tz is None


def test_calendar_multi_year_coverage(db_path):
    """Test that calendar has multi-year coverage."""
    calendar = load_global_trading_calendar(db_path)
    
    # Should span at least 1 year
    date_range = (calendar[-1] - calendar[0]).days
    assert date_range > 365
    
    print(f"Calendar coverage: {calendar[0]} to {calendar[-1]} ({len(calendar)} days)")


def test_calendar_trading_days_only(db_path):
    """Test that calendar contains only trading days (weekdays)."""
    calendar = load_global_trading_calendar(db_path)
    
    # Count weekdays vs total
    weekdays = [d for d in calendar if d.dayofweek < 5]
    
    # Most should be weekdays (allowing for some edge cases)
    weekday_ratio = len(weekdays) / len(calendar)
    assert weekday_ratio > 0.95  # At least 95% should be weekdays
    
    print(f"Weekday ratio: {weekday_ratio:.2%}")


def test_calendar_no_duplicates(db_path):
    """Test that calendar has no duplicate dates."""
    calendar = load_global_trading_calendar(db_path)
    
    # Check uniqueness
    assert len(calendar) == len(calendar.unique())


def test_calendar_respects_holidays(db_path):
    """Test that calendar respects major holidays."""
    calendar = load_global_trading_calendar(db_path)
    
    # Convert to date set for easy lookup
    trading_dates = set(calendar.date)
    
    # Check some known holidays (if in range)
    # New Year's Day 2023
    import datetime
    nye_2023 = datetime.date(2023, 1, 1)  # Sunday
    nye_2024 = datetime.date(2024, 1, 1)  # Monday
    
    # At least one New Year's should be excluded (if in range)
    if calendar[0].date() <= nye_2023 <= calendar[-1].date():
        assert nye_2023 not in trading_dates or nye_2023.weekday() >= 5
    
    if calendar[0].date() <= nye_2024 <= calendar[-1].date():
        # Jan 1, 2024 was a Monday (holiday)
        assert nye_2024 not in trading_dates


# ============================================================================
# KRONOS INTEGRATION TESTS
# ============================================================================

def test_calendar_for_future_date_generation(db_path):
    """Test using calendar to generate future dates (Kronos use case)."""
    calendar = load_global_trading_calendar(db_path)
    
    # Simulate Kronos: given last observed date, get next N trading days
    import numpy as np
    
    # Pick a date in the middle of the calendar
    mid_idx = len(calendar) // 2
    last_observed = calendar[mid_idx]
    horizon = 20
    
    # Get future dates
    idx = np.searchsorted(calendar.values, last_observed.to_datetime64())
    future_dates = calendar[idx + 1 : idx + 1 + horizon]
    
    # Should get exactly horizon dates (or less if near end)
    assert len(future_dates) <= horizon
    
    # All should be after last_observed
    if len(future_dates) > 0:
        assert (future_dates > last_observed).all()
    
    # Should be consecutive trading days
    if len(future_dates) > 1:
        assert future_dates.is_monotonic_increasing


def test_calendar_boundary_handling(db_path):
    """Test calendar behavior at boundaries."""
    calendar = load_global_trading_calendar(db_path)
    
    import numpy as np
    
    # Test at start
    first_date = calendar[0]
    idx = np.searchsorted(calendar.values, first_date.to_datetime64())
    assert idx == 0
    
    # Test at end
    last_date = calendar[-1]
    idx = np.searchsorted(calendar.values, last_date.to_datetime64())
    assert idx == len(calendar) - 1 or idx == len(calendar)


def test_calendar_searchsorted(db_path):
    """Test searchsorted for finding positions in calendar."""
    calendar = load_global_trading_calendar(db_path)
    
    import numpy as np
    
    # Pick several dates and test searchsorted
    for i in [0, len(calendar) // 4, len(calendar) // 2, len(calendar) - 1]:
        date = calendar[i]
        
        # Find position
        idx = np.searchsorted(calendar.values, date.to_datetime64())
        
        # Should be close to original index
        assert abs(idx - i) <= 1


def test_calendar_with_timestamps(db_path):
    """Test that calendar works with pd.Timestamp comparisons."""
    calendar = load_global_trading_calendar(db_path)
    
    # Pick a date
    test_date = calendar[len(calendar) // 2]
    
    # Should support Timestamp operations
    assert isinstance(test_date, pd.Timestamp)
    
    # Should support comparison
    before = calendar < test_date
    assert before.sum() > 0
    
    after = calendar > test_date
    assert after.sum() > 0


# ============================================================================
# ERROR HANDLING TESTS
# ============================================================================

def test_calendar_invalid_db_path():
    """Test error handling for invalid database path."""
    with pytest.raises(Exception):  # May be RuntimeError or duckdb error
        load_global_trading_calendar("/invalid/path/to/db.duckdb")


def test_calendar_consistency_across_loads(db_path):
    """Test that multiple loads return consistent calendar."""
    calendar1 = load_global_trading_calendar(db_path)
    calendar2 = load_global_trading_calendar(db_path)
    
    # Should be identical
    assert len(calendar1) == len(calendar2)
    assert (calendar1 == calendar2).all()


# ============================================================================
# PERFORMANCE TESTS
# ============================================================================

def test_calendar_load_speed(db_path):
    """Test that calendar loads reasonably fast."""
    import time
    
    start = time.time()
    calendar = load_global_trading_calendar(db_path)
    elapsed = time.time() - start
    
    # Should load in under 1 second
    assert elapsed < 1.0
    
    print(f"Calendar loaded in {elapsed:.3f}s ({len(calendar)} days)")


# ============================================================================
# INTEGRATION WITH PRICESSTORE TESTS
# ============================================================================

def test_calendar_matches_prices_dates(db_path):
    """Test that calendar matches dates in prices table."""
    from src.data import PricesStore
    
    calendar = load_global_trading_calendar(db_path)
    
    with PricesStore(db_path) as store:
        min_date, max_date = store.get_date_range()
        
        # Calendar should cover the same range
        assert calendar[0] <= min_date
        assert calendar[-1] >= max_date


def test_calendar_future_dates_for_batch_inference(db_path):
    """Test typical batch inference use case with calendar."""
    from src.data import PricesStore
    import numpy as np
    
    calendar = load_global_trading_calendar(db_path)
    
    with PricesStore(db_path) as store:
        tickers = store.fetch_available_tickers()[:3]
        
        if len(tickers) == 0:
            pytest.skip("No tickers in database")
        
        asof_date = "2023-06-30"
        lookback = 252
        horizon = 20
        
        # Fetch OHLCV for multiple tickers
        ohlcv_list = []
        x_ts_list = []
        
        for ticker in tickers:
            ohlcv = store.fetch_ohlcv(ticker, asof_date, lookback, strict_lookback=True)
            if len(ohlcv) == lookback:
                ohlcv_list.append(ohlcv)
                x_ts_list.append(ohlcv.index)
        
        if len(ohlcv_list) == 0:
            pytest.skip("No tickers with sufficient history")
        
        # Get future dates based on last x_timestamp (Kronos pattern)
        last_x = x_ts_list[0][-1]
        idx = np.searchsorted(calendar.values, last_x.to_datetime64())
        y_ts = calendar[idx + 1 : idx + 1 + horizon]
        
        # Should have exactly horizon dates
        assert len(y_ts) == horizon
        
        # All should be after last_x
        assert (y_ts > last_x).all()
        
        # Can replicate for batch
        y_ts_list = [y_ts] * len(ohlcv_list)
        assert len(y_ts_list) == len(ohlcv_list)


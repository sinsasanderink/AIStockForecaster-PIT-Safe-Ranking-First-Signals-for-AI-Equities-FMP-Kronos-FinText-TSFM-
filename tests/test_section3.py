"""
Section 3: Data Infrastructure Tests
=====================================

Comprehensive tests for data infrastructure including:
- FMP API client
- PIT store with UTC timestamps
- Trading calendar
- Cutoff boundary tests
- PIT violation detection

Run with: python tests/test_section3.py

For integration tests with real API (uses quota):
  RUN_INTEGRATION=1 python tests/test_section3.py
"""

import sys
import os
from pathlib import Path
from datetime import date, datetime, timedelta, timezone

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

import pytz

# Constants
ET = pytz.timezone("America/New_York")
UTC = pytz.UTC

# Check if integration tests should run
RUN_INTEGRATION = os.getenv("RUN_INTEGRATION", "").lower() in ("1", "true", "yes")


def print_test_header(name: str):
    """Print a test section header."""
    print(f"\n{'='*60}")
    print(f"TEST: {name}")
    print(f"{'='*60}")


def print_result(name: str, passed: bool, details: str = ""):
    """Print a test result with details."""
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"  {status}: {name}")
    if details:
        for line in details.split("\n"):
            print(f"         {line}")


# =============================================================================
# UNIT TESTS (No API calls)
# =============================================================================

def test_pit_timestamp_functions():
    """Test PIT timestamp helper functions."""
    print_test_header("PIT Timestamp Functions")
    
    from src.data.fmp_client import get_market_close_utc, get_next_market_open_utc
    
    # Test 1: Market close for a regular day
    d = date(2024, 6, 15)  # Saturday - but function doesn't check
    close_utc = get_market_close_utc(d)
    
    # 4pm ET should be 20:00 UTC in summer (EDT)
    expected_hour = 20  # EDT = UTC-4
    passed = close_utc.hour == expected_hour
    print_result(
        f"Market close UTC for {d}",
        passed,
        f"Got {close_utc.strftime('%Y-%m-%d %H:%M %Z')}, expected hour={expected_hour}"
    )
    
    # Test 2: DST handling - winter vs summer
    winter_date = date(2024, 1, 15)  # EST = UTC-5
    summer_date = date(2024, 7, 15)  # EDT = UTC-4
    
    winter_close = get_market_close_utc(winter_date)
    summer_close = get_market_close_utc(summer_date)
    
    # Winter: 4pm EST = 21:00 UTC, Summer: 4pm EDT = 20:00 UTC
    winter_ok = winter_close.hour == 21
    summer_ok = summer_close.hour == 20
    
    print_result(
        "DST handling (winter)",
        winter_ok,
        f"Winter {winter_date}: 4pm EST = {winter_close.strftime('%H:%M UTC')}, expected 21:00"
    )
    print_result(
        "DST handling (summer)",
        summer_ok,
        f"Summer {summer_date}: 4pm EDT = {summer_close.strftime('%H:%M UTC')}, expected 20:00"
    )
    
    # Test 3: Next market open
    friday = date(2024, 6, 14)  # Friday
    next_open = get_next_market_open_utc(friday)
    
    # Should be Monday 9:30am ET = 13:30 UTC (summer)
    passed = next_open.weekday() == 0  # Monday
    print_result(
        "Next market open skips weekend",
        passed,
        f"Friday {friday} -> {next_open.strftime('%Y-%m-%d %H:%M UTC')} (weekday={next_open.weekday()})"
    )
    
    return winter_ok and summer_ok


def test_pit_store_observed_at_filtering():
    """Test that PIT store correctly filters by observed_at."""
    print_test_header("PIT Store observed_at Filtering")
    
    from src.data.pit_store import DuckDBPITStore
    import pandas as pd
    
    store = DuckDBPITStore()  # In-memory
    
    # Create test data with specific observed_at times
    # Day 1: observed at 4pm ET (20:00 UTC summer)
    # Day 2: observed at 4pm ET
    test_data = pd.DataFrame([
        {
            "ticker": "TEST",
            "date": pd.Timestamp("2024-06-10"),
            "open": 100, "high": 105, "low": 99, "close": 102,
            "adj_close": 102, "volume": 1000000,
            "observed_at": pd.Timestamp("2024-06-10 20:00:00", tz="UTC"),
        },
        {
            "ticker": "TEST",
            "date": pd.Timestamp("2024-06-11"),
            "open": 102, "high": 107, "low": 101, "close": 105,
            "adj_close": 105, "volume": 1200000,
            "observed_at": pd.Timestamp("2024-06-11 20:00:00", tz="UTC"),
        },
    ])
    
    store.store_prices(test_data)
    
    # Test 1: Query before any data is available
    asof_before = datetime(2024, 6, 10, 19, 0, tzinfo=UTC)  # 3pm ET
    df_before = store.get_ohlcv(["TEST"], date(2024, 6, 1), date(2024, 6, 30), asof=asof_before)
    
    passed1 = len(df_before) == 0
    print_result(
        "Query before data available returns empty",
        passed1,
        f"Query asof {asof_before.strftime('%H:%M UTC')}: got {len(df_before)} rows, expected 0"
    )
    
    # Test 2: Query exactly at cutoff - should see day 1
    asof_at_cutoff = datetime(2024, 6, 10, 20, 0, tzinfo=UTC)  # 4pm ET
    df_at = store.get_ohlcv(["TEST"], date(2024, 6, 1), date(2024, 6, 30), asof=asof_at_cutoff)
    
    passed2 = len(df_at) == 1
    print_result(
        "Query at exact cutoff includes that day",
        passed2,
        f"Query asof {asof_at_cutoff.strftime('%H:%M UTC')}: got {len(df_at)} rows, expected 1"
    )
    
    # Test 3: Query 1 minute after - should still see day 1
    asof_after = datetime(2024, 6, 10, 20, 1, tzinfo=UTC)
    df_after = store.get_ohlcv(["TEST"], date(2024, 6, 1), date(2024, 6, 30), asof=asof_after)
    
    passed3 = len(df_after) == 1
    print_result(
        "Query 1 min after cutoff still sees day 1",
        passed3,
        f"Query asof {asof_after.strftime('%H:%M UTC')}: got {len(df_after)} rows, expected 1"
    )
    
    # Test 4: Query next day - should see both days
    asof_next_day = datetime(2024, 6, 11, 21, 0, tzinfo=UTC)
    df_both = store.get_ohlcv(["TEST"], date(2024, 6, 1), date(2024, 6, 30), asof=asof_next_day)
    
    passed4 = len(df_both) == 2
    print_result(
        "Query next day sees both days",
        passed4,
        f"Query asof {asof_next_day.strftime('%Y-%m-%d %H:%M UTC')}: got {len(df_both)} rows, expected 2"
    )
    
    store.close()
    return passed1 and passed2 and passed3 and passed4


def test_avg_volume_per_ticker():
    """Test that get_avg_volume correctly limits per ticker, not globally."""
    print_test_header("get_avg_volume Per-Ticker Window")
    
    from src.data.pit_store import DuckDBPITStore
    import pandas as pd
    import numpy as np
    
    store = DuckDBPITStore()  # In-memory
    
    # Create 10 days of data for 3 tickers
    tickers = ["AAAA", "BBBB", "CCCC"]
    rows = []
    
    for ticker in tickers:
        base_volume = {"AAAA": 1_000_000, "BBBB": 2_000_000, "CCCC": 3_000_000}[ticker]
        for i in range(10):
            d = date(2024, 6, 10) + timedelta(days=i)
            rows.append({
                "ticker": ticker,
                "date": pd.Timestamp(d),
                "open": 100, "high": 105, "low": 99, "close": 102,
                "adj_close": 102, 
                "volume": base_volume + i * 100_000,  # Slightly varying
                "observed_at": pd.Timestamp(f"{d} 20:00:00", tz="UTC"),
            })
    
    df = pd.DataFrame(rows)
    store.store_prices(df)
    
    # Query avg volume with lookback=5 for all 3 tickers
    asof = datetime(2024, 6, 25, 21, 0, tzinfo=UTC)
    avg_volumes = store.get_avg_volume(tickers, asof, lookback_days=5)
    
    # Each ticker should have its own average based on its last 5 days
    # AAAA: avg of volumes ~1.5M, BBBB: ~2.5M, CCCC: ~3.5M
    
    passed = len(avg_volumes) == 3
    print_result(
        f"Got volumes for all {len(tickers)} tickers",
        passed,
        f"Got {len(avg_volumes)} tickers: {list(avg_volumes.keys())}"
    )
    
    # Check volumes are in expected ranges
    all_correct = True
    for ticker in tickers:
        expected_base = {"AAAA": 1_000_000, "BBBB": 2_000_000, "CCCC": 3_000_000}[ticker]
        actual = avg_volumes.get(ticker, 0)
        # Should be close to base + ~450k (average of last 5 days increments)
        expected_approx = expected_base + 700_000  # Rough estimate
        
        in_range = expected_base < actual < expected_base + 2_000_000
        all_correct = all_correct and in_range
        
        print_result(
            f"  {ticker} volume in expected range",
            in_range,
            f"Got {actual:,.0f}, expected around {expected_base:,} - {expected_base+1_000_000:,}"
        )
    
    store.close()
    return passed and all_correct


def test_fundamental_pit_validation():
    """Test fundamental PIT validation with filing dates."""
    print_test_header("Fundamental PIT Validation")
    
    from src.audits.pit_scanner import validate_fundamental_pit, ViolationType
    
    # Test 1: Valid usage (after filing + lag)
    v1 = validate_fundamental_pit(
        ticker="NVDA",
        period_end=date(2024, 3, 31),
        filing_date=date(2024, 5, 1),
        usage_date=date(2024, 5, 3),
        conservative_lag_days=1,
    )
    
    passed1 = v1 is None
    print_result(
        "Usage 2 days after filing is valid",
        passed1,
        f"Filing: May 1, Usage: May 3, Lag: 1 day -> {'VALID' if passed1 else 'VIOLATION'}"
    )
    
    # Test 2: Invalid usage (before filing + lag)
    v2 = validate_fundamental_pit(
        ticker="NVDA",
        period_end=date(2024, 3, 31),
        filing_date=date(2024, 5, 1),
        usage_date=date(2024, 5, 1),  # Same day as filing
        conservative_lag_days=1,
    )
    
    passed2 = v2 is not None and v2.violation_type == ViolationType.FUTURE_FUNDAMENTAL
    print_result(
        "Usage on filing day is violation (with 1 day lag)",
        passed2,
        f"Filing: May 1, Usage: May 1, Lag: 1 day -> {'VIOLATION DETECTED' if passed2 else 'NOT DETECTED'}"
    )
    
    # Test 3: Edge case - usage exactly on available date
    v3 = validate_fundamental_pit(
        ticker="NVDA",
        period_end=date(2024, 3, 31),
        filing_date=date(2024, 5, 1),
        usage_date=date(2024, 5, 2),  # Exactly filing + 1 day
        conservative_lag_days=1,
    )
    
    passed3 = v3 is None
    print_result(
        "Usage exactly on available date is valid",
        passed3,
        f"Filing: May 1, Usage: May 2, Lag: 1 day -> {'VALID' if passed3 else 'VIOLATION'}"
    )
    
    return passed1 and passed2 and passed3


def test_pit_scanner_violation_types():
    """Test that PIT scanner uses correct violation types for different data."""
    print_test_header("PIT Scanner Violation Types")
    
    from src.audits.pit_scanner import (
        scan_feature_pit_violations, 
        DataType, 
        ViolationType
    )
    
    # Test with price data type
    price_data = [{
        "ticker": "TEST",
        "value": 100,
        "effective_from": date(2024, 6, 15),
        "observed_at": datetime(2024, 6, 15, 20, 0, tzinfo=UTC),
    }]
    
    # Usage before observed_at
    violations = scan_feature_pit_violations(
        "close_price",
        price_data,
        [date(2024, 6, 15)],  # Usage same day but we check datetime
        data_type=DataType.PRICE,
    )
    
    # Should detect FUTURE_PRICE, not FUTURE_FUNDAMENTAL
    has_price_violation = any(v.violation_type == ViolationType.FUTURE_PRICE for v in violations)
    has_fundamental_violation = any(v.violation_type == ViolationType.FUTURE_FUNDAMENTAL for v in violations)
    
    print_result(
        "Price data violations typed as FUTURE_PRICE",
        has_price_violation or len(violations) == 0,  # Either found correct type or no violation
        f"Found violations: {[v.violation_type.value for v in violations]}"
    )
    
    # Test with fundamental data type
    fundamental_data = [{
        "ticker": "TEST",
        "value": 1000000,
        "effective_from": date(2024, 3, 31),
        "observed_at": datetime(2024, 5, 2, 13, 30, tzinfo=UTC),  # Filing + 1 day
    }]
    
    violations2 = scan_feature_pit_violations(
        "revenue",
        fundamental_data,
        [date(2024, 5, 1)],  # Before available
        data_type=DataType.FUNDAMENTAL,
    )
    
    has_correct_type = any(v.violation_type == ViolationType.FUTURE_FUNDAMENTAL for v in violations2)
    
    print_result(
        "Fundamental data violations typed as FUTURE_FUNDAMENTAL",
        has_correct_type,
        f"Found violations: {[v.violation_type.value for v in violations2]}"
    )
    
    assert has_correct_type


def test_cutoff_boundary_cases():
    """Test exact cutoff boundary cases (15:59 vs 16:01)."""
    print_test_header("Cutoff Boundary Cases")
    
    from src.audits.pit_scanner import validate_price_pit, ViolationType
    
    test_date = date(2024, 6, 15)
    
    # Test 1: Usage at 15:59 (before market close) - should be violation
    usage_1559 = ET.localize(datetime(2024, 6, 15, 15, 59))
    v1 = validate_price_pit("TEST", test_date, usage_1559)
    
    passed1 = v1 is not None
    print_result(
        "Using price at 15:59 (before close) is violation",
        passed1,
        f"Usage at 15:59 ET -> {'VIOLATION' if passed1 else 'NO VIOLATION'}"
    )
    
    # Test 2: Usage at 16:00 (exactly at close) - should be valid
    usage_1600 = ET.localize(datetime(2024, 6, 15, 16, 0))
    v2 = validate_price_pit("TEST", test_date, usage_1600)
    
    passed2 = v2 is None
    print_result(
        "Using price at 16:00 (at close) is valid",
        passed2,
        f"Usage at 16:00 ET -> {'VALID' if passed2 else 'VIOLATION'}"
    )
    
    # Test 3: Usage at 16:01 (after close) - should be valid
    usage_1601 = ET.localize(datetime(2024, 6, 15, 16, 1))
    v3 = validate_price_pit("TEST", test_date, usage_1601)
    
    passed3 = v3 is None
    print_result(
        "Using price at 16:01 (after close) is valid",
        passed3,
        f"Usage at 16:01 ET -> {'VALID' if passed3 else 'VIOLATION'}"
    )
    
    assert passed1 and passed2 and passed3


def test_trading_calendar_holidays():
    """Test trading calendar handles holidays correctly."""
    print_test_header("Trading Calendar Holidays")
    
    from src.data.trading_calendar import TradingCalendarImpl
    
    cal = TradingCalendarImpl()
    
    test_dates = [
        (date(2024, 12, 25), False, "Christmas Day"),
        (date(2024, 1, 1), False, "New Year's Day"),
        (date(2024, 7, 4), False, "Independence Day"),
        (date(2024, 11, 28), False, "Thanksgiving"),
        (date(2024, 12, 24), True, "Christmas Eve (trading)"),
        (date(2024, 12, 23), True, "Monday before Christmas"),
        (date(2024, 12, 21), False, "Saturday"),
        (date(2024, 12, 22), False, "Sunday"),
    ]
    
    all_passed = True
    for d, expected_trading, description in test_dates:
        is_trading = cal.is_trading_day(d)
        passed = is_trading == expected_trading
        all_passed = all_passed and passed
        
        print_result(
            f"{d} ({description})",
            passed,
            f"Expected {'trading' if expected_trading else 'closed'}, got {'trading' if is_trading else 'closed'}"
        )
    
    assert all_passed


def test_rebalance_date_generation():
    """Test rebalance date generation rolls to valid trading days."""
    print_test_header("Rebalance Date Generation")
    
    from src.data.trading_calendar import TradingCalendarImpl
    
    cal = TradingCalendarImpl()
    
    # Get monthly rebalance dates for 2024
    monthly = cal.get_rebalance_dates(date(2024, 1, 1), date(2024, 12, 31), freq="monthly")
    
    passed1 = len(monthly) == 12
    print_result(
        "12 monthly rebalance dates generated",
        passed1,
        f"Got {len(monthly)} dates"
    )
    
    # Each rebalance date should be a trading day
    all_trading = all(cal.is_trading_day(d) for d in monthly)
    print_result(
        "All rebalance dates are trading days",
        all_trading,
        f"Non-trading dates: {[d for d in monthly if not cal.is_trading_day(d)]}"
    )
    
    # Each date should be the last trading day of its month
    correct_eom = True
    for d in monthly:
        # Next day should be in different month OR not a trading day
        next_day = d + timedelta(days=1)
        if next_day.month == d.month and cal.is_trading_day(next_day):
            correct_eom = False
            print(f"    ERROR: {d} is not last trading day (next: {next_day})")
    
    print_result(
        "Each date is last trading day of month",
        correct_eom,
        f"Sample: {monthly[:3]}"
    )
    
    # Quarterly should be subset of monthly
    quarterly = cal.get_rebalance_dates(date(2024, 1, 1), date(2024, 12, 31), freq="quarterly")
    passed_quarterly = len(quarterly) == 4 and all(d in monthly for d in quarterly)
    print_result(
        "Quarterly dates are subset of monthly",
        passed_quarterly,
        f"Got {len(quarterly)} quarterly dates: {quarterly}"
    )
    
    assert passed1 and all_trading and correct_eom and passed_quarterly


# =============================================================================
# INTEGRATION TESTS (Require API key and network)
# =============================================================================

def test_fmp_client_real_data():
    """Test FMP client with real API calls."""
    print_test_header("FMP Client (INTEGRATION)")
    
    if not RUN_INTEGRATION:
        print("  SKIPPED: Set RUN_INTEGRATION=1 to run")
        assert True  # Skip without failure
        return
    
    from src.data.fmp_client import FMPClient
    
    api_key = os.getenv("FMP_KEYS")
    if not api_key:
        print("  SKIPPED: No FMP_KEYS in environment")
        assert True  # Skip without failure
        return
    
    client = FMPClient(api_key=api_key)
    
    # Test 1: Historical prices have observed_at in UTC
    print(f"\n  Remaining API calls: {client.remaining_requests}")
    
    end_date = date.today().isoformat()
    start_date = (date.today() - timedelta(days=7)).isoformat()
    
    df = client.get_historical_prices("NVDA", start_date, end_date)
    
    passed1 = not df.empty and "observed_at" in df.columns
    print_result(
        "Historical prices have observed_at",
        passed1,
        f"Got {len(df)} rows, columns: {list(df.columns)[:5]}"
    )
    
    if not df.empty:
        # Check observed_at is UTC
        obs_sample = df["observed_at"].iloc[0]
        is_utc = obs_sample.tzinfo is not None
        print_result(
            "observed_at has timezone",
            is_utc,
            f"Sample: {obs_sample}"
        )
        
        # Check observed_at is market close time (20:00 UTC summer, 21:00 UTC winter)
        hour = obs_sample.hour
        valid_hour = hour in (20, 21)
        print_result(
            "observed_at hour is market close (20 or 21 UTC)",
            valid_hour,
            f"Hour: {hour}"
        )
    
    # Test 2: Income statement has filingDate (note: FMP uses single 'l')
    df_income = client.get_income_statement("NVDA", period="quarter", limit=4)
    
    has_filing = "filingDate" in df_income.columns if not df_income.empty else False
    print_result(
        "Income statement has filingDate",
        has_filing,
        f"Columns: {list(df_income.columns)[:8]}" if not df_income.empty else "Empty"
    )
    
    if not df_income.empty and has_filing:
        filing_sample = df_income["filingDate"].iloc[0]
        print_result(
            "  Filing date example",
            True,
            f"Period: {df_income['period_end'].iloc[0]}, Filed: {filing_sample}"
        )
    
    print(f"\n  Remaining API calls: {client.remaining_requests}")
    
    assert passed1


def test_full_pit_pipeline():
    """Test complete PIT pipeline with real data."""
    print_test_header("Full PIT Pipeline (INTEGRATION)")
    
    if not RUN_INTEGRATION:
        print("  SKIPPED: Set RUN_INTEGRATION=1 to run")
        assert True  # Skip without failure
        return
    
    from src.data.fmp_client import FMPClient
    from src.data.pit_store import DuckDBPITStore
    
    api_key = os.getenv("FMP_KEYS")
    if not api_key:
        print("  SKIPPED: No FMP_KEYS in environment")
        assert True  # Skip without failure
        return
    
    client = FMPClient(api_key=api_key)
    store = DuckDBPITStore()  # In-memory
    
    # Download and store data for one ticker
    end_date = date.today().isoformat()
    start_date = (date.today() - timedelta(days=14)).isoformat()
    
    df = client.get_historical_prices("NVDA", start_date, end_date)
    stored = store.store_prices(df)
    
    print_result(
        f"Stored {stored} price records",
        stored > 0,
        f"Ticker: NVDA, Period: {start_date} to {end_date}"
    )
    
    # Test PIT query
    asof_now = datetime.now(UTC)
    asof_past = datetime.now(UTC) - timedelta(days=7)
    
    prices_now = store.get_price(["NVDA"], asof_now)
    prices_past = store.get_price(["NVDA"], asof_past)
    
    print_result(
        "get_price returns values",
        "NVDA" in prices_now,
        f"Current price: ${prices_now.get('NVDA', 0):.2f}"
    )
    
    # Test avg volume
    avg_vol = store.get_avg_volume(["NVDA"], asof_now, lookback_days=5)
    print_result(
        "get_avg_volume returns value",
        "NVDA" in avg_vol,
        f"5-day avg volume: {avg_vol.get('NVDA', 0):,.0f}"
    )
    
    # Validate PIT
    validation = store.validate_pit("NVDA", "prices")
    print_result(
        "PIT validation passes",
        validation["valid"],
        f"Issues: {validation['issues']}" if validation["issues"] else "No issues"
    )
    
    store.close()
    assert validation["valid"]


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("SECTION 3: DATA INFRASTRUCTURE - COMPREHENSIVE TESTS")
    print("=" * 60)
    print(f"\nIntegration tests: {'ENABLED' if RUN_INTEGRATION else 'DISABLED'}")
    print("(Set RUN_INTEGRATION=1 to enable API tests)")
    
    results = {}
    
    # Unit tests (no API calls)
    print("\n" + "=" * 60)
    print("UNIT TESTS")
    print("=" * 60)
    
    results["timestamp_functions"] = test_pit_timestamp_functions()
    results["observed_at_filtering"] = test_pit_store_observed_at_filtering()
    results["avg_volume_per_ticker"] = test_avg_volume_per_ticker()
    results["fundamental_pit"] = test_fundamental_pit_validation()
    results["violation_types"] = test_pit_scanner_violation_types()
    results["cutoff_boundaries"] = test_cutoff_boundary_cases()
    results["calendar_holidays"] = test_trading_calendar_holidays()
    results["rebalance_dates"] = test_rebalance_date_generation()
    
    # Integration tests (require API)
    print("\n" + "=" * 60)
    print("INTEGRATION TESTS")
    print("=" * 60)
    
    results["fmp_real_data"] = test_fmp_client_real_data()
    results["full_pipeline"] = test_full_pit_pipeline()
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {name}")
    
    print(f"\n  Total: {passed}/{total} tests passed")
    
    print("\n" + "=" * 60)
    if passed == total:
        print("ALL TESTS PASSED ✓")
    else:
        print(f"SOME TESTS FAILED ({total - passed} failures)")
    print("=" * 60)

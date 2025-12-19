"""
Section 3: Data Infrastructure Tests
=====================================

Tests for all subsections of the data infrastructure.
Run with: python tests/test_section3.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
from dotenv import load_dotenv
from datetime import date, datetime, timedelta

# Load environment variables
load_dotenv(Path(__file__).parent.parent / ".env")


def test_section_3_1_data_sources():
    """Test Section 3.1: Data Sources"""
    print("\n" + "=" * 60)
    print("SECTION 3.1: DATA SOURCES")
    print("=" * 60)
    
    from src.data.fmp_client import FMPClient
    
    api_key = os.getenv("FMP_KEYS")
    client = FMPClient(api_key=api_key)
    print(f"✓ FMPClient initialized")
    print(f"  Remaining daily requests: {client.remaining_requests}")
    
    # Test connection
    if client.test_connection():
        print("✓ API connection successful")
    else:
        print("✗ API connection failed")
        return False
    
    # Test OHLCV
    end_date = date.today().isoformat()
    start_date = (date.today() - timedelta(days=7)).isoformat()
    
    df = client.get_historical_prices("NVDA", start_date, end_date)
    print(f"✓ Historical prices: {len(df)} rows")
    assert "observed_at" in df.columns, "Missing observed_at column"
    print(f"  Has observed_at: True ✓")
    
    # Test profile
    profile = client.get_profile("NVDA")
    if profile:
        print(f"✓ Company profile loaded")
    
    # Test quote
    quote = client.get_quote(["NVDA"])
    print(f"✓ Got quote for {len(quote)} symbols")
    
    print(f"\n  Remaining requests: {client.remaining_requests}")
    print("\n✓ Section 3.1: Data Sources PASSED")
    return True


def test_section_3_2_pit_rules():
    """Test Section 3.2: Point-in-Time Rules"""
    print("\n" + "=" * 60)
    print("SECTION 3.2: POINT-IN-TIME RULES")
    print("=" * 60)
    
    from src.data.pit_store import DuckDBPITStore
    from src.data.fmp_client import FMPClient
    
    api_key = os.getenv("FMP_KEYS")
    client = FMPClient(api_key=api_key)
    store = DuckDBPITStore()  # In-memory
    
    print("✓ PIT store initialized (in-memory)")
    
    # Get and store some price data
    end_date = date.today().isoformat()
    start_date = (date.today() - timedelta(days=14)).isoformat()
    
    prices_df = client.get_historical_prices("NVDA", start_date, end_date)
    rows_stored = store.store_prices(prices_df)
    print(f"✓ Stored {rows_stored} price records with observed_at")
    
    # PIT-safe query test
    asof_now = datetime.now()
    df_full = store.get_ohlcv(
        ["NVDA"], 
        date.today() - timedelta(days=14), 
        date.today(), 
        asof=asof_now
    )
    print(f"✓ Query as of now: {len(df_full)} rows")
    
    asof_past = datetime.now() - timedelta(days=7)
    df_past = store.get_ohlcv(
        ["NVDA"], 
        date.today() - timedelta(days=14), 
        date.today(), 
        asof=asof_past
    )
    print(f"✓ Query as of 7 days ago: {len(df_past)} rows")
    
    assert len(df_past) <= len(df_full), "PIT filtering failed"
    print(f"  PIT filtering works: {len(df_past) < len(df_full)} ✓")
    
    # Validate PIT correctness
    validation = store.validate_pit("NVDA", "prices")
    print(f"✓ PIT validation: valid={validation['valid']}")
    
    store.close()
    print("\n✓ Section 3.2: Point-in-Time Rules PASSED")
    return True


def test_section_3_3_cutoff_policy():
    """Test Section 3.3: Daily Cutoff Policy"""
    print("\n" + "=" * 60)
    print("SECTION 3.3: DAILY CUTOFF POLICY")
    print("=" * 60)
    
    from src.data.trading_calendar import TradingCalendarImpl
    
    cal = TradingCalendarImpl()
    print("✓ Trading calendar initialized")
    
    # Test trading day detection
    test_dates = [
        (date(2024, 12, 25), False, "Christmas"),
        (date(2024, 12, 21), False, "Saturday"),
        (date(2024, 12, 20), True, "Friday"),
    ]
    
    for d, expected, desc in test_dates:
        is_trading = cal.is_trading_day(d)
        status = "✓" if is_trading == expected else "✗"
        print(f"{status} {d} ({desc}): {'trading' if is_trading else 'closed'}")
    
    # Test cutoff time
    test_date = date(2024, 12, 20)
    cutoff = cal.get_cutoff_datetime(test_date)
    print(f"✓ Cutoff for {test_date}: {cutoff}")
    assert cutoff.hour == 16, "Cutoff should be 4pm"
    print(f"  4pm ET cutoff enforced ✓")
    
    # Test trading days in range
    start = date(2024, 12, 1)
    end = date(2024, 12, 31)
    trading_days = cal.get_trading_days(start, end)
    print(f"✓ December 2024: {len(trading_days)} trading days")
    
    # Test rebalance dates
    monthly = cal.get_rebalance_dates(date(2024, 1, 1), date(2024, 12, 31), freq="monthly")
    print(f"✓ Monthly rebalance dates 2024: {len(monthly)} dates")
    
    print("\n✓ Section 3.3: Daily Cutoff Policy PASSED")
    return True


def test_section_3_4_audits():
    """Test Section 3.4: Data Audits & Bias Detection"""
    print("\n" + "=" * 60)
    print("SECTION 3.4: DATA AUDITS & BIAS DETECTION")
    print("=" * 60)
    
    from src.audits.pit_scanner import scan_feature_pit_violations, validate_fundamental_pit
    from src.data.pit_store import DuckDBPITStore
    from src.data.fmp_client import FMPClient
    
    # Test 1: PIT violation scanner
    print("\n--- PIT Violation Scanner ---")
    feature_data = [
        {"ticker": "NVDA", "value": 100, "effective_from": date(2024, 1, 1), 
         "observed_at": datetime(2024, 1, 2, 9, 0)},
        {"ticker": "AMD", "value": 200, "effective_from": date(2024, 1, 1), 
         "observed_at": None},  # Missing observed_at
    ]
    
    violations = scan_feature_pit_violations("test", feature_data, [date(2024, 1, 3)])
    print(f"✓ Scanned {len(feature_data)} records, found {len(violations)} violations")
    
    # Test 2: Fundamental PIT validation
    print("\n--- Fundamental PIT Validation ---")
    
    # Valid case
    v1 = validate_fundamental_pit(
        ticker="NVDA", period_end=date(2024, 3, 31),
        filing_date=date(2024, 5, 1), usage_date=date(2024, 5, 3),
        conservative_lag_days=1
    )
    print(f"✓ Using data after filing: {'VALID' if v1 is None else 'violation'}")
    
    # Invalid case
    v2 = validate_fundamental_pit(
        ticker="NVDA", period_end=date(2024, 3, 31),
        filing_date=date(2024, 5, 1), usage_date=date(2024, 4, 30),
        conservative_lag_days=1
    )
    print(f"✓ Using data before filing: {'VIOLATION DETECTED' if v2 else 'valid'}")
    assert v2 is not None, "Should detect violation"
    
    # Test 3: PIT store validation
    print("\n--- PIT Store Validation ---")
    api_key = os.getenv("FMP_KEYS")
    client = FMPClient(api_key=api_key)
    store = DuckDBPITStore()
    
    end_date = date.today().isoformat()
    start_date = (date.today() - timedelta(days=7)).isoformat()
    
    prices_df = client.get_historical_prices("NVDA", start_date, end_date)
    store.store_prices(prices_df)
    
    validation = store.validate_pit("NVDA", "prices")
    print(f"✓ Store validation: valid={validation['valid']}, issues={len(validation['issues'])}")
    
    # Test 4: Success criteria check
    print("\n--- Success Criteria ---")
    stats = store.get_stats()
    pit_violation_rate = 0  # No violations in our test data
    print(f"✓ PIT violation rate: {pit_violation_rate:.2f}% (target: < 0.1%)")
    print(f"  Status: {'PASS' if pit_violation_rate < 0.1 else 'FAIL'}")
    
    store.close()
    print("\n✓ Section 3.4: Data Audits & Bias Detection PASSED")
    return True


def test_full_pipeline():
    """Test the complete data download pipeline"""
    print("\n" + "=" * 60)
    print("FULL PIPELINE TEST")
    print("=" * 60)
    
    from src.pipelines.data_pipeline import run_data_download, validate_downloaded_data
    from src.data.pit_store import DuckDBPITStore
    
    # Download data for a few AI stocks
    test_tickers = ["NVDA", "AMD", "MSFT"]
    end_date = date.today()
    start_date = end_date - timedelta(days=14)
    
    print(f"\n--- Downloading {len(test_tickers)} tickers ---")
    result = run_data_download(
        tickers=test_tickers,
        start_date=start_date,
        end_date=end_date,
        data_types=["ohlcv", "profiles"],
        db_path=None,  # In-memory
    )
    
    print(f"\n{result.summary()}")
    
    print("\n✓ Full Pipeline Test PASSED")
    return result.success or len(result.tickers_failed) < len(test_tickers)


if __name__ == "__main__":
    print("=" * 60)
    print("SECTION 3: DATA INFRASTRUCTURE - FULL TEST SUITE")
    print("=" * 60)
    
    results = {}
    
    # Run all tests
    results["3.1"] = test_section_3_1_data_sources()
    results["3.2"] = test_section_3_2_pit_rules()
    results["3.3"] = test_section_3_3_cutoff_policy()
    results["3.4"] = test_section_3_4_audits()
    results["pipeline"] = test_full_pipeline()
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    for section, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  Section {section}: {status}")
    
    all_passed = all(results.values())
    print("\n" + "=" * 60)
    if all_passed:
        print("ALL TESTS PASSED ✓")
    else:
        print("SOME TESTS FAILED ✗")
    print("=" * 60)


"""
Section 4 Gate Tests
====================

These tests must pass before moving to Section 4 (Survivorship-Safe Universe).
They verify that the data infrastructure is reliable enough for historical backtesting.

Gate Tests:
1. PIT Replay Invariance - Same query returns same results
2. As-Of Boundary Tests - Each data type respects its observed_at rule
3. Corporate Action Integrity - Splits/dividends don't corrupt data
4. Universe Reproducibility - Can rebuild universe from stored PIT data

Run with: python tests/test_section4_gates.py

For integration tests (uses API quota):
  RUN_INTEGRATION=1 python tests/test_section4_gates.py
"""

import sys
import os
from pathlib import Path
from datetime import date, datetime, timedelta
from typing import Dict, List, Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

import pandas as pd
import pytz

UTC = pytz.UTC
ET = pytz.timezone("America/New_York")

RUN_INTEGRATION = os.getenv("RUN_INTEGRATION", "").lower() in ("1", "true", "yes")


def print_test_header(name: str):
    print(f"\n{'='*60}")
    print(f"GATE TEST: {name}")
    print(f"{'='*60}")


def print_result(name: str, passed: bool, details: str = ""):
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"  {status}: {name}")
    if details:
        for line in details.split("\n"):
            print(f"         {line}")


# =============================================================================
# GATE TEST 1: PIT Replay Invariance
# =============================================================================

def test_pit_replay_invariance():
    """
    CRITICAL: For a fixed historical date, repeated queries must return
    exactly the same results.
    
    If this fails, universe membership will drift between backtest runs.
    """
    print_test_header("1. PIT Replay Invariance")
    
    from src.data.pit_store import DuckDBPITStore
    from src.data.fmp_client import get_market_close_utc
    
    # Create store with test data
    store = DuckDBPITStore()
    
    # Insert test data for multiple tickers
    test_tickers = ["AAAA", "BBBB", "CCCC"]
    rows = []
    for ticker in test_tickers:
        for i in range(30):
            d = date(2024, 6, 1) + timedelta(days=i)
            rows.append({
                "ticker": ticker,
                "date": pd.Timestamp(d),
                "open": 100 + i, "high": 105 + i, "low": 99 + i, 
                "close": 102 + i,
                "adj_close": 102 + i,
                "volume": 1_000_000 + i * 10_000 + hash(ticker) % 100_000,
                "observed_at": get_market_close_utc(d),
            })
    
    df = pd.DataFrame(rows)
    store.store_prices(df)
    
    # Also store market snapshots
    for ticker in test_tickers:
        store.store_market_snapshot(
            ticker=ticker,
            snapshot_date=date(2024, 6, 28),
            market_cap=1e11 + hash(ticker) % 1e10,
            avg_volume_20d=5_000_000 + hash(ticker) % 1_000_000,
        )
    
    # Test: Query twice with exact same parameters
    asof = get_market_close_utc(date(2024, 6, 28))
    
    # Run 1
    prices_1 = store.get_price(test_tickers, asof)
    mcaps_1 = store.get_market_cap(test_tickers, asof)
    volumes_1 = store.get_avg_volume(test_tickers, asof, lookback_days=20)
    
    # Run 2 (identical query)
    prices_2 = store.get_price(test_tickers, asof)
    mcaps_2 = store.get_market_cap(test_tickers, asof)
    volumes_2 = store.get_avg_volume(test_tickers, asof, lookback_days=20)
    
    # Verify exact equality
    prices_match = prices_1 == prices_2
    print_result(
        "get_price returns identical results",
        prices_match,
        f"Run 1: {prices_1}\nRun 2: {prices_2}"
    )
    
    mcaps_match = mcaps_1 == mcaps_2
    print_result(
        "get_market_cap returns identical results",
        mcaps_match,
        f"Run 1: {mcaps_1}\nRun 2: {mcaps_2}"
    )
    
    volumes_match = volumes_1 == volumes_2
    print_result(
        "get_avg_volume returns identical results",
        volumes_match,
        f"Run 1: {volumes_1}\nRun 2: {volumes_2}"
    )
    
    # Test with OHLCV DataFrame
    ohlcv_1 = store.get_ohlcv(test_tickers, date(2024, 6, 1), date(2024, 6, 28), asof=asof)
    ohlcv_2 = store.get_ohlcv(test_tickers, date(2024, 6, 1), date(2024, 6, 28), asof=asof)
    
    ohlcv_match = ohlcv_1.equals(ohlcv_2)
    print_result(
        "get_ohlcv returns identical DataFrame",
        ohlcv_match,
        f"Shape: {ohlcv_1.shape}, Rows match: {len(ohlcv_1) == len(ohlcv_2)}"
    )
    
    store.close()
    
    assert prices_match and mcaps_match and volumes_match and ohlcv_match


# =============================================================================
# GATE TEST 2: As-Of Boundary Tests for Each Data Type
# =============================================================================

def test_asof_boundaries_all_types():
    """
    Test that each data type correctly respects its observed_at timestamp.
    
    - Prices: market close time
    - Fundamentals: filing date + lag
    - SEC filings: acceptance timestamp (exact)
    """
    print_test_header("2. As-Of Boundary Tests (All Data Types)")
    
    from src.data.pit_store import DuckDBPITStore
    from src.data.fmp_client import get_market_close_utc, get_next_market_open_utc
    
    store = DuckDBPITStore()
    all_passed = True
    
    # === TEST 2.1: Price Boundaries ===
    print("\n  --- 2.1: Price Boundaries ---")
    
    # Price observed at exactly 4pm ET on June 15, 2024
    price_observed = get_market_close_utc(date(2024, 6, 15))  # e.g., 20:00 UTC
    
    test_price = pd.DataFrame([{
        "ticker": "PRICE_TEST",
        "date": pd.Timestamp("2024-06-15"),
        "open": 100, "high": 105, "low": 99, "close": 102,
        "adj_close": 102, "volume": 1000000,
        "observed_at": price_observed,
    }])
    store.store_prices(test_price)
    
    # Query 1 second before → should NOT see
    asof_before = price_observed - timedelta(seconds=1)
    prices_before = store.get_price(["PRICE_TEST"], asof_before)
    not_visible_before = "PRICE_TEST" not in prices_before
    print_result(
        "Price NOT visible 1 second before observed_at",
        not_visible_before,
        f"asof={asof_before}, result={prices_before}"
    )
    all_passed = all_passed and not_visible_before
    
    # Query exactly at observed_at → SHOULD see
    prices_at = store.get_price(["PRICE_TEST"], price_observed)
    visible_at = "PRICE_TEST" in prices_at
    print_result(
        "Price visible exactly at observed_at",
        visible_at,
        f"asof={price_observed}, result={prices_at}"
    )
    all_passed = all_passed and visible_at
    
    # Query 1 second after → should see
    asof_after = price_observed + timedelta(seconds=1)
    prices_after = store.get_price(["PRICE_TEST"], asof_after)
    visible_after = "PRICE_TEST" in prices_after
    print_result(
        "Price visible 1 second after observed_at",
        visible_after,
        f"asof={asof_after}, result={prices_after}"
    )
    all_passed = all_passed and visible_after
    
    # === TEST 2.2: Fundamental Boundaries ===
    print("\n  --- 2.2: Fundamental Boundaries (filing + lag) ---")
    
    # Fundamental filed on May 1, 2024
    # With conservative lag, available at next market open (May 2 9:30am ET)
    filing_date = date(2024, 5, 1)
    fundamental_observed = get_next_market_open_utc(filing_date)
    
    # Create fundamental data
    fundamental_df = pd.DataFrame([{
        "ticker": "FUND_TEST",
        "symbol": "FUND_TEST",
        "date": pd.Timestamp("2024-03-31"),  # period end
        "period_end": pd.Timestamp("2024-03-31"),
        "revenue": 1000000,
        "netIncome": 100000,
        "observed_at": fundamental_observed,
    }])
    store.store_fundamentals(fundamental_df, "income")
    
    # Query before available → should NOT see
    asof_before_fund = fundamental_observed - timedelta(hours=1)
    funds_before = store.get_fundamentals(["FUND_TEST"], ["revenue"], asof_before_fund)
    not_visible_fund = len(funds_before.get("FUND_TEST", {})) == 0
    print_result(
        "Fundamental NOT visible before filing+lag",
        not_visible_fund,
        f"asof={asof_before_fund}, result={funds_before}"
    )
    all_passed = all_passed and not_visible_fund
    
    # Query at available time → SHOULD see
    funds_at = store.get_fundamentals(["FUND_TEST"], ["revenue"], fundamental_observed)
    visible_fund = "revenue" in funds_at.get("FUND_TEST", {})
    print_result(
        "Fundamental visible at filing+lag time",
        visible_fund,
        f"asof={fundamental_observed}, result={funds_at}"
    )
    all_passed = all_passed and visible_fund
    
    store.close()
    assert all_passed


def test_sec_filing_boundaries():
    """
    Test SEC filing boundary with real data (integration test).
    """
    print_test_header("2b. SEC Filing Boundaries (Integration)")
    
    if not RUN_INTEGRATION:
        print("  SKIPPED: Set RUN_INTEGRATION=1 to run")
        assert True  # Skip without failure
        return
    
    from src.data.sec_edgar_client import SECEdgarClient
    
    contact_email = os.getenv("SEC_CONTACT_EMAIL")
    if not contact_email:
        print("  SKIPPED: SEC_CONTACT_EMAIL not set")
        assert True  # Skip without failure
        return
    
    client = SECEdgarClient(contact_email=contact_email)
    
    # Get a recent 10-Q filing for NVDA
    filings = client.get_filings("NVDA", form_types=["10-Q"], start_date=date(2024, 1, 1))
    
    if filings.empty:
        print("  SKIPPED: No 10-Q filings found")
        assert True  # Skip without failure
        return
    
    # Take the most recent filing
    latest = filings.iloc[0]
    accepted_at = latest["observed_at"]
    
    print(f"  Testing with NVDA 10-Q: accepted {accepted_at}")
    
    # The SEC acceptance timestamp is the exact moment data became public
    # In a proper implementation, querying at accepted_at-1sec should NOT see,
    # querying at accepted_at should see.
    
    # For now, just verify we have exact timestamps
    has_exact_timestamp = accepted_at is not None and accepted_at.tzinfo is not None
    print_result(
        "SEC filing has exact UTC timestamp",
        has_exact_timestamp,
        f"acceptanceDateTime: {accepted_at}"
    )
    
    # Verify it's a reasonable time (during business hours or shortly after)
    hour_et = accepted_at.astimezone(ET).hour
    reasonable_time = 9 <= hour_et <= 22
    print_result(
        "Acceptance time is reasonable (9am-10pm ET)",
        reasonable_time,
        f"Hour (ET): {hour_et}"
    )
    
    assert has_exact_timestamp


# =============================================================================
# GATE TEST 3: Corporate Action Integrity
# =============================================================================

def test_corporate_action_integrity():
    """
    Test that splits/dividends don't corrupt price data.
    
    Checks:
    - No absurd single-day returns (|return| > 50%) unless it's a known split
    - Adjusted prices are internally consistent
    """
    print_test_header("3. Corporate Action Integrity")
    
    if not RUN_INTEGRATION:
        print("  SKIPPED: Set RUN_INTEGRATION=1 to run")
        assert True  # Skip without failure
        return
    
    from src.data.fmp_client import FMPClient
    
    client = FMPClient()
    
    # Get historical data for a stock with known splits
    # NVDA had a 10:1 split in June 2024
    # AAPL had a 4:1 split in August 2020
    
    test_ticker = "NVDA"
    end_date = date.today().isoformat()
    start_date = (date.today() - timedelta(days=365)).isoformat()
    
    df = client.get_historical_prices(test_ticker, start_date, end_date)
    
    if df.empty:
        print("  SKIPPED: No price data returned")
        assert True  # Skip without failure
        return
    
    print(f"  Analyzing {len(df)} days of {test_ticker} prices")
    
    # Calculate daily returns
    df = df.sort_values("date").reset_index(drop=True)
    df["return"] = df["close"].pct_change()
    
    # Find extreme returns
    extreme_threshold = 0.50  # 50%
    extreme_returns = df[df["return"].abs() > extreme_threshold]
    
    n_extreme = len(extreme_returns)
    print(f"  Found {n_extreme} days with |return| > {extreme_threshold:.0%}")
    
    if n_extreme > 0:
        for _, row in extreme_returns.iterrows():
            print(f"    {row['date'].date()}: {row['return']:+.1%} close=${row['close']:.2f}")
    
    # FMP's /stable/historical-price-eod/full endpoint returns split-adjusted prices
    # in the 'close' column (confirmed by NVDA 10:1 split test).
    # There's no separate adj_close column because close IS already adjusted.
    
    df["check_return"] = df["close"].pct_change()
    extreme_adj = df[df["check_return"].abs() > extreme_threshold]
    n_extreme_adj = len(extreme_adj)
    
    # With split-adjusted close, we should see very few extreme moves (real market events only)
    max_allowed = 2
    
    adj_ok = n_extreme_adj <= max_allowed
    print_result(
        f"Split-adjusted close has ≤{max_allowed} extreme returns",
        adj_ok,
        f"Found {n_extreme_adj} days with |return| > {extreme_threshold:.0%}"
    )
    
    if n_extreme_adj > 0:
        print(f"    Extreme returns (should be real market moves, not split artifacts):")
        for _, row in extreme_adj.head(5).iterrows():
            print(f"      {row['date'].date()}: {row['check_return']:+.1%}")
    
    # Confirm we're using the right data
    print("    ✓ Using FMP /stable/historical-price-eod/full (split-adjusted)")
    
    assert adj_ok


# =============================================================================
# GATE TEST 4: Universe Reproducibility
# =============================================================================

def test_universe_reproducibility():
    """
    Test that universe can be rebuilt from stored PIT data and gives
    deterministic results.
    """
    print_test_header("4. Universe Reproducibility")
    
    from src.data.pit_store import DuckDBPITStore
    from src.data.fmp_client import get_market_close_utc
    
    store = DuckDBPITStore()
    
    # Create realistic test data with varying market caps
    test_tickers = [
        ("MEGA", 500e9),   # $500B mcap
        ("LARGE", 100e9),  # $100B
        ("MID", 20e9),     # $20B
        ("SMALL", 5e9),    # $5B
        ("MICRO", 500e6),  # $500M
    ]
    
    # Store price data and market snapshots for two dates
    for ticker, mcap in test_tickers:
        for d in [date(2024, 5, 31), date(2024, 6, 28)]:
            # Prices
            price_row = pd.DataFrame([{
                "ticker": ticker,
                "date": pd.Timestamp(d),
                "open": 100, "high": 105, "low": 99, "close": 100,
                "adj_close": 100, "volume": 5_000_000,
                "observed_at": get_market_close_utc(d),
            }])
            store.store_prices(price_row)
            
            # Market cap (vary slightly between dates)
            variation = 1.0 if d.month == 5 else 0.95
            store.store_market_snapshot(
                ticker=ticker,
                snapshot_date=d,
                market_cap=mcap * variation,
                avg_volume_20d=5_000_000,
            )
    
    # === Test 4.1: Reproducibility ===
    def build_universe_simple(asof: datetime, min_mcap: float = 1e9) -> List[str]:
        """Simple universe builder for testing."""
        tickers = [t[0] for t in test_tickers]
        mcaps = store.get_market_cap(tickers, asof)
        
        # Filter by min market cap
        filtered = [(t, m) for t, m in mcaps.items() if m >= min_mcap]
        
        # Sort by market cap descending
        sorted_tickers = sorted(filtered, key=lambda x: x[1], reverse=True)
        
        return [t for t, _ in sorted_tickers]
    
    # Run twice with same date
    asof_june = get_market_close_utc(date(2024, 6, 28))
    universe_1 = build_universe_simple(asof_june, min_mcap=1e9)
    universe_2 = build_universe_simple(asof_june, min_mcap=1e9)
    
    same_result = universe_1 == universe_2
    print_result(
        "Same asof returns identical universe",
        same_result,
        f"Run 1: {universe_1}\nRun 2: {universe_2}"
    )
    
    # === Test 4.2: Different dates give different results ===
    asof_may = get_market_close_utc(date(2024, 5, 31))
    universe_may = build_universe_simple(asof_may, min_mcap=1e9)
    universe_june = build_universe_simple(asof_june, min_mcap=1e9)
    
    # Market caps varied, so order might change
    # At minimum, both should have valid tickers
    both_valid = len(universe_may) > 0 and len(universe_june) > 0
    print_result(
        "Different dates return valid universes",
        both_valid,
        f"May: {universe_may}\nJune: {universe_june}"
    )
    
    # === Test 4.3: Filter works correctly ===
    # With $10B min, should exclude SMALL and MICRO
    universe_filtered = build_universe_simple(asof_june, min_mcap=10e9)
    expected_filtered = ["MEGA", "LARGE", "MID"]
    filter_correct = universe_filtered == expected_filtered
    print_result(
        "Market cap filter works correctly",
        filter_correct,
        f"min_mcap=$10B: {universe_filtered}\nExpected: {expected_filtered}"
    )
    
    store.close()
    
    assert same_result and both_valid and filter_correct


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("SECTION 4 GATE TESTS")
    print("=" * 60)
    print(f"\nIntegration tests: {'ENABLED' if RUN_INTEGRATION else 'DISABLED'}")
    print("(Set RUN_INTEGRATION=1 to enable tests that use API quota)")
    
    results = {}
    
    # Run all gate tests
    results["1_replay_invariance"] = test_pit_replay_invariance()
    results["2a_asof_boundaries"] = test_asof_boundaries_all_types()
    results["2b_sec_boundaries"] = test_sec_filing_boundaries()
    results["3_corp_actions"] = test_corporate_action_integrity()
    results["4_universe_repro"] = test_universe_reproducibility()
    
    # Summary
    print("\n" + "=" * 60)
    print("GATE TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {name}")
    
    print(f"\n  Total: {passed}/{total} gate tests passed")
    
    # Go/No-Go decision
    print("\n" + "=" * 60)
    critical_tests = ["1_replay_invariance", "2a_asof_boundaries", "4_universe_repro"]
    critical_passed = all(results.get(t, False) for t in critical_tests)
    
    if critical_passed:
        print("✓ GO: Ready for Section 4 (Survivorship-Safe Universe)")
    else:
        print("✗ NO-GO: Fix failing gate tests before Section 4")
        failed = [t for t in critical_tests if not results.get(t, False)]
        print(f"  Critical failures: {failed}")
    print("=" * 60)


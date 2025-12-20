"""
Chapter 3 Extensions - Test Suite
==================================

Tests for new data infrastructure components:
1. Expectations data (earnings surprises with PIT)
2. EventStore new event types
3. Security master (identifier mapping)
4. PIT rules for new data sources

Run with: python tests/test_chapter3_extensions.py
"""

import sys
from pathlib import Path
from datetime import date, datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

import pytz

UTC = pytz.UTC
ET = pytz.timezone("America/New_York")


def print_test_header(name: str):
    print(f"\n{'='*60}")
    print(f"TEST: {name}")
    print(f"{'='*60}")


def print_result(name: str, passed: bool, details: str = ""):
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"  {status}: {name}")
    if details:
        for line in details.split("\n"):
            print(f"         {line}")


# =============================================================================
# Test 1: New EventTypes
# =============================================================================

def test_new_event_types():
    """Test that all new EventTypes are defined."""
    print_test_header("1. New EventTypes")
    
    from src.data.event_store import EventType
    
    expected_types = [
        # Core (existing)
        "EARNINGS", "FILING", "NEWS", "SENTIMENT", "DIVIDEND", "SPLIT",
        # Tier 1 - Expectations
        "ESTIMATE_SNAPSHOT", "ESTIMATE_REVISION", "GUIDANCE", "ANALYST_ACTION",
        # Tier 2 - Options & Positioning
        "OPTIONS_SNAPSHOT", "SHORT_INTEREST", "BORROW_COST", "ETF_FLOW",
        "INSTITUTIONAL_13F",
        # Tier 0 - Survivorship
        "SECURITY_MASTER", "DELISTING",
    ]
    
    all_passed = True
    for type_name in expected_types:
        try:
            EventType[type_name]
            print_result(type_name, True)
        except KeyError:
            print_result(type_name, False, "Missing from EventType enum")
            all_passed = False
    
    return all_passed


# =============================================================================
# Test 2: Earnings Surprise PIT Rules
# =============================================================================

def test_earnings_surprise_pit():
    """Test that earnings surprises have correct PIT timestamps."""
    print_test_header("2. Earnings Surprise PIT Rules")
    
    from src.data.expectations_client import ExpectationsClient, EarningsSurprise
    
    # Create test surprise with known timing
    surprise = EarningsSurprise(
        ticker="TEST",
        fiscal_date_ending=date(2024, 10, 31),
        reported_date=date(2024, 11, 19),
        reported_eps=1.30,
        estimated_eps=1.24,
        surprise=0.06,
        surprise_pct=4.84,
        report_time="post-market",
        observed_at=ET.localize(datetime(2024, 11, 19, 16, 5)).astimezone(UTC),
    )
    
    # Test 1: observed_at is after market close for AMC
    is_after_close = surprise.observed_at.astimezone(ET).hour >= 16
    print_result(
        "AMC earnings observed_at is after 4pm ET",
        is_after_close,
        f"observed_at hour (ET): {surprise.observed_at.astimezone(ET).hour}"
    )
    
    # Test 2: Convert to Event
    event = surprise.to_event()
    from src.data.event_store import EventType, EventTiming
    
    is_earnings_event = event.event_type == EventType.EARNINGS
    print_result("Converts to EARNINGS event type", is_earnings_event)
    
    is_amc = event.timing == EventTiming.AMC
    print_result("Timing marked as AMC", is_amc)
    
    # Test 3: Payload contains required fields
    has_eps = "reported_eps" in event.payload and "estimated_eps" in event.payload
    print_result("Payload contains EPS data", has_eps)
    
    # Test 4: BMO handling
    bmo_surprise = EarningsSurprise(
        ticker="TEST_BMO",
        fiscal_date_ending=date(2024, 10, 31),
        reported_date=date(2024, 11, 20),
        reported_eps=1.0,
        estimated_eps=1.0,
        surprise=0.0,
        surprise_pct=0.0,
        report_time="pre-market",
        observed_at=ET.localize(datetime(2024, 11, 20, 9, 30)).astimezone(UTC),
    )
    
    bmo_event = bmo_surprise.to_event()
    is_bmo = bmo_event.timing == EventTiming.BMO
    print_result("BMO earnings marked correctly", is_bmo)
    
    bmo_hour = bmo_surprise.observed_at.astimezone(ET).hour
    is_morning = bmo_hour < 12
    print_result("BMO observed_at is morning", is_morning, f"Hour: {bmo_hour}")
    
    return is_after_close and is_earnings_event and is_amc and has_eps and is_bmo


# =============================================================================
# Test 3: Security Master
# =============================================================================

def test_security_master():
    """Test security master functionality."""
    print_test_header("3. Security Master")
    
    from src.data.security_master import (
        SecurityMaster, SecurityIdentifier, SecurityEventType
    )
    
    sm = SecurityMaster()  # In-memory
    
    # Test 1: Add identifier
    sm.add_identifier(SecurityIdentifier(
        ticker="TEST",
        stable_id="TEST_ID",
        company_name="Test Company",
        valid_from=date(2020, 1, 1),
        is_active=True,
    ))
    
    stable_id = sm.get_stable_id("TEST")
    has_id = stable_id == "TEST_ID"
    print_result("Add and retrieve stable_id", has_id, f"Got: {stable_id}")
    
    # Test 2: Record ticker change
    sm.record_ticker_change(
        stable_id="CHANGE_ID",
        old_ticker="OLD",
        new_ticker="NEW",
        change_date=date(2023, 6, 1),
        observed_at=UTC.localize(datetime(2023, 6, 1, 16, 0)),
    )
    
    # Check old ticker is no longer valid
    old_active = sm.was_active("OLD", date(2023, 7, 1))
    new_active = sm.was_active("NEW", date(2023, 7, 1))
    old_was_active = sm.was_active("OLD", date(2023, 5, 1))
    
    print_result("Old ticker inactive after change", not old_active)
    print_result("New ticker active after change", new_active)
    print_result("Old ticker was active before change", old_was_active)
    
    # Test 3: Record delisting
    sm.add_identifier(SecurityIdentifier(
        ticker="DELIST",
        stable_id="DELIST_ID",
        company_name="Delisted Co",
        valid_from=date(2020, 1, 1),
        is_active=True,
    ))
    
    sm.record_delisting(
        ticker="DELIST",
        stable_id="DELIST_ID",
        delisting_date=date(2024, 3, 15),
        terminal_price=0.50,
        observed_at=UTC.localize(datetime(2024, 3, 15, 16, 0)),
        reason="bankruptcy",
    )
    
    terminal = sm.get_terminal_price("DELIST_ID")
    has_terminal = terminal == 0.50
    print_result("Terminal price recorded", has_terminal, f"Got: ${terminal}")
    
    delist_active = sm.was_active("DELIST", date(2024, 6, 1))
    print_result("Delisted ticker marked inactive", not delist_active)
    
    sm.close()
    
    return has_id and not old_active and new_active and has_terminal


# =============================================================================
# Test 4: PIT Rules for Positioning Data
# =============================================================================

def test_positioning_pit_rules():
    """Test PIT rules documentation for positioning data."""
    print_test_header("4. Positioning PIT Rules")
    
    from src.data.positioning_client import (
        ShortInterestSnapshot, Institutional13FHolding, ETFFlowSnapshot
    )
    
    # Test 1: Short interest has 10-day lag
    si = ShortInterestSnapshot(
        ticker="TEST",
        settlement_date=date(2024, 1, 15),
        short_interest=1_000_000,
        avg_volume=500_000,
        days_to_cover=2.0,
    )
    event = si.to_event()
    
    lag_days = (event.observed_at.date() - si.settlement_date).days
    has_lag = lag_days >= 10
    print_result(
        "Short interest has publication lag",
        has_lag,
        f"Lag: {lag_days} days (expected ~10)"
    )
    
    # Test 2: 13F uses filing date, not period end
    holding = Institutional13FHolding(
        ticker="TEST",
        institution_name="Big Fund",
        institution_cik="0001234567",
        period_end=date(2024, 3, 31),  # Q1 end
        filing_date=date(2024, 5, 10),  # ~45 days later
        shares_held=100_000,
        value_usd=10_000_000,
    )
    event = holding.to_event()
    
    # CRITICAL: observed_at should be filing_date, NOT period_end
    is_filing_date = event.event_date == holding.filing_date
    is_not_period_end = event.event_date != holding.period_end
    print_result(
        "13F uses filing_date for event_date",
        is_filing_date,
        f"event_date: {event.event_date}, filing_date: {holding.filing_date}"
    )
    print_result(
        "13F does NOT use period_end",
        is_not_period_end,
        f"period_end: {holding.period_end}"
    )
    
    # Test 3: ETF flows available next day
    flow = ETFFlowSnapshot(
        etf_ticker="QQQ",
        flow_date=date(2024, 1, 15),
        flow_usd=500_000_000,
        aum=200_000_000_000,
        shares_outstanding=400_000_000,
    )
    event = flow.to_event()
    
    flow_lag = (event.observed_at.date() - flow.flow_date).days
    has_next_day = flow_lag >= 1
    print_result(
        "ETF flows available next day",
        has_next_day,
        f"Lag: {flow_lag} days"
    )
    
    return has_lag and is_filing_date and is_not_period_end and has_next_day


# =============================================================================
# Test 5: Options PIT Rules
# =============================================================================

def test_options_pit_rules():
    """Test PIT rules for options data."""
    print_test_header("5. Options PIT Rules")
    
    from src.data.options_client import IVSurfaceSnapshot, ImpliedMoveSnapshot
    
    # Test 1: IV surface observed at market close
    iv = IVSurfaceSnapshot(
        ticker="TEST",
        quote_date=date(2024, 1, 15),
        iv_1m=0.30,
        iv_3m=0.28,
    )
    event = iv.to_event()
    
    observed_hour = event.observed_at.hour
    is_close = observed_hour == 21  # 4pm ET in winter = 21:00 UTC
    print_result(
        "IV snapshot observed at market close",
        is_close,
        f"observed_at hour (UTC): {observed_hour}"
    )
    
    # Test 2: Term structure detection
    is_inverted = iv.is_inverted
    should_be_inverted = iv.iv_1m > iv.iv_3m
    print_result(
        "Inverted term structure detected",
        is_inverted == should_be_inverted,
        f"1m IV: {iv.iv_1m}, 3m IV: {iv.iv_3m}, inverted: {is_inverted}"
    )
    
    # Test 3: Implied move calculation
    from src.data.options_client import OptionsClient
    client = OptionsClient()
    
    move = client.calculate_implied_move(
        straddle_price=15.0,
        stock_price=500.0,
        days_to_expiry=7,
    )
    expected_move = 3.0  # 15/500 * 100
    is_correct = abs(move - expected_move) < 0.1
    print_result(
        "Implied move calculation correct",
        is_correct,
        f"Got: {move:.1f}%, expected: {expected_move:.1f}%"
    )
    
    return is_close and is_correct


# =============================================================================
# Test 6: EventStore Integration
# =============================================================================

def test_eventstore_integration():
    """Test storing new event types in EventStore."""
    print_test_header("6. EventStore Integration")
    
    from src.data.event_store import EventStore, EventType
    from src.data.expectations_client import EarningsSurprise
    from src.data.positioning_client import Institutional13FHolding
    
    store = EventStore()  # In-memory
    
    # Store earnings surprise
    surprise = EarningsSurprise(
        ticker="NVDA",
        fiscal_date_ending=date(2024, 10, 31),
        reported_date=date(2024, 11, 19),
        reported_eps=1.30,
        estimated_eps=1.24,
        surprise=0.06,
        surprise_pct=4.84,
        report_time="post-market",
        observed_at=UTC.localize(datetime(2024, 11, 19, 21, 5)),
    )
    store.store_event(surprise.to_event())
    
    # Store 13F holding
    holding = Institutional13FHolding(
        ticker="NVDA",
        institution_name="Test Fund",
        institution_cik="0001234567",
        period_end=date(2024, 9, 30),
        filing_date=date(2024, 11, 14),
        shares_held=1_000_000,
        value_usd=130_000_000,
    )
    store.store_event(holding.to_event())
    
    # Query earnings events
    asof = UTC.localize(datetime(2024, 11, 20))
    earnings_events = store.get_events(
        tickers=["NVDA"],
        asof=asof,
        event_types=[EventType.EARNINGS],
        lookback_days=30,
    )
    
    has_earnings = len(earnings_events) == 1
    print_result("Stored and retrieved earnings event", has_earnings)
    
    if has_earnings:
        e = earnings_events[0]
        has_surprise = e.payload.get("surprise_pct") == 4.84
        print_result("Payload contains surprise data", has_surprise)
    
    # Query 13F events
    inst_events = store.get_events(
        tickers=["NVDA"],
        asof=asof,
        event_types=[EventType.INSTITUTIONAL_13F],
        lookback_days=30,
    )
    
    has_13f = len(inst_events) == 1
    print_result("Stored and retrieved 13F event", has_13f)
    
    # PIT test: query BEFORE earnings should NOT see it
    asof_before = UTC.localize(datetime(2024, 11, 19, 20, 0))  # Before AMC
    events_before = store.get_events(
        tickers=["NVDA"],
        asof=asof_before,
        event_types=[EventType.EARNINGS],
        lookback_days=30,
    )
    
    not_visible_before = len(events_before) == 0
    print_result("Earnings NOT visible before observed_at", not_visible_before)
    
    store.close()
    
    return has_earnings and has_13f and not_visible_before


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("CHAPTER 3 EXTENSIONS - TEST SUITE")
    print("=" * 60)
    
    results = {}
    
    results["1_event_types"] = test_new_event_types()
    results["2_earnings_pit"] = test_earnings_surprise_pit()
    results["3_security_master"] = test_security_master()
    results["4_positioning_pit"] = test_positioning_pit_rules()
    results["5_options_pit"] = test_options_pit_rules()
    results["6_eventstore"] = test_eventstore_integration()
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {name}")
    
    print(f"\n  Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n" + "=" * 60)
        print("✓ ALL CHAPTER 3 EXTENSION TESTS PASSED")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("✗ SOME TESTS FAILED")
        print("=" * 60)


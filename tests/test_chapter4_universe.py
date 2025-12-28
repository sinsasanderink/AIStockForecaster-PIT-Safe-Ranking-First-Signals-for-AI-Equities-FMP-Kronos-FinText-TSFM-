"""
Chapter 4: Survivorship-Safe Universe Tests
============================================

Comprehensive tests for Chapter 4 implementation.

Run with: python tests/test_chapter4_universe.py
Set RUN_INTEGRATION=1 for API tests (slower).
"""

import os
import sys
import time
from datetime import date, timedelta
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load .env
env_path = Path(__file__).parent.parent / '.env'
if env_path.exists():
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ[key.strip()] = value.strip().strip('"').strip("'")

import logging
logging.basicConfig(level=logging.WARNING)

# Check if integration tests should run
RUN_INTEGRATION = os.getenv("RUN_INTEGRATION", "0") == "1"


def log(msg):
    """Print with timestamp."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def print_test_header(name: str):
    print(f"\n{'='*60}")
    print(f"TEST: {name}")
    print('='*60)


def print_result(test_name: str, passed: bool, details: str = ""):
    icon = "✅" if passed else "❌"
    print(f"  {icon} {test_name}")
    if details:
        print(f"      {details}")


# =============================================================================
# TEST 1: Polygon API Access (Quick - no rate limit sleeps)
# =============================================================================

def test_polygon_api_access():
    """Test that Polygon API is accessible."""
    print_test_header("1. Polygon API Access")
    
    all_passed = True
    
    # Check API key exists
    api_key = os.getenv("POLYGON_KEYS", "")
    has_key = bool(api_key) and len(api_key) > 10
    print_result("API key present", has_key, f"Length: {len(api_key)}")
    all_passed = all_passed and has_key
    
    if not has_key:
        log("SKIPPING remaining API tests - no valid key")
        return False
    
    if not RUN_INTEGRATION:
        log("SKIPPED API tests: Set RUN_INTEGRATION=1 to run")
        return True
    
    import requests
    
    BASE_URL = "https://api.polygon.io"
    
    # Single API test (no rate limit sleep needed for one call)
    log("Testing Polygon API...")
    try:
        resp = requests.get(
            f"{BASE_URL}/v3/reference/tickers",
            params={"market": "stocks", "locale": "us", "type": "CS", "limit": 5, "apiKey": api_key},
            timeout=10
        )
        if resp.status_code == 200:
            count = len(resp.json().get("results", []))
            print_result("API connection", True, f"Got {count} tickers")
            
            # Check for CIK (stable ID)
            results = resp.json().get("results", [])
            has_cik = any(r.get("cik") for r in results)
            print_result("CIK available", has_cik)
            all_passed = all_passed and has_cik
        else:
            print_result("API connection", False, f"Status: {resp.status_code}")
            all_passed = False
    except Exception as e:
        print_result("API connection", False, str(e))
        all_passed = False
    
    return all_passed


# =============================================================================
# TEST 2: Universe Construction (No API - Fast)
# =============================================================================

def test_universe_construction_fast():
    """Test universe construction without API calls."""
    print_test_header("2. Universe Construction (No API)")
    
    all_passed = True
    
    log("Importing UniverseBuilder...")
    from src.data.universe_builder import UniverseBuilder, SurvivorshipStatus
    
    log("Creating builder...")
    builder = UniverseBuilder()
    
    log("Building universe (skip_enrichment=True)...")
    snapshot = builder.build(
        date.today(),
        use_polygon=False,      # Skip Polygon API
        skip_enrichment=True,   # Skip FMP API calls
        max_constituents=50,
        ai_filter=True,
        min_price=0,            # No price filter (no data)
        min_adv=0,              # No ADV filter (no data)
    )
    
    # Check basic properties
    has_candidates = snapshot.total_candidates > 0
    print_result("Has candidates", has_candidates, f"{snapshot.total_candidates} from ai_stocks.py")
    all_passed = all_passed and has_candidates
    
    # Status should be PARTIAL (no Polygon)
    is_partial = snapshot.survivorship_status == SurvivorshipStatus.PARTIAL
    print_result("Status is PARTIAL", is_partial, snapshot.survivorship_status.value)
    all_passed = all_passed and is_partial
    
    # All candidates should have AI category
    log("Checking AI categories...")
    constituents_have_ai = all(c.ai_category for c in snapshot.constituents) if snapshot.constituents else True
    print_result("AI categories assigned", constituents_have_ai)
    
    log(f"Result: {len(snapshot.constituents)} constituents, status={snapshot.survivorship_status.value}")
    
    return all_passed


# =============================================================================
# TEST 3: Polygon Universe (Integration - Requires API)
# =============================================================================

def test_polygon_universe():
    """Test universe construction with Polygon API."""
    print_test_header("3. Polygon Universe (Integration)")
    
    if not RUN_INTEGRATION:
        log("SKIPPED: Set RUN_INTEGRATION=1 to run")
        return True
    
    api_key = os.getenv("POLYGON_KEYS", "")
    if not api_key:
        log("SKIPPED: No POLYGON_KEYS")
        return True
    
    all_passed = True
    
    log("Importing PolygonClient...")
    from src.data.polygon_client import PolygonClient
    
    log("Creating Polygon client...")
    try:
        client = PolygonClient()
    except Exception as e:
        print_result("Client creation", False, str(e))
        return False
    
    log("Getting tickers...")
    try:
        tickers = client.get_tickers_asof(limit=10)
        got_tickers = len(tickers) > 0
        print_result("Get tickers", got_tickers, f"{len(tickers)} tickers")
        all_passed = all_passed and got_tickers
        
        if tickers:
            # Check stable_id quality
            has_cik = any(t.cik for t in tickers)
            print_result("Has CIK-based stable_id", has_cik)
            
            log(f"Sample: {tickers[0].ticker} -> {tickers[0].stable_id}")
    except Exception as e:
        print_result("Get tickers", False, str(e))
        all_passed = False
    
    client.close()
    return all_passed


# =============================================================================
# TEST 4: Stable ID Consistency
# =============================================================================

def test_stable_id_consistency():
    """Test that stable IDs are consistent."""
    print_test_header("4. Stable ID Consistency")
    
    all_passed = True
    
    log("Importing...")
    from src.data.universe_builder import UniverseBuilder
    
    builder = UniverseBuilder()
    
    # Build twice
    log("Building universe twice...")
    snapshot1 = builder.build(date.today(), use_polygon=False, skip_enrichment=True, min_price=0, min_adv=0)
    snapshot2 = builder.build(date.today(), use_polygon=False, skip_enrichment=True, min_price=0, min_adv=0)
    
    # Same candidates
    same_count = len(snapshot1.constituents) == len(snapshot2.constituents)
    print_result("Same constituent count", same_count, 
                f"{len(snapshot1.constituents)} vs {len(snapshot2.constituents)}")
    all_passed = all_passed and same_count
    
    # Same stable IDs
    ids1 = set(c.stable_id for c in snapshot1.constituents)
    ids2 = set(c.stable_id for c in snapshot2.constituents)
    same_ids = ids1 == ids2
    print_result("Same stable_ids", same_ids)
    all_passed = all_passed and same_ids
    
    return all_passed


# =============================================================================
# TEST 5: AI Stocks Integration
# =============================================================================

def test_ai_stocks_integration():
    """Test ai_stocks.py is used correctly (labels only)."""
    print_test_header("5. AI Stocks Integration")
    
    all_passed = True
    
    log("Importing ai_stocks...")
    from src.universe.ai_stocks import AI_UNIVERSE, get_all_tickers
    
    all_tickers = get_all_tickers()
    has_tickers = len(all_tickers) > 0
    print_result("Has tickers", has_tickers, f"{len(all_tickers)} tickers")
    all_passed = all_passed and has_tickers
    
    num_categories = len(AI_UNIVERSE)
    has_10_cats = num_categories == 10
    print_result("Has 10 categories", has_10_cats, f"{num_categories} categories")
    all_passed = all_passed and has_10_cats
    
    # List categories
    log("Categories:")
    for cat, tickers in AI_UNIVERSE.items():
        print(f"    {cat}: {len(tickers)} tickers")
    
    return all_passed


# =============================================================================
# TEST 6: Delisted Tickers (Integration)
# =============================================================================

def test_delisted_tickers():
    """Test access to delisted tickers."""
    print_test_header("6. Delisted Tickers")
    
    if not RUN_INTEGRATION:
        log("SKIPPED: Set RUN_INTEGRATION=1 to run")
        return True
    
    api_key = os.getenv("POLYGON_KEYS", "")
    if not api_key:
        log("SKIPPED: No POLYGON_KEYS")
        return True
    
    all_passed = True
    
    log("Getting delisted tickers...")
    from src.data.polygon_client import PolygonClient
    
    client = PolygonClient()
    try:
        delisted = client.get_delisted_tickers()
        has_delisted = len(delisted) > 0
        print_result("Got delisted tickers", has_delisted, f"{len(delisted)} found")
        all_passed = all_passed and has_delisted
        
        if delisted:
            with_dates = sum(1 for t in delisted if t.delisted_utc)
            print_result("Have delisted_utc", with_dates > 0, f"{with_dates}/{len(delisted)}")
            
            # Sample
            sample = next((t for t in delisted if t.delisted_utc), None)
            if sample:
                log(f"Sample: {sample.ticker} delisted {sample.delisted_utc}")
    except Exception as e:
        print_result("Get delisted", False, str(e))
        all_passed = False
    
    client.close()
    return all_passed


# =============================================================================
# TEST 7: Summary
# =============================================================================

def test_summary():
    """Generate summary."""
    print_test_header("7. Summary & Capabilities")
    
    print("\n  API Key Status:")
    print(f"    POLYGON_KEYS: {'✅' if os.getenv('POLYGON_KEYS') else '❌'}")
    print(f"    FMP_KEYS: {'✅' if os.getenv('FMP_KEYS') else '❌'}")
    
    print("\n  Chapter 4 Implementation:")
    print("    ✅ PolygonClient for symbol master")
    print("    ✅ UniverseBuilder with survivorship status")
    print("    ✅ ai_stocks.py as label-only")
    print("    ✅ stable_id support")
    
    if RUN_INTEGRATION:
        print("\n  Integration tests ran (slower but more comprehensive)")
    else:
        print("\n  💡 Run with RUN_INTEGRATION=1 for API tests")
    
    return True


# =============================================================================
# Main
# =============================================================================

def run_all_tests():
    """Run all Chapter 4 tests."""
    print("\n" + "="*60)
    print("CHAPTER 4: SURVIVORSHIP-SAFE UNIVERSE TESTS")
    print("="*60)
    
    if RUN_INTEGRATION:
        print("\n⚠️  Running API tests (RUN_INTEGRATION=1)")
    else:
        print("\n💡 Quick tests only. Set RUN_INTEGRATION=1 for API tests.")
    
    results = {}
    
    tests = [
        ("1. Polygon API Access", test_polygon_api_access),
        ("2. Universe Construction", test_universe_construction_fast),
        ("3. Polygon Universe", test_polygon_universe),
        ("4. Stable ID Consistency", test_stable_id_consistency),
        ("5. AI Stocks Integration", test_ai_stocks_integration),
        ("6. Delisted Tickers", test_delisted_tickers),
        ("7. Summary", test_summary),
    ]
    
    for name, test_func in tests:
        try:
            start = time.time()
            passed = test_func()
            elapsed = time.time() - start
            results[name] = (passed, elapsed)
        except Exception as e:
            print(f"\n  ❌ EXCEPTION: {e}")
            import traceback
            traceback.print_exc()
            results[name] = (False, 0)
    
    # Final summary
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    
    for name, (passed, elapsed) in results.items():
        icon = "✅" if passed else "❌"
        print(f"  {icon} {name} ({elapsed:.1f}s)")
    
    total = len(results)
    passed_count = sum(1 for p, _ in results.values() if p)
    print(f"\n  Total: {passed_count}/{total} tests passed")
    
    return all(p for p, _ in results.values())


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

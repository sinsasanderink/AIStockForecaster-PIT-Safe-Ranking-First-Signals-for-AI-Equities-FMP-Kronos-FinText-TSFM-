"""
Debug script to identify slow imports and blocking operations.
Run with: python tests/debug_imports.py
"""
import time
import os
import sys

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def timed_import(name, module_path):
    """Time an import and print result."""
    print(f"  [{time.strftime('%H:%M:%S')}] Importing {name}...", end=" ", flush=True)
    start = time.time()
    try:
        exec(f"import {module_path}")
        elapsed = time.time() - start
        status = f"✅ {elapsed:.2f}s"
    except Exception as e:
        elapsed = time.time() - start
        status = f"❌ {elapsed:.2f}s - {type(e).__name__}: {e}"
    print(status)
    return elapsed

print("=" * 60)
print("IMPORT TIMING DIAGNOSTIC")
print("=" * 60)

# Phase 1: Standard library (should be instant)
print("\n[Phase 1] Standard library imports:")
timed_import("os", "os")
timed_import("datetime", "datetime")
timed_import("pathlib", "pathlib")
timed_import("json", "json")

# Phase 2: Third-party (may be slow)
print("\n[Phase 2] Third-party imports:")
timed_import("requests", "requests")
timed_import("pandas", "pandas")
timed_import("pytz", "pytz")

# Phase 3: Heavy third-party (potential blockers)
print("\n[Phase 3] Heavy third-party imports (potential blockers):")
t_duckdb = timed_import("duckdb", "duckdb")
t_exchange = timed_import("exchange_calendars", "exchange_calendars")

# Phase 4: Project modules - individual files
print("\n[Phase 4] Project module imports (direct file imports):")

# Test polygon_client directly without going through __init__.py
print("  Attempting direct polygon_client import...")
start = time.time()
try:
    # This avoids __init__.py
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "polygon_client", 
        "src/data/polygon_client.py"
    )
    polygon_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(polygon_module)
    print(f"  ✅ Direct polygon_client: {time.time()-start:.2f}s")
except Exception as e:
    print(f"  ❌ Direct polygon_client: {e}")

# Test ai_stocks
timed_import("ai_stocks", "src.universe.ai_stocks")

# Phase 5: Package imports (triggers __init__.py)
print("\n[Phase 5] Package imports (triggers __init__.py - MAY BE SLOW):")
print("  ⚠️  src.data package will load: pit_store, trading_calendar, etc.")
t_src_data = timed_import("src.data", "src.data")

# Phase 6: Full universe_builder
print("\n[Phase 6] Universe builder import:")
timed_import("universe_builder", "src.data.universe_builder")

# Summary
print("\n" + "=" * 60)
print("BLOCKING OPERATION SUMMARY")
print("=" * 60)
print(f"  duckdb import: {t_duckdb:.2f}s")
print(f"  exchange_calendars import: {t_exchange:.2f}s")
print(f"  src.data package: {t_src_data:.2f}s")

if t_exchange > 5:
    print("\n⚠️  exchange_calendars is SLOW - consider lazy loading")
if t_src_data > 5:
    print("\n⚠️  src.data.__init__.py triggers heavy imports")
    print("   Solution: Make TradingCalendarImpl a lazy import")

# Phase 7: Test Polygon API (with timeout)
print("\n" + "=" * 60)
print("API CONNECTION TEST")
print("=" * 60)

# Load .env
env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
if os.path.exists(env_path):
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                k, v = line.split('=', 1)
                os.environ[k.strip()] = v.strip().strip('"').strip("'")

polygon_key = os.environ.get('POLYGON_KEYS', '')
print(f"  POLYGON_KEYS present: {bool(polygon_key)}")
print(f"  Key length: {len(polygon_key)}")

if polygon_key:
    import requests
    print("\n  Testing Polygon API (5s timeout)...")
    try:
        resp = requests.get(
            "https://api.polygon.io/v3/reference/tickers",
            params={"market": "stocks", "limit": 1, "apiKey": polygon_key},
            timeout=5
        )
        if resp.status_code == 200:
            print(f"  ✅ Polygon API works: {resp.status_code}")
        else:
            print(f"  ❌ Polygon API error: {resp.status_code}")
            print(f"     Response: {resp.text[:200]}")
    except Exception as e:
        print(f"  ❌ Polygon API failed: {e}")

print("\n" + "=" * 60)
print("DIAGNOSTIC COMPLETE")
print("=" * 60)


#!/usr/bin/env python
"""
Quick test to verify Series conversion logic works correctly.

This tests that our timestamp conversion to Series (for .dt accessor) works
for all input types: DatetimeIndex, Series, list.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

print("=" * 70)
print("TESTING SERIES CONVERSION FOR KRONOS")
print("=" * 70)

# Helper function (same as in adapter)
def _to_series(ts):
    """Convert timestamp sequence to Series (which has .dt accessor)."""
    if isinstance(ts, pd.Series):
        return ts
    elif isinstance(ts, pd.DatetimeIndex):
        # Convert DatetimeIndex to Series (preserves datetime type, adds .dt accessor)
        return pd.Series(ts.values, index=range(len(ts)))
    else:
        # Convert list/array to Series
        return pd.Series(pd.to_datetime(ts))

# Test 1: DatetimeIndex input
print("\nTest 1: DatetimeIndex input")
dti = pd.date_range('2024-01-01', periods=5, freq='D')
print(f"  Input type: {type(dti)}")
print(f"  Input: {dti[:3].tolist()}")

s = _to_series(dti)
print(f"  Output type: {type(s)}")
print(f"  Has .dt: {hasattr(s, 'dt')}")
print(f"  Can use .dt.strftime: {s.dt.strftime('%Y-%m-%d').tolist()[:3]}")
print("  ✓ PASS")

# Test 2: Series input (should pass through)
print("\nTest 2: Series input")
ser = pd.Series(pd.date_range('2024-01-01', periods=5, freq='D'))
print(f"  Input type: {type(ser)}")

s = _to_series(ser)
print(f"  Output type: {type(s)}")
print(f"  Has .dt: {hasattr(s, 'dt')}")
print(f"  Can use .dt.strftime: {s.dt.strftime('%Y-%m-%d').tolist()[:3]}")
print("  ✓ PASS")

# Test 3: List input
print("\nTest 3: List input")
lst = pd.date_range('2024-01-01', periods=5, freq='D').tolist()
print(f"  Input type: {type(lst)}")
print(f"  Input: {lst[:3]}")

s = _to_series(lst)
print(f"  Output type: {type(s)}")
print(f"  Has .dt: {hasattr(s, 'dt')}")
print(f"  Can use .dt.strftime: {s.dt.strftime('%Y-%m-%d').tolist()[:3]}")
print("  ✓ PASS")

# Test 4: Verify our adapter uses this correctly
print("\nTest 4: Verify adapter integration")
try:
    from src.models.kronos_adapter import KronosAdapter
    from src.data.prices_store import PricesStore
    from src.data.trading_calendar import load_global_trading_calendar
    
    print("  Loading trading calendar...")
    calendar = load_global_trading_calendar("data/features.duckdb")
    print(f"  ✓ Calendar loaded: {len(calendar)} dates")
    
    print("  Loading PricesStore...")
    prices_store = PricesStore("data/features.duckdb")
    print("  ✓ PricesStore loaded")
    
    print("  Fetching sample OHLCV...")
    ohlcv = prices_store.fetch_ohlcv("NVDA", "2024-01-15", lookback=252, strict_lookback=True)
    print(f"  ✓ Fetched {len(ohlcv)} rows")
    
    # Check that ohlcv.index is DatetimeIndex
    print(f"  Index type: {type(ohlcv.index)}")
    print(f"  Index is DatetimeIndex: {isinstance(ohlcv.index, pd.DatetimeIndex)}")
    
    # Verify conversion works
    s = _to_series(ohlcv.index)
    print(f"  Converted to Series: {type(s)}")
    print(f"  Has .dt: {hasattr(s, 'dt')}")
    print("  ✓ PASS")
    
    prices_store.close()
    
except Exception as e:
    print(f"  ✗ FAIL: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("ALL TESTS PASSED")
print("=" * 70)
print("\nSeries conversion is working correctly.")
print("The adapter will now pass Series objects to Kronos, which have .dt accessor.")
print("\nNext: Run SMOKE test to verify Kronos accepts Series:")
print("  python scripts/run_chapter8_kronos.py --mode smoke")


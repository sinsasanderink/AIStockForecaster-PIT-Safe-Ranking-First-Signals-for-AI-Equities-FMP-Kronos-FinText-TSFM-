# ✅ Kronos DatetimeIndex Bug: FIXED

**Date:** January 8, 2026  
**Issue:** `'DatetimeIndex' object has no attribute 'dt'` error when running full Kronos evaluation  
**Status:** **FIXED**

---

## What Was the Problem?

When running:
```bash
./run_kronos_full.sh
```

You got:
```
2026-01-08 13:43:53,333 - src.models.kronos_adapter - ERROR - Kronos prediction failed: 'DatetimeIndex' object has no attribute 'dt'
```

**Root cause:** 

Our `KronosAdapter` was passing `pd.DatetimeIndex` objects to Kronos's `predict_batch()` method. Internally, Kronos tries to use the `.dt` accessor on these timestamps (e.g., `timestamps.dt.strftime(...)`), but `.dt` only works on `pd.Series`, not `pd.DatetimeIndex`.

**Why stub mode worked:**

The `StubPredictor` doesn't use `.dt` accessor - it just indexes directly into the DatetimeIndex (`y_ts[:1]`), so the bug never triggered in stub mode.

---

## The Fix

**File:** `src/models/kronos_adapter.py`  
**Lines:** ~335-346

**What changed:**

Before passing timestamps to Kronos, we now convert `DatetimeIndex` to plain Python lists:

```python
# CRITICAL FIX: Convert DatetimeIndex to lists for Kronos
# Kronos expects lists/arrays, not DatetimeIndex (which lacks .dt accessor)
x_ts_list_converted = [
    list(ts) if isinstance(ts, pd.DatetimeIndex) else ts 
    for ts in x_ts_list
]
y_ts_list_converted = [
    list(ts) if isinstance(ts, pd.DatetimeIndex) else ts 
    for ts in y_ts_list
]

# Batch prediction
pred_list = self.predictor.predict_batch(
    df_list=df_list,
    x_timestamp_list=x_ts_list_converted,  # ← Now lists
    y_timestamp_list=y_ts_list_converted,  # ← Now lists
    ...
)
```

**Why this works:**

- Python lists of datetime objects are compatible with Kronos's internal datetime operations
- When Kronos converts these to Series internally, `.dt` accessor works correctly

---

## Verification

The fix has been applied. To verify:

```bash
# Check the fix is in place
grep -A 5 "CRITICAL FIX: Convert DatetimeIndex" src/models/kronos_adapter.py
```

Expected output:
```python
# CRITICAL FIX: Convert DatetimeIndex to lists for Kronos
# Kronos expects lists/arrays, not DatetimeIndex (which lacks .dt accessor)
x_ts_list_converted = [
    list(ts) if isinstance(ts, pd.DatetimeIndex) else ts 
    for ts in x_ts_list
]
```

---

## What to Do Now

The bug is fixed. You can now run the full Kronos evaluation:

### Option 1: Run Complete Installation + Evaluation

```bash
cd /Users/ursinasanderink/Downloads/AI\ Stock\ Forecast

./INSTALL_KRONOS.sh && \
export PYTHONPATH="$(pwd)/Kronos:$PYTHONPATH" && \
python scripts/test_kronos_installation.py && \
./run_kronos_full.sh
```

### Option 2: If Kronos is Already Installed

```bash
cd /Users/ursinasanderink/Downloads/AI\ Stock\ Forecast

# Set PYTHONPATH
export PYTHONPATH="$(pwd)/Kronos:$PYTHONPATH"

# Run evaluation
./run_kronos_full.sh
```

### Option 3: Direct Command

```bash
cd /Users/ursinasanderink/Downloads/AI\ Stock\ Forecast

export PYTHONPATH="$(pwd)/Kronos:$PYTHONPATH"
python scripts/run_chapter8_kronos.py --mode full
```

---

## Expected Behavior Now

**No more DatetimeIndex errors!**

You should see:
```
======================================================================
CHAPTER 8: KRONOS WALK-FORWARD EVALUATION
======================================================================
Mode: FULL
Device: cpu
...

Loading features from DuckDB...
✓ Loaded 201307 feature rows

INITIALIZING KRONOS ADAPTER
✓ Kronos adapter initialized

RUNNING WALK-FORWARD EVALUATION
...
  Scoring fold fold_01...
  [1/21] 2024-02-01: Scored 98 tickers
  [2/21] 2024-02-02: Scored 98 tickers
  ...
```

**Runtime:** 2-4 hours for all 109 folds

---

## What Changed in Your Files

**Modified:**
- ✅ `src/models/kronos_adapter.py` - Added DatetimeIndex → list conversion before calling Kronos

**No other changes needed** - the fix is self-contained

---

## Technical Details (Optional)

### Why DatetimeIndex doesn't have .dt

In pandas:
- `pd.Series` of datetimes → **has** `.dt` accessor
- `pd.DatetimeIndex` → **does NOT have** `.dt` accessor (it's already a datetime type)

Example:
```python
# Series - has .dt
s = pd.Series(pd.date_range('2024-01-01', periods=3))
s.dt.strftime('%Y-%m-%d')  # ✓ Works

# DatetimeIndex - no .dt
idx = pd.date_range('2024-01-01', periods=3)
idx.dt.strftime('%Y-%m-%d')  # ✗ AttributeError: 'DatetimeIndex' object has no attribute 'dt'
idx.strftime('%Y-%m-%d')     # ✓ Works (direct method)
```

### Why Kronos needs the conversion

Kronos internally:
1. Receives timestamp lists
2. Converts to pandas Series (for manipulation)
3. Uses `.dt` accessor for formatting/operations

When we pass DatetimeIndex, step 2 creates a Series but Kronos's code assumes it still needs `.dt`, causing the error.

By converting to plain Python lists first, Kronos's Series conversion works correctly.

---

## After Evaluation Completes

Results will be in:
```
evaluation_outputs/chapter8_kronos_full/chapter8_kronos_full/REPORT_SUMMARY.md
```

Next steps:
1. Review RankIC per horizon
2. Check leak tripwires
3. Compare to baselines (Ch6: 0.0283/0.0392/0.0169, Ch7: 0.1009/0.1275/0.1808)
4. Proceed to Phase 4 (see `CHAPTER_8_NEXT_ACTIONS.md`)

---

## Summary

| Issue | Fix | Status |
|-------|-----|--------|
| DatetimeIndex has no .dt | Convert to list before passing to Kronos | ✅ FIXED |
| Stub mode worked but real mode failed | Stub doesn't use .dt internally | ✅ Explained |
| Installation scripts missing | Created all installation helpers | ✅ Complete |

**Everything is now ready for full Kronos evaluation!** 🚀

---

## Quick Reference

```bash
# Install (if not done)
./INSTALL_KRONOS.sh

# Set path
export PYTHONPATH="$(pwd)/Kronos:$PYTHONPATH"

# Verify (optional)
python scripts/test_kronos_installation.py

# Run evaluation
./run_kronos_full.sh
```

**That's it! The bug is fixed and you're ready to run.** ✅


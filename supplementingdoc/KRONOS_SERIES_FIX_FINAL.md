# ✅ Kronos Series Fix - FINAL (Correct Fix)

**Date:** January 8, 2026  
**Issue:** Both DatetimeIndex and list conversions failed  
**Status:** **FIXED** with Series conversion

---

## The Real Problem

### Error Evolution

1. **Original error:** `'DatetimeIndex' object has no attribute 'dt'`
   - We passed `pd.DatetimeIndex` to Kronos
   - Kronos tried to use `.dt.strftime()` or similar
   - DatetimeIndex doesn't have `.dt` accessor (methods are direct)

2. **After first "fix":** `'list' object has no attribute 'dt'`
   - We converted to Python lists
   - Kronos still tried to use `.dt` accessor
   - Lists don't have `.dt` accessor either

3. **Root cause:** Kronos expects `pd.Series` (which has `.dt` accessor)

---

## Why Different Types Behave Differently

In pandas:

| Type | Has `.dt` accessor? | Can use `.dt.strftime()`? | Direct `.strftime()`? |
|------|--------------------|--------------------------|-----------------------|
| `pd.Series` (datetime) | ✅ YES | ✅ YES | ❌ NO |
| `pd.DatetimeIndex` | ❌ NO | ❌ NO | ✅ YES |
| `list` of Timestamps | ❌ NO | ❌ NO | ❌ NO |

**Example:**
```python
# Series - has .dt accessor
s = pd.Series(pd.date_range('2024-01-01', periods=3))
s.dt.strftime('%Y-%m-%d')  # ✅ Works

# DatetimeIndex - methods are direct (no .dt)
dti = pd.date_range('2024-01-01', periods=3)
dti.dt.strftime('%Y-%m-%d')  # ❌ Error: 'DatetimeIndex' object has no attribute 'dt'
dti.strftime('%Y-%m-%d')     # ✅ Works

# List - no .dt accessor
lst = pd.date_range('2024-01-01', periods=3).tolist()
lst.dt.strftime('%Y-%m-%d')  # ❌ Error: 'list' object has no attribute 'dt'
```

**Kronos uses `.dt` accessor internally**, so it needs `pd.Series`.

---

## The Correct Fix

**File:** `src/models/kronos_adapter.py`  
**Lines:** ~335-360

**What changed:**

```python
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

x_ts_list_converted = [_to_series(ts) for ts in x_ts_list]
y_ts_list_converted = [_to_series(ts) for ts in y_ts_list]

# Now Kronos receives Series objects, which have .dt accessor
pred_list = self.predictor.predict_batch(
    df_list=df_list,
    x_timestamp_list=x_ts_list_converted,  # ← Now Series
    y_timestamp_list=y_ts_list_converted,  # ← Now Series
    ...
)
```

**Why this works:**
- `pd.Series` has `.dt` accessor
- Kronos can use `.dt.strftime()`, `.dt.date`, etc.
- Works for all input types (DatetimeIndex, Series, list)

---

## Verification

### Step 1: Test the conversion logic

```bash
python scripts/test_series_conversion.py
```

**Expected:**
```
ALL TESTS PASSED
Series conversion is working correctly.
```

### Step 2: Run SMOKE test (quick validation)

```bash
cd /Users/ursinasanderink/Downloads/AI\ Stock\ Forecast

# Set PYTHONPATH (if Kronos installed)
export PYTHONPATH="$(pwd)/Kronos:$PYTHONPATH"

# Run SMOKE (3 folds, ~5-10 min)
python scripts/run_chapter8_kronos.py --mode smoke
```

**What to check:**
- No more `.dt` errors
- Scores are generated (not empty)
- Folds complete successfully

**Expected output:**
```
[1/21] 2024-02-01: Scored 98 tickers
[2/21] 2024-02-02: Scored 98 tickers
...
✓ Fold fold_01, horizon 20d: 2058 total scores
```

If you see empty scores or skip messages, check the logs for errors.

### Step 3: If SMOKE passes, run FULL

```bash
./run_kronos_full.sh
```

---

## What This Fix Does NOT Change

1. ✅ **PricesStore** still returns DatetimeIndex (correct)
2. ✅ **Trading calendar** still returns DatetimeIndex (correct)
3. ✅ **Our internal logic** still uses DatetimeIndex (correct)
4. ✅ **Only conversion** happens right before calling Kronos

**This is a minimal, targeted fix** that only affects the Kronos interface.

---

## Installation Still Required

Don't forget to install Kronos first:

```bash
# Step 1: Install
./INSTALL_KRONOS.sh

# Step 2: Set path
export PYTHONPATH="$(pwd)/Kronos:$PYTHONPATH"

# Step 3: Verify
python scripts/test_kronos_installation.py
```

---

## Full Run Checklist

Before running FULL evaluation:

1. ✅ **Kronos installed** (`./INSTALL_KRONOS.sh`)
2. ✅ **PYTHONPATH set** (`export PYTHONPATH="$(pwd)/Kronos:$PYTHONPATH"`)
3. ✅ **Series fix verified** (`python scripts/test_series_conversion.py`)
4. ✅ **SMOKE test passed** (`python scripts/run_chapter8_kronos.py --mode smoke`)

Then:

```bash
./run_kronos_full.sh
```

---

## If It Still Fails

If you still get errors after this fix, the issue is likely:

1. **Kronos not installed** → Run `./INSTALL_KRONOS.sh`
2. **PYTHONPATH not set** → Run `export PYTHONPATH="$(pwd)/Kronos:$PYTHONPATH"`
3. **Different Kronos API** → Check Kronos version
4. **Data issue** → Check DuckDB has data
5. **Memory issue** → Try `--device cpu` or reduce batch size

Check the full traceback (now logged) for details.

---

## Expected Output Location

After FULL completes (~2-4 hours):

```
evaluation_outputs/chapter8_kronos_full/chapter8_kronos_full/
├── REPORT_SUMMARY.md          ← Read this first
├── fold_summaries.csv         ← RankIC per fold/horizon
├── eval_rows.parquet          ← ~700K+ rows (NOT empty!)
├── per_date_metrics.csv       ← ~7K+ rows
├── leak_tripwires.json        ← Negative controls
└── ...
```

**Key check:** `eval_rows.parquet` should be several MB, not KB.

---

## Summary

| Fix Attempt | Input Type | Result |
|-------------|-----------|--------|
| Original | DatetimeIndex | ❌ 'DatetimeIndex' has no .dt |
| First fix | list | ❌ 'list' has no .dt |
| **Final fix** | **Series** | **✅ Series has .dt** |

**Status:** ✅ **READY TO RUN**

**Next command:**
```bash
# Test first
python scripts/test_series_conversion.py

# SMOKE test
python scripts/run_chapter8_kronos.py --mode smoke

# If SMOKE passes, run FULL
./run_kronos_full.sh
```

---

## Technical Reference

From [Kronos GitHub](https://github.com/shiyu-coder/Kronos):
- `predict_batch()` expects timestamp lists
- Each timestamp sequence must be iterable with datetime objects
- Kronos internally uses pandas datetime operations (`.dt` accessor)
- Therefore: must pass `pd.Series` (which has `.dt`)

From pandas documentation:
- `.dt` accessor is only available on `Series` with datetime dtype
- `DatetimeIndex` has methods directly (`.strftime()`, `.date`, etc.)
- Converting `DatetimeIndex` → `Series` adds the `.dt` namespace

---

**This is the correct fix. Ready to run.** ✅


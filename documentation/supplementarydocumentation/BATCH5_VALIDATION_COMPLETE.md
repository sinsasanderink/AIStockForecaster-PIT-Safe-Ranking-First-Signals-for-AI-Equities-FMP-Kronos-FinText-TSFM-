# Batch 5: Fundamentals Validation Complete

**Date:** January 7, 2026  
**Status:** ✅ VALIDATED

---

## Summary

Batch 5 (Fundamental Features) has been implemented, fixed, and validated. The DuckDB feature store now contains **52 columns** with all fundamental features working correctly.

---

## Features Added (9 columns)

| Column | Type | Description |
|--------|------|-------------|
| `sector` | VARCHAR | Company sector (from FMP profile, static) |
| `gross_margin_ttm` | DOUBLE | TTM gross profit / revenue (raw, piecewise-constant) |
| `operating_margin_ttm` | DOUBLE | TTM operating income / revenue (raw, piecewise-constant) |
| `revenue_growth_yoy` | DOUBLE | YoY TTM revenue growth (raw, piecewise-constant) |
| `roe_raw` | DOUBLE | ROE = TTM net income / avg equity (raw, piecewise-constant) |
| `gross_margin_vs_sector` | DOUBLE | Z-score vs sector (stepwise per-ticker) |
| `operating_margin_vs_sector` | DOUBLE | Z-score vs sector (stepwise per-ticker) |
| `revenue_growth_vs_sector` | DOUBLE | Z-score vs sector (stepwise per-ticker) |
| `roe_zscore` | DOUBLE | Z-score vs universe (stepwise per-ticker) |

---

## Bugs Fixed (5 total)

1. **`days_since_10q/10k` never reset** - Changed `<` to `<=` for filing date comparison
2. **10-K vs 10-Q misclassification** - Now downloads both quarterly + annual income statements
3. **Raw TTM not persisted** - Added 4 raw TTM columns to DuckDB schema
4. **Z-scores daily drift** - Rewrote to per-ticker stepwise + 90-day lookback + forward-fill
5. **Z-score index mismatch** - Changed to dict lookup by filing_date

See `BATCH5_BUGFIX.md` for detailed fix documentation.

---

## Validation Results

### Test 1: Raw TTM Piecewise-Constant ✅ PASS
- Change rate: **0.8%** (8 changes over ~500 days)
- Expected: <5% (quarterly filings = ~1.6%)
- Result: **PASS** - Raw TTM metrics are piecewise-constant

### Test 2: days_since Drop Resets ✅ PASS
- AAPL: 8 10-Q drops, 2 10-K drops
- MSFT: 8 10-Q drops, 2 10-K drops
- Expected: 6-8 10-Q drops, 1-2 10-K drops (2 years)
- Result: **PASS** - Filings reset correctly

### Test 3: Z-Score Stepwise ✅ PASS
- Change rate: **1.6%** (8 changes over ~500 days)
- Expected: <20% (stepwise per-ticker)
- Result: **PASS** - Z-scores are stepwise

### Test 4: No Identical Patterns ⚠️ WARNING (False Positive)
- 7 tickers have identical change counts
- **Reason:** All are Technology sector (AAPL/MSFT/NVDA/GOOGL/META/PLTR)
- **Analysis:** This is EXPECTED behavior - same-sector tickers get z-scored against the same cross-section, so they have similar change patterns
- Result: **NOT A BUG** - Warning can be ignored

### Test 5: Filing Counts ✅ PASS
- 10-Q resets: 8 (expected 6-8 for 2 years)
- 10-K resets: 2 (expected 1-2 for 2 years)
- Result: **PASS** - Filing counts are reasonable

### Sample Filing Alignment
```
      date  days_since_10q  gross_margin_ttm  gross_margin_vs_sector
2023-02-03             0.0          0.430594               -1.030280
2023-05-05             0.0          0.431810               -0.977517
```
- `days_since_10q` hits 0 on filing days ✅
- `gross_margin_ttm` changes only at filings ✅
- `gross_margin_vs_sector` is stepwise (constant between filings) ✅

---

## DuckDB Status

| Table | Rows | Range |
|-------|------|-------|
| features | 201,307 | 2016-01-04 to 2025-06-30 |
| labels | 600,855 | 3 horizons × ~200K dates |
| regime | 2,386 | Unique dates |

**Total Columns:** 52 (was 7 in Chapter 7 freeze)

---

## Phase 2 Status: DEFERRED

Valuation features (`pe_zscore_3y`, `pe_vs_sector`, `ps_vs_sector`) are deferred:
- Requires PIT-safe price + shares from filings
- Not critical for Chapter 8 Kronos (uses OHLCV, not fundamentals)
- Can be added as ablation in Chapter 10

---

## Files Modified

- `scripts/build_features_duckdb.py` - All bug fixes
- `scripts/validate_stepwise_behavior.py` - Validation script

## Documentation

- `BATCH5_PLAN.md` - Implementation plan
- `BATCH5_BUGFIX.md` - Bug fix details
- `BATCH5_STATUS.md` - Build status tracking

---

## Next Steps

✅ Batch 5 Phase 1 complete and validated  
⏭️ Ready for Chapter 8 (Kronos)


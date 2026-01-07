# Batch 5: Fundamentals - Current Status

**Date:** January 2, 2026 21:41 PM  
**Status:** 🟡 BUILD IN PROGRESS

---

## Summary

Batch 5 Phase 1 (filings-only fundamental features) has been **implemented and integrated** into `scripts/build_features_duckdb.py`. The DuckDB rebuild is currently running to compute the new features.

---

## What's Been Completed ✅

### 1. Implementation (COMPLETE)
- [x] Created `download_fundamental_data()` function
  - Downloads income statements (60 quarters, ~15 years)
  - Downloads balance sheets for ROE calculation
  - Extracts sector from company profile (documented as static)
- [x] Created `compute_fundamental_features()` function
  - TTM metrics: sum of last 4 quarters where filing_date <= asof_date
  - YoY growth: TTM now vs TTM 4 quarters ago
  - Cross-sectional z-scores per date, per sector (min 5 tickers)
  - Forward-fill behavior: features constant until next filing
- [x] Updated DuckDB schema
  - Added 5 new columns: `sector`, `gross_margin_vs_sector`, `operating_margin_vs_sector`, `revenue_growth_vs_sector`, `roe_zscore`
- [x] Integrated into build script
  - Step 6c: Download fundamental data
  - Step 6d: Compute fundamental features
- [x] Tests pass: 429/429

### 2. Features Implemented (Phase 1: Filings-Only)

| Feature | Source | Method |
|---------|--------|--------|
| `sector` | Company profile | Static (not PIT), documented limitation |
| `gross_margin_vs_sector` | TTM gross profit / TTM revenue | Cross-sectional z-score per date/sector |
| `operating_margin_vs_sector` | TTM operating income / TTM revenue | Cross-sectional z-score per date/sector |
| `revenue_growth_vs_sector` | (TTM now - TTM 4Q ago) / TTM 4Q ago | Cross-sectional z-score per date/sector |
| `roe_zscore` | TTM net income / avg equity (4Q) | Universe-wide z-score per date |

---

## What's In Progress ⏳

### DuckDB Rebuild (RUNNING)
- **PID:** 58285
- **Log:** `build_batch5_final.log`
- **Progress Checkpoints:**
  - ✅ Price data downloaded (100 tickers)
  - ✅ Benchmark/market/VIX downloaded
  - ✅ Momentum features computed
  - ✅ Event features computed (5,065 filing records)
  - ✅ Regime features computed
  - ⏳ **Fundamental data download:** IN PROGRESS (0-100 tickers)
  - ⏳ Fundamental features computation: PENDING
  - ⏳ Labels computation: PENDING
  - ⏳ DuckDB write: PENDING

### Estimated Time Remaining
- **Fundamental download:** ~35-40 minutes (7-8 min per 20 tickers × 5 batches)
- **Fundamental computation:** ~10-15 minutes (201K rows × 5 features)
- **Labels + dedup + DuckDB write:** ~5 minutes
- **Total ETA:** ~50-60 minutes from now (22:30-22:40 PM)

---

## Monitoring Commands

```bash
# Check if build is still running
ps aux | grep build_features | grep -v grep

# Monitor progress in real-time
tail -f build_batch5_final.log

# Check current progress
tail -20 build_batch5_final.log

# Check DuckDB size (should grow when written)
ls -lh data/features.duckdb
```

---

## What's Pending (Phase 2) ⏸️

### Valuation Features (Deferred)
- `pe_zscore_3y` - Requires PIT-safe price + shares from filings
- `pe_vs_sector` - Requires PIT-safe price + shares from filings
- `ps_vs_sector` - Requires PIT-safe price + shares from filings

**Reason for Deferral:** These features require extracting `weightedAverageShsOutDiluted` from filings and using as-of close from `features_df`. To avoid PIT leakage, this needs careful implementation of:
1. Price from `features_df` (already PIT-safe)
2. Shares from filings, forward-filled between filings
3. Market cap = as-of close × filing shares

**Recommendation:** Implement Phase 2 after validating Phase 1 results.

---

## Validation Plan (After Build Completes) 

### 1. Basic Verification
```bash
# A) Check DuckDB was updated
ls -lh data/features.duckdb  # Should be ~90-100MB, timestamp after build

# B) Check schema
python -c "
import duckdb
con = duckdb.connect('data/features.duckdb', read_only=True)
cols = con.execute('PRAGMA table_info(features)').df()
print(f'Total columns: {len(cols)}')
print('Fundamental columns:', cols[cols['name'].str.contains('sector|margin|growth|roe')]['name'].tolist())
"

# C) Check feature coverage
python scripts/validate_fundamental_pit.py  # (to be created)
```

### 2. PIT Safety Tests
- [ ] Stepwise behavior: features constant between filings
- [ ] Coverage ramp: <50% in 2016, >80% in 2024
- [ ] Sector sanity: No sector with >50% NaN on any date

### 3. Integration Tests
- [ ] Unit tests still pass (429 tests)
- [ ] Smoke test: `python scripts/run_chapter7_tabular_lgb.py --smoke`

---

## Known Issues & Limitations

1. **Sector assignment is static** (from current profile, not PIT)
   - Impact: LOW - sector changes are rare
   - Documented in code and plan

2. **FMP API is slow** (~4 seconds per ticker for fundamentals)
   - Impact: Build time ~35-40 minutes just for fundamental download
   - Mitigation: Results are cached

3. **Phase 2 (valuation features) deferred**
   - P/E, P/S require additional PIT-safe implementation
   - Will be added after Phase 1 validation

---

## Next Steps

1. **Wait for build to complete** (~50-60 min from 21:40 PM)
2. **Run validation tests** (see plan above)
3. **Create `BATCH5_VALIDATION_COMPLETE.md`** (similar to Batch 3/4)
4. **Update `ROADMAP.md`** to mark Phase 1 complete
5. **Decide on Phase 2:** Implement valuation features or proceed to Chapter 8

---

## Files Modified

- `scripts/build_features_duckdb.py` - Added fundamental download/computation functions
- `BATCH5_PLAN.md` - Implementation plan and PIT safety requirements
- `ROADMAP.md` - Updated to show Phase 1 implementation
- `BATCH4_VALIDATION_COMPLETE.md` - Documented deferred sector rotation features

---

## Build Command (if restart needed)

```bash
cd /Users/ursinasanderink/Downloads/AI\ Stock\ Forecast
nohup python -u scripts/build_features_duckdb.py --auto-normalize-splits > build_batch5_final.log 2>&1 &
echo $! > build_batch5.pid
echo "Build started, PID: $(cat build_batch5.pid)"

# Monitor
tail -f build_batch5_final.log
```


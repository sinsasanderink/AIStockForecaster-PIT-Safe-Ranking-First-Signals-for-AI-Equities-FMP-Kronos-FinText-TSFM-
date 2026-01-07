# Feature Store Expansion: Implementation Plan

**Created:** January 2, 2026  
**Status:** 🔄 READY TO EXECUTE  
**Target:** Expand DuckDB from 7 → 50 features before Chapter 8

---

## Executive Summary

**Current:** 7 features in `data/features.duckdb`  
**Target:** 50 features (13 for frozen baseline + 37 for Chapter 8+)  
**Time Estimate:** 90-120 minutes  
**Risk:** LOW (with backup strategy)

---

## Pre-Flight Checklist

### ✅ Prerequisites Met
- [x] Chapter 7 frozen and tagged (`chapter7-tabular-lgb-freeze`)
- [x] All 50 features implemented in `src/features/`
- [x] FMP Premium API access available
- [x] AlphaVantage API access available
- [x] Current DuckDB exists at `data/features.duckdb`

### ⏳ Safety Steps (Execute First)
- [ ] Back up current 7-feature DuckDB
- [ ] Update `.gitignore` for backup
- [ ] Document hash change expectation in `CHAPTER_7_FREEZE.md`

---

## Implementation Steps

### Phase 1: Safety Backup (5 min)

**Step 1.1:** Back up current DuckDB
```bash
cp data/features.duckdb data/features_chapter7_freeze.duckdb
```

**Step 1.2:** Add to `.gitignore`
```
# Chapter 7 frozen baseline (7-feature snapshot for reproducibility)
data/features_chapter7_freeze.duckdb
```

**Step 1.3:** Document hash change
Add to `CHAPTER_7_FREEZE.md` under "Reproducibility" section:
```markdown
### Data Hash Note

The frozen Chapter 7 baseline was trained on a 7-feature DuckDB snapshot:
- **Hash:** `f3899b37cb9f34f1`
- **Features:** mom_1m, mom_3m, mom_6m, mom_12m, adv_20d, vol_20d, vol_60d

After Chapter 8 feature expansion (7 → 50 features), the production DuckDB hash changed.
The frozen 7-feature snapshot is preserved at `data/features_chapter7_freeze.duckdb`.

To reproduce Chapter 7 exactly:
```bash
python scripts/run_chapter7_tabular_lgb.py --db-path data/features_chapter7_freeze.duckdb
```
```

---

### Phase 2: Wire Features (60-90 min)

This is the main work. We need to modify `scripts/build_features_duckdb.py` in two places:

#### 2.1: Update DuckDB Schema (add 43 columns)

**File:** `scripts/build_features_duckdb.py` line ~554

**Current schema (7 features):**
```sql
CREATE TABLE features (
    date DATE,
    ticker VARCHAR,
    stable_id VARCHAR,
    mom_1m DOUBLE,
    mom_3m DOUBLE,
    mom_6m DOUBLE,
    mom_12m DOUBLE,
    adv_20d DOUBLE,
    vol_20d DOUBLE,
    vol_60d DOUBLE,
    PRIMARY KEY (date, ticker)
)
```

**New schema (50 features):**
```sql
CREATE TABLE features (
    date DATE,
    ticker VARCHAR,
    stable_id VARCHAR,
    -- Momentum (4) ✅ EXISTING
    mom_1m DOUBLE,
    mom_3m DOUBLE,
    mom_6m DOUBLE,
    mom_12m DOUBLE,
    -- Volatility (3) ✅ EXISTING + NEW
    vol_20d DOUBLE,
    vol_60d DOUBLE,
    vol_of_vol DOUBLE,
    -- Drawdown (2) NEW
    max_drawdown_60d DOUBLE,
    dist_from_high_60d DOUBLE,
    -- Liquidity (2) ✅ EXISTING + NEW
    adv_20d DOUBLE,
    adv_60d DOUBLE,
    -- Relative Strength (2) NEW
    rel_strength_1m DOUBLE,
    rel_strength_3m DOUBLE,
    -- Beta (1) NEW
    beta_252d DOUBLE,
    -- Events/Earnings (12) NEW
    days_to_earnings DOUBLE,
    days_since_earnings DOUBLE,
    in_pead_window BOOLEAN,
    pead_window_day DOUBLE,
    last_surprise_pct DOUBLE,
    avg_surprise_4q DOUBLE,
    surprise_streak DOUBLE,
    surprise_zscore DOUBLE,
    earnings_vol DOUBLE,
    days_since_10k DOUBLE,
    days_since_10q DOUBLE,
    reports_bmo BOOLEAN,
    -- Fundamentals (7) NEW
    pe_zscore_3y DOUBLE,
    pe_vs_sector DOUBLE,
    ps_vs_sector DOUBLE,
    gross_margin_vs_sector DOUBLE,
    operating_margin_vs_sector DOUBLE,
    revenue_growth_vs_sector DOUBLE,
    roe_zscore DOUBLE,
    -- Regime (15) NEW
    vix_level DOUBLE,
    vix_percentile DOUBLE,
    vix_change_5d DOUBLE,
    vix_regime VARCHAR,
    market_return_5d DOUBLE,
    market_return_21d DOUBLE,
    market_return_63d DOUBLE,
    market_vol_21d DOUBLE,
    market_regime VARCHAR,
    above_ma_50 BOOLEAN,
    above_ma_200 BOOLEAN,
    ma_50_200_cross VARCHAR,
    tech_vs_staples DOUBLE,
    tech_vs_utilities DOUBLE,
    risk_on_indicator DOUBLE,
    -- Missingness (2) NEW
    coverage_pct DOUBLE,
    is_new_stock BOOLEAN,
    PRIMARY KEY (date, ticker)
)
```

#### 2.2: Compute Features (add feature generation logic)

**File:** `scripts/build_features_duckdb.py` line ~700 (in `main()` function)

**Current logic:** Only computes momentum + volume
**Needed:** Add calls to feature computation functions

**Priority Order:**
1. **Price/Volume** (7 new) - Easy, no API calls
2. **Missingness** (2 new) - Easy, diagnostic only
3. **Events** (12 new) - Medium, requires event store
4. **Regime** (15 new) - Easy/Medium, requires index data
5. **Fundamentals** (7 new) - Medium, requires quarterly data

---

### Phase 3: Rebuild DuckDB (15-30 min)

**Step 3.1:** Rebuild with expanded schema
```bash
python scripts/build_features_duckdb.py --auto-normalize-splits
```

**Expected output:**
- New features computed
- New data hash generated
- Row counts should be similar (~192K rows)
- Column count: 7 → 50

**Step 3.2:** Verify schema
```bash
python - <<'PY'
import duckdb
conn = duckdb.connect("data/features.duckdb", read_only=True)
result = conn.execute("DESCRIBE features").fetchall()
print(f"Total columns: {len(result)}")
for col in result:
    print(f"  {col[0]}: {col[1]}")
conn.close()
PY
```

Should show 53 columns (date, ticker, stable_id + 50 features).

---

### Phase 4: Verification (10 min)

**Step 4.1:** Smoke test with new DuckDB
```bash
python scripts/run_chapter7_tabular_lgb.py --smoke
```

**Expected:**
- ✅ Loads successfully
- ⚠️ Different data hash (expected!)
- ✅ Evaluation runs
- ✅ Outputs written

**Step 4.2:** Verify frozen baseline still works
```bash
python scripts/run_chapter7_tabular_lgb.py --smoke --db-path data/features_chapter7_freeze.duckdb
```

**Expected:**
- ✅ Hash matches frozen: `f3899b37cb9f34f1`
- ✅ Same scores as frozen baseline

**Step 4.3:** Run full test suite
```bash
pytest tests/ -v
```

**Expected:**
- ✅ 429/429 tests pass (tests use synthetic data, not DuckDB)

---

### Phase 5: Documentation (10 min)

**Step 5.1:** Update `ROADMAP.md`
- Change "Pre-Chapter-8: Feature Store Expansion" from ⏳ TODO → ✅ COMPLETE
- Add completion date and new data hash

**Step 5.2:** Update `PROJECT_DOCUMENTATION.md`
- Chapter 5 header: Change "partial pipeline ⚠️" → "complete pipeline ✅"
- Update feature table to show 50/50 features in DuckDB

**Step 5.3:** Update `AI_Stock_Forecaster_(FinText_+_Kronos_+_Context).ipynb`
- Chapter 8.0 Prerequisites: Change status from "TODO" → "COMPLETE"
- Add note about new data hash

**Step 5.4:** Create completion summary
- Document new data hash
- List all 43 features added
- Confirm backward compatibility via frozen snapshot

---

## Expected Outcomes

### ✅ Success Criteria
- [ ] New DuckDB has 50 features (vs 7 before)
- [ ] New data hash documented
- [ ] Frozen 7-feature DuckDB backed up
- [ ] Smoke tests pass with both databases
- [ ] Full test suite passes (429/429)
- [ ] Documentation updated

### 📊 Metrics
- **Schema:** 7 columns → 50 columns (+43)
- **Rows:** ~192K (unchanged)
- **Data hash:** `f3899b37cb9f34f1` → `[new_hash]`
- **Backward compatibility:** ✅ via `data/features_chapter7_freeze.duckdb`

### 🎯 Unlocks
- ✅ Chapter 8 (Kronos) can proceed
- ✅ Earnings gap diagnostic can run
- ✅ Event features available for gap mitigation
- ✅ Regime features available for Chapter 12
- ✅ Fundamentals available for Chapter 11

---

## Rollback Plan

If anything breaks:

1. **Restore old DuckDB:**
   ```bash
   cp data/features_chapter7_freeze.duckdb data/features.duckdb
   ```

2. **Verify restoration:**
   ```bash
   python scripts/run_chapter7_tabular_lgb.py --smoke
   ```

3. **Debug issue:**
   - Check build script logs
   - Verify feature code imports
   - Test individual feature functions

---

## Implementation Complexity Estimate

| Phase | Time | Difficulty | Risk |
|-------|------|------------|------|
| 1. Safety Backup | 5 min | Easy | None |
| 2.1 Schema Update | 15 min | Easy | Low |
| 2.2 Feature Wiring | 60-90 min | Medium | Medium |
| 3. Rebuild | 15-30 min | Easy | Low |
| 4. Verification | 10 min | Easy | None |
| 5. Documentation | 10 min | Easy | None |
| **TOTAL** | **105-150 min** | **Medium** | **Low** |

**Complexity breakdown:**
- **Easy (30%):** Schema, backup, docs
- **Medium (60%):** Feature wiring, especially events + regime
- **Hard (10%):** Debugging if API calls fail

---

## Next Steps

**Immediate (this session):**
1. Execute Phase 1 (Safety Backup)
2. Begin Phase 2 (Feature Wiring)
3. Continue until complete or next natural break point

**After completion:**
1. Run earnings gap diagnostic (optional, 10 min)
2. Start Chapter 8 (Kronos implementation)

---

**Ready to proceed? Let me know and I'll start with Phase 1!**


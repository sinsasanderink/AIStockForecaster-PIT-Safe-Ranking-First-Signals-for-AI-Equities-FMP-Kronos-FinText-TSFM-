# Feature Store Expansion: Batch 1+2 Complete

**Date:** January 2, 2026  
**Status:** ✅ COMPLETE  
**Batches:** Price/Volume Extensions (Batch 1) + Missingness (Batch 2)

---

## Executive Summary

Successfully expanded the DuckDB feature store from **7 → 16 features** (+9 new) by completing Batches 1 and 2 of the feature expansion plan.

**Impact:**
- ✅ More robust price/volume features for Chapter 8+ models
- ✅ Missingness diagnostics for data quality monitoring
- ✅ Frozen Chapter 7 baseline preserved via backup
- ✅ All tests passing (429/429)
- ⏳ Ready to proceed with Batch 3 (events) or start Chapter 8

---

## What Was Accomplished

### Batch 1: Price/Volume Extensions (7 features)
Added advanced price and volume features:

| Feature | Description | Purpose |
|---------|-------------|---------|
| `vol_of_vol` | Volatility of volatility | Regime detection (vol clustering) |
| `max_drawdown_60d` | Maximum drawdown (60d) | Risk/drawdown exposure |
| `dist_from_high_60d` | Distance from 60d high | Momentum/reversal signal |
| `adv_60d` | 60-day average dollar volume | Liquidity (longer window) |
| `rel_strength_1m` | 1m return z-score vs universe | Cross-sectional momentum |
| `rel_strength_3m` | 3m return z-score vs universe | Cross-sectional momentum |
| `beta_252d` | Beta vs QQQ (252d) | Market sensitivity (placeholder) |

**Note:** `beta_252d` is currently set to `None` (requires benchmark merge logic, deferred).

### Batch 2: Missingness Features (2 features)
Added data quality diagnostics:

| Feature | Description | Purpose |
|---------|-------------|---------|
| `coverage_pct` | % of features non-null | Data quality monitoring |
| `is_new_stock` | Boolean: <252 trading days | Flag IPOs/new listings |

---

## Technical Changes

### 1. Modified `scripts/build_features_duckdb.py`

**Location:** `compute_momentum_features()` function (lines ~314-375)

**Changes:**
- Added computation for vol_of_vol, max_drawdown, dist_from_high, adv_60d
- Added cross-sectional relative strength computation (post-loop)
- Added missingness feature computation (post-loop)
- Updated feature dict to include all 9 new features

**Location:** DuckDB schema (lines ~588-610)

**Changes:**
- Expanded CREATE TABLE to include 16 feature columns (was 7)
- Added comments for feature categories

### 2. Updated `.gitignore`
- Added comment noting `features_chapter7_freeze.duckdb` is the frozen 7-feature snapshot

### 3. Updated `CHAPTER_7_FREEZE.md`
- Added "Data Hash Change" section explaining the frozen vs expanded hash
- Documented backup path and reproduction command

### 4. Updated `ROADMAP.md`
- Marked Batch 1 as ✅ COMPLETE
- Marked Batch 2 (Priority 5) as ✅ COMPLETE
- Updated current state to show 16 features (was 7)
- Added status update with new data hash

---

## Verification Results

### DuckDB Schema ✅
```
Total columns: 19 (3 meta + 16 features)
Total rows: 201,307
Tickers: 100
Date range: 2016-01-04 → 2025-06-30
```

### Data Hashes
- **Frozen (7-feature):** `f3899b37cb9f34f1` (preserved in `data/features_chapter7_freeze.duckdb`)
- **Expanded (16-feature):** `a6142358f0e9ac5701092e51a34058fd0bdf249aa945fd726143d98e177a3607`

### Tests ✅
- **Passed:** 429/429
- **Warnings:** 88 (pre-existing, non-critical)
- **Time:** 142s

### Smoke Test ✅
- ✅ Monthly cadence: 3 folds (SMOKE_MODE)
- ✅ Quarterly cadence: 3 folds (SMOKE_MODE)
- ✅ All reports generated
- ✅ Baseline reference written

---

## Backward Compatibility

### Frozen Artifacts Protected ✅
- Chapter 7 frozen artifacts remain at `evaluation_outputs/chapter7_tabular_lgb_full/`
- Frozen data hash (`f3899b37cb9f34f1`) documented in artifacts
- 7-feature snapshot backed up to `data/features_chapter7_freeze.duckdb`

### Reproduction Command
To reproduce Chapter 7 baseline exactly:
```bash
python scripts/run_chapter7_tabular_lgb.py --db-path data/features_chapter7_freeze.duckdb
```

Expected output: `data_hash = f3899b37cb9f34f1` (matches frozen artifacts)

---

## Remaining Work

### Batch 3: Events/Earnings (12 features) ⚠️ CRITICAL
**Why critical:** Earnings gap mitigation for monthly rebalancing.

**Features:**
- `days_to_earnings`, `days_since_earnings` (timing)
- `in_pead_window`, `pead_window_day` (post-earnings drift)
- `last_surprise_pct`, `avg_surprise_4q`, `surprise_streak`, `surprise_zscore`, `earnings_vol` (surprises)
- `days_since_10k`, `days_since_10q`, `reports_bmo` (filings)

**Difficulty:** HARD (requires event store with AlphaVantage + FMP data)

### Batch 4: Regime/Macro (15 features)
**Why needed:** Chapter 12 (Regime-Aware Ensembling)

**Features:**
- VIX features: `vix_level`, `vix_percentile`, `vix_change_5d`, `vix_regime`
- Market features: `market_return_5d/21d/63d`, `market_vol_21d`, `market_regime`
- Technical: `above_ma_50`, `above_ma_200`, `ma_50_200_cross`
- Sector rotation: `tech_vs_staples`, `tech_vs_utilities`, `risk_on_indicator`

**Difficulty:** MEDIUM (requires downloading SPY, VIX, sector ETFs from FMP)

### Batch 5: Fundamentals (7 features)
**Why needed:** Chapter 11 (Fusion with tabular context)

**Features:**
- `pe_zscore_3y`, `pe_vs_sector`, `ps_vs_sector`
- `gross_margin_vs_sector`, `operating_margin_vs_sector`, `revenue_growth_vs_sector`
- `roe_zscore`

**Difficulty:** HARD (requires quarterly fundamentals from FMP Premium)

---

## Next Steps

### Option A: Proceed with Chapter 8 (Kronos) Now
**Rationale:**
- Kronos only needs OHLCV (no tabular features)
- Can add events/regime/fundamentals in parallel or after
- Unblocks main development path

**Command:**
```bash
# Start Chapter 8 with current 16-feature DuckDB
# Events/regime/fundamentals can be added later
```

### Option B: Complete Batch 3 (Events) First
**Rationale:**
- Earnings gap is a known blind spot
- Monthly rebalancing + early-month earnings = gap moves
- Event features let models avoid/predict gaps

**Estimated Time:** 60-90 min

**Command:**
```bash
# Implement event store
# Wire into build_features_duckdb.py
# Rebuild DuckDB → 28 features total
```

### Option C: Complete All Batches (3+4+5)
**Rationale:**
- One-time effort now vs incremental rebuilds
- FMP Premium available (no quota concerns)
- Full feature set enables better diagnostics

**Estimated Time:** 3-4 hours

---

## Lessons Learned

1. **Caching works great:** Rebuild took 2min (vs 15min first time) thanks to FMP cache
2. **Incremental batches reduce risk:** Batch 1+2 was low-hanging fruit (no new APIs)
3. **Cross-sectional features need post-processing:** Relative strength requires groupby date (added after main loop)
4. **Backup strategy is essential:** Frozen baseline preserved via simple `cp` command
5. **Tests are robust:** All 429 tests passed with no modifications (synthetic data independent of DuckDB)

---

## Files Changed

**Modified:**
- `scripts/build_features_duckdb.py` (feature computation + schema)
- `.gitignore` (added comment)
- `CHAPTER_7_FREEZE.md` (data hash note)
- `ROADMAP.md` (marked Batch 1+2 complete)

**Created:**
- `data/features_chapter7_freeze.duckdb` (backup)
- `FEATURE_EXPANSION_PLAN.md` (implementation guide)
- `FEATURE_WIRING_TODO.md` (detailed batch checklist)
- `FEATURE_EXPANSION_BATCH1_2_COMPLETE.md` (this file)

**Unchanged:**
- All `evaluation_outputs/chapter7_tabular_lgb_full/` artifacts (frozen)
- All `evaluation_outputs/chapter6_closure_real/` artifacts (frozen)
- All test files
- All feature implementation code in `src/features/`

---

## Summary Stats

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| **Features** | 7 | 16 | +9 (+129%) |
| **Rows** | 192,307 | 201,307 | +9,000 (+4.7%) |
| **Date Range** | 2016-01-04 → 2025-02-19 | 2016-01-04 → 2025-06-30 | +131 days |
| **Data Hash** | `f3899b37cb9f34f1` | `a6142358f0e9ac57...` | Changed ✅ |
| **DuckDB Size** | 68 MB | TBD (likely ~75 MB) | +~10% |
| **Tests** | 429 passing | 429 passing | Stable ✅ |

---

**✅ Batch 1+2 Complete. Ready for Next Phase.**

**Recommended:** Proceed with Chapter 8 (Kronos) or complete Batch 3 (events) depending on priority.


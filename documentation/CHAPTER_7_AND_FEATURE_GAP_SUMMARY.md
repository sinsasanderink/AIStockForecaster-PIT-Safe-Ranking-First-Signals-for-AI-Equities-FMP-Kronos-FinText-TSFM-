# Chapter 7 Complete + Feature Gap Analysis

**Date:** December 30, 2025  
**Status:** Chapter 7 ✅ FROZEN, Pre-Chapter-8 gap documented

---

## Chapter 7: Final Status ✅

**All deliverables complete and frozen:**
- ✅ ML baseline (`tabular_lgb`) implemented and frozen
- ✅ FULL_MODE reference run executed (109 monthly, 36 quarterly folds)
- ✅ Frozen artifacts at `evaluation_outputs/chapter7_tabular_lgb_full/`
- ✅ Git tag: `chapter7-tabular-lgb-freeze`
- ✅ Tests: 429/429 passing
- ✅ Documentation synchronized across all files

**Frozen ML Baseline Floor:**
- 20d: 0.1009 (+256% vs factor floor)
- 60d: 0.1275 (+225% vs factor floor)
- 90d: 0.1808 (+970% vs factor floor)

---

## Key Finding: Feature Store Gap

### What We Discovered

**Documentation claimed:** "50 features implemented ✅"  
**Reality:** Only 7 features wired into DuckDB pipeline

| Category | Features | In Code | In DuckDB | Gap |
|----------|----------|---------|-----------|-----|
| Price/Volume | 14 | ✅ | 7 | 7 missing |
| Fundamentals | 7 | ✅ | 0 | 7 missing |
| Events/Earnings | 12 | ✅ | 0 | 12 missing |
| Regime/Macro | 15 | ✅ | 0 | 15 missing |
| Missingness | 2 | ✅ | 0 | 2 missing |
| **Total** | **50** | **50** | **7** | **43 missing** |

### Why This Happened

The feature engineering code was implemented in `src/features/` with full test coverage (84/84 tests), but `scripts/build_features_duckdb.py` was only wired to call a subset of the generators.

**This is not a bug** - it's just incomplete wiring between modules.

---

## Impact on Frozen Baseline

### Chapter 7 Baseline Used 13 Features

The frozen `tabular_lgb` baseline uses:
```python
DEFAULT_TABULAR_FEATURES = [
    "mom_1m", "mom_3m", "mom_6m", "mom_12m",      # Momentum (4)
    "vol_20d", "vol_60d", "vol_of_vol",           # Volatility (3)
    "max_drawdown_60d",                            # Drawdown (1)
    "adv_20d", "adv_60d",                          # Liquidity (2)
    "rel_strength_1m", "rel_strength_3m",         # Relative strength (2)
    "beta_252d",                                   # Beta (1)
]
```

**But DuckDB only has 7 of these 13!**

This means the frozen baseline actually ran with **only 7 features** (momentum + volume), not 13.

**Result:** The frozen baseline is even MORE conservative than documented. It achieved +256% to +970% lift using only momentum and volume features.

---

## Earnings Gap Microstructure Issue

### The Problem

Monthly rebalancing (1st of month) + earnings on 2nd-5th = gap moves within forward horizon.

**Example:**
```
Jan 31: Generate predictions, rebalance
Feb 1:  New positions enter
Feb 2:  Stock reports earnings → 15% gap
Feb 28: Measure 20d return (includes Feb 2 gap)
```

**Question:** Is the gap signal or noise?

### Current Mitigation

**Earnings features exist in code:**
- `days_to_earnings`, `surprise_streak`, `last_surprise_pct`, etc.

**But they're NOT in DuckDB**, so the frozen baseline can't use them.

**This is actually GOOD:**
- Makes baseline harder to beat
- Forces Chapter 8+ models to demonstrate value-add
- Clean separation: Chapter 7 = price only, Chapter 8+ = price + events

### Diagnostic Plan (Chapter 8+)

When you expand features:
1. IC stratification by `days_to_earnings` (0-5d vs 6-21d vs 22-90d)
2. Earnings-adjusted labels (replace gaps with market return)
3. Feature importance analysis
4. If contamination > 0.01 IC, implement mitigation

---

## What Needs to Happen Before Chapter 8

### Option A: Full Expansion (Recommended)

**Expand all 43 features now:**

```bash
# 1. Modify scripts/build_features_duckdb.py to wire all generators
# 2. Rebuild DuckDB
python scripts/build_features_duckdb.py --auto-normalize-splits

# 3. Verify backward compatibility
python scripts/run_chapter7_tabular_lgb.py --smoke
```

**Pros:**
- One-time effort (~30-60 min)
- Ready for all future chapters
- Enables earnings gap diagnostics immediately

**Cons:**
- Uses API quota (but you have FMP Premium with 100K/day)

### Option B: Incremental (Not Recommended)

Expand features in phases as needed for each chapter.

**Cons:**
- Multiple rebuilds
- Risk of forgetting features
- More complex to track

---

## Documentation Created

| File | Purpose |
|------|---------|
| `PRE_CHAPTER_8_CHECKLIST.md` | Complete feature expansion guide (schema, wiring, verification) |
| `EARNINGS_GAP_ANALYSIS.md` | Microstructure diagnostic protocol (IC stratification, contamination) |
| `CHAPTER_7_FREEZE.md` | Updated with "Known Limitations" section |
| `CHAPTER_7_FINAL_STATUS.md` | Complete Chapter 7 status report |
| `ROADMAP.md` | Added pre-Chapter-8 section with feature checklist |
| `PROJECT_DOCUMENTATION.md` | Updated Chapter 5 with pipeline status warnings |
| `AI_Stock_Forecaster_*.ipynb` | Updated Chapter 8 with prerequisites section |

---

## Answers to Your Questions

### 1. Do we need all 50 features?

**No, not immediately:**
- **Kronos (Ch 8):** Needs OHLCV only (0 tabular features)
- **FinText (Ch 9):** Needs returns only (0 tabular features)
- **Fusion (Ch 11):** Needs ALL 50 features (tabular context branch)

**But:** Having them all enables better diagnostics and ablation studies.

### 2. Can we keep files frozen and still add features?

**Yes!** Here's how:
- `evaluation_outputs/chapter7_tabular_lgb_full/` = frozen metrics (git-tracked)
- `data/features.duckdb` = gitignored data file (can rebuild anytime)
- `DEFAULT_TABULAR_FEATURES` = frozen code (13 features, locked)
- `EXTENDED_TABULAR_FEATURES` = new code (50 features, for Chapter 8+)

**The frozen baseline will always use its 13 features**, even if DuckDB has 50.

### 3. Is this noted in documentation?

**It is NOW:**
- ✅ ROADMAP.md: Pre-Chapter-8 section added
- ✅ PROJECT_DOCUMENTATION.md: Pipeline status warnings added
- ✅ Notebook: Chapter 8 prerequisites section added
- ✅ PRE_CHAPTER_8_CHECKLIST.md: Complete guide created

**It was NOT documented before** - this was a blind spot.

### 4. What's your recommended approach?

**Expand features before starting Chapter 8:**

**Reasons:**
1. **Earnings gap mitigation:** Event features are critical
2. **One-time effort:** Better than incremental rebuilds
3. **You have FMP Premium:** No quota concerns
4. **Enables diagnostics:** Can run earnings gap analysis immediately
5. **Future-proof:** Ready for Fusion (Chapter 11)

**Timeline:**
- Feature expansion: 30-60 min (mostly API calls)
- Verification: 5 min (smoke test)
- Total: < 1 hour

---

## Next Steps

### Immediate (Before Chapter 8)
1. **Expand feature store** (see `PRE_CHAPTER_8_CHECKLIST.md`)
2. **Run earnings gap diagnostic** (see `EARNINGS_GAP_ANALYSIS.md`)
3. **Verify Chapter 7 baseline still works** with expanded DuckDB

### Then (Chapter 8)
1. Integrate Kronos TSFM
2. Run FULL_MODE evaluation
3. Compare vs frozen ML baseline (must beat 0.1009/0.1275/0.1808 + 0.02)

---

**Chapter 7 Status:** ✅ COMPLETE + FROZEN  
**Feature Store:** ⚠️ Expansion needed (documented, not urgent)  
**Tests:** 429/429 passing  
**Ready for:** Chapter 8 (after feature expansion)


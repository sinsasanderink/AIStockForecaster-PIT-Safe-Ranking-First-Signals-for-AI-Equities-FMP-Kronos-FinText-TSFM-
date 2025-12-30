# Chapter 7: Baselines to Beat - COMPLETE

**Date:** December 30, 2025  
**Status:** ✅ COMPLETE (Ready for FULL_MODE execution)  
**Tests:** 429/429 passing

---

## Summary

Chapter 7 implements the tabular ML baseline (`tabular_lgb`) and establishes gating policies for future model development. The baseline is fully implemented, tested, and ready to run on REAL data.

---

## Deliverables

### 7.1-7.2: Factor Baselines ✅ (Frozen in Chapter 6)
- `mom_12m`, `momentum_composite`, `short_term_strength`, `naive_random`
- Frozen baseline floor: `evaluation_outputs/chapter6_closure_real/BASELINE_FLOOR.json`

### 7.3: Tabular ML Baseline ✅ COMPLETE
**Implementation:** `src/evaluation/baselines.py`

**Baseline:** `tabular_lgb`
- LightGBM Regressor with time-decay weighting
- Per-fold training using walk-forward splits
- Horizon-specific models (20/60/90d)
- Fixed hyperparameters (no tuning)
- Deterministic (`random_state=42`)

**Tests:** `tests/test_ml_baselines.py` (13 tests)

### 7.4: Baseline Gates ✅ COMPLETE
**Implementation:** `src/evaluation/run_evaluation.py`

**Gates:**
- Factor gate: `median_RankIC(best_factor) ≥ 0.02`
- ML gate: `median_RankIC(tabular_lgb) ≥ 0.05`
- TSFM rule: Must beat tuned ML baseline

**Acceptance Criteria:**
1. RankIC Lift: ≥ baseline + 0.02
2. Net-of-Cost: % positive folds ≥ baseline + 10pp (relative, not absolute 70%)
3. Churn: Top-10 median < 0.30
4. Regime Robustness: No negative fold

### 7.5: FULL_MODE Execution Script ✅ COMPLETE
**Implementation:** `scripts/run_chapter7_tabular_lgb.py`

**Features:**
- Loads REAL data from DuckDB
- Runs monthly + quarterly cadences
- Compares vs frozen Chapter 6 baseline floor
- Produces `BASELINE_REFERENCE.md`
- Does NOT modify Chapter 6 artifacts

**Tests:** `tests/test_chapter7_script.py` (3 tests)

---

## Test Results

```bash
$ pytest tests/ -q
429 passed, 99 warnings in 113.31s (0:01:53)
```

**Breakdown:**
- 413 existing tests (Chapter 1-6)
- 13 ML baseline tests (Chapter 7.3)
- 3 execution script tests (Chapter 7.5)

---

## How to Run

### Quick Smoke Test (Fast)
```bash
python scripts/run_chapter7_tabular_lgb.py --smoke
```

### Full Evaluation (Slow - trains models per fold)
```bash
# Ensure DuckDB exists
python scripts/build_features_duckdb.py --auto-normalize-splits

# Run FULL_MODE
python scripts/run_chapter7_tabular_lgb.py
```

**Expected Runtime:** 30-60 minutes (depends on CPU, number of folds)

**Output:** `evaluation_outputs/chapter7_tabular_lgb_real/`

---

## Key Files

### Implementation
- `src/evaluation/baselines.py`: ML baseline implementation
- `src/evaluation/run_evaluation.py`: Acceptance criteria
- `scripts/run_chapter7_tabular_lgb.py`: Execution script

### Tests
- `tests/test_ml_baselines.py`: ML baseline unit/integration tests
- `tests/test_chapter7_script.py`: Script smoke tests
- `tests/test_baselines.py`: Updated for 5 baselines

### Documentation
- `CHAPTER_7_IMPLEMENTATION_SUMMARY.md`: Implementation details
- `CHAPTER_7_COMPLETE.md`: This file
- `PROJECT_DOCUMENTATION.md`: Updated Chapter 7 section
- `ROADMAP.md`: Updated Chapter 7 status

---

## Design Decisions

### 1. LGBMRegressor (not Ranker)
- **Reason:** Labels are continuous excess returns, not integer relevance grades
- **Benefit:** Direct optimization of prediction error, natural ranking via predicted return

### 2. Time-Decay Weighting
- **Reason:** Financial data is non-stationary
- **Implementation:** Exponential decay with half-life = 252 trading days (1 year)
- **Effect:** Recent samples weighted higher, reduces impact of stale patterns

### 3. Fixed Hyperparameters
- **Reason:** This is a baseline, not a tuned model
- **Benefit:** Prevents "baseline shopping" (overfitting to validation)
- **Values:** Conservative choices (depth=5, lr=0.05) to avoid overfitting

### 4. No CLI for run_evaluation
- **Decision:** Keep CLI stub, document script as proper way
- **Reason:** Script pattern (like Chapter 6) is clearer and more maintainable
- **Benefit:** Explicit data loading, better error messages, easier to extend

---

## Documentation Fixes

### Net-Positive Folds Inconsistency (Fixed)
- **Before:** Mixed references to "≥ 70%" and "baseline + 10pp"
- **After:** Consistent relative gate: "% positive ≥ baseline + 10pp"
- **Rationale:** Frozen baselines show 5.8%-40.1% positive folds; 70% absolute would be unrealistic

**Files Updated:**
- `src/evaluation/run_evaluation.py`: Implementation uses relative threshold
- All documentation: Consistent "baseline + 10pp" language
- Chapter 6 freeze docs: Clarified relative vs absolute

---

## Acceptance Criteria Status

| Criterion | Target | Status |
|-----------|--------|--------|
| RankIC Lift | ≥ baseline + 0.02 | ⏳ Pending FULL_MODE run |
| Beats Baselines | 2 of 3 horizons | ⏳ Pending FULL_MODE run |
| Cost Survival | ≥ baseline + 10pp | ✅ Implemented (relative gate) |
| Churn | < 0.30 | ✅ Implemented |

**Note:** Acceptance can only be verified after running FULL_MODE on REAL data.

---

## What's Next

### Immediate (Chapter 7 Closure)
1. Run `python scripts/run_chapter7_tabular_lgb.py` (FULL_MODE)
2. Review `BASELINE_REFERENCE.md` output
3. Decide if `tabular_lgb` becomes new baseline floor
4. Update frozen artifacts if needed

### Future (Chapter 8+)
- **Chapter 8:** Kronos integration (TSFM for price dynamics)
- **Chapter 9:** FinText-TSFM integration (return structure prediction)
- **Chapter 10:** Context features (fundamentals, macro)
- **Chapter 11:** Fusion model (combine all signals)

**All future models must beat `tabular_lgb` baseline (ML gate: RankIC ≥ 0.05)**

---

## Files Changed

### New Files
- `scripts/run_chapter7_tabular_lgb.py` (423 lines)
- `tests/test_chapter7_script.py` (3 tests)
- `CHAPTER_7_IMPLEMENTATION_SUMMARY.md` (258 lines)
- `CHAPTER_7_COMPLETE.md` (this file)

### Modified Files
- `src/evaluation/baselines.py`: Added ML baseline implementation
- `src/evaluation/run_evaluation.py`: Added train_df extraction, relative gate
- `src/evaluation/__init__.py`: Added ML baseline exports
- `tests/test_baselines.py`: Updated for 5 baselines
- `tests/test_ml_baselines.py`: 13 new tests
- `tests/test_section4_gates.py`: Added assert statements
- `PROJECT_DOCUMENTATION.md`: Added Chapter 7 section
- `ROADMAP.md`: Updated Chapter 7 status

---

**Chapter 7 Implementation Complete:** December 30, 2025  
**Ready for FULL_MODE execution and baseline reference generation**


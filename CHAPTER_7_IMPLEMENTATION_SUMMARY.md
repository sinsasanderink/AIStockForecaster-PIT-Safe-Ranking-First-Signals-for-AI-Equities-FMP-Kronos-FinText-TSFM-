# Chapter 7.3 Implementation Summary

**Date:** December 30, 2025  
**Status:** ✅ COMPLETE  
**Tests:** 426/426 passing (413 existing + 13 new)

---

## What Was Implemented

### 1. Tabular ML Baseline (`tabular_lgb`)

**File:** `src/evaluation/baselines.py`

**Key Components:**
- `BASELINE_TABULAR_LGB`: Baseline definition with required features
- `train_lgbm_ranking_model()`: Trains LightGBM Regressor with time-decay weighting
- `predict_lgbm_scores()`: Generates predictions from trained model
- `generate_ml_baseline_scores()`: End-to-end train + score pipeline
- `_compute_time_decay_weights()`: Exponential time decay (recent samples weighted higher)

**Model Details:**
- **Type:** LightGBM Regressor (predicts excess return; higher = better)
- **Objective:** Regression (not ranking, because labels are continuous)
- **Training:** Per-fold using walk-forward splits
- **Time Decay:** Exponential weighting with half-life = 252 trading days
- **Features:** Momentum (1m/3m/6m/12m), volatility (20d/60d), drawdown, ADV, relative strength, beta
- **Horizon-Specific:** Separate models for 20/60/90d horizons
- **Deterministic:** Fixed `random_state=42`

**Fixed Hyperparameters:**
```python
n_estimators=100
learning_rate=0.05
max_depth=5
num_leaves=31
min_child_samples=20
```

### 2. Integration with Evaluation Pipeline

**File:** `src/evaluation/run_evaluation.py`

**Changes:**
- Added `fold_train_features` extraction for ML baselines
- Modified `generate_baseline_scores()` call to pass `train_df` parameter
- ML baselines now train on fold's training data and score on validation data

**File:** `src/evaluation/__init__.py`

**Changes:**
- Added `BASELINE_TABULAR_LGB` to exports
- Added `ML_BASELINES` list to exports
- Added `generate_ml_baseline_scores` to exports

### 3. Tests

**File:** `tests/test_ml_baselines.py` (NEW, 13 tests)

**Test Coverage:**
1. **Registration:** `tabular_lgb` is registered and listed
2. **Definition:** Has correct name, description, and required features
3. **Time Decay:** Recent dates get higher weights, deterministic
4. **Training:** Model can be trained successfully
5. **Prediction:** Trained model generates valid predictions
6. **Determinism:** Same inputs → same outputs (fixed random_state)
7. **Integration:** Produces valid EvaluationRow format
8. **No Duplicates:** No duplicate (as_of_date, stable_id, horizon) keys
9. **End-to-End Determinism:** Full pipeline is deterministic
10. **Train/Val Split:** Train and val dates are non-overlapping
11. **Guardrails:** Raises error if train_df is missing
12. **Smoke Test:** Works with baseline runner interface

**File:** `tests/test_baselines.py` (UPDATED)

**Changes:**
- Updated registry size check: 4 → 5 baselines
- Updated `test_all_baselines_deterministic` to skip ML baselines (require train_df)
- Updated `test_run_all_baselines` to only run factor + sanity baselines
- Updated `test_list_baselines` to expect 5 baselines

**File:** `tests/test_section4_gates.py` (TIDIED)

**Changes:**
- Added `assert` statements to `test_sec_filing_boundaries`, `test_corporate_action_integrity`, and `test_universe_reproducibility`

### 4. Documentation

**Files Updated:**
- `PROJECT_DOCUMENTATION.md`: Added Chapter 7 section with implementation details
- `ROADMAP.md`: Updated Chapter 7 status to "IN PROGRESS (7.3 COMPLETE)"
- `src/evaluation/baselines.py`: Updated module docstring to include ML baselines

---

## Test Results

```bash
$ pytest tests/ -q
426 passed, 99 warnings in 147.66s (0:02:27)
```

**Breakdown:**
- 413 existing tests (all passing)
- 13 new ML baseline tests (all passing)

---

## How to Use

### 1. Run `tabular_lgb` Baseline (SMOKE MODE)

```python
from src.evaluation import ExperimentSpec, run_experiment, SMOKE_MODE
from src.evaluation.data_loader import load_features_for_evaluation
from pathlib import Path

# Load features from DuckDB
features_df = load_features_for_evaluation(
    mode="duckdb",
    db_path="data/features.duckdb"
)

# Create experiment spec
spec = ExperimentSpec.baseline("tabular_lgb", cadence="monthly")

# Run in SMOKE MODE (1 fold for quick testing)
results = run_experiment(
    spec,
    features_df,
    Path("evaluation_outputs/test_tabular_lgb"),
    SMOKE_MODE
)
```

### 2. Run `tabular_lgb` Baseline (FULL MODE)

```bash
# Ensure DuckDB feature store exists
python scripts/build_features_duckdb.py --auto-normalize-splits

# Run tabular_lgb baseline (FULL MODE)
# Note: This will take significant time (trains model per fold per horizon)
python -m src.evaluation.run_evaluation \
  --baseline tabular_lgb \
  --cadence monthly \
  --mode full \
  --output-dir evaluation_outputs/chapter7_tabular_lgb
```

### 3. Compare vs Frozen Baseline Floor

```python
import json
from pathlib import Path

# Load frozen baseline floor
floor_path = Path("evaluation_outputs/chapter6_closure_real/BASELINE_FLOOR.json")
with floor_path.open() as f:
    baseline_floor = json.load(f)

# Load tabular_lgb results
results_path = Path("evaluation_outputs/chapter7_tabular_lgb/fold_summaries.csv")
import pandas as pd
results = pd.read_csv(results_path)

# Compare median RankIC per horizon
for horizon in [20, 60, 90]:
    baseline_rankic = baseline_floor["best_baseline_per_horizon"][str(horizon)]["median_rankic"]
    ml_rankic = results[results["horizon"] == horizon]["rankic_median"].median()
    
    print(f"Horizon {horizon}d:")
    print(f"  Baseline (frozen): {baseline_rankic:.4f}")
    print(f"  tabular_lgb: {ml_rankic:.4f}")
    print(f"  Lift: {ml_rankic - baseline_rankic:+.4f}")
    print()
```

---

## Next Steps (Chapter 7.5)

1. **Run FULL_MODE with `tabular_lgb`** on REAL DuckDB data
2. **Compare performance** vs frozen factor baselines
3. **Update `BASELINE_FLOOR.json`** if `tabular_lgb` beats momentum baselines
4. **Re-freeze baseline reference** if ML baseline becomes new floor
5. **Document ML baseline performance** in `BASELINE_REFERENCE.md`

---

## Key Design Decisions

### Why LGBMRegressor instead of LGBMRanker?

LightGBM's `LGBMRanker` requires **integer labels** (relevance grades like 0, 1, 2, 3).  
Our labels are **continuous excess returns** (e.g., 0.037, -0.012, 0.089).

Using `LGBMRegressor`:
- Predicts continuous excess return
- Higher predicted return = higher score (natural ranking)
- Allows direct optimization of prediction error
- Still produces valid ranking scores for cross-sectional comparison

### Why Time-Decay Weighting?

Financial data has **non-stationarity**: market dynamics change over time.

Time-decay weighting:
- Gives higher weight to recent observations
- Reduces impact of stale patterns from distant past
- Half-life of 252 trading days (1 year) balances recency vs sample size
- Implemented as exponential decay: `weight = 2^(-days_until_end / half_life)`

### Why Fixed Hyperparameters?

This is a **baseline**, not a tuned model.

Fixed hyperparameters:
- Prevent "baseline shopping" (overfitting to validation)
- Provide stable reference point for future models
- Conservative choices (depth=5, lr=0.05) avoid overfitting
- Future models (Chapter 8+) can tune hyperparameters and must beat this baseline

---

## Files Changed

### New Files
- `tests/test_ml_baselines.py` (13 tests)
- `CHAPTER_7_IMPLEMENTATION_SUMMARY.md` (this file)

### Modified Files
- `src/evaluation/baselines.py` (added ML baseline implementation)
- `src/evaluation/run_evaluation.py` (added train_df extraction and passing)
- `src/evaluation/__init__.py` (added ML baseline exports)
- `tests/test_baselines.py` (updated for 5 baselines)
- `tests/test_section4_gates.py` (added assert statements)
- `PROJECT_DOCUMENTATION.md` (added Chapter 7 section)
- `ROADMAP.md` (updated Chapter 7 status)

---

## Dependencies

**Required:**
- `lightgbm>=4.0.0` (already in `requirements/ml.txt`)

**Verified:**
- All 426 tests passing
- No new dependencies added
- Compatible with existing evaluation infrastructure

---

**Implementation Complete:** December 30, 2025  
**Ready for:** FULL_MODE evaluation run


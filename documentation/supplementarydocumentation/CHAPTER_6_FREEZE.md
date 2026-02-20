# Chapter 6: CLOSED & FROZEN ✅

**Status:** FROZEN  
**Freeze Date:** December 30, 2025  
**Commits:**
- Code freeze: `18bad8a` - Chapter 6: Closure fixes + freeze REAL baseline reference
- Artifacts freeze: `7e6fa3a` - Chapter 6: Freeze REAL baseline reference artifacts

---

## What Was Frozen

This freeze establishes the **immovable reference point** for all Chapter 7+ model comparisons. No future changes to Chapter 6 evaluation infrastructure are permitted without incrementing to a new evaluation version (and re-freezing all baselines).

### Frozen Evaluation Infrastructure

**Location:** `src/evaluation/`

| Module | Purpose | Tests |
|--------|---------|-------|
| `definitions.py` | Canonical time conventions (horizons/embargo = TRADING DAYS, UTC maturity, require_all_horizons) | ✅ |
| `walk_forward.py` | Expanding window splitter with enforced purging/embargo/maturity | ✅ |
| `sanity_checks.py` | IC parity + experiment naming validation | ✅ |
| `metrics.py` | EvaluationRow contract + RankIC/quintile/topK/churn/regime slicing | ✅ |
| `costs.py` | Diagnostic cost overlay (base + ADV-scaled slippage) + sensitivity bands | ✅ |
| `reports.py` | Stability reports (IC decay, regime tables, churn diagnostics, scorecard) | ✅ |
| `baselines.py` | 4 baselines (mom_12m, momentum_composite, short_term_strength, naive_random) | ✅ |
| `run_evaluation.py` | End-to-end orchestrator (SMOKE/FULL modes) | ✅ |
| `qlib_adapter.py` | Qlib shadow evaluator (IC parity verification) | ✅ |
| `data_loader.py` | Deterministic data loading (synthetic + DuckDB) | ✅ |

**Total Tests:** 413/413 passing

### Frozen Baseline Reference Artifacts

**Location:** `evaluation_outputs/chapter6_closure_real/` (tracked in git via `.gitignore` exception)

```
evaluation_outputs/chapter6_closure_real/
├── BASELINE_FLOOR.json          # Median RankIC per horizon/baseline (THE FLOOR TO BEAT)
├── BASELINE_REFERENCE.md        # Human-readable summary + usage instructions
├── CLOSURE_MANIFEST.json        # Commit hash, pip freeze, data_hash, run config
├── DATA_MANIFEST.json           # Data snapshot identity (192,307 rows, 2016-2025)
│
├── baseline_mom_12m_monthly/              # Full stability reports + figures
├── baseline_mom_12m_quarterly/
├── baseline_momentum_composite_monthly/
├── baseline_momentum_composite_quarterly/
├── baseline_naive_random_monthly/
├── baseline_naive_random_quarterly/
├── baseline_short_term_strength_monthly/
├── baseline_short_term_strength_quarterly/
│
└── qlib_shadow/                           # Qlib parity verification
    └── closure_verification/
        └── qlib/QLIB_SUMMARY.md
```

**Git Tracking:**
- `evaluation_outputs/*` is generally ignored (large regenerable files)
- **Exception:** `evaluation_outputs/chapter6_closure_real/` is explicitly tracked via:
  ```gitignore
  evaluation_outputs/*
  !evaluation_outputs/chapter6_closure_real/
  ```

---

## Baseline Floor Summary (The Numbers to Beat)

### Best Baseline per Horizon (Monthly Primary)

| Horizon | Best Baseline | Median RankIC | Quintile Spread | Hit Rate @10 | Avg ER @10 | N Folds |
|---------|---------------|---------------|-----------------|--------------|------------|---------|
| **20d** | `mom_12m_monthly` | **0.0283** | 0.0035 | 0.50 | 0.0125 | 109 |
| **60d** | `momentum_composite_monthly` | **0.0392** | 0.0370 | 0.60 | 0.0533 | 109 |
| **90d** | `momentum_composite_monthly` | **0.0169** | 0.0374 | 0.60 | 0.0882 | 109 |

### Quarterly Cadence (Robustness Lens)

| Horizon | Best Baseline | Median RankIC | N Folds |
|---------|---------------|---------------|---------|
| **20d** | `mom_12m_quarterly` | 0.0329 | 36 |
| **60d** | `momentum_composite_quarterly` | 0.0566 | 36 |
| **90d** | `momentum_composite_quarterly` | 0.0459 | 36 |

### Churn Diagnostics (Top-10)

| Horizon | Median Churn | P90 Churn | N Observations |
|---------|--------------|-----------|----------------|
| **20d** | 0.10 | 0.20 | 13,365 |
| **60d** | 0.10 | 0.20 | 13,365 |
| **90d** | 0.10 | 0.20 | 13,365 |

**Interpretation:** Very low churn (10% median turnover per rebalance) indicates high portfolio stability.

### Cost Survival (Base Cost Scenario: 20 bps round-trip)

| Horizon | % Positive Folds | Median Net ER | N Folds |
|---------|------------------|---------------|---------|
| **20d** | 5.8% | -0.092 | 327 |
| **60d** | 25.1% | -0.057 | 327 |
| **90d** | 40.1% | -0.033 | 327 |

**Interpretation:** Even simple momentum baselines struggle at the 20d horizon after costs. Longer horizons (60d/90d) show more cost resilience.

### Sanity Check: naive_random Baseline

| Cadence | Median RankIC (20d) | Median RankIC (60d) | Median RankIC (90d) |
|---------|---------------------|---------------------|---------------------|
| Monthly | 0.0019 | 0.0003 | -0.0031 |
| Quarterly | 0.0015 | 0.0002 | -0.0011 |

**Status:** ✅ PASSED (all RankIC values near zero, confirming no systematic bias in evaluation pipeline)

---

## Critical Bug Fixes Included in Freeze

These bugs were discovered during closure and fixed before freeze:

### 1. Per-Horizon Metrics Bug (HIGH SEVERITY)
**Root Cause:** `run_experiment()` used generic `excess_return` column (which defaults to 20d data) for ALL horizons (20/60/90).

**Impact:** 
- Metrics for 60d and 90d horizons were computed using 20d labels
- This caused identical or suspiciously similar RankIC across horizons
- Cost/churn calculations were also affected

**Fix:** Modified `src/evaluation/run_evaluation.py` to use horizon-specific columns:
```python
excess_return_col = f"excess_return_{horizon}d"  # e.g., excess_return_60d for horizon=60
```

**Verification:** `BASELINE_FLOOR.json` now shows distinct RankIC per horizon (0.0283/0.0392/0.0169 for 20/60/90d).

### 2. Baseline Floor Path Bug (MEDIUM SEVERITY)
**Root Cause:** `compute_baseline_floor()` used incorrect directory nesting patterns when loading outputs, causing `churn_medians` and `cost_survival` to be empty.

**Impact:**
- Churn diagnostics missing from baseline floor
- Cost survival metrics missing from baseline floor
- "Best baseline" selection sometimes prioritized quarterly over monthly (against spec)

**Fix:** Modified `scripts/run_chapter6_closure.py` to:
- Correctly iterate through `results["all_churn_series"]` and `results["all_cost_overlays"]`
- Populate `churn_medians` and `cost_survival` by aggregating across all folds
- Prioritize monthly cadences when selecting "best baseline"

**Verification:** `BASELINE_FLOOR.json` now has populated `churn_medians` and `cost_survival` sections.

### 3. Fold Date Alignment Bug (MEDIUM SEVERITY)
**Root Cause:** Rebalance date generation used calendar-based dates that sometimes fell on weekends/holidays, causing "0 eval rows" for many folds.

**Impact:** Many folds had no data, reducing effective sample size.

**Fix:** Modified `_generate_folds_from_features()` in `src/evaluation/run_evaluation.py` to snap rebalance dates to the nearest previous trading day present in `features_df['date']`.

**Verification:** Closure run now reports 109 monthly folds and 36 quarterly folds with non-zero eval rows.

### 4. Qlib Shadow Eval Input Bug (LOW SEVERITY)
**Root Cause:** `run_qlib_shadow_closure()` was passing raw `features_df` to Qlib instead of scored `eval_rows`.

**Impact:** Qlib shadow evaluator failed with "Missing required columns: ['as_of_date', 'score']".

**Fix:** 
- Modified `scripts/run_chapter6_closure.py` to collect and pass actual `eval_rows_df` to Qlib
- Enhanced `to_qlib_format()` in `src/evaluation/qlib_adapter.py` to handle column name aliases
- Added debug logging for column verification

**Verification:** Closure run now completes Qlib shadow evaluation and writes `qlib_shadow/closure_verification/qlib/QLIB_SUMMARY.md`.

**Note on Qlib Logs:** During Qlib initialization, you may see log lines mentioning default `cn_data` paths. This is normal Qlib initialization behavior and can be ignored—we feed Qlib our own converted DataFrame via `to_qlib_format()`, so it doesn't actually use the default Chinese market data.

---

## Data Snapshot

**Source:** DuckDB (`data/features.duckdb`)  
**Build Script:** `scripts/build_features_duckdb.py`  
**FMP API:** Premium (split-adjusted OHLCV + dividends)

| Attribute | Value |
|-----------|-------|
| DuckDB Features Table | 201,307 rows (full build, 2014-2025 with start buffer) |
| Evaluation Window | 192,307 rows (2016-01-04 → 2025-02-19) |
| Tickers | ~100 (AI universe) |
| Date Range | 2016-01-04 → 2025-02-19 (evaluation range) |
| Horizons | 20d, 60d, 90d (trading days) |
| Label Version | v2 (total return with dividends) |
| Data Hash | `5723d4c88b8ecba1...` (computed on evaluation window) |

**Note:** DuckDB build includes 2014-01-01 start buffer for feature lookbacks (e.g., `mom_12m` requires 12 months of history before first evaluation date).

**Features Available:**
- **Momentum:** `mom_1m`, `mom_3m`, `mom_6m`, `mom_12m`
- **ADV (Costs):** `adv_20d`
- **Regime:** `market_return_20d`, `market_vol_20d`, `vix_percentile_252d`

**Quality Checks:**
- ✅ Split discontinuities validated and normalized (`--auto-normalize-splits`)
- ✅ Duplicate daily bars removed (one row per ticker/date)
- ✅ Labels pivoted to wide format (one row per date/ticker)
- ✅ Point-in-time discipline enforced (no future data leakage)

---

## How to Use This Reference in Chapter 7+

### 1. Loading the Frozen Baseline Floor

```python
import json
from pathlib import Path

# Load frozen baseline floor
floor_path = Path("evaluation_outputs/chapter6_closure_real/BASELINE_FLOOR.json")
with floor_path.open() as f:
    baseline_floor = json.load(f)

# Get the RankIC to beat for a specific horizon
horizon = 60
best_baseline = baseline_floor["best_baseline_per_horizon"][str(horizon)]
rankic_to_beat = best_baseline["median_rankic"]
print(f"Your model must beat {best_baseline['baseline']} RankIC: {rankic_to_beat:.4f}")
# Output: Your model must beat momentum_composite_monthly RankIC: 0.0392
```

### 2. Running Your Model Through the Same Pipeline

```python
from src.evaluation import (
    load_features_for_evaluation,
    WalkForwardSplitter,
    run_experiment,
    ExperimentSpec,
    RunMode,
)

# Load REAL data (same DuckDB used for baseline freeze)
features_df = load_features_for_evaluation(
    mode="duckdb",
    db_path="data/features.duckdb",
    eval_start=date(2016, 1, 1),
    eval_end=date(2025, 6, 30),
    horizons=[20, 60, 90],
    cadence="monthly",
)

# Define your model experiment
model_spec = ExperimentSpec(
    name="tabular_lgb_v1",
    version="v1.0",
    model_type="tabular",
    horizons=[20, 60, 90],
    description="LightGBM baseline with time-decay weighting",
)

# Run through IDENTICAL pipeline as baselines
results = run_experiment(
    experiment_spec=model_spec,
    features_df=features_df,
    output_dir="evaluation_outputs/models/",
    mode=RunMode.FULL,
    cost_scenarios=["base", "low", "high"],
)
```

### 3. Comparing Against Baseline Floor

```python
from src.evaluation import compute_acceptance_verdict

# Load frozen baseline summaries
baseline_summaries = {
    horizon: baseline_floor["horizons"][str(horizon)]
    for horizon in [20, 60, 90]
}

# Compare your model
verdict = compute_acceptance_verdict(
    model_results=results["fold_summaries"],
    baseline_floor=baseline_summaries,
    cost_overlays=results["cost_overlays"],
    churn_series=results["churn_series"],
)

# Check if model passes Chapter 7 gates
print("Acceptance Criteria:")
for horizon in [20, 60, 90]:
    passed = verdict[horizon]["rankic_lift_vs_baseline"] >= 0.02
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"  {horizon}d: {status} (lift={verdict[horizon]['rankic_lift_vs_baseline']:.4f})")
```

---

## Chapter 7 Acceptance Criteria

Models evaluated in Chapter 7+ must meet these criteria relative to the frozen baseline floor:

### Primary Gates (Must Pass)

| Criterion | Threshold | Rationale |
|-----------|-----------|-----------|
| **RankIC Lift** | ≥ +0.02 vs best baseline | Meaningful improvement over momentum |
| **Net-of-Cost Positive** | ≥ baseline + 10pp for ≥2 horizons | Relative improvement over frozen floor (5.8%/25.1%/40.1%) |
| **Churn** | Median < 0.30 (30%) | Tradable turnover |
| **Regime Robustness** | No catastrophic collapse | Graceful degradation across VIX/bull/bear |

**Note:** Cost survival gate is relative to frozen baseline floor, not absolute 70%, because frozen baselines show 5.8%-40.1% positive folds under base cost scenario.

### Secondary Gates (Nice-to-Have)

| Criterion | Threshold | Rationale |
|-----------|-----------|-----------|
| **Qlib IC Parity** | Within ±0.001 of our RankIC | Independent verification |
| **Hit Rate @10** | > 0.55 | More than half of Top-10 are winners |
| **Quintile Spread** | > best baseline | Stronger signal separation |

---

## Reproducibility Guarantee

The frozen reference is deterministic and reproducible **given the same data hash, commit, and environment**:

### Commit Hash
```
18bad8a - Chapter 6: Closure fixes + freeze REAL baseline reference
7e6fa3a - Chapter 6: Freeze REAL baseline reference artifacts
```

### Environment
```bash
# Python version
python --version  # Python 3.11.5

# Dependencies (frozen)
pip install -r requirements.txt  # See CLOSURE_MANIFEST.json for full pip freeze

# Key packages
# - pandas==2.1.4
# - numpy==1.26.2
# - scipy==1.11.4
# - duckdb==0.9.2
# - pyqlib==0.9.5
```

### Data Hash
```
5723d4c88b8ecba1...  # Computed via data_loader.compute_data_hash()
```

### Re-running the Closure

```bash
# Rebuild DuckDB feature store (requires FMP_KEYS in .env)
python scripts/build_features_duckdb.py \
  --start-date 2014-01-01 \
  --end-date 2025-06-30 \
  --auto-normalize-splits

# Re-run Chapter 6 closure (should produce identical BASELINE_FLOOR.json)
python scripts/run_chapter6_closure.py
```

**Expected:** Identical `data_hash` and `BASELINE_FLOOR.json` with metrics matching to floating-point precision (±1e-10). **Note:** Different BLAS libraries or OS platforms may produce slightly different floating-point rounding in the final decimals, but results should be identical to at least 6 decimal places.

---

## What's Next: Chapter 7

With Chapter 6 frozen, Chapter 7 can now proceed safely:

### Remaining Baselines

| Baseline | Purpose | Status |
|----------|---------|--------|
| `mom_12m` | Naive baseline | ✅ FROZEN |
| `momentum_composite` | Stronger but transparent | ✅ FROZEN |
| `short_term_strength` | Diagnostic | ✅ FROZEN |
| `naive_random` | Sanity check | ✅ FROZEN |
| `tabular_lgb` | Tuned ML baseline | 🔧 TODO (Chapter 7) |

### Chapter 7 Deliverables

1. **Implement `tabular_lgb` Baseline**
   - LightGBM with time-decay sample weighting
   - Per-fold training using walk-forward splits
   - Horizon-specific models (separate for 20/60/90d)
   - One-time deterministic hyperparameter tuning (no "baseline shopping")
   - Re-run FULL_MODE and freeze as new baseline floor

2. **Formalize Gating Policy**
   - Factor gate: `median_RankIC(best_factor) ≥ 0.02`
   - ML gate: `median_RankIC(tabular_lgb) ≥ 0.05`
   - TSFM rule (later): Must beat tuned ML baseline

3. **Update Baseline Floor**
   - If `tabular_lgb` beats momentum baselines, it becomes the new floor
   - Re-freeze `BASELINE_FLOOR.json` with ML baseline included
   - All future models (Chapters 8-12) must beat this ML floor

---

## Important Constraints

### ⚠️ DO NOT Modify

The following are **FROZEN** and must not be changed without incrementing evaluation version:

1. **Evaluation definitions** (`src/evaluation/definitions.py`)
   - Time conventions (horizons/embargo in TRADING DAYS)
   - Maturity rules (UTC market close)
   - Eligibility rules (require_all_horizons)

2. **Evaluation pipeline** (`src/evaluation/*.py`)
   - Walk-forward splitting logic
   - Purging/embargo/maturity enforcement
   - EvaluationRow contract
   - Metrics computation
   - Cost model

3. **Baseline implementations** (`src/evaluation/baselines.py`)
   - mom_12m, momentum_composite, short_term_strength, naive_random

4. **Frozen artifacts** (`evaluation_outputs/chapter6_closure_real/`)
   - Never regenerate or modify these files
   - They are the immutable comparison anchor

### ✅ Can Be Modified

The following can be changed for Chapter 7+ without breaking the freeze:

1. **New baselines** (e.g., `tabular_lgb`)
   - As long as they run through the frozen pipeline
   - Must emit `EvaluationRow` format

2. **New models** (Chapters 8-12)
   - Kronos, FinText, Fusion, Ensemble
   - All must use the frozen evaluation pipeline

3. **Data updates**
   - Can extend date range forward (e.g., 2025-07-01 → 2026-12-31)
   - Must maintain schema compatibility
   - Must recompute `data_hash`

4. **Documentation**
   - Can update project docs, notebooks, READMEs
   - Cannot change frozen artifacts

---

## Verification Checklist

Before Chapter 7 proceeds, verify:

- [x] `evaluation_outputs/chapter6_closure_real/` exists and is tracked in git
- [x] `BASELINE_FLOOR.json` has distinct RankIC per horizon (20/60/90d)
- [x] `churn_medians` and `cost_survival` are populated (not empty)
- [x] `naive_random` RankIC ≈ 0 (sanity check passed)
- [x] Qlib shadow evaluation completed without errors
- [x] All 413 tests passing
- [x] Commits frozen: `18bad8a` + `7e6fa3a`
- [x] `.gitignore` updated to track closure artifacts

**Status:** ✅ ALL VERIFIED

---

**Frozen at:** 2025-12-30T05:45:21Z  
**Git Tag:** (optional: `git tag -a v6.0-baseline-freeze -m "Chapter 6 baseline reference freeze"`)  
**Next Chapter:** Chapter 7 - Baseline Models (tabular_lgb + gates)


# Chapter 6 Phase 6: Baselines + Qlib Shadow Evaluator — COMPLETE ✅

**Status:** CLOSED & FROZEN (December 30, 2025)  
**Phases:** 0, 1, 1.5, 2, 3, 4, 5, 6 ALL COMPLETE  
**Total Tests:** 413/413 passing  
**Commits:** `18bad8a` (code fixes) + `7e6fa3a` (artifacts freeze)  
**Reference Doc:** `CHAPTER_6_FREEZE.md`

---

## 🔒 FREEZE STATUS (December 30, 2025)

**Chapter 6 is now CLOSED & FROZEN** and serves as the immutable baseline reference for all Chapter 7+ model comparisons.

**Frozen Baseline Floor (REAL DuckDB Data):**

| Horizon | Best Baseline | Median RankIC | N Folds |
|---------|---------------|---------------|---------|
| **20d** | `mom_12m_monthly` | **0.0283** | 109 |
| **60d** | `momentum_composite_monthly` | **0.0392** | 109 |
| **90d** | `momentum_composite_monthly` | **0.0169** | 109 |

**Frozen Artifacts (tracked in git):**
- `evaluation_outputs/chapter6_closure_real/` - Complete baseline reference
- `BASELINE_FLOOR.json` - Metrics to beat
- `BASELINE_REFERENCE.md` - Usage instructions
- `CLOSURE_MANIFEST.json` - Commit hash + data hash

**Sanity Check:** ✅ PASSED (`naive_random` RankIC ≈ 0)

**Next:** Chapter 7 - Baseline Models (implement `tabular_lgb` + gates)

---

## Executive Summary

Phase 6 implemented:
1. **Phase 3 Baselines**: 4 baselines (mom_12m, momentum_composite, short_term_strength, naive_random) running through identical pipeline
2. **End-to-End Runner**: Orchestrates complete evaluation pipeline (walk-forward → scoring → metrics → costs → reports)
3. **Acceptance Criteria**: Verdict layer computing pass/fail per horizon
4. **Qlib Shadow Evaluator**: Adapter layer for "second opinion" IC analysis

All components are deterministic and use the canonical `EvaluationRow` contract.

---

## Implemented Components

### A) Phase 3 Baselines (`src/evaluation/baselines.py`)

**Baselines (exactly 3, locked):**

| Baseline | Description | Formula |
|----------|-------------|---------|
| `mom_12m` | Primary naive baseline | `score = mom_12m` |
| `momentum_composite` | Stronger but transparent | `score = (mom_1m + mom_3m + mom_6m + mom_12m) / 4` |
| `short_term_strength` | Diagnostic baseline | `score = mom_1m` |

**Key Functions:**
```python
from src.evaluation import (
    BASELINE_REGISTRY,
    generate_baseline_scores,
    run_all_baselines,
    list_baselines,
)

# Generate scores for one baseline
eval_rows = generate_baseline_scores(
    features_df=features,
    baseline_name="mom_12m",
    fold_id="fold_01",
    horizon=20
)

# Run all baselines at once
results = run_all_baselines(features_df, fold_id="fold_01", horizon=20)
```

**Tests:** 29 tests covering:
- ✅ Frozen definitions (immutable)
- ✅ Score computation correctness
- ✅ Higher score = better (monotonicity)
- ✅ No duplicates per (as_of_date, stable_id, horizon)
- ✅ Determinism (shuffle → identical output)

---

### B) End-to-End Runner (`src/evaluation/run_evaluation.py`)

**Modes:**
- `SMOKE_MODE`: 2024 only, max 3 folds (CI-friendly)
- `FULL_MODE`: 2016-2025, all folds

**Main Entry Point:**
```python
from src.evaluation import run_experiment, ExperimentSpec, SMOKE_MODE

# Create experiment spec
spec = ExperimentSpec.baseline("mom_12m", cadence="monthly")

# Run evaluation
results = run_experiment(
    experiment_spec=spec,
    features_df=features,
    output_dir=Path("evaluation_outputs"),
    mode=SMOKE_MODE
)

# Results include:
# - per_date_metrics: DataFrame
# - fold_summaries: DataFrame
# - cost_overlays: DataFrame
# - churn_series: DataFrame
# - output_paths: Dict[str, Path]
```

**Output Structure:**
```
experiment_name/
├── eval_rows.parquet           # All evaluation rows
├── per_date_metrics.csv        # RankIC, spread, etc. per date
├── fold_summaries.csv          # Aggregated metrics per fold
├── cost_overlays.csv           # Low/base/high cost scenarios
├── churn_series.csv            # Churn per date (if available)
├── experiment_metadata.json    # Run configuration
└── <experiment_name>/          # Stability reports
    ├── tables/
    ├── figures/
    └── REPORT_SUMMARY.md
```

**Tests:** 16 tests covering:
- ✅ Output directory creation
- ✅ All required files created
- ✅ Cost scenarios present and named consistently
- ✅ Determinism across runs
- ✅ Multiple baselines can run

---

### C) Acceptance Criteria (`compute_acceptance_verdict`)

**Criteria (per horizon):**
| Criterion | Threshold | Description |
|-----------|-----------|-------------|
| RankIC Lift | >= 0.02 | vs best baseline |
| Net-Positive Folds | >= 70% | base cost scenario |
| Top-10 Churn | < 30% | median churn |
| No Collapse | 0 | negative folds |

**Usage:**
```python
from src.evaluation import compute_acceptance_verdict, save_acceptance_summary

verdict = compute_acceptance_verdict(
    model_summary,
    baseline_summaries={"mom_12m": baseline_df},
    cost_overlays=cost_df,
    churn_df=churn_df
)

# Save as markdown + CSV
save_acceptance_summary(verdict, output_dir, "experiment_name")
```

**Output (ACCEPTANCE_SUMMARY.md):**
- Pass/fail per criterion per horizon
- Overall verdict (all must pass)
- Criterion definitions

---

### D) Qlib Shadow Evaluator (`src/evaluation/qlib_adapter.py`)

**Purpose:** Feed Qlib for "second opinion" IC analysis (SHADOW EVALUATOR only).

**Key Functions:**
```python
from src.evaluation import (
    to_qlib_format,
    validate_qlib_frame,
    check_ic_parity,
    run_qlib_shadow_evaluation,
    is_qlib_available,
)

# Convert to Qlib format
qlib_df = to_qlib_format(eval_rows)
# Result: MultiIndex (datetime, instrument) with UTC timezone

# Validate
is_valid, msg = validate_qlib_frame(qlib_df)

# Check IC parity (our implementation vs Qlib)
is_parity, qlib_ic, msg = check_ic_parity(eval_rows, our_ic, tolerance=0.001)

# Run shadow evaluation (if pyqlib installed)
if is_qlib_available():
    results = run_qlib_shadow_evaluation(qlib_df, output_dir, "experiment_name")
```

**Qlib Format:**
- MultiIndex: `(datetime, instrument)`
- datetime: UTC timezone-aware (market close)
- instrument: stable_id (preferred) or ticker
- Columns: `score`, `label`, optional `stable_id` for traceability

**Tests:** 21 tests covering:
- ✅ Format conversion correctness
- ✅ Timezone awareness (UTC)
- ✅ stable_id as instrument
- ✅ No duplicate index entries
- ✅ IC parity (perfect positive/negative/random)
- ✅ Index alignment pitfalls
- ✅ Determinism

---

## Test Summary

| Test File | Tests | Status |
|-----------|-------|--------|
| `test_baselines.py` | 29 | ✅ |
| `test_qlib_parity.py` | 21 | ✅ |
| `test_end_to_end_smoke.py` | 22 | ✅ |
| **Total New** | **72** | **✅** |

**Full Suite:** 345/345 tests passing

---

## Usage Examples

### 1. Run All Baselines (SMOKE mode)

```python
from pathlib import Path
from src.evaluation import (
    ExperimentSpec,
    run_experiment,
    SMOKE_MODE,
    list_baselines,
)

# Create sample features (in practice, load from your data)
features_df = ...  # DataFrame with date, ticker, stable_id, mom_* features, excess_return

# Run each baseline
for baseline_name in list_baselines():
    spec = ExperimentSpec.baseline(baseline_name, cadence="monthly")
    
    results = run_experiment(
        experiment_spec=spec,
        features_df=features_df,
        output_dir=Path("evaluation_outputs"),
        mode=SMOKE_MODE
    )
    
    print(f"{baseline_name}: {results['n_folds']} folds, "
          f"{len(results['per_date_metrics'])} date-metrics")
```

### 2. Compute Acceptance Criteria

```python
from src.evaluation import compute_acceptance_verdict, save_acceptance_summary

# Run model and baselines first...

# Compare model to baselines
verdict = compute_acceptance_verdict(
    model_summary=model_results["fold_summaries"],
    baseline_summaries={
        "mom_12m": baseline_results["mom_12m"]["fold_summaries"],
        "momentum_composite": baseline_results["momentum_composite"]["fold_summaries"],
    },
    cost_overlays=model_results["cost_overlays"],
    churn_df=model_results["churn_series"]
)

# Save verdict
save_acceptance_summary(verdict, Path("reports"), "kronos_v0")

# Check overall pass
if verdict["all_criteria_pass"].all():
    print("✅ ALL ACCEPTANCE CRITERIA PASS")
else:
    print("❌ Some criteria failed")
    print(verdict[~verdict["all_criteria_pass"]])
```

### 3. Qlib Shadow Evaluation

```python
from src.evaluation import (
    to_qlib_format,
    validate_qlib_frame,
    run_qlib_shadow_evaluation,
    is_qlib_available,
)

# Convert to Qlib format
qlib_df = to_qlib_format(eval_rows)

# Validate
is_valid, msg = validate_qlib_frame(qlib_df)
if not is_valid:
    raise ValueError(f"Invalid Qlib frame: {msg}")

# Run shadow evaluation (if Qlib installed)
if is_qlib_available():
    results = run_qlib_shadow_evaluation(
        qlib_df,
        output_dir=Path("reports"),
        experiment_name="kronos_v0"
    )
    
    # Results include IC summary, time series
    print(f"Qlib IC Summary: {results.get('summary', 'N/A')}")
else:
    print("pyqlib not installed, skipping shadow evaluation")
```

---

## CLI Entrypoint

```bash
# SMOKE mode (CI-friendly)
python -m src.evaluation.run_evaluation --mode smoke --baseline mom_12m

# FULL mode (local only, long run)
python -m src.evaluation.run_evaluation --mode full

# Quarterly cadence
python -m src.evaluation.run_evaluation --mode smoke --cadence quarterly
```

---

## Key Design Decisions

### 1. Baselines Run Through Identical Pipeline
- Same universe snapshots (stable_id)
- Same walk-forward folds (purging/embargo/maturity)
- Same EvaluationRow contract
- Same metrics, costs, stability reports

### 2. Qlib is SHADOW EVALUATOR Only
- Our system remains source-of-truth for universe, labels, PIT discipline
- Qlib provides "second opinion" IC analysis
- No Qlib backtest (requires full dataset provider)
- Parity tests ensure IC computation matches

### 3. Deterministic Outputs
- Shuffled inputs → identical outputs (tested)
- Same random seeds for reproducibility
- All aggregations deterministic (median, not random sample)

### 4. Cost Scenarios as Sensitivity Bands
- base_only: 0 bps slippage
- low_slippage: c=5
- base_slippage: c=10
- high_slippage: c=20

---

## Files Created/Modified

**New Files:**
- `src/evaluation/baselines.py` (315 lines)
- `src/evaluation/run_evaluation.py` (590 lines)
- `src/evaluation/qlib_adapter.py` (390 lines)
- `tests/test_baselines.py` (340 lines)
- `tests/test_qlib_parity.py` (480 lines)
- `tests/test_end_to_end_smoke.py` (510 lines)
- `CHAPTER_6_PHASE6_COMPLETE.md` (this file)

**Modified Files:**
- `src/evaluation/__init__.py` (updated exports)

---

## Definition Locks Respected

✅ Horizons and embargo are TRADING DAYS (explicitly named)  
✅ Maturity is UTC market close aware  
✅ No partial horizons at end-of-sample  
✅ Per-row-per-horizon purging remains correct  
✅ Churn never bridges fold boundaries  

---

## Chapter 6 Status

| Phase | Status | Tests |
|-------|--------|-------|
| Phase 0: Sanity Checks | ✅ | 16 |
| Phase 1: Walk-Forward | ✅ | 25 |
| Phase 1.5: Definition Lock | ✅ | 40 |
| Phase 2: Metrics | ✅ | 30 |
| Phase 4: Cost Realism | ✅ | 28 |
| Phase 5: Stability Reports | ✅ | 24 |
| **Phase 6: Baselines + Qlib** | ✅ | **72** |

**Total Evaluation Tests:** 235/235 passing  
**Full Suite:** 345/345 passing

---

## Next Steps

Ready for:
1. **FULL mode execution** (2016-2025) with actual features data
2. **Model integration** (Kronos, FinText) using same pipeline
3. **Acceptance criteria validation** comparing model vs baselines
4. **Qlib shadow validation** if pyqlib is installed

The evaluation framework is complete and production-ready.


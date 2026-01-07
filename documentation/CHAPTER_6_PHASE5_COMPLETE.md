# Chapter 6 Phase 5: Stability Reports — COMPLETE ✅

**Status:** Phase 0, 1, 1.5, 2, 4, 5 COMPLETE  
**Date:** December 29, 2025  
**Total Tests:** 163/163 passing (147 in evaluation suite)

---

## Executive Summary

Phase 5 (Stability Reports) implements a pure consumer layer that renders 6.3 metrics + 6.4 costs into deterministic, human-readable reporting artifacts. NO new APIs, NO new features, NO new modeling—just clean, explainable reporting.

### Core Principle

**Pure consumer of existing outputs:**
- Per-date metrics from `evaluate_fold` (6.3)
- Fold summaries from `evaluate_fold` (6.3)
- Regime-sliced summaries from `evaluate_with_regime_slicing` (6.3)
- Churn series from `compute_churn` (6.3)
- Cost overlays from 6.4

---

## Implementation Summary

### Files Created

**`src/evaluation/reports.py`** (768 lines, 24 tests)

Components:
- `STABILITY_THRESHOLDS`: Frozen dataclass with locked diagnostic thresholds
- `compute_ic_decay_stats()`: Early vs late performance (first third vs last third)
- `plot_ic_decay()`: IC time series with rolling mean (6-period default)
- `format_regime_performance()`: Regime tables with coverage stats + thin slice flags
- `plot_regime_bars()`: Grouped bar charts per regime feature
- `compute_churn_diagnostics()`: Churn summary (median, P90, high churn flags)
- `plot_churn_timeseries()`: Churn over time per fold/horizon
- `plot_churn_distribution()`: Churn histogram with median + threshold lines
- `generate_stability_scorecard()`: One-screen summary table
- `generate_stability_report()`: Main entry point → generates all artifacts

**`tests/test_reports.py`** (545 lines, 24 tests)

Test coverage:
- Frozen thresholds (immutability)
- IC decay statistics (early/late split, flags)
- Regime performance (thin slice detection, coverage)
- Churn diagnostics (flag computation, aggregation)
- Stability scorecard (with/without optional inputs)
- Full report generation (all artifacts created)
- Determinism (shuffled inputs → identical outputs)
- Fold boundary handling (no bridging)
- Regime bucket integrity (totals sum correctly)
- Exclusion logging (insufficient dates)

**Updated Files:**
- `src/evaluation/__init__.py`: Added report function exports
- `PROJECT_DOCUMENTATION.md`: Added Section 6.5 Stability Reports
- `PROJECT_STRUCTURE.md`: Updated Phase 5 status

---

## Locked Reporting Parameters

All parameters are frozen in `STABILITY_THRESHOLDS` dataclass:

### Diagnostic Thresholds
- **Rapid decay**: Late IC drops > 5% vs early IC
- **Noisy signal**: IQR / median > 2.0
- **High churn**: Churn > 50% (instability alarm)

### Coverage Requirements
- **Min dates per bucket**: 10 dates (regime slices)
- **Min names per date**: 10 stocks (cross-section)

### Smoothing
- **Rolling window**: 6 periods (monthly cadence)

---

## Report Contract

### Inputs (Pre-Computed by 6.3 + 6.4)

```python
from src.evaluation import StabilityReportInputs

inputs = StabilityReportInputs(
    # From evaluate_fold (6.3)
    per_date_metrics=pd.DataFrame(...),  # Columns: date, fold_id, horizon, rankic, quintile_spread, n_names
    fold_summaries=pd.DataFrame(...),    # Columns: fold_id, horizon, metric_median, metric_iqr, n_dates
    
    # From evaluate_with_regime_slicing (6.3)
    regime_summaries=pd.DataFrame(...),  # Columns: horizon, regime_feature, bucket, rankic_median, n_dates
    
    # From compute_churn (6.3)
    churn_series=pd.DataFrame(...),      # Columns: date, fold_id, horizon, k, churn, retention
    
    # From 6.4 cost overlay
    cost_overlays=pd.DataFrame(...)      # Columns: fold_id, horizon, scenario, net_avg_er, alpha_survives
)
```

### Outputs (Deterministic Artifacts)

```
experiment_name/
├── tables/
│   ├── ic_decay_stats.csv              # Early vs late stats
│   ├── regime_performance.csv          # RankIC by regime + coverage
│   ├── churn_diagnostics.csv           # Churn summary per fold
│   └── stability_scorecard.csv         # One-screen summary
├── figures/
│   ├── ic_decay.png                    # IC time series + rolling mean
│   ├── regime_bars.png                 # RankIC by regime (grouped bars)
│   ├── churn_timeseries.png            # Churn over time
│   └── churn_distribution.png          # Churn histogram
└── REPORT_SUMMARY.md                   # Human-readable summary
```

---

## Component Details

### A) IC Decay Analysis

**Purpose:** Answer "does this fade over time?"

```python
from src.evaluation import compute_ic_decay_stats, plot_ic_decay

# Compute statistics
stats = compute_ic_decay_stats(per_date_metrics, metric_col="rankic")

# Results per fold/horizon:
# - early_median: First 33% of dates
# - late_median: Last 33% of dates
# - decay: late - early (negative = fading)
# - decay_pct: % change
# - rapid_decay_flag: True if drops > 5%
# - noisy_flag: True if IQR/median > 2.0
# - pct_positive: % of dates with positive IC

# Plot with rolling mean
fig = plot_ic_decay(per_date_metrics, rolling_window=6)
```

**Flags:**
- ⚠️ **RAPID DECAY**: Late IC drops > 5% vs early (signal fading)
- ⚠️ **NOISY**: IQR/median > 2.0 (high relative noise)

### B) Regime-Conditional Performance

**Purpose:** Answer "does it survive regimes?"

```python
from src.evaluation import format_regime_performance, plot_regime_bars

# Format with coverage
formatted = format_regime_performance(regime_summaries)

# Adds:
# - thin_slice_flag: True if n_dates < 10 or n_names_median < 10

# Plot grouped bars
fig = plot_regime_bars(regime_summaries, metric_col="rankic_median")
```

**Coverage table example:**

| Horizon | Regime Feature | Bucket | RankIC Median | N Dates | N Names | Thin Slice? |
|---------|----------------|--------|---------------|---------|---------|-------------|
| 20 | vix_percentile_252d | low | 0.025 | 40 | 85 | ✓ OK |
| 20 | vix_percentile_252d | high | 0.010 | 8 | 80 | ⚠️ YES |

### C) Churn Diagnostics

**Purpose:** Answer "is it tradable?"

```python
from src.evaluation import compute_churn_diagnostics

# Summary statistics
diag = compute_churn_diagnostics(churn_series)

# Results per fold/horizon/k:
# - churn_median: Median churn
# - churn_p90: 90th percentile
# - pct_high_churn: % dates with churn > 50%
# - high_churn_flag: True if > 25% of dates have high churn
```

**Flags:**
- ⚠️ **HIGH CHURN**: >25% of dates have churn > 50% (unstable rankings)

**Target:** < 30% median churn for exploitability

### D) Stability Scorecard

**Purpose:** One-screen summary you paste into writeups

```python
from src.evaluation import generate_stability_scorecard

scorecard = generate_stability_scorecard(
    fold_summaries,
    churn_diagnostics=churn_diag,
    cost_overlays=cost_overlays
)

# Columns:
# - fold_id, horizon
# - rankic_median, rankic_iqr
# - quintile_spread_median
# - churn@10_median (if churn_diag provided)
# - net_avg_er, alpha_survives (if cost_overlays provided)
```

### E) Full Report Generation

**Main entry point:**

```python
from src.evaluation import generate_stability_report

outputs = generate_stability_report(
    inputs=StabilityReportInputs(...),
    experiment_name="kronos_v0_h20_monthly",
    output_dir=Path("reports")
)

# Returns: StabilityReportOutputs with paths to all artifacts
# - output_dir: Path to experiment folder
# - ic_decay_stats: Path to CSV
# - regime_performance: Path to CSV
# - churn_diagnostics: Path to CSV
# - stability_scorecard: Path to CSV
# - ic_decay_plot: Path to PNG
# - regime_bars: Path to PNG
# - churn_timeseries: Path to PNG
# - churn_distribution: Path to PNG
# - summary_report: Path to REPORT_SUMMARY.md
```

---

## Test Results (24/24 Passing)

### Test Categories

**1. Stability Thresholds (2 tests)**
- ✅ Frozen dataclass (immutable)
- ✅ Threshold values match spec

**2. IC Decay Analysis (4 tests)**
- ✅ Statistics computation (early/late split)
- ✅ Decay calculation correctness
- ✅ Minimum dates requirement (< 9 dates skipped)
- ✅ Plot creation

**3. Regime Performance (3 tests)**
- ✅ Formatting with coverage stats
- ✅ Thin slice detection (insufficient dates/names)
- ✅ Plot creation

**4. Churn Diagnostics (4 tests)**
- ✅ Summary statistics computation
- ✅ High churn flag (>25% of dates with churn>50%)
- ✅ Timeseries plot creation
- ✅ Distribution plot creation

**5. Stability Scorecard (3 tests)**
- ✅ Basic scorecard generation
- ✅ Scorecard with churn diagnostics
- ✅ Scorecard with cost overlays

**6. Full Report Generation (2 tests)**
- ✅ Complete report with all inputs
- ✅ Minimal report (only required inputs)

**7. Determinism (3 tests)**
- ✅ IC decay stats (shuffled → identical)
- ✅ Churn diagnostics (shuffled → identical)
- ✅ Scorecard (shuffled → identical)

**8. Fold Boundaries (1 test)**
- ✅ Churn never bridges folds

**9. Regime Buckets (1 test)**
- ✅ Bucket totals sum correctly

**10. Exclusion Reporting (1 test)**
- ✅ Insufficient dates logged (not silent)

---

## Usage Example

```python
from pathlib import Path
import pandas as pd
from src.evaluation import (
    generate_stability_report,
    StabilityReportInputs
)

# Assume you've already run evaluate_fold for each fold
# and collected the outputs

# Combine all fold outputs
per_date_metrics = pd.concat([fold1_per_date, fold2_per_date, ...])
fold_summaries = pd.concat([fold1_summary, fold2_summary, ...])
regime_summaries = pd.concat([fold1_regime, fold2_regime, ...])
churn_series = pd.concat([fold1_churn, fold2_churn, ...])
cost_overlays = pd.concat([fold1_costs, fold2_costs, ...])

# Create inputs
inputs = StabilityReportInputs(
    per_date_metrics=per_date_metrics,
    fold_summaries=fold_summaries,
    regime_summaries=regime_summaries,
    churn_series=churn_series,
    cost_overlays=cost_overlays
)

# Generate report
outputs = generate_stability_report(
    inputs=inputs,
    experiment_name="kronos_v0_h20_monthly_2016_2025",
    output_dir=Path("reports/chapter6")
)

# Read key outputs
scorecard = pd.read_csv(outputs.stability_scorecard)
print(scorecard)

# Read summary
with open(outputs.summary_report, 'r') as f:
    print(f.read())
```

---

## Enforced Invariants

### 1. Determinism
**Rule:** Same inputs (even shuffled) → identical outputs

**Test:** `test_ic_decay_determinism`, `test_churn_diagnostics_determinism`, `test_scorecard_determinism`

**Why:** Reports must be reproducible. No randomness allowed.

### 2. Fold Boundaries
**Rule:** Churn/timeseries never bridge folds

**Test:** `test_churn_never_bridges_folds`

**Why:** Each fold is independent. Churn is computed within fold only.

### 3. Regime Bucket Integrity
**Rule:** Bucket totals sum to overall total

**Test:** `test_regime_coverage_sums`

**Why:** Regime slicing must be exhaustive (no missing rows).

### 4. No Silent Drops
**Rule:** Exclusions are logged, not silent

**Test:** `test_insufficient_dates_logged`

**Why:** If rows are dropped (insufficient dates, missing fields), user must know.

### 5. Thin Slice Detection
**Rule:** Flag regime slices with insufficient coverage

**Implementation:** `format_regime_performance` adds `thin_slice_flag`

**Why:** Don't over-trust metrics from 5 dates or 3 stocks.

### 6. Rapid Decay Detection
**Rule:** Flag folds where IC drops > 5% from early to late

**Implementation:** `compute_ic_decay_stats` adds `rapid_decay_flag`

**Why:** Identify overfitting or regime shifts.

### 7. High Churn Detection
**Rule:** Flag folds where >25% of dates have churn > 50%

**Implementation:** `compute_churn_diagnostics` adds `high_churn_flag`

**Why:** Identify unstable rankings (not tradable).

---

## Key Design Decisions

### 1. Pure Consumer (No New Features)
- Reports ONLY consume existing outputs from 6.3 and 6.4
- NO new metrics, NO new computations, NO new modeling
- Just clean rendering + diagnostic flags

### 2. Deterministic Artifacts
- Same inputs → same outputs (even if shuffled)
- All plots use fixed random seeds (if any randomness)
- All aggregations are deterministic (median, not mean of random sample)

### 3. Locked Thresholds
- All diagnostic thresholds frozen in `STABILITY_THRESHOLDS`
- Cannot be changed at runtime
- Prevents "threshold shopping" to make results look better

### 4. Coverage-Aware
- Every regime slice includes `n_dates` and `n_names_median`
- Thin slice flags prevent over-trusting small samples
- Minimum requirements explicit and enforced

### 5. Fold-Aware
- Churn never bridges fold boundaries
- IC decay computed within fold only
- Regime slicing respects fold structure

### 6. Human-Readable
- Markdown summary with tables and interpretations
- CSV tables (easy to load into spreadsheets)
- PNG figures (easy to embed in docs)
- Clear flag indicators (⚠️ for warnings, ✓ for pass)

---

## Next Steps

### Phase 3: Baselines (TODO)
Implement 3 baselines using **identical pipeline**:
1. `mom_12m`: 12-month momentum
2. `momentum_composite`: Equal-weight average of 1/3/6/12m
3. `short_term_strength`: 1-month momentum (diagnostic)

**Critical:** All baselines must:
- Use same `EvaluationRow` format
- Go through same `evaluate_fold()`
- Generate same stability reports
- Use same cost overlays

### Phase 6: End-to-End Execution (TODO)
- Create Qlib adapter for standardized reporting
- Run full evaluation (2016-2025, monthly)
- Generate stability reports per experiment
- Compare model vs baselines
- Validate acceptance criteria

---

## Acceptance Criteria Met

✅ **Determinism**: Shuffled inputs → identical outputs  
✅ **Fold boundaries**: Churn never bridges folds  
✅ **Regime integrity**: Bucket totals sum correctly  
✅ **No silent drops**: Exclusions are logged  
✅ **Thin slice detection**: Flags insufficient coverage  
✅ **Rapid decay detection**: Flags early vs late drops  
✅ **High churn detection**: Flags unstable rankings  
✅ **All 24 tests passing**

---

## Key Deliverables

**Code:**
- `src/evaluation/reports.py` (768 lines)
- `tests/test_reports.py` (545 lines)

**Documentation:**
- `PROJECT_DOCUMENTATION.md` updated (Section 6.5)
- `PROJECT_STRUCTURE.md` updated (Phase 5 status)
- `CHAPTER_6_PHASE5_COMPLETE.md` (this file)

**Tests:**
- 24 new tests (all passing)
- Total Chapter 6: 147 tests passing

**Artifacts:**
- Frozen `STABILITY_THRESHOLDS` dataclass
- IC decay analysis (early vs late)
- Regime performance with coverage
- Churn diagnostics with flags
- Stability scorecard (one-screen summary)
- Full report generation (tables + figures + markdown)

---

## Summary

Phase 5 delivers a **pure consumer reporting layer** that:
- ✅ Consumes existing 6.3 metrics + 6.4 costs (no new features)
- ✅ Generates deterministic artifacts (same inputs → same outputs)
- ✅ Enforces coverage thresholds (thin slice detection)
- ✅ Detects rapid decay (early vs late IC drops)
- ✅ Detects high churn (unstable rankings)
- ✅ Respects fold boundaries (no bridging)
- ✅ Logs exclusions (no silent drops)
- ✅ Produces human-readable reports (tables + figures + markdown)

**Chapter 6 Progress:**
- ✅ Phase 0: Sanity Checks (16 tests)
- ✅ Phase 1: Walk-Forward (25 tests)
- ✅ Phase 1.5: Definition Lock (40 tests)
- ✅ Phase 2: Metrics (30 tests)
- ✅ Phase 4: Cost Realism (28 tests)
- ✅ Phase 5: Stability Reports (24 tests)
- 🔄 Phase 3: Baselines (TODO)
- 🔄 Phase 6: End-to-End Execution (TODO)

**Total: 163/163 tests passing (147 in evaluation suite)**

Ready to proceed to Phase 3 (Baselines) or Phase 6 (End-to-End) when needed.


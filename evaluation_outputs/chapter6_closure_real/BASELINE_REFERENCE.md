# Chapter 6 Baseline Reference

**Generated:** 2025-12-30T05:45:21.185964Z
**Commit:** bf2cf8e0e74a62100ac15314eaf74cbf4decbac0

---

## Baseline Floor Summary

This document records the frozen baseline floor for Chapter 7+ model comparisons.

### Best Baseline per Horizon

| Horizon | Best Baseline | Median RankIC |
|---------|---------------|---------------|
| 20d | mom_12m_monthly | 0.0283 |
| 60d | momentum_composite_monthly | 0.0392 |
| 90d | momentum_composite_monthly | 0.0169 |

### Sanity Check

**naive_random RankIC ≈ 0:** ✅ PASSED

### Data Snapshot

- **Source:** duckdb
- **Rows:** 192,307
- **Date Range:** 2016-01-04T00:00:00 to 2025-02-19T00:00:00
- **Data Hash:** 5723d4c88b8ecba1...

### Environment

- **Python:** 3.11.5
- **Platform:** macOS-26.1-x86_64-i386-64bit

---

## Usage

To compare a model against these baselines:

```python
from src.evaluation import compute_acceptance_verdict

# Load frozen baseline summaries
baseline_summaries = load_baseline_summaries('evaluation_outputs/chapter6_closure')

# Run your model through the same pipeline
model_results = run_experiment(model_spec, features_df, output_dir, FULL_MODE)

# Compare
verdict = compute_acceptance_verdict(
    model_results['fold_summaries'],
    baseline_summaries,
    model_results['cost_overlays'],
    model_results['churn_series']
)
```

---

**Frozen at commit:** `bf2cf8e0e74a62100ac15314eaf74cbf4decbac0`
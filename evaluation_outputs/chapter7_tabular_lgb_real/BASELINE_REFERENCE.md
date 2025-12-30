# Chapter 7 Tabular ML Baseline Reference

**Generated:** 2025-12-30T22:39:52.379060Z
**Commit:** a0368a63
**Data Hash:** f3899b37cb9f34f1

---

## Baseline: `tabular_lgb`

**Model:** LightGBM Regressor with time-decay weighting
**Training:** Per-fold, horizon-specific (20/60/90d)
**Features:** Momentum, volatility, drawdown, liquidity, relative strength, beta
**Hyperparameters:** Fixed (n_estimators=100, lr=0.05, depth=5, leaves=31)

---

## Performance Summary

### Monthly Cadence (Primary)

**Folds:** 3

| Horizon | Median RankIC | IC Stability | Churn (Top-10) | Cost Survival (% Positive) |
|---------|---------------|--------------|----------------|---------------------------|
| 20d | 0.0518 | 0.2007 | 0.20 | 0.0% |
| 60d | 0.1911 | 0.2636 | 0.20 | 33.3% |
| 90d | 0.2202 | 0.2813 | 0.20 | 33.3% |

### Quarterly Cadence (Robustness Check)

**Folds:** 3

| Horizon | Median RankIC | IC Stability | Churn (Top-10) | Cost Survival (% Positive) |
|---------|---------------|--------------|----------------|---------------------------|
| 20d | 0.0302 | 0.1662 | 0.20 | 0.0% |
| 60d | -0.0230 | 0.2611 | 0.20 | 66.7% |
| 90d | -0.0066 | 0.1988 | 0.20 | 33.3% |

---

## Comparison vs Chapter 6 Frozen Floor

| Horizon | Frozen Floor (Factor) | tabular_lgb (ML) | Lift |
|---------|----------------------|------------------|------|
| 20d | 0.0283 | 0.0518 | +0.0235 |
| 60d | 0.0392 | 0.1911 | +0.1519 |
| 90d | 0.0169 | 0.2202 | +0.2034 |

**Frozen Floor Baseline:** 20d=mom_12m_monthly, 60d=momentum_composite_monthly, 90d=momentum_composite_monthly

---

## Data Snapshot

**Rows:** 192,307
**Tickers:** 100
**Date Range:** 2016-01-04 00:00:00 → 2025-02-19 00:00:00
**Horizons:** 20, 60, 90d

---

## Output Artifacts

- `monthly/eval_rows.parquet`: Per-row scored observations
- `monthly/fold_summaries.csv`: Per-fold aggregate metrics
- `monthly/cost_overlays.csv`: Cost sensitivity analysis
- `monthly/churn_series.csv`: Portfolio turnover metrics
- `monthly/stability_report/`: IC decay, regime tables, diagnostics
- `quarterly/`: Same structure for quarterly cadence
- `BASELINE_REFERENCE.md`: This file
- `CLOSURE_MANIFEST.json`: Reproducibility metadata

---

**Note:** Chapter 6 frozen artifacts remain unchanged at `evaluation_outputs/chapter6_closure_real/`

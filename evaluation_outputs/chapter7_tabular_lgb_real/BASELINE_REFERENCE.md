# Chapter 7 Tabular ML Baseline Reference

**Generated:** 2025-12-30T21:20:04.886903Z
**Commit:** a264fd2a
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

**Folds:** 109

| Horizon | Median RankIC | IC Stability | Churn (Top-10) | Cost Survival (% Positive) |
|---------|---------------|--------------|----------------|---------------------------|
| 20d | 0.1009 | 0.1687 | 0.20 | 6.4% |
| 60d | 0.1275 | 0.1238 | 0.20 | 45.9% |
| 90d | 0.1808 | 0.1183 | 0.20 | 56.9% |

### Quarterly Cadence (Robustness Check)

**Folds:** 36

| Horizon | Median RankIC | IC Stability | Churn (Top-10) | Cost Survival (% Positive) |
|---------|---------------|--------------|----------------|---------------------------|
| 20d | 0.0462 | 0.2249 | 0.20 | 0.0% |
| 60d | 0.0744 | 0.2150 | 0.20 | 36.1% |
| 90d | 0.1035 | 0.1699 | 0.20 | 50.0% |

---

## Comparison vs Chapter 6 Frozen Floor

| Horizon | Frozen Floor (Factor) | tabular_lgb (ML) | Lift |
|---------|----------------------|------------------|------|
| 20d | 0.0283 | 0.1009 | +0.0726 |
| 60d | 0.0392 | 0.1275 | +0.0883 |
| 90d | 0.0169 | 0.1808 | +0.1639 |

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

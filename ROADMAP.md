# AI Stock Forecaster - Project Roadmap

**Last Updated:** December 30, 2025

---

## ✅ Completed (Production Ready)

### Chapter 1-4: Infrastructure & Data
- ✅ System outputs (signals, rankings, reports)
- ✅ FMP API client (split-adjusted OHLCV + dividends)
- ✅ Point-in-time data store (DuckDB)
- ✅ PIT scanner (zero CRITICAL violations)
- ✅ AI stock universe (100 stocks, 10 categories)
- ✅ Trading calendar (NYSE holidays, market close timing)

### Chapter 5: Feature Engineering
- ✅ Labels: v2 total return (dividends, mature-aware, PIT-safe)
- ✅ Price features: Momentum (1m/3m/6m/12m), volatility, drawdown
- ✅ Fundamental features: Relative ratios vs sector
- ✅ Event features: Earnings, filings, calendars
- ✅ Regime features: VIX, market trend, sector rotation
- ✅ Missingness: Coverage tracking, "known at time T" masks
- ✅ Hygiene: Standardization, correlation, VIF, IC stability
- ✅ Neutralization: Sector/beta/market neutral IC

### Chapter 6: Evaluation Realism 🔒 FROZEN
**Status:** CLOSED & FROZEN (December 30, 2025)  
**Tests:** 413/413 passing  
**Commits:** `18bad8a` + `7e6fa3a`

**Frozen Infrastructure:**
- ✅ Definition lock (horizons/embargo = TRADING DAYS, UTC maturity)
- ✅ Walk-forward splitter (expanding window + purging/embargo/maturity)
- ✅ Sanity checks (IC parity + experiment naming)
- ✅ Metrics (EvaluationRow contract + RankIC/churn/regime slicing)
- ✅ Cost overlay (base + ADV-scaled slippage + sensitivity bands)
- ✅ Stability reports (IC decay, regime tables, churn diagnostics)
- ✅ 4 baselines (mom_12m, momentum_composite, short_term_strength, naive_random)
- ✅ End-to-end orchestrator (SMOKE/FULL modes)
- ✅ Qlib shadow evaluator (IC parity verification)
- ✅ DuckDB feature store (192,307 rows, 2016-2025)

**Frozen Baseline Floor (REAL Data):**

| Horizon | Best Baseline | Median RankIC | Churn | Cost Survival |
|---------|---------------|---------------|-------|---------------|
| **20d** | `mom_12m_monthly` | 0.0283 | 0.10 | 5.8% positive |
| **60d** | `momentum_composite_monthly` | 0.0392 | 0.10 | 25.1% positive |
| **90d** | `momentum_composite_monthly` | 0.0169 | 0.10 | 40.1% positive |

**Frozen Artifacts:** `evaluation_outputs/chapter6_closure_real/` (tracked in git)

**Reference:** See `CHAPTER_6_FREEZE.md` for complete details.

---

## 🔧 In Progress

### Chapter 7: Baseline Models (Models to Beat)

**Goal:** Establish ML baseline floor and gating thresholds.

**Status:** 🔧 NEXT

**Deliverables:**
1. **Implement `tabular_lgb` Baseline**
   - LightGBM with time-decay sample weighting
   - Per-fold training using walk-forward splits (purging/embargo/maturity)
   - Horizon-specific models (separate for 20/60/90d)
   - One-time deterministic hyperparameter tuning (no "baseline shopping")
   
2. **Formalize Gating Policy**
   - Factor gate: `median_RankIC(best_factor) ≥ 0.02`
   - ML gate: `median_RankIC(tabular_lgb) ≥ 0.05`
   - TSFM rule (Chapters 8-12): Must beat tuned ML baseline
   
3. **Update Baseline Floor**
   - Re-run FULL_MODE with `tabular_lgb` included
   - Re-freeze `BASELINE_FLOOR.json` if `tabular_lgb` beats momentum baselines
   - Commit new frozen artifacts

**Acceptance Criteria (vs Chapter 6 Frozen Floor):**
- ✅ `tabular_lgb` RankIC ≥ 0.05 (ML gate)
- ✅ Beats momentum baselines on at least 2 of 3 horizons
- ✅ Cost survival improvement: % positive folds ≥ baseline + 10pp (frozen floor: 5.8%-40.1%)
- ✅ Churn ≤ 0.30 (tradable turnover)

---

## 📋 Planned (TODO)

### Chapter 8: Kronos Integration
**Goal:** Integrate Kronos foundation model for K-line price dynamics prediction.

**Deliverables:**
- [ ] Kronos model adapter (OHLCV → embedding → horizon-specific heads)
- [ ] Inference pipeline (batch processing, caching)
- [ ] Integration with evaluation pipeline (must use frozen Chapter 6 framework)
- [ ] Run FULL_MODE and compare vs `tabular_lgb` baseline

**Acceptance Gate:** Median RankIC ≥ 0.05 (must beat ML baseline)

---

### Chapter 9: FinText-TSFM Integration
**Goal:** Integrate FinText time series foundation model for return structure prediction.

**Deliverables:**
- [ ] FinText model adapter (time series → embedding → horizon-specific heads)
- [ ] Inference pipeline (batch processing, caching)
- [ ] Integration with evaluation pipeline (must use frozen Chapter 6 framework)
- [ ] Run FULL_MODE and compare vs `tabular_lgb` baseline

**Acceptance Gate:** Median RankIC ≥ 0.05 (must beat ML baseline)

---

### Chapter 10: Context Features (Tabular)
**Goal:** Add fundamentals and macro regime context to complement price/time-series signals.

**Deliverables:**
- [ ] Fundamental features: P/E, P/S, P/B relative to sector
- [ ] Macro features: Interest rates, unemployment, sentiment
- [ ] Tabular model trained on context features
- [ ] Integration with evaluation pipeline
- [ ] Run FULL_MODE and compare vs `tabular_lgb` baseline

**Acceptance Gate:** Median RankIC ≥ 0.05 (must beat ML baseline)

---

### Chapter 11: Fusion Model
**Goal:** Combine Kronos, FinText, and tabular context into a single fusion model.

**Deliverables:**
- [ ] Fusion architecture (stacking, weighted average, or learned fusion)
- [ ] Per-fold training with time-decay weighting
- [ ] Regime-aware fusion (different weights for different market conditions)
- [ ] Integration with evaluation pipeline
- [ ] Run FULL_MODE and compare vs individual models

**Acceptance Gate:** Median RankIC > best individual model by ≥ 0.01

---

### Chapter 12: Regime-Aware Ensembling
**Goal:** Dynamic weighting of models based on detected market regime.

**Deliverables:**
- [ ] Regime detection (VIX, trend, volatility, earnings window)
- [ ] Per-regime model performance analysis
- [ ] Dynamic ensemble weights (offline training, online application)
- [ ] Integration with evaluation pipeline
- [ ] Run FULL_MODE and compare vs static fusion

**Acceptance Gate:** Median RankIC > static fusion by ≥ 0.005

---

### Chapter 13: Calibration & Confidence
**Goal:** Produce calibrated probability distributions and confidence intervals.

**Deliverables:**
- [ ] Calibration layer (isotonic regression, Platt scaling)
- [ ] Uncertainty quantification (ensemble variance, dropout, conformal prediction)
- [ ] Confidence stratification (high/medium/low confidence signals)
- [ ] Backtest analysis stratified by confidence

**Acceptance Gate:** Calibration error < 5%, high-confidence signals outperform by ≥ 0.02 RankIC

---

### Chapter 14: Production Monitoring & Alerts
**Goal:** Real-time monitoring, drift detection, and automated alerts.

**Deliverables:**
- [ ] Data drift detection (feature distributions, missing data)
- [ ] Model drift detection (IC decay, prediction distribution shifts)
- [ ] Automated alerts (email, Slack, PagerDuty)
- [ ] Dashboard (Streamlit, Plotly Dash, or similar)
- [ ] Incident response runbook

**Acceptance Gate:** Alert on IC drop > 0.01 within 1 trading day

---

## 📊 Test Coverage

| Component | Tests | Status |
|-----------|-------|--------|
| **Infrastructure (Ch 1-4)** | 84/84 | ✅ |
| **Features (Ch 5)** | 60/60 | ✅ |
| **Evaluation (Ch 6)** | 269/269 | ✅ 🔒 FROZEN |
| **Total** | **413/413** | ✅ |

---

## 🎯 Success Metrics (Final System)

### Primary Goals
- **RankIC:** Median walk-forward RankIC ≥ 0.10 (0.05 ML baseline + 0.05 model lift)
- **Net-of-Cost:** ≥ 80% of folds profitable after base costs (20 bps round-trip)
- **Churn:** Median churn ≤ 0.25 (tradable portfolio turnover)
- **Regime Robustness:** No catastrophic collapse (RankIC > 0 in all VIX/bull/bear buckets)

### Secondary Goals
- **Hit Rate @10:** ≥ 60% of Top-10 stocks have positive excess returns
- **Quintile Spread:** Top - Bottom quintile spread ≥ 10% annualized
- **Calibration:** Prediction distributions are well-calibrated (< 5% calibration error)
- **Monitoring:** Automated alerts detect drift within 1 trading day

---

## 🚀 Quick Start

### Build DuckDB Feature Store (requires FMP Premium)
```bash
# Ensure FMP_KEYS is set in .env
python scripts/build_features_duckdb.py \
  --start-date 2014-01-01 \
  --end-date 2025-06-30 \
  --auto-normalize-splits
```

### Run Chapter 6 Closure (Baseline Reference)
```bash
# Uses frozen evaluation pipeline + real DuckDB data
python scripts/run_chapter6_closure.py
# Outputs: evaluation_outputs/chapter6_closure_real/
```

### Run Tests
```bash
# All 413 tests (includes Chapter 6 frozen tests)
pytest tests/ -v

# Chapter 6 only
pytest tests/test_*.py -k evaluation -v
```

### Load Frozen Baseline Floor
```python
import json
from pathlib import Path

# Load frozen baseline floor
floor_path = Path("evaluation_outputs/chapter6_closure_real/BASELINE_FLOOR.json")
with floor_path.open() as f:
    baseline_floor = json.load(f)

# Get RankIC to beat for a specific horizon
horizon = 60
best_baseline = baseline_floor["best_baseline_per_horizon"][str(horizon)]
print(f"Your model must beat {best_baseline['baseline']} RankIC: {best_baseline['median_rankic']:.4f}")
# Output: Your model must beat momentum_composite_monthly RankIC: 0.0392
```

---

## 📚 Key Documents

| Document | Purpose |
|----------|---------|
| `CHAPTER_6_FREEZE.md` | Complete Chapter 6 freeze details (baseline floor, bugs fixed, reproducibility) |
| `PROJECT_DOCUMENTATION.md` | Full system documentation (all chapters) |
| `PROJECT_STRUCTURE.md` | Directory structure + implementation status |
| `AI_Stock_Forecaster_(FinText_+_Kronos_+_Context).ipynb` | Main notebook (with Chapter 6 freeze banner) |
| `CHAPTER_6_PHASE6_COMPLETE.md` | Chapter 6 implementation summary |

---

**Next Action:** Implement Chapter 7 (`tabular_lgb` baseline + gates)  
**Questions?** See `CHAPTER_6_FREEZE.md` for complete details on the frozen baseline reference.


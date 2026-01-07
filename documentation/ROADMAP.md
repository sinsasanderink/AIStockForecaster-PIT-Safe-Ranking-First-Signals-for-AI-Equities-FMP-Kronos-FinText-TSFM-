# AI Stock Forecaster - Project Roadmap

**Last Updated:** January 7, 2026

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

### Chapter 7: Baseline Models (Models to Beat) ✅ FROZEN

**Goal:** Establish ML baseline floor and gating thresholds.

**Status:** ✅ COMPLETE + FROZEN (tag: `chapter7-tabular-lgb-freeze`)

**Deliverables:**
1. **Implement `tabular_lgb` Baseline** ✅ COMPLETE
   - LightGBM Regressor with time-decay sample weighting
   - Per-fold training using walk-forward splits (purging/embargo/maturity)
   - Horizon-specific models (separate for 20/60/90d)
   - Fixed hyperparameters (no tuning in baseline): n_estimators=100, lr=0.05, max_depth=5
   - **Implementation:** `src/evaluation/baselines.py` (ML baselines section)
   - **Tests:** `tests/test_ml_baselines.py` (13 tests, all passing)
   
2. **Formalize Gating Policy** ✅ COMPLETE
   - Factor gate: `median_RankIC(best_factor) ≥ 0.02`
   - ML gate: `median_RankIC(tabular_lgb) ≥ 0.05`
   - TSFM rule (Chapters 8-12): Must beat tuned ML baseline
   - **Implementation:** `src/evaluation/run_evaluation.py` (`compute_acceptance_verdict`)
   
3. **FULL_MODE Execution + Freeze** ✅ COMPLETE
   - Script to run `tabular_lgb` and generate baseline reference
   - Loads REAL DuckDB data, runs monthly + quarterly cadences
   - Compares vs frozen Chapter 6 baseline floor
   - **Implementation:** `scripts/run_chapter7_tabular_lgb.py`
   - **Tests:** `tests/test_chapter7_script.py` (3 tests, all passing)
   - **Total Tests:** 429/429 passing

**ML Baseline Floor (FROZEN):**

| **Horizon** | **tabular_lgb (Monthly)** | **Lift vs Factor Floor** |
|-------------|---------------------------|--------------------------|
| **20d** | 0.1009 | **+0.0726** (+256%) |
| **60d** | 0.1275 | **+0.0883** (+225%) |
| **90d** | 0.1808 | **+0.1639** (+970%) |

**Frozen Artifacts:** `evaluation_outputs/chapter7_tabular_lgb_full/` (tracked in git)

**Acceptance Criteria (PASSED):**
- ✅ `tabular_lgb` RankIC ≥ 0.05 (ML gate) - **PASSED** (all horizons)
- ✅ Beats momentum baselines on all 3 horizons - **PASSED**
- ✅ Cost survival: 6.4%/45.9%/56.9% (20d/60d/90d)
- ✅ Churn: 0.20 (well below 0.30 threshold)

**Reference:** See `evaluation_outputs/chapter7_tabular_lgb_full/BASELINE_REFERENCE.md`

---

## 📋 Planned (TODO)

### Pre-Chapter-8: Feature Store Expansion ✅ COMPLETE

**Final State (Jan 7, 2026):**
- ✅ DuckDB expanded to **52 features** (was 7 in Chapter 7 freeze)
- ✅ 201,307 feature rows, 600,855 labels, 2,386 regime rows
- ✅ Date range: 2016-01-04 to 2025-06-30
- ✅ Chapter 7 baseline frozen and preserved (backup at `data/features_chapter7_freeze.duckdb`)
- ✅ Data hash: `a6142358f0e9ac57...` (current production)

**Completed Batches:**
- Batch 1+2 (Jan 2): Price/Volume + Missingness (9 features)
- Batch 3 (Jan 2): Events/Earnings (12 features)
- Batch 4 (Jan 2): Regime/Macro (12 features)
- Batch 5 (Jan 7): Fundamentals Phase 1 (9 features) - validated stepwise behavior

**Required Before Chapter 8:**

**Priority 1: Price/Volume Features (14 total)** ✅ COMPLETE (Jan 2, 2026)
- [x] Add `vol_of_vol` (volatility of volatility)
- [x] Add `max_drawdown_60d` (drawdown features)
- [x] Add `dist_from_high_60d` (distance from high)
- [x] Add `adv_60d` (60-day ADV)
- [x] Add `rel_strength_1m`, `rel_strength_3m` (relative strength vs universe)
- [x] Add `beta_252d` (beta vs QQQ - placeholder, set to None pending benchmark merge)

**Priority 2: Events/Earnings Features (12 total) - Critical for Earnings Gap Issue** ✅ COMPLETE (Jan 2, 2026)
- [x] Add `days_to_earnings`, `days_since_earnings` (earnings timing) - 100% coverage
- [x] Add `in_pead_window`, `pead_window_day` (post-earnings drift) - 50% coverage (correct: only when within PEAD window)
- [x] Add `last_surprise_pct`, `avg_surprise_4q`, `surprise_streak`, `surprise_zscore`, `earnings_vol` (surprise history)
- [x] Add `days_since_10k`, `days_since_10q`, `reports_bmo` (filing recency)

**Priority 3: Regime/Macro Features (12 total)** ✅ COMPLETE (Jan 2, 2026)
- [x] Add VIX features: `vix_level`, `vix_percentile`, `vix_change_5d`, `vix_regime`
- [x] Add market features: `market_return_5d/21d/63d`, `market_vol_21d`, `market_regime`
- [x] Add technical: `above_ma_50`, `above_ma_200`, `ma_50_200_cross`
- [ ] ~~Sector rotation: `tech_vs_staples`, `tech_vs_utilities`, `risk_on_indicator`~~ (Deferred - needs sector ETF data)

**Batch 4 Validation (Jan 2, 2026):**
- [x] Unit tests pass (429 passed)
- [x] Chapter 7 smoke test passes
- [x] Regime PIT sniff tests pass (per-date consistency, backward windows, no leakage)
- [x] All risks mitigated (calendar alignment, merge semantics)
- See: `BATCH4_VALIDATION_COMPLETE.md`

**Priority 4: Fundamentals (7 total) - FMP Premium Available** ⚠️ HIGH RISK
- [x] Phase 1: `gross_margin_vs_sector`, `operating_margin_vs_sector` (filings-only) ✅ IMPLEMENTED
- [x] Phase 1: `revenue_growth_vs_sector`, `roe_zscore` (filings-only) ✅ IMPLEMENTED
- [x] Phase 1: `sector` (static from profile, documented limitation) ✅ IMPLEMENTED
- [ ] Phase 2: `pe_zscore_3y`, `pe_vs_sector`, `ps_vs_sector` (requires PIT-safe price + shares)

**⚠️ Batch 5 PIT Safety Implementation:**
- Phase 1 (filings-only): Uses TTM (sum of 4 quarters), forward-fill between filings
- Phase 2 (valuation): PENDING - requires as-of close from features_df + shares from filings
- Sector z-scores: Minimum 5 tickers per sector, else NaN
- See: `BATCH5_PLAN.md` for detailed implementation plan

**Deferred Features:**
- Sector rotation: `tech_vs_staples`, `tech_vs_utilities`, `risk_on_indicator` 
- Reason: Requires sector ETF data (XLK, XLP, XLU) not in pipeline

**Priority 5: Missingness (2 total) - Diagnostic Only** ✅ COMPLETE (Jan 2, 2026)
- [x] Add `coverage_pct` (% non-null features)
- [x] Add `is_new_stock` (<252 trading days)

**Implementation:**
```bash
# Expand scripts/build_features_duckdb.py to call:
# - src/features/price_features.py (Priority 1)
# - src/features/event_features.py (Priority 2)
# - src/features/regime_features.py (Priority 3)
# - src/features/fundamental_features.py (Priority 4, needs FMP Premium)
# - src/features/missingness.py (Priority 5)

# Then rebuild DuckDB:
python scripts/build_features_duckdb.py --auto-normalize-splits
```

**Status Update (Jan 7, 2026):**
- ✅ Batch 1+2 complete: 9 new features added (price/volume + missingness)
- ✅ Batch 3 complete: 12 event/earnings features added
- ✅ Batch 4 complete: 12 regime/macro features added
- ✅ Batch 5 Phase 1 COMPLETE + VALIDATED: 9 fundamental features (4 raw TTM + 4 z-scores + sector)
- ⏳ Batch 5 Phase 2 DEFERRED: 3 valuation features (P/E, P/S) - needs PIT-safe price/shares
- ⏳ Deferred: 3 sector rotation features (needs ETF data)
- ✅ DuckDB rebuild: COMPLETE (52 columns, 201K rows)
- ✅ Frozen baseline preserved: `data/features_chapter7_freeze.duckdb`
- ✅ Tests passing: 429/429
- ✅ Chapter 6 REAL closure complete with baseline floor frozen

**Note:** Chapter 7 baseline (`tabular_lgb`) is frozen and only uses 13 features. Expanding DuckDB does NOT invalidate frozen artifacts. The frozen 7-feature snapshot is backed up at `data/features_chapter7_freeze.duckdb` for exact reproducibility.

**Validation Documents:**
- `BATCH5_VALIDATION_COMPLETE.md` - Stepwise behavior validated
- `BATCH5_BUGFIX.md` - 5 bugs fixed (PIT safety, filing classification, etc.)

---

### Chapter 8: Kronos Integration 🟢 READY TO START
**Goal:** Integrate Kronos foundation model for K-line price dynamics prediction.

**Prerequisites:** ✅ ALL MET
- ✅ Chapter 6 closed with baseline floor frozen
- ✅ Chapter 7 baseline frozen (ML floor to beat: 0.1009/0.1275/0.1808)
- ✅ Feature store expanded (52 features, 201K rows)
- ✅ OHLCV data available in DuckDB
- ✅ Tests passing (429/429)

**Deliverables:**
- [ ] Kronos model adapter (OHLCV → embedding → horizon-specific heads)
- [ ] ReVIN normalization (rolling mean/std for price level invariance)
- [ ] Inference pipeline (batch processing, caching)
- [ ] Integration with evaluation pipeline (must use frozen Chapter 6 framework)
- [ ] Run FULL_MODE and compare vs `tabular_lgb` baseline

**Acceptance Gates:**
- Gate 1: Zero-shot RankIC ≥ 0.02 (factor baseline)
- Gate 2: RankIC ≥ 0.05 (ML gate)
- Gate 3: Approach `tabular_lgb` (≥0.08/0.10/0.15 for 20d/60d/90d)

**Reference:** See `CHAPTER_8_PLAN.md` for detailed implementation plan

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

**Next Action:** Implement Chapter 8 (Kronos integration)  
**Questions?** See `CHAPTER_8_PLAN.md` for detailed implementation plan.


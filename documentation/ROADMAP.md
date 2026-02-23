# AI Stock Forecaster - Project Roadmap

**Last Updated:** February 19, 2026
**Current Phase:** Chapter 13 IN PROGRESS — DEUP Uncertainty Quantification (13.0–13.5 complete, 13.6 next)

---

## Overview

This roadmap tracks the implementation status of the AI Stock Forecaster, a signal-only decision-support system for AI stock ranking using foundation models (Kronos + FinText-TSFM) with tabular context features.

---

## Chapter Status Summary

| Chapter | Status | Completion | Last Updated |
|---------|--------|------------|--------------|
| Ch1-2: Outputs & System Design | ✅ COMPLETE | 100% | Dec 2025 |
| Ch3-4: Data & Universe | ✅ COMPLETE | 100% | Dec 2025 |
| Ch5: Feature Engineering | ✅ COMPLETE | 100% | Dec 2025 |
| Ch6: Evaluation Framework | 🔒 FROZEN | 100% | Dec 30, 2025 |
| Ch7: Baseline Models | 🔒 FROZEN | 100% | Jan 5, 2026 |
| **Ch8: Kronos (TSFM)** | ✅ **COMPLETE** | **100%** | **Jan 9, 2026** |
| **Ch9: FinText-TSFM** | **✅ COMPLETE** | **100%** | **Feb 17, 2026** |
| **Ch10: NLP Sentiment** | **✅ COMPLETE** | **100%** | **Feb 18, 2026** |
| **Ch11: Fusion Model** | **✅ COMPLETE** | **100%** | **Feb 19, 2026** |
| **Ch12: Regime Analysis** | **✅ COMPLETE** | **100%** | **Feb 19, 2026** |
| Ch13: DEUP Uncertainty Quantification | ⏳ IN PROGRESS | 95% | Feb 2026 |
| Ch14: Monitoring | ⏳ TODO | 0% | - |
| Ch15: Interfaces | ⏳ TODO | 0% | - |
| Ch16: Acceptance Criteria & Factor Attribution | ⏳ TODO | 0% | - |
| Ch17: Bayesian UQ Extensions & Model Comparisons | ⏳ TODO | 0% | - |

---

## Chapter 1-2: System Outputs & Design ✅ COMPLETE

**Purpose:** Define signal-only outputs and establish project scope

**Key Deliverables:**
- ✅ Signal format specification (EvaluationRow contract)
- ✅ Per-stock outputs (expected excess return, distribution, ranking score)
- ✅ Cross-sectional outputs (ranked lists, confidence buckets)
- ✅ Scope boundaries (signals only, no execution/trading)

**Status:** Complete and frozen

---

## Chapter 3-4: Data & Universe Infrastructure ✅ COMPLETE

**Purpose:** Build point-in-time safe data infrastructure with survivorship-bias-free universe

**Key Deliverables:**
- ✅ FMP client (OHLCV, fundamentals, profiles)
- ✅ DuckDB PIT store (observed_at timestamps)
- ✅ Event store (earnings, filings, sentiment)
- ✅ Security master (stable IDs, ticker changes)
- ✅ Universe builder (survivorship-safe, Polygon-backed)
- ✅ Trading calendar (NYSE holidays, cutoffs)
- ✅ PIT violation scanner (0 CRITICAL violations)

**Tests:** 413/413 passing  

**Status:** Complete and frozen

---

## Chapter 5: Feature Engineering ✅ COMPLETE

**Purpose:** Implement 50+ PIT-safe features across 8 batches

**Batches:**
- ✅ Batch 1: Momentum (4 features)
- ✅ Batch 2: Liquidity + Volatility (5 features)
- ✅ Batch 3: Drawdown + Relative Strength + Beta (5 features)
- ✅ Batch 4: Regime & Macro (15 features)
- ✅ Batch 5: Fundamentals (9 features) - TTM + Z-scores
- ✅ Batch 6: Events & Earnings (12 features)
- ✅ Batch 7: Missingness Masks (2 features)
- ✅ Batch 8: Feature Hygiene & Neutralization (diagnostics)

**DuckDB Store:**
- Features: 201,307 rows (51 columns)
- Labels: 600,855 rows (excess_return_20d/60d/90d)
- Regime: 2,386 rows (15 macro features)
- Date range: 2016-01-04 to 2025-06-30

**Validation:**
- ✅ Step-function tests passed (fundamentals piecewise-constant between filings)
- ✅ PIT scanner: 0 CRITICAL violations
- ✅ Coverage > 95% for all feature batches

**Status:** Complete and frozen

---

## Chapter 6: Evaluation Framework 🔒 FROZEN

**Purpose:** Build walk-forward evaluation with frozen baseline reference

**Key Deliverables:**
- ✅ Walk-forward splitter (expanding window, purging/embargo/maturity)
- ✅ EvaluationRow contract
- ✅ Metrics (RankIC, quintile spread, hit rate, churn)
- ✅ Cost overlay diagnostics
- ✅ Stability reports (regime-conditional performance)
- ✅ Qlib shadow evaluator integration
- ✅ SMOKE and FULL evaluation modes

**Frozen Baseline Floor (REAL DuckDB Data):**

| Horizon | Best Baseline | Median RankIC | Quintile Spread | Hit Rate @10 | N Folds |
|---------|---------------|---------------|-----------------|--------------|---------|
| 20d | mom_12m_monthly | 0.0283 | 0.0035 | 0.50 | 109 |
| 60d | momentum_composite_monthly | 0.0392 | 0.0370 | 0.60 | 109 |
| 90d | momentum_composite_monthly | 0.0169 | 0.0374 | 0.60 | 109 |

**Artifacts:** `evaluation_outputs/chapter6_closure_real/` (tracked in git)

**Status:** FROZEN (Dec 30, 2025) - Immutable baseline reference

---

## Chapter 7: Baseline Models 🔒 FROZEN

**Purpose:** Establish factor and ML baselines that models must beat

**Baselines Implemented:**
1. ✅ `mom_12m` - 12-month momentum (primary naive baseline)
2. ✅ `momentum_composite` - Multi-horizon momentum
3. ✅ `short_term_strength` - 1-month momentum (diagnostic)
4. ✅ `naive_random` - Sanity check (RankIC ≈ 0)
5. ✅ `tabular_lgb` - LightGBM ML baseline

**Frozen ML Baseline (tabular_lgb):**

| Horizon | Median RankIC | Lift vs Factor Floor |
|---------|---------------|---------------------|
| 20d | 0.1009 | +0.0726 |
| 60d | 0.1275 | +0.0883 |
| 90d | 0.1808 | +0.1639 |

**Gates for Future Models:**
- Gate 1 (Factor): RankIC ≥ 0.02 for ≥2 horizons
- Gate 2 (ML): Any horizon RankIC ≥ 0.05 or within 0.03 of LGB
- Gate 3 (Practical): Churn ≤ 30%, stable across regimes

**Artifacts:** `evaluation_outputs/chapter7_tabular_lgb_full/` (frozen with git tag)

**Status:** FROZEN (Jan 5, 2026) - Immutable ML baseline reference

---

## Chapter 8: Kronos (Time Series Foundation Model) ✅ PHASE 2 COMPLETE

**Purpose:** Integrate Kronos foundation model for OHLCV/K-line price dynamics prediction

### Current Status: Phase 2 Kronos Adapter (100% Complete)

**Phase 1: Data Plumbing ✅ COMPLETE**
1. ✅ `PricesStore` implementation (`src/data/prices_store.py`)
2. ✅ Global trading calendar loader (`src/data/trading_calendar.py`)
3. ✅ Test suites (34 tests: 18 + 16, all passing)
4. ✅ Prices table added to DuckDB (243,101 rows, 100 tickers)
5. ✅ Critical bugs fixed (INSERT statement, date type handling)

**Phase 2: Kronos Adapter ✅ COMPLETE**
1. ✅ `KronosAdapter` class (`src/models/kronos_adapter.py`, 514 lines)
   - Batch inference with `predict_batch()`
   - Uses PricesStore (NOT features_df) for OHLCV
   - Uses global trading calendar (NO freq="B")
   - Deterministic settings: T=0.0, top_p=1.0, sample_count=1
   - Score formula: `(pred_close - spot_close) / spot_close`
2. ✅ `kronos_scoring_function` matching `run_experiment()` contract
3. ✅ Single-stock sanity test (`scripts/test_kronos_single_stock.py`)
   - Works in stub mode without Kronos installed
4. ✅ Comprehensive test suite (20 tests: 19 passed, 1 skipped)
   - `tests/test_kronos_adapter.py` (464 lines)
   - All non-negotiables verified
5. ✅ Documentation complete
   - `CHAPTER_8_PHASE2_COMPLETE.md`

**Phase 3: Evaluation Integration ✅ COMPLETE**

**Deliverables:**
- ✅ `scripts/run_chapter8_kronos.py` (745 lines) - Walk-forward evaluation runner
- ✅ Core metrics: RankIC/IC stability, quintile spread, churn, cost survival, regime slices
- ✅ Leak tripwires (negative controls):
  - Shuffle-within-date → RankIC ≈ 0
  - +1 trading-day lag → RankIC collapses
- ✅ Momentum-clone check (correlation vs frozen factor baselines)
- ✅ SMOKE evaluation verified (19,110 eval rows, 3 folds)
- ✅ Phase 2 robustness tweaks applied (per-ticker timestamps, stub predictor, diagnostics)

**Phase 4: Evaluation & Diagnosis ✅ COMPLETE**

**Micro-Test Results (60 predictions, 3 dates):**
- ✅ Technical integration: **WORKING** (all predictions generated)
- ❌ Signal quality: **INSUFFICIENT** (RankIC ~ -0.05, near zero)
- ⚠️ Root cause: **Mean-reversion bias** during momentum regime

**Key Finding:**
```
Date           RankIC    p-value   Verdict
──────────────────────────────────────────
2024-02-01    -0.6692    0.0013    ❌ Strongly INVERTED (significant!)
2024-03-01    +0.2586    0.2709    ⚠️ Weak positive
2024-04-01    +0.1805    0.4465    ⚠️ Weak positive
──────────────────────────────────────────
OVERALL       -0.0530    0.6875    ❌ Near zero
```

**Diagnosis:** Kronos predicted AMD (-27%), NVDA (-26%), META (-19%) would DROP during the Feb 2024 AI rally. Actually: AMD +13%, NVDA +25%, META +22%. Mean-reversion bias baked into model.

**Key Discovery (Jan 9, 2026):** The [Kronos paper](https://arxiv.org/html/2508.02739v1) uses **FINE-TUNED** models, not base pre-trained! From `Kronos/finetune/config.py`:
- Train for 30 epochs on target market (CSI300)
- Test on fine-tuned model, not base
- Even with paper's config (lookback=90, horizon=10, T=0.6): RankIC = -0.56 (still negative)

**Root Cause:** Base model not calibrated for US stocks without fine-tuning. Would require significant compute to fine-tune on our data.

**Deliverables:**
1. ✅ Micro-test script (`scripts/test_kronos_micro.py`)
2. ✅ RankIC computation (`scripts/compute_kronos_rankic.py`)
3. ✅ Diagnosis scripts (`scripts/diagnose_kronos_predictions.py`, `scripts/inspect_kronos_output.py`)
4. ✅ Paper config test (`scripts/test_kronos_paper_config.py`)
5. ✅ Full analysis documents:
   - `KRONOS_DIAGNOSIS_RESULTS.md`
   - `KRONOS_ROOT_CAUSE_ANALYSIS.md`
   - `KRONOS_FINAL_INVESTIGATION.md`
   - `documentation/CHAPTER_8_FINAL.md`

**Chapter 8 Conclusion:**
- **Integration:** ✅ Complete (technically working)
- **Signal Quality:** ❌ Does not meet gates (RankIC < 0.02)
- **Root Cause:** Base model needs fine-tuning (paper uses fine-tuned models)
- **Investigation:** Tested paper's config (lookback=90, horizon=10, T=0.6) - still fails
- **Decision:** Document negative result, proceed to Chapter 9 (FinText)

**Gate Results:**
- Gate 1 (RankIC ≥ 0.02): ❌ FAIL (-0.05)
- Gate 2 (RankIC ≥ 0.05 or within 0.03 of LGB): ❌ FAIL  
- Gate 3 (Churn ≤ 30%): N/A (not running full eval)

**Positive Takeaways:**
- Kronos IS learning something (Feb 1 RankIC = -0.67 is significant!)
- Not random noise - consistent mean-reversion view
- Technical integration complete and reusable
- Pipeline validated: PricesStore, trading calendar, scoring function
- **Future option:** Fine-tune Kronos on US data (requires GPU compute, 30 epochs)
- **Alternative:** Use as contrarian signal in mean-reverting regimes

**References:**
- GitHub: https://github.com/shiyu-coder/Kronos
- Paper: https://arxiv.org/html/2508.02739v1
- Full Analysis: `KRONOS_DIAGNOSIS_RESULTS.md`

**Status:** 100% COMPLETE - Negative result documented, proceeding to Chapter 9

---

## Chapter 9: FinText-TSFM (Return Structure) ✅ COMPLETE

**Purpose:** Integrate FinText-TSFM foundation models (finance-native pre-trained Chronos) for daily excess return forecasting

**Why FinText (not more Kronos tuning):**
- Kronos (Ch8) showed generic TSFM fails zero-shot on US stocks (mean-reversion bias)
- FinText models are pre-trained FROM SCRATCH on 2B+ financial excess return observations
- Year-specific models (2000-2023) are inherently PIT-safe
- Input = daily excess returns (matches our label definition exactly)
- Paper: "Re(Visiting) TSFMs in Finance" (Rahimikia et al., 2025, SSRN 5770562)

**Model Selection:**
- Primary: `FinText/Chronos_Small_{YEAR}_US` (46M params, U.S. excess returns)
- Architecture: Amazon Chronos (T5-based), finance-pre-trained by FinText team
- HuggingFace: [huggingface.co/FinText](https://huggingface.co/FinText) (613 models)
- GitHub: [DeepIntoStreams/TSFM_Finance](https://github.com/DeepIntoStreams/TSFM_Finance)

**Completed Deliverables:**

Phase 1: Data Plumbing ✅ (Feb 16, 2026)
- [x] Add QQQ benchmark to DuckDB prices table (2,890 rows, 2014-2025)
- [x] `src/data/excess_return_store.py` - Daily excess return sequences (384 lines)
- [x] Unit tests: 30 passing (init, correctness, PIT, batch, cache, diagnostics)

Phase 2: FinText Adapter ✅ (Feb 16, 2026)
- [x] Install `chronos-forecasting` 2.2.2 (torch upgraded to 2.2.2)
- [x] `src/models/fintext_adapter.py` - Year-aware model loading + inference (569 lines)
- [x] `fintext_scoring_function()` for `run_experiment()` integration
- [x] Stub mode + StubChronosPredictor for testing without model download
- [x] Configurable score aggregation (median/mean/trimmed_mean)
- [x] Unit tests: 48 passing

Phase 3: Evaluation Integration ✅ (Feb 16, 2026)
- [x] `scripts/run_chapter9_fintext.py` - Walk-forward evaluation runner (567 lines)
- [x] SMOKE evaluation (3 folds, pipeline verified)
- [x] Leak tripwires (shuffle + lag + year-mismatch) - all passing

Phase 4: Signal Quality Gates ✅ (Feb 16, 2026)
- [x] EMA score smoothing (half-life=5d) reduces churn from 65-75% → 10-20%
- [x] `scripts/evaluate_fintext_gates.py` - Gate evaluation & tripwire script
- [x] All 3 gates PASS, all 3 tripwires PASS

Phase 5: Ablation Studies ✅ (Feb 17, 2026)
- [x] 12-variant ablation matrix across 6 axes
- [x] `scripts/run_chapter9_ablations.py` — Systematic ablation runner
- [x] Optimal: trimmed_mean + US + 21d lookback + EMA(5) + Small model

Phase 6: Freeze & Documentation ✅ (Feb 17, 2026)
- [x] Sections 9.9–9.11 complete (implementation phases, file checklist, scope)
- [x] Honest FinText vs LGB comparison documented
- [x] All documentation updated

**Frozen Optimal Configuration:**
```python
FinTextAdapter.from_pretrained(
    model_size="Small",              # 46M params
    model_dataset="US",              # US excess returns
    lookback=21,                     # 1-month context
    num_samples=20,                  # Distribution samples
    score_aggregation="trimmed_mean",  # +25% vs median
    horizon_strategy="single_step",
)
# EMA smoothing: half-life = 5 trading days
```

**Final Metrics (Small + trimmed_mean + EMA, SMOKE 3 folds):**

| Metric | FinText | LGB Baseline | Gap |
|--------|---------|-------------|-----|
| 20d RankIC | 0.0742 | 0.1009 | -0.027 |
| 60d RankIC | 0.0820 | 0.1275 | -0.046 |
| 90d RankIC | 0.0504 | 0.1808 | -0.130 |
| Churn | 20% | 20% | same |
| IC Stability (20d) | 76.2% pos | 16.9% | +59.3% |

**Gate Results: ✅ ALL PASS**
- Gate 1 (Factor Baseline): ✅ PASS (all 3 horizons RankIC ≥ 0.02)
- Gate 2 (ML Baseline): ✅ PASS (20d within 0.027 of LGB; all ≥ 0.05)
- Gate 3 (Practical): ✅ PASS (churn 20%)

**Key Finding:** FinText does NOT beat LGB standalone (expected — zero-shot vs supervised). Value is as orthogonal signal for Chapter 11 fusion. FinText cannot overfit by construction; LGB's 90d RankIC 0.1808 likely has overfitting component.

**Test Coverage:** 116 Chapter 9 tests + 461 project tests = **577 total ✅**

**Files Created:** 12 files (6 required + 6 additional beyond outline)

**Status:** ✅ COMPLETE (Feb 17, 2026) — Artifacts frozen, ready for Chapter 10/11
**Documentation:** `documentation/CHAPTER_9.md`

---

## Chapter 10: NLP Sentiment Signal ✅ COMPLETE

**Purpose:** Add text-based sentiment signal orthogonal to price/fundamental features

**Model:** ProsusAI/FinBERT (pre-trained finance sentiment, zero-shot)

**Data Sources:**
- SEC EDGAR 8-K filings (free, unlimited, PIT-safe)
- FinnHub company news API (free tier, 60 req/min)

Phase 1: Sentiment Data Pipeline ✅ (Feb 17, 2026)
- [x] `src/data/sentiment_store.py` — SEC + FinnHub collection pipeline
- [x] `scripts/collect_sentiment_data.py` — Batch collection for universe
- [x] Collected 77,904 records (2,575 SEC 8-K + 75,329 news articles)
- [x] 100/100 tickers with news, 36/100 with SEC 8-K filings
- [x] PIT-safe timestamps: SEC acceptanceDateTime + FinnHub publication time
- [x] 28/28 tests passing

Phase 2: FinBERT Sentiment Scoring ✅ (Feb 18, 2026)
- [x] `src/models/finbert_scorer.py` — FinBERT scoring module (109M params)
- [x] `scripts/score_sentiment_finbert.py` — Batch scoring with MPS GPU
- [x] All 77,904 records scored using ProsusAI/finbert (zero-shot)
- [x] MPS acceleration: 20 rec/s average, ~65 min total
- [x] 22/22 tests passing (18 stub + 4 real model)

Phase 3: Sentiment Feature Engineering ✅ (Feb 18, 2026)
- [x] `src/features/sentiment_features.py` — 9 PIT-safe features
- [x] `enrich_features_df()` integration point for evaluation pipeline
- [x] 33/33 tests passing (30 stub + 3 real data)

Phase 4: Walk-Forward Evaluation & Gates ✅ (Feb 18, 2026)
- [x] `scripts/run_chapter10_sentiment.py` — Walk-forward evaluation runner
- [x] `scripts/evaluate_sentiment_gates.py` — Gate checking + orthogonality
- [x] SMOKE eval: 19,110 rows, 3 folds, 53 seconds
- [x] Gate 3 (Practical): PASS (10% churn)
- [x] Gates 1-2: FAIL standalone (expected for fusion-oriented signal)
- [x] **Orthogonality: ρ < 0.16 vs all signals (HIGH fusion value)**
- [x] 18/18 tests passing

Phase 5: Freeze & Documentation ✅ (Feb 18, 2026)
- [x] All artifacts frozen in `evaluation_outputs/chapter10_sentiment_smoke/`
- [x] `gate_results.json` with full metrics + orthogonality
- [x] `CHAPTER_10.md` complete (all 5 sections)
- [x] ROADMAP updated

**Key Finding:** Sentiment is weak standalone (negative RankIC in SMOKE window) but **highly orthogonal** to every existing signal (ρ < 0.16). This is the textbook use case for NLP sentiment in quant finance: fusion value, not standalone prediction.

**Test Coverage:** 101 Chapter 10 tests (28 + 22 + 33 + 18) all passing

**Status:** ✅ COMPLETE (Feb 18, 2026) — Ready for Chapter 11 Fusion
**Documentation:** `documentation/CHAPTER_10.md`

---

## Chapter 11: Fusion Model ✅ COMPLETE

**Purpose:** Combine LGB, FinText, and Sentiment signals; build expert interface for UQ pipeline

**Completed Deliverables:**
- Score alignment (SMOKE + FULL), fusion architecture (rank-avg, enriched LGB, stacking)
- Residual archive (DuckDB) + AIStockForecasterExpert interface
- 36 tests passing, gate evaluation with full metric profile
- Shadow portfolio for all variants

**Key Result:** LGB baseline wins across all metrics. Fusion Gate 4 FAIL — FinText (Chronos) has near-zero standalone signal (median RankIC 0.014 at 20d, ~0 at 60d). Learned Stacking nearly matches LGB (Ridge learned to discard weak sub-models). Infrastructure value delivered: residual archive, expert interface, disagreement proxy — all ready for Ch13 UQ.

**Final Metrics (FULL, 109 folds — ALL-period; see DEV/FINAL holdout protocol for split):**

| Model | 90d RankIC | IC Stability | Cost Survival | Shadow Sharpe |
|-------|-----------|-------------|--------------|--------------|
| LGB baseline | **0.1833** | **0.7972** | **79.8%** | **1.262** |
| Learned Stacking | 0.1802 | 0.7961 | 79.9% | 1.143 |
| Rank Avg 2 | 0.1173 | 0.6069 | 72.1% | 0.844 |

⚠️ **Holdout note:** 90d RankIC collapses to −0.02 in FINAL (2024+). 20d shadow
Sharpe holds at 1.91 in FINAL. See "DEV / FINAL Holdout Protocol" section.

**Status:** ✅ COMPLETE (Feb 19, 2026) — Documentation: `documentation/CHAPTER_11.md`

---

## Chapter 12: Regime-Aware Analysis & Heuristic Ensemble ✅ COMPLETE

**Purpose:** Understand when LGB fails by regime, build heuristic regime baseline for Ch13 DEUP comparison

**Completed Deliverables:**
- 12.1: Regime-conditional performance diagnostics (18 tests)
- 12.2: Regime stress tests on shadow portfolio (22 tests)
- 12.3: Regime-aware heuristic baselines — vol-sizing + regime blending (24 tests)
- 12.4: Freeze — `regime_context.parquet` (201K rows, 16 features) + documentation (11 tests)

**Key Results:**

| Finding | Detail |
|---------|--------|
| LGB 2.6× better in calm markets | 20d RankIC: 0.182 (low-VIX) vs 0.071 (high-VIX) |
| Bear > bull for stock differentiation | 20d RankIC: 0.110 (bear) vs 0.060 (bull) |
| VIX percentile predicts model error | ρ = −0.21 with 60d RankIC (significant) |
| Vol-sizing improves risk profile | Sharpe 2.65→2.73, max DD −22%→−18% |
| Regime blending fails | All metrics worse — momentum dilutes LGB signal |

**DEUP ablation baseline:** Vol-sized LGB (Sharpe 2.73, max DD −18.1% — ALL-period). Chapter 13 must beat this.

⚠️ **Holdout note:** The above regime diagnostics and heuristic metrics are computed
over the full 109-fold period. The DEV/FINAL holdout analysis (appended to CHAPTER_12.md)
reveals that LGB's signal collapses at 60d/90d in 2024+. All regime findings should be
understood as primarily driven by the DEV period (2016–2023).

**Test Coverage:** 75 tests (18 + 22 + 24 + 11) all passing

**Status:** ✅ COMPLETE (Feb 19, 2026) — Documentation: `documentation/CHAPTER_12.md`

---

## Chapter 13: Calibration & Confidence ⏳ TODO

**Purpose:** Calibrate return distributions and confidence scores

**Planned Deliverables:**
- Quantile calibration
- Confidence stratification
- High-confidence bucket evaluation
 - **Confidence-conditioned Shadow Portfolio slices** (evaluation-only): Sharpe/IR for high-confidence subset vs all signals, coverage/abstain diagnostics

**Target:** Quantile coverage error < 5%, high-confidence outperformance

**Status:** Not started

---

## Chapter 14: Monitoring & Research Ops ⏳ TODO

**Purpose:** Production monitoring and drift detection

**Planned Deliverables:**
- Prediction logging with timestamps
- Matured-label scoring
- Feature/performance drift detection
- Alerts (RankIC decay, calibration breakdown)
 - **Monitoring KPIs (Signal + Shadow Portfolio)**: rolling Sharpe/IR drift, drawdown alarms, turnover/cost-drag spikes (evaluation-only until productionized)

**Status:** Not started

---

## Chapter 15: Outputs & Interfaces ⏳ TODO

**Purpose:** Final output interfaces and traceability

**Planned Deliverables:**
- Ranked stock lists
- Per-stock explanation summaries
- Batch scoring interface
- Full traceability

**Status:** Not started

---

## Chapter 16: Global Research Acceptance Criteria ⏳ TODO

**Purpose:** Final acceptance gate — prove signal isn't repackaged factor exposure

**Planned Deliverables:**
- Fama-French 5-factor regression on shadow portfolio returns (Mkt-RF, SMB, HML, RMW, CMA)
- Alpha intercept significance (t-stat > 2)
- Factor loading documentation
- R² analysis (low = genuinely idiosyncratic alpha)
- Run on LGB baseline + vol-sized shadow portfolio returns
- Report on both DEV and FINAL period returns

**Prerequisite:** Chapters 11-13 complete (need shadow portfolio returns)

**Estimated effort:** ~1 day

**Status:** Not started

---

## Chapter 17: Bayesian UQ Extensions & Model Comparisons ⏳ TODO

**Purpose:** Extend DEUP-based UQ with Bayesian uncertainty estimation on the neural
sub-models (FinText/Chronos, FinBERT) and produce a definitive UQ method comparison.

**Context:** The Kotelevskii & Panov (ICLR 2025) risk decomposition framework offers 9
approximation variants for Bayesian risk estimation, most requiring multiple forward passes
through a neural network. Chapter 13 uses DEUP because the primary model (LightGBM)
doesn't support Bayesian inference. This chapter tests whether Bayesian approaches on the
available NNs add value.

**Planned Deliverables:**
- 17.1 MC Dropout on FinBERT (10-20 passes, sentiment variance as epistemic UQ)
- 17.2 MC Dropout on FinText/Chronos (return forecast distribution)
- 17.3 Bayesian Risk Estimates (R_Bayes aleatoric + R_Exc epistemic via Bregman divergence)
- 17.4 Seed Ensemble as Bregman Information (reframe Ch13 baseline #7 formally)
- 17.5 NGBoost Comparison (optional — natively probabilistic tree model)
- 17.6 Definitive UQ comparison table: DEUP vs Vol-sizing vs EPBD vs Bregman Info vs MC Dropout vs NGBoost

**Key thesis framing:** "We compare three UQ paradigms: (1) DEUP-based excess risk for tree
models, (2) ensemble disagreement as EPBD/Bregman Information approximations, (3) Bayesian
posterior sampling via MC Dropout on neural sub-models. Primary finding may be that DEUP on
a strong base model outperforms Bayesian UQ on weak sub-models — model strength matters more
than UQ sophistication."

**Prerequisite:** Chapter 13 DEUP complete with all diagnostics

**Estimated effort:** ~20-25 hours

**Status:** Not started

---

## Next Actions

**Completed (Feb 16-18) — Chapter 10:**
1. ✅ Sentiment data pipeline: 77,904 records from SEC + FinnHub (28 tests)
2. ✅ FinBERT scoring: MPS-accelerated, all records scored (22 tests)
3. ✅ Feature engineering: 9 PIT-safe sentiment features (33 tests)
4. ✅ Walk-forward evaluation: SMOKE mode + gates + orthogonality (18 tests)
5. ✅ Freeze & documentation: CHAPTER_10.md + ROADMAP.md complete

**Completed (Feb 19) — Chapter 12:**
6. ✅ Regime-conditional performance diagnostics (VIX + market regime slicing)
7. ✅ Shadow portfolio stress tests (corrected non-overlapping monthly Sharpe)
8. ✅ Heuristic baselines (vol-sizing PASS, regime blending FAIL)
9. ✅ regime_context.parquet frozen for Ch13 (201K rows, 100% coverage)

**Chapter 13 Progress (Feb 19):**
10. ✅ 13.0: Residual archive populated (591K LGB + 845K Rank Avg 2 + 845K Stacking rows)
11. ✅ 13.1: g(x) error predictor trained walk-forward (89 folds, ρ=0.19 at 20d, cross_sectional_rank dominates)
12. ✅ 13.2: Aleatoric baseline a(x) — 4 tiers tested, Tier 2 passes at 60d (ρ=0.317), empirical fallback at 20d/90d
13. ✅ 13.3: Epistemic signal ê(x) — perfect quintile monotonicity (ρ=1.0), FINAL > DEV at all horizons, 14/14 sanity checks passed
14. ✅ 13.4: Diagnostics — Disentanglement PASS (ê ≠ vol), baselines dominated 3–10×, stability PASS across all conditions, 98 total tests
15. ✅ 13.4b: Expert health H(t) — per-date regime throttle, G(t)→0 by Apr 2024, 20d crisis detection works (lag ~1 month), 116 total tests
16. ✅ 13.5: Conformal intervals — DEUP-norm reduces conditional coverage spread 25× (0.8% vs 20.2%), narrower intervals, 137 total tests
17. ✅ 13.6: Regime trust gate finalized (AUROC 0.72 / 0.75 FINAL), portfolio sizing evaluated, 154 total tests
18. ✅ 13.7: Deployment policy + ablation — COMPLETE. Structural conflict confirmed (ρ=0.616). Variant 6 (Gate+Vol+ê-Cap) is winner (ALL Sharpe 0.884, FINAL 0.316). Kill criterion K4 triggered for inverse sizing; ê-cap guard validated. 28 new tests.
19. ✅ 13.8: Multi-crisis G(t) diagnostic — COMPLETE. G(t) validated across 5 crisis + 3 calm windows (7/8 correct verdicts vs 5/8 for VIX gate). Critical finding: VIX produces 3 false alarms on calm periods where model IC > 0.10; G(t) correctly stays active. 2023 H2 is decisive distinguishing episode (IC=+0.034 at 20d; VIX=94.3%).
20. ⏳ 13.9: Freeze, final documentation, git tag — TODO
Chapter 13 overall: **95% complete**

**Chapter 13 outcome so far:**
- ✅ **PASS:** Regime trust gate works (AUROC 0.72, monotonic buckets, FINAL > DEV).
- ✅ **PASS:** G(t) validates across ALL stress episodes 2016–2025 (7/8 correct; VIX 5/8). Critical finding: G(t) stays active when model works despite elevated VIX (2019, 2023).
- ✅ **PASS:** ê-Cap tail-risk guard adds incremental value on top of vol-sizing (Variant 6 ALL Sharpe 0.884 vs Gate+Vol 0.817).
- ⚠️ **FAIL (honest, K4 triggered):** Per-stock inverse DEUP sizing does not beat vol sizing (ρ(ê, |score|) = 0.616 structural conflict confirmed).

**Notes on Ch13 Priority:**
- Vol-sized LGB is the ablation baseline (Sharpe 2.73, max DD −18.1%)
- VIX percentile is the strongest epistemic predictor (ρ = −0.21 with RankIC)
- Per-stock vol_20d available in regime_context.parquet for aleatoric baseline
- Risk attribution (Fama-French 5-factor) deferred to Chapter 16 acceptance gate
- **DEV/FINAL holdout protocol established:** All chapters must report both DEV (pre-2024)
  and FINAL (2024+) metrics. Signal collapses at 60d/90d in holdout; 20d is confirmed
  (FINAL Sharpe 1.91). See "DEV / FINAL Holdout Protocol" section above.
- **DEUP must detect the 2024 regime failure:** The holdout collapse is exactly the
  scenario epistemic uncertainty should flag and abstain from.

---

## DEV / FINAL Holdout Protocol

**Established:** February 19, 2026 (retroactive split)

### Definition

All evaluation is partitioned into two non-overlapping windows:

| Window | Date Range | Months | Purpose |
|--------|-----------|:------:|---------|
| **DEV** | Feb 2016 – Dec 2023 | 95 | Research iteration (walk-forward folds visible during development) |
| **FINAL** | Jan 2024 – Feb 2025 | 14 | One-shot confirmation (never optimized against) |

**Cutoff:** `HOLDOUT_START = 2024-01-01`

**Embargo clearance:** Last DEV fold's training window reaches ~Sep 2023
(90 trading day embargo). First FINAL evaluation date is Jan 2024. No
leakage from DEV training into FINAL labels.

### Caveat: Retroactive (soft) holdout

This split was established after 109-fold aggregate metrics were already
examined during Chapters 7–12 development. It is therefore a **soft holdout**,
not a true blind holdout. We never specifically optimized for 2024+ performance,
but researcher degrees of freedom (model choice, features, hyperparameters)
were informed by the full-period aggregate — which includes the holdout months.

A true blind holdout would require freezing all decisions before evaluating
the FINAL window. This protocol approximates that by committing to report
DEV and FINAL separately going forward, and iterating only on DEV metrics.

### Key findings (LGB baseline)

**Signal metrics collapse at longer horizons in the holdout:**

| Horizon | DEV Mean RankIC | FINAL Mean RankIC | Change |
|---------|:--------------:|:----------------:|:------:|
| 20d | 0.072 | 0.010 | −86% |
| 60d | 0.160 | −0.005 | flips negative |
| 90d | 0.192 | −0.021 | flips negative |

**Shadow portfolio (20d) degrades but remains strongly positive:**

| Split | Sharpe | Ann. Return | Max DD | Hit Rate |
|-------|:------:|:----------:|:------:|:--------:|
| DEV (95 mo) | 3.15 | 81.9% | −21.9% | 82.1% |
| FINAL (14 mo) | 1.91 | 119.1% | −16.6% | 71.4% |

**Year-by-year 90d RankIC reveals regime dependency, not pure overfitting:**

| Year | 90d RankIC | Interpretation |
|------|:---------:|----------------|
| 2016 | 0.405 | Very high — limited training data |
| 2017–2020 | 0.18–0.32 | Genuine strong signal across multiple years |
| **2021** | **−0.071** | Failure — meme stock / tech mania |
| 2022–2023 | 0.15–0.17 | Recovery, but weaker |
| **2024** | **−0.006** | Signal collapses — AI thematic rally |
| **2025** | **−0.139** | Actively wrong (32 days, insufficient sample) |

**Interpretation:** The model has real signal in normal and bear markets
(2017–2020, 2022–2023) but fails in strong thematic bull regimes (2021
meme stocks, 2024–2025 AI rally) where cross-sectional dispersion collapses.
This is regime dependency, not data leakage.

### All models show the same pattern

| Model | 20d FINAL | 60d FINAL | 90d FINAL |
|-------|:---------:|:---------:|:---------:|
| LGB baseline | 0.010 | −0.005 | −0.021 |
| Rank Avg 2 | **0.031** | **0.018** | −0.009 |
| Learned Stacking | 0.009 | −0.008 | −0.022 |

Rank Avg 2 (which includes FinText) holds up best in the holdout at 20d and
60d — the fusion with a zero-shot model provides some diversification value
during regime shifts.

### Overfitting vs Regime Shift Diagnostics (Feb 19, 2026)

Three diagnostics run to separate overfitting from regime shift:

1. **Retrain LGB on DEV-only → evaluate on FINAL:** Retrained model
   performs WORSE (90d: −0.075) than walk-forward (−0.021). **Verdict:
   regime shift, not leakage.** Walk-forward fold diversity actually helped.
2. **Feature importance stability:** Rank correlation 0.95+ across all
   time windows, same top-3 features (adv_20d, vol_60d, mom_12m) in every
   period. **Verdict: model learned real, stable patterns, not noise.**
3. **20d deep-dive on 2024:** 8/14 months positive at 20d vs only 4/14
   at 90d. Signal oscillates with regime (strong in Jan–Feb, Nov–Dec;
   negative in Mar–May, Jul). **Verdict: selective horizon failure confirms
   regime dependency, not blanket overfitting.**

**Combined conclusion:** The model learned genuine, economically interpretable
patterns (momentum, volume, volatility) but those patterns broke during the
2024 AI stock rally when cross-sectional dispersion collapsed. The "true"
RankIC is probably 0.04–0.10, not the headline 0.18.

Script: `scripts/run_holdout_diagnostics.py`
Output: `evaluation_outputs/chapter12/holdout_diagnostics/holdout_diagnostics.json`

### Rules going forward

1. **All chapters report both DEV and FINAL metrics** side by side
2. **Model iteration uses DEV only** — FINAL is evaluated once per chapter
3. **Chapter 16 factor regression** uses FINAL period shadow returns
4. **Failure threshold:** If FINAL shadow Sharpe drops below 1.0, the model
   needs fundamental changes before proceeding
5. **20d is the primary confirmed horizon** — longer horizons are promising
   but unconfirmed in the holdout

### Implications for the project

1. **UQ is even more critical:** DEUP (Ch13) must detect these failure regimes
   and reduce confidence / abstain. The 2024 collapse is exactly the scenario
   epistemic uncertainty should flag.
2. **20d is the most robust horizon:** Confirmed positive signal + Sharpe 1.91
   in holdout. 60d/90d carry regime risk.
3. **Factor regression (Ch16) becomes essential:** Need to confirm that the
   surviving 20d alpha isn't just repackaged momentum.
4. **The headline RankIC 0.18 at 90d is optimistically biased** by early-period
   high values and the absence of regime-failure years from the effective
   development narrative.
5. **This finding strengthens the thesis:** "The model needs uncertainty
   quantification" is exactly what the holdout degradation proves.

---

## Global Acceptance Criteria

A model is considered **valid** if:
- ✅ Median walk-forward RankIC exceeds best baseline by ≥ 0.02
- ✅ Net-of-cost performance positive in ≥ 70% of folds
- ✅ Top-10 ranking churn < 30% month-over-month
- ✅ Performance degrades gracefully under regime shifts
- ✅ No PIT or survivorship violations detected

**Institutional-grade add-on (evaluation-only, from Chapter 11+):**
- Shadow portfolio sanity: Sharpe/IR meaningfully > 0 and not driven by 1–2 months
  - **Robustness check:** Report rolling 12-month Sharpe/IR (or 36-month if available). Must not rely on a single regime.
  - This is evaluation-only reporting (no optimization target).

**Risk attribution gate (Chapter 16):**
- Fama-French 5-factor regression on shadow portfolio monthly returns
- Alpha intercept must be positive and significant (t-stat > 2)
- Factor loadings documented (momentum exposure acceptable if alpha survives)
- R² documented (low = genuinely idiosyncratic alpha)
- Existing `src/features/neutralization.py` covers feature-level attribution;
  Ch16 adds portfolio-level return attribution as complementary proof

---

## Repository Structure

```
AI_Stock_Forecast/
├── src/                           # Source code
│   ├── data/                      # Data infrastructure ✅
│   │   ├── prices_store.py       # NEW: Chapter 8 OHLCV store
│   │   └── trading_calendar.py   # EXTENDED: Global calendar
│   ├── features/                  # Feature engineering ✅
│   ├── evaluation/                # Frozen evaluation pipeline 🔒
│   ├── models/                    # Model implementations 🟡
│   │   ├── kronos_adapter.py     # Chapter 8: Kronos adapter
│   │   ├── fintext_adapter.py    # Chapter 9: FinText adapter
│   │   └── finbert_scorer.py    # Chapter 10: FinBERT scorer
│   ├── uncertainty/               # UQ components (Ch13) ⏳
│   │   ├── __init__.py
│   │   ├── deup_estimator.py    # g(x) error predictor (13.1)
│   │   └── aleatoric_baseline.py # a(x) aleatoric noise (13.2)
│   └── ...
├── scripts/                       # Build & evaluation scripts
│   ├── build_features_duckdb.py  # ✅ Feature store builder
│   ├── add_prices_table_to_duckdb.py  # NEW: Add prices table
│   └── ...
├── tests/                         # 755+ passing tests ✅
│   ├── test_prices_store.py      # Chapter 8: 18 tests
│   ├── test_trading_calendar_kronos.py  # Chapter 8: 16 tests
│   ├── test_excess_return_store.py     # Chapter 9: 29 tests
│   ├── test_fintext_adapter.py         # Chapter 9: 41 tests
│   ├── test_chapter9_evaluation.py     # Chapter 9: 7 tests
│   ├── test_fintext_gates_tripwires.py # Chapter 9: 18 tests
│   ├── test_ablation_framework.py      # Chapter 9: 17 tests
│   ├── test_sentiment_store.py        # Chapter 10: 28 tests
│   ├── test_finbert_scorer.py         # Chapter 10: 22 tests
│   ├── test_sentiment_features.py     # Chapter 10: 33 tests
│   └── test_chapter10_evaluation.py   # Chapter 10: 18 tests
├── data/                          # Data storage (gitignored)
│   └── features.duckdb            # Main feature store
├── evaluation_outputs/            # Evaluation artifacts
│   ├── chapter6_closure_real/    # 🔒 Frozen factor baseline
│   └── chapter7_tabular_lgb_full/  # 🔒 Frozen ML baseline
└── documentation/                 # Project documentation
    ├── ROADMAP.md                # This file
    ├── CHAPTER_8_*.md            # Chapter 8 docs
    └── ...
```

---

## Key Milestones

- ✅ **Dec 30, 2025:** Chapter 6 evaluation framework frozen
- ✅ **Jan 5, 2026:** Chapter 7 ML baseline frozen
- ✅ **Jan 9, 2026:** Chapter 8 Kronos complete (negative result documented)
- ✅ **Feb 17, 2026:** Chapter 9 FinText-TSFM COMPLETE (all sections 9.0–9.11, 545 tests, all gates pass)
- ✅ **Feb 18, 2026:** Chapter 10 NLP Sentiment COMPLETE (5 phases, 101 tests, orthogonality confirmed)
- ✅ **Feb 19, 2026:** Chapter 11 Fusion COMPLETE (LGB wins; fusion infrastructure + expert interface delivered)
- ✅ **Feb 19, 2026:** Chapter 12 Regime Analysis COMPLETE (75 tests, vol-sized heuristic baseline frozen)
- ⏳ **Feb 20, 2026 (target):** Chapter 13 Calibration & DEUP
- ⏳ **TBD:** Chapter 16 Acceptance Criteria & Factor Attribution
- ⏳ **TBD:** Chapter 17 Bayesian UQ Extensions & Model Comparisons

---

**END OF ROADMAP**

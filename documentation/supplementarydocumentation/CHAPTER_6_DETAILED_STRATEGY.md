# Chapter 6: Detailed Evaluation Strategy

**Status:** ✅ LOCKED AND READY FOR IMPLEMENTATION

**Last Updated:** December 29, 2025

---

## Table of Contents

1. [Pre-Implementation Sanity Checks](#1-pre-implementation-sanity-checks)
2. [Evaluation Parameters (LOCKED)](#2-evaluation-parameters-locked)
3. [Baselines (Models to Beat)](#3-baselines-models-to-beat)
4. [Top-K Metrics](#4-top-k-metrics)
5. [Acceptance Criteria](#5-acceptance-criteria)
6. [Implementation Checklist](#6-implementation-checklist)

---

## 1. Pre-Implementation Sanity Checks

> **CRITICAL**: These are not blockers—they are sanity locks. Complete BOTH before writing any Chapter 6 code.

### ✅ Sanity Check 1: Manual IC vs Qlib IC Parity Test

**Purpose:** Ensure adapter/indexing is correct before generating hundreds of evaluation reports.

**Test Protocol:**

```python
# One fold, one horizon, same predictions
fold = "2023-Q1"
horizon = 20

# Manual RankIC calculation
manual_rankic = df.groupby("date").apply(
    lambda x: spearmanr(x["prediction"], x["label"])[0]
).median()

# Qlib RankIC calculation
qlib_df = adapter.to_qlib_format(predictions, labels)
qlib_rankic = qlib.evaluate(qlib_df)["IC"].median()

# STOP if they don't match
assert abs(manual_rankic - qlib_rankic) < 0.001, "IC mismatch - fix adapter!"
```

**Acceptance:** Manual and Qlib RankIC must agree to **3 decimal places**.

**If they don't match → STOP immediately:**
- ❌ Check MultiIndex formatting (`datetime`, `instrument`)
- ❌ Check date alignment (T vs T+H)
- ❌ Check for missing data handling differences
- ❌ Check for sign flips (prediction vs label)

**Why this matters:**
- Prevents silent bugs that compound over hundreds of experiments
- Validates adapter logic before it becomes "too expensive to fix"
- Ensures Qlib reports are trustworthy

---

### ✅ Sanity Check 2: Experiment Naming Convention

**Purpose:** Prevent chaos when Recorder usage explodes across hundreds of experiments.

**Convention (LOCK THIS IN NOW):**

```
exp = ai_forecaster/
      horizon={20,60,90}/
      model={kronos_v0, fintext_v0, tabular_lgb, baseline_mom12m, baseline_momcomp, baseline_shortterm}/
      labels={v1_priceonly, v2_totalreturn}/
      fold={01, 02, ..., 40}/
```

**Example paths:**
```
ai_forecaster/horizon=20/model=kronos_v0/labels=v2/fold=03
ai_forecaster/horizon=60/model=baseline_mom12m/labels=v2/fold=12
ai_forecaster/horizon=90/model=tabular_lgb/labels=v2/fold=25
```

**Why this matters:**
- ✅ Enables bulk queries: "all Kronos results for horizon=20"
- ✅ Prevents accidental overwrites
- ✅ Makes debugging obvious (path tells you what broke)
- ✅ Enables automated comparison scripts
- ✅ Future-proof for ablation studies

**Implementation:**

```python
from qlib.workflow import R

exp_name = f"ai_forecaster/horizon={horizon}/model={model_name}/labels={label_version}/fold={fold_id:02d}"
with R.start(experiment_name=exp_name):
    recorder.save_objects(predictions=pred_df)
```

---

## 2. Evaluation Parameters (LOCKED)

### Rebalance Cadence

| Frequency | Purpose | Details |
|-----------|---------|---------|
| **Monthly (Primary)** | Main evaluation | First trading day of month, ~110 points (2016-2025), natural match to 20d horizon |
| **Quarterly (Secondary)** | Robustness check | Supplementary slice only, confirms not regime-specific |

**Rationale:**
- **20d horizon + monthly = one horizon per rebalance** (natural)
- **60/90d + monthly = overlapping positions** (realistic for evaluation, not execution sim)
- Quarterly gives too few points (<40) → noisier inference, easier to fool yourself
- Monthly balances statistical power with practical institutional constraints

**Rebalance date:** First trading day of each month (NYSE calendar)

### Evaluation Date Range

```python
EVAL_START = "2016-01-01"  # Earliest reliable fundamentals + universe snapshots
EVAL_END = "2025-06-30"    # Conservative: guarantees 90d label maturity
```

**Why these dates?**

| Date | Reason |
|------|--------|
| **2016-01-01** | FMP fundamentals become reliable, universe coverage sufficient |
| **2025-06-30** | All 90d labels mature (PIT-safe), includes 2023-25 AI rally |

**Regime Diversity (critical for robust evaluation):**
- Pre-COVID (2016-2019): Steady bull market
- COVID (2020): Extreme volatility
- 2021-2022: Drawdown and rate hikes
- 2023-2025: AI mania and recovery

**Result:** ~110 monthly rebalance points, multiple regimes, zero PIT violations.

### Walk-Forward Window

**Type:** Expanding (not rolling)
- Grows forward, never shrinks
- Preserves long-term signal stability
- Avoids artificial lookback dependency

**Embargo:** 90 trading days (max horizon)
- Conservative cushion for all horizons (20/60/90)
- Prevents label leakage across folds

**Purging:** Horizon-aware
- Remove overlapping labels between train/validation
- Prevents correlated label windows from leaking information

### Time-Decay Weighting (Training Only)

```python
# From src/features/time_decay.py
half_lives = {
    20: 2.5 * 252,  # 2.5 years for 20-day horizon
    60: 3.5 * 252,  # 3.5 years for 60-day horizon
    90: 4.5 * 252,  # 4.5 years for 90-day horizon
}
```

- Per-date normalization for cross-sectional ranking
- More recent data weighted higher, but not exclusively
- Balances recency with stability

---

## 3. Baselines (Models to Beat)

**3 baselines to beat:**

| Baseline | Feature(s) | Purpose | Expected IC |
|----------|-----------|---------|-------------|
| **A: `mom_12m`** | 12-month momentum | Primary naive baseline | ~0.02-0.03 |
| **B: `momentum_composite`** | `(mom_1m + mom_3m + mom_6m + mom_12m) / 4` | Stronger but transparent | ~0.03-0.04 |
| **C: `short_term_strength`** | `mom_1m` or `rel_strength_1m` | Diagnostic for horizon sensitivity | Varies by regime |

**+ 1 sanity baseline (not a target):**
- **`naive_random`**: Deterministic random scores (sanity check: RankIC ≈ 0)

### Why These 3 (+ 1 Sanity)?

**Baseline A (`mom_12m`):**
- ✅ Standard factor baseline (embarrassing if we can't beat)
- ✅ Uses existing feature from `src/features/price_features.py`
- ✅ Widely known benchmark in industry
- ✅ **Gate:** If Kronos/FinText can't beat this, something is fundamentally wrong

**Baseline B (`momentum_composite`):**
- ✅ Often materially stronger than single-horizon momentum
- ✅ Still no ML, still transparent
- ✅ Realistic bar for "is ML worth the complexity?"
- ✅ Tests whether multi-horizon fusion adds value

**Baseline C (`short_term_strength`):**
- ✅ Diagnostic for horizon/regime sensitivity
- ✅ Often weak in mean-reversion regimes (good thing—tells you something)
- ✅ If it beats you in certain windows → evaluation is revealing regime dependency
- ✅ If you can't beat it → may be over-penalizing longer-horizon signals

### Critical Guardrail

**All baselines run through IDENTICAL pipeline:**

| Component | Requirement |
|-----------|-------------|
| Universe | Same `stable_id` snapshots |
| Missingness | Same handling (no special treatment) |
| Neutralization | Same setting (raw or sector/beta-neutral) |
| Cost diagnostic | Same 20 bps + ADV-scaled slippage |
| Purging/Embargo | Same 90-day embargo |
| Walk-forward splits | Same expanding window |

**Why this matters:**
- Prevents accidentally giving your model special treatment
- Ensures comparisons are apples-to-apples
- Makes baseline exceedance meaningful

**No Baseline Shopping:**
- ❌ Don't add more baselines to find weak ones
- ❌ Don't remove baselines that beat you
- ❌ Don't cherry-pick good time periods for baseline comparison

---

## 4. Top-K Metrics

### Top-K Size

| Metric | Primary | Secondary | Purpose |
|--------|---------|-----------|---------|
| **Top-K size** | Top-10 | Top-20 | Fixed K, matches "shortlist" product narrative |

**Why Top-10 as primary?**
- ✅ Matches forecaster product narrative (shortlist of best stocks)
- ✅ Churn is interpretable ("did the list change?")
- ✅ Stable enough for month-to-month comparison
- ✅ Aligns with concentrated portfolio construction (realistic)

**Why Top-20 as secondary?**
- ✅ Robustness check for universe size changes
- ✅ Simpler than percentage-based (no dynamic computation)
- ✅ Still interpretable

**Alternative:** Top-10% (percentage-based) if universe size varies significantly (e.g., 80-150 stocks)

### Churn (Turnover)

**Definition:** Overlap-based metric between consecutive Top-K lists

**Formula:**

```python
# Jaccard similarity between consecutive Top-K
churn = 1 - len(set(top_k_t) & set(top_k_t_minus_1)) / len(set(top_k_t) | set(top_k_t_minus_1))

# Or simpler: % retained
retention = len(set(top_k_t) & set(top_k_t_minus_1)) / K
churn = 1 - retention
```

**Target:** < 30% month-over-month

**Why this matters:**
- High churn (>50%) → unstable, costly to execute
- Low churn (<10%) → may be stale, not adapting to regime shifts
- Sweet spot (20-30%) → stable enough to execute, dynamic enough to adapt

### Hit Rate

**Definition:** % of Top-K portfolios with excess return > 0 over horizon

**Formula:**

```python
# For each rebalance period t:
top_k_return = equal_weight_portfolio(top_k_stocks[t], horizon)
benchmark_return = benchmark_return_over_horizon[t]
hit = (top_k_return - benchmark_return) > 0

# Aggregate:
hit_rate = sum(hits) / total_rebalance_periods
```

**Target:** > 55%

**Why this matters:**
- 50% = coin flip (no skill)
- 55% = modest but meaningful edge
- 60%+ = very strong (don't expect this early)

**Alternative definition (if preferred):**
- Top-K beats equal-weight universe (instead of benchmark)
- Use if interpretability is more important than absolute performance

---

## 5. Acceptance Criteria

**Chapter 6 is COMPLETE when ALL of the following are true:**

### Quantitative Gates

| Criterion | Target | Purpose |
|-----------|--------|---------|
| **Median walk-forward RankIC** | > baseline + 0.02 | Meaningful improvement over baselines |
| **Net-of-cost positive** | % positive folds >= baseline + 10pp (relative) | Alpha survives better than frozen floor (5.8%-40.1%) |
| **Top-10 churn** | < 30% month-over-month | Exploitable without excessive turnover |
| **Regime robustness** | Graceful degradation | No catastrophic failures in any regime |
| **PIT violations** | 0 CRITICAL, 0 HIGH | Scanner enforced, no exceptions |

### Qualitative Guardrails

**What you CANNOT do during Chapter 6:**
- ❌ NO new features mid-evaluation (use what you have)
- ❌ NO retraining models to "fix" bad folds (accept results as-is)
- ❌ NO cherry-picking good time periods (report all folds)
- ❌ NO optimizing to costs (diagnostic only, not objective)
- ❌ NO hiding negative results (report failures too)

**Success = Boring Results That Don't Break**
- ✅ Median IC of 0.03-0.05 is **GOOD**
- ✅ Stable across regimes is **EXCELLENT**
- ✅ Survives costs is **SUFFICIENT**

**If results are disappointing:**
1. Document honestly (don't hide)
2. Diagnose (use 5.8 neutralization, 5.7 IC stability, regime slices)
3. **DO NOT** tweak features/models yet
4. Move to Chapter 7 with lessons learned

---

## 6. Implementation Checklist

### Phase 0: Pre-Implementation Validation (FIRST)

- [ ] **Sanity Check 1:** Manual IC vs Qlib IC parity test
  - [ ] One fold, one horizon, same predictions
  - [ ] Manual RankIC calculation
  - [ ] Qlib RankIC calculation
  - [ ] Assert agreement to 3 decimal places
  - [ ] Document any adapter fixes needed

- [ ] **Sanity Check 2:** Experiment naming convention
  - [ ] Define naming template
  - [ ] Test with dummy experiments
  - [ ] Verify bulk query capability
  - [ ] Document convention in code comments

### Phase 1: Infrastructure Setup

- [ ] Install Qlib: `pip install pyqlib==0.9.7`
- [ ] Create `src/evaluation/` directory
- [ ] Implement `src/evaluation/qlib_adapter.py`
  - [ ] `to_qlib_format()` function
  - [ ] `from_qlib_results()` function
  - [ ] Unit tests for adapter

- [ ] Implement `src/evaluation/walk_forward.py`
  - [ ] Expanding window generator
  - [ ] Purging logic (horizon-aware)
  - [ ] Embargo enforcement (90 days)
  - [ ] Universe snapshot integration

- [ ] Implement `src/evaluation/baselines.py`
  - [ ] `mom_12m` baseline
  - [ ] `momentum_composite` baseline
  - [ ] `short_term_strength` baseline
  - [ ] Ensure identical pipeline for all

### Phase 2: Metrics Implementation

- [ ] Implement `src/evaluation/metrics.py`
  - [ ] Top-K churn calculation (Jaccard/retention)
  - [ ] Hit rate calculation (excess return > 0)
  - [ ] Manual IC calculation (for parity test)
  - [ ] Regime slicing (VIX, bull/bear, sector)

- [ ] Implement `src/evaluation/reporting.py`
  - [ ] IC by fold report
  - [ ] IC by regime report
  - [ ] Top-K churn time series
  - [ ] Hit rate time series
  - [ ] Baseline comparison tables

### Phase 3: End-to-End Execution

- [ ] Run Phase 0 sanity checks (MANDATORY)
- [ ] Run walk-forward evaluation (2016-2025, monthly)
  - [ ] Baseline A (`mom_12m`)
  - [ ] Baseline B (`momentum_composite`)
  - [ ] Baseline C (`short_term_strength`)
  - [ ] [Future: Kronos, FinText, Tabular ML]

- [ ] Generate reports
  - [ ] Median IC by model/horizon
  - [ ] IC by regime (VIX, bull/bear)
  - [ ] Top-K churn and hit rate
  - [ ] Net-of-cost performance (20 bps)

- [ ] Validate acceptance criteria
  - [ ] Check median IC > baseline + 0.02
  - [ ] Check net-of-cost: % positive folds >= baseline + 10pp (relative threshold)
  - [ ] Check Top-10 churn < 30%
  - [ ] Check graceful regime degradation
  - [ ] Check PIT scanner (0 CRITICAL/HIGH)

### Phase 4: Documentation & Freeze

- [ ] Document evaluation results
  - [ ] Summary statistics table
  - [ ] Best/worst fold analysis
  - [ ] Regime conditional performance
  - [ ] Lessons learned

- [ ] Freeze Chapter 6 artifacts
  - [ ] Evaluation parameters (already locked)
  - [ ] Qlib adapter version
  - [ ] Walk-forward split dates
  - [ ] Git commit hash

- [ ] Update documentation
  - [ ] `PROJECT_DOCUMENTATION.md` (mark Chapter 6 complete)
  - [ ] `AI_Stock_Forecaster` notebook (results summary)
  - [ ] `CHAPTER_6_RESULTS.md` (detailed findings)

---

## Summary

**Chapter 6 is now LOCKED and ready for implementation.**

**Key Principles:**
1. **Sanity checks first** (IC parity, naming convention)
2. **Parameters locked** (no tweaking mid-evaluation)
3. **Baselines identical** (no special treatment)
4. **Results honest** (report failures too)
5. **Conservative acceptance** (boring is good)

**Expected Timeline:**
- Phase 0: 1 day (sanity checks)
- Phase 1: 2-3 days (infrastructure)
- Phase 2: 2-3 days (metrics)
- Phase 3: 3-5 days (execution + debugging)
- Phase 4: 1-2 days (documentation)

**Total:** ~2 weeks for complete Chapter 6 implementation and evaluation

**Next Steps:**
1. Complete Phase 0 sanity checks
2. Begin infrastructure implementation (Phase 1)
3. Do NOT proceed to Phase 3 until sanity checks pass

---

**Last Updated:** December 29, 2025  
**Status:** ✅ LOCKED — Ready for implementation  
**Frozen Parameters:** Rebalance, date range, baselines, Top-K, acceptance criteria


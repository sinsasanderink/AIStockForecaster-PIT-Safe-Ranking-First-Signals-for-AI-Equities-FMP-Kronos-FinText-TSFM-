# Chapter 6 Phase 4: Cost Realism — COMPLETE ✅

**Status:** Phase 0, 1, 1.5, 2, 4 COMPLETE  
**Date:** December 29, 2025  
**Total Tests:** 139/139 passing (123 in evaluation suite)

---

## Executive Summary

Phase 4 (Cost Realism) implements a diagnostic cost overlay to answer **"does alpha survive trading costs?"** using sensitivity bands instead of pretending we know exact slippage. This is the "right kind of boring" — minimal assumptions, maximum transparency.

### Key Principle

**Costs affect portfolio metrics (Top-K, spreads), NOT RankIC.**

- RankIC measures ranking skill
- Costs measure implementability
- Separation of concerns prevents contamination

---

## Implementation Summary

### Files Created

**`src/evaluation/costs.py`** (514 lines, 28 tests)

Core functionality:
- `TRADING_ASSUMPTIONS`: Frozen dataclass with locked trading parameters
- `compute_participation_rate()`: Trade value / ADV with 100% cap
- `compute_slippage_bps()`: Square-root impact model (standard in microstructure)
- `compute_trade_cost()`: Base cost + slippage + ADV missing penalty
- `compute_turnover()`: Weight changes using stable_id
- `compute_portfolio_costs()`: Full portfolio cost for one rebalance
- `compute_net_metrics()`: Net-of-cost overlay (gross ER - costs)
- `run_cost_sensitivity()`: Sensitivity analysis (low/base/high scenarios)
- Validation helpers for monotonicity and determinism

**`tests/test_costs.py`** (471 lines, 28 tests)

Test coverage:
- Frozen trading assumptions (immutability)
- Square-root impact law verification
- Monotonicity invariants (ADV, AUM)
- Turnover calculation (zero/full/partial)
- Portfolio cost computation
- Net metrics computation
- Determinism enforcement
- Edge cases (missing ADV, very small ADV)

**Updated Files:**
- `src/evaluation/__init__.py`: Added cost function exports
- `PROJECT_DOCUMENTATION.md`: Added Section 6.4 Cost Realism
- `PROJECT_STRUCTURE.md`: Updated Phase 4 status and implementation details

---

## Locked Trading Assumptions

All parameters are frozen in `TRADING_ASSUMPTIONS` dataclass:

### Portfolio Definition
- **Long-only Top-K equal-weight**
- K = 10 (primary), 20 (secondary)
- No shorts, no leverage, no volatility targeting

### Rebalance Timing
- On as-of close (first trading day of month/quarter)
- Consistent with label entry price convention

### Turnover Definition
- Weight changes between consecutive rebalances
- Uses `stable_id` (not ticker) to avoid rename noise
- Does NOT cross fold boundaries

### AUM Assumptions
- $1M (primary diagnostic)
- $10M (secondary diagnostic)
- Fixed notionals give meaning to ADV scaling

### Cost Model

**Base Cost:**
- 20 bps round-trip (10 bps entry + 10 bps exit)
- Always applied to liquid large caps

**Slippage (Square-Root Impact):**
```python
slippage_bps = c * sqrt(participation_rate)
where:
  participation_rate = trade_value / adv_dollars
  c = {5 (low), 10 (base), 20 (high)}
```

**Properties:**
- Monotone (higher participation → higher cost)
- Concave (diminishing marginal impact)
- Empirically validated in literature
- Simple and transparent

**Slippage Caps:**
- Floor: 0 bps (no negative slippage)
- Cap: 500 bps (5% max, prevents absurd tails)

**ADV Missing:**
- 100 bps penalty (conservative)
- Prevents silent optimism bias

---

## Test Results (28/28 Passing)

### Test Categories

**1. Trading Assumptions (5 tests)**
- ✅ Frozen dataclass (immutable)
- ✅ K values (10, 20)
- ✅ Base cost (20 bps round-trip)
- ✅ Slippage coefficients ordered (low < base < high)
- ✅ AUM values ($1M, $10M)

**2. Cost Model (7 tests)**
- ✅ Participation rate calculation (normal, high, missing ADV)
- ✅ Slippage follows square-root law (4x participation → 2x slippage)
- ✅ Floor and cap enforcement
- ✅ Trade cost breakdown (base + slippage)
- ✅ Missing ADV penalty applied

**3. Monotonicity Invariants (3 tests)**
- ✅ Lower ADV → higher cost
- ✅ Higher AUM → higher cost
- ✅ Zero trade → minimal cost (base only)

**4. Turnover (4 tests)**
- ✅ Identical portfolios → 0% turnover
- ✅ Complete replacement → 100% turnover
- ✅ Partial replacement → intermediate turnover
- ✅ Rebalancing weights → turnover from weight changes

**5. Portfolio Costs (3 tests)**
- ✅ Single date portfolio cost computation
- ✅ Insufficient stocks handling
- ✅ Zero turnover → minimal costs

**6. Net Metrics (2 tests)**
- ✅ Net average excess return = gross - costs
- ✅ Alpha survives flag (net > 0)

**7. Determinism (2 tests)**
- ✅ Same inputs → same outputs
- ✅ Portfolio costs deterministic

**8. Edge Cases (2 tests)**
- ✅ All missing ADV → penalty applied
- ✅ Very small ADV → high costs (high participation)

---

## Usage Examples

### Basic Cost Computation

```python
from src.evaluation import compute_trade_cost

# Compute cost for one trade
cost = compute_trade_cost(
    trade_value=100_000,  # $100K trade
    adv_dollars=1_000_000,  # $1M ADV
    slippage_coef=10.0  # Base scenario
)

# Result
cost["base_cost_bps"]  # 20 bps (always)
cost["slippage_bps"]   # ~3.2 bps (10 * sqrt(0.1))
cost["total_cost_bps"]  # ~23.2 bps
cost["participation_rate"]  # 0.1 (10% of ADV)
```

### Portfolio-Level Costs

```python
from src.evaluation import compute_portfolio_costs

# Compute costs for Top-10 portfolio at one rebalance
costs = compute_portfolio_costs(
    df=eval_df,  # EvaluationRow format
    k=10,
    aum=1_000_000,
    slippage_coef=10.0,  # Base scenario
    prev_portfolio=None  # First rebalance
)

# Results
costs["turnover"]  # 1.0 (full deployment)
costs["total_cost_pct"]  # 0.0023 (23 bps)
costs["avg_cost_bps"]  # 23 bps (average across stocks)
costs["n_missing_adv"]  # Count of stocks with missing ADV
costs["portfolio_weights"]  # Series indexed by stable_id
```

### Net-of-Cost Metrics

```python
from src.evaluation import compute_net_metrics

# Apply costs to gross metrics
gross_metrics = {
    "avg_excess_return": 0.05  # 5% gross
}

net = compute_net_metrics(
    gross_metrics=gross_metrics,
    cost_pct=0.002  # 20 bps = 0.2%
)

# Results
net["gross_avg_er"]  # 0.05 (5%)
net["cost_pct"]  # 0.002 (0.2%)
net["net_avg_er"]  # 0.048 (4.8%)
net["alpha_survives"]  # True (net > 0)
```

### Sensitivity Analysis

```python
from src.evaluation import run_cost_sensitivity

# Run all scenarios for one fold/horizon
sensitivity = run_cost_sensitivity(
    eval_df=eval_df,
    fold_id="fold_01",
    horizon=20,
    k=10,
    aum=1_000_000
)

# Results: DataFrame with scenarios
#   scenario       | slippage_coef | base_cost_bps | k  | aum
#   base_only      | 0             | 20            | 10 | 1M
#   low_slippage   | 5             | 20            | 10 | 1M
#   base_slippage  | 10            | 20            | 10 | 1M
#   high_slippage  | 20            | 20            | 10 | 1M
```

---

## Sensitivity Band Philosophy

**Why not one "true" slippage value?**

1. **Market Impact Varies:**
   - By liquidity regime
   - By volatility regime
   - By execution skill
   - By market structure

2. **Sensitivity Bands Answer the Right Question:**
   - Low scenario: "Best case with skilled execution"
   - Base scenario: "Realistic expectation"
   - High scenario: "Conservative / stressed case"

3. **Decision Framework:**
   - If alpha survives in **high** scenario → robust
   - If alpha dies in **base** scenario → reject signal
   - If alpha survives in **base** but dies in **high** → proceed with caution

4. **Transparency:**
   - No pretending we know exact slippage
   - Show where strategy breaks
   - Enables informed risk-taking

---

## Integration with Existing Metrics

### RankIC (Unchanged)
- Measures ranking skill
- Does NOT include costs
- Pure signal quality metric

### Top-K Metrics (Cost Overlay)
- **Gross Top-K Avg ER**: Original metric
- **Net Top-K Avg ER**: Gross - costs
- **Alpha Survives**: Net > 0 flag

### Quintile Spread (Cost Overlay)
- **Gross Spread**: Top 20% - bottom 20%
- **Net Spread**: After applying costs to both legs
- Shows if edge survives implementation

### Hit Rate (Informational)
- Keep as gross (% positive returns)
- Costs are systematic, not per-stock

---

## Next Steps

### Phase 3: Baselines (TODO)
Implement 3 baselines using **identical pipeline** (including cost overlay):
1. `mom_12m`: 12-month momentum
2. `momentum_composite`: Equal-weight average of 1/3/6/12m
3. `short_term_strength`: 1-month momentum (diagnostic)

**Critical:** All baselines MUST:
- Use same `EvaluationRow` format
- Go through same universe snapshots
- Use same cost calculation
- Use same purging/embargo

### Phase 5: End-to-End Execution (TODO)
- Integrate cost overlay into `evaluate_fold()`
- Generate net-of-cost reports per fold/horizon
- Run sensitivity analysis across all folds
- Create "alpha survival" summary tables
- Document where strategies break (regime, ADV, AUM)

---

## Acceptance Criteria Met

✅ **Zero turnover → minimal costs**  
✅ **Lower ADV → higher costs (monotonicity)**  
✅ **Higher AUM → higher costs (monotonicity)**  
✅ **Deterministic results (same inputs → same outputs)**  
✅ **ADV missing → conservative penalty (not optimistic)**  
✅ **Frozen assumptions (cannot drift)**  
✅ **All 28 tests passing**

---

## Key Deliverables

**Code:**
- `src/evaluation/costs.py` (514 lines)
- `tests/test_costs.py` (471 lines)

**Documentation:**
- `PROJECT_DOCUMENTATION.md` updated (Section 6.4)
- `PROJECT_STRUCTURE.md` updated (Phase 4 status)
- `CHAPTER_6_PHASE4_COMPLETE.md` (this file)

**Tests:**
- 28 new tests (all passing)
- Total Chapter 6: 123 tests passing

**Artifacts:**
- Frozen `TRADING_ASSUMPTIONS` dataclass
- Square-root slippage model (c * sqrt(participation))
- Sensitivity bands (low/base/high)
- Monotonicity validators
- Net-of-cost metric overlay

---

## Summary

Phase 4 delivers a **diagnostic cost overlay** that:
- ✅ Keeps ranking metrics pure (RankIC unaffected)
- ✅ Overlays costs on portfolio metrics (Top-K, spreads)
- ✅ Uses sensitivity bands (not false precision)
- ✅ Enforces monotonicity (lower ADV → higher cost)
- ✅ Applies conservative penalties (missing ADV → 100 bps)
- ✅ Maintains determinism (same inputs → same outputs)
- ✅ Freezes assumptions (cannot drift)

**Chapter 6 Progress:**
- ✅ Phase 0: Sanity Checks (16 tests)
- ✅ Phase 1: Walk-Forward (25 tests)
- ✅ Phase 1.5: Definition Lock (40 tests)
- ✅ Phase 2: Metrics (30 tests)
- ✅ Phase 4: Cost Realism (28 tests)
- 🔄 Phase 3: Baselines (TODO)
- 🔄 Phase 5: End-to-End Execution (TODO)

**Total: 139/139 tests passing (123 in evaluation suite)**

Ready to proceed to Phase 3 (Baselines) when needed.


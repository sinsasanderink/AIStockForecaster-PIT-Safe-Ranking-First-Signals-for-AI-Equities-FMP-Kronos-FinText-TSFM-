# Chapter 8: Kronos Integration - Executive Summary

**Date:** January 7, 2026  
**Status:** 🟢 READY TO START (post-critical-fixes)

---

## What We're Building

Integrate **Kronos** into the walk-forward evaluation pipeline to score AI stocks at monthly rebalance dates for horizons **20d / 60d / 90d**, then compare to frozen Chapter 6/7 baselines.

---

## Non-Negotiable Implementation Constraints (Locked)

### 1. Correct Kronos API
Initialize via:
```python
tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
model = Kronos.from_pretrained("NeoQuasar/Kronos-base")
predictor = KronosPredictor(model=model, tokenizer=tokenizer, max_context=512)
```

**NOT:** `KronosPredictor.from_pretrained(...)` ❌

### 2. Global Trading Calendar
- Load once from DuckDB `prices` table (all dates)
- No generic `freq="B"` calendars ❌
- No fold-filtered calendars ❌

```python
import duckdb
con = duckdb.connect("data/features.duckdb", read_only=True)
dates = con.execute("SELECT DISTINCT date FROM prices ORDER BY date").df()["date"]
trading_calendar = pd.DatetimeIndex(pd.to_datetime(dates))
```

### 3. Batch Inference
Score universes using `predict_batch()` (not per-ticker `predict()` loops)

**Performance:**
- Before (loops): 100 tickers × 5 sec = 500 sec per date
- After (batch): 100 tickers in one batch = ~10 sec per date
- **50x speedup**

### 4. Correct Evaluation Contract
Score **every (date, ticker)** row in fold validation features (monthly rebalance dates), not a single fold endpoint.

```python
def kronos_scoring_function(features_df, fold_id, horizon):
    # features_df already filtered to validation period
    unique_dates = features_df["date"].unique()
    
    for date in unique_dates:
        # Score all tickers for this date
        scores = adapter.score_universe_batch(...)
```

### 5. Correct Data Source for OHLCV History
Fetch OHLCV history from DuckDB `prices` (via `PricesStore`), **NOT** from fold-filtered `features_df` ❌

```python
class PricesStore:
    def fetch_ohlcv(self, ticker, asof_date, lookback):
        # Query DuckDB prices table directly
        # Returns full history even during fold validation
```

**Why:** `features_df` contains only validation-period rows, missing the full lookback history needed for Kronos (252 days).

---

## What Success Looks Like

### Gate 1 (Factor Baseline)
- ✅ Runs end-to-end without contract errors
- ✅ RankIC ≥ 0.02 for ≥2 horizons
- ✅ Signal not redundant with momentum (corr < 0.5)

### Gate 2 (ML Gate)
- ✅ Any horizon RankIC ≥ 0.05, OR
- ✅ Within 0.03 of LGB baseline on any horizon

### Gate 3 (Practical)
- ✅ Churn ≤ 0.30
- ✅ Cost survival ≥ 30% (60d/90d)
- ✅ Stable across volatility regimes

---

## Scoring Definition

Kronos predicts future OHLCV. We map its forecast to a ranking score:

```python
kronos_score = (pred_close - current_close) / current_close
```

This is treated as an **excess return proxy** suitable for **cross-sectional ranking**.

**Why this works:**
- We care about **ordering** (not absolute values)
- Dividend/benchmark components are smaller and uniform cross-sectionally
- For short horizons (20-90d), price return dominates total return

---

## Critical Fixes Applied

All issues from `CHAPTER_8_CRITICAL_FIXES.md` resolved:

| # | Issue | Status |
|---|-------|--------|
| 1 | Wrong Kronos constructor | ✅ Fixed |
| 2 | Broken timestamps/calendar | ✅ Fixed |
| 3 | Wrong scoring contract | ✅ Fixed |
| 4 | Slow sampling strategy | ✅ Fixed |
| 5 | Unclear return mapping | ✅ Fixed |
| 6 | CSV/JSON loading bugs | ✅ Fixed |
| 7 | Missing batch inference | ✅ Fixed |
| 8 | Factual claim attribution | ✅ Fixed |

**Additional:**
- ✅ OHLCV from `PricesStore` (not `features_df`)
- ✅ Future dates from last x_timestamp (not `asof_date`)

---

## Deliverables

### New Files
- `src/data/prices_store.py` - DuckDB-backed OHLCV fetcher
- `src/data/trading_calendar.py` - Global trading calendar loader
- `src/models/kronos_adapter.py` - Batch scoring adapter
- `scripts/run_chapter8_kronos.py` - Walk-forward evaluation runner
- `scripts/test_kronos_single_stock.py` - Single-stock sanity check
- `scripts/compare_kronos_vs_baselines.py` - Baseline comparison
- `tests/test_kronos_adapter.py` - Unit tests
- `tests/test_prices_store.py` - Unit tests

### Documentation
- `documentation/CHAPTER_8_KRONOS_RESULTS.md` - Results writeup
- `documentation/CHAPTER_8_FREEZE.md` - Freeze + reproducibility

### Artifacts
- `evaluation_outputs/chapter8_kronos_full/` - Full evaluation results
  - `fold_summaries.csv`
  - `per_date_metrics.csv`
  - `churn_series.csv`
  - `cost_overlays.csv`
  - Stability reports

---

## Decision Rule (After FULL)

```
Run FULL evaluation
  ↓
Check Gates
  ↓
Gate 2 Pass (RankIC ≥ 0.05)?
  ├─ YES → Freeze → Chapter 9 ✅
  └─ NO → Check Gate 1
      ↓
      Gate 1 Pass (RankIC ≥ 0.02)?
      ├─ YES → Consider fine-tuning OR proceed to Chapter 9
      └─ NO → Debug data/contract/inference
```

**Fine-tuning decision:**
- Only if RankIC ∈ [0.02, 0.05] (passed Gate 1, failed Gate 2)
- And we have GPU access
- And willing to invest 8-16 GPU hours

---

## Timeline

| Phase | Time | Deliverable |
|-------|------|-------------|
| Pre-implementation | 2-4 hours | Critical fixes review |
| Phase 1 (Zero-shot) | 2-4 days | Adapter + SMOKE test |
| Phase 2 (FULL eval) | 4-8 hours | Complete results |
| Phase 3 (Comparison) | 4 hours | Baseline comparison |
| Phase 4 (Analysis) | 4 hours | Gate checks + decision |
| Phase 5 (Freeze) | 4 hours | Documentation + commit |
| **Total (zero-shot)** | **4-6 days** | **Frozen Chapter 8** |
| Phase 6 (Fine-tuning, optional) | 4-7 days | Fine-tuned model |

---

## Key References

| Document | Purpose |
|----------|---------|
| `CHAPTER_8_CRITICAL_FIXES.md` | All 8 critical issues + fixes |
| `CHAPTER_8_IMPLEMENTATION_PLAN.md` | Detailed technical plan with code |
| `CHAPTER_8_TODO.md` | Step-by-step checklist |
| `CHAPTER_8_PLAN.md` | Original high-level plan |
| `CHAPTER_8_READINESS.md` | Prerequisites check |

**Kronos Resources:**
- GitHub: https://github.com/shiyu-coder/Kronos
- Paper: https://arxiv.org/html/2508.02739v1
- Model Zoo: `NeoQuasar/Kronos-Tokenizer-base`, `NeoQuasar/Kronos-base`

---

## What Changed vs Original Plan

### Fixed (Critical)
1. ✅ Corrected Kronos constructor pattern
2. ✅ Global trading calendar (not fold-filtered)
3. ✅ Batch inference (not single-ticker loops)
4. ✅ Scoring contract (all validation dates)
5. ✅ Data source (`PricesStore`, not `features_df`)
6. ✅ Future dates from last x_timestamp
7. ✅ CSV/JSON loader fixes

### Removed (Incorrect)
- ❌ `KronosPredictor.from_pretrained()`
- ❌ `shiyu-coder/*` HuggingFace IDs
- ❌ `freq="B"` generic business days
- ❌ Fold-filtered trading calendar
- ❌ Per-ticker `predict()` loops
- ❌ Scoring once per fold at `fold.train_end`
- ❌ Using `features_df` as OHLCV source

---

## Bottom Line

**You are ready to start Chapter 8.**

- ✅ All critical issues fixed
- ✅ Correct API usage (per Kronos README)
- ✅ Scalable inference (batch processing)
- ✅ Correct integration (evaluation contract)
- ✅ Clean data sources (`PricesStore` + global calendar)
- ✅ Clear decision gates

**No blockers. Begin implementation.**

---

**Next Action:** Start Phase 1 - Create `trading_calendar.py` and `prices_store.py`.

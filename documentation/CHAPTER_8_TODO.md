# Chapter 8: Kronos Integration - TODO List

**Date:** January 7, 2026  
**Status:** 🟢 READY TO START (post-critical-fixes)

---

## Pre-Implementation (Hard Requirements)

- [ ] Replace all old Kronos references:
  - [ ] Remove `from model.predictor import KronosPredictor` if still present
  - [ ] Remove `shiyu-coder/*` HF IDs if docs/code still mention them
  - [ ] Use constructor: `KronosTokenizer.from_pretrained(...)`, `Kronos.from_pretrained(...)`, `KronosPredictor(model=..., tokenizer=...)`

- [ ] Add global trading calendar loader from DuckDB `prices`
- [ ] Add `PricesStore` (DuckDB-backed OHLCV history fetcher)
- [ ] Confirm scorer contract: scores for every `(date, ticker)` in validation features
- [ ] Review CHAPTER_8_CRITICAL_FIXES.md for all issues

---

## Critical Issues Checklist (from CRITICAL_FIXES.md)

- [ ] **Issue 1:** Use correct Kronos constructor pattern ✅
- [ ] **Issue 2:** Load GLOBAL trading calendar from DuckDB (not fold-filtered) ✅
- [ ] **Issue 3:** Rewrite scoring function to score all validation dates ✅
- [ ] **Issue 4:** Use deterministic inference (T=0) for Phase 1 ✅
- [ ] **Issue 5:** Document return mapping (price → excess return proxy) ✅
- [ ] **Issue 6:** Fix CSV/JSON loading bugs ✅
- [ ] **Issue 7:** Implement batch inference with `predict_batch()` ✅
- [ ] **Issue 8:** Add proper attribution for claims ✅

**Additional:**
- [ ] OHLCV history from `PricesStore` (not `features_df`) ✅
- [ ] Future dates from last x_timestamp (not `asof_date`) ✅

---

## Phase 1: Zero-Shot Implementation

### 1) Data Plumbing

- [ ] Create `src/data/trading_calendar.py`
  - [ ] Implement `load_global_trading_calendar(db_path)` → pd.DatetimeIndex
  - [ ] Query: `SELECT DISTINCT date FROM prices ORDER BY date`
  - [ ] Return global calendar (all dates, not fold-filtered)

- [ ] Create `src/data/prices_store.py`
  - [ ] Implement `PricesStore` class with DuckDB connection
  - [ ] Implement `fetch_ohlcv(ticker, asof_date, lookback)` → DataFrame
  - [ ] Query: `SELECT ... FROM prices WHERE ticker=? AND date<=? ORDER BY date`
  - [ ] Return OHLCV with DatetimeIndex
  - [ ] Handle insufficient history (return empty or skip)

- [ ] Test data plumbing:
  ```bash
  python -c "
  from src.data.trading_calendar import load_global_trading_calendar
  from src.data.prices_store import PricesStore
  
  cal = load_global_trading_calendar()
  print(f'Trading calendar: {len(cal)} dates')
  
  store = PricesStore()
  ohlcv = store.fetch_ohlcv('NVDA', '2024-01-15', 252)
  print(f'NVDA OHLCV: {len(ohlcv)} rows')
  print(ohlcv.head())
  "
  ```

### 2) Kronos Adapter (Batch-First)

- [ ] Create `src/models/__init__.py` if not exists

- [ ] Create `src/models/kronos_adapter.py`
  - [ ] Import correct API: `from model import Kronos, KronosTokenizer, KronosPredictor`
  - [ ] Implement `__init__()`:
    - [ ] `tokenizer = KronosTokenizer.from_pretrained(tokenizer_id)`
    - [ ] `model = Kronos.from_pretrained(model_id)`
    - [ ] `predictor = KronosPredictor(model=model, tokenizer=tokenizer, max_context=512)`
    - [ ] Store global trading calendar
  - [ ] Implement `get_future_dates(last_x_date, horizon)`:
    - [ ] Use `np.searchsorted()` on global calendar
    - [ ] Return next `horizon` trading days
    - [ ] **Critical:** Use `last_x_date` from actual data, not `asof_date`
  - [ ] Implement `score_universe_batch(prices_store, tickers, asof_date, horizon)`:
    - [ ] Loop tickers to prepare OHLCV (via `prices_store.fetch_ohlcv()`)
    - [ ] Skip if `len(ohlcv) != lookback` (batch requires exact length)
    - [ ] Collect `df_list`, `x_timestamp_list`, `valid_tickers`
    - [ ] Compute `y_timestamp` from `x_timestamp_list[0][-1]` (last observed date)
    - [ ] Call `predictor.predict_batch()` with:
      - `T=0.0` (deterministic)
      - `top_p=1.0`
      - `sample_count=1`
    - [ ] Compute returns: `(pred_close - current_close) / current_close`
    - [ ] Return DataFrame with `[ticker, kronos_score]`

- [ ] Test adapter initialization:
  ```bash
  python -c "
  from src.data.trading_calendar import load_global_trading_calendar
  from src.models.kronos_adapter import KronosAdapter
  
  cal = load_global_trading_calendar()
  adapter = KronosAdapter(trading_calendar=cal, device='cpu')
  print('Kronos adapter initialized successfully')
  "
  ```

### 3) Scoring Function (Evaluation Contract)

- [ ] Create `scripts/run_chapter8_kronos.py`
  - [ ] Import all required modules
  - [ ] Implement `get_prices_store()` - cached global instance
  - [ ] Implement `get_kronos_adapter(device)` - cached global instance with global calendar
  - [ ] Implement `kronos_scoring_function(features_df, fold_id, horizon)`:
    - [ ] Get cached `prices_store` and `adapter`
    - [ ] Get unique dates: `features_df["date"].unique()`
    - [ ] Loop over dates:
      - [ ] Filter `features_df` to current date
      - [ ] Get tickers for that date
      - [ ] Call `adapter.score_universe_batch(prices_store, tickers, date, horizon)`
      - [ ] Merge with features to get labels
      - [ ] Format as EvaluationRow: rename columns, add `fold_id`, `horizon`
    - [ ] Concatenate all dates
    - [ ] Return complete DataFrame
  - [ ] Implement `main()`:
    - [ ] Parse args: `--mode`, `--device`
    - [ ] Load data from DuckDB
    - [ ] Create `ExperimentSpec`
    - [ ] Call `run_experiment()` with `kronos_scoring_function`
    - [ ] Print results summary

### 4) Single-Stock Sanity Script

- [ ] Create `scripts/test_kronos_single_stock.py`
  - [ ] Load global trading calendar
  - [ ] Create `PricesStore`
  - [ ] Fetch OHLCV for NVDA
  - [ ] Create `KronosAdapter`
  - [ ] Run 20d prediction for single stock
  - [ ] Print:
    - [ ] Current close
    - [ ] Predicted close
    - [ ] Computed return
    - [ ] Future dates used

- [ ] Run single-stock test:
  ```bash
  python scripts/test_kronos_single_stock.py
  # Should print prediction without errors
  ```

### 5) SMOKE Evaluation

- [ ] Run SMOKE mode:
  ```bash
  python scripts/run_chapter8_kronos.py --mode SMOKE --device cpu
  ```

- [ ] Verify outputs:
  - [ ] `evaluation_outputs/chapter8_kronos_smoke/` created
  - [ ] `fold_summaries.csv` exists and has all 3 horizons
  - [ ] `per_date_metrics.csv` exists
  - [ ] No NaNs in `score` column
  - [ ] Median RankIC values are reasonable (-0.5 to 0.5 range)

- [ ] Check for errors:
  - [ ] No contract violations
  - [ ] No missing timestamps
  - [ ] No batch size mismatches

---

## Phase 2: FULL Evaluation

- [ ] Run FULL evaluation:
  ```bash
  # Prefer GPU for speed
  python scripts/run_chapter8_kronos.py --mode FULL --device cuda
  
  # Or CPU if no GPU (slower, ~4-8 hours with batch)
  python scripts/run_chapter8_kronos.py --mode FULL --device cpu
  ```

- [ ] Monitor progress:
  - [ ] Check logs for errors
  - [ ] Estimate completion time
  - [ ] Monitor memory usage
  - [ ] Verify batch inference is working (fast per-date scoring)

- [ ] Verify completion:
  - [ ] All folds processed
  - [ ] All horizons (20d/60d/90d) completed
  - [ ] Stability reports generated
  - [ ] Cost overlays computed
  - [ ] Churn series computed

---

## Phase 3: Comparison vs Frozen Baselines (Correct Parsers)

- [ ] Fix comparison loaders (if not already correct):
  - [ ] Ch6 baseline floor: `json.loads(Path(...).read_text())`
  - [ ] Ch7 ML baseline: `pd.read_csv(...)`
  - [ ] Ch8 Kronos: `pd.read_csv(...)`

- [ ] Create `scripts/compare_kronos_vs_baselines.py`
  - [ ] Load Ch6 baseline floor (JSON)
  - [ ] Load Ch7 tabular_lgb fold summaries (CSV)
  - [ ] Load Ch8 Kronos fold summaries (CSV)
  - [ ] Compute median RankIC by horizon
  - [ ] Compute lifts vs factor and ML baselines
  - [ ] Print comparison table
  - [ ] Export markdown summary

- [ ] Run comparison:
  ```bash
  python scripts/compare_kronos_vs_baselines.py
  ```

- [ ] Review results:
  - [ ] Median RankIC by horizon
  - [ ] Lifts vs factor baseline
  - [ ] Lifts vs ML baseline
  - [ ] Cost survival rates
  - [ ] Churn analysis

---

## Phase 4: Analysis & Decision

### Gate 1 Check
- [ ] Kronos runs end-to-end? (no contract errors)
- [ ] RankIC ≥ 0.02 for ≥2 horizons?
- [ ] Signal independent from momentum (corr < 0.5)?

### Gate 2 Check
- [ ] Any horizon: RankIC ≥ 0.05?
- [ ] Any horizon: within 0.03 of LGB baseline?

### Gate 3 Check
- [ ] Churn ≤ 0.30?
- [ ] Cost survival ≥ 30% (60d/90d)?
- [ ] Stable across VIX regimes?

### Decision
- [ ] **If Gates 1+2 pass:** Freeze and proceed to Chapter 9 ✅
- [ ] **If Gate 1 passes, Gate 2 fails:** Consider fine-tuning
- [ ] **If Gate 1 fails:** Debug data/contract/inference

---

## Phase 5: Freeze & Documentation

### Create Freeze Document

- [ ] Create `documentation/CHAPTER_8_FREEZE.md`
  - [ ] Final RankIC results (all horizons)
  - [ ] Acceptance criteria verdict (which gates passed)
  - [ ] Model configuration (zero-shot or fine-tuned)
  - [ ] Frozen artifacts location
  - [ ] Reproducibility instructions
  - [ ] Git commit hash
  - [ ] Data hash

### Create Results Document

- [ ] Create `documentation/CHAPTER_8_KRONOS_RESULTS.md`
  - [ ] Executive summary
  - [ ] Median RankIC by horizon
  - [ ] Comparison vs baselines (table + charts)
  - [ ] Cost survival analysis
  - [ ] Regime stability analysis
  - [ ] Churn analysis
  - [ ] Key findings
  - [ ] Recommendations

### Update Project Documentation

- [ ] Update `documentation/ROADMAP.md`
  - [ ] Mark Chapter 8 as ✅ COMPLETE
  - [ ] Update model count
  - [ ] Add Kronos to completed models
  - [ ] Update next steps (Chapter 9)

- [ ] Update `documentation/PROJECT_STRUCTURE.md`
  - [ ] Document `src/models/` section
  - [ ] Document `src/data/prices_store.py`
  - [ ] Document Kronos adapter

- [ ] Update `outline.ipynb` (if needed)
  - [ ] Add Chapter 8 summary cell
  - [ ] Update status indicators

### Commit & Tag

- [ ] Stage all changes:
  ```bash
  git add evaluation_outputs/chapter8_kronos_full/
  git add src/models/
  git add src/data/prices_store.py
  git add src/data/trading_calendar.py
  git add scripts/run_chapter8_kronos.py
  git add documentation/CHAPTER_8_*.md
  ```

- [ ] Commit:
  ```bash
  git commit -m "Chapter 8: Kronos integration complete

  - Zero-shot evaluation with batch inference
  - Correct constructor pattern (per Kronos README)
  - Global trading calendar (not fold-filtered)
  - Scoring all validation dates (correct contract)
  - PricesStore for OHLCV history (not features_df)
  - Results: RankIC 20d/60d/90d = [fill in]
  "
  ```

- [ ] Tag release:
  ```bash
  git tag chapter8-kronos-freeze
  git push origin chapter8-kronos-freeze
  ```

---

## Unit Tests

- [ ] Create `tests/test_kronos_adapter.py`
  - [ ] Test `get_future_dates()` with mock calendar
  - [ ] Test `score_universe_batch()` with mock prices
  - [ ] Test handling of insufficient history
  - [ ] Test batch size consistency
  - [ ] Test return computation

- [ ] Create `tests/test_prices_store.py`
  - [ ] Test `fetch_ohlcv()` with mock DuckDB
  - [ ] Test DatetimeIndex is set correctly
  - [ ] Test lookback truncation
  - [ ] Test empty result handling

- [ ] Run all tests:
  ```bash
  pytest tests/test_kronos_adapter.py -v
  pytest tests/test_prices_store.py -v
  pytest tests/ -k kronos -v
  ```

---

## Optional: Fine-Tuning (Phase 6)

**Only proceed if zero-shot < Gate 2 threshold**

- [ ] Review fine-tuning requirements from Kronos repo
- [ ] Prepare Qlib data (convert DuckDB → Qlib format)
- [ ] Configure fine-tuning (epochs, lr, batch size)
- [ ] Fine-tune tokenizer (2-4 hours, 2 GPUs)
- [ ] Fine-tune predictor (6-12 hours, 2 GPUs)
- [ ] Re-evaluate fine-tuned model
- [ ] Compare fine-tuned vs zero-shot
- [ ] Document fine-tuning results

---

## Time Estimates

| Phase | Optimistic | Realistic | Pessimistic |
|-------|-----------|-----------|-------------|
| Pre-implementation | 1 hour | 2 hours | 4 hours |
| Phase 1 (Zero-shot) | 1 day | 2 days | 4 days |
| Phase 2 (FULL eval) | 4 hours | 8 hours | 1 day |
| Phase 3 (Comparison) | 2 hours | 4 hours | 1 day |
| Phase 4 (Analysis) | 2 hours | 4 hours | 1 day |
| Phase 5 (Freeze) | 2 hours | 4 hours | 1 day |
| **Total (zero-shot)** | **2 days** | **4 days** | **8 days** |
| Phase 6 (Fine-tuning) | 2 days | 4 days | 7 days |

---

## Checklist Summary

**Critical (must complete):**
- [ ] All pre-implementation items (8 critical fixes)
- [ ] Phase 1: Zero-shot implementation
- [ ] Phase 2: FULL evaluation
- [ ] Phase 3: Comparison vs baselines
- [ ] Phase 4: Gate checks + decision
- [ ] Phase 5: Freeze + documentation

**Optional:**
- [ ] Phase 6: Fine-tuning (only if needed)
- [ ] Unit tests (recommended but not blocking)

---

**Status:** Ready to start implementation with corrected API + scalable inference.

**Next Action:** Begin Phase 1 (Data Plumbing) - create `trading_calendar.py` and `prices_store.py`.

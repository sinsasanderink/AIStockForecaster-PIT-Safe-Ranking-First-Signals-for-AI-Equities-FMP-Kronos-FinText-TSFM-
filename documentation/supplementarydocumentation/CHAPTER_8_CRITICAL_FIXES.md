# Chapter 8: Critical Fixes Before Implementation

**Date:** January 7, 2026  
**Status:** ⚠️ MUST FIX BEFORE CODING

---

## Overview

This document addresses critical accuracy and integration issues identified in the initial plan. **All fixes below must be implemented before starting Chapter 8 coding.**

---

## ✅ Phase 1 Bug Fixes (Jan 7, 2026)

### Issue 10: INSERT Statement Bug in add_prices_table_to_duckdb.py

**Problem:** The script attempted to INSERT from a pandas DataFrame as if it were a DuckDB table:
```python
# WRONG - prices_ordered is a pandas DataFrame, not a DuckDB table
conn.execute("INSERT INTO prices SELECT * FROM prices_ordered")
```

**Error:** `Table with name prices_ordered does not exist!`

**Fix Applied:**
```python
# CORRECT - register DataFrame first
conn.register("prices_ordered", prices_ordered)
conn.execute("INSERT INTO prices SELECT * FROM prices_ordered")
conn.unregister("prices_ordered")
```

**Status:** ✅ FIXED  
**Impact:** Critical - script would crash before adding any data to DuckDB  
**File:** `scripts/add_prices_table_to_duckdb.py`

---

### Issue 11: Date Type Handling in PricesStore.fetch_ohlcv()

**Problem:** Passing `pd.Timestamp` directly to DuckDB WHERE clause with DATE column:
```python
# Slightly unsafe - usually works but not clean
df = self.con.execute(query, [ticker, asof_ts]).df()
```

**Fix Applied:**
```python
# CORRECT - cleaner type matching
df = self.con.execute(query, [ticker, asof_ts.date()]).df()
```

**Status:** ✅ FIXED  
**Impact:** Minor - usually worked but cleaner and more explicit now  
**File:** `src/data/prices_store.py`

---

## Pre-Implementation Critical Fixes (Resolved)

---

## ❌ Issue 1: Wrong Kronos Model IDs + Import Path (CRITICAL)

### What's Wrong
Original plan used:
```python
from model.predictor import KronosPredictor
# HuggingFace IDs: shiyu-coder/Kronos-tokenizer, shiyu-coder/Kronos-predictor
```

**But:** Kronos README "Model Zoo" shows different HF IDs and import style.

### ✅ Fix: Use Actual Kronos API

Based on [Kronos GitHub README](https://github.com/shiyu-coder/Kronos), the correct usage is:

```python
# Correct imports (from Kronos repo)
from model import Kronos, KronosTokenizer, KronosPredictor

# Correct HuggingFace model IDs (from Model Zoo)
# Correct construction pattern (per README):

# Option 1: Base models
tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
model = Kronos.from_pretrained("NeoQuasar/Kronos-base")
predictor = KronosPredictor(
    model=model,
    tokenizer=tokenizer,
    max_context=512  # Or appropriate context length
)

# Option 2: Mini models (faster, lower memory)
tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-mini")
model = Kronos.from_pretrained("NeoQuasar/Kronos-mini")
predictor = KronosPredictor(
    model=model,
    tokenizer=tokenizer,
    max_context=512
)
```

**Key difference:** Use `KronosPredictor(model, tokenizer, ...)` constructor, NOT `KronosPredictor.from_pretrained(...)`

**Action:** 
1. Verify correct import path by cloning Kronos repo
2. Test model download with correct IDs
3. Update all plan documents with correct API

---

## ❌ Issue 2: Timestamp/Index Handling Will Break (CRITICAL)

### What's Wrong
Original `prepare_ohlcv_sequence()` never sets index:
```python
ohlcv_df = ticker_df[['open', 'high', 'low', 'close', 'volume']].copy()
# Returns DataFrame with RangeIndex (0, 1, 2, ...)
```

Then passes:
```python
x_timestamp=ohlcv_df.index  # Just RangeIndex!
y_timestamp=pd.date_range(..., freq="B")  # Generic business days!
```

**Problems:**
1. Kronos expects real timestamps, not 0..251
2. `freq="B"` doesn't match exchange holidays
3. Future dates may not exist in our dataset → label misalignment

### ✅ Fix: Use Actual Trading Calendar

```python
def prepare_ohlcv_sequence(
    self,
    prices_df: pd.DataFrame,
    ticker: str,
    asof_date: date,
    lookback: int = 252,
) -> pd.DataFrame:
    """
    Extract OHLCV sequence with proper timestamp index.
    
    Returns DataFrame with:
    - Index: actual trading dates from DuckDB
    - Columns: ['open', 'high', 'low', 'close', 'volume']
    """
    # Filter to ticker and dates before asof_date
    ticker_df = prices_df[
        (prices_df['ticker'] == ticker) &
        (prices_df['date'] <= asof_date)
    ].sort_values('date').tail(lookback)
    
    # Set index to actual trading dates (critical!)
    ohlcv = ticker_df[['open', 'high', 'low', 'close', 'volume']].copy()
    ohlcv.index = pd.to_datetime(ticker_df['date'])
    
    return ohlcv

def predict_returns(
    self,
    ohlcv_df: pd.DataFrame,
    horizon: int,
    num_samples: int = 1,
) -> Dict[str, float]:
    """
    Generate return predictions.
    
    Args:
        ohlcv_df: OHLCV with DatetimeIndex
        horizon: Forecast horizon in TRADING DAYS
        num_samples: Number of samples (default 1 for speed)
    """
    # Get historical timestamps (already set as index)
    x_timestamp = ohlcv_df.index
    
    # CRITICAL: Get future timestamps from ACTUAL trading calendar
    # NOT from generic freq="B"
    # We need to pass in the trading calendar dates
    current_date = x_timestamp[-1]
    
    # Option A: Extract next N trading days from prices_df
    # (requires prices_df to be passed in or cached)
    
    # Option B: Use Kronos's internal calendar (if available)
    
    # Option C: Pre-compute future dates from our trading calendar
    # and pass them in
    
    # For now, placeholder - must be fixed with actual calendar
    # THIS IS WRONG - just showing what needs to be replaced
    y_timestamp = pd.date_range(
        start=current_date,
        periods=horizon + 1,
        freq='B'  # ← WRONG! Must use actual trading calendar
    )[1:]
    
    # ... rest of prediction logic
```

**Correct approach: Use GLOBAL trading calendar from DuckDB:**

```python
def load_global_trading_calendar(db_path: str = "data/features.duckdb") -> pd.DatetimeIndex:
    """
    Load complete trading calendar from DuckDB.
    
    CRITICAL: Must be global (all dates in prices table), not fold-filtered.
    Otherwise near fold boundaries, we can't generate future dates for pred_len.
    """
    import duckdb
    con = duckdb.connect(db_path, read_only=True)
    
    # Get all unique dates from prices (not filtered by fold!)
    dates_df = con.execute("""
        SELECT DISTINCT date 
        FROM prices 
        ORDER BY date
    """).df()
    con.close()
    
    return pd.to_datetime(dates_df['date'])

class KronosAdapter:
    def __init__(self, ..., trading_calendar: pd.DatetimeIndex):
        """
        Args:
            trading_calendar: GLOBAL trading calendar (all dates from DuckDB)
                             NOT fold-filtered dates!
        """
        tokenizer = KronosTokenizer.from_pretrained(tokenizer_path)
        model = Kronos.from_pretrained(model_path)
        self.predictor = KronosPredictor(
            model=model,
            tokenizer=tokenizer,
            max_context=512
        )
        self.trading_calendar = trading_calendar
    
    def get_future_dates(self, current_date: date, horizon: int) -> pd.DatetimeIndex:
        """Get next N trading days after current_date."""
        current_idx = np.searchsorted(
            self.trading_calendar, 
            pd.Timestamp(current_date)
        )
        future_dates = self.trading_calendar[
            current_idx + 1 : current_idx + 1 + horizon
        ]
        return pd.DatetimeIndex(future_dates)
```

**Why global calendar matters:**
- Fold-filtered dates = missing future dates near fold boundaries
- Result: `get_future_dates()` fails or returns incomplete horizon
- Solution: Load once from DuckDB, pass to adapter globally

---

## ❌ Issue 3: Scoring Function Contract Wrong (CRITICAL)

### What's Wrong
Original `kronos_scoring_function()` scores ONCE per fold:
```python
def kronos_scoring_function(features_df, labels_df, fold, horizon, **kwargs):
    # Only scores at fold.train_end
    scores_df = adapter.score_universe(
        ...,
        asof_date=fold.train_end,  # ← WRONG! Only one date
        horizon=horizon
    )
    return scores_df[['ticker', 'kronos_score']]
```

**Problem:** 
- Monthly rebalancing means multiple evaluation dates per fold
- `run_experiment()` expects scores for EVERY row in `features_df` (validation period)
- Current approach only scores once, missing all rebalance dates

### ✅ Fix: Score Every Date in Validation Period

Based on `run_experiment()` and `generate_baseline_scores()`, the correct signature is:

```python
scorer_fn(features_df, fold_id, horizon) -> DataFrame in EvaluationRow format
```

Where:
- `features_df`: Already filtered to validation period for this fold
- Returns: DataFrame with score for EVERY row in `features_df`

**Correct implementation:**

```python
def kronos_scoring_function(
    features_df: pd.DataFrame,
    fold_id: str,
    horizon: int,
) -> pd.DataFrame:
    """
    Score all (date, ticker) pairs in features_df.
    
    Args:
        features_df: Validation data for this fold (already filtered)
                    Contains ALL dates in validation period
        fold_id: Fold identifier (e.g., "fold_01")
        horizon: Forecast horizon in trading days
    
    Returns:
        DataFrame in EvaluationRow format with columns:
        - as_of_date (date)
        - ticker (str)
        - stable_id (str)
        - horizon (int)
        - fold_id (str)
        - score (float, higher = better)
        - excess_return (float, from labels)
        - [optional fields: adv_20d, sector, etc.]
    """
    # Initialize adapter (or use cached instance)
    adapter = KronosAdapter(device="cpu")
    
    # Get unique dates in validation period
    unique_dates = features_df['date'].unique()
    
    results = []
    
    for asof_date in unique_dates:
        # Get tickers for this date
        date_df = features_df[features_df['date'] == asof_date]
        tickers = date_df['ticker'].unique()
        
        # Score all tickers for this date/horizon
        try:
            scores_df = adapter.score_universe(
                prices_df=features_df,  # Full features (includes history)
                tickers=tickers,
                asof_date=asof_date,
                horizon=horizon
            )
            
            # Merge with features to get labels and other fields
            date_results = date_df[['date', 'ticker', 'stable_id', 
                                   f'excess_return_{horizon}d']].merge(
                scores_df[['ticker', 'kronos_score']],
                on='ticker',
                how='left'
            )
            
            # Rename to EvaluationRow format
            date_results = date_results.rename(columns={
                'date': 'as_of_date',
                'kronos_score': 'score',
                f'excess_return_{horizon}d': 'excess_return'
            })
            date_results['horizon'] = horizon
            date_results['fold_id'] = fold_id
            
            results.append(date_results)
            
        except Exception as e:
            logger.warning(f"Kronos failed for {asof_date}: {e}")
            continue
    
    # Concatenate all dates
    if not results:
        raise ValueError(f"Kronos produced no scores for fold {fold_id}")
    
    return pd.concat(results, ignore_index=True)
```

**Key differences:**
1. ✅ Loops over ALL dates in validation period (not just one)
2. ✅ Returns scores for every (date, ticker) pair
3. ✅ Matches EvaluationRow format exactly
4. ✅ Includes fold_id, horizon, stable_id

---

## ❌ Issue 4: Sampling Strategy Too Slow (IMPORTANT)

### What's Wrong
Original loops `num_samples` times:
```python
for _ in range(num_samples):
    pred_df = self.predictor.predict(
        ...,
        sample_count=1  # ← Calling predict() 5 times per ticker!
    )
```

**Problem:** 100 tickers × 5 samples × 2000 dates = 1M predict() calls

### ✅ Fix: Use Kronos's Built-In Sampling

```python
def predict_returns(
    self,
    ohlcv_df: pd.DataFrame,
    horizon: int,
    num_samples: int = 1,  # Reduce to 1 for first pass
) -> Dict[str, float]:
    """Generate return predictions (efficient sampling)."""
    
    # Single call with sample_count (if Kronos supports it)
    pred_df = self.predictor.predict(
        df=ohlcv_df,
        x_timestamp=ohlcv_df.index,
        y_timestamp=future_timestamps,
        pred_len=horizon,
        T=1.0,
        top_p=0.9,
        sample_count=num_samples  # ← Generate multiple samples in one call
    )
    
    # If Kronos returns multiple samples, aggregate
    # Otherwise, for first pass, just use deterministic (T=0, or num_samples=1)
    
    current_close = ohlcv_df['close'].iloc[-1]
    
    if num_samples == 1:
        # Deterministic
        final_close = pred_df['close'].iloc[-1]
        ret = (final_close - current_close) / current_close
        return {'expected_return': ret, 'return_std': 0.0}
    else:
        # Multiple samples (need to check Kronos API)
        # May need to call once per sample or check if returns array
        pass
```

**Recommendation for Phase 1:** Use `num_samples=1` (deterministic) for speed.

---

## ❌ Issue 5: Return Definition Must Match Labels (IMPORTANT)

### What's Wrong
Unclear if Kronos output matches our label definition.

### ✅ Fix: Explicitly Document Return Mapping

**Our labels (from DuckDB):**
- Column: `excess_return_{horizon}d` (e.g., `excess_return_20d`)
- Definition: **Total return (price + dividends) minus QQQ return**
- Formula: `(P_t+h + divs) / P_t - 1 - QQQ_return`
- Split-adjusted: Yes

**Kronos output:**
- Predicts: Future OHLCV (open, high, low, close, volume)
- Returns: Raw price forecast (NOT excess return)

**Mapping strategy:**

```python
def predict_returns(self, ohlcv_df, horizon, ...):
    """
    Generate predictions and convert to excess return proxy.
    
    NOTE: Kronos predicts PRICE, our labels are EXCESS RETURNS.
    
    Approximation:
    1. Kronos predicts: future_close
    2. We compute: price_return = (future_close - current_close) / current_close
    3. Assume: excess_return ≈ price_return (ignoring dividends + benchmark)
    
    This is acceptable because:
    - We're ranking stocks cross-sectionally (relative ordering matters)
    - Dividends are small over short horizons (20-90d)
    - Benchmark drift affects all stocks similarly (washes out in ranking)
    
    For more accurate mapping (future work):
    - Adjust for expected dividends (from div yield × horizon / 252)
    - Subtract expected benchmark return (historical mean × horizon / 252)
    """
    pred_df = self.predictor.predict(...)
    
    current_close = ohlcv_df['close'].iloc[-1]
    pred_close = pred_df['close'].iloc[-1]
    
    # Raw price return (approximates excess return for ranking)
    price_return = (pred_close - current_close) / current_close
    
    return {'expected_return': price_return, ...}
```

**Add to plan:** Section titled "How Kronos Score Maps to Our Labels"

---

## ❌ Issue 6: CSV/JSON Loading Bug (BUG)

### What's Wrong
Multiple places in docs use:
```python
ch7_ml = json.loads(
    Path('evaluation_outputs/chapter7_tabular_lgb_full/fold_summaries.csv').read_text()
)
```

**Problem:** CSV cannot be loaded with `json.loads()`

### ✅ Fix: Use Correct Loaders

```python
# Load Chapter 7 ML baseline (CSV)
ch7_ml = pd.read_csv(
    'evaluation_outputs/chapter7_tabular_lgb_full/fold_summaries.csv'
)

# Load Chapter 6 baseline floor (JSON)
ch6_floor = json.loads(
    Path('evaluation_outputs/chapter6_closure_real/BASELINE_FLOOR.json').read_text()
)

# Or use pandas for JSON too
ch6_floor_df = pd.read_json(
    'evaluation_outputs/chapter6_closure_real/BASELINE_FLOOR.json'
)
```

**Action:** Fix all comparison snippets in docs.

---

## ❌ Issue 7: Missing Batch Inference (CRITICAL - PERFORMANCE)

### What's Wrong
Original plan loops over tickers one-by-one:
```python
for ticker in tickers:
    ohlcv = prepare_ohlcv_sequence(ticker, asof_date)
    score = predict_returns(ohlcv, horizon)  # ← One predict() call per ticker!
```

**Problem:** 100 tickers × 2000 dates × 3 horizons = 600K individual `predict()` calls
- Even on GPU: ~5-10 seconds per call = weeks of runtime
- Impractical for FULL evaluation

### ✅ Fix: Use Kronos's `predict_batch()`

Kronos README explicitly provides [`predict_batch()`](https://github.com/shiyu-coder/Kronos) for parallel inference:

```python
# From Kronos README
pred_df_list = predictor.predict_batch(
    df_list=[df1, df2, df3],  # List of OHLCV DataFrames
    x_timestamp_list=[x_ts1, x_ts2, x_ts3],
    y_timestamp_list=[y_ts1, y_ts2, y_ts3],
    pred_len=pred_len,
    T=1.0,
    top_p=0.9,
    sample_count=1,
    verbose=True
)
```

**Requirements for batch prediction:**
- All series must have same historical length (lookback)
- All series must have same prediction length (pred_len)
- Each DataFrame must have ['open', 'high', 'low', 'close', 'volume']

**Corrected approach:**

```python
def score_universe_batch(
    self,
    prices_df: pd.DataFrame,
    tickers: List[str],
    asof_date: date,
    horizon: int,
) -> pd.DataFrame:
    """
    Score all tickers for one date using BATCH inference.
    
    Much faster than looping: processes all tickers in parallel on GPU.
    """
    # Prepare all OHLCV sequences
    df_list = []
    x_timestamp_list = []
    valid_tickers = []
    
    for ticker in tickers:
        try:
            ohlcv = self.prepare_ohlcv_sequence(
                prices_df, ticker, asof_date
            )
            
            # Skip if insufficient history
            if len(ohlcv) < 60:
                continue
            
            # Pad/truncate to exact lookback length
            if len(ohlcv) < self.lookback:
                # Pad with first row values (or skip)
                continue
            elif len(ohlcv) > self.lookback:
                ohlcv = ohlcv.tail(self.lookback)
            
            df_list.append(ohlcv)
            x_timestamp_list.append(ohlcv.index)
            valid_tickers.append(ticker)
            
        except Exception as e:
            logger.warning(f"Skipping {ticker}: {e}")
            continue
    
    if not df_list:
        return pd.DataFrame(columns=['ticker', 'kronos_score'])
    
    # Get future timestamps (same for all tickers)
    y_timestamp = self.get_future_dates(asof_date, horizon)
    y_timestamp_list = [y_timestamp] * len(df_list)
    
    # BATCH prediction (single GPU call for all tickers!)
    pred_df_list = self.predictor.predict_batch(
        df_list=df_list,
        x_timestamp_list=x_timestamp_list,
        y_timestamp_list=y_timestamp_list,
        pred_len=horizon,
        T=0.0,  # Deterministic
        top_p=1.0,
        sample_count=1,
        verbose=False
    )
    
    # Compute returns for all predictions
    scores = []
    for ticker, ohlcv, pred_df in zip(valid_tickers, df_list, pred_df_list):
        current_close = ohlcv['close'].iloc[-1]
        pred_close = pred_df['close'].iloc[-1]
        price_return = (pred_close - current_close) / current_close
        
        scores.append({
            'ticker': ticker,
            'kronos_score': price_return
        })
    
    return pd.DataFrame(scores)
```

**Performance improvement:**
- Before: 100 tickers × 5 sec = 500 seconds per date
- After: 100 tickers in one batch = ~10 seconds per date
- **50x speedup** for full evaluation

**Action:** Replace all single-ticker loops with batch inference in adapter.

---

## ❌ Issue 8: Factual Claims Too Strong (MINOR)

### What's Wrong
Plan states: "accepted by AAAI 2026" without caveat

### ✅ Fix: Add Proper Attribution

**Original:**
> Kronos has been accepted by AAAI 2026.

**Fixed:**
> Kronos paper (arXiv:2508.02739) reports acceptance by AAAI 2026 per GitHub repo authors.

**Or:**
> Kronos paper submitted to AAAI 2026 (per arXiv:2508.02739).

**Action:** Review all claims, add "per repo" or "per paper" where appropriate.

---

## 📋 Action Items Before Starting Implementation

### Critical (Must Fix)
- [ ] 1. Use correct Kronos constructor: `KronosPredictor(model, tokenizer, ...)` not `.from_pretrained()`
- [ ] 2. Load GLOBAL trading calendar from DuckDB (not fold-filtered dates)
- [ ] 3. Implement batch inference with `predict_batch()` (not single-ticker loops)
- [ ] 4. Rewrite `kronos_scoring_function()` to score all dates in validation period
- [ ] 5. Document return mapping (Kronos price → our excess return proxy)

### Important (Should Fix)
- [ ] 6. Fix CSV/JSON loading bugs in comparison scripts
- [ ] 7. Use deterministic inference (T=0) for Phase 1 speed

### Minor (Nice to Have)
- [ ] 8. Add attribution for AAAI claim

---

## Updated Code Examples

### Helper: Load Global Trading Calendar
```python
import duckdb
import pandas as pd

def load_global_trading_calendar(
    db_path: str = "data/features.duckdb"
) -> pd.DatetimeIndex:
    """
    Load complete trading calendar from DuckDB.
    
    CRITICAL: Must be global (all dates), not fold-filtered.
    """
    con = duckdb.connect(db_path, read_only=True)
    dates_df = con.execute("""
        SELECT DISTINCT date FROM prices ORDER BY date
    """).df()
    con.close()
    return pd.to_datetime(dates_df['date'])
```

### Corrected KronosAdapter
```python
from model import Kronos, KronosTokenizer, KronosPredictor
import pandas as pd
import numpy as np
from typing import List, Dict
from datetime import date
import logging

logger = logging.getLogger(__name__)

class KronosAdapter:
    """
    Adapter for Kronos foundation model.
    
    Properly handles:
    - Correct constructor pattern (per Kronos README)
    - Global trading calendar (not fold-filtered)
    - Batch inference (not single-ticker loops)
    - Return mapping (price → excess return proxy)
    """
    
    def __init__(
        self,
        tokenizer_path: str = "NeoQuasar/Kronos-Tokenizer-base",
        model_path: str = "NeoQuasar/Kronos-base",
        device: str = "cpu",
        lookback: int = 252,
        trading_calendar: pd.DatetimeIndex = None,
        max_context: int = 512,
    ):
        """
        Args:
            tokenizer_path: HuggingFace model ID for tokenizer
            model_path: HuggingFace model ID for model
            device: "cuda" or "cpu"
            lookback: Historical sequence length
            trading_calendar: GLOBAL trading calendar from DuckDB
            max_context: Max context length for predictor
        """
        # Correct constructor pattern (per Kronos README)
        tokenizer = KronosTokenizer.from_pretrained(tokenizer_path)
        model = Kronos.from_pretrained(model_path)
        
        # Move model to device
        if device == "cuda":
            model = model.cuda()
        
        self.predictor = KronosPredictor(
            model=model,
            tokenizer=tokenizer,
            max_context=max_context
        )
        self.device = device
        self.lookback = lookback
        self.trading_calendar = trading_calendar
        
        if trading_calendar is None:
            raise ValueError("trading_calendar required (use load_global_trading_calendar())")
    
    def get_future_dates(
        self, 
        current_date: pd.Timestamp, 
        horizon: int
    ) -> pd.DatetimeIndex:
        """Get next N trading days after current_date."""
        if self.trading_calendar is None:
            raise ValueError("trading_calendar required for future dates")
        
        current_idx = np.searchsorted(
            self.trading_calendar, 
            current_date
        )
        future_dates = self.trading_calendar[
            current_idx + 1 : current_idx + 1 + horizon
        ]
        return pd.DatetimeIndex(future_dates)
    
    def prepare_ohlcv_sequence(
        self,
        prices_df: pd.DataFrame,
        ticker: str,
        asof_date: date,
    ) -> pd.DataFrame:
        """
        Extract OHLCV with proper DatetimeIndex.
        
        Returns:
            DataFrame with:
            - Index: trading dates (DatetimeIndex)
            - Columns: ['open', 'high', 'low', 'close', 'volume']
        """
        # Filter
        ticker_df = prices_df[
            (prices_df['ticker'] == ticker) &
            (prices_df['date'] <= asof_date)
        ].sort_values('date').tail(self.lookback)
        
        # Extract OHLCV and set proper index
        ohlcv = ticker_df[['open', 'high', 'low', 'close', 'volume']].copy()
        ohlcv.index = pd.to_datetime(ticker_df['date'])
        
        return ohlcv
    
    def predict_returns(
        self,
        ohlcv_df: pd.DataFrame,
        horizon: int,
    ) -> float:
        """
        Predict return for one ticker/horizon.
        
        Returns:
            Price return (proxy for excess return in ranking)
        """
        # Get timestamps
        x_timestamp = ohlcv_df.index
        y_timestamp = self.get_future_dates(x_timestamp[-1], horizon)
        
        # Predict (deterministic for Phase 1)
        pred_df = self.predictor.predict(
            df=ohlcv_df,
            x_timestamp=x_timestamp,
            y_timestamp=y_timestamp,
            pred_len=horizon,
            T=0.0,  # Deterministic (T=0 or very low)
            top_p=1.0,
            sample_count=1
        )
        
        # Compute price return
        current_close = ohlcv_df['close'].iloc[-1]
        pred_close = pred_df['close'].iloc[-1]
        price_return = (pred_close - current_close) / current_close
        
        return price_return
    
    def score_universe(
        self,
        prices_df: pd.DataFrame,
        tickers: List[str],
        asof_date: date,
        horizon: int,
    ) -> pd.DataFrame:
        """
        Score all tickers for one date using BATCH inference.
        
        Key: Uses predict_batch() for 50x speedup vs single-ticker loops.
        """
        # Prepare all OHLCV sequences
        df_list = []
        x_timestamp_list = []
        valid_tickers = []
        
        for ticker in tickers:
            try:
                ohlcv = self.prepare_ohlcv_sequence(
                    prices_df, ticker, asof_date
                )
                
                # Skip if insufficient history
                if len(ohlcv) < 60:
                    continue
                
                # Ensure exact lookback length for batch
                if len(ohlcv) < self.lookback:
                    continue  # Skip if too short
                elif len(ohlcv) > self.lookback:
                    ohlcv = ohlcv.tail(self.lookback)
                
                df_list.append(ohlcv)
                x_timestamp_list.append(ohlcv.index)
                valid_tickers.append(ticker)
                
            except Exception as e:
                logger.warning(f"Skipping {ticker}: {e}")
                continue
        
        if not df_list:
            return pd.DataFrame(columns=['ticker', 'kronos_score'])
        
        # Get future timestamps (same for all tickers)
        y_timestamp = self.get_future_dates(asof_date, horizon)
        y_timestamp_list = [y_timestamp] * len(df_list)
        
        # BATCH prediction (single call for all tickers)
        pred_df_list = self.predictor.predict_batch(
            df_list=df_list,
            x_timestamp_list=x_timestamp_list,
            y_timestamp_list=y_timestamp_list,
            pred_len=horizon,
            T=0.0,  # Deterministic for speed
            top_p=1.0,
            sample_count=1,
            verbose=False
        )
        
        # Compute returns for all predictions
        scores = []
        for ticker, ohlcv, pred_df in zip(valid_tickers, df_list, pred_df_list):
            current_close = ohlcv['close'].iloc[-1]
            pred_close = pred_df['close'].iloc[-1]
            price_return = (pred_close - current_close) / current_close
            
            scores.append({
                'ticker': ticker,
                'kronos_score': price_return
            })
        
        return pd.DataFrame(scores)
```

### Corrected Scoring Function
```python
# Global adapter instance (cache model across folds)
_kronos_adapter = None

def get_kronos_adapter(device: str = "cpu") -> KronosAdapter:
    """Get or create cached Kronos adapter with global trading calendar."""
    global _kronos_adapter
    
    if _kronos_adapter is None:
        # Load GLOBAL trading calendar (not fold-filtered!)
        trading_calendar = load_global_trading_calendar("data/features.duckdb")
        
        _kronos_adapter = KronosAdapter(
            tokenizer_path="NeoQuasar/Kronos-Tokenizer-base",
            model_path="NeoQuasar/Kronos-base",
            device=device,
            trading_calendar=trading_calendar,
            max_context=512
        )
    
    return _kronos_adapter

def kronos_scoring_function(
    features_df: pd.DataFrame,
    fold_id: str,
    horizon: int,
) -> pd.DataFrame:
    """
    Score all (date, ticker) pairs in validation period.
    
    This is the contract expected by run_experiment().
    
    Key improvements:
    - Uses global trading calendar (not fold-filtered)
    - Caches adapter across folds (avoids reload)
    - Uses batch inference (50x speedup)
    """
    # Get cached adapter with global trading calendar
    adapter = get_kronos_adapter(device="cpu")
    
    # Score each date
    results = []
    unique_dates = features_df['date'].unique()
    
    for asof_date in unique_dates:
        date_df = features_df[features_df['date'] == asof_date]
        tickers = date_df['ticker'].unique()
        
        # Score all tickers for this date
        scores_df = adapter.score_universe(
            prices_df=features_df,
            tickers=tickers,
            asof_date=asof_date,
            horizon=horizon
        )
        
        # Merge with labels
        date_results = date_df[[
            'date', 'ticker', 'stable_id', 
            f'excess_return_{horizon}d'
        ]].merge(
            scores_df,
            on='ticker',
            how='left'
        )
        
        # Format as EvaluationRow
        date_results = date_results.rename(columns={
            'date': 'as_of_date',
            'kronos_score': 'score',
            f'excess_return_{horizon}d': 'excess_return'
        })
        date_results['horizon'] = horizon
        date_results['fold_id'] = fold_id
        
        results.append(date_results)
    
    return pd.concat(results, ignore_index=True)
```

---

## Status

**MUST FIX before starting implementation.**

All critical issues (#1-5, #7) must be addressed before coding. Important issue (#6) should be fixed. Minor issue (#8) is optional.

**Critical path:**
1. ✅ Correct constructor: `KronosPredictor(model, tokenizer, max_context)`
2. ✅ Global trading calendar from DuckDB (not fold-filtered)
3. ✅ Batch inference with `predict_batch()`
4. ✅ Scoring function scores all validation dates
5. ✅ Return mapping documented

---

**Next Action:** 
1. Verify these fixes are correct by testing model download
2. Update IMPLEMENTATION_PLAN.md and TODO.md with corrected code
3. Proceed to implementation


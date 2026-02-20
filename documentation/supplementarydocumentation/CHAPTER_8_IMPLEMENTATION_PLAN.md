# Chapter 8: Kronos Integration - Implementation Plan

**Date:** January 7, 2026  
**Status:** 🟢 READY TO START (post-critical-fixes)  
**Kronos Source:** https://github.com/shiyu-coder/Kronos  
**Paper:** https://arxiv.org/html/2508.02739v1

---

## Executive Summary

Integrate **Kronos** (foundation model for OHLCV/K-lines) into our walk-forward evaluation pipeline and compare against frozen baselines.

**Critical design decisions (locked):**
- ✅ Use **correct Kronos constructor pattern**: `KronosPredictor(model=..., tokenizer=...)` (no `KronosPredictor.from_pretrained`)
- ✅ Use **GLOBAL trading calendar** loaded from DuckDB `prices` table (not fold-filtered, no `freq="B"`)
- ✅ Use **batch inference** via `predict_batch()` (no per-ticker `predict()` loops)
- ✅ Scoring contract: score **every (date, ticker)** in the fold's validation features (monthly rebalance dates, not a single date)
- ✅ OHLCV history must come from **DuckDB prices**, not fold-filtered `features_df`

---

## What is Kronos? (Implementation-oriented)

Kronos has:
- **Tokenizer**: quantizes continuous OHLCV into discrete tokens
- **Predictor**: autoregressive Transformer generating future OHLCV tokens, decoded back to OHLCV

We will use Kronos **as a price forecaster** and map forecasted close → **price return proxy** for ranking.

---

## Prerequisites Check ✅

| Requirement | Status | Notes |
|-------------|--------|------|
| PyTorch | ✅ | Installed |
| DuckDB | ✅ | `data/features.duckdb` |
| Frozen Ch6 baseline | ✅ | Factor floor |
| Frozen Ch7 baseline | ✅ | Tabular LGB |
| Kronos HF access | ✅ | First run downloads |

---

## Implementation Strategy

### Phase 1: Zero-Shot (Primary Path)

**Goal:** Run zero-shot Kronos through walk-forward evaluation with correct contract + scalable inference.

#### 1) Create a "PricesStore" (Single Source of Truth)
**Purpose:** Ensure we always have full OHLCV history even when scoring inside fold validation windows.

**File:** `src/data/prices_store.py`

```python
import duckdb
import pandas as pd
from typing import Optional

class PricesStore:
    """
    Single source of truth for OHLCV history.
    
    Prevents fold-filtered leakage/holes by always fetching from full prices table.
    """
    
    def __init__(self, db_path: str = "data/features.duckdb"):
        self.db_path = db_path
        self.con = duckdb.connect(db_path, read_only=True)
    
    def fetch_ohlcv(
        self, 
        ticker: str, 
        asof_date, 
        lookback: int
    ) -> pd.DataFrame:
        """
        Fetch OHLCV for ticker up to asof_date.
        
        Returns:
            DataFrame with DatetimeIndex and columns: [open, high, low, close, volume]
            Empty if insufficient data.
        """
        query = """
            SELECT date, open, high, low, close, volume
            FROM prices
            WHERE ticker = ? AND date <= ?
            ORDER BY date
        """
        df = self.con.execute(query, [ticker, asof_date]).df().tail(lookback)
        
        if df.empty:
            return df
        
        # Extract OHLCV and set proper DatetimeIndex
        ohlcv = df[["open", "high", "low", "close", "volume"]].copy()
        ohlcv.index = pd.to_datetime(df["date"])
        
        return ohlcv
    
    def close(self):
        self.con.close()
```

**Key points:**
- Loads from `prices` table (full history, not fold-filtered)
- Returns DataFrame with proper DatetimeIndex
- Optionally cache per `(ticker, asof_date)` for speed later

#### 2) Load Global Trading Calendar Once
**Purpose:** Generate future timestamps that respect real market trading days and do not break near fold boundaries.

**File:** `src/data/trading_calendar.py`

```python
import duckdb
import pandas as pd

def load_global_trading_calendar(
    db_path: str = "data/features.duckdb"
) -> pd.DatetimeIndex:
    """
    Load complete trading calendar from DuckDB.
    
    CRITICAL: Must be global (all dates from prices), not fold-filtered.
    """
    con = duckdb.connect(db_path, read_only=True)
    dates_df = con.execute(
        "SELECT DISTINCT date FROM prices ORDER BY date"
    ).df()
    con.close()
    
    return pd.DatetimeIndex(pd.to_datetime(dates_df["date"]))
```

#### 3) Implement KronosAdapter (Batch-first)

**File:** `src/models/kronos_adapter.py`

**Responsibilities:**
- Instantiate Kronos correctly
- Build OHLCV sequences with DatetimeIndex
- Generate y_timestamp via global calendar
- Score a universe for one asof_date via `predict_batch()`
- Map forecasted close → price return proxy (`kronos_score`)

**Key constraint:** batch requires consistent lookback + pred_len across series.

```python
from typing import List
import pandas as pd
import numpy as np
from model import Kronos, KronosTokenizer, KronosPredictor
import logging

logger = logging.getLogger(__name__)

class KronosAdapter:
    """
    Adapter for Kronos foundation model with batch inference.
    
    Key features:
    - Correct constructor pattern (per Kronos README)
    - Global trading calendar (not fold-filtered)
    - Batch inference for scalability
    """
    
    def __init__(
        self,
        trading_calendar: pd.DatetimeIndex,
        device: str = "cpu",
        tokenizer_id: str = "NeoQuasar/Kronos-Tokenizer-base",
        model_id: str = "NeoQuasar/Kronos-base",
        lookback: int = 252,
        max_context: int = 512
    ):
        """
        Initialize Kronos adapter.
        
        Args:
            trading_calendar: GLOBAL trading calendar from DuckDB
            device: "cuda" or "cpu"
            tokenizer_id: HuggingFace tokenizer ID
            model_id: HuggingFace model ID
            lookback: Historical sequence length
            max_context: Max context length for predictor
        """
        # Correct constructor pattern (per Kronos README)
        tokenizer = KronosTokenizer.from_pretrained(tokenizer_id)
        model = Kronos.from_pretrained(model_id)
        
        if device == "cuda":
            model = model.cuda()
        
        self.predictor = KronosPredictor(
            model=model,
            tokenizer=tokenizer,
            max_context=max_context
        )
        
        self.calendar = trading_calendar
        self.lookback = lookback
        self.device = device
    
    def get_future_dates(
        self, 
        last_x_date: pd.Timestamp, 
        horizon: int
    ) -> pd.DatetimeIndex:
        """
        Get next N trading days after last_x_date.
        
        IMPORTANT: Uses actual last x_timestamp, not asof_date.
        """
        last_x_date = pd.Timestamp(last_x_date)
        idx = np.searchsorted(
            self.calendar.values, 
            last_x_date.to_datetime64()
        )
        future = self.calendar[idx + 1 : idx + 1 + horizon]
        return pd.DatetimeIndex(future)
    
    def score_universe_batch(
        self,
        prices_store,
        tickers: List[str],
        asof_date,
        horizon: int
    ) -> pd.DataFrame:
        """
        Score all tickers for one date using BATCH inference.
        
        Args:
            prices_store: PricesStore instance (fetches from DuckDB)
            tickers: List of tickers to score
            asof_date: Date to score at
            horizon: Forecast horizon in trading days
        
        Returns:
            DataFrame with columns: [ticker, kronos_score]
        """
        df_list, x_ts_list, valid = [], [], []
        
        for t in tickers:
            ohlcv = prices_store.fetch_ohlcv(t, asof_date, lookback=self.lookback)
            
            # Skip if wrong length (batch requires consistent lookback)
            if len(ohlcv) != self.lookback:
                continue
            
            df_list.append(ohlcv)
            x_ts_list.append(ohlcv.index)
            valid.append(t)
        
        if not df_list:
            return pd.DataFrame(columns=["ticker", "kronos_score"])
        
        # CRITICAL: Future dates must match the last x timestamp (not asof_date!)
        # In case of missing data, last observed date may differ from asof_date
        last_x = x_ts_list[0][-1]
        y_ts = self.get_future_dates(last_x, horizon)
        y_ts_list = [y_ts] * len(df_list)
        
        # Batch prediction (single call for all tickers)
        pred_list = self.predictor.predict_batch(
            df_list=df_list,
            x_timestamp_list=x_ts_list,
            y_timestamp_list=y_ts_list,
            pred_len=horizon,
            T=0.0,        # Deterministic for Phase 1
            top_p=1.0,
            sample_count=1,
            verbose=False
        )
        
        # Compute returns for all predictions
        scores = []
        for t, x_df, pred_df in zip(valid, df_list, pred_list):
            cur = x_df["close"].iloc[-1]
            fut = pred_df["close"].iloc[-1]
            price_return = (fut - cur) / cur
            
            scores.append({
                "ticker": t,
                "kronos_score": price_return
            })
        
        return pd.DataFrame(scores)
```

#### 4) Scoring Function Must Score Every Validation Date

**Purpose:** Match `run_experiment()` contract: return EvaluationRow format for every row in validation `features_df`.

**File:** `scripts/run_chapter8_kronos.py`

**Contract:**
- **Inputs:** `(features_df, fold_id, horizon)`
- **Output:** DataFrame with one row per `(as_of_date, ticker)` in `features_df`

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from src.data.prices_store import PricesStore
from src.data.trading_calendar import load_global_trading_calendar
from src.models.kronos_adapter import KronosAdapter
from src.evaluation.run_evaluation import run_experiment, ExperimentSpec, FULL_MODE, SMOKE_MODE
from src.evaluation.data_loader import load_features_from_duckdb
import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global caches (avoid reload per fold)
_prices_store = None
_kronos_adapter = None

def get_prices_store(db_path: str = "data/features.duckdb") -> PricesStore:
    """Get or create cached PricesStore."""
    global _prices_store
    if _prices_store is None:
        _prices_store = PricesStore(db_path)
    return _prices_store

def get_kronos_adapter(device: str = "cpu") -> KronosAdapter:
    """Get or create cached Kronos adapter with global trading calendar."""
    global _kronos_adapter
    
    if _kronos_adapter is None:
        # Load GLOBAL trading calendar (not fold-filtered!)
        trading_calendar = load_global_trading_calendar("data/features.duckdb")
        
        _kronos_adapter = KronosAdapter(
            trading_calendar=trading_calendar,
            device=device,
            tokenizer_id="NeoQuasar/Kronos-Tokenizer-base",
            model_id="NeoQuasar/Kronos-base",
            lookback=252,
            max_context=512
        )
    
    return _kronos_adapter

def kronos_scoring_function(
    features_df: pd.DataFrame,
    fold_id: str,
    horizon: int
) -> pd.DataFrame:
    """
    Score all (date, ticker) pairs in validation period.
    
    This is the contract expected by run_experiment().
    
    Args:
        features_df: Validation data for this fold (already filtered)
        fold_id: Fold identifier (e.g., "fold_01")
        horizon: Forecast horizon in trading days
    
    Returns:
        DataFrame in EvaluationRow format with columns:
        - as_of_date, ticker, stable_id, horizon, fold_id
        - score (kronos_score)
        - excess_return
    """
    # Get cached instances
    prices_store = get_prices_store()
    adapter = get_kronos_adapter(device="cpu")  # Use "cuda" if available
    
    # Get unique dates in validation period
    unique_dates = pd.Series(features_df["date"].unique()).sort_values().tolist()
    
    results = []
    
    for asof_date in unique_dates:
        logger.info(f"Scoring {asof_date} for horizon {horizon}d (fold {fold_id})")
        
        # Get tickers for this date
        date_df = features_df[features_df["date"] == asof_date]
        tickers = date_df["ticker"].unique().tolist()
        
        # Score all tickers for this date (batch inference)
        scores_df = adapter.score_universe_batch(
            prices_store=prices_store,
            tickers=tickers,
            asof_date=asof_date,
            horizon=horizon
        )
        
        # Merge with labels and format as EvaluationRow
        merged = date_df[[
            "date", "ticker", "stable_id", 
            f"excess_return_{horizon}d"
        ]].merge(
            scores_df,
            on="ticker",
            how="left"
        ).rename(columns={
            "date": "as_of_date",
            "kronos_score": "score",
            f"excess_return_{horizon}d": "excess_return"
        })
        
        merged["fold_id"] = fold_id
        merged["horizon"] = horizon
        
        results.append(merged)
    
    return pd.concat(results, ignore_index=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["SMOKE", "FULL"], default="SMOKE")
    parser.add_argument("--device", default="cpu", help="cuda or cpu")
    args = parser.parse_args()
    
    # Load data
    logger.info("Loading features from DuckDB...")
    features_df, labels_df, regime_df = load_features_from_duckdb(
        db_path="data/features.duckdb"
    )
    
    # Create experiment spec
    spec = ExperimentSpec(
        name="kronos_zeroshot",
        model_type="model",
        model_name="kronos_v0",
        horizons=[20, 60, 90],
        cadence="monthly"
    )
    
    # Set mode
    mode = FULL_MODE if args.mode == "FULL" else SMOKE_MODE
    
    # Run evaluation
    output_dir = Path(f"evaluation_outputs/chapter8_kronos_{args.mode.lower()}")
    
    logger.info(f"Running {args.mode} evaluation...")
    results = run_experiment(
        experiment_spec=spec,
        features_df=features_df,
        output_dir=output_dir,
        mode=mode,
        scorer_fn=kronos_scoring_function
    )
    
    logger.info(f"\n{'='*70}")
    logger.info(f"Kronos Zero-Shot Evaluation Complete")
    logger.info(f"{'='*70}")
    logger.info(f"Results saved to: {output_dir}")
    logger.info(f"\nMedian RankIC by horizon:")
    
    for h in [20, 60, 90]:
        fold_sum = results['fold_summaries']
        ic = fold_sum[fold_sum['horizon'] == h]['median_rankic'].median()
        logger.info(f"  {h}d: {ic:.4f}")

if __name__ == "__main__":
    main()
```

**Implementation notes:**
- Iterate unique rebalance dates from `features_df["date"]`
- For each date: score all tickers via `adapter.score_universe_batch()`
- Merge into EvaluationRow format
- Cache adapter + prices_store globally (avoid reload per fold)

---

## Phase 2: FULL Evaluation

Run FULL mode with the batch adapter + correct contract.

```bash
# Run FULL evaluation (prefer GPU for speed)
python scripts/run_chapter8_kronos.py --mode FULL --device cuda

# Or CPU (slower but works)
python scripts/run_chapter8_kronos.py --mode FULL --device cpu
```

---

## Comparisons vs Frozen Baselines (Correct Loaders)

```python
import json
import pandas as pd
from pathlib import Path

# Chapter 6 baseline floor (JSON)
ch6_floor = json.loads(
    Path("evaluation_outputs/chapter6_closure_real/BASELINE_FLOOR.json").read_text()
)

# Chapter 7 ML baseline (CSV)
ch7_ml = pd.read_csv(
    "evaluation_outputs/chapter7_tabular_lgb_full/fold_summaries.csv"
)

# Chapter 8 Kronos (CSV)
kronos = pd.read_csv(
    "evaluation_outputs/chapter8_kronos_full/fold_summaries.csv"
)

# Compare median RankIC
for h in [20, 60, 90]:
    factor_ic = ch6_floor['horizons'][str(h)]['best']['median_rankic']
    ml_ic = ch7_ml[ch7_ml['horizon'] == h]['median_rankic'].median()
    kronos_ic = kronos[kronos['horizon'] == h]['median_rankic'].median()
    
    print(f"{h}d horizon:")
    print(f"  Factor baseline: {factor_ic:.4f}")
    print(f"  ML baseline (LGB): {ml_ic:.4f}")
    print(f"  Kronos (zero-shot): {kronos_ic:.4f}")
    print(f"  Lift vs factor: {kronos_ic - factor_ic:+.4f}")
    print(f"  Lift vs ML: {kronos_ic - ml_ic:+.4f}")
```

---

## Return Mapping (Explicit)

**Labels are `excess_return_{horizon}d`.**

**Kronos predicts future prices.**

We use **price return as a cross-sectional ranking proxy:**

```
kronos_score = (pred_close - current_close) / current_close
```

**This is acceptable for ranking because:**
- We care about ordering (not absolute values)
- Dividend/benchmark components are smaller and more uniform cross-sectionally
- For short horizons (20-90d), price return dominates total return

---

## Acceptance Criteria

**Updated gates with corrected implementation:**

### Gate 1: Zero-Shot Performance
- [ ] Kronos runs end-to-end with no contract violations
- [ ] RankIC ≥ 0.02 for ≥2 horizons
- [ ] Signal not redundant with momentum (corr < 0.5)

### Gate 2: ML Comparison
- [ ] Any horizon RankIC ≥ 0.05, OR
- [ ] Within 0.03 of LGB baseline on any horizon

### Gate 3: Practical Viability
- [ ] Churn ≤ 0.30
- [ ] Cost survival acceptable (≥30% positive for 60d/90d)
- [ ] Stable across VIX regimes (no catastrophic collapse)

---

## Files to Create / Modify

**New files:**
- `src/data/prices_store.py` - DuckDB-backed OHLCV fetcher
- `src/data/trading_calendar.py` - Global calendar loader
- `src/models/kronos_adapter.py` - Batch scoring adapter
- `scripts/run_chapter8_kronos.py` - Evaluation runner
- `tests/test_kronos_adapter.py` - Unit tests

**Existing:** Evaluation pipeline remains frozen (Chapter 6/7)

---

## Status

✅ **READY TO START** (post-critical-fixes)

**All critical issues resolved:**
1. ✅ Correct Kronos constructor pattern
2. ✅ Global trading calendar from DuckDB
3. ✅ Batch inference for scalability
4. ✅ Correct scorer contract (all validation dates)
5. ✅ Correct data source (PricesStore, not features_df)
6. ✅ CSV/JSON loader fixes
7. ✅ Return mapping documented
8. ✅ Future dates from last x_timestamp

---

**Next Action:** Proceed to implementation with corrected API + scalable inference.

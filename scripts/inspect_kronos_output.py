#!/usr/bin/env python
"""
Direct Kronos Output Inspection
===============================

This script runs a SINGLE Kronos prediction and shows EXACTLY
what the model returns. Critical for debugging.

Key questions answered:
1. How many rows in pred_df? (should be = horizon)
2. What are the column values?
3. Is the close progression reasonable?
4. Does pred_close match expected scale?

Usage:
    python scripts/inspect_kronos_output.py
"""

# CRITICAL: Disable MPS before any PyTorch imports
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def inspect_single_prediction():
    """Run and inspect a single Kronos prediction."""
    
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║          DIRECT KRONOS OUTPUT INSPECTION                             ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    print()
    
    from src.models.kronos_adapter import KronosAdapter, KRONOS_AVAILABLE
    
    if not KRONOS_AVAILABLE:
        print("❌ Kronos not available. Install it first.")
        return
    
    # Load adapter
    print("Loading Kronos adapter...")
    adapter = KronosAdapter.from_pretrained(
        db_path='data/features.duckdb',
        use_stub=False,
        per_ticker_timeout=180,  # 3 min timeout for deep inspection
    )
    
    # Test parameters
    ticker = 'AAPL'
    asof_date = pd.Timestamp('2024-02-01')
    horizon = 20
    
    print(f"\n--- Test Parameters ---")
    print(f"Ticker: {ticker}")
    print(f"As-of Date: {asof_date}")
    print(f"Horizon: {horizon} trading days")
    
    # 1. Fetch OHLCV
    print("\n" + "="*70)
    print("STEP 1: INPUT DATA (OHLCV)")
    print("="*70)
    
    ohlcv = adapter.prices_store.fetch_ohlcv(
        ticker=ticker,
        asof_date=asof_date,
        lookback=252,
        strict_lookback=True,
        fill_missing=True
    )
    
    print(f"\nOHLCV shape: {ohlcv.shape}")
    print(f"Date range: {ohlcv.index[0]} to {ohlcv.index[-1]}")
    print(f"Columns: {ohlcv.columns.tolist()}")
    
    print(f"\nLast 5 rows of input:")
    print(ohlcv.tail())
    
    last_close = ohlcv["close"].iloc[-1]
    last_date = ohlcv.index[-1]
    print(f"\n>>> KEY: Last input close = {last_close:.2f} on {last_date}")
    
    # 2. Future timestamps
    print("\n" + "="*70)
    print("STEP 2: FUTURE TIMESTAMPS (Y_TIMESTAMP)")
    print("="*70)
    
    y_ts = adapter.get_future_dates(last_date, horizon)
    
    print(f"\nFuture timestamps (horizon={horizon}):")
    print(f"  First: {y_ts[0]}")
    print(f"  Last:  {y_ts[-1]}")
    print(f"  Count: {len(y_ts)}")
    
    if len(y_ts) != horizon:
        print(f"\n⚠️  WARNING: Expected {horizon} future dates, got {len(y_ts)}!")
    
    # 3. Run prediction
    print("\n" + "="*70)
    print("STEP 3: KRONOS PREDICTION")
    print("="*70)
    print(f"\nRunning prediction... (may take 30-60s)")
    
    # Convert timestamps to Series (as Kronos expects)
    x_ts = pd.Series(ohlcv.index.values, index=range(len(ohlcv.index)))
    y_ts_series = pd.Series(y_ts.values, index=range(len(y_ts)))
    
    # Get predictor
    if hasattr(adapter.predictor, '_predictor'):
        predictor = adapter.predictor._predictor
    else:
        predictor = adapter.predictor
    
    print(f"Predictor type: {type(predictor).__name__}")
    
    import time
    start = time.time()
    
    with torch.inference_mode():
        pred_list = predictor.predict_batch(
            df_list=[ohlcv],
            x_timestamp_list=[x_ts],
            y_timestamp_list=[y_ts_series],
            pred_len=horizon,
            T=0.0,
            top_p=1.0,
            sample_count=1,
            verbose=True,
        )
    
    elapsed = time.time() - start
    print(f"\n✓ Prediction completed in {elapsed:.1f}s")
    
    pred_df = pred_list[0]
    
    # 4. Inspect output
    print("\n" + "="*70)
    print("STEP 4: PREDICTION OUTPUT STRUCTURE")
    print("="*70)
    
    print(f"\npred_df type: {type(pred_df)}")
    print(f"pred_df shape: {pred_df.shape}")
    print(f"pred_df columns: {pred_df.columns.tolist()}")
    print(f"pred_df index type: {type(pred_df.index)}")
    
    print(f"\nFull prediction DataFrame:")
    print(pred_df.to_string())
    
    # 5. Analyze close progression
    print("\n" + "="*70)
    print("STEP 5: CLOSE PRICE PROGRESSION")
    print("="*70)
    
    print(f"\nInput last close: {last_close:.2f}")
    print(f"\nPredicted closes (step-by-step):")
    
    for i, (idx, row) in enumerate(pred_df.iterrows()):
        pred_close = row['close']
        change_vs_spot = (pred_close - last_close) / last_close
        
        if i == 0:
            step_change = 0
        else:
            prev_close = pred_df.iloc[i-1]['close']
            step_change = (pred_close - prev_close) / prev_close
        
        print(f"  Step {i+1:2d}: close={pred_close:8.2f}, vs_spot={change_vs_spot:+7.2%}, step_change={step_change:+6.2%}")
    
    # 6. Final score calculation
    print("\n" + "="*70)
    print("STEP 6: SCORE CALCULATION (AS IN ADAPTER)")
    print("="*70)
    
    final_pred_close = pred_df["close"].iloc[-1]
    score = (final_pred_close - last_close) / last_close
    
    print(f"\n>>> Calculation:")
    print(f"    spot_close = {last_close:.2f} (last input close)")
    print(f"    pred_close = {final_pred_close:.2f} (pred_df['close'].iloc[-1])")
    print(f"    score = ({final_pred_close:.2f} - {last_close:.2f}) / {last_close:.2f}")
    print(f"    score = {score:.4f} = {score*100:.2f}%")
    
    # 7. Sanity checks
    print("\n" + "="*70)
    print("STEP 7: SANITY CHECKS")
    print("="*70)
    
    issues = []
    
    # Check 1: pred_df shape
    if pred_df.shape[0] != horizon:
        issues.append(f"❌ pred_df has {pred_df.shape[0]} rows, expected {horizon}")
    else:
        print(f"✅ pred_df has correct shape ({horizon} rows)")
    
    # Check 2: columns
    expected_cols = {'open', 'high', 'low', 'close', 'volume'}
    actual_cols = set(pred_df.columns)
    if expected_cols.issubset(actual_cols):
        print(f"✅ All expected columns present")
    else:
        missing = expected_cols - actual_cols
        issues.append(f"❌ Missing columns: {missing}")
    
    # Check 3: price scale
    if pred_df['close'].iloc[-1] > 0.1 and pred_df['close'].iloc[-1] < 10000:
        print(f"✅ Predicted prices in reasonable range")
    else:
        issues.append(f"⚠️  Unusual price scale: {pred_df['close'].iloc[-1]:.2f}")
    
    # Check 4: monotonic decline?
    close_changes = pred_df['close'].diff().dropna()
    n_down = (close_changes < 0).sum()
    n_up = (close_changes > 0).sum()
    pct_down = n_down / len(close_changes)
    
    if pct_down > 0.8:
        issues.append(f"⚠️  {pct_down*100:.0f}% of steps are DECREASING (strong downward bias)")
    print(f"📊 Step-by-step: {n_up} up, {n_down} down ({pct_down*100:.0f}% decreasing)")
    
    # Check 5: magnitude of changes
    total_change = (pred_df['close'].iloc[-1] - pred_df['close'].iloc[0]) / pred_df['close'].iloc[0]
    print(f"📊 Total change (first to last pred): {total_change*100:.2f}%")
    
    if issues:
        print("\n⚠️  ISSUES FOUND:")
        for issue in issues:
            print(f"   {issue}")
    else:
        print("\n✅ No obvious issues with prediction structure")
    
    # 8. Interpretation
    print("\n" + "="*70)
    print("STEP 8: INTERPRETATION")
    print("="*70)
    
    print(f"""
What we see:
- Input: {ticker} OHLCV ending {last_date}, last close = ${last_close:.2f}
- Horizon: {horizon} trading days
- Prediction: ${final_pred_close:.2f} → {score*100:.2f}% return

Questions to investigate:
1. Does pred_df have {horizon} rows? {'✅ Yes' if pred_df.shape[0] == horizon else '❌ No - ' + str(pred_df.shape[0]) + ' rows'}
2. Is there a strong downward bias? {'⚠️  Yes - most steps decrease' if pct_down > 0.6 else '✅ No - balanced'}
3. Is the scale correct? {'✅ Appears correct' if 10 < final_pred_close < 500 else '⚠️  May have scale issue'}

If there's a strong downward bias in EVERY prediction:
- This is likely Kronos's mean-reversion prior
- Model may have learned "after big up-move, expect pullback"
- Feb-Apr 2024 was during a massive AI rally (NVDA, META, etc.)
- Kronos doesn't know about fundamental regime shifts
""")

    # 9. Compare to actual
    print("\n" + "="*70)
    print("STEP 9: WHAT ACTUALLY HAPPENED?")
    print("="*70)
    
    try:
        # Get actual 20d forward price
        actual_future_date = y_ts[-1]
        
        # Fetch actual price 20 days later
        future_prices = adapter.prices_store.fetch_ohlcv(
            ticker=ticker,
            asof_date=actual_future_date,
            lookback=5,
            strict_lookback=False,
            fill_missing=True
        )
        
        if len(future_prices) > 0:
            actual_future_close = future_prices["close"].iloc[-1]
            actual_return = (actual_future_close - last_close) / last_close
            
            print(f"\n>>> ACTUAL vs PREDICTED:")
            print(f"    {ticker} on {last_date}: ${last_close:.2f}")
            print(f"    ")
            print(f"    PREDICTED ({horizon}d horizon): ${final_pred_close:.2f} ({score*100:+.2f}%)")
            print(f"    ACTUAL    (as of ~{actual_future_date.date()}): ${actual_future_close:.2f} ({actual_return*100:+.2f}%)")
            print(f"    ")
            print(f"    Prediction error: {(score - actual_return)*100:.2f}pp")
            
            if score * actual_return > 0:
                print(f"    Direction: ✅ CORRECT (both {'positive' if score > 0 else 'negative'})")
            else:
                print(f"    Direction: ❌ WRONG (predicted {'up' if score > 0 else 'down'}, actual {'up' if actual_return > 0 else 'down'})")
        else:
            print(f"⚠️  Could not fetch actual future prices")
            
    except Exception as e:
        print(f"⚠️  Could not compare to actual: {e}")


if __name__ == "__main__":
    inspect_single_prediction()


#!/usr/bin/env python
"""
Test Kronos with Paper's Configuration
======================================

Key differences from our original setup:
1. lookback=90 (not 252)
2. horizon=10 (not 20)  
3. T=0.6 (not 0.0) - stochastic sampling
4. sample_count=5 (not 1) - multiple samples averaged
5. Signal = normalized_pred - normalized_current (not de-normalized)

This tests if the configuration mismatch is causing poor results.
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
from scipy.stats import spearmanr
import torch
import duckdb

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import PricesStore, load_global_trading_calendar


def test_paper_config():
    """Test Kronos with configuration matching the paper."""
    
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║       TESTING KRONOS WITH PAPER'S CONFIGURATION                      ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    
    # Paper's config
    LOOKBACK = 90   # Paper uses 90, we used 252
    HORIZON = 10    # Paper uses 10, we used 20
    TEMP = 0.6      # Paper uses 0.6, we used 0.0
    SAMPLE_COUNT = 5  # Paper uses 5, we used 1
    
    print(f"\nPaper's Config:")
    print(f"  lookback:     {LOOKBACK} days (we used 252)")
    print(f"  horizon:      {HORIZON} days (we used 20)")
    print(f"  temperature:  {TEMP} (we used 0.0)")
    print(f"  sample_count: {SAMPLE_COUNT} (we used 1)")
    
    # Try to import Kronos
    try:
        from model import Kronos, KronosTokenizer, KronosPredictor
        print("\n✓ Kronos imported successfully")
    except ImportError:
        print("\n❌ Kronos not available")
        return
    
    # Load models
    print("\nLoading Kronos models...")
    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    model = Kronos.from_pretrained("NeoQuasar/Kronos-base")
    
    model.eval()
    model = model.cpu()
    for param in model.parameters():
        param.requires_grad_(False)
    
    predictor = KronosPredictor(model, tokenizer, device="cpu", max_context=512)
    
    # Load price data
    prices_store = PricesStore(db_path='data/features.duckdb', enable_cache=True)
    calendar = load_global_trading_calendar('data/features.duckdb')
    
    # Test parameters
    TEST_DATE = '2024-02-01'
    TEST_TICKERS = ['AAPL', 'MSFT', 'NVDA', 'GOOGL', 'META', 'AMZN', 'AMD', 'INTC', 'CSCO', 'ORCL']
    
    print(f"\nTest setup:")
    print(f"  Date: {TEST_DATE}")
    print(f"  Tickers: {len(TEST_TICKERS)}")
    
    # Prepare data and predictions
    results_our_way = []
    results_paper_way = []
    
    print("\n" + "="*70)
    print("Running predictions...")
    print("="*70)
    
    for ticker in TEST_TICKERS:
        print(f"\n--- {ticker} ---")
        
        # Fetch OHLCV with paper's lookback
        ohlcv = prices_store.fetch_ohlcv(
            ticker=ticker,
            asof_date=pd.Timestamp(TEST_DATE),
            lookback=LOOKBACK,  # Paper uses 90
            strict_lookback=False,  # Allow less if needed
            fill_missing=True
        )
        
        if len(ohlcv) < LOOKBACK:
            print(f"  Skipping: only {len(ohlcv)} rows")
            continue
        
        # Get future dates for horizon
        last_date = ohlcv.index[-1]
        cal_idx = np.searchsorted(calendar, last_date)
        future_dates = calendar[cal_idx + 1 : cal_idx + 1 + HORIZON]
        
        if len(future_dates) < HORIZON:
            print(f"  Skipping: insufficient future dates")
            continue
        
        spot_close = ohlcv["close"].iloc[-1]
        
        # Prepare timestamps
        x_timestamp = pd.Series(ohlcv.index.values, index=range(len(ohlcv.index)))
        y_timestamp = pd.Series(future_dates.values, index=range(len(future_dates)))
        
        # Run prediction with paper's sampling config
        print(f"  Running with T={TEMP}, samples={SAMPLE_COUNT}...")
        
        try:
            with torch.inference_mode():
                pred_df = predictor.predict(
                    df=ohlcv,
                    x_timestamp=x_timestamp,
                    y_timestamp=y_timestamp,
                    pred_len=HORIZON,
                    T=TEMP,           # Paper's temperature
                    top_p=0.9,        # Paper's top_p
                    sample_count=SAMPLE_COUNT,  # Paper's sample count
                    verbose=False,
                )
            
            pred_close = pred_df["close"].iloc[-1]
            
            # OUR WAY: score = (pred - spot) / spot
            our_score = (pred_close - spot_close) / spot_close
            
            # PAPER'S WAY: signal in normalized space
            # The predictor internally normalizes and de-normalizes
            # To get paper's signal, we compute: (pred_norm - current_norm)
            # Which equals: (pred - mean) / std - (current - mean) / std
            # = (pred - current) / std
            
            x_close = ohlcv["close"].values
            x_mean = x_close.mean()
            x_std = x_close.std() + 1e-5
            
            paper_signal = (pred_close - spot_close) / x_std
            
            print(f"  spot_close:   ${spot_close:.2f}")
            print(f"  pred_close:   ${pred_close:.2f}")
            print(f"  our_score:    {our_score:.2%}")
            print(f"  paper_signal: {paper_signal:.4f}")
            
            results_our_way.append({
                'ticker': ticker,
                'spot_close': spot_close,
                'pred_close': pred_close,
                'score': our_score,
            })
            
            results_paper_way.append({
                'ticker': ticker,
                'signal': paper_signal,
            })
            
        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
    
    if not results_our_way:
        print("\n❌ No predictions generated")
        return
    
    # Convert to DataFrames
    df_our = pd.DataFrame(results_our_way)
    df_paper = pd.DataFrame(results_paper_way)
    
    print("\n" + "="*70)
    print("RESULTS COMPARISON")
    print("="*70)
    
    print("\nOur method (score = (pred - spot) / spot):")
    print(df_our.to_string(index=False))
    
    print(f"\nScore stats:")
    print(f"  Mean: {df_our['score'].mean():.4f}")
    print(f"  Std:  {df_our['score'].std():.4f}")
    print(f"  Positive: {(df_our['score'] > 0).sum()}/{len(df_our)}")
    
    # Get actual forward returns
    print("\n" + "="*70)
    print("COMPARING TO ACTUAL RETURNS")
    print("="*70)
    
    conn = duckdb.connect('data/features.duckdb', read_only=True)
    
    # Get 10d forward returns from labels (or compute from prices)
    try:
        # For horizon=10, we need to check if we have labels
        # Actually let's compute from prices directly
        actual_returns = []
        for ticker in df_our['ticker']:
            # Get price 10 days later
            future_price = conn.execute(f"""
                SELECT date, close FROM prices
                WHERE ticker = '{ticker}'
                AND date > '{TEST_DATE}'
                ORDER BY date
                LIMIT 15
            """).df()
            
            if len(future_price) >= HORIZON:
                spot = df_our[df_our['ticker'] == ticker]['spot_close'].iloc[0]
                future_close = future_price.iloc[HORIZON-1]['close']
                actual_return = (future_close - spot) / spot
                actual_returns.append({
                    'ticker': ticker,
                    'actual_return': actual_return
                })
        
        conn.close()
        
        if actual_returns:
            df_actual = pd.DataFrame(actual_returns)
            df_merged = df_our.merge(df_actual, on='ticker')
            
            print("\nPredictions vs Actual:")
            print(f"{'Ticker':<8} {'Score':>10} {'Actual':>10} {'Match?':>8}")
            print("-"*40)
            for _, row in df_merged.iterrows():
                match = (row['score'] > 0 and row['actual_return'] > 0) or \
                        (row['score'] < 0 and row['actual_return'] < 0)
                print(f"{row['ticker']:<8} {row['score']:>10.2%} {row['actual_return']:>10.2%} {'✅' if match else '❌'}")
            
            # Compute RankIC
            if len(df_merged) >= 5:
                rankic, pval = spearmanr(df_merged['score'], df_merged['actual_return'])
                print(f"\n10d RankIC: {rankic:.4f} (p={pval:.4f})")
                
                if rankic > 0.1:
                    print("✅ Positive RankIC with paper's config!")
                elif rankic > 0:
                    print("⚠️  Weak positive RankIC")
                else:
                    print("❌ Negative RankIC even with paper's config")
        
    except Exception as e:
        print(f"Error computing actual returns: {e}")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("""
Key Configuration Changes Tested:
  ✓ lookback = 90 (paper) vs 252 (ours)
  ✓ horizon = 10 (paper) vs 20 (ours)  
  ✓ T = 0.6 (paper) vs 0.0 (ours)
  ✓ sample_count = 5 (paper) vs 1 (ours)

Key Remaining Difference:
  ✗ Model: We use PRE-TRAINED base model
    Paper uses FINE-TUNED model on CSI300 data
  ✗ Market: US AI stocks vs Chinese A-shares (CSI300)
  
The paper FINE-TUNES Kronos on their specific market data before evaluation!
""")


if __name__ == "__main__":
    test_paper_config()


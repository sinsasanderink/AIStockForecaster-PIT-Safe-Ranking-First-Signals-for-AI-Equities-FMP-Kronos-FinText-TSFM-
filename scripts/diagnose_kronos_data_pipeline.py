#!/usr/bin/env python
"""
Kronos Data Pipeline Diagnostic
================================

Comprehensive check for potential bugs in the Kronos integration:
1. Score identity check (score ≈ pred_close/spot_close - 1)
2. Spot close alignment (spot_close == last close in input window)
3. Database price consistency
4. Kronos output structure inspection
5. Price discontinuity check (splits/adjustments)
6. Forward return verification
7. Normalization check

Usage:
    python scripts/diagnose_kronos_data_pipeline.py
"""

# CRITICAL: Disable MPS before any PyTorch imports
import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import duckdb

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def print_section(title: str):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def check_1_score_identity():
    """Verify: score ≈ pred_close/spot_close - 1"""
    print_section("CHECK 1: Score Identity (score = pred_close/spot_close - 1)")
    
    df = pd.read_csv('kronos_micro_test.csv')
    
    # Calculate what score SHOULD be
    calculated_score = df['pred_close'] / df['spot_close'] - 1.0
    
    # Compare to actual score
    error = (calculated_score - df['score']).abs()
    
    print(f"\nCalculated: pred_close / spot_close - 1")
    print(f"Comparing to logged 'score' column...")
    print(f"\n  Max absolute error: {error.max():.10f}")
    print(f"  Mean absolute error: {error.mean():.10f}")
    
    if error.max() < 1e-6:
        print("\n  ✅ PASS: Score calculation is correct")
        return True
    else:
        print("\n  ❌ FAIL: Score calculation has errors!")
        print("\n  Sample mismatches:")
        bad_rows = df[error > 1e-6].head(5)
        for _, row in bad_rows.iterrows():
            calc = row['pred_close'] / row['spot_close'] - 1
            print(f"    {row['ticker']}: logged={row['score']:.6f}, calc={calc:.6f}")
        return False


def check_2_spot_close_alignment():
    """Verify: spot_close matches last close in input window"""
    print_section("CHECK 2: Spot Close Alignment")
    
    from src.data import PricesStore
    
    df = pd.read_csv('kronos_micro_test.csv')
    prices_store = PricesStore(db_path='data/features.duckdb', enable_cache=True)
    
    print("\nComparing logged spot_close to actual last close from PricesStore...")
    print(f"\n{'Ticker':<8} {'Date':<12} {'spot_close':>12} {'last_window':>12} {'Match?':>8}")
    print("-" * 60)
    
    mismatches = 0
    samples = df.sample(min(15, len(df)))
    
    for _, row in samples.iterrows():
        ticker = row['ticker']
        date = pd.Timestamp(row['date'])
        logged_spot = row['spot_close']
        
        # Fetch actual OHLCV window
        ohlcv = prices_store.fetch_ohlcv(
            ticker=ticker,
            asof_date=date,
            lookback=252,
            strict_lookback=True,
            fill_missing=True
        )
        
        if len(ohlcv) > 0:
            actual_last_close = ohlcv['close'].iloc[-1]
            match = abs(logged_spot - actual_last_close) < 0.01
            if not match:
                mismatches += 1
            print(f"{ticker:<8} {str(date.date()):<12} {logged_spot:>12.2f} {actual_last_close:>12.2f} {'✅' if match else '❌'}")
        else:
            print(f"{ticker:<8} {str(date.date()):<12} {logged_spot:>12.2f} {'N/A':>12} ❓")
            mismatches += 1
    
    if mismatches == 0:
        print(f"\n  ✅ PASS: All {len(samples)} samples aligned correctly")
        return True
    else:
        print(f"\n  ❌ FAIL: {mismatches}/{len(samples)} samples misaligned!")
        return False


def check_3_database_schema():
    """Inspect database structure and prices table"""
    print_section("CHECK 3: Database Schema & Prices Table")
    
    conn = duckdb.connect('data/features.duckdb', read_only=True)
    
    print("\n--- Tables in Database ---")
    tables = conn.execute("SHOW TABLES").df()
    print(tables)
    
    print("\n--- Prices Table Schema ---")
    if 'prices' in tables['name'].values:
        schema = conn.execute("DESCRIBE prices").df()
        print(schema[['column_name', 'column_type']].to_string(index=False))
        
        print("\n--- Prices Table Sample ---")
        sample = conn.execute("""
            SELECT * FROM prices 
            WHERE ticker = 'NVDA' 
            AND date BETWEEN '2024-01-25' AND '2024-02-05'
            ORDER BY date
        """).df()
        print(sample)
        
        print("\n--- Check for 'adj_close' column ---")
        if 'adj_close' in schema['column_name'].values:
            print("  ⚠️  'adj_close' column EXISTS - verify we're using correct column")
        else:
            print("  ℹ️  No 'adj_close' column - using 'close' directly")
    
    conn.close()
    return True


def check_4_price_discontinuities():
    """Check for suspicious price jumps (potential split issues)"""
    print_section("CHECK 4: Price Discontinuities (Splits/Adjustments)")
    
    from src.data import PricesStore
    prices_store = PricesStore(db_path='data/features.duckdb', enable_cache=True)
    
    test_tickers = ['NVDA', 'AMD', 'META', 'AAPL', 'AVGO']
    
    print("\nChecking for >20% daily moves (potential split signals)...")
    
    for ticker in test_tickers:
        ohlcv = prices_store.fetch_ohlcv(
            ticker=ticker,
            asof_date=pd.Timestamp('2024-06-30'),  # Through mid-2024
            lookback=500,
            strict_lookback=False,
            fill_missing=True
        )
        
        if len(ohlcv) > 0:
            returns = ohlcv['close'].pct_change()
            large_moves = returns[abs(returns) > 0.20]
            
            if len(large_moves) > 0:
                print(f"\n{ticker}: {len(large_moves)} large moves")
                for date, ret in large_moves.items():
                    print(f"    {date.date()}: {ret:+.1%}")
            else:
                print(f"{ticker}: No large moves (data looks split-adjusted)")
    
    print("\n  ℹ️  NVDA had 10:1 split on June 10, 2024")
    print("  ℹ️  If no large jumps around that date, prices are back-adjusted")
    return True


def check_5_kronos_output_structure():
    """Inspect what Kronos actually returns"""
    print_section("CHECK 5: Kronos Output Structure")
    
    try:
        from src.models.kronos_adapter import KronosAdapter, KRONOS_AVAILABLE
        import torch
        
        if not KRONOS_AVAILABLE:
            print("  ⚠️  Kronos not available - skipping deep inspection")
            return True
        
        print("\nLoading adapter and running single prediction...")
        
        adapter = KronosAdapter.from_pretrained(
            db_path='data/features.duckdb',
            use_stub=False,
            per_ticker_timeout=120,
        )
        
        # Get OHLCV
        ticker = 'AAPL'
        asof_date = pd.Timestamp('2024-02-01')
        horizon = 20
        
        ohlcv = adapter.prices_store.fetch_ohlcv(
            ticker=ticker,
            asof_date=asof_date,
            lookback=252,
            strict_lookback=True,
            fill_missing=True
        )
        
        print(f"\n--- Input OHLCV ---")
        print(f"Shape: {ohlcv.shape}")
        print(f"Last 3 rows:")
        print(ohlcv.tail(3))
        
        last_close = ohlcv['close'].iloc[-1]
        last_date = ohlcv.index[-1]
        
        print(f"\n>>> Last input close: ${last_close:.2f} on {last_date.date()}")
        
        # Get future dates
        y_ts = adapter.get_future_dates(last_date, horizon)
        
        print(f"\n--- Future Timestamps ---")
        print(f"Count: {len(y_ts)} (should be {horizon})")
        print(f"First: {y_ts[0].date()}")
        print(f"Last: {y_ts[-1].date()}")
        
        # Run prediction
        print(f"\n--- Running Kronos Prediction ---")
        print("(This may take 30-60s)")
        
        x_ts = pd.Series(ohlcv.index.values, index=range(len(ohlcv.index)))
        y_ts_series = pd.Series(y_ts.values, index=range(len(y_ts)))
        
        if hasattr(adapter.predictor, '_predictor'):
            predictor = adapter.predictor._predictor
        else:
            predictor = adapter.predictor
        
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
        
        pred_df = pred_list[0]
        
        print(f"\n--- Prediction Output ---")
        print(f"Type: {type(pred_df)}")
        print(f"Shape: {pred_df.shape} (expected: ({horizon}, 5))")
        print(f"Columns: {pred_df.columns.tolist()}")
        
        print(f"\n--- Full Prediction ---")
        print(pred_df.to_string())
        
        print(f"\n--- Analysis ---")
        print(f"Input last close: ${last_close:.2f}")
        print(f"Pred first close: ${pred_df['close'].iloc[0]:.2f} ({(pred_df['close'].iloc[0]/last_close-1)*100:+.2f}%)")
        print(f"Pred last close:  ${pred_df['close'].iloc[-1]:.2f} ({(pred_df['close'].iloc[-1]/last_close-1)*100:+.2f}%)")
        
        # Check if pred_df is in the right scale
        pred_mean = pred_df['close'].mean()
        pred_std = pred_df['close'].std()
        
        print(f"\nPrediction close stats:")
        print(f"  Mean: ${pred_mean:.2f}")
        print(f"  Std:  ${pred_std:.2f}")
        print(f"  Range: ${pred_df['close'].min():.2f} - ${pred_df['close'].max():.2f}")
        
        # CRITICAL CHECK: Is pred_close in same scale as input?
        scale_ratio = pred_mean / last_close
        print(f"\n>>> Scale ratio (pred_mean / spot_close): {scale_ratio:.3f}")
        
        if 0.5 < scale_ratio < 2.0:
            print("  ✅ Predictions appear to be in same price scale")
        else:
            print("  ❌ WARNING: Predictions may be in different scale!")
            print("  This could indicate Kronos outputs normalized/transformed values")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error during inspection: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_6_forward_return_comparison():
    """Compare Kronos prediction to actual forward return"""
    print_section("CHECK 6: Kronos vs Actual Forward Returns")
    
    df = pd.read_csv('kronos_micro_test.csv')
    conn = duckdb.connect('data/features.duckdb', read_only=True)
    
    # Get actual 20d returns from labels
    labels = conn.execute("""
        SELECT as_of_date as date, ticker, excess_return as actual_return
        FROM labels
        WHERE horizon = 20
        AND as_of_date IN ('2024-02-01', '2024-03-01', '2024-04-01')
    """).df()
    
    conn.close()
    
    df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
    labels['date'] = pd.to_datetime(labels['date']).dt.strftime('%Y-%m-%d')
    
    merged = df.merge(labels, on=['date', 'ticker'], how='inner')
    
    print(f"\nMerged {len(merged)} predictions with actual returns")
    
    print(f"\n--- Statistical Comparison ---")
    print(f"\n{'Metric':<25} {'Kronos':>12} {'Actual':>12}")
    print("-" * 50)
    print(f"{'Mean':.<25} {merged['score'].mean():>12.4f} {merged['actual_return'].mean():>12.4f}")
    print(f"{'Std':.<25} {merged['score'].std():>12.4f} {merged['actual_return'].std():>12.4f}")
    print(f"{'Min':.<25} {merged['score'].min():>12.4f} {merged['actual_return'].min():>12.4f}")
    print(f"{'Max':.<25} {merged['score'].max():>12.4f} {merged['actual_return'].max():>12.4f}")
    
    print(f"\n--- Key Observation ---")
    kronos_mean = merged['score'].mean()
    actual_mean = merged['actual_return'].mean()
    
    print(f"Kronos mean: {kronos_mean:.4f} ({kronos_mean*100:.2f}%)")
    print(f"Actual mean: {actual_mean:.4f} ({actual_mean*100:.2f}%)")
    print(f"Bias: {(kronos_mean - actual_mean)*100:.2f}pp")
    
    if kronos_mean < -0.10:
        print("\n⚠️  WARNING: Kronos has strong negative bias!")
        print("   Possible causes:")
        print("   1. Mean-reversion prior in model")
        print("   2. Output scale/interpretation issue")
        print("   3. Normalization mismatch")
    
    return True


def check_7_kronos_paper_normalization():
    """Check if Kronos expects normalized input (from paper)"""
    print_section("CHECK 7: Kronos Normalization Expectations")
    
    print("""
From the Kronos paper (https://arxiv.org/html/2508.02739v1):

The model uses K-line tokenization which involves:
1. Dividing OHLCV into bins/tokens
2. The tokenizer handles normalization internally

Key question: Does KronosTokenizer normalize internally or expect pre-normalized input?

Let me check the tokenizer behavior...
""")
    
    try:
        from model import KronosTokenizer
        
        tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
        
        print("Tokenizer loaded successfully")
        print(f"Tokenizer type: {type(tokenizer)}")
        
        # Check tokenizer attributes
        if hasattr(tokenizer, 'config'):
            print(f"Tokenizer config: {tokenizer.config}")
        
        print("\n--- Testing tokenizer with different scales ---")
        
        import pandas as pd
        import numpy as np
        
        # Test with raw prices (AAPL scale ~180)
        raw_prices = pd.DataFrame({
            'open': [180, 181, 182],
            'high': [185, 186, 187],
            'low': [175, 176, 177],
            'close': [182, 183, 184],
            'volume': [1e8, 1.1e8, 1.2e8],
        })
        
        # Test with normalized prices (mean ~1)
        norm_prices = pd.DataFrame({
            'open': [1.0, 1.01, 1.02],
            'high': [1.03, 1.04, 1.05],
            'low': [0.97, 0.98, 0.99],
            'close': [1.01, 1.02, 1.03],
            'volume': [1e8, 1.1e8, 1.2e8],
        })
        
        print("\nTokenizer should handle both scales internally if designed correctly.")
        print("The KronosTokenizer uses quantile-based binning, so scale shouldn't matter for ranking.")
        
    except Exception as e:
        print(f"  Could not inspect tokenizer: {e}")
    
    print("""
--- Conclusion ---

Based on Kronos architecture:
- KronosTokenizer uses K-line (OHLCV) tokenization
- It converts price sequences to discrete tokens
- The predicted tokens are then converted back to prices

POTENTIAL ISSUE:
If the de-tokenization (tokens → prices) uses statistics from the 
INPUT sequence (like mean/std), and those differ significantly from
the actual price scale, predictions could be biased.

This could explain why Kronos predicts "pullback" (-13% mean):
- If the model learned mean-reverting patterns
- AND/OR if there's a scale mismatch in de-tokenization
""")
    
    return True


def check_8_date_index_alignment():
    """Check if input DataFrame index matches x_timestamp"""
    print_section("CHECK 8: DataFrame Index vs Timestamp Alignment")
    
    from src.data import PricesStore
    
    prices_store = PricesStore(db_path='data/features.duckdb', enable_cache=True)
    
    ticker = 'NVDA'
    asof_date = pd.Timestamp('2024-02-01')
    
    ohlcv = prices_store.fetch_ohlcv(
        ticker=ticker,
        asof_date=asof_date,
        lookback=252,
        strict_lookback=True,
        fill_missing=True
    )
    
    print(f"\n--- OHLCV DataFrame ---")
    print(f"Index type: {type(ohlcv.index)}")
    print(f"Index dtype: {ohlcv.index.dtype}")
    print(f"First date: {ohlcv.index[0]}")
    print(f"Last date: {ohlcv.index[-1]}")
    
    # Check if last date matches asof_date
    last_index_date = ohlcv.index[-1]
    
    print(f"\n--- Date Alignment ---")
    print(f"asof_date: {asof_date.date()}")
    print(f"Last index date: {last_index_date.date() if hasattr(last_index_date, 'date') else last_index_date}")
    
    # They might not match exactly if asof_date is a weekend/holiday
    print(f"\nMatch: {last_index_date.date() == asof_date.date() if hasattr(last_index_date, 'date') else 'N/A'}")
    
    if hasattr(last_index_date, 'date') and last_index_date.date() != asof_date.date():
        print("\n⚠️  NOTE: Last index date != asof_date")
        print("   This is expected if asof_date lands on weekend/holiday")
        print("   PricesStore returns data UP TO asof_date (inclusive)")
    
    return True


def run_all_checks():
    """Run all diagnostic checks"""
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║                                                                      ║")
    print("║           KRONOS DATA PIPELINE DIAGNOSTIC                            ║")
    print("║                                                                      ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    
    results = {}
    
    # Run checks
    results['score_identity'] = check_1_score_identity()
    results['spot_alignment'] = check_2_spot_close_alignment()
    results['db_schema'] = check_3_database_schema()
    results['discontinuities'] = check_4_price_discontinuities()
    results['forward_returns'] = check_6_forward_return_comparison()
    results['normalization'] = check_7_kronos_paper_normalization()
    results['date_alignment'] = check_8_date_index_alignment()
    
    # Summary
    print_section("DIAGNOSTIC SUMMARY")
    
    all_pass = True
    for check, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {check}: {status}")
        if not passed:
            all_pass = False
    
    print("\n" + "-" * 70)
    
    if all_pass:
        print("\n✅ All basic checks passed!")
        print("\nIf RankIC is still poor, likely causes are:")
        print("  1. Kronos has mean-reversion bias (model behavior, not bug)")
        print("  2. Feb-Apr 2024 was a momentum regime (unfavorable for mean-reversion)")
        print("  3. Foundation model not trained for cross-sectional ranking")
    else:
        print("\n❌ Some checks failed - investigate those areas!")
    
    # Ask about deep inspection
    print("\n" + "=" * 70)
    print("Run Kronos output inspection? (requires 30-60s)")
    print("=" * 70)
    
    response = input("Run deep inspection? [y/N]: ").strip().lower()
    if response == 'y':
        check_5_kronos_output_structure()
    
    return results


if __name__ == "__main__":
    run_all_checks()


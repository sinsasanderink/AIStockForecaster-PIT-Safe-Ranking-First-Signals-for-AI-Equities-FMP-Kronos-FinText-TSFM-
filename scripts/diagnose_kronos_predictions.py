#!/usr/bin/env python
"""
Kronos Prediction Diagnostics
=============================

This script investigates potential issues with Kronos predictions:
1. Spot close alignment (is last_x_date == asof_date?)
2. Prediction structure (how many rows, what's in pred_df?)
3. Price scale consistency
4. Compute actual RankIC vs forward returns
5. Check for systematic biases

Usage:
    python scripts/diagnose_kronos_predictions.py
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
import duckdb

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def print_section(title: str):
    """Print a section header."""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)


def diagnose_micro_test_results():
    """Analyze the micro-test CSV results."""
    print_section("1. MICRO-TEST RESULTS ANALYSIS")
    
    df = pd.read_csv('kronos_micro_test.csv')
    
    print(f"\nTotal predictions: {len(df)}")
    print(f"Unique dates: {df['date'].nunique()}")
    print(f"Unique tickers: {df['ticker'].nunique()}")
    
    # Score distribution
    print("\n--- Score Distribution ---")
    print(df['score'].describe())
    
    # Check negative bias
    n_positive = (df['score'] > 0).sum()
    n_negative = (df['score'] < 0).sum()
    print(f"\nPositive predictions: {n_positive} ({n_positive/len(df)*100:.1f}%)")
    print(f"Negative predictions: {n_negative} ({n_negative/len(df)*100:.1f}%)")
    
    # Check last_x_date alignment
    print("\n--- Date Alignment Check ---")
    print(f"Do all last_x_date == date (asof_date)?")
    
    df['last_x_date'] = pd.to_datetime(df['last_x_date'])
    df['date'] = pd.to_datetime(df['date'])
    aligned = (df['last_x_date'].dt.date == df['date'].dt.date).all()
    print(f"  Result: {'✅ YES - aligned' if aligned else '❌ NO - MISALIGNED!'}")
    
    if not aligned:
        misaligned = df[df['last_x_date'].dt.date != df['date'].dt.date]
        print(f"  Misaligned rows: {len(misaligned)}")
        print(misaligned[['ticker', 'date', 'last_x_date']].head(10))
    
    # Check n_history
    print("\n--- History Length Check ---")
    print(f"n_history values: {df['n_history'].unique()}")
    print(f"All have 252 days? {(df['n_history'] == 252).all()}")
    
    return df


def verify_spot_close_alignment():
    """Verify that spot_close matches the actual price in DuckDB."""
    print_section("2. SPOT CLOSE vs DATABASE VERIFICATION")
    
    df = pd.read_csv('kronos_micro_test.csv')
    conn = duckdb.connect('data/features.duckdb', read_only=True)
    
    # Check a few samples
    samples = df.sample(min(10, len(df)))
    
    print("\nComparing spot_close to prices table:")
    print(f"{'Ticker':<8} {'Date':<12} {'spot_close':>12} {'DB close':>12} {'Match?':>8}")
    print("-" * 60)
    
    mismatches = 0
    for _, row in samples.iterrows():
        ticker = row['ticker']
        date = row['date'][:10]  # YYYY-MM-DD
        spot_close = row['spot_close']
        
        # Get actual close from DB
        result = conn.execute(f"""
            SELECT close FROM prices
            WHERE ticker = '{ticker}' AND date = '{date}'
        """).df()
        
        if len(result) > 0:
            db_close = result['close'].iloc[0]
            match = abs(spot_close - db_close) < 0.01
            if not match:
                mismatches += 1
            print(f"{ticker:<8} {date:<12} {spot_close:>12.2f} {db_close:>12.2f} {'✅' if match else '❌'}")
        else:
            print(f"{ticker:<8} {date:<12} {spot_close:>12.2f} {'N/A':>12} {'❓ Not in DB'}")
            mismatches += 1
    
    conn.close()
    
    print(f"\nResult: {len(samples) - mismatches}/{len(samples)} prices match database")
    if mismatches > 0:
        print("⚠️  WARNING: Some spot_close values don't match database!")


def check_price_scale():
    """Check if prices look reasonable (adjusted vs unadjusted, splits, etc.)."""
    print_section("3. PRICE SCALE CHECK")
    
    df = pd.read_csv('kronos_micro_test.csv')
    
    print("\nSample predictions by ticker:")
    print(f"{'Ticker':<8} {'spot_close':>12} {'pred_close':>12} {'score':>10}")
    print("-" * 50)
    
    for _, row in df[df['date'] == '2024-02-01'].head(10).iterrows():
        print(f"{row['ticker']:<8} {row['spot_close']:>12.2f} {row['pred_close']:>12.2f} {row['score']:>10.2%}")
    
    # Check for suspicious patterns
    print("\n--- Suspicious Patterns ---")
    
    # All predictions lower?
    pct_lower = (df['pred_close'] < df['spot_close']).mean()
    print(f"% predictions below spot: {pct_lower*100:.1f}%")
    if pct_lower > 0.8:
        print("⚠️  WARNING: >80% of predictions are BELOW spot price - strong bearish bias!")
    
    # Check prediction range vs spot range
    spot_range = df['spot_close'].max() / df['spot_close'].min()
    pred_range = df['pred_close'].max() / df['pred_close'].min()
    print(f"\nSpot price range (max/min): {spot_range:.1f}x")
    print(f"Pred price range (max/min): {pred_range:.1f}x")
    
    # Check if predictions are in similar scale
    ratio = df['pred_close'] / df['spot_close']
    print(f"\npred/spot ratio: mean={ratio.mean():.3f}, std={ratio.std():.3f}")
    print(f"Range: [{ratio.min():.3f}, {ratio.max():.3f}]")


def analyze_kronos_prediction_structure():
    """Investigate what Kronos actually returns."""
    print_section("4. KRONOS PREDICTION STRUCTURE INVESTIGATION")
    
    print("""
To properly diagnose, we need to look at what Kronos returns.

In the current adapter:
- We call predict_batch() which returns a list of DataFrames
- Each DataFrame contains OHLCV predictions for the horizon period
- We take pred_df["close"].iloc[-1] as the predicted close at horizon

Questions to verify:
1. How many rows does pred_df have? (should be = horizon = 20)
2. Are all rows populated or just the last one?
3. Is the close column in the right scale?

Let me add a diagnostic print to show this...
""")


def compute_rankic_vs_actual_returns():
    """The REAL test: compute RankIC against actual forward returns."""
    print_section("5. RankIC vs ACTUAL FORWARD RETURNS")
    
    df = pd.read_csv('kronos_micro_test.csv')
    conn = duckdb.connect('data/features.duckdb', read_only=True)
    
    # Check what tables we have for labels
    print("\nChecking available tables for forward returns...")
    tables = conn.execute("SHOW TABLES").df()
    print(f"Tables: {tables['name'].tolist()}")
    
    # Try to get forward returns from labels table
    print("\nFetching actual forward returns from labels table...")
    
    try:
        # Get 20d forward returns for our test dates
        labels_df = conn.execute("""
            SELECT 
                as_of_date as date,
                ticker,
                value as actual_return
            FROM labels
            WHERE label_name = 'excess_return_20d'
            AND as_of_date IN ('2024-02-01', '2024-03-01', '2024-04-01')
        """).df()
        
        if len(labels_df) == 0:
            print("No labels found with that query. Trying alternative...")
            
            # Try different column names
            labels_df = conn.execute("""
                SELECT * FROM labels LIMIT 5
            """).df()
            print("Sample from labels table:")
            print(labels_df)
            
            # Check structure
            cols = conn.execute("DESCRIBE labels").df()
            print("\nLabels table structure:")
            print(cols)
            
        else:
            print(f"Found {len(labels_df)} label rows")
            
    except Exception as e:
        print(f"Error querying labels: {e}")
        
        # Try features table instead
        print("\nTrying features table for forward returns...")
        try:
            features_df = conn.execute("""
                SELECT date, ticker, excess_return_20d
                FROM features
                WHERE date IN ('2024-02-01', '2024-03-01', '2024-04-01')
                LIMIT 100
            """).df()
            
            if len(features_df) > 0:
                labels_df = features_df.rename(columns={'excess_return_20d': 'actual_return'})
                print(f"Found {len(labels_df)} rows in features table")
            else:
                print("No data found in features table either")
                labels_df = pd.DataFrame()
                
        except Exception as e2:
            print(f"Error: {e2}")
            labels_df = pd.DataFrame()
    
    conn.close()
    
    if len(labels_df) == 0:
        print("\n⚠️  Cannot compute RankIC - no forward return labels found!")
        print("You need actual realized returns to evaluate prediction quality.")
        return
    
    # Merge predictions with actual returns
    df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
    labels_df['date'] = pd.to_datetime(labels_df['date']).dt.strftime('%Y-%m-%d')
    
    merged = df.merge(labels_df, on=['date', 'ticker'], how='inner')
    
    print(f"\nMerged data: {len(merged)} rows")
    
    if len(merged) < 5:
        print("⚠️  Too few merged rows for meaningful RankIC calculation")
        return
    
    # Compute RankIC per date
    print("\n--- RankIC by Date ---")
    print(f"{'Date':<12} {'RankIC':>10} {'p-value':>10} {'n':>6}")
    print("-" * 45)
    
    per_date_ics = []
    for date in sorted(merged['date'].unique()):
        date_df = merged[merged['date'] == date]
        if len(date_df) >= 5:
            ic, pval = spearmanr(date_df['score'], date_df['actual_return'])
            per_date_ics.append(ic)
            print(f"{date:<12} {ic:>10.4f} {pval:>10.4f} {len(date_df):>6}")
    
    # Overall
    overall_ic, overall_p = spearmanr(merged['score'], merged['actual_return'])
    
    print("-" * 45)
    print(f"{'OVERALL':<12} {overall_ic:>10.4f} {overall_p:>10.4f} {len(merged):>6}")
    
    # Compare to baseline
    baseline_20d = 0.1009
    print(f"\n--- Comparison to Baseline ---")
    print(f"Your LGB baseline 20d RankIC: {baseline_20d:.4f}")
    print(f"Kronos 20d RankIC:            {overall_ic:.4f}")
    print(f"Difference:                   {overall_ic - baseline_20d:+.4f}")
    
    # Interpretation
    print("\n--- Interpretation ---")
    if overall_ic > 0.05:
        print("✅ GOOD: Kronos shows meaningful positive RankIC!")
    elif overall_ic > 0.02:
        print("⚠️  WEAK: Kronos shows weak positive RankIC (may be noise)")
    elif overall_ic > -0.02:
        print("❌ NO SIGNAL: Kronos RankIC is essentially zero")
    else:
        print("❌ NEGATIVE: Kronos rankings are INVERSELY correlated with returns!")
        print("   This could mean:")
        print("   1. Model is genuinely wrong (mean-reversion bias)")
        print("   2. Price scale mismatch (model expects different units)")
        print("   3. Horizon mismatch (not actually predicting 20d ahead)")


def check_cross_sectional_ranking_stability():
    """Check if rankings are stable across dates."""
    print_section("6. RANKING STABILITY ACROSS DATES")
    
    df = pd.read_csv('kronos_micro_test.csv')
    
    # Pivot to get rankings per date
    pivot = df.pivot(index='ticker', columns='date', values='score')
    
    # Compute rank correlations between dates
    print("\nRank correlation between dates:")
    print(pivot.corr(method='spearman').round(3))
    
    # Check if same tickers consistently ranked high/low
    print("\n--- Ticker Consistency ---")
    
    # Get average rank per ticker
    ranks = pivot.rank(ascending=False)  # Higher score = lower rank (better)
    avg_rank = ranks.mean(axis=1).sort_values()
    
    print("\nMost consistently BULLISH predictions (lowest avg rank):")
    print(avg_rank.head(5))
    
    print("\nMost consistently BEARISH predictions (highest avg rank):")
    print(avg_rank.tail(5))


def investigate_kronos_output_directly():
    """Run a single Kronos prediction and inspect the output structure."""
    print_section("7. DIRECT KRONOS OUTPUT INSPECTION")
    
    print("""
Let me create a minimal test to see exactly what Kronos returns...
""")
    
    try:
        from src.models.kronos_adapter import KronosAdapter
        import torch
        
        print("Loading adapter...")
        adapter = KronosAdapter.from_pretrained(
            db_path='data/features.duckdb',
            use_stub=False,
            per_ticker_timeout=120,  # Give it more time
        )
        
        # Get OHLCV for a single ticker
        print("\nFetching OHLCV for AAPL @ 2024-02-01...")
        ohlcv = adapter.prices_store.fetch_ohlcv(
            ticker='AAPL',
            asof_date=pd.Timestamp('2024-02-01'),
            lookback=252,
            strict_lookback=True,
            fill_missing=True
        )
        
        print(f"\n--- Input OHLCV ---")
        print(f"Shape: {ohlcv.shape}")
        print(f"Date range: {ohlcv.index[0]} to {ohlcv.index[-1]}")
        print(f"Last 5 rows:")
        print(ohlcv.tail())
        
        # Check last close
        last_close = ohlcv["close"].iloc[-1]
        last_date = ohlcv.index[-1]
        print(f"\nLast close: {last_close:.2f} on {last_date}")
        
        # Get future dates
        horizon = 20
        y_ts = adapter.get_future_dates(last_date, horizon)
        print(f"\n--- Future Timestamps (horizon={horizon}) ---")
        print(f"First future date: {y_ts[0]}")
        print(f"Last future date: {y_ts[-1]}")
        print(f"Total future dates: {len(y_ts)}")
        
        # Now run prediction and inspect output
        print("\n--- Running Kronos Prediction ---")
        print("(This will take ~1 minute)")
        
        # Convert to Series for Kronos
        x_ts = pd.Series(ohlcv.index.values, index=range(len(ohlcv.index)))
        y_ts_series = pd.Series(y_ts.values, index=range(len(y_ts)))
        
        # Check if predictor is stub or real
        if hasattr(adapter.predictor, '_predictor'):
            predictor = adapter.predictor._predictor
        else:
            predictor = adapter.predictor
        
        print(f"Predictor type: {type(predictor).__name__}")
        
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
        
        print(f"\n--- Prediction Output Structure ---")
        print(f"Type: {type(pred_df)}")
        print(f"Shape: {pred_df.shape}")
        print(f"Columns: {pred_df.columns.tolist()}")
        print(f"Index type: {type(pred_df.index)}")
        print(f"Index range: {pred_df.index[0]} to {pred_df.index[-1] if len(pred_df) > 1 else 'N/A'}")
        
        print(f"\n--- Full Prediction DataFrame ---")
        print(pred_df)
        
        print(f"\n--- Key Values ---")
        print(f"Input last close: {last_close:.2f}")
        print(f"Predicted close (first step): {pred_df['close'].iloc[0]:.2f}")
        print(f"Predicted close (last step): {pred_df['close'].iloc[-1]:.2f}")
        
        # Score calculation
        pred_close_final = pred_df["close"].iloc[-1]
        score = (pred_close_final - last_close) / last_close
        print(f"\nScore calculation:")
        print(f"  pred_close (step -1): {pred_close_final:.2f}")
        print(f"  spot_close: {last_close:.2f}")
        print(f"  score = ({pred_close_final:.2f} - {last_close:.2f}) / {last_close:.2f} = {score:.4f}")
        
        # Check progression
        print(f"\n--- Prediction Progression ---")
        print("Is Kronos predicting a steady decline?")
        closes = pred_df['close'].values
        for i, c in enumerate(closes):
            change = (c - last_close) / last_close
            print(f"  Step {i+1}: close={c:.2f}, return vs spot={change:.2%}")
        
    except Exception as e:
        print(f"Error during direct inspection: {e}")
        import traceback
        traceback.print_exc()


def main():
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║                                                                      ║")
    print("║           KRONOS PREDICTION DIAGNOSTICS                              ║")
    print("║                                                                      ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    
    # Check if micro-test results exist
    if not Path('kronos_micro_test.csv').exists():
        print("\n❌ ERROR: kronos_micro_test.csv not found!")
        print("Run 'python scripts/test_kronos_micro.py' first.")
        return
    
    # Run all diagnostics
    diagnose_micro_test_results()
    verify_spot_close_alignment()
    check_price_scale()
    analyze_kronos_prediction_structure()
    compute_rankic_vs_actual_returns()
    check_cross_sectional_ranking_stability()
    
    # This one takes longer, ask user
    print("\n" + "="*70)
    print("Would you like to run direct Kronos output inspection?")
    print("(This will run a single prediction and show full structure)")
    print("="*70)
    
    response = input("Run direct inspection? [y/N]: ").strip().lower()
    if response == 'y':
        investigate_kronos_output_directly()
    
    # Summary
    print_section("SUMMARY & RECOMMENDATIONS")
    print("""
Based on these diagnostics, likely issues are:

1. **Strong Negative Bias (-13.7% mean)**
   - Could be mean-reversion prior baked into Kronos
   - Could be price scale mismatch
   - Could be model expects different time horizon

2. **What to check next:**
   a. Run direct inspection to see pred_df structure
   b. Verify Kronos was trained on similar price scale
   c. Check if model expects raw vs adjusted prices
   d. Check if horizon interpretation is correct

3. **Even if biased, rankings might work:**
   - RankIC is what matters for cross-sectional signals
   - If rankings are correct, absolute levels don't matter
   - Compute RankIC vs actual forward returns (done above)

4. **Baseline comparison:**
   - Your LGB 20d RankIC: 0.1009
   - If Kronos RankIC < 0.05, it's likely not useful alone
   - But could still add value in ensemble (orthogonal errors)
""")


if __name__ == "__main__":
    main()


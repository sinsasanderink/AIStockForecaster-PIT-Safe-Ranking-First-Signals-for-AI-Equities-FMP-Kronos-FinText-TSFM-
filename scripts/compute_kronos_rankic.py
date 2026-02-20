#!/usr/bin/env python
"""
Quick RankIC Computation for Kronos Micro-Test
==============================================

The CRITICAL question: Does Kronos rank stocks correctly?

Even if predictions are biased (all negative), ranking quality is what matters.
This script computes RankIC against actual forward returns.

Usage:
    python scripts/compute_kronos_rankic.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import duckdb

def main():
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║          KRONOS RankIC vs ACTUAL RETURNS                             ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    print()
    
    # Load Kronos predictions
    if not Path('kronos_micro_test.csv').exists():
        print("❌ ERROR: kronos_micro_test.csv not found!")
        print("Run 'python scripts/test_kronos_micro.py' first.")
        return
    
    kronos = pd.read_csv('kronos_micro_test.csv')
    print(f"Loaded {len(kronos)} Kronos predictions")
    print(f"Dates: {kronos['date'].unique()}")
    print()
    
    # Connect to DuckDB
    conn = duckdb.connect('data/features.duckdb', read_only=True)
    
    # Get actual forward returns from features table
    print("Fetching actual 20d forward returns from features table...")
    
    dates_str = "', '".join(kronos['date'].unique())
    
    try:
        # Get 20d forward returns from labels table
        # Labels table has: as_of_date, ticker, horizon (20/60/90), excess_return
        actuals = conn.execute(f"""
            SELECT 
                as_of_date as date,
                ticker,
                excess_return as actual_return
            FROM labels
            WHERE horizon = 20
            AND as_of_date IN ('{dates_str}')
        """).df()
        
        print(f"Found {len(actuals)} rows with actual returns")
            
    except Exception as e:
        print(f"Error: {e}")
        actuals = pd.DataFrame()
    
    conn.close()
    
    if len(actuals) == 0:
        print("\n❌ Cannot compute RankIC - no actual returns found!")
        return
    
    # Merge predictions with actuals
    kronos['date'] = pd.to_datetime(kronos['date']).dt.strftime('%Y-%m-%d')
    actuals['date'] = pd.to_datetime(actuals['date']).dt.strftime('%Y-%m-%d')
    
    merged = kronos.merge(actuals, on=['date', 'ticker'], how='inner')
    print(f"\nMerged data: {len(merged)} rows")
    
    if len(merged) < 10:
        print("⚠️  Too few merged rows for meaningful analysis!")
        print("Checking what tickers are in both datasets...")
        
        kronos_tickers = set(kronos['ticker'])
        actual_tickers = set(actuals['ticker'])
        print(f"Kronos tickers: {kronos_tickers}")
        print(f"Actuals tickers: {actual_tickers}")
        print(f"Common: {kronos_tickers & actual_tickers}")
        return
    
    # Show sample of merged data
    print("\n" + "="*70)
    print("SAMPLE MERGED DATA (Kronos score vs Actual return)")
    print("="*70)
    print(f"{'Date':<12} {'Ticker':<8} {'Score':>10} {'Actual':>10} {'Match?':>8}")
    print("-" * 55)
    
    for _, row in merged.sample(min(15, len(merged))).iterrows():
        # "Match" = both positive or both negative
        match = (row['score'] > 0 and row['actual_return'] > 0) or \
                (row['score'] < 0 and row['actual_return'] < 0)
        print(f"{row['date']:<12} {row['ticker']:<8} {row['score']:>10.4f} {row['actual_return']:>10.4f} {'✅' if match else '❌'}")
    
    # Compute RankIC per date
    print("\n" + "="*70)
    print("RankIC BY DATE")
    print("="*70)
    print(f"{'Date':<12} {'RankIC':>10} {'p-value':>10} {'n':>6}")
    print("-" * 45)
    
    per_date_ics = []
    for date in sorted(merged['date'].unique()):
        date_df = merged[merged['date'] == date]
        if len(date_df) >= 5:
            ic, pval = spearmanr(date_df['score'], date_df['actual_return'])
            per_date_ics.append(ic)
            sig = "**" if pval < 0.05 else "*" if pval < 0.1 else ""
            print(f"{date:<12} {ic:>10.4f} {pval:>10.4f} {len(date_df):>6} {sig}")
    
    # Overall RankIC
    overall_ic, overall_p = spearmanr(merged['score'], merged['actual_return'])
    sig = "**" if overall_p < 0.05 else "*" if overall_p < 0.1 else ""
    
    print("-" * 45)
    print(f"{'OVERALL':<12} {overall_ic:>10.4f} {overall_p:>10.4f} {len(merged):>6} {sig}")
    
    # Statistics
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)
    
    print(f"\nKronos predictions:")
    print(f"  Mean:   {merged['score'].mean():>10.4f}")
    print(f"  Std:    {merged['score'].std():>10.4f}")
    print(f"  Range:  [{merged['score'].min():.4f}, {merged['score'].max():.4f}]")
    
    print(f"\nActual returns:")
    print(f"  Mean:   {merged['actual_return'].mean():>10.4f}")
    print(f"  Std:    {merged['actual_return'].std():>10.4f}")
    print(f"  Range:  [{merged['actual_return'].min():.4f}, {merged['actual_return'].max():.4f}]")
    
    # Baseline comparison
    baseline_20d = 0.1009
    
    print("\n" + "="*70)
    print("COMPARISON TO YOUR BASELINE")
    print("="*70)
    print(f"\nYour LGB baseline 20d RankIC:  {baseline_20d:.4f}")
    print(f"Kronos 20d RankIC:              {overall_ic:.4f}")
    print(f"Difference:                     {overall_ic - baseline_20d:+.4f}")
    
    # Interpretation
    print("\n" + "="*70)
    print("INTERPRETATION")
    print("="*70)
    
    if overall_ic > 0.10:
        print("\n✅ STRONG SIGNAL: Kronos RankIC > 0.10")
        print("   Kronos is competitive with your baseline!")
        print("   Worth running full evaluation.")
        
    elif overall_ic > 0.05:
        print("\n⚠️  MODERATE SIGNAL: Kronos RankIC 0.05-0.10")
        print("   Kronos shows some predictive power, but weaker than baseline.")
        print("   May still add value in ensemble (if orthogonal to baseline).")
        
    elif overall_ic > 0.02:
        print("\n⚠️  WEAK SIGNAL: Kronos RankIC 0.02-0.05")
        print("   Very weak predictive power, likely noise.")
        print("   Not recommended for standalone use.")
        
    elif overall_ic > -0.02:
        print("\n❌ NO SIGNAL: Kronos RankIC ~ 0")
        print("   Kronos rankings appear random.")
        print("   Model is not useful for this task.")
        
    else:
        print("\n❌ NEGATIVE SIGNAL: Kronos RankIC < -0.02")
        print("   Kronos rankings are INVERSELY correlated with returns!")
        print("   This suggests:")
        print("   1. Mean-reversion bias baked into model")
        print("   2. Model trained on different market regime")
        print("   3. Possible data/interpretation bug")
        print("   4. Or: inverse signal could be useful (short high-score stocks!)")
    
    # Additional diagnostics
    print("\n" + "="*70)
    print("ADDITIONAL DIAGNOSTICS")
    print("="*70)
    
    # Check if sign agreement is better than random
    sign_match = ((merged['score'] > 0) == (merged['actual_return'] > 0)).mean()
    print(f"\nSign agreement (both +/- same direction): {sign_match*100:.1f}%")
    print(f"  (Random = 50%, >55% is meaningful)")
    
    # Quintile spread
    merged['quintile'] = pd.qcut(merged['score'], 5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
    quintile_returns = merged.groupby('quintile')['actual_return'].mean()
    print(f"\nQuintile returns (Q5=highest score, should have highest return):")
    for q, ret in quintile_returns.items():
        print(f"  {q}: {ret:>8.4f}")
    
    q5_q1_spread = quintile_returns['Q5'] - quintile_returns['Q1']
    print(f"  Q5-Q1 spread: {q5_q1_spread:.4f}")
    
    if q5_q1_spread > 0.02:
        print("  ✅ Long-short spread is positive (Q5 > Q1)")
    elif q5_q1_spread > 0:
        print("  ⚠️  Slight positive spread (may be noise)")
    else:
        print("  ❌ Negative spread (Q5 < Q1) - rankings are inverted!")
    
    # Correlation check
    print("\n" + "="*70)
    print("WHAT THIS MEANS FOR YOUR PROJECT")
    print("="*70)
    
    if overall_ic > 0.05:
        print("""
✅ KRONOS SHOWS PROMISE

Next steps:
1. Run full SMOKE evaluation (1-3 days on CPU or 2-4 hours on GPU)
2. Compare to baseline across all horizons (20d, 60d, 90d)
3. If RankIC > 0.05, proceed to fusion experiments (Chapter 11)
""")
    else:
        print("""
⚠️  KRONOS SIGNAL IS WEAK/ABSENT

Options:
1. **Debug potential issues:**
   - Check price scale (adjusted vs unadjusted)
   - Verify horizon interpretation
   - Check if model expects different input format
   
2. **Accept limited value:**
   - Kronos as "price-only baseline" (expected to be weak)
   - Still document results for completeness
   - May add small value in ensemble even with low standalone RankIC
   
3. **Move to Chapter 9 (FinText):**
   - Text signals may be more orthogonal to your factor baseline
   - Kronos integration is technically complete either way
   
4. **Try different horizons:**
   - Kronos might be better at shorter/longer horizons
   - Your 90d baseline is very strong (0.18 RankIC)
""")


if __name__ == "__main__":
    main()


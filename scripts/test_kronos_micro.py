#!/usr/bin/env python
"""
Kronos Micro-Test: Quick validation of model output quality
============================================================

This script runs a minimal test to verify:
1. Kronos loads and runs successfully
2. Predictions are generated for multiple dates
3. Scores show reasonable variance (not degenerate)
4. Score distribution is sane (typical returns range)

Runtime: ~15 minutes (3 dates × 20 tickers)

Usage:
    python scripts/test_kronos_micro.py
    
Output:
    - Console: Progress and summary statistics
    - File: kronos_micro_test.csv (predictions for analysis)
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
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.kronos_adapter import KronosAdapter

def main():
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║                                                                      ║")
    print("║              KRONOS MICRO-TEST (~15 minutes)                         ║")
    print("║                                                                      ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    print()
    
    # Test configuration
    TEST_DATES = ['2024-02-01', '2024-03-01', '2024-04-01']
    TEST_TICKERS = [
        'AAPL', 'MSFT', 'NVDA', 'GOOGL', 'META', 
        'AMZN', 'AMD', 'AVGO', 'CRM', 'ORCL',
        'ADBE', 'INTC', 'CSCO', 'QCOM', 'TXN',
        'MU', 'AMAT', 'LRCX', 'KLAC', 'MRVL'
    ]
    HORIZON = 20
    TIMEOUT = 60
    
    print(f"Configuration:")
    print(f"  Dates:    {len(TEST_DATES)} ({TEST_DATES[0]} to {TEST_DATES[-1]})")
    print(f"  Tickers:  {len(TEST_TICKERS)} (top liquid AI stocks)")
    print(f"  Horizon:  {HORIZON} trading days")
    print(f"  Timeout:  {TIMEOUT}s per ticker")
    print(f"  Expected: ~15 minutes total")
    print()
    
    # Load adapter
    print("Loading Kronos adapter...")
    start_time = datetime.now()
    
    adapter = KronosAdapter.from_pretrained(
        db_path='data/features.duckdb',
        use_stub=False,
        per_ticker_timeout=TIMEOUT,
    )
    
    load_time = (datetime.now() - start_time).total_seconds()
    print(f"✓ Adapter loaded in {load_time:.1f}s")
    print()
    
    # Score tickers for each test date
    all_results = []
    total_predictions = 0
    
    for i, date in enumerate(TEST_DATES, 1):
        print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"Date {i}/{len(TEST_DATES)}: {date}")
        print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        
        date_start = datetime.now()
        
        scores_df = adapter.score_universe_batch(
            asof_date=pd.Timestamp(date),
            tickers=TEST_TICKERS,
            horizon=HORIZON,
            verbose=False,  # Reduce noise
        )
        
        date_elapsed = (datetime.now() - date_start).total_seconds()
        
        if not scores_df.empty:
            scores_df['date'] = date
            all_results.append(scores_df)
            total_predictions += len(scores_df)
            
            print(f"✓ Completed: {len(scores_df)}/{len(TEST_TICKERS)} tickers in {date_elapsed:.1f}s")
            print(f"  Score range: [{scores_df['score'].min():.4f}, {scores_df['score'].max():.4f}]")
            print(f"  Score mean:  {scores_df['score'].mean():.4f} ± {scores_df['score'].std():.4f}")
        else:
            print(f"✗ No predictions for {date}")
        
        print()
    
    # Combine results
    if not all_results:
        print("❌ ERROR: No predictions generated")
        print()
        print("Possible issues:")
        print("  - Kronos not installed correctly")
        print("  - DuckDB prices table missing data")
        print("  - All tickers timing out")
        return
    
    results = pd.concat(all_results, ignore_index=True)
    
    # Save to CSV
    output_path = Path(__file__).parent.parent / "kronos_micro_test.csv"
    results.to_csv(output_path, index=False)
    
    # Print summary
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║                                                                      ║")
    print("║              KRONOS MICRO-TEST RESULTS                               ║")
    print("║                                                                      ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    print()
    
    total_time = (datetime.now() - start_time).total_seconds()
    print(f"Total Runtime: {total_time / 60:.1f} minutes")
    print()
    
    print(f"Predictions Generated: {len(results)}")
    print(f"  Expected: {len(TEST_DATES) * len(TEST_TICKERS)} (if all succeed)")
    print(f"  Success rate: {len(results) / (len(TEST_DATES) * len(TEST_TICKERS)) * 100:.1f}%")
    print()
    
    print("Score Distribution:")
    print(results['score'].describe().to_string())
    print()
    
    # Sanity checks
    print("Sanity Checks:")
    print("─" * 70)
    
    score_std = results['score'].std()
    score_range = results['score'].max() - results['score'].min()
    
    # Check 1: Variance
    if score_std < 0.001:
        print("  ❌ FAIL: Scores have very low variance (model may be degenerate)")
    else:
        print(f"  ✓ PASS: Scores have reasonable variance (std={score_std:.4f})")
    
    # Check 2: Range
    if score_range < 0.01:
        print("  ❌ FAIL: Score range is too narrow (model may be broken)")
    else:
        print(f"  ✓ PASS: Score range is reasonable ({score_range:.4f})")
    
    # Check 3: Reasonable magnitude
    abs_mean = abs(results['score'].mean())
    if abs_mean > 0.5:
        print(f"  ⚠️  WARNING: Mean score magnitude is large ({abs_mean:.4f})")
        print("     (Expected: typical returns are -20% to +20%)")
    else:
        print(f"  ✓ PASS: Mean score magnitude is reasonable ({abs_mean:.4f})")
    
    # Check 4: Not all same sign
    n_positive = (results['score'] > 0).sum()
    n_negative = (results['score'] < 0).sum()
    if n_positive == 0 or n_negative == 0:
        print(f"  ❌ FAIL: All scores have same sign (pos={n_positive}, neg={n_negative})")
    else:
        print(f"  ✓ PASS: Scores have mixed signs (pos={n_positive}, neg={n_negative})")
    
    print()
    
    # Show sample predictions
    print("Sample Predictions:")
    print("─" * 70)
    sample = results[['date', 'ticker', 'score', 'pred_close', 'spot_close']].head(15)
    print(sample.to_string(index=False))
    print()
    
    # Per-date statistics
    print("Per-Date Statistics:")
    print("─" * 70)
    per_date = results.groupby('date')['score'].agg(['count', 'mean', 'std', 'min', 'max'])
    print(per_date.to_string())
    print()
    
    # Output file
    print(f"✓ Results saved to: {output_path}")
    print()
    
    # Next steps
    print("Next Steps:")
    print("─" * 70)
    print("1. Review the score distribution above")
    print("   - Should see reasonable variance (std > 0.01)")
    print("   - Typical range: -20% to +20%")
    print()
    print("2. Analyze predictions:")
    print("   import pandas as pd")
    print("   df = pd.read_csv('kronos_micro_test.csv')")
    print("   print(df.describe())")
    print()
    print("3. If results look good, run SMOKE or FULL mode:")
    print("   - SMOKE: ~3 days on CPU (can reduce to 1 day with single horizon)")
    print("   - FULL: ~2-3 weeks on CPU (or use CUDA GPU if available)")
    print()
    print("4. To run faster:")
    print("   - Use --horizons 20 (single horizon)")
    print("   - Filter to top 20 tickers")
    print("   - Use CUDA GPU (100x faster than CPU)")
    print()

if __name__ == "__main__":
    main()


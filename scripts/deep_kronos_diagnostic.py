#!/usr/bin/env python
"""
Deep Kronos Diagnostic: Is it a bug or is the model just bad?
=============================================================

This script checks ALL potential sources of error:
1. Score identity (is score = pred_close/spot_close - 1?)
2. Spot close alignment (does spot_close match last input close?)
3. Price scale (adjusted vs unadjusted, splits)
4. Kronos output interpretation (what does pred_df actually contain?)
5. Horizon indexing (are we grabbing the right timestep?)
6. Normalization (does Kronos expect normalized input?)

Run this BEFORE concluding the model is bad!
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
import duckdb

sys.path.insert(0, str(Path(__file__).parent.parent))


def print_section(title: str):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def check_score_identity():
    """CHECK 1: Is score = pred_close/spot_close - 1?"""
    print_section("CHECK 1: SCORE IDENTITY VERIFICATION")
    
    df = pd.read_csv('kronos_micro_test.csv')
    
    # Compute what score SHOULD be
    calculated_score = (df['pred_close'] / df['spot_close']) - 1.0
    
    # Compare to logged score
    error = (calculated_score - df['score']).abs()
    
    print(f"\nFormula: score = (pred_close / spot_close) - 1")
    print(f"\nMax absolute error:  {error.max():.10f}")
    print(f"Mean absolute error: {error.mean():.10f}")
    
    if error.max() < 1e-6:
        print("\n✅ PASS: Score identity is correct")
        return True
    else:
        print("\n❌ FAIL: Score calculation has errors!")
        print("Sample of mismatches:")
        print(df[error > 1e-6][['ticker', 'date', 'score', 'pred_close', 'spot_close']].head())
        return False


def check_spot_close_alignment():
    """CHECK 2: Does spot_close match the last close in the input OHLCV?"""
    print_section("CHECK 2: SPOT CLOSE ALIGNMENT")
    
    df = pd.read_csv('kronos_micro_test.csv')
    
    from src.data import PricesStore
    prices_store = PricesStore(db_path='data/features.duckdb', enable_cache=True)
    
    print("\nChecking if logged spot_close matches last close from PricesStore...")
    print(f"{'Ticker':<8} {'Date':<12} {'spot_close':>12} {'OHLCV last':>12} {'Match?':>8}")
    print("-" * 60)
    
    mismatches = 0
    for _, row in df.iterrows():
        ticker = row['ticker']
        date = pd.Timestamp(row['date'])
        spot_close = row['spot_close']
        
        # Fetch the EXACT same OHLCV that would go into Kronos
        ohlcv = prices_store.fetch_ohlcv(
            ticker=ticker,
            asof_date=date,
            lookback=252,
            strict_lookback=True,
            fill_missing=True
        )
        
        if len(ohlcv) == 252:
            ohlcv_last_close = ohlcv["close"].iloc[-1]
            match = abs(spot_close - ohlcv_last_close) < 0.01
            
            if not match:
                mismatches += 1
                print(f"{ticker:<8} {str(date.date()):<12} {spot_close:>12.2f} {ohlcv_last_close:>12.2f} {'❌'}")
        else:
            print(f"{ticker:<8} {str(date.date()):<12} {'N/A':>12} {'<252 rows':>12} {'⚠️'}")
    
    # Only print first 10
    if mismatches == 0:
        print(f"\n✅ PASS: All {len(df)} spot_close values match OHLCV last close")
        return True
    else:
        print(f"\n❌ FAIL: {mismatches} mismatches found!")
        return False


def check_price_scale_and_splits():
    """CHECK 3: Price scale consistency and split detection"""
    print_section("CHECK 3: PRICE SCALE & SPLIT DETECTION")
    
    from src.data import PricesStore
    prices_store = PricesStore(db_path='data/features.duckdb', enable_cache=True)
    
    # Check a few tickers known to have splits
    test_cases = [
        ("NVDA", "2024-02-01"),  # NVDA split 10:1 in June 2024, check if pre-split data is clean
        ("AVGO", "2024-02-01"),  # AVGO had splits
        ("AAPL", "2024-02-01"),  # Stable reference
    ]
    
    for ticker, date in test_cases:
        print(f"\n--- {ticker} @ {date} ---")
        
        ohlcv = prices_store.fetch_ohlcv(
            ticker=ticker,
            asof_date=pd.Timestamp(date),
            lookback=252,
            strict_lookback=True,
            fill_missing=True
        )
        
        if len(ohlcv) < 252:
            print(f"  Insufficient data: {len(ohlcv)} rows")
            continue
        
        print(f"  Date range: {ohlcv.index[0].date()} to {ohlcv.index[-1].date()}")
        print(f"  Close price range: ${ohlcv['close'].min():.2f} to ${ohlcv['close'].max():.2f}")
        print(f"  Last close: ${ohlcv['close'].iloc[-1]:.2f}")
        
        # Check for large daily moves (potential split artifacts)
        returns = ohlcv['close'].pct_change()
        large_moves = returns[abs(returns) > 0.15]  # >15% moves
        
        if len(large_moves) > 0:
            print(f"  ⚠️  Large daily moves (>15%): {len(large_moves)}")
            for idx, ret in large_moves.items():
                print(f"      {idx.date()}: {ret:.1%}")
        else:
            print(f"  ✅ No large daily moves (splits likely handled correctly)")


def check_kronos_output_structure():
    """CHECK 4: What does Kronos actually output?"""
    print_section("CHECK 4: KRONOS OUTPUT STRUCTURE (requires running Kronos)")
    
    print("""
To check this, we need to run Kronos and inspect pred_df directly.

Key questions:
1. How many rows does pred_df have? (should = horizon = 20)
2. What are the column names and types?
3. Are values in expected ranges?

The inspect_kronos_output.py script does this - run it if you haven't.
""")
    
    # Check if we have the detailed inspection results
    # For now, analyze what we can from the CSV
    df = pd.read_csv('kronos_micro_test.csv')
    
    print("\n--- Analyzing pred_close values ---")
    print(f"pred_close range: ${df['pred_close'].min():.2f} to ${df['pred_close'].max():.2f}")
    print(f"spot_close range: ${df['spot_close'].min():.2f} to ${df['spot_close'].max():.2f}")
    
    ratio = df['pred_close'] / df['spot_close']
    print(f"\npred_close/spot_close ratio:")
    print(f"  Mean:  {ratio.mean():.4f} (1.0 = unchanged)")
    print(f"  Std:   {ratio.std():.4f}")
    print(f"  Range: [{ratio.min():.4f}, {ratio.max():.4f}]")
    
    if ratio.mean() < 0.9:
        print(f"\n⚠️  WARNING: Mean ratio {ratio.mean():.4f} suggests systematic downward bias!")
        print("   This could be:")
        print("   - Mean-reversion model behavior")
        print("   - OR: Scale/normalization mismatch")


def check_database_schema():
    """CHECK 5: Database schema and data availability"""
    print_section("CHECK 5: DATABASE SCHEMA")
    
    conn = duckdb.connect('data/features.duckdb', read_only=True)
    
    print("\n--- Tables in database ---")
    tables = conn.execute("SHOW TABLES").df()
    print(tables)
    
    for table in tables['name'].tolist():
        print(f"\n--- Schema: {table} ---")
        schema = conn.execute(f"DESCRIBE {table}").df()
        print(schema[['column_name', 'column_type']].head(10).to_string(index=False))
    
    # Check prices table specifically
    print("\n--- Prices table sample ---")
    sample = conn.execute("SELECT * FROM prices ORDER BY date DESC LIMIT 5").df()
    print(sample)
    
    # Check if there's an adjusted close column
    price_cols = conn.execute("DESCRIBE prices").df()['column_name'].tolist()
    print(f"\nPrices columns: {price_cols}")
    
    if 'adj_close' in price_cols or 'adjusted_close' in price_cols:
        print("⚠️  Database has adjusted close column - are we using the right one?")
    else:
        print("✅ No separate adjusted close column (using 'close' directly)")
    
    conn.close()


def verify_actual_vs_predicted():
    """CHECK 6: Compare Kronos predictions to actual outcomes"""
    print_section("CHECK 6: PREDICTION vs ACTUAL OUTCOME")
    
    df = pd.read_csv('kronos_micro_test.csv')
    
    from src.data import PricesStore
    prices_store = PricesStore(db_path='data/features.duckdb', enable_cache=True)
    
    # Pick NVDA as test case (most dramatic mismatch)
    ticker = "NVDA"
    asof_date = "2024-02-01"
    horizon = 20
    
    print(f"\n--- {ticker} @ {asof_date}, horizon={horizon}d ---")
    
    # Get Kronos prediction
    kronos_row = df[(df['ticker'] == ticker) & (df['date'] == asof_date)]
    if len(kronos_row) == 0:
        print("No Kronos prediction found")
        return
    
    spot_close = kronos_row['spot_close'].iloc[0]
    pred_close = kronos_row['pred_close'].iloc[0]
    kronos_score = kronos_row['score'].iloc[0]
    
    print(f"Kronos prediction:")
    print(f"  spot_close:  ${spot_close:.2f}")
    print(f"  pred_close:  ${pred_close:.2f}")
    print(f"  score:       {kronos_score:.2%}")
    
    # Get actual forward price
    # Need to fetch data after asof_date
    conn = duckdb.connect('data/features.duckdb', read_only=True)
    
    future_prices = conn.execute(f"""
        SELECT date, close FROM prices
        WHERE ticker = '{ticker}'
        AND date > '{asof_date}'
        ORDER BY date
        LIMIT 30
    """).df()
    
    conn.close()
    
    if len(future_prices) >= horizon:
        actual_future_close = future_prices.iloc[horizon-1]['close']  # 20th trading day
        actual_future_date = future_prices.iloc[horizon-1]['date']
        actual_return = (actual_future_close - spot_close) / spot_close
        
        print(f"\nActual outcome:")
        print(f"  Date +{horizon}d:    {actual_future_date}")
        print(f"  Actual close:  ${actual_future_close:.2f}")
        print(f"  Actual return: {actual_return:.2%}")
        
        print(f"\nComparison:")
        print(f"  Kronos predicted: {kronos_score:.2%}")
        print(f"  Actual was:       {actual_return:.2%}")
        print(f"  Error:            {kronos_score - actual_return:.2%}")
        
        if kronos_score < 0 and actual_return > 0:
            print(f"\n  ❌ DIRECTION WRONG: Kronos said DOWN, actual was UP")
        elif kronos_score > 0 and actual_return < 0:
            print(f"\n  ❌ DIRECTION WRONG: Kronos said UP, actual was DOWN")
        else:
            print(f"\n  ✅ DIRECTION CORRECT")
    else:
        print(f"Insufficient future data: {len(future_prices)} days")


def check_normalization_expectation():
    """CHECK 7: Does Kronos expect normalized input?"""
    print_section("CHECK 7: NORMALIZATION CHECK")
    
    print("""
CRITICAL QUESTION: Does Kronos expect normalized input?

Looking at the Kronos paper and GitHub:
- Kronos uses a TOKENIZER that converts OHLCV to discrete tokens
- The tokenizer might expect:
  a) Raw prices (what we're feeding)
  b) Log prices
  c) Returns
  d) Z-scored/normalized values

Let me check what KronosTokenizer does...
""")
    
    try:
        from model import KronosTokenizer
        
        tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
        
        print("\n--- KronosTokenizer inspection ---")
        print(f"Type: {type(tokenizer)}")
        
        # Check if tokenizer has normalization attributes
        attrs = [a for a in dir(tokenizer) if not a.startswith('_')]
        print(f"Public attributes: {attrs[:20]}...")
        
        # Check if there's a normalize method or attribute
        norm_attrs = [a for a in attrs if 'norm' in a.lower() or 'scale' in a.lower()]
        print(f"Normalization-related: {norm_attrs}")
        
    except ImportError:
        print("\n⚠️  Kronos not installed - cannot inspect tokenizer")
    except Exception as e:
        print(f"\nError inspecting tokenizer: {e}")


def check_horizon_interpretation():
    """CHECK 8: Are we interpreting horizon correctly?"""
    print_section("CHECK 8: HORIZON INTERPRETATION")
    
    print("""
Our score calculation:
    score = (pred_df["close"].iloc[-1] - spot_close) / spot_close

Questions:
1. Does pred_df have `horizon` rows? (e.g., 20 rows for horizon=20)
2. Is iloc[-1] the correct row for "20 days ahead"?
3. Or does Kronos output just 1 prediction (the horizon close)?

This requires running Kronos directly to inspect pred_df structure.
Run: python scripts/inspect_kronos_output.py
""")


def check_last_x_date_alignment():
    """CHECK 9: Is last_x_date correct?"""
    print_section("CHECK 9: LAST_X_DATE vs ASOF_DATE ALIGNMENT")
    
    df = pd.read_csv('kronos_micro_test.csv')
    df['last_x_date'] = pd.to_datetime(df['last_x_date'])
    df['date'] = pd.to_datetime(df['date'])
    
    # Compare
    df['date_match'] = df['last_x_date'].dt.date == df['date'].dt.date
    
    n_match = df['date_match'].sum()
    n_total = len(df)
    
    print(f"\nlast_x_date == date (asof_date): {n_match}/{n_total}")
    
    if n_match == n_total:
        print("✅ PASS: All last_x_date values match asof_date")
    else:
        print("❌ FAIL: Some dates don't match!")
        print(df[~df['date_match']][['ticker', 'date', 'last_x_date']].head())


def run_all_checks():
    """Run all diagnostic checks."""
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║                                                                      ║")
    print("║         DEEP KRONOS DIAGNOSTIC: BUG vs MODEL BEHAVIOR                ║")
    print("║                                                                      ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    
    results = {}
    
    # Check 1: Score identity
    results['score_identity'] = check_score_identity()
    
    # Check 2: Spot close alignment
    results['spot_close_alignment'] = check_spot_close_alignment()
    
    # Check 3: Price scale and splits
    check_price_scale_and_splits()
    
    # Check 4: Output structure
    check_kronos_output_structure()
    
    # Check 5: Database schema
    check_database_schema()
    
    # Check 6: Prediction vs actual
    verify_actual_vs_predicted()
    
    # Check 7: Normalization
    check_normalization_expectation()
    
    # Check 8: Horizon interpretation
    check_horizon_interpretation()
    
    # Check 9: Date alignment
    check_last_x_date_alignment()
    
    # Summary
    print_section("DIAGNOSTIC SUMMARY")
    
    print("\nChecks completed:")
    for check, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {check}: {status}")
    
    print("\n" + "-" * 70)
    print("INTERPRETATION GUIDE:")
    print("-" * 70)
    print("""
If ALL checks pass but predictions are still bad:
  → Model genuinely has mean-reversion bias (not a bug)
  → Kronos was likely trained on patterns where "up a lot" → "pullback"

If score_identity FAILS:
  → Bug in score calculation code

If spot_close_alignment FAILS:
  → Bug in how we capture the reference price
  → Could be adjusted vs unadjusted price issue

If large price moves detected (CHECK 3):
  → Possible split adjustment issues
  → Compare to a known price source (Yahoo Finance)

If pred_close values are in wrong scale:
  → Kronos might output log-prices, normalized values, or returns
  → Need to check Kronos output interpretation

NEXT STEPS:
1. If any check fails → fix the bug and re-run micro test
2. If all checks pass → model is working, just predicts poorly
3. Run inspect_kronos_output.py to see raw Kronos output
""")


if __name__ == "__main__":
    run_all_checks()


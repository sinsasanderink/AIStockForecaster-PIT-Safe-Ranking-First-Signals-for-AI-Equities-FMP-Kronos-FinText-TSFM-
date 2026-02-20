#!/usr/bin/env python3
"""
Kronos Single-Stock Sanity Test
================================

Tests that Kronos adapter can:
1. Load model (or handle gracefully if not installed)
2. Fetch OHLCV from PricesStore
3. Run prediction (or stub if model not available)
4. Compute score

Usage:
    python scripts/test_kronos_single_stock.py
    
    # With specific ticker and date
    python scripts/test_kronos_single_stock.py --ticker NVDA --date 2024-01-15 --horizon 20
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from src.data import PricesStore, load_global_trading_calendar
from src.models.kronos_adapter import KRONOS_AVAILABLE, KronosAdapter


def test_single_stock(
    ticker: str = "NVDA",
    asof_date: str = "2024-01-15",
    horizon: int = 20,
    db_path: str = "data/features.duckdb",
):
    """
    Test Kronos prediction for a single stock.
    
    Args:
        ticker: Stock ticker
        asof_date: As-of date
        horizon: Forecast horizon in trading days
        db_path: Path to DuckDB database
    """
    print("=" * 70)
    print("KRONOS SINGLE-STOCK SANITY TEST")
    print("=" * 70)
    print()
    print(f"Ticker: {ticker}")
    print(f"As-of Date: {asof_date}")
    print(f"Horizon: {horizon}d")
    print(f"Database: {db_path}")
    print()
    
    # ========================================================================
    # Step 1: Load PricesStore and Trading Calendar
    # ========================================================================
    
    print("Step 1: Loading PricesStore and trading calendar...")
    try:
        prices_store = PricesStore(db_path=db_path, enable_cache=True)
        trading_calendar = load_global_trading_calendar(db_path)
        print(f"✓ PricesStore initialized")
        print(f"✓ Trading calendar loaded: {len(trading_calendar)} days ({trading_calendar[0]} to {trading_calendar[-1]})")
    except Exception as e:
        print(f"✗ Failed to load data infrastructure: {e}")
        return 1
    
    print()
    
    # ========================================================================
    # Step 2: Fetch OHLCV for Ticker
    # ========================================================================
    
    print(f"Step 2: Fetching OHLCV for {ticker}...")
    try:
        ohlcv = prices_store.fetch_ohlcv(
            ticker=ticker,
            asof_date=asof_date,
            lookback=252,
            strict_lookback=False,  # Allow partial for testing
            fill_missing=True
        )
        
        if ohlcv.empty:
            print(f"✗ No OHLCV data found for {ticker} @ {asof_date}")
            return 1
        
        print(f"✓ Fetched OHLCV: {len(ohlcv)} rows")
        print(f"  Date range: {ohlcv.index[0]} to {ohlcv.index[-1]}")
        print(f"  Spot close: ${ohlcv['close'].iloc[-1]:.2f}")
        print()
        print("Last 5 rows:")
        print(ohlcv.tail())
        
    except Exception as e:
        print(f"✗ Failed to fetch OHLCV: {e}")
        return 1
    
    print()
    
    # ========================================================================
    # Step 3: Check Kronos Availability
    # ========================================================================
    
    print("Step 3: Checking Kronos availability...")
    if not KRONOS_AVAILABLE:
        print("⚠ Kronos not installed (expected for initial testing)")
        print("  To install: Follow instructions at https://github.com/shiyu-coder/Kronos")
        print()
        print("STUB MODE: Simulating prediction...")
        
        # Simulate a simple prediction (random walk + small drift)
        import numpy as np
        np.random.seed(42)
        
        spot_close = ohlcv["close"].iloc[-1]
        # Simple stub: current close + small random change
        pred_close = spot_close * (1.0 + np.random.normal(0.02, 0.05))
        score = (pred_close - spot_close) / spot_close
        
        print(f"  Spot close: ${spot_close:.2f}")
        print(f"  Predicted close (stub): ${pred_close:.2f}")
        print(f"  Score: {score:.4f} ({score * 100:.2f}%)")
        print()
        print("✓ Stub prediction successful")
        print()
        print("=" * 70)
        print("SANITY TEST: PASSED (Stub Mode)")
        print("=" * 70)
        return 0
    
    # ========================================================================
    # Step 4: Load Kronos Model
    # ========================================================================
    
    print("✓ Kronos is available")
    print()
    print("Step 4: Loading Kronos model...")
    
    try:
        adapter = KronosAdapter.from_pretrained(
            db_path=db_path,
            tokenizer_id="NeoQuasar/Kronos-Tokenizer-base",
            model_id="NeoQuasar/Kronos-base",
            max_context=512,
            lookback=252,
            device="cpu",  # Use CPU for testing
            deterministic=True,
        )
        print("✓ Kronos model loaded successfully")
        
    except Exception as e:
        print(f"✗ Failed to load Kronos: {e}")
        print("  This is expected if HuggingFace models not downloaded yet")
        return 1
    
    print()
    
    # ========================================================================
    # Step 5: Run Prediction
    # ========================================================================
    
    print("Step 5: Running Kronos prediction...")
    try:
        scores_df = adapter.score_universe_batch(
            asof_date=pd.Timestamp(asof_date),
            tickers=[ticker],
            horizon=horizon,
            verbose=True,
        )
        
        if scores_df.empty:
            print(f"✗ No prediction generated (insufficient history?)")
            return 1
        
        print(f"✓ Prediction generated")
        print()
        print("Results:")
        print(scores_df)
        
        # Extract results
        score = scores_df["score"].iloc[0]
        pred_close = scores_df["pred_close"].iloc[0]
        spot_close = scores_df["spot_close"].iloc[0]
        
        print()
        print(f"  Spot close: ${spot_close:.2f}")
        print(f"  Predicted close ({horizon}d): ${pred_close:.2f}")
        print(f"  Score: {score:.4f} ({score * 100:.2f}%)")
        
    except Exception as e:
        print(f"✗ Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print()
    
    # ========================================================================
    # Success
    # ========================================================================
    
    print("=" * 70)
    print("SANITY TEST: PASSED")
    print("=" * 70)
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Test Kronos prediction for a single stock"
    )
    parser.add_argument(
        "--ticker",
        type=str,
        default="NVDA",
        help="Stock ticker (default: NVDA)"
    )
    parser.add_argument(
        "--date",
        type=str,
        default="2024-01-15",
        help="As-of date (default: 2024-01-15)"
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=20,
        help="Forecast horizon in trading days (default: 20)"
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default="data/features.duckdb",
        help="Path to DuckDB database (default: data/features.duckdb)"
    )
    
    args = parser.parse_args()
    
    return test_single_stock(
        ticker=args.ticker,
        asof_date=args.date,
        horizon=args.horizon,
        db_path=args.db_path,
    )


if __name__ == "__main__":
    sys.exit(main())


#!/usr/bin/env python3
"""
Add Prices Table to DuckDB
===========================

This script adds a `prices` table to the existing features.duckdb database.
This is required for Chapter 8 (Kronos) integration.

The `prices` table stores raw OHLCV data needed for model inference.

Usage:
    python scripts/add_prices_table_to_duckdb.py
    
    # Or with explicit DB path
    python scripts/add_prices_table_to_duckdb.py --db-path data/features.duckdb
    
    # Dry run (check what would be done)
    python scripts/add_prices_table_to_duckdb.py --dry-run
"""

import argparse
import json
import logging
import sys
from datetime import date
from pathlib import Path
from typing import List

import duckdb
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.env import load_repo_dotenv, resolve_fmp_key
load_repo_dotenv()

from src.universe.ai_stocks import get_all_tickers

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# DOWNLOAD PRICES FROM FMP (Reuses build_features_duckdb.py logic)
# ============================================================================

def download_historical_prices(
    tickers: List[str],
    start_date: date,
    end_date: date,
    cache_dir: Path,
    api_key: str,
) -> pd.DataFrame:
    """
    Download historical prices for all tickers from FMP.
    
    Uses local caching to avoid repeated API calls.
    """
    from src.data.fmp_client import FMPClient, FMPError
    
    client = FMPClient(api_key=api_key, cache_dir=cache_dir, use_cache=True)
    
    all_data = []
    failed_tickers = []
    
    total = len(tickers)
    for i, ticker in enumerate(tickers):
        logger.info(f"[{i+1}/{total}] Fetching {ticker}...")
        try:
            df = client.get_historical_prices(
                ticker,
                start=start_date.isoformat(),
                end=end_date.isoformat()
            )
            if df is not None and not df.empty:
                df["ticker"] = ticker
                all_data.append(df)
            else:
                logger.warning(f"  No data for {ticker}")
                failed_tickers.append(ticker)
        except FMPError as e:
            logger.warning(f"  Failed {ticker}: {e}")
            failed_tickers.append(ticker)
    
    if not all_data:
        raise RuntimeError("No price data downloaded!")
    
    prices_df = pd.concat(all_data, ignore_index=True)
    
    logger.info(f"Downloaded {len(prices_df)} price rows for {len(all_data)} tickers")
    if failed_tickers:
        logger.warning(f"Failed tickers: {failed_tickers}")
    
    return prices_df


def dedupe_daily_bars(
    df: pd.DataFrame,
    *,
    ticker_col: str = "ticker",
    date_col: str = "date",
) -> pd.DataFrame:
    """
    Ensure exactly one OHLCV row per (ticker, date).
    
    Keeps the row with the highest volume if available,
    otherwise keeps the last row deterministically.
    """
    out = df.copy()
    
    # Normalize date column to python date
    if date_col not in out.columns and "Date" in out.columns:
        out[date_col] = pd.to_datetime(out["Date"]).dt.date
    else:
        out[date_col] = pd.to_datetime(out[date_col]).dt.date
    
    before = len(out)
    
    # Find volume column if present
    vol_col = "volume" if "volume" in out.columns else ("Volume" if "Volume" in out.columns else None)
    
    sort_cols = [ticker_col, date_col]
    if vol_col:
        # Sort so the highest volume ends up last per (ticker, date)
        out = out.sort_values(sort_cols + [vol_col])
    else:
        out = out.sort_values(sort_cols)
    
    out = out.drop_duplicates([ticker_col, date_col], keep="last").reset_index(drop=True)
    
    removed = before - len(out)
    if removed > 0:
        logger.warning(f"Removed {removed} duplicate daily rows from prices.")
    
    return out


# ============================================================================
# ADD PRICES TABLE TO DUCKDB
# ============================================================================

def add_prices_table_to_duckdb(
    db_path: Path,
    prices_df: pd.DataFrame,
    dry_run: bool = False,
):
    """
    Add a prices table to the existing DuckDB database.
    
    Args:
        db_path: Path to DuckDB database
        prices_df: DataFrame with columns: date, ticker, open, high, low, close, volume
        dry_run: If True, don't actually modify the database
    """
    if dry_run:
        logger.info("DRY RUN: Would add prices table to DuckDB")
        logger.info(f"  Rows: {len(prices_df)}")
        logger.info(f"  Tickers: {prices_df['ticker'].nunique()}")
        logger.info(f"  Date range: {prices_df['date'].min()} to {prices_df['date'].max()}")
        return
    
    logger.info(f"Adding prices table to {db_path}...")
    
    conn = duckdb.connect(str(db_path))
    
    try:
        # Check if prices table already exists
        existing_tables = conn.execute("SHOW TABLES").df()
        if "prices" in existing_tables["name"].values:
            logger.warning("Prices table already exists! Dropping and recreating...")
            conn.execute("DROP TABLE prices")
        
        # Create prices table
        conn.execute("""
            CREATE TABLE prices (
                date DATE,
                ticker VARCHAR,
                open DOUBLE,
                high DOUBLE,
                low DOUBLE,
                close DOUBLE,
                volume DOUBLE,
                PRIMARY KEY (date, ticker)
            )
        """)
        
        # Ensure correct column order
        prices_cols = ["date", "ticker", "open", "high", "low", "close", "volume"]
        
        # Rename columns if needed (FMP sometimes uses different names)
        col_mapping = {
            "Date": "date",
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "adjClose": "close",  # Use adjusted close
            "volume": "volume",
        }
        
        prices_clean = prices_df.rename(columns=col_mapping)
        
        # Select and order columns
        if "adjClose" in prices_df.columns and "close" in prices_df.columns:
            # Use adjusted close if available
            prices_clean["close"] = prices_df["adjClose"]
        
        prices_ordered = prices_clean[prices_cols]
        
        # Insert into DuckDB (register DataFrame first)
        conn.register("prices_ordered", prices_ordered)
        conn.execute("INSERT INTO prices SELECT * FROM prices_ordered")
        conn.unregister("prices_ordered")
        
        # Create indexes for performance
        conn.execute("CREATE INDEX idx_prices_ticker ON prices(ticker)")
        conn.execute("CREATE INDEX idx_prices_date ON prices(date)")
        
        # Get stats
        row_count = conn.execute("SELECT COUNT(*) FROM prices").fetchone()[0]
        ticker_count = conn.execute("SELECT COUNT(DISTINCT ticker) FROM prices").fetchone()[0]
        date_range = conn.execute("SELECT MIN(date), MAX(date) FROM prices").fetchone()
        
        logger.info(f"✓ Prices table created successfully!")
        logger.info(f"  Rows: {row_count:,}")
        logger.info(f"  Tickers: {ticker_count}")
        logger.info(f"  Date range: {date_range[0]} to {date_range[1]}")
        
    finally:
        conn.close()


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Add prices table to features.duckdb for Chapter 8 (Kronos)"
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default="data/features.duckdb",
        help="Path to DuckDB database (default: data/features.duckdb)"
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="data/cache/fmp",
        help="FMP cache directory (default: data/cache/fmp)"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        help="FMP API key (reads from .env if not provided)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry run: show what would be done without modifying database"
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default="2014-01-01",
        help="Start date for price data (default: 2014-01-01)"
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default="2025-06-30",
        help="End date for price data (default: 2025-06-30)"
    )
    
    args = parser.parse_args()
    
    # Resolve paths
    db_path = Path(args.db_path)
    cache_dir = Path(args.cache_dir)
    
    if not db_path.exists():
        logger.error(f"DuckDB not found at {db_path}")
        logger.error("Please run scripts/build_features_duckdb.py first")
        return 1
    
    # Get API key
    api_key = args.api_key or resolve_fmp_key()
    if not api_key:
        logger.error("FMP API key not found. Set FMP_KEYS in .env or pass --api-key")
        return 1
    
    # Parse dates
    start_date = date.fromisoformat(args.start_date)
    end_date = date.fromisoformat(args.end_date)
    
    # Get tickers from universe
    tickers = get_all_tickers()
    logger.info(f"Universe: {len(tickers)} tickers")
    
    # Download prices
    logger.info(f"Downloading prices from {start_date} to {end_date}...")
    logger.info(f"Using cache at {cache_dir}")
    
    try:
        prices_df = download_historical_prices(
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
            cache_dir=cache_dir,
            api_key=api_key,
        )
        
        # Deduplicate
        prices_df = dedupe_daily_bars(prices_df)
        
        logger.info(f"Fetched {len(prices_df):,} price rows")
        
    except Exception as e:
        logger.error(f"Failed to download prices: {e}")
        return 1
    
    # Add to DuckDB
    try:
        add_prices_table_to_duckdb(db_path, prices_df, dry_run=args.dry_run)
        
        if not args.dry_run:
            logger.info("")
            logger.info("=" * 70)
            logger.info("✓ SUCCESS: Prices table added to DuckDB")
            logger.info("=" * 70)
            logger.info("")
            logger.info("Next steps:")
            logger.info("  1. Run tests: pytest tests/test_prices_store.py -v")
            logger.info("  2. Implement Kronos adapter: src/models/kronos_adapter.py")
            logger.info("")
        
        return 0
        
    except Exception as e:
        logger.error(f"Failed to add prices table: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())


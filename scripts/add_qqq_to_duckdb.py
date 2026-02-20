#!/usr/bin/env python3
"""
Add QQQ Benchmark to DuckDB Prices Table
==========================================

Chapter 9 (FinText-TSFM) requires daily excess returns: stock_return - QQQ_return.
This script adds QQQ to the existing `prices` table from the FMP cache.

Usage:
    python scripts/add_qqq_to_duckdb.py
    python scripts/add_qqq_to_duckdb.py --dry-run
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import duckdb
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_qqq_from_cache(cache_dir: Path) -> pd.DataFrame:
    """Load QQQ prices from FMP cache files."""
    cache_files = list(cache_dir.glob("*symbol=QQQ*"))
    if not cache_files:
        raise FileNotFoundError(
            f"No QQQ cache file found in {cache_dir}. "
            "Run the FMP client to fetch QQQ data first."
        )

    logger.info(f"Loading QQQ from cache: {cache_files[0].name}")
    with open(cache_files[0]) as f:
        records = json.load(f)

    if not records:
        raise ValueError("QQQ cache file is empty")

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["ticker"] = "QQQ"

    # Select and order columns to match prices table schema
    df = df[["date", "ticker", "open", "high", "low", "close", "volume"]].copy()

    # Ensure numeric types
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop duplicates (keep last by date)
    df = df.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)

    logger.info(f"Loaded {len(df)} QQQ price rows ({df['date'].min()} to {df['date'].max()})")
    return df


def add_qqq_to_duckdb(db_path: Path, qqq_df: pd.DataFrame, dry_run: bool = False):
    """Insert QQQ rows into the existing prices table."""
    if dry_run:
        logger.info("DRY RUN: Would add QQQ to prices table")
        logger.info(f"  Rows: {len(qqq_df)}")
        logger.info(f"  Date range: {qqq_df['date'].min()} to {qqq_df['date'].max()}")
        return

    conn = duckdb.connect(str(db_path))
    try:
        # Check that prices table exists
        tables = [t[0] for t in conn.execute("SHOW TABLES").fetchall()]
        if "prices" not in tables:
            raise RuntimeError(
                "prices table not found. Run scripts/add_prices_table_to_duckdb.py first."
            )

        # Check if QQQ already exists
        existing = conn.execute(
            "SELECT COUNT(*) FROM prices WHERE ticker = 'QQQ'"
        ).fetchone()[0]

        if existing > 0:
            logger.info(f"QQQ already has {existing} rows in prices table. Replacing...")
            conn.execute("DELETE FROM prices WHERE ticker = 'QQQ'")

        # Insert QQQ rows
        conn.register("qqq_data", qqq_df)
        conn.execute("INSERT INTO prices SELECT * FROM qqq_data")
        conn.unregister("qqq_data")

        # Verify
        new_count = conn.execute(
            "SELECT COUNT(*) FROM prices WHERE ticker = 'QQQ'"
        ).fetchone()[0]
        total_count = conn.execute("SELECT COUNT(*) FROM prices").fetchone()[0]
        total_tickers = conn.execute(
            "SELECT COUNT(DISTINCT ticker) FROM prices"
        ).fetchone()[0]

        logger.info(f"QQQ rows inserted: {new_count}")
        logger.info(f"Total prices rows: {total_count:,}")
        logger.info(f"Total tickers: {total_tickers}")

    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser(description="Add QQQ benchmark to DuckDB prices table")
    parser.add_argument("--db-path", default="data/features.duckdb")
    parser.add_argument("--cache-dir", default="data/cache/fmp")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    db_path = Path(args.db_path)
    cache_dir = Path(args.cache_dir)

    if not db_path.exists():
        logger.error(f"DuckDB not found at {db_path}")
        return 1

    try:
        qqq_df = load_qqq_from_cache(cache_dir)
        add_qqq_to_duckdb(db_path, qqq_df, dry_run=args.dry_run)
        logger.info("SUCCESS: QQQ benchmark added to prices table")
        return 0
    except Exception as e:
        logger.error(f"Failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

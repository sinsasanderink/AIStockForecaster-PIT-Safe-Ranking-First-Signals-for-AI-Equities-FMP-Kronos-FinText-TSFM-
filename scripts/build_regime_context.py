#!/usr/bin/env python
"""
Chapter 12.4 — Build regime_context.parquet
============================================

Freezes regime context features for Chapter 13's DEUP pipeline.

Combines:
  - Per-stock: vol_20d, vol_60d, vol_of_vol, mom_1m, beta_252d, sector
    (from features table — stock-level noise estimates for DEUP's aleatoric baseline)
  - Market-level: vix_percentile_252d, market_regime, market_vol_21d, etc.
    (from regime table — macro context for epistemic uncertainty prediction)

Output: data/regime_context.parquet
  Keyed by (date, stable_id) — one row per stock per trading day.

Usage:
    python scripts/build_regime_context.py
"""

import argparse
import logging
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Per-stock features (from features table)
STOCK_COLS = [
    "date", "stable_id", "ticker",
    "vol_20d", "vol_60d", "vol_of_vol",
    "mom_1m", "sector",
]

# Market-level features (from regime table)
REGIME_COLS = [
    "date",
    "vix_percentile_252d",
    "vix_regime",
    "market_regime",
    "market_vol_21d",
    "market_return_5d",
    "market_return_21d",
    "above_ma_50",
    "above_ma_200",
]


def build_regime_context(db_path: str, output_path: str) -> pd.DataFrame:
    """Build and save the regime context parquet."""
    conn = duckdb.connect(db_path, read_only=True)

    stock_cols_str = ", ".join(STOCK_COLS)
    stock_df = conn.execute(f"SELECT {stock_cols_str} FROM features").df()
    logger.info("Loaded %d stock-level rows from features table", len(stock_df))

    regime_cols_str = ", ".join(REGIME_COLS)
    regime_df = conn.execute(f"SELECT {regime_cols_str} FROM regime").df()
    logger.info("Loaded %d regime rows from regime table", len(regime_df))

    conn.close()

    stock_df["date"] = pd.to_datetime(stock_df["date"])
    regime_df["date"] = pd.to_datetime(regime_df["date"])

    merged = stock_df.merge(regime_df, on="date", how="left")
    logger.info(
        "Merged: %d rows, %d columns, %d unique dates, %d unique stocks",
        len(merged), len(merged.columns),
        merged["date"].nunique(), merged["stable_id"].nunique(),
    )

    # Validate
    vol_coverage = merged["vol_20d"].notna().mean()
    vix_coverage = merged["vix_percentile_252d"].notna().mean()
    regime_coverage = merged["market_regime"].notna().mean()
    logger.info(
        "Coverage: vol_20d=%.1f%%, vix_pctile=%.1f%%, market_regime=%.1f%%",
        100 * vol_coverage, 100 * vix_coverage, 100 * regime_coverage,
    )

    if vol_coverage < 0.90:
        logger.warning("vol_20d coverage below 90%% — check features table")
    if vix_coverage < 0.90:
        logger.warning("vix_percentile_252d coverage below 90%% — check regime table")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(out, index=False)
    logger.info("Saved regime_context.parquet to %s (%.1f MB)", out, out.stat().st_size / 1e6)

    # Summary statistics
    print("\n" + "=" * 70)
    print("REGIME CONTEXT — SUMMARY")
    print("=" * 70)
    print(f"  Rows:          {len(merged):,}")
    print(f"  Columns:       {len(merged.columns)}")
    print(f"  Date range:    {merged['date'].min().date()} to {merged['date'].max().date()}")
    print(f"  Unique dates:  {merged['date'].nunique()}")
    print(f"  Unique stocks: {merged['stable_id'].nunique()}")
    print(f"\n  Per-stock features (for DEUP aleatoric baseline):")
    for col in ["vol_20d", "vol_60d", "vol_of_vol", "mom_1m"]:
        if col in merged.columns:
            s = merged[col]
            print(f"    {col:<16} coverage={s.notna().mean():.1%}  median={s.median():.4f}")
    print(f"\n  Market-level features (for DEUP epistemic context):")
    for col in ["vix_percentile_252d", "market_regime", "market_vol_21d"]:
        if col in merged.columns:
            s = merged[col]
            print(f"    {col:<24} coverage={s.notna().mean():.1%}  median={s.median():.4f}")
    print("=" * 70)

    return merged


def main():
    parser = argparse.ArgumentParser(description="Build regime_context.parquet for Ch13")
    parser.add_argument("--db-path", default="data/features.duckdb")
    parser.add_argument("--output", default="data/regime_context.parquet")
    args = parser.parse_args()
    build_regime_context(args.db_path, args.output)


if __name__ == "__main__":
    main()

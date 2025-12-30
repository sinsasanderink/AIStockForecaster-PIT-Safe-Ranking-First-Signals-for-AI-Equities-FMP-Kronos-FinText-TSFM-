#!/usr/bin/env python3
"""
Build Features DuckDB

Materializes a DuckDB feature store at data/features.duckdb from FMP data.

This script:
1. Downloads historical prices from FMP (cached locally)
2. Computes PIT-safe features (momentum, ADV, regime)
3. Computes v2 total-return labels (with dividends)
4. Stores everything in DuckDB with schema versioning

Usage:
    # .env is auto-loaded from repo root
    python scripts/build_features_duckdb.py
    
    # Or provide key via CLI
    python scripts/build_features_duckdb.py --api-key YOUR_KEY

Requirements:
    - FMP Premium API key (in .env as FMP_KEYS, or via CLI)
    - Network access for initial download (cached thereafter)

Output:
    - data/features.duckdb (NOT committed to git)
    - data/cache/fmp/ (cached API responses)
    - DATA_MANIFEST.json in data/ folder
"""

import argparse
import hashlib
import json
import logging
import os
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# AUTO-LOAD .env FROM REPO ROOT (must happen before any key resolution)
from src.utils.env import load_repo_dotenv, resolve_fmp_key
load_repo_dotenv()

from src.universe.ai_stocks import get_all_tickers, get_category_for_ticker
from src.features.labels import HORIZONS
from src.utils.price_validation import (
    validate_price_series_consistency,
    normalize_split_discontinuities,
    SplitDiscontinuityError,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# CONSTANTS
# ============================================================================

# Schema version for reproducibility
SCHEMA_VERSION = "1.0.0"

# Date ranges
RAW_DATA_START = date(2014, 1, 1)  # Buffer for lookbacks (252d for 12m momentum)
RAW_DATA_END = date(2025, 6, 30)
EVAL_START = date(2016, 1, 1)
EVAL_END = date(2025, 6, 30)

# Horizons for labels
LABEL_HORIZONS = [20, 60, 90]

# Benchmark
BENCHMARK_TICKER = "QQQ"

# Regime data
MARKET_BENCHMARK = "SPY"
VIX_PROXY = "VIXY"  # VIX ETF (FMP doesn't have ^VIX easily)

# Feature windows (trading days)
MOMENTUM_WINDOWS = {"mom_1m": 21, "mom_3m": 63, "mom_6m": 126, "mom_12m": 252}
ADV_WINDOW = 20
BETA_WINDOW = 252
VOL_WINDOW_20D = 20
VOL_WINDOW_60D = 60
VIX_PERCENTILE_WINDOW = 252  # 1 year for percentile


# ============================================================================
# FMP API KEY HANDLING (using shared helper from src.utils.env)
# ============================================================================

# resolve_fmp_key() is imported from src.utils.env
# It handles: CLI arg > FMP_KEYS env > FMP_API_KEY env
# and auto-loads .env from repo root


# ============================================================================
# DATA DOWNLOAD
# ============================================================================

def dedupe_daily_bars(
    df: pd.DataFrame,
    *,
    ticker_col: str = "ticker",
    date_col: str = "date",
) -> pd.DataFrame:
    """
    Ensure exactly one OHLCV row per (ticker, date).
    
    FMP sometimes returns duplicate daily bars; if we don't dedupe,
    we create duplicate features and labels.
    
    Keeps the row with the highest volume if available,
    otherwise keeps the last row deterministically.
    
    Args:
        df: DataFrame with price data
        ticker_col: Name of ticker column
        date_col: Name of date column
    
    Returns:
        Deduplicated DataFrame
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
        logger.warning(f"Removed {removed} duplicate daily rows from prices (one row kept per (ticker,date)).")
    
    return out


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
        logger.info(f"[{i+1}/{total}] Downloading {ticker}...")
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


def download_dividends(
    tickers: List[str],
    start_date: date,
    end_date: date,
    cache_dir: Path,
    api_key: str,
) -> pd.DataFrame:
    """
    Download dividend data for all tickers.
    """
    from src.data.fmp_client import FMPClient, FMPError
    
    client = FMPClient(api_key=api_key, cache_dir=cache_dir, use_cache=True)
    
    all_dividends = []
    
    total = len(tickers)
    for i, ticker in enumerate(tickers):
        if (i + 1) % 20 == 0:
            logger.info(f"[{i+1}/{total}] Downloading dividends for {ticker}...")
        try:
            # FMP historical dividends endpoint
            df = client.get_stock_dividend(ticker)
            if df is not None and not df.empty:
                df["ticker"] = ticker
                all_dividends.append(df)
        except (FMPError, AttributeError) as e:
            # Many stocks don't have dividends - that's fine
            pass
    
    if all_dividends:
        div_df = pd.concat(all_dividends, ignore_index=True)
        logger.info(f"Downloaded {len(div_df)} dividend records")
        return div_df
    else:
        logger.warning("No dividend data downloaded")
        return pd.DataFrame()


# ============================================================================
# FEATURE COMPUTATION
# ============================================================================

def compute_momentum_features(prices_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute momentum features from price data.
    
    Returns DataFrame with:
    - date, ticker, stable_id
    - mom_1m, mom_3m, mom_6m, mom_12m
    - adv_20d
    - vol_20d, vol_60d
    - beta_252d (if benchmark data available)
    """
    logger.info("Computing momentum features...")
    
    # Ensure date is datetime for groupby
    df = prices_df.copy()
    if "date" not in df.columns and "Date" in df.columns:
        df["date"] = pd.to_datetime(df["Date"]).dt.date
    else:
        df["date"] = pd.to_datetime(df["date"]).dt.date
    
    # Sort for rolling calculations
    df = df.sort_values(["ticker", "date"])
    
    features = []
    
    for ticker, group in df.groupby("ticker"):
        group = group.sort_values("date").reset_index(drop=True)
        
        if len(group) < 30:
            continue
        
        # Use close price for returns
        close_col = "close" if "close" in group.columns else "Close"
        volume_col = "volume" if "volume" in group.columns else "Volume"
        
        closes = group[close_col].values
        volumes = group[volume_col].values if volume_col in group.columns else None
        dates = group["date"].values
        
        for i in range(MOMENTUM_WINDOWS["mom_12m"], len(group)):
            row_date = dates[i]
            
            # Skip if before evaluation start (allow some buffer for early dates)
            if row_date < RAW_DATA_START:
                continue
            
            # Momentum returns
            mom_1m = (closes[i] / closes[i - MOMENTUM_WINDOWS["mom_1m"]] - 1) if i >= MOMENTUM_WINDOWS["mom_1m"] else None
            mom_3m = (closes[i] / closes[i - MOMENTUM_WINDOWS["mom_3m"]] - 1) if i >= MOMENTUM_WINDOWS["mom_3m"] else None
            mom_6m = (closes[i] / closes[i - MOMENTUM_WINDOWS["mom_6m"]] - 1) if i >= MOMENTUM_WINDOWS["mom_6m"] else None
            mom_12m = (closes[i] / closes[i - MOMENTUM_WINDOWS["mom_12m"]] - 1) if i >= MOMENTUM_WINDOWS["mom_12m"] else None
            
            # ADV (average daily dollar volume)
            adv_20d = None
            if volumes is not None and i >= ADV_WINDOW:
                dollar_volumes = closes[i-ADV_WINDOW:i] * volumes[i-ADV_WINDOW:i]
                adv_20d = float(np.mean(dollar_volumes))
            
            # Volatility (annualized)
            vol_20d = None
            vol_60d = None
            if i >= VOL_WINDOW_20D:
                returns = np.diff(np.log(closes[i-VOL_WINDOW_20D:i+1]))
                vol_20d = float(np.std(returns) * np.sqrt(252))
            if i >= VOL_WINDOW_60D:
                returns = np.diff(np.log(closes[i-VOL_WINDOW_60D:i+1]))
                vol_60d = float(np.std(returns) * np.sqrt(252))
            
            # Create stable_id
            stable_id = f"STABLE_{ticker}"
            
            features.append({
                "date": row_date,
                "ticker": ticker,
                "stable_id": stable_id,
                "mom_1m": mom_1m,
                "mom_3m": mom_3m,
                "mom_6m": mom_6m,
                "mom_12m": mom_12m,
                "adv_20d": adv_20d,
                "vol_20d": vol_20d,
                "vol_60d": vol_60d,
            })
    
    features_df = pd.DataFrame(features)
    logger.info(f"Computed features for {features_df['ticker'].nunique()} tickers, {len(features_df)} rows")
    
    return features_df


def compute_regime_features(
    market_prices: pd.DataFrame,
    vix_prices: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Compute market regime features.
    
    Returns DataFrame with:
    - date
    - market_return_20d, market_vol_20d
    - vix_percentile_252d (if VIX data available)
    """
    logger.info("Computing regime features...")
    
    df = market_prices.copy()
    if "date" not in df.columns and "Date" in df.columns:
        df["date"] = pd.to_datetime(df["Date"]).dt.date
    else:
        df["date"] = pd.to_datetime(df["date"]).dt.date
    
    df = df.sort_values("date").reset_index(drop=True)
    
    close_col = "close" if "close" in df.columns else "Close"
    closes = df[close_col].values
    dates = df["date"].values
    
    regime_features = []
    
    for i in range(VOL_WINDOW_60D, len(df)):
        row_date = dates[i]
        
        # Market return 20d
        market_return_20d = (closes[i] / closes[i - 20] - 1) if i >= 20 else None
        
        # Market volatility 20d (annualized)
        market_vol_20d = None
        if i >= 20:
            returns = np.diff(np.log(closes[i-20:i+1]))
            market_vol_20d = float(np.std(returns) * np.sqrt(252))
        
        regime_features.append({
            "date": row_date,
            "market_return_20d": market_return_20d,
            "market_vol_20d": market_vol_20d,
        })
    
    regime_df = pd.DataFrame(regime_features)
    
    # Add VIX percentile if available
    if vix_prices is not None and len(vix_prices) > VIX_PERCENTILE_WINDOW:
        vix_df = vix_prices.copy()
        if "date" not in vix_df.columns and "Date" in vix_df.columns:
            vix_df["date"] = pd.to_datetime(vix_df["Date"]).dt.date
        else:
            vix_df["date"] = pd.to_datetime(vix_df["date"]).dt.date
        
        vix_df = vix_df.sort_values("date")
        close_col = "close" if "close" in vix_df.columns else "Close"
        
        # Compute rolling percentile
        vix_series = vix_df.set_index("date")[close_col]
        
        def rolling_percentile(x):
            if len(x) < VIX_PERCENTILE_WINDOW:
                return np.nan
            return (x.values[-1] <= x.values).sum() / len(x) * 100
        
        vix_pct = vix_series.rolling(window=VIX_PERCENTILE_WINDOW).apply(rolling_percentile, raw=False)
        vix_pct_df = vix_pct.reset_index()
        vix_pct_df.columns = ["date", "vix_percentile_252d"]
        
        regime_df = regime_df.merge(vix_pct_df, on="date", how="left")
    else:
        # Use market volatility as proxy for VIX percentile
        logger.warning("VIX data not available, using market volatility proxy")
        regime_df["vix_percentile_252d"] = regime_df["market_vol_20d"].rank(pct=True) * 100
    
    logger.info(f"Computed regime features for {len(regime_df)} dates")
    
    return regime_df


def compute_labels(
    prices_df: pd.DataFrame,
    benchmark_prices: pd.DataFrame,
    dividends_df: Optional[pd.DataFrame],
    horizons: List[int] = LABEL_HORIZONS,
) -> pd.DataFrame:
    """
    Compute v2 total-return excess labels.
    
    Returns DataFrame with:
    - as_of_date, ticker, stable_id, horizon
    - excess_return
    - label_matured_at (UTC timestamp)
    - label_version
    """
    import pytz
    from src.data.fmp_client import get_market_close_utc
    
    logger.info("Computing v2 total-return labels...")
    
    # Prepare prices
    stock_df = prices_df.copy()
    if "date" not in stock_df.columns and "Date" in stock_df.columns:
        stock_df["date"] = pd.to_datetime(stock_df["Date"]).dt.date
    else:
        stock_df["date"] = pd.to_datetime(stock_df["date"]).dt.date
    
    bench_df = benchmark_prices.copy()
    if "date" not in bench_df.columns and "Date" in bench_df.columns:
        bench_df["date"] = pd.to_datetime(bench_df["Date"]).dt.date
    else:
        bench_df["date"] = pd.to_datetime(bench_df["date"]).dt.date
    
    close_col = "close" if "close" in stock_df.columns else "Close"
    
    # Sort and index
    stock_df = stock_df.sort_values(["ticker", "date"])
    bench_df = bench_df.sort_values("date").set_index("date")
    
    labels = []
    max_horizon = max(horizons)
    
    for ticker, group in stock_df.groupby("ticker"):
        group = group.sort_values("date").reset_index(drop=True)
        dates = group["date"].values
        closes = group[close_col].values
        
        if len(group) < max_horizon + 10:
            continue
        
        stable_id = f"STABLE_{ticker}"
        
        for i in range(len(group) - max_horizon):
            entry_date = dates[i]
            entry_price = closes[i]
            
            # Skip if entry date not in benchmark
            if entry_date not in bench_df.index:
                continue
            
            bench_entry = bench_df.loc[entry_date][close_col]
            
            for horizon in horizons:
                exit_idx = i + horizon
                if exit_idx >= len(group):
                    continue
                
                exit_date = dates[exit_idx]
                exit_price = closes[exit_idx]
                
                # Skip if exit date not in benchmark
                if exit_date not in bench_df.index:
                    continue
                
                bench_exit = bench_df.loc[exit_date][close_col]
                
                # Compute returns (price-only for now; dividends TODO if available)
                stock_return = (exit_price / entry_price) - 1
                bench_return = (bench_exit / bench_entry) - 1
                excess_return = stock_return - bench_return
                
                # Label matured at exit date market close (UTC)
                label_matured_at = get_market_close_utc(exit_date)
                
                labels.append({
                    "as_of_date": entry_date,
                    "ticker": ticker,
                    "stable_id": stable_id,
                    "horizon": horizon,
                    "excess_return": excess_return,
                    "label_matured_at": label_matured_at,
                    "label_version": "v2_price_only",  # Would be "v2" with dividends
                })
    
    labels_df = pd.DataFrame(labels)
    logger.info(f"Computed {len(labels_df)} labels for {labels_df['ticker'].nunique()} tickers")
    
    return labels_df


# ============================================================================
# DUCKDB STORAGE
# ============================================================================

def create_duckdb_store(
    db_path: Path,
    features_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    regime_df: pd.DataFrame,
    metadata: Dict[str, Any],
):
    """
    Create DuckDB feature store.
    """
    import duckdb
    
    logger.info(f"Creating DuckDB store at {db_path}...")
    
    # Remove existing
    if db_path.exists():
        db_path.unlink()
    
    conn = duckdb.connect(str(db_path))
    
    try:
        # Create features table
        conn.execute("""
            CREATE TABLE features (
                date DATE,
                ticker VARCHAR,
                stable_id VARCHAR,
                mom_1m DOUBLE,
                mom_3m DOUBLE,
                mom_6m DOUBLE,
                mom_12m DOUBLE,
                adv_20d DOUBLE,
                vol_20d DOUBLE,
                vol_60d DOUBLE,
                PRIMARY KEY (date, ticker)
            )
        """)
        
        conn.execute("INSERT INTO features SELECT * FROM features_df")
        
        # Create labels table
        conn.execute("""
            CREATE TABLE labels (
                as_of_date DATE,
                ticker VARCHAR,
                stable_id VARCHAR,
                horizon INTEGER,
                excess_return DOUBLE,
                label_matured_at TIMESTAMPTZ,
                label_version VARCHAR,
                PRIMARY KEY (as_of_date, ticker, horizon)
            )
        """)
        
        conn.execute("INSERT INTO labels SELECT * FROM labels_df")
        
        # Create regime table
        conn.execute("""
            CREATE TABLE regime (
                date DATE PRIMARY KEY,
                market_return_20d DOUBLE,
                market_vol_20d DOUBLE,
                vix_percentile_252d DOUBLE
            )
        """)
        
        conn.execute("INSERT INTO regime SELECT * FROM regime_df")
        
        # Create metadata table
        conn.execute("""
            CREATE TABLE metadata (
                key VARCHAR PRIMARY KEY,
                value VARCHAR
            )
        """)
        
        for key, value in metadata.items():
            conn.execute(
                "INSERT INTO metadata VALUES (?, ?)",
                [key, json.dumps(value) if not isinstance(value, str) else value]
            )
        
        # Create indexes
        conn.execute("CREATE INDEX idx_features_ticker ON features(ticker)")
        conn.execute("CREATE INDEX idx_features_date ON features(date)")
        conn.execute("CREATE INDEX idx_labels_ticker ON labels(ticker)")
        conn.execute("CREATE INDEX idx_labels_horizon ON labels(horizon)")
        
        # Verify
        feature_count = conn.execute("SELECT COUNT(*) FROM features").fetchone()[0]
        label_count = conn.execute("SELECT COUNT(*) FROM labels").fetchone()[0]
        regime_count = conn.execute("SELECT COUNT(*) FROM regime").fetchone()[0]
        
        logger.info(f"Created DuckDB with {feature_count} features, {label_count} labels, {regime_count} regime rows")
        
    finally:
        conn.close()


def compute_data_hash(features_df: pd.DataFrame, labels_df: pd.DataFrame) -> str:
    """Compute deterministic hash of data for reproducibility."""
    hash_str = (
        f"features:{len(features_df)},"
        f"features_tickers:{features_df['ticker'].nunique()},"
        f"features_min_date:{features_df['date'].min()},"
        f"features_max_date:{features_df['date'].max()},"
        f"labels:{len(labels_df)},"
        f"labels_tickers:{labels_df['ticker'].nunique()},"
        f"labels_min_date:{labels_df['as_of_date'].min()},"
        f"labels_max_date:{labels_df['as_of_date'].max()}"
    )
    return hashlib.sha256(hash_str.encode()).hexdigest()


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Build features DuckDB from FMP data")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/features.duckdb"),
        help="Output DuckDB path"
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/cache/fmp"),
        help="Cache directory for FMP responses"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="FMP API key (overrides .env and environment variables)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate setup without downloading"
    )
    parser.add_argument(
        "--skip-split-check",
        action="store_true",
        help="Skip split discontinuity validation (DANGEROUS)."
    )
    parser.add_argument(
        "--auto-normalize-splits",
        action="store_true",
        help="If split discontinuities are detected, attempt deterministic normalization and re-validate (recommended)."
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("BUILD FEATURES DUCKDB")
    print("=" * 60)
    
    # Validate API key (priority: CLI > env var > .env, already auto-loaded)
    try:
        api_key = resolve_fmp_key(cli_key=args.api_key)
        print(f"✓ FMP API key found")
    except RuntimeError as e:
        print(f"✗ {e}")
        return 1
    
    if args.dry_run:
        print("\nDry run - setup validated, exiting")
        return 0
    
    # Create directories
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Get universe
    tickers = get_all_tickers()
    print(f"\n✓ Universe: {len(tickers)} tickers")
    
    # Download prices
    print(f"\nStep 1: Downloading historical prices ({RAW_DATA_START} to {RAW_DATA_END})...")
    try:
        prices_df = download_historical_prices(
            tickers=tickers,
            start_date=RAW_DATA_START,
            end_date=RAW_DATA_END,
            cache_dir=args.cache_dir,
            api_key=api_key,
        )
        # Dedupe immediately after download (FMP may return duplicate bars)
        prices_df = dedupe_daily_bars(prices_df, ticker_col="ticker", date_col="date")
    except Exception as e:
        print(f"✗ Failed to download prices: {e}")
        return 1
    
    # Validate price series for split-adjustment consistency
    if not args.skip_split_check:
        print("\nStep 1b: Validating price series for split discontinuities...")

        # IMPORTANT: don't raise here; we need the discontinuities object if we want to auto-fix
        is_valid, discontinuities = validate_price_series_consistency(
            prices_df,
            price_col="close",
            date_col="date",
            ticker_col="ticker",
            raise_on_error=False,
        )

        if is_valid:
            print("  ✓ Price series are consistently split-adjusted")
        else:
            if args.auto_normalize_splits:
                logger.warning(
                    "Split discontinuities detected; attempting deterministic normalization (auto-normalize enabled)."
                )

                # FIX: pass discontinuities into normalization
                prices_df = normalize_split_discontinuities(
                    prices_df,
                    discontinuities,
                    price_cols=["close", "open", "high", "low"],
                    date_col="date",
                    ticker_col="ticker",
                )

                # Dedupe again after normalization (may preserve duplicates)
                prices_df = dedupe_daily_bars(prices_df, ticker_col="ticker", date_col="date")
                
                # Re-validate, now we DO want to fail hard if it still isn't clean
                validate_price_series_consistency(
                    prices_df,
                    price_col="close",
                    date_col="date",
                    ticker_col="ticker",
                    raise_on_error=True,
                )
                print("  ✓ Split discontinuities normalized deterministically")
            else:
                # Build error message from discontinuities
                error_lines = []
                for d in (discontinuities or []):
                    error_lines.append(
                        f"  - {d.ticker} on {d.date}: ${d.price_before:.2f} -> ${d.price_after:.2f} "
                        f"(ratio {d.ratio:.2f}, likely {d.likely_split_ratio}:1 split)"
                    )
                error_msg = "Split discontinuities detected in price series:\n" + "\n".join(error_lines)
                print(f"✗ {error_msg}")
                print("\nThe FMP price data contains split discontinuities.")
                print("This will cause 10x errors in momentum/return calculations.")
                print("\nOptions:")
                print("  1. Check FMP endpoint being used (prefer /historical-price-eod/full)")
                print("  2. Clear cache and re-download: rm -rf data/cache/fmp/")
                print("  3. Use --auto-normalize-splits to fix automatically")
                print("  4. Skip check (DANGEROUS): --skip-split-check")
                return 1
    else:
        print("\nStep 1b: ⚠️  Skipping split-discontinuity check (--skip-split-check)")
        print("  WARNING: This may result in incorrect momentum/return calculations!")
    
    # Download benchmark prices
    print(f"\nStep 2: Downloading benchmark ({BENCHMARK_TICKER}) prices...")
    try:
        benchmark_prices = download_historical_prices(
            tickers=[BENCHMARK_TICKER],
            start_date=RAW_DATA_START,
            end_date=RAW_DATA_END,
            cache_dir=args.cache_dir,
            api_key=api_key,
        )
        benchmark_prices = dedupe_daily_bars(benchmark_prices, ticker_col="ticker", date_col="date")
    except Exception as e:
        print(f"✗ Failed to download benchmark: {e}")
        return 1
    
    # Download market prices for regime
    print(f"\nStep 3: Downloading market ({MARKET_BENCHMARK}) prices for regime...")
    try:
        market_prices = download_historical_prices(
            tickers=[MARKET_BENCHMARK],
            start_date=RAW_DATA_START,
            end_date=RAW_DATA_END,
            cache_dir=args.cache_dir,
            api_key=api_key,
        )
        market_prices = dedupe_daily_bars(market_prices, ticker_col="ticker", date_col="date")
    except Exception as e:
        print(f"✗ Failed to download market prices: {e}")
        return 1
    
    # Try to download VIX proxy
    print(f"\nStep 4: Downloading VIX proxy ({VIX_PROXY})...")
    vix_prices = None
    try:
        vix_prices = download_historical_prices(
            tickers=[VIX_PROXY],
            start_date=RAW_DATA_START,
            end_date=RAW_DATA_END,
            cache_dir=args.cache_dir,
            api_key=api_key,
        )
        if vix_prices is not None and not vix_prices.empty:
            vix_prices = dedupe_daily_bars(vix_prices, ticker_col="ticker", date_col="date")
    except Exception as e:
        print(f"  Warning: VIX proxy not available: {e}")
    
    # Compute features
    print("\nStep 5: Computing momentum features...")
    features_df = compute_momentum_features(prices_df)
    
    # Filter to evaluation range
    features_df = features_df[
        (features_df["date"] >= EVAL_START) &
        (features_df["date"] <= EVAL_END)
    ]
    print(f"  Features after filtering to eval range: {len(features_df)} rows")
    
    # Compute regime features
    print("\nStep 6: Computing regime features...")
    regime_df = compute_regime_features(market_prices, vix_prices)
    regime_df = regime_df[
        (regime_df["date"] >= EVAL_START) &
        (regime_df["date"] <= EVAL_END)
    ]
    
    # Compute labels
    print("\nStep 7: Computing v2 labels...")
    labels_df = compute_labels(prices_df, benchmark_prices, None, LABEL_HORIZONS)
    labels_df = labels_df[
        (labels_df["as_of_date"] >= EVAL_START) &
        (labels_df["as_of_date"] <= EVAL_END)
    ]
    print(f"  Labels after filtering to eval range: {len(labels_df)} rows")
    
    # =========================================================================
    # FINAL DEDUPLICATION SAFETY NET
    # =========================================================================
    # Even if upstream is clean, guard against any edge cases
    print("\nStep 7b: Final deduplication safety check...")
    
    features_df = features_df.sort_values(["ticker", "date"]).drop_duplicates(
        ["date", "ticker"], keep="last"
    ).reset_index(drop=True)
    
    labels_df = labels_df.sort_values(["ticker", "as_of_date", "horizon"]).drop_duplicates(
        ["as_of_date", "ticker", "horizon"], keep="last"
    ).reset_index(drop=True)
    
    # Hard guard: raise if duplicates still exist
    if features_df.duplicated(["date", "ticker"]).any():
        raise ValueError("features_df still has duplicate (date, ticker) after dedupe!")
    if labels_df.duplicated(["as_of_date", "ticker", "horizon"]).any():
        raise ValueError("labels_df still has duplicate (as_of_date, ticker, horizon) after dedupe!")
    
    print(f"  ✓ No duplicates in features ({len(features_df):,} rows)")
    print(f"  ✓ No duplicates in labels ({len(labels_df):,} rows)")
    
    # Compute data hash
    data_hash = compute_data_hash(features_df, labels_df)
    
    # Prepare metadata
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "build_timestamp": datetime.utcnow().isoformat() + "Z",
        "eval_start": EVAL_START.isoformat(),
        "eval_end": EVAL_END.isoformat(),
        "raw_data_start": RAW_DATA_START.isoformat(),
        "raw_data_end": RAW_DATA_END.isoformat(),
        "n_features": len(features_df),
        "n_labels": len(labels_df),
        "n_regime": len(regime_df),
        "n_tickers": features_df["ticker"].nunique(),
        "horizons": LABEL_HORIZONS,
        "data_hash": data_hash,
    }
    
    # Create DuckDB
    print(f"\nStep 8: Creating DuckDB at {args.output}...")
    create_duckdb_store(args.output, features_df, labels_df, regime_df, metadata)
    
    # Write manifest
    manifest_path = args.output.parent / "DATA_MANIFEST.json"
    with open(manifest_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Wrote manifest to {manifest_path}")
    
    # Summary
    print("\n" + "=" * 60)
    print("BUILD COMPLETE")
    print("=" * 60)
    print(f"\n✓ DuckDB: {args.output}")
    print(f"✓ Manifest: {manifest_path}")
    print(f"\n  Features: {len(features_df):,} rows ({features_df['ticker'].nunique()} tickers)")
    print(f"  Labels: {len(labels_df):,} rows")
    print(f"  Regime: {len(regime_df):,} rows")
    print(f"  Date range: {features_df['date'].min()} to {features_df['date'].max()}")
    print(f"  Data hash: {data_hash[:16]}...")
    
    print("\n" + "-" * 60)
    print("Next: Run Chapter 6 closure with real data:")
    print("  python scripts/run_chapter6_closure.py")
    print("-" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
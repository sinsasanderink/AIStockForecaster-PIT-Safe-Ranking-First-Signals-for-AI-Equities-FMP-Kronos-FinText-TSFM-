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
# EARNINGS DATA DOWNLOAD
# ============================================================================

def download_earnings_data(
    tickers: List[str],
    cache_dir: Path,
    api_key: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Download earnings-related data for all tickers from FMP.
    
    Returns:
        earnings_df: DataFrame with earnings dates and surprises
        filings_df: DataFrame with SEC filing dates (both 10-Q and 10-K)
    """
    from src.data.fmp_client import FMPClient, FMPError
    
    client = FMPClient(api_key=api_key, cache_dir=cache_dir, use_cache=True)
    
    all_earnings = []
    all_filings = []
    
    total = len(tickers)
    logger.info(f"Downloading earnings data for {total} tickers...")
    
    for i, ticker in enumerate(tickers):
        if (i + 1) % 20 == 0:
            logger.info(f"[{i+1}/{total}] Downloading earnings for {ticker}...")
        
        try:
            # Get QUARTERLY income statements (10-Q filings)
            # Use limit=60 to get ~15 years of quarterly filings (2010-2025)
            quarterly_data = client._request("income-statement", 
                                             {"symbol": ticker, "period": "quarter", "limit": 60})
            
            if quarterly_data:
                for stmt in quarterly_data:
                    filing_date_str = stmt.get("fillingDate") or stmt.get("filingDate")
                    period_end_str = stmt.get("date")
                    
                    if filing_date_str and period_end_str:
                        eps_actual = stmt.get("epsdiluted") or stmt.get("eps")
                        
                        all_filings.append({
                            "ticker": ticker,
                            "period_end": period_end_str,
                            "filing_date": filing_date_str,
                            "eps": eps_actual,
                            "revenue": stmt.get("revenue"),
                            "filing_type": "10-Q",  # Quarterly filings are 10-Q
                        })
            
            # Get ANNUAL income statements (10-K filings)
            # These have different filing dates than quarterly
            annual_data = client._request("income-statement", 
                                          {"symbol": ticker, "period": "annual", "limit": 20})
            
            if annual_data:
                for stmt in annual_data:
                    filing_date_str = stmt.get("fillingDate") or stmt.get("filingDate")
                    period_end_str = stmt.get("date")
                    
                    if filing_date_str and period_end_str:
                        all_filings.append({
                            "ticker": ticker,
                            "period_end": period_end_str,
                            "filing_date": filing_date_str,
                            "eps": stmt.get("epsdiluted") or stmt.get("eps"),
                            "revenue": stmt.get("revenue"),
                            "filing_type": "10-K",  # Annual filings are 10-K
                        })
            
            # Try to get earnings calendar/surprises if Premium (endpoint may not exist)
            try:
                surprise_data = client._request("earnings-surprises", {"symbol": ticker})
                if surprise_data:
                    for s in surprise_data:
                        all_earnings.append({
                            "ticker": ticker,
                            "date": s.get("date"),
                            "actual_eps": s.get("actualEarningResult"),
                            "estimated_eps": s.get("estimatedEarning"),
                            "surprise_pct": ((s.get("actualEarningResult") or 0) - (s.get("estimatedEarning") or 0)) 
                                           / abs(s.get("estimatedEarning") or 1) * 100 
                                           if s.get("estimatedEarning") else None,
                        })
            except Exception:
                pass  # Premium endpoint may not be available
                
        except (FMPError, Exception) as e:
            logger.debug(f"Could not download earnings for {ticker}: {e}")
    
    earnings_df = pd.DataFrame(all_earnings) if all_earnings else pd.DataFrame()
    filings_df = pd.DataFrame(all_filings) if all_filings else pd.DataFrame()
    
    # Convert dates
    if not filings_df.empty:
        filings_df["filing_date"] = pd.to_datetime(filings_df["filing_date"]).dt.date
        filings_df["period_end"] = pd.to_datetime(filings_df["period_end"]).dt.date
        # Sort for efficient lookup
        filings_df = filings_df.sort_values(["ticker", "filing_date"])
    
    if not earnings_df.empty:
        earnings_df["date"] = pd.to_datetime(earnings_df["date"]).dt.date
        earnings_df = earnings_df.sort_values(["ticker", "date"])
    
    logger.info(f"Downloaded {len(filings_df)} filing records, {len(earnings_df)} earnings surprises")
    
    return earnings_df, filings_df


def download_fundamental_data(
    tickers: List[str],
    cache_dir: Path,
    api_key: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, str]]:
    """
    Download full financial statement data for fundamental features (Batch 5).
    
    Returns:
        income_df: Full income statement data with all fields
        balance_df: Balance sheet data with equity for ROE
        sector_map: Dict of ticker -> sector (from profile, static)
    """
    from src.data.fmp_client import FMPClient, FMPError
    
    client = FMPClient(api_key=api_key, cache_dir=cache_dir, use_cache=True)
    
    all_income = []
    all_balance = []
    sector_map = {}
    
    total = len(tickers)
    logger.info(f"Downloading fundamental data for {total} tickers (Batch 5)...")
    
    for i, ticker in enumerate(tickers):
        if (i + 1) % 20 == 0:
            logger.info(f"[{i+1}/{total}] Downloading fundamentals for {ticker}...")
        
        try:
            # Get company profile for sector (document: static, not PIT)
            profile = client._request("profile", {"symbol": ticker})
            if profile and len(profile) > 0:
                sector_map[ticker] = profile[0].get("sector", "Unknown")
            
            # Get income statement with ALL fields for fundamentals
            income_data = client._request("income-statement", 
                                         {"symbol": ticker, "period": "quarter", "limit": 60})
            
            if income_data:
                for stmt in income_data:
                    filing_date_str = stmt.get("fillingDate") or stmt.get("filingDate")
                    period_end_str = stmt.get("date")
                    
                    if filing_date_str and period_end_str:
                        all_income.append({
                            "ticker": ticker,
                            "period_end": period_end_str,
                            "filing_date": filing_date_str,
                            # Revenue and profit metrics
                            "revenue": stmt.get("revenue"),
                            "gross_profit": stmt.get("grossProfit"),
                            "operating_income": stmt.get("operatingIncome"),
                            "net_income": stmt.get("netIncome"),
                            # EPS and shares
                            "eps": stmt.get("epsdiluted") or stmt.get("eps"),
                            "shares_diluted": stmt.get("weightedAverageShsOutDiluted"),
                        })
            
            # Get balance sheet for equity (ROE calculation)
            balance_data = client._request("balance-sheet-statement",
                                          {"symbol": ticker, "period": "quarter", "limit": 60})
            
            if balance_data:
                for stmt in balance_data:
                    filing_date_str = stmt.get("fillingDate") or stmt.get("filingDate")
                    period_end_str = stmt.get("date")
                    
                    if filing_date_str and period_end_str:
                        all_balance.append({
                            "ticker": ticker,
                            "period_end": period_end_str,
                            "filing_date": filing_date_str,
                            "total_equity": stmt.get("totalStockholdersEquity"),
                            "total_assets": stmt.get("totalAssets"),
                        })
                        
        except (FMPError, Exception) as e:
            logger.debug(f"Could not download fundamentals for {ticker}: {e}")
    
    income_df = pd.DataFrame(all_income) if all_income else pd.DataFrame()
    balance_df = pd.DataFrame(all_balance) if all_balance else pd.DataFrame()
    
    # Convert dates
    if not income_df.empty:
        income_df["filing_date"] = pd.to_datetime(income_df["filing_date"]).dt.date
        income_df["period_end"] = pd.to_datetime(income_df["period_end"]).dt.date
        income_df = income_df.sort_values(["ticker", "filing_date"])
    
    if not balance_df.empty:
        balance_df["filing_date"] = pd.to_datetime(balance_df["filing_date"]).dt.date
        balance_df["period_end"] = pd.to_datetime(balance_df["period_end"]).dt.date
        balance_df = balance_df.sort_values(["ticker", "filing_date"])
    
    logger.info(f"Downloaded {len(income_df)} income records, {len(balance_df)} balance records")
    logger.info(f"Sector mappings: {len(sector_map)} tickers (NOTE: static, not PIT)")
    
    return income_df, balance_df, sector_map


def compute_fundamental_features(
    features_df: pd.DataFrame,
    income_df: pd.DataFrame,
    balance_df: pd.DataFrame,
    sector_map: Dict[str, str],
) -> pd.DataFrame:
    """
    Compute fundamental features (Batch 5 - Phase 1: filings-only).
    
    RAW features (piecewise-constant between filings, stored for debugging):
    - gross_margin_ttm: TTM gross margin (gross_profit / revenue)
    - operating_margin_ttm: TTM operating margin (operating_income / revenue)
    - revenue_growth_yoy: YoY TTM revenue growth
    - roe_raw: ROE (TTM net income / avg equity)
    
    Z-SCORE features (cross-sectional, also stepwise per-ticker):
    - gross_margin_vs_sector: z-score vs sector at ticker's filing date, forward-filled
    - operating_margin_vs_sector: z-score vs sector at ticker's filing date, forward-filled
    - revenue_growth_vs_sector: z-score vs sector at ticker's filing date, forward-filled
    - roe_zscore: z-score vs universe at ticker's filing date, forward-filled
    
    Implementation:
    1. Compute raw TTM at each ticker's filing dates (piecewise-constant)
    2. At each ticker's filing, compute z-score using 90-day lookback cross-section
    3. Forward-fill both raw and z-score values between filings
    
    This ensures BOTH raw and z-score features are stepwise per-ticker.
    """
    logger.info("Computing fundamental features (Phase 1: filings-only)...")
    
    # Initialize columns - RAW + Z-SCORE
    features_df["sector"] = None
    # Raw TTM (piecewise-constant)
    features_df["gross_margin_ttm"] = None
    features_df["operating_margin_ttm"] = None
    features_df["revenue_growth_yoy"] = None
    features_df["roe_raw"] = None
    # Z-scores (stepwise per-ticker)
    features_df["gross_margin_vs_sector"] = None
    features_df["operating_margin_vs_sector"] = None
    features_df["revenue_growth_vs_sector"] = None
    features_df["roe_zscore"] = None
    
    if income_df.empty:
        logger.warning("No income data, fundamental features will be null")
        return features_df
    
    # Convert dates to Timestamp for comparison
    income_df = income_df.copy()
    income_df["filing_date"] = pd.to_datetime(income_df["filing_date"])
    income_df["period_end"] = pd.to_datetime(income_df["period_end"])
    
    if not balance_df.empty:
        balance_df = balance_df.copy()
        balance_df["filing_date"] = pd.to_datetime(balance_df["filing_date"])
        balance_df["period_end"] = pd.to_datetime(balance_df["period_end"])
    
    # Build per-ticker lookup
    income_by_ticker = income_df.groupby("ticker")
    balance_by_ticker = balance_df.groupby("ticker") if not balance_df.empty else None
    
    MIN_SECTOR_SIZE = 5
    
    # ==========================================================================
    # STEP 1: Compute raw TTM at each ticker's FILING DATES (not every row)
    # ==========================================================================
    logger.info("  Step 1: Computing raw TTM at filing dates...")
    ticker_filings_data = {}  # ticker -> list of {filing_date, gross_margin, ...}
    
    for ticker in features_df["ticker"].unique():
        sector = sector_map.get(ticker, "Unknown")
        
        if ticker not in income_by_ticker.groups:
            ticker_filings_data[ticker] = []
            continue
        
        ticker_income = income_by_ticker.get_group(ticker)
        # Get unique filing dates, sorted
        filing_dates = sorted(ticker_income["filing_date"].unique())
        
        filings_list = []
        for filing_date in filing_dates:
            # Get all filings up to and including this filing_date
            past = ticker_income[ticker_income["filing_date"] <= filing_date]
            
            if len(past) < 4:
                continue
            
            # Sort by period_end descending
            past = past.sort_values("period_end", ascending=False)
            
            # TTM (last 4 quarters)
            ttm = past.head(4)
            ttm_revenue = ttm["revenue"].sum()
            ttm_gross_profit = ttm["gross_profit"].sum()
            ttm_operating_income = ttm["operating_income"].sum()
            ttm_net_income = ttm["net_income"].sum()
            
            metrics = {
                "filing_date": filing_date,
                "ticker": ticker,
                "sector": sector,
                "gross_margin": ttm_gross_profit / ttm_revenue if ttm_revenue > 0 else None,
                "operating_margin": ttm_operating_income / ttm_revenue if ttm_revenue > 0 else None,
                "revenue_growth": None,
                "roe": None,
            }
            
            # YoY revenue growth
            if len(past) >= 8:
                prior_ttm = past.iloc[4:8]
                prior_revenue = prior_ttm["revenue"].sum()
                if prior_revenue > 0:
                    metrics["revenue_growth"] = (ttm_revenue / prior_revenue) - 1
            
            # ROE
            if balance_by_ticker and ticker in balance_by_ticker.groups:
                ticker_balance = balance_by_ticker.get_group(ticker)
                past_balance = ticker_balance[ticker_balance["filing_date"] <= filing_date]
                
                if not past_balance.empty:
                    past_balance = past_balance.sort_values("period_end", ascending=False)
                    avg_equity = past_balance.head(4)["total_equity"].mean()
                    
                    if avg_equity and avg_equity > 0:
                        metrics["roe"] = ttm_net_income / avg_equity
            
            filings_list.append(metrics)
        
        ticker_filings_data[ticker] = filings_list
    
    n_filings_total = sum(len(v) for v in ticker_filings_data.values())
    logger.info(f"  Computed TTM for {n_filings_total} filings across {len(ticker_filings_data)} tickers")
    
    # ==========================================================================
    # STEP 2: At each ticker's filing, compute z-scores using 90-day lookback
    # ==========================================================================
    logger.info("  Step 2: Computing z-scores at filing dates (90-day lookback)...")
    
    ticker_zscore_data = {}  # ticker -> list of {filing_date, *_vs_sector, roe_zscore}
    
    for ticker, filings_list in ticker_filings_data.items():
        zscore_list = []
        sector = sector_map.get(ticker, "Unknown")
        
        for metrics in filings_list:
            filing_date = metrics["filing_date"]
            
            # Build cross-section: get other tickers' most recent filing within 90-day lookback
            lookback = pd.Timedelta(days=90)
            sector_cross_section = []
            universe_cross_section = []
            
            for other_ticker, other_filings in ticker_filings_data.items():
                other_sector = sector_map.get(other_ticker, "Unknown")
                
                # Find other_ticker's most recent filing as of this filing_date
                valid = [f for f in other_filings
                        if f["filing_date"] <= filing_date
                        and f["filing_date"] >= filing_date - lookback]
                if valid:
                    recent = max(valid, key=lambda x: x["filing_date"])
                    universe_cross_section.append(recent)
                    if other_sector == sector:
                        sector_cross_section.append(recent)
            
            zscore_metrics = {
                "filing_date": filing_date,
                "gross_margin_vs_sector": None,
                "operating_margin_vs_sector": None,
                "revenue_growth_vs_sector": None,
                "roe_zscore": None,
            }
            
            # Sector z-scores
            if len(sector_cross_section) >= MIN_SECTOR_SIZE:
                # Gross margin
                vals = [f["gross_margin"] for f in sector_cross_section if f["gross_margin"] is not None]
                if len(vals) >= MIN_SECTOR_SIZE and metrics["gross_margin"] is not None:
                    mean_val, std_val = np.mean(vals), np.std(vals, ddof=1)
                    if std_val > 1e-8:
                        zscore_metrics["gross_margin_vs_sector"] = (metrics["gross_margin"] - mean_val) / std_val
                
                # Operating margin
                vals = [f["operating_margin"] for f in sector_cross_section if f["operating_margin"] is not None]
                if len(vals) >= MIN_SECTOR_SIZE and metrics["operating_margin"] is not None:
                    mean_val, std_val = np.mean(vals), np.std(vals, ddof=1)
                    if std_val > 1e-8:
                        zscore_metrics["operating_margin_vs_sector"] = (metrics["operating_margin"] - mean_val) / std_val
                
                # Revenue growth
                vals = [f["revenue_growth"] for f in sector_cross_section if f["revenue_growth"] is not None]
                if len(vals) >= MIN_SECTOR_SIZE and metrics["revenue_growth"] is not None:
                    mean_val, std_val = np.mean(vals), np.std(vals, ddof=1)
                    if std_val > 1e-8:
                        zscore_metrics["revenue_growth_vs_sector"] = (metrics["revenue_growth"] - mean_val) / std_val
            
            # Universe ROE z-score
            if len(universe_cross_section) >= MIN_SECTOR_SIZE:
                vals = [f["roe"] for f in universe_cross_section if f["roe"] is not None]
                if len(vals) >= MIN_SECTOR_SIZE and metrics["roe"] is not None:
                    mean_val, std_val = np.mean(vals), np.std(vals, ddof=1)
                    if std_val > 1e-8:
                        zscore_metrics["roe_zscore"] = (metrics["roe"] - mean_val) / std_val
            
            zscore_list.append(zscore_metrics)
        
        ticker_zscore_data[ticker] = zscore_list
    
    # ==========================================================================
    # STEP 3: Forward-fill raw and z-score values into features_df
    # ==========================================================================
    logger.info("  Step 3: Forward-filling features from filing dates...")
    
    for ticker in features_df["ticker"].unique():
        sector = sector_map.get(ticker, "Unknown")
        ticker_mask = features_df["ticker"] == ticker
        features_df.loc[ticker_mask, "sector"] = sector
        
        filings_list = ticker_filings_data.get(ticker, [])
        zscore_list = ticker_zscore_data.get(ticker, [])
        
        if not filings_list:
            continue
        
        # Build dicts keyed by filing_date for robust lookup (not positional index)
        filings_by_date = {f["filing_date"]: f for f in filings_list}
        zscore_by_date = {z["filing_date"]: z for z in zscore_list}
        
        # Convert to sorted array for binary search
        filing_dates = np.array(sorted(filings_by_date.keys()))
        
        # For each date in features_df for this ticker
        for row_idx in features_df[ticker_mask].index:
            row_date = pd.Timestamp(features_df.at[row_idx, "date"])
            
            # Find most recent filing <= row_date
            valid_mask = filing_dates <= row_date
            if not valid_mask.any():
                continue
            
            # Get the filing_date (not index)
            matched_filing_date = filing_dates[valid_mask][-1]
            
            # Set raw TTM values (lookup by filing_date)
            raw = filings_by_date.get(matched_filing_date)
            if raw:
                if raw["gross_margin"] is not None:
                    features_df.at[row_idx, "gross_margin_ttm"] = raw["gross_margin"]
                if raw["operating_margin"] is not None:
                    features_df.at[row_idx, "operating_margin_ttm"] = raw["operating_margin"]
                if raw["revenue_growth"] is not None:
                    features_df.at[row_idx, "revenue_growth_yoy"] = raw["revenue_growth"]
                if raw["roe"] is not None:
                    features_df.at[row_idx, "roe_raw"] = raw["roe"]
            
            # Set z-score values (lookup by filing_date, not positional index)
            zs = zscore_by_date.get(matched_filing_date)
            if zs:
                if zs["gross_margin_vs_sector"] is not None:
                    features_df.at[row_idx, "gross_margin_vs_sector"] = zs["gross_margin_vs_sector"]
                if zs["operating_margin_vs_sector"] is not None:
                    features_df.at[row_idx, "operating_margin_vs_sector"] = zs["operating_margin_vs_sector"]
                if zs["revenue_growth_vs_sector"] is not None:
                    features_df.at[row_idx, "revenue_growth_vs_sector"] = zs["revenue_growth_vs_sector"]
                if zs["roe_zscore"] is not None:
                    features_df.at[row_idx, "roe_zscore"] = zs["roe_zscore"]
    
    # Log coverage
    raw_cols = ["gross_margin_ttm", "operating_margin_ttm", "revenue_growth_yoy", "roe_raw"]
    zscore_cols = ["gross_margin_vs_sector", "operating_margin_vs_sector", "revenue_growth_vs_sector", "roe_zscore"]
    
    logger.info("  Raw TTM coverage:")
    for col in raw_cols:
        non_null = features_df[col].notna().sum()
        logger.info(f"    {col}: {non_null:,} non-null ({100*non_null/len(features_df):.1f}%)")
    
    logger.info("  Z-score coverage:")
    for col in zscore_cols:
        non_null = features_df[col].notna().sum()
        logger.info(f"    {col}: {non_null:,} non-null ({100*non_null/len(features_df):.1f}%)")
    
    logger.info(f"Computed fundamental features for {len(features_df)} rows")
    
    return features_df


def compute_event_features(
    features_df: pd.DataFrame,
    earnings_df: pd.DataFrame,
    filings_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add event/earnings features to the features DataFrame.
    
    Features added:
    - days_to_earnings: Days until next expected earnings (estimated)
    - days_since_earnings: Days since last earnings release
    - in_pead_window: Boolean if within 63-day post-earnings window
    - pead_window_day: Which day of PEAD window (1-63)
    - last_surprise_pct: Last quarter's EPS surprise %
    - avg_surprise_4q: Average surprise last 4 quarters
    - surprise_streak: Consecutive beats (positive) or misses (negative)
    - surprise_zscore: Cross-sectional z-score of surprise (computed later)
    - earnings_vol: Std dev of surprises over 8 quarters
    - days_since_10k: Days since last annual filing
    - days_since_10q: Days since last quarterly filing
    - reports_bmo: Boolean if typically reports before market open
    """
    logger.info("Computing event features...")
    
    PEAD_WINDOW = 63  # Post-earnings announcement drift window
    
    # Initialize event columns
    features_df["days_to_earnings"] = None
    features_df["days_since_earnings"] = None
    features_df["in_pead_window"] = False
    features_df["pead_window_day"] = None
    features_df["last_surprise_pct"] = None
    features_df["avg_surprise_4q"] = None
    features_df["surprise_streak"] = 0
    features_df["earnings_vol"] = None
    features_df["days_since_10k"] = None
    features_df["days_since_10q"] = None
    features_df["reports_bmo"] = None
    
    if filings_df.empty:
        logger.warning("No filings data available, event features will be null")
        features_df["surprise_zscore"] = None
        return features_df
    
    # Build lookup structures for efficient computation
    filings_by_ticker = filings_df.groupby("ticker")
    
    # If we have earnings surprises, build surprise lookup
    surprises_by_ticker = {}
    if not earnings_df.empty:
        for ticker, group in earnings_df.groupby("ticker"):
            surprises_by_ticker[ticker] = group.sort_values("date", ascending=False)
    
    # Convert filing_date to Timestamp for consistent comparison
    # (features_df['date'] is Timestamp, filings_df['filing_date'] is datetime.date)
    if not filings_df.empty:
        filings_df = filings_df.copy()
        filings_df["filing_date"] = pd.to_datetime(filings_df["filing_date"])
        filings_df["period_end"] = pd.to_datetime(filings_df["period_end"])
        # Rebuild the groupby with converted dates
        filings_by_ticker = filings_df.groupby("ticker")
    
    # Compute features for each row
    for idx, row in features_df.iterrows():
        ticker = row["ticker"]
        # Convert asof_date to Timestamp for consistent comparison
        asof_date = pd.Timestamp(row["date"]) if not isinstance(row["date"], pd.Timestamp) else row["date"]
        
        if ticker not in filings_by_ticker.groups:
            continue
        
        ticker_filings = filings_by_ticker.get_group(ticker)
        
        # Get filings on or before asof_date (PIT-safe)
        # Use <= to include filings on the same day (days_since = 0 on filing day)
        # Both are now Timestamps, so comparison works correctly
        past_filings = ticker_filings[ticker_filings["filing_date"] <= asof_date]
        future_filings = ticker_filings[ticker_filings["filing_date"] > asof_date]
        
        if not past_filings.empty:
            # Most recent filing (any type)
            last_filing = past_filings.iloc[-1]
            days_since = (asof_date - last_filing["filing_date"]).days
            features_df.at[idx, "days_since_earnings"] = days_since
            
            # PEAD window
            if days_since <= PEAD_WINDOW:
                features_df.at[idx, "in_pead_window"] = True
                features_df.at[idx, "pead_window_day"] = days_since
            
            # 10-K vs 10-Q: Use actual filing_type field (set in download_earnings_data)
            # Filter by filing_type directly instead of inferring from period_end month
            quarterly_filings = past_filings[past_filings["filing_type"] == "10-Q"]
            annual_filings = past_filings[past_filings["filing_type"] == "10-K"]
            
            # days_since_10q: Days since most recent 10-Q filing
            if not quarterly_filings.empty:
                most_recent_10q = quarterly_filings.iloc[-1]
                features_df.at[idx, "days_since_10q"] = (
                    asof_date - most_recent_10q["filing_date"]
                ).days
            
            # days_since_10k: Days since most recent 10-K filing
            if not annual_filings.empty:
                most_recent_10k = annual_filings.iloc[-1]
                features_df.at[idx, "days_since_10k"] = (
                    asof_date - most_recent_10k["filing_date"]
                ).days
        
        # Days to next earnings (estimated from quarterly pattern)
        # Use clean pandas diff() approach for spacing calculation
        if not past_filings.empty and len(past_filings) >= 2:
            # Sort filing dates and compute spacings using pandas (clean approach)
            fd = past_filings["filing_date"].sort_values()
            spacings_series = fd.diff().dt.days.dropna()
            
            # Filter to reasonable quarterly range [60, 120] days
            valid_spacings = spacings_series[(spacings_series >= 60) & (spacings_series <= 120)]
            
            if len(valid_spacings) > 0:
                avg_spacing = valid_spacings.tail(4).mean()  # Use last 4 spacings
            else:
                avg_spacing = 91  # Default quarterly if no valid spacings
            
            if pd.notna(avg_spacing) and avg_spacing > 0:
                last_date = fd.iloc[-1]
                days_since_last = (asof_date - last_date).days
                days_to_next = int(avg_spacing - (days_since_last % avg_spacing))
                if days_to_next <= 0:
                    days_to_next += int(avg_spacing)
                features_df.at[idx, "days_to_earnings"] = days_to_next
        
        # Surprise features (if available)
        if ticker in surprises_by_ticker:
            surprises = surprises_by_ticker[ticker]
            # Get surprises before asof_date
            past_surprises = surprises[surprises["date"] < asof_date]
            
            if not past_surprises.empty:
                # Last surprise
                valid_surprises = past_surprises[past_surprises["surprise_pct"].notna()]
                
                if not valid_surprises.empty:
                    features_df.at[idx, "last_surprise_pct"] = valid_surprises.iloc[0]["surprise_pct"]
                    
                    # Average 4Q
                    if len(valid_surprises) >= 4:
                        features_df.at[idx, "avg_surprise_4q"] = valid_surprises.iloc[:4]["surprise_pct"].mean()
                    
                    # Surprise streak
                    streak = 0
                    for _, s in valid_surprises.iterrows():
                        val = s["surprise_pct"]
                        if val > 0:
                            if streak >= 0:
                                streak += 1
                            else:
                                break
                        elif val < 0:
                            if streak <= 0:
                                streak -= 1
                            else:
                                break
                        else:
                            break
                    features_df.at[idx, "surprise_streak"] = streak
                    
                    # Earnings volatility (8Q)
                    if len(valid_surprises) >= 3:
                        features_df.at[idx, "earnings_vol"] = valid_surprises.iloc[:8]["surprise_pct"].std()
    
    # Cross-sectional surprise z-score (by date)
    logger.info("Computing cross-sectional surprise z-scores...")
    features_df["surprise_zscore"] = None
    
    for date_val in features_df["date"].unique():
        date_mask = features_df["date"] == date_val
        surprise_vals = features_df.loc[date_mask, "last_surprise_pct"]
        
        if surprise_vals.notna().sum() >= 5:
            mean_s = surprise_vals.mean()
            std_s = surprise_vals.std()
            if std_s > 1e-8:
                features_df.loc[date_mask, "surprise_zscore"] = (
                    features_df.loc[date_mask, "last_surprise_pct"] - mean_s
                ) / std_s
    
    # reports_bmo: We don't have timing data, leave as None
    # (Could estimate from market price gaps at filing dates in the future)
    
    logger.info(f"Computed event features for {len(features_df)} rows")
    
    return features_df


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
            
            # Vol of vol (volatility of volatility)
            vol_of_vol = None
            if i >= VOL_WINDOW_60D:
                # Compute rolling 20d vol over 60d window (3 observations)
                vol_series = []
                for j in range(i - VOL_WINDOW_60D + VOL_WINDOW_20D, i + 1, VOL_WINDOW_20D):
                    if j >= VOL_WINDOW_20D:
                        rets = np.diff(np.log(closes[j-VOL_WINDOW_20D:j+1]))
                        if len(rets) > 0:
                            vol_series.append(np.std(rets) * np.sqrt(252))
                if len(vol_series) >= 2:
                    vol_of_vol = float(np.std(vol_series))
            
            # Max drawdown (60d)
            max_drawdown_60d = None
            if i >= VOL_WINDOW_60D:
                window_prices = closes[i-VOL_WINDOW_60D:i+1]
                running_max = np.maximum.accumulate(window_prices)
                drawdowns = (window_prices - running_max) / running_max
                max_drawdown_60d = float(np.min(drawdowns))
            
            # Distance from high (60d)
            dist_from_high_60d = None
            if i >= VOL_WINDOW_60D:
                window_high = np.max(closes[i-VOL_WINDOW_60D:i+1])
                dist_from_high_60d = float((closes[i] - window_high) / window_high)
            
            # ADV 60d
            adv_60d = None
            if volumes is not None and i >= VOL_WINDOW_60D:
                dollar_volumes = closes[i-VOL_WINDOW_60D:i] * volumes[i-VOL_WINDOW_60D:i]
                adv_60d = float(np.mean(dollar_volumes))
            
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
                "adv_60d": adv_60d,
                "vol_20d": vol_20d,
                "vol_60d": vol_60d,
                "vol_of_vol": vol_of_vol,
                "max_drawdown_60d": max_drawdown_60d,
                "dist_from_high_60d": dist_from_high_60d,
            })
    
    features_df = pd.DataFrame(features)
    logger.info(f"Computed features for {features_df['ticker'].nunique()} tickers, {len(features_df)} rows")
    
    # Compute cross-sectional features (relative strength)
    logger.info("Computing cross-sectional relative strength features...")
    for date_val in features_df['date'].unique():
        date_mask = features_df['date'] == date_val
        
        # 1-month relative strength (z-score)
        mom_1m_vals = features_df.loc[date_mask, 'mom_1m']
        if mom_1m_vals.notna().sum() >= 5:  # Need at least 5 stocks for meaningful z-score
            mean_mom = mom_1m_vals.mean()
            std_mom = mom_1m_vals.std()
            if std_mom > 1e-8:
                features_df.loc[date_mask, 'rel_strength_1m'] = (mom_1m_vals - mean_mom) / std_mom
        
        # 3-month relative strength (z-score)
        mom_3m_vals = features_df.loc[date_mask, 'mom_3m']
        if mom_3m_vals.notna().sum() >= 5:
            mean_mom = mom_3m_vals.mean()
            std_mom = mom_3m_vals.std()
            if std_mom > 1e-8:
                features_df.loc[date_mask, 'rel_strength_3m'] = (mom_3m_vals - mean_mom) / std_mom
    
    # Initialize columns if not present
    if 'rel_strength_1m' not in features_df.columns:
        features_df['rel_strength_1m'] = None
    if 'rel_strength_3m' not in features_df.columns:
        features_df['rel_strength_3m'] = None
    if 'beta_252d' not in features_df.columns:
        features_df['beta_252d'] = None  # TODO: Requires benchmark data, defer for now
    
    logger.info(f"Added cross-sectional features (rel_strength_1m, rel_strength_3m)")
    
    # Compute missingness features
    logger.info("Computing missingness features...")
    feature_cols = [c for c in features_df.columns if c not in ['date', 'ticker', 'stable_id']]
    features_df['coverage_pct'] = features_df[feature_cols].notna().sum(axis=1) / len(feature_cols)
    
    # New stock indicator (has < 252 trading days of history)
    stock_lengths = features_df.groupby('ticker').size()
    new_stocks = stock_lengths[stock_lengths < 252].index
    features_df['is_new_stock'] = features_df['ticker'].isin(new_stocks)
    
    logger.info(f"Added missingness features (coverage_pct, is_new_stock)")
    
    return features_df


# NOTE: Old AlphaVantage-based download_earnings_data and compute_event_features removed
# (superseded by FMP-based versions at lines 255 and 345)


# NOTE: Old AlphaVantage-based earnings code has been removed.
# Event features are now computed using FMP data (see download_earnings_data and
# compute_event_features functions above, around lines 255 and 345).


# ============================================================================
# REGIME FEATURES
# ============================================================================


def compute_regime_features_placeholder_removed():
    """REMOVED - This was a duplicate. See compute_regime_features below."""
    pass


# NOTE: The old AlphaVantage-based compute_event_features was here.
# It has been removed and replaced by the FMP-based version at line 345.


def _old_compute_event_features_removed(
    features_df,
    earnings_data,
):
    """
    Compute event/earnings features for all rows in features_df.
    
    Features computed:
    - days_to_earnings: Days until next earnings (estimated if no calendar)
    - days_since_earnings: Days since last earnings
    - in_pead_window: Boolean, within 63-day post-earnings drift window
    - pead_window_day: Which day of PEAD (1-63)
    - last_surprise_pct: Last quarter's surprise %
    - avg_surprise_4q: Average surprise over last 4 quarters
    - surprise_streak: Consecutive beats (>0) or misses (<0)
    - surprise_zscore: Cross-sectional z-score of surprise
    - earnings_vol: Std dev of surprises (8Q)
    - days_since_10k: Days since last 10-K (approximated as ~365d after year-end)
    - days_since_10q: Days since last 10-Q (approximated as ~45d after quarter-end)
    - reports_bmo: Boolean, typically reports before market open
    
    All features are PIT-safe: only use data that was available at the as_of_date.
    """
    logger.info("Computing event features...")
    
    PEAD_WINDOW = 63  # Post-Earnings Announcement Drift window (trading days)
    
    # Initialize all event columns with None
    event_cols = [
        'days_to_earnings', 'days_since_earnings', 'in_pead_window', 'pead_window_day',
        'last_surprise_pct', 'avg_surprise_4q', 'surprise_streak', 'surprise_zscore',
        'earnings_vol', 'days_since_10k', 'days_since_10q', 'reports_bmo'
    ]
    for col in event_cols:
        features_df[col] = None
    
    if not earnings_data:
        logger.warning("No earnings data available, event features will be None")
        return features_df
    
    # Process each ticker
    processed = 0
    for ticker in features_df['ticker'].unique():
        ticker_mask = features_df['ticker'] == ticker
        ticker_earnings = earnings_data.get(ticker, [])
        
        if not ticker_earnings:
            continue
        
        # Parse and sort earnings dates
        earnings_records = []
        for rec in ticker_earnings:
            try:
                reported_date = pd.to_datetime(rec['reported_date']).date()
                fiscal_date = pd.to_datetime(rec['fiscal_date']).date()
                earnings_records.append({
                    'reported_date': reported_date,
                    'fiscal_date': fiscal_date,
                    'surprise_pct': rec.get('surprise_pct', 0) or 0,
                    'report_time': rec.get('report_time', 'post-market'),
                })
            except (ValueError, TypeError):
                continue
        
        if not earnings_records:
            continue
        
        # Sort by reported date (most recent first)
        earnings_records.sort(key=lambda x: x['reported_date'], reverse=True)
        
        # Get ticker rows
        ticker_rows = features_df[ticker_mask].copy()
        
        for idx in ticker_rows.index:
            asof_date = features_df.loc[idx, 'date']
            if isinstance(asof_date, str):
                asof_date = pd.to_datetime(asof_date).date()
            elif hasattr(asof_date, 'date'):
                asof_date = asof_date.date() if callable(getattr(asof_date, 'date', None)) else asof_date
            
            # Get past earnings (PIT-safe: reported_date <= asof_date)
            past_earnings = [e for e in earnings_records if e['reported_date'] <= asof_date]
            
            if past_earnings:
                # Days since last earnings
                last_earnings = past_earnings[0]
                days_since = (asof_date - last_earnings['reported_date']).days
                features_df.loc[idx, 'days_since_earnings'] = days_since
                
                # PEAD window
                if days_since <= PEAD_WINDOW:
                    features_df.loc[idx, 'in_pead_window'] = True
                    features_df.loc[idx, 'pead_window_day'] = days_since
                else:
                    features_df.loc[idx, 'in_pead_window'] = False
                
                # Last surprise
                features_df.loc[idx, 'last_surprise_pct'] = last_earnings['surprise_pct']
                
                # Average surprise (4Q)
                if len(past_earnings) >= 4:
                    surprises_4q = [e['surprise_pct'] for e in past_earnings[:4]]
                    features_df.loc[idx, 'avg_surprise_4q'] = sum(surprises_4q) / len(surprises_4q)
                
                # Surprise streak (consecutive beats or misses)
                streak = 0
                for e in past_earnings:
                    if e['surprise_pct'] > 0:
                        if streak >= 0:
                            streak += 1
                        else:
                            break
                    elif e['surprise_pct'] < 0:
                        if streak <= 0:
                            streak -= 1
                        else:
                            break
                    else:
                        break  # Zero breaks streak
                features_df.loc[idx, 'surprise_streak'] = streak
                
                # Earnings volatility (std of surprises over 8Q)
                if len(past_earnings) >= 3:
                    surprises = [e['surprise_pct'] for e in past_earnings[:8]]
                    features_df.loc[idx, 'earnings_vol'] = np.std(surprises)
                
                # Reports BMO (before market open) tendency
                bmo_count = sum(1 for e in past_earnings[:8] if e['report_time'] in ['pre-market', 'bmo'])
                if len(past_earnings) >= 2:
                    features_df.loc[idx, 'reports_bmo'] = bmo_count > len(past_earnings[:8]) / 2
            
            # Days to next earnings (estimate from past pattern)
            future_earnings = [e for e in earnings_records if e['reported_date'] > asof_date]
            if future_earnings:
                # Use known future date if available in data
                next_earnings = min(future_earnings, key=lambda x: x['reported_date'])
                features_df.loc[idx, 'days_to_earnings'] = (next_earnings['reported_date'] - asof_date).days
            elif past_earnings:
                # Estimate: ~91 days (1 quarter) from last earnings
                estimated_next = last_earnings['reported_date'] + timedelta(days=91)
                if estimated_next > asof_date:
                    features_df.loc[idx, 'days_to_earnings'] = (estimated_next - asof_date).days
            
            # Filing features (approximate based on fiscal dates)
            # 10-Q: ~45 days after quarter end, 10-K: ~60-90 days after fiscal year end
            if past_earnings:
                # Approximate 10-Q: most recent quarter filing
                features_df.loc[idx, 'days_since_10q'] = days_since + 45  # Approx filing lag
                
                # Approximate 10-K: look for year-end fiscal date
                for e in past_earnings:
                    if e['fiscal_date'].month in [12, 1]:  # Year-end
                        days_since_10k = (asof_date - e['reported_date']).days
                        features_df.loc[idx, 'days_since_10k'] = days_since_10k
                        break
        
        processed += 1
        if processed % 20 == 0:
            logger.info(f"  Processed event features for {processed} tickers...")
    
    # Cross-sectional surprise z-score (per date)
    logger.info("Computing cross-sectional surprise z-scores...")
    for date_val in features_df['date'].unique():
        date_mask = features_df['date'] == date_val
        surprises = features_df.loc[date_mask, 'last_surprise_pct'].dropna()
        
        if len(surprises) >= 5:
            mean_s = surprises.mean()
            std_s = surprises.std()
            if std_s > 1e-8:
                for idx in features_df[date_mask & features_df['last_surprise_pct'].notna()].index:
                    features_df.loc[idx, 'surprise_zscore'] = (
                        features_df.loc[idx, 'last_surprise_pct'] - mean_s
                    ) / std_s
    
    # Count coverage
    non_null = features_df['days_since_earnings'].notna().sum()
    logger.info(f"Event features computed: {non_null}/{len(features_df)} rows have earnings data ({100*non_null/len(features_df):.1f}%)")
    
    return features_df


def compute_regime_features(
    market_prices: pd.DataFrame,
    vix_prices: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Compute market regime features (Batch 4 expanded).
    
    All windows are BACKWARD-LOOKING to avoid leakage.
    
    Returns DataFrame with:
    - date
    - VIX features (4): vix_level, vix_percentile, vix_change_5d, vix_regime
    - Market features (5): market_return_5d, market_return_21d, market_return_63d, 
                           market_vol_21d, market_regime
    - Technical (3): above_ma_50, above_ma_200, ma_50_200_cross
    - Legacy: market_return_20d, market_vol_20d, vix_percentile_252d (for backward compat)
    """
    logger.info("Computing regime features (Batch 4 expanded)...")
    
    df = market_prices.copy()
    if "date" not in df.columns and "Date" in df.columns:
        df["date"] = pd.to_datetime(df["Date"]).dt.date
    else:
        df["date"] = pd.to_datetime(df["date"]).dt.date
    
    df = df.sort_values("date").reset_index(drop=True)
    
    close_col = "close" if "close" in df.columns else "Close"
    closes = df[close_col].values
    dates = df["date"].values
    
    # Pre-compute moving averages for efficiency (backward-looking)
    ma_50 = pd.Series(closes).rolling(window=50, min_periods=50).mean().values
    ma_200 = pd.Series(closes).rolling(window=200, min_periods=200).mean().values
    
    regime_features = []
    
    # Start after longest lookback (200 days for MA200)
    min_lookback = 200
    
    for i in range(min_lookback, len(df)):
        row_date = dates[i]
        current_close = closes[i]
        
        # === Market Return Features (backward-looking) ===
        # IMPORTANT: We use closes[i] / closes[i-N] - 1, which is return from N days ago to today
        market_return_5d = (closes[i] / closes[i - 5] - 1) if i >= 5 else None
        market_return_20d = (closes[i] / closes[i - 20] - 1) if i >= 20 else None
        market_return_21d = (closes[i] / closes[i - 21] - 1) if i >= 21 else None
        market_return_63d = (closes[i] / closes[i - 63] - 1) if i >= 63 else None
        
        # === Market Volatility (backward-looking, annualized) ===
        market_vol_20d = None
        market_vol_21d = None
        if i >= 21:
            returns_21d = np.diff(np.log(closes[i-21:i+1]))
            market_vol_21d = float(np.std(returns_21d) * np.sqrt(252))
            returns_20d = np.diff(np.log(closes[i-20:i+1]))
            market_vol_20d = float(np.std(returns_20d) * np.sqrt(252))
        
        # === Market Regime (based on recent trend + volatility) ===
        # Simple regime: 1 = bull (return > 0, low vol), -1 = bear, 0 = neutral
        market_regime = 0  # neutral
        if market_return_21d is not None and market_vol_21d is not None:
            if market_return_21d > 0.02 and market_vol_21d < 0.20:
                market_regime = 1  # bull
            elif market_return_21d < -0.02 or market_vol_21d > 0.30:
                market_regime = -1  # bear
        
        # === Technical Features (backward-looking MAs) ===
        above_ma_50 = 1 if current_close > ma_50[i] else 0 if not np.isnan(ma_50[i]) else None
        above_ma_200 = 1 if current_close > ma_200[i] else 0 if not np.isnan(ma_200[i]) else None
        
        # MA crossover: 1 if MA50 > MA200 (golden cross), -1 if MA50 < MA200 (death cross)
        ma_50_200_cross = None
        if not np.isnan(ma_50[i]) and not np.isnan(ma_200[i]):
            ma_50_200_cross = 1 if ma_50[i] > ma_200[i] else -1
        
        regime_features.append({
            "date": row_date,
            # Market returns (backward-looking)
            "market_return_5d": market_return_5d,
            "market_return_20d": market_return_20d,  # Legacy
            "market_return_21d": market_return_21d,
            "market_return_63d": market_return_63d,
            # Volatility
            "market_vol_20d": market_vol_20d,  # Legacy
            "market_vol_21d": market_vol_21d,
            # Regime
            "market_regime": market_regime,
            # Technical
            "above_ma_50": above_ma_50,
            "above_ma_200": above_ma_200,
            "ma_50_200_cross": ma_50_200_cross,
        })
    
    regime_df = pd.DataFrame(regime_features)
    
    # === VIX Features ===
    if vix_prices is not None and len(vix_prices) > VIX_PERCENTILE_WINDOW:
        vix_df = vix_prices.copy()
        if "date" not in vix_df.columns and "Date" in vix_df.columns:
            vix_df["date"] = pd.to_datetime(vix_df["Date"]).dt.date
        else:
            vix_df["date"] = pd.to_datetime(vix_df["date"]).dt.date
        
        vix_df = vix_df.sort_values("date")
        vix_close_col = "close" if "close" in vix_df.columns else "Close"
        
        # VIX level (current value)
        vix_df["vix_level"] = vix_df[vix_close_col]
        
        # VIX 5-day change (backward-looking)
        vix_df["vix_change_5d"] = vix_df[vix_close_col].pct_change(5)
        
        # VIX percentile (rolling 252-day)
        def rolling_percentile(x):
            if len(x) < VIX_PERCENTILE_WINDOW:
                return np.nan
            return (x.values[-1] <= x.values).sum() / len(x) * 100
        
        vix_df["vix_percentile"] = vix_df[vix_close_col].rolling(
            window=VIX_PERCENTILE_WINDOW, min_periods=VIX_PERCENTILE_WINDOW
        ).apply(rolling_percentile, raw=False)
        
        # VIX regime: 0 = low (<25th pct), 1 = normal, 2 = elevated (>75th pct), 3 = extreme (>90th pct)
        def vix_regime_from_percentile(pct):
            if pd.isna(pct):
                return None
            if pct > 90:
                return 3  # extreme
            elif pct > 75:
                return 2  # elevated
            elif pct < 25:
                return 0  # low
            else:
                return 1  # normal
        
        vix_df["vix_regime"] = vix_df["vix_percentile"].apply(vix_regime_from_percentile)
        
        # Legacy column
        vix_df["vix_percentile_252d"] = vix_df["vix_percentile"]
        
        # Merge VIX features into regime_df
        vix_cols = ["date", "vix_level", "vix_percentile", "vix_change_5d", "vix_regime", "vix_percentile_252d"]
        regime_df = regime_df.merge(vix_df[vix_cols], on="date", how="left")
    else:
        # Use market volatility as proxy
        logger.warning("VIX data not available, using market volatility proxy")
        regime_df["vix_level"] = None
        regime_df["vix_percentile"] = regime_df["market_vol_21d"].rank(pct=True) * 100
        regime_df["vix_change_5d"] = None
        regime_df["vix_regime"] = None
        regime_df["vix_percentile_252d"] = regime_df["vix_percentile"]
    
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
        # Create features table (with Batch 4 regime columns)
        conn.execute("""
            CREATE TABLE features (
                date DATE,
                ticker VARCHAR,
                stable_id VARCHAR,
                -- Momentum (4)
                mom_1m DOUBLE,
                mom_3m DOUBLE,
                mom_6m DOUBLE,
                mom_12m DOUBLE,
                -- Liquidity (2)
                adv_20d DOUBLE,
                adv_60d DOUBLE,
                -- Volatility (3)
                vol_20d DOUBLE,
                vol_60d DOUBLE,
                vol_of_vol DOUBLE,
                -- Drawdown (2)
                max_drawdown_60d DOUBLE,
                dist_from_high_60d DOUBLE,
                -- Relative Strength (2)
                rel_strength_1m DOUBLE,
                rel_strength_3m DOUBLE,
                -- Beta (1)
                beta_252d DOUBLE,
                -- Events/Earnings (12)
                days_to_earnings DOUBLE,
                days_since_earnings DOUBLE,
                in_pead_window BOOLEAN,
                pead_window_day DOUBLE,
                last_surprise_pct DOUBLE,
                avg_surprise_4q DOUBLE,
                surprise_streak DOUBLE,
                surprise_zscore DOUBLE,
                earnings_vol DOUBLE,
                days_since_10k DOUBLE,
                days_since_10q DOUBLE,
                reports_bmo BOOLEAN,
                -- Missingness (2)
                coverage_pct DOUBLE,
                is_new_stock BOOLEAN,
                -- Regime/Macro (12) - Batch 4
                vix_level DOUBLE,
                vix_percentile DOUBLE,
                vix_change_5d DOUBLE,
                vix_regime INTEGER,
                market_return_5d DOUBLE,
                market_return_21d DOUBLE,
                market_return_63d DOUBLE,
                market_vol_21d DOUBLE,
                market_regime INTEGER,
                above_ma_50 INTEGER,
                above_ma_200 INTEGER,
                ma_50_200_cross INTEGER,
                -- Fundamentals (9) - Batch 5 Phase 1: Raw TTM + Z-scores
                sector VARCHAR,
                -- Raw TTM (piecewise-constant between filings)
                gross_margin_ttm DOUBLE,
                operating_margin_ttm DOUBLE,
                revenue_growth_yoy DOUBLE,
                roe_raw DOUBLE,
                -- Z-scores (stepwise per-ticker)
                gross_margin_vs_sector DOUBLE,
                operating_margin_vs_sector DOUBLE,
                revenue_growth_vs_sector DOUBLE,
                roe_zscore DOUBLE,
                PRIMARY KEY (date, ticker)
            )
        """)
        
        # Explicitly specify column order to match DuckDB schema
        features_cols = [
            "date", "ticker", "stable_id",
            "mom_1m", "mom_3m", "mom_6m", "mom_12m",
            "adv_20d", "adv_60d",
            "vol_20d", "vol_60d", "vol_of_vol",
            "max_drawdown_60d", "dist_from_high_60d",
            "rel_strength_1m", "rel_strength_3m",
            "beta_252d",
            "days_to_earnings", "days_since_earnings",
            "in_pead_window", "pead_window_day",
            "last_surprise_pct", "avg_surprise_4q",
            "surprise_streak", "surprise_zscore", "earnings_vol",
            "days_since_10k", "days_since_10q", "reports_bmo",
            "coverage_pct", "is_new_stock",
            # Regime (Batch 4)
            "vix_level", "vix_percentile", "vix_change_5d", "vix_regime",
            "market_return_5d", "market_return_21d", "market_return_63d",
            "market_vol_21d", "market_regime",
            "above_ma_50", "above_ma_200", "ma_50_200_cross",
            # Fundamentals (Batch 5) - Raw TTM + Z-scores
            "sector",
            "gross_margin_ttm", "operating_margin_ttm", "revenue_growth_yoy", "roe_raw",
            "gross_margin_vs_sector", "operating_margin_vs_sector",
            "revenue_growth_vs_sector", "roe_zscore",
        ]
        features_df_ordered = features_df[features_cols]
        conn.execute("INSERT INTO features SELECT * FROM features_df_ordered")
        
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
        
        # Create regime table (expanded for Batch 4)
        conn.execute("""
            CREATE TABLE regime (
                date DATE PRIMARY KEY,
                -- VIX features
                vix_level DOUBLE,
                vix_percentile DOUBLE,
                vix_change_5d DOUBLE,
                vix_regime INTEGER,
                -- Market returns
                market_return_5d DOUBLE,
                market_return_20d DOUBLE,
                market_return_21d DOUBLE,
                market_return_63d DOUBLE,
                -- Market volatility
                market_vol_20d DOUBLE,
                market_vol_21d DOUBLE,
                -- Regime
                market_regime INTEGER,
                -- Technical
                above_ma_50 INTEGER,
                above_ma_200 INTEGER,
                ma_50_200_cross INTEGER,
                -- Legacy
                vix_percentile_252d DOUBLE
            )
        """)
        
        # Explicitly specify column order for regime table
        regime_cols = [
            "date",
            "vix_level", "vix_percentile", "vix_change_5d", "vix_regime",
            "market_return_5d", "market_return_20d", "market_return_21d", "market_return_63d",
            "market_vol_20d", "market_vol_21d",
            "market_regime",
            "above_ma_50", "above_ma_200", "ma_50_200_cross",
            "vix_percentile_252d",
        ]
        regime_df_ordered = regime_df[regime_cols]
        conn.execute("INSERT INTO regime SELECT * FROM regime_df_ordered")
        
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
    
    # Download and compute event features
    print("\nStep 5b: Downloading earnings data...")
    earnings_df, filings_df = download_earnings_data(
        tickers=list(features_df['ticker'].unique()),
        cache_dir=args.cache_dir,
        api_key=api_key,
    )
    n_with_filings = filings_df['ticker'].nunique() if not filings_df.empty else 0
    n_with_earnings = earnings_df['ticker'].nunique() if not earnings_df.empty else 0
    print(f"  Downloaded filings for {n_with_filings} tickers, earnings surprises for {n_with_earnings}")
    
    print("\nStep 5c: Computing event features...")
    features_df = compute_event_features(features_df, earnings_df, filings_df)
    
    # Compute regime features
    print("\nStep 6: Computing regime features...")
    regime_df = compute_regime_features(market_prices, vix_prices)
    regime_df = regime_df[
        (regime_df["date"] >= EVAL_START) &
        (regime_df["date"] <= EVAL_END)
    ]
    
    # Merge regime features into features_df (same value for all tickers on each date)
    print("\nStep 6b: Merging regime features into features...")
    regime_cols_for_features = [
        "date",
        "vix_level", "vix_percentile", "vix_change_5d", "vix_regime",
        "market_return_5d", "market_return_21d", "market_return_63d",
        "market_vol_21d", "market_regime",
        "above_ma_50", "above_ma_200", "ma_50_200_cross",
    ]
    regime_for_merge = regime_df[regime_cols_for_features].copy()
    
    # Convert features_df date to same type as regime_df date for merge
    features_df["date"] = pd.to_datetime(features_df["date"]).dt.date
    regime_for_merge["date"] = pd.to_datetime(regime_for_merge["date"]).dt.date
    
    features_df = features_df.merge(regime_for_merge, on="date", how="left")
    print(f"  Merged regime features: {len(regime_cols_for_features) - 1} columns")
    
    # Compute fundamental features (Batch 5)
    print("\nStep 6c: Downloading fundamental data (Batch 5)...")
    income_df, balance_df, sector_map = download_fundamental_data(
        tickers=list(features_df['ticker'].unique()),
        cache_dir=args.cache_dir,
        api_key=api_key,
    )
    
    print("\nStep 6d: Computing fundamental features...")
    features_df = compute_fundamental_features(features_df, income_df, balance_df, sector_map)
    
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
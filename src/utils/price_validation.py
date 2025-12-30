"""
Price Series Validation
=======================

Utilities for detecting and handling stock split discontinuities
and ensuring consistent split-adjusted price series.

The Problem:
    Some FMP endpoints return prices that aren't consistently split-adjusted.
    For example, NVDA's 10-for-1 split on 2024-06-07 could appear as:
    - Correctly adjusted: $120.89 (post-split equivalent)
    - Incorrectly raw: $1208.90 (pre-split value)
    
    Mixing these creates 10x errors in momentum/return calculations.

Solution:
    1. Detect potential split discontinuities (ratios near 2,3,4,5,10,etc.)
    2. Validate that price series are consistently split-adjusted
    3. Reject or normalize inconsistent series
"""

import logging
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class SplitDiscontinuityError(Exception):
    """Raised when a split discontinuity is detected in price series."""
    pass


# Common split ratios to detect (forward splits)
COMMON_SPLIT_RATIOS = [2, 3, 4, 5, 10, 20]
# Tolerance for ratio matching (e.g., 9.8 to 10.2 matches ratio=10)
RATIO_TOLERANCE = 0.05


@dataclass
class SplitDiscontinuity:
    """Details of a detected split discontinuity."""
    date: str
    price_before: float
    price_after: float
    ratio: float
    likely_split_ratio: Optional[int]
    ticker: str = ""


def detect_split_discontinuities(
    prices_df: pd.DataFrame,
    price_col: str = "close",
    date_col: str = "date",
    ticker_col: Optional[str] = "ticker",
    threshold_ratio: float = 1.5,
) -> List[SplitDiscontinuity]:
    """
    Detect potential split discontinuities in a price series.
    
    A discontinuity is detected when the day-over-day price ratio
    is close to a common split ratio (2, 3, 4, 5, 10, 20).
    
    Args:
        prices_df: DataFrame with price data
        price_col: Column name for price (default: "close")
        date_col: Column name for date (default: "date")
        ticker_col: Column name for ticker (optional, for grouping)
        threshold_ratio: Minimum ratio to consider (default: 1.5)
    
    Returns:
        List of detected SplitDiscontinuity objects
    
    Notes:
        - Properly split-adjusted series should have NO discontinuities
        - A single discontinuity suggests the series has unadjusted pre-split data
    """
    discontinuities = []
    
    df = prices_df.copy()
    
    # Ensure date column is datetime
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        df[date_col] = pd.to_datetime(df[date_col])
    
    # Process per ticker if ticker column exists
    if ticker_col and ticker_col in df.columns:
        groups = df.groupby(ticker_col)
    else:
        groups = [(None, df)]
    
    for ticker, group in groups:
        group = group.sort_values(date_col).reset_index(drop=True)
        
        if len(group) < 2:
            continue
        
        prices = group[price_col].values
        dates = group[date_col].values
        
        for i in range(1, len(prices)):
            if prices[i] == 0 or prices[i-1] == 0:
                continue
            
            # Check for large price jumps (both directions)
            ratio = prices[i-1] / prices[i]  # Price drop (forward split)
            
            if ratio > threshold_ratio:
                # Check if ratio matches a common split ratio
                likely_ratio = None
                for split_ratio in COMMON_SPLIT_RATIOS:
                    if abs(ratio - split_ratio) / split_ratio < RATIO_TOLERANCE:
                        likely_ratio = split_ratio
                        break
                
                if likely_ratio:
                    disc = SplitDiscontinuity(
                        date=str(dates[i])[:10],
                        price_before=float(prices[i-1]),
                        price_after=float(prices[i]),
                        ratio=float(ratio),
                        likely_split_ratio=likely_ratio,
                        ticker=ticker or "",
                    )
                    discontinuities.append(disc)
                    logger.warning(
                        f"Split discontinuity detected: {disc.ticker} on {disc.date} "
                        f"(ratio {disc.ratio:.2f}, likely {disc.likely_split_ratio}:1 split)"
                    )
            
            # Also check reverse (price spike - reverse split or data error)
            reverse_ratio = prices[i] / prices[i-1]
            if reverse_ratio > threshold_ratio:
                likely_ratio = None
                for split_ratio in COMMON_SPLIT_RATIOS:
                    if abs(reverse_ratio - split_ratio) / split_ratio < RATIO_TOLERANCE:
                        likely_ratio = split_ratio
                        break
                
                if likely_ratio:
                    disc = SplitDiscontinuity(
                        date=str(dates[i])[:10],
                        price_before=float(prices[i-1]),
                        price_after=float(prices[i]),
                        ratio=float(reverse_ratio),
                        likely_split_ratio=-likely_ratio,  # Negative for reverse
                        ticker=ticker or "",
                    )
                    discontinuities.append(disc)
                    logger.warning(
                        f"Reverse split/error detected: {disc.ticker} on {disc.date} "
                        f"(ratio {disc.ratio:.2f})"
                    )
    
    return discontinuities


def validate_price_series_consistency(
    prices_df: pd.DataFrame,
    price_col: str = "close",
    adj_close_col: Optional[str] = "adj_close",
    date_col: str = "date",
    ticker_col: Optional[str] = "ticker",
    raise_on_error: bool = True,
) -> Tuple[bool, List[SplitDiscontinuity]]:
    """
    Validate that a price series is consistently split-adjusted.
    
    A properly adjusted series should:
    1. Have no split discontinuities in the 'close' column
    2. If adj_close exists, it should be very close to close (FMP's /full endpoint)
    
    Args:
        prices_df: DataFrame with price data
        price_col: Column name for close price
        adj_close_col: Column name for adjusted close (optional)
        date_col: Column name for date
        ticker_col: Column name for ticker
        raise_on_error: If True, raise SplitDiscontinuityError on detection
    
    Returns:
        Tuple of (is_valid, list_of_discontinuities)
    
    Raises:
        SplitDiscontinuityError if raise_on_error=True and issues found
    """
    issues = []
    
    # Check for split discontinuities in close price
    discontinuities = detect_split_discontinuities(
        prices_df,
        price_col=price_col,
        date_col=date_col,
        ticker_col=ticker_col,
    )
    
    if discontinuities:
        issues.extend(discontinuities)
    
    # If adj_close exists, check consistency with close
    if adj_close_col and adj_close_col in prices_df.columns:
        df = prices_df.copy()
        
        # Check if close and adj_close diverge significantly
        mask = (df[price_col] > 0) & (df[adj_close_col] > 0)
        if mask.any():
            ratio = df.loc[mask, price_col] / df.loc[mask, adj_close_col]
            
            # For FMP /full endpoint, close IS adj_close, so ratio should be ~1
            # Large deviations suggest mixing endpoints
            outliers = (ratio < 0.5) | (ratio > 2.0)
            if outliers.any():
                n_outliers = outliers.sum()
                logger.warning(
                    f"Found {n_outliers} rows where close/adj_close ratio is extreme "
                    f"(suggests mixed data sources)"
                )
    
    is_valid = len(issues) == 0
    
    if not is_valid and raise_on_error:
        error_msg = "Split discontinuities detected in price series:\n"
        for disc in issues[:5]:  # Limit error message
            error_msg += (
                f"  - {disc.ticker} on {disc.date}: "
                f"${disc.price_before:.2f} -> ${disc.price_after:.2f} "
                f"(ratio {disc.ratio:.1f}x, likely {abs(disc.likely_split_ratio) if disc.likely_split_ratio else '?'}:1 split)\n"
            )
        if len(issues) > 5:
            error_msg += f"  ... and {len(issues) - 5} more\n"
        error_msg += "\nThis suggests the price series is not consistently split-adjusted."
        raise SplitDiscontinuityError(error_msg)
    
    return is_valid, issues


def normalize_split_discontinuities(
    prices_df: pd.DataFrame,
    discontinuities: List[SplitDiscontinuity],
    price_cols: List[str] = None,
    date_col: str = "date",
    ticker_col: Optional[str] = "ticker",
) -> pd.DataFrame:
    """
    Attempt to normalize split discontinuities by adjusting pre-split prices.
    
    WARNING: This is a fallback. The preferred approach is to use a data source
    that provides properly split-adjusted prices.
    
    Args:
        prices_df: DataFrame with price data
        discontinuities: List of detected discontinuities
        price_cols: Columns to adjust (default: ["close", "open", "high", "low"])
        date_col: Date column name
        ticker_col: Ticker column name
    
    Returns:
        Adjusted DataFrame
    """
    if price_cols is None:
        price_cols = ["close", "open", "high", "low", "adj_close"]
    
    df = prices_df.copy()
    
    # Ensure date is datetime for comparison
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        df[date_col] = pd.to_datetime(df[date_col])
    
    for disc in discontinuities:
        if disc.likely_split_ratio is None:
            continue
        
        split_date = pd.to_datetime(disc.date)
        ratio = abs(disc.likely_split_ratio)
        
        if disc.likely_split_ratio > 0:
            # Forward split: divide pre-split prices
            mask = df[date_col] < split_date
            if ticker_col and ticker_col in df.columns and disc.ticker:
                mask = mask & (df[ticker_col] == disc.ticker)
            
            for col in price_cols:
                if col in df.columns:
                    df.loc[mask, col] = df.loc[mask, col] / ratio
            
            logger.info(f"Normalized {mask.sum()} pre-split rows for {disc.ticker} {ratio}:1 split on {disc.date}")
    
    return df
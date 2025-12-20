"""
Options & Implied Volatility Client
===================================

Provides PIT-safe access to options-implied risk data:
- EOD options chain snapshots
- Implied volatility (ATM, skew, term structure)
- Implied move around earnings

PIT RULES:
- Options data: observed_at = market close on quote date
- IV surfaces: derived from EOD options, same timing

DATA SOURCES:
- CBOE: Options chains (PAID)
- FMP: Limited options data (check availability)
- OptionMetrics: Academic/institutional (PAID)

NOTE: Options data typically requires paid subscriptions.
This module provides structure for future implementation.
"""

import os
import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import math

import pandas as pd
import pytz

from .event_store import Event, EventType, EventStore
from .trading_calendar import TradingCalendarImpl

logger = logging.getLogger(__name__)

UTC = pytz.UTC
ET = pytz.timezone("America/New_York")


@dataclass
class IVSurfaceSnapshot:
    """
    Implied volatility surface snapshot.
    
    Captures key points on the IV surface:
    - ATM IV for different expirations
    - Skew (25 delta put - call IV)
    - Term structure slope
    
    PIT Rule: Available at market close.
    """
    ticker: str
    quote_date: date
    
    # ATM implied volatility by expiration
    iv_1m: Optional[float] = None   # 1 month
    iv_2m: Optional[float] = None   # 2 months
    iv_3m: Optional[float] = None   # 3 months
    iv_6m: Optional[float] = None   # 6 months
    iv_1y: Optional[float] = None   # 1 year
    
    # Skew (25 delta put IV - 25 delta call IV)
    skew_1m: Optional[float] = None
    skew_3m: Optional[float] = None
    
    # Term structure slope (3m - 1m IV)
    term_slope: Optional[float] = None
    
    observed_at: Optional[datetime] = None
    source: str = "options_vendor"
    
    @property
    def atm_iv(self) -> Optional[float]:
        """Front month ATM IV."""
        return self.iv_1m
    
    @property
    def is_inverted(self) -> bool:
        """Check if term structure is inverted (short > long)."""
        if self.iv_1m and self.iv_3m:
            return self.iv_1m > self.iv_3m
        return False
    
    def to_event(self) -> Event:
        """Convert to Event for storage."""
        return Event(
            ticker=self.ticker,
            event_type=EventType.OPTIONS_SNAPSHOT,
            event_date=self.quote_date,
            observed_at=self.observed_at or UTC.localize(
                datetime.combine(self.quote_date, datetime.min.time().replace(hour=21))
            ),
            source=self.source,
            payload={
                "iv_1m": self.iv_1m,
                "iv_2m": self.iv_2m,
                "iv_3m": self.iv_3m,
                "iv_6m": self.iv_6m,
                "iv_1y": self.iv_1y,
                "skew_1m": self.skew_1m,
                "skew_3m": self.skew_3m,
                "term_slope": self.term_slope,
            },
        )


@dataclass
class ImpliedMoveSnapshot:
    """
    Implied move for an earnings event.
    
    Derived from straddle pricing around earnings.
    """
    ticker: str
    earnings_date: date
    quote_date: date  # When this was calculated
    
    implied_move_pct: float  # Expected move as percentage
    implied_move_1sd: float  # 1 standard deviation move
    
    front_expiry: date       # Nearest expiry used
    atm_straddle_price: float
    atm_strike: float
    stock_price: float
    
    observed_at: Optional[datetime] = None
    source: str = "options_vendor"
    
    @property
    def implied_move_ratio(self) -> float:
        """Implied move / historical avg (if known)."""
        # Would compare to historical earnings moves
        return 1.0  # Placeholder
    
    def to_event(self) -> Event:
        """Convert to Event for storage."""
        return Event(
            ticker=self.ticker,
            event_type=EventType.OPTIONS_SNAPSHOT,
            event_date=self.quote_date,
            observed_at=self.observed_at or UTC.localize(
                datetime.combine(self.quote_date, datetime.min.time().replace(hour=21))
            ),
            source=self.source,
            payload={
                "event_type": "implied_move",
                "earnings_date": self.earnings_date.isoformat(),
                "implied_move_pct": self.implied_move_pct,
                "implied_move_1sd": self.implied_move_1sd,
                "front_expiry": self.front_expiry.isoformat(),
                "atm_straddle_price": self.atm_straddle_price,
                "atm_strike": self.atm_strike,
                "stock_price": self.stock_price,
            },
        )


class OptionsClient:
    """
    Client for options and implied volatility data.
    
    Most endpoints require paid subscriptions - this provides
    structure and calculation utilities for future implementation.
    
    Usage:
        client = OptionsClient()
        
        # IV surface (requires paid data)
        iv = client.get_iv_surface("NVDA", date(2024, 1, 15))
        
        # Implied move for earnings
        move = client.get_implied_move("NVDA", earnings_date)
    """
    
    def __init__(
        self,
        cache_dir: Optional[Path] = None,
    ):
        self.cache_dir = cache_dir or Path("data/cache/options")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.calendar = TradingCalendarImpl()
    
    def get_iv_surface(
        self,
        ticker: str,
        quote_date: date,
    ) -> Optional[IVSurfaceSnapshot]:
        """
        Get IV surface snapshot for a date.
        
        NOTE: Requires paid options data subscription.
        Returns None (stub).
        """
        logger.warning(
            f"get_iv_surface requires paid options data. "
            f"Returning None for {ticker}."
        )
        return None
    
    def get_iv_history(
        self,
        ticker: str,
        start_date: date,
        end_date: date,
    ) -> List[IVSurfaceSnapshot]:
        """
        Get IV surface history.
        
        NOTE: Requires paid options data.
        Returns empty (stub).
        """
        logger.warning(
            f"get_iv_history requires paid options data. "
            f"Returning empty for {ticker}."
        )
        return []
    
    def get_implied_move(
        self,
        ticker: str,
        earnings_date: date,
        quote_date: Optional[date] = None,
    ) -> Optional[ImpliedMoveSnapshot]:
        """
        Calculate implied move for earnings.
        
        NOTE: Requires paid options data.
        Returns None (stub).
        
        Calculation (when data available):
        1. Find nearest expiry after earnings
        2. Get ATM straddle price
        3. Implied move = straddle / stock price
        """
        logger.warning(
            f"get_implied_move requires paid options data. "
            f"Returning None for {ticker}."
        )
        return None
    
    def calculate_implied_move(
        self,
        straddle_price: float,
        stock_price: float,
        days_to_expiry: int,
    ) -> float:
        """
        Calculate implied move from straddle.
        
        Formula: Implied Move = Straddle Price / Stock Price
        
        This is a simplified calculation. For earnings, use
        front-week straddle to capture the event.
        """
        if stock_price <= 0:
            return 0.0
        return (straddle_price / stock_price) * 100
    
    def calculate_iv_from_price(
        self,
        option_price: float,
        stock_price: float,
        strike: float,
        days_to_expiry: int,
        risk_free_rate: float = 0.05,
        is_call: bool = True,
    ) -> Optional[float]:
        """
        Calculate implied volatility from option price.
        
        Uses Newton-Raphson iteration on Black-Scholes.
        Returns annualized IV as decimal (e.g., 0.30 = 30%).
        
        NOTE: Requires scipy for accurate calculation.
        This is a placeholder for the structure.
        """
        try:
            from scipy.stats import norm
            from scipy.optimize import brentq
            
            T = days_to_expiry / 365.0
            if T <= 0:
                return None
            
            def bs_price(sigma):
                d1 = (math.log(stock_price / strike) + (risk_free_rate + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
                d2 = d1 - sigma * math.sqrt(T)
                
                if is_call:
                    return stock_price * norm.cdf(d1) - strike * math.exp(-risk_free_rate * T) * norm.cdf(d2)
                else:
                    return strike * math.exp(-risk_free_rate * T) * norm.cdf(-d2) - stock_price * norm.cdf(-d1)
            
            def objective(sigma):
                return bs_price(sigma) - option_price
            
            # Find IV using Brent's method
            iv = brentq(objective, 0.01, 5.0)
            return iv
            
        except ImportError:
            logger.warning("scipy not available for IV calculation")
            return None
        except Exception as e:
            logger.debug(f"IV calculation failed: {e}")
            return None
    
    def get_high_iv_stocks(
        self,
        tickers: List[str],
        min_iv_percentile: float = 80,
    ) -> List[Tuple[str, float]]:
        """
        Find stocks with high IV percentile.
        
        Useful for identifying potential volatility plays
        or regime detection.
        
        NOTE: Requires paid data and historical IV.
        Returns empty (stub).
        """
        logger.warning(
            f"get_high_iv_stocks requires paid options data. "
            f"Returning empty."
        )
        return []


# =============================================================================
# Feature Calculation Utilities
# =============================================================================

def calculate_iv_features(
    iv_surfaces: List[IVSurfaceSnapshot],
) -> Dict[str, float]:
    """
    Calculate IV-based features from surface history.
    
    Returns:
        Dictionary of feature values
    """
    if not iv_surfaces:
        return {}
    
    features = {}
    
    # Current IV
    current = iv_surfaces[0]
    if current.iv_1m:
        features["iv_1m"] = current.iv_1m
    if current.iv_3m:
        features["iv_3m"] = current.iv_3m
    
    # Skew
    if current.skew_1m:
        features["skew_1m"] = current.skew_1m
    
    # Term structure
    if current.term_slope:
        features["term_slope"] = current.term_slope
    features["is_inverted"] = 1.0 if current.is_inverted else 0.0
    
    # IV change (if history available)
    if len(iv_surfaces) > 5 and iv_surfaces[5].iv_1m and current.iv_1m:
        features["iv_change_5d"] = current.iv_1m - iv_surfaces[5].iv_1m
    
    if len(iv_surfaces) > 20 and iv_surfaces[20].iv_1m and current.iv_1m:
        features["iv_change_20d"] = current.iv_1m - iv_surfaces[20].iv_1m
    
    return features


# =============================================================================
# PIT Rules Documentation
# =============================================================================

"""
PIT RULES FOR OPTIONS DATA
==========================

1. EOD OPTIONS CHAINS
   - Snapshot taken at market close (4pm ET)
   - observed_at = quote_date at 21:00 UTC (4pm ET winter)
   - Includes all listed options with bid/ask/volume

2. IMPLIED VOLATILITY SURFACE
   - Derived from EOD chain
   - Same PIT as underlying chain data
   - observed_at = quote_date at market close

3. IMPLIED MOVE (EARNINGS)
   - Calculated from straddle pricing
   - Changes rapidly near earnings
   - observed_at = when calculation was made (EOD)

4. IV PERCENTILE / RANK
   - Requires historical IV data
   - Lookback window defines percentile
   - observed_at = current date (uses past data only)

FEATURE ENGINEERING NOTES:
- IV level: Absolute IV (ATM front month)
- IV skew: Put-call IV spread, indicates sentiment
- IV term structure: Front vs back, indicates event pricing
- IV change: Momentum in volatility expectations
- IV percentile: Current IV vs historical (52-week)
"""


def test_options_client():
    """Test options client (stubs)."""
    print("Testing OptionsClient...")
    
    client = OptionsClient()
    
    # All should return None/empty (stubs)
    print("\n1. Testing IV surface (stub)...")
    iv = client.get_iv_surface("NVDA", date(2024, 1, 15))
    print(f"  Returned: {iv} (expected None)")
    
    print("\n2. Testing IV history (stub)...")
    history = client.get_iv_history("NVDA", date(2024, 1, 1), date(2024, 1, 31))
    print(f"  Returned: {len(history)} records (expected 0)")
    
    print("\n3. Testing implied move (stub)...")
    move = client.get_implied_move("NVDA", date(2024, 2, 21))
    print(f"  Returned: {move} (expected None)")
    
    print("\n4. Testing IV calculation utility...")
    implied_move = client.calculate_implied_move(
        straddle_price=15.0,
        stock_price=500.0,
        days_to_expiry=7,
    )
    print(f"  Implied move from $15 straddle / $500 stock: {implied_move:.1f}%")
    
    print("\nOptionsClient tests complete (mostly stubs)! ✓")


if __name__ == "__main__":
    test_options_client()


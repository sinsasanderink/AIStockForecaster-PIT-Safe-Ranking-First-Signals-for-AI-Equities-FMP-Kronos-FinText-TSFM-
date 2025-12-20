"""
Positioning & Constraints Data Client
=====================================

Provides PIT-safe access to positioning and constraints data:
- Short interest history
- Borrow cost / fee rates
- ETF flows (QQQ, XLK, SMH)
- 13F institutional holdings

PIT RULES:
- Short interest: observed_at = settlement date + 2 business days
- 13F filings: observed_at = SEC filing date (not period end)
- ETF flows: observed_at = NAV publication time

DATA SOURCES:
- FMP: Short interest, 13F (PAID TIER)
- SEC: 13F filings (free, but complex parsing)
- ETF providers: Flow data (limited free access)

NOTE: Most positioning data requires paid APIs.
This module provides structure and stubs for future implementation.
"""

import os
import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import List, Dict, Any, Optional
from pathlib import Path
from enum import Enum

import pandas as pd
import pytz

from .event_store import Event, EventType, EventStore
from .trading_calendar import TradingCalendarImpl

logger = logging.getLogger(__name__)

UTC = pytz.UTC
ET = pytz.timezone("America/New_York")


@dataclass
class ShortInterestSnapshot:
    """
    Short interest data at a point in time.
    
    PIT Rule: Short interest is reported bi-weekly with ~10 day lag.
    observed_at = settlement_date + publication_lag
    """
    ticker: str
    settlement_date: date
    short_interest: int  # Number of shares short
    avg_volume: int
    days_to_cover: float
    short_percent_of_float: Optional[float] = None
    observed_at: Optional[datetime] = None
    source: str = "fmp"
    
    def to_event(self) -> Event:
        """Convert to Event for storage."""
        return Event(
            ticker=self.ticker,
            event_type=EventType.SHORT_INTEREST,
            event_date=self.settlement_date,
            observed_at=self.observed_at or UTC.localize(
                datetime.combine(self.settlement_date + timedelta(days=10), 
                               datetime.min.time())
            ),
            source=self.source,
            payload={
                "short_interest": self.short_interest,
                "avg_volume": self.avg_volume,
                "days_to_cover": self.days_to_cover,
                "short_percent_of_float": self.short_percent_of_float,
            },
        )


@dataclass
class BorrowCostSnapshot:
    """
    Stock borrow cost / fee rate.
    
    PIT Rule: Borrow rates are typically known same-day.
    """
    ticker: str
    rate_date: date
    borrow_rate: float  # Annualized rate (e.g., 0.05 = 5%)
    availability: str = "easy"  # "easy", "medium", "hard"
    observed_at: Optional[datetime] = None
    source: str = "prime_broker"
    
    def to_event(self) -> Event:
        """Convert to Event for storage."""
        return Event(
            ticker=self.ticker,
            event_type=EventType.BORROW_COST,
            event_date=self.rate_date,
            observed_at=self.observed_at or UTC.localize(
                datetime.combine(self.rate_date, datetime.min.time().replace(hour=16))
            ),
            source=self.source,
            payload={
                "borrow_rate": self.borrow_rate,
                "availability": self.availability,
            },
        )


@dataclass
class ETFFlowSnapshot:
    """
    ETF daily flow data.
    
    PIT Rule: Flows derived from NAV changes, available next day.
    """
    etf_ticker: str
    flow_date: date
    flow_usd: float  # Positive = inflow, negative = outflow
    aum: float  # Assets under management
    shares_outstanding: int
    observed_at: Optional[datetime] = None
    source: str = "etf_provider"
    
    @property
    def flow_percent(self) -> float:
        return (self.flow_usd / self.aum) * 100 if self.aum > 0 else 0
    
    def to_event(self) -> Event:
        """Convert to Event for storage."""
        return Event(
            ticker=self.etf_ticker,
            event_type=EventType.ETF_FLOW,
            event_date=self.flow_date,
            observed_at=self.observed_at or UTC.localize(
                datetime.combine(self.flow_date + timedelta(days=1), 
                               datetime.min.time().replace(hour=9, minute=30))
            ),
            source=self.source,
            payload={
                "flow_usd": self.flow_usd,
                "flow_percent": self.flow_percent,
                "aum": self.aum,
                "shares_outstanding": self.shares_outstanding,
            },
        )


@dataclass
class Institutional13FHolding:
    """
    Institutional holding from 13F filing.
    
    PIT Rule: 13F has 45-day filing deadline after quarter end.
    observed_at = SEC filing date (not period end!)
    """
    ticker: str
    institution_name: str
    institution_cik: str
    period_end: date  # Quarter end
    filing_date: date  # SEC filing date
    shares_held: int
    value_usd: float
    shares_change: Optional[int] = None  # vs prior filing
    observed_at: Optional[datetime] = None
    source: str = "sec_13f"
    
    @property
    def is_new_position(self) -> bool:
        return self.shares_change is not None and self.shares_change == self.shares_held
    
    @property
    def is_exit(self) -> bool:
        return self.shares_held == 0 and self.shares_change is not None
    
    def to_event(self) -> Event:
        """Convert to Event for storage."""
        return Event(
            ticker=self.ticker,
            event_type=EventType.INSTITUTIONAL_13F,
            event_date=self.filing_date,  # Use filing date, not period end!
            observed_at=self.observed_at or UTC.localize(
                datetime.combine(self.filing_date, datetime.min.time().replace(hour=16))
            ),
            source=self.source,
            payload={
                "institution_name": self.institution_name,
                "institution_cik": self.institution_cik,
                "period_end": self.period_end.isoformat(),
                "shares_held": self.shares_held,
                "value_usd": self.value_usd,
                "shares_change": self.shares_change,
            },
        )


class PositioningClient:
    """
    Client for positioning and constraints data.
    
    Most endpoints require paid APIs - this provides structure
    and stubs for future implementation.
    
    Usage:
        client = PositioningClient()
        
        # Short interest (requires FMP paid tier)
        si = client.get_short_interest("NVDA")
        
        # ETF flows (stub)
        flows = client.get_etf_flows("QQQ", lookback_days=30)
    """
    
    # AI-relevant ETFs for flow tracking
    AI_ETFS = ["QQQ", "XLK", "SMH", "SOXX", "IGV", "ARKK"]
    
    def __init__(
        self,
        fmp_key: Optional[str] = None,
        cache_dir: Optional[Path] = None,
    ):
        self.fmp_key = fmp_key or os.getenv("FMP_KEYS", "")
        self.cache_dir = cache_dir or Path("data/cache/positioning")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.calendar = TradingCalendarImpl()
    
    def get_short_interest(
        self,
        ticker: str,
        limit: int = 12,
    ) -> List[ShortInterestSnapshot]:
        """
        Get short interest history.
        
        NOTE: Requires FMP Starter/Pro tier.
        Returns empty on free tier.
        
        PIT Rule: Data has ~10 day publication lag.
        """
        logger.warning(
            f"get_short_interest requires FMP paid tier. "
            f"Returning empty for {ticker}."
        )
        return []
    
    def get_borrow_cost(
        self,
        ticker: str,
    ) -> Optional[BorrowCostSnapshot]:
        """
        Get current borrow cost / fee rate.
        
        NOTE: Requires prime broker data feed.
        Returns None (stub).
        """
        logger.warning(
            f"get_borrow_cost requires prime broker data. "
            f"Returning None for {ticker}."
        )
        return None
    
    def get_etf_flows(
        self,
        etf_ticker: str,
        lookback_days: int = 30,
    ) -> List[ETFFlowSnapshot]:
        """
        Get ETF flow history.
        
        NOTE: Requires paid ETF data feed.
        Returns empty (stub).
        
        For future implementation:
        - ETF.com has some free data
        - Bloomberg/Refinitiv for institutional feeds
        """
        logger.warning(
            f"get_etf_flows requires paid data feed. "
            f"Returning empty for {etf_ticker}."
        )
        return []
    
    def get_13f_holdings(
        self,
        ticker: str,
        top_n: int = 20,
    ) -> List[Institutional13FHolding]:
        """
        Get top 13F institutional holders.
        
        NOTE: Requires FMP paid tier or SEC EDGAR parsing.
        Returns empty (stub).
        
        PIT Rule: Use filing_date (not period_end) for observed_at!
        """
        logger.warning(
            f"get_13f_holdings requires FMP paid tier or SEC parsing. "
            f"Returning empty for {ticker}."
        )
        return []
    
    def get_13f_holdings_changes(
        self,
        ticker: str,
        min_change_pct: float = 0.1,
    ) -> List[Institutional13FHolding]:
        """
        Get significant 13F holding changes.
        
        Filters for positions with >min_change_pct change.
        
        NOTE: Requires FMP paid tier.
        Returns empty (stub).
        """
        logger.warning(
            f"get_13f_holdings_changes requires FMP paid tier. "
            f"Returning empty for {ticker}."
        )
        return []
    
    def get_short_squeeze_candidates(
        self,
        tickers: List[str],
        min_short_interest: float = 0.15,
        min_days_to_cover: float = 5.0,
    ) -> List[str]:
        """
        Identify potential short squeeze candidates.
        
        Criteria:
        - High short interest (>15% of float)
        - High days to cover (>5 days)
        - Hard to borrow
        
        NOTE: Requires paid data.
        Returns empty (stub).
        """
        logger.warning(
            f"get_short_squeeze_candidates requires paid tier. "
            f"Returning empty."
        )
        return []


# =============================================================================
# PIT Rules Documentation
# =============================================================================

"""
PIT RULES FOR POSITIONING DATA
==============================

1. SHORT INTEREST
   - Settlement date: Date shorts were counted
   - Publication lag: ~10 business days
   - observed_at = settlement_date + 10 days
   - Source: FINRA / exchanges, via FMP

2. BORROW COST
   - Updates throughout trading day
   - observed_at = rate_date at 4pm ET
   - Source: Prime brokers

3. ETF FLOWS
   - Derived from NAV and shares outstanding
   - Available next trading day morning
   - observed_at = flow_date + 1 day at 9:30am ET
   - Source: ETF providers, fund websites

4. 13F INSTITUTIONAL HOLDINGS
   - Period end: Quarter end (Mar 31, Jun 30, Sep 30, Dec 31)
   - Filing deadline: 45 days after quarter end
   - observed_at = SEC filing date (NOT period end!)
   - Source: SEC EDGAR 13F filings

IMPORTANT: 13F PIT
   Many systems incorrectly use period_end for 13F data.
   This creates lookahead bias because you're using data
   before it was filed with the SEC.
   
   CORRECT: Use filing_date (acceptedDate from SEC)
   WRONG: Use period_end (quarter end date)
"""


def test_positioning_client():
    """Test positioning client (stubs)."""
    print("Testing PositioningClient...")
    
    client = PositioningClient()
    
    # All should return empty/None (stubs)
    print("\n1. Testing short interest (stub)...")
    si = client.get_short_interest("NVDA")
    print(f"  Returned: {len(si)} records (expected 0)")
    
    print("\n2. Testing borrow cost (stub)...")
    bc = client.get_borrow_cost("NVDA")
    print(f"  Returned: {bc} (expected None)")
    
    print("\n3. Testing ETF flows (stub)...")
    flows = client.get_etf_flows("QQQ")
    print(f"  Returned: {len(flows)} records (expected 0)")
    
    print("\n4. Testing 13F holdings (stub)...")
    holdings = client.get_13f_holdings("NVDA")
    print(f"  Returned: {len(holdings)} records (expected 0)")
    
    print("\nPositioningClient tests complete (all stubs)! ✓")


if __name__ == "__main__":
    test_positioning_client()


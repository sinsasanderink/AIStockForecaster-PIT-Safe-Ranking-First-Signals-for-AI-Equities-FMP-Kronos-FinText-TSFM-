"""
Expectations Data Client
========================

Provides PIT-safe access to forward-looking expectations data:
- Earnings estimates (consensus EPS/Revenue)
- Earnings surprises (actual vs estimate)
- Analyst actions (ratings, price targets)
- Guidance events

Data Sources:
- Alpha Vantage: EARNINGS endpoint (surprise data with reportTime!)
- FMP: Analyst estimates (PAID TIER ONLY)
- SEC: 8-K filings for guidance

PIT RULES:
- Estimates: observed_at = when the estimate was published
- Surprises: observed_at = reportedDate + reportTime (BMO/AMC handling)
- Analyst actions: observed_at = publication timestamp

NOTE: Many advanced endpoints require FMP Starter/Pro tier.
This module implements what's available on free tiers and provides
stubs for paid-tier features.
"""

import os
import time
import logging
from datetime import date, datetime, timedelta
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path

import requests
import pandas as pd
import pytz

from .event_store import Event, EventType, EventTiming, EventStore
from .trading_calendar import TradingCalendarImpl

logger = logging.getLogger(__name__)

UTC = pytz.UTC
ET = pytz.timezone("America/New_York")


# =============================================================================
# New EventTypes for expectations data
# =============================================================================

# These are already defined in event_store.py, but we'll extend them there


@dataclass
class EarningsSurprise:
    """
    Earnings surprise data with PIT-safe timestamps.
    
    From Alpha Vantage EARNINGS endpoint.
    """
    ticker: str
    fiscal_date_ending: date
    reported_date: date
    reported_eps: float
    estimated_eps: float
    surprise: float
    surprise_pct: float
    report_time: str  # "pre-market" or "post-market"
    observed_at: datetime  # PIT timestamp (derived from reported_date + report_time)
    source: str = "alphavantage"
    
    @property
    def is_beat(self) -> bool:
        return self.surprise > 0
    
    @property
    def is_miss(self) -> bool:
        return self.surprise < 0
    
    def to_event(self) -> Event:
        """Convert to Event for storage in EventStore."""
        timing = EventTiming.AMC if self.report_time == "post-market" else EventTiming.BMO
        
        return Event(
            ticker=self.ticker,
            event_type=EventType.EARNINGS,
            event_date=self.reported_date,
            observed_at=self.observed_at,
            source=self.source,
            payload={
                "fiscal_date_ending": self.fiscal_date_ending.isoformat(),
                "reported_eps": self.reported_eps,
                "estimated_eps": self.estimated_eps,
                "surprise": self.surprise,
                "surprise_pct": self.surprise_pct,
                "report_time": self.report_time,
            },
            timing=timing,
        )


@dataclass
class EstimateSnapshot:
    """
    Consensus estimate snapshot (EPS/Revenue).
    
    NOTE: Requires FMP Starter/Pro tier for full history.
    """
    ticker: str
    period_end: date
    estimate_date: date  # When this estimate was recorded
    eps_estimate: Optional[float] = None
    revenue_estimate: Optional[float] = None
    num_analysts: Optional[int] = None
    observed_at: Optional[datetime] = None
    source: str = "fmp"
    
    def to_event(self) -> Event:
        """Convert to Event for storage."""
        return Event(
            ticker=self.ticker,
            event_type=EventType.ESTIMATE_SNAPSHOT,
            event_date=self.estimate_date,
            observed_at=self.observed_at or UTC.localize(
                datetime.combine(self.estimate_date, datetime.min.time())
            ),
            source=self.source,
            payload={
                "period_end": self.period_end.isoformat(),
                "eps_estimate": self.eps_estimate,
                "revenue_estimate": self.revenue_estimate,
                "num_analysts": self.num_analysts,
            },
        )


@dataclass
class AnalystAction:
    """
    Analyst rating change or price target update.
    
    NOTE: Requires FMP Starter/Pro tier.
    """
    ticker: str
    action_date: date
    analyst_firm: str
    action_type: str  # "upgrade", "downgrade", "initiate", "reiterate"
    old_rating: Optional[str] = None
    new_rating: Optional[str] = None
    old_price_target: Optional[float] = None
    new_price_target: Optional[float] = None
    observed_at: Optional[datetime] = None
    source: str = "fmp"
    
    def to_event(self) -> Event:
        """Convert to Event for storage."""
        return Event(
            ticker=self.ticker,
            event_type=EventType.ANALYST_ACTION,
            event_date=self.action_date,
            observed_at=self.observed_at or UTC.localize(
                datetime.combine(self.action_date, datetime.min.time())
            ),
            source=self.source,
            payload={
                "analyst_firm": self.analyst_firm,
                "action_type": self.action_type,
                "old_rating": self.old_rating,
                "new_rating": self.new_rating,
                "old_price_target": self.old_price_target,
                "new_price_target": self.new_price_target,
            },
        )


class ExpectationsClient:
    """
    Client for expectations data (estimates, surprises, analyst actions).
    
    Uses Alpha Vantage for earnings surprises (free tier).
    FMP endpoints require paid tier for full functionality.
    
    Usage:
        client = ExpectationsClient()
        
        # Get earnings surprises (works on free tier!)
        surprises = client.get_earnings_surprises("NVDA")
        
        # Store in EventStore
        for s in surprises:
            event_store.store_event(s.to_event())
    """
    
    def __init__(
        self,
        alphavantage_key: Optional[str] = None,
        fmp_key: Optional[str] = None,
        cache_dir: Optional[Path] = None,
    ):
        self.av_key = alphavantage_key or os.getenv("ALPHAVANTAGE_KEYS", "")
        self.fmp_key = fmp_key or os.getenv("FMP_KEYS", "")
        
        if not self.av_key:
            logger.warning("No Alpha Vantage key - earnings surprises unavailable")
        
        self.cache_dir = cache_dir or Path("data/cache/expectations")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.calendar = TradingCalendarImpl()
        
        # Rate limiting
        self._av_last_request = 0.0
        self._av_min_interval = 12.0  # 5 req/min for free tier
        
        self._session = requests.Session()
    
    def _av_rate_limit(self):
        """Enforce Alpha Vantage rate limit."""
        elapsed = time.time() - self._av_last_request
        if elapsed < self._av_min_interval:
            time.sleep(self._av_min_interval - elapsed)
        self._av_last_request = time.time()
    
    def _get_observed_at_from_report(
        self, 
        reported_date: date, 
        report_time: str
    ) -> datetime:
        """
        Derive PIT-safe observed_at from reported_date and report_time.
        
        Rules:
        - pre-market (BMO): data available at market open (9:30 ET)
        - post-market (AMC): data available at market close + buffer (4:05 ET)
        - unknown: conservative = next market open
        """
        if report_time == "pre-market":
            # Available at market open
            dt = ET.localize(datetime.combine(reported_date, datetime.min.time().replace(hour=9, minute=30)))
        elif report_time == "post-market":
            # Available shortly after close
            dt = ET.localize(datetime.combine(reported_date, datetime.min.time().replace(hour=16, minute=5)))
        else:
            # Unknown - conservative: next market open
            next_open = self.calendar.get_next_trading_day(reported_date)
            dt = ET.localize(datetime.combine(next_open, datetime.min.time().replace(hour=9, minute=30)))
        
        return dt.astimezone(UTC)
    
    def get_earnings_surprises(
        self,
        ticker: str,
        limit: int = 20,
    ) -> List[EarningsSurprise]:
        """
        Get historical earnings surprises from Alpha Vantage.
        
        Returns surprise data with PIT-safe timestamps derived from
        reportedDate + reportTime (BMO/AMC).
        
        Args:
            ticker: Stock symbol
            limit: Max records to return
        
        Returns:
            List of EarningsSurprise objects
        """
        if not self.av_key:
            logger.warning("No Alpha Vantage key configured")
            return []
        
        self._av_rate_limit()
        
        url = f"https://www.alphavantage.co/query?function=EARNINGS&symbol={ticker}&apikey={self.av_key}"
        
        try:
            resp = self._session.get(url, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            
            if "quarterlyEarnings" not in data:
                logger.warning(f"No quarterly earnings data for {ticker}")
                return []
            
            surprises = []
            for item in data["quarterlyEarnings"][:limit]:
                try:
                    fiscal_date = date.fromisoformat(item["fiscalDateEnding"])
                    reported_date = date.fromisoformat(item["reportedDate"])
                    report_time = item.get("reportTime", "post-market")  # Default AMC
                    
                    # Parse numeric fields safely
                    reported_eps = float(item.get("reportedEPS", 0) or 0)
                    estimated_eps = float(item.get("estimatedEPS", 0) or 0)
                    surprise = float(item.get("surprise", 0) or 0)
                    surprise_pct = float(item.get("surprisePercentage", 0) or 0)
                    
                    observed_at = self._get_observed_at_from_report(reported_date, report_time)
                    
                    surprises.append(EarningsSurprise(
                        ticker=ticker,
                        fiscal_date_ending=fiscal_date,
                        reported_date=reported_date,
                        reported_eps=reported_eps,
                        estimated_eps=estimated_eps,
                        surprise=surprise,
                        surprise_pct=surprise_pct,
                        report_time=report_time,
                        observed_at=observed_at,
                    ))
                except (KeyError, ValueError) as e:
                    logger.debug(f"Skipping malformed earnings record: {e}")
                    continue
            
            logger.info(f"Retrieved {len(surprises)} earnings surprises for {ticker}")
            return surprises
            
        except Exception as e:
            logger.error(f"Error fetching earnings surprises for {ticker}: {e}")
            return []
    
    def get_estimate_snapshots(
        self,
        ticker: str,
        periods: int = 4,
    ) -> List[EstimateSnapshot]:
        """
        Get consensus estimate snapshots.
        
        NOTE: Requires FMP Starter/Pro tier for full history.
        This is a STUB that returns empty on free tier.
        
        Args:
            ticker: Stock symbol
            periods: Number of periods
        
        Returns:
            List of EstimateSnapshot objects (empty on free tier)
        """
        logger.warning(
            f"get_estimate_snapshots requires FMP Starter/Pro tier. "
            f"Returning empty for {ticker}."
        )
        return []
    
    def get_analyst_actions(
        self,
        ticker: str,
        days: int = 365,
    ) -> List[AnalystAction]:
        """
        Get analyst rating changes and price target updates.
        
        NOTE: Requires FMP Starter/Pro tier.
        This is a STUB that returns empty on free tier.
        
        Args:
            ticker: Stock symbol
            days: Lookback days
        
        Returns:
            List of AnalystAction objects (empty on free tier)
        """
        logger.warning(
            f"get_analyst_actions requires FMP Starter/Pro tier. "
            f"Returning empty for {ticker}."
        )
        return []
    
    def store_surprises_to_eventstore(
        self,
        tickers: List[str],
        event_store: EventStore,
    ) -> int:
        """
        Fetch and store earnings surprises for multiple tickers.
        
        Args:
            tickers: List of stock symbols
            event_store: EventStore instance
        
        Returns:
            Count of events stored
        """
        total = 0
        for ticker in tickers:
            surprises = self.get_earnings_surprises(ticker)
            for s in surprises:
                event_store.store_event(s.to_event())
                total += 1
        
        logger.info(f"Stored {total} earnings surprise events")
        return total


# =============================================================================
# Testing
# =============================================================================

def test_expectations_client():
    """Test expectations client functionality."""
    from dotenv import load_dotenv
    load_dotenv()
    
    print("Testing ExpectationsClient...")
    
    client = ExpectationsClient()
    
    # Test earnings surprises (should work on free tier)
    print("\n1. Testing get_earnings_surprises...")
    surprises = client.get_earnings_surprises("NVDA", limit=5)
    
    if surprises:
        print(f"  ✓ Got {len(surprises)} earnings surprises")
        for s in surprises[:2]:
            print(f"    {s.reported_date}: EPS={s.reported_eps} vs Est={s.estimated_eps}, "
                  f"Surprise={s.surprise_pct:.1f}%, Time={s.report_time}")
            print(f"      observed_at: {s.observed_at}")
        
        # Test conversion to Event
        event = surprises[0].to_event()
        print(f"  ✓ Converted to Event: {event.event_type.value}")
    else:
        print("  ✗ No surprises returned (check API key)")
    
    # Test stubs
    print("\n2. Testing stub endpoints (should warn about paid tier)...")
    estimates = client.get_estimate_snapshots("NVDA")
    print(f"  Estimates returned: {len(estimates)} (expected 0)")
    
    actions = client.get_analyst_actions("NVDA")
    print(f"  Analyst actions returned: {len(actions)} (expected 0)")
    
    print("\nExpectationsClient tests complete!")


if __name__ == "__main__":
    test_expectations_client()


"""
Event & Calendar Features (Section 5.4)
=======================================

Computes event-based and calendar features for the AI Stock Forecaster.

FEATURES:
- Days to next earnings (forward-looking calendar feature)
- Days since last earnings
- Post-earnings drift window indicator (PEAD)
- Earnings surprise magnitude (last N quarters)
- Filing recency (days since last 10-Q/10-K)

ALL FEATURES ARE:
- PIT-safe: Use only events with observed_at <= asof
- Cross-sectionally standardized where appropriate
- Designed to capture earnings momentum and information flow

CONVENTION:
- Days are calendar days
- Surprises are in percentage terms
- Missing values indicate no event in lookback window

DATA SOURCES:
- EventStore: Historical earnings, filings from Alpha Vantage / SEC
- ExpectationsClient: Earnings surprises with BMO/AMC timing
- TradingCalendar: For trading day calculations
"""

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

import pandas as pd
import numpy as np
import pytz

logger = logging.getLogger(__name__)

# Standard timezone
ET = pytz.timezone("America/New_York")
UTC = pytz.UTC

# PEAD window (Post-Earnings Announcement Drift)
# Research shows drift extends ~60 trading days post-announcement
PEAD_WINDOW_DAYS = 63  # ~3 months

# Earnings lookback for surprise features
EARNINGS_LOOKBACK_QUARTERS = 8  # 2 years

# Filing lookback
FILING_LOOKBACK_DAYS = 180  # ~6 months


@dataclass
class EventFeatures:
    """
    Event and calendar features for a single stock on a single date.
    """
    ticker: str
    date: date
    
    # Days to/since earnings
    days_to_earnings: Optional[int] = None      # Days until next expected earnings
    days_since_earnings: Optional[int] = None   # Days since last earnings
    
    # PEAD (Post-Earnings Announcement Drift) features
    in_pead_window: bool = False                # True if within PEAD window
    pead_window_day: Optional[int] = None       # Which day of PEAD window (1-63)
    
    # Surprise features (from last earnings)
    last_surprise_pct: Optional[float] = None   # Last quarter surprise %
    avg_surprise_4q: Optional[float] = None     # Average surprise last 4Q
    surprise_streak: int = 0                    # Consecutive beats (positive) or misses (negative)
    surprise_zscore: Optional[float] = None     # Cross-sectional z-score of surprise
    
    # Filing features
    days_since_10k: Optional[int] = None        # Days since last annual report
    days_since_10q: Optional[int] = None        # Days since last quarterly report
    days_since_any_filing: Optional[int] = None # Days since any SEC filing
    
    # Earnings volatility
    earnings_vol: Optional[float] = None        # Std dev of surprises (8Q)
    
    # Announcement timing
    reports_bmo: Optional[bool] = None          # True if typically reports before market open
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "ticker": self.ticker,
            "date": self.date.isoformat(),
            "days_to_earnings": self.days_to_earnings,
            "days_since_earnings": self.days_since_earnings,
            "in_pead_window": self.in_pead_window,
            "pead_window_day": self.pead_window_day,
            "last_surprise_pct": self.last_surprise_pct,
            "avg_surprise_4q": self.avg_surprise_4q,
            "surprise_streak": self.surprise_streak,
            "surprise_zscore": self.surprise_zscore,
            "days_since_10k": self.days_since_10k,
            "days_since_10q": self.days_since_10q,
            "days_since_any_filing": self.days_since_any_filing,
            "earnings_vol": self.earnings_vol,
            "reports_bmo": self.reports_bmo,
        }


class EventFeatureGenerator:
    """
    Generates event-based and calendar features.
    
    Uses EventStore for historical events and ExpectationsClient for
    earnings calendar / surprises.
    
    Usage:
        from src.features.event_features import EventFeatureGenerator
        from src.data import get_event_store, get_expectations_client
        
        generator = EventFeatureGenerator(
            event_store=get_event_store(),
            expectations_client=get_expectations_client(),
        )
        
        features = generator.compute_features(
            tickers=["NVDA", "AMD"],
            asof_date=date(2024, 1, 15),
        )
    """
    
    def __init__(
        self,
        event_store=None,
        expectations_client=None,
        trading_calendar=None,
        earnings_calendar: Optional[Dict[str, List[date]]] = None,
    ):
        """
        Initialize event feature generator.
        
        Args:
            event_store: EventStore instance for historical events
            expectations_client: ExpectationsClient for earnings data
            trading_calendar: TradingCalendarImpl for date calculations
            earnings_calendar: Optional pre-loaded earnings calendar
                               {ticker: [date1, date2, ...]} sorted by date
        """
        # Lazy imports to avoid circular dependencies
        if event_store is None:
            from src.data import get_event_store
            event_store = get_event_store()
        if expectations_client is None:
            from src.data import get_expectations_client
            expectations_client = get_expectations_client()
        if trading_calendar is None:
            from src.data import get_trading_calendar
            trading_calendar = get_trading_calendar()
        
        self._events = event_store
        self._expectations = expectations_client
        self._calendar = trading_calendar
        self._earnings_calendar = earnings_calendar or {}
        
        # Cache for earnings data
        self._surprise_cache: Dict[str, List[Dict]] = {}
        
        logger.info("EventFeatureGenerator initialized")
    
    def _get_cutoff_datetime(self, asof_date: date) -> datetime:
        """Get market close cutoff time for a date."""
        return self._calendar.get_market_close(asof_date).astimezone(UTC)
    
    def _load_earnings_surprises(self, ticker: str) -> List[Dict]:
        """
        Load and cache earnings surprises for a ticker.
        
        Returns list of dicts with keys:
            reported_date, fiscal_date, surprise_pct, observed_at, report_time
        """
        if ticker in self._surprise_cache:
            return self._surprise_cache[ticker]
        
        # Try EventStore first (already PIT-safe)
        try:
            from src.data.event_store import EventType
            events = self._events.get_events(
                tickers=[ticker],
                asof=datetime.now(UTC),  # Get all historical
                event_types=[EventType.EARNINGS],
                lookback_days=365 * 3,  # 3 years
            )
            
            if events:
                surprises = []
                for e in events:
                    surprises.append({
                        "reported_date": e.event_date,
                        "fiscal_date": date.fromisoformat(e.payload.get("fiscal_date_ending", e.event_date.isoformat())),
                        "surprise_pct": e.payload.get("surprise_pct", 0),
                        "observed_at": e.observed_at,
                        "report_time": e.timing.value if hasattr(e.timing, 'value') else str(e.timing),
                    })
                self._surprise_cache[ticker] = sorted(surprises, key=lambda x: x["reported_date"], reverse=True)
                return self._surprise_cache[ticker]
        except Exception as e:
            logger.debug(f"EventStore lookup failed for {ticker}: {e}")
        
        # Fallback to ExpectationsClient
        try:
            from src.data.expectations_client import EarningsSurprise
            raw_surprises = self._expectations.get_earnings_surprises(ticker, limit=20)
            
            surprises = []
            for s in raw_surprises:
                surprises.append({
                    "reported_date": s.reported_date,
                    "fiscal_date": s.fiscal_date_ending,
                    "surprise_pct": s.surprise_pct,
                    "observed_at": s.observed_at,
                    "report_time": s.report_time,
                })
            
            self._surprise_cache[ticker] = sorted(surprises, key=lambda x: x["reported_date"], reverse=True)
            return self._surprise_cache[ticker]
        except Exception as e:
            logger.warning(f"Could not load earnings surprises for {ticker}: {e}")
            self._surprise_cache[ticker] = []
            return []
    
    def _get_past_earnings(
        self, 
        ticker: str, 
        asof: datetime,
        limit: int = EARNINGS_LOOKBACK_QUARTERS,
    ) -> List[Dict]:
        """
        Get past earnings events visible as-of the asof datetime.
        
        PIT-safe: Only returns earnings where observed_at <= asof.
        """
        all_surprises = self._load_earnings_surprises(ticker)
        
        # Filter for PIT safety
        visible = [
            s for s in all_surprises
            if s["observed_at"] <= asof
        ]
        
        return visible[:limit]
    
    def _get_next_earnings_date(self, ticker: str, asof_date: date) -> Optional[date]:
        """
        Get next expected earnings date for a ticker.
        
        IMPORTANT: This is a forward-looking feature!
        We use the earnings calendar (scheduled dates) which are typically
        announced weeks in advance and are "known" at asof.
        
        If no calendar data, we estimate based on fiscal quarter pattern.
        """
        # Check if we have a pre-loaded calendar
        if ticker in self._earnings_calendar:
            future_dates = [d for d in self._earnings_calendar[ticker] if d > asof_date]
            if future_dates:
                return min(future_dates)
        
        # Estimate based on past earnings pattern
        past_earnings = self._get_past_earnings(ticker, self._get_cutoff_datetime(asof_date), limit=4)
        
        if not past_earnings:
            return None
        
        # Estimate next date as ~90 days from last earnings
        last_date = past_earnings[0]["reported_date"]
        
        # If we have multiple quarters, use average spacing
        if len(past_earnings) >= 2:
            spacings = []
            for i in range(len(past_earnings) - 1):
                delta = (past_earnings[i]["reported_date"] - past_earnings[i+1]["reported_date"]).days
                if 60 <= delta <= 120:  # Valid quarterly spacing
                    spacings.append(delta)
            
            if spacings:
                avg_spacing = sum(spacings) / len(spacings)
            else:
                avg_spacing = 91  # Default quarterly
        else:
            avg_spacing = 91
        
        estimated_next = last_date + timedelta(days=int(avg_spacing))
        
        # Only return if it's in the future
        if estimated_next > asof_date:
            return estimated_next
        
        # Otherwise, estimate from asof + remainder of quarter
        return asof_date + timedelta(days=int(avg_spacing - (asof_date - last_date).days) % int(avg_spacing))
    
    def compute_features(
        self,
        tickers: List[str],
        asof_date: date,
        pead_window: int = PEAD_WINDOW_DAYS,
    ) -> List[EventFeatures]:
        """
        Compute event features for a list of tickers on a given date.
        
        Args:
            tickers: List of ticker symbols
            asof_date: Date to compute features for
            pead_window: Number of days for post-earnings drift window
        
        Returns:
            List of EventFeatures objects
        """
        asof_datetime = self._get_cutoff_datetime(asof_date)
        results = []
        
        for ticker in tickers:
            try:
                features = self._compute_single(ticker, asof_date, asof_datetime, pead_window)
                results.append(features)
            except Exception as e:
                logger.warning(f"Error computing event features for {ticker}: {e}")
                results.append(EventFeatures(ticker=ticker, date=asof_date))
        
        # Cross-sectional standardization of surprise
        self._standardize_surprises(results)
        
        return results
    
    def _compute_single(
        self,
        ticker: str,
        asof_date: date,
        asof_datetime: datetime,
        pead_window: int,
    ) -> EventFeatures:
        """Compute features for a single ticker."""
        features = EventFeatures(ticker=ticker, date=asof_date)
        
        # Get past earnings (PIT-safe)
        past_earnings = self._get_past_earnings(ticker, asof_datetime, limit=EARNINGS_LOOKBACK_QUARTERS)
        
        # Days since last earnings
        if past_earnings:
            last_earnings_date = past_earnings[0]["reported_date"]
            features.days_since_earnings = (asof_date - last_earnings_date).days
            
            # PEAD window
            if features.days_since_earnings <= pead_window:
                features.in_pead_window = True
                features.pead_window_day = features.days_since_earnings
            
            # Last surprise
            features.last_surprise_pct = past_earnings[0]["surprise_pct"]
            
            # Average surprise (4Q)
            if len(past_earnings) >= 4:
                surprises_4q = [e["surprise_pct"] for e in past_earnings[:4] if e["surprise_pct"] is not None]
                if surprises_4q:
                    features.avg_surprise_4q = sum(surprises_4q) / len(surprises_4q)
            
            # Surprise streak
            streak = 0
            for e in past_earnings:
                if e["surprise_pct"] is None:
                    break
                if e["surprise_pct"] > 0:
                    if streak >= 0:
                        streak += 1
                    else:
                        break
                elif e["surprise_pct"] < 0:
                    if streak <= 0:
                        streak -= 1
                    else:
                        break
                else:
                    break  # Exactly zero breaks streak
            features.surprise_streak = streak
            
            # Earnings volatility (std of surprises)
            all_surprises = [e["surprise_pct"] for e in past_earnings if e["surprise_pct"] is not None]
            if len(all_surprises) >= 3:
                features.earnings_vol = float(np.std(all_surprises))
            
            # Report timing (BMO vs AMC tendency)
            bmo_count = sum(1 for e in past_earnings if e.get("report_time") == "pre-market")
            if len(past_earnings) >= 2:
                features.reports_bmo = bmo_count > len(past_earnings) / 2
        
        # Days to next earnings (forward-looking but using known calendar)
        next_earnings = self._get_next_earnings_date(ticker, asof_date)
        if next_earnings:
            features.days_to_earnings = (next_earnings - asof_date).days
        
        # Filing features (from EventStore)
        self._compute_filing_features(features, ticker, asof_datetime)
        
        return features
    
    def _compute_filing_features(
        self,
        features: EventFeatures,
        ticker: str,
        asof_datetime: datetime,
    ) -> None:
        """Compute SEC filing-related features."""
        try:
            from src.data.event_store import EventType
            
            # Days since 10-K (annual)
            days_10k = self._events.days_since_event(
                ticker, EventType.FILING, asof_datetime
            )
            # Note: This gets any FILING. For specific types, we'd need to check payload
            
            # Get all recent filings to distinguish types
            filings = self._events.get_events(
                tickers=[ticker],
                asof=asof_datetime,
                event_types=[EventType.FILING],
                lookback_days=FILING_LOOKBACK_DAYS,
            )
            
            for f in filings:
                filing_type = f.payload.get("filing_type", "").upper()
                days_since = (asof_datetime.date() - f.event_date).days
                
                if "10-K" in filing_type and features.days_since_10k is None:
                    features.days_since_10k = days_since
                elif "10-Q" in filing_type and features.days_since_10q is None:
                    features.days_since_10q = days_since
                
                # Any filing
                if features.days_since_any_filing is None:
                    features.days_since_any_filing = days_since
        
        except Exception as e:
            logger.debug(f"Could not compute filing features for {ticker}: {e}")
    
    def _standardize_surprises(self, features_list: List[EventFeatures]) -> None:
        """
        Compute cross-sectional z-score for surprise.
        
        This helps identify relative earnings momentum.
        """
        surprises = [f.last_surprise_pct for f in features_list if f.last_surprise_pct is not None]
        
        if len(surprises) < 3:
            return
        
        mean_surprise = np.mean(surprises)
        std_surprise = np.std(surprises)
        
        if std_surprise > 0:
            for f in features_list:
                if f.last_surprise_pct is not None:
                    f.surprise_zscore = (f.last_surprise_pct - mean_surprise) / std_surprise
    
    def to_dataframe(self, features_list: List[EventFeatures]) -> pd.DataFrame:
        """Convert list of EventFeatures to DataFrame."""
        return pd.DataFrame([f.to_dict() for f in features_list])
    
    def compute_batch(
        self,
        tickers: List[str],
        start_date: date,
        end_date: date,
        freq: str = "weekly",
    ) -> pd.DataFrame:
        """
        Compute event features for multiple dates (batch mode).
        
        Args:
            tickers: List of ticker symbols
            start_date: Start date
            end_date: End date
            freq: 'daily' or 'weekly'
        
        Returns:
            DataFrame with all features for all dates
        """
        dates = pd.date_range(start_date, end_date, freq="B" if freq == "daily" else "W-FRI")
        
        all_features = []
        for d in dates:
            d_date = d.date()
            try:
                # Skip non-trading days
                if not self._calendar.is_trading_day(d_date):
                    continue
                
                features = self.compute_features(tickers, d_date)
                all_features.extend(features)
            except Exception as e:
                logger.warning(f"Error computing features for {d_date}: {e}")
        
        return self.to_dataframe(all_features)


# =============================================================================
# Helper Functions
# =============================================================================

def cross_sectional_zscore(series: pd.Series) -> pd.Series:
    """Compute cross-sectional z-score."""
    mean = series.mean()
    std = series.std()
    if std == 0 or pd.isna(std):
        return pd.Series(0, index=series.index)
    return (series - mean) / std


def cross_sectional_rank(series: pd.Series) -> pd.Series:
    """Compute cross-sectional rank (0-1 scale)."""
    return series.rank(pct=True, na_option="keep")


def get_event_feature_generator(
    event_store=None,
    expectations_client=None,
    trading_calendar=None,
) -> EventFeatureGenerator:
    """Factory function for EventFeatureGenerator."""
    return EventFeatureGenerator(
        event_store=event_store,
        expectations_client=expectations_client,
        trading_calendar=trading_calendar,
    )


# =============================================================================
# CLI / Demo
# =============================================================================

if __name__ == "__main__":
    import sys
    from datetime import date
    
    print("=" * 60)
    print("EVENT FEATURE GENERATOR DEMO")
    print("=" * 60)
    
    try:
        generator = EventFeatureGenerator()
        
        # Demo with a few tickers
        tickers = ["NVDA", "AMD", "MSFT"]
        asof = date(2024, 12, 15)
        
        print(f"\nComputing event features for {tickers} as of {asof}...")
        features = generator.compute_features(tickers, asof)
        
        for f in features:
            print(f"\n{f.ticker}:")
            print(f"  Days to earnings: {f.days_to_earnings}")
            print(f"  Days since earnings: {f.days_since_earnings}")
            print(f"  In PEAD window: {f.in_pead_window}")
            print(f"  Last surprise: {f.last_surprise_pct:.2f}%" if f.last_surprise_pct else "  Last surprise: N/A")
            print(f"  Avg surprise (4Q): {f.avg_surprise_4q:.2f}%" if f.avg_surprise_4q else "  Avg surprise (4Q): N/A")
            print(f"  Surprise streak: {f.surprise_streak}")
            print(f"  Reports BMO: {f.reports_bmo}")
        
        print("\n" + "=" * 60)
        print("DEMO COMPLETE")
        print("=" * 60)
        
    except Exception as e:
        print(f"Error running demo: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


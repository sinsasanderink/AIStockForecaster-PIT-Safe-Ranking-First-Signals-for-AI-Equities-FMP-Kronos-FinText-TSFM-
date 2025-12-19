"""
Interfaces (Protocols)
======================

Abstract interfaces that pipelines depend on, allowing for:
- Testability with stubs/mocks
- Swappable implementations (DuckDB, SQLite, in-memory)
- Clear contracts between components

These are Python Protocols (structural subtyping) - implementations
don't need to explicitly inherit, just implement the methods.
"""

from abc import abstractmethod
from datetime import date, datetime, time
from typing import Protocol, List, Dict, Optional, Any, runtime_checkable
from dataclasses import dataclass


# =============================================================================
# Trading Calendar Interface
# =============================================================================

@runtime_checkable
class TradingCalendar(Protocol):
    """
    Interface for trading calendar operations.
    
    Implementations:
    - exchange_calendars (production)
    - InMemoryCalendar (testing)
    
    NOTE: Full implementation deferred to Section 3 (Data Infrastructure)
    """
    
    @abstractmethod
    def is_trading_day(self, dt: date) -> bool:
        """Check if a date is a trading day."""
        ...
    
    @abstractmethod
    def get_trading_days(self, start: date, end: date) -> List[date]:
        """Get all trading days in a date range."""
        ...
    
    @abstractmethod
    def get_next_trading_day(self, dt: date) -> date:
        """Get the next trading day after dt."""
        ...
    
    @abstractmethod
    def get_prev_trading_day(self, dt: date) -> date:
        """Get the previous trading day before dt."""
        ...
    
    @abstractmethod
    def get_cutoff_datetime(self, dt: date) -> datetime:
        """Get the cutoff datetime for a trading day (e.g., 4pm ET)."""
        ...
    
    @abstractmethod
    def get_rebalance_dates(
        self, 
        start: date, 
        end: date, 
        freq: str = "monthly"
    ) -> List[date]:
        """
        Get rebalance dates in a range.
        
        Args:
            freq: 'daily', 'weekly', 'monthly', 'quarterly'
        """
        ...


# =============================================================================
# PIT Store Interface
# =============================================================================

@dataclass
class PITRecord:
    """
    A single point-in-time record.
    
    Stores the value along with when it became known (observed_at)
    and what period it's effective for (effective_from).
    """
    ticker: str
    field: str
    value: Any
    effective_from: date
    observed_at: datetime
    source: str = "fmp"


@runtime_checkable
class PITStore(Protocol):
    """
    Interface for point-in-time safe data storage.
    
    All queries respect the `asof` parameter - only returning data
    that was actually available at that point in time.
    
    Implementations:
    - DuckDBPITStore (production) - Section 3
    - InMemoryPITStore (testing)
    
    NOTE: Full implementation deferred to Section 3 (Data Infrastructure)
    """
    
    @abstractmethod
    def get_ohlcv(
        self,
        tickers: List[str],
        start: date,
        end: date,
        asof: Optional[datetime] = None,
    ) -> Dict[str, Any]:  # Returns DataFrame-like
        """
        Get OHLCV data for tickers, respecting PIT rules.
        
        Args:
            asof: Only return data observed before this datetime.
                  If None, returns all available data.
        """
        ...
    
    @abstractmethod
    def get_fundamentals(
        self,
        tickers: List[str],
        fields: List[str],
        asof: datetime,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Get fundamental data (earnings, ratios, etc.) as of a datetime.
        
        Returns: {ticker: {field: value}}
        """
        ...
    
    @abstractmethod
    def get_market_cap(
        self,
        tickers: List[str],
        asof: date,
    ) -> Dict[str, float]:
        """Get market caps as of a date."""
        ...
    
    @abstractmethod
    def get_avg_volume(
        self,
        tickers: List[str],
        asof: date,
        lookback_days: int = 20,
    ) -> Dict[str, float]:
        """Get average daily volume over lookback period."""
        ...
    
    @abstractmethod
    def get_price(
        self,
        tickers: List[str],
        asof: date,
    ) -> Dict[str, float]:
        """Get closing prices as of a date."""
        ...
    
    @abstractmethod
    def get_sector_industry(
        self,
        tickers: List[str],
    ) -> Dict[str, Dict[str, str]]:
        """Get sector/industry classification."""
        ...
    
    @abstractmethod
    def store_records(self, records: List[PITRecord]) -> int:
        """Store PIT records, returns count stored."""
        ...
    
    @abstractmethod
    def get_last_observed_date(self, ticker: str, field: str) -> Optional[date]:
        """Get the last date we have data for a ticker/field."""
        ...


# =============================================================================
# Universe Store Interface
# =============================================================================

@runtime_checkable
class UniverseStore(Protocol):
    """
    Interface for storing and retrieving historical universe snapshots.
    
    This enables survivorship-safe backtesting by preserving
    what the universe looked like at each point in time.
    
    NOTE: Full implementation deferred to Section 4 (Universe Construction)
    """
    
    @abstractmethod
    def store_universe(
        self,
        asof_date: date,
        tickers: List[str],
        metadata: Dict[str, Dict],  # ticker -> metadata
    ) -> None:
        """Store a universe snapshot."""
        ...
    
    @abstractmethod
    def get_universe(
        self,
        asof_date: date,
    ) -> Optional[Dict]:
        """
        Get universe snapshot for a date.
        
        Returns: {tickers: [...], metadata: {...}} or None if not found
        """
        ...
    
    @abstractmethod
    def get_all_historical_tickers(
        self,
        start: date,
        end: date,
    ) -> List[str]:
        """Get all tickers that appeared in any universe in the date range."""
        ...


# =============================================================================
# Stub Implementations for Testing
# =============================================================================

class StubTradingCalendar:
    """
    Stub trading calendar for testing.
    
    Treats all weekdays as trading days (ignores holidays).
    """
    
    def __init__(self, cutoff_hour: int = 16, timezone: str = "America/New_York"):
        self.cutoff_hour = cutoff_hour
        self.timezone = timezone
    
    def is_trading_day(self, dt: date) -> bool:
        return dt.weekday() < 5  # Mon-Fri
    
    def get_trading_days(self, start: date, end: date) -> List[date]:
        from datetime import timedelta
        days = []
        current = start
        while current <= end:
            if self.is_trading_day(current):
                days.append(current)
            current += timedelta(days=1)
        return days
    
    def get_next_trading_day(self, dt: date) -> date:
        from datetime import timedelta
        next_day = dt + timedelta(days=1)
        while not self.is_trading_day(next_day):
            next_day += timedelta(days=1)
        return next_day
    
    def get_prev_trading_day(self, dt: date) -> date:
        from datetime import timedelta
        prev_day = dt - timedelta(days=1)
        while not self.is_trading_day(prev_day):
            prev_day -= timedelta(days=1)
        return prev_day
    
    def get_cutoff_datetime(self, dt: date) -> datetime:
        import pytz
        tz = pytz.timezone(self.timezone)
        return tz.localize(datetime.combine(dt, time(self.cutoff_hour, 0)))
    
    def get_rebalance_dates(
        self,
        start: date,
        end: date,
        freq: str = "monthly"
    ) -> List[date]:
        trading_days = self.get_trading_days(start, end)
        if not trading_days:
            return []
        
        if freq == "daily":
            return trading_days
        elif freq == "weekly":
            # Last trading day of each week
            result = []
            for i, day in enumerate(trading_days):
                if i == len(trading_days) - 1 or trading_days[i+1].isocalendar()[1] != day.isocalendar()[1]:
                    result.append(day)
            return result
        elif freq == "monthly":
            # Last trading day of each month
            result = []
            for i, day in enumerate(trading_days):
                if i == len(trading_days) - 1 or trading_days[i+1].month != day.month:
                    result.append(day)
            return result
        elif freq == "quarterly":
            # Last trading day of each quarter
            monthly = self.get_rebalance_dates(start, end, "monthly")
            return [d for d in monthly if d.month in (3, 6, 9, 12)]
        else:
            raise ValueError(f"Unknown frequency: {freq}")


class StubPITStore:
    """
    In-memory stub PIT store for testing.
    
    Pre-populate with test data, then query as normal.
    """
    
    def __init__(self):
        self._ohlcv: Dict[str, List[Dict]] = {}  # ticker -> list of records
        self._fundamentals: Dict[str, Dict[str, Any]] = {}
        self._metadata: Dict[str, Dict] = {}  # ticker -> {sector, industry, ...}
    
    def add_test_data(
        self,
        ticker: str,
        price: float = 100.0,
        market_cap: float = 10e9,
        avg_volume: float = 1_000_000,
        sector: str = "Technology",
        industry: str = "Semiconductors",
    ):
        """Add test data for a ticker."""
        self._metadata[ticker] = {
            "price": price,
            "market_cap": market_cap,
            "avg_volume": avg_volume,
            "sector": sector,
            "industry": industry,
        }
    
    def get_ohlcv(self, tickers, start, end, asof=None):
        return {}  # Stub
    
    def get_fundamentals(self, tickers, fields, asof):
        return {}  # Stub
    
    def get_market_cap(self, tickers, asof):
        return {t: self._metadata.get(t, {}).get("market_cap", 0) for t in tickers}
    
    def get_avg_volume(self, tickers, asof, lookback_days=20):
        return {t: self._metadata.get(t, {}).get("avg_volume", 0) for t in tickers}
    
    def get_price(self, tickers, asof):
        return {t: self._metadata.get(t, {}).get("price", 0) for t in tickers}
    
    def get_sector_industry(self, tickers):
        return {
            t: {
                "sector": self._metadata.get(t, {}).get("sector", "Unknown"),
                "industry": self._metadata.get(t, {}).get("industry", "Unknown"),
            }
            for t in tickers
        }
    
    def store_records(self, records):
        return len(records)
    
    def get_last_observed_date(self, ticker, field):
        return None


class StubUniverseStore:
    """In-memory stub universe store for testing."""
    
    def __init__(self):
        self._universes: Dict[date, Dict] = {}
    
    def store_universe(self, asof_date, tickers, metadata):
        self._universes[asof_date] = {"tickers": tickers, "metadata": metadata}
    
    def get_universe(self, asof_date):
        return self._universes.get(asof_date)
    
    def get_all_historical_tickers(self, start, end):
        all_tickers = set()
        for d, data in self._universes.items():
            if start <= d <= end:
                all_tickers.update(data["tickers"])
        return list(all_tickers)


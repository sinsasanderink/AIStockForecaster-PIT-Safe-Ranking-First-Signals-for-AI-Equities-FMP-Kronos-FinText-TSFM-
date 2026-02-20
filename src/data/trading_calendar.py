"""
Trading Calendar Implementation
===============================

Provides accurate trading day calculations using exchange-calendars.

Features:
- NYSE trading days and holidays
- Market open/close times
- Cutoff datetime calculations (for PIT rules)
- Rebalance date generation

This implements the TradingCalendar protocol from src/interfaces.py
"""

from datetime import date, datetime, time, timedelta
from typing import List, Optional
import logging

import pytz

logger = logging.getLogger(__name__)

# Try to import exchange_calendars, fall back to basic implementation
try:
    import exchange_calendars as xcals
    HAS_XCALS = True
except ImportError:
    HAS_XCALS = False
    logger.warning(
        "exchange_calendars not installed. Using basic calendar. "
        "Install with: pip install exchange-calendars"
    )


class TradingCalendarImpl:
    """
    Trading calendar implementation using exchange-calendars.
    
    Implements the TradingCalendar protocol for NYSE.
    
    Usage:
        cal = TradingCalendarImpl()
        
        # Check if trading day
        cal.is_trading_day(date(2024, 1, 15))  # True (Monday)
        cal.is_trading_day(date(2024, 1, 1))   # False (New Year's)
        
        # Get trading days
        days = cal.get_trading_days(date(2024, 1, 1), date(2024, 1, 31))
        
        # Get cutoff time
        cutoff = cal.get_cutoff_datetime(date(2024, 1, 15))  # 4pm ET
    """
    
    def __init__(
        self,
        exchange: str = "XNYS",  # NYSE
        cutoff_hour: int = 16,
        cutoff_minute: int = 0,
    ):
        """
        Initialize trading calendar.
        
        Args:
            exchange: Exchange code (XNYS=NYSE, XNAS=NASDAQ)
            cutoff_hour: Hour for daily cutoff (default 16 = 4pm)
            cutoff_minute: Minute for daily cutoff
        """
        self.exchange = exchange
        self.cutoff_hour = cutoff_hour
        self.cutoff_minute = cutoff_minute
        self.timezone = pytz.timezone("America/New_York")
        
        if HAS_XCALS:
            self._calendar = xcals.get_calendar(exchange)
            logger.info(f"Using exchange-calendars for {exchange}")
        else:
            self._calendar = None
            logger.info("Using basic weekday calendar (no holiday handling)")
    
    def is_trading_day(self, dt: date) -> bool:
        """
        Check if a date is a trading day.
        
        Args:
            dt: Date to check
        
        Returns:
            True if trading day, False otherwise
        """
        if self._calendar is not None:
            # exchange_calendars uses pandas Timestamp
            import pandas as pd
            ts = pd.Timestamp(dt)
            return self._calendar.is_session(ts)
        else:
            # Fallback: weekdays only (ignores holidays)
            return dt.weekday() < 5
    
    def get_trading_days(self, start: date, end: date) -> List[date]:
        """
        Get all trading days in a date range (inclusive).
        
        Args:
            start: Start date
            end: End date
        
        Returns:
            List of trading days
        """
        if self._calendar is not None:
            import pandas as pd
            sessions = self._calendar.sessions_in_range(
                pd.Timestamp(start),
                pd.Timestamp(end),
            )
            return [s.date() for s in sessions]
        else:
            # Fallback: all weekdays
            days = []
            current = start
            while current <= end:
                if current.weekday() < 5:
                    days.append(current)
                current += timedelta(days=1)
            return days
    
    def get_next_trading_day(self, dt: date) -> date:
        """
        Get the next trading day after dt.
        
        Args:
            dt: Reference date
        
        Returns:
            Next trading day
        """
        if self._calendar is not None:
            import pandas as pd
            ts = pd.Timestamp(dt)
            # Get next session after this date
            next_ts = self._calendar.next_session(ts)
            return next_ts.date()
        else:
            next_day = dt + timedelta(days=1)
            while not self.is_trading_day(next_day):
                next_day += timedelta(days=1)
            return next_day
    
    def get_prev_trading_day(self, dt: date) -> date:
        """
        Get the previous trading day before dt.
        
        Args:
            dt: Reference date
        
        Returns:
            Previous trading day
        """
        if self._calendar is not None:
            import pandas as pd
            ts = pd.Timestamp(dt)
            prev_ts = self._calendar.previous_session(ts)
            return prev_ts.date()
        else:
            prev_day = dt - timedelta(days=1)
            while not self.is_trading_day(prev_day):
                prev_day -= timedelta(days=1)
            return prev_day
    
    def get_cutoff_datetime(self, dt: date) -> datetime:
        """
        Get the cutoff datetime for a trading day.
        
        Data for date T is considered available after the cutoff
        on that day (typically 4:00 PM ET for US markets).
        
        Args:
            dt: Trading date
        
        Returns:
            Timezone-aware datetime for the cutoff
        """
        cutoff_time = time(self.cutoff_hour, self.cutoff_minute)
        naive_dt = datetime.combine(dt, cutoff_time)
        return self.timezone.localize(naive_dt)
    
    def get_market_open(self, dt: date) -> datetime:
        """Get market open time for a trading day."""
        if self._calendar is not None:
            import pandas as pd
            ts = pd.Timestamp(dt)
            if self._calendar.is_session(ts):
                return self._calendar.session_open(ts).to_pydatetime()
        
        # Default: 9:30 AM ET
        open_time = time(9, 30)
        naive_dt = datetime.combine(dt, open_time)
        return self.timezone.localize(naive_dt)
    
    def get_market_close(self, dt: date) -> datetime:
        """Get market close time for a trading day."""
        if self._calendar is not None:
            import pandas as pd
            ts = pd.Timestamp(dt)
            if self._calendar.is_session(ts):
                return self._calendar.session_close(ts).to_pydatetime()
        
        # Default: 4:00 PM ET
        close_time = time(16, 0)
        naive_dt = datetime.combine(dt, close_time)
        return self.timezone.localize(naive_dt)
    
    def get_rebalance_dates(
        self,
        start: date,
        end: date,
        freq: str = "monthly",
    ) -> List[date]:
        """
        Get rebalance dates in a range.
        
        Args:
            start: Start date
            end: End date
            freq: Frequency - 'daily', 'weekly', 'monthly', 'quarterly'
        
        Returns:
            List of rebalance dates
        """
        trading_days = self.get_trading_days(start, end)
        
        if not trading_days:
            return []
        
        if freq == "daily":
            return trading_days
        
        elif freq == "weekly":
            # Last trading day of each week
            result = []
            for i, day in enumerate(trading_days):
                if i == len(trading_days) - 1:
                    result.append(day)
                elif trading_days[i + 1].isocalendar()[1] != day.isocalendar()[1]:
                    result.append(day)
            return result
        
        elif freq == "monthly":
            # Last trading day of each month
            result = []
            for i, day in enumerate(trading_days):
                if i == len(trading_days) - 1:
                    result.append(day)
                elif trading_days[i + 1].month != day.month:
                    result.append(day)
            return result
        
        elif freq == "quarterly":
            # Last trading day of each quarter
            monthly = self.get_rebalance_dates(start, end, "monthly")
            return [d for d in monthly if d.month in (3, 6, 9, 12)]
        
        else:
            raise ValueError(f"Unknown frequency: {freq}")
    
    def get_n_trading_days_back(self, dt: date, n: int) -> date:
        """
        Get the date N trading days before dt.
        
        Args:
            dt: Reference date
            n: Number of trading days to go back
        
        Returns:
            Date N trading days before dt
        """
        if self._calendar is not None:
            import pandas as pd
            ts = pd.Timestamp(dt)
            # Get N sessions before
            sessions = self._calendar.sessions_window(ts, -n)
            if len(sessions) > 0:
                return sessions[0].date()
        
        # Fallback
        current = dt
        count = 0
        while count < n:
            current -= timedelta(days=1)
            if self.is_trading_day(current):
                count += 1
        return current
    
    def get_n_trading_days_forward(self, dt: date, n: int) -> date:
        """
        Get the date N trading days after dt.
        
        Args:
            dt: Reference date
            n: Number of trading days to go forward
        
        Returns:
            Date N trading days after dt
        """
        if self._calendar is not None:
            import pandas as pd
            ts = pd.Timestamp(dt)
            sessions = self._calendar.sessions_window(ts, n)
            if len(sessions) > 0:
                return sessions[-1].date()
        
        # Fallback
        current = dt
        count = 0
        while count < n:
            current += timedelta(days=1)
            if self.is_trading_day(current):
                count += 1
        return current
    
    def count_trading_days(self, start: date, end: date) -> int:
        """
        Count trading days between two dates (inclusive).
        
        Args:
            start: Start date
            end: End date
        
        Returns:
            Number of trading days
        """
        return len(self.get_trading_days(start, end))


# Convenience function
def get_trading_calendar() -> TradingCalendarImpl:
    """Get a configured trading calendar instance."""
    return TradingCalendarImpl()


# ============================================================================
# CHAPTER 8: KRONOS-SPECIFIC UTILITIES
# ============================================================================

def load_global_trading_calendar(db_path: str = "data/features.duckdb") -> 'pd.DatetimeIndex':
    """
    Load the complete trading calendar from DuckDB.
    
    **CRITICAL FOR CHAPTER 8 (KRONOS):**
    The calendar is defined as the set of distinct `date` values present in the
    `prices` table. This ensures future timestamps respect actual market trading
    days and do not break near fold boundaries.
    
    **This must be GLOBAL (all dates), NOT fold-filtered.**
    
    Args:
        db_path: Path to DuckDB database.
    
    Returns:
        A sorted, unique, timezone-naive pd.DatetimeIndex of trading dates.
    
    Raises:
        RuntimeError: If no dates are found in the `prices` table.
    
    Example:
        >>> import pandas as pd
        >>> calendar = load_global_trading_calendar()
        >>> print(f"Calendar has {len(calendar)} trading days")
        >>> print(f"Range: {calendar[0]} to {calendar[-1]}")
    """
    import duckdb
    import pandas as pd
    
    con = duckdb.connect(db_path, read_only=True)
    try:
        dates_df = con.execute("SELECT DISTINCT date FROM prices ORDER BY date").df()
    finally:
        con.close()
    
    if dates_df.empty or "date" not in dates_df.columns:
        raise RuntimeError("No trading dates found in DuckDB table `prices`.")
    
    dates = pd.to_datetime(dates_df["date"], errors="coerce").dropna()
    calendar = pd.DatetimeIndex(dates).unique().sort_values()
    
    if len(calendar) == 0:
        raise RuntimeError("Trading calendar resolved to empty after parsing dates.")
    
    logger.info(f"Loaded global trading calendar: {len(calendar)} days ({calendar[0]} to {calendar[-1]})")
    
    return calendar


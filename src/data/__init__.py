"""
Data Infrastructure Module
==========================

Section 3: Data & Point-in-Time Infrastructure

This module provides multi-source data acquisition with PIT-safe timestamps:

1. FMP Client (Financial Modeling Prep)
   - Historical prices (OHLCV) - split-adjusted via /stable/historical-price-eod/full
   - Fundamentals (income, balance, cashflow with filingDate)
   - Company profiles
   
2. Alpha Vantage Client
   - Earnings calendar (dates, but not BMO/AMC timing)
   
3. SEC EDGAR Client (GOLD STANDARD for PIT)
   - Filing timestamps (acceptanceDateTime - exact public availability)
   - XBRL fundamentals
   - 8-K filings for earnings releases

4. DuckDB PIT Store
   - All timestamps in UTC
   - observed_at filtering on all queries
   
5. Trading Calendar
   - NYSE holidays and trading days
   - Cutoff time handling (4pm ET)

6. Event Store (NEW)
   - Discrete events (earnings, filings, news, sentiment)
   - PIT-safe with observed_at filtering
   - Sentiment aggregation utilities
"""

from .fmp_client import FMPClient, FMPError, RateLimitError
from .pit_store import DuckDBPITStore
from .trading_calendar import TradingCalendarImpl
from .event_store import EventStore, Event, EventType, EventTiming

# Lazy imports for optional clients
def get_alphavantage_client():
    """Get Alpha Vantage client (requires ALPHAVANTAGE_KEYS in .env)."""
    from .alphavantage_client import AlphaVantageClient
    return AlphaVantageClient()

def get_sec_client(contact_email=None):
    """Get SEC EDGAR client (no API key required, but needs User-Agent)."""
    from .sec_edgar_client import SECEdgarClient
    return SECEdgarClient(contact_email=contact_email)

__all__ = [
    "FMPClient",
    "FMPError", 
    "RateLimitError",
    "DuckDBPITStore",
    "TradingCalendarImpl",
    "EventStore",
    "Event",
    "EventType",
    "EventTiming",
    "get_alphavantage_client",
    "get_sec_client",
]

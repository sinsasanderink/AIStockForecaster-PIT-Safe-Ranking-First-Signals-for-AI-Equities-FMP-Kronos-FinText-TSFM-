"""
Data Infrastructure Module
==========================

Section 3: Data & Point-in-Time Infrastructure (Extended)

Core Data Sources:
1. FMP Client - Prices (split-adjusted), fundamentals, profiles
2. Alpha Vantage Client - Earnings calendar, earnings surprises
3. SEC EDGAR Client - Filing timestamps (gold standard for PIT)

Storage:
4. DuckDB PIT Store - All queries filter by observed_at <= asof
5. Event Store - Discrete events with PIT-safe timestamps
6. Security Master - Identifier mapping, ticker changes, delistings

Forward-Looking Data (Chapter 3 Extensions):
7. Expectations Client - Earnings surprises, estimates, analyst actions
8. Positioning Client - Short interest, 13F, ETF flows (stubs for paid APIs)
9. Options Client - IV surfaces, implied moves (stubs for paid APIs)

Calendar:
10. Trading Calendar - NYSE holidays, cutoffs, DST handling

NOTE: Many advanced endpoints require FMP Starter/Pro tier.
Free tier provides: prices, fundamentals, profiles, earnings surprises (via AV).
"""

from .fmp_client import FMPClient, FMPError, RateLimitError
from .pit_store import DuckDBPITStore
from .trading_calendar import TradingCalendarImpl
from .event_store import EventStore, Event, EventType, EventTiming
from .security_master import SecurityMaster, SecurityIdentifier, SecurityEventType

# Lazy imports for optional clients (avoids import errors if deps missing)
def get_alphavantage_client():
    """Get Alpha Vantage client (requires ALPHAVANTAGE_KEYS in .env)."""
    from .alphavantage_client import AlphaVantageClient
    return AlphaVantageClient()

def get_sec_client(contact_email=None):
    """Get SEC EDGAR client (no API key required, but needs User-Agent)."""
    from .sec_edgar_client import SECEdgarClient
    return SECEdgarClient(contact_email=contact_email)

def get_expectations_client():
    """Get expectations data client (earnings surprises, estimates)."""
    from .expectations_client import ExpectationsClient
    return ExpectationsClient()

def get_positioning_client():
    """Get positioning data client (short interest, 13F, flows)."""
    from .positioning_client import PositioningClient
    return PositioningClient()

def get_options_client():
    """Get options data client (IV, implied moves)."""
    from .options_client import OptionsClient
    return OptionsClient()

__all__ = [
    # Core clients
    "FMPClient",
    "FMPError", 
    "RateLimitError",
    # Storage
    "DuckDBPITStore",
    "EventStore",
    "Event",
    "EventType",
    "EventTiming",
    "SecurityMaster",
    "SecurityIdentifier",
    "SecurityEventType",
    # Calendar
    "TradingCalendarImpl",
    # Factory functions
    "get_alphavantage_client",
    "get_sec_client",
    "get_expectations_client",
    "get_positioning_client",
    "get_options_client",
]

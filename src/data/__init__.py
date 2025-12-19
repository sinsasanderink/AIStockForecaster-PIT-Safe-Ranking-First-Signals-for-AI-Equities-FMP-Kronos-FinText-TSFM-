"""
Data Infrastructure Module
==========================

Section 3: Data & Point-in-Time Infrastructure

This module provides:
- FMP API client for market data, fundamentals, events
- DuckDB-based Point-in-Time (PIT) safe storage
- Trading calendar with accurate market hours
- Data validation and audit tools
"""

from .fmp_client import FMPClient, FMPError, RateLimitError
from .pit_store import DuckDBPITStore
from .trading_calendar import TradingCalendarImpl

__all__ = [
    "FMPClient",
    "FMPError", 
    "RateLimitError",
    "DuckDBPITStore",
    "TradingCalendarImpl",
]


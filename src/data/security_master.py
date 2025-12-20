"""
Security Master & Identifier Mapping
=====================================

Provides stable identifier mapping and corporate action tracking for
survivorship-safe universe construction.

KEY CONCEPTS:

1. Stable ID: Permanent identifier that doesn't change through:
   - Ticker changes (e.g., FB → META)
   - Name changes
   - Exchange changes

2. Security Master Events:
   - Ticker changes
   - Delistings (with terminal price)
   - Mergers/acquisitions
   - Spin-offs

3. PIT Rules:
   - Events observed_at = SEC filing date or announcement date
   - Terminal prices = last trading day close

DATA SOURCES:
- FMP: Symbol changes (PAID TIER)
- SEC: 8-K filings for corporate actions
- Manual maintenance for critical tickers

NOTE: Full security master requires FMP paid tier.
This module provides the structure and handles what's available.
"""

import os
import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from enum import Enum
import json

import duckdb
import pandas as pd
import pytz

logger = logging.getLogger(__name__)

UTC = pytz.UTC
ET = pytz.timezone("America/New_York")


class SecurityEventType(Enum):
    """Types of security master events."""
    TICKER_CHANGE = "ticker_change"     # Symbol changed
    DELISTING = "delisting"             # Removed from exchange
    MERGER = "merger"                   # Acquired by another company
    SPINOFF = "spinoff"                 # Spun off from parent
    BANKRUPTCY = "bankruptcy"           # Filed for bankruptcy
    NAME_CHANGE = "name_change"         # Company name changed
    EXCHANGE_CHANGE = "exchange_change" # Moved to different exchange


@dataclass
class SecurityIdentifier:
    """
    Security identifier mapping.
    
    Maps ticker to stable_id with validity period.
    """
    ticker: str
    stable_id: str  # Permanent ID (e.g., CIK or CUSIP)
    company_name: str
    valid_from: date
    valid_to: Optional[date] = None  # None = currently valid
    exchange: str = "NYSE"
    is_active: bool = True
    
    @property
    def is_current(self) -> bool:
        return self.valid_to is None and self.is_active


@dataclass  
class SecurityMasterEvent:
    """
    A corporate action or security master event.
    """
    stable_id: str
    event_type: SecurityEventType
    event_date: date
    observed_at: datetime
    old_ticker: Optional[str] = None
    new_ticker: Optional[str] = None
    terminal_price: Optional[float] = None
    details: Dict[str, Any] = field(default_factory=dict)
    source: str = "manual"


class SecurityMaster:
    """
    Security master database for identifier mapping and corporate action tracking.
    
    Enables survivorship-safe universe construction by:
    1. Maintaining stable IDs across ticker changes
    2. Tracking delisted securities
    3. Recording terminal prices for delistings
    
    Usage:
        sm = SecurityMaster()
        
        # Get current ticker for a stable ID
        ticker = sm.get_ticker("0001045810", asof=date(2024, 1, 1))
        
        # Get stable ID for ticker
        stable_id = sm.get_stable_id("NVDA")
        
        # Check if ticker was active on a date
        is_active = sm.was_active("NVDA", date(2020, 1, 1))
    """
    
    def __init__(
        self,
        db_path: Optional[Path] = None,
        read_only: bool = False,
    ):
        """
        Initialize security master.
        
        Args:
            db_path: Path to DuckDB file. None = in-memory.
            read_only: Open in read-only mode.
        """
        if db_path:
            self.db_path = Path(db_path)
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            conn_str = str(self.db_path)
        else:
            self.db_path = None
            conn_str = ":memory:"
        
        self._conn = duckdb.connect(conn_str, read_only=read_only)
        self._init_schema()
        self._load_seed_data()
        
        logger.info(f"SecurityMaster initialized: {conn_str}")
    
    def _init_schema(self):
        """Create security master tables."""
        # Identifier mappings (composite key on ticker + valid_from)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS security_identifiers (
                ticker VARCHAR NOT NULL,
                stable_id VARCHAR NOT NULL,
                company_name VARCHAR,
                valid_from DATE NOT NULL,
                valid_to DATE,
                exchange VARCHAR,
                is_active BOOLEAN DEFAULT TRUE,
                updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (ticker, valid_from)
            )
        """)
        
        # Security events (ticker changes, delistings, etc.)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS security_events (
                event_id VARCHAR PRIMARY KEY,
                stable_id VARCHAR NOT NULL,
                event_type VARCHAR NOT NULL,
                event_date DATE NOT NULL,
                observed_at TIMESTAMPTZ NOT NULL,
                old_ticker VARCHAR,
                new_ticker VARCHAR,
                terminal_price DOUBLE,
                details JSON,
                source VARCHAR
            )
        """)
        
        # Indexes
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_sec_ticker 
            ON security_identifiers (ticker, valid_from, valid_to)
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_sec_stable_id 
            ON security_identifiers (stable_id, valid_from)
        """)
    
    def _load_seed_data(self):
        """Load seed data for AI stock universe."""
        # Check if already seeded
        count = self._conn.execute("SELECT COUNT(*) FROM security_identifiers").fetchone()[0]
        if count > 0:
            return
        
        # Import AI stock universe
        try:
            from src.universe.ai_stocks import AI_TOP100_BY_BUCKET, get_all_tickers
            tickers = get_all_tickers()
        except ImportError:
            try:
                # Fallback for different import contexts
                from ..universe.ai_stocks import AI_TOP100_BY_BUCKET, get_all_tickers
                tickers = get_all_tickers()
            except ImportError:
                logger.warning("Could not import AI stock universe")
                return
        
        # Seed with current tickers (stable_id = ticker for now)
        # In production, would use CIK or CUSIP
        for ticker in tickers:
            self.add_identifier(SecurityIdentifier(
                ticker=ticker,
                stable_id=ticker,  # Placeholder - would use CIK
                company_name=ticker,
                valid_from=date(2020, 1, 1),
                valid_to=None,
                is_active=True,
            ))
        
        # Add known ticker changes in AI universe
        known_changes = [
            # (old_ticker, new_ticker, change_date, stable_id)
            ("FB", "META", date(2022, 6, 9), "META"),
        ]
        
        for old, new, change_date, stable_id in known_changes:
            self.record_ticker_change(
                stable_id=stable_id,
                old_ticker=old,
                new_ticker=new,
                change_date=change_date,
                observed_at=UTC.localize(datetime.combine(change_date, datetime.min.time())),
            )
        
        logger.info(f"Seeded security master with {len(tickers)} tickers")
    
    def add_identifier(self, identifier: SecurityIdentifier) -> None:
        """Add or update a security identifier mapping."""
        self._conn.execute("""
            INSERT OR REPLACE INTO security_identifiers 
            (ticker, stable_id, company_name, valid_from, valid_to, exchange, is_active)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, [
            identifier.ticker,
            identifier.stable_id,
            identifier.company_name,
            identifier.valid_from,
            identifier.valid_to,
            identifier.exchange,
            identifier.is_active,
        ])
    
    def get_stable_id(self, ticker: str, asof: Optional[date] = None) -> Optional[str]:
        """
        Get stable ID for a ticker.
        
        Args:
            ticker: Stock symbol
            asof: Date to check (default: today)
        
        Returns:
            Stable ID or None if not found
        """
        asof = asof or date.today()
        
        result = self._conn.execute("""
            SELECT stable_id FROM security_identifiers
            WHERE ticker = ?
              AND valid_from <= ?
              AND (valid_to IS NULL OR valid_to >= ?)
            ORDER BY valid_from DESC
            LIMIT 1
        """, [ticker, asof, asof]).fetchone()
        
        return result[0] if result else None
    
    def get_ticker(self, stable_id: str, asof: Optional[date] = None) -> Optional[str]:
        """
        Get current ticker for a stable ID.
        
        Args:
            stable_id: Permanent identifier
            asof: Date to check (default: today)
        
        Returns:
            Ticker symbol or None if not found
        """
        asof = asof or date.today()
        
        result = self._conn.execute("""
            SELECT ticker FROM security_identifiers
            WHERE stable_id = ?
              AND valid_from <= ?
              AND (valid_to IS NULL OR valid_to >= ?)
            ORDER BY valid_from DESC
            LIMIT 1
        """, [stable_id, asof, asof]).fetchone()
        
        return result[0] if result else None
    
    def was_active(self, ticker: str, asof: date) -> bool:
        """Check if ticker was actively trading on a date."""
        result = self._conn.execute("""
            SELECT is_active FROM security_identifiers
            WHERE ticker = ?
              AND valid_from <= ?
              AND (valid_to IS NULL OR valid_to >= ?)
            LIMIT 1
        """, [ticker, asof, asof]).fetchone()
        
        return result[0] if result else False
    
    def get_all_active_tickers(self, asof: Optional[date] = None) -> List[str]:
        """Get all active tickers as of a date."""
        asof = asof or date.today()
        
        result = self._conn.execute("""
            SELECT DISTINCT ticker FROM security_identifiers
            WHERE valid_from <= ?
              AND (valid_to IS NULL OR valid_to >= ?)
              AND is_active = TRUE
        """, [asof, asof]).fetchall()
        
        return [r[0] for r in result]
    
    def get_ticker_history(self, stable_id: str) -> List[Tuple[str, date, Optional[date]]]:
        """
        Get ticker history for a stable ID.
        
        Returns:
            List of (ticker, valid_from, valid_to) tuples
        """
        result = self._conn.execute("""
            SELECT ticker, valid_from, valid_to FROM security_identifiers
            WHERE stable_id = ?
            ORDER BY valid_from
        """, [stable_id]).fetchall()
        
        return [(r[0], r[1], r[2]) for r in result]
    
    def record_ticker_change(
        self,
        stable_id: str,
        old_ticker: str,
        new_ticker: str,
        change_date: date,
        observed_at: datetime,
        source: str = "manual",
    ) -> None:
        """
        Record a ticker change event.
        
        Updates identifier mappings and logs the event.
        """
        # End validity of old ticker
        self._conn.execute("""
            UPDATE security_identifiers
            SET valid_to = ?, is_active = FALSE
            WHERE stable_id = ? AND ticker = ? AND valid_to IS NULL
        """, [change_date - timedelta(days=1), stable_id, old_ticker])
        
        # Add new ticker
        self.add_identifier(SecurityIdentifier(
            ticker=new_ticker,
            stable_id=stable_id,
            company_name=new_ticker,
            valid_from=change_date,
            valid_to=None,
            is_active=True,
        ))
        
        # Log event
        event_id = f"{stable_id}_ticker_change_{change_date}"
        self._conn.execute("""
            INSERT OR REPLACE INTO security_events
            (event_id, stable_id, event_type, event_date, observed_at, 
             old_ticker, new_ticker, source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            event_id,
            stable_id,
            SecurityEventType.TICKER_CHANGE.value,
            change_date,
            observed_at.isoformat(),
            old_ticker,
            new_ticker,
            source,
        ])
        
        logger.info(f"Recorded ticker change: {old_ticker} → {new_ticker} on {change_date}")
    
    def record_delisting(
        self,
        ticker: str,
        stable_id: str,
        delisting_date: date,
        terminal_price: float,
        observed_at: datetime,
        reason: str = "unknown",
        source: str = "manual",
    ) -> None:
        """
        Record a delisting event with terminal price.
        
        Critical for survivorship-safe backtesting - ensures
        delistings are properly accounted for.
        """
        # Mark identifier as inactive
        self._conn.execute("""
            UPDATE security_identifiers
            SET valid_to = ?, is_active = FALSE
            WHERE stable_id = ? AND valid_to IS NULL
        """, [delisting_date, stable_id])
        
        # Log event with terminal price
        event_id = f"{stable_id}_delisting_{delisting_date}"
        self._conn.execute("""
            INSERT OR REPLACE INTO security_events
            (event_id, stable_id, event_type, event_date, observed_at,
             old_ticker, terminal_price, details, source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            event_id,
            stable_id,
            SecurityEventType.DELISTING.value,
            delisting_date,
            observed_at.isoformat(),
            ticker,
            terminal_price,
            json.dumps({"reason": reason}),
            source,
        ])
        
        logger.info(f"Recorded delisting: {ticker} on {delisting_date}, terminal=${terminal_price}")
    
    def get_terminal_price(self, stable_id: str) -> Optional[float]:
        """Get terminal price for a delisted security."""
        result = self._conn.execute("""
            SELECT terminal_price FROM security_events
            WHERE stable_id = ? AND event_type = 'delisting'
            ORDER BY event_date DESC
            LIMIT 1
        """, [stable_id]).fetchone()
        
        return result[0] if result else None
    
    def get_delistings(
        self,
        start_date: date,
        end_date: date,
    ) -> List[Dict[str, Any]]:
        """Get all delistings in a date range."""
        result = self._conn.execute("""
            SELECT * FROM security_events
            WHERE event_type = 'delisting'
              AND event_date >= ?
              AND event_date <= ?
            ORDER BY event_date
        """, [start_date, end_date]).fetchdf()
        
        return result.to_dict('records')
    
    def close(self):
        """Close database connection."""
        self._conn.close()


# =============================================================================
# Testing
# =============================================================================

def test_security_master():
    """Test security master functionality."""
    print("Testing SecurityMaster...")
    
    sm = SecurityMaster()  # In-memory
    
    # Test stable ID lookup
    print("\n1. Testing stable ID lookup...")
    stable_id = sm.get_stable_id("NVDA")
    print(f"  NVDA stable_id: {stable_id}")
    
    # Test ticker lookup
    ticker = sm.get_ticker("NVDA")
    print(f"  Ticker for NVDA: {ticker}")
    
    # Test active tickers
    print("\n2. Testing active tickers...")
    active = sm.get_all_active_tickers()
    print(f"  Active tickers: {len(active)}")
    
    # Test ticker change
    print("\n3. Testing ticker change...")
    history = sm.get_ticker_history("META")
    print(f"  META ticker history: {history}")
    
    # Test delisting
    print("\n4. Testing delisting...")
    sm.record_delisting(
        ticker="TEST_DELISTED",
        stable_id="TEST_DELISTED",
        delisting_date=date(2023, 6, 15),
        terminal_price=0.50,
        observed_at=UTC.localize(datetime(2023, 6, 15, 16, 0)),
        reason="bankruptcy",
    )
    
    terminal = sm.get_terminal_price("TEST_DELISTED")
    print(f"  TEST_DELISTED terminal price: ${terminal}")
    
    was_active = sm.was_active("TEST_DELISTED", date(2023, 1, 1))
    print(f"  Was active on 2023-01-01: {was_active}")
    
    sm.close()
    print("\nSecurityMaster tests complete! ✓")


if __name__ == "__main__":
    test_security_master()


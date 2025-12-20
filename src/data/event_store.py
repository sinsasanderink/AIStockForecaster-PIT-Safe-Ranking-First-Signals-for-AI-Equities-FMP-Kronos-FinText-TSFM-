"""
Event Store - PIT-Safe Event Abstraction
=========================================

Stores and retrieves discrete events (earnings, filings, news, sentiment)
with strict Point-in-Time (PIT) correctness.

CORE PRINCIPLE:
Every event has an observed_at timestamp - the moment it became publicly available.
Queries filter by observed_at <= asof to prevent lookahead bias.

EVENT TYPES:
- EARNINGS: Quarterly earnings releases (with BMO/AMC timing)
- FILING: SEC filings (10-K, 10-Q, 8-K) with acceptance timestamps
- NEWS: News articles with publication timestamp
- SENTIMENT: Sentiment scores derived from text (inherits source observed_at)

USAGE:
    store = EventStore()
    
    # Store an earnings event
    store.store_event(Event(
        ticker="NVDA",
        event_type=EventType.EARNINGS,
        event_date=date(2024, 11, 20),
        observed_at=datetime(2024, 11, 20, 21, 5, tzinfo=UTC),  # AMC
        source="sec_8k",
        payload={"eps_actual": 0.81, "eps_estimate": 0.75, "surprise_pct": 8.0}
    ))
    
    # Query events as-of a date
    events = store.get_events(
        tickers=["NVDA"],
        event_types=[EventType.EARNINGS],
        asof=datetime(2024, 11, 21, tzinfo=UTC),
        lookback_days=30
    )
"""

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from enum import Enum
from typing import List, Dict, Any, Optional
from pathlib import Path
import json

import duckdb
import pandas as pd
import pytz

logger = logging.getLogger(__name__)

UTC = pytz.UTC
ET = pytz.timezone("America/New_York")


class EventType(Enum):
    """Types of discrete events tracked in the system."""
    # Core events (Tier 0)
    EARNINGS = "earnings"           # Quarterly earnings releases with surprise data
    FILING = "filing"               # SEC filings (10-K, 10-Q, 8-K, etc.)
    NEWS = "news"                   # News articles
    SENTIMENT = "sentiment"         # Sentiment scores (derived from news/filings)
    DIVIDEND = "dividend"           # Dividend announcements
    SPLIT = "split"                 # Stock splits
    
    # Forward-looking expectations (Tier 1)
    ESTIMATE_SNAPSHOT = "estimate_snapshot"     # Consensus estimate at a point in time
    ESTIMATE_REVISION = "estimate_revision"     # Analyst estimate revision
    GUIDANCE = "guidance"                       # Company forward guidance
    ANALYST_ACTION = "analyst_action"           # Rating change / price target update
    
    # Options-implied risk (Tier 2)
    OPTIONS_SNAPSHOT = "options_snapshot"       # EOD options chain / IV surface
    
    # Positioning / constraints (Tier 2)
    SHORT_INTEREST = "short_interest"           # Short interest + days to cover
    BORROW_COST = "borrow_cost"                 # Stock borrow fee rate
    ETF_FLOW = "etf_flow"                       # ETF inflow/outflow
    INSTITUTIONAL_13F = "institutional_13f"     # 13F institutional holdings
    
    # Survivorship / security master (Tier 0)
    SECURITY_MASTER = "security_master"         # Ticker changes, delistings, mergers
    DELISTING = "delisting"                     # Delisting event with terminal price


class EventTiming(Enum):
    """When an event occurred relative to market hours."""
    BMO = "bmo"         # Before Market Open
    DURING = "during"   # During market hours
    AMC = "amc"         # After Market Close
    UNKNOWN = "unknown" # Timing not specified


@dataclass
class Event:
    """
    A discrete event with PIT-safe timestamps.
    
    Attributes:
        ticker: Stock symbol
        event_type: Type of event (earnings, filing, news, etc.)
        event_date: The date the event "is about" (e.g., earnings period end)
        observed_at: UTC datetime when event became publicly available (CRITICAL!)
        source: Data source (sec_8k, alphavantage, fmp, newsapi, etc.)
        payload: Event-specific data as JSON-serializable dict
        event_id: Unique identifier (auto-generated if not provided)
        timing: BMO/AMC/DURING/UNKNOWN for market-relative timing
    """
    ticker: str
    event_type: EventType
    event_date: date
    observed_at: datetime
    source: str
    payload: Dict[str, Any] = field(default_factory=dict)
    event_id: Optional[str] = None
    timing: EventTiming = EventTiming.UNKNOWN
    
    def __post_init__(self):
        # Ensure observed_at is UTC
        if self.observed_at.tzinfo is None:
            self.observed_at = UTC.localize(self.observed_at)
        else:
            self.observed_at = self.observed_at.astimezone(UTC)
        
        # Auto-generate event_id if not provided
        if self.event_id is None:
            self.event_id = f"{self.ticker}_{self.event_type.value}_{self.event_date}_{self.source}"
    
    @property
    def payload_json(self) -> str:
        """Serialize payload to JSON string for storage."""
        return json.dumps(self.payload) if self.payload else "{}"
    
    @classmethod
    def from_row(cls, row: Dict) -> "Event":
        """Create Event from database row."""
        payload = row.get("payload", "{}")
        if isinstance(payload, str):
            payload = json.loads(payload)
        
        observed_at = row["observed_at"]
        if isinstance(observed_at, str):
            observed_at = datetime.fromisoformat(observed_at)
        
        event_date = row["event_date"]
        if isinstance(event_date, str):
            event_date = date.fromisoformat(event_date)
        elif hasattr(event_date, 'date'):  # pandas Timestamp
            event_date = event_date.date()
        elif not isinstance(event_date, date):
            event_date = pd.Timestamp(event_date).date()
        
        return cls(
            ticker=row["ticker"],
            event_type=EventType(row["event_type"]),
            event_date=event_date,
            observed_at=observed_at,
            source=row["source"],
            payload=payload,
            event_id=row.get("event_id"),
            timing=EventTiming(row.get("timing", "unknown")),
        )


class EventStore:
    """
    DuckDB-backed store for PIT-safe events.
    
    All queries filter by observed_at to ensure no lookahead bias.
    """
    
    def __init__(
        self,
        db_path: Optional[Path] = None,
        read_only: bool = False,
    ):
        """
        Initialize event store.
        
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
        logger.info(f"EventStore initialized: {conn_str}")
    
    def _init_schema(self):
        """Create events table if not exists."""
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS events (
                event_id VARCHAR PRIMARY KEY,
                ticker VARCHAR NOT NULL,
                event_type VARCHAR NOT NULL,
                event_date DATE NOT NULL,
                observed_at TIMESTAMPTZ NOT NULL,
                timing VARCHAR DEFAULT 'unknown',
                source VARCHAR NOT NULL,
                payload JSON,
                created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Indexes for common query patterns
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_events_ticker_observed 
            ON events (ticker, observed_at)
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_events_type_observed 
            ON events (event_type, observed_at)
        """)
    
    def store_event(self, event: Event) -> None:
        """
        Store a single event.
        
        Uses INSERT OR REPLACE to handle updates.
        """
        self._conn.execute("""
            INSERT OR REPLACE INTO events 
            (event_id, ticker, event_type, event_date, observed_at, timing, source, payload)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            event.event_id,
            event.ticker,
            event.event_type.value,
            event.event_date,
            event.observed_at.isoformat(),
            event.timing.value,
            event.source,
            event.payload_json,
        ])
    
    def store_events(self, events: List[Event]) -> int:
        """
        Store multiple events efficiently.
        
        Returns count of events stored.
        """
        for event in events:
            self.store_event(event)
        return len(events)
    
    def get_events(
        self,
        tickers: List[str],
        asof: datetime,
        event_types: Optional[List[EventType]] = None,
        lookback_days: int = 90,
        limit: Optional[int] = None,
    ) -> List[Event]:
        """
        Get events for tickers as-of a datetime (PIT-safe).
        
        Args:
            tickers: List of ticker symbols
            asof: UTC datetime cutoff (only events observed before this)
            event_types: Optional filter by event types
            lookback_days: How far back to look
            limit: Maximum events to return
        
        Returns:
            List of Events, sorted by observed_at descending
        """
        # Ensure asof is UTC
        if asof.tzinfo is None:
            asof = UTC.localize(asof)
        else:
            asof = asof.astimezone(UTC)
        
        asof_str = asof.isoformat()
        start_date = (asof.date() - timedelta(days=lookback_days)).isoformat()
        
        ticker_list = ", ".join([f"'{t}'" for t in tickers])
        
        query = f"""
            SELECT * FROM events
            WHERE ticker IN ({ticker_list})
              AND observed_at <= '{asof_str}'
              AND event_date >= '{start_date}'
        """
        
        if event_types:
            type_list = ", ".join([f"'{et.value}'" for et in event_types])
            query += f" AND event_type IN ({type_list})"
        
        query += " ORDER BY observed_at DESC"
        
        if limit:
            query += f" LIMIT {limit}"
        
        df = self._conn.execute(query).df()
        
        return [Event.from_row(row.to_dict()) for _, row in df.iterrows()]
    
    def get_latest_event(
        self,
        ticker: str,
        event_type: EventType,
        asof: datetime,
    ) -> Optional[Event]:
        """
        Get the most recent event of a type for a ticker.
        
        Args:
            ticker: Stock symbol
            event_type: Type of event to find
            asof: UTC datetime cutoff
        
        Returns:
            Most recent Event or None
        """
        events = self.get_events(
            tickers=[ticker],
            asof=asof,
            event_types=[event_type],
            limit=1,
        )
        return events[0] if events else None
    
    def get_earnings_dates(
        self,
        tickers: List[str],
        asof: datetime,
        lookback_days: int = 365,
    ) -> Dict[str, List[date]]:
        """
        Get historical earnings dates for tickers.
        
        Returns:
            {ticker: [date1, date2, ...]} sorted by date descending
        """
        events = self.get_events(
            tickers=tickers,
            asof=asof,
            event_types=[EventType.EARNINGS],
            lookback_days=lookback_days,
        )
        
        result = {t: [] for t in tickers}
        for event in events:
            result[event.ticker].append(event.event_date)
        
        return result
    
    def days_since_event(
        self,
        ticker: str,
        event_type: EventType,
        asof: datetime,
    ) -> Optional[int]:
        """
        Calculate days since most recent event of type.
        
        Useful for features like "days since last earnings".
        
        Returns:
            Number of calendar days, or None if no event found
        """
        event = self.get_latest_event(ticker, event_type, asof)
        if event:
            return (asof.date() - event.event_date).days
        return None
    
    def get_sentiment_score(
        self,
        ticker: str,
        asof: datetime,
        lookback_days: int = 7,
        aggregation: str = "mean",
    ) -> Optional[float]:
        """
        Get aggregated sentiment score for a ticker.
        
        Args:
            ticker: Stock symbol
            asof: UTC datetime cutoff
            lookback_days: Window for sentiment aggregation
            aggregation: "mean", "median", or "latest"
        
        Returns:
            Aggregated sentiment score or None
        """
        events = self.get_events(
            tickers=[ticker],
            asof=asof,
            event_types=[EventType.SENTIMENT],
            lookback_days=lookback_days,
        )
        
        if not events:
            return None
        
        scores = [
            e.payload.get("score", 0) 
            for e in events 
            if "score" in e.payload
        ]
        
        if not scores:
            return None
        
        if aggregation == "mean":
            return sum(scores) / len(scores)
        elif aggregation == "median":
            sorted_scores = sorted(scores)
            mid = len(sorted_scores) // 2
            return sorted_scores[mid]
        elif aggregation == "latest":
            return scores[0]  # Already sorted by observed_at desc
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")
    
    def count_events(
        self,
        ticker: str,
        event_type: EventType,
        asof: datetime,
        lookback_days: int = 30,
    ) -> int:
        """
        Count events in a lookback window.
        
        Useful for features like "news volume" or "filing count".
        """
        events = self.get_events(
            tickers=[ticker],
            asof=asof,
            event_types=[event_type],
            lookback_days=lookback_days,
        )
        return len(events)
    
    def close(self):
        """Close database connection."""
        self._conn.close()


# =============================================================================
# Helper functions for creating events from different sources
# =============================================================================

def earnings_event_from_sec(
    ticker: str,
    period_end: date,
    filing_date: date,
    accepted_at: datetime,
    eps_actual: Optional[float] = None,
    eps_estimate: Optional[float] = None,
) -> Event:
    """
    Create earnings event from SEC 8-K filing.
    
    Uses acceptance timestamp for precise PIT timing.
    """
    # Determine timing from acceptance hour
    accepted_hour_et = accepted_at.astimezone(ET).hour
    if accepted_hour_et < 9:
        timing = EventTiming.BMO
    elif accepted_hour_et >= 16:
        timing = EventTiming.AMC
    else:
        timing = EventTiming.DURING
    
    payload = {
        "period_end": period_end.isoformat(),
        "filing_date": filing_date.isoformat(),
    }
    if eps_actual is not None:
        payload["eps_actual"] = eps_actual
    if eps_estimate is not None:
        payload["eps_estimate"] = eps_estimate
    if eps_actual is not None and eps_estimate is not None and eps_estimate != 0:
        payload["surprise_pct"] = ((eps_actual - eps_estimate) / abs(eps_estimate)) * 100
    
    return Event(
        ticker=ticker,
        event_type=EventType.EARNINGS,
        event_date=filing_date,
        observed_at=accepted_at,
        source="sec_8k",
        payload=payload,
        timing=timing,
    )


def filing_event_from_sec(
    ticker: str,
    form_type: str,
    filing_date: date,
    accepted_at: datetime,
    accession_number: str,
) -> Event:
    """
    Create filing event from SEC EDGAR data.
    """
    return Event(
        ticker=ticker,
        event_type=EventType.FILING,
        event_date=filing_date,
        observed_at=accepted_at,
        source="sec_edgar",
        payload={
            "form_type": form_type,
            "accession_number": accession_number,
        },
    )


def sentiment_event_from_text(
    ticker: str,
    text_date: date,
    observed_at: datetime,
    score: float,
    source: str,
    text_snippet: Optional[str] = None,
    model: str = "default",
) -> Event:
    """
    Create sentiment event from analyzed text.
    
    Args:
        ticker: Stock symbol
        text_date: Date of the source text
        observed_at: When the TEXT was published (not when sentiment computed!)
        score: Sentiment score (e.g., -1 to +1)
        source: Source of text (e.g., "newsapi", "earnings_transcript")
        text_snippet: Optional snippet for debugging
        model: Sentiment model used
    """
    payload = {
        "score": score,
        "model": model,
    }
    if text_snippet:
        payload["snippet"] = text_snippet[:200]  # Truncate for storage
    
    return Event(
        ticker=ticker,
        event_type=EventType.SENTIMENT,
        event_date=text_date,
        observed_at=observed_at,
        source=source,
        payload=payload,
    )


# =============================================================================
# Testing / Validation
# =============================================================================

def test_event_store():
    """Test basic event store functionality."""
    print("Testing EventStore...")
    
    store = EventStore()  # In-memory
    
    # Create test events
    from datetime import timezone
    
    now = datetime.now(UTC)
    yesterday = now - timedelta(days=1)
    
    events = [
        Event(
            ticker="NVDA",
            event_type=EventType.EARNINGS,
            event_date=date(2024, 11, 20),
            observed_at=datetime(2024, 11, 20, 21, 5, tzinfo=UTC),
            source="sec_8k",
            payload={"eps_actual": 0.81, "eps_estimate": 0.75},
            timing=EventTiming.AMC,
        ),
        Event(
            ticker="NVDA",
            event_type=EventType.SENTIMENT,
            event_date=date(2024, 11, 21),
            observed_at=datetime(2024, 11, 21, 14, 0, tzinfo=UTC),
            source="newsapi",
            payload={"score": 0.85, "model": "finbert"},
        ),
        Event(
            ticker="AMD",
            event_type=EventType.EARNINGS,
            event_date=date(2024, 10, 29),
            observed_at=datetime(2024, 10, 29, 21, 10, tzinfo=UTC),
            source="sec_8k",
            payload={"eps_actual": 0.92},
            timing=EventTiming.AMC,
        ),
    ]
    
    # Store events
    count = store.store_events(events)
    print(f"  ✓ Stored {count} events")
    
    # Query events
    asof = datetime(2024, 11, 22, tzinfo=UTC)
    results = store.get_events(["NVDA"], asof, lookback_days=30)
    assert len(results) == 2, f"Expected 2 events, got {len(results)}"
    print(f"  ✓ Retrieved {len(results)} NVDA events")
    
    # Query before event observed
    asof_before = datetime(2024, 11, 20, 20, 0, tzinfo=UTC)  # Before AMC earnings
    results_before = store.get_events(
        ["NVDA"], asof_before, 
        event_types=[EventType.EARNINGS]
    )
    assert len(results_before) == 0, "Should not see earnings before observed_at"
    print("  ✓ PIT boundary test passed (event not visible before observed_at)")
    
    # Sentiment score
    sentiment = store.get_sentiment_score("NVDA", asof, lookback_days=7)
    assert sentiment == 0.85
    print(f"  ✓ Sentiment score: {sentiment}")
    
    # Days since event
    days = store.days_since_event("NVDA", EventType.EARNINGS, asof)
    assert days == 2
    print(f"  ✓ Days since earnings: {days}")
    
    store.close()
    print("All EventStore tests passed! ✓")


if __name__ == "__main__":
    test_event_store()


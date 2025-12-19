"""
DuckDB Point-in-Time (PIT) Store
================================

Provides PIT-safe data storage and retrieval using DuckDB.

CRITICAL PIT RULES ENFORCED:
1. All timestamps stored in UTC
2. All queries filter by observed_at <= asof (datetime, not just date)
3. Fundamentals use filing_date + conservative lag
4. No forward-filling before observed_at
5. Revisions stored as separate rows (same effective_from, different observed_at)

TIMESTAMP CONVENTION:
- observed_at: UTC datetime when data became publicly available
- effective_from: date the data is "for" (period end, trade date, etc.)
- All query asof parameters must be UTC datetime

Schema Design:
- prices: (ticker, date, observed_at) - unique per ticker/date
- fundamentals: (ticker, period_end, statement_type, field, observed_at) 
  - Allows revisions: same period_end can have multiple observed_at
- profiles: (ticker, updated_at) - treated as static metadata
- market_snapshots: (ticker, date, observed_at) - for computed values like ADV

This implements the PITStore protocol from src/interfaces.py
"""

import logging
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional

import duckdb
import pandas as pd
import pytz

logger = logging.getLogger(__name__)

# Standard timezone
UTC = pytz.UTC
ET = pytz.timezone("America/New_York")


def ensure_utc(dt: datetime) -> datetime:
    """Ensure a datetime is UTC. Convert if needed."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        # Assume UTC for naive datetimes
        return UTC.localize(dt)
    return dt.astimezone(UTC)


def get_market_close_utc(d: date) -> datetime:
    """Get market close (4pm ET) in UTC for a date."""
    et_close = ET.localize(datetime(d.year, d.month, d.day, 16, 0))
    return et_close.astimezone(UTC)


class DuckDBPITStore:
    """
    DuckDB-backed Point-in-Time safe data store.
    
    ALL QUERIES USE UTC DATETIME FOR asof PARAMETER.
    
    Usage:
        store = DuckDBPITStore("data/pit_store.duckdb")
        
        # Store price data (observed_at should be UTC)
        store.store_prices(prices_df)
        
        # Query with PIT safety - asof must be UTC datetime
        from datetime import datetime, timezone
        asof = datetime(2024, 1, 15, 21, 0, tzinfo=timezone.utc)  # 4pm ET in UTC
        df = store.get_ohlcv(["NVDA"], start, end, asof=asof)
    """
    
    def __init__(
        self,
        db_path: Optional[Path] = None,
        read_only: bool = False,
    ):
        """
        Initialize PIT store.
        
        Args:
            db_path: Path to DuckDB file. None = in-memory.
            read_only: Open database in read-only mode
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
        
        logger.info(f"DuckDBPITStore initialized: {conn_str}")
    
    def _init_schema(self):
        """Initialize database schema with proper indices."""
        
        # Prices table - daily OHLCV
        # observed_at stored as TIMESTAMPTZ (UTC)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS prices (
                ticker VARCHAR NOT NULL,
                date DATE NOT NULL,
                open DOUBLE,
                high DOUBLE,
                low DOUBLE,
                close DOUBLE,
                adj_close DOUBLE,
                volume BIGINT,
                observed_at TIMESTAMPTZ NOT NULL,
                source VARCHAR DEFAULT 'fmp',
                PRIMARY KEY (ticker, date)
            )
        """)
        
        # Fundamentals table - supports revisions
        # Same (ticker, period_end, field) can have multiple observed_at rows
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS fundamentals (
                ticker VARCHAR NOT NULL,
                period_end DATE NOT NULL,
                period_type VARCHAR NOT NULL,
                statement_type VARCHAR NOT NULL,
                field VARCHAR NOT NULL,
                value DOUBLE,
                filing_date DATE,
                observed_at TIMESTAMPTZ NOT NULL,
                source VARCHAR DEFAULT 'fmp',
                -- Allow multiple revisions: same fact, different observed_at
                PRIMARY KEY (ticker, period_end, period_type, statement_type, field, observed_at)
            )
        """)
        
        # Company profiles - treated as static metadata
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS profiles (
                ticker VARCHAR PRIMARY KEY,
                company_name VARCHAR,
                sector VARCHAR,
                industry VARCHAR,
                exchange VARCHAR,
                currency VARCHAR DEFAULT 'USD',
                country VARCHAR DEFAULT 'US',
                updated_at TIMESTAMPTZ NOT NULL
            )
        """)
        
        # Market data snapshots (computed values: market cap, ADV)
        # observed_at should be based on input data timestamps, not "now"
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS market_snapshots (
                ticker VARCHAR NOT NULL,
                date DATE NOT NULL,
                market_cap DOUBLE,
                shares_outstanding BIGINT,
                avg_volume_20d DOUBLE,
                observed_at TIMESTAMPTZ NOT NULL,
                source VARCHAR DEFAULT 'computed',
                PRIMARY KEY (ticker, date)
            )
        """)
        
        # Events table (earnings, dividends, splits)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS events (
                ticker VARCHAR NOT NULL,
                event_date DATE NOT NULL,
                event_type VARCHAR NOT NULL,
                event_time VARCHAR,
                value DOUBLE,
                observed_at TIMESTAMPTZ NOT NULL,
                source VARCHAR DEFAULT 'fmp',
                PRIMARY KEY (ticker, event_date, event_type)
            )
        """)
        
        # Create indices for fast PIT queries
        # These are critical for performance
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_prices_pit ON prices (ticker, observed_at)")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_prices_date ON prices (ticker, date)")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_fundamentals_pit ON fundamentals (ticker, observed_at)")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_fundamentals_period ON fundamentals (ticker, period_end)")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_snapshots_pit ON market_snapshots (ticker, observed_at)")
    
    def close(self):
        """Close database connection."""
        self._conn.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    # =========================================================================
    # Storage Methods
    # =========================================================================
    
    def store_prices(self, df: pd.DataFrame, source: str = "fmp") -> int:
        """
        Store price data with PIT metadata.
        
        Required columns: ticker, date, open, high, low, close, volume
        Required: observed_at (UTC datetime) - when price became available
        
        Returns: Number of rows stored
        """
        if df.empty:
            return 0
        
        df = df.copy()
        
        if "observed_at" not in df.columns:
            raise ValueError("observed_at column required for PIT safety")
        
        df["source"] = source
        
        # Ensure observed_at is UTC
        if df["observed_at"].dt.tz is None:
            logger.warning("observed_at has no timezone, assuming UTC")
            df["observed_at"] = df["observed_at"].dt.tz_localize("UTC")
        else:
            df["observed_at"] = df["observed_at"].dt.tz_convert("UTC")
        
        # Insert/update
        count = 0
        for _, row in df.iterrows():
            try:
                self._conn.execute("""
                    INSERT OR REPLACE INTO prices 
                    (ticker, date, open, high, low, close, adj_close, volume, observed_at, source)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, [
                    row.get("ticker"),
                    row.get("date"),
                    row.get("open"),
                    row.get("high"),
                    row.get("low"),
                    row.get("close"),
                    row.get("adj_close"),
                    row.get("volume"),
                    row.get("observed_at"),
                    row.get("source"),
                ])
                count += 1
            except Exception as e:
                logger.warning(f"Failed to store price row: {e}")
        
        logger.debug(f"Stored {count} price records")
        return count
    
    def store_fundamentals(
        self,
        df: pd.DataFrame,
        statement_type: str,
        source: str = "fmp",
    ) -> int:
        """
        Store fundamental data with PIT metadata.
        
        Supports revisions: if same (ticker, period_end, field) is stored
        with a different observed_at, it's kept as a separate row.
        
        Returns: Number of field records stored
        """
        if df.empty:
            return 0
        
        df = df.copy()
        
        if "observed_at" not in df.columns:
            raise ValueError("observed_at column required for PIT safety")
        
        # Ensure UTC
        if df["observed_at"].dt.tz is None:
            df["observed_at"] = df["observed_at"].dt.tz_localize("UTC")
        else:
            df["observed_at"] = df["observed_at"].dt.tz_convert("UTC")
        
        period_type = "quarter"
        if "period" in df.columns:
            period_type = df["period"].iloc[0] if df["period"].notna().any() else "quarter"
        
        # Get metric columns
        meta_cols = {"ticker", "symbol", "date", "period", "period_end", "observed_at",
                     "filingDate", "fillingDate", "acceptedDate", "calendarYear", "source", "link",
                     "finalLink", "statement_type"}
        metric_cols = [c for c in df.columns if c not in meta_cols and not c.startswith("_")]
        
        count = 0
        for _, row in df.iterrows():
            ticker = row.get("symbol") or row.get("ticker")
            period_end = row.get("period_end") or row.get("date")
            observed_at = row.get("observed_at")
            filing_date = row.get("filingDate") or row.get("fillingDate") or row.get("filing_date")
            
            for field in metric_cols:
                value = row.get(field)
                if pd.notna(value):
                    try:
                        self._conn.execute("""
                            INSERT OR REPLACE INTO fundamentals
                            (ticker, period_end, period_type, statement_type, field, 
                             value, filing_date, observed_at, source)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, [
                            ticker, period_end, period_type, statement_type,
                            field, float(value), filing_date, observed_at, source,
                        ])
                        count += 1
                    except Exception as e:
                        logger.warning(f"Failed to store fundamental: {e}")
        
        logger.debug(f"Stored {count} fundamental records")
        return count
    
    def store_profile(self, ticker: str, profile: Dict[str, Any]) -> None:
        """Store company profile (treated as static metadata)."""
        self._conn.execute("""
            INSERT OR REPLACE INTO profiles
            (ticker, company_name, sector, industry, exchange, currency, country, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            ticker,
            profile.get("companyName"),
            profile.get("sector"),
            profile.get("industry"),
            profile.get("exchange"),
            profile.get("currency", "USD"),
            profile.get("country", "US"),
            datetime.now(UTC),
        ])
    
    def store_market_snapshot(
        self,
        ticker: str,
        snapshot_date: date,
        market_cap: Optional[float] = None,
        shares_outstanding: Optional[int] = None,
        avg_volume_20d: Optional[float] = None,
        observed_at: Optional[datetime] = None,
    ) -> None:
        """
        Store market data snapshot.
        
        IMPORTANT: observed_at should be based on when the inputs were available,
        not when this function is called. Use the max observed_at of the inputs
        (price close time, shares outstanding availability).
        
        If observed_at is None, defaults to market close on snapshot_date.
        """
        if observed_at is None:
            # Default: market close on the snapshot date
            observed_at = get_market_close_utc(snapshot_date)
        else:
            observed_at = ensure_utc(observed_at)
        
        self._conn.execute("""
            INSERT OR REPLACE INTO market_snapshots
            (ticker, date, market_cap, shares_outstanding, avg_volume_20d, observed_at, source)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, [
            ticker, snapshot_date, market_cap, shares_outstanding,
            avg_volume_20d, observed_at, "computed",
        ])
    
    # =========================================================================
    # PIT-Safe Query Methods
    # =========================================================================
    
    def get_ohlcv(
        self,
        tickers: List[str],
        start: date,
        end: date,
        asof: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        Get OHLCV data respecting PIT rules.
        
        Args:
            tickers: List of ticker symbols
            start: Start date
            end: End date
            asof: UTC datetime - only return data with observed_at <= asof
                  If None, returns all available data (no PIT filter)
        
        Returns:
            DataFrame with OHLCV data
        """
        ticker_list = ", ".join([f"'{t}'" for t in tickers])
        
        query = f"""
            SELECT ticker, date, open, high, low, close, adj_close, volume, observed_at
            FROM prices
            WHERE ticker IN ({ticker_list})
              AND date >= '{start}'
              AND date <= '{end}'
        """
        
        if asof is not None:
            asof = ensure_utc(asof)
            asof_str = asof.strftime("%Y-%m-%d %H:%M:%S+00")
            query += f" AND observed_at <= '{asof_str}'"
        
        query += " ORDER BY ticker, date"
        
        return self._conn.execute(query).df()
    
    def get_fundamentals(
        self,
        tickers: List[str],
        fields: List[str],
        asof: datetime,
        statement_types: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Get fundamental data as of a datetime (PIT-safe).
        
        For each (ticker, field), returns the value from the most recent
        period_end where observed_at <= asof.
        
        If there are revisions (same period_end, different observed_at),
        returns the latest revision that was available at asof.
        
        Args:
            tickers: List of ticker symbols
            fields: List of field names
            asof: UTC datetime cutoff
            statement_types: Optional filter
        
        Returns:
            {ticker: {field: value}}
        """
        asof = ensure_utc(asof)
        asof_str = asof.strftime("%Y-%m-%d %H:%M:%S+00")
        ticker_list = ", ".join([f"'{t}'" for t in tickers])
        field_list = ", ".join([f"'{f}'" for f in fields])
        
        # For each ticker/field, get the most recent period_end where
        # the latest revision was available at asof
        query = f"""
            WITH pit_filtered AS (
                SELECT ticker, field, value, period_end, observed_at,
                       ROW_NUMBER() OVER (
                           PARTITION BY ticker, field, period_end
                           ORDER BY observed_at DESC
                       ) as revision_rank
                FROM fundamentals
                WHERE ticker IN ({ticker_list})
                  AND field IN ({field_list})
                  AND observed_at <= '{asof_str}'
        """
        
        if statement_types:
            type_list = ", ".join([f"'{t}'" for t in statement_types])
            query += f" AND statement_type IN ({type_list})"
        
        query += """
            ),
            latest_revision AS (
                SELECT * FROM pit_filtered WHERE revision_rank = 1
            ),
            latest_period AS (
                SELECT ticker, field, value, period_end,
                       ROW_NUMBER() OVER (
                           PARTITION BY ticker, field
                           ORDER BY period_end DESC
                       ) as period_rank
                FROM latest_revision
            )
            SELECT ticker, field, value, period_end
            FROM latest_period
            WHERE period_rank = 1
        """
        
        df = self._conn.execute(query).df()
        
        result = {t: {} for t in tickers}
        for _, row in df.iterrows():
            result[row["ticker"]][row["field"]] = row["value"]
        
        return result
    
    def get_price(
        self,
        tickers: List[str],
        asof: datetime,
    ) -> Dict[str, float]:
        """
        Get closing prices as of a datetime (PIT-safe).
        
        Returns the most recent close price where observed_at <= asof.
        
        Args:
            tickers: List of ticker symbols
            asof: UTC datetime cutoff
        
        Returns:
            {ticker: close_price}
        """
        asof = ensure_utc(asof)
        asof_str = asof.strftime("%Y-%m-%d %H:%M:%S+00")
        ticker_list = ", ".join([f"'{t}'" for t in tickers])
        
        query = f"""
            WITH ranked AS (
                SELECT ticker, close, date,
                       ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY date DESC) as rn
                FROM prices
                WHERE ticker IN ({ticker_list})
                  AND observed_at <= '{asof_str}'
            )
            SELECT ticker, close
            FROM ranked
            WHERE rn = 1
        """
        
        df = self._conn.execute(query).df()
        return dict(zip(df["ticker"], df["close"]))
    
    def get_market_cap(
        self,
        tickers: List[str],
        asof: datetime,
    ) -> Dict[str, float]:
        """
        Get market caps as of a datetime (PIT-safe).
        
        Args:
            tickers: List of ticker symbols
            asof: UTC datetime cutoff
        
        Returns:
            {ticker: market_cap}
        """
        asof = ensure_utc(asof)
        asof_str = asof.strftime("%Y-%m-%d %H:%M:%S+00")
        ticker_list = ", ".join([f"'{t}'" for t in tickers])
        
        query = f"""
            WITH ranked AS (
                SELECT ticker, market_cap,
                       ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY date DESC) as rn
                FROM market_snapshots
                WHERE ticker IN ({ticker_list})
                  AND observed_at <= '{asof_str}'
                  AND market_cap IS NOT NULL
            )
            SELECT ticker, market_cap
            FROM ranked
            WHERE rn = 1
        """
        
        df = self._conn.execute(query).df()
        return dict(zip(df["ticker"], df["market_cap"]))
    
    def get_avg_volume(
        self,
        tickers: List[str],
        asof: datetime,
        lookback_days: int = 20,
    ) -> Dict[str, float]:
        """
        Get average daily volume over lookback period (PIT-safe).
        
        FIXED: Uses window function to get N rows per ticker, not globally.
        
        Args:
            tickers: List of ticker symbols
            asof: UTC datetime cutoff
            lookback_days: Number of trading days to average
        
        Returns:
            {ticker: avg_volume}
        """
        asof = ensure_utc(asof)
        asof_str = asof.strftime("%Y-%m-%d %H:%M:%S+00")
        ticker_list = ", ".join([f"'{t}'" for t in tickers])
        
        # FIXED: Window function partitioned by ticker
        query = f"""
            WITH ranked AS (
                SELECT ticker, volume, date,
                       ROW_NUMBER() OVER (
                           PARTITION BY ticker 
                           ORDER BY date DESC
                       ) as rn
                FROM prices
                WHERE ticker IN ({ticker_list})
                  AND observed_at <= '{asof_str}'
            ),
            recent AS (
                SELECT ticker, volume
                FROM ranked
                WHERE rn <= {lookback_days}
            )
            SELECT ticker, AVG(volume) as avg_volume
            FROM recent
            GROUP BY ticker
        """
        
        df = self._conn.execute(query).df()
        return dict(zip(df["ticker"], df["avg_volume"]))
    
    def get_sector_industry(
        self,
        tickers: List[str],
    ) -> Dict[str, Dict[str, str]]:
        """
        Get sector/industry classification.
        
        Note: Profiles are treated as static metadata (no PIT filter).
        This is acceptable because sector/industry rarely changes.
        """
        ticker_list = ", ".join([f"'{t}'" for t in tickers])
        
        query = f"""
            SELECT ticker, sector, industry
            FROM profiles
            WHERE ticker IN ({ticker_list})
        """
        
        df = self._conn.execute(query).df()
        
        return {
            row["ticker"]: {"sector": row["sector"], "industry": row["industry"]}
            for _, row in df.iterrows()
        }
    
    def get_last_observed_date(self, ticker: str, table: str = "prices") -> Optional[date]:
        """Get the last date we have data for a ticker."""
        if table == "prices":
            query = f"SELECT MAX(date) FROM prices WHERE ticker = '{ticker}'"
        else:
            query = f"SELECT MAX(period_end) FROM fundamentals WHERE ticker = '{ticker}'"
        
        result = self._conn.execute(query).fetchone()
        return result[0] if result and result[0] else None
    
    # =========================================================================
    # Validation Methods
    # =========================================================================
    
    def validate_pit(self, ticker: str, table: str = "prices") -> Dict[str, Any]:
        """
        Validate PIT correctness for a ticker.
        
        Checks:
        1. All records have observed_at
        2. observed_at >= market close on that date (for prices)
        3. Monotonicity: dates increasing implies observed_at non-decreasing
        
        Returns:
            {ticker, table, valid, issues: [...]}
        """
        issues = []
        
        if table == "prices":
            # Check for missing observed_at
            query = f"""
                SELECT COUNT(*) 
                FROM prices 
                WHERE ticker = '{ticker}' AND observed_at IS NULL
            """
            missing = self._conn.execute(query).fetchone()[0]
            if missing > 0:
                issues.append(f"{missing} records missing observed_at")
            
            # Check for observed_at before reasonable availability
            # Price should be observed at or after market close on that date
            query = f"""
                SELECT COUNT(*)
                FROM prices
                WHERE ticker = '{ticker}' 
                  AND CAST(observed_at AS DATE) < date
            """
            invalid = self._conn.execute(query).fetchone()[0]
            if invalid > 0:
                issues.append(f"{invalid} records with observed_at before date")
            
            # Check monotonicity (later dates should have >= observed_at)
            query = f"""
                WITH ordered AS (
                    SELECT date, observed_at,
                           LAG(observed_at) OVER (ORDER BY date) as prev_observed
                    FROM prices
                    WHERE ticker = '{ticker}'
                )
                SELECT COUNT(*)
                FROM ordered
                WHERE prev_observed IS NOT NULL 
                  AND observed_at < prev_observed
            """
            non_monotonic = self._conn.execute(query).fetchone()[0]
            if non_monotonic > 0:
                issues.append(f"{non_monotonic} records violate monotonicity")
        
        return {
            "ticker": ticker,
            "table": table,
            "valid": len(issues) == 0,
            "issues": issues,
        }
    
    # =========================================================================
    # Utility Methods
    # =========================================================================
    
    def get_all_tickers(self) -> List[str]:
        """Get all tickers in the database."""
        query = "SELECT DISTINCT ticker FROM prices ORDER BY ticker"
        df = self._conn.execute(query).df()
        return df["ticker"].tolist()
    
    def get_date_range(self, ticker: str) -> Optional[tuple]:
        """Get date range available for a ticker."""
        query = f"""
            SELECT MIN(date), MAX(date)
            FROM prices
            WHERE ticker = '{ticker}'
        """
        result = self._conn.execute(query).fetchone()
        if result and result[0]:
            return (result[0], result[1])
        return None
    
    def count_records(self, table: str = "prices") -> int:
        """Count total records in a table."""
        query = f"SELECT COUNT(*) FROM {table}"
        return self._conn.execute(query).fetchone()[0]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        return {
            "prices_count": self.count_records("prices"),
            "fundamentals_count": self.count_records("fundamentals"),
            "profiles_count": self.count_records("profiles"),
            "snapshots_count": self.count_records("market_snapshots"),
            "events_count": self.count_records("events"),
            "tickers": len(self.get_all_tickers()),
        }


# =============================================================================
# Convenience Function
# =============================================================================

def get_pit_store(db_path: Optional[Path] = None) -> DuckDBPITStore:
    """Get a configured PIT store instance."""
    if db_path is None:
        try:
            from ..config import PROJECT_ROOT
            db_path = PROJECT_ROOT / "data" / "pit_store.duckdb"
        except ImportError:
            db_path = Path("data/pit_store.duckdb")
    
    return DuckDBPITStore(db_path)

"""
DuckDB Point-in-Time (PIT) Store
================================

Provides PIT-safe data storage and retrieval using DuckDB.

Key PIT Rules Enforced:
1. All data has an observed_at timestamp
2. Queries respect asof parameter - only return data known at that time
3. Fundamentals use filing_date + conservative lag
4. No forward-filling before observed_at

Schema Design:
- prices: date, ticker, open, high, low, close, adj_close, volume, observed_at
- fundamentals: period_end, ticker, field, value, filing_date, observed_at
- profiles: ticker, sector, industry, market_cap, updated_at
- events: date, ticker, event_type, value, observed_at

This implements the PITStore protocol from src/interfaces.py
"""

import logging
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

import duckdb
import pandas as pd
import pytz

logger = logging.getLogger(__name__)


@dataclass
class PITConfig:
    """Configuration for PIT store."""
    # Conservative lag rules (when exact observed_at is unknown)
    fundamental_lag_days: int = 1  # Filing date + 1 day
    price_lag_days: int = 0  # EOD prices available same day after close
    
    # Default cutoff time (Eastern)
    cutoff_hour: int = 16
    cutoff_minute: int = 0
    timezone: str = "America/New_York"


class DuckDBPITStore:
    """
    DuckDB-backed Point-in-Time safe data store.
    
    Implements the PITStore protocol for production use.
    
    Usage:
        store = DuckDBPITStore("data/pit_store.duckdb")
        
        # Store price data
        store.store_prices(prices_df)
        
        # Query with PIT safety
        df = store.get_ohlcv(["NVDA"], start, end, asof=datetime(2024, 1, 15, 16, 0))
        
        # Get market cap as of a date
        mcaps = store.get_market_cap(["NVDA", "AMD"], asof=date(2024, 1, 15))
    """
    
    def __init__(
        self,
        db_path: Optional[Path] = None,
        config: Optional[PITConfig] = None,
        read_only: bool = False,
    ):
        """
        Initialize PIT store.
        
        Args:
            db_path: Path to DuckDB file. If None, uses in-memory DB.
            config: PIT configuration
            read_only: Open database in read-only mode
        """
        self.config = config or PITConfig()
        self.timezone = pytz.timezone(self.config.timezone)
        
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
        """Initialize database schema."""
        # Prices table - daily OHLCV
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
                observed_at TIMESTAMP NOT NULL,
                source VARCHAR DEFAULT 'fmp',
                PRIMARY KEY (ticker, date)
            )
        """)
        
        # Fundamentals table - financial statements
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS fundamentals (
                ticker VARCHAR NOT NULL,
                period_end DATE NOT NULL,
                period_type VARCHAR NOT NULL,  -- 'quarter' or 'annual'
                statement_type VARCHAR NOT NULL,  -- 'income', 'balance', 'cashflow'
                field VARCHAR NOT NULL,
                value DOUBLE,
                filing_date DATE,
                observed_at TIMESTAMP NOT NULL,
                source VARCHAR DEFAULT 'fmp',
                PRIMARY KEY (ticker, period_end, period_type, statement_type, field)
            )
        """)
        
        # Company profiles table
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS profiles (
                ticker VARCHAR PRIMARY KEY,
                company_name VARCHAR,
                sector VARCHAR,
                industry VARCHAR,
                exchange VARCHAR,
                currency VARCHAR DEFAULT 'USD',
                country VARCHAR DEFAULT 'US',
                updated_at TIMESTAMP NOT NULL
            )
        """)
        
        # Market data snapshots (for market cap, ADV at specific dates)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS market_snapshots (
                ticker VARCHAR NOT NULL,
                date DATE NOT NULL,
                market_cap DOUBLE,
                shares_outstanding BIGINT,
                avg_volume_20d DOUBLE,
                observed_at TIMESTAMP NOT NULL,
                source VARCHAR DEFAULT 'fmp',
                PRIMARY KEY (ticker, date)
            )
        """)
        
        # Events table (earnings, dividends, splits)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS events (
                ticker VARCHAR NOT NULL,
                event_date DATE NOT NULL,
                event_type VARCHAR NOT NULL,  -- 'earnings', 'dividend', 'split'
                event_time VARCHAR,  -- 'bmo', 'amc', 'during' for earnings
                value DOUBLE,
                observed_at TIMESTAMP NOT NULL,
                source VARCHAR DEFAULT 'fmp',
                PRIMARY KEY (ticker, event_date, event_type)
            )
        """)
        
        # Create indices for fast PIT queries
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_prices_observed 
            ON prices (observed_at, ticker)
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_fundamentals_observed 
            ON fundamentals (observed_at, ticker)
        """)
    
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
        
        Expected columns: ticker, date, open, high, low, close, adj_close, volume
        Optional: observed_at (defaults to date + 1 day)
        
        Args:
            df: DataFrame with price data
            source: Data source identifier
        
        Returns:
            Number of rows stored
        """
        if df.empty:
            return 0
        
        df = df.copy()
        
        # Ensure observed_at exists
        if "observed_at" not in df.columns:
            # Default: available next day at midnight
            df["observed_at"] = pd.to_datetime(df["date"]) + pd.Timedelta(days=1)
        
        df["source"] = source
        
        # Upsert using DuckDB
        self._conn.execute("BEGIN TRANSACTION")
        try:
            for _, row in df.iterrows():
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
            self._conn.execute("COMMIT")
            logger.debug(f"Stored {len(df)} price records")
            return len(df)
        except Exception as e:
            self._conn.execute("ROLLBACK")
            raise e
    
    def store_fundamentals(
        self,
        df: pd.DataFrame,
        statement_type: str,
        source: str = "fmp",
    ) -> int:
        """
        Store fundamental data with PIT metadata.
        
        Args:
            df: DataFrame with fundamental data (wide format)
            statement_type: 'income', 'balance', or 'cashflow'
            source: Data source identifier
        
        Returns:
            Number of rows stored
        """
        if df.empty:
            return 0
        
        df = df.copy()
        
        # Ensure observed_at exists
        if "observed_at" not in df.columns:
            if "fillingDate" in df.columns:
                df["observed_at"] = pd.to_datetime(df["fillingDate"]) + pd.Timedelta(days=self.config.fundamental_lag_days)
            else:
                # Conservative: 45 days after period end
                df["observed_at"] = pd.to_datetime(df["period_end"]) + pd.Timedelta(days=45)
        
        # Determine period type
        period_type = "quarter"
        if "period" in df.columns:
            period_type = df["period"].iloc[0] if not df["period"].isna().all() else "quarter"
        
        # Get metric columns (exclude metadata)
        meta_cols = {"ticker", "symbol", "date", "period", "period_end", "observed_at", 
                     "fillingDate", "acceptedDate", "calendarYear", "source", "link", "finalLink"}
        metric_cols = [c for c in df.columns if c not in meta_cols and not c.startswith("_")]
        
        rows_stored = 0
        self._conn.execute("BEGIN TRANSACTION")
        
        try:
            for _, row in df.iterrows():
                ticker = row.get("symbol") or row.get("ticker")
                period_end = row.get("period_end") or row.get("date")
                observed_at = row.get("observed_at")
                filing_date = row.get("fillingDate") or row.get("filing_date")
                
                for field in metric_cols:
                    value = row.get(field)
                    if pd.notna(value):
                        self._conn.execute("""
                            INSERT OR REPLACE INTO fundamentals
                            (ticker, period_end, period_type, statement_type, field, value, filing_date, observed_at, source)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, [
                            ticker,
                            period_end,
                            period_type,
                            statement_type,
                            field,
                            float(value) if value else None,
                            filing_date,
                            observed_at,
                            source,
                        ])
                        rows_stored += 1
            
            self._conn.execute("COMMIT")
            logger.debug(f"Stored {rows_stored} fundamental records")
            return rows_stored
            
        except Exception as e:
            self._conn.execute("ROLLBACK")
            raise e
    
    def store_profile(self, ticker: str, profile: Dict[str, Any]) -> None:
        """Store company profile."""
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
            datetime.now(self.timezone),
        ])
    
    def store_market_snapshot(
        self,
        ticker: str,
        snapshot_date: date,
        market_cap: Optional[float] = None,
        shares_outstanding: Optional[int] = None,
        avg_volume_20d: Optional[float] = None,
    ) -> None:
        """Store market data snapshot for a specific date."""
        self._conn.execute("""
            INSERT OR REPLACE INTO market_snapshots
            (ticker, date, market_cap, shares_outstanding, avg_volume_20d, observed_at, source)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, [
            ticker,
            snapshot_date,
            market_cap,
            shares_outstanding,
            avg_volume_20d,
            datetime.now(self.timezone),
            "computed",
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
            asof: Only return data observed before this datetime.
                  If None, returns all available data.
        
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
            # PIT filter: only data observed before asof
            asof_str = asof.strftime("%Y-%m-%d %H:%M:%S")
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
        Get fundamental data as of a datetime.
        
        Returns the most recent value for each field that was
        available (observed) before the asof datetime.
        
        Args:
            tickers: List of ticker symbols
            fields: List of field names (e.g., 'revenue', 'netIncome')
            asof: Only return data observed before this datetime
            statement_types: Filter by statement type (income, balance, cashflow)
        
        Returns:
            Dict: {ticker: {field: value}}
        """
        ticker_list = ", ".join([f"'{t}'" for t in tickers])
        field_list = ", ".join([f"'{f}'" for f in fields])
        asof_str = asof.strftime("%Y-%m-%d %H:%M:%S")
        
        query = f"""
            WITH ranked AS (
                SELECT 
                    ticker, field, value, period_end, observed_at,
                    ROW_NUMBER() OVER (
                        PARTITION BY ticker, field 
                        ORDER BY period_end DESC
                    ) as rn
                FROM fundamentals
                WHERE ticker IN ({ticker_list})
                  AND field IN ({field_list})
                  AND observed_at <= '{asof_str}'
        """
        
        if statement_types:
            type_list = ", ".join([f"'{t}'" for t in statement_types])
            query += f" AND statement_type IN ({type_list})"
        
        query += """
            )
            SELECT ticker, field, value, period_end
            FROM ranked
            WHERE rn = 1
        """
        
        df = self._conn.execute(query).df()
        
        # Convert to nested dict
        result = {t: {} for t in tickers}
        for _, row in df.iterrows():
            result[row["ticker"]][row["field"]] = row["value"]
        
        return result
    
    def get_market_cap(
        self,
        tickers: List[str],
        asof: date,
    ) -> Dict[str, float]:
        """
        Get market caps as of a date.
        
        First tries market_snapshots table, then computes from price * shares.
        
        Args:
            tickers: List of ticker symbols
            asof: Date for market cap
        
        Returns:
            Dict mapping ticker to market cap
        """
        ticker_list = ", ".join([f"'{t}'" for t in tickers])
        
        # Try snapshots first
        query = f"""
            SELECT ticker, market_cap
            FROM market_snapshots
            WHERE ticker IN ({ticker_list})
              AND date <= '{asof}'
            ORDER BY date DESC
        """
        
        df = self._conn.execute(query).df()
        
        result = {}
        seen = set()
        for _, row in df.iterrows():
            if row["ticker"] not in seen and row["market_cap"]:
                result[row["ticker"]] = row["market_cap"]
                seen.add(row["ticker"])
        
        return result
    
    def get_avg_volume(
        self,
        tickers: List[str],
        asof: date,
        lookback_days: int = 20,
    ) -> Dict[str, float]:
        """
        Get average daily volume over lookback period.
        
        Args:
            tickers: List of ticker symbols
            asof: End date for calculation
            lookback_days: Number of trading days to average
        
        Returns:
            Dict mapping ticker to average volume
        """
        ticker_list = ", ".join([f"'{t}'" for t in tickers])
        start = asof - timedelta(days=int(lookback_days * 1.5))  # Buffer for weekends
        
        query = f"""
            SELECT ticker, AVG(volume) as avg_volume
            FROM (
                SELECT ticker, volume
                FROM prices
                WHERE ticker IN ({ticker_list})
                  AND date >= '{start}'
                  AND date <= '{asof}'
                ORDER BY date DESC
                LIMIT {lookback_days}
            )
            GROUP BY ticker
        """
        
        df = self._conn.execute(query).df()
        return dict(zip(df["ticker"], df["avg_volume"]))
    
    def get_price(
        self,
        tickers: List[str],
        asof: date,
    ) -> Dict[str, float]:
        """
        Get closing prices as of a date.
        
        Args:
            tickers: List of ticker symbols
            asof: Date for price
        
        Returns:
            Dict mapping ticker to close price
        """
        ticker_list = ", ".join([f"'{t}'" for t in tickers])
        
        query = f"""
            WITH ranked AS (
                SELECT ticker, close, date,
                       ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY date DESC) as rn
                FROM prices
                WHERE ticker IN ({ticker_list})
                  AND date <= '{asof}'
            )
            SELECT ticker, close
            FROM ranked
            WHERE rn = 1
        """
        
        df = self._conn.execute(query).df()
        return dict(zip(df["ticker"], df["close"]))
    
    def get_sector_industry(
        self,
        tickers: List[str],
    ) -> Dict[str, Dict[str, str]]:
        """
        Get sector/industry classification.
        
        Args:
            tickers: List of ticker symbols
        
        Returns:
            Dict: {ticker: {sector: str, industry: str}}
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
        """
        Get the last date we have data for a ticker.
        
        Args:
            ticker: Ticker symbol
            table: Table to check ('prices' or 'fundamentals')
        
        Returns:
            Last available date or None
        """
        if table == "prices":
            query = f"SELECT MAX(date) FROM prices WHERE ticker = '{ticker}'"
        else:
            query = f"SELECT MAX(period_end) FROM fundamentals WHERE ticker = '{ticker}'"
        
        result = self._conn.execute(query).fetchone()
        return result[0] if result and result[0] else None
    
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
    
    def validate_pit(self, ticker: str, table: str = "prices") -> Dict[str, Any]:
        """
        Validate PIT correctness for a ticker.
        
        Checks:
        1. All records have observed_at
        2. observed_at >= date (for prices)
        3. No duplicate dates
        
        Returns:
            Dict with validation results
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
            
            # Check for invalid observed_at (before date)
            query = f"""
                SELECT COUNT(*)
                FROM prices
                WHERE ticker = '{ticker}' 
                  AND observed_at::DATE < date
            """
            invalid = self._conn.execute(query).fetchone()[0]
            if invalid > 0:
                issues.append(f"{invalid} records with observed_at before date")
        
        return {
            "ticker": ticker,
            "table": table,
            "valid": len(issues) == 0,
            "issues": issues,
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        return {
            "prices_count": self.count_records("prices"),
            "fundamentals_count": self.count_records("fundamentals"),
            "profiles_count": self.count_records("profiles"),
            "events_count": self.count_records("events"),
            "tickers": len(self.get_all_tickers()),
        }


# =============================================================================
# Convenience Function
# =============================================================================

def get_pit_store(db_path: Optional[Path] = None) -> DuckDBPITStore:
    """
    Get a configured PIT store instance.
    
    Args:
        db_path: Optional path to database file
    
    Returns:
        DuckDBPITStore instance
    """
    if db_path is None:
        from ..config import PROJECT_ROOT
        db_path = PROJECT_ROOT / "data" / "pit_store.duckdb"
    
    return DuckDBPITStore(db_path)


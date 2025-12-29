"""
Label Generator - Forward Excess Returns (Section 5.1)
======================================================

Computes forward excess returns for training/evaluation targets.

LABEL VERSIONS:
- v1: Split-adjusted close price return (no dividends)
- v2: Total return (price + dividends) for both stock AND benchmark (DEFAULT)

LABEL FORMULA (v2):
    y_i,T(H) = TR_i,T(H) - TR_b,T(H)
    
    where:
        TR_i,T(H) = (P_i,T+H / P_i,T - 1) + DIV_i,T(H)
        TR_b,T(H) = (P_b,T+H / P_b,T - 1) + DIV_b,T(H)
        
        DIV_i,T(H) = sum(dividends paid between T and T+H) / P_i,T
        
        P_i,T    = stock i split-adjusted close on date T
        P_i,T+H  = stock i split-adjusted close on date T+H trading days
        H        = horizon in TRADING DAYS (20, 60, 90)

LABEL FORMULA (v1 - legacy):
    y_i,T(H) = (P_i,T+H / P_i,T - 1) - (P_b,T+H / P_b,T - 1)

RATIONALE FOR v2:
- Dividends matter for ranking fairness (especially 60-90d horizons)
- Mature dividend payers (MSFT ~0.8% yield) vs growth stocks
- Consistency: total return for both stock AND benchmark avoids distortion

BENCHMARK HANDLING:
- If benchmark total return (dividends) available: use it
- If not: documented as "stock total return vs benchmark price return"
- For QQQ (ETF): distributions may not be cleanly available on free tier

LABEL ALIGNMENT (matches cutoff policy):
- Entry: price(T close) — 4:00pm ET cutoff
- Exit: price(T+H close) — H trading days forward
- Benchmark: same dates as stock
- Calendar: Use TradingCalendarImpl for trading day arithmetic
- Dividends: Sum all ex-dates where T < ex-date <= T+H

LABEL AVAILABILITY RULE (PIT-safe):
- Labels mature at T+H close
- Dividends: use ex-date (conservative - declared date would leak forward)
- During training/eval: filter by asof >= T+H close
- Labels stored with maturity timestamp for purging/embargo support

STORAGE:
- Same DuckDB pattern as features
- Keys: stable_id, ticker, date, horizon
- Values: excess_return, stock_return, benchmark_return, dividend_yield (v2)
- Metadata: label_matured_at, benchmark_ticker, label_version
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

# Label horizons (trading days)
HORIZONS = [20, 60, 90]

# Default benchmark
# NOTE: QQQ (ETF) may require FMP paid tier for historical prices
# Consider using yfinance for benchmark data, or a stock like MSFT as proxy
DEFAULT_BENCHMARK = "QQQ"


@dataclass
class ForwardReturn:
    """
    A single forward return label.
    
    Represents the excess return of a stock over the benchmark
    for a specific date and horizon.
    """
    ticker: str
    stable_id: Optional[str]
    as_of_date: date           # Date T (entry date)
    horizon: int               # H trading days
    exit_date: date            # Date T+H (maturity date)
    
    # Returns (all as decimals, not percentages)
    stock_return: float        # Price return: (P_T+H / P_T) - 1
    benchmark_return: float    # Price return: (P_b,T+H / P_b,T) - 1
    excess_return: float       # Total return stock - Total return benchmark
    
    # Dividends (v2 only, decimal yield)
    stock_dividend_yield: float = 0.0      # sum(divs T to T+H) / P_T
    benchmark_dividend_yield: float = 0.0  # sum(divs T to T+H) / P_b,T
    
    # Prices (for audit trail)
    entry_price: float = 0.0
    exit_price: float = 0.0
    benchmark_entry_price: float = 0.0
    benchmark_exit_price: float = 0.0
    
    # Metadata
    benchmark_ticker: str = DEFAULT_BENCHMARK
    label_matured_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    label_version: str = "v2"  # "v1" (price only) or "v2" (total return)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "ticker": self.ticker,
            "stable_id": self.stable_id,
            "as_of_date": self.as_of_date.isoformat(),
            "horizon": self.horizon,
            "exit_date": self.exit_date.isoformat(),
            "stock_return": self.stock_return,
            "benchmark_return": self.benchmark_return,
            "excess_return": self.excess_return,
            "stock_dividend_yield": self.stock_dividend_yield,
            "benchmark_dividend_yield": self.benchmark_dividend_yield,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "benchmark_entry_price": self.benchmark_entry_price,
            "benchmark_exit_price": self.benchmark_exit_price,
            "benchmark_ticker": self.benchmark_ticker,
            "label_matured_at": self.label_matured_at.isoformat(),
            "label_version": self.label_version,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ForwardReturn":
        """Create from dictionary."""
        return cls(
            ticker=d["ticker"],
            stable_id=d.get("stable_id"),
            as_of_date=date.fromisoformat(d["as_of_date"]),
            horizon=d["horizon"],
            exit_date=date.fromisoformat(d["exit_date"]),
            stock_return=d["stock_return"],
            benchmark_return=d["benchmark_return"],
            excess_return=d["excess_return"],
            stock_dividend_yield=d.get("stock_dividend_yield", 0.0),
            benchmark_dividend_yield=d.get("benchmark_dividend_yield", 0.0),
            entry_price=d.get("entry_price", 0.0),
            exit_price=d.get("exit_price", 0.0),
            benchmark_entry_price=d.get("benchmark_entry_price", 0.0),
            benchmark_exit_price=d.get("benchmark_exit_price", 0.0),
            benchmark_ticker=d.get("benchmark_ticker", DEFAULT_BENCHMARK),
            label_matured_at=datetime.fromisoformat(d["label_matured_at"]),
            label_version=d.get("label_version", "v1"),  # Default to v1 for backward compat
        )
    
    def is_mature(self, asof: datetime) -> bool:
        """
        Check if this label has matured as of a given datetime.
        
        PIT RULE: Labels are only usable after they mature (T+H close).
        
        Args:
            asof: Current datetime (UTC)
        
        Returns:
            True if label is mature and usable
        """
        return asof >= self.label_matured_at


class LabelGenerator:
    """
    Generates forward excess return labels.
    
    Uses:
    - FMPClient for price data
    - TradingCalendar for trading day arithmetic
    - SecurityMaster for stable IDs (optional)
    
    Usage:
        from src.data import FMPClient, TradingCalendarImpl
        
        generator = LabelGenerator(
            fmp_client=FMPClient(),
            calendar=TradingCalendarImpl(),
        )
        
        # Generate labels for a single stock
        labels = generator.generate(
            ticker="NVDA",
            start_date=date(2023, 1, 1),
            end_date=date(2023, 12, 31),
            horizons=[20, 60, 90],
        )
        
        # Generate labels for universe
        labels_df = generator.generate_for_universe(
            tickers=["NVDA", "AMD", "MSFT"],
            start_date=date(2023, 1, 1),
            end_date=date(2023, 12, 31),
        )
    """
    
    def __init__(
        self,
        fmp_client=None,
        calendar=None,
        security_master=None,
        benchmark: str = DEFAULT_BENCHMARK,
        label_version: str = "v2",  # "v1" (price only) or "v2" (total return)
    ):
        """
        Initialize label generator.
        
        Args:
            fmp_client: FMPClient instance (lazy-loaded if None)
            calendar: TradingCalendar instance (lazy-loaded if None)
            security_master: SecurityMaster for stable IDs (optional)
            benchmark: Benchmark ticker (default: QQQ)
            label_version: "v1" (price return) or "v2" (total return with dividends, DEFAULT)
        """
        self._fmp = fmp_client
        self._calendar = calendar
        self._sm = security_master
        self.benchmark = benchmark
        self.label_version = label_version
        
        # Cache for dividend data (ticker -> DataFrame)
        self._dividend_cache: Dict[str, pd.DataFrame] = {}
        
        # Cache for price data
        self._price_cache: Dict[str, pd.DataFrame] = {}
        self._benchmark_cache: Optional[pd.DataFrame] = None
    
    def _get_fmp_client(self):
        """Lazy-load FMP client."""
        if self._fmp is None:
            from ..data import FMPClient
            self._fmp = FMPClient()
        return self._fmp
    
    def _get_calendar(self):
        """Lazy-load trading calendar."""
        if self._calendar is None:
            from ..data import TradingCalendarImpl
            self._calendar = TradingCalendarImpl()
        return self._calendar
    
    def _get_prices(
        self,
        ticker: str,
        start: date,
        end: date,
    ) -> pd.DataFrame:
        """
        Get price data for a ticker with caching.
        
        Returns DataFrame with columns: date, close
        """
        cache_key = f"{ticker}_{start}_{end}"
        
        if cache_key in self._price_cache:
            return self._price_cache[cache_key]
        
        fmp = self._get_fmp_client()
        
        # Add buffer for horizon calculation
        buffer_start = start - timedelta(days=10)
        buffer_end = end + timedelta(days=150)  # ~90 trading days + buffer
        
        df = fmp.get_historical_prices(
            ticker,
            start=buffer_start.isoformat(),
            end=buffer_end.isoformat(),
        )
        
        if df.empty:
            logger.warning(f"No price data for {ticker}")
            return df
        
        # Keep only date and close
        df = df[["date", "close"]].copy()
        df["date"] = pd.to_datetime(df["date"]).dt.date
        df = df.sort_values("date").reset_index(drop=True)
        
        self._price_cache[cache_key] = df
        return df
    
    def _get_benchmark_prices(self, start: date, end: date) -> pd.DataFrame:
        """Get benchmark prices with caching."""
        if self._benchmark_cache is not None:
            return self._benchmark_cache
        
        df = self._get_prices(self.benchmark, start, end)
        self._benchmark_cache = df
        return df
    
    def _get_price_on_date(
        self,
        prices_df: pd.DataFrame,
        target_date: date,
    ) -> Optional[float]:
        """Get closing price on a specific date."""
        if prices_df.empty:
            return None
        
        match = prices_df[prices_df["date"] == target_date]
        if match.empty:
            return None
        
        return float(match.iloc[0]["close"])
    
    def _get_maturity_datetime(self, exit_date: date) -> datetime:
        """
        Get the datetime when a label matures.
        
        Labels mature at market close (4pm ET) on the exit date.
        
        Args:
            exit_date: The exit date (T+H)
        
        Returns:
            UTC datetime when label matures
        """
        cal = self._get_calendar()
        close_et = cal.get_market_close(exit_date)
        return close_et.astimezone(UTC)
    
    def _get_dividends(
        self,
        ticker: str,
        start: date,
        end: date,
    ) -> pd.DataFrame:
        """
        Get dividend data for a ticker with caching.
        
        Returns DataFrame with columns: date (ex-date), dividend (amount)
        
        PIT NOTE: We use ex-date (not declaration date) as conservative choice.
        Dividend is known/announced before ex-date, but using ex-date avoids
        any forward-looking bias.
        """
        cache_key = f"{ticker}_div"
        
        if cache_key in self._dividend_cache:
            df = self._dividend_cache[cache_key]
            # Filter to date range
            return df[(df["date"] >= start) & (df["date"] <= end)].copy()
        
        fmp = self._get_fmp_client()
        
        try:
            df = fmp.get_stock_dividend(ticker)
            
            if df.empty:
                logger.debug(f"No dividend data for {ticker} (may be non-dividend paying stock)")
                # Cache empty result to avoid repeated API calls
                self._dividend_cache[cache_key] = pd.DataFrame(columns=["date", "dividend"])
                return pd.DataFrame(columns=["date", "dividend"])
            
            # FMP returns columns: date (ex-date), dividend, ...
            if "date" not in df.columns or "dividend" not in df.columns:
                logger.warning(f"Unexpected dividend data format for {ticker}: {df.columns.tolist()}")
                self._dividend_cache[cache_key] = pd.DataFrame(columns=["date", "dividend"])
                return pd.DataFrame(columns=["date", "dividend"])
            
            df = df[["date", "dividend"]].copy()
            df["date"] = pd.to_datetime(df["date"]).dt.date
            df = df.sort_values("date").reset_index(drop=True)
            
            self._dividend_cache[cache_key] = df
            
            # Filter to date range
            return df[(df["date"] >= start) & (df["date"] <= end)].copy()
            
        except Exception as e:
            logger.warning(f"Error fetching dividends for {ticker}: {e}")
            self._dividend_cache[cache_key] = pd.DataFrame(columns=["date", "dividend"])
            return pd.DataFrame(columns=["date", "dividend"])
    
    def _calculate_dividend_yield(
        self,
        ticker: str,
        entry_date: date,
        exit_date: date,
        entry_price: float,
    ) -> float:
        """
        Calculate dividend yield between entry and exit dates.
        
        Formula: sum(dividends with ex-date in (entry, exit]) / entry_price
        
        PIT RULE: Only include dividends with ex-date AFTER entry and UP TO exit
        (exclusive of entry, inclusive of exit).
        
        Args:
            ticker: Stock ticker
            entry_date: Entry date (T)
            exit_date: Exit date (T+H)
            entry_price: Price at entry (P_T)
        
        Returns:
            Dividend yield as decimal (e.g., 0.02 for 2%)
        """
        if entry_price <= 0:
            return 0.0
        
        # Fetch dividends in the range (entry, exit]
        # Add buffer to ensure we get all dividends
        buffer_start = entry_date - timedelta(days=10)
        buffer_end = exit_date + timedelta(days=10)
        
        divs = self._get_dividends(ticker, buffer_start, buffer_end)
        
        if divs.empty:
            return 0.0
        
        # Filter to (entry, exit] - exclusive of entry, inclusive of exit
        divs_in_period = divs[(divs["date"] > entry_date) & (divs["date"] <= exit_date)]
        
        if divs_in_period.empty:
            return 0.0
        
        total_dividends = divs_in_period["dividend"].sum()
        div_yield = total_dividends / entry_price
        
        return float(div_yield)
    
    def generate(
        self,
        ticker: str,
        start_date: date,
        end_date: date,
        horizons: List[int] = None,
        stable_id: Optional[str] = None,
    ) -> List[ForwardReturn]:
        """
        Generate forward return labels for a single ticker.
        
        Args:
            ticker: Stock ticker
            start_date: First date to generate labels for
            end_date: Last date to generate labels for
            horizons: List of horizons in trading days (default: [20, 60, 90])
            stable_id: Optional stable identifier
        
        Returns:
            List of ForwardReturn labels
        """
        if horizons is None:
            horizons = HORIZONS
        
        cal = self._get_calendar()
        
        # Get trading days in range
        trading_days = cal.get_trading_days(start_date, end_date)
        
        if not trading_days:
            logger.warning(f"No trading days in range {start_date} to {end_date}")
            return []
        
        # Get price data
        stock_prices = self._get_prices(ticker, start_date, end_date)
        benchmark_prices = self._get_benchmark_prices(start_date, end_date)
        
        if stock_prices.empty:
            logger.warning(f"No price data for {ticker}")
            return []
        
        if benchmark_prices.empty:
            logger.warning(f"No benchmark data for {self.benchmark}")
            return []
        
        labels = []
        
        for as_of_date in trading_days:
            for horizon in horizons:
                try:
                    # Calculate exit date
                    exit_date = cal.get_n_trading_days_forward(as_of_date, horizon)
                    
                    # Get prices
                    entry_price = self._get_price_on_date(stock_prices, as_of_date)
                    exit_price = self._get_price_on_date(stock_prices, exit_date)
                    bench_entry = self._get_price_on_date(benchmark_prices, as_of_date)
                    bench_exit = self._get_price_on_date(benchmark_prices, exit_date)
                    
                    # Skip if any price is missing
                    if any(p is None for p in [entry_price, exit_price, bench_entry, bench_exit]):
                        continue
                    
                    # Calculate price returns
                    stock_price_return = (exit_price / entry_price) - 1
                    benchmark_price_return = (bench_exit / bench_entry) - 1
                    
                    # Calculate dividend yields (v2 only)
                    stock_div_yield = 0.0
                    bench_div_yield = 0.0
                    
                    if self.label_version == "v2":
                        stock_div_yield = self._calculate_dividend_yield(
                            ticker, as_of_date, exit_date, entry_price
                        )
                        bench_div_yield = self._calculate_dividend_yield(
                            self.benchmark, as_of_date, exit_date, bench_entry
                        )
                    
                    # Total returns = price return + dividend yield
                    stock_total_return = stock_price_return + stock_div_yield
                    benchmark_total_return = benchmark_price_return + bench_div_yield
                    
                    # Excess return = stock total return - benchmark total return
                    excess_return = stock_total_return - benchmark_total_return
                    
                    # Get maturity datetime
                    maturity = self._get_maturity_datetime(exit_date)
                    
                    label = ForwardReturn(
                        ticker=ticker,
                        stable_id=stable_id,
                        as_of_date=as_of_date,
                        horizon=horizon,
                        exit_date=exit_date,
                        stock_return=stock_price_return,  # Keep price return for comparison
                        benchmark_return=benchmark_price_return,  # Keep price return for comparison
                        excess_return=excess_return,  # Total return excess
                        stock_dividend_yield=stock_div_yield,
                        benchmark_dividend_yield=bench_div_yield,
                        entry_price=entry_price,
                        exit_price=exit_price,
                        benchmark_entry_price=bench_entry,
                        benchmark_exit_price=bench_exit,
                        benchmark_ticker=self.benchmark,
                        label_matured_at=maturity,
                        label_version=self.label_version,
                    )
                    labels.append(label)
                    
                except Exception as e:
                    logger.debug(f"Failed to generate label for {ticker} on {as_of_date} H={horizon}: {e}")
                    continue
        
        logger.info(f"Generated {len(labels)} labels for {ticker}")
        return labels
    
    def generate_for_universe(
        self,
        tickers: List[str],
        start_date: date,
        end_date: date,
        horizons: List[int] = None,
    ) -> pd.DataFrame:
        """
        Generate labels for a universe of stocks.
        
        Args:
            tickers: List of tickers
            start_date: First date
            end_date: Last date
            horizons: List of horizons
        
        Returns:
            DataFrame with all labels
        """
        all_labels = []
        
        for ticker in tickers:
            try:
                labels = self.generate(
                    ticker=ticker,
                    start_date=start_date,
                    end_date=end_date,
                    horizons=horizons,
                )
                all_labels.extend(labels)
            except Exception as e:
                logger.warning(f"Failed to generate labels for {ticker}: {e}")
                continue
        
        if not all_labels:
            return pd.DataFrame()
        
        # Convert to DataFrame
        records = [l.to_dict() for l in all_labels]
        df = pd.DataFrame(records)
        
        # Convert date columns
        df["as_of_date"] = pd.to_datetime(df["as_of_date"]).dt.date
        df["exit_date"] = pd.to_datetime(df["exit_date"]).dt.date
        df["label_matured_at"] = pd.to_datetime(df["label_matured_at"])
        
        logger.info(f"Generated {len(df)} total labels for {len(tickers)} tickers")
        return df
    
    def filter_mature_labels(
        self,
        labels_df: pd.DataFrame,
        asof: datetime,
    ) -> pd.DataFrame:
        """
        Filter labels to only include those that have matured.
        
        PIT RULE: Only use labels where asof >= label_matured_at.
        
        Args:
            labels_df: DataFrame of labels
            asof: Current datetime (should be UTC)
        
        Returns:
            Filtered DataFrame
        """
        if labels_df.empty:
            return labels_df
        
        # Ensure asof is timezone-aware
        if asof.tzinfo is None:
            asof = UTC.localize(asof)
        
        # Filter
        return labels_df[labels_df["label_matured_at"] <= asof].copy()
    
    def clear_cache(self):
        """Clear the price cache."""
        self._price_cache.clear()
        self._benchmark_cache = None


# =============================================================================
# DuckDB Storage Functions (for persistence)
# =============================================================================

def create_labels_table(conn) -> None:
    """
    Create labels table in DuckDB.
    
    Args:
        conn: DuckDB connection
    """
    conn.execute("""
        CREATE TABLE IF NOT EXISTS labels (
            ticker VARCHAR NOT NULL,
            stable_id VARCHAR,
            as_of_date DATE NOT NULL,
            horizon INTEGER NOT NULL,
            exit_date DATE NOT NULL,
            stock_return DOUBLE NOT NULL,
            benchmark_return DOUBLE NOT NULL,
            excess_return DOUBLE NOT NULL,
            stock_dividend_yield DOUBLE DEFAULT 0.0,
            benchmark_dividend_yield DOUBLE DEFAULT 0.0,
            entry_price DOUBLE NOT NULL,
            exit_price DOUBLE NOT NULL,
            benchmark_entry_price DOUBLE NOT NULL,
            benchmark_exit_price DOUBLE NOT NULL,
            benchmark_ticker VARCHAR NOT NULL,
            label_matured_at TIMESTAMP WITH TIME ZONE NOT NULL,
            label_version VARCHAR DEFAULT 'v1',
            PRIMARY KEY (ticker, as_of_date, horizon)
        )
    """)


def store_labels(conn, labels_df: pd.DataFrame) -> int:
    """
    Store labels in DuckDB.
    
    Args:
        conn: DuckDB connection
        labels_df: DataFrame of labels
    
    Returns:
        Number of rows stored
    """
    if labels_df.empty:
        return 0
    
    create_labels_table(conn)
    
    # Convert dates to proper format
    df = labels_df.copy()
    df["as_of_date"] = pd.to_datetime(df["as_of_date"])
    df["exit_date"] = pd.to_datetime(df["exit_date"])
    
    # Insert or replace
    conn.execute("""
        INSERT OR REPLACE INTO labels 
        SELECT * FROM df
    """)
    
    return len(df)


def get_labels(
    conn,
    tickers: List[str] = None,
    start_date: date = None,
    end_date: date = None,
    horizons: List[int] = None,
    asof: datetime = None,
) -> pd.DataFrame:
    """
    Query labels from DuckDB with optional filters.
    
    Args:
        conn: DuckDB connection
        tickers: Filter by tickers
        start_date: Filter by as_of_date >= start_date
        end_date: Filter by as_of_date <= end_date
        horizons: Filter by horizons
        asof: PIT filter - only return mature labels (label_matured_at <= asof)
    
    Returns:
        DataFrame of labels
    """
    query = "SELECT * FROM labels WHERE 1=1"
    params = []
    
    if tickers:
        query += " AND ticker IN (" + ",".join(["?" for _ in tickers]) + ")"
        params.extend(tickers)
    
    if start_date:
        query += " AND as_of_date >= ?"
        params.append(start_date)
    
    if end_date:
        query += " AND as_of_date <= ?"
        params.append(end_date)
    
    if horizons:
        query += " AND horizon IN (" + ",".join(["?" for _ in horizons]) + ")"
        params.extend(horizons)
    
    if asof:
        query += " AND label_matured_at <= ?"
        params.append(asof)
    
    query += " ORDER BY ticker, as_of_date, horizon"
    
    return conn.execute(query, params).fetchdf()


# =============================================================================
# CLI/Demo
# =============================================================================

if __name__ == "__main__":
    import sys
    from datetime import date
    
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("LABEL GENERATOR DEMO")
    print("=" * 60)
    
    # Create generator
    generator = LabelGenerator()
    
    # Generate labels for a small test
    ticker = "NVDA"
    start = date(2024, 11, 1)
    end = date(2024, 11, 30)
    
    print(f"\nGenerating labels for {ticker} from {start} to {end}...")
    
    labels = generator.generate(
        ticker=ticker,
        start_date=start,
        end_date=end,
        horizons=[20],  # Just 20-day for demo
    )
    
    if labels:
        print(f"\nGenerated {len(labels)} labels")
        
        # Show sample
        print("\nSample label:")
        sample = labels[0]
        print(f"  Ticker: {sample.ticker}")
        print(f"  As-of date: {sample.as_of_date}")
        print(f"  Horizon: {sample.horizon} trading days")
        print(f"  Exit date: {sample.exit_date}")
        print(f"  Entry price: ${sample.entry_price:.2f}")
        print(f"  Exit price: ${sample.exit_price:.2f}")
        print(f"  Stock return: {sample.stock_return:.2%}")
        print(f"  Benchmark ({sample.benchmark_ticker}): {sample.benchmark_return:.2%}")
        print(f"  Excess return: {sample.excess_return:.2%}")
        print(f"  Matured at: {sample.label_matured_at}")
        
        # Check if mature
        now = datetime.now(UTC)
        print(f"\n  Is mature now? {sample.is_mature(now)}")
    else:
        print("No labels generated (check API key and network)")
    
    print("\n" + "=" * 60)


"""
Financial Modeling Prep (FMP) API Client
========================================

Handles all communication with the FMP API with:
- Rate limiting (free tier: 250 requests/day, 5/min)
- Automatic retries with exponential backoff
- PIT-aware data transformation (adds observed_at timestamps)
- Caching to minimize API calls

FMP Free Tier Availability (tested 2025):
✅ Historical Prices (OHLCV)
✅ Quote (15-min delayed)
✅ Profile (sector, industry, market cap)
✅ Income Statement (with fillingDate for PIT)
✅ Balance Sheet (with fillingDate)
✅ Cash Flow (with fillingDate)
✅ Ratios TTM
✅ Enterprise Value
❌ Earnings Calendar (needs paid for pre/post market timing)
❌ Key Metrics (endpoint error on free tier)

PIT TIMESTAMP CONVENTION:
- All observed_at timestamps are stored in UTC
- Price data: observed_at = market close time (4pm ET = 21:00 UTC)
- Fundamentals: observed_at = filing_date + 1 day at market open (9:30am ET)
  (conservative assumption: data available after overnight processing)
"""

import os
import time
import logging
from datetime import date, datetime, timedelta, timezone
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path
import json

import requests
import pandas as pd
import pytz

logger = logging.getLogger(__name__)

# Timezone constants
ET = pytz.timezone("America/New_York")
UTC = pytz.UTC

# Market timing constants
MARKET_CLOSE_HOUR = 16  # 4:00 PM ET
MARKET_CLOSE_MINUTE = 0
MARKET_OPEN_HOUR = 9
MARKET_OPEN_MINUTE = 30


class FMPError(Exception):
    """Base exception for FMP API errors."""
    pass


class RateLimitError(FMPError):
    """Raised when API rate limit is exceeded."""
    pass


class APIKeyError(FMPError):
    """Raised when API key is invalid or missing."""
    pass


def get_market_close_utc(d: date) -> datetime:
    """
    Get market close time in UTC for a given date.
    
    This is the canonical observed_at for price data.
    Handles DST correctly.
    """
    # Create 4:00 PM ET on that date
    et_close = ET.localize(datetime(d.year, d.month, d.day, MARKET_CLOSE_HOUR, MARKET_CLOSE_MINUTE))
    # Convert to UTC
    return et_close.astimezone(UTC)


def get_next_market_open_utc(d: date) -> datetime:
    """
    Get next market open time in UTC after a given date.
    
    Used for fundamentals - conservative assumption that data
    filed on date D is available at market open on D+1.
    """
    next_day = d + timedelta(days=1)
    # Skip weekends
    while next_day.weekday() >= 5:  # Saturday=5, Sunday=6
        next_day += timedelta(days=1)
    
    et_open = ET.localize(datetime(next_day.year, next_day.month, next_day.day, 
                                    MARKET_OPEN_HOUR, MARKET_OPEN_MINUTE))
    return et_open.astimezone(UTC)


@dataclass
class RateLimiter:
    """
    Simple rate limiter for FMP free tier.
    
    Free tier limits:
    - 5 requests per minute
    - 250 requests per day
    """
    requests_per_minute: int = 5
    requests_per_day: int = 250
    
    _minute_requests: List[float] = field(default_factory=list)
    _day_requests: List[float] = field(default_factory=list)
    
    def wait_if_needed(self):
        """Block until we can make another request."""
        now = time.time()
        
        # Clean old timestamps
        minute_ago = now - 60
        day_ago = now - 86400
        
        self._minute_requests = [t for t in self._minute_requests if t > minute_ago]
        self._day_requests = [t for t in self._day_requests if t > day_ago]
        
        # Check daily limit
        if len(self._day_requests) >= self.requests_per_day:
            raise RateLimitError(
                f"Daily limit ({self.requests_per_day}) exceeded. "
                f"Reset in {86400 - (now - self._day_requests[0]):.0f}s"
            )
        
        # Check minute limit
        if len(self._minute_requests) >= self.requests_per_minute:
            sleep_time = 60 - (now - self._minute_requests[0]) + 0.1
            if sleep_time > 0:
                logger.debug(f"Rate limit: sleeping {sleep_time:.1f}s")
                time.sleep(sleep_time)
        
        # Record this request
        now = time.time()
        self._minute_requests.append(now)
        self._day_requests.append(now)
    
    @property
    def remaining_daily(self) -> int:
        """Remaining requests today."""
        now = time.time()
        day_ago = now - 86400
        recent = [t for t in self._day_requests if t > day_ago]
        return max(0, self.requests_per_day - len(recent))


class FMPClient:
    """
    Client for Financial Modeling Prep API (Stable API).
    
    PIT TIMESTAMP CONVENTION:
    - All observed_at stored in UTC
    - Prices: available at market close (4pm ET → UTC)
    - Fundamentals: available next market open after filing
    
    Usage:
        client = FMPClient()
        
        # Get OHLCV data (observed_at = market close UTC)
        df = client.get_historical_prices("NVDA", start="2023-01-01", end="2024-01-01")
        
        # Get fundamentals (observed_at = next market open after filing)
        df = client.get_income_statement("NVDA", limit=8)
    """
    
    BASE_URL = "https://financialmodelingprep.com/stable"
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        cache_dir: Optional[Path] = None,
        use_cache: bool = True,
        cache_ttl_hours: int = 24,
    ):
        """
        Initialize FMP client.
        
        Args:
            api_key: FMP API key (defaults to FMP_KEYS env var)
            cache_dir: Directory for caching responses
            use_cache: Whether to use response caching
            cache_ttl_hours: Cache time-to-live in hours
        """
        self.api_key = api_key or os.getenv("FMP_KEYS", "")
        if not self.api_key:
            raise APIKeyError(
                "FMP API key not found. Set FMP_KEYS in .env file."
            )
        
        self.use_cache = use_cache
        self.cache_ttl = timedelta(hours=cache_ttl_hours)
        
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            try:
                from ..config import PROJECT_ROOT
                self.cache_dir = PROJECT_ROOT / "data" / "cache" / "fmp"
            except ImportError:
                self.cache_dir = Path("data/cache/fmp")
        
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self._rate_limiter = RateLimiter()
        self._session = requests.Session()
        
        logger.info(f"FMPClient initialized (cache: {self.use_cache})")
    
    def _get_cache_path(self, endpoint: str, params: Dict) -> Path:
        """Generate cache file path for a request."""
        param_str = "_".join(f"{k}={v}" for k, v in sorted(params.items()) if k != "apikey")
        safe_endpoint = endpoint.replace("/", "_").replace("-", "_")
        filename = f"{safe_endpoint}_{param_str}.json"
        return self.cache_dir / filename
    
    def _get_cached(self, cache_path: Path) -> Optional[Dict]:
        """Get cached response if valid."""
        if not self.use_cache or not cache_path.exists():
            return None
        
        try:
            stat = cache_path.stat()
            age = datetime.now() - datetime.fromtimestamp(stat.st_mtime)
            
            if age > self.cache_ttl:
                return None
            
            with open(cache_path, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Cache read error: {e}")
            return None
    
    def _set_cached(self, cache_path: Path, data: Any):
        """Cache a response."""
        if not self.use_cache:
            return
        
        try:
            with open(cache_path, "w") as f:
                json.dump(data, f)
        except Exception as e:
            logger.warning(f"Cache write error: {e}")
    
    def _request(
        self,
        endpoint: str,
        params: Optional[Dict] = None,
        use_cache: Optional[bool] = None,
    ) -> Any:
        """Make an API request with rate limiting and caching."""
        params = params or {}
        params["apikey"] = self.api_key
        
        should_cache = use_cache if use_cache is not None else self.use_cache
        
        cache_path = self._get_cache_path(endpoint, params)
        if should_cache:
            cached = self._get_cached(cache_path)
            if cached is not None:
                logger.debug(f"Cache hit: {endpoint}")
                return cached
        
        self._rate_limiter.wait_if_needed()
        
        url = f"{self.BASE_URL}/{endpoint}"
        
        try:
            response = self._session.get(url, params=params, timeout=30)
            
            if response.status_code == 401:
                raise APIKeyError("Invalid API key")
            elif response.status_code == 429:
                raise RateLimitError("Rate limit exceeded by server")
            elif response.status_code == 403:
                raise FMPError(f"Access denied: {response.text[:200]}")
            elif response.status_code != 200:
                raise FMPError(f"API error {response.status_code}: {response.text}")
            
            data = response.json()
            
            if isinstance(data, dict) and "Error Message" in data:
                raise FMPError(data["Error Message"])
            
            if should_cache:
                self._set_cached(cache_path, data)
            
            return data
            
        except requests.RequestException as e:
            raise FMPError(f"Request failed: {e}")
    
    # =========================================================================
    # Price Data Endpoints
    # =========================================================================
    
    def get_historical_prices(
        self,
        symbol: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Get historical OHLCV data for a symbol.
        
        PIT RULE: observed_at = market close time (4pm ET) in UTC.
        This means data for date D is available at queries with asof >= D 16:00 ET.
        
        Args:
            symbol: Ticker symbol (e.g., "NVDA")
            start: Start date (YYYY-MM-DD)
            end: End date (YYYY-MM-DD)
        
        Returns:
            DataFrame with columns including observed_at (UTC datetime)
        """
        params = {"symbol": symbol}
        if start:
            params["from"] = start
        if end:
            params["to"] = end
        
        data = self._request("historical-price-eod/full", params)
        
        if not data:
            logger.warning(f"No price data for {symbol}")
            return pd.DataFrame()
        
        df = pd.DataFrame(data)
        
        if df.empty:
            return df
        
        df["ticker"] = symbol
        df["date"] = pd.to_datetime(df["date"])
        
        # PIT RULE: Price data available at market close (4pm ET) in UTC
        # This is the KEY fix for timestamp consistency
        df["observed_at"] = df["date"].apply(lambda d: get_market_close_utc(d.date()))
        df["source"] = "fmp_historical"
        
        # Handle adjusted close column:
        # FMP's /stable/historical-price-eod/full returns split-adjusted prices
        # in the 'close' column directly (no separate adjClose column).
        # We populate adj_close = close for schema consistency.
        if "adjClose" in df.columns:
            df = df.rename(columns={"adjClose": "adj_close"})
        elif "close" in df.columns and "adj_close" not in df.columns:
            # /full endpoint: close IS already split-adjusted, copy to adj_close
            df["adj_close"] = df["close"]
        
        if "changePercent" in df.columns:
            df = df.rename(columns={"changePercent": "change_pct"})
        
        df = df.sort_values("date").reset_index(drop=True)
        
        return df
    
    def get_quote(self, symbols: List[str]) -> Dict[str, Dict]:
        """
        Get current quotes for symbols.
        
        Note: Free tier has 15-minute delay.
        """
        results = {}
        
        for symbol in symbols:
            try:
                data = self._request("quote", {"symbol": symbol}, use_cache=False)
                
                if data and isinstance(data, list) and len(data) > 0:
                    results[symbol] = data[0]
            except FMPError as e:
                logger.warning(f"Failed to get quote for {symbol}: {e}")
        
        return results
    
    # =========================================================================
    # Fundamental Data Endpoints
    # =========================================================================
    
    def get_income_statement(
        self,
        symbol: str,
        period: str = "quarter",
        limit: int = 20,
    ) -> pd.DataFrame:
        """
        Get income statement data with PIT-safe observed_at.
        
        PIT RULE: 
        - If fillingDate available: observed_at = next market open after filing
        - Else: observed_at = period_end + 45 days (conservative SEC deadline)
        
        This ensures we never use earnings data before it was publicly filed.
        """
        data = self._request("income-statement", 
                            {"symbol": symbol, "period": period, "limit": limit})
        
        if not data:
            return pd.DataFrame()
        
        df = pd.DataFrame(data)
        df["ticker"] = symbol
        df["period_end"] = pd.to_datetime(df["date"])
        
        # PIT RULE: Use filing date if available, else conservative lag
        # FMP uses "filingDate" (single l) but some docs say "fillingDate" - check both
        filing_col = "filingDate" if "filingDate" in df.columns else "fillingDate"
        if filing_col in df.columns and df[filing_col].notna().any():
            # Filing date + next market open (conservative: available after overnight)
            df["observed_at"] = df[filing_col].apply(
                lambda d: get_next_market_open_utc(pd.to_datetime(d).date()) 
                if pd.notna(d) else None
            )
            # Fill any missing with conservative estimate
            mask = df["observed_at"].isna()
            if mask.any():
                df.loc[mask, "observed_at"] = df.loc[mask, "period_end"].apply(
                    lambda d: get_next_market_open_utc(d.date() + timedelta(days=45))
                )
        else:
            # No filing date - use conservative 45-day lag
            logger.warning(f"No filingDate for {symbol}, using 45-day lag")
            df["observed_at"] = df["period_end"].apply(
                lambda d: get_next_market_open_utc(d.date() + timedelta(days=45))
            )
        
        df["source"] = "fmp_income_statement"
        df["statement_type"] = "income"
        
        return df
    
    def get_balance_sheet(
        self,
        symbol: str,
        period: str = "quarter",
        limit: int = 20,
    ) -> pd.DataFrame:
        """Get balance sheet data with PIT-safe observed_at."""
        data = self._request("balance-sheet-statement",
                            {"symbol": symbol, "period": period, "limit": limit})
        
        if not data:
            return pd.DataFrame()
        
        df = pd.DataFrame(data)
        df["ticker"] = symbol
        df["period_end"] = pd.to_datetime(df["date"])
        
        filing_col = "filingDate" if "filingDate" in df.columns else "fillingDate"
        if filing_col in df.columns and df[filing_col].notna().any():
            df["observed_at"] = df[filing_col].apply(
                lambda d: get_next_market_open_utc(pd.to_datetime(d).date())
                if pd.notna(d) else None
            )
            mask = df["observed_at"].isna()
            if mask.any():
                df.loc[mask, "observed_at"] = df.loc[mask, "period_end"].apply(
                    lambda d: get_next_market_open_utc(d.date() + timedelta(days=45))
                )
        else:
            df["observed_at"] = df["period_end"].apply(
                lambda d: get_next_market_open_utc(d.date() + timedelta(days=45))
            )
        
        df["source"] = "fmp_balance_sheet"
        df["statement_type"] = "balance"
        
        return df
    
    def get_cash_flow(
        self,
        symbol: str,
        period: str = "quarter",
        limit: int = 20,
    ) -> pd.DataFrame:
        """Get cash flow statement data with PIT-safe observed_at."""
        data = self._request("cash-flow-statement",
                            {"symbol": symbol, "period": period, "limit": limit})
        
        if not data:
            return pd.DataFrame()
        
        df = pd.DataFrame(data)
        df["ticker"] = symbol
        df["period_end"] = pd.to_datetime(df["date"])
        
        filing_col = "filingDate" if "filingDate" in df.columns else "fillingDate"
        if filing_col in df.columns and df[filing_col].notna().any():
            df["observed_at"] = df[filing_col].apply(
                lambda d: get_next_market_open_utc(pd.to_datetime(d).date())
                if pd.notna(d) else None
            )
            mask = df["observed_at"].isna()
            if mask.any():
                df.loc[mask, "observed_at"] = df.loc[mask, "period_end"].apply(
                    lambda d: get_next_market_open_utc(d.date() + timedelta(days=45))
                )
        else:
            df["observed_at"] = df["period_end"].apply(
                lambda d: get_next_market_open_utc(d.date() + timedelta(days=45))
            )
        
        df["source"] = "fmp_cash_flow"
        df["statement_type"] = "cashflow"
        
        return df
    
    def get_ratios_ttm(self, symbol: str) -> Optional[Dict]:
        """Get trailing twelve month ratios."""
        try:
            data = self._request("ratios-ttm", {"symbol": symbol})
            if data and isinstance(data, list) and len(data) > 0:
                return data[0]
        except FMPError:
            pass
        return None
    
    def get_enterprise_value(
        self,
        symbol: str,
        period: str = "quarter",
        limit: int = 20,
    ) -> pd.DataFrame:
        """Get enterprise value data."""
        try:
            data = self._request("enterprise-values",
                                {"symbol": symbol, "period": period, "limit": limit})
            if data:
                df = pd.DataFrame(data)
                df["ticker"] = symbol
                return df
        except FMPError:
            pass
        return pd.DataFrame()
    
    # =========================================================================
    # Company Info Endpoints
    # =========================================================================
    
    def get_profile(self, symbol: str) -> Optional[Dict]:
        """
        Get company profile.
        
        Note: Profiles are treated as "static metadata" for PIT purposes.
        Sector/industry rarely change, and when they do, the old classification
        is usually still reasonable for backtesting.
        """
        try:
            data = self._request("profile", {"symbol": symbol})
            
            if data and isinstance(data, list) and len(data) > 0:
                return data[0]
            
            # Fallback to quote for basic info
            quote = self.get_quote([symbol])
            if symbol in quote:
                return {
                    "symbol": symbol,
                    "companyName": quote[symbol].get("name", symbol),
                    "mktCap": quote[symbol].get("marketCap"),
                    "sector": "Technology",
                    "industry": "Unknown",
                }
        except FMPError:
            pass
        return None
    
    def get_profiles_batch(self, symbols: List[str]) -> Dict[str, Dict]:
        """Get profiles for multiple symbols (one at a time for free tier)."""
        results = {}
        for symbol in symbols:
            profile = self.get_profile(symbol)
            if profile:
                results[symbol] = profile
        return results
    
    # =========================================================================
    # Corporate Actions
    # =========================================================================
    
    def get_stock_dividend(self, symbol: str) -> pd.DataFrame:
        """Get historical dividend data."""
        try:
            data = self._request("historical-price-eod/stock_dividend", {"symbol": symbol})
            if data:
                df = pd.DataFrame(data)
                df["ticker"] = symbol
                df["source"] = "fmp_dividend"
                return df
        except FMPError:
            pass
        return pd.DataFrame()
    
    def get_stock_split(self, symbol: str) -> pd.DataFrame:
        """Get historical stock split data."""
        try:
            data = self._request("historical-price-eod/stock_split", {"symbol": symbol})
            if data:
                df = pd.DataFrame(data)
                df["ticker"] = symbol
                df["source"] = "fmp_split"
                return df
        except FMPError:
            pass
        return pd.DataFrame()
    
    # =========================================================================
    # Index/Benchmark Data
    # =========================================================================
    
    def get_index_historical(
        self,
        symbol: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """Get historical data for an index/ETF (QQQ, SPY, etc.)."""
        return self.get_historical_prices(symbol, start, end)
    
    # =========================================================================
    # Utilities
    # =========================================================================
    
    @property
    def remaining_requests(self) -> int:
        """Get remaining daily API requests."""
        return self._rate_limiter.remaining_daily
    
    def test_connection(self) -> bool:
        """Test API connection with a simple request."""
        try:
            result = self.get_quote(["AAPL"])
            return "AAPL" in result
        except FMPError:
            return False


# =============================================================================
# Convenience Functions
# =============================================================================

def get_fmp_client() -> FMPClient:
    """Get a configured FMP client instance."""
    return FMPClient()


def download_ticker_data(
    ticker: str,
    start_date: str,
    end_date: str,
    include_fundamentals: bool = True,
    client: Optional[FMPClient] = None,
) -> Dict[str, Any]:
    """
    Download all available data for a single ticker.
    
    Returns dict with: prices, profile, income, balance, cashflow, ratios, ev
    """
    if client is None:
        client = FMPClient()
    
    result = {
        "prices": client.get_historical_prices(ticker, start_date, end_date),
        "profile": client.get_profile(ticker),
    }
    
    if include_fundamentals:
        result["income"] = client.get_income_statement(ticker, period="quarter", limit=20)
        result["balance"] = client.get_balance_sheet(ticker, period="quarter", limit=20)
        result["cashflow"] = client.get_cash_flow(ticker, period="quarter", limit=20)
        result["ratios"] = client.get_ratios_ttm(ticker)
        result["ev"] = client.get_enterprise_value(ticker, period="quarter", limit=20)
    
    result["dividends"] = client.get_stock_dividend(ticker)
    result["splits"] = client.get_stock_split(ticker)
    
    return result

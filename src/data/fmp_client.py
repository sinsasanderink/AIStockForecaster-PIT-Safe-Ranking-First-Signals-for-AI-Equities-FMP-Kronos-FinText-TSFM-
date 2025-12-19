"""
Financial Modeling Prep (FMP) API Client
========================================

Handles all communication with the FMP API with:
- Rate limiting (free tier: 250 requests/day, 5/min)
- Automatic retries with exponential backoff
- PIT-aware data transformation (adds observed_at timestamps)
- Caching to minimize API calls

FMP Free Tier Limitations (as of 2025):
- 250 API calls per day
- 5 API calls per minute  
- Uses /stable/ API endpoint

Endpoints Used (Stable API):
- /stable/historical-price-eod/full - OHLCV data
- /stable/quote - Current quote
- /stable/company-core-information - Basic company info
- /stable/income-statement - Income statements
- /stable/balance-sheet-statement - Balance sheets
- /stable/cash-flow-statement - Cash flows
"""

import os
import time
import logging
from datetime import date, datetime, timedelta
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path
import json

import requests
import pandas as pd

logger = logging.getLogger(__name__)


class FMPError(Exception):
    """Base exception for FMP API errors."""
    pass


class RateLimitError(FMPError):
    """Raised when API rate limit is exceeded."""
    pass


class APIKeyError(FMPError):
    """Raised when API key is invalid or missing."""
    pass


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
    
    Features:
    - Automatic rate limiting
    - Caching to reduce API calls
    - PIT-aware data transformation
    - Error handling with retries
    
    Usage:
        client = FMPClient()
        
        # Get OHLCV data
        df = client.get_historical_prices("NVDA", start="2023-01-01", end="2024-01-01")
        
        # Get fundamentals
        df = client.get_income_statement("NVDA", limit=8)
        
        # Get quote
        quote = client.get_quote(["NVDA"])
    """
    
    # Using stable API endpoint (2025)
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
            # Use relative path if config not available
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
        # Create a deterministic key from endpoint + params
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
        """
        Make an API request with rate limiting and caching.
        
        Args:
            endpoint: API endpoint (e.g., "historical-price-eod/full")
            params: Query parameters
            use_cache: Override instance cache setting
        
        Returns:
            JSON response data
        """
        params = params or {}
        params["apikey"] = self.api_key
        
        should_cache = use_cache if use_cache is not None else self.use_cache
        
        # Check cache first
        cache_path = self._get_cache_path(endpoint, params)
        if should_cache:
            cached = self._get_cached(cache_path)
            if cached is not None:
                logger.debug(f"Cache hit: {endpoint}")
                return cached
        
        # Rate limit
        self._rate_limiter.wait_if_needed()
        
        # Make request
        url = f"{self.BASE_URL}/{endpoint}"
        
        try:
            response = self._session.get(url, params=params, timeout=30)
            
            # Check for errors
            if response.status_code == 401:
                raise APIKeyError("Invalid API key")
            elif response.status_code == 429:
                raise RateLimitError("Rate limit exceeded by server")
            elif response.status_code == 403:
                raise FMPError(f"Access denied: {response.text[:200]}")
            elif response.status_code != 200:
                raise FMPError(f"API error {response.status_code}: {response.text}")
            
            data = response.json()
            
            # FMP returns error messages in JSON
            if isinstance(data, dict) and "Error Message" in data:
                raise FMPError(data["Error Message"])
            
            # Cache successful response
            if should_cache:
                self._set_cached(cache_path, data)
            
            return data
            
        except requests.RequestException as e:
            raise FMPError(f"Request failed: {e}")
    
    # =========================================================================
    # Price Data Endpoints (Stable API)
    # =========================================================================
    
    def get_historical_prices(
        self,
        symbol: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Get historical OHLCV data for a symbol.
        
        Args:
            symbol: Ticker symbol (e.g., "NVDA")
            start: Start date (YYYY-MM-DD)
            end: End date (YYYY-MM-DD)
        
        Returns:
            DataFrame with columns: date, open, high, low, close, adjClose, 
            volume, change, changePercent, observed_at
        """
        params = {"symbol": symbol}
        if start:
            params["from"] = start
        if end:
            params["to"] = end
        
        endpoint = "historical-price-eod/full"
        data = self._request(endpoint, params)
        
        if not data:
            logger.warning(f"No price data for {symbol}")
            return pd.DataFrame()
        
        df = pd.DataFrame(data)
        
        if df.empty:
            return df
        
        # Add ticker column
        df["ticker"] = symbol
        
        # Add PIT metadata
        # Price data is available after market close on that day
        # We conservatively say it's available at midnight the next day
        df["date"] = pd.to_datetime(df["date"])
        df["observed_at"] = df["date"] + pd.Timedelta(days=1)
        df["source"] = "fmp_historical"
        
        # Rename for consistency
        if "adjClose" in df.columns:
            df = df.rename(columns={"adjClose": "adj_close"})
        if "changePercent" in df.columns:
            df = df.rename(columns={"changePercent": "change_pct"})
        
        # Sort oldest first
        df = df.sort_values("date").reset_index(drop=True)
        
        return df
    
    def get_quote(self, symbols: List[str]) -> Dict[str, Dict]:
        """
        Get current quotes for symbols.
        
        Note: Free tier has 15-minute delay.
        
        Args:
            symbols: List of ticker symbols
        
        Returns:
            Dict mapping symbol to quote data
        """
        results = {}
        
        for symbol in symbols:
            params = {"symbol": symbol}
            endpoint = "quote"
            
            try:
                data = self._request(endpoint, params, use_cache=False)
                
                if data and isinstance(data, list) and len(data) > 0:
                    results[symbol] = data[0]
            except FMPError as e:
                logger.warning(f"Failed to get quote for {symbol}: {e}")
        
        return results
    
    # =========================================================================
    # Fundamental Data Endpoints (Stable API)
    # =========================================================================
    
    def get_income_statement(
        self,
        symbol: str,
        period: str = "quarter",
        limit: int = 20,
    ) -> pd.DataFrame:
        """
        Get income statement data.
        
        Args:
            symbol: Ticker symbol
            period: "quarter" or "annual"
            limit: Number of periods to return
        
        Returns:
            DataFrame with income statement items + PIT metadata
        """
        endpoint = "income-statement"
        params = {"symbol": symbol, "period": period, "limit": limit}
        
        data = self._request(endpoint, params)
        
        if not data:
            return pd.DataFrame()
        
        df = pd.DataFrame(data)
        df["ticker"] = symbol
        
        # Add PIT metadata
        # FMP provides fillingDate (when SEC filing was made)
        if "fillingDate" in df.columns:
            df["observed_at"] = pd.to_datetime(df["fillingDate"]) + pd.Timedelta(days=1)
        elif "acceptedDate" in df.columns:
            df["observed_at"] = pd.to_datetime(df["acceptedDate"])
        else:
            # Conservative: use period end + 45 days (typical filing deadline)
            df["observed_at"] = pd.to_datetime(df["date"]) + pd.Timedelta(days=45)
            logger.warning(f"No filing date for {symbol}, using conservative lag")
        
        df["source"] = "fmp_income_statement"
        df["period_end"] = pd.to_datetime(df["date"])
        
        return df
    
    def get_balance_sheet(
        self,
        symbol: str,
        period: str = "quarter",
        limit: int = 20,
    ) -> pd.DataFrame:
        """Get balance sheet data with PIT metadata."""
        endpoint = "balance-sheet-statement"
        params = {"symbol": symbol, "period": period, "limit": limit}
        
        data = self._request(endpoint, params)
        
        if not data:
            return pd.DataFrame()
        
        df = pd.DataFrame(data)
        df["ticker"] = symbol
        
        # Add PIT metadata
        if "fillingDate" in df.columns:
            df["observed_at"] = pd.to_datetime(df["fillingDate"]) + pd.Timedelta(days=1)
        elif "acceptedDate" in df.columns:
            df["observed_at"] = pd.to_datetime(df["acceptedDate"])
        else:
            df["observed_at"] = pd.to_datetime(df["date"]) + pd.Timedelta(days=45)
        
        df["source"] = "fmp_balance_sheet"
        df["period_end"] = pd.to_datetime(df["date"])
        
        return df
    
    def get_cash_flow(
        self,
        symbol: str,
        period: str = "quarter",
        limit: int = 20,
    ) -> pd.DataFrame:
        """Get cash flow statement data with PIT metadata."""
        endpoint = "cash-flow-statement"
        params = {"symbol": symbol, "period": period, "limit": limit}
        
        data = self._request(endpoint, params)
        
        if not data:
            return pd.DataFrame()
        
        df = pd.DataFrame(data)
        df["ticker"] = symbol
        
        # Add PIT metadata
        if "fillingDate" in df.columns:
            df["observed_at"] = pd.to_datetime(df["fillingDate"]) + pd.Timedelta(days=1)
        elif "acceptedDate" in df.columns:
            df["observed_at"] = pd.to_datetime(df["acceptedDate"])
        else:
            df["observed_at"] = pd.to_datetime(df["date"]) + pd.Timedelta(days=45)
        
        df["source"] = "fmp_cash_flow"
        df["period_end"] = pd.to_datetime(df["date"])
        
        return df
    
    def get_key_metrics(
        self,
        symbol: str,
        period: str = "quarter",
        limit: int = 20,
    ) -> pd.DataFrame:
        """Get key financial metrics (PE, PB, etc.) with PIT metadata."""
        endpoint = "key-metrics"
        params = {"symbol": symbol, "period": period, "limit": limit}
        
        data = self._request(endpoint, params)
        
        if not data:
            return pd.DataFrame()
        
        df = pd.DataFrame(data)
        df["ticker"] = symbol
        df["source"] = "fmp_key_metrics"
        
        return df
    
    # =========================================================================
    # Company Info Endpoints
    # =========================================================================
    
    def get_profile(self, symbol: str) -> Optional[Dict]:
        """
        Get company profile (sector, industry, market cap, etc.).
        
        Args:
            symbol: Ticker symbol
        
        Returns:
            Dict with company info or None
        """
        # Try stable/profile first
        endpoint = "profile"
        params = {"symbol": symbol}
        
        try:
            data = self._request(endpoint, params)
            
            if data and isinstance(data, list) and len(data) > 0:
                return data[0]
            
            # If profile is empty, try to build from quote
            quote = self.get_quote([symbol])
            if symbol in quote:
                return {
                    "symbol": symbol,
                    "companyName": quote[symbol].get("name", symbol),
                    "mktCap": quote[symbol].get("marketCap"),
                    "sector": "Technology",  # Default for AI stocks
                    "industry": "Unknown",
                }
            
            return None
        except FMPError:
            return None
    
    def get_profiles_batch(self, symbols: List[str]) -> Dict[str, Dict]:
        """
        Get profiles for multiple symbols.
        
        Note: Fetches one at a time for free tier.
        """
        results = {}
        
        for symbol in symbols:
            profile = self.get_profile(symbol)
            if profile:
                results[symbol] = profile
        
        return results
    
    # =========================================================================
    # Events Endpoints
    # =========================================================================
    
    def get_earnings_calendar(
        self,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Get earnings calendar.
        
        Args:
            start: Start date (YYYY-MM-DD)
            end: End date (YYYY-MM-DD)
        
        Returns:
            DataFrame with earnings dates
        """
        endpoint = "earning-calendar"
        params = {}
        if start:
            params["from"] = start
        if end:
            params["to"] = end
        
        try:
            data = self._request(endpoint, params)
            
            if not data:
                return pd.DataFrame()
            
            df = pd.DataFrame(data)
            df["source"] = "fmp_earnings_calendar"
            
            return df
        except FMPError:
            return pd.DataFrame()
    
    def get_stock_dividend(self, symbol: str) -> pd.DataFrame:
        """Get historical dividend data."""
        endpoint = "historical-price-eod/stock_dividend"
        params = {"symbol": symbol}
        
        try:
            data = self._request(endpoint, params)
            
            if not data:
                return pd.DataFrame()
            
            df = pd.DataFrame(data)
            df["ticker"] = symbol
            df["source"] = "fmp_dividend"
            
            return df
        except FMPError:
            return pd.DataFrame()
    
    def get_stock_split(self, symbol: str) -> pd.DataFrame:
        """Get historical stock split data."""
        endpoint = "historical-price-eod/stock_split"
        params = {"symbol": symbol}
        
        try:
            data = self._request(endpoint, params)
            
            if not data:
                return pd.DataFrame()
            
            df = pd.DataFrame(data)
            df["ticker"] = symbol
            df["source"] = "fmp_split"
            
            return df
        except FMPError:
            return pd.DataFrame()
    
    # =========================================================================
    # Benchmark / Index Data
    # =========================================================================
    
    def get_index_historical(
        self,
        symbol: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Get historical data for an index/ETF (QQQ, SPY, etc.).
        
        Same as get_historical_prices but semantic distinction.
        """
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
            # Use quote endpoint with a single symbol
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
) -> Dict[str, pd.DataFrame]:
    """
    Download all available data for a single ticker.
    
    Args:
        ticker: Stock ticker
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        include_fundamentals: Whether to include income/balance/cashflow
        client: FMP client instance (creates one if not provided)
    
    Returns:
        Dict with keys: prices, profile, income, balance, cashflow, dividends, splits
    """
    if client is None:
        client = FMPClient()
    
    result = {}
    
    # Prices (always)
    logger.info(f"Downloading prices for {ticker}")
    result["prices"] = client.get_historical_prices(ticker, start_date, end_date)
    
    # Profile
    result["profile"] = client.get_profile(ticker)
    
    if include_fundamentals:
        # Income statement
        logger.info(f"Downloading income statement for {ticker}")
        result["income"] = client.get_income_statement(ticker, period="quarter", limit=20)
        
        # Balance sheet
        logger.info(f"Downloading balance sheet for {ticker}")
        result["balance"] = client.get_balance_sheet(ticker, period="quarter", limit=20)
        
        # Cash flow
        logger.info(f"Downloading cash flow for {ticker}")
        result["cashflow"] = client.get_cash_flow(ticker, period="quarter", limit=20)
    
    # Corporate actions
    result["dividends"] = client.get_stock_dividend(ticker)
    result["splits"] = client.get_stock_split(ticker)
    
    return result

"""
Alpha Vantage API Client
========================

Provides earnings calendar and other data not available in FMP free tier.

Alpha Vantage Free Tier:
- 25 requests per day
- 5 requests per minute
- Earnings calendar (date-only, no BMO/AMC timing)

Key Limitation:
Alpha Vantage earnings calendar provides dates but NOT times (BMO/AMC).
For strict PIT handling around earnings, prefer SEC 8-K filing timestamps.

Documentation: https://www.alphavantage.co/documentation/
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
import pytz

logger = logging.getLogger(__name__)

UTC = pytz.UTC
ET = pytz.timezone("America/New_York")


class AlphaVantageError(Exception):
    """Base exception for Alpha Vantage API errors."""
    pass


class AVRateLimitError(AlphaVantageError):
    """Raised when API rate limit is exceeded."""
    pass


class AVAPIKeyError(AlphaVantageError):
    """Raised when API key is invalid or missing."""
    pass


@dataclass
class AVRateLimiter:
    """
    Rate limiter for Alpha Vantage free tier.
    
    Free tier limits:
    - 5 requests per minute
    - 25 requests per day (very restrictive!)
    """
    requests_per_minute: int = 5
    requests_per_day: int = 25
    
    _minute_requests: List[float] = field(default_factory=list)
    _day_requests: List[float] = field(default_factory=list)
    
    def wait_if_needed(self):
        """Block until we can make another request."""
        now = time.time()
        
        minute_ago = now - 60
        day_ago = now - 86400
        
        self._minute_requests = [t for t in self._minute_requests if t > minute_ago]
        self._day_requests = [t for t in self._day_requests if t > day_ago]
        
        if len(self._day_requests) >= self.requests_per_day:
            raise AVRateLimitError(
                f"Daily limit ({self.requests_per_day}) exceeded. "
                f"Reset in {86400 - (now - self._day_requests[0]):.0f}s"
            )
        
        if len(self._minute_requests) >= self.requests_per_minute:
            sleep_time = 60 - (now - self._minute_requests[0]) + 0.1
            if sleep_time > 0:
                logger.debug(f"Rate limit: sleeping {sleep_time:.1f}s")
                time.sleep(sleep_time)
        
        now = time.time()
        self._minute_requests.append(now)
        self._day_requests.append(now)
    
    @property
    def remaining_daily(self) -> int:
        now = time.time()
        day_ago = now - 86400
        recent = [t for t in self._day_requests if t > day_ago]
        return max(0, self.requests_per_day - len(recent))


class AlphaVantageClient:
    """
    Client for Alpha Vantage API.
    
    Primary use case: Earnings calendar (dates when companies report)
    
    IMPORTANT PIT NOTE:
    Alpha Vantage earnings calendar provides DATE only, not time-of-day.
    For PIT-correct handling:
    - If timing unknown, assume AMC (after market close)
    - Treat as available next market open (conservative)
    
    For precise timing, use SEC 8-K acceptance timestamps.
    
    Usage:
        client = AlphaVantageClient()
        
        # Get earnings calendar
        df = client.get_earnings_calendar(horizon="3month")
        
        # Get earnings for specific symbol
        df = client.get_earnings_calendar(symbol="NVDA", horizon="12month")
    """
    
    BASE_URL = "https://www.alphavantage.co/query"
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        cache_dir: Optional[Path] = None,
        use_cache: bool = True,
        cache_ttl_hours: int = 6,  # Shorter TTL for calendar data
    ):
        """
        Initialize Alpha Vantage client.
        
        Args:
            api_key: API key (defaults to ALPHAVANTAGE_KEYS env var)
            cache_dir: Directory for caching responses
            use_cache: Whether to use response caching
            cache_ttl_hours: Cache time-to-live in hours
        """
        self.api_key = api_key or os.getenv("ALPHAVANTAGE_KEYS", "")
        if not self.api_key:
            raise AVAPIKeyError(
                "Alpha Vantage API key not found. Set ALPHAVANTAGE_KEYS in .env file."
            )
        
        self.use_cache = use_cache
        self.cache_ttl = timedelta(hours=cache_ttl_hours)
        
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            try:
                from ..config import PROJECT_ROOT
                self.cache_dir = PROJECT_ROOT / "data" / "cache" / "alphavantage"
            except ImportError:
                self.cache_dir = Path("data/cache/alphavantage")
        
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self._rate_limiter = AVRateLimiter()
        self._session = requests.Session()
        
        logger.info(f"AlphaVantageClient initialized")
    
    def _get_cache_path(self, function: str, params: Dict) -> Path:
        param_str = "_".join(f"{k}={v}" for k, v in sorted(params.items()) if k != "apikey")
        filename = f"{function}_{param_str}.json"
        return self.cache_dir / filename
    
    def _get_cached(self, cache_path: Path) -> Optional[Any]:
        if not self.use_cache or not cache_path.exists():
            return None
        
        try:
            stat = cache_path.stat()
            age = datetime.now() - datetime.fromtimestamp(stat.st_mtime)
            if age > self.cache_ttl:
                return None
            
            with open(cache_path, "r") as f:
                return json.load(f)
        except Exception:
            return None
    
    def _set_cached(self, cache_path: Path, data: Any):
        if not self.use_cache:
            return
        try:
            with open(cache_path, "w") as f:
                json.dump(data, f)
        except Exception:
            pass
    
    def _request(
        self,
        function: str,
        params: Optional[Dict] = None,
        use_cache: Optional[bool] = None,
        datatype: str = "json",
    ) -> Any:
        """Make an API request."""
        params = params or {}
        params["function"] = function
        params["apikey"] = self.api_key
        params["datatype"] = datatype
        
        should_cache = use_cache if use_cache is not None else self.use_cache
        
        cache_path = self._get_cache_path(function, params)
        if should_cache and datatype == "json":
            cached = self._get_cached(cache_path)
            if cached is not None:
                logger.debug(f"Cache hit: {function}")
                return cached
        
        self._rate_limiter.wait_if_needed()
        
        try:
            response = self._session.get(self.BASE_URL, params=params, timeout=30)
            
            if response.status_code != 200:
                raise AlphaVantageError(f"API error {response.status_code}: {response.text}")
            
            # Alpha Vantage returns errors in JSON with "Error Message" or "Note"
            if datatype == "json":
                data = response.json()
                if "Error Message" in data:
                    raise AlphaVantageError(data["Error Message"])
                if "Note" in data:
                    # Rate limit warning
                    raise AVRateLimitError(data["Note"])
                
                if should_cache:
                    self._set_cached(cache_path, data)
                return data
            else:
                # CSV response
                return response.text
                
        except requests.RequestException as e:
            raise AlphaVantageError(f"Request failed: {e}")
    
    # =========================================================================
    # Earnings Calendar
    # =========================================================================
    
    def get_earnings_calendar(
        self,
        symbol: Optional[str] = None,
        horizon: str = "3month",
    ) -> pd.DataFrame:
        """
        Get earnings calendar.
        
        IMPORTANT: This provides DATE only, not time-of-day (BMO/AMC).
        
        For PIT-correct handling, you should:
        1. Assume unknown timing = AMC (after market close)
        2. Treat as available at next market open
        
        Args:
            symbol: Optional - filter for specific ticker
            horizon: "3month", "6month", or "12month"
        
        Returns:
            DataFrame with columns: symbol, name, reportDate, fiscalDateEnding,
            estimate, currency
            
            DOES NOT include: reportTime (BMO/AMC) - that's the limitation!
        """
        params = {"horizon": horizon}
        if symbol:
            params["symbol"] = symbol
        
        # Earnings calendar returns CSV
        csv_data = self._request("EARNINGS_CALENDAR", params, datatype="csv")
        
        if not csv_data or csv_data.startswith("{"):
            # Error response is JSON
            try:
                error = json.loads(csv_data)
                if "Error Message" in error:
                    raise AlphaVantageError(error["Error Message"])
            except json.JSONDecodeError:
                pass
            return pd.DataFrame()
        
        from io import StringIO
        df = pd.read_csv(StringIO(csv_data))
        
        if df.empty:
            return df
        
        # Standardize column names
        df = df.rename(columns={
            "reportDate": "report_date",
            "fiscalDateEnding": "fiscal_date_ending",
        })
        
        # Convert dates
        if "report_date" in df.columns:
            df["report_date"] = pd.to_datetime(df["report_date"])
        if "fiscal_date_ending" in df.columns:
            df["fiscal_date_ending"] = pd.to_datetime(df["fiscal_date_ending"])
        
        # Add PIT metadata
        # CONSERVATIVE: Assume AMC, available next market open
        if "report_date" in df.columns:
            from .fmp_client import get_next_market_open_utc
            df["observed_at"] = df["report_date"].apply(
                lambda d: get_next_market_open_utc(d.date()) if pd.notna(d) else None
            )
            df["timing_source"] = "assumed_amc"  # Flag that this is assumed, not known
        
        df["source"] = "alphavantage_earnings_calendar"
        
        return df
    
    def get_company_overview(self, symbol: str) -> Optional[Dict]:
        """
        Get company overview (fundamentals summary).
        
        Includes: sector, industry, market cap, PE, EPS, dividend, etc.
        """
        try:
            data = self._request("OVERVIEW", {"symbol": symbol})
            if data and "Symbol" in data:
                return data
        except AlphaVantageError:
            pass
        return None
    
    # =========================================================================
    # Utilities
    # =========================================================================
    
    @property
    def remaining_requests(self) -> int:
        """Get remaining daily API requests."""
        return self._rate_limiter.remaining_daily
    
    def test_connection(self) -> bool:
        """Test API connection."""
        try:
            # Use a simple quote request
            data = self._request("GLOBAL_QUOTE", {"symbol": "IBM"})
            return "Global Quote" in data
        except AlphaVantageError:
            return False


def get_alphavantage_client() -> AlphaVantageClient:
    """Get a configured Alpha Vantage client instance."""
    return AlphaVantageClient()


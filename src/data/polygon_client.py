"""
Polygon.io Client - Symbol Master for Survivorship-Safe Universe
=================================================================

Provides point-in-time universe membership queries using Polygon's reference 
tickers API. This is the AUTHORITATIVE source for "which equities existed 
at time T" (including later-delisted names).

KEY CONCEPTS:

1. Universe-as-of-T: Polygon's reference tickers endpoint supports:
   - `date=`: retrieve tickers active on that date
   - `active=`: true/false for current status (false = includes delisted)
   
2. Symbol Master Role:
   - Polygon determines WHO is in the universe at each date
   - FMP provides OHLCV/fundamentals for those tickers
   - This separation ensures survivorship-bias-free universe construction

3. PIT Rules:
   - Ticker data returned is as-of the query date
   - Delisted tickers included via active=false
   - CIK/CUSIP used as stable_id where available

DATA SOURCES:
- Polygon Reference Tickers: /v3/reference/tickers
- Polygon Ticker Details: /v3/reference/tickers/{ticker}
- Polygon Ticker Events: /vX/reference/tickers/{ticker}/events

FREE TIER LIMITATIONS:
- 5 API calls per minute
- Basic historical data access (varies by endpoint)
- Need to test historical date queries

API KEYS:
- POLYGON_KEYS: API key for REST endpoints
- POLYGON_SECRETKEYS: Secret key for S3/flat files
"""

import os
import logging
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import List, Dict, Any, Optional, Set, Tuple
from pathlib import Path
from enum import Enum
import json
import hashlib

import requests
import pandas as pd
import pytz

logger = logging.getLogger(__name__)

UTC = pytz.UTC
ET = pytz.timezone("America/New_York")


class PolygonError(Exception):
    """Base exception for Polygon API errors."""
    pass


class RateLimitError(PolygonError):
    """Rate limit exceeded."""
    pass


class APIKeyError(PolygonError):
    """API key missing or invalid."""
    pass


class PlanLimitationError(PolygonError):
    """Feature not available on current plan."""
    pass


@dataclass
class TickerInfo:
    """
    Information about a single ticker from Polygon.
    
    Used for universe membership determination.
    """
    ticker: str
    name: str
    market: str  # stocks, crypto, fx, otc
    locale: str  # us, global
    primary_exchange: str
    type: str  # CS (common stock), ETF, ADR, etc.
    active: bool
    currency_name: str
    cik: Optional[str] = None  # SEC CIK - use as stable_id
    composite_figi: Optional[str] = None
    share_class_figi: Optional[str] = None
    last_updated_utc: Optional[datetime] = None
    delisted_utc: Optional[datetime] = None
    
    # Derived fields
    sic_code: Optional[str] = None
    sic_description: Optional[str] = None
    
    @property
    def stable_id(self) -> str:
        """
        Get stable identifier for this ticker.
        Priority: CIK > composite_figi > ticker
        """
        if self.cik:
            return f"CIK:{self.cik}"
        if self.composite_figi:
            return f"FIGI:{self.composite_figi}"
        return f"TICKER:{self.ticker}"
    
    @property
    def is_common_stock(self) -> bool:
        """Check if this is a common stock (CS)."""
        return self.type == "CS"
    
    @property
    def is_us_equity(self) -> bool:
        """Check if this is a US equity (common stock or similar)."""
        return (
            self.market == "stocks" and 
            self.locale == "us" and
            self.type in ("CS", "ADRC", "PFD")  # Common stock, ADR, Preferred
        )


@dataclass
class TickerEvent:
    """A ticker-related event (name change, ticker change, delisting, etc.)."""
    ticker: str
    event_type: str
    event_date: date
    old_ticker: Optional[str] = None
    new_ticker: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)


class RateLimiter:
    """Simple rate limiter for API calls."""
    
    def __init__(self, max_per_minute: int = 5):
        self.max_per_minute = max_per_minute
        self.calls: List[float] = []
    
    def wait_if_needed(self):
        """Wait if we've exceeded the rate limit."""
        now = time.time()
        minute_ago = now - 60
        
        # Remove old calls
        self.calls = [t for t in self.calls if t > minute_ago]
        
        if len(self.calls) >= self.max_per_minute:
            sleep_time = self.calls[0] - minute_ago + 0.1
            if sleep_time > 0:
                logger.debug(f"Rate limiting: sleeping {sleep_time:.1f}s")
                time.sleep(sleep_time)
        
        self.calls.append(time.time())


class PolygonClient:
    """
    Polygon.io API client for symbol master queries.
    
    Primary use: determining universe membership at historical dates.
    
    Usage:
        client = PolygonClient()
        
        # Get all US stocks active on a specific date
        tickers = client.get_tickers_asof(date(2023, 1, 15))
        
        # Check if ticker was active
        is_active = client.was_active("NVDA", date(2023, 1, 15))
        
        # Get delisted tickers
        delisted = client.get_delisted_tickers(since=date(2020, 1, 1))
    """
    
    BASE_URL = "https://api.polygon.io"
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        cache_dir: Optional[Path] = None,
        rate_limit: int = 5,  # Free tier: 5/min
    ):
        """
        Initialize Polygon client.
        
        Args:
            api_key: Polygon API key. If None, reads from POLYGON_KEYS env var.
            cache_dir: Directory for caching responses.
            rate_limit: Max API calls per minute.
        """
        self.api_key = api_key or os.getenv("POLYGON_KEYS", "")
        # Strip any quotes that might be in the env value
        self.api_key = self.api_key.strip().strip('"').strip("'")
        
        if not self.api_key:
            raise APIKeyError(
                "Polygon API key not found. Set POLYGON_KEYS in .env file."
            )
        
        self.cache_dir = cache_dir or Path("data/cache/polygon")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.rate_limiter = RateLimiter(rate_limit)
        self._session = requests.Session()
        
        logger.info("PolygonClient initialized")
    
    def _get_cache_key(self, endpoint: str, params: Dict) -> str:
        """Generate cache key for request."""
        param_str = json.dumps(params, sort_keys=True)
        key = f"{endpoint}:{param_str}"
        return hashlib.md5(key.encode()).hexdigest()
    
    def _get_cached(self, cache_key: str, max_age_days: int = 1) -> Optional[Dict]:
        """Get cached response if valid."""
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        if cache_file.exists():
            # Check age
            age = time.time() - cache_file.stat().st_mtime
            if age < max_age_days * 86400:
                try:
                    with open(cache_file) as f:
                        return json.load(f)
                except Exception:
                    pass
        return None
    
    def _save_cache(self, cache_key: str, data: Dict):
        """Save response to cache."""
        cache_file = self.cache_dir / f"{cache_key}.json"
        try:
            with open(cache_file, "w") as f:
                json.dump(data, f)
        except Exception as e:
            logger.warning(f"Failed to cache response: {e}")
    
    def _request(
        self, 
        endpoint: str, 
        params: Optional[Dict] = None,
        use_cache: bool = True,
        cache_days: int = 1,
    ) -> Dict:
        """
        Make authenticated request to Polygon API.
        
        Args:
            endpoint: API endpoint (e.g., "/v3/reference/tickers")
            params: Query parameters
            use_cache: Whether to use cached responses
            cache_days: Max age of cache in days
            
        Returns:
            API response as dict
        """
        params = params or {}
        params["apiKey"] = self.api_key
        
        # Check cache first
        if use_cache:
            cache_key = self._get_cache_key(endpoint, {k: v for k, v in params.items() if k != "apiKey"})
            cached = self._get_cached(cache_key, cache_days)
            if cached:
                logger.debug(f"Cache hit: {endpoint}")
                return cached
        
        # Rate limit
        self.rate_limiter.wait_if_needed()
        
        url = f"{self.BASE_URL}{endpoint}"
        
        try:
            response = self._session.get(url, params=params, timeout=30)
            
            if response.status_code == 429:
                raise RateLimitError("Rate limit exceeded")
            
            if response.status_code == 403:
                data = response.json()
                if "not authorized" in data.get("message", "").lower():
                    raise PlanLimitationError(
                        f"Feature requires higher plan tier: {data.get('message')}"
                    )
                raise APIKeyError(f"API key invalid or unauthorized: {data.get('message')}")
            
            response.raise_for_status()
            data = response.json()
            
            # Check for API-level errors
            if data.get("status") == "ERROR":
                raise PolygonError(f"API error: {data.get('message', 'Unknown error')}")
            
            # Cache successful response
            if use_cache:
                self._save_cache(cache_key, data)
            
            return data
            
        except requests.exceptions.RequestException as e:
            raise PolygonError(f"Request failed: {e}")
    
    def _paginate_results(
        self,
        endpoint: str,
        params: Dict,
        max_results: int = 10000,
    ) -> List[Dict]:
        """
        Paginate through API results.
        
        Args:
            endpoint: API endpoint
            params: Query parameters
            max_results: Maximum results to return
            
        Returns:
            List of all results
        """
        results = []
        cursor = None
        
        while len(results) < max_results:
            req_params = params.copy()
            if cursor:
                req_params["cursor"] = cursor
            
            data = self._request(endpoint, req_params, use_cache=False)
            
            batch = data.get("results", [])
            results.extend(batch)
            
            # Check for next page
            next_url = data.get("next_url")
            if not next_url or not batch:
                break
            
            # Extract cursor from next_url
            if "cursor=" in next_url:
                cursor = next_url.split("cursor=")[1].split("&")[0]
            else:
                break
        
        return results[:max_results]
    
    def get_tickers_asof(
        self,
        asof_date: Optional[date] = None,
        market: str = "stocks",
        locale: str = "us",
        ticker_type: str = "CS",  # Common Stock
        active_only: bool = False,
        limit: int = 1000,
    ) -> List[TickerInfo]:
        """
        Get tickers that were active on a specific date.
        
        This is the core method for survivorship-safe universe construction.
        
        Args:
            asof_date: Date to query (None = current)
            market: Market type (stocks, crypto, fx, otc)
            locale: Market locale (us, global)
            ticker_type: Ticker type (CS, ETF, ADR, etc.)
            active_only: If False, includes currently delisted tickers
            limit: Max results per page
            
        Returns:
            List of TickerInfo objects
        """
        params = {
            "market": market,
            "locale": locale,
            "type": ticker_type,
            "limit": limit,
            "order": "asc",
            "sort": "ticker",
        }
        
        if asof_date:
            params["date"] = asof_date.isoformat()
        
        if not active_only:
            params["active"] = "false"  # Include delisted
        
        try:
            results = self._paginate_results("/v3/reference/tickers", params)
        except PlanLimitationError as e:
            logger.warning(f"Plan limitation on tickers endpoint: {e}")
            # Fallback to simpler query
            params.pop("date", None)
            results = self._paginate_results("/v3/reference/tickers", params)
        
        tickers = []
        for item in results:
            try:
                ticker_info = TickerInfo(
                    ticker=item.get("ticker", ""),
                    name=item.get("name", ""),
                    market=item.get("market", ""),
                    locale=item.get("locale", ""),
                    primary_exchange=item.get("primary_exchange", ""),
                    type=item.get("type", ""),
                    active=item.get("active", True),
                    currency_name=item.get("currency_name", ""),
                    cik=item.get("cik"),
                    composite_figi=item.get("composite_figi"),
                    share_class_figi=item.get("share_class_figi"),
                    last_updated_utc=pd.to_datetime(item.get("last_updated_utc"), utc=True) if item.get("last_updated_utc") else None,
                    delisted_utc=pd.to_datetime(item.get("delisted_utc"), utc=True) if item.get("delisted_utc") else None,
                )
                tickers.append(ticker_info)
            except Exception as e:
                logger.warning(f"Failed to parse ticker: {item.get('ticker')}: {e}")
        
        logger.info(f"Got {len(tickers)} tickers as of {asof_date or 'current'}")
        return tickers
    
    def get_ticker_details(self, ticker: str, asof_date: Optional[date] = None) -> Optional[TickerInfo]:
        """
        Get detailed information for a specific ticker.
        
        Args:
            ticker: Ticker symbol
            asof_date: Date for historical lookup (if supported)
            
        Returns:
            TickerInfo or None if not found
        """
        params = {}
        if asof_date:
            params["date"] = asof_date.isoformat()
        
        try:
            data = self._request(f"/v3/reference/tickers/{ticker}", params)
            
            item = data.get("results", {})
            if not item:
                return None
            
            return TickerInfo(
                ticker=item.get("ticker", ticker),
                name=item.get("name", ""),
                market=item.get("market", "stocks"),
                locale=item.get("locale", "us"),
                primary_exchange=item.get("primary_exchange", ""),
                type=item.get("type", ""),
                active=item.get("active", True),
                currency_name=item.get("currency_name", "usd"),
                cik=item.get("cik"),
                composite_figi=item.get("composite_figi"),
                share_class_figi=item.get("share_class_figi"),
                sic_code=item.get("sic_code"),
                sic_description=item.get("sic_description"),
                last_updated_utc=pd.to_datetime(item.get("last_updated_utc"), utc=True) if item.get("last_updated_utc") else None,
                delisted_utc=pd.to_datetime(item.get("delisted_utc"), utc=True) if item.get("delisted_utc") else None,
            )
        except Exception as e:
            logger.warning(f"Failed to get ticker details for {ticker}: {e}")
            return None
    
    def was_ticker_active(self, ticker: str, asof_date: date) -> bool:
        """
        Check if a ticker was active (tradable) on a specific date.
        
        Args:
            ticker: Ticker symbol
            asof_date: Date to check
            
        Returns:
            True if ticker was active on that date
        """
        details = self.get_ticker_details(ticker, asof_date)
        if not details:
            return False
        
        # If delisted_utc exists and is before asof_date, ticker was not active
        if details.delisted_utc:
            delisted_date = details.delisted_utc.date()
            if delisted_date <= asof_date:
                return False
        
        return details.active or details.delisted_utc is not None
    
    def get_delisted_tickers(
        self,
        since: Optional[date] = None,
        until: Optional[date] = None,
    ) -> List[TickerInfo]:
        """
        Get tickers that have been delisted.
        
        Args:
            since: Start date for delisting filter
            until: End date for delisting filter
            
        Returns:
            List of delisted TickerInfo objects
        """
        params = {
            "market": "stocks",
            "locale": "us",
            "active": "false",
            "limit": 1000,
        }
        
        try:
            results = self._paginate_results("/v3/reference/tickers", params)
        except Exception as e:
            logger.error(f"Failed to get delisted tickers: {e}")
            return []
        
        delisted = []
        for item in results:
            delisted_utc = pd.to_datetime(item.get("delisted_utc"), utc=True) if item.get("delisted_utc") else None
            
            # Filter by date range if specified
            if delisted_utc:
                delisted_date = delisted_utc.date()
                if since and delisted_date < since:
                    continue
                if until and delisted_date > until:
                    continue
            
            try:
                ticker_info = TickerInfo(
                    ticker=item.get("ticker", ""),
                    name=item.get("name", ""),
                    market=item.get("market", ""),
                    locale=item.get("locale", ""),
                    primary_exchange=item.get("primary_exchange", ""),
                    type=item.get("type", ""),
                    active=False,
                    currency_name=item.get("currency_name", ""),
                    cik=item.get("cik"),
                    composite_figi=item.get("composite_figi"),
                    delisted_utc=delisted_utc,
                )
                delisted.append(ticker_info)
            except Exception as e:
                logger.warning(f"Failed to parse delisted ticker: {e}")
        
        logger.info(f"Got {len(delisted)} delisted tickers")
        return delisted
    
    def get_ticker_events(
        self,
        ticker: str,
        event_types: Optional[List[str]] = None,
    ) -> List[TickerEvent]:
        """
        Get events for a ticker (name changes, ticker changes, etc.).
        
        Note: This endpoint may require a higher tier plan.
        
        Args:
            ticker: Ticker symbol
            event_types: Filter by event types
            
        Returns:
            List of TickerEvent objects
        """
        try:
            data = self._request(f"/vX/reference/tickers/{ticker}/events")
            
            events = []
            for item in data.get("results", {}).get("events", []):
                event = TickerEvent(
                    ticker=ticker,
                    event_type=item.get("type", ""),
                    event_date=date.fromisoformat(item.get("date", "1970-01-01")),
                    details=item,
                )
                
                if event_types and event.event_type not in event_types:
                    continue
                
                events.append(event)
            
            return events
            
        except PlanLimitationError:
            logger.warning(f"Ticker events endpoint requires higher plan tier")
            return []
        except Exception as e:
            logger.warning(f"Failed to get ticker events for {ticker}: {e}")
            return []
    
    def test_api_access(self) -> Dict[str, Any]:
        """
        Test API access and determine available features.
        
        Returns:
            Dict with test results for various endpoints
        """
        results = {
            "api_key_valid": False,
            "basic_tickers": False,
            "historical_date_query": False,
            "delisted_tickers": False,
            "ticker_details": False,
            "ticker_events": False,
            "errors": [],
        }
        
        # Test 1: Basic tickers query
        try:
            tickers = self.get_tickers_asof(limit=5)
            results["api_key_valid"] = True
            results["basic_tickers"] = len(tickers) > 0
        except APIKeyError as e:
            results["errors"].append(f"API key error: {e}")
            return results
        except Exception as e:
            results["errors"].append(f"Basic tickers failed: {e}")
        
        # Test 2: Historical date query
        try:
            test_date = date.today() - timedelta(days=365)
            tickers = self.get_tickers_asof(asof_date=test_date, limit=5)
            results["historical_date_query"] = len(tickers) > 0
        except PlanLimitationError:
            results["errors"].append("Historical date query requires higher plan")
        except Exception as e:
            results["errors"].append(f"Historical query failed: {e}")
        
        # Test 3: Delisted tickers
        try:
            delisted = self.get_delisted_tickers()
            results["delisted_tickers"] = len(delisted) > 0
        except Exception as e:
            results["errors"].append(f"Delisted tickers failed: {e}")
        
        # Test 4: Ticker details
        try:
            details = self.get_ticker_details("AAPL")
            results["ticker_details"] = details is not None
        except Exception as e:
            results["errors"].append(f"Ticker details failed: {e}")
        
        # Test 5: Ticker events
        try:
            events = self.get_ticker_events("META")
            results["ticker_events"] = True  # Endpoint accessible even if no events
        except PlanLimitationError:
            results["errors"].append("Ticker events requires higher plan")
        except Exception as e:
            results["errors"].append(f"Ticker events failed: {e}")
        
        return results
    
    def get_universe_candidates(
        self,
        asof_date: date,
        include_delisted: bool = True,
    ) -> Tuple[List[TickerInfo], str]:
        """
        Get candidate tickers for universe construction.
        
        Returns both the tickers and the survivorship status based on
        data availability.
        
        Args:
            asof_date: Date to build universe for
            include_delisted: Whether to include tickers that are now delisted
            
        Returns:
            Tuple of (list of TickerInfo, survivorship_status)
        """
        from src.pipelines.universe_pipeline import SurvivorshipStatus
        
        status = SurvivorshipStatus.UNKNOWN
        
        try:
            # Try historical date query first
            tickers = self.get_tickers_asof(
                asof_date=asof_date,
                active_only=not include_delisted,
            )
            
            if tickers:
                # If we can query historical dates with delisted, we're FULL
                if include_delisted:
                    status = SurvivorshipStatus.FULL
                else:
                    status = SurvivorshipStatus.PARTIAL
                
                return tickers, status
                
        except PlanLimitationError:
            logger.warning("Historical date query requires higher plan, falling back")
        
        # Fallback: get current tickers (includes active=false for delisted)
        try:
            tickers = self.get_tickers_asof(active_only=not include_delisted)
            status = SurvivorshipStatus.PARTIAL
            return tickers, status
        except Exception as e:
            logger.error(f"Failed to get universe candidates: {e}")
            return [], SurvivorshipStatus.UNKNOWN
    
    def close(self):
        """Close the client session."""
        self._session.close()


def test_polygon_access():
    """
    Test Polygon API access and print available features.
    
    Run this to determine if free tier is sufficient.
    """
    print("=" * 60)
    print("POLYGON API ACCESS TEST")
    print("=" * 60)
    
    try:
        client = PolygonClient()
    except APIKeyError as e:
        print(f"\n❌ API KEY ERROR: {e}")
        print("\nTo fix: Add POLYGON_KEYS=your_key to .env file")
        return False
    
    results = client.test_api_access()
    
    print("\nTest Results:")
    print("-" * 40)
    
    # Print results
    for key, value in results.items():
        if key == "errors":
            continue
        status = "✅" if value else "❌"
        print(f"  {status} {key}: {value}")
    
    if results["errors"]:
        print("\nErrors/Warnings:")
        for err in results["errors"]:
            print(f"  ⚠️  {err}")
    
    # Determine survivorship capability
    print("\n" + "=" * 60)
    print("SURVIVORSHIP CAPABILITY ASSESSMENT")
    print("=" * 60)
    
    if results["historical_date_query"] and results["delisted_tickers"]:
        print("\n✅ FULL survivorship capability available")
        print("   - Can query tickers active on historical dates")
        print("   - Can access delisted ticker information")
        print("   → survivorship_status = FULL is achievable")
    elif results["delisted_tickers"]:
        print("\n⚠️  PARTIAL survivorship capability")
        print("   - Can access delisted tickers")
        print("   - Historical date queries limited")
        print("   → survivorship_status = PARTIAL")
        print("\n   To upgrade to FULL: Consider Polygon paid tier")
    else:
        print("\n❌ LIMITED survivorship capability")
        print("   - Cannot reliably access delisted tickers")
        print("   → survivorship_status = UNKNOWN recommended")
        print("\n   Consider: Polygon paid tier or Sharadar (Nasdaq Data Link)")
    
    client.close()
    
    return results["api_key_valid"] and results["basic_tickers"]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_polygon_access()


"""
SEC EDGAR API Client
====================

Provides PIT-accurate filing timestamps from SEC EDGAR.

This is the GOLD STANDARD for Point-in-Time correctness because:
1. SEC provides exact acceptance timestamps (to the second)
2. 8-K filings often accompany earnings releases
3. Timestamps are auditable and legally significant

No API key required, but MUST include User-Agent header.

Documentation: https://www.sec.gov/search-filings/edgar-application-programming-interfaces

Key Endpoints:
- /submissions/CIK{cik}.json - Company filing history
- /api/xbrl/companyfacts/CIK{cik}.json - XBRL facts (fundamentals)
"""

import os
import time
import logging
from datetime import date, datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import json

import requests
import pandas as pd
import pytz

logger = logging.getLogger(__name__)

UTC = pytz.UTC
ET = pytz.timezone("America/New_York")


class SECError(Exception):
    """Base exception for SEC EDGAR API errors."""
    pass


class SECRateLimitError(SECError):
    """Raised when SEC rate limit is hit."""
    pass


# Ticker to CIK mapping (will be populated from SEC)
_TICKER_CIK_CACHE: Dict[str, str] = {}


@dataclass
class SECRateLimiter:
    """
    Rate limiter for SEC EDGAR.
    
    SEC guidelines:
    - Max 10 requests per second
    - Recommend < 10 requests per second to avoid blocks
    """
    requests_per_second: float = 8  # Conservative
    
    _last_request: float = 0.0
    
    def wait_if_needed(self):
        """Ensure minimum delay between requests."""
        now = time.time()
        min_interval = 1.0 / self.requests_per_second
        elapsed = now - self._last_request
        
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        
        self._last_request = time.time()


class SECEdgarClient:
    """
    Client for SEC EDGAR API.
    
    Provides:
    1. Filing timestamps (acceptanceDateTime) - gold standard for PIT
    2. Company facts (XBRL data)
    3. Filing history
    
    NO API KEY REQUIRED - but must include User-Agent header.
    
    Usage:
        client = SECEdgarClient("your_app@email.com")
        
        # Get filing history
        filings = client.get_filings("NVDA", form_types=["10-K", "10-Q", "8-K"])
        
        # Get observed_at for earnings (from 8-K filings)
        observed_at = client.get_earnings_observed_at("NVDA", period_end=date(2024, 1, 31))
    """
    
    BASE_URL = "https://data.sec.gov"
    
    def __init__(
        self,
        contact_email: Optional[str] = None,
        app_name: str = "AIStockForecast",
        cache_dir: Optional[Path] = None,
        use_cache: bool = True,
        cache_ttl_hours: int = 24,
    ):
        """
        Initialize SEC EDGAR client.
        
        Args:
            contact_email: Email for User-Agent (REQUIRED by SEC)
            app_name: Application name for User-Agent
            cache_dir: Directory for caching responses
            use_cache: Whether to use response caching
            cache_ttl_hours: Cache time-to-live
        """
        self.contact_email = contact_email or os.getenv("SEC_CONTACT_EMAIL", "")
        if not self.contact_email:
            logger.warning(
                "No contact email provided for SEC User-Agent. "
                "Set SEC_CONTACT_EMAIL in .env or pass contact_email parameter. "
                "SEC may block requests without proper User-Agent."
            )
            self.contact_email = "anonymous@example.com"
        
        self.app_name = app_name
        self.user_agent = f"{app_name}/1.0 ({self.contact_email})"
        
        self.use_cache = use_cache
        self.cache_ttl = timedelta(hours=cache_ttl_hours)
        
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            try:
                from ..config import PROJECT_ROOT
                self.cache_dir = PROJECT_ROOT / "data" / "cache" / "sec"
            except ImportError:
                self.cache_dir = Path("data/cache/sec")
        
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self._rate_limiter = SECRateLimiter()
        self._session = requests.Session()
        # Don't set Host header globally - it varies by endpoint
        self._session.headers.update({
            "User-Agent": self.user_agent,
            "Accept-Encoding": "gzip, deflate",
        })
        
        logger.info(f"SECEdgarClient initialized (User-Agent: {self.user_agent})")
    
    def _get_cache_path(self, endpoint: str) -> Path:
        safe_endpoint = endpoint.replace("/", "_").replace(".", "_")
        return self.cache_dir / f"{safe_endpoint}.json"
    
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
    
    def _request(self, endpoint: str, use_cache: Optional[bool] = None) -> Any:
        """Make an API request to SEC EDGAR."""
        should_cache = use_cache if use_cache is not None else self.use_cache
        
        cache_path = self._get_cache_path(endpoint)
        if should_cache:
            cached = self._get_cached(cache_path)
            if cached is not None:
                logger.debug(f"Cache hit: {endpoint}")
                return cached
        
        self._rate_limiter.wait_if_needed()
        
        url = f"{self.BASE_URL}/{endpoint}"
        
        try:
            response = self._session.get(url, timeout=30)
            
            if response.status_code == 403:
                raise SECRateLimitError(
                    "SEC blocked request. Check User-Agent header and rate limits."
                )
            elif response.status_code == 404:
                return None
            elif response.status_code != 200:
                raise SECError(f"SEC API error {response.status_code}: {response.text[:200]}")
            
            data = response.json()
            
            if should_cache:
                self._set_cached(cache_path, data)
            
            return data
            
        except requests.RequestException as e:
            raise SECError(f"Request failed: {e}")
    
    # =========================================================================
    # CIK Lookup
    # =========================================================================
    
    def get_cik(self, ticker: str) -> Optional[str]:
        """
        Get CIK for a ticker symbol.
        
        Uses SEC's ticker-to-CIK mapping.
        """
        global _TICKER_CIK_CACHE
        
        ticker = ticker.upper()
        
        if ticker in _TICKER_CIK_CACHE:
            return _TICKER_CIK_CACHE[ticker]
        
        # Load the full mapping from www.sec.gov (different from data.sec.gov)
        if not _TICKER_CIK_CACHE:
            try:
                # This endpoint is on www.sec.gov, not data.sec.gov
                url = "https://www.sec.gov/files/company_tickers.json"
                response = self._session.get(url, timeout=30)
                response.raise_for_status()
                data = response.json()
                
                if data:
                    for entry in data.values():
                        t = entry.get("ticker", "").upper()
                        cik = str(entry.get("cik_str", ""))
                        if t and cik:
                            _TICKER_CIK_CACHE[t] = cik.zfill(10)
                    logger.info(f"Loaded {len(_TICKER_CIK_CACHE)} ticker-CIK mappings")
            except Exception as e:
                logger.warning(f"Failed to load CIK mapping: {e}")
        
        return _TICKER_CIK_CACHE.get(ticker)
    
    # =========================================================================
    # Company Submissions (Filing History)
    # =========================================================================
    
    def get_submissions(self, ticker: str) -> Optional[Dict]:
        """
        Get company filing history (submissions).
        
        Returns filing metadata including:
        - accessionNumber
        - filingDate
        - acceptanceDateTime (GOLD STANDARD for PIT!)
        - form (10-K, 10-Q, 8-K, etc.)
        
        Args:
            ticker: Stock ticker
        
        Returns:
            Dict with company info and filings
        """
        cik = self.get_cik(ticker)
        if not cik:
            logger.warning(f"No CIK found for {ticker}")
            return None
        
        endpoint = f"submissions/CIK{cik}.json"
        return self._request(endpoint)
    
    def get_filings(
        self,
        ticker: str,
        form_types: Optional[List[str]] = None,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> pd.DataFrame:
        """
        Get filing history with acceptance timestamps.
        
        This is the GOLD STANDARD for PIT timestamps because:
        - acceptanceDateTime is when SEC accepted the filing
        - This is the exact moment the filing became public
        
        Args:
            ticker: Stock ticker
            form_types: Filter by form type (e.g., ["10-K", "10-Q", "8-K"])
            start_date: Filter filings after this date
            end_date: Filter filings before this date
        
        Returns:
            DataFrame with columns: ticker, form, filingDate, acceptanceDateTime,
            primaryDocument, accessionNumber, observed_at
        """
        data = self.get_submissions(ticker)
        if not data or "filings" not in data:
            return pd.DataFrame()
        
        recent = data["filings"].get("recent", {})
        
        if not recent:
            return pd.DataFrame()
        
        # Build DataFrame from recent filings
        df = pd.DataFrame({
            "form": recent.get("form", []),
            "filingDate": recent.get("filingDate", []),
            "acceptanceDateTime": recent.get("acceptanceDateTime", []),
            "primaryDocument": recent.get("primaryDocument", []),
            "accessionNumber": recent.get("accessionNumber", []),
        })
        
        df["ticker"] = ticker
        
        # Parse dates
        df["filingDate"] = pd.to_datetime(df["filingDate"])
        
        # Parse acceptance datetime - THIS IS THE GOLD STANDARD FOR PIT
        # Format: "2024-01-31T16:05:23.000Z"
        df["acceptanceDateTime"] = pd.to_datetime(df["acceptanceDateTime"], utc=True)
        
        # observed_at = acceptanceDateTime (exact public availability)
        df["observed_at"] = df["acceptanceDateTime"]
        
        # Filter by form type
        if form_types:
            df = df[df["form"].isin(form_types)]
        
        # Filter by date range
        if start_date:
            df = df[df["filingDate"] >= pd.Timestamp(start_date)]
        if end_date:
            df = df[df["filingDate"] <= pd.Timestamp(end_date)]
        
        df["source"] = "sec_edgar"
        
        return df.reset_index(drop=True)
    
    def get_earnings_observed_at(
        self,
        ticker: str,
        period_end: date,
        tolerance_days: int = 60,
    ) -> Optional[datetime]:
        """
        Get the observed_at timestamp for an earnings release.
        
        This looks for 8-K filings (which typically accompany earnings)
        or 10-Q/10-K filings for the given period.
        
        Args:
            ticker: Stock ticker
            period_end: End of fiscal period (e.g., 2024-03-31 for Q1 2024)
            tolerance_days: Days after period_end to search
        
        Returns:
            UTC datetime when earnings became available, or None
        """
        # Get filings around the expected date
        start_date = period_end
        end_date = period_end + timedelta(days=tolerance_days)
        
        # Look for 8-K (earnings release) or 10-Q/10-K
        filings = self.get_filings(
            ticker,
            form_types=["8-K", "10-Q", "10-K"],
            start_date=start_date,
            end_date=end_date,
        )
        
        if filings.empty:
            return None
        
        # Prefer 8-K (usually filed same day as earnings)
        eight_k = filings[filings["form"] == "8-K"]
        if not eight_k.empty:
            # Return the earliest 8-K after period_end
            return eight_k["observed_at"].min()
        
        # Fall back to 10-Q/10-K
        return filings["observed_at"].min()
    
    # =========================================================================
    # Company Facts (XBRL)
    # =========================================================================
    
    def get_company_facts(self, ticker: str) -> Optional[Dict]:
        """
        Get XBRL company facts (fundamentals data).
        
        This provides historical fundamental data with filing dates.
        
        Args:
            ticker: Stock ticker
        
        Returns:
            Dict with XBRL facts
        """
        cik = self.get_cik(ticker)
        if not cik:
            return None
        
        endpoint = f"api/xbrl/companyfacts/CIK{cik}.json"
        return self._request(endpoint)
    
    def get_fundamental_with_pit(
        self,
        ticker: str,
        concept: str,
        taxonomy: str = "us-gaap",
    ) -> pd.DataFrame:
        """
        Get fundamental data with PIT-accurate timestamps.
        
        Args:
            ticker: Stock ticker
            concept: XBRL concept (e.g., "Revenues", "NetIncomeLoss")
            taxonomy: XBRL taxonomy (usually "us-gaap")
        
        Returns:
            DataFrame with value, period_end, filing_date, observed_at
        """
        facts = self.get_company_facts(ticker)
        if not facts:
            return pd.DataFrame()
        
        # Navigate to the concept
        try:
            concept_data = facts["facts"][taxonomy][concept]
            units = concept_data.get("units", {})
            
            # Usually in USD for financial data
            unit_data = units.get("USD", [])
            if not unit_data:
                # Try other units
                for unit_key, unit_values in units.items():
                    if unit_values:
                        unit_data = unit_values
                        break
            
            if not unit_data:
                return pd.DataFrame()
            
            df = pd.DataFrame(unit_data)
            df["ticker"] = ticker
            df["concept"] = concept
            
            # Rename columns
            df = df.rename(columns={
                "val": "value",
                "end": "period_end",
                "filed": "filing_date",
            })
            
            # Parse dates
            df["period_end"] = pd.to_datetime(df["period_end"])
            df["filing_date"] = pd.to_datetime(df["filing_date"])
            
            # observed_at = filing_date (conservative: same day at midnight UTC)
            # For more precision, we'd need to look up the actual acceptanceDateTime
            df["observed_at"] = df["filing_date"].apply(
                lambda d: UTC.localize(datetime.combine(d.date(), datetime.min.time()))
                if pd.notna(d) else None
            )
            
            df["source"] = "sec_xbrl"
            
            return df
            
        except KeyError:
            logger.debug(f"Concept {concept} not found for {ticker}")
            return pd.DataFrame()
    
    # =========================================================================
    # Utilities
    # =========================================================================
    
    def test_connection(self) -> bool:
        """Test API connection."""
        try:
            # Try to get Apple's CIK (well-known)
            cik = self.get_cik("AAPL")
            return cik is not None
        except SECError:
            return False


def get_sec_client(contact_email: Optional[str] = None) -> SECEdgarClient:
    """Get a configured SEC EDGAR client instance."""
    return SECEdgarClient(contact_email=contact_email)


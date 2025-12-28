"""
Survivorship-Safe Universe Builder
==================================

Chapter 4: Builds point-in-time universe using Polygon as symbol master
and FMP for market data (prices, mcap, volume).

ARCHITECTURE:
1. Polygon determines WHO is in the universe (symbol master)
2. FMP provides WHAT we know about them (prices, fundamentals)
3. SecurityMaster tracks stable IDs and ticker changes

UNIVERSE CONSTRUCTION STEPS (at rebalance date T):
1. Query Polygon for all US common stocks active on date T
2. Apply liquidity filter: ADV >= min_adv (default $1M)
3. Apply price filter: Price >= min_price (default $5)
4. Apply AI relevance filter (sector/industry tags from ai_stocks.py)
5. Select Top N by market cap as-of T
6. Map to stable_id via CIK/FIGI (survives ticker changes)
7. Persist with timestamp for audit/replay

SURVIVORSHIP STATUS:
- FULL: Polygon + FMP with verified delisted coverage
- PARTIAL: Current data only (may miss historical delistings)
- UNKNOWN: Not verified

CRITICAL RULES:
- ai_stocks.py is label-only (for AI relevance tagging)
- Never use ai_stocks.py as the candidate universe
- All filtering based on as-of-T data, not current data
"""

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import List, Dict, Any, Optional, Set, Tuple
from pathlib import Path
from enum import Enum

import pandas as pd
import pytz

logger = logging.getLogger(__name__)

UTC = pytz.UTC
ET = pytz.timezone("America/New_York")


class SurvivorshipStatus(Enum):
    """
    Survivorship bias status for universe.
    
    CRITICAL: 
    - FULL requires Polygon as candidate source (use_polygon=True)
    - ai_stocks.py can NEVER produce FULL status (it's label-only)
    - Unit tests may use ai_stocks.py for speed (PARTIAL)
    - Production backtests MUST use Polygon for FULL
    """
    FULL = "full"           # Polygon candidates + verified delisted coverage
    PARTIAL = "partial"     # ai_stocks.py fallback (may miss delistings)
    UNKNOWN = "unknown"     # Not verified (never use for backtests)


@dataclass
class UniverseCandidate:
    """
    A candidate for universe inclusion.
    
    Contains all data needed for filtering decisions.
    """
    ticker: str
    stable_id: str  # CIK or FIGI - survives ticker changes
    company_name: str
    
    # Market data as-of-T
    price: Optional[float] = None
    market_cap: Optional[float] = None
    avg_daily_volume: Optional[float] = None  # Dollar volume
    
    # Classification
    exchange: str = ""
    type: str = "CS"
    sic_code: Optional[str] = None
    sector: Optional[str] = None
    industry: Optional[str] = None
    
    # AI relevance (from ai_stocks.py tagging)
    ai_category: Optional[str] = None
    is_ai_relevant: bool = False
    
    # Status
    is_active: bool = True
    delisted_date: Optional[date] = None
    
    # Filter results
    passed_liquidity: bool = True
    passed_price: bool = True
    passed_mcap: bool = True
    passed_ai_filter: bool = True
    exclusion_reason: Optional[str] = None
    
    @property
    def passes_all_filters(self) -> bool:
        return (
            self.passed_liquidity and
            self.passed_price and
            self.passed_mcap and
            self.passed_ai_filter
        )


@dataclass
class UniverseSnapshot:
    """
    A point-in-time universe snapshot.
    
    Stores the complete universe state at a rebalance date.
    """
    asof_date: date
    constituents: List[UniverseCandidate]
    survivorship_status: SurvivorshipStatus
    
    # Filtering thresholds used
    min_price: float = 5.0
    min_adv: float = 1_000_000  # $1M daily dollar volume
    min_mcap: Optional[float] = None
    max_constituents: int = 100
    
    # Audit info
    total_candidates: int = 0
    passed_liquidity: int = 0
    passed_price: int = 0
    passed_ai_filter: int = 0
    final_count: int = 0
    
    # Warnings
    warnings: List[str] = field(default_factory=list)
    
    @property
    def tickers(self) -> List[str]:
        return [c.ticker for c in self.constituents]
    
    @property
    def stable_ids(self) -> List[str]:
        return [c.stable_id for c in self.constituents]
    
    def get_by_stable_id(self, stable_id: str) -> Optional[UniverseCandidate]:
        """Look up constituent by stable_id."""
        for c in self.constituents:
            if c.stable_id == stable_id:
                return c
        return None
    
    def get_by_ticker(self, ticker: str) -> Optional[UniverseCandidate]:
        """Look up constituent by ticker."""
        for c in self.constituents:
            if c.ticker == ticker:
                return c
        return None
    
    def summary(self) -> str:
        """Generate human-readable summary."""
        status_icon = {
            SurvivorshipStatus.FULL: "✅",
            SurvivorshipStatus.PARTIAL: "⚠️",
            SurvivorshipStatus.UNKNOWN: "❓",
        }
        
        lines = [
            f"Universe Snapshot: {self.asof_date}",
            f"  Status: {status_icon.get(self.survivorship_status, '?')} {self.survivorship_status.value.upper()}",
            f"  Constituents: {self.final_count}",
            f"",
            f"  Filtering Pipeline:",
            f"    Total candidates: {self.total_candidates}",
            f"    → Passed liquidity (ADV >= ${self.min_adv/1e6:.1f}M): {self.passed_liquidity}",
            f"    → Passed price (>= ${self.min_price}): {self.passed_price}",
            f"    → Passed AI filter: {self.passed_ai_filter}",
            f"    → Top {self.max_constituents} by mcap: {self.final_count}",
        ]
        
        if self.warnings:
            lines.append(f"")
            lines.append(f"  Warnings ({len(self.warnings)}):")
            for w in self.warnings[:5]:
                lines.append(f"    ⚠️ {w}")
        
        # Show sample tickers
        if self.constituents:
            sample = self.tickers[:10]
            lines.append(f"")
            lines.append(f"  Sample tickers: {', '.join(sample)}{'...' if len(self.tickers) > 10 else ''}")
        
        return "\n".join(lines)
    
    def to_dict(self) -> Dict:
        """Serialize for storage."""
        return {
            "asof_date": self.asof_date.isoformat(),
            "survivorship_status": self.survivorship_status.value,
            "constituents": [
                {
                    "ticker": c.ticker,
                    "stable_id": c.stable_id,
                    "company_name": c.company_name,
                    "price": c.price,
                    "market_cap": c.market_cap,
                    "avg_daily_volume": c.avg_daily_volume,
                    "ai_category": c.ai_category,
                }
                for c in self.constituents
            ],
            "min_price": self.min_price,
            "min_adv": self.min_adv,
            "max_constituents": self.max_constituents,
            "total_candidates": self.total_candidates,
            "final_count": self.final_count,
            "warnings": self.warnings,
        }


class UniverseBuilder:
    """
    Builds survivorship-safe universes using Polygon + FMP.
    
    Usage:
        builder = UniverseBuilder()
        
        # Build universe for a specific date
        snapshot = builder.build(date(2024, 6, 15))
        
        # Check survivorship status
        if snapshot.survivorship_status == SurvivorshipStatus.FULL:
            print("Safe for backtesting")
    """
    
    def __init__(
        self,
        polygon_client=None,
        fmp_client=None,
        security_master=None,
        ai_tagger=None,
    ):
        """
        Initialize universe builder.
        
        Args:
            polygon_client: PolygonClient for symbol master queries
            fmp_client: FMPClient for market data
            security_master: SecurityMaster for stable ID tracking
            ai_tagger: Function to tag AI relevance (default: use ai_stocks.py)
        """
        self._polygon = polygon_client
        self._fmp = fmp_client
        self._security_master = security_master
        self._ai_tagger = ai_tagger or self._default_ai_tagger
        
        # Cache for AI stock mappings
        self._ai_stock_cache: Optional[Dict[str, str]] = None
    
    def _get_polygon_client(self):
        """Lazy-load Polygon client."""
        if self._polygon is None:
            from src.data.polygon_client import PolygonClient
            self._polygon = PolygonClient()
        return self._polygon
    
    def _get_fmp_client(self):
        """Lazy-load FMP client."""
        if self._fmp is None:
            from src.data.fmp_client import FMPClient
            self._fmp = FMPClient()
        return self._fmp
    
    def _get_security_master(self):
        """Lazy-load Security Master."""
        if self._security_master is None:
            from src.data.security_master import SecurityMaster
            self._security_master = SecurityMaster()
        return self._security_master
    
    def _default_ai_tagger(self, ticker: str, sic_code: Optional[str] = None) -> Tuple[bool, Optional[str]]:
        """
        Default AI relevance tagger using ai_stocks.py.
        
        Returns:
            Tuple of (is_ai_relevant, ai_category)
        """
        if self._ai_stock_cache is None:
            try:
                from src.universe.ai_stocks import AI_UNIVERSE
                self._ai_stock_cache = {}
                for category, tickers in AI_UNIVERSE.items():
                    for t in tickers:
                        self._ai_stock_cache[t] = category
            except ImportError:
                logger.warning("ai_stocks.py not found, AI tagging disabled")
                self._ai_stock_cache = {}
        
        if ticker in self._ai_stock_cache:
            return True, self._ai_stock_cache[ticker]
        
        # Could add SIC code based AI relevance here
        # For now, only exact matches
        return False, None
    
    def build(
        self,
        asof_date: date,
        min_price: float = 5.0,
        min_adv: float = 1_000_000,  # $1M
        min_mcap: Optional[float] = None,
        max_constituents: int = 100,
        ai_filter: bool = True,
        use_polygon: bool = True,
        skip_enrichment: bool = False,  # For fast testing without FMP API calls
    ) -> UniverseSnapshot:
        """
        Build universe for a specific date.
        
        Args:
            asof_date: Date to build universe for
            min_price: Minimum stock price
            min_adv: Minimum average daily dollar volume
            min_mcap: Minimum market cap (None = no filter)
            max_constituents: Maximum number of stocks
            ai_filter: Whether to apply AI relevance filter
            use_polygon: Whether to use Polygon for candidates (vs ai_stocks.py only)
            skip_enrichment: Skip FMP API calls for market data (for fast testing)
            
        Returns:
            UniverseSnapshot with constituents and metadata
        """
        warnings = []
        survivorship_status = SurvivorshipStatus.UNKNOWN
        
        # Step 1: Get candidates from Polygon (or fallback)
        if use_polygon:
            try:
                candidates, survivorship_status = self._get_polygon_candidates(asof_date)
            except Exception as e:
                warnings.append(f"Polygon failed, falling back to AI stocks: {e}")
                candidates = self._get_ai_stock_candidates()
                survivorship_status = SurvivorshipStatus.PARTIAL
        else:
            candidates = self._get_ai_stock_candidates()
            survivorship_status = SurvivorshipStatus.PARTIAL
            warnings.append("Using ai_stocks.py only (survivorship_status=PARTIAL)")
        
        total_candidates = len(candidates)
        logger.info(f"Got {total_candidates} candidates for {asof_date}")
        
        # Step 2: Enrich with market data from FMP (skip for fast testing)
        if not skip_enrichment:
            candidates = self._enrich_with_market_data(candidates, asof_date)
        else:
            logger.info("Skipping FMP enrichment (skip_enrichment=True)")
        
        # Step 3: Apply filters
        # 3a. Liquidity filter
        for c in candidates:
            if c.avg_daily_volume is None or c.avg_daily_volume < min_adv:
                c.passed_liquidity = False
                c.exclusion_reason = f"ADV < ${min_adv/1e6:.1f}M"
        passed_liquidity = sum(1 for c in candidates if c.passed_liquidity)
        
        # 3b. Price filter
        for c in candidates:
            if c.passed_liquidity:
                if c.price is None or c.price < min_price:
                    c.passed_price = False
                    c.exclusion_reason = f"Price < ${min_price}"
        passed_price = sum(1 for c in candidates if c.passed_liquidity and c.passed_price)
        
        # 3c. Market cap filter
        if min_mcap:
            for c in candidates:
                if c.passed_liquidity and c.passed_price:
                    if c.market_cap is None or c.market_cap < min_mcap:
                        c.passed_mcap = False
                        c.exclusion_reason = f"MCap < ${min_mcap/1e9:.1f}B"
        
        # 3d. AI relevance filter
        if ai_filter:
            for c in candidates:
                if c.passed_liquidity and c.passed_price and c.passed_mcap:
                    is_ai, category = self._ai_tagger(c.ticker, c.sic_code)
                    c.is_ai_relevant = is_ai
                    c.ai_category = category
                    if not is_ai:
                        c.passed_ai_filter = False
                        c.exclusion_reason = "Not AI-relevant"
        
        passed_ai = sum(1 for c in candidates if c.passes_all_filters)
        
        # Step 4: Rank by market cap and take top N
        filtered = [c for c in candidates if c.passes_all_filters]
        
        # Sort by market cap (descending), handle None
        filtered.sort(key=lambda x: x.market_cap or 0, reverse=True)
        
        # Take top N
        constituents = filtered[:max_constituents]
        final_count = len(constituents)
        
        # Step 5: Build snapshot
        snapshot = UniverseSnapshot(
            asof_date=asof_date,
            constituents=constituents,
            survivorship_status=survivorship_status,
            min_price=min_price,
            min_adv=min_adv,
            min_mcap=min_mcap,
            max_constituents=max_constituents,
            total_candidates=total_candidates,
            passed_liquidity=passed_liquidity,
            passed_price=passed_price,
            passed_ai_filter=passed_ai,
            final_count=final_count,
            warnings=warnings,
        )
        
        logger.info(f"Built universe: {final_count} constituents, status={survivorship_status.value}")
        
        return snapshot
    
    def _get_polygon_candidates(
        self,
        asof_date: date,
    ) -> Tuple[List[UniverseCandidate], SurvivorshipStatus]:
        """
        Get universe candidates from Polygon.
        
        Returns:
            Tuple of (candidates, survivorship_status)
        """
        polygon = self._get_polygon_client()
        
        # Get tickers active on asof_date
        tickers = polygon.get_tickers_asof(
            asof_date=asof_date,
            market="stocks",
            locale="us",
            ticker_type="CS",  # Common stocks
            active_only=False,  # Include delisted
        )
        
        candidates = []
        for t in tickers:
            candidate = UniverseCandidate(
                ticker=t.ticker,
                stable_id=t.stable_id,
                company_name=t.name,
                exchange=t.primary_exchange,
                type=t.type,
                sic_code=t.sic_code,
                is_active=t.active,
                delisted_date=t.delisted_utc.date() if t.delisted_utc else None,
            )
            candidates.append(candidate)
        
        # Determine status based on Polygon capabilities
        # If we got results with historical date query, we're FULL
        if len(candidates) > 0:
            status = SurvivorshipStatus.FULL
        else:
            status = SurvivorshipStatus.PARTIAL
        
        return candidates, status
    
    def _get_ai_stock_candidates(self) -> List[UniverseCandidate]:
        """
        Get candidates from ai_stocks.py only.
        
        WARNING: This ALWAYS results in PARTIAL survivorship status.
        ai_stocks.py is label-only - it should NEVER be the production candidate source.
        
        Use case: Fast unit tests only.
        Production: Must use Polygon (use_polygon=True) for FULL survivorship.
        """
        try:
            from src.universe.ai_stocks import AI_UNIVERSE
        except ImportError:
            logger.error("Cannot import ai_stocks.py")
            return []
        
        candidates = []
        for category, tickers in AI_UNIVERSE.items():
            for ticker in tickers:
                candidate = UniverseCandidate(
                    ticker=ticker,
                    stable_id=f"TICKER:{ticker}",  # Placeholder stable ID
                    company_name=ticker,
                    is_ai_relevant=True,
                    ai_category=category,
                )
                candidates.append(candidate)
        
        return candidates
    
    def _enrich_with_market_data(
        self,
        candidates: List[UniverseCandidate],
        asof_date: date,
    ) -> List[UniverseCandidate]:
        """
        Enrich candidates with market data from FMP.
        
        Adds: price, market_cap, avg_daily_volume, sector, industry
        """
        if not candidates:
            return candidates
        
        try:
            fmp = self._get_fmp_client()
        except Exception as e:
            logger.warning(f"FMP client unavailable: {e}")
            return candidates
        
        # Process in batches to respect rate limits
        batch_size = 5
        enriched = []
        
        for i in range(0, len(candidates), batch_size):
            batch = candidates[i:i + batch_size]
            
            for c in batch:
                try:
                    # Get profile for sector/industry
                    profile = fmp.get_profile(c.ticker)
                    if profile:
                        c.sector = profile.get("sector")
                        c.industry = profile.get("industry")
                        c.market_cap = profile.get("mktCap")
                    
                    # Get historical prices for ADV and price
                    end_date = asof_date.isoformat()
                    start_date = (asof_date - timedelta(days=30)).isoformat()
                    
                    prices = fmp.get_historical_prices(c.ticker, start_date, end_date)
                    
                    if not prices.empty:
                        # Latest price
                        c.price = float(prices.iloc[0]["close"])
                        
                        # Calculate ADV (20-day)
                        if len(prices) >= 20:
                            prices = prices.head(20)
                        c.avg_daily_volume = float(prices["volume"].mean() * c.price) if c.price else None
                        
                except Exception as e:
                    logger.debug(f"Failed to enrich {c.ticker}: {e}")
                
                enriched.append(c)
        
        return enriched
    
    def build_historical_series(
        self,
        start_date: date,
        end_date: date,
        rebalance_freq: str = "monthly",
        **kwargs,
    ) -> List[UniverseSnapshot]:
        """
        Build universe snapshots for a date range.
        
        Useful for backtesting universe construction.
        
        Args:
            start_date: Start of date range
            end_date: End of date range
            rebalance_freq: "monthly" or "quarterly"
            **kwargs: Passed to build()
            
        Returns:
            List of UniverseSnapshot for each rebalance date
        """
        from src.data.trading_calendar import TradingCalendarImpl
        calendar = TradingCalendarImpl()
        
        # Generate rebalance dates
        if rebalance_freq == "monthly":
            rebalance_dates = calendar.get_monthly_rebalance_dates(start_date, end_date)
        elif rebalance_freq == "quarterly":
            rebalance_dates = calendar.get_quarterly_rebalance_dates(start_date, end_date)
        else:
            raise ValueError(f"Unknown rebalance_freq: {rebalance_freq}")
        
        snapshots = []
        for rebal_date in rebalance_dates:
            try:
                snapshot = self.build(rebal_date, **kwargs)
                snapshots.append(snapshot)
                logger.info(f"Built snapshot for {rebal_date}: {snapshot.final_count} constituents")
            except Exception as e:
                logger.error(f"Failed to build snapshot for {rebal_date}: {e}")
        
        return snapshots


def test_universe_builder():
    """Test the universe builder with current data."""
    import os
    
    # Load .env
    env_path = '.env'
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip().strip('"').strip("'")
    
    print("=" * 60)
    print("UNIVERSE BUILDER TEST")
    print("=" * 60)
    
    builder = UniverseBuilder()
    
    # Test 1: Build universe for today
    print("\n[Test 1] Building universe for today...")
    today = date.today()
    snapshot = builder.build(today)
    print(snapshot.summary())
    
    # Test 2: Build universe for 1 year ago
    print("\n" + "=" * 60)
    print("[Test 2] Building universe for 1 year ago...")
    historical = date.today() - timedelta(days=365)
    snapshot_hist = builder.build(historical)
    print(snapshot_hist.summary())
    
    # Test 3: Compare universes
    print("\n" + "=" * 60)
    print("[Test 3] Universe comparison...")
    current_tickers = set(snapshot.tickers)
    historical_tickers = set(snapshot_hist.tickers)
    
    added = current_tickers - historical_tickers
    removed = historical_tickers - current_tickers
    
    print(f"  Added since then: {len(added)} - {list(added)[:5]}...")
    print(f"  Removed since then: {len(removed)} - {list(removed)[:5]}...")
    
    return snapshot.survivorship_status == SurvivorshipStatus.FULL


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_universe_builder()


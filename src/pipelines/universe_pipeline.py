"""
Universe Pipeline
=================

Builds survivorship-safe dynamic universe at each rebalance date.

The default universe is 100 AI stocks across 10 subcategories:
1. AI Compute & Core Semiconductors
2. Semiconductor Manufacturing, Equipment & EDA  
3. Networking, Optics & Data-Center Hardware
4. Power, Cooling & Data Center Infrastructure
5. Cloud/Hyperscalers & AI Platforms
6. Data Platforms & Enterprise Software
7. Cybersecurity & Identity
8. Robotics, Automation & Industrial AI
9. Autonomy, Drones & Defense AI
10. AI-as-a-Product, Ads & Other Adopters

This pipeline depends on interfaces (not concrete implementations):
- PITStore: For querying historical market data
- TradingCalendar: For trading days and cutoffs

NOTE: Actual PITStore/TradingCalendar implementations are in Section 3.
      Until then, stubs are used for testing.
"""

from dataclasses import dataclass, field
from datetime import date
from typing import List, Optional, Dict, Any, TYPE_CHECKING
import logging

# Type checking imports (avoid circular imports)
if TYPE_CHECKING:
    from ..interfaces import PITStore, TradingCalendar

logger = logging.getLogger(__name__)


# Import AI universe seed
try:
    from ..universe import (
        AI_UNIVERSE,
        AI_CATEGORIES,
        CATEGORY_DESCRIPTIONS,
        get_all_tickers,
        get_tickers_by_category,
        get_tickers_by_categories,
        get_category_for_ticker,
    )
except ImportError:
    # Fallback for direct execution
    from universe import (
        AI_UNIVERSE,
        AI_CATEGORIES,
        CATEGORY_DESCRIPTIONS,
        get_all_tickers,
        get_tickers_by_category,
        get_tickers_by_categories,
        get_category_for_ticker,
    )


@dataclass
class TickerMetadata:
    """
    Metadata for a single ticker explaining inclusion/exclusion.
    
    Stores the data AS OF the universe construction date,
    enabling audit and debugging of universe decisions.
    
    CRITICAL: stable_id is the primary identity, NOT ticker.
    Ticker can change (e.g., FB → META), stable_id cannot.
    """
    ticker: str
    stable_id: Optional[str] = None  # Primary identity (survives ticker changes)
    
    # Data as of the construction date
    asof_price: Optional[float] = None
    asof_adv_20: Optional[float] = None  # 20-day average daily volume
    asof_mcap: Optional[float] = None    # Market cap
    
    # Classification
    sector: Optional[str] = None
    industry: Optional[str] = None
    
    # Filter results
    passed_filters: bool = True
    exclusion_reason: Optional[str] = None
    
    # Rank (if included)
    mcap_rank: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "ticker": self.ticker,
            "stable_id": self.stable_id,
            "asof_price": self.asof_price,
            "asof_adv_20": self.asof_adv_20,
            "asof_mcap": self.asof_mcap,
            "sector": self.sector,
            "industry": self.industry,
            "passed_filters": self.passed_filters,
            "exclusion_reason": self.exclusion_reason,
            "mcap_rank": self.mcap_rank,
        }


class SurvivorshipStatus:
    """Status of survivorship data availability."""
    FULL = "full"          # Using survivorship-bias-free data feed
    PARTIAL = "partial"    # Seed universe only (may miss historical delistings)
    UNKNOWN = "unknown"    # Not verified


@dataclass
class UniverseResult:
    """
    Result of universe construction with full audit trail.
    
    Includes metadata for EVERY ticker considered (not just included),
    enabling survivorship audits and debugging.
    
    CRITICAL: stable_ids is the canonical representation for historical replay.
    Use stable_ids (not tickers) when reconstructing historical universes.
    """
    asof_date: date
    tickers: List[str]  # Included tickers (final universe)
    stable_ids: List[str] = field(default_factory=list)  # Canonical IDs for replay
    
    # Full metadata for all tickers (included AND excluded)
    ticker_metadata: Dict[str, TickerMetadata] = field(default_factory=dict)
    
    # Summary info
    filters_applied: List[str] = field(default_factory=list)
    candidates_considered: int = 0
    excluded_count: int = 0
    warnings: List[str] = field(default_factory=list)
    
    # Survivorship status (IMPORTANT for backtest validity)
    survivorship_status: str = SurvivorshipStatus.PARTIAL
    
    def __len__(self) -> int:
        return len(self.tickers)
    
    def get_included_metadata(self) -> Dict[str, TickerMetadata]:
        """Get metadata only for included tickers."""
        return {t: self.ticker_metadata[t] for t in self.tickers if t in self.ticker_metadata}
    
    def get_excluded_metadata(self) -> Dict[str, TickerMetadata]:
        """Get metadata for excluded tickers."""
        return {
            t: m for t, m in self.ticker_metadata.items() 
            if not m.passed_filters
        }
    
    def get_exclusion_reasons(self) -> Dict[str, int]:
        """Count exclusions by reason."""
        reasons: Dict[str, int] = {}
        for m in self.ticker_metadata.values():
            if not m.passed_filters and m.exclusion_reason:
                reasons[m.exclusion_reason] = reasons.get(m.exclusion_reason, 0) + 1
        return reasons
    
    def summary(self) -> str:
        # Survivorship warning
        surv_warning = ""
        if self.survivorship_status == SurvivorshipStatus.PARTIAL:
            surv_warning = " ⚠️ PARTIAL SURVIVORSHIP"
        elif self.survivorship_status == SurvivorshipStatus.UNKNOWN:
            surv_warning = " ⚠️ SURVIVORSHIP UNKNOWN"
        
        lines = [
            f"📊 Universe as of {self.asof_date}{surv_warning}",
            f"  Constituents: {len(self.tickers)}",
            f"  Stable IDs: {len(self.stable_ids)}",
            f"  Candidates considered: {self.candidates_considered}",
            f"  Excluded: {self.excluded_count}",
            f"  Filters: {', '.join(self.filters_applied)}",
            f"  Survivorship: {self.survivorship_status}",
        ]
        
        # Exclusion breakdown
        exclusion_reasons = self.get_exclusion_reasons()
        if exclusion_reasons:
            lines.append("  Exclusion breakdown:")
            for reason, count in sorted(exclusion_reasons.items(), key=lambda x: -x[1]):
                lines.append(f"    {reason}: {count}")
        
        if self.tickers:
            lines.append(f"  Sample: {', '.join(self.tickers[:5])}...")
        
        if self.warnings:
            lines.append(f"  ⚠️ Warnings: {len(self.warnings)}")
            for w in self.warnings[:3]:
                lines.append(f"    - {w}")
        
        return "\n".join(lines)
    
    def get_ticker_for_stable_id(self, stable_id: str) -> Optional[str]:
        """Look up ticker by stable_id (for historical replay)."""
        for meta in self.ticker_metadata.values():
            if meta.stable_id == stable_id and meta.passed_filters:
                return meta.ticker
        return None
    
    def get_stable_id_for_ticker(self, ticker: str) -> Optional[str]:
        """Look up stable_id by ticker."""
        if ticker in self.ticker_metadata:
            return self.ticker_metadata[ticker].stable_id
        return None
    
    def to_dict(self) -> Dict:
        """Serialize for storage/export."""
        return {
            "asof_date": self.asof_date.isoformat(),
            "tickers": self.tickers,
            "stable_ids": self.stable_ids,
            "ticker_metadata": {t: m.to_dict() for t, m in self.ticker_metadata.items()},
            "filters_applied": self.filters_applied,
            "candidates_considered": self.candidates_considered,
            "excluded_count": self.excluded_count,
            "warnings": self.warnings,
            "survivorship_status": self.survivorship_status,
        }


def run_universe_construction(
    asof_date: date,
    max_size: int = 100,
    min_price: float = 5.0,
    min_avg_volume: int = 100_000,
    min_market_cap: float = 1e9,
    categories: Optional[List[str]] = None,  # AI subcategories to include
    use_seed_universe: bool = True,  # Use predefined AI 100 stocks
    # Legacy filters (used if use_seed_universe=False)
    sectors: Optional[List[str]] = None,
    industries: Optional[List[str]] = None,
    # Dependency injection (interfaces)
    pit_store: Optional["PITStore"] = None,
    calendar: Optional["TradingCalendar"] = None,
) -> UniverseResult:
    """
    Construct the investment universe as of a specific date.
    
    By default, uses the predefined AI 100 universe (10 subcategories).
    Can optionally filter by subcategory for subsector-specific forecasts.
    
    This is survivorship-safe: it uses only information available
    at the asof_date, not current data.
    
    Args:
        asof_date: The date for which to construct universe
        max_size: Maximum number of stocks (top N by market cap)
        min_price: Minimum stock price filter
        min_avg_volume: Minimum average daily volume
        min_market_cap: Minimum market capitalization
        categories: AI subcategories to include (None = all 10)
                   Options: ai_compute_core_semis, semicap_eda_manufacturing,
                   networking_optics_dc_hw, datacenter_power_cooling_reits,
                   cloud_platforms_model_owners, data_platforms_enterprise_sw,
                   cybersecurity, robotics_industrial_ai, autonomy_defense_ai,
                   ai_apps_ads_misc
        use_seed_universe: If True, use predefined AI 100 stocks
        sectors: Legacy filter (used if use_seed_universe=False)
        industries: Legacy filter (used if use_seed_universe=False)
        pit_store: PIT data store (None = use stub for now)
        calendar: Trading calendar (None = use stub)
    
    Returns:
        UniverseResult with constituent list and full metadata
    """
    logger.info(f"Constructing universe as of {asof_date}")
    
    # Use stubs if no real implementation provided
    # TODO [Section 3]: Replace with real implementations
    if pit_store is None:
        from ..interfaces import StubPITStore
        pit_store = StubPITStore()
        logger.warning("Using StubPITStore - universe will use placeholder data")
    
    if calendar is None:
        from ..interfaces import StubTradingCalendar
        calendar = StubTradingCalendar()
    
    # Record filters applied
    filters_applied = []
    
    # =========================================================================
    # Get seed universe tickers
    # =========================================================================
    if use_seed_universe:
        if categories:
            # Filter to specific subcategories
            seed_tickers = get_tickers_by_categories(categories)
            filters_applied.append(f"AI categories: {', '.join(categories)}")
        else:
            # All 100 AI stocks
            seed_tickers = get_all_tickers()
            filters_applied.append("AI seed universe (all 10 categories)")
        
        logger.info(f"Using seed universe: {len(seed_tickers)} tickers")
    else:
        # Legacy mode: would query from broader market
        seed_tickers = []
        if sectors:
            filters_applied.append(f"sectors: {len(sectors)}")
        if industries:
            filters_applied.append(f"industries: {len(industries)}")
    
    if min_price > 0:
        filters_applied.append(f"price >= ${min_price}")
    if min_avg_volume > 0:
        filters_applied.append(f"ADV >= {min_avg_volume:,}")
    if min_market_cap > 0:
        filters_applied.append(f"mcap >= ${min_market_cap/1e9:.1f}B")
    filters_applied.append(f"top {max_size} by mcap")
    
    # =========================================================================
    # TODO [Section 3]: Implement real PIT data fetching
    # 
    # When PITStore is implemented, this should:
    # 1. For each seed ticker, query price/volume/mcap as of asof_date
    # 2. Apply liquidity filters using data as of asof_date
    # 3. Record exclusion reasons
    # 4. Rank by market cap and select top N
    # =========================================================================
    
    # Build placeholder data from seed universe
    # These are approximate market caps/prices for demonstration
    # TODO [Section 3]: Replace with real PIT data
    placeholder_mcaps = {
        # Top tier (>$500B)
        "AAPL": 3000e9, "MSFT": 2800e9, "NVDA": 1200e9, "GOOGL": 1700e9, "AMZN": 1600e9,
        "META": 900e9, "TSM": 600e9, "AVGO": 400e9, "ORCL": 320e9,
        # Large cap ($100B-$500B)
        "AMD": 220e9, "CRM": 250e9, "ADBE": 240e9, "INTC": 190e9, "QCOM": 180e9,
        "TXN": 155e9, "NOW": 140e9, "ASML": 350e9, "AMAT": 130e9, "IBM": 150e9,
        "HON": 140e9, "LMT": 120e9, "RTX": 130e9, "NOC": 70e9,
        # Mid cap ($20B-$100B)
        "MU": 95e9, "LRCX": 100e9, "KLAC": 85e9, "SNPS": 75e9, "CDNS": 75e9,
        "PANW": 95e9, "CRWD": 60e9, "MRVL": 52e9, "ANET": 80e9, "DELL": 50e9,
        "PLTR": 40e9, "SNOW": 50e9, "DDOG": 35e9, "FTNT": 45e9, "ZS": 25e9,
        "UBER": 120e9, "SHOP": 90e9, "INTU": 170e9, "TSLA": 700e9,
        # Others - assign reasonable defaults
    }
    
    placeholder_data = []
    for ticker in seed_tickers:
        category = get_category_for_ticker(ticker)
        mcap = placeholder_mcaps.get(ticker, 20e9)  # Default $20B if not specified
        # Estimate price/volume from mcap
        price = max(50.0, mcap / 1e9 * 0.5)  # Rough estimate
        adv = max(500_000, mcap / 1e6)  # Rough estimate
        
        placeholder_data.append((
            ticker, 
            price, 
            adv, 
            mcap, 
            category or "unknown",
            CATEGORY_DESCRIPTIONS.get(category, "Unknown")
        ))
    
    # Build metadata for all candidates
    ticker_metadata: Dict[str, TickerMetadata] = {}
    included = []
    excluded_count = 0
    
    for ticker, price, adv, mcap, category, category_desc in placeholder_data:
        meta = TickerMetadata(
            ticker=ticker,
            asof_price=price,
            asof_adv_20=adv,
            asof_mcap=mcap,
            sector=category,  # Using category as sector for now
            industry=category_desc,  # Using category description as industry
        )
        
        # Apply filters
        if price < min_price:
            meta.passed_filters = False
            meta.exclusion_reason = f"price < ${min_price}"
            excluded_count += 1
        elif adv < min_avg_volume:
            meta.passed_filters = False
            meta.exclusion_reason = f"ADV < {min_avg_volume:,}"
            excluded_count += 1
        elif mcap < min_market_cap:
            meta.passed_filters = False
            meta.exclusion_reason = f"mcap < ${min_market_cap/1e9:.1f}B"
            excluded_count += 1
        else:
            included.append((ticker, mcap, meta))
        
        ticker_metadata[ticker] = meta
    
    # Sort by market cap and take top N
    included.sort(key=lambda x: x[1], reverse=True)
    final_tickers = []
    
    for rank, (ticker, mcap, meta) in enumerate(included[:max_size], 1):
        meta.mcap_rank = rank
        final_tickers.append(ticker)
    
    # Mark those that passed filters but didn't make the cut
    for ticker, mcap, meta in included[max_size:]:
        meta.passed_filters = False
        meta.exclusion_reason = f"mcap rank > {max_size}"
        excluded_count += 1
    
    warnings = []
    if pit_store.__class__.__name__ == "StubPITStore":
        warnings.append("Using placeholder data - real PITStore not implemented yet")
    
    return UniverseResult(
        asof_date=asof_date,
        tickers=final_tickers,
        ticker_metadata=ticker_metadata,
        filters_applied=filters_applied,
        candidates_considered=len(placeholder_data),
        excluded_count=excluded_count,
        warnings=warnings,
    )


def get_historical_universes(
    start_date: date,
    end_date: date,
    rebalance_freq: str = "monthly",
    pit_store: Optional["PITStore"] = None,
    calendar: Optional["TradingCalendar"] = None,
    universe_store: Optional[Any] = None,
) -> Dict[date, UniverseResult]:
    """
    Get universe constituents for all rebalance dates in a range.
    
    Used for backtesting to ensure survivorship-safe evaluation.
    
    Args:
        start_date: Start of period
        end_date: End of period
        rebalance_freq: 'monthly', 'quarterly', or 'weekly'
        pit_store: PIT data store
        calendar: Trading calendar
        universe_store: Optional cached universe store
    
    Returns:
        Dict mapping rebalance dates to UniverseResult
    
    NOTE: Full implementation deferred to Section 4 (Dynamic Universe)
    """
    # Use stub calendar if not provided
    if calendar is None:
        from ..interfaces import StubTradingCalendar
        calendar = StubTradingCalendar()
    
    rebalance_dates = calendar.get_rebalance_dates(start_date, end_date, rebalance_freq)
    
    results = {}
    for rebal_date in rebalance_dates:
        logger.info(f"Building universe for {rebal_date}")
        results[rebal_date] = run_universe_construction(
            asof_date=rebal_date,
            pit_store=pit_store,
            calendar=calendar,
        )
    
    return results

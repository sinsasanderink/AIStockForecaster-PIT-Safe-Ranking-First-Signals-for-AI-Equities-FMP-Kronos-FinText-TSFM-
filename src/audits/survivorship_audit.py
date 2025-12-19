"""
Survivorship Bias Audit
=======================

Validates that universe construction does not suffer from survivorship bias.

Key Checks:
- Universe includes delisted/merged stocks in historical periods
- Universe constituents vary through time (not hardcoded)
- Backtests include both winners and losers
"""

from dataclasses import dataclass, field
from datetime import date
from typing import List, Dict, Set, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class SurvivorshipAuditResult:
    """Result of a survivorship bias audit."""
    start_date: date
    end_date: date
    universes_checked: int
    unique_tickers_seen: int
    currently_delisted: int
    universe_turnover_rate: float  # Average monthly change
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    duration_seconds: float = 0.0
    
    @property
    def passed(self) -> bool:
        """Audit passes if no critical issues."""
        return len(self.issues) == 0
    
    def summary(self) -> str:
        status = "✅ PASSED" if self.passed else "❌ FAILED"
        lines = [
            f"{status}: Survivorship Audit",
            f"  Period: {self.start_date} to {self.end_date}",
            f"  Universes checked: {self.universes_checked}",
            f"  Unique tickers seen: {self.unique_tickers_seen}",
            f"  Currently delisted: {self.currently_delisted}",
            f"  Monthly turnover: {self.universe_turnover_rate:.1%}",
            f"  Duration: {self.duration_seconds:.1f}s",
        ]
        
        if self.issues:
            lines.append(f"  Issues ({len(self.issues)}):")
            for issue in self.issues[:5]:
                lines.append(f"    ❌ {issue}")
            if len(self.issues) > 5:
                lines.append(f"    ... and {len(self.issues) - 5} more")
        
        if self.warnings:
            lines.append(f"  Warnings ({len(self.warnings)}):")
            for warn in self.warnings[:3]:
                lines.append(f"    ⚠️ {warn}")
        
        return "\n".join(lines)


def run_survivorship_audit(
    start_date: date,
    end_date: date,
    current_universe: Optional[List[str]] = None,
) -> SurvivorshipAuditResult:
    """
    Audit universe construction for survivorship bias.
    
    This checks:
    1. Universe varies meaningfully over time
    2. Historical universes include stocks no longer trading
    3. No hardcoded "winners list"
    
    Args:
        start_date: Start of audit period
        end_date: End of audit period
        current_universe: Current universe for comparison
    
    Returns:
        SurvivorshipAuditResult with findings
    """
    import time
    start_time = time.time()
    
    logger.info(f"Starting survivorship audit: {start_date} to {end_date}")
    
    issues = []
    warnings = []
    
    # TODO: Implement actual audit
    # This requires:
    # 1. Load all historical universes from start_date to end_date
    # 2. Compute unique tickers across all universes
    # 3. Check which of those tickers are currently delisted
    # 4. Compute monthly turnover rates
    # 5. Flag if turnover is suspiciously low (hardcoded list?)
    # 6. Flag if no delisted stocks appear (survivorship bias)
    
    # Placeholder checks
    universes_checked = 0
    unique_tickers = set()
    delisted_count = 0
    turnover_rate = 0.0
    
    # Check 1: Universe should have some turnover
    if turnover_rate < 0.01:
        warnings.append(
            "Universe turnover very low (<1%/month) - may be hardcoded"
        )
    
    # Check 2: Should see some delisted stocks in historical data
    # (indicates we're not just using current winners)
    
    logger.warning("Survivorship audit not yet fully implemented")
    warnings.append("Audit incomplete - awaiting universe history")
    
    duration = time.time() - start_time
    
    return SurvivorshipAuditResult(
        start_date=start_date,
        end_date=end_date,
        universes_checked=universes_checked,
        unique_tickers_seen=len(unique_tickers),
        currently_delisted=delisted_count,
        universe_turnover_rate=turnover_rate,
        issues=issues,
        warnings=warnings,
        duration_seconds=duration,
    )


def check_universe_contains_delisted(
    historical_universes: Dict[date, List[str]],
    current_active_tickers: Set[str],
) -> Dict[str, date]:
    """
    Find delisted stocks that appear in historical universes.
    
    A healthy backtest should include some stocks that were
    later delisted/merged/failed.
    
    Args:
        historical_universes: Map of date -> ticker list
        current_active_tickers: Set of currently trading tickers
    
    Returns:
        Dict mapping delisted ticker -> last date in universe
    """
    all_historical = set()
    last_seen = {}
    
    for asof_date, tickers in sorted(historical_universes.items()):
        for ticker in tickers:
            all_historical.add(ticker)
            last_seen[ticker] = asof_date
    
    # Find tickers that were in universe but are no longer active
    delisted = {
        ticker: last_seen[ticker]
        for ticker in all_historical
        if ticker not in current_active_tickers
    }
    
    return delisted


def compute_universe_turnover(
    historical_universes: Dict[date, List[str]],
) -> float:
    """
    Compute average monthly turnover rate of universe.
    
    Low turnover might indicate a hardcoded list.
    High turnover is expected as market caps change.
    
    Args:
        historical_universes: Map of date -> ticker list
    
    Returns:
        Average monthly turnover rate (0-1)
    """
    if len(historical_universes) < 2:
        return 0.0
    
    sorted_dates = sorted(historical_universes.keys())
    turnovers = []
    
    for i in range(1, len(sorted_dates)):
        prev_set = set(historical_universes[sorted_dates[i-1]])
        curr_set = set(historical_universes[sorted_dates[i]])
        
        if not prev_set:
            continue
        
        # Turnover = (additions + deletions) / (2 * average size)
        additions = len(curr_set - prev_set)
        deletions = len(prev_set - curr_set)
        avg_size = (len(prev_set) + len(curr_set)) / 2
        
        if avg_size > 0:
            turnover = (additions + deletions) / (2 * avg_size)
            turnovers.append(turnover)
    
    return sum(turnovers) / len(turnovers) if turnovers else 0.0


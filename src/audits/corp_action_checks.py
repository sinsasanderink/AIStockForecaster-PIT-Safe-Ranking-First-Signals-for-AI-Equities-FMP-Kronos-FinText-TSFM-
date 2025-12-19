"""
Corporate Action Checks
=======================

Validates that price data is correctly adjusted for corporate actions
(splits, dividends, spin-offs) and that adjustments are applied consistently.
"""

from dataclasses import dataclass, field
from datetime import date
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class CorpActionIssue:
    """A single corporate action data issue."""
    ticker: str
    action_date: date
    action_type: str  # "split", "dividend", "spinoff"
    issue: str
    severity: str = "warning"
    
    def __str__(self) -> str:
        return f"[{self.severity}] {self.ticker} {self.action_date}: {self.issue}"


@dataclass
class CorpActionAuditResult:
    """Result of corporate action audit."""
    start_date: date
    end_date: date
    tickers_checked: int
    actions_found: int
    issues: List[CorpActionIssue] = field(default_factory=list)
    duration_seconds: float = 0.0
    
    @property
    def passed(self) -> bool:
        return not any(i.severity == "error" for i in self.issues)
    
    def summary(self) -> str:
        status = "✅ PASSED" if self.passed else "❌ FAILED"
        lines = [
            f"{status}: Corporate Action Audit",
            f"  Period: {self.start_date} to {self.end_date}",
            f"  Tickers checked: {self.tickers_checked}",
            f"  Actions found: {self.actions_found}",
            f"  Issues: {len(self.issues)}",
            f"  Duration: {self.duration_seconds:.1f}s",
        ]
        
        if self.issues:
            lines.append("  Issues:")
            for issue in self.issues[:5]:
                lines.append(f"    {issue}")
        
        return "\n".join(lines)


def run_corp_action_audit(
    start_date: date,
    end_date: date,
    tickers: Optional[List[str]] = None,
) -> CorpActionAuditResult:
    """
    Audit price data for corporate action handling.
    
    Checks:
    1. Splits are reflected in adjusted prices
    2. Price jumps coincide with known corporate actions
    3. Volume is adjusted consistently with price
    4. No unexplained large price discontinuities
    
    Args:
        start_date: Start of audit period
        end_date: End of audit period
        tickers: Specific tickers to audit (None = all)
    
    Returns:
        CorpActionAuditResult with findings
    """
    import time
    start_time = time.time()
    
    logger.info(f"Starting corporate action audit: {start_date} to {end_date}")
    
    issues = []
    actions_found = 0
    tickers_checked = 0
    
    # TODO: Implement actual audit
    # This requires:
    # 1. Load price data and known corporate actions
    # 2. For each split, verify price adjusted correctly
    # 3. For each dividend, verify ex-date handling
    # 4. Flag unexplained >20% overnight moves
    # 5. Verify adjustment factors are consistent
    
    logger.warning("Corporate action audit not yet fully implemented")
    
    duration = time.time() - start_time
    
    return CorpActionAuditResult(
        start_date=start_date,
        end_date=end_date,
        tickers_checked=tickers_checked,
        actions_found=actions_found,
        issues=issues,
        duration_seconds=duration,
    )


def detect_price_discontinuities(
    prices: List[float],
    dates: List[date],
    threshold: float = 0.20,
) -> List[Dict]:
    """
    Detect large price jumps that might indicate corporate actions.
    
    Args:
        prices: List of adjusted close prices
        dates: Corresponding dates
        threshold: Minimum overnight return to flag (default 20%)
    
    Returns:
        List of dicts with discontinuity details
    """
    discontinuities = []
    
    for i in range(1, len(prices)):
        if prices[i-1] == 0:
            continue
        
        ret = (prices[i] - prices[i-1]) / prices[i-1]
        
        if abs(ret) >= threshold:
            discontinuities.append({
                "date": dates[i],
                "prev_date": dates[i-1],
                "prev_price": prices[i-1],
                "price": prices[i],
                "return": ret,
            })
    
    return discontinuities


def validate_split_adjustment(
    pre_split_price: float,
    post_split_price: float,
    split_ratio: float,
    tolerance: float = 0.01,
) -> bool:
    """
    Validate that a split was correctly applied.
    
    Args:
        pre_split_price: Adjusted price before split
        post_split_price: Adjusted price after split
        split_ratio: Split ratio (e.g., 4.0 for 4:1 split)
        tolerance: Allowed error percentage
    
    Returns:
        True if adjustment is correct
    """
    expected_ratio = pre_split_price / post_split_price
    actual_ratio = split_ratio
    
    error = abs(expected_ratio - actual_ratio) / actual_ratio
    return error <= tolerance


"""
Point-in-Time (PIT) Scanner
===========================

Detects lookahead bias by validating that all data points were actually
available at the time they claim to be available.

Key Rules:
- Fundamentals cannot be used before their filing/announcement date
- Price data cannot be used before market close
- Events cannot be used before their occurrence timestamp
"""

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import List, Dict, Optional, Any
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ViolationType(Enum):
    """Types of PIT violations."""
    FUTURE_FUNDAMENTAL = "future_fundamental"  # Using earnings before filed
    FUTURE_PRICE = "future_price"              # Using price before close
    MISSING_OBSERVED_AT = "missing_observed_at" # No timestamp for data point
    INCONSISTENT_DATES = "inconsistent_dates"  # effective_from > observed_at
    RESTATEMENT_LEAK = "restatement_leak"      # Using restated value before restatement


@dataclass(frozen=True)
class PITViolation:
    """A single point-in-time violation."""
    violation_type: ViolationType
    ticker: str
    feature_name: str
    feature_date: date           # Date the feature claims to be for
    observed_at: Optional[datetime]  # When data was actually available
    used_at: date                # When data was used in a prediction
    severity: str = "error"      # "error", "warning", "info"
    details: str = ""
    
    def __str__(self) -> str:
        return (
            f"[{self.severity.upper()}] {self.violation_type.value}: "
            f"{self.ticker}/{self.feature_name} for {self.feature_date}, "
            f"used at {self.used_at}, observed at {self.observed_at}"
        )


@dataclass
class PITAuditResult:
    """Result of a PIT audit."""
    start_date: date
    end_date: date
    total_checks: int
    violations: List[PITViolation]
    violations_by_type: Dict[str, int] = field(default_factory=dict)
    duration_seconds: float = 0.0
    
    @property
    def passed(self) -> bool:
        """Audit passes if no error-level violations."""
        return not any(v.severity == "error" for v in self.violations)
    
    @property
    def violation_rate(self) -> float:
        """Percentage of checks that found violations."""
        if self.total_checks == 0:
            return 0.0
        return 100 * len(self.violations) / self.total_checks
    
    def summary(self) -> str:
        status = "✅ PASSED" if self.passed else "❌ FAILED"
        lines = [
            f"{status}: PIT Audit",
            f"  Period: {self.start_date} to {self.end_date}",
            f"  Total checks: {self.total_checks:,}",
            f"  Violations: {len(self.violations)} ({self.violation_rate:.2f}%)",
            f"  Duration: {self.duration_seconds:.1f}s",
        ]
        
        if self.violations_by_type:
            lines.append("  By type:")
            for vtype, count in sorted(self.violations_by_type.items()):
                lines.append(f"    {vtype}: {count}")
        
        return "\n".join(lines)


def scan_feature_pit_violations(
    feature_name: str,
    feature_data: List[Dict[str, Any]],
    usage_dates: List[date],
) -> List[PITViolation]:
    """
    Scan a single feature for PIT violations.
    
    Args:
        feature_name: Name of the feature being checked
        feature_data: List of dicts with keys:
            - ticker: str
            - value: any
            - effective_from: date (when value is "for")
            - observed_at: datetime (when value became known)
        usage_dates: Dates when this feature was used
    
    Returns:
        List of detected violations
    """
    violations = []
    
    for record in feature_data:
        ticker = record.get("ticker", "UNKNOWN")
        effective_from = record.get("effective_from")
        observed_at = record.get("observed_at")
        
        # Check 1: Missing observed_at
        if observed_at is None:
            violations.append(PITViolation(
                violation_type=ViolationType.MISSING_OBSERVED_AT,
                ticker=ticker,
                feature_name=feature_name,
                feature_date=effective_from,
                observed_at=None,
                used_at=effective_from,
                severity="warning",
                details="No observed_at timestamp recorded",
            ))
            continue
        
        # Check 2: Inconsistent dates (effective_from > observed_at is wrong)
        if effective_from and observed_at:
            if effective_from > observed_at.date():
                violations.append(PITViolation(
                    violation_type=ViolationType.INCONSISTENT_DATES,
                    ticker=ticker,
                    feature_name=feature_name,
                    feature_date=effective_from,
                    observed_at=observed_at,
                    used_at=effective_from,
                    severity="error",
                    details=f"effective_from ({effective_from}) > observed_at ({observed_at.date()})",
                ))
        
        # Check 3: Usage before availability
        for usage_date in usage_dates:
            if observed_at and usage_date < observed_at.date():
                violations.append(PITViolation(
                    violation_type=ViolationType.FUTURE_FUNDAMENTAL,
                    ticker=ticker,
                    feature_name=feature_name,
                    feature_date=effective_from,
                    observed_at=observed_at,
                    used_at=usage_date,
                    severity="error",
                    details=f"Used on {usage_date} but not available until {observed_at.date()}",
                ))
    
    return violations


def run_pit_audit(
    start_date: date,
    end_date: date,
    tickers: Optional[List[str]] = None,
    feature_names: Optional[List[str]] = None,
) -> PITAuditResult:
    """
    Run a comprehensive PIT audit over a date range.
    
    This scans the feature store and validates that all features
    used at each date were actually available at that date.
    
    Args:
        start_date: Start of audit period
        end_date: End of audit period
        tickers: Specific tickers to audit (None = all)
        feature_names: Specific features to audit (None = all)
    
    Returns:
        PITAuditResult with all findings
    """
    import time
    start_time = time.time()
    
    logger.info(f"Starting PIT audit: {start_date} to {end_date}")
    
    violations = []
    total_checks = 0
    
    # TODO: Implement actual audit logic
    # This requires:
    # 1. Query feature store for all features in date range
    # 2. For each feature value, verify observed_at <= usage_date
    # 3. Special handling for fundamentals (lag rules)
    # 4. Log all violations with full context
    
    # Placeholder: In real implementation, would scan DuckDB
    logger.warning("PIT audit not yet fully implemented - awaiting data store")
    
    duration = time.time() - start_time
    
    # Aggregate violations by type
    violations_by_type = {}
    for v in violations:
        vtype = v.violation_type.value
        violations_by_type[vtype] = violations_by_type.get(vtype, 0) + 1
    
    return PITAuditResult(
        start_date=start_date,
        end_date=end_date,
        total_checks=total_checks,
        violations=violations,
        violations_by_type=violations_by_type,
        duration_seconds=duration,
    )


def validate_fundamental_pit(
    ticker: str,
    period_end: date,
    filing_date: date,
    usage_date: date,
    conservative_lag_days: int = 1,
) -> Optional[PITViolation]:
    """
    Validate that a fundamental data point respects PIT rules.
    
    FMP may not always provide exact filing times, so we apply
    a conservative lag rule: data is considered available
    filing_date + lag_days.
    
    Args:
        ticker: Stock ticker
        period_end: End of reporting period (e.g., 2023-12-31)
        filing_date: Date the filing was made
        usage_date: Date when this data is being used
        conservative_lag_days: Days to add to filing_date for safety
    
    Returns:
        PITViolation if violated, None otherwise
    """
    # Apply conservative lag
    available_date = filing_date + timedelta(days=conservative_lag_days)
    
    if usage_date < available_date:
        return PITViolation(
            violation_type=ViolationType.FUTURE_FUNDAMENTAL,
            ticker=ticker,
            feature_name="fundamental",
            feature_date=period_end,
            observed_at=datetime.combine(filing_date, datetime.min.time()),
            used_at=usage_date,
            severity="error",
            details=(
                f"Period ending {period_end}, filed {filing_date}, "
                f"available {available_date}, used {usage_date}"
            ),
        )
    
    return None


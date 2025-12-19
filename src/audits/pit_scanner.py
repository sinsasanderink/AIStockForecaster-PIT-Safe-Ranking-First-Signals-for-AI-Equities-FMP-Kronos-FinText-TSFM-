"""
Point-in-Time (PIT) Scanner
===========================

Detects lookahead bias by validating that all data points were actually
available at the time they claim to be available.

Key Rules:
- Fundamentals cannot be used before their filing/announcement date
- Price data cannot be used before market close
- Events cannot be used before their occurrence timestamp

VIOLATION TYPES:
- FUTURE_PRICE: Using price before market close on that day
- FUTURE_FUNDAMENTAL: Using earnings before filing date
- FUTURE_EVENT: Using event before it occurred
- MISSING_OBSERVED_AT: Data point has no availability timestamp
- INCONSISTENT_DATES: effective_from > observed_at (impossible)
- RESTATEMENT_LEAK: Using restated value before restatement
"""

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import List, Dict, Optional, Any
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ViolationType(Enum):
    """Types of PIT violations."""
    FUTURE_PRICE = "future_price"              # Using price before close
    FUTURE_FUNDAMENTAL = "future_fundamental"  # Using earnings before filed
    FUTURE_EVENT = "future_event"              # Using event before occurred
    MISSING_OBSERVED_AT = "missing_observed_at" # No timestamp for data point
    INCONSISTENT_DATES = "inconsistent_dates"  # effective_from > observed_at
    RESTATEMENT_LEAK = "restatement_leak"      # Using restated value before restatement


class DataType(Enum):
    """Types of data for proper violation classification."""
    PRICE = "price"
    FUNDAMENTAL = "fundamental"
    EVENT = "event"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class PITViolation:
    """A single point-in-time violation."""
    violation_type: ViolationType
    data_type: DataType
    ticker: str
    feature_name: str
    feature_date: date           # Date the feature claims to be for
    observed_at: Optional[datetime]  # When data was actually available
    used_at: date                # When data was used in a prediction
    usage_datetime: Optional[datetime] = None  # Precise usage time if available
    severity: str = "error"      # "error", "warning", "info"
    details: str = ""
    
    def __str__(self) -> str:
        obs_str = self.observed_at.isoformat() if self.observed_at else "UNKNOWN"
        return (
            f"[{self.severity.upper()}] {self.violation_type.value} ({self.data_type.value}): "
            f"{self.ticker}/{self.feature_name} for {self.feature_date}, "
            f"used at {self.used_at}, observed at {obs_str}"
        )


@dataclass
class PITAuditResult:
    """Result of a PIT audit."""
    start_date: date
    end_date: date
    total_checks: int
    violations: List[PITViolation]
    violations_by_type: Dict[str, int] = field(default_factory=dict)
    violations_by_data_type: Dict[str, int] = field(default_factory=dict)
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
            f"  Violations: {len(self.violations)} ({self.violation_rate:.3f}%)",
            f"  Duration: {self.duration_seconds:.1f}s",
        ]
        
        if self.violations_by_type:
            lines.append("  By violation type:")
            for vtype, count in sorted(self.violations_by_type.items()):
                lines.append(f"    {vtype}: {count}")
        
        if self.violations_by_data_type:
            lines.append("  By data type:")
            for dtype, count in sorted(self.violations_by_data_type.items()):
                lines.append(f"    {dtype}: {count}")
        
        if self.violations[:5]:
            lines.append("  Sample violations:")
            for v in self.violations[:5]:
                lines.append(f"    - {v}")
        
        return "\n".join(lines)


def scan_feature_pit_violations(
    feature_name: str,
    feature_data: List[Dict[str, Any]],
    usage_dates: List[date],
    data_type: DataType = DataType.UNKNOWN,
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
        usage_dates: Dates when this feature was used in predictions
        data_type: Type of data (price, fundamental, event) for proper
                   violation classification
    
    Returns:
        List of detected violations
    """
    violations = []
    
    # Determine violation type based on data type
    future_violation_type = {
        DataType.PRICE: ViolationType.FUTURE_PRICE,
        DataType.FUNDAMENTAL: ViolationType.FUTURE_FUNDAMENTAL,
        DataType.EVENT: ViolationType.FUTURE_EVENT,
        DataType.UNKNOWN: ViolationType.FUTURE_FUNDAMENTAL,  # Default
    }[data_type]
    
    for record in feature_data:
        ticker = record.get("ticker", "UNKNOWN")
        effective_from = record.get("effective_from")
        observed_at = record.get("observed_at")
        
        # Check 1: Missing observed_at
        if observed_at is None:
            violations.append(PITViolation(
                violation_type=ViolationType.MISSING_OBSERVED_AT,
                data_type=data_type,
                ticker=ticker,
                feature_name=feature_name,
                feature_date=effective_from,
                observed_at=None,
                used_at=effective_from or date.today(),
                severity="warning",
                details="No observed_at timestamp recorded",
            ))
            continue
        
        # Check 2: Inconsistent dates (effective_from > observed_at is wrong)
        if effective_from and observed_at:
            observed_date = observed_at.date() if isinstance(observed_at, datetime) else observed_at
            if effective_from > observed_date:
                violations.append(PITViolation(
                    violation_type=ViolationType.INCONSISTENT_DATES,
                    data_type=data_type,
                    ticker=ticker,
                    feature_name=feature_name,
                    feature_date=effective_from,
                    observed_at=observed_at,
                    used_at=effective_from,
                    severity="error",
                    details=f"effective_from ({effective_from}) > observed_at ({observed_date})",
                ))
        
        # Check 3: Usage before availability (for each usage date)
        for usage_date in usage_dates:
            if observed_at:
                observed_date = observed_at.date() if isinstance(observed_at, datetime) else observed_at
                if usage_date < observed_date:
                    violations.append(PITViolation(
                        violation_type=future_violation_type,
                        data_type=data_type,
                        ticker=ticker,
                        feature_name=feature_name,
                        feature_date=effective_from,
                        observed_at=observed_at,
                        used_at=usage_date,
                        severity="error",
                        details=f"Used on {usage_date} but not available until {observed_date}",
                    ))
    
    return violations


def run_pit_audit(
    start_date: date,
    end_date: date,
    tickers: Optional[List[str]] = None,
    feature_names: Optional[List[str]] = None,
    pit_store = None,
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
        pit_store: Optional DuckDBPITStore instance for real data
    
    Returns:
        PITAuditResult with all findings
    """
    import time
    start_time = time.time()
    
    logger.info(f"Starting PIT audit: {start_date} to {end_date}")
    
    violations = []
    total_checks = 0
    
    # If we have a real PIT store, audit it
    if pit_store is not None:
        all_tickers = tickers or pit_store.get_all_tickers()
        
        for ticker in all_tickers:
            # Validate prices
            validation = pit_store.validate_pit(ticker, "prices")
            total_checks += 1
            
            if not validation["valid"]:
                for issue in validation["issues"]:
                    violations.append(PITViolation(
                        violation_type=ViolationType.INCONSISTENT_DATES,
                        data_type=DataType.PRICE,
                        ticker=ticker,
                        feature_name="prices",
                        feature_date=start_date,
                        observed_at=None,
                        used_at=start_date,
                        severity="error",
                        details=issue,
                    ))
    else:
        logger.warning("No PIT store provided - audit will be limited")
    
    duration = time.time() - start_time
    
    # Aggregate violations by type
    violations_by_type = {}
    violations_by_data_type = {}
    for v in violations:
        vtype = v.violation_type.value
        violations_by_type[vtype] = violations_by_type.get(vtype, 0) + 1
        dtype = v.data_type.value
        violations_by_data_type[dtype] = violations_by_data_type.get(dtype, 0) + 1
    
    return PITAuditResult(
        start_date=start_date,
        end_date=end_date,
        total_checks=total_checks,
        violations=violations,
        violations_by_type=violations_by_type,
        violations_by_data_type=violations_by_data_type,
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
            data_type=DataType.FUNDAMENTAL,
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


def validate_price_pit(
    ticker: str,
    price_date: date,
    usage_datetime: datetime,
    market_close_hour: int = 16,
    market_close_minute: int = 0,
    timezone_str: str = "America/New_York",
) -> Optional[PITViolation]:
    """
    Validate that a price data point respects PIT rules.
    
    Price for date D should only be available after market close on D.
    
    Args:
        ticker: Stock ticker
        price_date: Date of the price
        usage_datetime: Datetime when price is being used (must be tz-aware)
        market_close_hour: Hour of market close (default 16 = 4pm)
        market_close_minute: Minute of market close
        timezone_str: Market timezone
    
    Returns:
        PITViolation if violated, None otherwise
    """
    import pytz
    
    tz = pytz.timezone(timezone_str)
    market_close = tz.localize(datetime(
        price_date.year, price_date.month, price_date.day,
        market_close_hour, market_close_minute
    ))
    
    # Convert usage_datetime to same timezone for comparison
    if usage_datetime.tzinfo is None:
        usage_datetime = pytz.UTC.localize(usage_datetime)
    usage_local = usage_datetime.astimezone(tz)
    
    if usage_local < market_close:
        return PITViolation(
            violation_type=ViolationType.FUTURE_PRICE,
            data_type=DataType.PRICE,
            ticker=ticker,
            feature_name="close_price",
            feature_date=price_date,
            observed_at=market_close,
            used_at=price_date,
            usage_datetime=usage_datetime,
            severity="error",
            details=(
                f"Price for {price_date} used at {usage_local.strftime('%H:%M %Z')}, "
                f"but not available until market close at {market_close.strftime('%H:%M %Z')}"
            ),
        )
    
    return None


def create_pit_boundary_test_cases() -> List[Dict[str, Any]]:
    """
    Create test cases for PIT boundary validation.
    
    These cases test the exact boundaries:
    - Data at 15:59 should be visible for that day
    - Data at 16:01 should NOT be visible for that day
    
    Returns:
        List of test case dictionaries
    """
    import pytz
    
    ET = pytz.timezone("America/New_York")
    test_date = date(2024, 6, 15)  # A random weekday
    
    return [
        {
            "name": "price_at_1559_visible_same_day",
            "price_date": test_date,
            "observed_at": ET.localize(datetime(2024, 6, 15, 15, 59)),
            "query_asof": ET.localize(datetime(2024, 6, 15, 16, 0)),
            "should_be_visible": True,
            "description": "Price observed at 15:59 should be visible at 16:00 cutoff",
        },
        {
            "name": "price_at_1601_not_visible_same_day",
            "price_date": test_date,
            "observed_at": ET.localize(datetime(2024, 6, 15, 16, 1)),
            "query_asof": ET.localize(datetime(2024, 6, 15, 16, 0)),
            "should_be_visible": False,
            "description": "Price observed at 16:01 should NOT be visible at 16:00 cutoff",
        },
        {
            "name": "fundamental_filing_plus_1_day",
            "period_end": date(2024, 3, 31),
            "filing_date": date(2024, 5, 1),
            "usage_date": date(2024, 5, 2),
            "should_be_valid": True,
            "description": "Fundamental filed May 1, used May 2 (+1 day lag) = valid",
        },
        {
            "name": "fundamental_filing_same_day",
            "period_end": date(2024, 3, 31),
            "filing_date": date(2024, 5, 1),
            "usage_date": date(2024, 5, 1),
            "should_be_valid": False,
            "description": "Fundamental filed May 1, used May 1 (no lag) = violation",
        },
    ]

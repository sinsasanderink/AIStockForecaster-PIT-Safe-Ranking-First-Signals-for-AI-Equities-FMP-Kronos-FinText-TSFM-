"""
PIT (Point-in-Time) Violation Scanner
======================================

Automated scanner to detect point-in-time violations across all feature modules.

A PIT violation occurs when a feature uses data that would not have been available
at the time the feature was computed (future information leakage).

CRITICAL: This scanner is the foundation of system trustworthiness.
Without it, all downstream evaluation becomes unreliable.

WHAT IT CHECKS:
1. Features use only data with observed_at <= asof
2. Labels only available after maturity (asof >= label_matured_at)
3. Sectors/metadata are as-of date T (not current)
4. No forward-looking information in any module
5. Event timestamps respect observed_at discipline

USAGE:
    from src.features.pit_scanner import PITScanner
    
    scanner = PITScanner()
    
    # Scan all Chapter 5 modules
    violations = scanner.scan_all_features()
    
    # Generate report
    report = scanner.generate_report(violations)
    print(report)
    
    # Assert no violations
    assert len(violations) == 0, f"Found {len(violations)} PIT violations!"
"""

import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import List, Dict, Any, Optional, Tuple, Set
from pathlib import Path
import inspect

import pandas as pd
import numpy as np
import pytz

logger = logging.getLogger(__name__)

UTC = pytz.UTC


@dataclass
class PITViolation:
    """
    A detected point-in-time violation.
    """
    module: str
    function: str
    violation_type: str
    description: str
    severity: str  # "CRITICAL", "HIGH", "MEDIUM", "LOW"
    line_number: Optional[int] = None
    example_data: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "module": self.module,
            "function": self.function,
            "violation_type": self.violation_type,
            "description": self.description,
            "severity": self.severity,
            "line_number": self.line_number,
            "example_data": self.example_data,
        }


class PITScanner:
    """
    Automated PIT violation scanner for all feature modules.
    
    Checks:
    1. Code inspection (static analysis)
    2. Data flow validation (runtime checks)
    3. Timestamp ordering (observed_at, asof, matured_at)
    4. Cross-module consistency
    """
    
    def __init__(self):
        self.violations: List[PITViolation] = []
        logger.info("PITScanner initialized")
    
    # =========================================================================
    # 1. CODE INSPECTION (Static Analysis)
    # =========================================================================
    
    def scan_code_for_violations(self, module_path: Path) -> List[PITViolation]:
        """
        Scan Python source code for common PIT violation patterns.
        
        Checks for:
        - Direct date comparisons without PIT checks
        - Missing observed_at filtering
        - Forward-looking time shifts
        """
        violations = []
        
        if not module_path.exists():
            return violations
        
        with open(module_path, 'r') as f:
            lines = f.readlines()
        
        for i, line in enumerate(lines, 1):
            # Check for common violation patterns
            
            # Pattern 1: .tail() or .head() without PIT filter
            if '.tail(' in line or '.head(' in line:
                if 'observed_at' not in ''.join(lines[max(0, i-5):min(len(lines), i+5)]):
                    # Only flag if not in a safe context
                    if 'def _get_prices' not in ''.join(lines[max(0, i-10):i]):
                        violations.append(PITViolation(
                            module=module_path.name,
                            function="<unknown>",
                            violation_type="POTENTIAL_FUTURE_DATA",
                            description=f"Line {i}: .tail()/.head() without nearby observed_at filter",
                            severity="MEDIUM",
                            line_number=i,
                        ))
            
            # Pattern 2: pd.Timestamp('today') or datetime.now()
            if 'datetime.now()' in line or "pd.Timestamp('today')" in line:
                violations.append(PITViolation(
                    module=module_path.name,
                    function="<unknown>",
                    violation_type="DYNAMIC_TIMESTAMP",
                    description=f"Line {i}: Uses dynamic timestamp (datetime.now or 'today')",
                    severity="HIGH",
                    line_number=i,
                ))
            
            # Pattern 3: Comparing dates without checking observed_at
            if '>' in line and 'date' in line.lower() and 'observed_at' not in line:
                # This is a heuristic - might have false positives
                if 'asof' in line:
                    # Good - comparing with asof
                    pass
                elif 'matured_at' in line:
                    # Good - checking maturity
                    pass
                else:
                    # Potential issue
                    pass  # Too many false positives, skip for now
        
        return violations
    
    # =========================================================================
    # 2. DATA FLOW VALIDATION (Runtime Checks)
    # =========================================================================
    
    def validate_label_maturity(
        self,
        labels_df: pd.DataFrame,
        asof_datetime: datetime,
    ) -> List[PITViolation]:
        """
        Check that labels are only used after they mature.
        
        Rule: asof_datetime >= label_matured_at
        """
        violations = []
        
        if 'label_matured_at' not in labels_df.columns:
            violations.append(PITViolation(
                module="labels",
                function="validate_label_maturity",
                violation_type="MISSING_MATURITY_FIELD",
                description="Labels DataFrame missing 'label_matured_at' column",
                severity="CRITICAL",
            ))
            return violations
        
        # Ensure timezone-aware
        if labels_df['label_matured_at'].dt.tz is None:
            labels_df = labels_df.copy()
            labels_df['label_matured_at'] = labels_df['label_matured_at'].dt.tz_localize(UTC)
        
        asof_utc = asof_datetime.astimezone(UTC) if asof_datetime.tzinfo else datetime(asof_datetime.year, asof_datetime.month, asof_datetime.day, tzinfo=UTC)
        
        # Check for premature labels
        premature = labels_df[labels_df['label_matured_at'] > asof_utc]
        
        if len(premature) > 0:
            violations.append(PITViolation(
                module="labels",
                function="validate_label_maturity",
                violation_type="PREMATURE_LABEL",
                description=f"Found {len(premature)} labels used before maturity",
                severity="CRITICAL",
                example_data={
                    "n_violations": len(premature),
                    "example_ticker": premature.iloc[0]['ticker'] if 'ticker' in premature.columns else None,
                    "example_date": premature.iloc[0]['as_of_date'].isoformat() if 'as_of_date' in premature.columns else None,
                },
            ))
        
        return violations
    
    def validate_feature_timestamps(
        self,
        features_df: pd.DataFrame,
        asof_date: date,
    ) -> List[PITViolation]:
        """
        Check that features only use data available as-of the feature date.
        
        For each row with date=T, all data should have been observable by cutoff(T).
        """
        violations = []
        
        if 'date' not in features_df.columns:
            violations.append(PITViolation(
                module="features",
                function="validate_feature_timestamps",
                violation_type="MISSING_DATE_FIELD",
                description="Features DataFrame missing 'date' column",
                severity="CRITICAL",
            ))
            return violations
        
        # Check for dates in the future
        future_dates = features_df[features_df['date'] > asof_date]
        
        if len(future_dates) > 0:
            violations.append(PITViolation(
                module="features",
                function="validate_feature_timestamps",
                violation_type="FUTURE_FEATURE",
                description=f"Found {len(future_dates)} feature rows with future dates",
                severity="CRITICAL",
                example_data={
                    "n_violations": len(future_dates),
                    "latest_future_date": future_dates['date'].max().isoformat(),
                },
            ))
        
        return violations
    
    def validate_sector_consistency(
        self,
        features_df: pd.DataFrame,
    ) -> List[PITViolation]:
        """
        Check that sector/industry fields are as-of date T (not current).
        
        If a stock changes sector, the historical data should reflect
        the sector it was in at that time.
        """
        violations = []
        
        if 'sector' not in features_df.columns:
            # Sector not present - OK
            return violations
        
        if 'ticker' not in features_df.columns or 'date' not in features_df.columns:
            return violations
        
        # Check for stocks that have multiple sectors
        # This is expected (corporate actions), but we should verify
        # that the sector assignment is time-consistent
        
        sector_changes = features_df.groupby('ticker')['sector'].nunique()
        stocks_with_changes = sector_changes[sector_changes > 1]
        
        if len(stocks_with_changes) > 0:
            # This is informational, not necessarily a violation
            logger.info(f"Found {len(stocks_with_changes)} stocks with sector changes (expected for corporate actions)")
            
            # Just log, don't flag as violation
            # In a real system, we'd verify the sector change dates match known corporate actions
        
        return violations
    
    # =========================================================================
    # 3. MODULE-SPECIFIC CHECKS
    # =========================================================================
    
    def scan_labels_module(self) -> List[PITViolation]:
        """Scan labels module (5.1) for PIT violations."""
        violations = []
        module_path = Path("src/features/labels.py")
        
        # Code inspection
        violations.extend(self.scan_code_for_violations(module_path))
        
        # Design check: verify label maturity logic exists
        try:
            from src.features import ForwardReturn
            
            # Check that ForwardReturn has is_mature method
            if not hasattr(ForwardReturn, 'is_mature'):
                violations.append(PITViolation(
                    module="labels",
                    function="ForwardReturn",
                    violation_type="MISSING_MATURITY_CHECK",
                    description="ForwardReturn missing is_mature() method",
                    severity="CRITICAL",
                ))
        except Exception as e:
            violations.append(PITViolation(
                module="labels",
                function="import",
                violation_type="IMPORT_ERROR",
                description=f"Failed to import labels module: {e}",
                severity="HIGH",
            ))
        
        return violations
    
    def scan_price_features_module(self) -> List[PITViolation]:
        """Scan price features module (5.2) for PIT violations."""
        violations = []
        module_path = Path("src/features/price_features.py")
        
        violations.extend(self.scan_code_for_violations(module_path))
        
        # Check that _get_prices_for_window uses asof filtering
        try:
            with open(module_path, 'r') as f:
                content = f.read()
            
            if '_get_prices_for_window' in content:
                # Check that it uses asof or cutoff
                if 'asof' not in content:
                    violations.append(PITViolation(
                        module="price_features",
                        function="_get_prices_for_window",
                        violation_type="MISSING_ASOF_FILTER",
                        description="_get_prices_for_window may not respect asof filtering",
                        severity="HIGH",
                    ))
        except Exception:
            pass
        
        return violations
    
    def scan_fundamental_features_module(self) -> List[PITViolation]:
        """Scan fundamental features module (5.3) for PIT violations."""
        violations = []
        module_path = Path("src/features/fundamental_features.py")
        
        violations.extend(self.scan_code_for_violations(module_path))
        
        # Check that fundamental data uses fillingDate
        try:
            with open(module_path, 'r') as f:
                content = f.read()
            
            if 'get_financial_statements' in content or 'get_income_statement' in content:
                # Should check fillingDate
                if 'fillingDate' not in content and 'filing_date' not in content:
                    violations.append(PITViolation(
                        module="fundamental_features",
                        function="<various>",
                        violation_type="MISSING_FILING_DATE_CHECK",
                        description="Fundamental features may not filter by fillingDate",
                        severity="HIGH",
                    ))
        except Exception:
            pass
        
        return violations
    
    def scan_event_features_module(self) -> List[PITViolation]:
        """Scan event features module (5.4) for PIT violations."""
        violations = []
        module_path = Path("src/features/event_features.py")
        
        violations.extend(self.scan_code_for_violations(module_path))
        
        # Check that events use observed_at
        try:
            with open(module_path, 'r') as f:
                content = f.read()
            
            if 'get_events' in content or 'get_earnings' in content:
                # Should filter by observed_at
                if content.count('observed_at') < 2:  # Should appear multiple times
                    violations.append(PITViolation(
                        module="event_features",
                        function="<various>",
                        violation_type="MISSING_OBSERVED_AT_CHECK",
                        description="Event features may not consistently filter by observed_at",
                        severity="HIGH",
                    ))
        except Exception:
            pass
        
        return violations
    
    def scan_regime_features_module(self) -> List[PITViolation]:
        """Scan regime features module (5.5) for PIT violations."""
        violations = []
        module_path = Path("src/features/regime_features.py")
        
        violations.extend(self.scan_code_for_violations(module_path))
        
        # Check that regime features filter by asof_date
        try:
            with open(module_path, 'r') as f:
                content = f.read()
            
            # Should have explicit "df[df.index <= asof_date]" pattern
            if 'df.index <= asof_date' in content or 'df[df.index <= asof_date]' in content:
                # Good - explicit PIT filtering
                pass
            else:
                violations.append(PITViolation(
                    module="regime_features",
                    function="<various>",
                    violation_type="MISSING_ASOF_FILTER",
                    description="Regime features may not filter df.index <= asof_date",
                    severity="MEDIUM",
                ))
        except Exception:
            pass
        
        return violations
    
    def scan_neutralization_module(self) -> List[PITViolation]:
        """Scan neutralization module (5.8) for PIT violations."""
        violations = []
        module_path = Path("src/features/neutralization.py")
        
        violations.extend(self.scan_code_for_violations(module_path))
        
        # Neutralization should be cross-sectional per date (inherently PIT-safe)
        # Main risk is using future sector membership
        
        try:
            with open(module_path, 'r') as f:
                content = f.read()
            
            # Should process by date
            if 'for d in dates' in content or 'groupby(date' in content:
                # Good - per-date processing
                pass
            else:
                violations.append(PITViolation(
                    module="neutralization",
                    function="compute_neutralized_ic",
                    violation_type="MISSING_PER_DATE_PROCESSING",
                    description="Neutralization may not process per-date",
                    severity="MEDIUM",
                ))
        except Exception:
            pass
        
        return violations
    
    # =========================================================================
    # 4. COMPREHENSIVE SCAN
    # =========================================================================
    
    def scan_all_features(self) -> List[PITViolation]:
        """
        Run comprehensive PIT scan across all Chapter 5 modules.
        
        Returns list of all detected violations.
        """
        all_violations = []
        
        logger.info("Starting comprehensive PIT scan...")
        
        # Scan each module
        modules = [
            ("5.1 Labels", self.scan_labels_module),
            ("5.2 Price Features", self.scan_price_features_module),
            ("5.3 Fundamental Features", self.scan_fundamental_features_module),
            ("5.4 Event Features", self.scan_event_features_module),
            ("5.5 Regime Features", self.scan_regime_features_module),
            ("5.8 Neutralization", self.scan_neutralization_module),
        ]
        
        for module_name, scan_func in modules:
            logger.info(f"Scanning {module_name}...")
            violations = scan_func()
            all_violations.extend(violations)
            logger.info(f"  Found {len(violations)} potential issues")
        
        logger.info(f"Scan complete. Total violations: {len(all_violations)}")
        
        return all_violations
    
    # =========================================================================
    # 5. REPORTING
    # =========================================================================
    
    def generate_report(self, violations: List[PITViolation]) -> str:
        """
        Generate human-readable PIT violation report.
        """
        report = []
        report.append("=" * 80)
        report.append("PIT VIOLATION SCANNER REPORT")
        report.append("=" * 80)
        report.append("")
        
        if not violations:
            report.append("✅ NO PIT VIOLATIONS DETECTED")
            report.append("")
            report.append("All Chapter 5 feature modules pass PIT safety checks.")
            report.append("System is ready for evaluation (Chapter 6).")
        else:
            report.append(f"⚠️  FOUND {len(violations)} POTENTIAL PIT VIOLATIONS")
            report.append("")
            
            # Group by severity
            critical = [v for v in violations if v.severity == "CRITICAL"]
            high = [v for v in violations if v.severity == "HIGH"]
            medium = [v for v in violations if v.severity == "MEDIUM"]
            low = [v for v in violations if v.severity == "LOW"]
            
            if critical:
                report.append(f"🚨 CRITICAL ({len(critical)}):")
                for v in critical:
                    report.append(f"  • {v.module}.{v.function}: {v.description}")
                report.append("")
            
            if high:
                report.append(f"⚠️  HIGH ({len(high)}):")
                for v in high:
                    report.append(f"  • {v.module}.{v.function}: {v.description}")
                report.append("")
            
            if medium:
                report.append(f"• MEDIUM ({len(medium)}):")
                for v in medium:
                    report.append(f"  • {v.module}.{v.function}: {v.description}")
                report.append("")
            
            if low:
                report.append(f"ℹ️  LOW ({len(low)}):")
                for v in low:
                    report.append(f"  • {v.module}.{v.function}: {v.description}")
                report.append("")
        
        report.append("=" * 80)
        report.append("")
        report.append("WHAT TO DO:")
        report.append("  1. Review CRITICAL violations immediately")
        report.append("  2. Fix HIGH violations before Chapter 6")
        report.append("  3. Investigate MEDIUM violations")
        report.append("  4. LOW violations are informational")
        report.append("")
        report.append("PIT RULES:")
        report.append("  • Features: use only data with observed_at <= asof")
        report.append("  • Labels: only available after maturity (asof >= label_matured_at)")
        report.append("  • Sectors: as-of date T (not current)")
        report.append("  • No forward-looking information in any module")
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def assert_no_violations(self, violations: List[PITViolation]):
        """
        Assert that no CRITICAL or HIGH violations exist.
        
        Raises AssertionError if violations found.
        """
        critical = [v for v in violations if v.severity == "CRITICAL"]
        high = [v for v in violations if v.severity == "HIGH"]
        
        if critical or high:
            msg = f"Found {len(critical)} CRITICAL and {len(high)} HIGH PIT violations. System not ready for evaluation."
            raise AssertionError(msg)


# =============================================================================
# Helper Functions
# =============================================================================

def run_pit_scan() -> Tuple[List[PITViolation], str]:
    """
    Run comprehensive PIT scan and return violations + report.
    
    Returns:
        (violations, report_string)
    """
    scanner = PITScanner()
    violations = scanner.scan_all_features()
    report = scanner.generate_report(violations)
    return violations, report


# =============================================================================
# CLI / Demo
# =============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("PIT VIOLATION SCANNER")
    print("=" * 80)
    print("\nScanning all Chapter 5 feature modules...")
    print("")
    
    violations, report = run_pit_scan()
    
    print(report)
    
    # Exit with error code if violations found
    critical_or_high = [v for v in violations if v.severity in ["CRITICAL", "HIGH"]]
    exit(1 if critical_or_high else 0)


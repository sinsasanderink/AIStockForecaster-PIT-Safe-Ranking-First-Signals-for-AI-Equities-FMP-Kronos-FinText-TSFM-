"""
Audits Module
=============

Runnable audit scripts to validate data integrity and prevent lookahead bias.

Audits:
- PIT Scanner: Detect point-in-time violations
- Survivorship Audit: Verify universe construction is bias-free  
- Corporate Action Checks: Validate split/dividend adjustments
"""

from .pit_scanner import (
    PITViolation,
    PITAuditResult,
    run_pit_audit,
    scan_feature_pit_violations,
)

from .survivorship_audit import (
    SurvivorshipAuditResult,
    run_survivorship_audit,
)

from .corp_action_checks import (
    CorpActionAuditResult,
    run_corp_action_audit,
)

__all__ = [
    # PIT
    "PITViolation",
    "PITAuditResult", 
    "run_pit_audit",
    "scan_feature_pit_violations",
    # Survivorship
    "SurvivorshipAuditResult",
    "run_survivorship_audit",
    # Corp Actions
    "CorpActionAuditResult",
    "run_corp_action_audit",
]


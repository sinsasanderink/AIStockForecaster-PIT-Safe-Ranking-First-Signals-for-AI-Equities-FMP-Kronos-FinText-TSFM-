"""
CHAPTER 6 DEFINITION LOCK — SINGLE SOURCE OF TRUTH

This file contains the canonical definitions for all evaluation parameters.
All other documentation and code MUST reference this file, not define their own.

CRITICAL: Do not modify these definitions without updating all dependent code.
Any change here should trigger a full re-evaluation.

Last Updated: December 29, 2025
Frozen for Chapter 6 Evaluation.
"""

from dataclasses import dataclass
from datetime import date, datetime, timezone, time
from typing import List, NamedTuple
import pytz

# ============================================================================
# TIME CONVENTIONS (LOCKED)
# ============================================================================

# Trading days per year (NYSE)
TRADING_DAYS_PER_YEAR = 252

# Calendar to trading day ratio (approximate)
# 365 calendar days / 252 trading days ≈ 1.45
CALENDAR_TO_TRADING_RATIO = 365 / 252  # ≈ 1.449

# CRITICAL: All horizons are in TRADING DAYS
# Do NOT interpret these as calendar days
HORIZONS_TRADING_DAYS = [20, 60, 90]

# Market close time (Eastern Time)
# This is when a trading day "ends" and prices are finalized
MARKET_CLOSE_ET = time(16, 0, 0)  # 4:00 PM ET
TIMEZONE_ET = pytz.timezone("US/Eastern")
TIMEZONE_UTC = pytz.UTC


@dataclass(frozen=True)
class TimeConventions:
    """
    CANONICAL TIME CONVENTIONS — LOCKED
    
    These define how we interpret all time-related parameters.
    Reference this, do not duplicate.
    """
    
    # Horizons are TRADING DAYS (not calendar days)
    horizons_trading_days: tuple = (20, 60, 90)
    
    # Embargo is TRADING DAYS (not calendar days)
    # Must be >= max horizon to prevent label leakage
    embargo_trading_days: int = 90
    
    # Rebalance occurs on first trading day of period
    rebalance_rule: str = "first_trading_day_of_month"
    
    # Prices used for labels: close-to-close
    # Label at T+H uses: close(T+H) / close(T) - benchmark
    pricing_convention: str = "close_to_close"
    
    # Market close time when a trading day "ends"
    market_close_time: time = MARKET_CLOSE_ET
    market_timezone: str = "US/Eastern"
    
    # Label maturity: A label at T with horizon H matures at T+H market close
    # It can ONLY be used for training/validation AFTER maturity
    maturity_rule: str = "label_matured_at_market_close_utc"


# Singleton instance
TIME_CONVENTIONS = TimeConventions()


# ============================================================================
# EVALUATION DATE RANGE (LOCKED)
# ============================================================================

@dataclass(frozen=True)
class EvaluationRange:
    """
    CANONICAL EVALUATION DATE RANGE — LOCKED
    
    These dates define the evaluation period.
    """
    
    # First as-of date for evaluation
    # Chosen for: sufficient data quality, universe coverage, fundamentals
    eval_start: date = date(2016, 1, 1)
    
    # Last as-of date for evaluation (MUST allow 90d labels to mature)
    # This means actual "usable" date is eval_end minus ~130 calendar days
    eval_end: date = date(2025, 6, 30)
    
    # Minimum training history before first fold (calendar days)
    min_train_days: int = 730  # ~2 years


EVALUATION_RANGE = EvaluationRange()


# ============================================================================
# EMBARGO & PURGING RULES (LOCKED)
# ============================================================================

@dataclass(frozen=True)
class EmbargoRules:
    """
    CANONICAL EMBARGO & PURGING RULES — LOCKED
    
    CRITICAL: These are HARD CONSTRAINTS, not guidelines.
    Code MUST enforce these rules and FAIL LOUDLY if violated.
    """
    
    # Embargo period in TRADING DAYS (not calendar days)
    # Must be >= max horizon to prevent ANY label overlap
    embargo_trading_days: int = 90
    
    # Purging rule for training labels:
    # A training label at date T with horizon H is PURGED if:
    #   T + H (trading days) > train_end
    # This prevents training labels whose forward window extends into embargo/val
    train_purge_rule: str = "purge_if_maturity_extends_past_train_end"
    
    # Purging rule for validation labels:
    # A validation label at date T with horizon H is PURGED if:
    #   T - H (trading days) < train_end
    # This prevents validation labels that "look back" into training period
    val_purge_rule: str = "purge_if_lookback_overlaps_train_period"
    
    # CRITICAL: Purging is PER-ROW-PER-HORIZON
    # Each (date, ticker, horizon) is evaluated independently
    # NOT a global rule that happens to work because embargo = max horizon
    purging_granularity: str = "per_row_per_horizon"


EMBARGO_RULES = EmbargoRules()


# ============================================================================
# END-OF-SAMPLE ELIGIBILITY RULE (LOCKED)
# ============================================================================

@dataclass(frozen=True)
class EligibilityRules:
    """
    CANONICAL END-OF-SAMPLE ELIGIBILITY RULES — LOCKED
    
    These rules define which as-of dates are ELIGIBLE for evaluation.
    This prevents "late sample distortion" where some horizons silently drop out.
    """
    
    # OPTION A (RECOMMENDED): All-horizons-valid rule
    # An as-of date T is ELIGIBLE if and only if:
    #   1. Labels exist for ALL horizons (20, 60, 90)
    #   2. ALL labels have matured (label_matured_at <= cutoff)
    #   3. ALL required fields are populated (including dividend yields for v2)
    # This means NO partial horizons near the end.
    eligibility_rule: str = "all_horizons_must_be_valid"
    
    # What makes a label "valid"?
    valid_label_criteria: tuple = (
        "label_exists",              # Row exists in labels table
        "label_matured",             # label_matured_at <= cutoff
        "required_fields_populated", # label, stock_dividend_yield (if v2), etc.
    )
    
    # Per-horizon grids are NOT allowed in Chapter 6
    # (i.e., no "20d present but 90d missing" scenarios)
    allow_partial_horizons: bool = False


ELIGIBILITY_RULES = EligibilityRules()


# ============================================================================
# LABEL MATURITY (UTC CLOSE)
# ============================================================================

def get_market_close_utc(as_of_date: date) -> datetime:
    """
    Get the UTC datetime for market close on a given date.
    
    This is the canonical cutoff for label maturity checks.
    A label is mature if label_matured_at <= get_market_close_utc(as_of_date)
    
    Args:
        as_of_date: The as-of date (in US/Eastern timezone)
        
    Returns:
        datetime in UTC for market close
    """
    # Create datetime at market close in Eastern Time
    et_close = datetime.combine(as_of_date, MARKET_CLOSE_ET)
    et_close = TIMEZONE_ET.localize(et_close)
    
    # Convert to UTC
    utc_close = et_close.astimezone(TIMEZONE_UTC)
    
    return utc_close


def is_label_mature(
    label_matured_at: datetime,
    cutoff_date: date
) -> bool:
    """
    Check if a label has matured by a given cutoff date.
    
    CANONICAL RULE: label_matured_at <= cutoff_close_utc
    
    Args:
        label_matured_at: When the label matured (must be timezone-aware UTC)
        cutoff_date: The as-of date for the fold
        
    Returns:
        True if label has matured, False otherwise
    """
    cutoff_utc = get_market_close_utc(cutoff_date)
    
    # Ensure label_matured_at is timezone-aware
    if label_matured_at.tzinfo is None:
        raise ValueError(
            f"label_matured_at must be timezone-aware UTC, got naive datetime: {label_matured_at}"
        )
    
    return label_matured_at <= cutoff_utc


# ============================================================================
# TRADING DAYS CONVERSION
# ============================================================================

def trading_days_to_calendar_days(trading_days: int) -> int:
    """
    Convert trading days to calendar days (CONSERVATIVE estimate).
    
    This is used when we don't have a trading calendar available.
    The result is CONSERVATIVE (slightly more calendar days than needed)
    to ensure we don't accidentally include overlapping labels.
    
    Args:
        trading_days: Number of trading days
        
    Returns:
        Approximate number of calendar days (conservative)
    """
    # Conservative: multiply by ratio + 1 day buffer
    return int(trading_days * CALENDAR_TO_TRADING_RATIO) + 1


def calendar_days_to_trading_days(calendar_days: int) -> int:
    """
    Convert calendar days to trading days (CONSERVATIVE estimate).
    
    Args:
        calendar_days: Number of calendar days
        
    Returns:
        Approximate number of trading days (conservative = fewer)
    """
    # Conservative: divide by ratio and round down
    return int(calendar_days / CALENDAR_TO_TRADING_RATIO)


# ============================================================================
# VALIDATION HELPERS
# ============================================================================

def validate_horizon(horizon: int) -> None:
    """Validate that a horizon is one of the allowed values."""
    if horizon not in HORIZONS_TRADING_DAYS:
        raise ValueError(
            f"Horizon must be one of {HORIZONS_TRADING_DAYS} trading days, got {horizon}"
        )


def validate_embargo(embargo_days: int) -> None:
    """Validate that embargo meets minimum requirement."""
    if embargo_days < EMBARGO_RULES.embargo_trading_days:
        raise ValueError(
            f"Embargo must be at least {EMBARGO_RULES.embargo_trading_days} TRADING DAYS, "
            f"got {embargo_days}. This is a HARD requirement."
        )


# ============================================================================
# DOCUMENTATION STRING (FOR EMBEDDING IN OTHER DOCS)
# ============================================================================

DEFINITION_LOCK_SUMMARY = """
## Chapter 6 Definition Lock (Canonical Reference)

**Source of Truth:** `src/evaluation/definitions.py`

### Time Conventions
- **Horizons:** 20, 60, 90 TRADING DAYS (not calendar days)
- **Embargo:** 90 TRADING DAYS (must be >= max horizon)
- **Rebalance:** First trading day of month (or quarter)
- **Pricing:** Close-to-close (labels use closing prices)
- **Market Close:** 4:00 PM Eastern Time

### Maturity Rule
- **Canonical:** label_matured_at <= cutoff_close_utc
- **Cutoff:** Market close time converted to UTC
- **Enforcement:** HARD (code fails loudly if violated)

### Purging Rules
- **Train labels:** Purged if T + H > train_end (maturity extends past)
- **Val labels:** Purged if T - H < train_end (lookback overlaps)
- **Granularity:** Per-row-per-horizon (NOT global)

### End-of-Sample Eligibility
- **Rule:** All horizons must be valid (no partial horizons)
- **Valid:** Label exists + matured + required fields populated
- **Enforcement:** Evaluation grid only includes eligible as-of dates

### Evaluation Range
- **Start:** 2016-01-01
- **End:** 2025-06-30
- **Min Training:** 730 calendar days (~2 years)
"""


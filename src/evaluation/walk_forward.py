"""
Walk-Forward Validation Engine (Chapter 6.1)

Implements expanding window walk-forward splits with:
- Strict purging of overlapping label windows (PER-ROW-PER-HORIZON)
- Hard embargo period (90 TRADING DAYS) between train and validation
- PIT-safe label maturity enforcement (UTC market close)
- End-of-sample eligibility (all horizons must be valid)

CRITICAL: This is NOT a rolling window - the training window GROWS forward.

REFERENCES: src/evaluation/definitions.py for canonical time conventions.
"""

from dataclasses import dataclass, field
from datetime import date, datetime, time
from typing import List, Optional, Set, Dict, Any
import pandas as pd
import numpy as np
import logging

from .definitions import (
    TIME_CONVENTIONS,
    EVALUATION_RANGE,
    EMBARGO_RULES,
    ELIGIBILITY_RULES,
    HORIZONS_TRADING_DAYS,
    trading_days_to_calendar_days,
    get_market_close_utc,
    validate_embargo,
    validate_horizon,
)

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardFold:
    """
    A single fold in walk-forward validation.
    
    CRITICAL: All date fields are dates (not datetimes).
    For maturity checks, use get_cutoff_utc() to get the UTC datetime.
    
    Attributes:
        fold_id: Unique identifier (e.g., "fold_01_202301")
        train_start: Start date of training period (inclusive)
        train_end: End date of training period (EXCLUSIVE - last day NOT included)
        val_start: Start date of validation period (inclusive)
        val_end: End date of validation period (EXCLUSIVE)
        embargo_trading_days: Number of TRADING DAYS between train_end and val_start
        purged_labels: Set of (date, ticker, horizon) tuples purged for overlap
    """
    fold_id: str
    train_start: date
    train_end: date
    val_start: date
    val_end: date
    embargo_trading_days: int
    purged_labels: Set[tuple] = field(default_factory=set)
    
    def __post_init__(self):
        """Validate fold dates and embargo with LOUD failures."""
        # Date ordering validation
        if self.train_start >= self.train_end:
            raise ValueError(
                f"train_start ({self.train_start}) must be < train_end ({self.train_end})"
            )
        if self.val_start >= self.val_end:
            raise ValueError(
                f"val_start ({self.val_start}) must be < val_end ({self.val_end})"
            )
        if self.train_end >= self.val_start:
            raise ValueError(
                f"train_end ({self.train_end}) must be < val_start ({self.val_start}). "
                f"EMBARGO NOT ENFORCED! This is a critical anti-leakage violation."
            )
        
        # Embargo validation (HARD requirement)
        validate_embargo(self.embargo_trading_days)
        
        logger.debug(
            f"Fold {self.fold_id} validated: "
            f"train=[{self.train_start}, {self.train_end}), "
            f"val=[{self.val_start}, {self.val_end}), "
            f"embargo={self.embargo_trading_days} trading days"
        )
    
    def get_train_cutoff_utc(self) -> datetime:
        """Get UTC datetime for train period cutoff (market close on train_end - 1 day)."""
        # train_end is exclusive, so cutoff is the day before
        from datetime import timedelta
        cutoff_date = self.train_end - timedelta(days=1)
        return get_market_close_utc(cutoff_date)
    
    def get_val_cutoff_utc(self) -> datetime:
        """Get UTC datetime for validation period cutoff (market close on val_end - 1 day)."""
        from datetime import timedelta
        cutoff_date = self.val_end - timedelta(days=1)
        return get_market_close_utc(cutoff_date)
    
    def filter_labels(
        self, 
        labels_df: pd.DataFrame,
        split: str = "train",
        horizons: Optional[List[int]] = None
    ) -> pd.DataFrame:
        """
        Filter labels for this fold with purging, maturity, and eligibility enforcement.
        
        Args:
            labels_df: DataFrame with columns: 
                - date: as-of date
                - ticker: stock identifier
                - horizon: forecast horizon in TRADING DAYS
                - label: the target value
                - label_matured_at: datetime (UTC) when label matured
            split: "train" or "val"
            horizons: List of horizons to include (default: all from definitions)
            
        Returns:
            Filtered DataFrame with purged overlapping labels
            
        ENFORCES:
        1. Date range filtering
        2. Label maturity (label_matured_at <= cutoff_utc)
        3. Purging (overlapping label windows removed)
        4. Per-horizon validation
        """
        if split not in ["train", "val"]:
            raise ValueError(f"split must be 'train' or 'val', got {split}")
        
        # Default horizons from definitions
        if horizons is None:
            horizons = list(HORIZONS_TRADING_DAYS)
        
        # Validate horizons
        for h in horizons:
            validate_horizon(h)
        
        # Get date range and cutoff
        if split == "train":
            start, end = self.train_start, self.train_end
            cutoff_utc = self.get_train_cutoff_utc()
        else:
            start, end = self.val_start, self.val_end
            cutoff_utc = self.get_val_cutoff_utc()
        
        # Step 1: Filter by date range
        mask = (labels_df["date"] >= start) & (labels_df["date"] < end)
        
        # Step 2: Filter by horizon
        mask &= labels_df["horizon"].isin(horizons)
        
        df = labels_df[mask].copy()
        
        if len(df) == 0:
            logger.warning(f"Fold {self.fold_id} {split}: No labels in date range")
            return df
        
        # Step 3: CRITICAL - Enforce label maturity (PIT safety)
        # Labels can only be used if label_matured_at <= cutoff_utc
        if "label_matured_at" in df.columns:
            # Handle both datetime and date columns
            def check_maturity(row_matured_at):
                if pd.isna(row_matured_at):
                    return False
                
                # Convert to datetime if needed
                if isinstance(row_matured_at, date) and not isinstance(row_matured_at, datetime):
                    # Assume market close UTC
                    row_matured_at = get_market_close_utc(row_matured_at)
                elif isinstance(row_matured_at, datetime):
                    # Ensure timezone-aware
                    if row_matured_at.tzinfo is None:
                        # Assume UTC
                        import pytz
                        row_matured_at = pytz.UTC.localize(row_matured_at)
                
                return row_matured_at <= cutoff_utc
            
            mature_mask = df["label_matured_at"].apply(check_maturity)
            n_immature = (~mature_mask).sum()
            
            if n_immature > 0:
                logger.warning(
                    f"Fold {self.fold_id} {split}: Dropping {n_immature} IMMATURE labels "
                    f"(label_matured_at > {cutoff_utc}). This is PIT enforcement."
                )
                df = df[mature_mask]
        
        # Step 4: CRITICAL - Remove purged labels (PER-ROW-PER-HORIZON overlap check)
        if self.purged_labels:
            n_before = len(df)
            
            # Create lookup set for O(1) checking
            purge_set = self.purged_labels
            
            def is_not_purged(row):
                key = (row["date"], row["ticker"], row["horizon"])
                return key not in purge_set
            
            purge_mask = df.apply(is_not_purged, axis=1)
            df = df[purge_mask]
            
            n_purged = n_before - len(df)
            if n_purged > 0:
                logger.info(
                    f"Fold {self.fold_id} {split}: PURGED {n_purged} overlapping labels "
                    f"(per-row-per-horizon anti-leakage)"
                )
        
        logger.debug(
            f"Fold {self.fold_id} {split}: {len(df)} labels after filtering "
            f"(horizons={horizons})"
        )
        
        return df


class WalkForwardSplitter:
    """
    Walk-forward validation splitter with expanding window.
    
    ENFORCES (HARD CONSTRAINTS):
    1. Purging: Labels with overlapping forward windows are removed (per-row-per-horizon)
    2. Embargo: Hard 90 TRADING DAY gap between train and validation
    3. Maturity: Only mature labels (label_matured_at <= cutoff_utc) are used
    4. Eligibility: Only as-of dates with ALL horizons valid are included
    
    REFERENCES: src/evaluation/definitions.py for canonical parameters.
    
    Parameters:
        eval_start: Start date of evaluation period (default from definitions)
        eval_end: End date of evaluation period (default from definitions)
        rebalance_freq: "monthly" or "quarterly"
        embargo_trading_days: Embargo in TRADING DAYS (NOT calendar days!)
        min_train_days: Minimum training period in calendar days
        trading_calendar: Optional TradingCalendar for accurate day counting
    """
    
    def __init__(
        self,
        eval_start: Optional[date] = None,
        eval_end: Optional[date] = None,
        rebalance_freq: str = "monthly",
        embargo_trading_days: int = 90,
        min_train_days: int = 730,
        trading_calendar = None,
    ):
        """
        Initialize walk-forward splitter with LOCKED parameters.
        
        CRITICAL: embargo_trading_days is in TRADING DAYS, not calendar days.
        """
        self.eval_start = eval_start or EVALUATION_RANGE.eval_start
        self.eval_end = eval_end or EVALUATION_RANGE.eval_end
        self.rebalance_freq = rebalance_freq.lower()
        self.embargo_trading_days = embargo_trading_days
        self.min_train_days = min_train_days
        self.trading_calendar = trading_calendar
        
        # Validate rebalance frequency
        if self.rebalance_freq not in ["monthly", "quarterly"]:
            raise ValueError(
                f"rebalance_freq must be 'monthly' or 'quarterly', got {rebalance_freq}"
            )
        
        # HARD validation of embargo (must be >= max horizon)
        validate_embargo(self.embargo_trading_days)
        
        logger.info(
            f"WalkForwardSplitter initialized:\n"
            f"  eval_range: [{self.eval_start}, {self.eval_end}]\n"
            f"  rebalance: {self.rebalance_freq}\n"
            f"  embargo: {self.embargo_trading_days} TRADING DAYS\n"
            f"  min_train: {self.min_train_days} calendar days\n"
            f"  PURGING: per-row-per-horizon (ENFORCED)\n"
            f"  MATURITY: label_matured_at <= cutoff_utc (ENFORCED)"
        )
    
    # Keep backward compatibility with old parameter name
    @property
    def embargo_days(self) -> int:
        """Backward compatibility alias. embargo_trading_days is preferred."""
        return self.embargo_trading_days
    
    def generate_folds(
        self,
        labels_df: pd.DataFrame,
        horizons: List[int] = None,
        require_all_horizons: bool = True,
    ) -> List[WalkForwardFold]:
        """
        Generate walk-forward folds with purging and embargo.
        
        Args:
            labels_df: DataFrame with labels (must have: date, ticker, horizon, label_matured_at)
            horizons: List of forecast horizons in TRADING DAYS (default: [20, 60, 90])
            require_all_horizons: If True, only include as-of dates with ALL horizons valid
                                  (ELIGIBILITY_RULES.allow_partial_horizons = False)
            
        Returns:
            List of WalkForwardFold objects
            
        CRITICAL: This implements EXPANDING window (not rolling).
        The training set GROWS with each fold.
        """
        # Default horizons from definitions
        if horizons is None:
            horizons = list(HORIZONS_TRADING_DAYS)
        
        # Validate horizons
        for h in horizons:
            validate_horizon(h)
        
        # Generate rebalance dates
        rebalance_dates = self._generate_rebalance_dates()
        
        # Filter for eligible as-of dates if required
        if require_all_horizons and not ELIGIBILITY_RULES.allow_partial_horizons:
            rebalance_dates = self._filter_eligible_dates(
                rebalance_dates, labels_df, horizons
            )
        
        folds = []
        for i, val_start in enumerate(rebalance_dates):
            # Skip if not enough training history
            if (val_start - self.eval_start).days < self.min_train_days:
                logger.debug(
                    f"Skipping fold starting {val_start}: "
                    f"insufficient training history ({(val_start - self.eval_start).days} < {self.min_train_days})"
                )
                continue
            
            # Expanding window: train_start is ALWAYS eval_start
            train_start = self.eval_start
            
            # Convert embargo from trading days to calendar days (CONSERVATIVE)
            embargo_calendar_days = trading_days_to_calendar_days(self.embargo_trading_days)
            
            # train_end = val_start - embargo (in calendar days, conservative)
            train_end = val_start - pd.Timedelta(days=embargo_calendar_days)
            train_end = train_end.date() if isinstance(train_end, pd.Timestamp) else train_end
            
            # Validation period: one rebalance period
            if i + 1 < len(rebalance_dates):
                val_end = rebalance_dates[i + 1]
            else:
                val_end = self.eval_end
            
            # Create fold ID
            fold_id = f"fold_{i+1:02d}_{val_start.strftime('%Y%m')}"
            
            # CRITICAL: Compute purged labels (PER-ROW-PER-HORIZON)
            purged_labels = self._compute_purged_labels(
                labels_df=labels_df,
                train_end=train_end,
                val_start=val_start,
                horizons=horizons
            )
            
            fold = WalkForwardFold(
                fold_id=fold_id,
                train_start=train_start,
                train_end=train_end,
                val_start=val_start,
                val_end=val_end,
                embargo_trading_days=self.embargo_trading_days,
                purged_labels=purged_labels
            )
            
            folds.append(fold)
            
            logger.info(
                f"Created {fold_id}: "
                f"train=[{train_start}, {train_end}), "
                f"val=[{val_start}, {val_end}), "
                f"purged={len(purged_labels)} labels"
            )
        
        logger.info(
            f"Generated {len(folds)} walk-forward folds "
            f"(embargo={self.embargo_trading_days} TRADING DAYS)"
        )
        
        return folds
    
    def _generate_rebalance_dates(self) -> List[date]:
        """
        Generate rebalance dates (first day of month/quarter).
        
        CONVENTION: First calendar day of period (first trading day if calendar available).
        """
        dates = []
        current = self.eval_start
        
        while current < self.eval_end:
            dates.append(current)
            
            if self.rebalance_freq == "monthly":
                # Move to first day of next month
                if current.month == 12:
                    current = date(current.year + 1, 1, 1)
                else:
                    current = date(current.year, current.month + 1, 1)
            else:  # quarterly
                # Move to first day of next quarter
                next_month = ((current.month - 1) // 3 + 1) * 3 + 1
                if next_month > 12:
                    current = date(current.year + 1, 1, 1)
                else:
                    current = date(current.year, next_month, 1)
        
        return dates
    
    def _filter_eligible_dates(
        self,
        rebalance_dates: List[date],
        labels_df: pd.DataFrame,
        horizons: List[int]
    ) -> List[date]:
        """
        Filter rebalance dates for eligibility (all horizons must be valid).
        
        END-OF-SAMPLE ELIGIBILITY RULE:
        An as-of date is ELIGIBLE only if ALL horizons have valid labels.
        This prevents partial horizons near the end of the evaluation period.
        """
        eligible_dates = []
        
        for as_of_date in rebalance_dates:
            # Check if all horizons have labels on this date
            date_labels = labels_df[labels_df["date"] == as_of_date]
            
            if len(date_labels) == 0:
                logger.debug(f"Skipping {as_of_date}: no labels")
                continue
            
            # Check all horizons
            horizons_present = set(date_labels["horizon"].unique())
            required_horizons = set(horizons)
            
            if not required_horizons.issubset(horizons_present):
                missing = required_horizons - horizons_present
                logger.debug(
                    f"Skipping {as_of_date}: missing horizons {missing} "
                    f"(end-of-sample eligibility rule)"
                )
                continue
            
            # Check labels exist for each horizon for at least some tickers
            for h in horizons:
                horizon_labels = date_labels[date_labels["horizon"] == h]
                if len(horizon_labels) == 0:
                    logger.debug(
                        f"Skipping {as_of_date}: no labels for horizon {h}"
                    )
                    continue
            
            eligible_dates.append(as_of_date)
        
        logger.info(
            f"End-of-sample eligibility: {len(eligible_dates)}/{len(rebalance_dates)} "
            f"dates have all horizons valid"
        )
        
        return eligible_dates
    
    def _compute_purged_labels(
        self,
        labels_df: pd.DataFrame,
        train_end: date,
        val_start: date,
        horizons: List[int]
    ) -> Set[tuple]:
        """
        Compute labels to purge due to overlapping forward windows.
        
        CRITICAL: This is PER-ROW-PER-HORIZON purging, not a global rule.
        Each (date, ticker, horizon) combination is evaluated independently.
        
        PURGE RULES (from definitions):
        1. Training labels: Purge if T + H (trading days) > train_end
           (maturity extends past training period)
        2. Validation labels: Purge if T - H (trading days) < train_end
           (lookback overlaps training period)
        
        This ensures NO forward-looking information leakage.
        """
        purged = set()
        
        # Convert to timestamps for arithmetic
        train_end_ts = pd.Timestamp(train_end)
        val_start_ts = pd.Timestamp(val_start)
        
        # Process training labels (purge if maturity extends past train_end)
        train_labels = labels_df[
            (labels_df["date"] < train_end) & 
            (labels_df["horizon"].isin(horizons))
        ]
        
        for _, row in train_labels.iterrows():
            label_date = pd.Timestamp(row["date"])
            horizon = row["horizon"]
            
            # Convert horizon (trading days) to calendar days (conservative)
            horizon_calendar = trading_days_to_calendar_days(horizon)
            
            # Label maturity date (approximate)
            label_maturity = label_date + pd.Timedelta(days=horizon_calendar)
            
            # PURGE RULE 1: Training label purged if maturity extends past train_end
            if label_maturity > train_end_ts:
                purged.add((row["date"], row["ticker"], horizon))
        
        # Process validation labels (purge if lookback overlaps training)
        val_labels = labels_df[
            (labels_df["date"] >= val_start) & 
            (labels_df["horizon"].isin(horizons))
        ]
        
        for _, row in val_labels.iterrows():
            label_date = pd.Timestamp(row["date"])
            horizon = row["horizon"]
            
            # Convert horizon (trading days) to calendar days (conservative)
            horizon_calendar = trading_days_to_calendar_days(horizon)
            
            # Label "lookback" (where the forward window started)
            label_lookback = label_date - pd.Timedelta(days=horizon_calendar)
            
            # PURGE RULE 2: Validation label purged if lookback overlaps training
            if label_lookback < train_end_ts:
                purged.add((row["date"], row["ticker"], horizon))
        
        return purged

"""
Pre-Implementation Sanity Checks (Chapter 6.0.1)

CRITICAL: These must pass BEFORE any Chapter 6 evaluation code is written.

Sanity Check 1: Manual IC vs Qlib IC Parity Test
- Ensures adapter/indexing is correct before generating hundreds of reports
- Manual and Qlib RankIC must agree to 3 decimal places

Sanity Check 2: Experiment Naming Convention
- Enforces standardized naming to prevent chaos when experiments explode
- Format: ai_forecaster/horizon={H}/model={M}/labels={L}/fold={F}
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import logging
import re

logger = logging.getLogger(__name__)


def verify_ic_parity(
    predictions: pd.DataFrame,
    labels: pd.DataFrame,
    qlib_ic: Optional[float] = None,
    tolerance: float = 0.001
) -> Dict[str, Any]:
    """
    Sanity Check 1: Verify Manual IC matches Qlib IC.
    
    CRITICAL: If these don't match, STOP and fix the adapter immediately.
    
    Args:
        predictions: DataFrame with columns: date, ticker, prediction
        labels: DataFrame with columns: date, ticker, label
        qlib_ic: Qlib-computed IC (median over dates). If None, skip Qlib comparison.
        tolerance: Maximum absolute difference allowed (default: 0.001)
        
    Returns:
        Dictionary with:
            - manual_ic: Manually computed median RankIC
            - qlib_ic: Qlib IC (if provided)
            - match: True if within tolerance
            - diff: Absolute difference
            - details: Per-date IC values
            
    Raises:
        ValueError: If ICs don't match within tolerance
    """
    # Merge predictions and labels
    df = pd.merge(
        predictions[["date", "ticker", "prediction"]],
        labels[["date", "ticker", "label"]],
        on=["date", "ticker"],
        how="inner"
    )
    
    if len(df) == 0:
        raise ValueError("No matching predictions and labels found")
    
    # Compute IC per date (Spearman rank correlation)
    def compute_ic(group):
        if len(group) < 2:
            return np.nan
        return spearmanr(group["prediction"], group["label"])[0]
    
    daily_ic = df.groupby("date").apply(compute_ic)
    daily_ic = daily_ic.dropna()
    
    if len(daily_ic) == 0:
        raise ValueError("No valid IC values computed (all dates had < 2 observations)")
    
    manual_ic = daily_ic.median()
    
    logger.info(f"Manual IC: {manual_ic:.4f} (median over {len(daily_ic)} dates)")
    logger.info(f"Manual IC mean: {daily_ic.mean():.4f}, std: {daily_ic.std():.4f}")
    
    result = {
        "manual_ic": manual_ic,
        "qlib_ic": qlib_ic,
        "match": None,
        "diff": None,
        "details": {
            "daily_ic_median": manual_ic,
            "daily_ic_mean": daily_ic.mean(),
            "daily_ic_std": daily_ic.std(),
            "n_dates": len(daily_ic),
            "n_observations": len(df),
        }
    }
    
    # Compare with Qlib if provided
    if qlib_ic is not None:
        diff = abs(manual_ic - qlib_ic)
        match = diff <= tolerance
        
        result["diff"] = diff
        result["match"] = match
        
        logger.info(f"Qlib IC: {qlib_ic:.4f}")
        logger.info(f"Difference: {diff:.6f}")
        
        if match:
            logger.info(f"✅ IC PARITY CHECK PASSED (diff={diff:.6f} <= {tolerance})")
        else:
            error_msg = (
                f"❌ IC PARITY CHECK FAILED!\n"
                f"   Manual IC: {manual_ic:.4f}\n"
                f"   Qlib IC:   {qlib_ic:.4f}\n"
                f"   Diff:      {diff:.6f} > {tolerance}\n"
                f"\n"
                f"STOP: Fix adapter/indexing before proceeding.\n"
                f"Check:\n"
                f"  - MultiIndex formatting (datetime, instrument)\n"
                f"  - Date alignment (T vs T+H)\n"
                f"  - Missing data handling\n"
                f"  - Sign flips (prediction vs label)"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)
    else:
        logger.warning("No Qlib IC provided for comparison. Skipping parity check.")
    
    return result


@dataclass
class ExperimentNameBuilder:
    """
    Builds standardized experiment names for Qlib Recorder.
    
    Convention (LOCKED):
        ai_forecaster/horizon={H}/model={M}/labels={L}/fold={F}
    
    Example:
        ai_forecaster/horizon=20/model=kronos_v0/labels=v2/fold=03
    
    This prevents chaos when experiments explode to hundreds of runs.
    """
    
    horizon: int
    model: str
    label_version: str
    fold_id: str
    
    # Valid values (for validation)
    VALID_HORIZONS = [20, 60, 90]
    VALID_LABEL_VERSIONS = ["v1_priceonly", "v2_totalreturn", "v1", "v2"]
    
    def __post_init__(self):
        """Validate experiment name components."""
        if self.horizon not in self.VALID_HORIZONS:
            raise ValueError(
                f"horizon must be one of {self.VALID_HORIZONS}, got {self.horizon}"
            )
        
        # Normalize label version
        if self.label_version == "v1":
            self.label_version = "v1_priceonly"
        elif self.label_version == "v2":
            self.label_version = "v2_totalreturn"
        
        if self.label_version not in self.VALID_LABEL_VERSIONS:
            raise ValueError(
                f"label_version must be one of {self.VALID_LABEL_VERSIONS}, "
                f"got {self.label_version}"
            )
        
        # Validate model name (alphanumeric + underscore only)
        if not re.match(r'^[a-z0-9_]+$', self.model):
            raise ValueError(
                f"model name must be lowercase alphanumeric + underscore, got {self.model}"
            )
        
        # Validate fold_id (alphanumeric + underscore + hyphen)
        if not re.match(r'^[a-z0-9_\-]+$', self.fold_id):
            raise ValueError(
                f"fold_id must be lowercase alphanumeric + underscore/hyphen, "
                f"got {self.fold_id}"
            )
    
    def build(self) -> str:
        """Build the experiment name."""
        return (
            f"ai_forecaster/"
            f"horizon={self.horizon}/"
            f"model={self.model}/"
            f"labels={self.label_version}/"
            f"fold={self.fold_id}"
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Export as dictionary for metadata logging."""
        return {
            "horizon": self.horizon,
            "model": self.model,
            "label_version": self.label_version,
            "fold_id": self.fold_id,
            "experiment_name": self.build()
        }


def validate_experiment_name(exp_name: str) -> Dict[str, str]:
    """
    Validate and parse an experiment name.
    
    Args:
        exp_name: Experiment name string
        
    Returns:
        Dictionary with parsed components: horizon, model, labels, fold
        
    Raises:
        ValueError: If name doesn't match convention
    """
    pattern = (
        r'^ai_forecaster/'
        r'horizon=(?P<horizon>\d+)/'
        r'model=(?P<model>[a-z0-9_]+)/'
        r'labels=(?P<labels>[a-z0-9_]+)/'
        r'fold=(?P<fold>[a-z0-9_\-]+)$'
    )
    
    match = re.match(pattern, exp_name)
    if not match:
        raise ValueError(
            f"Experiment name does not match convention:\n"
            f"  Got: {exp_name}\n"
            f"  Expected: ai_forecaster/horizon={{H}}/model={{M}}/labels={{L}}/fold={{F}}"
        )
    
    components = match.groupdict()
    
    # Validate horizon
    horizon = int(components["horizon"])
    if horizon not in ExperimentNameBuilder.VALID_HORIZONS:
        raise ValueError(
            f"Invalid horizon in experiment name: {horizon}. "
            f"Must be one of {ExperimentNameBuilder.VALID_HORIZONS}"
        )
    
    logger.info(f"✅ Experiment name validated: {exp_name}")
    return components


def run_sanity_checks(
    predictions: pd.DataFrame,
    labels: pd.DataFrame,
    qlib_ic: Optional[float] = None,
    experiment_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run all pre-implementation sanity checks.
    
    Args:
        predictions: Predictions DataFrame
        labels: Labels DataFrame
        qlib_ic: Qlib IC for parity check (optional)
        experiment_name: Experiment name to validate (optional)
        
    Returns:
        Dictionary with results from all checks
        
    Raises:
        ValueError: If any check fails
    """
    logger.info("=" * 70)
    logger.info("RUNNING PRE-IMPLEMENTATION SANITY CHECKS (Chapter 6.0.1)")
    logger.info("=" * 70)
    
    results = {}
    
    # Sanity Check 1: IC Parity
    logger.info("\nSanity Check 1: Manual IC vs Qlib IC Parity Test")
    logger.info("-" * 70)
    ic_result = verify_ic_parity(predictions, labels, qlib_ic)
    results["ic_parity"] = ic_result
    
    # Sanity Check 2: Experiment Naming
    if experiment_name:
        logger.info("\nSanity Check 2: Experiment Naming Convention")
        logger.info("-" * 70)
        name_components = validate_experiment_name(experiment_name)
        results["experiment_name"] = name_components
    
    logger.info("\n" + "=" * 70)
    logger.info("✅ ALL SANITY CHECKS PASSED")
    logger.info("=" * 70)
    
    return results


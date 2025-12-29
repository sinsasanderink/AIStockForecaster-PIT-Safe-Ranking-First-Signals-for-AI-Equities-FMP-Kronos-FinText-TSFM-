"""
Feature Neutralization (Section 5.8)
=====================================

Neutralization diagnostics to reveal WHERE alpha comes from.

This module is for EVALUATION/DIAGNOSTICS ONLY, not for training.

KEY PURPOSE:
- Understand if feature alpha is:
  • Sector rotation effect
  • Market beta exposure
  • Genuinely stock-specific

DESIGN CHOICES:
1. We neutralize the FEATURE (not the label) before computing IC
   "Is this feature just a proxy for sector/beta?"

2. Neutralization is cross-sectional PER DATE
   Sectors/betas are as-of date T (PIT-safe)

3. We report IC deltas:
   Δ_sector = neutral_IC - raw_IC
   Δ_beta = neutral_IC - raw_IC
   
   If Δ_sector is large negative, the feature was mostly sector rotation.
   If Δ remains strong, it's stock-specific alpha.

4. Beta-neutral uses beta_252d from price_features.py (consistent definition)

5. Market-neutral = beta-neutral here (linear market exposure removed)

USAGE:
    from src.features.neutralization import (
        neutralize_cross_section,
        compute_neutralized_ic,
        neutralization_report,
    )
    
    # Neutralize a feature cross-sectionally
    neutral_values = neutralize_cross_section(
        values=features_df["mom_1m"],
        exposures=sector_dummies,
        method="ols",
    )
    
    # Compute IC with different neutralizations
    results = compute_neutralized_ic(
        features_df=features_df,
        labels_df=labels_df,
        feature_col="mom_1m",
        label_col="excess_return",
        sector_col="sector",
        beta_col="beta_252d",
    )
    
    # Full report
    report = neutralization_report(
        features_df=features_df,
        labels_df=labels_df,
        feature_cols=["mom_1m", "mom_3m", "pe_vs_sector"],
    )
"""

import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

import pandas as pd
import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)

# Default IC method
DEFAULT_IC_METHOD = "spearman"


@dataclass
class NeutralizationResult:
    """
    IC results with different neutralizations for a single feature.
    """
    feature: str
    horizon: Optional[int] = None
    
    # Sample info
    n_dates: int = 0
    n_observations: int = 0
    
    # Raw IC
    raw_ic: float = 0.0
    raw_ic_std: float = 0.0
    
    # Sector-neutral IC
    sector_neutral_ic: Optional[float] = None
    sector_neutral_ic_std: Optional[float] = None
    delta_sector: Optional[float] = None  # neutral - raw
    
    # Beta-neutral IC
    beta_neutral_ic: Optional[float] = None
    beta_neutral_ic_std: Optional[float] = None
    delta_beta: Optional[float] = None  # neutral - raw
    
    # Sector + Beta neutral IC
    sector_beta_neutral_ic: Optional[float] = None
    sector_beta_neutral_ic_std: Optional[float] = None
    delta_sector_beta: Optional[float] = None  # neutral - raw
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature": self.feature,
            "horizon": self.horizon,
            "n_dates": self.n_dates,
            "n_observations": self.n_observations,
            "raw_ic": self.raw_ic,
            "raw_ic_std": self.raw_ic_std,
            "sector_neutral_ic": self.sector_neutral_ic,
            "sector_neutral_ic_std": self.sector_neutral_ic_std,
            "delta_sector": self.delta_sector,
            "beta_neutral_ic": self.beta_neutral_ic,
            "beta_neutral_ic_std": self.beta_neutral_ic_std,
            "delta_beta": self.delta_beta,
            "sector_beta_neutral_ic": self.sector_beta_neutral_ic,
            "sector_beta_neutral_ic_std": self.sector_beta_neutral_ic_std,
            "delta_sector_beta": self.delta_sector_beta,
        }


# =============================================================================
# Core Neutralization Functions
# =============================================================================

def neutralize_cross_section(
    values: pd.Series,
    exposures: pd.DataFrame,
    method: str = "ols",
    add_intercept: bool = True,
    alpha: float = 1.0,  # Ridge penalty (only if method="ridge")
) -> pd.Series:
    """
    Neutralize values against exposures via cross-sectional regression.
    
    Returns residuals from: values = exposures @ coeffs + residuals
    
    Args:
        values: Series of feature values (indexed by stock)
        exposures: DataFrame of exposures (rows=stocks, cols=factors)
                   e.g., sector dummies, beta values
        method: "ols" or "ridge"
        add_intercept: Whether to add constant term
        alpha: Ridge penalty (only used if method="ridge")
    
    Returns:
        Series of residuals (same index as values)
    """
    # Align values and exposures
    common_idx = values.index.intersection(exposures.index)
    
    if len(common_idx) < 5:
        logger.warning("Not enough observations for neutralization")
        return values
    
    values_clean = values.loc[common_idx].dropna()
    exposures_clean = exposures.loc[values_clean.index].dropna(how="any")
    
    # Further align after dropping NaNs
    common_idx_clean = values_clean.index.intersection(exposures_clean.index)
    
    if len(common_idx_clean) < 5:
        return values
    
    y = values_clean.loc[common_idx_clean].values
    X = exposures_clean.loc[common_idx_clean].values
    
    # Add intercept if requested
    if add_intercept:
        X = np.column_stack([np.ones(len(X)), X])
    
    # Fit regression
    try:
        if method == "ols":
            from numpy.linalg import lstsq
            coeffs, residuals_sum, rank, s = lstsq(X, y, rcond=None)
            residuals = y - X @ coeffs
        elif method == "ridge":
            from numpy.linalg import inv
            # Ridge: (X'X + αI)^(-1) X'y
            XtX = X.T @ X
            XtX_ridge = XtX + alpha * np.eye(XtX.shape[0])
            coeffs = inv(XtX_ridge) @ (X.T @ y)
            residuals = y - X @ coeffs
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Return as Series with original index
        residuals_series = pd.Series(residuals, index=common_idx_clean)
        
        # Fill non-overlapping indices with original values (or NaN)
        result = values.copy()
        result.loc[common_idx_clean] = residuals_series
        
        return result
    
    except Exception as e:
        logger.warning(f"Neutralization failed: {e}")
        return values


def create_sector_dummies(
    sector_series: pd.Series,
    drop_first: bool = True,
) -> pd.DataFrame:
    """
    Create one-hot encoded sector dummies.
    
    Args:
        sector_series: Series of sector labels (indexed by stock)
        drop_first: Drop first category to avoid multicollinearity
    
    Returns:
        DataFrame of dummy variables
    """
    return pd.get_dummies(sector_series, drop_first=drop_first, dtype=float)


def compute_ic(
    feature: pd.Series,
    label: pd.Series,
    method: str = "spearman",
) -> float:
    """
    Compute Information Coefficient (IC) between feature and label.
    
    Args:
        feature: Feature values
        label: Label values (e.g., forward returns)
        method: "spearman" or "pearson"
    
    Returns:
        IC value (correlation coefficient)
    """
    # Align and drop NaNs
    aligned = pd.DataFrame({"feature": feature, "label": label}).dropna()
    
    if len(aligned) < 10:
        return np.nan
    
    if method == "spearman":
        ic, _ = stats.spearmanr(aligned["feature"], aligned["label"])
    else:
        ic, _ = stats.pearsonr(aligned["feature"], aligned["label"])
    
    return ic


# =============================================================================
# IC Computation with Neutralization
# =============================================================================

def compute_neutralized_ic(
    features_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    feature_col: str,
    label_col: str = "excess_return",
    date_col: str = "date",
    ticker_col: str = "ticker",
    sector_col: Optional[str] = "sector",
    beta_col: Optional[str] = "beta_252d",
    ic_method: str = DEFAULT_IC_METHOD,
    horizon: Optional[int] = None,
) -> NeutralizationResult:
    """
    Compute IC with different neutralizations for a single feature.
    
    This is the MAIN function for Section 5.8.
    
    Args:
        features_df: Features DataFrame (must have date, ticker, feature_col)
        labels_df: Labels DataFrame (must have date, ticker, label_col)
        feature_col: Name of feature column
        label_col: Name of label column (default: "excess_return")
        date_col: Date column name
        ticker_col: Ticker column name
        sector_col: Sector column name (None to skip sector neutralization)
        beta_col: Beta column name (None to skip beta neutralization)
        ic_method: "spearman" or "pearson"
        horizon: Optional horizon for filtering labels
    
    Returns:
        NeutralizationResult with all ICs and deltas
    """
    result = NeutralizationResult(feature=feature_col, horizon=horizon)
    
    # Merge features and labels
    merged = features_df.merge(
        labels_df[[ticker_col, date_col, label_col]],
        on=[ticker_col, date_col],
        how="inner",
    )
    
    if horizon is not None and "horizon" in labels_df.columns:
        merged = merged[labels_df["horizon"] == horizon]
    
    if merged.empty:
        logger.warning(f"No data for {feature_col} after merge")
        return result
    
    # Get unique dates
    dates = sorted(merged[date_col].unique())
    result.n_dates = len(dates)
    result.n_observations = len(merged)
    
    # Compute ICs per date
    raw_ics = []
    sector_neutral_ics = []
    beta_neutral_ics = []
    sector_beta_neutral_ics = []
    
    for d in dates:
        date_df = merged[merged[date_col] == d].copy()
        
        if len(date_df) < 10:
            continue
        
        # Set index to ticker for easier alignment
        date_df = date_df.set_index(ticker_col)
        
        feature_values = date_df[feature_col]
        label_values = date_df[label_col]
        
        # Raw IC
        raw_ic = compute_ic(feature_values, label_values, method=ic_method)
        if not np.isnan(raw_ic):
            raw_ics.append(raw_ic)
        
        # Sector-neutral IC
        if sector_col and sector_col in date_df.columns:
            sector_series = date_df[sector_col]
            sector_dummies = create_sector_dummies(sector_series)
            
            neutral_feature = neutralize_cross_section(
                feature_values,
                sector_dummies,
                method="ols",
            )
            
            ic = compute_ic(neutral_feature, label_values, method=ic_method)
            if not np.isnan(ic):
                sector_neutral_ics.append(ic)
        
        # Beta-neutral IC
        if beta_col and beta_col in date_df.columns:
            beta_series = date_df[[beta_col]].dropna()
            
            if len(beta_series) >= 10:
                neutral_feature = neutralize_cross_section(
                    feature_values,
                    beta_series,
                    method="ols",
                )
                
                ic = compute_ic(neutral_feature, label_values, method=ic_method)
                if not np.isnan(ic):
                    beta_neutral_ics.append(ic)
        
        # Sector + Beta neutral IC
        if sector_col and beta_col and sector_col in date_df.columns and beta_col in date_df.columns:
            sector_series = date_df[sector_col]
            sector_dummies = create_sector_dummies(sector_series)
            beta_series = date_df[[beta_col]].dropna()
            
            # Combine exposures
            combined_exposures = sector_dummies.join(beta_series, how="inner")
            
            if len(combined_exposures) >= 10:
                neutral_feature = neutralize_cross_section(
                    feature_values,
                    combined_exposures,
                    method="ols",
                )
                
                ic = compute_ic(neutral_feature, label_values, method=ic_method)
                if not np.isnan(ic):
                    sector_beta_neutral_ics.append(ic)
    
    # Aggregate ICs
    if raw_ics:
        result.raw_ic = np.mean(raw_ics)
        result.raw_ic_std = np.std(raw_ics)
    
    if sector_neutral_ics:
        result.sector_neutral_ic = np.mean(sector_neutral_ics)
        result.sector_neutral_ic_std = np.std(sector_neutral_ics)
        result.delta_sector = result.sector_neutral_ic - result.raw_ic
    
    if beta_neutral_ics:
        result.beta_neutral_ic = np.mean(beta_neutral_ics)
        result.beta_neutral_ic_std = np.std(beta_neutral_ics)
        result.delta_beta = result.beta_neutral_ic - result.raw_ic
    
    if sector_beta_neutral_ics:
        result.sector_beta_neutral_ic = np.mean(sector_beta_neutral_ics)
        result.sector_beta_neutral_ic_std = np.std(sector_beta_neutral_ics)
        result.delta_sector_beta = result.sector_beta_neutral_ic - result.raw_ic
    
    return result


# =============================================================================
# Reporting
# =============================================================================

def neutralization_report(
    features_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    horizons: Optional[List[int]] = None,
    **kwargs,
) -> List[NeutralizationResult]:
    """
    Generate neutralization report for multiple features and horizons.
    
    Args:
        features_df: Features DataFrame
        labels_df: Labels DataFrame
        feature_cols: List of feature columns (auto-detect if None)
        horizons: List of horizons to analyze (all if None)
        **kwargs: Additional arguments passed to compute_neutralized_ic
    
    Returns:
        List of NeutralizationResult objects
    """
    if feature_cols is None:
        # Auto-detect feature columns
        meta_cols = {"date", "ticker", "stable_id", "sector"}
        feature_cols = [c for c in features_df.columns if c not in meta_cols
                       and features_df[c].dtype in ['float64', 'float32', 'int64', 'int32']]
    
    if horizons is None and "horizon" in labels_df.columns:
        horizons = sorted(labels_df["horizon"].unique())
    
    results = []
    
    for feature_col in feature_cols:
        if horizons:
            for h in horizons:
                result = compute_neutralized_ic(
                    features_df=features_df,
                    labels_df=labels_df,
                    feature_col=feature_col,
                    horizon=h,
                    **kwargs,
                )
                results.append(result)
        else:
            result = compute_neutralized_ic(
                features_df=features_df,
                labels_df=labels_df,
                feature_col=feature_col,
                **kwargs,
            )
            results.append(result)
    
    return results


def format_neutralization_report(results: List[NeutralizationResult]) -> str:
    """
    Format neutralization results as a human-readable report.
    
    Args:
        results: List of NeutralizationResult objects
    
    Returns:
        Formatted report string
    """
    report = []
    report.append("=" * 80)
    report.append("NEUTRALIZATION DIAGNOSTICS REPORT")
    report.append("=" * 80)
    report.append("")
    report.append("Purpose: Understand WHERE alpha comes from")
    report.append("  - Large negative Δ_sector → feature was mostly sector rotation")
    report.append("  - Large negative Δ_beta → feature was mostly market exposure")
    report.append("  - Δ remains small → alpha is stock-specific")
    report.append("")
    report.append("=" * 80)
    
    for result in results:
        report.append("")
        horizon_str = f" (H={result.horizon}d)" if result.horizon else ""
        report.append(f"Feature: {result.feature}{horizon_str}")
        report.append("-" * 60)
        report.append(f"  Sample: {result.n_observations} obs across {result.n_dates} dates")
        report.append(f"  Raw IC:          {result.raw_ic:>7.3f} (±{result.raw_ic_std:.3f})")
        
        if result.sector_neutral_ic is not None:
            delta_str = f"{result.delta_sector:+.3f}"
            report.append(f"  Sector-neutral:  {result.sector_neutral_ic:>7.3f} (Δ={delta_str})")
        
        if result.beta_neutral_ic is not None:
            delta_str = f"{result.delta_beta:+.3f}"
            report.append(f"  Beta-neutral:    {result.beta_neutral_ic:>7.3f} (Δ={delta_str})")
        
        if result.sector_beta_neutral_ic is not None:
            delta_str = f"{result.delta_sector_beta:+.3f}"
            report.append(f"  Sector+Beta:     {result.sector_beta_neutral_ic:>7.3f} (Δ={delta_str})")
        
        # Interpretation
        report.append("")
        report.append("  Interpretation:")
        
        if result.delta_sector is not None:
            if abs(result.delta_sector) < 0.005:
                report.append("    ✓ Sector-independent (pure stock selection)")
            elif result.delta_sector < -0.01:
                report.append("    ⚠️ Heavy sector rotation component")
            else:
                report.append("    • Some sector exposure")
        
        if result.delta_beta is not None:
            if abs(result.delta_beta) < 0.005:
                report.append("    ✓ Beta-independent (market-neutral)")
            elif result.delta_beta < -0.01:
                report.append("    ⚠️ Heavy market exposure")
            else:
                report.append("    • Some market beta")
    
    report.append("")
    report.append("=" * 80)
    
    return "\n".join(report)


# =============================================================================
# Helper Functions
# =============================================================================

def get_neutralization_summary_df(results: List[NeutralizationResult]) -> pd.DataFrame:
    """Convert list of NeutralizationResults to DataFrame."""
    return pd.DataFrame([r.to_dict() for r in results])


# =============================================================================
# CLI / Demo
# =============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("FEATURE NEUTRALIZATION DEMO (Section 5.8)")
    print("=" * 80)
    print("\nThis module provides neutralization diagnostics.")
    print("See tests/test_neutralization.py for working examples.")
    print("")
    print("Key functions:")
    print("  • neutralize_cross_section() - Remove exposures via OLS/Ridge")
    print("  • compute_neutralized_ic() - Compute IC with neutralization")
    print("  • neutralization_report() - Full report for multiple features")
    print("")
    print("Output format:")
    print("  Raw IC, Sector-neutral IC, Beta-neutral IC, Sector+Beta IC")
    print("  Plus deltas (Δ) showing IC change after neutralization")
    print("")
    print("Interpretation:")
    print("  Large negative Δ → feature was mostly that factor")
    print("  Small Δ → alpha is genuine stock-specific")
    print("=" * 80)


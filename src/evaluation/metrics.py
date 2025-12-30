"""
Evaluation Metrics (Chapter 6.3)

Implements ranking-first metrics with locked definitions:
- RankIC (Spearman correlation)
- Quintile/decile spread
- Top-K hit rate and excess return
- Churn (using stable_id, consecutive dates only)
- Regime slicing (using existing regime features)

CRITICAL: All metrics computed per-date, then aggregated per fold.
This ensures cross-sectional ranking evaluation (not time-series).

CANONICAL DATA CONTRACT: See EvaluationRow for required fields.
"""

from dataclasses import dataclass
from datetime import date
from typing import List, Optional, Dict, Any, Tuple
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import logging

from .definitions import HORIZONS_TRADING_DAYS, validate_horizon

logger = logging.getLogger(__name__)


# ============================================================================
# CANONICAL EVALUATION DATA CONTRACT
# ============================================================================

@dataclass
class EvaluationRow:
    """
    CANONICAL EVALUATION ROW FORMAT
    
    Every model and baseline MUST produce this format.
    This prevents "everyone computing metrics slightly differently" chaos.
    
    Required fields:
        as_of_date: Evaluation date
        ticker: Stock ticker symbol
        stable_id: Survivorship-safe identifier (for churn)
        horizon: Forecast horizon in TRADING DAYS (20, 60, or 90)
        fold_id: Walk-forward fold identifier
        score: Model's ranking score (HIGHER = BETTER)
        excess_return: v2 total return relative to benchmark
        
    Optional (for cost realism, regime slicing):
        adv_20d: Average daily dollar volume (20d)
        adv_60d: Average daily dollar volume (60d)
        sector: GICS sector
        beta_252d: Rolling 252d beta
        vix_percentile_252d: VIX percentile
        market_return_20d: Market return (20d)
        market_vol_20d: Market volatility (20d)
    
    RULES:
    1. score direction: Higher = better (already rank-normalized if needed)
    2. Duplicates: NOT ALLOWED per (as_of_date, stable_id, horizon)
    3. Missing score: Row DROPPED (logged as warning)
    4. Missing excess_return: Row DROPPED (logged as warning)
    5. Tie-breaking: Deterministic via stable_id sorting
    """
    as_of_date: date
    ticker: str
    stable_id: str
    horizon: int
    fold_id: str
    score: float
    excess_return: float
    
    # Optional
    adv_20d: Optional[float] = None
    adv_60d: Optional[float] = None
    sector: Optional[str] = None
    beta_252d: Optional[float] = None
    vix_percentile_252d: Optional[float] = None
    market_return_20d: Optional[float] = None
    market_vol_20d: Optional[float] = None


# Minimum cross-section size (skip dates with fewer names)
MIN_CROSS_SECTION_SIZE = 10

# Aggregation methods
DEFAULT_AGG_METHOD = "median"  # Primary
SECONDARY_AGG_METHOD = "mean"


# ============================================================================
# TIE-BREAKING (DETERMINISTIC)
# ============================================================================

def rank_with_ties(df: pd.DataFrame, score_col: str = "score") -> pd.Series:
    """
    Rank scores with DETERMINISTIC tie-breaking.
    
    Ties are broken by stable_id (alphabetical order).
    This ensures churn/hit-rate don't wobble randomly.
    
    Args:
        df: DataFrame with 'score' and 'stable_id'
        score_col: Column name for scores
        
    Returns:
        Series of ranks (1 = highest score)
    """
    # Sort by score (desc) then stable_id (asc) for deterministic tie-breaking
    df_sorted = df.sort_values([score_col, "stable_id"], ascending=[False, True])
    
    # Assign ranks (1 = best)
    df_sorted["rank"] = range(1, len(df_sorted) + 1)
    
    # Return ranks in original order
    return df_sorted.sort_index()["rank"]


# ============================================================================
# CORE METRICS (PER-DATE)
# ============================================================================

def compute_rankic_per_date(
    df: pd.DataFrame,
    score_col: str = "score",
    label_col: str = "excess_return"
) -> float:
    """
    Compute RankIC (Spearman correlation) for a single date.
    
    RankIC = Spearman(rank(score), rank(excess_return))
    
    Args:
        df: DataFrame for single date with score and excess_return
        score_col: Column name for scores
        label_col: Column name for labels
        
    Returns:
        RankIC value (np.nan if < 2 observations)
    """
    # Filter out missing values
    df_clean = df[[score_col, label_col]].dropna()
    
    if len(df_clean) < 2:
        return np.nan
    
    # Spearman correlation (rank-based)
    ic, _ = spearmanr(df_clean[score_col], df_clean[label_col])
    
    return ic


def compute_quintile_spread_per_date(
    df: pd.DataFrame,
    score_col: str = "score",
    label_col: str = "excess_return",
    n_buckets: int = 5
) -> Dict[str, float]:
    """
    Compute top-bottom quintile spread for a single date.
    
    Spread = mean(excess_return in top bucket) - mean(excess_return in bottom bucket)
    
    Args:
        df: DataFrame for single date
        score_col: Column name for scores
        label_col: Column name for labels
        n_buckets: Number of buckets (5 = quintiles, 10 = deciles)
        
    Returns:
        Dictionary with spread and bucket means
    """
    df_clean = df[[score_col, label_col]].dropna()
    
    if len(df_clean) < n_buckets:
        return {"spread": np.nan, "top_mean": np.nan, "bottom_mean": np.nan}
    
    # Assign buckets based on score rank
    df_clean = df_clean.copy()
    df_clean["bucket"] = pd.qcut(
        df_clean[score_col], 
        q=n_buckets, 
        labels=False, 
        duplicates="drop"
    )
    
    # Top bucket = highest scores (bucket == n_buckets - 1)
    # Bottom bucket = lowest scores (bucket == 0)
    top_mean = df_clean[df_clean["bucket"] == n_buckets - 1][label_col].mean()
    bottom_mean = df_clean[df_clean["bucket"] == 0][label_col].mean()
    
    return {
        "spread": top_mean - bottom_mean,
        "top_mean": top_mean,
        "bottom_mean": bottom_mean
    }


def compute_topk_metrics_per_date(
    df: pd.DataFrame,
    k: int,
    score_col: str = "score",
    label_col: str = "excess_return"
) -> Dict[str, float]:
    """
    Compute Top-K hit rate and average excess return for a single date.
    
    Hit rate = fraction of Top-K with excess_return > 0
    Average ER = mean(excess_return of Top-K)
    
    Args:
        df: DataFrame for single date with score, excess_return, stable_id
        k: Top-K size
        score_col: Column name for scores
        label_col: Column name for labels
        
    Returns:
        Dictionary with hit_rate and avg_excess_return
    """
    df_clean = df[[score_col, label_col, "stable_id"]].dropna()
    
    if len(df_clean) < k:
        return {
            "hit_rate": np.nan,
            "avg_excess_return": np.nan,
            "n_positive": np.nan,
            "n_total": len(df_clean)
        }
    
    # Rank with deterministic tie-breaking
    df_clean = df_clean.copy()
    df_clean["rank"] = rank_with_ties(df_clean, score_col)
    
    # Select Top-K
    top_k = df_clean[df_clean["rank"] <= k]
    
    # Hit rate = fraction with positive excess return
    n_positive = (top_k[label_col] > 0).sum()
    hit_rate = n_positive / k
    
    # Average excess return
    avg_er = top_k[label_col].mean()
    
    return {
        "hit_rate": hit_rate,
        "avg_excess_return": avg_er,
        "n_positive": int(n_positive),
        "n_total": k
    }


# ============================================================================
# CHURN (CONSECUTIVE DATES ONLY)
# ============================================================================

def compute_churn(
    df: pd.DataFrame,
    k: int,
    date_col: str = "as_of_date",
    id_col: str = "stable_id",
    score_col: str = "score"
) -> pd.DataFrame:
    """
    Compute Top-K churn across consecutive rebalance dates.
    
    CRITICAL:
    - Uses stable_id (not ticker) to avoid rename noise
    - Only computes on consecutive dates within the input DataFrame
    - Deterministic tie-breaking via stable_id
    
    Retention@K = |TopK(t) ∩ TopK(t-1)| / K
    Churn@K = 1 - Retention@K
    
    Args:
        df: DataFrame with as_of_date, stable_id, score
        k: Top-K size
        date_col: Column name for dates
        id_col: Column name for identifiers (stable_id)
        score_col: Column name for scores
        
    Returns:
        DataFrame with (date, retention, churn, n_overlap, n_total)
        First date is excluded (no previous to compare)
    """
    dates = sorted(df[date_col].unique())
    
    if len(dates) < 2:
        logger.warning("Churn requires at least 2 dates")
        return pd.DataFrame(columns=[date_col, "retention", "churn", "n_overlap", "n_total"])
    
    churn_results = []
    prev_top_k = None
    
    for i, current_date in enumerate(dates):
        # Get current date data
        df_current = df[df[date_col] == current_date].copy()
        
        if len(df_current) < k:
            logger.warning(f"Date {current_date}: only {len(df_current)} < {k} names, skipping churn")
            continue
        
        # Rank with deterministic tie-breaking
        df_current["rank"] = rank_with_ties(df_current, score_col)
        
        # Select Top-K
        current_top_k = set(df_current[df_current["rank"] <= k][id_col])
        
        # Compute churn if we have previous
        if prev_top_k is not None:
            overlap = current_top_k & prev_top_k
            retention = len(overlap) / k
            churn = 1 - retention
            
            churn_results.append({
                date_col: current_date,
                "retention": retention,
                "churn": churn,
                "n_overlap": len(overlap),
                "n_total": k
            })
        
        prev_top_k = current_top_k
    
    return pd.DataFrame(churn_results)


# ============================================================================
# REGIME SLICING
# ============================================================================

# LOCKED REGIME DEFINITIONS (using existing features)
REGIME_DEFINITIONS = {
    "vix_percentile_252d": {
        "low": lambda x: x <= 33,
        "mid": lambda x: (x > 33) & (x <= 67),
        "high": lambda x: x > 67
    },
    "market_regime": {
        "bull": lambda x: x > 0,  # market_return_20d > 0
        "bear": lambda x: x <= 0
    },
    "market_vol_20d": {
        "low": lambda x: x <= 33,
        "high": lambda x: x > 67
    }
}


def assign_regime_bucket(
    df: pd.DataFrame,
    regime_feature: str
) -> pd.Series:
    """
    Assign regime bucket based on locked definitions.
    
    Args:
        df: DataFrame with regime feature column
        regime_feature: Feature name (e.g., "vix_percentile_252d")
        
    Returns:
        Series with regime bucket labels
    """
    if regime_feature not in df.columns:
        raise ValueError(f"Regime feature {regime_feature} not in DataFrame")
    
    if regime_feature not in REGIME_DEFINITIONS:
        raise ValueError(
            f"Regime feature {regime_feature} not in locked definitions. "
            f"Available: {list(REGIME_DEFINITIONS.keys())}"
        )
    
    buckets = REGIME_DEFINITIONS[regime_feature]
    values = df[regime_feature]
    
    # Assign buckets
    result = pd.Series("unknown", index=df.index)
    for bucket_name, condition_func in buckets.items():
        mask = condition_func(values)
        result[mask] = bucket_name
    
    return result


# ============================================================================
# FOLD-LEVEL AGGREGATION
# ============================================================================

def aggregate_per_date_metrics(
    per_date_df: pd.DataFrame,
    metric_cols: List[str],
    agg_method: str = "median"
) -> Dict[str, Any]:
    """
    Aggregate per-date metrics to fold summary.
    
    Args:
        per_date_df: DataFrame with per-date metrics
        metric_cols: List of metric column names to aggregate
        agg_method: "median" (primary) or "mean" (secondary)
        
    Returns:
        Dictionary with aggregated metrics (median/mean, IQR, n_dates)
    """
    result = {}
    
    for col in metric_cols:
        if col not in per_date_df.columns:
            continue
        
        values = per_date_df[col].dropna()
        
        if len(values) == 0:
            result[f"{col}_{agg_method}"] = np.nan
            result[f"{col}_iqr"] = np.nan
            result[f"{col}_n_dates"] = 0
            continue
        
        if agg_method == "median":
            result[f"{col}_median"] = values.median()
        elif agg_method == "mean":
            result[f"{col}_mean"] = values.mean()
        
        # IQR for all
        result[f"{col}_iqr"] = values.quantile(0.75) - values.quantile(0.25)
        result[f"{col}_n_dates"] = len(values)
    
    return result


# ============================================================================
# HIGH-LEVEL EVALUATION FUNCTIONS
# ============================================================================

def evaluate_fold(
    eval_df: pd.DataFrame,
    fold_id: str,
    horizon: int,
    k_values: List[int] = [10, 20],
    min_cross_section: int = MIN_CROSS_SECTION_SIZE
) -> Dict[str, pd.DataFrame]:
    """
    Evaluate a single fold with all metrics.
    
    Args:
        eval_df: DataFrame in EvaluationRow format
        fold_id: Fold identifier
        horizon: Horizon in trading days
        k_values: List of K values for Top-K metrics
        min_cross_section: Minimum names per date
        
    Returns:
        Dictionary with:
            - per_date_metrics: Per-date metric values
            - fold_summary: Aggregated fold metrics
            - churn: Churn results for each K
    """
    validate_horizon(horizon)
    
    # Filter for this fold and horizon
    df = eval_df[
        (eval_df["fold_id"] == fold_id) & 
        (eval_df["horizon"] == horizon)
    ].copy()
    
    if len(df) == 0:
        logger.warning(f"No data for fold {fold_id}, horizon {horizon}")
        return {
            "per_date_metrics": pd.DataFrame(),
            "fold_summary": pd.DataFrame(),
            "churn": {}
        }
    
    # Validate data contract
    required_cols = ["as_of_date", "stable_id", "score", "excess_return"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    # Drop rows with missing score or excess_return
    n_before = len(df)
    df = df.dropna(subset=["score", "excess_return"])
    n_dropped = n_before - len(df)
    if n_dropped > 0:
        logger.warning(
            f"Fold {fold_id}, horizon {horizon}: Dropped {n_dropped} rows "
            f"with missing score/excess_return"
        )
    
    # Check for duplicates
    dupe_mask = df.duplicated(subset=["as_of_date", "stable_id"], keep=False)
    if dupe_mask.any():
        n_dupes = dupe_mask.sum()
        logger.error(
            f"Fold {fold_id}, horizon {horizon}: Found {n_dupes} duplicate "
            f"(as_of_date, stable_id) pairs. This violates data contract!"
        )
        # Keep first occurrence (deterministic)
        df = df.drop_duplicates(subset=["as_of_date", "stable_id"], keep="first")
    
    # Compute per-date metrics
    per_date_results = []
    
    for as_of_date in sorted(df["as_of_date"].unique()):
        df_date = df[df["as_of_date"] == as_of_date]
        n_names = len(df_date)
        
        # Skip if too few names
        if n_names < min_cross_section:
            logger.debug(
                f"Fold {fold_id}, date {as_of_date}: only {n_names} < {min_cross_section}, skipping"
            )
            continue
        
        # RankIC
        rankic = compute_rankic_per_date(df_date)
        
        # Quintile spread
        spread = compute_quintile_spread_per_date(df_date)
        
        # Top-K metrics for each K
        topk_metrics = {}
        for k in k_values:
            topk = compute_topk_metrics_per_date(df_date, k)
            topk_metrics[k] = topk
        
        # Collect results
        result = {
            "as_of_date": as_of_date,
            "n_names": n_names,
            "rankic": rankic,
            "quintile_spread": spread["spread"],
            "top_quintile_mean": spread["top_mean"],
            "bottom_quintile_mean": spread["bottom_mean"],
        }
        
        for k in k_values:
            result[f"top{k}_hit_rate"] = topk_metrics[k]["hit_rate"]
            result[f"top{k}_avg_er"] = topk_metrics[k]["avg_excess_return"]
        
        per_date_results.append(result)
    
    per_date_df = pd.DataFrame(per_date_results)
    
    # Aggregate to fold summary
    metric_cols = ["rankic", "quintile_spread"]
    for k in k_values:
        metric_cols.extend([f"top{k}_hit_rate", f"top{k}_avg_er"])
    
    fold_summary = aggregate_per_date_metrics(per_date_df, metric_cols, agg_method="median")
    fold_summary["fold_id"] = fold_id
    fold_summary["horizon"] = horizon
    
    # Compute churn for each K
    churn_results = {}
    for k in k_values:
        churn_df = compute_churn(df, k)
        churn_results[k] = churn_df
        
        # Add churn summary to fold summary
        if len(churn_df) > 0:
            fold_summary[f"top{k}_churn_median"] = churn_df["churn"].median()
            fold_summary[f"top{k}_retention_median"] = churn_df["retention"].median()
    
    return {
        "per_date_metrics": per_date_df,
        "fold_summary": pd.DataFrame([fold_summary]),
        "churn": churn_results
    }


def evaluate_with_regime_slicing(
    eval_df: pd.DataFrame,
    fold_id: str,
    horizon: int,
    regime_feature: str,
    k_values: List[int] = [10, 20]
) -> pd.DataFrame:
    """
    Evaluate with regime slicing.
    
    Computes metrics within each regime bucket.
    
    Args:
        eval_df: DataFrame in EvaluationRow format
        fold_id: Fold identifier
        horizon: Horizon in trading days
        regime_feature: Regime feature name (from locked definitions)
        k_values: List of K values for Top-K metrics
        
    Returns:
        DataFrame with regime-sliced metrics
    """
    # Filter for this fold and horizon
    df = eval_df[
        (eval_df["fold_id"] == fold_id) & 
        (eval_df["horizon"] == horizon)
    ].copy()
    
    if regime_feature not in df.columns:
        logger.warning(f"Regime feature {regime_feature} not in DataFrame")
        return pd.DataFrame()
    
    # Assign regime buckets
    df["regime_bucket"] = assign_regime_bucket(df, regime_feature)
    
    # Evaluate within each bucket
    results = []
    for bucket in df["regime_bucket"].unique():
        if bucket == "unknown":
            continue
        
        df_bucket = df[df["regime_bucket"] == bucket]
        
        # Evaluate this bucket as a mini-fold
        bucket_eval = evaluate_fold(
            df_bucket, 
            fold_id=f"{fold_id}_{regime_feature}_{bucket}",
            horizon=horizon,
            k_values=k_values
        )
        
        if len(bucket_eval["fold_summary"]) > 0:
            summary = bucket_eval["fold_summary"].iloc[0].to_dict()
            summary["regime_feature"] = regime_feature
            summary["regime_bucket"] = bucket
            results.append(summary)
    
    return pd.DataFrame(results)


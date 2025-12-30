"""
Chapter 7 Baselines

Implements baselines that run through IDENTICAL pipeline:

FACTOR BASELINES (transparent, no ML):
1. mom_12m: Rank by 12-month momentum (primary naive baseline)
2. momentum_composite: Average of mom_1m, mom_3m, mom_6m, mom_12m (stronger baseline)
3. short_term_strength: Rank by mom_1m (diagnostic baseline)

SANITY BASELINES (pipeline verification):
4. naive_random: Deterministic random scores seeded by (as_of_date, horizon)
   - Should produce ~0 RankIC over time
   - If it doesn't, your evaluation is hallucinating alpha

CRITICAL: All baselines MUST:
- Use same universe snapshots (stable_id)
- Use same walk-forward folds (purging/embargo/maturity)
- Use same EvaluationRow contract
- Use same metrics + costs + stability reports

NO ML, NO optimization in factor baselines. Just transparent feature-based scoring.
"""

from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Optional, Callable
import pandas as pd
import numpy as np
import logging

from .metrics import EvaluationRow

logger = logging.getLogger(__name__)


# ============================================================================
# BASELINE DEFINITIONS (LOCKED)
# ============================================================================

@dataclass(frozen=True)
class BaselineDefinition:
    """
    Locked definition for a baseline model.
    
    Attributes:
        name: Unique baseline identifier
        description: Human-readable description
        required_features: List of features required from input data
        score_formula: Description of how score is computed
    """
    name: str
    description: str
    required_features: tuple
    score_formula: str


# LOCKED BASELINE DEFINITIONS (do not modify)
BASELINE_MOM_12M = BaselineDefinition(
    name="mom_12m",
    description="12-month momentum (primary naive baseline)",
    required_features=("mom_12m",),
    score_formula="score = mom_12m"
)

BASELINE_MOMENTUM_COMPOSITE = BaselineDefinition(
    name="momentum_composite",
    description="Equal-weight average of mom_1m, mom_3m, mom_6m, mom_12m",
    required_features=("mom_1m", "mom_3m", "mom_6m", "mom_12m"),
    score_formula="score = (mom_1m + mom_3m + mom_6m + mom_12m) / 4"
)

BASELINE_SHORT_TERM_STRENGTH = BaselineDefinition(
    name="short_term_strength",
    description="1-month momentum (diagnostic baseline)",
    required_features=("mom_1m",),
    score_formula="score = mom_1m"
)

# SANITY BASELINES (pipeline verification, not "to beat")
BASELINE_NAIVE_RANDOM = BaselineDefinition(
    name="naive_random",
    description="Deterministic random scores (pipeline sanity check)",
    required_features=(),  # No features required - pure random
    score_formula="score = deterministic_random(seed=hash(as_of_date, horizon, stable_id))"
)

# Registry of all baselines
# NOTE: naive_random is included but should NOT be used as a "bar to clear"
# It's purely a sanity check that evaluation isn't hallucinating alpha
BASELINE_REGISTRY: Dict[str, BaselineDefinition] = {
    "mom_12m": BASELINE_MOM_12M,
    "momentum_composite": BASELINE_MOMENTUM_COMPOSITE,
    "short_term_strength": BASELINE_SHORT_TERM_STRENGTH,
    "naive_random": BASELINE_NAIVE_RANDOM,
}

# Factor baselines (models must beat these)
FACTOR_BASELINES = ["mom_12m", "momentum_composite", "short_term_strength"]

# Sanity baselines (pipeline verification only)
SANITY_BASELINES = ["naive_random"]


# ============================================================================
# BASELINE SCORER
# ============================================================================

def compute_baseline_score(
    row: pd.Series,
    baseline_name: str
) -> Optional[float]:
    """
    Compute score for a single row using the specified baseline.
    
    Args:
        row: Series with feature values
        baseline_name: Name of baseline to use
        
    Returns:
        Score value (higher = better) or None if missing data
    """
    if baseline_name not in BASELINE_REGISTRY:
        raise ValueError(f"Unknown baseline: {baseline_name}. Valid options: {list(BASELINE_REGISTRY.keys())}")
    
    baseline = BASELINE_REGISTRY[baseline_name]
    
    # Check for missing features
    for feat in baseline.required_features:
        if feat not in row.index or pd.isna(row[feat]):
            return None
    
    # Compute score based on baseline type
    if baseline_name == "mom_12m":
        return row["mom_12m"]
    
    elif baseline_name == "momentum_composite":
        # Equal-weight average of 4 momentum features
        return (row["mom_1m"] + row["mom_3m"] + row["mom_6m"] + row["mom_12m"]) / 4.0
    
    elif baseline_name == "short_term_strength":
        return row["mom_1m"]
    
    elif baseline_name == "naive_random":
        # Deterministic random score using hash of identifying info
        # Must be deterministic: same inputs -> same output
        # Note: This is typically computed in generate_baseline_scores_batch for efficiency
        # If called row-by-row, we use the row's index as part of seed
        return None  # Handled in batch function
    
    else:
        raise ValueError(f"No scoring logic for baseline: {baseline_name}")


def _compute_naive_random_score(
    as_of_date: date,
    horizon: int,
    stable_id: str
) -> float:
    """
    Compute deterministic random score for naive_random baseline.
    
    Uses hash of (as_of_date, horizon, stable_id) as seed for reproducibility.
    This ensures the same inputs ALWAYS produce the same output.
    
    Args:
        as_of_date: The as-of date
        horizon: Forecast horizon
        stable_id: Stable identifier for the stock
        
    Returns:
        Random score in [0, 1]
    """
    # Create deterministic seed from inputs
    # Convert date to string for consistent hashing
    date_str = as_of_date.isoformat() if hasattr(as_of_date, 'isoformat') else str(as_of_date)
    seed_str = f"{date_str}_{horizon}_{stable_id}"
    
    # Use hash to create seed (make it positive)
    seed = abs(hash(seed_str)) % (2**31)
    
    # Generate deterministic random number
    rng = np.random.RandomState(seed)
    return rng.random()


def generate_baseline_scores(
    features_df: pd.DataFrame,
    baseline_name: str,
    fold_id: str,
    horizon: int,
    date_col: str = "date",
    ticker_col: str = "ticker",
    stable_id_col: str = "stable_id",
    excess_return_col: str = "excess_return",
    adv_20d_col: str = "adv_20d",
    adv_60d_col: str = "adv_60d",
    sector_col: str = "sector",
    vix_col: str = "vix_percentile_252d",
    market_return_col: str = "market_return_20d",
    market_vol_col: str = "market_vol_20d",
    beta_col: str = "beta_252d"
) -> pd.DataFrame:
    """
    Generate baseline scores for all rows in a features DataFrame.
    
    This produces evaluation rows in the CANONICAL EvaluationRow format.
    
    Args:
        features_df: DataFrame with features and labels
        baseline_name: Which baseline to use
        fold_id: Fold identifier (e.g., "fold_01")
        horizon: Forecast horizon in TRADING DAYS (20, 60, or 90)
        date_col: Column name for as_of_date
        ticker_col: Column name for ticker
        stable_id_col: Column name for stable_id
        excess_return_col: Column name for excess_return (v2 total return)
        *_col: Optional column names for optional fields
        
    Returns:
        DataFrame in EvaluationRow format:
        - as_of_date, ticker, stable_id, horizon, fold_id
        - score (HIGHER = BETTER)
        - excess_return
        - Optional: adv_20d, adv_60d, sector, vix_percentile_252d, market_return_20d, market_vol_20d, beta_252d
    """
    if baseline_name not in BASELINE_REGISTRY:
        raise ValueError(f"Unknown baseline: {baseline_name}. Valid options: {list(BASELINE_REGISTRY.keys())}")
    
    baseline = BASELINE_REGISTRY[baseline_name]
    
    # Validate required columns
    required_cols = [date_col, ticker_col, stable_id_col, excess_return_col]
    required_cols.extend(baseline.required_features)
    
    missing_cols = [col for col in required_cols if col not in features_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Compute scores
    df = features_df.copy()
    
    if baseline_name == "naive_random":
        # Deterministic random scores using hash of (as_of_date, horizon, stable_id)
        # This ensures reproducibility: same inputs -> same outputs
        df["score"] = df.apply(
            lambda row: _compute_naive_random_score(
                row[date_col], 
                horizon, 
                row[stable_id_col]
            ),
            axis=1
        )
    else:
        # Feature-based baselines
        scores = []
        for idx, row in df.iterrows():
            score = compute_baseline_score(row, baseline_name)
            scores.append(score)
        df["score"] = scores
    
    # Filter out rows with missing scores
    n_before = len(df)
    df = df.dropna(subset=["score"])
    n_dropped = n_before - len(df)
    
    if n_dropped > 0:
        logger.warning(f"Baseline {baseline_name}: Dropped {n_dropped} rows due to missing features")
    
    # Check for missing excess_return
    n_before = len(df)
    df = df.dropna(subset=[excess_return_col])
    n_dropped = n_before - len(df)
    
    if n_dropped > 0:
        logger.warning(f"Baseline {baseline_name}: Dropped {n_dropped} rows due to missing {excess_return_col}")
    
    # Build output DataFrame
    output = pd.DataFrame({
        "as_of_date": df[date_col],
        "ticker": df[ticker_col],
        "stable_id": df[stable_id_col],
        "horizon": horizon,
        "fold_id": fold_id,
        "score": df["score"],
        "excess_return": df[excess_return_col]
    })
    
    # Add optional columns if present
    optional_mappings = [
        (adv_20d_col, "adv_20d"),
        (adv_60d_col, "adv_60d"),
        (sector_col, "sector"),
        (vix_col, "vix_percentile_252d"),
        (market_return_col, "market_return_20d"),
        (market_vol_col, "market_vol_20d"),
        (beta_col, "beta_252d")
    ]
    
    for src_col, dst_col in optional_mappings:
        if src_col in df.columns:
            output[dst_col] = df[src_col].values
        else:
            output[dst_col] = None
    
    # Check for duplicates (CRITICAL: not allowed per contract)
    duplicate_check = output.groupby(["as_of_date", "stable_id", "horizon"]).size()
    duplicates = duplicate_check[duplicate_check > 1]
    
    if len(duplicates) > 0:
        raise ValueError(
            f"Duplicate entries detected for (as_of_date, stable_id, horizon)! "
            f"First duplicates: {duplicates.head()}"
        )
    
    logger.info(
        f"Baseline {baseline_name}: Generated {len(output)} evaluation rows "
        f"for fold {fold_id}, horizon {horizon}"
    )
    
    return output


# ============================================================================
# BASELINE RUNNER (BATCH)
# ============================================================================

def run_all_baselines(
    features_df: pd.DataFrame,
    fold_id: str,
    horizon: int,
    baselines: Optional[List[str]] = None,
    **kwargs
) -> Dict[str, pd.DataFrame]:
    """
    Run all (or specified) baselines and return their evaluation rows.
    
    Args:
        features_df: DataFrame with features and labels
        fold_id: Fold identifier
        horizon: Forecast horizon in TRADING DAYS
        baselines: List of baseline names to run (None = all 3)
        **kwargs: Additional arguments passed to generate_baseline_scores
        
    Returns:
        Dictionary mapping baseline_name -> DataFrame of evaluation rows
    """
    if baselines is None:
        baselines = list(BASELINE_REGISTRY.keys())
    
    results = {}
    
    for baseline_name in baselines:
        logger.info(f"Running baseline: {baseline_name}")
        
        try:
            eval_rows = generate_baseline_scores(
                features_df=features_df,
                baseline_name=baseline_name,
                fold_id=fold_id,
                horizon=horizon,
                **kwargs
            )
            results[baseline_name] = eval_rows
            
        except Exception as e:
            logger.error(f"Baseline {baseline_name} failed: {e}")
            raise
    
    return results


# ============================================================================
# VALIDATION HELPERS
# ============================================================================

def validate_baseline_monotonicity(
    features_df: pd.DataFrame,
    baseline_name: str,
    feature_col: str
) -> bool:
    """
    Validate that baseline score is monotone with the underlying feature.
    
    This is a sanity check: higher feature value should always give higher score.
    
    Args:
        features_df: DataFrame with features
        baseline_name: Baseline to validate
        feature_col: Primary feature (e.g., "mom_12m" for mom_12m baseline)
        
    Returns:
        True if monotone, False otherwise
    """
    df = features_df.copy()
    
    # Compute scores
    scores = []
    for idx, row in df.iterrows():
        score = compute_baseline_score(row, baseline_name)
        scores.append(score)
    
    df["_score"] = scores
    df = df.dropna(subset=["_score", feature_col])
    
    # Check monotonicity (Spearman correlation should be ~1)
    from scipy.stats import spearmanr
    corr, _ = spearmanr(df[feature_col], df["_score"])
    
    # For simple baselines (mom_12m, short_term_strength), should be exactly 1
    # For composite, should be positive and high
    if baseline_name in ["mom_12m", "short_term_strength"]:
        return abs(corr - 1.0) < 1e-10
    else:
        # Composite: should be highly positive correlated with primary feature
        return corr > 0.5


def get_baseline_description(baseline_name: str) -> str:
    """Get human-readable description of a baseline."""
    if baseline_name not in BASELINE_REGISTRY:
        raise ValueError(f"Unknown baseline: {baseline_name}")
    
    baseline = BASELINE_REGISTRY[baseline_name]
    return f"{baseline.name}: {baseline.description}\nFormula: {baseline.score_formula}"


def list_baselines() -> List[str]:
    """List all available baseline names."""
    return list(BASELINE_REGISTRY.keys())


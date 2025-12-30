"""
Chapter 7 Baselines

Implements baselines that run through IDENTICAL pipeline:

FACTOR BASELINES (transparent, no ML):
1. mom_12m: Rank by 12-month momentum (primary naive baseline)
2. momentum_composite: Average of mom_1m, mom_3m, mom_6m, mom_12m (stronger baseline)
3. short_term_strength: Rank by mom_1m (diagnostic baseline)

ML BASELINES (Chapter 7.3):
4. tabular_lgb: LightGBM ranking model with time-decay weighting
   - Trained per fold using walk-forward splits
   - Horizon-specific models (separate for 20/60/90d)
   - Deterministic hyperparameters (no tuning in baseline)

SANITY BASELINES (pipeline verification):
5. naive_random: Deterministic random scores seeded by (as_of_date, horizon)
   - Should produce ~0 RankIC over time
   - If it doesn't, your evaluation is hallucinating alpha

CRITICAL: All baselines MUST:
- Use same universe snapshots (stable_id)
- Use same walk-forward folds (purging/embargo/maturity)
- Use same EvaluationRow contract
- Use same metrics + costs + stability reports

NO optimization in factor/sanity baselines. ML baselines use fixed hyperparameters.
"""

from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Optional, Callable, Tuple
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

# ML BASELINES (Chapter 7.3)
BASELINE_TABULAR_LGB = BaselineDefinition(
    name="tabular_lgb",
    description="LightGBM ranking model with time-decay weighting",
    required_features=(
        "mom_1m", "mom_3m", "mom_6m", "mom_12m",
        "vol_20d", "vol_60d",
        "max_drawdown_60d",
        "adv_20d"
    ),  # Core features; model handles missing gracefully
    score_formula="score = LGBMRegressor.predict(features) [trained per fold with time-decay]"
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
    "tabular_lgb": BASELINE_TABULAR_LGB,
    "naive_random": BASELINE_NAIVE_RANDOM,
}

# Factor baselines (models must beat these)
FACTOR_BASELINES = ["mom_12m", "momentum_composite", "short_term_strength"]

# ML baselines (Chapter 7+)
ML_BASELINES = ["tabular_lgb"]

# Sanity baselines (pipeline verification only)
SANITY_BASELINES = ["naive_random"]


# ============================================================================
# ML BASELINE HELPERS (Chapter 7.3)
# ============================================================================

# Default features for tabular_lgb (can be extended in future)
DEFAULT_TABULAR_FEATURES = [
    # Momentum
    "mom_1m", "mom_3m", "mom_6m", "mom_12m",
    # Volatility
    "vol_20d", "vol_60d", "vol_of_vol",
    # Drawdown
    "max_drawdown_60d",
    # Liquidity
    "adv_20d", "adv_60d",
    # Relative strength (if available)
    "rel_strength_1m", "rel_strength_3m",
    # Beta (if available)
    "beta_252d",
]


def _compute_time_decay_weights(
    dates: pd.Series,
    half_life_days: float = 252.0
) -> np.ndarray:
    """
    Compute exponential time-decay sample weights.
    
    More recent samples get higher weight. Uses exponential decay with
    configurable half-life.
    
    Args:
        dates: Series of dates (as date objects or datetime)
        half_life_days: Half-life for exponential decay (default: 1 year = 252 trading days)
        
    Returns:
        Array of weights (normalized to sum to len(dates))
    """
    # Convert to days since earliest date
    date_numeric = pd.to_datetime(dates).astype(np.int64) // 10**9 // (24*3600)  # days since epoch
    days_since_start = date_numeric - date_numeric.min()
    max_days = days_since_start.max()
    
    # Exponential decay: weight = 2^(-days_until_end / half_life)
    # Recent dates (days_until_end = 0) get weight = 1
    # Old dates (days_until_end = max) get weight = 2^(-max/half_life)
    days_until_end = max_days - days_since_start.values
    weights = np.power(2.0, -days_until_end / half_life_days)
    
    # Normalize so sum = len(dates) (to match unweighted case)
    weights = weights * len(dates) / weights.sum()
    
    return weights


def train_lgbm_ranking_model(
    train_df: pd.DataFrame,
    feature_cols: List[str],
    label_col: str = "excess_return",
    date_col: str = "date",
    group_col: str = "date",  # Group by date for ranking objective
    time_decay_half_life: float = 252.0,
    random_state: int = 42,
    n_estimators: int = 100,
    learning_rate: float = 0.05,
    max_depth: int = 5,
    num_leaves: int = 31,
    min_child_samples: int = 20,
    verbose: int = -1
):
    """
    Train LightGBM regression model with time-decay sample weighting.
    
    Note: Uses LGBMRegressor instead of LGBMRanker because our labels are continuous
    excess returns, not integer relevance grades. The model still produces ranking
    scores (higher predicted return = higher score).
    
    Args:
        train_df: Training data with features and labels
        feature_cols: List of feature column names
        label_col: Label column name (excess return)
        date_col: Date column for time-decay weighting
        group_col: Column to group by for ranking (typically date) - not used in regression
        time_decay_half_life: Half-life for exponential time decay (trading days)
        random_state: Fixed random state for determinism
        n_estimators: Number of boosting rounds
        learning_rate: Learning rate
        max_depth: Max tree depth
        num_leaves: Max number of leaves
        min_child_samples: Minimum samples per leaf
        verbose: Verbosity level (-1 = silent)
        
    Returns:
        Trained LightGBM model
    """
    try:
        from lightgbm import LGBMRegressor
    except ImportError:
        raise ImportError(
            "LightGBM not installed. Install with: pip install lightgbm>=4.0.0"
        )
    
    # Filter to rows with non-null labels and required features
    df = train_df.copy()
    required_cols = feature_cols + [label_col, date_col]
    df = df.dropna(subset=[label_col])
    
    # Check which features are actually available
    available_features = [f for f in feature_cols if f in df.columns]
    if len(available_features) == 0:
        raise ValueError(f"No features available from requested list: {feature_cols}")
    
    # Handle missing feature values (LightGBM can handle NaNs natively)
    X = df[available_features].values
    y = df[label_col].values
    
    # Compute time-decay weights
    sample_weights = _compute_time_decay_weights(
        df[date_col],
        half_life_days=time_decay_half_life
    )
    
    # Train LGBMRegressor (predicts excess return; higher = better)
    model = LGBMRegressor(
        objective='regression',
        metric='l2',
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        num_leaves=num_leaves,
        min_child_samples=min_child_samples,
        random_state=random_state,
        verbose=verbose,
        force_col_wise=True,  # Faster for wide datasets
    )
    
    model.fit(
        X, y,
        sample_weight=sample_weights
    )
    
    # Store feature names for prediction
    model.feature_names_ = available_features
    
    return model


def predict_lgbm_scores(
    model,
    val_df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None
) -> np.ndarray:
    """
    Generate predictions using trained LightGBM model.
    
    Args:
        model: Trained LGBMRegressor model (from train_lgbm_ranking_model)
        val_df: Validation data with features
        feature_cols: Feature column names (if None, use model.feature_names_)
        
    Returns:
        Array of predicted scores (higher = better)
    """
    if feature_cols is None:
        if not hasattr(model, 'feature_names_'):
            raise ValueError("Model doesn't have feature_names_ attribute; provide feature_cols explicitly")
        feature_cols = model.feature_names_
    
    # Extract features (handle missing columns gracefully)
    X = val_df[feature_cols].values
    
    # Predict
    scores = model.predict(X)
    
    return scores


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


def generate_ml_baseline_scores(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
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
    beta_col: str = "beta_252d",
    feature_cols: Optional[List[str]] = None,
    lgbm_params: Optional[Dict] = None
) -> pd.DataFrame:
    """
    Generate ML baseline scores by training on train_df and scoring val_df.
    
    This is used for baselines that require training (e.g., tabular_lgb).
    
    Args:
        train_df: Training data with features and labels
        val_df: Validation data to score
        baseline_name: Which ML baseline to use (currently only "tabular_lgb")
        fold_id: Fold identifier
        horizon: Forecast horizon in TRADING DAYS
        date_col: Column name for date
        ticker_col: Column name for ticker
        stable_id_col: Column name for stable_id
        excess_return_col: Column name for label
        *_col: Optional column names
        feature_cols: List of features to use (None = use defaults)
        lgbm_params: Optional LightGBM hyperparameters
        
    Returns:
        DataFrame in EvaluationRow format (same as factor baselines)
    """
    if baseline_name != "tabular_lgb":
        raise ValueError(f"Unknown ML baseline: {baseline_name}. Only 'tabular_lgb' is currently supported.")
    
    # Use default features if not specified
    if feature_cols is None:
        feature_cols = DEFAULT_TABULAR_FEATURES
    
    # Filter to available features
    available_features = [f for f in feature_cols if f in train_df.columns and f in val_df.columns]
    
    if len(available_features) == 0:
        raise ValueError(f"No features available from requested list: {feature_cols}")
    
    logger.info(f"Training {baseline_name} with {len(available_features)} features: {available_features}")
    
    # Train model
    lgbm_params = lgbm_params or {}
    model = train_lgbm_ranking_model(
        train_df=train_df,
        feature_cols=available_features,
        label_col=excess_return_col,
        date_col=date_col,
        group_col=date_col,  # Group by date for ranking
        **lgbm_params
    )
    
    # Predict on validation set
    scores = predict_lgbm_scores(
        model=model,
        val_df=val_df,
        feature_cols=available_features
    )
    
    # Build output DataFrame
    output = pd.DataFrame({
        "as_of_date": val_df[date_col],
        "ticker": val_df[ticker_col],
        "stable_id": val_df[stable_id_col],
        "horizon": horizon,
        "fold_id": fold_id,
        "score": scores,
        "excess_return": val_df[excess_return_col]
    })
    
    # Add optional columns
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
        if src_col in val_df.columns:
            output[dst_col] = val_df[src_col].values
        else:
            output[dst_col] = None
    
    # Drop rows with missing labels
    n_before = len(output)
    output = output.dropna(subset=["excess_return"])
    n_dropped = n_before - len(output)
    
    if n_dropped > 0:
        logger.warning(f"ML baseline {baseline_name}: Dropped {n_dropped} val rows due to missing labels")
    
    # Check for duplicates
    duplicate_check = output.groupby(["as_of_date", "stable_id", "horizon"]).size()
    duplicates = duplicate_check[duplicate_check > 1]
    
    if len(duplicates) > 0:
        raise ValueError(
            f"Duplicate entries detected for (as_of_date, stable_id, horizon)! "
            f"First duplicates: {duplicates.head()}"
        )
    
    logger.info(
        f"ML baseline {baseline_name}: Generated {len(output)} evaluation rows "
        f"for fold {fold_id}, horizon {horizon}"
    )
    
    return output


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
    beta_col: str = "beta_252d",
    train_df: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Generate baseline scores for all rows in a features DataFrame.
    
    This produces evaluation rows in the CANONICAL EvaluationRow format.
    
    For factor baselines (mom_12m, etc): Uses features_df directly.
    For ML baselines (tabular_lgb): Requires train_df to be provided.
    
    Args:
        features_df: DataFrame with features and labels (for factor baselines, this is val data)
        baseline_name: Which baseline to use
        fold_id: Fold identifier (e.g., "fold_01")
        horizon: Forecast horizon in TRADING DAYS (20, 60, or 90)
        date_col: Column name for as_of_date
        ticker_col: Column name for ticker
        stable_id_col: Column name for stable_id
        excess_return_col: Column name for excess_return (v2 total return)
        *_col: Optional column names for optional fields
        train_df: Training data (REQUIRED for ML baselines like tabular_lgb)
        
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
    
    # ML baselines require training data
    if baseline_name in ML_BASELINES:
        if train_df is None:
            raise ValueError(f"ML baseline '{baseline_name}' requires train_df to be provided")
        
        return generate_ml_baseline_scores(
            train_df=train_df,
            val_df=features_df,
            baseline_name=baseline_name,
            fold_id=fold_id,
            horizon=horizon,
            date_col=date_col,
            ticker_col=ticker_col,
            stable_id_col=stable_id_col,
            excess_return_col=excess_return_col,
            adv_20d_col=adv_20d_col,
            adv_60d_col=adv_60d_col,
            sector_col=sector_col,
            vix_col=vix_col,
            market_return_col=market_return_col,
            market_vol_col=market_vol_col,
            beta_col=beta_col
        )
    
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


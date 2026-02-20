#!/usr/bin/env python
"""
Chapter 9: FinText-TSFM Walk-Forward Evaluation Script

This script runs the FinText adapter through the frozen Chapter 6 evaluation pipeline.

Usage:
    # SMOKE mode (fast plumbing check, 3 folds)
    python scripts/run_chapter9_fintext.py --mode smoke --stub

    # FULL mode (complete evaluation, 109 folds × 3 horizons)
    python scripts/run_chapter9_fintext.py --mode full

    # With custom output directory
    python scripts/run_chapter9_fintext.py --mode full --output-dir evaluation_outputs/fintext_small

    # Model size ablation
    python scripts/run_chapter9_fintext.py --mode smoke --model-size Tiny
    python scripts/run_chapter9_fintext.py --mode smoke --model-size Mini
    python scripts/run_chapter9_fintext.py --mode smoke --model-size Small  # default

    # Lookback window ablation
    python scripts/run_chapter9_fintext.py --mode smoke --lookback 252
"""

import sys
from pathlib import Path
import argparse
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.run_evaluation import (
    run_experiment,
    ExperimentSpec,
    SMOKE_MODE,
    FULL_MODE,
)
from src.models.fintext_adapter import (
    FinTextAdapter,
    initialize_fintext_adapter,
)
from src.data.excess_return_store import ExcessReturnStore

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger(__name__)


# ============================================================================
# FINTEXT SCORER FUNCTION
# ============================================================================

# Global instance (initialized once, reused across folds)
_fintext_adapter: Optional[FinTextAdapter] = None


def setup_fintext_adapter(
    db_path: str = "data/features.duckdb",
    model_size: str = "Small",
    model_dataset: str = "US",
    lookback: int = 21,
    num_samples: int = 20,
    device: str = "cpu",
    use_stub: bool = False,
    score_aggregation: str = "trimmed_mean",
) -> FinTextAdapter:
    """
    Initialize the FinText adapter (once per evaluation run).
    
    Args:
        db_path: Path to DuckDB database
        model_size: "Tiny" (8M), "Mini" (20M), "Small" (46M)
        model_dataset: "US", "Global", "Augmented"
        lookback: Context window in trading days (default 21)
        num_samples: Distribution samples per prediction (default 20)
        device: Device for inference ("cpu" or "cuda")
        use_stub: If True, use StubChronosPredictor for testing
        score_aggregation: "median", "mean", or "trimmed_mean"
    
    Returns:
        FinTextAdapter instance
    """
    global _fintext_adapter
    
    if _fintext_adapter is not None:
        logger.info("FinText adapter already initialized. Reusing.")
        return _fintext_adapter
    
    logger.info("=" * 70)
    logger.info("INITIALIZING FINTEXT ADAPTER")
    logger.info("=" * 70)
    
    _fintext_adapter = initialize_fintext_adapter(
        db_path=db_path,
        model_size=model_size,
        model_dataset=model_dataset,
        lookback=lookback,
        num_samples=num_samples,
        device=device,
        use_stub=use_stub,
        score_aggregation=score_aggregation,
    )
    
    # Log configuration
    logger.info(f"✓ FinText adapter initialized")
    logger.info(f"  Model family: FinText/Chronos_{model_size}_{'{YEAR}'}_{model_dataset}")
    logger.info(f"  Device: {device}")
    logger.info(f"  Stub mode: {use_stub}")
    logger.info(f"  Lookback: {lookback} trading days")
    logger.info(f"  Samples: {num_samples} per prediction")
    logger.info(f"  Score aggregation: {score_aggregation}")
    
    # Log data store stats
    store = _fintext_adapter.excess_return_store
    tickers = store.get_available_tickers()
    date_range = store.get_date_range()
    logger.info(f"  Tickers: {len(tickers)}")
    logger.info(f"  Date range: {date_range[0].date()} to {date_range[1].date()}")
    logger.info("=" * 70)
    
    return _fintext_adapter


def fintext_scoring_function(
    features_df: pd.DataFrame,
    fold_id: str,
    horizon: int,
) -> pd.DataFrame:
    """
    Scoring function for FinText, matching the run_experiment() contract.
    
    This function is designed to be passed to run_experiment(scorer_fn=...).
    It expects the global _fintext_adapter to be initialized.
    
    Args:
        features_df: DataFrame containing validation features for a specific fold.
                     Includes 'date', 'ticker', 'stable_id', and horizon-specific excess_return column.
        fold_id: Identifier for the current walk-forward fold.
        horizon: The forecast horizon in trading days (e.g., 20, 60, 90).
    
    Returns:
        DataFrame in EvaluationRow format with columns:
        - as_of_date: pd.Timestamp
        - ticker: str
        - stable_id: str
        - fold_id: str
        - horizon: int
        - score: float (median predicted excess return)
        - excess_return: float (actual realized excess return)
        - [optional] adv_20d, adv_60d, sector, vix_percentile, etc.
    """
    if _fintext_adapter is None:
        raise RuntimeError(
            "FinText adapter not initialized. "
            "Call setup_fintext_adapter() before running evaluation."
        )
    
    logger.info(
        f"FinText scoring: fold={fold_id}, horizon={horizon}d, "
        f"rows={len(features_df)}"
    )
    
    # Get unique dates in this fold
    unique_dates = sorted(features_df["date"].unique())
    logger.info(f"  Scoring {len(unique_dates)} unique dates")
    
    all_scores = []
    
    for i, asof_date in enumerate(unique_dates):
        date_df = features_df[features_df["date"] == asof_date]
        tickers = date_df["ticker"].unique().tolist()
        
        logger.info(
            f"  [{i + 1}/{len(unique_dates)}] {asof_date}: {len(tickers)} tickers"
        )
        
        # Score all tickers for this date
        scores_df = _fintext_adapter.score_universe(
            asof_date=pd.Timestamp(asof_date),
            tickers=tickers,
            verbose=False,  # Already logging at higher level
        )
        
        if scores_df.empty:
            logger.warning(f"  No scores for {asof_date}")
            continue
        
        # Determine excess return column
        excess_return_col = f"excess_return_{horizon}d"
        if excess_return_col not in date_df.columns:
            excess_return_col = "excess_return"
        
        # Merge scores with evaluation metadata
        merge_cols = ["date", "ticker", "stable_id", excess_return_col]
        optional_cols = [
            "adv_20d",
            "adv_60d",
            "sector",
            "vix_percentile",
            "market_return_5d",
            "beta_252d",
        ]
        for col in optional_cols:
            if col in date_df.columns:
                merge_cols.append(col)
        
        merged = date_df[merge_cols].merge(
            scores_df[["ticker", "score", "pred_mean", "pred_std"]],
            on="ticker",
            how="inner"
        )
        
        merged = merged.rename(
            columns={"date": "as_of_date", excess_return_col: "excess_return"}
        )
        merged["fold_id"] = fold_id
        merged["horizon"] = horizon
        
        all_scores.append(merged)
    
    if not all_scores:
        raise ValueError(
            f"No scores generated for fold {fold_id}, horizon {horizon}d"
        )
    
    result = pd.concat(all_scores, ignore_index=True)
    
    # Apply EMA smoothing to reduce daily churn (half-life = 5 trading days)
    from src.models.fintext_adapter import _apply_ema_smoothing
    result = _apply_ema_smoothing(result, halflife_days=5)
    
    logger.info(f"Generated {len(result)} evaluation rows (EMA smoothed)")
    return result


# ============================================================================
# LEAK TRIPWIRES (negative controls)
# ============================================================================

def shuffle_within_date_control(
    features_df: pd.DataFrame,
    fold_id: str,
    horizon: int,
) -> pd.DataFrame:
    """
    Negative control: shuffle scores within each date.
    
    Expected: RankIC ≈ 0 (confirms signal is stock-specific, not systematic).
    """
    logger.info(f"[LEAK TRIPWIRE] Shuffle-within-date control: fold={fold_id}, horizon={horizon}d")
    
    # Get real scores
    result = fintext_scoring_function(features_df, fold_id, horizon)
    
    # Shuffle scores within each date
    np.random.seed(42)
    result["score"] = (
        result.groupby("as_of_date")["score"]
        .transform(lambda x: np.random.permutation(x))
    )
    
    return result


def lag_control(
    features_df: pd.DataFrame,
    fold_id: str,
    horizon: int,
) -> pd.DataFrame:
    """
    Negative control: shift scores forward by 1 trading day.
    
    Expected: RankIC collapses (confirms time alignment).
    """
    logger.info(f"[LEAK TRIPWIRE] +1 day lag control: fold={fold_id}, horizon={horizon}d")
    
    # Get real scores
    result = fintext_scoring_function(features_df, fold_id, horizon)
    
    # Shift scores forward by 1 trading day
    result = result.sort_values(["ticker", "as_of_date"])
    result["score"] = result.groupby("ticker")["score"].shift(1)
    result = result.dropna(subset=["score"])
    
    return result


def year_mismatch_control(
    features_df: pd.DataFrame,
    fold_id: str,
    horizon: int,
) -> pd.DataFrame:
    """
    Negative control: use deliberately wrong model year (e.g., 2023 model for 2018 dates).
    
    Expected: RankIC degrades (confirms year-specific training matters).
    
    Implementation: Temporarily override adapter's get_model_id() to return a fixed wrong year.
    """
    logger.info(f"[LEAK TRIPWIRE] Year-mismatch control: fold={fold_id}, horizon={horizon}d")
    
    if _fintext_adapter is None:
        raise RuntimeError("FinText adapter not initialized")
    
    # Save original method
    original_get_model_id = _fintext_adapter.get_model_id
    
    # Override to always use 2023 model (regardless of actual date)
    def wrong_year_model_id(asof_date):
        return f"{_fintext_adapter.model_family}_{_fintext_adapter.model_size}_2023_{_fintext_adapter.model_dataset}"
    
    _fintext_adapter.get_model_id = wrong_year_model_id
    
    try:
        result = fintext_scoring_function(features_df, fold_id, horizon)
    finally:
        # Restore original method
        _fintext_adapter.get_model_id = original_get_model_id
    
    return result


# ============================================================================
# MAIN EVALUATION RUNNER
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Chapter 9: FinText-TSFM Walk-Forward Evaluation"
    )
    
    # Evaluation mode
    parser.add_argument(
        "--mode",
        choices=["smoke", "full"],
        default="smoke",
        help="Evaluation mode: smoke (3 folds) or full (109 folds)"
    )
    
    # FinText configuration
    parser.add_argument(
        "--model-size",
        choices=["Tiny", "Mini", "Small"],
        default="Small",
        help="Model size: Tiny (8M), Mini (20M), Small (46M)"
    )
    parser.add_argument(
        "--model-dataset",
        choices=["US", "Global", "Augmented"],
        default="US",
        help="Model dataset variant"
    )
    parser.add_argument(
        "--lookback",
        type=int,
        default=21,
        help="Context window in trading days (default: 21)"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=20,
        help="Distribution samples per prediction (default: 20)"
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="Device for inference"
    )
    parser.add_argument(
        "--score-aggregation",
        choices=["median", "mean", "trimmed_mean"],
        default="trimmed_mean",
        help="Score aggregation method (default: trimmed_mean)"
    )
    parser.add_argument(
        "--stub",
        action="store_true",
        help="Use stub predictor (no model download)"
    )
    
    # Database path
    parser.add_argument(
        "--db-path",
        type=Path,
        default=Path("data/features.duckdb"),
        help="Path to DuckDB database"
    )
    
    # Output configuration
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Custom output directory (default: auto-generated)"
    )
    
    args = parser.parse_args()
    
    # ========================================================================
    # SETUP
    # ========================================================================
    
    logger.info("=" * 70)
    logger.info("CHAPTER 9: FINTEXT-TSFM EVALUATION")
    logger.info("=" * 70)
    logger.info(f"Mode: {args.mode.upper()}")
    logger.info(f"Model: FinText/Chronos_{args.model_size}_{'{YEAR}'}_{args.model_dataset}")
    logger.info(f"Lookback: {args.lookback} days")
    logger.info(f"Samples: {args.num_samples}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Stub mode: {args.stub}")
    logger.info(f"Database: {args.db_path}")
    logger.info("=" * 70)
    
    # Set evaluation mode
    eval_mode = SMOKE_MODE if args.mode == "smoke" else FULL_MODE
    
    # Set output directory
    if args.output_dir is None:
        stub_suffix = "_stub" if args.stub else ""
        size_suffix = args.model_size.lower()
        args.output_dir = Path(
            f"evaluation_outputs/chapter9_fintext_{size_suffix}_{args.mode}{stub_suffix}"
        )
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {args.output_dir}")
    
    # ========================================================================
    # LOAD FEATURES
    # ========================================================================
    
    logger.info("\nLoading features from DuckDB...")
    
    import duckdb
    con = duckdb.connect(str(args.db_path), read_only=True)
    
    try:
        # Load features
        features_df = con.execute("SELECT * FROM features").df()
        
        # Load labels (long format: horizon column)
        labels_df = con.execute("SELECT * FROM labels").df()
        
        logger.info(f"✓ Loaded {len(features_df)} feature rows")
        logger.info(f"  Date range: {features_df['date'].min()} to {features_df['date'].max()}")
        logger.info(f"  Tickers: {features_df['ticker'].nunique()}")
        logger.info(f"✓ Loaded {len(labels_df)} label rows")
        
    finally:
        con.close()
    
    # Convert date columns to datetime
    features_df["date"] = pd.to_datetime(features_df["date"])
    labels_df["as_of_date"] = pd.to_datetime(labels_df["as_of_date"])
    
    # Pivot labels to wide format (one column per horizon)
    labels_wide = labels_df.pivot_table(
        index=["as_of_date", "ticker"],
        columns="horizon",
        values="excess_return",
        aggfunc="first",
    ).reset_index()
    
    # Rename columns
    labels_wide.columns.name = None
    labels_wide = labels_wide.rename(columns={
        20: "excess_return_20d",
        60: "excess_return_60d",
        90: "excess_return_90d",
    })
    
    # Merge features with labels
    features_df = features_df.merge(
        labels_wide,
        left_on=["date", "ticker"],
        right_on=["as_of_date", "ticker"],
        how="left",
    )
    
    # Drop duplicate as_of_date column
    if "as_of_date" in features_df.columns:
        features_df = features_df.drop(columns=["as_of_date"])
    
    logger.info(f"✓ Merged features with labels: {len(features_df)} rows")
    
    # ========================================================================
    # INITIALIZE FINTEXT ADAPTER
    # ========================================================================
    
    setup_fintext_adapter(
        db_path=str(args.db_path),
        model_size=args.model_size,
        model_dataset=args.model_dataset,
        lookback=args.lookback,
        num_samples=args.num_samples,
        device=args.device,
        use_stub=args.stub,
        score_aggregation=args.score_aggregation,
    )
    
    # ========================================================================
    # RUN EVALUATION
    # ========================================================================
    
    experiment_spec = ExperimentSpec(
        name=f"chapter9_fintext_{args.model_size.lower()}_{args.mode}",
        model_type="model",
        model_name=f"fintext_{args.model_size.lower()}",
        horizons=[20, 60, 90],
        cadence="monthly",
    )
    
    logger.info("\n" + "=" * 70)
    logger.info("RUNNING WALK-FORWARD EVALUATION")
    logger.info("=" * 70)
    
    results = run_experiment(
        experiment_spec=experiment_spec,
        features_df=features_df,
        output_dir=args.output_dir,
        mode=eval_mode,
        scorer_fn=fintext_scoring_function,
    )
    
    logger.info("\n" + "=" * 70)
    logger.info("EVALUATION COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Experiment: {results['experiment_name']}")
    logger.info(f"Folds: {results['n_folds']}")
    logger.info(f"Evaluation rows: {results['n_eval_rows']}")
    logger.info(f"Output directory: {results['output_paths']['root']}")
    logger.info("=" * 70)
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    
    logger.info("")
    logger.info("=" * 70)
    logger.info("CHAPTER 9 EVALUATION COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Output directory: {args.output_dir}")
    logger.info("")
    logger.info("Next steps:")
    logger.info("  1. Review metrics: RankIC, quintile spread, hit rate")
    logger.info("  2. Compare vs frozen baselines (Chapter 6 & 7)")
    logger.info("  3. Check gate results (RankIC ≥ 0.02 for ≥2 horizons)")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()

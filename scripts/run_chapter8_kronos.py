#!/usr/bin/env python
"""
Chapter 8: Kronos Walk-Forward Evaluation Script

This script runs the Kronos adapter through the frozen Chapter 6 evaluation pipeline.

Usage:
    # SMOKE mode (fast plumbing check)
    python scripts/run_chapter8_kronos.py --mode smoke --stub

    # FULL mode (complete evaluation, ~2-4 hours)
    python scripts/run_chapter8_kronos.py --mode full

    # With custom output directory
    python scripts/run_chapter8_kronos.py --mode full --output-dir evaluation_outputs/kronos_custom
"""

# CRITICAL: Disable MPS (Apple GPU) globally BEFORE any PyTorch/Kronos imports
# This prevents device mismatch errors on Mac where Kronos creates MPS tensors
# while model is on CPU
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"  # Disable MPS entirely

import sys
from pathlib import Path
import argparse
import logging
import pandas as pd
import duckdb
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.run_evaluation import (
    run_experiment,
    ExperimentSpec,
    SMOKE_MODE,
    FULL_MODE,
)
from src.models.kronos_adapter import (
    KronosAdapter,
    KRONOS_AVAILABLE,
)
from src.data.prices_store import PricesStore
from src.data.trading_calendar import load_global_trading_calendar

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
# KRONOS SCORER FUNCTION
# ============================================================================

# Global instances (initialized once, reused across folds)
_kronos_adapter = None
_prices_store = None


def initialize_kronos_adapter(
    db_path: str = "data/features.duckdb",
    device: str = "cpu",
    use_stub: bool = False,
    batch_size: int = 12,
) -> KronosAdapter:
    """
    Initialize the Kronos adapter (once per evaluation run).
    
    Args:
        db_path: Path to DuckDB database
        device: Device for inference ("cpu" or "cuda")
        use_stub: If True, use StubPredictor for testing
        batch_size: Max tickers per Kronos call (MPS: 8-12, CPU: 4-8, CUDA: 32+)
    
    Returns:
        KronosAdapter instance
    """
    global _kronos_adapter, _prices_store
    
    if _kronos_adapter is not None:
        logger.info("Kronos adapter already initialized. Reusing.")
        return _kronos_adapter
    
    logger.info("=" * 70)
    logger.info("INITIALIZING KRONOS ADAPTER")
    logger.info("=" * 70)
    
    # Initialize adapter
    _kronos_adapter = KronosAdapter.from_pretrained(
        db_path=db_path,
        device=device,
        use_stub=use_stub,
        lookback=252,
        deterministic=True,
        batch_size=batch_size,
    )
    
    # Store prices_store reference for diagnostics
    _prices_store = _kronos_adapter.prices_store
    
    logger.info(f"✓ Kronos adapter initialized")
    logger.info(f"  Device: {device}")
    logger.info(f"  Stub mode: {use_stub}")
    logger.info(f"  Batch size: {batch_size} tickers per call (memory optimization)")
    logger.info(f"  Lookback: {_kronos_adapter.lookback} trading days")
    logger.info(f"  Trading calendar: {len(_kronos_adapter.trading_calendar)} dates")
    logger.info("=" * 70)
    
    return _kronos_adapter


def kronos_scoring_function(
    features_df: pd.DataFrame,
    fold_id: str,
    horizon: int,
) -> pd.DataFrame:
    """
    Scoring function for Kronos, matching the run_experiment() contract.
    
    This function is designed to be passed to run_experiment(scorer_fn=...).
    It expects the global _kronos_adapter to be initialized.
    
    Args:
        features_df: DataFrame containing validation features for a specific fold and date range.
                     It includes 'date', 'ticker', 'stable_id', and horizon-specific excess_return column.
        fold_id: Identifier for the current walk-forward fold.
        horizon: The forecast horizon in trading days (e.g., 20, 60, 90).
    
    Returns:
        DataFrame in EvaluationRow format, with one row per (as_of_date, ticker) in features_df.
        
        Required columns:
        - as_of_date: pd.Timestamp
        - ticker: str
        - stable_id: str
        - fold_id: str
        - horizon: int
        - score: float (Kronos score = price return proxy)
        - excess_return: float (actual realized excess return for this horizon)
        - adv_20d: float (average dollar volume)
        - adv_60d: float (average dollar volume)
        - sector: str (optional)
    """
    if _kronos_adapter is None:
        raise RuntimeError(
            "Kronos adapter not initialized. "
            "Call initialize_kronos_adapter() before running evaluation."
        )
    
    # Prepare output list
    all_scores = []
    
    # Get unique dates from features_df (already filtered to validation period)
    unique_dates = pd.Series(features_df["date"].unique()).sort_values().values
    
    logger.info(f"Scoring fold {fold_id}, horizon {horizon}d: {len(unique_dates)} dates")
    
    for i, asof_date_np in enumerate(unique_dates):
        asof_date = pd.Timestamp(asof_date_np)
        
        # Get tickers for this date
        date_features = features_df[features_df["date"] == asof_date_np].copy()
        tickers = date_features["ticker"].unique().tolist()
        
        if not tickers:
            logger.warning(f"  [{i+1}/{len(unique_dates)}] {asof_date.date()}: No tickers found. Skipping.")
            continue
        
        # Score this date's universe via batch inference
        scores_df = _kronos_adapter.score_universe_batch(
            tickers=tickers,
            asof_date=asof_date,
            horizon=horizon,
            verbose=(i % 10 == 0),  # Log every 10th date
        )
        
        if scores_df.empty:
            logger.warning(f"  [{i+1}/{len(unique_dates)}] {asof_date.date()}: No scores returned. Skipping.")
            continue
        
        # Merge scores with features to get labels + metadata
        # Rename 'date' to 'as_of_date' for merge
        date_features_renamed = date_features.rename(columns={"date": "as_of_date"})
        
        merged = date_features_renamed.merge(
            scores_df,
            on="ticker",
            how="inner",  # Only keep tickers we successfully scored
        )
        
        # Get horizon-specific excess return column
        horizon_col = f"excess_return_{horizon}d"
        if horizon_col not in merged.columns:
            logger.error(f"  [{i+1}/{len(unique_dates)}] {asof_date.date()}: Missing {horizon_col}. Skipping.")
            continue
        
        # Build EvaluationRow format
        eval_rows = merged[[
            "as_of_date", "ticker", "stable_id", horizon_col,
            "score", "adv_20d", "adv_60d", "sector"
        ]].rename(columns={
            horizon_col: "excess_return",
            "as_of_date": "as_of_date",  # Keep name
        }).copy()
        
        eval_rows["fold_id"] = fold_id
        eval_rows["horizon"] = horizon
        
        all_scores.append(eval_rows)
        
        if (i + 1) % 10 == 0 or (i + 1) == len(unique_dates):
            logger.info(
                f"  [{i+1}/{len(unique_dates)}] {asof_date.date()}: "
                f"Scored {len(eval_rows)} tickers"
            )
    
    if not all_scores:
        logger.warning(f"No scores generated for fold {fold_id}, horizon {horizon}d")
        # Return empty DataFrame with correct columns
        return pd.DataFrame(columns=[
            "as_of_date", "ticker", "stable_id", "fold_id", "horizon",
            "score", "excess_return", "adv_20d", "adv_60d", "sector"
        ])
    
    result = pd.concat(all_scores, ignore_index=True)
    logger.info(f"✓ Fold {fold_id}, horizon {horizon}d: {len(result)} total scores")
    
    return result


# ============================================================================
# NEGATIVE CONTROLS (LEAK TRIPWIRES)
# ============================================================================

def compute_leak_tripwires(
    eval_df: pd.DataFrame,
    output_dir: Path,
) -> dict:
    """
    Compute negative control checks (leak tripwires).
    
    1. Shuffle-within-date: Shuffle scores within each date, recompute RankIC → expect ~0
    2. +1 trading-day lag: Shift scores forward by +1 trading day, recompute RankIC → expect collapse
    
    Args:
        eval_df: Evaluation rows DataFrame (output from run_experiment)
        output_dir: Directory to save tripwire results
    
    Returns:
        Dictionary with tripwire results
    """
    from src.evaluation.metrics import compute_rankic_per_date
    
    logger.info("=" * 70)
    logger.info("COMPUTING LEAK TRIPWIRES (Negative Controls)")
    logger.info("=" * 70)
    
    results = {}
    
    # Compute original RankIC by horizon
    original_rankic = {}
    for horizon in eval_df["horizon"].unique():
        horizon_df = eval_df[eval_df["horizon"] == horizon].copy()
        
        # Compute RankIC per date
        ic_list = []
        for date in horizon_df["as_of_date"].unique():
            date_df = horizon_df[horizon_df["as_of_date"] == date]
            ic = compute_rankic_per_date(date_df)
            ic_list.append({"date": date, "rankic": ic})
        
        ic_df = pd.DataFrame(ic_list)
        original_rankic[int(horizon)] = {
            "median": ic_df["rankic"].median(),
            "mean": ic_df["rankic"].mean(),
            "std": ic_df["rankic"].std(),
        }
    
    logger.info("Original RankIC (baseline):")
    for h, stats in original_rankic.items():
        logger.info(f"  {h}d: median={stats['median']:.4f}, mean={stats['mean']:.4f}, std={stats['std']:.4f}")
    
    results["original_rankic"] = original_rankic
    
    # ========================================================================
    # TRIPWIRE 1: Shuffle-within-date
    # ========================================================================
    
    logger.info("\nTRIPWIRE 1: Shuffle-within-date")
    
    shuffled_rankic = {}
    for horizon in eval_df["horizon"].unique():
        horizon_df = eval_df[eval_df["horizon"] == horizon].copy()
        
        # Shuffle scores within each date
        horizon_df["score_shuffled"] = (
            horizon_df.groupby("as_of_date")["score"]
            .transform(lambda x: x.sample(frac=1.0, random_state=42).values)
        )
        
        # Replace score with shuffled
        horizon_df_shuffled = horizon_df.copy()
        horizon_df_shuffled["score"] = horizon_df_shuffled["score_shuffled"]
        
        # Recompute RankIC
        ic_list = []
        for date in horizon_df_shuffled["as_of_date"].unique():
            date_df = horizon_df_shuffled[horizon_df_shuffled["as_of_date"] == date]
            ic = compute_rankic_per_date(date_df)
            ic_list.append({"date": date, "rankic": ic})
        
        ic_df = pd.DataFrame(ic_list)
        shuffled_rankic[int(horizon)] = {
            "median": ic_df["rankic"].median(),
            "mean": ic_df["rankic"].mean(),
            "std": ic_df["rankic"].std(),
        }
    
    logger.info("Shuffled RankIC (expect ~0):")
    for h, stats in shuffled_rankic.items():
        logger.info(f"  {h}d: median={stats['median']:.4f}, mean={stats['mean']:.4f}, std={stats['std']:.4f}")
    
    results["shuffled_rankic"] = shuffled_rankic
    
    # Verdict
    shuffle_pass = all(
        abs(stats["median"]) < 0.01 for stats in shuffled_rankic.values()
    )
    
    if shuffle_pass:
        logger.info("✓ TRIPWIRE 1 PASSED: Shuffled RankIC ≈ 0")
    else:
        logger.warning("⚠️  TRIPWIRE 1 FAILED: Shuffled RankIC not close to 0. Possible leak.")
    
    results["shuffle_pass"] = shuffle_pass
    
    # ========================================================================
    # TRIPWIRE 2: +1 trading-day lag
    # ========================================================================
    
    logger.info("\nTRIPWIRE 2: +1 trading-day lag")
    
    lagged_rankic = {}
    for horizon in eval_df["horizon"].unique():
        horizon_df = eval_df[eval_df["horizon"] == horizon].copy()
        
        # Sort by ticker, date
        horizon_df = horizon_df.sort_values(["ticker", "as_of_date"])
        
        # Shift scores forward by +1 trading day per ticker
        horizon_df["score_lagged"] = (
            horizon_df.groupby("ticker")["score"]
            .shift(1)  # Lag by 1 row = 1 trading day (assuming daily rebalance)
        )
        
        # Drop rows with NaN lagged scores
        horizon_df_lagged = horizon_df.dropna(subset=["score_lagged"]).copy()
        horizon_df_lagged["score"] = horizon_df_lagged["score_lagged"]
        
        if len(horizon_df_lagged) == 0:
            logger.warning(f"  {horizon}d: No lagged data available. Skipping.")
            continue
        
        # Recompute RankIC
        ic_list = []
        for date in horizon_df_lagged["as_of_date"].unique():
            date_df = horizon_df_lagged[horizon_df_lagged["as_of_date"] == date]
            ic = compute_rankic_per_date(date_df)
            ic_list.append({"date": date, "rankic": ic})
        
        ic_df = pd.DataFrame(ic_list)
        lagged_rankic[int(horizon)] = {
            "median": ic_df["rankic"].median(),
            "mean": ic_df["rankic"].mean(),
            "std": ic_df["rankic"].std(),
        }
    
    logger.info("Lagged RankIC (expect collapse):")
    for h, stats in lagged_rankic.items():
        logger.info(f"  {h}d: median={stats['median']:.4f}, mean={stats['mean']:.4f}, std={stats['std']:.4f}")
    
    results["lagged_rankic"] = lagged_rankic
    
    # Verdict: lagged RankIC should be materially lower than original
    lag_pass = all(
        lagged_rankic.get(h, {"median": 0})["median"] < original_rankic[h]["median"] * 0.5
        for h in original_rankic.keys()
        if h in lagged_rankic
    )
    
    if lag_pass:
        logger.info("✓ TRIPWIRE 2 PASSED: Lagged RankIC collapses")
    else:
        logger.warning("⚠️  TRIPWIRE 2 FAILED: Lagged RankIC does not collapse. Possible leak.")
    
    results["lag_pass"] = lag_pass
    
    # ========================================================================
    # Save results
    # ========================================================================
    
    tripwire_path = output_dir / "leak_tripwires.json"
    import json
    with open(tripwire_path, "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\n✓ Leak tripwire results saved to: {tripwire_path}")
    logger.info("=" * 70)
    
    return results


# ============================================================================
# CORRELATION CHECK (Momentum Clone Detection)
# ============================================================================

def compute_momentum_correlation(
    eval_df: pd.DataFrame,
    baseline_eval_rows: pd.DataFrame,
    output_dir: Path,
) -> dict:
    """
    Compute cross-sectional correlation between Kronos scores and frozen factor baselines.
    
    Args:
        eval_df: Kronos evaluation rows
        baseline_eval_rows: Baseline evaluation rows (e.g., mom_12m_monthly from frozen baseline)
        output_dir: Directory to save correlation results
    
    Returns:
        Dictionary with correlation stats
    """
    logger.info("=" * 70)
    logger.info("MOMENTUM CLONE CHECK (Correlation Analysis)")
    logger.info("=" * 70)
    
    results = {}
    
    # Merge Kronos scores with baseline scores on (date, ticker, horizon)
    merged = eval_df.merge(
        baseline_eval_rows[["as_of_date", "ticker", "horizon", "score"]],
        on=["as_of_date", "ticker", "horizon"],
        how="inner",
        suffixes=("_kronos", "_baseline"),
    )
    
    if len(merged) == 0:
        logger.warning("No overlapping (date, ticker, horizon) pairs. Cannot compute correlation.")
        return {"correlation_per_horizon": {}}
    
    # Compute correlation per horizon
    for horizon in merged["horizon"].unique():
        horizon_df = merged[merged["horizon"] == horizon]
        
        # Cross-sectional correlation per date, then average
        corr_per_date = (
            horizon_df.groupby("as_of_date")
            .apply(lambda x: x["score_kronos"].corr(x["score_baseline"]))
            .dropna()
        )
        
        results[int(horizon)] = {
            "median_corr": corr_per_date.median(),
            "mean_corr": corr_per_date.mean(),
            "std_corr": corr_per_date.std(),
        }
        
        logger.info(
            f"  {horizon}d: median_corr={corr_per_date.median():.3f}, "
            f"mean_corr={corr_per_date.mean():.3f}, std={corr_per_date.std():.3f}"
        )
    
    # Verdict: correlation should be < 0.5 (not a pure momentum clone)
    clone_pass = all(
        abs(stats["median_corr"]) < 0.5 for stats in results.values()
    )
    
    if clone_pass:
        logger.info("✓ MOMENTUM CLONE CHECK PASSED: Kronos is not a pure momentum clone (corr < 0.5)")
    else:
        logger.warning("⚠️  MOMENTUM CLONE CHECK WARNING: High correlation with momentum baseline (corr ≥ 0.5)")
    
    results["clone_pass"] = clone_pass
    
    # Save results
    corr_path = output_dir / "momentum_correlation.json"
    import json
    with open(corr_path, "w") as f:
        json.dump({"correlation_per_horizon": results, "clone_pass": clone_pass}, f, indent=2)
    
    logger.info(f"\n✓ Momentum correlation results saved to: {corr_path}")
    logger.info("=" * 70)
    
    return results


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run Chapter 8 Kronos evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        choices=["smoke", "full"],
        default="smoke",
        help="Evaluation mode: 'smoke' (fast, 3 folds) or 'full' (all folds)",
    )
    
    parser.add_argument(
        "--stub",
        action="store_true",
        help="Use StubPredictor instead of real Kronos (for testing)",
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device for Kronos inference",
    )
    
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Custom output directory (default: evaluation_outputs/chapter8_kronos_{mode})",
    )
    
    parser.add_argument(
        "--db-path",
        type=Path,
        default=Path("data/features.duckdb"),
        help="Path to DuckDB database",
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=12,
        help="Max tickers per Kronos call (MPS: 8-12, CPU: 4-8, CUDA: 32+)",
    )
    
    args = parser.parse_args()
    
    # ========================================================================
    # SETUP
    # ========================================================================
    
    logger.info("=" * 70)
    logger.info("CHAPTER 8: KRONOS WALK-FORWARD EVALUATION")
    logger.info("=" * 70)
    logger.info(f"Mode: {args.mode.upper()}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Stub mode: {args.stub}")
    logger.info(f"Batch size: {args.batch_size} tickers per call")
    logger.info(f"Database: {args.db_path}")
    logger.info("=" * 70)
    
    # Check if Kronos is available
    if not args.stub and not KRONOS_AVAILABLE:
        logger.error(
            "Kronos not installed. Install from https://github.com/shiyu-coder/Kronos "
            "or use --stub for testing."
        )
        return 1
    
    # Set evaluation mode
    mode = SMOKE_MODE if args.mode == "smoke" else FULL_MODE
    
    # Set output directory
    if args.output_dir is None:
        stub_suffix = "_stub" if args.stub else ""
        args.output_dir = Path(f"evaluation_outputs/chapter8_kronos_{args.mode}{stub_suffix}")
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # ========================================================================
    # LOAD FEATURES
    # ========================================================================
    
    logger.info("\nLoading features from DuckDB...")
    
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
    # INITIALIZE KRONOS ADAPTER
    # ========================================================================
    
    initialize_kronos_adapter(
        db_path=str(args.db_path),
        device=args.device,
        use_stub=args.stub,
        batch_size=args.batch_size,
    )
    
    # ========================================================================
    # RUN EVALUATION
    # ========================================================================
    
    experiment_spec = ExperimentSpec(
        name=f"chapter8_kronos_{args.mode}",
        model_type="model",
        model_name="kronos",
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
        mode=mode,
        scorer_fn=kronos_scoring_function,
    )
    
    logger.info("\n" + "=" * 70)
    logger.info("EVALUATION COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Experiment: {results['experiment_name']}")
    logger.info(f"Folds: {results['n_folds']}")
    logger.info(f"Evaluation rows: {results['n_eval_rows']}")
    logger.info(f"Output directory: {results['output_paths']['root']}")
    
    # ========================================================================
    # RUN LEAK TRIPWIRES
    # ========================================================================
    
    eval_df = results["eval_rows_df"]
    
    tripwire_results = compute_leak_tripwires(
        eval_df=eval_df,
        output_dir=results["output_paths"]["root"],
    )
    
    # ========================================================================
    # MOMENTUM CORRELATION CHECK (if baseline available)
    # ========================================================================
    
    # Try to load frozen baseline for momentum correlation check
    baseline_path = Path("evaluation_outputs/chapter6_closure_real/baseline_mom_12m_monthly/eval_rows.parquet")
    
    if baseline_path.exists():
        logger.info(f"\nLoading frozen baseline from: {baseline_path}")
        baseline_eval_rows = pd.read_parquet(baseline_path)
        
        # Ensure date column is renamed to as_of_date
        if "date" in baseline_eval_rows.columns and "as_of_date" not in baseline_eval_rows.columns:
            baseline_eval_rows["as_of_date"] = pd.to_datetime(baseline_eval_rows["date"])
        
        corr_results = compute_momentum_correlation(
            eval_df=eval_df,
            baseline_eval_rows=baseline_eval_rows,
            output_dir=results["output_paths"]["root"],
        )
    else:
        logger.warning(f"Frozen baseline not found at: {baseline_path}")
        logger.warning("Skipping momentum correlation check.")
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    
    logger.info("\n" + "=" * 70)
    logger.info("CHAPTER 8 EVALUATION SUMMARY")
    logger.info("=" * 70)
    
    # Print fold summaries
    fold_summaries = results["fold_summaries"]
    
    logger.info("\nMedian RankIC per horizon:")
    for horizon in [20, 60, 90]:
        horizon_folds = fold_summaries[fold_summaries["horizon"] == horizon]
        if len(horizon_folds) > 0:
            median_ic = horizon_folds["rankic_median"].median()
            logger.info(f"  {horizon}d: {median_ic:.4f}")
    
    logger.info("\nLeak Tripwires:")
    logger.info(f"  Shuffle-within-date: {'✓ PASS' if tripwire_results['shuffle_pass'] else '✗ FAIL'}")
    logger.info(f"  +1 trading-day lag: {'✓ PASS' if tripwire_results['lag_pass'] else '✗ FAIL'}")
    
    if baseline_path.exists():
        logger.info("\nMomentum Clone Check:")
        logger.info(f"  {'✓ PASS' if corr_results['clone_pass'] else '⚠️  WARNING'}")
    
    logger.info("\n" + "=" * 70)
    logger.info("✓ CHAPTER 8 PHASE 3 COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Next: Review results in {results['output_paths']['root']}")
    logger.info("      Compare vs frozen baselines (Chapter 6/7)")
    logger.info("      Freeze if passing gates (Phase 4)")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())


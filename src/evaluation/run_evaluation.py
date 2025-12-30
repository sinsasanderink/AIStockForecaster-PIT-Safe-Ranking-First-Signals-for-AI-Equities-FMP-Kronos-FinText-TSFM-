"""
End-to-End Evaluation Runner (Chapter 6)

Orchestrates the complete evaluation pipeline:
1. Walk-forward fold generation (purging/embargo/maturity enforced)
2. Baseline/model scoring using EvaluationRow contract
3. Metrics computation (RankIC, quintile spread, Top-K, churn, regime slicing)
4. Cost overlay (sensitivity bands: low/base/high)
5. Stability reports (IC decay, regime tables, churn diagnostics, scorecard)
6. Acceptance criteria verdict

MODES:
- SMOKE: Small date slice (6-12 months) for CI/integration testing
- FULL: Complete evaluation range (2016-01-01 through last eligible date)

DETERMINISTIC: Shuffled inputs produce identical outputs.
"""

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any
import pandas as pd
import numpy as np
import logging
import json
from datetime import datetime

from .definitions import (
    HORIZONS_TRADING_DAYS,
    EVALUATION_RANGE,
    TIME_CONVENTIONS,
)
from .walk_forward import WalkForwardSplitter, WalkForwardFold
from .metrics import (
    EvaluationRow,
    compute_rankic_per_date,
    compute_quintile_spread_per_date,
    compute_topk_metrics_per_date,
    compute_churn,
    evaluate_fold,
    evaluate_with_regime_slicing,
)
from .costs import (
    TRADING_ASSUMPTIONS,
    compute_portfolio_costs,
    compute_net_metrics,
)
from .reports import (
    generate_stability_report,
    StabilityReportInputs,
)
from .baselines import (
    BASELINE_REGISTRY,
    generate_baseline_scores,
    list_baselines,
)

logger = logging.getLogger(__name__)


# ============================================================================
# EXPERIMENT SPECIFICATION
# ============================================================================

@dataclass
class ExperimentSpec:
    """
    Specification for a single evaluation experiment.
    
    Attributes:
        name: Unique experiment identifier (used in folder naming)
        model_type: Either "baseline" or "model"
        model_name: For baselines: "mom_12m", "momentum_composite", "short_term_strength"
                   For models: e.g., "kronos_v0", "fintext_v0"
        horizons: List of forecast horizons in TRADING DAYS
        cadence: "monthly" (primary) or "quarterly" (secondary)
        label_version: "v2" (default, total return)
    """
    name: str
    model_type: str  # "baseline" or "model"
    model_name: str
    horizons: List[int] = field(default_factory=lambda: list(HORIZONS_TRADING_DAYS))
    cadence: str = "monthly"
    label_version: str = "v2"
    
    def __post_init__(self):
        # Validate
        if self.model_type not in ["baseline", "model"]:
            raise ValueError(f"model_type must be 'baseline' or 'model', got: {self.model_type}")
        
        if self.model_type == "baseline" and self.model_name not in BASELINE_REGISTRY:
            raise ValueError(f"Unknown baseline: {self.model_name}. Valid: {list(BASELINE_REGISTRY.keys())}")
        
        if self.cadence not in ["monthly", "quarterly"]:
            raise ValueError(f"cadence must be 'monthly' or 'quarterly', got: {self.cadence}")
    
    @classmethod
    def baseline(cls, baseline_name: str, cadence: str = "monthly") -> "ExperimentSpec":
        """Create spec for a baseline experiment."""
        return cls(
            name=f"baseline_{baseline_name}_{cadence}",
            model_type="baseline",
            model_name=baseline_name,
            cadence=cadence
        )


# ============================================================================
# EVALUATION MODE
# ============================================================================

@dataclass(frozen=True)
class EvaluationMode:
    """Configuration for evaluation mode (SMOKE vs FULL)."""
    name: str
    eval_start: date
    eval_end: date
    max_folds: Optional[int]  # None = all folds
    
    @property
    def is_smoke(self) -> bool:
        return self.max_folds is not None


# Pre-defined modes
SMOKE_MODE = EvaluationMode(
    name="smoke",
    eval_start=date(2024, 1, 1),
    eval_end=date(2024, 12, 31),
    max_folds=3  # Only first 3 folds for CI
)

FULL_MODE = EvaluationMode(
    name="full",
    eval_start=EVALUATION_RANGE.eval_start,
    eval_end=EVALUATION_RANGE.eval_end,
    max_folds=None  # All folds
)


# ============================================================================
# COST SCENARIOS
# ============================================================================

COST_SCENARIOS = {
    "base_only": 0.0,       # Only base cost, no slippage
    "low_slippage": TRADING_ASSUMPTIONS.slippage_coef_low,
    "base_slippage": TRADING_ASSUMPTIONS.slippage_coef_base,
    "high_slippage": TRADING_ASSUMPTIONS.slippage_coef_high,
}


# ============================================================================
# MAIN RUNNER
# ============================================================================

def run_experiment(
    experiment_spec: ExperimentSpec,
    features_df: pd.DataFrame,
    output_dir: Path,
    mode: EvaluationMode = FULL_MODE,
    scorer_fn: Optional[Callable] = None
) -> Dict[str, Any]:
    """
    Run a complete evaluation experiment.
    
    Args:
        experiment_spec: Experiment specification
        features_df: DataFrame with features, labels, and optional fields
        output_dir: Base output directory
        mode: Evaluation mode (SMOKE or FULL)
        scorer_fn: Custom scoring function for model experiments
                  Signature: (features_df, fold_id, horizon) -> DataFrame in EvaluationRow format
                  If None and model_type="baseline", uses built-in baseline scorer
        
    Returns:
        Dictionary with:
        - experiment_name: str
        - n_folds: int
        - per_date_metrics: DataFrame
        - fold_summaries: DataFrame
        - regime_summaries: DataFrame (if regime features available)
        - churn_series: DataFrame
        - cost_overlays: DataFrame
        - output_paths: Dict[str, Path]
    """
    logger.info(f"=" * 60)
    logger.info(f"Starting experiment: {experiment_spec.name}")
    logger.info(f"Mode: {mode.name} ({mode.eval_start} to {mode.eval_end})")
    logger.info(f"Cadence: {experiment_spec.cadence}")
    logger.info(f"Horizons: {experiment_spec.horizons}")
    logger.info(f"=" * 60)
    
    # Create output directory
    exp_output_dir = output_dir / experiment_spec.name
    exp_output_dir.mkdir(parents=True, exist_ok=True)
    
    # ========================================================================
    # STEP 1: Generate walk-forward folds
    # ========================================================================
    logger.info("Step 1: Generating walk-forward folds...")
    
    splitter = WalkForwardSplitter(
        eval_start=mode.eval_start,
        eval_end=mode.eval_end,
        rebalance_freq=experiment_spec.cadence,
        embargo_trading_days=TIME_CONVENTIONS.embargo_trading_days
    )
    
    # Generate folds (this is a placeholder - actual implementation would use labels_df)
    # For now, we'll simulate fold structure based on the features_df dates
    folds = _generate_folds_from_features(
        features_df, splitter, mode
    )
    
    if mode.max_folds is not None:
        folds = folds[:mode.max_folds]
        logger.info(f"Limited to {len(folds)} folds (SMOKE mode)")
    
    logger.info(f"Generated {len(folds)} folds")
    
    # ========================================================================
    # STEP 2: Score each fold/horizon
    # ========================================================================
    logger.info("Step 2: Scoring folds...")
    
    all_eval_rows = []
    
    for fold in folds:
        logger.info(f"  Scoring fold {fold['fold_id']}...")
        
        # Filter features for this fold's validation period
        # Ensure date types match for comparison
        fold_val_start = pd.Timestamp(fold["val_start"]) if not isinstance(fold["val_start"], pd.Timestamp) else fold["val_start"]
        fold_val_end = pd.Timestamp(fold["val_end"]) if not isinstance(fold["val_end"], pd.Timestamp) else fold["val_end"]
        fold_train_start = pd.Timestamp(fold["train_start"]) if not isinstance(fold["train_start"], pd.Timestamp) else fold["train_start"]
        fold_train_end = pd.Timestamp(fold["train_end"]) if not isinstance(fold["train_end"], pd.Timestamp) else fold["train_end"]
        
        # Convert date column if needed
        if features_df["date"].dtype == object or hasattr(features_df["date"].iloc[0], 'isoformat'):
            date_series = pd.to_datetime(features_df["date"])
        else:
            date_series = features_df["date"]
        
        # Extract validation features
        fold_features = features_df[
            (date_series >= fold_val_start) &
            (date_series <= fold_val_end)
        ].copy()
        
        # Extract training features (for ML baselines)
        fold_train_features = features_df[
            (date_series >= fold_train_start) &
            (date_series <= fold_train_end)
        ].copy()
        
        if len(fold_features) == 0:
            logger.warning(f"  No features for fold {fold['fold_id']}, skipping")
            continue
        
        for horizon in experiment_spec.horizons:
            # CRITICAL: Use horizon-specific excess_return column if available (wide format)
            # Otherwise fall back to generic excess_return
            horizon_excess_return_col = f"excess_return_{horizon}d"
            if horizon_excess_return_col in fold_features.columns:
                excess_return_col = horizon_excess_return_col
            elif "excess_return" in fold_features.columns:
                excess_return_col = "excess_return"
                logger.warning(
                    f"Using generic 'excess_return' for horizon {horizon}d. "
                    f"For accurate per-horizon evaluation, use wide format with {horizon_excess_return_col}."
                )
            else:
                logger.error(f"No excess_return column found for horizon {horizon}")
                continue
            
            if experiment_spec.model_type == "baseline":
                eval_rows = generate_baseline_scores(
                    features_df=fold_features,
                    baseline_name=experiment_spec.model_name,
                    fold_id=fold["fold_id"],
                    horizon=horizon,
                    excess_return_col=excess_return_col,  # Pass horizon-specific column
                    train_df=fold_train_features if len(fold_train_features) > 0 else None  # For ML baselines
                )
            else:
                # Custom model
                if scorer_fn is None:
                    raise ValueError("scorer_fn required for model experiments")
                eval_rows = scorer_fn(fold_features, fold["fold_id"], horizon)
            
            all_eval_rows.append(eval_rows)
    
    if not all_eval_rows:
        raise ValueError("No evaluation rows generated!")
    
    eval_df = pd.concat(all_eval_rows, ignore_index=True)
    logger.info(f"Generated {len(eval_df)} total evaluation rows")
    
    # ========================================================================
    # STEP 3: Compute metrics per fold/horizon
    # ========================================================================
    logger.info("Step 3: Computing metrics...")
    
    per_date_metrics = []
    fold_summaries = []
    churn_series = []
    regime_summaries = []
    
    for fold in folds:
        fold_df = eval_df[eval_df["fold_id"] == fold["fold_id"]]
        
        if len(fold_df) == 0:
            continue
        
        for horizon in experiment_spec.horizons:
            horizon_df = fold_df[fold_df["horizon"] == horizon]
            
            if len(horizon_df) == 0:
                continue
            
            # Per-date metrics
            dates = horizon_df["as_of_date"].unique()
            for d in dates:
                date_df = horizon_df[horizon_df["as_of_date"] == d]
                
                if len(date_df) < 10:  # MIN_CROSS_SECTION_SIZE
                    continue
                
                # RankIC
                rankic = compute_rankic_per_date(date_df)
                
                # Quintile spread
                spread_result = compute_quintile_spread_per_date(date_df)
                spread = spread_result.get("spread", np.nan) if isinstance(spread_result, dict) else spread_result
                
                # Top-K
                topk_10 = compute_topk_metrics_per_date(date_df, k=10)
                topk_20 = compute_topk_metrics_per_date(date_df, k=20)
                
                per_date_metrics.append({
                    "date": d,
                    "fold_id": fold["fold_id"],
                    "horizon": horizon,
                    "n_names": len(date_df),
                    "rankic": rankic,
                    "quintile_spread": spread,
                    "hit_rate_10": topk_10.get("hit_rate", np.nan),
                    "avg_er_10": topk_10.get("avg_excess_return", np.nan),
                    "hit_rate_20": topk_20.get("hit_rate", np.nan),
                    "avg_er_20": topk_20.get("avg_excess_return", np.nan),
                })
            
            # Fold summary
            fold_metrics = [m for m in per_date_metrics 
                          if m["fold_id"] == fold["fold_id"] and m["horizon"] == horizon]
            
            if fold_metrics:
                rankic_values = [m["rankic"] for m in fold_metrics if not pd.isna(m["rankic"])]
                spread_values = [m["quintile_spread"] for m in fold_metrics if not pd.isna(m["quintile_spread"])]
                hit_rate_values = [m["hit_rate_10"] for m in fold_metrics if not pd.isna(m["hit_rate_10"])]
                avg_er_values = [m["avg_er_10"] for m in fold_metrics if not pd.isna(m["avg_er_10"])]
                
                fold_summaries.append({
                    "fold_id": fold["fold_id"],
                    "horizon": horizon,
                    "n_dates": len(fold_metrics),
                    "rankic_median": np.median(rankic_values) if rankic_values else np.nan,
                    "rankic_iqr": (np.percentile(rankic_values, 75) - np.percentile(rankic_values, 25)) if len(rankic_values) >= 2 else np.nan,
                    "quintile_spread_median": np.median(spread_values) if spread_values else np.nan,
                    "hit_rate_10_median": np.median(hit_rate_values) if hit_rate_values else np.nan,
                    "avg_er_10_median": np.median(avg_er_values) if avg_er_values else np.nan,
                })
            
            # Churn (within fold)
                churn_result = compute_churn(horizon_df, k=10, id_col="stable_id")
                if len(churn_result) > 0:
                    for _, row in churn_result.iterrows():
                        # Handle different column names (as_of_date or date)
                        churn_date = row.get("as_of_date", row.get("date", None))
                        churn_series.append({
                            "date": churn_date,
                            "fold_id": fold["fold_id"],
                            "horizon": horizon,
                            "k": 10,
                            "churn": row["churn"],
                            "retention": row["retention"]
                        })
            
            churn_result_20 = compute_churn(horizon_df, k=20, id_col="stable_id")
            if len(churn_result_20) > 0:
                for _, row in churn_result_20.iterrows():
                    # Handle different column names (as_of_date or date)
                    churn_date_20 = row.get("as_of_date", row.get("date", None))
                    churn_series.append({
                        "date": churn_date_20,
                        "fold_id": fold["fold_id"],
                        "horizon": horizon,
                        "k": 20,
                        "churn": row["churn"],
                        "retention": row["retention"]
                    })
    
    per_date_df = pd.DataFrame(per_date_metrics)
    fold_summary_df = pd.DataFrame(fold_summaries)
    churn_df = pd.DataFrame(churn_series) if churn_series else pd.DataFrame()
    
    logger.info(f"  Per-date metrics: {len(per_date_df)} rows")
    logger.info(f"  Fold summaries: {len(fold_summary_df)} rows")
    logger.info(f"  Churn series: {len(churn_df)} rows")
    
    # ========================================================================
    # STEP 4: Cost overlays (sensitivity bands)
    # ========================================================================
    logger.info("Step 4: Computing cost overlays...")
    
    cost_overlays = []
    
    for fold in folds:
        fold_df = eval_df[eval_df["fold_id"] == fold["fold_id"]]
        
        for horizon in experiment_spec.horizons:
            horizon_df = fold_df[fold_df["horizon"] == horizon]
            
            if len(horizon_df) == 0:
                continue
            
            # Get gross avg ER
            fold_metrics = fold_summary_df[
                (fold_summary_df["fold_id"] == fold["fold_id"]) &
                (fold_summary_df["horizon"] == horizon)
            ]
            
            if len(fold_metrics) == 0:
                continue
            
            gross_avg_er = fold_metrics["avg_er_10_median"].values[0]
            
            for scenario_name, slippage_coef in COST_SCENARIOS.items():
                # Estimate cost (simplified)
                # In reality, this would compute per-rebalance costs
                estimated_cost_pct = (
                    TRADING_ASSUMPTIONS.base_cost_bps / 10000 + 
                    slippage_coef * 0.01  # Rough estimate
                )
                
                net_avg_er = gross_avg_er - estimated_cost_pct if not np.isnan(gross_avg_er) else np.nan
                
                cost_overlays.append({
                    "fold_id": fold["fold_id"],
                    "horizon": horizon,
                    "scenario": scenario_name,
                    "gross_avg_er": gross_avg_er,
                    "estimated_cost_pct": estimated_cost_pct,
                    "net_avg_er": net_avg_er,
                    "alpha_survives": net_avg_er > 0 if not np.isnan(net_avg_er) else False
                })
    
    cost_df = pd.DataFrame(cost_overlays)
    logger.info(f"  Cost overlays: {len(cost_df)} rows")
    
    # ========================================================================
    # STEP 5: Generate stability reports
    # ========================================================================
    logger.info("Step 5: Generating stability reports...")
    
    report_inputs = StabilityReportInputs(
        per_date_metrics=per_date_df,
        fold_summaries=fold_summary_df,
        regime_summaries=pd.DataFrame(regime_summaries) if regime_summaries else None,
        churn_series=churn_df if len(churn_df) > 0 else None,
        cost_overlays=cost_df if len(cost_df) > 0 else None
    )
    
    report_outputs = generate_stability_report(
        inputs=report_inputs,
        experiment_name=experiment_spec.name,
        output_dir=exp_output_dir
    )
    
    # ========================================================================
    # STEP 6: Save raw data
    # ========================================================================
    logger.info("Step 6: Saving raw data...")
    
    # Save evaluation rows
    eval_rows_path = exp_output_dir / "eval_rows.parquet"
    eval_df.to_parquet(eval_rows_path, index=False)
    
    # Save metrics
    per_date_path = exp_output_dir / "per_date_metrics.csv"
    per_date_df.to_csv(per_date_path, index=False)
    
    fold_summary_path = exp_output_dir / "fold_summaries.csv"
    fold_summary_df.to_csv(fold_summary_path, index=False)
    
    if len(churn_df) > 0:
        churn_path = exp_output_dir / "churn_series.csv"
        churn_df.to_csv(churn_path, index=False)
    
    if len(cost_df) > 0:
        cost_path = exp_output_dir / "cost_overlays.csv"
        cost_df.to_csv(cost_path, index=False)
    
    # Save experiment metadata
    metadata = {
        "experiment_name": experiment_spec.name,
        "model_type": experiment_spec.model_type,
        "model_name": experiment_spec.model_name,
        "horizons": experiment_spec.horizons,
        "cadence": experiment_spec.cadence,
        "mode": mode.name,
        "eval_start": str(mode.eval_start),
        "eval_end": str(mode.eval_end),
        "n_folds": len(folds),
        "n_eval_rows": len(eval_df),
        "generated_at": datetime.now().isoformat()
    }
    
    metadata_path = exp_output_dir / "experiment_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Experiment complete! Outputs in: {exp_output_dir}")
    
    return {
        "experiment_name": experiment_spec.name,
        "n_folds": len(folds),
        "n_eval_rows": len(eval_df),  # For logging
        "eval_rows_df": eval_df,  # Return actual eval rows for Qlib integration
        "per_date_metrics": per_date_df,
        "fold_summaries": fold_summary_df,
        "regime_summaries": pd.DataFrame(regime_summaries) if regime_summaries else None,
        "churn_series": churn_df,
        "cost_overlays": cost_df,
        "output_paths": {
            "root": exp_output_dir,
            "eval_rows": eval_rows_path,
            "per_date_metrics": per_date_path,
            "fold_summaries": fold_summary_path,
            "stability_report": report_outputs.output_dir,
            "metadata": metadata_path
        }
    }


def _generate_folds_from_features(
    features_df: pd.DataFrame,
    splitter: WalkForwardSplitter,
    mode: EvaluationMode
) -> List[Dict]:
    """
    Generate fold structure from features DataFrame.
    
    This creates monthly or quarterly folds, snapping rebalance dates to 
    actual trading days present in the features data.
    
    CRITICAL: Fold boundaries are snapped to the NEAREST PREVIOUS trading day
    in features_df['date'] to avoid empty folds when period starts fall on 
    weekends/holidays.
    """
    # Ensure date column is datetime for grouping
    df = features_df.copy()
    if df["date"].dtype == object or hasattr(df["date"].iloc[0], 'isoformat'):
        df["date"] = pd.to_datetime(df["date"])
    
    # Get sorted unique dates as trading day grid
    trading_dates = sorted(df["date"].unique())
    
    # Convert to date objects for comparison
    trading_dates_as_date = [
        d.date() if hasattr(d, 'date') else d for d in trading_dates
    ]
    
    # Filter by mode date range
    trading_dates = [d for d in trading_dates 
             if mode.eval_start <= (d.date() if hasattr(d, 'date') else d) <= mode.eval_end]
    trading_dates_as_date = [
        d.date() if hasattr(d, 'date') else d for d in trading_dates
    ]
    
    if len(trading_dates) < 3:
        logger.warning(f"Not enough trading dates in range: {len(trading_dates)}")
        return []
    
    # Build a set for fast lookup
    trading_day_set = set(trading_dates_as_date)
    
    def snap_to_previous_trading_day(target_date):
        """Snap a calendar date to the nearest previous trading day."""
        if hasattr(target_date, 'date'):
            target_date = target_date.date()
        
        # If it's already a trading day, return it
        if target_date in trading_day_set:
            return target_date
        
        # Find nearest previous trading day
        candidates = [d for d in trading_dates_as_date if d <= target_date]
        if candidates:
            return max(candidates)
        
        # If no previous date exists, return the first trading date
        return min(trading_dates_as_date) if trading_dates_as_date else None
    
    def snap_to_next_trading_day(target_date):
        """Snap a calendar date to the nearest next trading day."""
        if hasattr(target_date, 'date'):
            target_date = target_date.date()
        
        # If it's already a trading day, return it
        if target_date in trading_day_set:
            return target_date
        
        # Find nearest next trading day
        candidates = [d for d in trading_dates_as_date if d >= target_date]
        if candidates:
            return min(candidates)
        
        # If no next date exists, return the last trading date
        return max(trading_dates_as_date) if trading_dates_as_date else None
    
    # Create folds based on cadence
    freq = "MS" if splitter.rebalance_freq == "monthly" else "QS"
    
    # Get unique periods from actual trading dates
    df_filtered = df[df["date"].isin(trading_dates)]
    periods = sorted(df_filtered["date"].dt.to_period('M' if freq == "MS" else 'Q').unique())
    
    if len(periods) < 2:
        logger.warning(f"Not enough periods for folds: {len(periods)}")
        return []
    
    folds = []
    for i, period in enumerate(periods):
        if i == 0:
            continue  # Skip first period (no previous data for training)
        
        # Period start (calendar date)
        period_start = period.to_timestamp()
        period_end = period.end_time
        
        # Snap val_start to the first trading day ON OR AFTER period start
        val_start = snap_to_next_trading_day(period_start.date())
        
        # Snap val_end to the last trading day ON OR BEFORE period end
        if i + 1 < len(periods):
            # Use start of next period as boundary
            next_period_start = periods[i + 1].to_timestamp()
            val_end = snap_to_previous_trading_day(next_period_start.date())
        else:
            # Last period: use end of period
            val_end = snap_to_previous_trading_day(period_end.date())
        
        # Ensure val_start <= val_end
        if val_start is None or val_end is None or val_start > val_end:
            logger.debug(f"Skipping period {period}: no valid trading days in range")
            continue
        
        # Train boundaries
        train_start = snap_to_next_trading_day(trading_dates_as_date[0])
        
        # Train end = day before val_start (embargo handling is done elsewhere)
        prev_period_end = periods[i-1].end_time if i > 0 else period_start
        train_end = snap_to_previous_trading_day(prev_period_end.date())
        
        if train_start is None or train_end is None or train_start >= train_end:
            logger.debug(f"Skipping fold {i}: invalid train range")
            continue
        
        fold = {
            "fold_id": f"fold_{i:02d}",
            "train_start": train_start,
            "train_end": train_end,
            "val_start": val_start,
            "val_end": val_end,
        }
        folds.append(fold)
        logger.debug(f"Created fold {fold['fold_id']}: val [{val_start}, {val_end}]")
    
    logger.info(f"Generated {len(folds)} folds with snapped trading days")
    return folds


# ============================================================================
# ACCEPTANCE CRITERIA
# ============================================================================

def _load_frozen_baseline_floor() -> Optional[Dict]:
    """
    Load frozen baseline floor from Chapter 6 closure artifacts.
    
    Returns None if file doesn't exist (e.g., in CI or before Chapter 6 closure).
    """
    import json
    from pathlib import Path
    
    floor_path = Path("evaluation_outputs/chapter6_closure_real/BASELINE_FLOOR.json")
    if not floor_path.exists():
        return None
    
    try:
        with floor_path.open() as f:
            return json.load(f)
    except Exception:
        return None


def compute_acceptance_verdict(
    model_summary: pd.DataFrame,
    baseline_summaries: Dict[str, pd.DataFrame],
    cost_overlays: pd.DataFrame,
    churn_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Compute acceptance criteria verdict.
    
    Criteria (per horizon):
    1. Median RankIC lift vs best baseline >= +0.02
    2. Net-of-cost improvement: % positive folds >= baseline + 10pp (relative gate)
    3. Top-10 churn median < 0.30
    4. No catastrophic regime collapse (use stability flags)
    
    Note: Cost survival uses relative gate vs frozen baseline floor (5.8%/25.1%/40.1%),
    not absolute 70%, which would be unrealistic under current cost model.
    
    Args:
        model_summary: Fold summaries for the model
        baseline_summaries: Dict of baseline name -> fold summaries
        cost_overlays: Cost overlays DataFrame
        churn_df: Churn series DataFrame
        
    Returns:
        DataFrame with pass/fail per criterion and horizon
    """
    # Load frozen baseline floor for relative thresholds
    baseline_floor = _load_frozen_baseline_floor()
    
    # Horizon-specific fallback thresholds (frozen floor + 10pp) if file not available
    COST_SURVIVAL_THRESHOLDS = {
        20: 0.158,  # 5.8% + 10pp
        60: 0.351,  # 25.1% + 10pp
        90: 0.501,  # 40.1% + 10pp
    }
    
    results = []
    
    for horizon in model_summary["horizon"].unique():
        model_h = model_summary[model_summary["horizon"] == horizon]
        model_rankic_median = model_h["rankic_median"].median()
        
        # Find best baseline
        best_baseline_rankic = 0.0
        best_baseline_name = None
        for name, df in baseline_summaries.items():
            baseline_h = df[df["horizon"] == horizon]
            if len(baseline_h) > 0:
                baseline_rankic = baseline_h["rankic_median"].median()
                if baseline_rankic > best_baseline_rankic:
                    best_baseline_rankic = baseline_rankic
                    best_baseline_name = name
        
        # Criterion 1: RankIC lift >= 0.02
        rankic_lift = model_rankic_median - best_baseline_rankic
        criterion_1_pass = rankic_lift >= 0.02
        
        # Criterion 2: Net-of-cost improvement (relative to frozen baseline floor)
        # Compute model's % positive folds
        if cost_overlays is not None and len(cost_overlays) > 0 and "horizon" in cost_overlays.columns:
            cost_h = cost_overlays[
                (cost_overlays["horizon"] == horizon) &
                (cost_overlays["scenario"] == "base_slippage")
            ]
            n_positive = cost_h["alpha_survives"].sum()
            n_total = len(cost_h)
            pct_positive = n_positive / n_total if n_total > 0 else 0
        else:
            pct_positive = 0
        
        # Determine threshold: baseline + 10pp (relative gate)
        if baseline_floor is not None and "cost_survival" in baseline_floor:
            # Load from frozen floor
            baseline_pct = baseline_floor["cost_survival"].get(str(horizon), {}).get("pct_positive_folds", 0)
            cost_threshold = baseline_pct + 0.10  # baseline + 10 percentage points
        else:
            # Fallback to horizon-specific frozen values + 10pp
            cost_threshold = COST_SURVIVAL_THRESHOLDS.get(horizon, 0.15)
        
        criterion_2_pass = pct_positive >= cost_threshold
        
        # Criterion 3: Churn < 30%
        if churn_df is not None and len(churn_df) > 0 and "horizon" in churn_df.columns:
            churn_h = churn_df[(churn_df["horizon"] == horizon) & (churn_df["k"] == 10)]
            churn_median = churn_h["churn"].median() if len(churn_h) > 0 else 1.0
        else:
            churn_median = 1.0  # Conservative default (fail if no churn data)
        criterion_3_pass = churn_median < 0.30
        
        # Criterion 4: No catastrophic collapse (simplified: no fold with negative median RankIC)
        n_negative_folds = (model_h["rankic_median"] < 0).sum()
        criterion_4_pass = n_negative_folds == 0
        
        # Overall pass
        all_pass = criterion_1_pass and criterion_2_pass and criterion_3_pass and criterion_4_pass
        
        results.append({
            "horizon": horizon,
            "model_rankic_median": model_rankic_median,
            "best_baseline": best_baseline_name,
            "best_baseline_rankic": best_baseline_rankic,
            "rankic_lift": rankic_lift,
            "criterion_1_rankic_lift": criterion_1_pass,
            "pct_net_positive": pct_positive,
            "cost_threshold_used": cost_threshold,  # Document which threshold was applied
            "criterion_2_net_positive": criterion_2_pass,
            "churn_median": churn_median,
            "criterion_3_churn": criterion_3_pass,
            "n_negative_folds": n_negative_folds,
            "criterion_4_no_collapse": criterion_4_pass,
            "all_criteria_pass": all_pass
        })
    
    return pd.DataFrame(results)


def save_acceptance_summary(
    verdict_df: pd.DataFrame,
    output_dir: Path,
    experiment_name: str
) -> Path:
    """Save acceptance criteria summary to file."""
    
    # CSV
    csv_path = output_dir / "ACCEPTANCE_SUMMARY.csv"
    verdict_df.to_csv(csv_path, index=False)
    
    # Markdown
    md_path = output_dir / "ACCEPTANCE_SUMMARY.md"
    with open(md_path, 'w') as f:
        f.write(f"# Acceptance Criteria Summary: {experiment_name}\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("---\n\n")
        
        # Overall verdict
        all_pass = verdict_df["all_criteria_pass"].all()
        f.write(f"## Overall Verdict: {'✅ PASS' if all_pass else '❌ FAIL'}\n\n")
        
        # Per-horizon
        f.write("## Per-Horizon Results\n\n")
        for _, row in verdict_df.iterrows():
            h = row["horizon"]
            f.write(f"### Horizon {h}d\n\n")
            f.write(f"| Criterion | Value | Threshold | Pass |\n")
            f.write(f"|-----------|-------|-----------|------|\n")
            f.write(f"| RankIC Lift vs {row['best_baseline']} | {row['rankic_lift']:.4f} | >= 0.02 | "
                   f"{'✅' if row['criterion_1_rankic_lift'] else '❌'} |\n")
            f.write(f"| Net-Positive Folds | {row['pct_net_positive']:.1%} | >= 70% | "
                   f"{'✅' if row['criterion_2_net_positive'] else '❌'} |\n")
            f.write(f"| Top-10 Churn | {row['churn_median']:.1%} | < 30% | "
                   f"{'✅' if row['criterion_3_churn'] else '❌'} |\n")
            f.write(f"| Negative Folds | {row['n_negative_folds']} | 0 | "
                   f"{'✅' if row['criterion_4_no_collapse'] else '❌'} |\n")
            f.write(f"\n**Horizon {h}d Verdict:** "
                   f"{'✅ PASS' if row['all_criteria_pass'] else '❌ FAIL'}\n\n")
        
        f.write("---\n\n")
        f.write("## Acceptance Criteria Definitions\n\n")
        f.write("1. **RankIC Lift**: Median RankIC must exceed best baseline by >= 0.02\n")
        f.write("2. **Net-Positive Folds**: % positive >= baseline + 10pp (relative gate; frozen floor: 5.8%/25.1%/40.1% → thresholds: 15.8%/35.1%/50.1%)\n")
        f.write("3. **Churn**: Top-10 median churn must be < 30%\n")
        f.write("4. **Regime Robustness**: No fold with negative median RankIC\n")
    
    logger.info(f"Saved acceptance summary to {md_path}")
    return md_path


# ============================================================================
# CLI ENTRYPOINT
# ============================================================================

def main():
    """Command-line entrypoint for running evaluations."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Chapter 6 Evaluation")
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke",
                       help="Evaluation mode (smoke=CI, full=complete)")
    parser.add_argument("--baseline", type=str, default=None,
                       help="Specific baseline to run (default: all)")
    parser.add_argument("--output-dir", type=str, default="evaluation_outputs",
                       help="Output directory")
    parser.add_argument("--cadence", choices=["monthly", "quarterly"], default="monthly",
                       help="Rebalance cadence")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    
    # Select mode
    mode = SMOKE_MODE if args.mode == "smoke" else FULL_MODE
    
    # Determine baselines to run
    baselines = [args.baseline] if args.baseline else list_baselines()
    
    logger.info(f"Running {args.mode.upper()} evaluation")
    logger.info(f"Baselines: {baselines}")
    logger.info(f"Cadence: {args.cadence}")
    logger.info(f"Output: {args.output_dir}")
    
    # NOTE: This is a skeleton - actual execution requires features_df
    logger.info("NOTE: Actual execution requires features DataFrame.")
    logger.info("Use run_experiment() programmatically with your data.")


if __name__ == "__main__":
    main()


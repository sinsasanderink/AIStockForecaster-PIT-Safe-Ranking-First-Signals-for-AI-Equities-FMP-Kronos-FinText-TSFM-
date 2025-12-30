"""
Stability Reports (Chapter 6.5)

Pure consumer of 6.3 metrics + 6.4 costs → deterministic reporting artifacts.

NO new APIs, NO new features, NO new modeling.
Just clean rendering of what we already have.

PHILOSOPHY: Every report must be deterministic and explainable.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# Set plot style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10


# ============================================================================
# LOCKED REPORTING PARAMETERS
# ============================================================================

@dataclass(frozen=True)
class StabilityThresholds:
    """
    Locked thresholds for stability flags.
    
    These are diagnostic indicators, not hard rules.
    Once locked, DO NOT change them to make results look better.
    """
    
    # IC decay threshold
    rapid_decay_threshold: float = 0.05  # If last-third median drops > 5% vs first-third
    
    # Noise threshold
    noisy_threshold: float = 2.0  # If IQR / median > 2.0 (high relative noise)
    
    # Churn alarm threshold
    high_churn_threshold: float = 0.50  # If churn > 50%, flag as unstable
    
    # Minimum coverage for regime slicing
    min_dates_per_bucket: int = 10  # Need at least 10 dates to trust a regime slice
    min_names_per_date: int = 10  # Need at least 10 stocks per date
    
    # Rolling window for smoothing
    rolling_window: int = 6  # 6-month rolling for monthly cadence


STABILITY_THRESHOLDS = StabilityThresholds()


# ============================================================================
# REPORT CONTRACT (INPUTS/OUTPUTS)
# ============================================================================

@dataclass
class StabilityReportInputs:
    """
    Contract for stability report inputs.
    
    All inputs must be pre-computed by 6.3 (metrics) or 6.4 (costs).
    """
    
    # Per-date metrics (from evaluate_fold)
    per_date_metrics: pd.DataFrame  # Columns: date, fold_id, horizon, rankic, quintile_spread, n_names, ...
    
    # Fold summaries (from evaluate_fold)
    fold_summaries: pd.DataFrame  # Columns: fold_id, horizon, metric_median, metric_iqr, n_dates
    
    # Regime-sliced metrics (from evaluate_with_regime_slicing)
    regime_summaries: Optional[pd.DataFrame] = None  # Columns: horizon, regime_feature, bucket, metric_median, n_dates, ...
    
    # Churn series (from compute_churn)
    churn_series: Optional[pd.DataFrame] = None  # Columns: date, fold_id, horizon, k, churn, retention
    
    # Cost overlays (from 6.4)
    cost_overlays: Optional[pd.DataFrame] = None  # Columns: fold_id, horizon, scenario, net_avg_er, alpha_survives


@dataclass
class StabilityReportOutputs:
    """
    Contract for stability report outputs.
    
    All outputs are deterministic artifacts (no randomness).
    """
    
    # Output directory
    output_dir: Path
    
    # Tables (CSV)
    ic_decay_stats: Path  # Early vs late IC stats
    regime_performance: Path  # RankIC by regime bucket
    churn_diagnostics: Path  # Churn summary per fold
    stability_scorecard: Path  # One-screen summary
    
    # Figures (PNG)
    ic_decay_plot: Path  # IC time series with rolling mean
    regime_bars: Path  # RankIC by regime (grouped bars)
    churn_timeseries: Path  # Churn over time
    churn_distribution: Path  # Churn histogram
    
    # Summary report (Markdown)
    summary_report: Path  # Human-readable summary


# ============================================================================
# IC DECAY ANALYSIS
# ============================================================================

def compute_ic_decay_stats(
    per_date_metrics: pd.DataFrame,
    metric_col: str = "rankic"
) -> pd.DataFrame:
    """
    Compute early vs late stability statistics.
    
    Splits each fold into thirds:
    - Early: first 33% of dates
    - Middle: middle 33% of dates
    - Late: last 33% of dates
    
    Args:
        per_date_metrics: Per-date metrics from evaluate_fold
        metric_col: Column name for metric to analyze
        
    Returns:
        DataFrame with decay statistics per fold/horizon
    """
    results = []
    
    for (fold_id, horizon), group in per_date_metrics.groupby(["fold_id", "horizon"]):
        group = group.sort_values("date")
        n = len(group)
        
        if n < 9:  # Need at least 9 dates for thirds
            logger.warning(f"Fold {fold_id} horizon {horizon}: only {n} dates, skipping decay analysis")
            continue
        
        # Split into thirds
        third = n // 3
        early = group.iloc[:third]
        middle = group.iloc[third:2*third]
        late = group.iloc[2*third:]
        
        # Compute statistics
        early_median = early[metric_col].median()
        late_median = late[metric_col].median()
        full_median = group[metric_col].median()
        full_iqr = group[metric_col].quantile(0.75) - group[metric_col].quantile(0.25)
        
        # Decay metric
        decay = late_median - early_median
        decay_pct = decay / abs(early_median) if early_median != 0 else np.nan
        
        # Stability flags
        rapid_decay_flag = decay < -STABILITY_THRESHOLDS.rapid_decay_threshold
        noisy_flag = (full_iqr / abs(full_median)) > STABILITY_THRESHOLDS.noisy_threshold if full_median != 0 else True
        
        # Positive rate
        pct_positive = (group[metric_col] > 0).mean()
        
        results.append({
            "fold_id": fold_id,
            "horizon": horizon,
            "n_dates": n,
            "early_median": early_median,
            "late_median": late_median,
            "full_median": full_median,
            "full_iqr": full_iqr,
            "decay": decay,
            "decay_pct": decay_pct,
            "pct_positive": pct_positive,
            "rapid_decay_flag": rapid_decay_flag,
            "noisy_flag": noisy_flag
        })
    
    return pd.DataFrame(results)


def plot_ic_decay(
    per_date_metrics: pd.DataFrame,
    metric_col: str = "rankic",
    rolling_window: int = STABILITY_THRESHOLDS.rolling_window,
    output_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plot IC time series with rolling mean.
    
    Args:
        per_date_metrics: Per-date metrics from evaluate_fold
        metric_col: Column name for metric to plot
        rolling_window: Window size for rolling mean
        output_path: Path to save figure (optional)
        
    Returns:
        Matplotlib figure
    """
    n_folds = per_date_metrics["fold_id"].nunique()
    n_horizons = per_date_metrics["horizon"].nunique()
    
    fig, axes = plt.subplots(n_horizons, 1, figsize=(12, 4 * n_horizons), sharex=False)
    if n_horizons == 1:
        axes = [axes]
    
    for idx, horizon in enumerate(sorted(per_date_metrics["horizon"].unique())):
        ax = axes[idx]
        
        horizon_data = per_date_metrics[per_date_metrics["horizon"] == horizon].copy()
        
        for fold_id in sorted(horizon_data["fold_id"].unique()):
            fold_data = horizon_data[horizon_data["fold_id"] == fold_id].sort_values("date")
            
            # Plot raw IC
            ax.plot(fold_data["date"], fold_data[metric_col], 
                   alpha=0.3, linewidth=0.5, color="gray")
            
            # Plot rolling mean
            rolling_mean = fold_data[metric_col].rolling(window=rolling_window, center=True).mean()
            ax.plot(fold_data["date"], rolling_mean, 
                   linewidth=2, label=f"{fold_id}")
        
        # Add zero line
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        
        ax.set_ylabel(f"{metric_col.upper()}")
        ax.set_title(f"Horizon {horizon}d - {metric_col.upper()} Time Series (Rolling {rolling_window}-period mean)")
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved IC decay plot to {output_path}")
    
    return fig


# ============================================================================
# REGIME-CONDITIONAL PERFORMANCE
# ============================================================================

def format_regime_performance(
    regime_summaries: pd.DataFrame,
    metric_col: str = "rankic"
) -> pd.DataFrame:
    """
    Format regime performance for clean display.
    
    Adds coverage statistics (n_dates, n_names_median) and flags
    for thin slices.
    
    Args:
        regime_summaries: Regime-sliced summaries from evaluate_with_regime_slicing
        metric_col: Column name for metric
        
    Returns:
        Formatted DataFrame with coverage stats
    """
    formatted = regime_summaries.copy()
    
    # Add thin slice flag
    formatted["thin_slice_flag"] = (
        (formatted["n_dates"] < STABILITY_THRESHOLDS.min_dates_per_bucket) |
        (formatted.get("n_names_median", 100) < STABILITY_THRESHOLDS.min_names_per_date)
    )
    
    # Sort for clean display
    formatted = formatted.sort_values(["horizon", "regime_feature", "bucket"])
    
    return formatted


def plot_regime_bars(
    regime_summaries: pd.DataFrame,
    metric_col: str = "rankic_median",
    output_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plot regime-conditional performance as grouped bars.
    
    Args:
        regime_summaries: Regime-sliced summaries
        metric_col: Column name for metric
        output_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    # Get unique regime features and horizons
    regime_features = sorted(regime_summaries["regime_feature"].unique())
    horizons = sorted(regime_summaries["horizon"].unique())
    
    n_features = len(regime_features)
    fig, axes = plt.subplots(n_features, 1, figsize=(10, 4 * n_features))
    if n_features == 1:
        axes = [axes]
    
    for idx, regime_feature in enumerate(regime_features):
        ax = axes[idx]
        
        feature_data = regime_summaries[regime_summaries["regime_feature"] == regime_feature]
        
        # Pivot for grouped bars
        pivot = feature_data.pivot(index="bucket", columns="horizon", values=metric_col)
        
        pivot.plot(kind="bar", ax=ax, width=0.8)
        
        # Add zero line
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        
        ax.set_ylabel(metric_col.replace("_", " ").title())
        ax.set_title(f"Regime: {regime_feature}")
        ax.set_xlabel("")
        ax.legend(title="Horizon (days)", bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Rotate x labels
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved regime bars to {output_path}")
    
    return fig


# ============================================================================
# CHURN DIAGNOSTICS
# ============================================================================

def compute_churn_diagnostics(
    churn_series: pd.DataFrame
) -> pd.DataFrame:
    """
    Compute churn summary statistics per fold/horizon/k.
    
    Args:
        churn_series: Churn time series from compute_churn
        
    Returns:
        DataFrame with churn diagnostics
    """
    results = []
    
    for (fold_id, horizon, k), group in churn_series.groupby(["fold_id", "horizon", "k"]):
        churn_median = group["churn"].median()
        churn_p90 = group["churn"].quantile(0.90)
        churn_mean = group["churn"].mean()
        
        # High churn alarm
        pct_high_churn = (group["churn"] > STABILITY_THRESHOLDS.high_churn_threshold).mean()
        high_churn_flag = pct_high_churn > 0.25  # If >25% of dates have high churn
        
        results.append({
            "fold_id": fold_id,
            "horizon": horizon,
            "k": k,
            "n_dates": len(group),
            "churn_median": churn_median,
            "churn_mean": churn_mean,
            "churn_p90": churn_p90,
            "pct_high_churn": pct_high_churn,
            "high_churn_flag": high_churn_flag
        })
    
    return pd.DataFrame(results)


def plot_churn_timeseries(
    churn_series: pd.DataFrame,
    k: int = 10,
    output_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plot churn time series per fold/horizon.
    
    Args:
        churn_series: Churn time series
        k: Top-K size to plot
        output_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    churn_k = churn_series[churn_series["k"] == k].copy()
    
    horizons = sorted(churn_k["horizon"].unique())
    n_horizons = len(horizons)
    
    fig, axes = plt.subplots(n_horizons, 1, figsize=(12, 4 * n_horizons))
    if n_horizons == 1:
        axes = [axes]
    
    for idx, horizon in enumerate(horizons):
        ax = axes[idx]
        
        horizon_data = churn_k[churn_k["horizon"] == horizon]
        
        for fold_id in sorted(horizon_data["fold_id"].unique()):
            fold_data = horizon_data[horizon_data["fold_id"] == fold_id].sort_values("date")
            
            ax.plot(fold_data["date"], fold_data["churn"], 
                   linewidth=1.5, marker='o', markersize=3, label=f"{fold_id}")
        
        # Add threshold line
        ax.axhline(y=STABILITY_THRESHOLDS.high_churn_threshold, 
                  color='red', linestyle='--', linewidth=1, 
                  label=f"High Churn Threshold ({STABILITY_THRESHOLDS.high_churn_threshold:.0%})")
        
        ax.set_ylabel("Churn (fraction)")
        ax.set_title(f"Horizon {horizon}d - Top-{k} Churn Over Time")
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved churn timeseries to {output_path}")
    
    return fig


def plot_churn_distribution(
    churn_series: pd.DataFrame,
    k: int = 10,
    output_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plot churn distribution (histogram).
    
    Args:
        churn_series: Churn time series
        k: Top-K size to plot
        output_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    churn_k = churn_series[churn_series["k"] == k].copy()
    
    horizons = sorted(churn_k["horizon"].unique())
    n_horizons = len(horizons)
    
    fig, axes = plt.subplots(1, n_horizons, figsize=(6 * n_horizons, 5))
    if n_horizons == 1:
        axes = [axes]
    
    for idx, horizon in enumerate(horizons):
        ax = axes[idx]
        
        horizon_data = churn_k[churn_k["horizon"] == horizon]
        
        ax.hist(horizon_data["churn"], bins=20, alpha=0.7, edgecolor='black')
        
        # Add median line
        median_churn = horizon_data["churn"].median()
        ax.axvline(x=median_churn, color='blue', linestyle='-', linewidth=2, 
                  label=f"Median: {median_churn:.1%}")
        
        # Add threshold line
        ax.axvline(x=STABILITY_THRESHOLDS.high_churn_threshold, 
                  color='red', linestyle='--', linewidth=2, 
                  label=f"High Threshold: {STABILITY_THRESHOLDS.high_churn_threshold:.0%}")
        
        ax.set_xlabel("Churn (fraction)")
        ax.set_ylabel("Count")
        ax.set_title(f"Horizon {horizon}d - Top-{k} Churn Distribution")
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved churn distribution to {output_path}")
    
    return fig


# ============================================================================
# STABILITY SCORECARD
# ============================================================================

def generate_stability_scorecard(
    fold_summaries: pd.DataFrame,
    churn_diagnostics: Optional[pd.DataFrame] = None,
    cost_overlays: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Generate one-screen stability scorecard.
    
    This is the thing you paste into your Chapter 6 writeup.
    
    Args:
        fold_summaries: Fold-level summaries from evaluate_fold
        churn_diagnostics: Churn diagnostics (optional)
        cost_overlays: Cost overlays from 6.4 (optional)
        
    Returns:
        Scorecard DataFrame
    """
    # Start with fold summaries
    scorecard = fold_summaries.copy()
    
    # Add churn if available
    if churn_diagnostics is not None:
        churn_k10 = churn_diagnostics[churn_diagnostics["k"] == 10][
            ["fold_id", "horizon", "churn_median"]
        ].rename(columns={"churn_median": "churn@10_median"})
        
        scorecard = scorecard.merge(churn_k10, on=["fold_id", "horizon"], how="left")
    
    # Add cost overlay if available
    if cost_overlays is not None:
        # Use base scenario for scorecard
        cost_base = cost_overlays[cost_overlays["scenario"] == "base_slippage"][
            ["fold_id", "horizon", "net_avg_er", "alpha_survives"]
        ]
        
        scorecard = scorecard.merge(cost_base, on=["fold_id", "horizon"], how="left")
    
    # Sort for clean display
    scorecard = scorecard.sort_values(["horizon", "fold_id"])
    
    return scorecard


# ============================================================================
# MAIN REPORT GENERATION
# ============================================================================

def generate_stability_report(
    inputs: StabilityReportInputs,
    experiment_name: str,
    output_dir: Path
) -> StabilityReportOutputs:
    """
    Generate complete stability report.
    
    This is the main entry point for 6.5 reporting.
    
    Args:
        inputs: Pre-computed metrics from 6.3 and 6.4
        experiment_name: Experiment identifier
        output_dir: Base output directory
        
    Returns:
        Paths to all generated artifacts
    """
    # Create output directories
    exp_dir = output_dir / experiment_name
    tables_dir = exp_dir / "tables"
    figures_dir = exp_dir / "figures"
    
    for d in [exp_dir, tables_dir, figures_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Generating stability report for {experiment_name}")
    logger.info(f"Output directory: {exp_dir}")
    
    # ========================================================================
    # IC DECAY ANALYSIS
    # ========================================================================
    
    logger.info("Computing IC decay statistics...")
    ic_decay_stats = compute_ic_decay_stats(inputs.per_date_metrics, metric_col="rankic")
    ic_decay_stats_path = tables_dir / "ic_decay_stats.csv"
    ic_decay_stats.to_csv(ic_decay_stats_path, index=False)
    logger.info(f"Saved IC decay stats to {ic_decay_stats_path}")
    
    logger.info("Plotting IC decay...")
    ic_decay_plot_path = figures_dir / "ic_decay.png"
    plot_ic_decay(inputs.per_date_metrics, metric_col="rankic", output_path=ic_decay_plot_path)
    plt.close()
    
    # ========================================================================
    # REGIME-CONDITIONAL PERFORMANCE
    # ========================================================================
    
    regime_performance_path = None
    regime_bars_path = None
    
    if inputs.regime_summaries is not None:
        logger.info("Formatting regime performance...")
        regime_performance = format_regime_performance(inputs.regime_summaries, metric_col="rankic_median")
        regime_performance_path = tables_dir / "regime_performance.csv"
        regime_performance.to_csv(regime_performance_path, index=False)
        logger.info(f"Saved regime performance to {regime_performance_path}")
        
        logger.info("Plotting regime bars...")
        regime_bars_path = figures_dir / "regime_bars.png"
        plot_regime_bars(inputs.regime_summaries, metric_col="rankic_median", output_path=regime_bars_path)
        plt.close()
    
    # ========================================================================
    # CHURN DIAGNOSTICS
    # ========================================================================
    
    churn_diagnostics_path = None
    churn_timeseries_path = None
    churn_distribution_path = None
    churn_diag = None
    
    if inputs.churn_series is not None:
        logger.info("Computing churn diagnostics...")
        churn_diag = compute_churn_diagnostics(inputs.churn_series)
        churn_diagnostics_path = tables_dir / "churn_diagnostics.csv"
        churn_diag.to_csv(churn_diagnostics_path, index=False)
        logger.info(f"Saved churn diagnostics to {churn_diagnostics_path}")
        
        logger.info("Plotting churn timeseries...")
        churn_timeseries_path = figures_dir / "churn_timeseries.png"
        plot_churn_timeseries(inputs.churn_series, k=10, output_path=churn_timeseries_path)
        plt.close()
        
        logger.info("Plotting churn distribution...")
        churn_distribution_path = figures_dir / "churn_distribution.png"
        plot_churn_distribution(inputs.churn_series, k=10, output_path=churn_distribution_path)
        plt.close()
    
    # ========================================================================
    # STABILITY SCORECARD
    # ========================================================================
    
    logger.info("Generating stability scorecard...")
    scorecard = generate_stability_scorecard(
        inputs.fold_summaries,
        churn_diagnostics=churn_diag,
        cost_overlays=inputs.cost_overlays
    )
    scorecard_path = tables_dir / "stability_scorecard.csv"
    scorecard.to_csv(scorecard_path, index=False)
    logger.info(f"Saved stability scorecard to {scorecard_path}")
    
    # ========================================================================
    # SUMMARY REPORT (MARKDOWN)
    # ========================================================================
    
    logger.info("Generating summary report...")
    summary_path = exp_dir / "REPORT_SUMMARY.md"
    _write_summary_report(
        summary_path,
        experiment_name=experiment_name,
        ic_decay_stats=ic_decay_stats,
        scorecard=scorecard,
        churn_diag=churn_diag
    )
    logger.info(f"Saved summary report to {summary_path}")
    
    # ========================================================================
    # RETURN OUTPUT PATHS
    # ========================================================================
    
    return StabilityReportOutputs(
        output_dir=exp_dir,
        ic_decay_stats=ic_decay_stats_path,
        regime_performance=regime_performance_path or tables_dir / "regime_performance.csv",
        churn_diagnostics=churn_diagnostics_path or tables_dir / "churn_diagnostics.csv",
        stability_scorecard=scorecard_path,
        ic_decay_plot=ic_decay_plot_path,
        regime_bars=regime_bars_path or figures_dir / "regime_bars.png",
        churn_timeseries=churn_timeseries_path or figures_dir / "churn_timeseries.png",
        churn_distribution=churn_distribution_path or figures_dir / "churn_distribution.png",
        summary_report=summary_path
    )


def _write_summary_report(
    output_path: Path,
    experiment_name: str,
    ic_decay_stats: pd.DataFrame,
    scorecard: pd.DataFrame,
    churn_diag: Optional[pd.DataFrame]
) -> None:
    """Write human-readable summary report in Markdown."""
    
    with open(output_path, 'w') as f:
        f.write(f"# Stability Report: {experiment_name}\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("---\n\n")
        
        # IC Decay Summary
        f.write("## IC Decay Summary\n\n")
        f.write("Early vs late performance:\n\n")
        f.write("| Fold | Horizon | Early Median | Late Median | Decay | % Positive | Flags |\n")
        f.write("|------|---------|--------------|-------------|-------|------------|-------|\n")
        
        for _, row in ic_decay_stats.iterrows():
            flags = []
            if row["rapid_decay_flag"]:
                flags.append("⚠️ RAPID DECAY")
            if row["noisy_flag"]:
                flags.append("⚠️ NOISY")
            
            f.write(f"| {row['fold_id']} | {row['horizon']} | "
                   f"{row['early_median']:.3f} | {row['late_median']:.3f} | "
                   f"{row['decay']:+.3f} | {row['pct_positive']:.1%} | "
                   f"{', '.join(flags) if flags else '✓'} |\n")
        
        f.write("\n")
        
        # Stability Scorecard
        f.write("## Stability Scorecard\n\n")
        f.write("One-screen summary:\n\n")
        f.write(scorecard.to_markdown(index=False))
        f.write("\n\n")
        
        # Churn Summary
        if churn_diag is not None:
            f.write("## Churn Summary\n\n")
            f.write("Top-10 churn diagnostics:\n\n")
            churn_k10 = churn_diag[churn_diag["k"] == 10]
            f.write("| Fold | Horizon | Median Churn | P90 Churn | % High Churn | Flags |\n")
            f.write("|------|---------|--------------|-----------|--------------|-------|\n")
            
            for _, row in churn_k10.iterrows():
                flags = "⚠️ HIGH CHURN" if row["high_churn_flag"] else "✓"
                f.write(f"| {row['fold_id']} | {row['horizon']} | "
                       f"{row['churn_median']:.1%} | {row['churn_p90']:.1%} | "
                       f"{row['pct_high_churn']:.1%} | {flags} |\n")
            
            f.write("\n")
        
        # Interpretation Guide
        f.write("---\n\n")
        f.write("## Interpretation Guide\n\n")
        f.write("**IC Decay Flags:**\n")
        f.write(f"- RAPID DECAY: Late IC drops > {STABILITY_THRESHOLDS.rapid_decay_threshold:.1%} vs early\n")
        f.write(f"- NOISY: IQR/median > {STABILITY_THRESHOLDS.noisy_threshold:.1f}\n\n")
        f.write("**Churn Flags:**\n")
        f.write(f"- HIGH CHURN: >25% of dates have churn > {STABILITY_THRESHOLDS.high_churn_threshold:.0%}\n\n")
        f.write("**Coverage:**\n")
        f.write(f"- Minimum dates per regime bucket: {STABILITY_THRESHOLDS.min_dates_per_bucket}\n")
        f.write(f"- Minimum names per date: {STABILITY_THRESHOLDS.min_names_per_date}\n")


# ============================================================================
# VALIDATION HELPERS
# ============================================================================

def validate_report_determinism(
    inputs: StabilityReportInputs,
    experiment_name: str,
    output_dir: Path,
    n_runs: int = 2
) -> bool:
    """
    Validate that reports are deterministic.
    
    Same inputs (even shuffled) → identical outputs.
    
    Args:
        inputs: Report inputs
        experiment_name: Experiment name
        output_dir: Output directory
        n_runs: Number of runs to compare
        
    Returns:
        True if deterministic, False otherwise
    """
    outputs_list = []
    
    for run in range(n_runs):
        # Shuffle input rows (but keep them)
        shuffled_inputs = StabilityReportInputs(
            per_date_metrics=inputs.per_date_metrics.sample(frac=1.0, random_state=run),
            fold_summaries=inputs.fold_summaries.sample(frac=1.0, random_state=run),
            regime_summaries=inputs.regime_summaries.sample(frac=1.0, random_state=run) if inputs.regime_summaries is not None else None,
            churn_series=inputs.churn_series.sample(frac=1.0, random_state=run) if inputs.churn_series is not None else None,
            cost_overlays=inputs.cost_overlays.sample(frac=1.0, random_state=run) if inputs.cost_overlays is not None else None
        )
        
        # Generate report
        exp_name = f"{experiment_name}_determinism_test_{run}"
        outputs = generate_stability_report(shuffled_inputs, exp_name, output_dir)
        outputs_list.append(outputs)
    
    # Compare outputs
    # Read first scorecard
    scorecard_0 = pd.read_csv(outputs_list[0].stability_scorecard).sort_values(["fold_id", "horizon"]).reset_index(drop=True)
    
    # Compare with others
    for i in range(1, n_runs):
        scorecard_i = pd.read_csv(outputs_list[i].stability_scorecard).sort_values(["fold_id", "horizon"]).reset_index(drop=True)
        
        # Check if identical
        if not scorecard_0.equals(scorecard_i):
            logger.error(f"Determinism check failed: run 0 vs run {i}")
            return False
    
    logger.info("Determinism check passed: all runs produce identical results")
    return True


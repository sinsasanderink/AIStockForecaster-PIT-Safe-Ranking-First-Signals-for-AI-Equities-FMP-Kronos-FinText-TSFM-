#!/usr/bin/env python3
"""
Chapter 7.5 Execution Script - Tabular ML Baseline

Executes FULL_MODE reference run for tabular_lgb baseline.

This script:
1. Loads features data (REAL from DuckDB by default)
2. Runs tabular_lgb baseline (monthly + quarterly)
3. Produces cost overlays and stability reports
4. Compares vs frozen Chapter 6 baseline floor
5. Produces Chapter 7 baseline reference summary

Usage:
    # REAL DATA (default if data/features.duckdb exists)
    python scripts/run_chapter7_tabular_lgb.py
    
    # SMOKE TEST (quick test, 1 fold per cadence)
    python scripts/run_chapter7_tabular_lgb.py --smoke

Output directory:
    evaluation_outputs/chapter7_tabular_lgb_real/

Note: This does NOT modify Chapter 6 frozen artifacts.
"""

import argparse
import json
import hashlib
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

import pandas as pd
import numpy as np
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# AUTO-LOAD .env FROM REPO ROOT
from src.utils.env import load_repo_dotenv
load_repo_dotenv()

from src.evaluation import (
    ExperimentSpec,
    run_experiment,
    SMOKE_MODE,
    FULL_MODE,
    EVALUATION_RANGE,
    HORIZONS_TRADING_DAYS,
)

from src.evaluation.data_loader import (
    load_features_for_evaluation,
    check_duckdb_available,
)


def get_git_commit_hash() -> str:
    """Get current git commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()[:8]
    except Exception as e:
        logger.warning(f"Could not get git commit hash: {e}")
        return "unknown"


def compute_data_hash(features_df: pd.DataFrame) -> str:
    """Compute deterministic hash of features dataframe."""
    # Sort by date + ticker for determinism
    df_sorted = features_df.sort_values(["date", "ticker"]).reset_index(drop=True)
    
    # Hash key columns + shapes
    hash_input = {
        "n_rows": len(df_sorted),
        "n_cols": len(df_sorted.columns),
        "columns": sorted(df_sorted.columns.tolist()),
        "min_date": str(df_sorted["date"].min()),
        "max_date": str(df_sorted["date"].max()),
        "n_tickers": df_sorted["ticker"].nunique(),
    }
    
    hash_str = json.dumps(hash_input, sort_keys=True)
    return hashlib.sha256(hash_str.encode()).hexdigest()[:16]


def load_frozen_baseline_floor(floor_path: Path) -> Dict:
    """Load frozen Chapter 6 baseline floor."""
    if not floor_path.exists():
        logger.warning(f"Frozen baseline floor not found: {floor_path}")
        return {}
    
    with floor_path.open() as f:
        return json.load(f)


def compute_baseline_summary(
    output_dir: Path,
    cadence: str
) -> Dict[str, Any]:
    """Compute summary metrics for a baseline run."""
    # run_experiment creates a nested directory structure:
    # output_dir/baseline_tabular_lgb_{cadence}/...
    experiment_dir = output_dir / f"baseline_tabular_lgb_{cadence}"
    
    # Load fold summaries
    fold_summaries_path = experiment_dir / "fold_summaries.csv"
    if not fold_summaries_path.exists():
        logger.warning(f"Fold summaries not found: {fold_summaries_path}")
        return {}
    
    fold_summaries = pd.read_csv(fold_summaries_path)
    
    # Compute per-horizon medians
    summary = {
        "cadence": cadence,
        "n_folds": len(fold_summaries["fold_id"].unique()),
        "horizons": {},
    }
    
    for horizon in HORIZONS_TRADING_DAYS:
        h_data = fold_summaries[fold_summaries["horizon"] == horizon]
        if len(h_data) == 0:
            continue
        
        summary["horizons"][horizon] = {
            "median_rankic": float(h_data["rankic_median"].median()),
            "ic_stability": float(h_data["rankic_iqr"].mean()) if "rankic_iqr" in h_data.columns else 0.0,
        }
    
    # Load cost overlays if available
    cost_path = experiment_dir / "cost_overlays.csv"
    if cost_path.exists():
        cost_df = pd.read_csv(cost_path)
        base_cost = cost_df[cost_df["scenario"] == "base_slippage"]
        
        for horizon in HORIZONS_TRADING_DAYS:
            h_cost = base_cost[base_cost["horizon"] == horizon]
            if len(h_cost) > 0 and horizon in summary["horizons"]:
                n_positive = h_cost["alpha_survives"].sum()
                pct_positive = n_positive / len(h_cost) if len(h_cost) > 0 else 0
                summary["horizons"][horizon]["cost_survival_pct_positive"] = float(pct_positive)
                # Use net_avg_er column (not net_excess_return_median)
                if "net_avg_er" in h_cost.columns:
                    summary["horizons"][horizon]["cost_survival_median_net_er"] = float(h_cost["net_avg_er"].median())
    
    # Load churn if available
    churn_path = experiment_dir / "churn_series.csv"
    if churn_path.exists():
        churn_df = pd.read_csv(churn_path)
        
        for horizon in HORIZONS_TRADING_DAYS:
            h_churn = churn_df[(churn_df["horizon"] == horizon) & (churn_df["k"] == 10)]
            if len(h_churn) > 0 and horizon in summary["horizons"]:
                summary["horizons"][horizon]["churn_top10_median"] = float(h_churn["churn"].median())
    
    return summary


def write_baseline_reference(
    output_dir: Path,
    monthly_summary: Dict,
    quarterly_summary: Dict,
    data_manifest: Dict,
    commit_hash: str,
    frozen_floor: Dict
):
    """Write Chapter 7 baseline reference document."""
    md_path = output_dir / "BASELINE_REFERENCE.md"
    
    with md_path.open("w") as f:
        f.write("# Chapter 7 Tabular ML Baseline Reference\n\n")
        f.write(f"**Generated:** {datetime.utcnow().isoformat()}Z\n")
        f.write(f"**Commit:** {commit_hash}\n")
        f.write(f"**Data Hash:** {data_manifest.get('data_hash', 'unknown')}\n\n")
        
        f.write("---\n\n")
        f.write("## Baseline: `tabular_lgb`\n\n")
        f.write("**Model:** LightGBM Regressor with time-decay weighting\n")
        f.write("**Training:** Per-fold, horizon-specific (20/60/90d)\n")
        f.write("**Features:** Momentum, volatility, drawdown, liquidity, relative strength, beta\n")
        f.write("**Hyperparameters:** Fixed (n_estimators=100, lr=0.05, depth=5, leaves=31)\n\n")
        
        f.write("---\n\n")
        f.write("## Performance Summary\n\n")
        
        # Monthly cadence
        f.write("### Monthly Cadence (Primary)\n\n")
        f.write(f"**Folds:** {monthly_summary.get('n_folds', 0)}\n\n")
        f.write("| Horizon | Median RankIC | IC Stability | Churn (Top-10) | Cost Survival (% Positive) |\n")
        f.write("|---------|---------------|--------------|----------------|---------------------------|\n")
        
        for horizon in HORIZONS_TRADING_DAYS:
            h_data = monthly_summary.get("horizons", {}).get(horizon, {})
            rankic = h_data.get("median_rankic", 0.0)
            stability = h_data.get("ic_stability", 0.0)
            churn = h_data.get("churn_top10_median", 0.0)
            cost_pct = h_data.get("cost_survival_pct_positive", 0.0)
            f.write(f"| {horizon}d | {rankic:.4f} | {stability:.4f} | {churn:.2f} | {cost_pct:.1%} |\n")
        
        f.write("\n")
        
        # Quarterly cadence
        f.write("### Quarterly Cadence (Robustness Check)\n\n")
        f.write(f"**Folds:** {quarterly_summary.get('n_folds', 0)}\n\n")
        f.write("| Horizon | Median RankIC | IC Stability | Churn (Top-10) | Cost Survival (% Positive) |\n")
        f.write("|---------|---------------|--------------|----------------|---------------------------|\n")
        
        for horizon in HORIZONS_TRADING_DAYS:
            h_data = quarterly_summary.get("horizons", {}).get(horizon, {})
            rankic = h_data.get("median_rankic", 0.0)
            stability = h_data.get("ic_stability", 0.0)
            churn = h_data.get("churn_top10_median", 0.0)
            cost_pct = h_data.get("cost_survival_pct_positive", 0.0)
            f.write(f"| {horizon}d | {rankic:.4f} | {stability:.4f} | {churn:.2f} | {cost_pct:.1%} |\n")
        
        f.write("\n")
        
        # Comparison vs frozen baseline floor
        if frozen_floor and "best_baseline_per_horizon" in frozen_floor:
            f.write("---\n\n")
            f.write("## Comparison vs Chapter 6 Frozen Floor\n\n")
            f.write("| Horizon | Frozen Floor (Factor) | tabular_lgb (ML) | Lift |\n")
            f.write("|---------|----------------------|------------------|------|\n")
            
            for horizon in HORIZONS_TRADING_DAYS:
                frozen_h = frozen_floor["best_baseline_per_horizon"].get(str(horizon), {})
                frozen_rankic = frozen_h.get("median_rankic", 0.0)
                
                ml_rankic = monthly_summary.get("horizons", {}).get(horizon, {}).get("median_rankic", 0.0)
                lift = ml_rankic - frozen_rankic
                
                f.write(f"| {horizon}d | {frozen_rankic:.4f} | {ml_rankic:.4f} | {lift:+.4f} |\n")
            
            f.write("\n")
            f.write("**Frozen Floor Baseline:** ")
            frozen_20d = frozen_floor["best_baseline_per_horizon"].get("20", {}).get("baseline", "")
            frozen_60d = frozen_floor["best_baseline_per_horizon"].get("60", {}).get("baseline", "")
            frozen_90d = frozen_floor["best_baseline_per_horizon"].get("90", {}).get("baseline", "")
            f.write(f"20d={frozen_20d}, 60d={frozen_60d}, 90d={frozen_90d}\n\n")
        
        f.write("---\n\n")
        f.write("## Data Snapshot\n\n")
        f.write(f"**Rows:** {data_manifest.get('n_rows', 0):,}\n")
        f.write(f"**Tickers:** {data_manifest.get('n_tickers', 0)}\n")
        f.write(f"**Date Range:** {data_manifest.get('date_range', {}).get('start', 'N/A')} → {data_manifest.get('date_range', {}).get('end', 'N/A')}\n")
        f.write(f"**Horizons:** {', '.join(map(str, data_manifest.get('horizons', [])))}d\n\n")
        
        f.write("---\n\n")
        f.write("## Output Artifacts\n\n")
        f.write("- `monthly/eval_rows.parquet`: Per-row scored observations\n")
        f.write("- `monthly/fold_summaries.csv`: Per-fold aggregate metrics\n")
        f.write("- `monthly/cost_overlays.csv`: Cost sensitivity analysis\n")
        f.write("- `monthly/churn_series.csv`: Portfolio turnover metrics\n")
        f.write("- `monthly/stability_report/`: IC decay, regime tables, diagnostics\n")
        f.write("- `quarterly/`: Same structure for quarterly cadence\n")
        f.write("- `BASELINE_REFERENCE.md`: This file\n")
        f.write("- `CLOSURE_MANIFEST.json`: Reproducibility metadata\n\n")
        
        f.write("---\n\n")
        f.write("**Note:** Chapter 6 frozen artifacts remain unchanged at `evaluation_outputs/chapter6_closure_real/`\n")
    
    logger.info(f"✓ Wrote baseline reference: {md_path}")


def main():
    parser = argparse.ArgumentParser(description="Chapter 7.5: Run tabular_lgb baseline")
    parser.add_argument("--smoke", action="store_true", help="Run in SMOKE_MODE (1 fold per cadence)")
    args = parser.parse_args()
    
    # Determine mode
    mode = SMOKE_MODE if args.smoke else FULL_MODE
    mode_name = "SMOKE" if args.smoke else "FULL"
    
    logger.info(f"="*60)
    logger.info(f"Chapter 7.5: Tabular ML Baseline Execution ({mode_name} MODE)")
    logger.info(f"="*60)
    
    # Check for DuckDB
    db_path = Path("data/features.duckdb")
    if not db_path.exists():
        logger.error(f"DuckDB feature store not found: {db_path}")
        logger.error("Run: python scripts/build_features_duckdb.py --auto-normalize-splits")
        sys.exit(1)
    
    # Load features from DuckDB
    logger.info(f"\n{'='*60}")
    logger.info("Step 1: Load Features from DuckDB")
    logger.info(f"{'='*60}\n")
    
    # Note: function returns (dataframe, metadata) tuple - unpack it
    features_df, data_metadata = load_features_for_evaluation(
        source="duckdb",  # Parameter name is "source", not "mode"
        db_path=db_path,  # Can be Path, not just string
        eval_start=EVALUATION_RANGE.eval_start,
        eval_end=EVALUATION_RANGE.eval_end,
        horizons=HORIZONS_TRADING_DAYS,
        require_all_horizons=True
    )
    
    logger.info(f"✓ Loaded {len(features_df):,} rows")
    logger.info(f"  Date range: {features_df['date'].min()} → {features_df['date'].max()}")
    logger.info(f"  Tickers: {features_df['ticker'].nunique()}")
    
    # Use metadata from load function and compute hash
    data_hash = compute_data_hash(features_df)
    data_manifest = {
        "n_rows": len(features_df),
        "n_tickers": features_df["ticker"].nunique(),
        "date_range": {
            "start": str(features_df["date"].min()),
            "end": str(features_df["date"].max()),
        },
        "horizons": list(HORIZONS_TRADING_DAYS),
        "source": data_metadata.get("source", "duckdb"),
        "data_hash": data_hash,
    }
    
    # Get git commit
    commit_hash = get_git_commit_hash()
    
    # Create output directory
    output_base = Path("evaluation_outputs/chapter7_tabular_lgb_real")
    output_base.mkdir(parents=True, exist_ok=True)
    
    # Run tabular_lgb for both cadences
    results = {}
    
    for cadence in ["monthly", "quarterly"]:
        logger.info(f"\n{'='*60}")
        logger.info(f"Step 2.{['1', '2'][cadence == 'quarterly']}: Run tabular_lgb Baseline ({cadence.upper()})")
        logger.info(f"{'='*60}\n")
        
        spec = ExperimentSpec.baseline("tabular_lgb", cadence=cadence)
        output_dir = output_base / cadence
        output_dir.mkdir(parents=True, exist_ok=True)
        
        experiment_results = run_experiment(
            experiment_spec=spec,
            features_df=features_df,
            output_dir=output_dir,
            mode=mode
        )
        
        results[cadence] = experiment_results
        logger.info(f"✓ Completed {cadence} cadence: {len(experiment_results)} artifacts")
    
    # Compute summaries
    logger.info(f"\n{'='*60}")
    logger.info("Step 3: Compute Baseline Summaries")
    logger.info(f"{'='*60}\n")
    
    monthly_summary = compute_baseline_summary(output_base / "monthly", "monthly")
    quarterly_summary = compute_baseline_summary(output_base / "quarterly", "quarterly")
    
    logger.info(f"✓ Monthly: {monthly_summary.get('n_folds', 0)} folds")
    logger.info(f"✓ Quarterly: {quarterly_summary.get('n_folds', 0)} folds")
    
    # Load frozen baseline floor
    logger.info(f"\n{'='*60}")
    logger.info("Step 4: Compare vs Frozen Baseline Floor")
    logger.info(f"{'='*60}\n")
    
    frozen_floor_path = Path("evaluation_outputs/chapter6_closure_real/BASELINE_FLOOR.json")
    frozen_floor = load_frozen_baseline_floor(frozen_floor_path)
    
    if frozen_floor and "best_baseline_per_horizon" in frozen_floor:
        logger.info("Comparison vs Chapter 6 Frozen Floor:")
        for horizon in HORIZONS_TRADING_DAYS:
            frozen_h = frozen_floor["best_baseline_per_horizon"].get(str(horizon), {})
            frozen_rankic = frozen_h.get("median_rankic", 0.0)
            frozen_baseline = frozen_h.get("baseline", "N/A")
            
            ml_rankic = monthly_summary.get("horizons", {}).get(horizon, {}).get("median_rankic", 0.0)
            lift = ml_rankic - frozen_rankic
            
            logger.info(f"  {horizon}d: {frozen_baseline} {frozen_rankic:.4f} → tabular_lgb {ml_rankic:.4f} (lift: {lift:+.4f})")
    else:
        logger.warning("Frozen baseline floor not found or incomplete")
    
    # Write baseline reference
    logger.info(f"\n{'='*60}")
    logger.info("Step 5: Write Baseline Reference")
    logger.info(f"{'='*60}\n")
    
    write_baseline_reference(
        output_dir=output_base,
        monthly_summary=monthly_summary,
        quarterly_summary=quarterly_summary,
        data_manifest=data_manifest,
        commit_hash=commit_hash,
        frozen_floor=frozen_floor
    )
    
    # Write closure manifest
    closure_manifest = {
        "chapter": "7.5",
        "baseline": "tabular_lgb",
        "mode": mode_name,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "commit_hash": commit_hash,
        "data_hash": data_hash,
        "data_manifest": data_manifest,
        "monthly_summary": monthly_summary,
        "quarterly_summary": quarterly_summary,
    }
    
    manifest_path = output_base / "CLOSURE_MANIFEST.json"
    with manifest_path.open("w") as f:
        json.dump(closure_manifest, f, indent=2)
    
    logger.info(f"✓ Wrote closure manifest: {manifest_path}")
    
    # Write data manifest
    data_manifest_path = output_base / "DATA_MANIFEST.json"
    with data_manifest_path.open("w") as f:
        json.dump(data_manifest, f, indent=2)
    
    logger.info(f"✓ Wrote data manifest: {data_manifest_path}")
    
    # Final summary
    logger.info(f"\n{'='*60}")
    logger.info("✓ Chapter 7.5 Execution Complete")
    logger.info(f"{'='*60}\n")
    logger.info(f"Output directory: {output_base}")
    logger.info(f"Baseline reference: {output_base / 'BASELINE_REFERENCE.md'}")
    logger.info(f"Closure manifest: {manifest_path}")
    logger.info(f"\nCommit hash: {commit_hash}")
    logger.info(f"Data hash: {data_hash}")
    logger.info(f"\nChapter 6 frozen artifacts remain unchanged.")


if __name__ == "__main__":
    main()


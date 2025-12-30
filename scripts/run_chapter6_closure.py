#!/usr/bin/env python3
"""
Chapter 6 Closure Script

Executes FULL_MODE baseline reference run and freezes artifacts.

This script:
1. Loads features data (REAL from DuckDB by default, synthetic only with explicit flag)
2. Runs all baselines (factor + sanity)
3. Produces cost overlays and stability reports
4. Runs Qlib shadow evaluation
5. Freezes reference artifacts (commit hash, manifest, etc.)
6. Produces baseline floor summary

Usage:
    # REAL DATA (default if data/features.duckdb exists)
    # .env is auto-loaded from repo root
    python scripts/run_chapter6_closure.py
    
    # SYNTHETIC DATA (explicit flag required, outputs to chapter6_closure_synth/)
    python scripts/run_chapter6_closure.py --mode synthetic
    
    # SMOKE TEST (quick test with either mode)
    python scripts/run_chapter6_closure.py --smoke
    python scripts/run_chapter6_closure.py --mode synthetic --smoke

Output directories:
    - Real data:      evaluation_outputs/chapter6_closure_real/
    - Synthetic data: evaluation_outputs/chapter6_closure_synth/
"""

import argparse
import json
import hashlib
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd
import numpy as np
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# AUTO-LOAD .env FROM REPO ROOT (must happen before any imports that need env vars)
from src.utils.env import load_repo_dotenv
load_repo_dotenv()

from src.evaluation import (
    # Baselines
    BASELINE_REGISTRY,
    FACTOR_BASELINES,
    SANITY_BASELINES,
    generate_baseline_scores,
    run_all_baselines,
    # Run evaluation
    ExperimentSpec,
    run_experiment,
    SMOKE_MODE,
    FULL_MODE,
    COST_SCENARIOS,
    compute_acceptance_verdict,
    save_acceptance_summary,
    # Metrics
    compute_rankic_per_date,
    compute_quintile_spread_per_date,
    compute_topk_metrics_per_date,
    compute_churn,
    # Costs
    compute_net_metrics,
    run_cost_sensitivity,
    # Reports
    generate_stability_report,
    generate_stability_scorecard,
    StabilityReportInputs,
    # Qlib
    is_qlib_available,
    to_qlib_format,
    validate_qlib_frame,
    run_qlib_shadow_evaluation,
    check_ic_parity,
    # Definitions
    EVALUATION_RANGE,
    HORIZONS_TRADING_DAYS,
)

from src.evaluation.data_loader import (
    load_features_for_evaluation,
    load_features_from_duckdb,
    generate_data_manifest,
    validate_features_for_evaluation,
    check_duckdb_available,
    SYNTHETIC_CONFIG,
    SMOKE_CONFIG,
    DataLoaderConfig,
    DEFAULT_DUCKDB_PATH,
)


# ============================================================================
# CONFIGURATION
# ============================================================================

OUTPUT_BASE = Path("evaluation_outputs")
CLOSURE_DIR_REAL = OUTPUT_BASE / "chapter6_closure_real"
CLOSURE_DIR_SYNTH = OUTPUT_BASE / "chapter6_closure_synth"

# For backward compatibility
CLOSURE_DIR = CLOSURE_DIR_REAL


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_git_commit_hash() -> str:
    """Get current git commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def get_git_status() -> Dict[str, Any]:
    """Get git status information."""
    try:
        # Check for uncommitted changes
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True
        )
        has_changes = len(result.stdout.strip()) > 0
        
        return {
            "commit_hash": get_git_commit_hash(),
            "has_uncommitted_changes": has_changes,
            "branch": subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                capture_output=True,
                text=True,
                check=True
            ).stdout.strip()
        }
    except Exception as e:
        return {"error": str(e)}


def get_environment_snapshot() -> Dict[str, Any]:
    """Get environment information."""
    import platform
    
    try:
        pip_freeze = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            capture_output=True,
            text=True,
            check=True
        ).stdout.strip().split("\n")
    except Exception:
        pip_freeze = []
    
    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "pip_packages": pip_freeze[:50],  # Limit for readability
    }


# ============================================================================
# MAIN CLOSURE RUNNER
# ============================================================================

def run_baseline_closure(
    features_df: pd.DataFrame,
    output_dir: Path,
    mode: str = "full",
    cadences: List[str] = None,
) -> Dict[str, Any]:
    """
    Run full baseline closure evaluation.
    
    Args:
        features_df: Features DataFrame
        output_dir: Output directory
        mode: "full" or "smoke"
        cadences: List of cadences to run ["monthly", "quarterly"]
    
    Returns:
        Results dict with all outputs
    """
    if cadences is None:
        cadences = ["monthly"] if mode == "smoke" else ["monthly", "quarterly"]
    
    eval_mode = SMOKE_MODE if mode == "smoke" else FULL_MODE
    
    results = {
        "mode": mode,
        "cadences": cadences,
        "baselines": {},
        "summaries": {},
        "sanity_checks": {},
        "all_eval_rows": [],  # Collect eval_rows for Qlib shadow
    }
    
    print(f"\n{'='*60}")
    print(f"CHAPTER 6 CLOSURE: {mode.upper()} MODE")
    print(f"{'='*60}")
    print(f"Output directory: {output_dir}")
    print(f"Cadences: {cadences}")
    print(f"Date range: {features_df['date'].min()} to {features_df['date'].max()}")
    print(f"Stocks: {features_df['ticker'].nunique()}")
    print(f"Rows: {len(features_df):,}")
    print()
    
    # Run each baseline
    all_baselines = FACTOR_BASELINES + SANITY_BASELINES
    
    for cadence in cadences:
        print(f"\n--- Running {cadence} cadence ---")
        
        for baseline_name in all_baselines:
            print(f"\n  Running baseline: {baseline_name}")
            
            try:
                spec = ExperimentSpec.baseline(baseline_name, cadence=cadence)
                
                baseline_output = output_dir / f"baseline_{baseline_name}_{cadence}"
                baseline_output.mkdir(parents=True, exist_ok=True)
                
                result = run_experiment(
                    experiment_spec=spec,
                    features_df=features_df,
                    output_dir=baseline_output,
                    mode=eval_mode,
                )
                
                key = f"{baseline_name}_{cadence}"
                results["baselines"][key] = {
                    "output_dir": str(baseline_output),
                    "n_folds": result.get("n_folds", 0),
                    "n_eval_rows": result.get("n_eval_rows", 0),
                    "metrics_computed": True,
                    "costs_computed": True,
                    "reports_generated": True,
                }
                
                # Store summary metrics
                if "fold_summaries" in result:
                    results["summaries"][key] = result["fold_summaries"]
                
                # Collect eval_rows for Qlib shadow evaluation
                if "eval_rows_df" in result and len(result["eval_rows_df"]) > 0:
                    eval_rows_df = result["eval_rows_df"].copy()
                    eval_rows_df["baseline_name"] = baseline_name
                    eval_rows_df["cadence"] = cadence
                    results["all_eval_rows"].append(eval_rows_df)
                
                print(f"    ✓ {result.get('n_folds', 0)} folds, {result.get('n_eval_rows', 0)} eval rows")
                
            except Exception as e:
                print(f"    ✗ Failed: {e}")
                results["baselines"][f"{baseline_name}_{cadence}"] = {
                    "error": str(e)
                }
    
    # Sanity check: naive_random should have ~0 RankIC
    print("\n--- Sanity Checks ---")
    for cadence in cadences:
        key = f"naive_random_{cadence}"
        if key in results["summaries"]:
            summary = results["summaries"][key]
            if isinstance(summary, pd.DataFrame) and "rankic_median" in summary.columns:
                median_ic = summary["rankic_median"].median()
                results["sanity_checks"][key] = {
                    "median_rankic": float(median_ic),
                    "passed": abs(median_ic) < 0.05,
                }
                status = "✓ PASSED" if abs(median_ic) < 0.05 else "✗ FAILED"
                print(f"  {key}: median RankIC = {median_ic:.4f} {status}")
    
    return results


def compute_baseline_floor(
    results: Dict[str, Any],
    output_dir: Path,
) -> Dict[str, Any]:
    """
    Compute and save baseline floor summary.
    
    This is the reference point for Chapter 7+ model comparisons.
    
    CRITICAL: Uses MONTHLY as primary cadence, quarterly as robustness only.
    Each horizon (20/60/90) should have independently computed metrics.
    """
    floor = {
        "computed_at": datetime.utcnow().isoformat() + "Z",
        "horizons": {},
        "best_baseline_per_horizon": {},
        "churn_medians": {},
        "cost_survival": {},
        "sanity_passed": True,
        "primary_cadence": "monthly",
    }
    
    print("\n--- Computing Baseline Floor ---")
    
    # =========================================================================
    # STEP 1: Collect per-horizon metrics from all baselines
    # =========================================================================
    
    for horizon in HORIZONS_TRADING_DAYS:
        floor["horizons"][horizon] = {}
        
        for key, summary in results.get("summaries", {}).items():
            if "naive_random" in key:
                continue  # Skip sanity baseline
            
            if isinstance(summary, pd.DataFrame) and "horizon" in summary.columns:
                horizon_data = summary[summary["horizon"] == horizon]
                if not horizon_data.empty and "rankic_median" in horizon_data.columns:
                    # Compute median of fold-level medians for this horizon
                    median_ic = float(horizon_data["rankic_median"].median())
                    
                    # Also get other metrics if available
                    quintile_spread = (
                        float(horizon_data["quintile_spread_median"].median())
                        if "quintile_spread_median" in horizon_data.columns else None
                    )
                    hit_rate = (
                        float(horizon_data["hit_rate_10_median"].median())
                        if "hit_rate_10_median" in horizon_data.columns else None
                    )
                    avg_er = (
                        float(horizon_data["avg_er_10_median"].median())
                        if "avg_er_10_median" in horizon_data.columns else None
                    )
                    
                    floor["horizons"][horizon][key] = {
                        "median_rankic": median_ic,
                        "quintile_spread": quintile_spread,
                        "hit_rate_10": hit_rate,
                        "avg_er_10": avg_er,
                        "n_folds": len(horizon_data),
                    }
    
    # =========================================================================
    # STEP 2: Find best baseline per horizon (MONTHLY PRIMARY)
    # =========================================================================
    
    for horizon in HORIZONS_TRADING_DAYS:
        best_baseline = None
        best_ic = -999
        best_baseline_quarterly = None
        best_ic_quarterly = -999
        
        horizon_baselines = floor["horizons"].get(horizon, {})
        
        for key, metrics in horizon_baselines.items():
            median_ic = metrics.get("median_rankic", -999)
            
            if "monthly" in key:
                # Primary: monthly cadence
                if median_ic > best_ic:
                    best_ic = median_ic
                    best_baseline = key
            elif "quarterly" in key:
                # Secondary: quarterly cadence (robustness)
                if median_ic > best_ic_quarterly:
                    best_ic_quarterly = median_ic
                    best_baseline_quarterly = key
        
        floor["best_baseline_per_horizon"][horizon] = {
            "baseline": best_baseline,
            "median_rankic": float(best_ic) if best_ic > -999 else None,
            "quarterly_baseline": best_baseline_quarterly,
            "quarterly_rankic": float(best_ic_quarterly) if best_ic_quarterly > -999 else None,
        }
        
        print(f"  Horizon {horizon}d: best MONTHLY = {best_baseline} (RankIC = {best_ic:.4f})")
        if best_baseline_quarterly:
            print(f"            quarterly = {best_baseline_quarterly} (RankIC = {best_ic_quarterly:.4f})")
    
    # =========================================================================
    # STEP 3: Compute churn medians per horizon (from baseline outputs)
    # =========================================================================
    
    for horizon in HORIZONS_TRADING_DAYS:
        churn_values = []
        
        # Look for churn data in baseline output directories
        for key, baseline_info in results.get("baselines", {}).items():
            if "naive_random" in key:
                continue
            
            output_dir_baseline = baseline_info.get("output_dir", "")
            if output_dir_baseline:
                # The experiment name is baseline_<name>_<cadence>, stored nested
                # key = "<name>_<cadence>" but experiment_name = "baseline_<name>_<cadence>"
                experiment_name = f"baseline_{key}"
                
                # Try multiple path patterns (run_experiment nests by experiment name)
                possible_paths = [
                    Path(output_dir_baseline) / experiment_name / "churn_series.csv",  # nested with experiment_name
                    Path(output_dir_baseline) / key / "churn_series.csv",              # nested with key
                    Path(output_dir_baseline) / "churn_series.csv",                    # flat
                ]
                
                for churn_path in possible_paths:
                    if churn_path.exists():
                        try:
                            churn_df = pd.read_csv(churn_path)
                            if "horizon" in churn_df.columns and "churn" in churn_df.columns:
                                horizon_churn = churn_df[
                                    (churn_df["horizon"] == horizon) & 
                                    (churn_df["k"] == 10)
                                ]["churn"]
                                if len(horizon_churn) > 0:
                                    churn_values.extend(horizon_churn.dropna().tolist())
                                    break  # Found data, don't check other paths
                        except Exception as e:
                            logger.debug(f"Could not read churn from {churn_path}: {e}")
        
        if churn_values:
            floor["churn_medians"][horizon] = {
                "median": float(np.median(churn_values)),
                "p90": float(np.percentile(churn_values, 90)),
                "n_observations": len(churn_values),
            }
        else:
            floor["churn_medians"][horizon] = {
                "median": None,
                "p90": None,
                "n_observations": 0,
            }
    
    # =========================================================================
    # STEP 4: Compute cost survival per horizon
    # =========================================================================
    
    for horizon in HORIZONS_TRADING_DAYS:
        cost_positive_folds = 0
        cost_total_folds = 0
        net_ers = []
        
        # Look for cost overlay data in baseline output directories
        for key, baseline_info in results.get("baselines", {}).items():
            if "naive_random" in key or "quarterly" in key:  # Monthly primary
                continue
            
            output_dir_baseline = baseline_info.get("output_dir", "")
            if output_dir_baseline:
                # The experiment name is baseline_<name>_<cadence>, stored nested
                # key = "<name>_<cadence>" but experiment_name = "baseline_<name>_<cadence>"
                experiment_name = f"baseline_{key}"
                
                # Try multiple path patterns (run_experiment nests by experiment name)
                possible_paths = [
                    Path(output_dir_baseline) / experiment_name / "cost_overlays.csv",  # nested with experiment_name
                    Path(output_dir_baseline) / key / "cost_overlays.csv",              # nested with key
                    Path(output_dir_baseline) / "cost_overlays.csv",                    # flat
                ]
                
                for cost_path in possible_paths:
                    if cost_path.exists():
                        try:
                            cost_df = pd.read_csv(cost_path)
                            if "horizon" in cost_df.columns:
                                # Filter to this horizon and base_slippage scenario
                                horizon_cost = cost_df[
                                    (cost_df["horizon"] == horizon) &
                                    (cost_df["scenario"] == "base_slippage")
                                ]
                                if len(horizon_cost) > 0:
                                    cost_total_folds += len(horizon_cost)
                                    cost_positive_folds += horizon_cost["alpha_survives"].sum()
                                    net_ers.extend(horizon_cost["net_avg_er"].dropna().tolist())
                                    break  # Found data, don't check other paths
                        except Exception as e:
                            logger.debug(f"Could not read costs from {cost_path}: {e}")
        
        pct_positive = cost_positive_folds / cost_total_folds if cost_total_folds > 0 else None
        median_net = float(np.median(net_ers)) if net_ers else None
        
        floor["cost_survival"][horizon] = {
            "pct_positive_folds": pct_positive,
            "median_net_er": median_net,
            "n_folds_evaluated": cost_total_folds,
        }
    
    # =========================================================================
    # STEP 5: Check sanity baseline
    # =========================================================================
    
    for key, check in results.get("sanity_checks", {}).items():
        if not check.get("passed", True):
            floor["sanity_passed"] = False
            print(f"  ⚠️  Sanity check failed: {key}")
    
    # Print summary
    print("\n  --- Churn Medians ---")
    for horizon, churn_info in floor["churn_medians"].items():
        median_churn = churn_info.get("median")
        if median_churn is not None:
            print(f"    Horizon {horizon}d: median churn = {median_churn:.3f}")
        else:
            print(f"    Horizon {horizon}d: no churn data")
    
    print("\n  --- Cost Survival (base slippage, monthly) ---")
    for horizon, cost_info in floor["cost_survival"].items():
        pct = cost_info.get("pct_positive_folds")
        if pct is not None:
            print(f"    Horizon {horizon}d: {pct*100:.1f}% positive folds")
        else:
            print(f"    Horizon {horizon}d: no cost data")
    
    # Save floor summary
    floor_path = output_dir / "BASELINE_FLOOR.json"
    with open(floor_path, "w") as f:
        json.dump(floor, f, indent=2, default=str)
    
    print(f"\n  Saved baseline floor to: {floor_path}")
    
    return floor


def run_qlib_shadow_closure(
    results: Dict[str, Any],
    features_df: pd.DataFrame,
    output_dir: Path,
) -> Dict[str, Any]:
    """
    Run Qlib shadow evaluation for closure verification.
    
    Uses the actual eval_rows from baseline scoring (which have as_of_date, score, excess_return)
    NOT the raw features_df.
    """
    qlib_results = {
        "qlib_available": is_qlib_available(),
        "parity_checks": {},
    }
    
    print("\n--- Qlib Shadow Evaluation ---")
    
    if not is_qlib_available():
        print("  Qlib not installed - skipping shadow evaluation")
        print("  (This is OK - parity was verified in unit tests)")
        qlib_results["status"] = "skipped_not_installed"
        return qlib_results
    
    # Collect eval_rows from baseline results
    all_eval_rows = results.get("all_eval_rows", [])
    
    if not all_eval_rows:
        print("  ⚠ No eval rows collected from baselines - skipping Qlib shadow")
        qlib_results["status"] = "no_eval_rows"
        return qlib_results
    
    # Concatenate all eval rows
    eval_rows_df = pd.concat(all_eval_rows, ignore_index=True)
    print(f"  Collected {len(eval_rows_df)} eval rows from {len(all_eval_rows)} baseline runs")
    
    # DEBUG: Log columns present
    print(f"  DEBUG: eval_rows columns = {list(eval_rows_df.columns)}")
    if len(eval_rows_df) > 0:
        print(f"  DEBUG: sample row:\n{eval_rows_df.head(2).to_string()}")
    
    # =========================================================================
    # IMPORTANT: For Qlib shadow evaluation, use ONE baseline + ONE horizon
    # to avoid duplicate (datetime, instrument) entries in the MultiIndex.
    # We pick the first factor baseline (e.g., mom_12m) at horizon=20.
    # =========================================================================
    
    # Filter to a single baseline and horizon for clean Qlib evaluation
    if "baseline_name" in eval_rows_df.columns and "horizon" in eval_rows_df.columns:
        # Pick the first factor baseline (not naive_random)
        factor_baselines = [b for b in eval_rows_df["baseline_name"].unique() 
                           if "random" not in b.lower()]
        if factor_baselines:
            selected_baseline = factor_baselines[0]
        else:
            selected_baseline = eval_rows_df["baseline_name"].iloc[0]
        
        selected_horizon = 20  # Primary horizon
        
        sample_df = eval_rows_df[
            (eval_rows_df["baseline_name"] == selected_baseline) &
            (eval_rows_df["horizon"] == selected_horizon)
        ].copy()
        
        print(f"  Selected {selected_baseline} at horizon={selected_horizon} for Qlib: {len(sample_df)} rows")
    else:
        # Fallback: take a sample (might have duplicates)
        sample_df = eval_rows_df.head(5000).copy() if len(eval_rows_df) > 5000 else eval_rows_df.copy()
    
    # Final safety: drop duplicates on (as_of_date, stable_id) keeping last
    if len(sample_df) > 0:
        before = len(sample_df)
        sample_df = sample_df.drop_duplicates(
            subset=["as_of_date", "stable_id"], keep="last"
        ).reset_index(drop=True)
        if len(sample_df) < before:
            print(f"  Deduped: {before} -> {len(sample_df)} rows")
    
    # Run parity check on the sample
    try:
        # Convert to Qlib format (handles column aliases like date->as_of_date)
        qlib_df = to_qlib_format(sample_df)
        
        # Validate format
        validation = validate_qlib_frame(qlib_df)
        qlib_results["format_valid"] = validation[0] if isinstance(validation, tuple) else validation.get("valid", False)
        
        validation_msg = validation[1] if isinstance(validation, tuple) else validation.get("message", "")
        
        if qlib_results["format_valid"]:
            # Run shadow evaluation
            shadow_output = output_dir / "qlib_shadow"
            shadow_output.mkdir(parents=True, exist_ok=True)
            
            shadow_results = run_qlib_shadow_evaluation(
                eval_rows_df=sample_df,
                output_dir=shadow_output,
                experiment_name="closure_verification"
            )
            
            qlib_results["shadow_outputs"] = str(shadow_output)
            qlib_results["status"] = "completed"
            qlib_results["n_eval_rows_used"] = len(sample_df)
            print(f"  ✓ Qlib shadow evaluation completed")
            print(f"    Outputs: {shadow_output}")
        else:
            qlib_results["status"] = "validation_failed"
            qlib_results["issues"] = validation_msg
            print(f"  ✗ Qlib format validation failed: {validation_msg}")
    
    except Exception as e:
        import traceback
        qlib_results["status"] = "error"
        qlib_results["error"] = str(e)
        print(f"  ✗ Qlib shadow evaluation failed: {e}")
        traceback.print_exc()
    
    return qlib_results


def generate_closure_manifest(
    output_dir: Path,
    data_metadata: Dict[str, Any],
    baseline_results: Dict[str, Any],
    baseline_floor: Dict[str, Any],
    qlib_results: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Generate comprehensive closure manifest.
    """
    manifest = {
        "chapter": "6",
        "type": "closure_reference",
        "generated_at": datetime.utcnow().isoformat() + "Z",
        
        # Git state
        "git": get_git_status(),
        
        # Environment
        "environment": get_environment_snapshot(),
        
        # Data
        "data": data_metadata,
        
        # Baseline results summary
        "baselines": {
            "count": len(baseline_results.get("baselines", {})),
            "factor_baselines": FACTOR_BASELINES,
            "sanity_baselines": SANITY_BASELINES,
            "cadences": baseline_results.get("cadences", []),
            "sanity_passed": all(
                c.get("passed", True) 
                for c in baseline_results.get("sanity_checks", {}).values()
            ),
        },
        
        # Baseline floor
        "baseline_floor": baseline_floor,
        
        # Qlib shadow
        "qlib_shadow": qlib_results,
        
        # Output paths
        "output_dir": str(output_dir),
        "artifacts": [
            "DATA_MANIFEST.json",
            "BASELINE_FLOOR.json",
            "CLOSURE_MANIFEST.json",
            "BASELINE_REFERENCE.md",
        ],
    }
    
    # Save manifest
    manifest_path = output_dir / "CLOSURE_MANIFEST.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    
    return manifest


def generate_baseline_reference_doc(
    output_dir: Path,
    manifest: Dict[str, Any],
    baseline_floor: Dict[str, Any],
) -> None:
    """
    Generate human-readable baseline reference document.
    """
    doc_lines = [
        "# Chapter 6 Baseline Reference",
        "",
        f"**Generated:** {manifest['generated_at']}",
        f"**Commit:** {manifest['git'].get('commit_hash', 'unknown')}",
        "",
        "---",
        "",
        "## Baseline Floor Summary",
        "",
        "This document records the frozen baseline floor for Chapter 7+ model comparisons.",
        "",
        "### Best Baseline per Horizon",
        "",
        "| Horizon | Best Baseline | Median RankIC |",
        "|---------|---------------|---------------|",
    ]
    
    for horizon in [20, 60, 90]:
        info = baseline_floor.get("best_baseline_per_horizon", {}).get(horizon, {})
        baseline = info.get("baseline", "N/A")
        ic = info.get("median_rankic", 0)
        ic_str = f"{ic:.4f}" if ic else "N/A"
        doc_lines.append(f"| {horizon}d | {baseline} | {ic_str} |")
    
    doc_lines.extend([
        "",
        "### Sanity Check",
        "",
        f"**naive_random RankIC ≈ 0:** {'✅ PASSED' if baseline_floor.get('sanity_passed', False) else '❌ FAILED'}",
        "",
        "### Data Snapshot",
        "",
        f"- **Source:** {manifest['data'].get('source', 'unknown')}",
        f"- **Rows:** {manifest['data'].get('n_rows', 'N/A'):,}",
        f"- **Date Range:** {manifest['data'].get('date_range', {}).get('min', 'N/A')} to {manifest['data'].get('date_range', {}).get('max', 'N/A')}",
        f"- **Data Hash:** {manifest['data'].get('data_hash', 'unknown')[:16]}...",
        "",
        "### Environment",
        "",
        f"- **Python:** {manifest['environment'].get('python_version', 'unknown').split()[0]}",
        f"- **Platform:** {manifest['environment'].get('platform', 'unknown')}",
        "",
        "---",
        "",
        "## Usage",
        "",
        "To compare a model against these baselines:",
        "",
        "```python",
        "from src.evaluation import compute_acceptance_verdict",
        "",
        "# Load frozen baseline summaries",
        "baseline_summaries = load_baseline_summaries('evaluation_outputs/chapter6_closure')",
        "",
        "# Run your model through the same pipeline",
        "model_results = run_experiment(model_spec, features_df, output_dir, FULL_MODE)",
        "",
        "# Compare",
        "verdict = compute_acceptance_verdict(",
        "    model_results['fold_summaries'],",
        "    baseline_summaries,",
        "    model_results['cost_overlays'],",
        "    model_results['churn_series']",
        ")",
        "```",
        "",
        "---",
        "",
        f"**Frozen at commit:** `{manifest['git'].get('commit_hash', 'unknown')}`",
    ])
    
    doc_path = output_dir / "BASELINE_REFERENCE.md"
    with open(doc_path, "w") as f:
        f.write("\n".join(doc_lines))
    
    print(f"\n  Generated reference doc: {doc_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Run Chapter 6 closure")
    parser.add_argument(
        "--mode", 
        choices=["synthetic", "duckdb", "auto"],
        default="auto",
        help="Data source mode: 'auto' uses duckdb if available else fails, 'synthetic' for pipeline testing, 'duckdb' for real data"
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run smoke test (smaller data, faster)"
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=DEFAULT_DUCKDB_PATH,
        help="Path to DuckDB database"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory (default: based on mode)"
    )
    
    args = parser.parse_args()
    
    # Determine actual mode
    if args.mode == "auto":
        if check_duckdb_available(args.db_path):
            actual_mode = "duckdb"
        else:
            # FAIL loudly if auto mode and no duckdb
            print("=" * 60)
            print("ERROR: DuckDB feature store not found!")
            print("=" * 60)
            print(f"\nExpected: {args.db_path}")
            print("\nTo build the feature store from FMP data, run:")
            print("  export FMP_KEYS='your_fmp_premium_api_key'")
            print("  python scripts/build_features_duckdb.py")
            print("\nTo run with synthetic data (for testing only):")
            print("  python scripts/run_chapter6_closure.py --mode synthetic")
            print()
            return 1
    else:
        actual_mode = args.mode
    
    # Set output directory based on mode
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = CLOSURE_DIR_REAL if actual_mode == "duckdb" else CLOSURE_DIR_SYNTH
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("CHAPTER 6 CLOSURE EXECUTION")
    print("=" * 60)
    print(f"Data mode: {actual_mode.upper()}")
    print(f"Run mode: {'SMOKE' if args.smoke else 'FULL'}")
    print(f"Output: {output_dir}")
    if actual_mode == "duckdb":
        print(f"DuckDB: {args.db_path}")
    print()
    
    # Step 1: Load features
    print("Step 1: Loading features data...")
    
    if actual_mode == "synthetic":
        config = SMOKE_CONFIG if args.smoke else SYNTHETIC_CONFIG
        features_df, data_metadata = load_features_for_evaluation(
            source="synthetic",
            config=config,
        )
    else:
        # Load from DuckDB
        try:
            features_df, data_metadata = load_features_for_evaluation(
                source="duckdb",
                db_path=args.db_path,
            )
        except RuntimeError as e:
            print(f"\n✗ Failed to load from DuckDB: {e}")
            return 1
    
    # Validate features
    validation = validate_features_for_evaluation(features_df, strict=True)
    print(f"  ✓ Loaded {len(features_df):,} rows")
    print(f"  ✓ {data_metadata['n_dates']} dates, {data_metadata['n_stocks']} stocks")
    print(f"  ✓ Source: {data_metadata.get('source', 'unknown').upper()}")
    
    # Save data manifest
    generate_data_manifest(features_df, data_metadata, output_dir)
    
    # Step 2: Run baseline evaluation
    print("\nStep 2: Running baseline evaluation...")
    baseline_results = run_baseline_closure(
        features_df=features_df,
        output_dir=output_dir,
        mode="smoke" if args.smoke else "full",
    )
    
    # Step 3: Compute baseline floor
    print("\nStep 3: Computing baseline floor...")
    baseline_floor = compute_baseline_floor(baseline_results, output_dir)
    
    # Step 4: Run Qlib shadow (optional)
    print("\nStep 4: Running Qlib shadow evaluation...")
    qlib_results = run_qlib_shadow_closure(
        baseline_results, features_df, output_dir
    )
    
    # Step 5: Generate closure manifest
    print("\nStep 5: Generating closure manifest...")
    manifest = generate_closure_manifest(
        output_dir,
        data_metadata,
        baseline_results,
        baseline_floor,
        qlib_results,
    )
    
    # Step 6: Generate reference document
    print("\nStep 6: Generating baseline reference document...")
    generate_baseline_reference_doc(output_dir, manifest, baseline_floor)
    
    # Summary
    print("\n" + "=" * 60)
    print("CHAPTER 6 CLOSURE COMPLETE")
    print("=" * 60)
    data_source = "REAL (DuckDB)" if actual_mode == "duckdb" else "SYNTHETIC"
    print(f"\n✅ Data source: {data_source}")
    print(f"✅ Output directory: {output_dir}")
    print(f"✅ Commit hash: {manifest['git'].get('commit_hash', 'unknown')[:12]}")
    print(f"✅ Data hash: {data_metadata.get('data_hash', 'unknown')[:12]}")
    print(f"✅ Sanity passed: {baseline_floor.get('sanity_passed', False)}")
    
    print("\nArtifacts generated:")
    for artifact in manifest.get("artifacts", []):
        print(f"  - {artifact}")
    
    print("\n" + "-" * 60)
    if actual_mode == "duckdb":
        print("To freeze this REAL DATA baseline reference, run:")
        print(f"  git add {output_dir}")
        print("  git commit -m 'Chapter 6: Freeze REAL baseline reference'")
    else:
        print("⚠️  This is a SYNTHETIC data run (for testing only)")
        print("   For production baseline reference, run with real data:")
        print("   python scripts/run_chapter6_closure.py")
    print("-" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())


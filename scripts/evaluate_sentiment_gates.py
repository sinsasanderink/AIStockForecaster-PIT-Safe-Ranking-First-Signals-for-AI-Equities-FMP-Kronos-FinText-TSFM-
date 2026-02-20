#!/usr/bin/env python
"""
Chapter 10.4 — Gate Evaluation & Orthogonality Check
======================================================

Evaluates sentiment signal against the standard gates, then checks
orthogonality with existing signals (LGB, FinText, factor baselines).

Usage:
    python scripts/evaluate_sentiment_gates.py --eval-dir evaluation_outputs/chapter10_sentiment_smoke
    python scripts/evaluate_sentiment_gates.py --eval-dir evaluation_outputs/chapter10_sentiment_full
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

LGB_BASELINES = {20: 0.1009, 60: 0.1275, 90: 0.1808}
FACTOR_BASELINES = {20: 0.0283, 60: 0.0392, 90: 0.0169}


def load_eval_rows(eval_dir: Path) -> pd.DataFrame:
    """Load evaluation rows from parquet."""
    parquet_path = eval_dir / "eval_rows.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(f"No eval_rows.parquet in {eval_dir}")
    return pd.read_parquet(parquet_path)


def compute_metrics(eval_rows: pd.DataFrame) -> dict:
    """Compute per-horizon metrics from evaluation rows."""
    metrics = {}

    for horizon in [20, 60, 90]:
        h_rows = eval_rows[eval_rows["horizon"] == horizon]
        if h_rows.empty:
            continue

        # Per-date RankIC
        per_date_ic = h_rows.groupby("as_of_date")[["score", "excess_return"]].apply(
            lambda g: stats.spearmanr(
                g["score"], g["excess_return"]
            ).statistic
            if len(g) >= 5
            else np.nan
        )
        per_date_ic = per_date_ic.dropna()

        # Churn: fraction of top-10 that changes between dates
        dates_sorted = sorted(h_rows["as_of_date"].unique())
        churns = []
        for i in range(1, len(dates_sorted)):
            prev_date = dates_sorted[i - 1]
            curr_date = dates_sorted[i]
            prev_top = set(
                h_rows[h_rows["as_of_date"] == prev_date]
                .nlargest(10, "score")["stable_id"]
            )
            curr_top = set(
                h_rows[h_rows["as_of_date"] == curr_date]
                .nlargest(10, "score")["stable_id"]
            )
            if prev_top and curr_top:
                overlap = len(prev_top & curr_top)
                churns.append(1.0 - overlap / 10.0)

        metrics[horizon] = {
            "mean_rankic": float(per_date_ic.mean()),
            "median_rankic": float(per_date_ic.median()),
            "std_rankic": float(per_date_ic.std()),
            "pct_positive": float((per_date_ic > 0).mean()),
            "n_dates": len(per_date_ic),
            "median_churn": float(np.median(churns)) if churns else 0.0,
            "mean_churn": float(np.mean(churns)) if churns else 0.0,
        }

    return metrics


def evaluate_gates(metrics: dict) -> dict:
    """Evaluate the three standard gates."""
    gates = {}

    # Gate 1: Factor Baseline — Mean RankIC >= 0.02 for >= 2 horizons
    passing_horizons = [
        h for h, m in metrics.items() if m["mean_rankic"] >= 0.02
    ]
    gates["gate_1_factor"] = {
        "description": "Mean RankIC >= 0.02 for >= 2 horizons",
        "passing_horizons": passing_horizons,
        "pass": len(passing_horizons) >= 2,
    }

    # Gate 2: ML Baseline — any horizon RankIC >= 0.05 OR within 0.03 of LGB
    gate2_pass = False
    gate2_details = {}
    for h, m in metrics.items():
        abs_pass = m["mean_rankic"] >= 0.05
        lgb = LGB_BASELINES.get(h, 0.0)
        rel_pass = m["mean_rankic"] >= (lgb - 0.03)
        gate2_details[h] = {
            "mean_rankic": m["mean_rankic"],
            "abs_pass": abs_pass,
            "rel_pass": rel_pass,
            "lgb_baseline": lgb,
        }
        if abs_pass or rel_pass:
            gate2_pass = True

    gates["gate_2_ml"] = {
        "description": "Any horizon RankIC >= 0.05 OR within 0.03 of LGB",
        "details": gate2_details,
        "pass": gate2_pass,
    }

    # Gate 3: Practical — median churn <= 30%
    churn_values = {h: m["median_churn"] for h, m in metrics.items()}
    gates["gate_3_practical"] = {
        "description": "Median churn <= 30%",
        "churn_by_horizon": churn_values,
        "pass": all(c <= 0.30 for c in churn_values.values()),
    }

    return gates


def compute_orthogonality(eval_dir: Path, eval_rows: pd.DataFrame) -> dict:
    """
    Compute score correlations with other signals if their
    evaluation outputs exist.
    """
    ortho = {}

    other_signals = {
        "fintext": "evaluation_outputs/chapter9_fintext_small_smoke/chapter9_fintext_small_smoke",
        "lgb": "evaluation_outputs/chapter7_tabular_lgb_full/monthly/baseline_tabular_lgb_monthly",
        "momentum": "evaluation_outputs/chapter6_closure_real/baseline_momentum_composite_monthly/baseline_momentum_composite_monthly",
    }

    for name, other_dir_str in other_signals.items():
        other_dir = Path(other_dir_str)
        other_parquet = other_dir / "eval_rows.parquet"
        if not other_parquet.exists():
            logger.info(f"  {name}: no eval_rows found at {other_parquet}")
            continue

        other_rows = pd.read_parquet(other_parquet)

        for horizon in [20, 60, 90]:
            sent_h = eval_rows[eval_rows["horizon"] == horizon][
                ["as_of_date", "stable_id", "score"]
            ].rename(columns={"score": "sent_score"})
            other_h = other_rows[other_rows["horizon"] == horizon][
                ["as_of_date", "stable_id", "score"]
            ].rename(columns={"score": "other_score"})

            merged = sent_h.merge(
                other_h, on=["as_of_date", "stable_id"], how="inner"
            )

            if len(merged) < 20:
                continue

            per_date_corr = merged.groupby("as_of_date")[["sent_score", "other_score"]].apply(
                lambda g: stats.spearmanr(
                    g["sent_score"], g["other_score"]
                ).statistic
                if len(g) >= 5
                else np.nan
            )
            per_date_corr = per_date_corr.dropna()

            if len(per_date_corr) == 0:
                continue

            key = f"{name}_{horizon}d"
            ortho[key] = {
                "mean_corr": float(per_date_corr.mean()),
                "median_corr": float(per_date_corr.median()),
                "n_dates": len(per_date_corr),
            }

    # Correlation with momentum factor (from features)
    for horizon in [20, 60, 90]:
        h_rows = eval_rows[eval_rows["horizon"] == horizon]
        if "momentum_composite" in h_rows.columns:
            mom_col = "momentum_composite"
        elif "mom_12m" in h_rows.columns:
            mom_col = "mom_12m"
        else:
            continue

        per_date_corr = h_rows.groupby("as_of_date").apply(
            lambda g: stats.spearmanr(g["score"], g[mom_col]).statistic
            if len(g) >= 5 and g[mom_col].notna().sum() >= 5
            else np.nan
        )
        per_date_corr = per_date_corr.dropna()
        if len(per_date_corr) > 0:
            key = f"momentum_{horizon}d"
            ortho[key] = {
                "mean_corr": float(per_date_corr.mean()),
                "median_corr": float(per_date_corr.median()),
                "n_dates": len(per_date_corr),
            }

    return ortho


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate sentiment gates and orthogonality"
    )
    parser.add_argument(
        "--eval-dir",
        required=True,
        help="Path to evaluation output directory",
    )
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    logger.info("=" * 60)
    logger.info("CHAPTER 10.4: GATE EVALUATION & ORTHOGONALITY")
    logger.info("=" * 60)
    logger.info(f"Eval dir: {eval_dir}")

    eval_rows = load_eval_rows(eval_dir)
    logger.info(f"Loaded {len(eval_rows):,} evaluation rows")

    # Compute metrics
    metrics = compute_metrics(eval_rows)
    logger.info("\n--- Per-Horizon Metrics ---")
    for h, m in sorted(metrics.items()):
        logger.info(
            f"  {h}d: Mean RankIC={m['mean_rankic']:.4f}, "
            f"Median={m['median_rankic']:.4f}, "
            f"Std={m['std_rankic']:.4f}, "
            f"Positive={m['pct_positive']*100:.0f}%, "
            f"Churn={m['median_churn']*100:.0f}%, "
            f"N={m['n_dates']}"
        )

    # Evaluate gates
    gates = evaluate_gates(metrics)
    logger.info("\n--- Gate Results ---")
    for gate_name, gate_result in gates.items():
        status = "PASS" if gate_result["pass"] else "FAIL"
        logger.info(
            f"  {gate_name}: {'✅' if gate_result['pass'] else '❌'} "
            f"{status} — {gate_result['description']}"
        )

    all_pass = all(g["pass"] for g in gates.values())
    logger.info(
        f"\n  Overall: {'✅ ALL GATES PASS' if all_pass else '❌ SOME GATES FAILED'}"
    )

    # Orthogonality check
    logger.info("\n--- Orthogonality Check ---")
    ortho = compute_orthogonality(eval_dir, eval_rows)
    if ortho:
        for key, vals in sorted(ortho.items()):
            corr = vals["mean_corr"]
            verdict = (
                "HIGH fusion value"
                if abs(corr) < 0.3
                else "MODERATE fusion value"
                if abs(corr) < 0.5
                else "LOW fusion value"
            )
            logger.info(
                f"  {key}: mean ρ = {corr:.3f} → {verdict}"
            )
    else:
        logger.info("  No other signal outputs found for comparison")

    # Save results
    results = {
        "metrics": {str(k): v for k, v in metrics.items()},
        "gates": gates,
        "orthogonality": ortho,
        "all_gates_pass": all_pass,
    }
    output_path = eval_dir / "gate_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()

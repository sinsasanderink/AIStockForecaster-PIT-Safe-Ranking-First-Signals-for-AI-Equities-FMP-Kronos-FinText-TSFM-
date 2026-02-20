#!/usr/bin/env python
"""
Chapter 11.3 - Fusion Gate Evaluation
=====================================

Evaluates fusion variants against Chapter 11 gates:
1) Factor gate: Mean RankIC >= 0.02 for >= 2 horizons
2) ML gate: any horizon RankIC >= 0.05 or within 0.03 of LGB
3) Practical gate: median churn <= 30%
4) Fusion gate: best fusion RankIC beats best single model by >= 0.02
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import pandas as pd
from scipy import stats

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

HORIZONS = (20, 60, 90)


def load_eval_rows(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing eval_rows: {path}")
    return pd.read_parquet(path)


def compute_metrics(eval_rows: pd.DataFrame) -> Dict[int, dict]:
    """Compute per-horizon RankIC, IC stability, churn, and cost survival."""
    metrics: Dict[int, dict] = {}

    for horizon in HORIZONS:
        h_rows = eval_rows[eval_rows["horizon"] == horizon].copy()
        if h_rows.empty:
            continue

        per_date_ic = h_rows.groupby("as_of_date")[["score", "excess_return"]].apply(
            lambda g: stats.spearmanr(g["score"], g["excess_return"]).statistic
            if len(g) >= 5
            else np.nan
        )
        per_date_ic = per_date_ic.dropna()

        # IC stability = mean / std (higher = more stable)
        ic_mean = float(per_date_ic.mean()) if len(per_date_ic) else np.nan
        ic_std = float(per_date_ic.std()) if len(per_date_ic) > 1 else np.nan
        if not np.isnan(ic_mean) and not np.isnan(ic_std) and ic_std > 0:
            ic_stability = ic_mean / ic_std
        else:
            ic_stability = np.nan

        # Cost survival: % of folds where top-10 avg excess return > 0
        fold_ids = h_rows["fold_id"].unique() if "fold_id" in h_rows.columns else []
        fold_positive = 0
        fold_total = 0
        for fid in fold_ids:
            fold_h = h_rows[h_rows["fold_id"] == fid]
            if fold_h.empty:
                continue
            top10_er = (
                fold_h.groupby("as_of_date", group_keys=False)
                .apply(
                    lambda g: pd.Series(
                        [g.nlargest(10, "score")["excess_return"].mean()],
                        index=[g.name],
                    )
                    if len(g) >= 10
                    else pd.Series([np.nan], index=[g.name]),
                    include_groups=False,
                )
                .squeeze()
                .dropna()
            )
            if len(top10_er) == 0:
                continue
            fold_total += 1
            if top10_er.median() > 0:
                fold_positive += 1
        cost_survival = fold_positive / fold_total if fold_total > 0 else np.nan

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
                churns.append(1.0 - len(prev_top & curr_top) / 10.0)

        metrics[horizon] = {
            "mean_rankic": ic_mean,
            "median_rankic": float(per_date_ic.median()) if len(per_date_ic) else np.nan,
            "ic_stability": float(ic_stability),
            "pct_positive": float((per_date_ic > 0).mean()) if len(per_date_ic) else np.nan,
            "cost_survival": float(cost_survival),
            "n_dates": int(len(per_date_ic)),
            "n_folds": int(fold_total),
            "median_churn": float(np.median(churns)) if churns else np.nan,
            "mean_churn": float(np.mean(churns)) if churns else np.nan,
        }
    return metrics


def evaluate_gates(
    fusion_metrics: Dict[int, dict],
    lgb_metrics: Dict[int, dict],
    single_model_metrics: Dict[str, Dict[int, dict]],
) -> dict:
    """Evaluate all four Chapter 11 gates."""
    # Gate 1
    passing_gate1 = [
        h for h, m in fusion_metrics.items()
        if not np.isnan(m["mean_rankic"]) and m["mean_rankic"] >= 0.02
    ]
    gate_1 = {
        "description": "Mean RankIC >= 0.02 for >= 2 horizons",
        "passing_horizons": passing_gate1,
        "pass": len(passing_gate1) >= 2,
    }

    # Gate 2
    gate2_pass = False
    gate2_details = {}
    for h, m in fusion_metrics.items():
        lgb_ic = lgb_metrics.get(h, {}).get("mean_rankic", np.nan)
        abs_pass = (not np.isnan(m["mean_rankic"])) and m["mean_rankic"] >= 0.05
        rel_pass = (
            (not np.isnan(m["mean_rankic"]))
            and (not np.isnan(lgb_ic))
            and m["mean_rankic"] >= (lgb_ic - 0.03)
        )
        gate2_details[h] = {
            "fusion_rankic": m["mean_rankic"],
            "lgb_rankic": lgb_ic,
            "abs_pass": abs_pass,
            "rel_pass": rel_pass,
        }
        gate2_pass = gate2_pass or abs_pass or rel_pass
    gate_2 = {
        "description": "Any horizon RankIC >= 0.05 or within 0.03 of LGB",
        "details": gate2_details,
        "pass": gate2_pass,
    }

    # Gate 3
    churn_values = {
        h: m["median_churn"]
        for h, m in fusion_metrics.items()
        if not np.isnan(m["median_churn"])
    }
    gate_3 = {
        "description": "Median churn <= 30% across horizons",
        "churn_by_horizon": churn_values,
        "pass": bool(churn_values) and all(c <= 0.30 for c in churn_values.values()),
    }

    # Gate 4: compare best fusion vs best single model (including LGB)
    fusion_best = max(
        (
            m["mean_rankic"]
            for m in fusion_metrics.values()
            if not np.isnan(m["mean_rankic"])
        ),
        default=np.nan,
    )
    single_best = np.nan
    all_single_metrics = dict(single_model_metrics)
    all_single_metrics["lgb"] = lgb_metrics
    for model_metrics in all_single_metrics.values():
        for m in model_metrics.values():
            v = m["mean_rankic"]
            if not np.isnan(v):
                single_best = v if np.isnan(single_best) else max(single_best, v)
    improvement = fusion_best - single_best if not np.isnan(fusion_best) and not np.isnan(single_best) else np.nan
    gate_4 = {
        "description": "Best fusion RankIC >= best single model RankIC + 0.02",
        "best_fusion_rankic": fusion_best,
        "best_single_rankic": single_best,
        "improvement": improvement,
        "pass": (not np.isnan(improvement)) and (improvement >= 0.02),
    }

    return {
        "gate_1_factor": gate_1,
        "gate_2_ml": gate_2,
        "gate_3_practical": gate_3,
        "gate_4_fusion_specific": gate_4,
    }


def metrics_to_rows(model_name: str, metrics: Dict[int, dict]) -> list:
    rows = []
    for h in HORIZONS:
        m = metrics.get(h)
        if not m:
            continue
        rows.append(
            {
                "model": model_name,
                "horizon": h,
                "mean_rankic": m["mean_rankic"],
                "median_rankic": m["median_rankic"],
                "ic_stability": m.get("ic_stability", np.nan),
                "pct_positive": m["pct_positive"],
                "cost_survival": m.get("cost_survival", np.nan),
                "median_churn": m["median_churn"],
                "n_dates": m["n_dates"],
                "n_folds": m.get("n_folds", np.nan),
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser(description="Evaluate Chapter 11 fusion gates.")
    parser.add_argument(
        "--fusion-eval",
        required=True,
        help="Path to fusion eval_rows.parquet",
    )
    parser.add_argument(
        "--lgb-eval",
        required=True,
        help="Path to LGB baseline eval_rows.parquet",
    )
    parser.add_argument(
        "--single-evals",
        nargs="+",
        required=True,
        help="Single-model baselines in format name=path",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write gate outputs",
    )
    args = parser.parse_args()

    fusion_eval = load_eval_rows(Path(args.fusion_eval))
    lgb_eval = load_eval_rows(Path(args.lgb_eval))

    singles = {}
    for pair in args.single_evals:
        if "=" not in pair:
            raise ValueError(f"Invalid --single-evals entry: {pair}")
        name, path = pair.split("=", 1)
        singles[name] = load_eval_rows(Path(path))

    fusion_metrics = compute_metrics(fusion_eval)
    lgb_metrics = compute_metrics(lgb_eval)
    single_metrics = {name: compute_metrics(df) for name, df in singles.items()}
    gates = evaluate_gates(fusion_metrics, lgb_metrics, single_metrics)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    rows.extend(metrics_to_rows("fusion", fusion_metrics))
    rows.extend(metrics_to_rows("lgb", lgb_metrics))
    for name, metrics in single_metrics.items():
        rows.extend(metrics_to_rows(name, metrics))
    table = pd.DataFrame(rows)
    table_path = output_dir / "comparison_table.csv"
    table.to_csv(table_path, index=False)

    gate_path = output_dir / "fusion_gate_results.json"
    with gate_path.open("w") as f:
        json.dump(gates, f, indent=2, default=str)

    logger.info("Saved comparison table to %s", table_path)
    logger.info("Saved gate results to %s", gate_path)
    logger.info("Gate summary:")
    for gate_name, gate_result in gates.items():
        status = "PASS" if gate_result["pass"] else "FAIL"
        logger.info("  %s: %s", gate_name, status)


if __name__ == "__main__":
    main()

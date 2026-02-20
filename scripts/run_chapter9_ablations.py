#!/usr/bin/env python
"""
Chapter 9 — Section 9.8: Ablation Studies

Systematically varies FinText hyper-parameters in SMOKE mode and collects
per-variant signal-quality metrics (mean/median RankIC, churn) to find the
optimal configuration.

Ablation axes (from outline):
  1. Lookback window   : 21 vs 252 vs 512
  2. Model size         : Tiny (8M) vs Mini (20M) vs Small (46M)
  3. Model dataset      : US vs Global vs Augmented
  4. Score aggregation  : median vs mean vs trimmed_mean
  5. Num samples        : 5 vs 20 vs 50
  6. EMA half-life      : 0 (none) vs 3 vs 5 vs 10

Strategy: use Tiny model for most ablations (fast, ~1.5s/date).
Only re-run promising combos with Small model.

Usage:
    python scripts/run_chapter9_ablations.py           # default matrix
    python scripts/run_chapter9_ablations.py --stub     # fast stub run for testing
    python scripts/run_chapter9_ablations.py --quick    # reduced matrix (3 variants)
"""

import gc
import json
import logging
import sys
import time
import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.fintext_adapter import (
    FinTextAdapter,
    _apply_ema_smoothing,
)
from src.data.excess_return_store import ExcessReturnStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ============================================================================
# ABLATION VARIANT DEFINITION
# ============================================================================

def get_ablation_matrix(quick: bool = False) -> List[Dict[str, Any]]:
    """
    Build the ablation matrix.

    Each entry is a dict of parameters that differ from the baseline config.
    The baseline is: Small/US/lookback=21/samples=20/median/ema=5.
    """
    baseline = {
        "model_size": "Tiny",       # Use Tiny for speed in ablations
        "model_dataset": "US",
        "lookback": 21,
        "num_samples": 20,
        "score_aggregation": "median",
        "ema_halflife": 5,
        "horizon_strategy": "single_step",
    }

    if quick:
        return [
            {**baseline, "tag": "baseline_tiny"},
            {**baseline, "score_aggregation": "mean", "tag": "agg_mean"},
            {**baseline, "ema_halflife": 0, "tag": "ema_off"},
        ]

    variants = []

    # --- Baseline (Tiny, for reference) ---
    variants.append({**baseline, "tag": "baseline_tiny"})

    # --- 1. Score aggregation (fast, same model/lookback) ---
    variants.append({**baseline, "score_aggregation": "mean", "tag": "agg_mean"})
    variants.append({**baseline, "score_aggregation": "trimmed_mean", "tag": "agg_trimmed"})

    # --- 2. Num samples (fast, same model/lookback) ---
    variants.append({**baseline, "num_samples": 5, "tag": "samples_5"})
    variants.append({**baseline, "num_samples": 50, "tag": "samples_50"})

    # --- 3. EMA half-life (fast, same model/lookback) ---
    variants.append({**baseline, "ema_halflife": 0, "tag": "ema_off"})
    variants.append({**baseline, "ema_halflife": 3, "tag": "ema_3"})
    variants.append({**baseline, "ema_halflife": 10, "tag": "ema_10"})

    # --- 4. Model dataset (moderate cost — same model size) ---
    variants.append({**baseline, "model_dataset": "Global", "tag": "dataset_global"})
    variants.append({**baseline, "model_dataset": "Augmented", "tag": "dataset_augmented"})

    # --- 5. Model size (costly — Small already evaluated in 9.6) ---
    # Mini included here; Small is added from 9.6 SMOKE results separately.
    variants.append({**baseline, "model_size": "Mini", "tag": "model_mini"})

    # --- 6. Lookback window (very costly with large windows) ---
    # 21 is already baseline; 252 was tested and slightly worse but worth
    # including for completeness. Skip 512 (too slow, marginal gains).
    variants.append({**baseline, "lookback": 252, "tag": "lookback_252"})

    return variants


# ============================================================================
# SINGLE ABLATION EVALUATOR
# ============================================================================

def evaluate_variant(
    variant: Dict[str, Any],
    features_df: pd.DataFrame,
    db_path: str = "data/features.duckdb",
    device: str = "cpu",
    use_stub: bool = False,
    max_folds: int = 3,
) -> Dict[str, Any]:
    """
    Run one ablation variant through SMOKE evaluation and return metrics.
    """
    tag = variant["tag"]
    logger.info(f"\n{'='*60}")
    logger.info(f"ABLATION: {tag}")
    logger.info(f"  Config: {json.dumps({k: v for k, v in variant.items() if k != 'tag'}, indent=2)}")
    logger.info(f"{'='*60}")

    start_time = time.time()

    # Create a fresh adapter for this variant
    adapter = FinTextAdapter.from_pretrained(
        db_path=db_path,
        model_size=variant["model_size"],
        model_dataset=variant["model_dataset"],
        lookback=variant["lookback"],
        num_samples=variant["num_samples"],
        device=device,
        use_stub=use_stub,
        horizon_strategy=variant["horizon_strategy"],
        score_aggregation=variant["score_aggregation"],
    )

    # Generate folds (simple monthly splits from features_df)
    date_col = pd.to_datetime(features_df["date"])
    all_months = sorted(date_col.dt.to_period("M").unique())

    # Use last months for evaluation (similar to SMOKE)
    eval_months = all_months[-max_folds:] if len(all_months) > max_folds else all_months

    horizons = [20, 60, 90]
    all_rows = []

    for fold_idx, month in enumerate(eval_months):
        month_mask = date_col.dt.to_period("M") == month
        fold_df = features_df[month_mask].copy()
        fold_id = f"fold_{fold_idx+1:02d}"

        if len(fold_df) == 0:
            continue

        unique_dates = sorted(fold_df["date"].unique())

        for h in horizons:
            fold_rows = []
            for asof_date in unique_dates:
                date_df = fold_df[fold_df["date"] == asof_date]
                tickers = date_df["ticker"].unique().tolist()

                scores_df = adapter.score_universe(
                    asof_date=pd.Timestamp(asof_date),
                    tickers=tickers,
                    horizon=h,
                    verbose=False,
                )

                if scores_df.empty:
                    continue

                # Determine excess return column
                er_col = f"excess_return_{h}d"
                if er_col not in date_df.columns:
                    er_col = "excess_return"
                    if er_col not in date_df.columns:
                        continue

                merged = date_df[["date", "ticker", "stable_id", er_col]].merge(
                    scores_df[["ticker", "score"]], on="ticker", how="inner"
                )
                merged = merged.rename(
                    columns={"date": "as_of_date", er_col: "excess_return"}
                )
                merged["fold_id"] = fold_id
                merged["horizon"] = h
                fold_rows.append(merged)

            if fold_rows:
                result = pd.concat(fold_rows, ignore_index=True)
                # Apply EMA smoothing
                ema_hl = variant["ema_halflife"]
                if ema_hl > 0:
                    result = _apply_ema_smoothing(result, halflife_days=ema_hl)
                all_rows.append(result)

    adapter.close()
    gc.collect()

    elapsed = time.time() - start_time

    if not all_rows:
        logger.warning(f"  No evaluation rows generated for {tag}")
        return {"tag": tag, "error": "no_rows", "elapsed_s": elapsed}

    eval_df = pd.concat(all_rows, ignore_index=True)

    # Compute metrics
    metrics = _compute_ablation_metrics(eval_df, horizons)
    metrics["tag"] = tag
    metrics["elapsed_s"] = round(elapsed, 1)
    metrics["n_eval_rows"] = len(eval_df)

    # Add config params
    for k, v in variant.items():
        if k != "tag":
            metrics[f"cfg_{k}"] = v

    logger.info(f"  Completed in {elapsed:.1f}s ({len(eval_df)} eval rows)")
    for h in horizons:
        hk = f"rankic_mean_{h}d"
        ck = f"churn_median_{h}d"
        logger.info(
            f"  {h}d: mean_RankIC={metrics.get(hk, 'N/A'):.4f}, "
            f"churn={metrics.get(ck, 'N/A'):.1%}"
        )

    return metrics


def _compute_ablation_metrics(
    eval_df: pd.DataFrame, horizons: List[int]
) -> Dict[str, Any]:
    """Compute RankIC and churn metrics for an ablation variant."""
    metrics: Dict[str, Any] = {}

    for h in horizons:
        h_df = eval_df[eval_df["horizon"] == h]

        # RankIC per date
        ics = []
        for _, group in h_df.groupby("as_of_date"):
            if len(group) < 10:
                continue
            ic, _ = stats.spearmanr(group["score"], group["excess_return"])
            if not np.isnan(ic):
                ics.append(ic)

        metrics[f"rankic_mean_{h}d"] = float(np.mean(ics)) if ics else float("nan")
        metrics[f"rankic_median_{h}d"] = float(np.median(ics)) if ics else float("nan")
        metrics[f"rankic_pct_pos_{h}d"] = float(np.mean([x > 0 for x in ics])) if ics else 0.0
        metrics[f"rankic_n_dates_{h}d"] = len(ics)

        # Churn
        dates = sorted(h_df["as_of_date"].unique())
        churns = []
        for i in range(1, len(dates)):
            prev_top = set(
                h_df[h_df["as_of_date"] == dates[i - 1]]
                .nlargest(10, "score")["ticker"]
            )
            curr_top = set(
                h_df[h_df["as_of_date"] == dates[i]]
                .nlargest(10, "score")["ticker"]
            )
            if prev_top and curr_top:
                churns.append(1 - len(prev_top & curr_top) / 10)

        metrics[f"churn_median_{h}d"] = float(np.median(churns)) if churns else 1.0

    # Aggregate across horizons
    all_ics = [
        metrics.get(f"rankic_mean_{h}d", float("nan")) for h in horizons
    ]
    valid_ics = [x for x in all_ics if not np.isnan(x)]
    metrics["rankic_mean_avg"] = float(np.mean(valid_ics)) if valid_ics else float("nan")

    all_churns = [metrics.get(f"churn_median_{h}d", 1.0) for h in horizons]
    metrics["churn_median_avg"] = float(np.mean(all_churns))

    # Gate checks
    gate1_pass = sum(1 for h in horizons if metrics.get(f"rankic_mean_{h}d", 0) >= 0.02) >= 2
    gate2_pass = any(metrics.get(f"rankic_mean_{h}d", 0) >= 0.05 for h in horizons)
    gate3_pass = all(metrics.get(f"churn_median_{h}d", 1) <= 0.30 for h in horizons)

    metrics["gate_1_pass"] = gate1_pass
    metrics["gate_2_pass"] = gate2_pass
    metrics["gate_3_pass"] = gate3_pass
    metrics["all_gates_pass"] = gate1_pass and gate2_pass and gate3_pass

    return metrics


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Chapter 9: Ablation Studies")
    parser.add_argument("--stub", action="store_true", help="Use stub predictor")
    parser.add_argument("--quick", action="store_true", help="Reduced 3-variant matrix")
    parser.add_argument("--device", default="cpu", help="Device for inference")
    parser.add_argument("--db-path", default="data/features.duckdb")
    parser.add_argument("--max-folds", type=int, default=3, help="Folds per variant")
    parser.add_argument(
        "--output", type=Path,
        default=Path("evaluation_outputs/chapter9_ablations"),
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    # Load features + labels
    import duckdb
    logger.info("Loading features and labels from DuckDB …")
    con = duckdb.connect(args.db_path, read_only=True)
    features_df = con.execute(
        "SELECT * FROM features ORDER BY date, ticker"
    ).fetchdf()

    # Labels table uses as_of_date + horizon (long format) → pivot to wide
    labels_df = con.execute(
        "SELECT as_of_date, ticker, stable_id, horizon, excess_return "
        "FROM labels ORDER BY as_of_date, ticker"
    ).fetchdf()
    con.close()

    # Pivot labels to wide format: excess_return_20d, excess_return_60d, excess_return_90d
    labels_wide = labels_df.pivot_table(
        index=["as_of_date", "ticker", "stable_id"],
        columns="horizon",
        values="excess_return",
    ).reset_index()
    labels_wide.columns = [
        f"excess_return_{c}d" if isinstance(c, int) else c
        for c in labels_wide.columns
    ]
    labels_wide = labels_wide.rename(columns={"as_of_date": "date"})

    # Merge
    merge_keys = ["date", "ticker", "stable_id"]
    features_df = features_df.merge(labels_wide, on=merge_keys, how="inner")
    logger.info(f"Combined dataset: {len(features_df)} rows, {features_df['ticker'].nunique()} tickers")

    # Build ablation matrix
    matrix = get_ablation_matrix(quick=args.quick)
    logger.info(f"\nAblation matrix: {len(matrix)} variants")
    for v in matrix:
        logger.info(f"  - {v['tag']}")

    # Run each variant
    all_results = []
    for i, variant in enumerate(matrix):
        logger.info(f"\n>>> Variant {i+1}/{len(matrix)}: {variant['tag']} <<<")
        result = evaluate_variant(
            variant,
            features_df=features_df,
            db_path=args.db_path,
            device=args.device,
            use_stub=args.stub,
            max_folds=args.max_folds,
        )
        all_results.append(result)

        # Save incrementally
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(args.output / "ablation_results.csv", index=False)

    # Final summary
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(args.output / "ablation_results.csv", index=False)

    with open(args.output / "ablation_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Print summary table
    logger.info("\n" + "=" * 80)
    logger.info("ABLATION RESULTS SUMMARY")
    logger.info("=" * 80)

    display_cols = [
        "tag", "rankic_mean_avg",
        "rankic_mean_20d", "rankic_mean_60d", "rankic_mean_90d",
        "churn_median_avg", "all_gates_pass", "elapsed_s",
    ]
    avail_cols = [c for c in display_cols if c in results_df.columns]
    summary = results_df[avail_cols].copy()

    # Sort by average RankIC descending
    if "rankic_mean_avg" in summary.columns:
        summary = summary.sort_values("rankic_mean_avg", ascending=False)

    logger.info("\n" + summary.to_string(index=False))

    # Best variant
    if "rankic_mean_avg" in results_df.columns:
        best_idx = results_df["rankic_mean_avg"].idxmax()
        best = results_df.iloc[best_idx]
        logger.info(f"\n🏆 BEST VARIANT: {best['tag']}")
        logger.info(f"   Avg RankIC: {best['rankic_mean_avg']:.4f}")
        logger.info(f"   Avg Churn:  {best['churn_median_avg']:.1%}")
        logger.info(f"   All Gates:  {best['all_gates_pass']}")

    logger.info(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()

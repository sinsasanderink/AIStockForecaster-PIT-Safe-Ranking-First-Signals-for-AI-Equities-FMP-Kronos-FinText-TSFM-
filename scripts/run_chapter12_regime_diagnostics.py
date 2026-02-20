#!/usr/bin/env python
"""
Chapter 12.1 — Regime-Conditional Performance Diagnostics
=========================================================

Slices the LGB baseline (and optional other models) by regime buckets
and computes per-regime signal quality metrics.

Outputs:
  - evaluation_outputs/chapter12/regime_diagnostics.csv
  - evaluation_outputs/chapter12/regime_diagnostics.json
  - evaluation_outputs/chapter12/rankic_vs_vix_correlation.json
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import duckdb
import numpy as np
import pandas as pd
from scipy import stats

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

HORIZONS = (20, 60, 90)

REGIME_AXES = {
    "vix_percentile_252d": {
        "low": lambda x: x <= 33,
        "mid": lambda x: (x > 33) & (x <= 67),
        "high": lambda x: x > 67,
    },
    "market_regime_label": {
        "bull": lambda x: x == "bull",
        "neutral": lambda x: x == "neutral",
        "bear": lambda x: x == "bear",
    },
}


def load_regime_data(db_path: str = "data/features.duckdb") -> pd.DataFrame:
    """Load regime features from DuckDB, decode integer columns."""
    conn = duckdb.connect(db_path, read_only=True)
    df = conn.execute("SELECT * FROM regime").df()
    conn.close()

    df["date"] = pd.to_datetime(df["date"]).dt.date

    market_regime_map = {-1: "bear", 0: "neutral", 1: "bull"}
    df["market_regime_label"] = df["market_regime"].map(market_regime_map)

    return df


def compute_per_date_rankic(eval_rows: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Compute per-date Spearman RankIC for a given horizon."""
    hdf = eval_rows[eval_rows["horizon"] == horizon]

    records = []
    for dt, grp in hdf.groupby("as_of_date"):
        if len(grp) < 5:
            continue
        ic = stats.spearmanr(grp["score"], grp["excess_return"]).statistic
        top10 = grp.nlargest(10, "score")
        top10_er = top10["excess_return"].mean() if len(top10) >= 10 else np.nan
        records.append({
            "as_of_date": dt,
            "horizon": horizon,
            "rankic": ic,
            "n_stocks": len(grp),
            "top10_mean_er": top10_er,
        })

    return pd.DataFrame(records)


def compute_regime_metrics(
    per_date_ic: pd.DataFrame,
    regime_series: pd.Series,
    bucket_name: str,
) -> dict:
    """Compute signal-quality metrics for a subset of dates in a regime bucket."""
    subset = per_date_ic[regime_series].copy()
    if len(subset) < 5:
        return {
            "bucket": bucket_name,
            "n_dates": len(subset),
            "mean_rankic": np.nan,
            "median_rankic": np.nan,
            "ic_stability": np.nan,
            "pct_positive": np.nan,
            "cost_survival": np.nan,
        }

    ics = subset["rankic"].dropna()
    mean_ic = float(ics.mean())
    std_ic = float(ics.std())
    ic_stability = mean_ic / std_ic if std_ic > 0 else 0.0

    top10_ers = subset["top10_mean_er"].dropna()
    cost_surv = float((top10_ers > 0).mean()) if len(top10_ers) > 0 else np.nan

    return {
        "bucket": bucket_name,
        "n_dates": int(len(subset)),
        "mean_rankic": float(mean_ic),
        "median_rankic": float(ics.median()),
        "ic_stability": float(ic_stability),
        "pct_positive": float((ics > 0).mean()),
        "cost_survival": float(cost_surv),
    }


def run_diagnostics(
    eval_rows_path: str,
    model_name: str,
    regime_df: pd.DataFrame,
    output_dir: Path,
) -> Tuple[pd.DataFrame, dict]:
    """Run regime diagnostics for one model."""
    eval_rows = pd.read_parquet(eval_rows_path)
    eval_rows["as_of_date"] = pd.to_datetime(eval_rows["as_of_date"]).dt.date
    n_folds = eval_rows["fold_id"].nunique()
    n_dates = eval_rows["as_of_date"].nunique()
    logger.info(
        "Loaded %s: %d rows, %d folds, %d dates",
        model_name, len(eval_rows), n_folds, n_dates,
    )

    regime_lookup = regime_df.set_index("date")

    all_rows = []
    correlation_results = {}

    for horizon in HORIZONS:
        per_date_ic = compute_per_date_rankic(eval_rows, horizon)
        if per_date_ic.empty:
            continue

        per_date_ic["as_of_date_dt"] = pd.to_datetime(per_date_ic["as_of_date"])

        regime_cols_to_join = ["vix_percentile_252d", "vix_level", "market_regime_label"]
        for col in regime_cols_to_join:
            per_date_ic[col] = per_date_ic["as_of_date"].map(
                regime_lookup[col].to_dict() if col in regime_lookup.columns else {}
            )

        matched = per_date_ic["vix_percentile_252d"].notna().sum()
        logger.info(
            "  %dd: %d eval dates, %d matched to regime data (%.0f%%)",
            horizon, len(per_date_ic), matched,
            100 * matched / len(per_date_ic) if len(per_date_ic) > 0 else 0,
        )

        # --- Overall (all regimes) ---
        overall = compute_regime_metrics(
            per_date_ic,
            pd.Series(True, index=per_date_ic.index),
            "ALL",
        )
        overall.update({"model": model_name, "horizon": horizon, "regime_axis": "overall"})
        all_rows.append(overall)

        # --- Per regime axis ---
        for axis_name, buckets in REGIME_AXES.items():
            for bucket_name, condition_fn in buckets.items():
                col_vals = per_date_ic[axis_name]
                mask = condition_fn(col_vals) & col_vals.notna()
                metrics = compute_regime_metrics(per_date_ic, mask, bucket_name)
                metrics.update({
                    "model": model_name,
                    "horizon": horizon,
                    "regime_axis": axis_name,
                })
                all_rows.append(metrics)

        # --- Continuous correlation: RankIC vs VIX ---
        valid = per_date_ic.dropna(subset=["rankic", "vix_level"])
        if len(valid) >= 20:
            rho, pval = stats.spearmanr(valid["rankic"], valid["vix_level"])
            correlation_results[f"{horizon}d_rankic_vs_vix"] = {
                "spearman_rho": float(rho),
                "p_value": float(pval),
                "n": int(len(valid)),
            }
            rho_pct, pval_pct = stats.spearmanr(
                valid["rankic"], valid["vix_percentile_252d"]
            )
            correlation_results[f"{horizon}d_rankic_vs_vix_pctile"] = {
                "spearman_rho": float(rho_pct),
                "p_value": float(pval_pct),
                "n": int(len(valid)),
            }

    results_df = pd.DataFrame(all_rows)
    return results_df, correlation_results


def main():
    parser = argparse.ArgumentParser(
        description="Chapter 12.1 — Regime-Conditional Performance Diagnostics"
    )
    parser.add_argument(
        "--eval-paths",
        nargs="+",
        required=True,
        help="name=path pairs, e.g. lgb=evaluation_outputs/.../eval_rows.parquet",
    )
    parser.add_argument(
        "--db-path",
        default="data/features.duckdb",
        help="Path to DuckDB features database",
    )
    parser.add_argument(
        "--output-dir",
        default="evaluation_outputs/chapter12",
        help="Output directory",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    regime_df = load_regime_data(args.db_path)
    logger.info(
        "Loaded regime data: %d dates (%s to %s)",
        len(regime_df), regime_df["date"].min(), regime_df["date"].max(),
    )

    all_results = []
    all_correlations = {}

    for pair in args.eval_paths:
        if "=" not in pair:
            raise ValueError(f"Expected name=path format, got: {pair}")
        name, path = pair.split("=", 1)
        results_df, correlations = run_diagnostics(path, name, regime_df, output_dir)
        all_results.append(results_df)
        all_correlations[name] = correlations

    combined = pd.concat(all_results, ignore_index=True)

    col_order = [
        "model", "regime_axis", "bucket", "horizon",
        "n_dates", "mean_rankic", "median_rankic",
        "ic_stability", "pct_positive", "cost_survival",
    ]
    combined = combined[[c for c in col_order if c in combined.columns]]

    csv_path = output_dir / "regime_diagnostics.csv"
    combined.to_csv(csv_path, index=False, float_format="%.4f")
    logger.info("Saved diagnostics to %s", csv_path)

    json_path = output_dir / "regime_diagnostics.json"
    with json_path.open("w") as f:
        json.dump(
            {
                "diagnostics": combined.to_dict(orient="records"),
                "correlations": all_correlations,
            },
            f,
            indent=2,
            default=str,
        )
    logger.info("Saved JSON to %s", json_path)

    corr_path = output_dir / "rankic_vs_vix_correlation.json"
    with corr_path.open("w") as f:
        json.dump(all_correlations, f, indent=2, default=str)
    logger.info("Saved correlation results to %s", corr_path)

    # Print summary
    print("\n" + "=" * 80)
    print("CHAPTER 12.1 — REGIME-CONDITIONAL PERFORMANCE DIAGNOSTICS")
    print("=" * 80)

    for model in combined["model"].unique():
        mdf = combined[combined["model"] == model]
        print(f"\n--- {model} ---")

        for axis in mdf["regime_axis"].unique():
            adf = mdf[mdf["regime_axis"] == axis]
            print(f"\n  [{axis}]")
            print(f"  {'Bucket':<10} {'Hz':>4} {'N':>6} {'MedIC':>8} {'ICStab':>8} {'CostSurv':>10} {'%Pos':>6}")
            print(f"  {'-'*58}")
            for _, row in adf.iterrows():
                print(
                    f"  {row['bucket']:<10} {row['horizon']:>4}d"
                    f" {row['n_dates']:>6}"
                    f" {row['median_rankic']:>8.4f}"
                    f" {row['ic_stability']:>8.4f}"
                    f" {row['cost_survival']:>9.1%}"
                    f" {row['pct_positive']:>5.1%}"
                )

    if all_correlations:
        print("\n--- RankIC vs VIX Correlations ---")
        for model, corrs in all_correlations.items():
            print(f"\n  {model}:")
            for key, vals in corrs.items():
                print(f"    {key}: ρ={vals['spearman_rho']:.4f} (p={vals['p_value']:.4f}, n={vals['n']})")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()

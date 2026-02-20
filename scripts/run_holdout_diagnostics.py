#!/usr/bin/env python
"""
Holdout diagnostics: Separate overfitting from regime shift.

Three diagnostics:
  1. Retrain LGB on DEV-only (pre-2024), evaluate on FINAL-only
  2. Feature importance stability across time periods
  3. Deep-dive 20d model on 2024 (monthly RankIC, top-10 returns)
"""

import logging
import json
import sys
from pathlib import Path
from datetime import date

import numpy as np
import pandas as pd
from scipy import stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.baselines import (
    train_lgbm_ranking_model,
    predict_lgbm_scores,
    DEFAULT_TABULAR_FEATURES,
)
from src.evaluation.data_loader import load_features_from_duckdb

HOLDOUT_START = date(2024, 1, 1)
HORIZONS = [20, 60, 90]
DB_PATH = PROJECT_ROOT / "data" / "features.duckdb"

EVAL_ROWS_PATH = (
    PROJECT_ROOT
    / "evaluation_outputs"
    / "chapter7_tabular_lgb_real"
    / "monthly"
    / "baseline_tabular_lgb_monthly"
    / "eval_rows.parquet"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def compute_per_date_rankic(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dt, grp in df.groupby("as_of_date"):
        if len(grp) < 5:
            continue
        ic = stats.spearmanr(grp["score"], grp["excess_return"]).statistic
        top10 = grp.nlargest(10, "score")["excess_return"].mean() if len(grp) >= 10 else np.nan
        rows.append({"as_of_date": dt, "rankic": ic, "top10_er": top10, "n_stocks": len(grp)})
    return pd.DataFrame(rows)


def summarise_signal(per_date: pd.DataFrame, label: str = "") -> dict:
    ic = per_date["rankic"]
    return {
        "label": label,
        "n_dates": len(ic),
        "mean_rankic": round(ic.mean(), 4),
        "median_rankic": round(ic.median(), 4),
        "ic_stability": round(ic.mean() / ic.std(), 3) if ic.std() > 0 else 0,
        "pct_positive": round((ic > 0).mean(), 3),
        "top10_mean_er": round(per_date["top10_er"].mean(), 4),
    }


# ---------------------------------------------------------------------------
# Diagnostic 1: Retrain LGB on DEV-only, evaluate on FINAL-only
# ---------------------------------------------------------------------------

def diagnostic_1_retrain(features_df: pd.DataFrame) -> dict:
    """Train ONE model on all pre-2024 data, score 2024+ data."""
    logger.info("=== DIAGNOSTIC 1: Retrain LGB on DEV-only, eval on FINAL ===")

    train_df = features_df[features_df["date"] < pd.Timestamp(HOLDOUT_START)].copy()
    test_df = features_df[features_df["date"] >= pd.Timestamp(HOLDOUT_START)].copy()

    avail_feats = [f for f in DEFAULT_TABULAR_FEATURES if f in train_df.columns]
    logger.info(f"Train rows: {len(train_df):,}, Test rows: {len(test_df):,}, Features: {len(avail_feats)}")

    results = {}
    for h in HORIZONS:
        label_col = f"excess_return_{h}d"
        if label_col not in train_df.columns:
            logger.warning(f"Skipping {h}d — {label_col} not in columns")
            continue

        tr = train_df.dropna(subset=[label_col]).copy()
        te = test_df.dropna(subset=[label_col]).copy()
        logger.info(f"  {h}d — train: {len(tr):,} rows, test: {len(te):,} rows")

        model = train_lgbm_ranking_model(
            train_df=tr,
            feature_cols=avail_feats,
            label_col=label_col,
            date_col="date",
        )

        scores = predict_lgbm_scores(model, te, feature_cols=avail_feats)
        te = te.copy()
        te["score"] = scores
        te["excess_return"] = te[label_col]
        te["as_of_date"] = te["date"]

        per_date = compute_per_date_rankic(te)
        summary = summarise_signal(per_date, label=f"Retrain {h}d FINAL")
        results[h] = summary
        logger.info(f"  {h}d FINAL — MeanIC: {summary['mean_rankic']:.4f}, "
                     f"MedIC: {summary['median_rankic']:.4f}")

    return results


# ---------------------------------------------------------------------------
# Diagnostic 2: Feature importance stability
# ---------------------------------------------------------------------------

def diagnostic_2_feature_importance(features_df: pd.DataFrame) -> dict:
    """Train on 3 time windows, compare feature importances."""
    logger.info("=== DIAGNOSTIC 2: Feature importance stability ===")

    windows = {
        "early_2016_2019": (date(2016, 1, 1), date(2020, 1, 1)),
        "mid_2020_2021": (date(2020, 1, 1), date(2022, 1, 1)),
        "late_2022_2023": (date(2022, 1, 1), date(2024, 1, 1)),
    }

    avail_feats = [f for f in DEFAULT_TABULAR_FEATURES if f in features_df.columns]
    results = {}

    for h in HORIZONS:
        label_col = f"excess_return_{h}d"
        if label_col not in features_df.columns:
            continue

        imp_by_window = {}
        for win_name, (win_start, win_end) in windows.items():
            mask = (features_df["date"] >= pd.Timestamp(win_start)) & (
                features_df["date"] < pd.Timestamp(win_end)
            )
            tr = features_df[mask].dropna(subset=[label_col]).copy()
            if len(tr) < 100:
                logger.warning(f"  {h}d {win_name}: too few rows ({len(tr)})")
                continue

            model = train_lgbm_ranking_model(
                train_df=tr,
                feature_cols=avail_feats,
                label_col=label_col,
                date_col="date",
            )

            importances = dict(zip(model.feature_names_, model.feature_importances_))
            total = sum(importances.values()) or 1
            importances = {k: round(v / total, 4) for k, v in importances.items()}
            imp_by_window[win_name] = importances

        if len(imp_by_window) < 2:
            results[h] = {"error": "not enough windows"}
            continue

        all_feats_set = sorted(set().union(*[set(v.keys()) for v in imp_by_window.values()]))
        imp_df = pd.DataFrame(
            {w: {f: imp_by_window[w].get(f, 0) for f in all_feats_set} for w in imp_by_window}
        )

        rank_df = imp_df.rank(ascending=False)

        early_key = list(imp_by_window.keys())[0]
        late_key = list(imp_by_window.keys())[-1]
        early_ranks = rank_df[early_key]
        late_ranks = rank_df[late_key]
        rank_corr = stats.spearmanr(early_ranks, late_ranks).statistic

        top3_early = rank_df[early_key].nsmallest(3).index.tolist()
        top3_late = rank_df[late_key].nsmallest(3).index.tolist()

        results[h] = {
            "rank_correlation_early_vs_late": round(rank_corr, 3),
            "top3_early": top3_early,
            "top3_late": top3_late,
            "top3_overlap": len(set(top3_early) & set(top3_late)),
            "importance_by_window": {
                w: dict(sorted(v.items(), key=lambda x: -x[1])[:5])
                for w, v in imp_by_window.items()
            },
        }
        logger.info(
            f"  {h}d — rank corr (early vs late): {rank_corr:.3f}, "
            f"top3 overlap: {results[h]['top3_overlap']}/3"
        )

    return results


# ---------------------------------------------------------------------------
# Diagnostic 3: 20d deep-dive on 2024
# ---------------------------------------------------------------------------

def diagnostic_3_deep_dive_20d() -> dict:
    """Monthly RankIC and top-10 excess returns for 20d in 2024-2025."""
    logger.info("=== DIAGNOSTIC 3: 20d deep-dive on FINAL period ===")

    er = pd.read_parquet(EVAL_ROWS_PATH)
    er["as_of_date"] = pd.to_datetime(er["as_of_date"])

    final_mask = er["as_of_date"] >= pd.Timestamp(HOLDOUT_START)
    results = {}

    for h in HORIZONS:
        hr = er[(er["horizon"] == h) & final_mask]
        if hr.empty:
            continue

        per_date = compute_per_date_rankic(hr)
        per_date["ym"] = per_date["as_of_date"].dt.to_period("M")

        monthly = (
            per_date.groupby("ym")
            .agg(
                mean_rankic=("rankic", "mean"),
                median_rankic=("rankic", "median"),
                pct_positive=("rankic", lambda x: (x > 0).mean()),
                top10_mean_er=("top10_er", "mean"),
                n_dates=("rankic", "count"),
            )
            .reset_index()
        )
        monthly["ym"] = monthly["ym"].astype(str)

        results[h] = {
            "monthly": monthly.to_dict(orient="records"),
            "summary": summarise_signal(per_date, label=f"WalkFwd {h}d FINAL"),
        }

        logger.info(f"\n  {h}d monthly FINAL:")
        for _, row in monthly.iterrows():
            ic_str = f"{row['mean_rankic']:+.3f}"
            er_str = f"{row['top10_mean_er']:+.4f}" if not np.isnan(row["top10_mean_er"]) else "N/A"
            logger.info(f"    {row['ym']}  RankIC={ic_str}  Top10ER={er_str}  "
                         f"%Pos={row['pct_positive']:.0%}  n={int(row['n_dates'])}")

    return results


# ---------------------------------------------------------------------------
# Compare retrained vs walk-forward
# ---------------------------------------------------------------------------

def compare_retrained_vs_walkforward(retrain_results: dict) -> dict:
    """Load walk-forward FINAL metrics and compare with retrained model."""
    logger.info("=== COMPARISON: Retrained (single model) vs Walk-Forward (multi-fold) ===")

    er = pd.read_parquet(EVAL_ROWS_PATH)
    er["as_of_date"] = pd.to_datetime(er["as_of_date"])
    final = er[er["as_of_date"] >= pd.Timestamp(HOLDOUT_START)]

    comparison = {}
    for h in HORIZONS:
        hr = final[final["horizon"] == h]
        if hr.empty:
            continue
        wf_per_date = compute_per_date_rankic(hr)
        wf_summary = summarise_signal(wf_per_date, label=f"WalkFwd {h}d FINAL")

        retrain_summary = retrain_results.get(h, {})
        comparison[h] = {
            "walkforward_FINAL": wf_summary,
            "retrained_FINAL": retrain_summary,
            "delta_mean_rankic": round(
                retrain_summary.get("mean_rankic", 0) - wf_summary.get("mean_rankic", 0), 4
            ),
        }
        logger.info(
            f"  {h}d — WF MeanIC: {wf_summary.get('mean_rankic', 0):.4f}, "
            f"Retrained MeanIC: {retrain_summary.get('mean_rankic', 0):.4f}, "
            f"Delta: {comparison[h]['delta_mean_rankic']:+.4f}"
        )

    return comparison


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    output_dir = PROJECT_ROOT / "evaluation_outputs" / "chapter12" / "holdout_diagnostics"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading features from DuckDB...")
    features_df, meta = load_features_from_duckdb(DB_PATH)
    logger.info(f"Loaded {len(features_df):,} rows, {features_df['date'].nunique()} dates, "
                f"{features_df['ticker'].nunique()} tickers")

    # Diagnostic 1
    retrain_results = diagnostic_1_retrain(features_df)

    # Comparison
    comparison = compare_retrained_vs_walkforward(retrain_results)

    # Diagnostic 2
    feat_imp_results = diagnostic_2_feature_importance(features_df)

    # Diagnostic 3
    deep_dive_results = diagnostic_3_deep_dive_20d()

    # Assemble full report
    report = {
        "holdout_start": HOLDOUT_START.isoformat(),
        "diagnostic_1_retrain_on_dev": retrain_results,
        "comparison_retrained_vs_walkforward": comparison,
        "diagnostic_2_feature_importance_stability": feat_imp_results,
        "diagnostic_3_deep_dive_final": deep_dive_results,
    }

    out_path = output_dir / "holdout_diagnostics.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info(f"\nFull report saved to {out_path}")

    # Print summary
    print("\n" + "=" * 80)
    print("HOLDOUT DIAGNOSTICS SUMMARY")
    print("=" * 80)

    print("\n--- DIAGNOSTIC 1: Retrain on DEV-only, eval on FINAL ---")
    print(f"{'Hz':>5}  {'Retrained FINAL':>15}  {'WalkFwd FINAL':>14}  {'Delta':>7}  {'Verdict'}")
    print("-" * 65)
    for h in HORIZONS:
        c = comparison.get(h, {})
        rt = c.get("retrained_FINAL", {}).get("mean_rankic", 0)
        wf = c.get("walkforward_FINAL", {}).get("mean_rankic", 0)
        delta = c.get("delta_mean_rankic", 0)
        if rt > 0.02:
            verdict = "SIGNAL"
        elif rt > 0:
            verdict = "weak"
        else:
            verdict = "NO SIGNAL"
        print(f"  {h}d  {rt:>+14.4f}  {wf:>+13.4f}  {delta:>+6.4f}  {verdict}")

    print("\n--- DIAGNOSTIC 2: Feature importance stability ---")
    for h in HORIZONS:
        fi = feat_imp_results.get(h, {})
        rc = fi.get("rank_correlation_early_vs_late", 0)
        overlap = fi.get("top3_overlap", 0)
        t3e = fi.get("top3_early", [])
        t3l = fi.get("top3_late", [])
        if rc > 0.7:
            verdict = "STABLE"
        elif rc > 0.4:
            verdict = "moderate"
        else:
            verdict = "UNSTABLE"
        print(f"  {h}d — rank corr: {rc:.3f} ({verdict}), "
              f"top3 overlap: {overlap}/3")
        print(f"         early top3: {t3e}")
        print(f"         late  top3: {t3l}")

    print("\n--- DIAGNOSTIC 3: 20d deep-dive FINAL monthly ---")
    dd = deep_dive_results.get(20, {})
    monthly = dd.get("monthly", [])
    if monthly:
        pos_months = sum(1 for m in monthly if m["mean_rankic"] > 0)
        neg_months = sum(1 for m in monthly if m["mean_rankic"] <= 0)
        print(f"  Positive months: {pos_months}, Negative months: {neg_months}")
        for m in monthly:
            ic = m["mean_rankic"]
            er_val = m.get("top10_mean_er", float("nan"))
            er_str = f"{er_val:+.4f}" if not (isinstance(er_val, float) and np.isnan(er_val)) else "N/A"
            print(f"    {m['ym']}  RankIC={ic:+.3f}  Top10ER={er_str}")

    print("\n--- INTERPRETATION ---")
    rt_90 = comparison.get(90, {}).get("retrained_FINAL", {}).get("mean_rankic", 0)
    wf_90 = comparison.get(90, {}).get("walkforward_FINAL", {}).get("mean_rankic", 0)
    rt_20 = comparison.get(20, {}).get("retrained_FINAL", {}).get("mean_rankic", 0)

    if rt_90 < 0:
        print("  90d: Retrained model ALSO fails on 2024 → REGIME SHIFT (not just leakage)")
    else:
        print("  90d: Retrained model works on 2024 → WALK-FORWARD HAD LEAKAGE ISSUE")

    if rt_20 > 0:
        print("  20d: Retrained model positive → CONFIRMS selective horizon failure")
    else:
        print("  20d: Retrained model also fails → BROAD MODEL FAILURE")

    fi_90 = feat_imp_results.get(90, {}).get("rank_correlation_early_vs_late", 0)
    if fi_90 > 0.5:
        print(f"  Features: Stable (ρ={fi_90:.2f}) → model learned consistent patterns")
    else:
        print(f"  Features: Unstable (ρ={fi_90:.2f}) → model chased different noise each period")


if __name__ == "__main__":
    main()

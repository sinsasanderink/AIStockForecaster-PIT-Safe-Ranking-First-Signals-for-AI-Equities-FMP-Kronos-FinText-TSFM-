#!/usr/bin/env python
"""
Chapter 11.1 — Sub-Model Score Preparation
=============================================

Loads evaluation rows from LGB (Ch7), FinText (Ch9), and Sentiment (Ch10),
aligns them on (as_of_date, stable_id, horizon), and produces a single
DataFrame for score-level fusion.

Usage:
    python scripts/prepare_fusion_scores.py
    python scripts/prepare_fusion_scores.py --mode full
"""

import argparse
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

EVAL_ROW_PATHS = {
    "smoke": {
        "lgb": "evaluation_outputs/chapter7_tabular_lgb_full/monthly/baseline_tabular_lgb_monthly/eval_rows.parquet",
        "fintext": "evaluation_outputs/chapter9_fintext_small_smoke/chapter9_fintext_small_smoke/eval_rows.parquet",
        "sentiment": "evaluation_outputs/chapter10_sentiment_smoke/chapter10_sentiment_smoke/eval_rows.parquet",
    },
    "full": {
        "lgb": "evaluation_outputs/chapter7_tabular_lgb_real/monthly/baseline_tabular_lgb_monthly/eval_rows.parquet",
        "fintext": "evaluation_outputs/chapter9_fintext_small_full/chapter9_fintext_small_full/eval_rows.parquet",
        "sentiment": "evaluation_outputs/chapter10_sentiment_full/chapter10_sentiment_full/eval_rows.parquet",
    },
}

JOIN_KEYS = ["as_of_date", "stable_id", "horizon"]


def load_and_align(mode: str = "smoke") -> pd.DataFrame:
    """
    Load eval rows from all sub-models, align on (date, ticker, horizon).

    Returns DataFrame with columns:
        as_of_date, ticker, stable_id, fold_id, horizon, excess_return,
        lgb_score, fintext_score, sentiment_score
    """
    paths = EVAL_ROW_PATHS[mode]
    dfs = {}

    for name, path in paths.items():
        p = Path(path)
        if not p.exists():
            logger.warning(f"{name}: not found at {p}")
            continue
        df = pd.read_parquet(p)
        df = df.rename(columns={"score": f"{name}_score"})
        keep_cols = JOIN_KEYS + [f"{name}_score"]
        if name == "lgb":
            keep_cols += ["ticker", "fold_id", "excess_return"]
        dfs[name] = df[keep_cols].copy()

    if "lgb" not in dfs:
        raise FileNotFoundError("LGB eval rows required but not found")

    aligned = dfs["lgb"]
    for name in ["fintext", "sentiment"]:
        if name in dfs:
            aligned = aligned.merge(dfs[name], on=JOIN_KEYS, how="left")

    return aligned


def coverage_report(aligned: pd.DataFrame) -> dict:
    """Compute coverage statistics."""
    score_cols = [c for c in aligned.columns if c.endswith("_score")]
    n_total = len(aligned)
    report = {"total_rows": n_total}

    for col in score_cols:
        n_valid = aligned[col].notna().sum()
        report[f"{col}_coverage"] = n_valid / n_total

    all_valid = aligned[score_cols].notna().all(axis=1).sum()
    report["all_three_coverage"] = all_valid / n_total
    report["n_dates"] = aligned["as_of_date"].nunique()
    report["n_tickers"] = aligned["stable_id"].nunique()
    report["n_horizons"] = aligned["horizon"].nunique()
    return report


def correlation_report(aligned: pd.DataFrame) -> pd.DataFrame:
    """Compute per-horizon Spearman correlations between sub-model scores."""
    results = []
    score_cols = [c for c in aligned.columns if c.endswith("_score")]

    for horizon in sorted(aligned["horizon"].unique()):
        h_df = aligned[aligned["horizon"] == horizon]
        for i, col_a in enumerate(score_cols):
            for col_b in score_cols[i + 1 :]:
                valid = h_df[[col_a, col_b]].dropna()
                if len(valid) >= 10:
                    corr, pval = stats.spearmanr(valid[col_a], valid[col_b])
                else:
                    corr, pval = np.nan, np.nan
                results.append(
                    {
                        "horizon": horizon,
                        "pair": f"{col_a} vs {col_b}",
                        "spearman_rho": corr,
                        "p_value": pval,
                        "n_valid": len(valid),
                    }
                )
    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(
        description="Prepare aligned fusion scores"
    )
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument(
        "--output",
        default=None,
        help="Output parquet path",
    )
    args = parser.parse_args()

    output_path = Path(
        args.output or f"data/fusion_scores_{args.mode}.parquet"
    )

    logger.info("=" * 60)
    logger.info("CHAPTER 11.1: SUB-MODEL SCORE PREPARATION")
    logger.info("=" * 60)
    logger.info(f"Mode: {args.mode}")

    aligned = load_and_align(args.mode)
    logger.info(f"Aligned rows: {len(aligned):,}")

    # Coverage
    cov = coverage_report(aligned)
    logger.info("\n--- Coverage ---")
    for k, v in cov.items():
        if isinstance(v, float):
            logger.info(f"  {k}: {v:.1%}")
        else:
            logger.info(f"  {k}: {v}")

    # Correlations
    corr_df = correlation_report(aligned)
    logger.info("\n--- Score Correlations (Spearman) ---")
    for _, row in corr_df.iterrows():
        logger.info(
            f"  {row['horizon']}d {row['pair']}: "
            f"ρ={row['spearman_rho']:.3f} (p={row['p_value']:.4f}, n={row['n_valid']})"
        )

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    aligned.to_parquet(output_path, index=False)
    logger.info(f"\nSaved to {output_path}")

    # Save correlation report
    corr_path = output_path.with_name(f"fusion_correlations_{args.mode}.csv")
    corr_df.to_csv(corr_path, index=False)
    logger.info(f"Correlations saved to {corr_path}")


if __name__ == "__main__":
    main()

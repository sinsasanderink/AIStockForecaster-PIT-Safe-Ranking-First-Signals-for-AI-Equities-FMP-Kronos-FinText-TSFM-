#!/usr/bin/env python
"""
Chapter 11.2–11.3 — Fusion Walk-Forward Evaluation
=====================================================

Runs the walk-forward evaluation for all fusion variants:
  A: Rank-Average (parameter-free)
  B: Enriched LGB (tabular + sentiment, regression or lambdarank)
  C: Learned Stacking (meta-learner on sub-model scores)

Usage:
    python scripts/run_chapter11_fusion.py --mode smoke
    python scripts/run_chapter11_fusion.py --mode smoke --variant all
    python scripts/run_chapter11_fusion.py --mode full --variant enriched_lgb
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.run_evaluation import (
    run_experiment,
    ExperimentSpec,
    SMOKE_MODE,
    FULL_MODE,
)
from src.models.fusion_scorer import (
    rank_average_scores,
    make_enriched_lgb_scorer,
    make_enriched_xgb_scorer,
    train_stacking_meta,
    stacking_predict,
    DEFAULT_TABULAR_FEATURES,
    SENTIMENT_FEATURES,
    SCORE_MODELS,
)
from src.features.sentiment_features import SentimentFeatureGenerator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Globals
_sentiment_gen = None
_fusion_scores = None
_enriched_features_df = None


def setup_globals(sentiment_db: str, fusion_scores_path: str):
    """Initialize global state."""
    global _sentiment_gen, _fusion_scores
    logger.info("Loading sentiment generator...")
    _sentiment_gen = SentimentFeatureGenerator(db_path=sentiment_db)

    if Path(fusion_scores_path).exists():
        logger.info(f"Loading aligned fusion scores from {fusion_scores_path}")
        _fusion_scores = pd.read_parquet(fusion_scores_path)
        logger.info(f"  {len(_fusion_scores):,} aligned score rows")
    else:
        logger.warning(f"Fusion scores not found at {fusion_scores_path}")


def build_enriched_features_cache(
    features_df: pd.DataFrame,
    cache_path: Path,
) -> pd.DataFrame:
    """
    Build/load a single enriched feature matrix for Approach B.

    This avoids repeated per-fold enrichment of the full training history,
    which is expensive and can destabilize runtime.
    """
    global _sentiment_gen
    if cache_path.exists():
        logger.info(f"Loading enriched feature cache: {cache_path}")
        cached = pd.read_parquet(cache_path)
        if "date" in cached.columns:
            cached["date"] = pd.to_datetime(cached["date"])
        return cached

    logger.info("Building enriched feature cache (one-time)...")
    enriched = _sentiment_gen.enrich_features_df(features_df.copy())
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_parquet(cache_path, index=False)
    logger.info(f"Saved enriched feature cache: {cache_path}")
    return enriched


# ---------------------------------------------------------------------------
# Scoring functions for run_experiment
# ---------------------------------------------------------------------------


def rank_average_scoring_fn(
    features_df: pd.DataFrame,
    fold_id: str,
    horizon: int,
) -> pd.DataFrame:
    """Approach A: Rank-average of 3 sub-model scores."""
    global _fusion_scores
    if _fusion_scores is None:
        return pd.DataFrame()

    er_col = f"excess_return_{horizon}d"
    if er_col not in features_df.columns:
        er_col = "excess_return"

    dates = pd.to_datetime(features_df["date"])
    date_set = set(dates.dt.date if hasattr(dates.iloc[0], "date") else dates)

    h_scores = _fusion_scores[_fusion_scores["horizon"] == horizon].copy()
    h_scores["_date"] = pd.to_datetime(h_scores["as_of_date"])

    # Filter to this fold's dates
    fold_scores = h_scores[h_scores["_date"].isin(dates.values)]
    if fold_scores.empty:
        return pd.DataFrame()

    composite = rank_average_scores(fold_scores)
    fold_scores = fold_scores.copy()
    fold_scores["score"] = composite.values

    results = []
    for _, row in fold_scores.iterrows():
        er = row.get("excess_return")
        if pd.isna(er):
            continue
        results.append(
            {
                "as_of_date": row["as_of_date"],
                "ticker": row["ticker"],
                "stable_id": row["stable_id"],
                "fold_id": fold_id,
                "horizon": horizon,
                "score": float(row["score"]),
                "excess_return": float(er),
            }
        )
    return pd.DataFrame(results)


def rank_average_2model_scoring_fn(
    features_df: pd.DataFrame,
    fold_id: str,
    horizon: int,
) -> pd.DataFrame:
    """Approach A variant: Rank-average of LGB + FinText only."""
    global _fusion_scores
    if _fusion_scores is None:
        return pd.DataFrame()

    er_col = f"excess_return_{horizon}d"
    dates = pd.to_datetime(features_df["date"])
    h_scores = _fusion_scores[_fusion_scores["horizon"] == horizon].copy()
    h_scores["_date"] = pd.to_datetime(h_scores["as_of_date"])
    fold_scores = h_scores[h_scores["_date"].isin(dates.values)]
    if fold_scores.empty:
        return pd.DataFrame()

    composite = rank_average_scores(
        fold_scores, score_cols=["lgb_score", "fintext_score"]
    )
    fold_scores = fold_scores.copy()
    fold_scores["score"] = composite.values

    results = []
    for _, row in fold_scores.iterrows():
        er = row.get("excess_return")
        if pd.isna(er):
            continue
        results.append(
            {
                "as_of_date": row["as_of_date"],
                "ticker": row["ticker"],
                "stable_id": row["stable_id"],
                "fold_id": fold_id,
                "horizon": horizon,
                "score": float(row["score"]),
                "excess_return": float(er),
            }
        )
    return pd.DataFrame(results)


def _make_enriched_lgb_scorer(features_df, objective="regression"):
    """Create Approach B scorer with access to full features."""
    global _sentiment_gen
    return make_enriched_lgb_scorer(
        full_features_df=features_df,
        sentiment_gen=_sentiment_gen,
        objective=objective,
    )


def _make_enriched_xgb_scorer(features_df):
    """Create Approach B XGBoost scorer with access to full features."""
    global _sentiment_gen
    return make_enriched_xgb_scorer(
        full_features_df=features_df,
        sentiment_gen=_sentiment_gen,
    )


def learned_stacking_scoring_fn(
    features_df: pd.DataFrame,
    fold_id: str,
    horizon: int,
) -> pd.DataFrame:
    """Approach C: Learned stacking meta-learner."""
    global _fusion_scores
    if _fusion_scores is None:
        return pd.DataFrame()

    dates = pd.to_datetime(features_df["date"])
    h_scores = _fusion_scores[_fusion_scores["horizon"] == horizon].copy()
    h_scores["_date"] = pd.to_datetime(h_scores["as_of_date"])

    # Split: train on dates BEFORE this fold, predict on fold dates
    fold_dates = set(dates.values)
    train_scores = h_scores[~h_scores["_date"].isin(fold_dates)]
    val_scores = h_scores[h_scores["_date"].isin(fold_dates)]

    if train_scores.empty or val_scores.empty:
        # Fallback to rank average if insufficient training data
        return rank_average_scoring_fn(features_df, fold_id, horizon)

    score_cols = [c for c in SCORE_MODELS if c in h_scores.columns]
    meta = train_stacking_meta(
        train_scores, score_cols=score_cols, method="ridge"
    )
    preds = stacking_predict(meta, val_scores)
    val_scores = val_scores.copy()
    val_scores["score"] = preds

    results = []
    for _, row in val_scores.iterrows():
        er = row.get("excess_return")
        if pd.isna(er):
            continue
        results.append(
            {
                "as_of_date": row["as_of_date"],
                "ticker": row["ticker"],
                "stable_id": row["stable_id"],
                "fold_id": fold_id,
                "horizon": horizon,
                "score": float(row["score"]),
                "excess_return": float(er),
            }
        )
    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Feature loading
# ---------------------------------------------------------------------------


def load_features(db_path: str) -> pd.DataFrame:
    """Load features + labels from DuckDB."""
    import duckdb

    con = duckdb.connect(db_path, read_only=True)
    features_df = con.execute("SELECT * FROM features").df()
    labels_df = con.execute("SELECT * FROM labels").df()
    con.close()

    features_df["date"] = pd.to_datetime(features_df["date"])
    labels_df["as_of_date"] = pd.to_datetime(labels_df["as_of_date"])

    labels_wide = (
        labels_df.pivot_table(
            index=["as_of_date", "ticker"],
            columns="horizon",
            values="excess_return",
            aggfunc="first",
        )
        .reset_index()
    )
    labels_wide.columns.name = None
    labels_wide = labels_wide.rename(
        columns={20: "excess_return_20d", 60: "excess_return_60d", 90: "excess_return_90d"}
    )

    features_df = features_df.merge(
        labels_wide,
        left_on=["date", "ticker"],
        right_on=["as_of_date", "ticker"],
        how="left",
    )
    if "as_of_date" in features_df.columns:
        features_df.drop(columns=["as_of_date"], inplace=True)

    return features_df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

VARIANTS = {
    "rank_avg_3": {
        "name": "rank_avg_3",
        "desc": "Rank Average (3 models)",
        "approach": "A",
        "scorer": "rank_average_scoring_fn",
        "needs_train": False,
        "experimental": False,
    },
    "rank_avg_2": {
        "name": "rank_avg_2",
        "desc": "Rank Average (LGB + FinText)",
        "approach": "A",
        "scorer": "rank_average_2model_scoring_fn",
        "needs_train": False,
        "experimental": False,
    },
    "enriched_lgb": {
        "name": "enriched_lgb",
        "desc": "LGB + Sentiment (22 features, regression)",
        "approach": "B",
        "scorer": "enriched_lgb_regression",
        "needs_train": True,
        "experimental": True,
    },
    "enriched_lgb_rank": {
        "name": "enriched_lgb_rank",
        "desc": "LGB + Sentiment (22 features, LambdaRank)",
        "approach": "B",
        "scorer": "enriched_lgb_lambdarank",
        "needs_train": True,
        "experimental": True,
    },
    "enriched_xgb": {
        "name": "enriched_xgb",
        "desc": "XGB + Sentiment (22 features, larger tree model)",
        "approach": "B",
        "scorer": "enriched_xgb_regression",
        "needs_train": True,
        "experimental": True,
    },
    "learned_stacking": {
        "name": "learned_stacking",
        "desc": "Learned Stacking (Ridge, 3 scores)",
        "approach": "C",
        "scorer": "learned_stacking_scoring_fn",
        "needs_train": False,
        "experimental": False,
    },
}


def main():
    global _enriched_features_df
    parser = argparse.ArgumentParser(
        description="Chapter 11: Fusion Walk-Forward Evaluation"
    )
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument(
        "--variant",
        choices=list(VARIANTS.keys()) + ["all"],
        default="all",
    )
    parser.add_argument(
        "--include-experimental",
        action="store_true",
        help="Include experimental variants (enriched_lgb*).",
    )
    parser.add_argument("--db-path", default="data/features.duckdb")
    parser.add_argument("--sentiment-db", default="data/sentiment.duckdb")
    parser.add_argument(
        "--fusion-scores",
        default=None,
        help="Path to aligned fusion scores parquet",
    )
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    eval_mode = SMOKE_MODE if args.mode == "smoke" else FULL_MODE
    fusion_scores_path = args.fusion_scores or f"data/fusion_scores_{args.mode}.parquet"
    base_output = Path(args.output_dir or f"evaluation_outputs/chapter11_fusion_{args.mode}")

    logger.info("=" * 60)
    logger.info("CHAPTER 11: FUSION WALK-FORWARD EVALUATION")
    logger.info("=" * 60)

    # Load features for Approach B
    logger.info("Loading features...")
    features_df = load_features(args.db_path)
    logger.info(f"Features: {len(features_df):,} rows")

    # Setup globals
    setup_globals(args.sentiment_db, fusion_scores_path)

    # Determine which variants to run
    if args.variant == "all":
        variants_to_run = list(VARIANTS.keys())
        if not args.include_experimental:
            variants_to_run = [
                v for v in variants_to_run
                if not VARIANTS[v].get("experimental", False)
            ]
    else:
        variants_to_run = [args.variant]

    logger.info(f"Variants to run: {variants_to_run}")

    results_summary = []

    for variant_key in variants_to_run:
        variant = VARIANTS[variant_key]
        logger.info(f"\n{'='*60}")
        logger.info(f"VARIANT: {variant['desc']} (Approach {variant['approach']})")
        logger.info(f"{'='*60}")

        output_dir = base_output / variant_key
        spec = ExperimentSpec(
            name=f"ch11_{variant_key}_{args.mode}",
            model_type="model",
            model_name=f"fusion_{variant_key}",
            horizons=[20, 60, 90],
            cadence="monthly",
        )

        # Select scorer
        features_input = features_df
        if variant_key == "rank_avg_3":
            scorer = rank_average_scoring_fn
        elif variant_key == "rank_avg_2":
            scorer = rank_average_2model_scoring_fn
        elif variant_key == "enriched_lgb":
            if _enriched_features_df is None:
                cache_path = Path(f"data/enriched_features_{args.mode}.parquet")
                _enriched_features_df = build_enriched_features_cache(
                    features_df, cache_path
                )
            features_input = _enriched_features_df
            scorer = _make_enriched_lgb_scorer(features_input, "regression")
        elif variant_key == "enriched_lgb_rank":
            if _enriched_features_df is None:
                cache_path = Path(f"data/enriched_features_{args.mode}.parquet")
                _enriched_features_df = build_enriched_features_cache(
                    features_df, cache_path
                )
            features_input = _enriched_features_df
            scorer = _make_enriched_lgb_scorer(features_input, "lambdarank")
        elif variant_key == "enriched_xgb":
            if _enriched_features_df is None:
                cache_path = Path(f"data/enriched_features_{args.mode}.parquet")
                _enriched_features_df = build_enriched_features_cache(
                    features_df, cache_path
                )
            features_input = _enriched_features_df
            scorer = _make_enriched_xgb_scorer(features_input)
        elif variant_key == "learned_stacking":
            scorer = learned_stacking_scoring_fn
        else:
            continue

        try:
            result = run_experiment(
                experiment_spec=spec,
                features_df=features_input,
                output_dir=output_dir,
                mode=eval_mode,
                scorer_fn=scorer,
            )

            # Quick metrics
            if "eval_rows" in result and result["eval_rows"] is not None:
                er = result["eval_rows"]
                from scipy import stats as sp_stats
                for h in [20, 60, 90]:
                    h_rows = er[er["horizon"] == h]
                    if h_rows.empty:
                        continue
                    per_date = h_rows.groupby("as_of_date")[["score", "excess_return"]].apply(
                        lambda g: sp_stats.spearmanr(g["score"], g["excess_return"]).statistic
                        if len(g) >= 5 else np.nan
                    )
                    med_ic = per_date.dropna().median()
                    mean_ic = per_date.dropna().mean()
                    results_summary.append({
                        "variant": variant_key,
                        "horizon": h,
                        "mean_rankic": mean_ic,
                        "median_rankic": med_ic,
                        "n_dates": per_date.notna().sum(),
                    })
                    logger.info(
                        f"  {h}d: Mean RankIC={mean_ic:.4f}, Median={med_ic:.4f}"
                    )
        except Exception as e:
            logger.error(f"  FAILED: {e}")
            continue

    # Summary table
    if results_summary:
        summary_df = pd.DataFrame(results_summary)
        logger.info("\n" + "=" * 60)
        logger.info("ABLATION SUMMARY")
        logger.info("=" * 60)
        pivot = summary_df.pivot_table(
            index="variant", columns="horizon", values="mean_rankic"
        )
        logger.info(f"\n{pivot.to_string()}")

        summary_path = base_output / "ablation_summary.csv"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(summary_path, index=False)
        logger.info(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()

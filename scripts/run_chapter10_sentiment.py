#!/usr/bin/env python
"""
Chapter 10.4 — Walk-Forward Evaluation: Sentiment Signal
==========================================================

Runs the Chapter 6 walk-forward framework using sentiment features as
the ranking signal. Produces evaluation rows for gate checking and
orthogonality analysis.

Usage:
    # SMOKE mode (3 folds, fast)
    python scripts/run_chapter10_sentiment.py --mode smoke

    # FULL mode (all folds)
    python scripts/run_chapter10_sentiment.py --mode full
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
from src.features.sentiment_features import (
    SentimentFeatureGenerator,
    SENTIMENT_FEATURE_NAMES,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Global generator — loaded once, reused across folds
_sentiment_gen: SentimentFeatureGenerator = None

# Features used for composite sentiment score, in priority order.
SCORE_FEATURES = [
    "news_sentiment_30d",
    "news_sentiment_7d",
    "news_sentiment_momentum",
    "news_volume_30d",
    "filing_sentiment_latest",
    "filing_sentiment_90d",
    "sentiment_zscore",
]


def setup_sentiment_generator(sentiment_db: str):
    """Initialize the global sentiment feature generator once."""
    global _sentiment_gen
    logger.info("Loading sentiment data...")
    _sentiment_gen = SentimentFeatureGenerator(db_path=sentiment_db)
    summary = _sentiment_gen.get_summary()
    logger.info(
        f"Sentiment data: {summary['n_filings']:,} filings, "
        f"{summary['n_news']:,} news articles, "
        f"{summary['n_filing_tickers']} filing tickers, "
        f"{summary['n_news_tickers']} news tickers"
    )


def sentiment_scoring_function(
    features_df: pd.DataFrame,
    fold_id: str,
    horizon: int,
) -> pd.DataFrame:
    """
    Score all (date, ticker) pairs using a composite of sentiment features.

    Called by run_experiment for each fold/horizon. Enriches only the small
    per-fold features_df with sentiment features, then computes a composite
    rank-based score.

    Composite score approach:
    1. Extracts available sentiment features per ticker on each date
    2. Rank-normalizes each feature cross-sectionally (percentile rank)
    3. Averages the ranks into a single composite score in [0, 1]

    Tickers with zero available sentiment features receive a neutral 0.5.
    """
    # Enrich this fold's features with sentiment data (typically ~80 rows)
    enriched = _sentiment_gen.enrich_features_df(features_df)

    results = []

    er_col = f"excess_return_{horizon}d"
    if er_col not in enriched.columns:
        er_col = "excess_return"

    for asof_date, date_df in enriched.groupby("date"):
        rank_matrix = pd.DataFrame(index=date_df.index)
        available_features = []

        for feat in SCORE_FEATURES:
            if feat not in date_df.columns:
                continue
            col = date_df[feat]
            n_valid = col.notna().sum()
            if n_valid < 3:
                continue
            available_features.append(feat)
            rank_matrix[feat] = col.rank(pct=True)

        if available_features:
            composite = rank_matrix[available_features].mean(axis=1).fillna(0.5)
        else:
            composite = pd.Series(0.5, index=date_df.index)

        for idx, row in date_df.iterrows():
            er_val = row.get(er_col)
            if pd.isna(er_val):
                continue

            results.append(
                {
                    "as_of_date": asof_date,
                    "ticker": row["ticker"],
                    "stable_id": row["stable_id"],
                    "fold_id": fold_id,
                    "horizon": horizon,
                    "score": float(composite.loc[idx]),
                    "excess_return": float(er_val),
                }
            )

    return pd.DataFrame(results)


def load_features(db_path: str) -> pd.DataFrame:
    """Load features + labels from DuckDB, merge to wide format."""
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
        columns={
            20: "excess_return_20d",
            60: "excess_return_60d",
            90: "excess_return_90d",
        }
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


def main():
    parser = argparse.ArgumentParser(
        description="Chapter 10: Sentiment Walk-Forward Evaluation"
    )
    parser.add_argument(
        "--mode",
        choices=["smoke", "full"],
        default="smoke",
    )
    parser.add_argument(
        "--db-path",
        default="data/features.duckdb",
    )
    parser.add_argument(
        "--sentiment-db",
        default="data/sentiment.duckdb",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override output directory",
    )
    args = parser.parse_args()

    eval_mode = SMOKE_MODE if args.mode == "smoke" else FULL_MODE
    output_dir = Path(
        args.output_dir
        or f"evaluation_outputs/chapter10_sentiment_{args.mode}"
    )

    logger.info("=" * 60)
    logger.info("CHAPTER 10.4: SENTIMENT WALK-FORWARD EVALUATION")
    logger.info("=" * 60)
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Output: {output_dir}")

    # Load features
    logger.info("Loading features from DuckDB...")
    features_df = load_features(args.db_path)
    logger.info(
        f"Features: {len(features_df):,} rows, "
        f"{len(features_df.columns)} columns"
    )

    # Initialize sentiment generator (loaded once, reused per fold)
    setup_sentiment_generator(args.sentiment_db)

    # Run evaluation
    experiment_spec = ExperimentSpec(
        name=f"chapter10_sentiment_{args.mode}",
        model_type="model",
        model_name="sentiment_composite",
        horizons=[20, 60, 90],
        cadence="monthly",
    )

    logger.info("Starting walk-forward evaluation...")
    results = run_experiment(
        experiment_spec=experiment_spec,
        features_df=features_df,
        output_dir=output_dir,
        mode=eval_mode,
        scorer_fn=sentiment_scoring_function,
    )

    logger.info("=" * 60)
    logger.info("EVALUATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Output directory: {output_dir}")

    # Quick summary
    if "eval_rows" in results and results["eval_rows"] is not None:
        er = results["eval_rows"]
        logger.info(f"Total eval rows: {len(er):,}")
        for h in [20, 60, 90]:
            h_rows = er[er["horizon"] == h]
            if h_rows.empty:
                continue
            from scipy import stats
            per_date = h_rows.groupby("as_of_date")[["score", "excess_return"]].apply(
                lambda g: stats.spearmanr(g["score"], g["excess_return"]).statistic
                if len(g) >= 5 else np.nan
            )
            med_ic = per_date.dropna().median()
            mean_ic = per_date.dropna().mean()
            logger.info(
                f"  {h}d: Median RankIC={med_ic:.4f}, "
                f"Mean RankIC={mean_ic:.4f}, "
                f"N dates={per_date.notna().sum()}"
            )


if __name__ == "__main__":
    main()

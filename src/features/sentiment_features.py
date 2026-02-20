"""
Sentiment Feature Engineering — Chapter 10.3
==============================================

Computes 9 PIT-safe sentiment features from FinBERT-scored text records:

Filing Sentiment (3):
  filing_sentiment_latest     — Score of most recent 8-K filing
  filing_sentiment_change     — Change between last two filings
  filing_sentiment_90d        — Mean score of filings in past 90 calendar days

News Sentiment (4):
  news_sentiment_7d           — Mean score of news in past 7 calendar days
  news_sentiment_30d          — Mean score of news in past 30 calendar days
  news_sentiment_momentum     — 7d minus 30d (acceleration)
  news_volume_30d             — Count of articles in past 30 days (attention proxy)

Cross-Sectional (2):
  sentiment_zscore            — Z-score of news_sentiment_30d across universe
  sentiment_vs_momentum       — Residual: sentiment minus rank-matched momentum

All features use `event_date < asof_date` (strict) to enforce PIT safety.
"""

import logging
from datetime import date, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

SENTIMENT_FEATURE_NAMES = [
    "filing_sentiment_latest",
    "filing_sentiment_change",
    "filing_sentiment_90d",
    "news_sentiment_7d",
    "news_sentiment_30d",
    "news_sentiment_momentum",
    "news_volume_30d",
    "sentiment_zscore",
    "sentiment_vs_momentum",
]


class SentimentFeatureGenerator:
    """
    Computes sentiment features for the walk-forward evaluation pipeline.

    Preloads all scored text records into memory for fast lookups.
    All features are PIT-safe: only data with event_date < asof_date is used.
    """

    def __init__(
        self,
        db_path: str = "data/sentiment.duckdb",
        preload: bool = True,
    ):
        self._db_path = db_path
        self._filings: Optional[pd.DataFrame] = None
        self._news: Optional[pd.DataFrame] = None

        if preload:
            self._load_data()

    def _load_data(self):
        """Load all scored texts into memory, split by source."""
        import duckdb

        conn = duckdb.connect(self._db_path, read_only=True)

        self._filings = conn.execute("""
            SELECT ticker, event_date, sentiment_score
            FROM sentiment_texts
            WHERE scored = TRUE AND source = 'sec_8k'
            ORDER BY event_date ASC
        """).df()

        self._news = conn.execute("""
            SELECT ticker, event_date, sentiment_score
            FROM sentiment_texts
            WHERE scored = TRUE AND source = 'finnhub_news'
            ORDER BY event_date ASC
        """).df()

        conn.close()

        # Convert to date type for efficient comparison
        self._filings["event_date"] = pd.to_datetime(
            self._filings["event_date"]
        ).dt.date
        self._news["event_date"] = pd.to_datetime(
            self._news["event_date"]
        ).dt.date

        # Pre-index by ticker for fast lookups
        self._filings_by_ticker = {
            t: g.sort_values("event_date")
            for t, g in self._filings.groupby("ticker")
        }
        self._news_by_ticker = {
            t: g.sort_values("event_date")
            for t, g in self._news.groupby("ticker")
        }

        logger.info(
            f"Loaded {len(self._filings):,} filings, "
            f"{len(self._news):,} news articles"
        )

    def compute_ticker_features(
        self, ticker: str, asof_date: date
    ) -> Dict[str, float]:
        """
        Compute 7 per-ticker sentiment features (PIT-safe).

        Args:
            ticker: Stock ticker
            asof_date: Evaluation date (features use data strictly before this)

        Returns:
            Dict with 7 feature values (NaN for missing)
        """
        features = {}

        # ---- Filing features (3) ----
        filings = self._filings_by_ticker.get(ticker, pd.DataFrame())
        if not filings.empty:
            pit_filings = filings[filings["event_date"] < asof_date]
        else:
            pit_filings = pd.DataFrame()

        if not pit_filings.empty:
            scores = pit_filings["sentiment_score"].values
            dates = pit_filings["event_date"].values

            features["filing_sentiment_latest"] = float(scores[-1])

            if len(scores) >= 2:
                features["filing_sentiment_change"] = float(
                    scores[-1] - scores[-2]
                )
            else:
                features["filing_sentiment_change"] = 0.0

            cutoff_90 = asof_date - timedelta(days=90)
            mask_90 = pit_filings["event_date"] >= cutoff_90
            filings_90 = pit_filings.loc[mask_90, "sentiment_score"]
            if len(filings_90) > 0:
                features["filing_sentiment_90d"] = float(filings_90.mean())
            else:
                features["filing_sentiment_90d"] = float(scores[-1])
        else:
            features["filing_sentiment_latest"] = np.nan
            features["filing_sentiment_change"] = np.nan
            features["filing_sentiment_90d"] = np.nan

        # ---- News features (4) ----
        news = self._news_by_ticker.get(ticker, pd.DataFrame())
        if not news.empty:
            pit_news = news[news["event_date"] < asof_date]
        else:
            pit_news = pd.DataFrame()

        if not pit_news.empty:
            cutoff_7 = asof_date - timedelta(days=7)
            cutoff_30 = asof_date - timedelta(days=30)

            news_7d = pit_news.loc[
                pit_news["event_date"] >= cutoff_7, "sentiment_score"
            ]
            news_30d = pit_news.loc[
                pit_news["event_date"] >= cutoff_30, "sentiment_score"
            ]

            features["news_sentiment_7d"] = (
                float(news_7d.mean()) if len(news_7d) > 0 else np.nan
            )
            features["news_sentiment_30d"] = (
                float(news_30d.mean()) if len(news_30d) > 0 else np.nan
            )

            s7 = features["news_sentiment_7d"]
            s30 = features["news_sentiment_30d"]
            if np.isfinite(s7) and np.isfinite(s30):
                features["news_sentiment_momentum"] = s7 - s30
            else:
                features["news_sentiment_momentum"] = np.nan

            features["news_volume_30d"] = float(len(news_30d))
        else:
            features["news_sentiment_7d"] = np.nan
            features["news_sentiment_30d"] = np.nan
            features["news_sentiment_momentum"] = np.nan
            features["news_volume_30d"] = 0.0

        return features

    def compute_for_universe(
        self,
        tickers: List[str],
        asof_date: date,
        momentum_values: Optional[Dict[str, float]] = None,
    ) -> pd.DataFrame:
        """
        Compute all 9 sentiment features for a universe on a given date.

        Includes cross-sectional features (z-score, momentum residual).

        Args:
            tickers: List of stock tickers
            asof_date: Evaluation date
            momentum_values: Optional dict {ticker: momentum_score} for
                sentiment_vs_momentum residual. If None, uses rank-based proxy.

        Returns:
            DataFrame with columns: ticker, date, + 9 sentiment features
        """
        rows = []
        for ticker in tickers:
            feats = self.compute_ticker_features(ticker, asof_date)
            feats["ticker"] = ticker
            feats["date"] = asof_date
            rows.append(feats)

        df = pd.DataFrame(rows)

        # ---- Cross-sectional: sentiment_zscore ----
        s30 = df["news_sentiment_30d"]
        valid_mask = s30.notna()
        if valid_mask.sum() >= 3:
            mean = s30[valid_mask].mean()
            std = s30[valid_mask].std()
            if std > 1e-8:
                df.loc[valid_mask, "sentiment_zscore"] = (
                    (s30[valid_mask] - mean) / std
                )
            else:
                df["sentiment_zscore"] = 0.0
        else:
            df["sentiment_zscore"] = np.nan

        df.loc[~valid_mask, "sentiment_zscore"] = np.nan

        # ---- Cross-sectional: sentiment_vs_momentum ----
        if momentum_values is not None:
            mom_series = df["ticker"].map(momentum_values)
        else:
            mom_series = pd.Series(np.nan, index=df.index)

        both_valid = s30.notna() & mom_series.notna()
        if both_valid.sum() >= 3:
            # Rank-based residual: rank(sentiment) - rank(momentum)
            sent_rank = s30[both_valid].rank(pct=True)
            mom_rank = mom_series[both_valid].rank(pct=True)
            df.loc[both_valid, "sentiment_vs_momentum"] = (
                sent_rank.values - mom_rank.values
            )
        else:
            df["sentiment_vs_momentum"] = np.nan

        df.loc[~both_valid, "sentiment_vs_momentum"] = np.nan

        return df

    def enrich_features_df(
        self, features_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Add sentiment features to an existing features DataFrame.

        This is the main integration point for the evaluation pipeline.
        Computes sentiment features for every unique (date, ticker) and
        merges them into features_df.

        Args:
            features_df: DataFrame with 'date', 'ticker' columns and
                optionally 'mom_12m' or 'momentum_composite' for
                sentiment_vs_momentum.

        Returns:
            features_df with 9 new sentiment feature columns added.
        """
        if features_df.empty:
            for col in SENTIMENT_FEATURE_NAMES:
                features_df[col] = np.nan
            return features_df

        # Normalize date column
        date_col = features_df["date"]
        if hasattr(date_col.iloc[0], "date") and callable(
            date_col.iloc[0].date
        ):
            dates = date_col.dt.date
        else:
            dates = date_col

        unique_dates = sorted(dates.unique())
        tickers_per_date = {
            d: features_df.loc[dates == d, "ticker"].unique().tolist()
            for d in unique_dates
        }

        # Check for momentum column for sentiment_vs_momentum
        mom_col = None
        for candidate in [
            "momentum_composite",
            "mom_12m",
            "momentum_composite_monthly",
        ]:
            if candidate in features_df.columns:
                mom_col = candidate
                break

        all_sentiment = []
        for d in unique_dates:
            tickers = tickers_per_date[d]

            # Get momentum values if available
            mom_dict = None
            if mom_col:
                date_mask = dates == d
                mom_data = features_df.loc[date_mask, ["ticker", mom_col]]
                mom_dict = dict(
                    zip(mom_data["ticker"], mom_data[mom_col])
                )

            sent_df = self.compute_for_universe(
                tickers, d, momentum_values=mom_dict
            )
            all_sentiment.append(sent_df)

        sentiment_all = pd.concat(all_sentiment, ignore_index=True)

        # Merge on (ticker, date)
        features_df = features_df.copy()
        features_df["_merge_date"] = dates.values

        result = features_df.merge(
            sentiment_all,
            left_on=["ticker", "_merge_date"],
            right_on=["ticker", "date"],
            how="left",
            suffixes=("", "_sent"),
        )
        result.drop(
            columns=["_merge_date", "date_sent"],
            errors="ignore",
            inplace=True,
        )

        # Ensure all sentiment columns exist
        for col in SENTIMENT_FEATURE_NAMES:
            if col not in result.columns:
                result[col] = np.nan

        return result

    def get_feature_names(self) -> List[str]:
        """Return list of sentiment feature column names."""
        return list(SENTIMENT_FEATURE_NAMES)

    def get_summary(self) -> Dict[str, int]:
        """Return counts of loaded data."""
        return {
            "n_filings": len(self._filings) if self._filings is not None else 0,
            "n_news": len(self._news) if self._news is not None else 0,
            "n_filing_tickers": (
                len(self._filings_by_ticker) if self._filings_by_ticker else 0
            ),
            "n_news_tickers": (
                len(self._news_by_ticker) if self._news_by_ticker else 0
            ),
        }

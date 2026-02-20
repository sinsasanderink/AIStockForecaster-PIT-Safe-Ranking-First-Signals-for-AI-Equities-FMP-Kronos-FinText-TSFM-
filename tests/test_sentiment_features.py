"""
Tests for Chapter 10.3 — Sentiment Feature Engineering
========================================================

Tests cover:
1. Per-ticker filing features (3)
2. Per-ticker news features (4)
3. Cross-sectional features (2)
4. PIT safety (strict < asof_date)
5. Edge cases (missing data, single record, empty universe)
6. enrich_features_df integration
7. Real data integration (if sentiment DB exists)
"""

import os
import tempfile
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import pytest
import pytz

UTC = pytz.UTC


# ---------------------------------------------------------------------------
# Fixtures: create a temp DuckDB with known sentiment data
# ---------------------------------------------------------------------------


def _create_test_db(path: str):
    """Build a small sentiment DB with controlled data for deterministic tests."""
    import duckdb

    conn = duckdb.connect(path)
    conn.execute("""
        CREATE TABLE sentiment_texts (
            record_id VARCHAR PRIMARY KEY,
            ticker VARCHAR NOT NULL,
            source VARCHAR NOT NULL,
            text VARCHAR NOT NULL,
            observed_at TIMESTAMPTZ NOT NULL,
            event_date DATE NOT NULL,
            metadata JSON,
            scored BOOLEAN DEFAULT FALSE,
            sentiment_score FLOAT DEFAULT NULL,
            created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # --- SEC 8-K filings for NVDA ---
    filings = [
        ("f1", "NVDA", "sec_8k", "Q1 results strong",
         datetime(2024, 5, 10, 21, 0, tzinfo=UTC), date(2024, 5, 10), 0.7),
        ("f2", "NVDA", "sec_8k", "Q2 results record",
         datetime(2024, 8, 10, 21, 0, tzinfo=UTC), date(2024, 8, 10), 0.9),
        ("f3", "NVDA", "sec_8k", "Q3 results missed",
         datetime(2024, 11, 15, 21, 0, tzinfo=UTC), date(2024, 11, 15), -0.3),
    ]

    # --- SEC 8-K filings for AAPL ---
    filings += [
        ("f4", "AAPL", "sec_8k", "Q2 results ok",
         datetime(2024, 7, 20, 21, 0, tzinfo=UTC), date(2024, 7, 20), 0.1),
    ]

    # --- FinnHub news for NVDA ---
    news = [
        # Old news (before 30d window)
        ("n1", "NVDA", "finnhub_news", "Old NVDA news",
         datetime(2024, 10, 1, 12, 0, tzinfo=UTC), date(2024, 10, 1), 0.5),
        # Within 30d window
        ("n2", "NVDA", "finnhub_news", "NVDA demand strong",
         datetime(2024, 11, 5, 10, 0, tzinfo=UTC), date(2024, 11, 5), 0.8),
        ("n3", "NVDA", "finnhub_news", "NVDA chip shortage",
         datetime(2024, 11, 10, 14, 0, tzinfo=UTC), date(2024, 11, 10), -0.4),
        # Within 7d window
        ("n4", "NVDA", "finnhub_news", "NVDA AI rally",
         datetime(2024, 11, 18, 9, 0, tzinfo=UTC), date(2024, 11, 18), 0.6),
        ("n5", "NVDA", "finnhub_news", "NVDA beat estimates",
         datetime(2024, 11, 19, 22, 0, tzinfo=UTC), date(2024, 11, 19), 0.9),
    ]

    # --- FinnHub news for AMD ---
    news += [
        ("n6", "AMD", "finnhub_news", "AMD gaming weak",
         datetime(2024, 11, 5, 10, 0, tzinfo=UTC), date(2024, 11, 5), -0.6),
        ("n7", "AMD", "finnhub_news", "AMD AI chips coming",
         datetime(2024, 11, 15, 14, 0, tzinfo=UTC), date(2024, 11, 15), 0.3),
        ("n8", "AMD", "finnhub_news", "AMD results",
         datetime(2024, 11, 19, 20, 0, tzinfo=UTC), date(2024, 11, 19), 0.1),
    ]

    # --- FinnHub news for MSFT ---
    news += [
        ("n9", "MSFT", "finnhub_news", "MSFT cloud strong",
         datetime(2024, 11, 10, 10, 0, tzinfo=UTC), date(2024, 11, 10), 0.7),
        ("n10", "MSFT", "finnhub_news", "MSFT copilot revenue",
         datetime(2024, 11, 18, 10, 0, tzinfo=UTC), date(2024, 11, 18), 0.5),
    ]

    for rid, ticker, source, text, observed, edate, score in filings + news:
        conn.execute(
            """INSERT INTO sentiment_texts
               (record_id, ticker, source, text, observed_at, event_date,
                scored, sentiment_score)
               VALUES (?, ?, ?, ?, ?, ?, TRUE, ?)""",
            [rid, ticker, source, text, observed, edate, score],
        )

    conn.close()


@pytest.fixture
def test_db():
    """Create temp sentiment DB with controlled test data."""
    tmpdir = tempfile.mkdtemp()
    path = os.path.join(tmpdir, "test_sentiment.duckdb")
    _create_test_db(path)
    yield path


@pytest.fixture
def generator(test_db):
    """Create SentimentFeatureGenerator with test data."""
    from src.features.sentiment_features import SentimentFeatureGenerator
    return SentimentFeatureGenerator(db_path=test_db)


# Evaluation date: 2024-11-20 (all data before this is visible)
ASOF = date(2024, 11, 20)


# ===========================================================================
# Filing Feature Tests
# ===========================================================================


class TestFilingFeatures:
    """Test the 3 filing sentiment features."""

    def test_filing_sentiment_latest(self, generator):
        """Latest filing score for NVDA should be Q3 = -0.3."""
        feats = generator.compute_ticker_features("NVDA", ASOF)
        assert feats["filing_sentiment_latest"] == pytest.approx(-0.3)

    def test_filing_sentiment_change(self, generator):
        """Change should be Q3 - Q2 = -0.3 - 0.9 = -1.2."""
        feats = generator.compute_ticker_features("NVDA", ASOF)
        assert feats["filing_sentiment_change"] == pytest.approx(-1.2)

    def test_filing_sentiment_90d(self, generator):
        """90d mean: Q2 (Aug 10) is within 90d, Q3 (Nov 15) too.
        Mean = (0.9 + -0.3) / 2 = 0.3."""
        feats = generator.compute_ticker_features("NVDA", ASOF)
        # 90 days before Nov 20 = Aug 22. Q2 is Aug 10 (outside).
        # Q3 is Nov 15 (inside). So only Q3.
        assert feats["filing_sentiment_90d"] == pytest.approx(-0.3)

    def test_filing_single_record(self, generator):
        """AAPL has one filing → change should be 0."""
        feats = generator.compute_ticker_features("AAPL", ASOF)
        assert feats["filing_sentiment_latest"] == pytest.approx(0.1)
        assert feats["filing_sentiment_change"] == pytest.approx(0.0)

    def test_filing_no_filings(self, generator):
        """AMD has no filings → all filing features are NaN."""
        feats = generator.compute_ticker_features("AMD", ASOF)
        assert np.isnan(feats["filing_sentiment_latest"])
        assert np.isnan(feats["filing_sentiment_change"])
        assert np.isnan(feats["filing_sentiment_90d"])


# ===========================================================================
# News Feature Tests
# ===========================================================================


class TestNewsFeatures:
    """Test the 4 news sentiment features."""

    def test_news_sentiment_7d(self, generator):
        """7d window (Nov 13-19): n4 (0.6) + n5 (0.9) → mean = 0.75."""
        feats = generator.compute_ticker_features("NVDA", ASOF)
        assert feats["news_sentiment_7d"] == pytest.approx(0.75)

    def test_news_sentiment_30d(self, generator):
        """30d window (Oct 21-Nov 19): n2 (0.8), n3 (-0.4), n4 (0.6), n5 (0.9).
        Mean = (0.8 - 0.4 + 0.6 + 0.9) / 4 = 0.475."""
        feats = generator.compute_ticker_features("NVDA", ASOF)
        assert feats["news_sentiment_30d"] == pytest.approx(0.475)

    def test_news_sentiment_momentum(self, generator):
        """Momentum = 7d - 30d = 0.75 - 0.475 = 0.275."""
        feats = generator.compute_ticker_features("NVDA", ASOF)
        assert feats["news_sentiment_momentum"] == pytest.approx(0.275)

    def test_news_volume_30d(self, generator):
        """NVDA has 4 articles in 30d window."""
        feats = generator.compute_ticker_features("NVDA", ASOF)
        assert feats["news_volume_30d"] == 4.0

    def test_news_no_recent(self, generator):
        """With asof before all news → NaN."""
        feats = generator.compute_ticker_features("NVDA", date(2024, 1, 1))
        assert np.isnan(feats["news_sentiment_7d"])
        assert np.isnan(feats["news_sentiment_30d"])
        assert feats["news_volume_30d"] == 0.0

    def test_amd_news(self, generator):
        """AMD: 30d has n6 (-0.6), n7 (0.3), n8 (0.1) → mean = -0.0667."""
        feats = generator.compute_ticker_features("AMD", ASOF)
        expected_30d = (-0.6 + 0.3 + 0.1) / 3
        assert feats["news_sentiment_30d"] == pytest.approx(expected_30d, abs=0.01)
        assert feats["news_volume_30d"] == 3.0


# ===========================================================================
# PIT Safety Tests
# ===========================================================================


class TestPITSafety:
    """Verify strict PIT enforcement: event_date < asof_date."""

    def test_same_day_excluded(self, generator):
        """Data on asof_date itself must NOT be included (strict <)."""
        # NVDA filing on Nov 15, asof = Nov 15 → should NOT include it
        feats = generator.compute_ticker_features("NVDA", date(2024, 11, 15))
        # Latest should be Q2 (Aug 10), not Q3 (Nov 15)
        assert feats["filing_sentiment_latest"] == pytest.approx(0.9)

    def test_day_after_included(self, generator):
        """Data one day before asof should be included."""
        # NVDA filing Nov 15, asof = Nov 16 → should include Q3
        feats = generator.compute_ticker_features("NVDA", date(2024, 11, 16))
        assert feats["filing_sentiment_latest"] == pytest.approx(-0.3)

    def test_news_pit_boundary(self, generator):
        """News on Nov 19, asof = Nov 19 → NOT included."""
        feats = generator.compute_ticker_features("NVDA", date(2024, 11, 19))
        # n5 (Nov 19) should be excluded, n4 (Nov 18) included
        # 7d window: Nov 12-18 → n4 only → 0.6
        assert feats["news_sentiment_7d"] == pytest.approx(0.6)


# ===========================================================================
# Cross-Sectional Feature Tests
# ===========================================================================


class TestCrossSectionalFeatures:
    """Test the 2 cross-sectional features."""

    def test_sentiment_zscore(self, generator):
        """Z-scores should have mean ≈ 0 and std ≈ 1 across universe."""
        df = generator.compute_for_universe(
            ["NVDA", "AMD", "MSFT"], ASOF
        )
        zscores = df["sentiment_zscore"].dropna()
        if len(zscores) >= 3:
            assert abs(zscores.mean()) < 0.01
            assert abs(zscores.std(ddof=0) - 1.0) < 0.3

    def test_zscore_direction(self, generator):
        """NVDA (positive news) should have higher z-score than AMD (negative)."""
        df = generator.compute_for_universe(
            ["NVDA", "AMD", "MSFT"], ASOF
        )
        nvda_z = df.loc[df["ticker"] == "NVDA", "sentiment_zscore"].iloc[0]
        amd_z = df.loc[df["ticker"] == "AMD", "sentiment_zscore"].iloc[0]
        assert nvda_z > amd_z

    def test_sentiment_vs_momentum(self, generator):
        """With momentum values, should compute rank residual."""
        mom = {"NVDA": 0.25, "AMD": -0.10, "MSFT": 0.15}
        df = generator.compute_for_universe(
            ["NVDA", "AMD", "MSFT"], ASOF, momentum_values=mom
        )
        residuals = df["sentiment_vs_momentum"].dropna()
        assert len(residuals) == 3
        # Residuals should be bounded roughly in [-1, 1]
        assert residuals.min() >= -1.1
        assert residuals.max() <= 1.1

    def test_sentiment_vs_momentum_no_data(self, generator):
        """Without momentum, sentiment_vs_momentum should be NaN."""
        df = generator.compute_for_universe(
            ["NVDA", "AMD", "MSFT"], ASOF, momentum_values=None
        )
        assert df["sentiment_vs_momentum"].isna().all()


# ===========================================================================
# Edge Cases
# ===========================================================================


class TestEdgeCases:
    """Edge cases: unknown tickers, empty data, single ticker."""

    def test_unknown_ticker(self, generator):
        """Unknown ticker should have all NaN features."""
        feats = generator.compute_ticker_features("ZZZZ", ASOF)
        assert np.isnan(feats["filing_sentiment_latest"])
        assert np.isnan(feats["news_sentiment_7d"])
        assert feats["news_volume_30d"] == 0.0

    def test_single_ticker_universe(self, generator):
        """Single ticker universe should still compute z-score."""
        df = generator.compute_for_universe(["NVDA"], ASOF)
        assert len(df) == 1
        # Z-score with single value: can't compute (NaN or 0)
        assert df["sentiment_zscore"].iloc[0] == 0.0 or np.isnan(
            df["sentiment_zscore"].iloc[0]
        )

    def test_all_features_present(self, generator):
        """All 9 feature columns should be present in output."""
        from src.features.sentiment_features import SENTIMENT_FEATURE_NAMES

        df = generator.compute_for_universe(
            ["NVDA", "AMD", "MSFT"], ASOF
        )
        for col in SENTIMENT_FEATURE_NAMES:
            assert col in df.columns, f"Missing column: {col}"

    def test_feature_names_list(self, generator):
        """get_feature_names should return 9 names."""
        names = generator.get_feature_names()
        assert len(names) == 9


# ===========================================================================
# enrich_features_df Integration Tests
# ===========================================================================


class TestEnrichFeaturesDF:
    """Test the main integration point for evaluation pipeline."""

    def _make_features_df(self):
        """Create a minimal features_df matching evaluation format."""
        rows = []
        for ticker in ["NVDA", "AMD", "MSFT"]:
            rows.append({
                "date": ASOF,
                "ticker": ticker,
                "stable_id": f"STB_{ticker}",
                "excess_return_20d": 0.05,
                "mom_12m": {"NVDA": 0.3, "AMD": -0.1, "MSFT": 0.2}[ticker],
            })
        return pd.DataFrame(rows)

    def test_enrichment_adds_columns(self, generator):
        """Should add 9 sentiment columns to features_df."""
        from src.features.sentiment_features import SENTIMENT_FEATURE_NAMES

        df = self._make_features_df()
        result = generator.enrich_features_df(df)

        for col in SENTIMENT_FEATURE_NAMES:
            assert col in result.columns

    def test_enrichment_preserves_rows(self, generator):
        """Should not add or remove rows."""
        df = self._make_features_df()
        result = generator.enrich_features_df(df)
        assert len(result) == len(df)

    def test_enrichment_preserves_columns(self, generator):
        """Original columns should still be present."""
        df = self._make_features_df()
        result = generator.enrich_features_df(df)
        for col in ["date", "ticker", "stable_id", "excess_return_20d"]:
            assert col in result.columns

    def test_enrichment_uses_momentum(self, generator):
        """With mom_12m present, sentiment_vs_momentum should be computed."""
        df = self._make_features_df()
        result = generator.enrich_features_df(df)
        residuals = result["sentiment_vs_momentum"]
        assert not residuals.isna().all()

    def test_enrichment_empty_df(self, generator):
        """Empty features_df should return empty with sentiment columns."""
        from src.features.sentiment_features import SENTIMENT_FEATURE_NAMES

        df = pd.DataFrame(columns=["date", "ticker", "stable_id"])
        result = generator.enrich_features_df(df)
        assert len(result) == 0
        for col in SENTIMENT_FEATURE_NAMES:
            assert col in result.columns

    def test_enrichment_multiple_dates(self, generator):
        """Should handle features_df with multiple dates."""
        rows = []
        for d in [date(2024, 11, 10), date(2024, 11, 20)]:
            for ticker in ["NVDA", "AMD"]:
                rows.append({
                    "date": d,
                    "ticker": ticker,
                    "stable_id": f"STB_{ticker}",
                    "excess_return_20d": 0.05,
                })
        df = pd.DataFrame(rows)
        result = generator.enrich_features_df(df)
        assert len(result) == 4

        # Nov 10 NVDA should have less news than Nov 20 NVDA
        nvda_10 = result[
            (result["ticker"] == "NVDA")
            & (result["date"] == date(2024, 11, 10))
        ]
        nvda_20 = result[
            (result["ticker"] == "NVDA")
            & (result["date"] == date(2024, 11, 20))
        ]
        assert nvda_10["news_volume_30d"].iloc[0] <= nvda_20["news_volume_30d"].iloc[0]


# ===========================================================================
# Summary & Data Loading Tests
# ===========================================================================


class TestDataLoading:
    """Test data loading and summary."""

    def test_summary(self, generator):
        """Summary should report correct counts."""
        summary = generator.get_summary()
        assert summary["n_filings"] == 4  # 3 NVDA + 1 AAPL
        assert summary["n_news"] == 10
        assert summary["n_filing_tickers"] == 2  # NVDA, AAPL
        assert summary["n_news_tickers"] == 3  # NVDA, AMD, MSFT

    def test_preload(self, test_db):
        """Preloading should work without errors."""
        from src.features.sentiment_features import SentimentFeatureGenerator
        gen = SentimentFeatureGenerator(db_path=test_db, preload=True)
        assert gen._filings is not None
        assert gen._news is not None


# ===========================================================================
# Real Data Integration Test (requires sentiment.duckdb)
# ===========================================================================


@pytest.mark.skipif(
    not os.path.exists("data/sentiment.duckdb"),
    reason="No sentiment DB found",
)
class TestRealData:
    """Integration tests with real scored sentiment data."""

    @pytest.fixture
    def real_generator(self):
        from src.features.sentiment_features import SentimentFeatureGenerator
        return SentimentFeatureGenerator(db_path="data/sentiment.duckdb")

    def test_real_data_loads(self, real_generator):
        summary = real_generator.get_summary()
        assert summary["n_news"] > 50000
        assert summary["n_filings"] > 1000

    def test_real_features_computed(self, real_generator):
        """Compute features for NVDA on a recent date."""
        feats = real_generator.compute_ticker_features(
            "NVDA", date(2025, 12, 1)
        )
        assert np.isfinite(feats["news_sentiment_30d"])
        assert feats["news_volume_30d"] > 0

    def test_real_universe_features(self, real_generator):
        """Compute for a small universe."""
        from src.features.sentiment_features import SENTIMENT_FEATURE_NAMES

        tickers = ["NVDA", "AAPL", "MSFT", "AMD", "GOOGL"]
        df = real_generator.compute_for_universe(
            tickers, date(2025, 12, 1)
        )
        assert len(df) == 5
        for col in SENTIMENT_FEATURE_NAMES:
            assert col in df.columns

        # Z-scores should vary
        zscores = df["sentiment_zscore"].dropna()
        assert len(zscores) >= 4

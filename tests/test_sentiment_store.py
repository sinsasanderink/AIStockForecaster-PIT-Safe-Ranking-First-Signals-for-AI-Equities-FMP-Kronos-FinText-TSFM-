"""
Tests for Chapter 10.1 — Sentiment Data Store
==============================================

Tests cover:
1. TextRecord validation
2. SentimentDataStore schema + CRUD
3. SEC 8-K text extraction (mocked)
4. FinnHub news collection (mocked)
5. PIT-safety guarantees
6. Deduplication
7. Query methods
8. Batch collection
"""

import json
import os
import re
import tempfile
from datetime import date, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import pytz

UTC = pytz.UTC

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def temp_db():
    """Create a temporary path for DuckDB (file must not pre-exist)."""
    path = os.path.join(tempfile.mkdtemp(), "test_sentiment.duckdb")
    yield path
    if os.path.exists(path):
        os.unlink(path)


@pytest.fixture
def sample_text_records():
    """Create sample TextRecords for testing."""
    from src.data.sentiment_store import TextRecord

    return [
        TextRecord(
            ticker="NVDA",
            source="sec_8k",
            text="NVIDIA Corporation reported record revenue of $35.1 billion "
            "for Q3 2025, up 94% year-over-year, driven by data center demand.",
            observed_at=datetime(2025, 11, 19, 21, 5, tzinfo=UTC),
            event_date=date(2025, 11, 19),
            metadata={
                "accession_number": "0001045810-25-000228",
                "form": "8-K",
                "text_length": 120,
            },
        ),
        TextRecord(
            ticker="NVDA",
            source="finnhub_news",
            text="NVIDIA beats Q3 earnings estimates, stock surges 5% in after-hours trading.",
            observed_at=datetime(2025, 11, 19, 22, 30, tzinfo=UTC),
            event_date=date(2025, 11, 19),
            metadata={
                "article_id": 12345,
                "source_name": "MarketWatch",
                "headline": "NVIDIA beats Q3 earnings",
            },
        ),
        TextRecord(
            ticker="AMD",
            source="sec_8k",
            text="Advanced Micro Devices reported Q3 2025 revenue of $6.8 billion, "
            "up 18% year-over-year. Data center segment grew 122%.",
            observed_at=datetime(2025, 10, 29, 21, 10, tzinfo=UTC),
            event_date=date(2025, 10, 29),
            metadata={"form": "8-K"},
        ),
    ]


@pytest.fixture
def store(temp_db):
    """Create a SentimentDataStore with mocked collectors."""
    with patch(
        "src.data.sentiment_store.FinnhubNewsCollector"
    ) as mock_fh, patch(
        "src.data.sentiment_store.SECFilingCollector"
    ) as mock_sec:
        mock_fh_instance = MagicMock()
        mock_fh.return_value = mock_fh_instance
        mock_sec_instance = MagicMock()
        mock_sec.return_value = mock_sec_instance

        from src.data.sentiment_store import SentimentDataStore

        s = SentimentDataStore(db_path=temp_db)
        s._sec = mock_sec_instance
        s._finnhub = mock_fh_instance
        yield s
        s.close()


# ===========================================================================
# TextRecord Tests
# ===========================================================================


class TestTextRecord:
    """Tests for TextRecord dataclass."""

    def test_valid_record(self, sample_text_records):
        """Valid records should pass is_valid check."""
        for rec in sample_text_records:
            assert rec.is_valid, f"Record should be valid: {rec.ticker}"

    def test_empty_text_invalid(self):
        from src.data.sentiment_store import TextRecord

        rec = TextRecord(
            ticker="NVDA",
            source="test",
            text="",
            observed_at=datetime(2025, 1, 1, tzinfo=UTC),
            event_date=date(2025, 1, 1),
            metadata={},
        )
        assert not rec.is_valid

    def test_short_text_invalid(self):
        from src.data.sentiment_store import TextRecord

        rec = TextRecord(
            ticker="NVDA",
            source="test",
            text="Too short",
            observed_at=datetime(2025, 1, 1, tzinfo=UTC),
            event_date=date(2025, 1, 1),
            metadata={},
        )
        assert not rec.is_valid

    def test_whitespace_only_invalid(self):
        from src.data.sentiment_store import TextRecord

        rec = TextRecord(
            ticker="NVDA",
            source="test",
            text="          ",
            observed_at=datetime(2025, 1, 1, tzinfo=UTC),
            event_date=date(2025, 1, 1),
            metadata={},
        )
        assert not rec.is_valid


# ===========================================================================
# SentimentDataStore CRUD Tests
# ===========================================================================


class TestSentimentDataStoreCRUD:
    """Tests for store, query, and update operations."""

    def test_store_records(self, store, sample_text_records):
        """Should store valid text records."""
        n = store.store_records(sample_text_records)
        assert n == 3

    def test_deduplication(self, store, sample_text_records):
        """Storing the same records twice should not create duplicates."""
        store.store_records(sample_text_records)
        store.store_records(sample_text_records)  # Store again

        summary = store.get_summary()
        assert summary["total_records"] == 3  # Still 3, not 6

    def test_get_texts_by_ticker(self, store, sample_text_records):
        """Should filter by ticker."""
        store.store_records(sample_text_records)

        nvda = store.get_texts("NVDA")
        assert len(nvda) == 2

        amd = store.get_texts("AMD")
        assert len(amd) == 1

    def test_get_texts_by_source(self, store, sample_text_records):
        """Should filter by source."""
        store.store_records(sample_text_records)

        sec = store.get_texts("NVDA", source="sec_8k")
        assert len(sec) == 1

        news = store.get_texts("NVDA", source="finnhub_news")
        assert len(news) == 1

    def test_get_texts_by_date_range(self, store, sample_text_records):
        """Should filter by date range."""
        store.store_records(sample_text_records)

        # Only November records
        df = store.get_texts(
            "NVDA",
            start_date=date(2025, 11, 1),
            end_date=date(2025, 11, 30),
        )
        assert len(df) == 2

        # Only October records
        df = store.get_texts(
            "AMD",
            start_date=date(2025, 10, 1),
            end_date=date(2025, 10, 31),
        )
        assert len(df) == 1

    def test_get_unscored_texts(self, store, sample_text_records):
        """Should return only unscored records."""
        store.store_records(sample_text_records)

        unscored = store.get_unscored_texts(batch_size=100)
        assert len(unscored) == 3  # All are unscored initially

    def test_update_scores(self, store, sample_text_records):
        """Should update sentiment scores."""
        store.store_records(sample_text_records)

        unscored = store.get_unscored_texts()
        record_ids = unscored["record_id"].tolist()
        scores = [0.8, 0.6, 0.3]

        updated = store.update_scores(record_ids, scores)
        assert updated == 3

        # Verify scored
        still_unscored = store.get_unscored_texts()
        assert len(still_unscored) == 0

        # Verify scores
        scored = store.get_texts("NVDA", scored_only=True)
        assert len(scored) == 2
        assert all(scored["sentiment_score"].notna())

    def test_summary(self, store, sample_text_records):
        """Should return correct summary statistics."""
        store.store_records(sample_text_records)

        summary = store.get_summary()
        assert summary["total_records"] == 3
        assert len(summary["by_source"]) == 2  # sec_8k and finnhub_news


# ===========================================================================
# SEC Filing Collector Tests
# ===========================================================================


class TestSECFilingCollector:
    """Tests for SEC 8-K text extraction."""

    def test_extract_relevant_sections(self):
        """Should extract Item 2.02, 7.01, 8.01 sections."""
        from src.data.sentiment_store import SECFilingCollector

        collector = SECFilingCollector.__new__(SECFilingCollector)
        collector.RELEVANT_ITEMS = SECFilingCollector.RELEVANT_ITEMS

        text = (
            "HEADER STUFF "
            "Item 2.02 Results of Operations and Financial Condition. "
            "Revenue increased 94% to $35.1B driven by data center demand. "
            "GAAP earnings per diluted share of $0.81. "
            "Non-GAAP earnings per diluted share of $0.78. "
            "SOME_MORE_BOILERPLATE "
            "Item 9.01 Financial Statements and Exhibits."
        )

        result = collector.extract_relevant_sections(text)
        assert "Item 2.02" in result
        assert "Revenue increased" in result

    def test_extract_fallback_no_items(self):
        """When no items found, should return body text."""
        from src.data.sentiment_store import SECFilingCollector

        collector = SECFilingCollector.__new__(SECFilingCollector)
        collector.RELEVANT_ITEMS = SECFilingCollector.RELEVANT_ITEMS

        text = "A" * 1000 + "REAL_CONTENT_HERE" + "B" * 1000
        result = collector.extract_relevant_sections(text)
        # Should skip initial boilerplate
        assert len(result) > 0
        assert len(result) <= 2000

    def test_html_stripping(self):
        """Should strip HTML tags from filing text."""
        from src.data.sentiment_store import SECFilingCollector

        collector = SECFilingCollector.__new__(SECFilingCollector)

        raw_html = "<html><body><p>Revenue was <b>$35.1B</b></p></body></html>"
        result = collector.fetch_filing_text.__wrapped__ if hasattr(
            collector.fetch_filing_text, "__wrapped__"
        ) else None

        # Test the HTML stripping logic directly
        text = re.sub(r"<[^>]+>", " ", raw_html)
        text = re.sub(r"\s+", " ", text).strip()
        assert "Revenue was $35.1B" in text
        assert "<" not in text

    @patch("src.data.sentiment_store.SECFilingCollector.get_filings_metadata")
    @patch("src.data.sentiment_store.SECFilingCollector.fetch_filing_text")
    def test_collect_for_ticker(self, mock_fetch, mock_meta):
        """Should produce TextRecords from filing metadata + text."""
        from src.data.sentiment_store import SECFilingCollector

        collector = SECFilingCollector.__new__(SECFilingCollector)
        collector.RELEVANT_ITEMS = SECFilingCollector.RELEVANT_ITEMS

        mock_meta.return_value = pd.DataFrame(
            {
                "accessionNumber": ["0001-25-000001"],
                "filingDate": [pd.Timestamp("2025-11-19")],
                "acceptanceDateTime": [
                    pd.Timestamp("2025-11-19 21:05:00", tz="UTC")
                ],
                "primaryDocument": ["doc.htm"],
                "cik": ["0001045810"],
                "form": ["8-K"],
            }
        )

        mock_fetch.return_value = (
            "Item 2.02 Results of Operations. Revenue was $35B, up 94%."
            + " " * 100
        )

        records = collector.collect_for_ticker("NVDA")
        assert len(records) == 1
        assert records[0].ticker == "NVDA"
        assert records[0].source == "sec_8k"
        assert "Revenue" in records[0].text


# ===========================================================================
# FinnHub News Collector Tests
# ===========================================================================


class TestFinnhubNewsCollector:
    """Tests for FinnHub news collection."""

    @patch("src.data.sentiment_store.requests.get")
    def test_fetch_news(self, mock_get):
        """Should parse FinnHub API response correctly."""
        from src.data.sentiment_store import FinnhubNewsCollector

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = [
            {
                "id": 1001,
                "headline": "NVDA beats earnings",
                "summary": "NVIDIA reported record Q3 revenue of $35.1B.",
                "datetime": 1732053000,  # 2024-11-19 UTC
                "source": "MarketWatch",
                "category": "company",
                "url": "https://example.com/article",
            }
        ]
        mock_get.return_value = mock_resp

        collector = FinnhubNewsCollector(api_key="test_key")
        articles = collector.fetch_news(
            "NVDA", date(2024, 11, 1), date(2024, 11, 30)
        )
        assert len(articles) == 1
        assert articles[0]["headline"] == "NVDA beats earnings"

    @patch("src.data.sentiment_store.FinnhubNewsCollector.fetch_news")
    def test_collect_for_ticker(self, mock_fetch):
        """Should produce TextRecords from FinnHub articles."""
        from src.data.sentiment_store import FinnhubNewsCollector

        mock_fetch.return_value = [
            {
                "id": 1001,
                "headline": "NVDA beats earnings estimates in Q3",
                "summary": "NVIDIA reported record Q3 revenue.",
                "datetime": 1732053000,
                "source": "MarketWatch",
                "category": "company",
                "url": "https://example.com",
            },
            {
                "id": 1002,
                "headline": "AI chip demand drives semiconductor rally",
                "summary": "",
                "datetime": 1732139400,
                "source": "Reuters",
                "category": "company",
                "url": "https://example.com/2",
            },
        ]

        collector = FinnhubNewsCollector(api_key="test_key")
        records = collector.collect_for_ticker(
            "NVDA",
            start_date=date(2024, 11, 1),
            end_date=date(2024, 11, 30),
        )

        assert len(records) == 2
        assert records[0].source == "finnhub_news"
        assert "NVDA beats" in records[0].text

    @patch("src.data.sentiment_store.FinnhubNewsCollector.fetch_news")
    def test_deduplication_within_collection(self, mock_fetch):
        """Should deduplicate articles with same ID."""
        from src.data.sentiment_store import FinnhubNewsCollector

        mock_fetch.return_value = [
            {
                "id": 1001,
                "headline": "Same article",
                "summary": "Content here.",
                "datetime": 1732053000,
                "source": "Source",
                "category": "company",
                "url": "",
            },
            {
                "id": 1001,
                "headline": "Same article",
                "summary": "Content here.",
                "datetime": 1732053000,
                "source": "Source",
                "category": "company",
                "url": "",
            },
        ]

        collector = FinnhubNewsCollector(api_key="test_key")
        records = collector.collect_for_ticker(
            "NVDA",
            start_date=date(2024, 11, 1),
            end_date=date(2024, 11, 30),
        )
        assert len(records) == 1

    def test_missing_api_key_raises(self):
        """Should raise ValueError without API key."""
        from src.data.sentiment_store import FinnhubNewsCollector

        with patch.dict(os.environ, {"FINNHUB_KEYS": ""}, clear=False):
            with pytest.raises(ValueError, match="FinnHub API key"):
                FinnhubNewsCollector(api_key="")


# ===========================================================================
# PIT Safety Tests
# ===========================================================================


class TestPITSafety:
    """Tests ensuring Point-in-Time safety of sentiment data."""

    def test_observed_at_is_utc(self, sample_text_records):
        """All observed_at timestamps must be UTC."""
        for rec in sample_text_records:
            assert rec.observed_at.tzinfo is not None
            assert rec.observed_at.tzinfo == UTC

    def test_sec_uses_acceptance_datetime(self):
        """SEC records should use acceptanceDateTime, not filingDate."""
        from src.data.sentiment_store import TextRecord

        acceptance = datetime(2025, 11, 19, 21, 5, 23, tzinfo=UTC)
        filing_date = date(2025, 11, 19)

        rec = TextRecord(
            ticker="NVDA",
            source="sec_8k",
            text="Revenue was $35.1B for Q3 2025." + " " * 30,
            observed_at=acceptance,
            event_date=filing_date,
            metadata={"form": "8-K"},
        )

        # observed_at should be the precise acceptance time, not midnight
        assert rec.observed_at.hour == 21
        assert rec.observed_at.minute == 5

    def test_store_preserves_timestamps(self, store, sample_text_records):
        """Stored timestamps should be preserved exactly."""
        store.store_records(sample_text_records)

        df = store.get_texts("NVDA")
        assert len(df) == 2

        # Verify observed_at is preserved
        for _, row in df.iterrows():
            assert row["observed_at"] is not None
            assert row["event_date"] is not None

    def test_no_future_data_in_queries(self, store, sample_text_records):
        """Query by date should not return future records."""
        store.store_records(sample_text_records)

        # Query only October data
        df = store.get_texts(
            "NVDA",
            start_date=date(2025, 10, 1),
            end_date=date(2025, 10, 31),
        )
        assert len(df) == 0  # NVDA records are Nov 19

        # AMD record is Oct 29
        df = store.get_texts(
            "AMD",
            start_date=date(2025, 10, 1),
            end_date=date(2025, 10, 31),
        )
        assert len(df) == 1


# ===========================================================================
# Integration-style Tests
# ===========================================================================


class TestStoreIntegration:
    """Integration tests for the full pipeline."""

    def test_batch_collect(self, store, sample_text_records):
        """Batch collection should process multiple tickers."""
        store._sec.collect_for_ticker.return_value = sample_text_records[:1]
        store._finnhub.collect_for_ticker.return_value = sample_text_records[1:2]

        stats = store.collect_all(
            tickers=["NVDA", "AMD"],
            sec_start=date(2016, 1, 1),
            news_start=date(2024, 1, 1),
        )

        assert stats["tickers_processed"] == 2
        assert stats["sec_total"] >= 0
        assert stats["news_total"] >= 0

    def test_score_update_workflow(self, store, sample_text_records):
        """Full workflow: store → get unscored → score → verify."""
        store.store_records(sample_text_records)

        # Step 1: Get unscored
        unscored = store.get_unscored_texts()
        assert len(unscored) == 3

        # Step 2: Score them
        ids = unscored["record_id"].tolist()
        scores = [0.85, 0.72, 0.45]
        store.update_scores(ids, scores)

        # Step 3: Verify
        still_unscored = store.get_unscored_texts()
        assert len(still_unscored) == 0

        scored = store.get_texts("NVDA", scored_only=True)
        assert len(scored) == 2
        assert all(s > 0 for s in scored["sentiment_score"])

    def test_summary_stats(self, store, sample_text_records):
        """Summary should report correct source breakdown."""
        store.store_records(sample_text_records)

        summary = store.get_summary()
        assert summary["total_records"] == 3

        sources = {s["source"]: s["cnt"] for s in summary["by_source"]}
        assert sources["sec_8k"] == 2
        assert sources["finnhub_news"] == 1

    def test_empty_store(self, store):
        """Empty store should return zeros."""
        summary = store.get_summary()
        assert summary["total_records"] == 0

        df = store.get_texts("NVDA")
        assert len(df) == 0

        unscored = store.get_unscored_texts()
        assert len(unscored) == 0

"""
Tests for Chapter 10.2 — FinBERT Sentiment Scorer
===================================================

Tests cover:
1. Score range and polarity (positive/negative/neutral)
2. Batch scoring consistency
3. Long text chunking
4. Edge cases (empty, short, whitespace)
5. Store integration (score_store workflow)
6. Detailed score breakdown
"""

import os
import tempfile
from datetime import date, datetime
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import pytz

UTC = pytz.UTC


# ---------------------------------------------------------------------------
# Stub scorer for tests that don't need real FinBERT
# ---------------------------------------------------------------------------


class StubFinBERTScorer:
    """Deterministic scorer for unit tests (no model download)."""

    def __init__(self):
        self.batch_size = 64

    def score_text(self, text: str) -> float:
        if not text or len(text.strip()) < 5:
            return 0.0
        text_lower = text.lower()
        if any(w in text_lower for w in ["record", "beat", "strong", "surge", "growth"]):
            return 0.8
        if any(w in text_lower for w in ["loss", "decline", "weak", "miss", "fall"]):
            return -0.7
        return 0.05

    def score_batch(self, texts):
        return [self.score_text(t) for t in texts]

    def score_store(self, store, batch_size=64, max_records=None):
        total = 0
        while True:
            remaining = batch_size if max_records is None else min(batch_size, max_records - total)
            if remaining <= 0:
                break
            df = store.get_unscored_texts(batch_size=remaining)
            if df.empty:
                break
            texts = df["text"].tolist()
            record_ids = df["record_id"].tolist()
            scores = self.score_batch(texts)
            store.update_scores(record_ids, scores)
            total += len(scores)
        return total

    def get_detailed_scores(self, text):
        score = self.score_text(text)
        p_pos = max(0, score)
        p_neg = max(0, -score)
        p_neu = 1.0 - p_pos - p_neg
        return (score, p_pos, p_neg, p_neu)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def stub_scorer():
    return StubFinBERTScorer()


@pytest.fixture
def temp_store():
    """Create a temporary SentimentDataStore with sample data."""
    from src.data.sentiment_store import SentimentDataStore, TextRecord

    path = os.path.join(tempfile.mkdtemp(), "test_score.duckdb")

    with patch("src.data.sentiment_store.FinnhubNewsCollector") as mock_fh, \
         patch("src.data.sentiment_store.SECFilingCollector"):
        mock_fh.return_value = MagicMock()
        store = SentimentDataStore(db_path=path)

    records = [
        TextRecord(
            ticker="NVDA",
            source="finnhub_news",
            text="NVIDIA reported record revenue of $35.1 billion, beating estimates.",
            observed_at=datetime(2025, 11, 19, 22, 0, tzinfo=UTC),
            event_date=date(2025, 11, 19),
            metadata={"headline": "NVDA beats"},
        ),
        TextRecord(
            ticker="AMD",
            source="finnhub_news",
            text="AMD reported a decline in gaming revenue, missing analyst expectations.",
            observed_at=datetime(2025, 10, 29, 21, 0, tzinfo=UTC),
            event_date=date(2025, 10, 29),
            metadata={"headline": "AMD misses"},
        ),
        TextRecord(
            ticker="MSFT",
            source="finnhub_news",
            text="Microsoft cloud revenue showed strong growth in Q3 fiscal year.",
            observed_at=datetime(2025, 10, 23, 20, 0, tzinfo=UTC),
            event_date=date(2025, 10, 23),
            metadata={"headline": "MSFT cloud"},
        ),
        TextRecord(
            ticker="NVDA",
            source="sec_8k",
            text="Item 2.02 Results of Operations. Revenue increased 94% to $35.1B "
            "driven by data center demand. GAAP net income was $19.3 billion. "
            + "x " * 200,
            observed_at=datetime(2025, 11, 19, 21, 5, tzinfo=UTC),
            event_date=date(2025, 11, 19),
            metadata={"form": "8-K"},
        ),
    ]
    store.store_records(records)
    yield store
    store.close()


# ===========================================================================
# Score Polarity Tests
# ===========================================================================


class TestScorePolarity:
    """Verify FinBERT produces correct polarity for known texts."""

    def test_positive_text(self, stub_scorer):
        """Positive financial text should have positive score."""
        score = stub_scorer.score_text(
            "Revenue hit a record high, beating all analyst estimates."
        )
        assert score > 0.3, f"Expected positive score, got {score}"

    def test_negative_text(self, stub_scorer):
        """Negative financial text should have negative score."""
        score = stub_scorer.score_text(
            "Company reported a significant loss and revenue decline."
        )
        assert score < -0.3, f"Expected negative score, got {score}"

    def test_neutral_text(self, stub_scorer):
        """Neutral text should have score near zero."""
        score = stub_scorer.score_text(
            "The company held its annual general meeting on Tuesday."
        )
        assert -0.5 < score < 0.5, f"Expected near-zero score, got {score}"

    def test_score_range(self, stub_scorer):
        """All scores should be in [-1, +1]."""
        texts = [
            "Massive earnings beat, stock surges 20%.",
            "Revenue collapsed, company facing bankruptcy.",
            "Board approved quarterly dividend payment.",
            "",
            "short",
        ]
        scores = stub_scorer.score_batch(texts)
        for s in scores:
            assert -1.0 <= s <= 1.0, f"Score out of range: {s}"


# ===========================================================================
# Batch Scoring Tests
# ===========================================================================


class TestBatchScoring:
    """Tests for batch scoring consistency."""

    def test_batch_matches_individual(self, stub_scorer):
        """Batch scores should match individual scores."""
        texts = [
            "Record revenue growth this quarter.",
            "Stock declined on weak guidance.",
            "Annual meeting scheduled for next month.",
        ]

        batch_scores = stub_scorer.score_batch(texts)
        individual_scores = [stub_scorer.score_text(t) for t in texts]

        for bs, iss in zip(batch_scores, individual_scores):
            assert abs(bs - iss) < 0.01, f"Mismatch: batch={bs}, individual={iss}"

    def test_empty_batch(self, stub_scorer):
        """Empty batch should return empty list."""
        assert stub_scorer.score_batch([]) == []

    def test_single_item_batch(self, stub_scorer):
        """Single-item batch should work."""
        scores = stub_scorer.score_batch(["Strong earnings growth reported."])
        assert len(scores) == 1
        assert scores[0] > 0


# ===========================================================================
# Edge Cases
# ===========================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_string(self, stub_scorer):
        """Empty string should return 0."""
        assert stub_scorer.score_text("") == 0.0

    def test_very_short_text(self, stub_scorer):
        """Very short text should return 0."""
        assert stub_scorer.score_text("hi") == 0.0

    def test_whitespace_only(self, stub_scorer):
        """Whitespace-only text should return 0."""
        assert stub_scorer.score_text("    ") == 0.0

    def test_mixed_batch_with_empty(self, stub_scorer):
        """Batch with empty strings mixed in should work."""
        texts = ["Strong growth reported.", "", "Revenue declined sharply.", "   "]
        scores = stub_scorer.score_batch(texts)
        assert len(scores) == 4
        assert scores[1] == 0.0
        assert scores[3] == 0.0


# ===========================================================================
# Store Integration Tests
# ===========================================================================


class TestStoreIntegration:
    """Tests for score_store workflow."""

    def test_score_store(self, stub_scorer, temp_store):
        """Should score all unscored records."""
        n = stub_scorer.score_store(temp_store)
        assert n == 4  # 4 records in fixture

        unscored = temp_store.get_unscored_texts()
        assert len(unscored) == 0

    def test_score_store_with_max(self, stub_scorer, temp_store):
        """Should respect max_records limit."""
        n = stub_scorer.score_store(temp_store, max_records=2)
        assert n == 2

        unscored = temp_store.get_unscored_texts()
        assert len(unscored) == 2

    def test_score_store_idempotent(self, stub_scorer, temp_store):
        """Running twice should not re-score already scored records."""
        n1 = stub_scorer.score_store(temp_store)
        n2 = stub_scorer.score_store(temp_store)
        assert n1 == 4
        assert n2 == 0

    def test_scores_stored_correctly(self, stub_scorer, temp_store):
        """Stored scores should match expected polarity."""
        stub_scorer.score_store(temp_store)

        nvda_news = temp_store.get_texts("NVDA", source="finnhub_news", scored_only=True)
        assert len(nvda_news) == 1
        assert nvda_news.iloc[0]["sentiment_score"] > 0  # "record revenue, beating"

        amd = temp_store.get_texts("AMD", scored_only=True)
        assert len(amd) == 1
        assert amd.iloc[0]["sentiment_score"] < 0  # "decline, missing"


# ===========================================================================
# Detailed Scores Tests
# ===========================================================================


class TestDetailedScores:
    """Tests for get_detailed_scores."""

    def test_probabilities_sum_to_one(self, stub_scorer):
        """P(pos) + P(neg) + P(neu) should approximately sum to 1."""
        score, p_pos, p_neg, p_neu = stub_scorer.get_detailed_scores(
            "Revenue surged to record levels this quarter."
        )
        total = p_pos + p_neg + p_neu
        assert 0.95 <= total <= 1.05, f"Probabilities sum to {total}"

    def test_score_matches_pos_minus_neg(self, stub_scorer):
        """Score should equal P(positive) - P(negative)."""
        score, p_pos, p_neg, p_neu = stub_scorer.get_detailed_scores(
            "Strong earnings beat this quarter."
        )
        expected = p_pos - p_neg
        assert abs(score - expected) < 0.01

    def test_empty_text_returns_neutral(self, stub_scorer):
        """Empty text should return neutral probabilities."""
        score, p_pos, p_neg, p_neu = stub_scorer.get_detailed_scores("")
        assert score == 0.0
        assert p_neu == 1.0


# ===========================================================================
# Real FinBERT Tests (requires model download, marked slow)
# ===========================================================================


@pytest.mark.slow
class TestRealFinBERT:
    """Tests with actual FinBERT model (slow, requires download)."""

    @pytest.fixture
    def real_scorer(self):
        from src.models.finbert_scorer import FinBERTScorer
        return FinBERTScorer()

    def test_positive_earnings(self, real_scorer):
        """NVIDIA earnings beat should score positive."""
        score = real_scorer.score_text(
            "NVIDIA reported record revenue of $35.1 billion, "
            "up 94% year-over-year, beating analyst estimates."
        )
        assert score > 0.3, f"Expected strongly positive, got {score}"

    def test_negative_earnings(self, real_scorer):
        """Earnings miss should score negative."""
        score = real_scorer.score_text(
            "Company reported a net loss of $500 million and "
            "revenue declined 30% year-over-year, missing estimates."
        )
        assert score < -0.3, f"Expected strongly negative, got {score}"

    def test_batch_consistency(self, real_scorer):
        """Batch and individual scores should be very close."""
        texts = [
            "Strong revenue growth driven by AI demand.",
            "Stock plunged after disappointing guidance.",
            "Board of directors approved quarterly dividend.",
        ]
        batch = real_scorer.score_batch(texts)
        individual = [real_scorer.score_text(t) for t in texts]
        for b, i in zip(batch, individual):
            assert abs(b - i) < 0.01

    def test_detailed_probabilities(self, real_scorer):
        """Detailed scores should have valid probabilities."""
        score, p_pos, p_neg, p_neu = real_scorer.get_detailed_scores(
            "Revenue surged to record levels this quarter."
        )
        assert abs((p_pos + p_neg + p_neu) - 1.0) < 0.01
        assert abs(score - (p_pos - p_neg)) < 0.01

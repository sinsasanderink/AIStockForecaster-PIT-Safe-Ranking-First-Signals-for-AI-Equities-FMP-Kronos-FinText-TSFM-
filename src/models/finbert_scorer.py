"""
FinBERT Sentiment Scorer — Chapter 10.2
========================================

Scores financial text using ProsusAI/finbert (110M params, zero-shot).

Output: sentiment_score = P(positive) - P(negative) ∈ [-1, +1]

Texts are pre-truncated to 400 chars and tokenized to max 128 tokens for
speed. FinBERT was designed for short financial text (headlines, brief
passages); the first ~100 tokens capture the dominant sentiment for longer
documents (SEC filings, earnings summaries).

Usage:
    scorer = FinBERTScorer()

    # Score a single text
    score = scorer.score_text("Revenue increased 94% to $35.1B.")
    # → 0.91

    # Score a batch
    scores = scorer.score_batch(["text1", "text2", ...])
    # → [0.91, -0.45, ...]

    # Score all unscored records in SentimentDataStore
    n = scorer.score_store(store, batch_size=64)
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)

MODEL_NAME = "ProsusAI/finbert"
MAX_TOKEN_LENGTH = 128
MAX_CHAR_LENGTH = 400


@dataclass
class FinBERTScorer:
    """
    Wraps ProsusAI/finbert for batch sentiment scoring.

    Attributes:
        model_name: HuggingFace model name
        device: 'cpu' or 'cuda'
        batch_size: batch size for inference
    """

    model_name: str = MODEL_NAME
    device: str = "cpu"
    batch_size: int = 64

    _model: object = field(default=None, init=False, repr=False)
    _tokenizer: object = field(default=None, init=False, repr=False)

    def _ensure_loaded(self):
        """Lazy-load model on first use."""
        if self._model is not None:
            return

        from transformers import (
            AutoTokenizer,
            BertForSequenceClassification,
        )

        logger.info(f"Loading {self.model_name}...")
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = BertForSequenceClassification.from_pretrained(
            self.model_name, use_safetensors=True
        )
        self._model.to(self.device)
        self._model.eval()

        n_params = sum(p.numel() for p in self._model.parameters())
        logger.info(
            f"FinBERT loaded: {n_params/1e6:.0f}M params, "
            f"labels={self._model.config.id2label}, device={self.device}"
        )

    def _score_short_batch(self, texts: List[str]) -> np.ndarray:
        """
        Score a batch of texts with pre-truncation for speed.

        Returns array of scores in [-1, +1].
        """
        self._ensure_loaded()

        truncated = [t[:MAX_CHAR_LENGTH] for t in texts]

        inputs = self._tokenizer(
            truncated,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_TOKEN_LENGTH,
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)

        # Label order: 0=positive, 1=negative, 2=neutral
        scores = (probs[:, 0] - probs[:, 1]).cpu().numpy()
        return scores

    def score_text(self, text: str) -> float:
        """
        Score a single text. Returns sentiment_score in [-1, +1].
        """
        if not text or len(text.strip()) < 5:
            return 0.0

        return float(self._score_short_batch([text])[0])

    def score_batch(self, texts: List[str]) -> List[float]:
        """
        Score a batch of texts efficiently.

        All texts are pre-truncated to 400 chars and tokenized to 128 tokens.
        This covers news headlines fully (avg 271 chars) and captures the
        lead sentiment of longer SEC filings.

        Returns list of scores in [-1, +1].
        """
        if not texts:
            return []

        self._ensure_loaded()

        scores = [0.0] * len(texts)

        valid_indices = []
        valid_texts = []
        for i, text in enumerate(texts):
            if text and len(text.strip()) >= 5:
                valid_indices.append(i)
                valid_texts.append(text)

        for batch_start in range(0, len(valid_texts), self.batch_size):
            batch = valid_texts[batch_start : batch_start + self.batch_size]
            batch_idx = valid_indices[batch_start : batch_start + self.batch_size]
            batch_scores = self._score_short_batch(batch)
            for idx, s in zip(batch_idx, batch_scores):
                scores[idx] = float(s)

        return scores

    def score_store(
        self,
        store,
        batch_size: int = 64,
        max_records: Optional[int] = None,
    ) -> int:
        """
        Score all unscored records in a SentimentDataStore.

        Fetches large chunks from DB, scores in GPU/CPU batches,
        then writes back scores in bulk.

        Args:
            store: SentimentDataStore instance
            batch_size: inference batch size for FinBERT
            max_records: optional cap on total records to score

        Returns:
            Total records scored.
        """
        import time

        self._ensure_loaded()
        total_scored = 0
        db_fetch_size = max(batch_size * 32, 2048)
        start_time = time.time()

        while True:
            remaining = (
                db_fetch_size
                if max_records is None
                else min(db_fetch_size, max_records - total_scored)
            )
            if remaining <= 0:
                break

            df = store.get_unscored_texts(batch_size=remaining)
            if df.empty:
                break

            texts = df["text"].tolist()
            record_ids = df["record_id"].tolist()

            scores = self.score_batch(texts)
            store.update_scores(record_ids, scores)
            total_scored += len(scores)

            elapsed = time.time() - start_time
            rate = total_scored / elapsed if elapsed > 0 else 0
            logger.info(
                f"Scored {total_scored:,} records "
                f"({elapsed:.0f}s, {rate:.0f} rec/s)"
            )

        elapsed = time.time() - start_time
        logger.info(
            f"Scoring complete: {total_scored:,} records "
            f"in {elapsed:.1f}s ({total_scored/max(elapsed,1):.0f} rec/s)"
        )
        return total_scored

    def get_detailed_scores(
        self, text: str
    ) -> Tuple[float, float, float, float]:
        """
        Get detailed probability breakdown for a text.

        Returns (score, p_positive, p_negative, p_neutral).
        """
        self._ensure_loaded()

        if not text or len(text.strip()) < 5:
            return (0.0, 0.0, 0.0, 1.0)

        inputs = self._tokenizer(
            [text[:MAX_CHAR_LENGTH]],
            return_tensors="pt",
            truncation=True,
            max_length=MAX_TOKEN_LENGTH,
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)[0]

        p_pos = float(probs[0].cpu())
        p_neg = float(probs[1].cpu())
        p_neu = float(probs[2].cpu())
        score = p_pos - p_neg

        return (score, p_pos, p_neg, p_neu)

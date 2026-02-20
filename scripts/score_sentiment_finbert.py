#!/usr/bin/env python
"""
Chapter 10.2 — FinBERT Sentiment Scoring
==========================================

Scores all text records in the sentiment store using FinBERT.
Loads all texts into memory, scores in batches on MPS/CPU,
then writes all scores back in a single bulk operation.

Usage:
    # Score all records
    python scripts/score_sentiment_finbert.py

    # Score a limited number (for testing)
    python scripts/score_sentiment_finbert.py --max-records 100

    # Quality report only
    python scripts/score_sentiment_finbert.py --report-only
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def quality_report(db_path: str):
    """Print quality report on scored sentiment data."""
    import duckdb

    conn = duckdb.connect(db_path, read_only=True)

    total = conn.execute("SELECT COUNT(*) FROM sentiment_texts").fetchone()[0]
    scored = conn.execute(
        "SELECT COUNT(*) FROM sentiment_texts WHERE scored = TRUE"
    ).fetchone()[0]
    unscored = total - scored

    logger.info("=" * 60)
    logger.info("SENTIMENT SCORING QUALITY REPORT")
    logger.info("=" * 60)
    logger.info(f"Total records: {total:,}")
    logger.info(f"Scored:        {scored:,}")
    logger.info(f"Unscored:      {unscored:,}")

    if scored == 0:
        logger.info("No scored records yet. Run scoring first.")
        conn.close()
        return

    stats = conn.execute("""
        SELECT
            source,
            COUNT(*) as n,
            ROUND(AVG(sentiment_score), 4) as mean_score,
            ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP
                (ORDER BY sentiment_score), 4) as median_score,
            ROUND(STDDEV(sentiment_score), 4) as std_score,
            ROUND(MIN(sentiment_score), 4) as min_score,
            ROUND(MAX(sentiment_score), 4) as max_score,
            SUM(CASE WHEN sentiment_score > 0.1 THEN 1 ELSE 0 END) as n_positive,
            SUM(CASE WHEN sentiment_score < -0.1 THEN 1 ELSE 0 END) as n_negative,
            SUM(CASE WHEN sentiment_score BETWEEN -0.1 AND 0.1
                THEN 1 ELSE 0 END) as n_neutral
        FROM sentiment_texts
        WHERE scored = TRUE
        GROUP BY source
    """).df()

    logger.info("\nScore distribution by source:")
    for _, row in stats.iterrows():
        logger.info(f"\n  {row['source']}:")
        logger.info(f"    N scored:  {row['n']:,.0f}")
        logger.info(
            f"    Mean:      {row['mean_score']:.4f}  "
            f"Median: {row['median_score']:.4f}  "
            f"Std: {row['std_score']:.4f}"
        )
        logger.info(
            f"    Range:     [{row['min_score']:.4f}, {row['max_score']:.4f}]"
        )
        pct_pos = row["n_positive"] / row["n"] * 100
        pct_neg = row["n_negative"] / row["n"] * 100
        pct_neu = row["n_neutral"] / row["n"] * 100
        logger.info(
            f"    Positive (>0.1): {row['n_positive']:,.0f} ({pct_pos:.1f}%)"
        )
        logger.info(
            f"    Negative (<-0.1): {row['n_negative']:,.0f} ({pct_neg:.1f}%)"
        )
        logger.info(
            f"    Neutral:  {row['n_neutral']:,.0f} ({pct_neu:.1f}%)"
        )

    # Spot-check: NVDA filings
    logger.info("\nSpot-check: NVDA recent filings")
    nvda = conn.execute("""
        SELECT event_date, source, sentiment_score,
               LEFT(text, 80) as text_preview
        FROM sentiment_texts
        WHERE ticker = 'NVDA' AND scored = TRUE
          AND source = 'sec_8k'
        ORDER BY event_date DESC
        LIMIT 5
    """).df()
    if not nvda.empty:
        for _, row in nvda.iterrows():
            score = row["sentiment_score"]
            indicator = "+" if score > 0.1 else ("-" if score < -0.1 else "~")
            logger.info(
                f"  [{indicator}] {row['event_date']} "
                f"score={score:.3f}: {row['text_preview']}..."
            )

    logger.info("\nSpot-check: Top 5 most positive news")
    pos = conn.execute("""
        SELECT ticker, event_date, sentiment_score,
               LEFT(text, 80) as text_preview
        FROM sentiment_texts
        WHERE scored = TRUE AND source = 'finnhub_news'
        ORDER BY sentiment_score DESC
        LIMIT 5
    """).df()
    for _, row in pos.iterrows():
        logger.info(
            f"  [{row['ticker']}] {row['event_date']} "
            f"score={row['sentiment_score']:.3f}: {row['text_preview']}..."
        )

    logger.info("\nSpot-check: Top 5 most negative news")
    neg = conn.execute("""
        SELECT ticker, event_date, sentiment_score,
               LEFT(text, 80) as text_preview
        FROM sentiment_texts
        WHERE scored = TRUE AND source = 'finnhub_news'
        ORDER BY sentiment_score ASC
        LIMIT 5
    """).df()
    for _, row in neg.iterrows():
        logger.info(
            f"  [{row['ticker']}] {row['event_date']} "
            f"score={row['sentiment_score']:.3f}: {row['text_preview']}..."
        )

    conn.close()


def run_scoring(db_path: str, batch_size: int, device: str, max_records: int = None):
    """Load all texts, score with FinBERT, write back in bulk."""
    import duckdb
    import pandas as pd
    import torch
    from transformers import AutoTokenizer, BertForSequenceClassification

    MAX_CHAR_LENGTH = 400
    MAX_TOKEN_LENGTH = 128

    # -- Load all unscored texts from DB --
    conn = duckdb.connect(db_path)

    limit_clause = f"LIMIT {max_records}" if max_records else ""
    logger.info("Loading unscored texts from database...")
    df = conn.execute(f"""
        SELECT record_id, text
        FROM sentiment_texts
        WHERE scored = FALSE
        ORDER BY LENGTH(text) ASC
        {limit_clause}
    """).df()

    if df.empty:
        logger.info("No unscored records found.")
        conn.close()
        return 0

    logger.info(f"Loaded {len(df):,} texts (sorted by length for efficient batching)")

    texts = df["text"].tolist()
    record_ids = df["record_id"].tolist()

    # -- Load FinBERT --
    logger.info(f"Loading FinBERT on {device}...")
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = BertForSequenceClassification.from_pretrained(
        "ProsusAI/finbert", use_safetensors=True
    )
    model.to(device)
    model.eval()
    logger.info("FinBERT loaded.")

    # -- Score all texts in batches --
    all_scores = np.zeros(len(texts), dtype=np.float32)
    start = time.time()

    for batch_start in range(0, len(texts), batch_size):
        batch_end = min(batch_start + batch_size, len(texts))
        batch_texts = texts[batch_start:batch_end]

        # Pre-truncate and tokenize
        truncated = [t[:MAX_CHAR_LENGTH] if t else "neutral" for t in batch_texts]

        inputs = tokenizer(
            truncated,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_TOKEN_LENGTH,
            padding=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            scores = (probs[:, 0] - probs[:, 1]).cpu().numpy()

        all_scores[batch_start:batch_end] = scores

        # Log every 5000 records
        if (batch_end % 5000 < batch_size) or batch_end == len(texts):
            elapsed = time.time() - start
            rate = batch_end / elapsed if elapsed > 0 else 0
            remaining = (len(texts) - batch_end) / max(rate, 1)
            logger.info(
                f"  Scored {batch_end:,}/{len(texts):,} "
                f"({elapsed:.0f}s, {rate:.0f} rec/s, "
                f"~{remaining:.0f}s remaining)"
            )

    scoring_time = time.time() - start
    logger.info(
        f"Scoring complete: {len(texts):,} records in {scoring_time:.1f}s "
        f"({len(texts)/scoring_time:.0f} rec/s)"
    )

    # -- Write all scores back to DB in bulk --
    logger.info("Writing scores to database...")
    write_start = time.time()

    score_df = pd.DataFrame({
        "record_id": record_ids,
        "sentiment_score": all_scores.tolist(),
    })

    conn.execute("DROP TABLE IF EXISTS _score_updates")
    conn.execute("CREATE TEMP TABLE _score_updates AS SELECT * FROM score_df")
    conn.execute("""
        UPDATE sentiment_texts s
        SET sentiment_score = u.sentiment_score,
            scored = TRUE
        FROM _score_updates u
        WHERE s.record_id = u.record_id
    """)
    conn.execute("DROP TABLE IF EXISTS _score_updates")

    write_time = time.time() - write_start
    logger.info(f"DB write complete in {write_time:.1f}s")

    conn.close()
    return len(texts)


def main():
    parser = argparse.ArgumentParser(
        description="Score sentiment records with FinBERT"
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default="data/sentiment.duckdb",
        help="Path to sentiment DuckDB file",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for FinBERT inference",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device: 'cpu', 'mps', 'cuda', or 'auto' (default)",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Max records to score (default: all)",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Only print quality report (no scoring)",
    )
    args = parser.parse_args()

    if args.report_only:
        quality_report(args.db_path)
        return

    import torch

    device = args.device
    if device == "auto":
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

    logger.info("=" * 60)
    logger.info("CHAPTER 10.2: FINBERT SENTIMENT SCORING")
    logger.info("=" * 60)
    logger.info(f"Database: {args.db_path}")
    logger.info(f"Device: {device}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Max records: {args.max_records or 'all'}")

    start = time.time()
    n_scored = run_scoring(
        args.db_path, args.batch_size, device, args.max_records
    )
    total = time.time() - start

    logger.info(f"\nTotal time: {total:.1f}s")
    if n_scored > 0:
        quality_report(args.db_path)


if __name__ == "__main__":
    main()

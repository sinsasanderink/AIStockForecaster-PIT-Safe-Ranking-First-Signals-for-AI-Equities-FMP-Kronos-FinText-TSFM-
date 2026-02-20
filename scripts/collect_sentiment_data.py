#!/usr/bin/env python
"""
Chapter 10 — Sentiment Data Collection
========================================

Collects historical sentiment data for the AI stock universe from:
1. SEC EDGAR 8-K filings (2016–present, free, unlimited)
2. FinnHub company news (2024–present, free tier)

Usage:
    # Collect both sources for full universe
    python scripts/collect_sentiment_data.py

    # SEC filings only (slower but unlimited history)
    python scripts/collect_sentiment_data.py --sec-only

    # FinnHub news only (faster but limited history)
    python scripts/collect_sentiment_data.py --news-only

    # Single ticker test
    python scripts/collect_sentiment_data.py --ticker NVDA

    # Custom date range for news
    python scripts/collect_sentiment_data.py --news-start 2025-01-01
"""

import argparse
import json
import logging
import sys
from datetime import date
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

from src.data.sentiment_store import SentimentDataStore
from src.universe.ai_stocks import get_all_tickers

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Collect sentiment data for AI stock universe"
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default="data/sentiment.duckdb",
        help="Path to sentiment DuckDB file",
    )
    parser.add_argument(
        "--sec-only",
        action="store_true",
        help="Collect SEC filings only (skip FinnHub news)",
    )
    parser.add_argument(
        "--news-only",
        action="store_true",
        help="Collect FinnHub news only (skip SEC filings)",
    )
    parser.add_argument(
        "--ticker",
        type=str,
        default=None,
        help="Collect for a single ticker only (for testing)",
    )
    parser.add_argument(
        "--sec-start",
        type=str,
        default="2016-01-01",
        help="Start date for SEC filings (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--news-start",
        type=str,
        default="2024-01-01",
        help="Start date for FinnHub news (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="End date (YYYY-MM-DD, default: today)",
    )

    args = parser.parse_args()

    sec_start = date.fromisoformat(args.sec_start)
    news_start = date.fromisoformat(args.news_start)
    end_date = date.fromisoformat(args.end_date) if args.end_date else None

    collect_sec = not args.news_only
    collect_news = not args.sec_only

    # Determine tickers
    if args.ticker:
        tickers = [args.ticker.upper()]
    else:
        tickers = sorted(get_all_tickers())

    logger.info("=" * 70)
    logger.info("CHAPTER 10: SENTIMENT DATA COLLECTION")
    logger.info("=" * 70)
    logger.info(f"Tickers: {len(tickers)}")
    logger.info(f"SEC filings: {'YES' if collect_sec else 'NO'} (from {sec_start})")
    logger.info(f"FinnHub news: {'YES' if collect_news else 'NO'} (from {news_start})")
    logger.info(f"Database: {args.db_path}")
    logger.info("=" * 70)

    store = SentimentDataStore(db_path=args.db_path)

    try:
        stats = store.collect_all(
            tickers=tickers,
            sec_start=sec_start,
            news_start=news_start,
            end_date=end_date,
            collect_sec=collect_sec,
            collect_news=collect_news,
        )

        # Print summary
        logger.info("\n" + "=" * 70)
        logger.info("COLLECTION SUMMARY")
        logger.info("=" * 70)
        logger.info(f"Tickers processed: {stats['tickers_processed']}/{len(tickers)}")
        logger.info(f"SEC 8-K filings:   {stats['sec_total']}")
        logger.info(f"FinnHub articles:  {stats['news_total']}")
        if stats["errors"]:
            logger.warning(f"Errors: {len(stats['errors'])}")
            for err in stats["errors"][:5]:
                logger.warning(f"  {err['ticker']}: {err['error']}")

        # Database summary
        summary = store.get_summary()
        logger.info(f"\nDatabase total: {summary['total_records']} records")
        for src_info in summary["by_source"]:
            logger.info(
                f"  {src_info['source']}: {src_info['cnt']} records, "
                f"{src_info['n_tickers']} tickers, "
                f"{src_info['min_date']} to {src_info['max_date']}"
            )

        # Save stats
        stats_path = Path(args.db_path).parent / "sentiment_collection_stats.json"
        with open(stats_path, "w") as f:
            json.dump(
                {
                    "tickers_processed": stats["tickers_processed"],
                    "sec_total": stats["sec_total"],
                    "news_total": stats["news_total"],
                    "errors": stats["errors"],
                    "summary": {
                        "total_records": summary["total_records"],
                        "by_source": summary["by_source"],
                    },
                },
                f,
                indent=2,
                default=str,
            )
        logger.info(f"\nStats saved to: {stats_path}")

    finally:
        store.close()

    logger.info("\nDone.")


if __name__ == "__main__":
    main()

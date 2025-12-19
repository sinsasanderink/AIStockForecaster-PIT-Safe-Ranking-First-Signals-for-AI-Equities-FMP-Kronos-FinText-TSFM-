"""
AI Stock Forecaster CLI
=======================

Command-line interface for running forecaster pipelines.

Usage:
    python -m src.cli download-data --start 2020-01-01 --end 2024-01-01
    python -m src.cli build-universe --asof 2024-01-15
    python -m src.cli score --asof 2024-01-15
    python -m src.cli make-report --asof 2024-01-15
    python -m src.cli audit-pit --start 2023-01-01 --end 2024-01-01
"""

import argparse
import logging
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("cli")


def parse_date(date_str: str) -> date:
    """Parse a date string in YYYY-MM-DD format."""
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Invalid date format: {date_str}. Use YYYY-MM-DD."
        )


def cmd_download_data(args):
    """Download market data from FMP."""
    from src.pipelines import run_data_download
    
    logger.info("=" * 60)
    logger.info("DOWNLOAD DATA")
    logger.info("=" * 60)
    
    # Get tickers from universe or explicit list
    if args.tickers:
        tickers = args.tickers.split(",")
    else:
        # Use default AI universe
        from src.pipelines import run_universe_construction
        universe = run_universe_construction(args.end)
        tickers = universe.tickers
    
    result = run_data_download(
        tickers=tickers,
        start_date=args.start,
        end_date=args.end,
        data_types=args.types.split(",") if args.types else None,
        force_refresh=args.force,
        dry_run=args.dry_run,
    )
    
    print(result.summary())
    return 0 if result.success else 1


def cmd_build_universe(args):
    """Build stock universe for a given date."""
    from src.pipelines import run_universe_construction
    
    logger.info("=" * 60)
    logger.info("BUILD UNIVERSE")
    logger.info("=" * 60)
    
    # Parse categories if provided
    categories = None
    if args.categories:
        categories = args.categories.split(",")
        logger.info(f"Filtering to categories: {categories}")
    
    result = run_universe_construction(
        asof_date=args.asof,
        max_size=args.max_size,
        categories=categories,
    )
    
    print(result.summary())
    print("\nConstituents:")
    for i, ticker in enumerate(result.tickers, 1):
        # Show category for each ticker
        meta = result.ticker_metadata.get(ticker)
        cat = meta.sector if meta else "unknown"
        print(f"  {i:3d}. {ticker:6s} ({cat})")
    
    return 0


def cmd_build_features(args):
    """Build features for scoring."""
    logger.info("=" * 60)
    logger.info("BUILD FEATURES")
    logger.info("=" * 60)
    
    # TODO: Implement feature building pipeline
    logger.warning("Feature building not yet implemented")
    return 0


def cmd_train_baselines(args):
    """Train baseline models."""
    logger.info("=" * 60)
    logger.info("TRAIN BASELINES")
    logger.info("=" * 60)
    
    # TODO: Implement baseline training
    logger.warning("Baseline training not yet implemented")
    return 0


def cmd_score(args):
    """Generate signals for a given date."""
    from src.pipelines import run_scoring
    
    logger.info("=" * 60)
    logger.info("SCORE")
    logger.info("=" * 60)
    
    tickers = args.tickers.split(",") if args.tickers else None
    
    result = run_scoring(
        asof_date=args.asof,
        tickers=tickers,
        horizons=[20, 60, 90],
    )
    
    print(result.summary())
    return 0


def cmd_make_report(args):
    """Generate reports for a scoring date."""
    from src.pipelines import run_report_generation
    
    logger.info("=" * 60)
    logger.info("MAKE REPORT")
    logger.info("=" * 60)
    
    output_dir = Path(args.output) if args.output else None
    formats = args.formats.split(",") if args.formats else None
    
    result = run_report_generation(
        asof_date=args.asof,
        output_dir=output_dir,
        formats=formats,
    )
    
    print(result.summary())
    return 0


def cmd_audit_pit(args):
    """Run PIT validation audit."""
    from src.audits import run_pit_audit
    
    logger.info("=" * 60)
    logger.info("PIT AUDIT")
    logger.info("=" * 60)
    
    result = run_pit_audit(
        start_date=args.start,
        end_date=args.end,
    )
    
    print(result.summary())
    return 0 if result.passed else 1


def cmd_audit_survivorship(args):
    """Run survivorship bias audit."""
    from src.audits import run_survivorship_audit
    
    logger.info("=" * 60)
    logger.info("SURVIVORSHIP AUDIT")
    logger.info("=" * 60)
    
    result = run_survivorship_audit(
        start_date=args.start,
        end_date=args.end,
    )
    
    print(result.summary())
    return 0 if result.passed else 1


def cmd_list_universe(args):
    """List the AI stock universe and categories."""
    from src.universe import (
        AI_UNIVERSE,
        CATEGORY_DESCRIPTIONS,
        get_all_tickers,
    )
    
    all_tickers = get_all_tickers()
    
    print("=" * 70)
    print("AI STOCK UNIVERSE")
    print("=" * 70)
    print(f"\nTotal: {len(all_tickers)} unique tickers across {len(AI_UNIVERSE)} categories\n")
    
    if args.category:
        # Show specific category
        if args.category not in AI_UNIVERSE:
            print(f"Unknown category: {args.category}")
            print(f"Valid categories: {list(AI_UNIVERSE.keys())}")
            return 1
        
        tickers = AI_UNIVERSE[args.category]
        desc = CATEGORY_DESCRIPTIONS[args.category]
        print(f"Category: {args.category}")
        print(f"Description: {desc}")
        print(f"Tickers ({len(tickers)}):")
        for t in tickers:
            print(f"  {t}")
    else:
        # Show all categories
        for cat, tickers in AI_UNIVERSE.items():
            desc = CATEGORY_DESCRIPTIONS[cat]
            print(f"{cat} ({len(tickers)} stocks)")
            print(f"  {desc}")
            print(f"  Tickers: {', '.join(tickers)}")
            print()
    
    return 0


def cmd_full_pipeline(args):
    """Run the full pipeline: universe → features → score → report."""
    logger.info("=" * 60)
    logger.info("FULL PIPELINE")
    logger.info("=" * 60)
    
    asof = args.asof
    
    # Step 1: Build universe
    logger.info("Step 1/4: Building universe...")
    from src.pipelines import run_universe_construction
    universe = run_universe_construction(asof)
    print(universe.summary())
    
    # Step 2: Build features (placeholder)
    logger.info("Step 2/4: Building features...")
    logger.warning("Feature building not yet implemented")
    
    # Step 3: Score
    logger.info("Step 3/4: Scoring...")
    from src.pipelines import run_scoring
    scoring = run_scoring(asof, tickers=universe.tickers)
    print(scoring.summary())
    
    # Step 4: Generate report
    logger.info("Step 4/4: Generating report...")
    from src.pipelines import run_report_generation
    report = run_report_generation(asof)
    print(report.summary())
    
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)
    
    return 0


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="AI Stock Forecaster CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.cli download-data --start 2020-01-01 --end 2024-01-01
  python -m src.cli build-universe --asof 2024-01-15
  python -m src.cli score --asof 2024-01-15
  python -m src.cli make-report --asof 2024-01-15
  python -m src.cli run --asof 2024-01-15  # Full pipeline
        """,
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # download-data
    p_download = subparsers.add_parser("download-data", help="Download market data")
    p_download.add_argument("--start", type=parse_date, required=True, help="Start date (YYYY-MM-DD)")
    p_download.add_argument("--end", type=parse_date, required=True, help="End date (YYYY-MM-DD)")
    p_download.add_argument("--tickers", type=str, help="Comma-separated tickers (default: universe)")
    p_download.add_argument("--types", type=str, help="Data types: ohlcv,fundamentals,events")
    p_download.add_argument("--force", action="store_true", help="Force re-download")
    p_download.add_argument("--dry-run", action="store_true", help="Validate without downloading")
    p_download.set_defaults(func=cmd_download_data)
    
    # build-universe
    p_universe = subparsers.add_parser("build-universe", help="Build stock universe")
    p_universe.add_argument("--asof", type=parse_date, required=True, help="As-of date (YYYY-MM-DD)")
    p_universe.add_argument("--max-size", type=int, default=100, help="Max universe size (default: 100)")
    p_universe.add_argument(
        "--categories", 
        type=str, 
        help="Comma-separated AI categories (default: all). Options: "
             "ai_compute_core_semis, semicap_eda_manufacturing, "
             "networking_optics_dc_hw, datacenter_power_cooling_reits, "
             "cloud_platforms_model_owners, data_platforms_enterprise_sw, "
             "cybersecurity, robotics_industrial_ai, autonomy_defense_ai, ai_apps_ads_misc"
    )
    p_universe.set_defaults(func=cmd_build_universe)
    
    # build-features
    p_features = subparsers.add_parser("build-features", help="Build features")
    p_features.add_argument("--asof", type=parse_date, required=True, help="As-of date")
    p_features.set_defaults(func=cmd_build_features)
    
    # train-baselines
    p_train = subparsers.add_parser("train-baselines", help="Train baseline models")
    p_train.add_argument("--start", type=parse_date, help="Training start date")
    p_train.add_argument("--end", type=parse_date, help="Training end date")
    p_train.set_defaults(func=cmd_train_baselines)
    
    # score
    p_score = subparsers.add_parser("score", help="Generate signals")
    p_score.add_argument("--asof", type=parse_date, required=True, help="As-of date")
    p_score.add_argument("--tickers", type=str, help="Comma-separated tickers (default: universe)")
    p_score.set_defaults(func=cmd_score)
    
    # make-report
    p_report = subparsers.add_parser("make-report", help="Generate reports")
    p_report.add_argument("--asof", type=parse_date, required=True, help="As-of date")
    p_report.add_argument("--output", type=str, help="Output directory")
    p_report.add_argument("--formats", type=str, help="Output formats: text,csv,json,html")
    p_report.set_defaults(func=cmd_make_report)
    
    # audit-pit
    p_pit = subparsers.add_parser("audit-pit", help="Run PIT audit")
    p_pit.add_argument("--start", type=parse_date, required=True, help="Start date")
    p_pit.add_argument("--end", type=parse_date, required=True, help="End date")
    p_pit.set_defaults(func=cmd_audit_pit)
    
    # audit-survivorship
    p_surv = subparsers.add_parser("audit-survivorship", help="Run survivorship audit")
    p_surv.add_argument("--start", type=parse_date, required=True, help="Start date")
    p_surv.add_argument("--end", type=parse_date, required=True, help="End date")
    p_surv.set_defaults(func=cmd_audit_survivorship)
    
    # list-universe
    p_list = subparsers.add_parser("list-universe", help="List AI stock universe")
    p_list.add_argument(
        "--category",
        type=str,
        help="Show only this category"
    )
    p_list.set_defaults(func=cmd_list_universe)
    
    # run (full pipeline)
    p_run = subparsers.add_parser("run", help="Run full pipeline")
    p_run.add_argument("--asof", type=parse_date, required=True, help="As-of date")
    p_run.set_defaults(func=cmd_full_pipeline)
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return 1
    
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())


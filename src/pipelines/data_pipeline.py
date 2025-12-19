"""
Data Pipeline
=============

Orchestrates data download, validation, and PIT-safe storage.

This pipeline:
1. Downloads OHLCV data from FMP API
2. Stores with PIT-safe metadata (observed_at timestamps)
3. Runs validation checks
4. Supports incremental updates
"""

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import List, Optional, Dict, Any, Set
import logging
import time

logger = logging.getLogger(__name__)


@dataclass
class DataDownloadResult:
    """Result of a data download operation."""
    success: bool
    tickers_processed: int
    tickers_failed: List[str]
    rows_downloaded: int
    start_date: date
    end_date: date
    duration_seconds: float
    pit_violations_detected: int = 0
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    
    def summary(self) -> str:
        status = "✅ SUCCESS" if self.success else "❌ FAILED"
        lines = [
            f"{status}: Data Download Pipeline",
            f"  Period: {self.start_date} to {self.end_date}",
            f"  Tickers: {self.tickers_processed} processed, {len(self.tickers_failed)} failed",
            f"  Rows: {self.rows_downloaded:,}",
            f"  Duration: {self.duration_seconds:.1f}s",
            f"  PIT Violations: {self.pit_violations_detected}",
        ]
        if self.warnings:
            lines.append(f"  Warnings: {len(self.warnings)}")
            for w in self.warnings[:3]:
                lines.append(f"    - {w}")
        if self.errors:
            lines.append(f"  Errors: {len(self.errors)}")
            for e in self.errors[:3]:
                lines.append(f"    - {e}")
        return "\n".join(lines)


def run_data_download(
    tickers: List[str],
    start_date: date,
    end_date: date,
    data_types: Optional[List[str]] = None,
    force_refresh: bool = False,
    dry_run: bool = False,
    db_path: Optional[Path] = None,
) -> DataDownloadResult:
    """
    Download market data for specified tickers and date range.
    
    This is the main entry point for populating the data store.
    All downloaded data is stored with PIT metadata.
    
    Args:
        tickers: List of ticker symbols to download
        start_date: Start of date range
        end_date: End of date range  
        data_types: Types to download ['ohlcv', 'fundamentals', 'profiles']
                   If None, downloads all types
        force_refresh: If True, re-download even if data exists
        dry_run: If True, validate inputs but don't download
        db_path: Path to DuckDB database (None = in-memory)
    
    Returns:
        DataDownloadResult with status and statistics
    """
    from ..data.fmp_client import FMPClient, FMPError, RateLimitError
    from ..data.pit_store import DuckDBPITStore
    
    start_time = time.time()
    
    if data_types is None:
        data_types = ["ohlcv", "profiles"]  # Default to most common
    
    logger.info(
        f"Starting data download: {len(tickers)} tickers, "
        f"{start_date} to {end_date}, types={data_types}"
    )
    
    if dry_run:
        logger.info("DRY RUN - no data will be downloaded")
        return DataDownloadResult(
            success=True,
            tickers_processed=len(tickers),
            tickers_failed=[],
            rows_downloaded=0,
            start_date=start_date,
            end_date=end_date,
            duration_seconds=time.time() - start_time,
            warnings=["Dry run - no data downloaded"],
        )
    
    # Initialize FMP client and PIT store
    try:
        client = FMPClient()
        store = DuckDBPITStore(db_path)
    except Exception as e:
        return DataDownloadResult(
            success=False,
            tickers_processed=0,
            tickers_failed=tickers,
            rows_downloaded=0,
            start_date=start_date,
            end_date=end_date,
            duration_seconds=time.time() - start_time,
            errors=[f"Initialization failed: {e}"],
        )
    
    rows_downloaded = 0
    failed_tickers = []
    warnings = []
    errors = []
    pit_violations = 0
    
    start_str = start_date.isoformat()
    end_str = end_date.isoformat()
    
    for i, ticker in enumerate(tickers):
        logger.info(f"[{i+1}/{len(tickers)}] Downloading {ticker}...")
        
        try:
            # Download OHLCV data
            if "ohlcv" in data_types:
                prices_df = client.get_historical_prices(ticker, start_str, end_str)
                
                if not prices_df.empty:
                    stored = store.store_prices(prices_df)
                    rows_downloaded += stored
                    
                    # Validate PIT
                    validation = store.validate_pit(ticker, "prices")
                    if not validation["valid"]:
                        pit_violations += len(validation["issues"])
                        warnings.append(f"{ticker}: {validation['issues']}")
                else:
                    warnings.append(f"{ticker}: No price data returned")
            
            # Download profile
            if "profiles" in data_types:
                profile = client.get_profile(ticker)
                if profile:
                    store.store_profile(ticker, profile)
            
            # Download fundamentals (expensive - use sparingly)
            if "fundamentals" in data_types:
                try:
                    income_df = client.get_income_statement(ticker, period="quarter", limit=8)
                    if not income_df.empty:
                        stored = store.store_fundamentals(income_df, "income")
                        rows_downloaded += stored
                except FMPError as e:
                    warnings.append(f"{ticker} income: {e}")
        
        except RateLimitError as e:
            errors.append(f"Rate limit hit: {e}")
            logger.error(f"Rate limit exceeded, stopping download")
            failed_tickers.extend(tickers[i:])
            break
        
        except FMPError as e:
            failed_tickers.append(ticker)
            errors.append(f"{ticker}: {e}")
            logger.warning(f"Failed to download {ticker}: {e}")
        
        except Exception as e:
            failed_tickers.append(ticker)
            errors.append(f"{ticker}: Unexpected error - {e}")
            logger.exception(f"Unexpected error for {ticker}")
    
    # Close store
    store.close()
    
    duration = time.time() - start_time
    success = len(failed_tickers) == 0 and len(errors) == 0
    
    result = DataDownloadResult(
        success=success,
        tickers_processed=len(tickers) - len(failed_tickers),
        tickers_failed=failed_tickers,
        rows_downloaded=rows_downloaded,
        start_date=start_date,
        end_date=end_date,
        duration_seconds=duration,
        pit_violations_detected=pit_violations,
        warnings=warnings,
        errors=errors,
    )
    
    logger.info(result.summary())
    return result


def run_incremental_update(
    end_date: Optional[date] = None,
    tickers: Optional[List[str]] = None,
    db_path: Optional[Path] = None,
) -> DataDownloadResult:
    """
    Run incremental update to bring data store current.
    
    Detects the last available date in the store and downloads
    new data up to end_date (defaults to yesterday).
    
    Args:
        end_date: End date for update (defaults to yesterday)
        tickers: Specific tickers to update (None = all in store)
        db_path: Path to DuckDB database
    
    Returns:
        DataDownloadResult with status and statistics
    """
    from ..data.pit_store import DuckDBPITStore
    from ..universe.ai_stocks import get_all_tickers
    
    start_time = time.time()
    
    if end_date is None:
        end_date = date.today() - timedelta(days=1)
    
    logger.info(f"Starting incremental update through {end_date}")
    
    # Open store to check existing data
    store = DuckDBPITStore(db_path)
    
    if tickers is None:
        # Get tickers from store, or use AI universe
        existing = store.get_all_tickers()
        if existing:
            tickers = existing
        else:
            tickers = get_all_tickers()
            logger.info(f"No existing data - using AI universe ({len(tickers)} tickers)")
    
    # Find the earliest needed start date
    start_date = end_date - timedelta(days=30)  # Default lookback
    
    for ticker in tickers[:5]:  # Sample a few tickers
        last_date = store.get_last_observed_date(ticker, "prices")
        if last_date and last_date < start_date:
            start_date = last_date + timedelta(days=1)
    
    store.close()
    
    logger.info(f"Updating {len(tickers)} tickers from {start_date} to {end_date}")
    
    # Run download for the missing period
    return run_data_download(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        data_types=["ohlcv"],
        db_path=db_path,
    )


def download_benchmark_data(
    benchmarks: Optional[List[str]] = None,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    db_path: Optional[Path] = None,
) -> DataDownloadResult:
    """
    Download benchmark data (QQQ, SPY, etc.).
    
    Args:
        benchmarks: List of benchmark tickers (default: QQQ, XLK, SMH)
        start_date: Start date (default: 5 years ago)
        end_date: End date (default: yesterday)
        db_path: Path to DuckDB database
    
    Returns:
        DataDownloadResult
    """
    if benchmarks is None:
        benchmarks = ["QQQ", "SPY", "XLK", "SMH", "SOXX"]
    
    if end_date is None:
        end_date = date.today() - timedelta(days=1)
    
    if start_date is None:
        start_date = end_date - timedelta(days=5*365)  # 5 years
    
    logger.info(f"Downloading benchmark data: {benchmarks}")
    
    return run_data_download(
        tickers=benchmarks,
        start_date=start_date,
        end_date=end_date,
        data_types=["ohlcv"],
        db_path=db_path,
    )


def validate_downloaded_data(
    tickers: List[str],
    db_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Validate downloaded data for PIT correctness and completeness.
    
    Args:
        tickers: Tickers to validate
        db_path: Path to DuckDB database
    
    Returns:
        Dict with validation results
    """
    from ..data.pit_store import DuckDBPITStore
    
    store = DuckDBPITStore(db_path)
    
    results = {
        "valid": True,
        "tickers_checked": len(tickers),
        "pit_violations": 0,
        "missing_data": [],
        "issues": [],
    }
    
    for ticker in tickers:
        # Check PIT validity
        validation = store.validate_pit(ticker, "prices")
        if not validation["valid"]:
            results["pit_violations"] += len(validation["issues"])
            results["issues"].extend(
                f"{ticker}: {issue}" for issue in validation["issues"]
            )
        
        # Check data availability
        date_range = store.get_date_range(ticker)
        if date_range is None:
            results["missing_data"].append(ticker)
    
    results["valid"] = (
        results["pit_violations"] == 0 and 
        len(results["missing_data"]) == 0
    )
    
    store.close()
    return results


def get_data_coverage_report(
    db_path: Optional[Path] = None,
) -> str:
    """
    Generate a report on data coverage in the store.
    
    Args:
        db_path: Path to DuckDB database
    
    Returns:
        Formatted report string
    """
    from ..data.pit_store import DuckDBPITStore
    
    store = DuckDBPITStore(db_path)
    stats = store.get_stats()
    tickers = store.get_all_tickers()
    
    lines = [
        "=" * 60,
        "DATA COVERAGE REPORT",
        "=" * 60,
        "",
        f"Total tickers: {stats['tickers']}",
        f"Price records: {stats['prices_count']:,}",
        f"Fundamental records: {stats['fundamentals_count']:,}",
        f"Profile records: {stats['profiles_count']:,}",
        "",
        "Date ranges by ticker (sample):",
    ]
    
    for ticker in tickers[:10]:
        date_range = store.get_date_range(ticker)
        if date_range:
            lines.append(f"  {ticker}: {date_range[0]} to {date_range[1]}")
    
    if len(tickers) > 10:
        lines.append(f"  ... and {len(tickers) - 10} more tickers")
    
    store.close()
    return "\n".join(lines)

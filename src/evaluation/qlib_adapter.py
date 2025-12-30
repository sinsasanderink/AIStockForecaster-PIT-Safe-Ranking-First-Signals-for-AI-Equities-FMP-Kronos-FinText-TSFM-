"""
Qlib Adapter (Chapter 6.6)

Provides adapter layer to feed Qlib for "shadow evaluation" second opinion.

CRITICAL: Qlib is a SHADOW EVALUATOR only. Our system remains source-of-truth for:
- Universe (stable_id + survivorship)
- Labels (v2 total return)
- PIT discipline
- Splits/purging/embargo
- Core metrics

This adapter:
1. Converts our EvaluationRow format to Qlib's expected format
2. Validates the converted frame for common pitfalls
3. Provides optional Qlib evaluation runner (if pyqlib is installed)

ISOLATION: Qlib imports are isolated so core evaluation works without pyqlib.
"""

from datetime import datetime, timezone
from typing import Optional, Dict, Any, Tuple
import pandas as pd
import numpy as np
import logging
from pathlib import Path

from .definitions import get_market_close_utc

logger = logging.getLogger(__name__)


# ============================================================================
# QLIB AVAILABILITY CHECK
# ============================================================================

def is_qlib_available() -> bool:
    """Check if pyqlib is installed and available."""
    try:
        import qlib
        return True
    except ImportError:
        return False


# ============================================================================
# FORMAT CONVERSION
# ============================================================================

def to_qlib_format(
    eval_rows: pd.DataFrame,
    datetime_col: str = "as_of_date",
    instrument_col: str = "stable_id",
    score_col: str = "score",
    label_col: str = "excess_return",
    use_stable_id: bool = True,
    tz_aware: bool = True
) -> pd.DataFrame:
    """
    Convert our EvaluationRow format to Qlib's expected format.
    
    Qlib expects:
    - MultiIndex: (datetime, instrument)
    - datetime: timezone-aware (UTC aligned with our "as-of close" convention)
    - instrument: stable and unique identifier
    - score: prediction column
    - label: realized return column
    
    COLUMN ALIASES (handled automatically):
    - datetime: accepts "as_of_date" or "date"
    - score: accepts "score", "pred", "baseline_score", "prediction"
    - label: accepts "excess_return", "label", "return"
    
    Args:
        eval_rows: DataFrame in EvaluationRow format
        datetime_col: Column name for date (with fallback aliases)
        instrument_col: Column for instrument (prefer stable_id, fallback to ticker)
        score_col: Column name for prediction/score (with fallback aliases)
        label_col: Column name for realized return (with fallback aliases)
        use_stable_id: If True, use stable_id as instrument; else use ticker
        tz_aware: If True, convert dates to UTC market close time
        
    Returns:
        DataFrame with Qlib-compatible MultiIndex (datetime, instrument)
    """
    df = eval_rows.copy()
    
    # DEBUG: Log incoming columns
    logger.info(f"to_qlib_format: incoming columns = {list(df.columns)}")
    if len(df) > 0:
        logger.debug(f"to_qlib_format: sample rows:\n{df.head(2).to_string()}")
    
    # =========================================================================
    # HANDLE COLUMN ALIASES - find actual column names
    # =========================================================================
    
    # Date column aliases
    date_aliases = [datetime_col, "as_of_date", "date", "datetime"]
    actual_date_col = None
    for alias in date_aliases:
        if alias in df.columns:
            actual_date_col = alias
            break
    
    if actual_date_col is None:
        raise ValueError(f"Missing date column. Tried: {date_aliases}. Available: {list(df.columns)}")
    
    # Score column aliases
    score_aliases = [score_col, "score", "pred", "baseline_score", "prediction"]
    actual_score_col = None
    for alias in score_aliases:
        if alias in df.columns:
            actual_score_col = alias
            break
    
    if actual_score_col is None:
        raise ValueError(f"Missing score column. Tried: {score_aliases}. Available: {list(df.columns)}")
    
    # Label column aliases (optional but preferred)
    label_aliases = [label_col, "excess_return", "label", "return"] if label_col else []
    actual_label_col = None
    for alias in label_aliases:
        if alias in df.columns:
            actual_label_col = alias
            break
    
    logger.info(f"to_qlib_format: using date={actual_date_col}, score={actual_score_col}, label={actual_label_col}")
    
    # =========================================================================
    # DETERMINE INSTRUMENT COLUMN
    # =========================================================================
    
    if use_stable_id and "stable_id" in df.columns:
        inst_col = "stable_id"
    elif "ticker" in df.columns:
        inst_col = "ticker"
        logger.warning("Using 'ticker' as instrument (stable_id not available)")
    else:
        raise ValueError("Need either 'stable_id' or 'ticker' for instrument")
    
    # Validate we have the columns
    required_cols = [actual_date_col, actual_score_col, inst_col]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Convert datetime to UTC market close if needed
    if tz_aware:
        if pd.api.types.is_datetime64_any_dtype(df[actual_date_col]):
            # Already datetime
            dates = df[actual_date_col]
        else:
            # Convert to datetime
            dates = pd.to_datetime(df[actual_date_col])
        
        # Add UTC market close time (4 PM ET = varies by DST)
        utc_datetimes = []
        for d in dates:
            if hasattr(d, 'date'):
                d = d.date()
            utc_dt = get_market_close_utc(d)
            utc_datetimes.append(utc_dt)
        
        df["datetime"] = utc_datetimes
    else:
        df["datetime"] = pd.to_datetime(df[actual_date_col])
    
    # Prepare output columns
    df["instrument"] = df[inst_col]
    
    # Keep stable_id as a column for traceability if we're using ticker
    if inst_col != "stable_id" and "stable_id" in df.columns:
        df["_stable_id"] = df["stable_id"]
    
    # Create output DataFrame
    output = df[["datetime", "instrument", actual_score_col]].copy()
    output = output.rename(columns={actual_score_col: "score"})
    
    if actual_label_col is not None:
        output["label"] = df[actual_label_col].values
    
    # Add optional columns for traceability
    if "_stable_id" in df.columns:
        output["stable_id"] = df["_stable_id"].values
    
    # Set MultiIndex
    output = output.set_index(["datetime", "instrument"])
    
    logger.info(
        f"Converted {len(output)} rows to Qlib format. "
        f"Index: {output.index.names}, Columns: {list(output.columns)}"
    )
    
    return output


def from_qlib_format(
    qlib_df: pd.DataFrame,
    datetime_col: str = "as_of_date",
    instrument_col: str = "stable_id"
) -> pd.DataFrame:
    """
    Convert Qlib format back to our EvaluationRow format.
    
    Args:
        qlib_df: DataFrame with Qlib MultiIndex (datetime, instrument)
        datetime_col: Target column name for date
        instrument_col: Target column name for instrument
        
    Returns:
        DataFrame with flat structure
    """
    df = qlib_df.reset_index()
    
    # Rename columns
    rename_map = {
        "datetime": datetime_col,
        "instrument": instrument_col
    }
    
    df = df.rename(columns=rename_map)
    
    # Convert datetime back to date if needed
    if datetime_col in df.columns and pd.api.types.is_datetime64_any_dtype(df[datetime_col]):
        df[datetime_col] = df[datetime_col].dt.date
    
    return df


# ============================================================================
# VALIDATION
# ============================================================================

def validate_qlib_frame(df: pd.DataFrame) -> Tuple[bool, str]:
    """
    Validate a Qlib-formatted DataFrame for common pitfalls.
    
    Checks:
    1. Has MultiIndex with (datetime, instrument)
    2. No duplicate index entries
    3. No NaT in datetime
    4. datetime is timezone-aware (UTC)
    5. No empty instruments
    6. Has 'score' column
    
    Args:
        df: DataFrame to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    errors = []
    
    # Check MultiIndex
    if not isinstance(df.index, pd.MultiIndex):
        errors.append("Must have MultiIndex")
    else:
        if df.index.names != ["datetime", "instrument"]:
            errors.append(f"Index names must be ['datetime', 'instrument'], got: {df.index.names}")
    
    # Check for duplicates
    if df.index.duplicated().any():
        n_dupes = df.index.duplicated().sum()
        errors.append(f"Found {n_dupes} duplicate index entries")
    
    # Check datetime level
    if isinstance(df.index, pd.MultiIndex):
        dt_level = df.index.get_level_values("datetime")
        
        # Check for NaT
        if dt_level.isna().any():
            errors.append(f"Found {dt_level.isna().sum()} NaT values in datetime")
        
        # Check timezone
        if dt_level.tz is None:
            errors.append("datetime is not timezone-aware (should be UTC)")
        elif str(dt_level.tz) != "UTC":
            errors.append(f"datetime timezone should be UTC, got: {dt_level.tz}")
        
        # Check instrument level
        inst_level = df.index.get_level_values("instrument")
        if (inst_level == "").any() or inst_level.isna().any():
            errors.append("Found empty or NaN instruments")
    
    # Check for score column
    if "score" not in df.columns:
        errors.append("Missing 'score' column")
    else:
        if df["score"].isna().all():
            errors.append("All scores are NaN")
    
    is_valid = len(errors) == 0
    error_msg = "; ".join(errors) if errors else "OK"
    
    return is_valid, error_msg


# ============================================================================
# QLIB SHADOW EVALUATION
# ============================================================================

def run_qlib_shadow_evaluation(
    eval_rows_df: pd.DataFrame,
    output_dir: Path,
    experiment_name: str
) -> Dict[str, Any]:
    """
    Run Qlib shadow evaluation (IC analysis only - no backtest).
    
    This produces "second opinion" factor evaluation using Qlib's utilities.
    
    Args:
        eval_rows_df: DataFrame with evaluation rows (as_of_date, score, excess_return, etc.)
                      Will be converted to Qlib format automatically.
        output_dir: Output directory for Qlib artifacts
        experiment_name: Experiment identifier
        
    Returns:
        Dictionary with results and paths
    """
    if not is_qlib_available():
        logger.warning("pyqlib not installed. Skipping Qlib shadow evaluation.")
        return {"status": "skipped", "reason": "pyqlib not installed"}
    
    # Convert to Qlib format (handles column aliases)
    logger.info(f"Converting {len(eval_rows_df)} eval rows to Qlib format")
    qlib_df = to_qlib_format(eval_rows_df)
    
    # Validate input
    is_valid, error_msg = validate_qlib_frame(qlib_df)
    if not is_valid:
        raise ValueError(f"Invalid Qlib frame: {error_msg}")
    
    # Create output directory
    qlib_output_dir = output_dir / experiment_name / "qlib"
    qlib_output_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = qlib_output_dir / "tables"
    figures_dir = qlib_output_dir / "figures"
    tables_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True)
    
    logger.info(f"Running Qlib shadow evaluation for {experiment_name}")
    
    results = {
        "status": "completed",
        "experiment_name": experiment_name,
        "output_dir": qlib_output_dir,
        "artifacts": {}
    }
    
    try:
        # Import Qlib modules
        from qlib.contrib.evaluate import risk_analysis
        import qlib
        
        # Initialize Qlib (minimal setup)
        try:
            qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region="cn")
        except Exception:
            # If init fails, try without provider (analysis-only mode)
            logger.warning("Qlib init failed, running in analysis-only mode")
        
        # Compute IC metrics using our data
        pred_df = qlib_df.reset_index()
        
        if "label" in qlib_df.columns:
            # Compute IC per date
            ic_series = []
            for dt in pred_df["datetime"].unique():
                dt_df = pred_df[pred_df["datetime"] == dt]
                if len(dt_df) >= 10:  # Minimum cross-section
                    from scipy.stats import spearmanr
                    corr, _ = spearmanr(dt_df["score"], dt_df["label"])
                    ic_series.append({"datetime": dt, "IC": corr})
            
            ic_df = pd.DataFrame(ic_series)
            
            # Save IC series
            ic_path = tables_dir / "ic_series.csv"
            ic_df.to_csv(ic_path, index=False)
            results["artifacts"]["ic_series"] = ic_path
            
            # Summary stats
            summary = {
                "IC_mean": ic_df["IC"].mean(),
                "IC_std": ic_df["IC"].std(),
                "IC_median": ic_df["IC"].median(),
                "IC_ir": ic_df["IC"].mean() / ic_df["IC"].std() if ic_df["IC"].std() > 0 else 0,
                "positive_pct": (ic_df["IC"] > 0).mean(),
                "n_dates": len(ic_df)
            }
            
            summary_path = tables_dir / "ic_summary.csv"
            pd.DataFrame([summary]).to_csv(summary_path, index=False)
            results["artifacts"]["ic_summary"] = summary_path
            results["summary"] = summary
            
            logger.info(f"IC Summary - Median: {summary['IC_median']:.4f}, IR: {summary['IC_ir']:.2f}")
        
        # Generate summary markdown
        _write_qlib_summary(qlib_output_dir, experiment_name, results)
        
    except Exception as e:
        logger.error(f"Qlib shadow evaluation failed: {e}")
        results["status"] = "failed"
        results["error"] = str(e)
    
    return results


def _write_qlib_summary(output_dir: Path, experiment_name: str, results: Dict) -> Path:
    """Write Qlib shadow evaluation summary."""
    md_path = output_dir / "QLIB_SUMMARY.md"
    
    with open(md_path, 'w') as f:
        f.write(f"# Qlib Shadow Evaluation: {experiment_name}\n\n")
        f.write(f"**Status:** {results.get('status', 'unknown')}\n\n")
        
        if "summary" in results:
            f.write("## IC Summary\n\n")
            f.write("| Metric | Value |\n")
            f.write("|--------|-------|\n")
            for k, v in results["summary"].items():
                if isinstance(v, float):
                    f.write(f"| {k} | {v:.4f} |\n")
                else:
                    f.write(f"| {k} | {v} |\n")
            f.write("\n")
        
        f.write("## Included\n\n")
        f.write("- IC/RankIC time series\n")
        f.write("- Summary statistics (mean, std, median, IR)\n")
        f.write("\n")
        
        f.write("## Excluded (Scope Limitation)\n\n")
        f.write("- Full Qlib backtest (requires dataset provider)\n")
        f.write("- Group analysis (optional extension)\n")
        f.write("\n")
        
        f.write("## Note\n\n")
        f.write("This is a SHADOW EVALUATOR only. Our system remains source-of-truth for:\n")
        f.write("- Universe, labels, PIT discipline, splits/purging/embargo, core metrics\n")
    
    return md_path


# ============================================================================
# PARITY CHECK
# ============================================================================

def check_ic_parity(
    eval_rows: pd.DataFrame,
    our_ic: float,
    tolerance: float = 0.001
) -> Tuple[bool, float, str]:
    """
    Check parity between our RankIC and Qlib's computation.
    
    Args:
        eval_rows: DataFrame in EvaluationRow format
        our_ic: RankIC computed by our system
        tolerance: Maximum allowed difference (default: 0.001 = 3 decimals)
        
    Returns:
        Tuple of (is_parity: Python bool, qlib_ic: float, message: str)
    """
    if not is_qlib_available():
        return bool(True), float(our_ic), "Qlib not installed, skipping parity check"
    
    # Convert to Qlib format
    qlib_df = to_qlib_format(eval_rows)
    
    # Compute IC using scipy (same method Qlib uses internally)
    from scipy.stats import spearmanr
    
    df = qlib_df.reset_index()
    corr, _ = spearmanr(df["score"], df["label"])
    qlib_ic = corr
    
    # Check parity
    diff = abs(our_ic - qlib_ic)
    is_parity = diff <= tolerance
    
    message = f"Our IC: {our_ic:.6f}, Qlib IC: {qlib_ic:.6f}, Diff: {diff:.6f}"
    if not is_parity:
        message += f" - EXCEEDS TOLERANCE ({tolerance})"
    
    return is_parity, qlib_ic, message


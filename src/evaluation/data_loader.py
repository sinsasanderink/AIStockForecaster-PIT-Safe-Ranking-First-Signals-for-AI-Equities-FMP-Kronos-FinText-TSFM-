"""
Evaluation Data Loader

Provides deterministic data loading for the evaluation pipeline:
1. Synthetic data generation for pipeline verification
2. Real data loading from DuckDB (when available)

CRITICAL: This module ensures consistent, reproducible data across runs.
"""

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
import pandas as pd
import numpy as np
import hashlib
import json
import logging

from .definitions import (
    EVALUATION_RANGE,
    HORIZONS_TRADING_DAYS,
    TRADING_DAYS_PER_YEAR,
)

logger = logging.getLogger(__name__)


# ============================================================================
# DATA LOADER CONFIGURATION
# ============================================================================

@dataclass(frozen=True)
class DataLoaderConfig:
    """Configuration for data loading."""
    start_date: date
    end_date: date
    n_stocks: int = 50  # Number of stocks in universe
    random_seed: int = 42  # For reproducibility
    
    # Feature noise parameters (for synthetic data)
    momentum_drift: float = 0.03  # Annual drift
    momentum_vol: float = 0.15   # Annual volatility
    return_noise: float = 0.02   # Noise in returns


# Default configurations
SYNTHETIC_CONFIG = DataLoaderConfig(
    start_date=EVALUATION_RANGE.eval_start,
    end_date=EVALUATION_RANGE.eval_end,
    n_stocks=50,
    random_seed=42,
)

SMOKE_CONFIG = DataLoaderConfig(
    start_date=date(2024, 1, 1),
    end_date=date(2024, 12, 31),
    n_stocks=20,
    random_seed=42,
)


# ============================================================================
# UNIVERSE GENERATION
# ============================================================================

def generate_universe(
    n_stocks: int = 50,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate a stable universe of stocks for evaluation.
    
    Returns DataFrame with:
    - ticker: Stock symbol
    - stable_id: Survivorship-safe identifier
    - sector: Sector assignment
    - base_adv: Base average daily volume
    """
    rng = np.random.RandomState(seed)
    
    # Create deterministic tickers
    sectors = ["Technology", "Semiconductors", "Software", "Cloud", "AI"]
    
    data = []
    for i in range(n_stocks):
        ticker = f"STOCK_{i:03d}"
        stable_id = f"STABLE_{ticker}"
        sector = sectors[i % len(sectors)]
        base_adv = rng.lognormal(mean=17, sigma=1.5)  # $1M - $1B range
        
        data.append({
            "ticker": ticker,
            "stable_id": stable_id,
            "sector": sector,
            "base_adv": base_adv,
        })
    
    return pd.DataFrame(data)


# ============================================================================
# SYNTHETIC FEATURE GENERATION
# ============================================================================

def generate_synthetic_features(
    config: DataLoaderConfig = SYNTHETIC_CONFIG,
) -> pd.DataFrame:
    """
    Generate synthetic features DataFrame for evaluation.
    
    This produces a deterministic dataset that matches the EvaluationRow contract:
    - date: As-of date
    - ticker: Stock symbol
    - stable_id: Survivorship-safe identifier
    - horizon: Forecast horizon (20, 60, 90)
    - excess_return: Forward excess return
    - mom_1m, mom_3m, mom_6m, mom_12m: Momentum features
    - adv_20d: Average daily volume
    - sector: Sector assignment
    - vix_percentile_252d, market_return_20d, market_vol_20d: Regime features
    - beta_252d: Beta to benchmark
    
    Returns:
        DataFrame with all required features for evaluation
    """
    rng = np.random.RandomState(config.random_seed)
    
    # Generate universe
    universe = generate_universe(config.n_stocks, config.random_seed)
    
    # Generate date grid (first trading day of each month)
    date_range = pd.date_range(
        start=config.start_date,
        end=config.end_date,
        freq='MS'  # Month start
    )
    
    logger.info(f"Generating synthetic data: {len(date_range)} dates x {config.n_stocks} stocks")
    
    all_data = []
    
    # Generate market regime time series (affects all stocks)
    market_returns = rng.normal(0.01, 0.04, size=len(date_range))  # Monthly market returns
    vix_levels = np.clip(rng.normal(20, 8, size=len(date_range)), 10, 80)
    market_vol = np.clip(rng.normal(0.15, 0.05, size=len(date_range)), 0.05, 0.5)
    
    for date_idx, d in enumerate(date_range):
        as_of_date = d.date()
        
        # Market regime for this date
        vix = vix_levels[date_idx]
        mkt_ret = market_returns[date_idx]
        mkt_vol = market_vol[date_idx]
        
        # VIX percentile (rolling estimation)
        vix_pct = min(100, max(0, (vix - 10) / 70 * 100))
        
        for _, stock in universe.iterrows():
            ticker = stock["ticker"]
            stable_id = stock["stable_id"]
            sector = stock["sector"]
            base_adv = stock["base_adv"]
            
            # Stock-specific seed for reproducibility
            stock_seed = hash(f"{ticker}_{as_of_date}") % (2**31)
            stock_rng = np.random.RandomState(stock_seed)
            
            # Generate momentum features with mean-reversion and momentum
            base_mom = stock_rng.normal(0, config.momentum_vol)
            
            mom_1m = stock_rng.normal(base_mom * 0.3, config.momentum_vol / 4)
            mom_3m = stock_rng.normal(base_mom * 0.5, config.momentum_vol / 3)
            mom_6m = stock_rng.normal(base_mom * 0.7, config.momentum_vol / 2)
            mom_12m = stock_rng.normal(base_mom, config.momentum_vol)
            
            # Beta (relatively stable)
            beta = 0.8 + stock_rng.normal(0, 0.3)
            beta = np.clip(beta, 0.3, 2.0)
            
            # ADV (with some variation)
            adv_20d = base_adv * (1 + stock_rng.normal(0, 0.2))
            
            # Generate excess returns for each horizon
            # Returns have weak signal from momentum (realistic IC ~0.03-0.05)
            
            # For simplicity, use horizon=20 as the primary excess_return
            # (The evaluation pipeline handles multi-horizon evaluation)
            horizon = 20  # Primary horizon for features
            
            # True signal component (weak)
            signal = 0.02 * mom_12m + 0.01 * mom_3m  # Weak momentum signal
            
            # Noise component (dominant)
            noise = stock_rng.normal(0, config.return_noise * np.sqrt(horizon / 20))
            
            # Market component
            market_component = beta * mkt_ret * (horizon / 20)
            
            excess_return = signal + noise + market_component * 0.1
            
            # Also generate horizon-specific returns for later use
            excess_return_20d = excess_return
            excess_return_60d = signal + stock_rng.normal(0, config.return_noise * np.sqrt(60 / 20)) + market_component * 3 * 0.1
            excess_return_90d = signal + stock_rng.normal(0, config.return_noise * np.sqrt(90 / 20)) + market_component * 4.5 * 0.1
            
            all_data.append({
                "date": as_of_date,
                "ticker": ticker,
                "stable_id": stable_id,
                "excess_return": excess_return,  # Primary excess return
                "excess_return_20d": excess_return_20d,
                "excess_return_60d": excess_return_60d,
                "excess_return_90d": excess_return_90d,
                "mom_1m": mom_1m,
                "mom_3m": mom_3m,
                "mom_6m": mom_6m,
                "mom_12m": mom_12m,
                "adv_20d": adv_20d,
                "sector": sector,
                "vix_percentile_252d": vix_pct,
                "market_return_20d": mkt_ret,
                "market_vol_20d": mkt_vol,
                "beta_252d": beta,
            })
    
    df = pd.DataFrame(all_data)
    
    logger.info(f"Generated {len(df)} rows ({df['date'].nunique()} dates x {df['ticker'].nunique()} stocks)")
    
    return df


# ============================================================================
# DUCKDB DEFAULT PATH
# ============================================================================

DEFAULT_DUCKDB_PATH = Path("data/features.duckdb")


# ============================================================================
# DATA LOADING UTILITIES
# ============================================================================

def load_features_from_duckdb(
    db_path: Path,
    eval_start: Optional[date] = None,
    eval_end: Optional[date] = None,
    horizons: List[int] = None,
    require_all_horizons: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Load features from DuckDB feature store.
    
    This function:
    1. Loads features, labels, and regime tables SEPARATELY
    2. Pivots labels from long to WIDE format (one column per horizon)
    3. Merges to produce EXACTLY ONE ROW per (date, ticker)
    4. Returns DataFrame matching run_experiment contract
    
    Output format:
    - One row per (date, ticker)
    - Label columns: excess_return_20d, excess_return_60d, excess_return_90d
    - Primary excess_return = excess_return_20d (for backward compatibility)
    
    Args:
        db_path: Path to DuckDB database
        eval_start: Evaluation start date (default: EVALUATION_RANGE.eval_start)
        eval_end: Evaluation end date (default: EVALUATION_RANGE.eval_end)
        horizons: Label horizons to include (default: all)
        require_all_horizons: If True, only include dates where all horizons have labels
    
    Returns:
        Tuple of (features_df, metadata_dict)
    
    Raises:
        RuntimeError: If database doesn't exist or is invalid
    """
    import duckdb
    
    # Default values
    if eval_start is None:
        eval_start = EVALUATION_RANGE.eval_start
    if eval_end is None:
        eval_end = EVALUATION_RANGE.eval_end
    if horizons is None:
        horizons = list(HORIZONS_TRADING_DAYS)
    
    # Validate db exists
    if not db_path.exists():
        raise RuntimeError(
            f"DuckDB feature store not found at: {db_path}\n\n"
            f"To build the feature store, run:\n"
            f"  export FMP_KEYS='your_fmp_api_key'\n"
            f"  python scripts/build_features_duckdb.py\n\n"
            f"This requires an FMP Premium API key."
        )
    
    logger.info(f"Loading features from DuckDB: {db_path}")
    
    conn = duckdb.connect(str(db_path), read_only=True)
    
    try:
        # Verify tables exist
        tables = conn.execute("SHOW TABLES").fetchall()
        table_names = {t[0] for t in tables}
        required_tables = {"features", "labels", "regime", "metadata"}
        missing_tables = required_tables - table_names
        
        if missing_tables:
            raise RuntimeError(f"DuckDB missing required tables: {missing_tables}")
        
        # Load metadata
        metadata_rows = conn.execute("SELECT key, value FROM metadata").fetchall()
        db_metadata = {row[0]: row[1] for row in metadata_rows}
        
        # Parse JSON values
        for key in ["horizons"]:
            if key in db_metadata and isinstance(db_metadata[key], str):
                try:
                    db_metadata[key] = json.loads(db_metadata[key])
                except json.JSONDecodeError:
                    pass
        
        # =====================================================================
        # STEP 1: Load features (already one row per (date, ticker))
        # =====================================================================
        features_query = f"""
            SELECT 
                date,
                ticker,
                stable_id,
                mom_1m,
                mom_3m,
                mom_6m,
                mom_12m,
                adv_20d,
                vol_20d,
                vol_60d
            FROM features
            WHERE date >= '{eval_start.isoformat()}'
              AND date <= '{eval_end.isoformat()}'
            ORDER BY date, ticker
        """
        features_df = conn.execute(features_query).fetchdf()
        logger.info(f"Loaded {len(features_df)} feature rows ({features_df['ticker'].nunique()} tickers)")
        
        # =====================================================================
        # STEP 2: Load labels (long format: multiple rows per (as_of_date, ticker))
        # =====================================================================
        horizons_str = ",".join(str(h) for h in horizons)
        labels_query = f"""
            SELECT 
                as_of_date,
                ticker,
                stable_id,
                horizon,
                excess_return,
                label_matured_at,
                label_version
            FROM labels
            WHERE as_of_date >= '{eval_start.isoformat()}'
              AND as_of_date <= '{eval_end.isoformat()}'
              AND horizon IN ({horizons_str})
            ORDER BY as_of_date, ticker, horizon
        """
        labels_df = conn.execute(labels_query).fetchdf()
        logger.info(f"Loaded {len(labels_df)} label rows (long format)")
        
        # DEBUG: Log horizon distribution BEFORE pivot
        logger.info("Label horizon distribution per (as_of_date, ticker):")
        horizon_counts = labels_df.groupby(["as_of_date", "ticker"]).size()
        logger.info(f"  Value counts:\n{horizon_counts.value_counts().head(10)}")
        
        # =====================================================================
        # STEP 3: Pivot labels to WIDE format
        # =====================================================================
        # We want one row per (as_of_date, ticker) with columns:
        #   excess_return_20d, excess_return_60d, excess_return_90d
        #   label_matured_at_20d, label_matured_at_60d, label_matured_at_90d
        
        # First dedupe labels (keep last if duplicates exist)
        labels_df = labels_df.sort_values(["as_of_date", "ticker", "horizon"])
        labels_df = labels_df.drop_duplicates(
            ["as_of_date", "ticker", "horizon"], keep="last"
        ).reset_index(drop=True)
        
        # Pivot excess_return
        excess_return_wide = labels_df.pivot_table(
            index=["as_of_date", "ticker", "stable_id"],
            columns="horizon",
            values="excess_return",
            aggfunc="last",  # Deterministic: use last value if duplicates
        ).reset_index()
        
        # Rename columns: horizon -> excess_return_{horizon}d
        excess_return_wide.columns = [
            f"excess_return_{c}d" if isinstance(c, (int, np.integer)) else c
            for c in excess_return_wide.columns
        ]
        
        # Pivot label_matured_at
        matured_at_wide = labels_df.pivot_table(
            index=["as_of_date", "ticker"],
            columns="horizon",
            values="label_matured_at",
            aggfunc="last",
        ).reset_index()
        
        matured_at_wide.columns = [
            f"label_matured_at_{c}d" if isinstance(c, (int, np.integer)) else c
            for c in matured_at_wide.columns
        ]
        
        # Merge the wide tables
        labels_wide = excess_return_wide.merge(
            matured_at_wide,
            on=["as_of_date", "ticker"],
            how="left",
        )
        
        # Rename as_of_date -> date for merge with features
        labels_wide = labels_wide.rename(columns={"as_of_date": "date"})
        
        logger.info(f"Pivoted labels to wide: {len(labels_wide)} rows (one per (date, ticker))")
        
        # Verify no duplicates in labels_wide
        n_label_dups = labels_wide.duplicated(["date", "ticker"]).sum()
        if n_label_dups > 0:
            logger.error(f"Labels still have {n_label_dups} duplicates after pivot!")
            # Debug: show sample duplicates
            dup_mask = labels_wide.duplicated(["date", "ticker"], keep=False)
            sample_dups = labels_wide[dup_mask].head(20)
            logger.error(f"Sample duplicate labels:\n{sample_dups}")
            raise RuntimeError(f"Labels have {n_label_dups} duplicate (date, ticker) after pivot")
        
        # =====================================================================
        # STEP 4: Load regime data
        # =====================================================================
        regime_query = f"""
            SELECT 
                date,
                market_return_20d,
                market_vol_20d,
                vix_percentile_252d
            FROM regime
            WHERE date >= '{eval_start.isoformat()}'
              AND date <= '{eval_end.isoformat()}'
            ORDER BY date
        """
        regime_df = conn.execute(regime_query).fetchdf()
        logger.info(f"Loaded {len(regime_df)} regime rows")
        
        # =====================================================================
        # STEP 5: Merge features + labels_wide + regime
        # =====================================================================
        # Merge features with labels (one-to-one)
        df = features_df.merge(
            labels_wide,
            on=["date", "ticker"],
            how="inner",  # Only keep rows with both features AND labels
            suffixes=("", "_label"),
        )
        
        # Use stable_id from features if both have it
        if "stable_id_label" in df.columns:
            df = df.drop(columns=["stable_id_label"])
        
        # Merge with regime (left join, regime may have fewer dates)
        df = df.merge(
            regime_df,
            on="date",
            how="left",
        )
        
        logger.info(f"After merge: {len(df)} rows")
        
        # =====================================================================
        # STEP 6: If require_all_horizons, filter appropriately
        # =====================================================================
        if require_all_horizons and len(horizons) > 1:
            # Check all horizon columns are non-null
            horizon_cols = [f"excess_return_{h}d" for h in horizons]
            available_cols = [c for c in horizon_cols if c in df.columns]
            
            if available_cols:
                before_filter = len(df)
                mask = df[available_cols].notna().all(axis=1)
                df = df[mask].reset_index(drop=True)
                logger.info(f"Filtered to rows with all horizons: {before_filter} -> {len(df)}")
        
        # =====================================================================
        # STEP 7: Add backward-compatible excess_return column (primary = 20d)
        # =====================================================================
        if "excess_return_20d" in df.columns:
            df["excess_return"] = df["excess_return_20d"]
        elif "excess_return_60d" in df.columns:
            df["excess_return"] = df["excess_return_60d"]
        elif "excess_return_90d" in df.columns:
            df["excess_return"] = df["excess_return_90d"]
        
        # =====================================================================
        # STEP 8: Add sector and beta
        # =====================================================================
        try:
            from ..universe.ai_stocks import get_category_for_ticker
            df["sector"] = df["ticker"].apply(lambda t: get_category_for_ticker(t) or "unknown")
        except ImportError:
            df["sector"] = "unknown"
        
        # Add beta placeholder
        df["beta_252d"] = 1.0
        
        if len(df) == 0:
            raise RuntimeError(
                f"No data found in DuckDB for date range {eval_start} to {eval_end}"
            )
        
        # =====================================================================
        # FINAL VALIDATION: Ensure exactly one row per (date, ticker)
        # =====================================================================
        n_final_dups = df.duplicated(["date", "ticker"]).sum()
        unique_date_ticker = df[["date", "ticker"]].drop_duplicates()
        
        logger.info(f"Final validation: {len(df)} total rows, {len(unique_date_ticker)} unique (date, ticker)")
        
        if n_final_dups > 0:
            logger.error(f"STILL have {n_final_dups} duplicate (date, ticker) after all processing!")
            # Show samples
            dup_mask = df.duplicated(["date", "ticker"], keep=False)
            dup_samples = df[dup_mask].groupby(["date", "ticker"]).size().reset_index(name="count")
            logger.error(f"Sample duplicates:\n{dup_samples.head(10)}")
            raise RuntimeError(f"Failed to produce unique (date, ticker): {n_final_dups} duplicates remain")
        
        # Build metadata
        metadata = {
            "source": "duckdb",
            "db_path": str(db_path),
            "schema_version": db_metadata.get("schema_version", "unknown"),
            "build_timestamp": db_metadata.get("build_timestamp", "unknown"),
            "n_rows": len(df),
            "n_dates": df["date"].nunique(),
            "n_stocks": df["ticker"].nunique(),
            "horizons": horizons,
            "date_range": {
                "min": df["date"].min().isoformat() if len(df) > 0 else None,
                "max": df["date"].max().isoformat() if len(df) > 0 else None,
            },
            "db_data_hash": db_metadata.get("data_hash", "unknown"),
            "label_format": "wide",  # Document we're using wide format
        }
        
    finally:
        conn.close()
    
    # Compute hash of loaded data
    data_hash = compute_data_hash(df)
    metadata["data_hash"] = data_hash
    
    logger.info(f"Loaded {len(df)} rows from DuckDB (wide format), hash={data_hash[:12]}")
    
    return df, metadata


def load_features_for_evaluation(
    source: str = "synthetic",
    config: Optional[DataLoaderConfig] = None,
    db_path: Optional[Path] = None,
    eval_start: Optional[date] = None,
    eval_end: Optional[date] = None,
    horizons: List[int] = None,
    require_all_horizons: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Load features for evaluation.
    
    Args:
        source: "synthetic", "duckdb", or "auto" (uses duckdb if available, else synthetic)
        config: DataLoaderConfig (uses SYNTHETIC_CONFIG if None, only for synthetic)
        db_path: Path to DuckDB database (uses DEFAULT_DUCKDB_PATH if None)
        eval_start: Evaluation start date (only for duckdb)
        eval_end: Evaluation end date (only for duckdb)
        horizons: Label horizons (only for duckdb)
        require_all_horizons: Filter to dates with all horizons (only for duckdb)
    
    Returns:
        Tuple of (features_df, metadata_dict)
    
    Raises:
        RuntimeError: If duckdb source and database doesn't exist
    """
    if config is None:
        config = SYNTHETIC_CONFIG
    
    if db_path is None:
        db_path = DEFAULT_DUCKDB_PATH
    
    # Handle auto mode
    if source == "auto":
        if db_path.exists():
            source = "duckdb"
            logger.info(f"Auto mode: found {db_path}, using duckdb")
        else:
            source = "synthetic"
            logger.info(f"Auto mode: {db_path} not found, using synthetic")
    
    if source == "synthetic":
        features_df = generate_synthetic_features(config)
        metadata = {
            "source": "synthetic",
            "config": {
                "start_date": config.start_date.isoformat(),
                "end_date": config.end_date.isoformat(),
                "n_stocks": config.n_stocks,
                "random_seed": config.random_seed,
            },
            "n_rows": len(features_df),
            "n_dates": features_df["date"].nunique(),
            "n_stocks": features_df["ticker"].nunique(),
            "date_range": {
                "min": features_df["date"].min().isoformat(),
                "max": features_df["date"].max().isoformat(),
            },
        }
        
        # Compute data hash for reproducibility tracking
        data_hash = compute_data_hash(features_df)
        metadata["data_hash"] = data_hash
        
    elif source == "duckdb":
        features_df, metadata = load_features_from_duckdb(
            db_path=db_path,
            eval_start=eval_start,
            eval_end=eval_end,
            horizons=horizons,
            require_all_horizons=require_all_horizons,
        )
    
    else:
        raise ValueError(f"Unknown source: {source}. Use 'synthetic', 'duckdb', or 'auto'")
    
    logger.info(f"Loaded features: {metadata['n_rows']} rows, hash={metadata.get('data_hash', 'N/A')[:12]}")
    
    return features_df, metadata


def check_duckdb_available(db_path: Optional[Path] = None) -> bool:
    """
    Check if DuckDB feature store is available.
    
    Args:
        db_path: Path to check (uses DEFAULT_DUCKDB_PATH if None)
    
    Returns:
        True if database exists and is valid
    """
    if db_path is None:
        db_path = DEFAULT_DUCKDB_PATH
    
    if not db_path.exists():
        return False
    
    try:
        import duckdb
        conn = duckdb.connect(str(db_path), read_only=True)
        tables = conn.execute("SHOW TABLES").fetchall()
        conn.close()
        
        table_names = {t[0] for t in tables}
        required = {"features", "labels", "regime", "metadata"}
        return required.issubset(table_names)
    except Exception:
        return False


def compute_data_hash(df: pd.DataFrame) -> str:
    """
    Compute deterministic hash of DataFrame for reproducibility tracking.
    """
    # Sort for determinism
    sort_cols = ["date", "ticker"]
    if "horizon" in df.columns:
        sort_cols.append("horizon")
    df_sorted = df.sort_values(sort_cols).reset_index(drop=True)
    
    # Create hash from key columns
    hash_cols = ["date", "ticker", "stable_id", "excess_return", "mom_12m"]
    available_cols = [c for c in hash_cols if c in df_sorted.columns]
    
    # Convert to string and hash
    data_str = df_sorted[available_cols].to_csv(index=False)
    return hashlib.sha256(data_str.encode()).hexdigest()


def validate_features_for_evaluation(
    df: pd.DataFrame,
    strict: bool = True,
) -> Dict[str, Any]:
    """
    Validate features DataFrame meets evaluation requirements.
    
    Expects WIDE format: one row per (date, ticker) with:
    - excess_return_20d, excess_return_60d, excess_return_90d
    - excess_return (backward-compatible, typically = excess_return_20d)
    
    Args:
        df: Features DataFrame
        strict: If True, raise on validation failures
    
    Returns:
        Validation result dict
    """
    issues = []
    warnings = []
    
    # DEBUG: Log shape and unique counts before validation
    logger.info(f"Validating features: {len(df)} rows")
    logger.info(f"  Unique (date, ticker): {df[['date', 'ticker']].drop_duplicates().shape[0] if 'date' in df.columns and 'ticker' in df.columns else 'N/A'}")
    
    # Required columns (horizon is NOT required - we use wide format)
    # excess_return is required for backward compatibility
    required_cols = [
        "date", "ticker", "stable_id",
        "excess_return", "mom_12m", "adv_20d"
    ]
    
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        issues.append(f"Missing required columns: {missing_cols}")
    
    # Check for momentum columns (needed for baselines)
    momentum_cols = ["mom_1m", "mom_3m", "mom_6m", "mom_12m"]
    missing_mom = [c for c in momentum_cols if c not in df.columns]
    if missing_mom:
        issues.append(f"Missing momentum columns: {missing_mom}")
    
    # Check for duplicates - CRITICAL for wide format
    if "date" in df.columns and "ticker" in df.columns:
        dup_check = df.groupby(["date", "ticker"]).size()
        n_dups = (dup_check > 1).sum()
        
        if n_dups > 0:
            total_extra_rows = (dup_check - 1).clip(lower=0).sum()
            issues.append(f"Found {total_extra_rows} duplicate (date, ticker) combinations")
            
            # DEBUG: Show sample duplicates
            dup_keys = dup_check[dup_check > 1].head(10)
            logger.error(f"Sample duplicate (date, ticker) keys with counts:\n{dup_keys}")
            
            # Check if this looks like a long-format issue (horizon column present)
            if "horizon" in df.columns:
                logger.error("DataFrame has 'horizon' column - appears to be LONG format, not WIDE!")
                logger.error("This is likely a bug in data loading - labels should be pivoted WIDE before merge")
                horizon_dist = df.groupby(["date", "ticker"])["horizon"].nunique()
                logger.error(f"Horizons per (date, ticker): \n{horizon_dist.value_counts().head()}")
    
    # Check date range
    if "date" in df.columns:
        min_date = df["date"].min()
        max_date = df["date"].max()
        
        # Convert to date for comparison if needed
        if hasattr(min_date, 'date'):
            min_date = min_date.date()
        if hasattr(max_date, 'date'):
            max_date = max_date.date()
        
        if min_date > EVALUATION_RANGE.eval_start:
            warnings.append(f"Data starts after evaluation range: {min_date} > {EVALUATION_RANGE.eval_start}")
        
        if max_date < EVALUATION_RANGE.eval_end:
            warnings.append(f"Data ends before evaluation range: {max_date} < {EVALUATION_RANGE.eval_end}")
    
    # Check for wide label columns (new format)
    wide_label_cols = [f"excess_return_{h}d" for h in HORIZONS_TRADING_DAYS]
    available_wide_cols = [c for c in wide_label_cols if c in df.columns]
    if available_wide_cols:
        logger.info(f"  Wide label columns found: {available_wide_cols}")
    
    # Check for NaN values
    for col in ["excess_return", "mom_12m"]:
        if col in df.columns:
            nan_pct = df[col].isna().mean() * 100
            if nan_pct > 10:
                warnings.append(f"High NaN rate in {col}: {nan_pct:.1f}%")
    
    result = {
        "valid": len(issues) == 0,
        "issues": issues,
        "warnings": warnings,
        "stats": {
            "n_rows": len(df),
            "n_dates": df["date"].nunique() if "date" in df.columns else 0,
            "n_stocks": df["ticker"].nunique() if "ticker" in df.columns else 0,
            "unique_date_ticker": df[["date", "ticker"]].drop_duplicates().shape[0] if "date" in df.columns and "ticker" in df.columns else 0,
        }
    }
    
    if strict and not result["valid"]:
        raise ValueError(f"Features validation failed: {issues}")
    
    return result


# ============================================================================
# MANIFEST GENERATION
# ============================================================================

def generate_data_manifest(
    features_df: pd.DataFrame,
    metadata: Dict[str, Any],
    output_path: Path,
) -> Dict[str, Any]:
    """
    Generate and save a data manifest for reproducibility.
    
    Args:
        features_df: Features DataFrame
        metadata: Metadata from data loading
        output_path: Path to save manifest
    
    Returns:
        Manifest dict
    """
    import platform
    import sys
    
    manifest = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "data": metadata,
        "environment": {
            "python_version": sys.version,
            "platform": platform.platform(),
        },
        "validation": validate_features_for_evaluation(features_df, strict=False),
    }
    
    # Save manifest
    manifest_path = output_path / "DATA_MANIFEST.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    
    logger.info(f"Saved data manifest to {manifest_path}")
    
    return manifest


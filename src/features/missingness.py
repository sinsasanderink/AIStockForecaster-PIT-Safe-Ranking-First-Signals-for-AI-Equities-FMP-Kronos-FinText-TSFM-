"""
Missingness Masks (Section 5.6)
===============================

Tracks data availability and creates explicit "known at time T" indicators.

PHILOSOPHY:
- Missingness is a SIGNAL, not just noise to impute away
- "Not known yet" ≠ "zero" or "mean"
- Missing data patterns often correlate with other factors
- Track coverage statistics for quality monitoring

FEATURES:
- Per-feature availability masks
- Data staleness indicators
- Coverage statistics by date/ticker
- Freshness metrics

ALL FEATURES ARE:
- PIT-safe: Only consider data with observed_at <= asof
- First-class signals: Missingness itself can be predictive
- Monitored: Track coverage over time for drift detection

USAGE:
    from src.features.missingness import MissingnessTracker
    
    tracker = MissingnessTracker()
    
    # Check feature availability
    masks = tracker.compute_masks(
        features_df=my_features,
        asof_date=date(2024, 1, 15),
    )
    
    # Get coverage statistics
    stats = tracker.compute_coverage_stats(features_df)
"""

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple, Set
from pathlib import Path

import pandas as pd
import numpy as np
import pytz

logger = logging.getLogger(__name__)

# Standard timezone
UTC = pytz.UTC

# Feature categories for tracking
FEATURE_CATEGORIES = {
    "price": [
        "mom_1m", "mom_3m", "mom_6m", "mom_12m",
        "vol_20d", "vol_60d", "vol_of_vol",
        "max_drawdown_60d", "dist_from_high_60d",
        "rel_strength_1m", "rel_strength_3m",
        "beta_252d", "adv_20d", "adv_60d", "vol_adj_adv",
    ],
    "fundamental": [
        "pe_zscore_3y", "ps_zscore_3y",
        "pe_vs_sector", "ps_vs_sector",
        "gross_margin_vs_sector", "operating_margin_vs_sector",
        "revenue_growth_vs_sector",
        "roe_zscore", "roa_zscore",
    ],
    "event": [
        "days_to_earnings", "days_since_earnings",
        "in_pead_window", "pead_window_day",
        "last_surprise_pct", "avg_surprise_4q",
        "surprise_streak", "surprise_zscore",
        "earnings_vol", "days_since_10k", "days_since_10q",
    ],
    "regime": [
        "vix_level", "vix_percentile", "vix_change_5d", "vix_regime",
        "market_return_5d", "market_return_21d", "market_return_63d",
        "market_vol_21d", "market_regime",
        "above_ma_50", "above_ma_200", "ma_50_200_cross",
        "tech_vs_staples", "tech_vs_utilities", "risk_on_indicator",
    ],
}

# Staleness thresholds (in days)
STALENESS_THRESHOLDS = {
    "price": 1,        # Price data should be daily
    "fundamental": 95,  # Quarterly, ~90 days + buffer
    "event": 365,       # Events can be sparse
    "regime": 1,        # Market data should be daily
}


@dataclass
class MissingnessFeatures:
    """
    Missingness features for a single stock on a single date.
    """
    ticker: str
    date: date
    
    # Overall coverage metrics
    total_features: int = 0
    missing_features: int = 0
    coverage_pct: float = 1.0  # 1 = fully available
    
    # Category-level coverage
    price_coverage: float = 1.0
    fundamental_coverage: float = 1.0
    event_coverage: float = 1.0
    regime_coverage: float = 1.0
    
    # Staleness indicators (days since last update)
    fundamental_staleness_days: Optional[int] = None
    event_staleness_days: Optional[int] = None
    
    # Specific missing flags (most important features)
    has_price_data: bool = True
    has_fundamental_data: bool = True
    has_earnings_data: bool = True
    has_sector_data: bool = True
    
    # New stock indicator (limited history)
    is_new_stock: bool = False  # < 252 days of history
    history_days: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "ticker": self.ticker,
            "date": self.date.isoformat(),
            "total_features": self.total_features,
            "missing_features": self.missing_features,
            "coverage_pct": self.coverage_pct,
            "price_coverage": self.price_coverage,
            "fundamental_coverage": self.fundamental_coverage,
            "event_coverage": self.event_coverage,
            "regime_coverage": self.regime_coverage,
            "fundamental_staleness_days": self.fundamental_staleness_days,
            "event_staleness_days": self.event_staleness_days,
            "has_price_data": self.has_price_data,
            "has_fundamental_data": self.has_fundamental_data,
            "has_earnings_data": self.has_earnings_data,
            "has_sector_data": self.has_sector_data,
            "is_new_stock": self.is_new_stock,
            "history_days": self.history_days,
        }


@dataclass
class CoverageStats:
    """
    Aggregate coverage statistics across universe.
    """
    date: date
    
    # Universe-level stats
    n_stocks: int = 0
    avg_coverage: float = 0.0
    min_coverage: float = 0.0
    max_coverage: float = 0.0
    
    # Category coverage
    avg_price_coverage: float = 0.0
    avg_fundamental_coverage: float = 0.0
    avg_event_coverage: float = 0.0
    avg_regime_coverage: float = 0.0
    
    # Problem counts
    n_missing_price: int = 0
    n_missing_fundamental: int = 0
    n_missing_earnings: int = 0
    n_new_stocks: int = 0
    
    # Feature-level stats
    features_above_95pct: int = 0
    features_below_50pct: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "date": self.date.isoformat(),
            "n_stocks": self.n_stocks,
            "avg_coverage": self.avg_coverage,
            "min_coverage": self.min_coverage,
            "max_coverage": self.max_coverage,
            "avg_price_coverage": self.avg_price_coverage,
            "avg_fundamental_coverage": self.avg_fundamental_coverage,
            "avg_event_coverage": self.avg_event_coverage,
            "avg_regime_coverage": self.avg_regime_coverage,
            "n_missing_price": self.n_missing_price,
            "n_missing_fundamental": self.n_missing_fundamental,
            "n_missing_earnings": self.n_missing_earnings,
            "n_new_stocks": self.n_new_stocks,
            "features_above_95pct": self.features_above_95pct,
            "features_below_50pct": self.features_below_50pct,
        }


class MissingnessTracker:
    """
    Tracks data availability and computes missingness masks.
    
    Key principle: Missing data is a SIGNAL, not just noise.
    
    Usage:
        tracker = MissingnessTracker()
        
        # Compute masks for a features DataFrame
        masks_df = tracker.compute_masks(features_df)
        
        # Get coverage statistics
        stats = tracker.compute_coverage_stats(features_df)
        
        # Add missingness features to feature DataFrame
        enhanced_df = tracker.enhance_features(features_df)
    """
    
    def __init__(
        self,
        feature_categories: Optional[Dict[str, List[str]]] = None,
        staleness_thresholds: Optional[Dict[str, int]] = None,
    ):
        """
        Initialize missingness tracker.
        
        Args:
            feature_categories: Mapping of category name to feature list
            staleness_thresholds: Mapping of category name to staleness days
        """
        self._categories = feature_categories or FEATURE_CATEGORIES
        self._staleness = staleness_thresholds or STALENESS_THRESHOLDS
        
        # Build reverse mapping (feature -> category)
        self._feature_to_category: Dict[str, str] = {}
        for category, features in self._categories.items():
            for feature in features:
                self._feature_to_category[feature] = category
        
        logger.info(f"MissingnessTracker initialized with {len(self._feature_to_category)} tracked features")
    
    def _count_category_coverage(
        self,
        row: pd.Series,
        category: str,
    ) -> Tuple[int, int]:
        """
        Count available and total features for a category.
        
        Returns:
            (available_count, total_count)
        """
        features = self._categories.get(category, [])
        total = 0
        available = 0
        
        for f in features:
            if f in row.index:
                total += 1
                if pd.notna(row[f]):
                    available += 1
        
        return available, total
    
    def compute_masks(
        self,
        df: pd.DataFrame,
        date_col: str = "date",
        ticker_col: str = "ticker",
    ) -> pd.DataFrame:
        """
        Compute missingness masks for each feature.
        
        Args:
            df: Features DataFrame (rows = stock-date, columns = features)
            date_col: Name of date column
            ticker_col: Name of ticker column
        
        Returns:
            DataFrame with same index, columns are {feature}_available (bool)
        """
        # Identify feature columns
        meta_cols = {date_col, ticker_col, "stable_id"}
        feature_cols = [c for c in df.columns if c not in meta_cols]
        
        # Create mask DataFrame
        masks = pd.DataFrame(index=df.index)
        
        for col in feature_cols:
            masks[f"{col}_available"] = df[col].notna()
        
        return masks
    
    def compute_stock_features(
        self,
        row: pd.Series,
        asof_date: date,
        ticker: str,
        history_days: Optional[int] = None,
        last_fundamental_date: Optional[date] = None,
        last_earnings_date: Optional[date] = None,
    ) -> MissingnessFeatures:
        """
        Compute missingness features for a single stock-date.
        
        Args:
            row: Row from features DataFrame
            asof_date: The date we're computing for
            ticker: Stock ticker
            history_days: Days of price history available
            last_fundamental_date: Last fundamental data date
            last_earnings_date: Last earnings date
        
        Returns:
            MissingnessFeatures object
        """
        features = MissingnessFeatures(ticker=ticker, date=asof_date)
        
        # Count total and missing features
        meta_cols = {"date", "ticker", "stable_id"}
        feature_cols = [c for c in row.index if c not in meta_cols and not c.endswith("_available")]
        
        features.total_features = len(feature_cols)
        features.missing_features = sum(1 for c in feature_cols if pd.isna(row.get(c)))
        features.coverage_pct = 1 - (features.missing_features / max(features.total_features, 1))
        
        # Category coverage
        for category in ["price", "fundamental", "event", "regime"]:
            available, total = self._count_category_coverage(row, category)
            coverage = available / max(total, 1)
            setattr(features, f"{category}_coverage", coverage)
        
        # Staleness indicators
        if last_fundamental_date:
            features.fundamental_staleness_days = (asof_date - last_fundamental_date).days
        
        if last_earnings_date:
            features.event_staleness_days = (asof_date - last_earnings_date).days
        
        # Specific availability flags
        price_avail, price_total = self._count_category_coverage(row, "price")
        features.has_price_data = price_avail > 0
        
        fund_avail, fund_total = self._count_category_coverage(row, "fundamental")
        features.has_fundamental_data = fund_avail > 0
        
        # Check for specific earnings fields
        earnings_fields = ["days_since_earnings", "last_surprise_pct"]
        features.has_earnings_data = any(
            pd.notna(row.get(f)) for f in earnings_fields if f in row.index
        )
        
        # Sector data check
        sector_fields = ["pe_vs_sector", "ps_vs_sector", "gross_margin_vs_sector"]
        features.has_sector_data = any(
            pd.notna(row.get(f)) for f in sector_fields if f in row.index
        )
        
        # New stock indicator
        if history_days is not None:
            features.history_days = history_days
            features.is_new_stock = history_days < 252  # Less than 1 year
        
        return features
    
    def compute_coverage_stats(
        self,
        df: pd.DataFrame,
        asof_date: Optional[date] = None,
        date_col: str = "date",
        ticker_col: str = "ticker",
    ) -> CoverageStats:
        """
        Compute aggregate coverage statistics for the universe.
        
        Args:
            df: Features DataFrame
            asof_date: Date to report stats for (defaults to max date)
            date_col: Name of date column
            ticker_col: Name of ticker column
        
        Returns:
            CoverageStats object
        """
        if df.empty:
            return CoverageStats(date=asof_date or date.today())
        
        # Filter to asof_date if provided
        if asof_date and date_col in df.columns:
            df = df[df[date_col] == asof_date]
        
        if df.empty:
            return CoverageStats(date=asof_date or date.today())
        
        # Determine date for stats
        if date_col in df.columns:
            stats_date = pd.to_datetime(df[date_col]).max().date()
        else:
            stats_date = asof_date or date.today()
        
        stats = CoverageStats(date=stats_date)
        stats.n_stocks = df[ticker_col].nunique() if ticker_col in df.columns else len(df)
        
        # Identify feature columns
        meta_cols = {date_col, ticker_col, "stable_id"}
        feature_cols = [c for c in df.columns if c not in meta_cols and not c.endswith("_available")]
        
        if not feature_cols:
            return stats
        
        # Per-row coverage
        row_coverage = df[feature_cols].notna().mean(axis=1)
        stats.avg_coverage = float(row_coverage.mean())
        stats.min_coverage = float(row_coverage.min())
        stats.max_coverage = float(row_coverage.max())
        
        # Category coverage
        for category in ["price", "fundamental", "event", "regime"]:
            cat_features = [f for f in self._categories.get(category, []) if f in feature_cols]
            if cat_features:
                cat_coverage = df[cat_features].notna().mean(axis=1).mean()
                setattr(stats, f"avg_{category}_coverage", float(cat_coverage))
        
        # Problem counts
        price_features = [f for f in self._categories.get("price", []) if f in feature_cols]
        if price_features:
            stats.n_missing_price = int((df[price_features].isna().all(axis=1)).sum())
        
        fund_features = [f for f in self._categories.get("fundamental", []) if f in feature_cols]
        if fund_features:
            stats.n_missing_fundamental = int((df[fund_features].isna().all(axis=1)).sum())
        
        if "days_since_earnings" in df.columns:
            stats.n_missing_earnings = int(df["days_since_earnings"].isna().sum())
        
        # Feature-level stats
        feature_coverage = df[feature_cols].notna().mean()
        stats.features_above_95pct = int((feature_coverage >= 0.95).sum())
        stats.features_below_50pct = int((feature_coverage < 0.50).sum())
        
        return stats
    
    def enhance_features(
        self,
        df: pd.DataFrame,
        date_col: str = "date",
        ticker_col: str = "ticker",
    ) -> pd.DataFrame:
        """
        Add missingness features to the feature DataFrame.
        
        This adds columns that can be used as features themselves,
        since missingness often carries information.
        
        Args:
            df: Features DataFrame
            date_col: Name of date column
            ticker_col: Name of ticker column
        
        Returns:
            DataFrame with additional missingness feature columns
        """
        result = df.copy()
        
        # Identify feature columns
        meta_cols = {date_col, ticker_col, "stable_id"}
        feature_cols = [c for c in df.columns if c not in meta_cols]
        
        # Overall coverage
        result["miss_coverage_pct"] = df[feature_cols].notna().mean(axis=1)
        result["miss_n_missing"] = df[feature_cols].isna().sum(axis=1)
        
        # Category coverage
        for category in ["price", "fundamental", "event", "regime"]:
            cat_features = [f for f in self._categories.get(category, []) if f in feature_cols]
            if cat_features:
                result[f"miss_{category}_coverage"] = df[cat_features].notna().mean(axis=1)
        
        # Key availability flags
        if "mom_1m" in df.columns:
            result["miss_has_momentum"] = df["mom_1m"].notna()
        
        if "pe_vs_sector" in df.columns:
            result["miss_has_sector_val"] = df["pe_vs_sector"].notna()
        
        if "days_since_earnings" in df.columns:
            result["miss_has_earnings"] = df["days_since_earnings"].notna()
        
        if "vix_level" in df.columns:
            result["miss_has_regime"] = df["vix_level"].notna()
        
        return result
    
    def get_missing_features_for_row(
        self,
        row: pd.Series,
        date_col: str = "date",
        ticker_col: str = "ticker",
    ) -> List[str]:
        """
        Get list of missing feature names for a single row.
        
        Useful for debugging and data quality investigation.
        """
        meta_cols = {date_col, ticker_col, "stable_id"}
        feature_cols = [c for c in row.index if c not in meta_cols]
        
        return [c for c in feature_cols if pd.isna(row.get(c))]
    
    def generate_coverage_report(
        self,
        df: pd.DataFrame,
        date_col: str = "date",
        ticker_col: str = "ticker",
    ) -> str:
        """
        Generate a human-readable coverage report.
        
        Args:
            df: Features DataFrame
            date_col: Name of date column
            ticker_col: Name of ticker column
        
        Returns:
            Formatted report string
        """
        stats = self.compute_coverage_stats(df, date_col=date_col, ticker_col=ticker_col)
        
        report = []
        report.append("=" * 60)
        report.append("FEATURE COVERAGE REPORT")
        report.append("=" * 60)
        report.append(f"Date: {stats.date}")
        report.append(f"Stocks: {stats.n_stocks}")
        report.append("")
        report.append("OVERALL COVERAGE:")
        report.append(f"  Average: {stats.avg_coverage:.1%}")
        report.append(f"  Min:     {stats.min_coverage:.1%}")
        report.append(f"  Max:     {stats.max_coverage:.1%}")
        report.append("")
        report.append("CATEGORY COVERAGE:")
        report.append(f"  Price:       {stats.avg_price_coverage:.1%}")
        report.append(f"  Fundamental: {stats.avg_fundamental_coverage:.1%}")
        report.append(f"  Event:       {stats.avg_event_coverage:.1%}")
        report.append(f"  Regime:      {stats.avg_regime_coverage:.1%}")
        report.append("")
        report.append("PROBLEMS:")
        report.append(f"  Missing all price data:       {stats.n_missing_price}")
        report.append(f"  Missing all fundamental data: {stats.n_missing_fundamental}")
        report.append(f"  Missing earnings data:        {stats.n_missing_earnings}")
        report.append(f"  New stocks (< 1y history):    {stats.n_new_stocks}")
        report.append("")
        report.append("FEATURE QUALITY:")
        report.append(f"  Features >= 95% coverage: {stats.features_above_95pct}")
        report.append(f"  Features < 50% coverage:  {stats.features_below_50pct}")
        report.append("=" * 60)
        
        return "\n".join(report)


# =============================================================================
# Helper Functions
# =============================================================================

def create_availability_mask(
    series: pd.Series,
) -> pd.Series:
    """Create boolean availability mask for a Series."""
    return series.notna()


def compute_feature_coverage(
    df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
) -> pd.Series:
    """
    Compute coverage percentage for each feature.
    
    Returns:
        Series with feature names as index, coverage pct as values
    """
    if feature_cols is None:
        feature_cols = df.columns.tolist()
    
    return df[feature_cols].notna().mean()


def get_missingness_tracker(
    feature_categories: Optional[Dict[str, List[str]]] = None,
) -> MissingnessTracker:
    """Factory function for MissingnessTracker."""
    return MissingnessTracker(feature_categories=feature_categories)


# =============================================================================
# CLI / Demo
# =============================================================================

if __name__ == "__main__":
    import sys
    
    print("=" * 60)
    print("MISSINGNESS TRACKER DEMO")
    print("=" * 60)
    
    # Create sample data with some missing values
    np.random.seed(42)
    
    sample_data = pd.DataFrame({
        "ticker": ["NVDA", "AMD", "MSFT", "INTC"],
        "date": [date(2024, 1, 15)] * 4,
        "mom_1m": [0.05, 0.03, np.nan, 0.02],
        "mom_3m": [0.15, 0.10, 0.08, np.nan],
        "vol_20d": [0.3, 0.35, 0.25, 0.28],
        "pe_vs_sector": [1.2, 0.8, np.nan, np.nan],
        "ps_vs_sector": [1.5, np.nan, 1.1, 0.9],
        "days_since_earnings": [30, 45, np.nan, 60],
        "vix_level": [15.0] * 4,  # Market-level, same for all
    })
    
    print(f"\nSample data with {len(sample_data)} stocks:")
    print(sample_data)
    
    tracker = MissingnessTracker()
    
    # Compute masks
    print("\n" + "-" * 40)
    print("AVAILABILITY MASKS:")
    print("-" * 40)
    masks = tracker.compute_masks(sample_data)
    print(masks)
    
    # Coverage stats
    print("\n" + "-" * 40)
    print("COVERAGE REPORT:")
    print("-" * 40)
    print(tracker.generate_coverage_report(sample_data))
    
    # Enhanced features
    print("\n" + "-" * 40)
    print("ENHANCED FEATURES (with missingness indicators):")
    print("-" * 40)
    enhanced = tracker.enhance_features(sample_data)
    miss_cols = [c for c in enhanced.columns if c.startswith("miss_")]
    print(enhanced[["ticker"] + miss_cols])
    
    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)


"""
Feature Hygiene & Redundancy Control (Section 5.7)
===================================================

Provides tools for ensuring feature quality and understanding redundancy.

KEY PRINCIPLE:
- Stability > VIF
- A feature with IC 0.04 once and −0.01 later is WORSE than IC 0.02 stable forever

COMPONENTS:
1. Cross-sectional standardization (z-score, rank)
2. Rolling Spearman correlation matrix
3. Feature clustering & block identification
4. VIF diagnostics (for tabular features)
5. Rolling IC stability checks
6. Sign consistency analysis

USAGE:
    from src.features.hygiene import FeatureHygiene
    
    hygiene = FeatureHygiene()
    
    # Standardize features
    std_df = hygiene.standardize_cross_sectional(features_df)
    
    # Compute correlation matrix
    corr_matrix = hygiene.compute_correlation_matrix(features_df)
    
    # Identify feature blocks
    blocks = hygiene.identify_feature_blocks(features_df)
    
    # VIF diagnostics
    vif_df = hygiene.compute_vif(features_df)
    
    # IC stability
    ic_stability = hygiene.compute_ic_stability(features_df, labels_df)
"""

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple, Set
from pathlib import Path
import warnings

import pandas as pd
import numpy as np
from scipy import stats
from scipy.cluster import hierarchy

logger = logging.getLogger(__name__)

# Default correlation threshold for feature clustering
CORRELATION_THRESHOLD = 0.7

# Default VIF threshold (diagnostic, not hard filter)
VIF_THRESHOLD = 5.0

# IC stability threshold (% of periods with consistent sign)
IC_SIGN_CONSISTENCY_THRESHOLD = 0.70


@dataclass
class FeatureBlock:
    """
    A group of highly correlated features.
    
    We identify blocks but DON'T automatically drop features.
    Blocks are for understanding, not auto-deletion.
    """
    block_id: int
    features: List[str]
    avg_correlation: float
    representative: str  # Feature with highest avg correlation to others
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "block_id": self.block_id,
            "features": self.features,
            "avg_correlation": self.avg_correlation,
            "representative": self.representative,
        }


@dataclass
class ICStabilityResult:
    """
    IC stability metrics for a single feature.
    """
    feature: str
    n_periods: int
    ic_values: List[float]
    ic_mean: float
    ic_std: float
    ic_sign_consistency: float  # % of periods with same sign as mean
    ic_positive_pct: float      # % of periods with positive IC
    is_stable: bool             # IC sign consistent >= threshold
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature": self.feature,
            "n_periods": self.n_periods,
            "ic_mean": self.ic_mean,
            "ic_std": self.ic_std,
            "ic_sign_consistency": self.ic_sign_consistency,
            "ic_positive_pct": self.ic_positive_pct,
            "is_stable": self.is_stable,
        }


@dataclass 
class VIFResult:
    """
    Variance Inflation Factor result for a feature.
    
    VIF > 5 suggests high multicollinearity.
    VIF > 10 is severe.
    
    NOTE: VIF is a DIAGNOSTIC, not a hard filter.
    High VIF features may still be valuable if stable.
    """
    feature: str
    vif: float
    is_high: bool  # VIF > threshold
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature": self.feature,
            "vif": self.vif,
            "is_high": self.is_high,
        }


class FeatureHygiene:
    """
    Feature hygiene and redundancy analysis.
    
    Provides tools for:
    - Cross-sectional standardization
    - Correlation analysis
    - Feature clustering
    - VIF diagnostics
    - IC stability checks
    """
    
    def __init__(
        self,
        correlation_threshold: float = CORRELATION_THRESHOLD,
        vif_threshold: float = VIF_THRESHOLD,
        ic_sign_threshold: float = IC_SIGN_CONSISTENCY_THRESHOLD,
    ):
        """
        Initialize feature hygiene.
        
        Args:
            correlation_threshold: Threshold for grouping features (default: 0.7)
            vif_threshold: VIF threshold for flagging (default: 5.0)
            ic_sign_threshold: IC sign consistency threshold (default: 0.70)
        """
        self.correlation_threshold = correlation_threshold
        self.vif_threshold = vif_threshold
        self.ic_sign_threshold = ic_sign_threshold
        
        logger.info(
            f"FeatureHygiene initialized: corr_thresh={correlation_threshold}, "
            f"vif_thresh={vif_threshold}, ic_sign_thresh={ic_sign_threshold}"
        )
    
    # =========================================================================
    # Cross-Sectional Standardization
    # =========================================================================
    
    def standardize_cross_sectional(
        self,
        df: pd.DataFrame,
        feature_cols: Optional[List[str]] = None,
        method: str = "zscore",
        date_col: str = "date",
    ) -> pd.DataFrame:
        """
        Apply cross-sectional standardization within each date.
        
        Args:
            df: Features DataFrame (rows = stock-date)
            feature_cols: Columns to standardize (auto-detect if None)
            method: 'zscore' or 'rank'
            date_col: Name of date column
        
        Returns:
            DataFrame with standardized features
        """
        result = df.copy()
        
        # Auto-detect feature columns
        if feature_cols is None:
            meta_cols = {date_col, "ticker", "stable_id"}
            feature_cols = [c for c in df.columns if c not in meta_cols 
                          and df[c].dtype in ['float64', 'float32', 'int64', 'int32']]
        
        if date_col not in df.columns:
            # No date column - standardize across all rows
            for col in feature_cols:
                if method == "zscore":
                    result[col] = self._zscore(df[col])
                else:
                    result[col] = self._rank(df[col])
        else:
            # Standardize within each date
            for col in feature_cols:
                if method == "zscore":
                    result[col] = df.groupby(date_col)[col].transform(self._zscore)
                else:
                    result[col] = df.groupby(date_col)[col].transform(self._rank)
        
        return result
    
    def _zscore(self, series: pd.Series) -> pd.Series:
        """Compute z-score, handling NaN."""
        mean = series.mean()
        std = series.std()
        if std == 0 or pd.isna(std):
            return pd.Series(0, index=series.index)
        return (series - mean) / std
    
    def _rank(self, series: pd.Series) -> pd.Series:
        """Compute percentile rank (0-1)."""
        return series.rank(pct=True, na_option="keep")
    
    # =========================================================================
    # Correlation Analysis
    # =========================================================================
    
    def compute_correlation_matrix(
        self,
        df: pd.DataFrame,
        feature_cols: Optional[List[str]] = None,
        method: str = "spearman",
    ) -> pd.DataFrame:
        """
        Compute correlation matrix for features.
        
        Args:
            df: Features DataFrame
            feature_cols: Columns to include (auto-detect if None)
            method: 'spearman' (rank) or 'pearson' (linear)
        
        Returns:
            Correlation matrix as DataFrame
        """
        if feature_cols is None:
            meta_cols = {"date", "ticker", "stable_id"}
            feature_cols = [c for c in df.columns if c not in meta_cols
                          and df[c].dtype in ['float64', 'float32', 'int64', 'int32']]
        
        return df[feature_cols].corr(method=method)
    
    def compute_rolling_correlation(
        self,
        df: pd.DataFrame,
        feature1: str,
        feature2: str,
        date_col: str = "date",
        window_days: int = 63,
    ) -> pd.DataFrame:
        """
        Compute rolling correlation between two features over time.
        
        Args:
            df: Features DataFrame
            feature1: First feature name
            feature2: Second feature name
            date_col: Date column name
            window_days: Rolling window in days
        
        Returns:
            DataFrame with date and correlation
        """
        # Group by date and compute correlation within each date
        dates = df[date_col].unique()
        dates = sorted(dates)
        
        correlations = []
        for i, d in enumerate(dates):
            if i < window_days:
                continue
            
            window_dates = dates[i-window_days:i+1]
            window_df = df[df[date_col].isin(window_dates)]
            
            if len(window_df) >= 20:  # Minimum samples
                corr = window_df[feature1].corr(window_df[feature2], method="spearman")
                correlations.append({"date": d, "correlation": corr})
        
        return pd.DataFrame(correlations)
    
    # =========================================================================
    # Feature Clustering
    # =========================================================================
    
    def identify_feature_blocks(
        self,
        df: pd.DataFrame,
        feature_cols: Optional[List[str]] = None,
        threshold: Optional[float] = None,
    ) -> List[FeatureBlock]:
        """
        Identify clusters of highly correlated features.
        
        Uses hierarchical clustering on correlation matrix.
        
        Args:
            df: Features DataFrame
            feature_cols: Columns to analyze
            threshold: Correlation threshold (default: self.correlation_threshold)
        
        Returns:
            List of FeatureBlock objects
        """
        threshold = threshold or self.correlation_threshold
        
        # Compute correlation matrix
        corr_matrix = self.compute_correlation_matrix(df, feature_cols)
        
        if corr_matrix.empty or len(corr_matrix) < 2:
            return []
        
        features = corr_matrix.columns.tolist()
        
        # Convert correlation to distance (1 - |correlation|)
        dist_matrix = 1 - np.abs(corr_matrix.values)
        np.fill_diagonal(dist_matrix, 0)
        
        # Handle any NaN values
        dist_matrix = np.nan_to_num(dist_matrix, nan=1.0)
        
        # Hierarchical clustering
        try:
            condensed = hierarchy.distance.squareform(dist_matrix)
            linkage = hierarchy.linkage(condensed, method="average")
            
            # Cut tree at threshold distance (1 - threshold)
            clusters = hierarchy.fcluster(linkage, t=1-threshold, criterion="distance")
        except Exception as e:
            logger.warning(f"Clustering failed: {e}")
            return []
        
        # Group features by cluster
        cluster_groups: Dict[int, List[str]] = {}
        for feature, cluster_id in zip(features, clusters):
            if cluster_id not in cluster_groups:
                cluster_groups[cluster_id] = []
            cluster_groups[cluster_id].append(feature)
        
        # Create FeatureBlock objects
        blocks = []
        for block_id, block_features in cluster_groups.items():
            if len(block_features) >= 2:  # Only blocks with multiple features
                # Compute average correlation within block
                block_corr = corr_matrix.loc[block_features, block_features]
                # Get upper triangle (excluding diagonal)
                mask = np.triu(np.ones_like(block_corr, dtype=bool), k=1)
                avg_corr = block_corr.where(mask).stack().mean()
                
                # Find representative (highest avg correlation to others)
                avg_corrs = block_corr.mean()
                representative = avg_corrs.idxmax()
                
                blocks.append(FeatureBlock(
                    block_id=block_id,
                    features=block_features,
                    avg_correlation=float(avg_corr),
                    representative=representative,
                ))
        
        return sorted(blocks, key=lambda b: -b.avg_correlation)
    
    # =========================================================================
    # VIF Diagnostics
    # =========================================================================
    
    def compute_vif(
        self,
        df: pd.DataFrame,
        feature_cols: Optional[List[str]] = None,
    ) -> List[VIFResult]:
        """
        Compute Variance Inflation Factor for features.
        
        VIF measures multicollinearity. High VIF (>5) suggests redundancy.
        
        NOTE: This is a DIAGNOSTIC. Don't auto-drop high VIF features.
        
        Args:
            df: Features DataFrame
            feature_cols: Columns to analyze
        
        Returns:
            List of VIFResult objects sorted by VIF descending
        """
        if feature_cols is None:
            meta_cols = {"date", "ticker", "stable_id"}
            feature_cols = [c for c in df.columns if c not in meta_cols
                          and df[c].dtype in ['float64', 'float32', 'int64', 'int32']]
        
        # Drop rows with any NaN
        clean_df = df[feature_cols].dropna()
        
        if len(clean_df) < len(feature_cols) + 10:
            logger.warning("Not enough data for VIF computation")
            return []
        
        results = []
        
        for i, col in enumerate(feature_cols):
            try:
                # VIF = 1 / (1 - R^2) where R^2 is from regressing col on other cols
                y = clean_df[col]
                X = clean_df[[c for c in feature_cols if c != col]]
                
                # Add constant
                X = X.assign(_const=1)
                
                # Compute R^2 using OLS
                try:
                    from numpy.linalg import lstsq
                    coeffs, residuals, rank, s = lstsq(X.values, y.values, rcond=None)
                    
                    ss_res = np.sum((y.values - X.values @ coeffs) ** 2)
                    ss_tot = np.sum((y.values - y.mean()) ** 2)
                    
                    if ss_tot == 0:
                        r_squared = 0
                    else:
                        r_squared = 1 - (ss_res / ss_tot)
                    
                    if r_squared >= 1:
                        vif = float('inf')
                    else:
                        vif = 1 / (1 - r_squared)
                except Exception:
                    vif = float('nan')
                
                results.append(VIFResult(
                    feature=col,
                    vif=vif,
                    is_high=vif > self.vif_threshold,
                ))
            except Exception as e:
                logger.debug(f"VIF computation failed for {col}: {e}")
        
        return sorted(results, key=lambda r: -r.vif if not np.isnan(r.vif) else 0)
    
    # =========================================================================
    # IC Stability Analysis
    # =========================================================================
    
    def compute_ic_stability(
        self,
        features_df: pd.DataFrame,
        labels_df: pd.DataFrame,
        feature_cols: Optional[List[str]] = None,
        label_col: str = "excess_return",
        date_col: str = "date",
        ticker_col: str = "ticker",
        min_periods: int = 12,
    ) -> List[ICStabilityResult]:
        """
        Compute IC (Information Coefficient) stability for features.
        
        IC = Spearman correlation between feature and forward return.
        
        Stability = % of periods where IC has same sign as mean IC.
        
        IMPORTANT: This is the most critical hygiene metric.
        A feature with IC 0.04 once and −0.01 later is worse than
        IC 0.02 stable forever.
        
        Args:
            features_df: Features DataFrame
            labels_df: Labels DataFrame with forward returns
            feature_cols: Feature columns to analyze
            label_col: Column with forward returns
            date_col: Date column
            ticker_col: Ticker column
            min_periods: Minimum periods for stability calc
        
        Returns:
            List of ICStabilityResult objects
        """
        if feature_cols is None:
            meta_cols = {date_col, ticker_col, "stable_id", label_col}
            feature_cols = [c for c in features_df.columns if c not in meta_cols
                          and features_df[c].dtype in ['float64', 'float32', 'int64', 'int32']]
        
        # Merge features and labels
        merged = features_df.merge(
            labels_df[[ticker_col, date_col, label_col]],
            on=[ticker_col, date_col],
            how="inner",
        )
        
        if merged.empty:
            logger.warning("No matching feature-label data")
            return []
        
        # Get unique dates
        dates = sorted(merged[date_col].unique())
        
        results = []
        
        for col in feature_cols:
            ic_values = []
            
            for d in dates:
                date_df = merged[merged[date_col] == d]
                
                if len(date_df) < 10:  # Need enough stocks
                    continue
                
                # Remove NaN
                valid = date_df[[col, label_col]].dropna()
                
                if len(valid) < 10:
                    continue
                
                # Compute Spearman IC
                ic, _ = stats.spearmanr(valid[col], valid[label_col])
                
                if not np.isnan(ic):
                    ic_values.append(ic)
            
            if len(ic_values) < min_periods:
                continue
            
            ic_mean = np.mean(ic_values)
            ic_std = np.std(ic_values)
            
            # Sign consistency: % of periods with same sign as mean
            if ic_mean > 0:
                sign_consistent = sum(1 for ic in ic_values if ic > 0) / len(ic_values)
            elif ic_mean < 0:
                sign_consistent = sum(1 for ic in ic_values if ic < 0) / len(ic_values)
            else:
                sign_consistent = 0.5
            
            # % positive
            positive_pct = sum(1 for ic in ic_values if ic > 0) / len(ic_values)
            
            results.append(ICStabilityResult(
                feature=col,
                n_periods=len(ic_values),
                ic_values=ic_values,
                ic_mean=ic_mean,
                ic_std=ic_std,
                ic_sign_consistency=sign_consistent,
                ic_positive_pct=positive_pct,
                is_stable=sign_consistent >= self.ic_sign_threshold,
            ))
        
        return sorted(results, key=lambda r: (-r.ic_sign_consistency, -abs(r.ic_mean)))
    
    def compute_single_ic(
        self,
        features_df: pd.DataFrame,
        labels_df: pd.DataFrame,
        feature_col: str,
        label_col: str = "excess_return",
        date_col: str = "date",
        ticker_col: str = "ticker",
    ) -> float:
        """
        Compute single IC value across all data.
        
        Returns:
            Spearman IC value
        """
        merged = features_df.merge(
            labels_df[[ticker_col, date_col, label_col]],
            on=[ticker_col, date_col],
            how="inner",
        )
        
        valid = merged[[feature_col, label_col]].dropna()
        
        if len(valid) < 30:
            return float('nan')
        
        ic, _ = stats.spearmanr(valid[feature_col], valid[label_col])
        return ic
    
    # =========================================================================
    # Reporting
    # =========================================================================
    
    def generate_hygiene_report(
        self,
        df: pd.DataFrame,
        feature_cols: Optional[List[str]] = None,
        labels_df: Optional[pd.DataFrame] = None,
    ) -> str:
        """
        Generate a comprehensive feature hygiene report.
        
        Args:
            df: Features DataFrame
            feature_cols: Feature columns to analyze
            labels_df: Optional labels for IC analysis
        
        Returns:
            Formatted report string
        """
        if feature_cols is None:
            meta_cols = {"date", "ticker", "stable_id"}
            feature_cols = [c for c in df.columns if c not in meta_cols
                          and df[c].dtype in ['float64', 'float32', 'int64', 'int32']]
        
        report = []
        report.append("=" * 60)
        report.append("FEATURE HYGIENE REPORT")
        report.append("=" * 60)
        report.append(f"Total features: {len(feature_cols)}")
        report.append(f"Total rows: {len(df)}")
        report.append("")
        
        # Correlation analysis
        report.append("-" * 40)
        report.append("CORRELATION ANALYSIS")
        report.append("-" * 40)
        
        corr_matrix = self.compute_correlation_matrix(df, feature_cols)
        
        # Find highly correlated pairs
        high_corr_pairs = []
        for i, f1 in enumerate(feature_cols):
            for f2 in feature_cols[i+1:]:
                if f1 in corr_matrix.index and f2 in corr_matrix.columns:
                    corr = abs(corr_matrix.loc[f1, f2])
                    if corr >= self.correlation_threshold:
                        high_corr_pairs.append((f1, f2, corr))
        
        report.append(f"Pairs with |corr| >= {self.correlation_threshold}: {len(high_corr_pairs)}")
        for f1, f2, corr in sorted(high_corr_pairs, key=lambda x: -x[2])[:10]:
            report.append(f"  {f1} <-> {f2}: {corr:.3f}")
        
        # Feature blocks
        report.append("")
        report.append("-" * 40)
        report.append("FEATURE BLOCKS (clusters)")
        report.append("-" * 40)
        
        blocks = self.identify_feature_blocks(df, feature_cols)
        report.append(f"Identified {len(blocks)} feature blocks:")
        for block in blocks[:5]:
            report.append(f"  Block {block.block_id}: {block.features}")
            report.append(f"    Avg correlation: {block.avg_correlation:.3f}")
            report.append(f"    Representative: {block.representative}")
        
        # VIF analysis
        report.append("")
        report.append("-" * 40)
        report.append("VIF DIAGNOSTICS")
        report.append("-" * 40)
        
        vif_results = self.compute_vif(df, feature_cols)
        high_vif = [v for v in vif_results if v.is_high]
        report.append(f"Features with VIF > {self.vif_threshold}: {len(high_vif)}")
        
        for v in vif_results[:10]:
            status = "⚠️ HIGH" if v.is_high else "✓"
            report.append(f"  {v.feature}: VIF={v.vif:.1f} {status}")
        
        # IC stability (if labels provided)
        if labels_df is not None:
            report.append("")
            report.append("-" * 40)
            report.append("IC STABILITY (most critical)")
            report.append("-" * 40)
            
            ic_results = self.compute_ic_stability(df, labels_df, feature_cols)
            stable = [r for r in ic_results if r.is_stable]
            unstable = [r for r in ic_results if not r.is_stable]
            
            report.append(f"Stable features (sign consistent >= {self.ic_sign_threshold:.0%}): {len(stable)}")
            report.append(f"Unstable features: {len(unstable)}")
            
            report.append("\nTop stable features:")
            for r in stable[:10]:
                report.append(f"  {r.feature}: IC={r.ic_mean:.3f}, sign_cons={r.ic_sign_consistency:.1%}")
            
            if unstable:
                report.append("\n⚠️ Unstable features (consider dropping or investigating):")
                for r in unstable[:5]:
                    report.append(f"  {r.feature}: IC={r.ic_mean:.3f}, sign_cons={r.ic_sign_consistency:.1%}")
        
        report.append("")
        report.append("=" * 60)
        
        return "\n".join(report)


# =============================================================================
# Helper Functions
# =============================================================================

def winsorize_cross_sectional(
    series: pd.Series,
    lower_pct: float = 0.01,
    upper_pct: float = 0.99,
) -> pd.Series:
    """
    Winsorize a series at given percentiles.
    
    Clips extreme values to reduce outlier influence.
    """
    lower = series.quantile(lower_pct)
    upper = series.quantile(upper_pct)
    return series.clip(lower=lower, upper=upper)


def rank_transform(series: pd.Series) -> pd.Series:
    """Transform to percentile ranks (0-1)."""
    return series.rank(pct=True, na_option="keep")


def zscore_transform(series: pd.Series) -> pd.Series:
    """Transform to z-scores."""
    mean = series.mean()
    std = series.std()
    if std == 0 or pd.isna(std):
        return pd.Series(0, index=series.index)
    return (series - mean) / std


def get_feature_hygiene(
    correlation_threshold: float = CORRELATION_THRESHOLD,
    vif_threshold: float = VIF_THRESHOLD,
) -> FeatureHygiene:
    """Factory function for FeatureHygiene."""
    return FeatureHygiene(
        correlation_threshold=correlation_threshold,
        vif_threshold=vif_threshold,
    )


# =============================================================================
# CLI / Demo
# =============================================================================

if __name__ == "__main__":
    import sys
    
    print("=" * 60)
    print("FEATURE HYGIENE DEMO")
    print("=" * 60)
    
    # Create sample data
    np.random.seed(42)
    n = 500
    
    # Create correlated features
    base = np.random.randn(n)
    sample_data = pd.DataFrame({
        "date": pd.date_range("2023-01-01", periods=n//5, freq="B").repeat(5),
        "ticker": ["NVDA", "AMD", "MSFT", "INTC", "AVGO"] * (n//5),
        "mom_1m": base + np.random.randn(n) * 0.3,
        "mom_3m": base * 0.9 + np.random.randn(n) * 0.3,  # Correlated with mom_1m
        "vol_20d": np.abs(np.random.randn(n)),
        "pe_vs_sector": np.random.randn(n) * 0.5,
        "beta": 0.8 + np.random.randn(n) * 0.3,
    })
    
    # Sample labels
    sample_labels = pd.DataFrame({
        "date": sample_data["date"],
        "ticker": sample_data["ticker"],
        "excess_return": sample_data["mom_1m"] * 0.1 + np.random.randn(n) * 0.05,
    })
    
    hygiene = FeatureHygiene()
    
    # Standardize
    print("\n1. Cross-sectional standardization:")
    std_df = hygiene.standardize_cross_sectional(sample_data, method="zscore")
    print(f"  Before: mom_1m mean={sample_data['mom_1m'].mean():.2f}, std={sample_data['mom_1m'].std():.2f}")
    print(f"  After:  mom_1m mean={std_df['mom_1m'].mean():.2f}, std={std_df['mom_1m'].std():.2f}")
    
    # Correlation matrix
    print("\n2. Correlation matrix:")
    corr = hygiene.compute_correlation_matrix(sample_data)
    print(f"  mom_1m vs mom_3m: {corr.loc['mom_1m', 'mom_3m']:.3f}")
    
    # Feature blocks
    print("\n3. Feature blocks:")
    blocks = hygiene.identify_feature_blocks(sample_data, threshold=0.5)
    for b in blocks:
        print(f"  Block {b.block_id}: {b.features} (avg_corr={b.avg_correlation:.2f})")
    
    # VIF
    print("\n4. VIF diagnostics:")
    vif_results = hygiene.compute_vif(sample_data)
    for v in vif_results:
        print(f"  {v.feature}: VIF={v.vif:.1f}")
    
    # IC stability
    print("\n5. IC stability:")
    ic_results = hygiene.compute_ic_stability(sample_data, sample_labels)
    for r in ic_results:
        status = "✓ STABLE" if r.is_stable else "⚠️ UNSTABLE"
        print(f"  {r.feature}: IC={r.ic_mean:.3f}, sign_cons={r.ic_sign_consistency:.1%} {status}")
    
    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)


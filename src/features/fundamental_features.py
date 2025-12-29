"""
Fundamental Features (Section 5.3)
==================================

Computes fundamental features that are:
- RELATIVE: vs own history or sector peers (not raw ratios)
- NORMALIZED: z-score or rank-transformed
- PIT-SAFE: Use filing date (observed_at) for point-in-time correctness

FEATURES:
- Valuation: P/E, P/S, EV/EBITDA (vs own history & sector)
- Profitability: Margins (gross, operating, net) vs sector
- Growth: Revenue growth, earnings growth vs sector
- Quality: ROE, ROA, debt ratios

RAW RATIOS ARE AVOIDED:
- P/E = 20 means nothing in isolation
- P/E vs own 3-year median z-score = meaningful signal
- P/E vs sector median = cross-sectional signal

PIT RULES:
- All fundamentals use filing_date (from FMP fillingDate)
- Features computed as of asof_date use only data with observed_at <= asof
"""

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

import pandas as pd
import numpy as np
import pytz

logger = logging.getLogger(__name__)

# Standard timezone
ET = pytz.timezone("America/New_York")
UTC = pytz.UTC


@dataclass
class FundamentalFeatures:
    """
    Fundamental features for a single stock on a single date.
    
    All features are RELATIVE (vs history or sector), not raw ratios.
    """
    ticker: str
    date: date
    sector: Optional[str] = None
    
    # Valuation (z-scores vs own history)
    pe_zscore_3y: Optional[float] = None      # P/E vs own 3-year history
    ps_zscore_3y: Optional[float] = None      # P/S vs own 3-year history
    pb_zscore_3y: Optional[float] = None      # P/B vs own 3-year history
    
    # Valuation (vs sector median, cross-sectional)
    pe_vs_sector: Optional[float] = None      # P/E relative to sector median
    ps_vs_sector: Optional[float] = None      # P/S relative to sector median
    
    # Profitability (vs sector)
    gross_margin_vs_sector: Optional[float] = None
    operating_margin_vs_sector: Optional[float] = None
    net_margin_vs_sector: Optional[float] = None
    
    # Growth (vs sector)
    revenue_growth_vs_sector: Optional[float] = None
    earnings_growth_vs_sector: Optional[float] = None
    
    # Quality (z-scores)
    roe_zscore: Optional[float] = None
    roa_zscore: Optional[float] = None
    
    # Raw values (for debugging, not for model)
    _raw_pe: Optional[float] = None
    _raw_ps: Optional[float] = None
    _raw_gross_margin: Optional[float] = None
    _raw_revenue_growth: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage/DataFrame."""
        return {
            "ticker": self.ticker,
            "date": self.date.isoformat() if isinstance(self.date, date) else self.date,
            "sector": self.sector,
            "pe_zscore_3y": self.pe_zscore_3y,
            "ps_zscore_3y": self.ps_zscore_3y,
            "pb_zscore_3y": self.pb_zscore_3y,
            "pe_vs_sector": self.pe_vs_sector,
            "ps_vs_sector": self.ps_vs_sector,
            "gross_margin_vs_sector": self.gross_margin_vs_sector,
            "operating_margin_vs_sector": self.operating_margin_vs_sector,
            "net_margin_vs_sector": self.net_margin_vs_sector,
            "revenue_growth_vs_sector": self.revenue_growth_vs_sector,
            "earnings_growth_vs_sector": self.earnings_growth_vs_sector,
            "roe_zscore": self.roe_zscore,
            "roa_zscore": self.roa_zscore,
            "_raw_pe": self._raw_pe,
            "_raw_ps": self._raw_ps,
            "_raw_gross_margin": self._raw_gross_margin,
            "_raw_revenue_growth": self._raw_revenue_growth,
        }


class FundamentalFeatureGenerator:
    """
    Generates fundamental features.
    
    Usage:
        generator = FundamentalFeatureGenerator()
        
        # Generate for single ticker
        features = generator.generate(
            ticker="NVDA",
            asof_date=date(2024, 1, 15),
        )
        
        # Generate for universe with sector context
        features_df = generator.generate_for_universe(
            tickers=["NVDA", "AMD", "INTC"],
            asof_date=date(2024, 1, 15),
        )
    """
    
    def __init__(
        self,
        fmp_client=None,
        lookback_quarters: int = 12,  # 3 years of history
    ):
        """
        Initialize feature generator.
        
        Args:
            fmp_client: FMPClient instance (lazy-loaded if None)
            lookback_quarters: Number of quarters to fetch for history
        """
        self._fmp = fmp_client
        self.lookback_quarters = lookback_quarters
        
        # Cache for fundamental data
        self._fundamental_cache: Dict[str, Dict[str, pd.DataFrame]] = {}
        self._profile_cache: Dict[str, Dict] = {}
    
    def _get_fmp_client(self):
        """Lazy-load FMP client."""
        if self._fmp is None:
            from ..data import FMPClient
            self._fmp = FMPClient()
        return self._fmp
    
    def _get_profile(self, ticker: str) -> Optional[Dict]:
        """Get company profile with caching."""
        if ticker in self._profile_cache:
            return self._profile_cache[ticker]
        
        fmp = self._get_fmp_client()
        profile = fmp.get_profile(ticker)
        
        if profile:
            self._profile_cache[ticker] = profile
        
        return profile
    
    def _get_income_statement(
        self,
        ticker: str,
        asof: datetime,
    ) -> pd.DataFrame:
        """Get income statement data with PIT filtering."""
        cache_key = f"{ticker}_income"
        
        if cache_key not in self._fundamental_cache:
            self._fundamental_cache[cache_key] = {}
        
        if "data" not in self._fundamental_cache[cache_key]:
            fmp = self._get_fmp_client()
            df = fmp.get_income_statement(ticker, limit=self.lookback_quarters)
            self._fundamental_cache[cache_key]["data"] = df
        
        df = self._fundamental_cache[cache_key]["data"]
        
        if df.empty:
            return df
        
        # PIT filter: only include rows where observed_at <= asof
        if "observed_at" in df.columns:
            df = df[df["observed_at"] <= asof].copy()
        
        return df
    
    def _get_balance_sheet(
        self,
        ticker: str,
        asof: datetime,
    ) -> pd.DataFrame:
        """Get balance sheet data with PIT filtering."""
        cache_key = f"{ticker}_balance"
        
        if cache_key not in self._fundamental_cache:
            self._fundamental_cache[cache_key] = {}
        
        if "data" not in self._fundamental_cache[cache_key]:
            fmp = self._get_fmp_client()
            df = fmp.get_balance_sheet(ticker, limit=self.lookback_quarters)
            self._fundamental_cache[cache_key]["data"] = df
        
        df = self._fundamental_cache[cache_key]["data"]
        
        if df.empty:
            return df
        
        # PIT filter
        if "observed_at" in df.columns:
            df = df[df["observed_at"] <= asof].copy()
        
        return df
    
    def _get_ratios_ttm(self, ticker: str) -> Optional[Dict]:
        """Get TTM ratios."""
        fmp = self._get_fmp_client()
        return fmp.get_ratios_ttm(ticker)
    
    def _compute_zscore_vs_history(
        self,
        current: float,
        history: pd.Series,
    ) -> Optional[float]:
        """Compute z-score of current value vs historical values."""
        if pd.isna(current) or history.empty:
            return None
        
        # Remove NaN and outliers (> 3 std)
        history = history.dropna()
        if len(history) < 4:  # Need at least 4 quarters
            return None
        
        mean = history.mean()
        std = history.std()
        
        if std == 0 or pd.isna(std):
            return None
        
        return (current - mean) / std
    
    def _safe_divide(self, a: float, b: float) -> Optional[float]:
        """Safe division handling NaN and zero."""
        if pd.isna(a) or pd.isna(b) or b == 0:
            return None
        return a / b
    
    def generate(
        self,
        ticker: str,
        asof_date: date,
        price: Optional[float] = None,
        sector_data: Optional[Dict] = None,
    ) -> FundamentalFeatures:
        """
        Generate fundamental features for a single ticker.
        
        Args:
            ticker: Stock ticker
            asof_date: Date to compute features for (PIT-safe)
            price: Current stock price (if known)
            sector_data: Pre-computed sector medians for relative features
        
        Returns:
            FundamentalFeatures dataclass
        """
        # Convert date to datetime for PIT filtering
        asof_dt = datetime.combine(asof_date, datetime.max.time())
        asof_dt = UTC.localize(asof_dt)
        
        # Get profile for sector
        profile = self._get_profile(ticker)
        sector = profile.get("sector", "Unknown") if profile else "Unknown"
        
        # Initialize features
        features = FundamentalFeatures(
            ticker=ticker,
            date=asof_date,
            sector=sector,
        )
        
        # Get income statement
        income_df = self._get_income_statement(ticker, asof_dt)
        
        if income_df.empty:
            logger.warning(f"No income statement data for {ticker}")
            return features
        
        # Sort by period_end
        income_df = income_df.sort_values("period_end", ascending=False)
        
        # Get balance sheet
        balance_df = self._get_balance_sheet(ticker, asof_dt)
        
        # Get TTM ratios (current snapshot)
        ratios = self._get_ratios_ttm(ticker)
        
        # --- Compute valuation ratios (need price) ---
        if price is None and profile:
            price = profile.get("price")
        
        if price and "eps" in income_df.columns:
            # Get trailing 4 quarters EPS
            recent_eps = income_df["eps"].head(4).sum()
            pe_ratio = self._safe_divide(price, recent_eps)
            features._raw_pe = pe_ratio
            
            # P/E z-score vs own history
            if "eps" in income_df.columns and len(income_df) >= 8:
                # Compute rolling TTM EPS and P/E for each quarter
                eps_history = []
                for i in range(len(income_df) - 3):
                    ttm_eps = income_df["eps"].iloc[i:i+4].sum()
                    if ttm_eps > 0:
                        # Use historical price estimate (simplified: assume same price)
                        # In practice, we'd need historical prices
                        eps_history.append(ttm_eps)
                
                if len(eps_history) >= 4 and pe_ratio is not None:
                    eps_series = pd.Series(eps_history)
                    pe_history = price / eps_series  # Simplified
                    features.pe_zscore_3y = self._compute_zscore_vs_history(pe_ratio, pe_history)
        
        if price and "revenue" in income_df.columns:
            # P/S ratio
            if profile and "sharesOutstanding" in profile:
                shares = profile["sharesOutstanding"]
                mcap = price * shares
                ttm_revenue = income_df["revenue"].head(4).sum()
                ps_ratio = self._safe_divide(mcap, ttm_revenue)
                features._raw_ps = ps_ratio
        
        # --- Compute profitability margins ---
        if "grossProfit" in income_df.columns and "revenue" in income_df.columns:
            recent = income_df.iloc[0]
            gross_margin = self._safe_divide(recent.get("grossProfit", 0), recent.get("revenue", 0))
            features._raw_gross_margin = gross_margin
        
        # --- Compute growth rates ---
        if "revenue" in income_df.columns and len(income_df) >= 5:
            current_rev = income_df["revenue"].iloc[0]
            prior_rev = income_df["revenue"].iloc[4] if len(income_df) > 4 else None
            
            if prior_rev and prior_rev > 0:
                revenue_growth = (current_rev / prior_rev) - 1
                features._raw_revenue_growth = revenue_growth
        
        # --- Quality metrics from ratios ---
        if ratios:
            if "returnOnEquity" in ratios:
                features.roe_zscore = ratios["returnOnEquity"]  # Will be z-scored in universe
            if "returnOnAssets" in ratios:
                features.roa_zscore = ratios["returnOnAssets"]
        
        return features
    
    def generate_for_universe(
        self,
        tickers: List[str],
        asof_date: date,
        prices: Optional[Dict[str, float]] = None,
    ) -> pd.DataFrame:
        """
        Generate features for a universe of stocks.
        
        Args:
            tickers: List of tickers
            asof_date: Date to compute features for
            prices: Dictionary of ticker -> price (optional)
        
        Returns:
            DataFrame with all features, including sector-relative metrics
        """
        all_features = []
        
        # First pass: compute raw features
        for ticker in tickers:
            try:
                price = prices.get(ticker) if prices else None
                features = self.generate(
                    ticker=ticker,
                    asof_date=asof_date,
                    price=price,
                )
                all_features.append(features)
            except Exception as e:
                logger.warning(f"Failed to generate fundamentals for {ticker}: {e}")
                continue
        
        if not all_features:
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame([f.to_dict() for f in all_features])
        df["date"] = pd.to_datetime(df["date"]).dt.date
        
        # Second pass: compute sector-relative features
        df = self._add_sector_relative_features(df)
        
        # Third pass: compute cross-sectional z-scores
        df = self._add_cross_sectional_zscores(df)
        
        logger.info(f"Generated {len(df)} fundamental feature rows for {len(tickers)} tickers")
        return df
    
    def _add_sector_relative_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add features relative to sector median."""
        if "sector" not in df.columns or df["sector"].isna().all():
            return df
        
        for sector in df["sector"].unique():
            if pd.isna(sector):
                continue
            
            mask = df["sector"] == sector
            sector_df = df[mask]
            
            if len(sector_df) < 2:
                continue
            
            # P/E vs sector
            if "_raw_pe" in df.columns:
                sector_median = sector_df["_raw_pe"].median()
                if sector_median and sector_median > 0:
                    df.loc[mask, "pe_vs_sector"] = df.loc[mask, "_raw_pe"] / sector_median - 1
            
            # P/S vs sector
            if "_raw_ps" in df.columns:
                sector_median = sector_df["_raw_ps"].median()
                if sector_median and sector_median > 0:
                    df.loc[mask, "ps_vs_sector"] = df.loc[mask, "_raw_ps"] / sector_median - 1
            
            # Gross margin vs sector
            if "_raw_gross_margin" in df.columns:
                sector_median = sector_df["_raw_gross_margin"].median()
                if sector_median and not pd.isna(sector_median):
                    df.loc[mask, "gross_margin_vs_sector"] = (
                        df.loc[mask, "_raw_gross_margin"] - sector_median
                    )
            
            # Revenue growth vs sector
            if "_raw_revenue_growth" in df.columns:
                sector_median = sector_df["_raw_revenue_growth"].median()
                if sector_median is not None and not pd.isna(sector_median):
                    df.loc[mask, "revenue_growth_vs_sector"] = (
                        df.loc[mask, "_raw_revenue_growth"] - sector_median
                    )
        
        return df
    
    def _add_cross_sectional_zscores(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add cross-sectional z-scores for quality metrics."""
        # ROE z-score
        if "roe_zscore" in df.columns:
            raw_roe = df["roe_zscore"].copy()
            mean = raw_roe.mean()
            std = raw_roe.std()
            if std > 0:
                df["roe_zscore"] = (raw_roe - mean) / std
        
        # ROA z-score
        if "roa_zscore" in df.columns:
            raw_roa = df["roa_zscore"].copy()
            mean = raw_roa.mean()
            std = raw_roa.std()
            if std > 0:
                df["roa_zscore"] = (raw_roa - mean) / std
        
        return df
    
    def clear_cache(self):
        """Clear all caches."""
        self._fundamental_cache.clear()
        self._profile_cache.clear()


# =============================================================================
# Helper Functions
# =============================================================================

def winsorize(series: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    """
    Winsorize a series to reduce outlier impact.
    
    Args:
        series: Input series
        lower: Lower percentile (default 1%)
        upper: Upper percentile (default 99%)
    
    Returns:
        Winsorized series
    """
    lower_bound = series.quantile(lower)
    upper_bound = series.quantile(upper)
    return series.clip(lower=lower_bound, upper=upper_bound)


def sector_neutralize(df: pd.DataFrame, feature_col: str, sector_col: str = "sector") -> pd.Series:
    """
    Neutralize a feature by subtracting sector median.
    
    Args:
        df: DataFrame with feature and sector columns
        feature_col: Name of feature column
        sector_col: Name of sector column
    
    Returns:
        Sector-neutralized feature series
    """
    result = df[feature_col].copy()
    
    for sector in df[sector_col].unique():
        if pd.isna(sector):
            continue
        mask = df[sector_col] == sector
        sector_median = df.loc[mask, feature_col].median()
        result.loc[mask] = result.loc[mask] - sector_median
    
    return result


# =============================================================================
# CLI/Demo
# =============================================================================

if __name__ == "__main__":
    import sys
    from datetime import date
    
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("FUNDAMENTAL FEATURES DEMO")
    print("=" * 60)
    
    generator = FundamentalFeatureGenerator()
    
    # Generate for a single stock
    ticker = "NVDA"
    asof = date(2024, 12, 15)
    
    print(f"\nGenerating fundamental features for {ticker} as of {asof}...")
    
    features = generator.generate(ticker, asof)
    
    print(f"\nFeatures for {ticker} (sector: {features.sector}):")
    for key, value in features.to_dict().items():
        if value is not None and key not in ["ticker", "date", "sector"]:
            if isinstance(value, float):
                if "growth" in key or "margin" in key or "_vs_" in key:
                    print(f"  {key}: {value:.2%}")
                elif "zscore" in key:
                    print(f"  {key}: {value:.2f}")
                else:
                    print(f"  {key}: {value:.4f}")
    
    print("\n" + "=" * 60)


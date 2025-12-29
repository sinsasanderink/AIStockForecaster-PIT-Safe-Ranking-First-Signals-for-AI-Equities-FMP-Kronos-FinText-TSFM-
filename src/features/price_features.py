"""
Price & Volume Features (Section 5.2)
=====================================

Computes price and volume-based features for the AI Stock Forecaster.

FEATURES:
- Momentum (1m, 3m, 6m, 12m returns)
- Volatility (realized vol, vol-of-vol)
- Drawdown (max drawdown, distance from high)
- Relative strength (vs universe median)
- Beta (vs benchmark, rolling window)
- Liquidity (ADV, volatility-adjusted ADV)

ALL FEATURES ARE:
- PIT-safe: Use only data available at asof date
- Cross-sectionally standardized: z-score or rank within universe
- Computed from split-adjusted prices

CONVENTION:
- All returns are decimal (0.10 = 10%)
- Volatility is annualized
- Windows are in trading days
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

# Trading days per year (for annualization)
TRADING_DAYS_PER_YEAR = 252

# Standard momentum windows (trading days)
MOMENTUM_WINDOWS = {
    "mom_1m": 21,     # ~1 month
    "mom_3m": 63,     # ~3 months
    "mom_6m": 126,    # ~6 months
    "mom_12m": 252,   # ~12 months
}

# Volatility windows
VOLATILITY_WINDOWS = {
    "vol_20d": 20,
    "vol_60d": 60,
}

# ADV windows
ADV_WINDOWS = {
    "adv_20d": 20,
    "adv_60d": 60,
}

# Beta window
BETA_WINDOW = 252


@dataclass
class PriceFeatures:
    """
    Price and volume features for a single stock on a single date.
    """
    ticker: str
    date: date
    
    # Momentum features (returns, decimal)
    mom_1m: Optional[float] = None
    mom_3m: Optional[float] = None
    mom_6m: Optional[float] = None
    mom_12m: Optional[float] = None
    
    # Volatility features (annualized)
    vol_20d: Optional[float] = None
    vol_60d: Optional[float] = None
    vol_of_vol: Optional[float] = None  # Volatility of volatility
    
    # Drawdown features
    max_drawdown_60d: Optional[float] = None
    dist_from_high_60d: Optional[float] = None  # Current price vs 60d high
    
    # Relative strength (vs universe)
    rel_strength_1m: Optional[float] = None  # z-score vs universe
    rel_strength_3m: Optional[float] = None
    
    # Beta (vs benchmark)
    beta_252d: Optional[float] = None
    
    # Liquidity features
    adv_20d: Optional[float] = None  # Average daily dollar volume
    adv_60d: Optional[float] = None
    vol_adj_adv: Optional[float] = None  # ADV / volatility
    
    # Price level
    price: Optional[float] = None
    log_mcap: Optional[float] = None  # Log market cap (if available)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage/DataFrame."""
        return {
            "ticker": self.ticker,
            "date": self.date.isoformat() if isinstance(self.date, date) else self.date,
            "mom_1m": self.mom_1m,
            "mom_3m": self.mom_3m,
            "mom_6m": self.mom_6m,
            "mom_12m": self.mom_12m,
            "vol_20d": self.vol_20d,
            "vol_60d": self.vol_60d,
            "vol_of_vol": self.vol_of_vol,
            "max_drawdown_60d": self.max_drawdown_60d,
            "dist_from_high_60d": self.dist_from_high_60d,
            "rel_strength_1m": self.rel_strength_1m,
            "rel_strength_3m": self.rel_strength_3m,
            "beta_252d": self.beta_252d,
            "adv_20d": self.adv_20d,
            "adv_60d": self.adv_60d,
            "vol_adj_adv": self.vol_adj_adv,
            "price": self.price,
            "log_mcap": self.log_mcap,
        }


class PriceFeatureGenerator:
    """
    Generates price and volume features.
    
    Usage:
        generator = PriceFeatureGenerator()
        
        # Generate features for a single ticker
        features = generator.generate(
            ticker="NVDA",
            asof_date=date(2024, 1, 15),
        )
        
        # Generate features for universe
        features_df = generator.generate_for_universe(
            tickers=["NVDA", "AMD", "MSFT"],
            asof_date=date(2024, 1, 15),
        )
    """
    
    def __init__(
        self,
        fmp_client=None,
        calendar=None,
        benchmark: str = "QQQ",
    ):
        """
        Initialize feature generator.
        
        Args:
            fmp_client: FMPClient instance (lazy-loaded if None)
            calendar: TradingCalendar instance (lazy-loaded if None)
            benchmark: Benchmark ticker for beta calculation
        """
        self._fmp = fmp_client
        self._calendar = calendar
        self.benchmark = benchmark
        
        # Price cache
        self._price_cache: Dict[str, pd.DataFrame] = {}
        self._benchmark_cache: Optional[pd.DataFrame] = None
    
    def _get_fmp_client(self):
        """Lazy-load FMP client."""
        if self._fmp is None:
            from ..data import FMPClient
            self._fmp = FMPClient()
        return self._fmp
    
    def _get_calendar(self):
        """Lazy-load trading calendar."""
        if self._calendar is None:
            from ..data import TradingCalendarImpl
            self._calendar = TradingCalendarImpl()
        return self._calendar
    
    def _get_prices(
        self,
        ticker: str,
        asof_date: date,
        lookback_days: int = 300,
    ) -> pd.DataFrame:
        """
        Get price data for a ticker with sufficient history.
        
        Returns DataFrame with columns: date, open, high, low, close, volume
        """
        cache_key = f"{ticker}_{asof_date}_{lookback_days}"
        
        if cache_key in self._price_cache:
            return self._price_cache[cache_key]
        
        fmp = self._get_fmp_client()
        
        # Get extra buffer for lookback
        start = asof_date - timedelta(days=lookback_days + 50)
        end = asof_date
        
        df = fmp.get_historical_prices(
            ticker,
            start=start.isoformat(),
            end=end.isoformat(),
        )
        
        if df.empty:
            logger.warning(f"No price data for {ticker}")
            return df
        
        # Ensure we have required columns
        df["date"] = pd.to_datetime(df["date"]).dt.date
        df = df.sort_values("date").reset_index(drop=True)
        
        # Filter to asof_date (PIT-safe)
        df = df[df["date"] <= asof_date].copy()
        
        self._price_cache[cache_key] = df
        return df
    
    def _get_benchmark_prices(self, asof_date: date, lookback_days: int = 300) -> pd.DataFrame:
        """Get benchmark prices."""
        return self._get_prices(self.benchmark, asof_date, lookback_days)
    
    def _compute_returns(self, prices: pd.Series, window: int) -> Optional[float]:
        """Compute return over window."""
        if len(prices) < window + 1:
            return None
        
        current = prices.iloc[-1]
        past = prices.iloc[-(window + 1)]
        
        if past == 0 or pd.isna(past) or pd.isna(current):
            return None
        
        return (current / past) - 1
    
    def _compute_volatility(self, prices: pd.Series, window: int) -> Optional[float]:
        """Compute annualized volatility from price series."""
        if len(prices) < window + 1:
            return None
        
        # Use last 'window' prices
        recent = prices.iloc[-(window + 1):]
        returns = recent.pct_change().dropna()
        
        if len(returns) < window * 0.8:  # Need 80% of window
            return None
        
        # Annualize
        daily_vol = returns.std()
        return daily_vol * np.sqrt(TRADING_DAYS_PER_YEAR)
    
    def _compute_max_drawdown(self, prices: pd.Series, window: int) -> Optional[float]:
        """Compute maximum drawdown over window."""
        if len(prices) < window:
            return None
        
        recent = prices.iloc[-window:]
        
        # Running max
        running_max = recent.expanding().max()
        drawdown = (recent - running_max) / running_max
        
        return drawdown.min()
    
    def _compute_dist_from_high(self, prices: pd.Series, window: int) -> Optional[float]:
        """Compute distance from high (as fraction)."""
        if len(prices) < window:
            return None
        
        recent = prices.iloc[-window:]
        high = recent.max()
        current = prices.iloc[-1]
        
        if high == 0 or pd.isna(high):
            return None
        
        return (current - high) / high
    
    def _compute_beta(
        self,
        stock_prices: pd.Series,
        benchmark_prices: pd.Series,
        window: int = BETA_WINDOW,
    ) -> Optional[float]:
        """Compute beta vs benchmark."""
        if len(stock_prices) < window or len(benchmark_prices) < window:
            return None
        
        # Align dates
        stock_ret = stock_prices.pct_change().dropna()
        bench_ret = benchmark_prices.pct_change().dropna()
        
        # Take last 'window' returns
        stock_ret = stock_ret.iloc[-window:]
        bench_ret = bench_ret.iloc[-window:]
        
        if len(stock_ret) < window * 0.8 or len(bench_ret) < window * 0.8:
            return None
        
        # Compute beta = Cov(stock, bench) / Var(bench)
        cov = stock_ret.cov(bench_ret)
        var = bench_ret.var()
        
        if var == 0 or pd.isna(var):
            return None
        
        return cov / var
    
    def _compute_adv(
        self,
        prices: pd.DataFrame,
        window: int,
    ) -> Optional[float]:
        """Compute average daily dollar volume."""
        if len(prices) < window:
            return None
        
        recent = prices.iloc[-window:]
        
        if "volume" not in recent.columns or "close" not in recent.columns:
            return None
        
        dollar_volume = recent["close"] * recent["volume"]
        return dollar_volume.mean()
    
    def generate(
        self,
        ticker: str,
        asof_date: date,
        benchmark_prices: pd.DataFrame = None,
    ) -> PriceFeatures:
        """
        Generate price features for a single ticker on a single date.
        
        Args:
            ticker: Stock ticker
            asof_date: Date to compute features for (PIT-safe)
            benchmark_prices: Pre-fetched benchmark prices (optional)
        
        Returns:
            PriceFeatures dataclass
        """
        # Get price data
        df = self._get_prices(ticker, asof_date, lookback_days=300)
        
        if df.empty or len(df) < 30:
            logger.warning(f"Insufficient price data for {ticker} on {asof_date}")
            return PriceFeatures(ticker=ticker, date=asof_date)
        
        prices = df["close"]
        
        # Initialize features
        features = PriceFeatures(ticker=ticker, date=asof_date)
        
        # Current price
        features.price = float(prices.iloc[-1])
        
        # Momentum
        features.mom_1m = self._compute_returns(prices, MOMENTUM_WINDOWS["mom_1m"])
        features.mom_3m = self._compute_returns(prices, MOMENTUM_WINDOWS["mom_3m"])
        features.mom_6m = self._compute_returns(prices, MOMENTUM_WINDOWS["mom_6m"])
        features.mom_12m = self._compute_returns(prices, MOMENTUM_WINDOWS["mom_12m"])
        
        # Volatility
        features.vol_20d = self._compute_volatility(prices, VOLATILITY_WINDOWS["vol_20d"])
        features.vol_60d = self._compute_volatility(prices, VOLATILITY_WINDOWS["vol_60d"])
        
        # Vol of vol (rolling 20d vol computed over last 60 days)
        if len(prices) >= 80:
            rolling_vol = prices.pct_change().rolling(20).std() * np.sqrt(TRADING_DAYS_PER_YEAR)
            features.vol_of_vol = rolling_vol.iloc[-60:].std() if len(rolling_vol) >= 60 else None
        
        # Drawdown
        features.max_drawdown_60d = self._compute_max_drawdown(prices, 60)
        features.dist_from_high_60d = self._compute_dist_from_high(prices, 60)
        
        # Beta (needs benchmark)
        if benchmark_prices is None:
            benchmark_prices = self._get_benchmark_prices(asof_date, lookback_days=300)
        
        if not benchmark_prices.empty:
            bench_close = benchmark_prices["close"]
            features.beta_252d = self._compute_beta(prices, bench_close)
        
        # ADV
        features.adv_20d = self._compute_adv(df, ADV_WINDOWS["adv_20d"])
        features.adv_60d = self._compute_adv(df, ADV_WINDOWS["adv_60d"])
        
        # Vol-adjusted ADV
        if features.adv_60d is not None and features.vol_60d is not None and features.vol_60d > 0:
            features.vol_adj_adv = features.adv_60d / features.vol_60d
        
        return features
    
    def generate_for_universe(
        self,
        tickers: List[str],
        asof_date: date,
        compute_relative: bool = True,
    ) -> pd.DataFrame:
        """
        Generate features for a universe of stocks.
        
        Args:
            tickers: List of tickers
            asof_date: Date to compute features for
            compute_relative: If True, compute relative strength (z-scores)
        
        Returns:
            DataFrame with all features
        """
        # Pre-fetch benchmark
        benchmark_df = self._get_benchmark_prices(asof_date, lookback_days=300)
        
        all_features = []
        
        for ticker in tickers:
            try:
                features = self.generate(
                    ticker=ticker,
                    asof_date=asof_date,
                    benchmark_prices=benchmark_df,
                )
                all_features.append(features)
            except Exception as e:
                logger.warning(f"Failed to generate features for {ticker}: {e}")
                continue
        
        if not all_features:
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame([f.to_dict() for f in all_features])
        df["date"] = pd.to_datetime(df["date"]).dt.date
        
        # Compute relative strength (z-scores within universe)
        if compute_relative and len(df) > 1:
            df = self._add_relative_features(df)
        
        logger.info(f"Generated {len(df)} feature rows for {len(tickers)} tickers")
        return df
    
    def _add_relative_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add cross-sectional relative features (z-scores)."""
        # Relative strength (momentum z-scores)
        if "mom_1m" in df.columns:
            mean = df["mom_1m"].mean()
            std = df["mom_1m"].std()
            if std > 0:
                df["rel_strength_1m"] = (df["mom_1m"] - mean) / std
        
        if "mom_3m" in df.columns:
            mean = df["mom_3m"].mean()
            std = df["mom_3m"].std()
            if std > 0:
                df["rel_strength_3m"] = (df["mom_3m"] - mean) / std
        
        return df
    
    def clear_cache(self):
        """Clear the price cache."""
        self._price_cache.clear()
        self._benchmark_cache = None


def cross_sectional_rank(series: pd.Series) -> pd.Series:
    """
    Convert values to cross-sectional percentile ranks (0 to 1).
    
    Useful for making features comparable across time.
    """
    return series.rank(pct=True)


def cross_sectional_zscore(series: pd.Series) -> pd.Series:
    """
    Convert values to cross-sectional z-scores.
    
    Centers at 0, scales by standard deviation.
    """
    mean = series.mean()
    std = series.std()
    if std == 0 or pd.isna(std):
        return pd.Series(0, index=series.index)
    return (series - mean) / std


# =============================================================================
# CLI/Demo
# =============================================================================

if __name__ == "__main__":
    import sys
    from datetime import date
    
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("PRICE FEATURES DEMO")
    print("=" * 60)
    
    generator = PriceFeatureGenerator()
    
    # Generate for a single stock
    ticker = "NVDA"
    asof = date(2024, 12, 15)
    
    print(f"\nGenerating price features for {ticker} as of {asof}...")
    
    features = generator.generate(ticker, asof)
    
    print(f"\nFeatures for {ticker}:")
    for key, value in features.to_dict().items():
        if value is not None and key not in ["ticker", "date"]:
            if isinstance(value, float):
                if "mom" in key or "drawdown" in key or "dist" in key:
                    print(f"  {key}: {value:.2%}")
                elif "vol" in key and "adv" not in key:
                    print(f"  {key}: {value:.2%}")
                elif "adv" in key:
                    print(f"  {key}: ${value/1e6:.1f}M")
                else:
                    print(f"  {key}: {value:.4f}")
    
    print("\n" + "=" * 60)


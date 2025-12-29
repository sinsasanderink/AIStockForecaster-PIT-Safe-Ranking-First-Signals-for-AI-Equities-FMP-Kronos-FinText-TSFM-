"""
Regime & Macro Features (Section 5.5)
=====================================

Computes regime and macro-level features for the AI Stock Forecaster.

FEATURES:
- VIX level and percentile (volatility regime)
- Market trend regime (bull/bear/neutral from SPY)
- Market momentum (SPY returns at various windows)
- Sector rotation indicators (tech vs defensive)
- Market breadth proxies

ALL FEATURES ARE:
- PIT-safe: Use only data available at asof date
- Common across all stocks in universe (market-level features)
- Timestamped with cutoff enforcement

CONVENTION:
- VIX in raw level (typically 10-80 range)
- VIX percentile as decimal (0-1)
- Returns as decimal (0.10 = 10%)
- Regime as categorical: 'bull', 'bear', 'neutral'

DATA SOURCES:
- VIX: ^VIX or VIXY via FMP
- Market: SPY or QQQ via FMP
- Sectors: XLK (tech), XLY (consumer), XLP (staples), XLU (utilities)
"""

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from enum import Enum

import pandas as pd
import numpy as np
import pytz

logger = logging.getLogger(__name__)

# Standard timezone
ET = pytz.timezone("America/New_York")
UTC = pytz.UTC

# Trading days per year
TRADING_DAYS_PER_YEAR = 252

# VIX percentile lookback (2 years of data)
VIX_PERCENTILE_WINDOW = 504  # ~2 years of trading days

# Market regime thresholds
BULL_THRESHOLD = 0.0   # Positive momentum
BEAR_THRESHOLD = -0.0  # Negative momentum (use MA for cleaner signal)

# Sector ETFs for rotation
SECTOR_ETFS = {
    "tech": "XLK",       # Technology
    "consumer_disc": "XLY",  # Consumer Discretionary
    "consumer_staples": "XLP",  # Consumer Staples (defensive)
    "utilities": "XLU",   # Utilities (defensive)
    "financials": "XLF",  # Financials
    "healthcare": "XLV",  # Healthcare
}

# Market benchmark
MARKET_BENCHMARK = "SPY"
VOLATILITY_INDEX = "^VIX"  # CBOE VIX


class MarketRegime(Enum):
    """Market regime classification."""
    BULL = "bull"
    BEAR = "bear"
    NEUTRAL = "neutral"


class VolatilityRegime(Enum):
    """Volatility regime classification."""
    LOW = "low"           # VIX < 15
    NORMAL = "normal"     # 15 <= VIX < 25
    ELEVATED = "elevated" # 25 <= VIX < 35
    HIGH = "high"         # VIX >= 35


@dataclass
class RegimeFeatures:
    """
    Regime and macro features for a single date.
    
    These are MARKET-LEVEL features, common to all stocks in the universe.
    """
    date: date
    
    # VIX features
    vix_level: Optional[float] = None           # Raw VIX level
    vix_percentile: Optional[float] = None      # VIX percentile (0-1) over 2y
    vix_change_5d: Optional[float] = None       # 5-day VIX change
    vix_regime: Optional[str] = None            # low/normal/elevated/high
    
    # Market trend features (SPY-based)
    market_return_5d: Optional[float] = None    # 5-day SPY return
    market_return_21d: Optional[float] = None   # 21-day (~1 month) return
    market_return_63d: Optional[float] = None   # 63-day (~3 month) return
    market_vol_21d: Optional[float] = None      # 21-day realized volatility
    
    # Market regime
    market_regime: Optional[str] = None         # bull/bear/neutral
    above_ma_50: Optional[bool] = None          # Price > 50-day MA
    above_ma_200: Optional[bool] = None         # Price > 200-day MA
    ma_50_200_cross: Optional[float] = None     # (MA50 - MA200) / MA200
    
    # Sector rotation features
    tech_vs_staples: Optional[float] = None     # XLK/XLP relative strength
    tech_vs_utilities: Optional[float] = None   # XLK/XLU relative strength
    risk_on_indicator: Optional[float] = None   # Composite risk-on/off signal
    
    # Market breadth (simplified proxy)
    market_breadth_proxy: Optional[float] = None  # Approximated from returns
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "date": self.date.isoformat(),
            "vix_level": self.vix_level,
            "vix_percentile": self.vix_percentile,
            "vix_change_5d": self.vix_change_5d,
            "vix_regime": self.vix_regime,
            "market_return_5d": self.market_return_5d,
            "market_return_21d": self.market_return_21d,
            "market_return_63d": self.market_return_63d,
            "market_vol_21d": self.market_vol_21d,
            "market_regime": self.market_regime,
            "above_ma_50": self.above_ma_50,
            "above_ma_200": self.above_ma_200,
            "ma_50_200_cross": self.ma_50_200_cross,
            "tech_vs_staples": self.tech_vs_staples,
            "tech_vs_utilities": self.tech_vs_utilities,
            "risk_on_indicator": self.risk_on_indicator,
            "market_breadth_proxy": self.market_breadth_proxy,
        }


class RegimeFeatureGenerator:
    """
    Generates regime and macro-level features.
    
    These features are common to all stocks in the universe - they describe
    the overall market environment rather than individual stocks.
    
    Usage:
        from src.features.regime_features import RegimeFeatureGenerator
        
        generator = RegimeFeatureGenerator()
        
        features = generator.compute_features(
            asof_date=date(2024, 1, 15),
        )
    """
    
    def __init__(
        self,
        fmp_client=None,
        trading_calendar=None,
        vix_ticker: str = "^VIX",
        market_ticker: str = "SPY",
    ):
        """
        Initialize regime feature generator.
        
        Args:
            fmp_client: FMPClient instance for data fetching
            trading_calendar: TradingCalendarImpl for date calculations
            vix_ticker: Ticker for volatility index (^VIX or VIXY)
            market_ticker: Ticker for market benchmark (SPY)
        """
        # Lazy imports
        if fmp_client is None:
            from src.data import get_fmp_client
            fmp_client = get_fmp_client()
        if trading_calendar is None:
            from src.data import get_trading_calendar
            trading_calendar = get_trading_calendar()
        
        self._fmp = fmp_client
        self._calendar = trading_calendar
        self._vix_ticker = vix_ticker
        self._market_ticker = market_ticker
        
        # Cache for market data
        self._price_cache: Dict[str, pd.DataFrame] = {}
        
        logger.info(f"RegimeFeatureGenerator initialized (VIX={vix_ticker}, Market={market_ticker})")
    
    def _get_cutoff_datetime(self, asof_date: date) -> datetime:
        """Get market close cutoff time for a date."""
        return self._calendar.get_market_close(asof_date).astimezone(UTC)
    
    def _get_prices(self, ticker: str, lookback_days: int, asof_date: date) -> pd.DataFrame:
        """
        Get historical prices for a ticker.
        
        Args:
            ticker: Ticker symbol
            lookback_days: Number of calendar days to look back
            asof_date: The date to compute features for
        
        Returns:
            DataFrame with date index and 'close' column
        """
        cache_key = f"{ticker}_{asof_date}_{lookback_days}"
        
        if cache_key in self._price_cache:
            return self._price_cache[cache_key]
        
        start_date = asof_date - timedelta(days=lookback_days)
        
        try:
            # Use get_index_historical for indices and ETFs
            df = self._fmp.get_index_historical(
                ticker.replace("^", ""),  # FMP doesn't use ^ prefix
                start=start_date.isoformat(),
                end=asof_date.isoformat(),
            )
            
            if df.empty:
                logger.warning(f"No price data for {ticker}")
                return pd.DataFrame()
            
            # Standardize column names
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"]).dt.date
                df = df.set_index("date")
            
            # Get close price (handle different column names)
            close_col = None
            for col in ["adjClose", "adj_close", "close", "Close"]:
                if col in df.columns:
                    close_col = col
                    break
            
            if close_col is None:
                logger.warning(f"No close column found for {ticker}")
                return pd.DataFrame()
            
            result = df[[close_col]].rename(columns={close_col: "close"}).sort_index()
            self._price_cache[cache_key] = result
            return result
            
        except Exception as e:
            logger.warning(f"Error fetching prices for {ticker}: {e}")
            return pd.DataFrame()
    
    def _compute_vix_features(self, asof_date: date) -> Dict[str, Any]:
        """Compute VIX-based volatility features."""
        features = {}
        
        # Get VIX data (use longer lookback for percentile)
        vix_df = self._get_prices(self._vix_ticker, lookback_days=VIX_PERCENTILE_WINDOW * 2, asof_date=asof_date)
        
        if vix_df.empty or asof_date not in vix_df.index:
            # Try alternative VIX ticker
            vix_df = self._get_prices("VIX", lookback_days=VIX_PERCENTILE_WINDOW * 2, asof_date=asof_date)
        
        if vix_df.empty:
            logger.warning("No VIX data available")
            return features
        
        # Filter to data <= asof_date (PIT safety)
        vix_df = vix_df[vix_df.index <= asof_date]
        
        if len(vix_df) < 5:
            return features
        
        # Current VIX level
        if asof_date in vix_df.index:
            current_vix = vix_df.loc[asof_date, "close"]
        else:
            current_vix = vix_df.iloc[-1]["close"]
        
        features["vix_level"] = float(current_vix)
        
        # VIX percentile (over 2 years)
        lookback_vix = vix_df.iloc[-min(VIX_PERCENTILE_WINDOW, len(vix_df)):]
        percentile = (lookback_vix["close"] <= current_vix).mean()
        features["vix_percentile"] = float(percentile)
        
        # VIX 5-day change
        if len(vix_df) >= 6:
            vix_5d_ago = vix_df.iloc[-6]["close"]
            features["vix_change_5d"] = float((current_vix - vix_5d_ago) / vix_5d_ago)
        
        # VIX regime classification
        if current_vix < 15:
            features["vix_regime"] = VolatilityRegime.LOW.value
        elif current_vix < 25:
            features["vix_regime"] = VolatilityRegime.NORMAL.value
        elif current_vix < 35:
            features["vix_regime"] = VolatilityRegime.ELEVATED.value
        else:
            features["vix_regime"] = VolatilityRegime.HIGH.value
        
        return features
    
    def _compute_market_features(self, asof_date: date) -> Dict[str, Any]:
        """Compute market trend and regime features."""
        features = {}
        
        # Get SPY data (need enough for 200-day MA)
        spy_df = self._get_prices(self._market_ticker, lookback_days=300, asof_date=asof_date)
        
        if spy_df.empty:
            logger.warning("No SPY data available")
            return features
        
        # Filter to data <= asof_date (PIT safety)
        spy_df = spy_df[spy_df.index <= asof_date]
        
        if len(spy_df) < 21:
            return features
        
        closes = spy_df["close"]
        
        # Market returns at various windows
        if len(closes) >= 6:
            features["market_return_5d"] = float((closes.iloc[-1] / closes.iloc[-6]) - 1)
        
        if len(closes) >= 22:
            features["market_return_21d"] = float((closes.iloc[-1] / closes.iloc[-22]) - 1)
        
        if len(closes) >= 64:
            features["market_return_63d"] = float((closes.iloc[-1] / closes.iloc[-64]) - 1)
        
        # Market volatility (21-day)
        if len(closes) >= 22:
            daily_returns = closes.pct_change().dropna()
            vol_21d = daily_returns.iloc[-21:].std() * np.sqrt(TRADING_DAYS_PER_YEAR)
            features["market_vol_21d"] = float(vol_21d)
        
        # Moving averages
        current_price = closes.iloc[-1]
        
        if len(closes) >= 50:
            ma_50 = closes.iloc[-50:].mean()
            features["above_ma_50"] = bool(current_price > ma_50)
        
        if len(closes) >= 200:
            ma_200 = closes.iloc[-200:].mean()
            features["above_ma_200"] = bool(current_price > ma_200)
            
            if len(closes) >= 50:
                ma_50 = closes.iloc[-50:].mean()
                features["ma_50_200_cross"] = float((ma_50 - ma_200) / ma_200)
        
        # Market regime classification
        above_ma_50 = features.get("above_ma_50", None)
        above_ma_200 = features.get("above_ma_200", None)
        return_21d = features.get("market_return_21d", 0)
        
        if above_ma_50 is not None and above_ma_200 is not None:
            if above_ma_50 and above_ma_200 and return_21d > 0:
                features["market_regime"] = MarketRegime.BULL.value
            elif not above_ma_50 and not above_ma_200 and return_21d < 0:
                features["market_regime"] = MarketRegime.BEAR.value
            else:
                features["market_regime"] = MarketRegime.NEUTRAL.value
        
        return features
    
    def _compute_sector_features(self, asof_date: date) -> Dict[str, Any]:
        """Compute sector rotation features."""
        features = {}
        
        # Get sector ETF data
        sector_returns = {}
        
        for sector_name, etf in [("tech", "XLK"), ("staples", "XLP"), ("utilities", "XLU")]:
            df = self._get_prices(etf, lookback_days=30, asof_date=asof_date)
            
            if df.empty or len(df) < 22:
                continue
            
            # Filter to data <= asof_date (PIT safety)
            df = df[df.index <= asof_date]
            
            if len(df) >= 22:
                ret_21d = (df["close"].iloc[-1] / df["close"].iloc[-22]) - 1
                sector_returns[sector_name] = ret_21d
        
        # Compute relative strength
        if "tech" in sector_returns and "staples" in sector_returns:
            features["tech_vs_staples"] = float(sector_returns["tech"] - sector_returns["staples"])
        
        if "tech" in sector_returns and "utilities" in sector_returns:
            features["tech_vs_utilities"] = float(sector_returns["tech"] - sector_returns["utilities"])
        
        # Risk-on indicator (composite)
        # Positive = risk-on (tech outperforming defensives)
        # Negative = risk-off (defensives outperforming tech)
        if features.get("tech_vs_staples") is not None and features.get("tech_vs_utilities") is not None:
            features["risk_on_indicator"] = float(
                (features["tech_vs_staples"] + features["tech_vs_utilities"]) / 2
            )
        
        return features
    
    def compute_features(self, asof_date: date) -> RegimeFeatures:
        """
        Compute all regime features for a given date.
        
        Args:
            asof_date: Date to compute features for
        
        Returns:
            RegimeFeatures object
        """
        features = RegimeFeatures(date=asof_date)
        
        # VIX features
        try:
            vix_features = self._compute_vix_features(asof_date)
            for k, v in vix_features.items():
                setattr(features, k, v)
        except Exception as e:
            logger.warning(f"Error computing VIX features: {e}")
        
        # Market features
        try:
            market_features = self._compute_market_features(asof_date)
            for k, v in market_features.items():
                setattr(features, k, v)
        except Exception as e:
            logger.warning(f"Error computing market features: {e}")
        
        # Sector rotation features
        try:
            sector_features = self._compute_sector_features(asof_date)
            for k, v in sector_features.items():
                setattr(features, k, v)
        except Exception as e:
            logger.warning(f"Error computing sector features: {e}")
        
        return features
    
    def compute_batch(
        self,
        start_date: date,
        end_date: date,
        freq: str = "weekly",
    ) -> pd.DataFrame:
        """
        Compute regime features for multiple dates (batch mode).
        
        Args:
            start_date: Start date
            end_date: End date
            freq: 'daily' or 'weekly'
        
        Returns:
            DataFrame with all features for all dates
        """
        dates = pd.date_range(start_date, end_date, freq="B" if freq == "daily" else "W-FRI")
        
        all_features = []
        for d in dates:
            d_date = d.date()
            try:
                # Skip non-trading days
                if not self._calendar.is_trading_day(d_date):
                    continue
                
                features = self.compute_features(d_date)
                all_features.append(features.to_dict())
            except Exception as e:
                logger.warning(f"Error computing features for {d_date}: {e}")
        
        return pd.DataFrame(all_features)
    
    def to_dataframe(self, features: RegimeFeatures) -> pd.DataFrame:
        """Convert RegimeFeatures to single-row DataFrame."""
        return pd.DataFrame([features.to_dict()])


# =============================================================================
# Helper Functions
# =============================================================================

def classify_vix_regime(vix_level: float) -> str:
    """Classify VIX level into regime."""
    if vix_level < 15:
        return VolatilityRegime.LOW.value
    elif vix_level < 25:
        return VolatilityRegime.NORMAL.value
    elif vix_level < 35:
        return VolatilityRegime.ELEVATED.value
    else:
        return VolatilityRegime.HIGH.value


def classify_market_regime(
    above_ma_50: bool,
    above_ma_200: bool,
    return_21d: float,
) -> str:
    """Classify market into regime based on MA and momentum."""
    if above_ma_50 and above_ma_200 and return_21d > 0:
        return MarketRegime.BULL.value
    elif not above_ma_50 and not above_ma_200 and return_21d < 0:
        return MarketRegime.BEAR.value
    else:
        return MarketRegime.NEUTRAL.value


def get_regime_feature_generator(
    fmp_client=None,
    trading_calendar=None,
) -> RegimeFeatureGenerator:
    """Factory function for RegimeFeatureGenerator."""
    return RegimeFeatureGenerator(
        fmp_client=fmp_client,
        trading_calendar=trading_calendar,
    )


# =============================================================================
# CLI / Demo
# =============================================================================

if __name__ == "__main__":
    import sys
    
    print("=" * 60)
    print("REGIME FEATURE GENERATOR DEMO")
    print("=" * 60)
    
    try:
        generator = RegimeFeatureGenerator()
        
        asof = date(2024, 12, 15)
        print(f"\nComputing regime features for {asof}...")
        
        features = generator.compute_features(asof)
        
        print(f"\nVIX Features:")
        print(f"  Level: {features.vix_level}")
        print(f"  Percentile: {features.vix_percentile:.1%}" if features.vix_percentile else "  Percentile: N/A")
        print(f"  5d Change: {features.vix_change_5d:.1%}" if features.vix_change_5d else "  5d Change: N/A")
        print(f"  Regime: {features.vix_regime}")
        
        print(f"\nMarket Features:")
        print(f"  5d Return: {features.market_return_5d:.1%}" if features.market_return_5d else "  5d Return: N/A")
        print(f"  21d Return: {features.market_return_21d:.1%}" if features.market_return_21d else "  21d Return: N/A")
        print(f"  63d Return: {features.market_return_63d:.1%}" if features.market_return_63d else "  63d Return: N/A")
        print(f"  21d Vol: {features.market_vol_21d:.1%}" if features.market_vol_21d else "  21d Vol: N/A")
        print(f"  Above MA50: {features.above_ma_50}")
        print(f"  Above MA200: {features.above_ma_200}")
        print(f"  Regime: {features.market_regime}")
        
        print(f"\nSector Rotation:")
        print(f"  Tech vs Staples: {features.tech_vs_staples:.1%}" if features.tech_vs_staples else "  Tech vs Staples: N/A")
        print(f"  Tech vs Utilities: {features.tech_vs_utilities:.1%}" if features.tech_vs_utilities else "  Tech vs Utilities: N/A")
        print(f"  Risk-On Indicator: {features.risk_on_indicator:.1%}" if features.risk_on_indicator else "  Risk-On Indicator: N/A")
        
        print("\n" + "=" * 60)
        print("DEMO COMPLETE")
        print("=" * 60)
        
    except Exception as e:
        print(f"Error running demo: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


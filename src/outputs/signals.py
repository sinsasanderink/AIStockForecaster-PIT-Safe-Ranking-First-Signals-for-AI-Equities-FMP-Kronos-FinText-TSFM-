"""
Signal Data Structures
======================

Core data classes defining the per-stock and aggregate signal outputs.
These represent the fundamental outputs of the forecasting system.

Design Philosophy:
- Immutable data structures (frozen dataclasses)
- Full traceability (timestamps, model versions, inputs)
- Distribution-aware (not just point estimates)

IMPORTANT CONVENTIONS:
- expected_excess_return is ALWAYS the distribution mean (not median)
- P50 (median) may differ from mean for skewed distributions
- prob_outperform uses normal approximation; for TSFM outputs, use from_samples()

Dependencies:
- scipy.stats is used for prob_positive calculation (normal CDF)
- If you need a lightweight runtime, see the _prob_positive_no_scipy alternative
"""

from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Dict, List, Optional, Tuple
from enum import Enum
import numpy as np
import math


class SignalSource(Enum):
    """Source model that generated the signal."""
    KRONOS = "kronos"              # Price dynamics model
    FINTEXT = "fintext"            # Return structure model
    TABULAR = "tabular"            # Tabular ML baseline
    FUSION = "fusion"              # Combined fusion model
    ENSEMBLE = "ensemble"          # Regime-aware ensemble


@dataclass(frozen=True)
class ReturnDistribution:
    """
    Predicted return distribution for a stock.
    
    Represents the full uncertainty around the forecast,
    not just a point estimate.
    
    Attributes:
        percentile_5: 5th percentile (downside risk)
        percentile_25: 25th percentile (lower quartile)
        percentile_50: Median expected return
        percentile_75: 75th percentile (upper quartile)
        percentile_95: 95th percentile (upside potential)
        mean: Mean expected return
        std: Standard deviation of the distribution
    """
    percentile_5: float
    percentile_25: float
    percentile_50: float  # Median
    percentile_75: float
    percentile_95: float
    mean: float
    std: float
    
    @property
    def iqr(self) -> float:
        """Interquartile range - measure of dispersion."""
        return self.percentile_75 - self.percentile_25
    
    @property
    def upside_ratio(self) -> float:
        """Ratio of upside to downside potential."""
        downside = abs(self.percentile_5 - self.percentile_50)
        upside = abs(self.percentile_95 - self.percentile_50)
        return upside / downside if downside > 0 else float('inf')
    
    @property
    def risk_adjusted_return(self) -> float:
        """Sharpe-like ratio: mean / std."""
        return self.mean / self.std if self.std > 0 else 0.0
    
    @property
    def prob_positive(self) -> float:
        """
        Probability of positive return (excess return > 0).
        
        CURRENT IMPLEMENTATION: Normal approximation using mean/std.
        This is appropriate for initial development but should be replaced
        with empirical probability once we have TSFM quantile outputs or
        Monte Carlo samples.
        
        TODO [Section 8/9]: When Kronos/FinText provide sample-based predictions,
        use from_samples() and compute empirical prob as (samples > 0).mean()
        
        Uses scipy.stats.norm which is in requirements/base.txt.
        If you need a no-scipy version, use _prob_positive_approx().
        """
        if self.std <= 0:
            return 1.0 if self.mean > 0 else 0.0
        # Using normal CDF: P(X > 0) = 1 - Phi(-mean/std) = Phi(mean/std)
        from scipy.stats import norm
        return float(norm.cdf(self.mean / self.std))
    
    def _prob_positive_approx(self) -> float:
        """
        No-scipy approximation of prob_positive using error function.
        
        Uses the standard normal CDF approximation via math.erf.
        Accuracy: within 0.001 of scipy.stats.norm.cdf for typical values.
        """
        if self.std <= 0:
            return 1.0 if self.mean > 0 else 0.0
        z = self.mean / self.std
        # CDF(z) = 0.5 * (1 + erf(z / sqrt(2)))
        return 0.5 * (1 + math.erf(z / math.sqrt(2)))
    
    @property
    def prob_outperform(self) -> float:
        """
        Probability of outperforming the benchmark.
        
        Same as prob_positive since we're measuring excess returns
        (return - benchmark_return > 0).
        """
        return self.prob_positive
    
    @property
    def value_at_risk_95(self) -> float:
        """95% VaR: 5th percentile loss."""
        return -self.percentile_5 if self.percentile_5 < 0 else 0.0
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for serialization."""
        return {
            "p5": self.percentile_5,
            "p25": self.percentile_25,
            "p50": self.percentile_50,
            "p75": self.percentile_75,
            "p95": self.percentile_95,
            "mean": self.mean,
            "std": self.std,
            "iqr": self.iqr,
            "upside_ratio": self.upside_ratio,
            "prob_outperform": self.prob_outperform,
            "var_95": self.value_at_risk_95,
        }
    
    @classmethod
    def from_samples(cls, samples: np.ndarray) -> "ReturnDistribution":
        """Create distribution from Monte Carlo samples."""
        return cls(
            percentile_5=float(np.percentile(samples, 5)),
            percentile_25=float(np.percentile(samples, 25)),
            percentile_50=float(np.percentile(samples, 50)),
            percentile_75=float(np.percentile(samples, 75)),
            percentile_95=float(np.percentile(samples, 95)),
            mean=float(np.mean(samples)),
            std=float(np.std(samples)),
        )
    
    def format_range(self, style: str = "p5_p50_p95") -> str:
        """
        Format the distribution as a readable range string.
        
        Args:
            style: "p5_p50_p95" or "p10_p90" or "iqr"
        """
        if style == "p5_p50_p95":
            return f"[{self.percentile_5:+.1%} / {self.percentile_50:+.1%} / {self.percentile_95:+.1%}]"
        elif style == "iqr":
            return f"[{self.percentile_25:+.1%} to {self.percentile_75:+.1%}]"
        else:
            return f"[{self.percentile_5:+.1%}, {self.percentile_95:+.1%}]"


@dataclass(frozen=True)
class SignalDriver:
    """
    Explanation of what drives a stock's signal/ranking.
    
    Provides interpretability by identifying key contributing factors.
    
    Attributes:
        feature_name: Name of the feature/signal component
        category: Category (price, fundamental, event, regime, model)
        contribution: Contribution to the overall score (can be negative)
        description: Human-readable explanation
    """
    feature_name: str
    category: str  # "price", "fundamental", "event", "regime", "model"
    contribution: float
    description: str
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "feature": self.feature_name,
            "category": self.category,
            "contribution": self.contribution,
            "description": self.description,
        }


class LiquidityFlag(Enum):
    """Liquidity/capacity warning flags."""
    OK = "ok"                    # No concerns
    LOW_VOLUME = "low_volume"   # Below average volume threshold
    WIDE_SPREAD = "wide_spread" # Bid-ask spread concern
    SMALL_CAP = "small_cap"     # Market cap may limit position size


@dataclass(frozen=True)
class StockSignal:
    """
    Complete signal output for a single stock at a single horizon.
    
    This is the primary per-stock output of the forecasting system.
    
    IMPORTANT CONVENTION:
    - expected_excess_return MUST equal return_distribution.mean
    - This is enforced via __post_init__ validation
    - Use StockSignal.create() factory method to auto-set from distribution
    - P50 (median) may differ from expected_excess_return for skewed distributions
    
    Attributes:
        ticker: Stock ticker symbol
        rebalance_date: Date when signal was generated
        horizon_days: Forecast horizon in trading days
        benchmark: Benchmark used for excess return calculation
        
        # Core predictions
        expected_excess_return: Mean of return distribution (NOT median)
        return_distribution: Full distribution of predicted returns
        
        # Ranking & confidence
        alpha_rank_score: Cross-sectional ranking score (higher = better)
        confidence_score: Calibrated uncertainty (0-1, higher = more confident)
        
        # Liquidity
        liquidity_flag: Capacity/liquidity warning
        avg_daily_volume: Average daily volume (shares)
        
        # Interpretability
        key_drivers: Top factors influencing the signal
        
        # Metadata
        source: Model that generated the signal
        generated_at: Timestamp of generation
        model_version: Version of the model used
    """
    # Identification
    ticker: str
    rebalance_date: date
    horizon_days: int
    benchmark: str
    
    # Core predictions
    # NOTE: expected_excess_return = return_distribution.mean (enforced)
    expected_excess_return: float
    return_distribution: ReturnDistribution
    
    # Ranking & confidence
    alpha_rank_score: float
    confidence_score: float
    
    # Liquidity (optional, with defaults)
    liquidity_flag: LiquidityFlag = LiquidityFlag.OK
    avg_daily_volume: Optional[float] = None
    
    # Interpretability
    key_drivers: Tuple[SignalDriver, ...] = field(default_factory=tuple)
    
    # Metadata
    source: SignalSource = SignalSource.FUSION
    generated_at: datetime = field(default_factory=datetime.utcnow)
    model_version: str = "0.1.0"
    
    def __post_init__(self):
        """
        Validate that expected_excess_return equals distribution mean.
        
        This prevents confusion between mean and median, ensuring reports
        always show consistent values.
        """
        # Allow small floating point tolerance
        tolerance = 1e-6
        diff = abs(self.expected_excess_return - self.return_distribution.mean)
        if diff > tolerance:
            raise ValueError(
                f"expected_excess_return ({self.expected_excess_return:.6f}) must equal "
                f"return_distribution.mean ({self.return_distribution.mean:.6f}). "
                f"Use StockSignal.create() to auto-set from distribution."
            )
    
    @classmethod
    def create(
        cls,
        ticker: str,
        rebalance_date: date,
        horizon_days: int,
        benchmark: str,
        return_distribution: ReturnDistribution,
        alpha_rank_score: float,
        confidence_score: float,
        liquidity_flag: LiquidityFlag = LiquidityFlag.OK,
        avg_daily_volume: Optional[float] = None,
        key_drivers: Tuple[SignalDriver, ...] = (),
        source: SignalSource = SignalSource.FUSION,
        generated_at: Optional[datetime] = None,
        model_version: str = "0.1.0",
    ) -> "StockSignal":
        """
        Factory method that auto-sets expected_excess_return from distribution.mean.
        
        This is the preferred way to create StockSignal instances as it
        ensures the expected_excess_return convention is followed.
        """
        return cls(
            ticker=ticker,
            rebalance_date=rebalance_date,
            horizon_days=horizon_days,
            benchmark=benchmark,
            expected_excess_return=return_distribution.mean,  # Auto-set
            return_distribution=return_distribution,
            alpha_rank_score=alpha_rank_score,
            confidence_score=confidence_score,
            liquidity_flag=liquidity_flag,
            avg_daily_volume=avg_daily_volume,
            key_drivers=key_drivers,
            source=source,
            generated_at=generated_at or datetime.utcnow(),
            model_version=model_version,
        )
    
    @property
    def is_high_confidence(self) -> bool:
        """Check if signal meets high confidence threshold."""
        return self.confidence_score >= 0.7
    
    @property
    def signal_direction(self) -> str:
        """Get signal direction: buy, neutral, or avoid."""
        if self.expected_excess_return > 0.02:  # > 2% expected excess
            return "buy"
        elif self.expected_excess_return < -0.02:  # < -2% expected excess
            return "avoid"
        return "neutral"
    
    @property
    def prob_outperform(self) -> float:
        """Probability of outperforming the benchmark."""
        return self.return_distribution.prob_outperform
    
    @property
    def has_liquidity_concern(self) -> bool:
        """Check if there's a liquidity/capacity concern."""
        return self.liquidity_flag != LiquidityFlag.OK
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "ticker": self.ticker,
            "rebalance_date": self.rebalance_date.isoformat(),
            "horizon_days": self.horizon_days,
            "benchmark": self.benchmark,
            "expected_excess_return": self.expected_excess_return,
            "return_distribution": self.return_distribution.to_dict(),
            "prob_outperform": self.prob_outperform,
            "alpha_rank_score": self.alpha_rank_score,
            "confidence_score": self.confidence_score,
            "signal_direction": self.signal_direction,
            "is_high_confidence": self.is_high_confidence,
            "liquidity_flag": self.liquidity_flag.value,
            "avg_daily_volume": self.avg_daily_volume,
            "key_drivers": [d.to_dict() for d in self.key_drivers],
            "source": self.source.value,
            "generated_at": self.generated_at.isoformat(),
            "model_version": self.model_version,
        }
    
    def summary(self) -> str:
        """
        Generate human-readable decision-support summary.
        
        Shows the minimum useful set for making a decision:
        - Expected return (mean or median)
        - P5/P50/P95 range
        - Prob(outperform)
        - Confidence bucket
        - Liquidity flag
        - Key drivers
        """
        direction_emoji = {"buy": "🟢", "neutral": "🟡", "avoid": "🔴"}
        conf_label = "HIGH" if self.is_high_confidence else "MEDIUM" if self.confidence_score >= 0.4 else "LOW"
        liq_warning = f" ⚠️{self.liquidity_flag.value}" if self.has_liquidity_concern else ""
        
        dist = self.return_distribution
        
        lines = [
            f"{direction_emoji.get(self.signal_direction, '⚪')} {self.ticker} "
            f"({self.horizon_days}d horizon){liq_warning}",
            f"",
            f"  Expected Excess Return: {self.expected_excess_return:+.1%} vs {self.benchmark}",
            f"  Return Range (P5/P50/P95): {dist.format_range()}",
            f"  Prob(Outperform):  {self.prob_outperform:.0%}",
            f"",
            f"  Alpha Rank Score:  {self.alpha_rank_score:.3f}",
            f"  Confidence:        {self.confidence_score:.2f} ({conf_label})",
        ]
        
        if self.avg_daily_volume:
            adv_str = f"{self.avg_daily_volume/1e6:.1f}M" if self.avg_daily_volume >= 1e6 else f"{self.avg_daily_volume/1e3:.0f}K"
            lines.append(f"  Avg Daily Volume:  {adv_str}")
        
        if self.key_drivers:
            lines.append("")
            lines.append("  Key Drivers:")
            for driver in self.key_drivers[:3]:
                sign = "+" if driver.contribution > 0 else ""
                lines.append(f"    • {driver.feature_name}: {sign}{driver.contribution:.3f}")
        
        return "\n".join(lines)
    
    def one_liner(self) -> str:
        """Compact one-line summary for tables."""
        dist = self.return_distribution
        conf = "★" if self.is_high_confidence else "○"
        liq = "!" if self.has_liquidity_concern else " "
        return (
            f"{self.ticker:6s} {self.expected_excess_return:+6.1%} "
            f"{dist.format_range():28s} "
            f"P(>0)={self.prob_outperform:4.0%} {conf}{liq}"
        )


@dataclass
class HorizonSignals:
    """
    Collection of signals for all stocks at a specific horizon.
    
    Attributes:
        horizon_days: The forecast horizon
        signals: Dictionary mapping ticker to signal
        rebalance_date: Date when signals were generated
    """
    horizon_days: int
    signals: Dict[str, StockSignal]
    rebalance_date: date
    benchmark: str
    
    def __len__(self) -> int:
        return len(self.signals)
    
    def get_top_n(self, n: int = 10) -> List[StockSignal]:
        """Get top N stocks by alpha rank score."""
        sorted_signals = sorted(
            self.signals.values(),
            key=lambda s: s.alpha_rank_score,
            reverse=True
        )
        return sorted_signals[:n]
    
    def get_bottom_n(self, n: int = 10) -> List[StockSignal]:
        """Get bottom N stocks by alpha rank score."""
        sorted_signals = sorted(
            self.signals.values(),
            key=lambda s: s.alpha_rank_score
        )
        return sorted_signals[:n]
    
    def get_high_confidence(self) -> List[StockSignal]:
        """Get all high-confidence signals."""
        return [s for s in self.signals.values() if s.is_high_confidence]
    
    def get_by_direction(self, direction: str) -> List[StockSignal]:
        """Get signals by direction (buy/neutral/avoid)."""
        return [s for s in self.signals.values() if s.signal_direction == direction]


@dataclass
class RebalanceSignals:
    """
    Complete output for a single rebalance date across all horizons.
    
    This is the top-level output structure that contains everything
    produced by the system at a single point in time.
    
    Attributes:
        rebalance_date: Date of signal generation
        horizon_signals: Signals for each horizon (20/60/90 days)
        universe_tickers: List of all tickers in the universe
        benchmark: Primary benchmark used
        metadata: Additional metadata
    """
    rebalance_date: date
    horizon_signals: Dict[int, HorizonSignals]  # {20: HorizonSignals, 60: ..., 90: ...}
    universe_tickers: List[str]
    benchmark: str
    generated_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate that all horizons have consistent data."""
        for horizon, signals in self.horizon_signals.items():
            assert signals.rebalance_date == self.rebalance_date, \
                f"Inconsistent rebalance dates for horizon {horizon}"
            assert signals.benchmark == self.benchmark, \
                f"Inconsistent benchmark for horizon {horizon}"
    
    @property
    def horizons(self) -> List[int]:
        """Get list of available horizons."""
        return sorted(self.horizon_signals.keys())
    
    def get_signal(self, ticker: str, horizon: int) -> Optional[StockSignal]:
        """Get signal for a specific ticker and horizon."""
        if horizon not in self.horizon_signals:
            return None
        return self.horizon_signals[horizon].signals.get(ticker)
    
    def get_all_signals_for_ticker(self, ticker: str) -> Dict[int, StockSignal]:
        """Get signals for a ticker across all horizons."""
        return {
            horizon: signals.signals[ticker]
            for horizon, signals in self.horizon_signals.items()
            if ticker in signals.signals
        }
    
    def summary(self) -> str:
        """Generate summary of the rebalance signals."""
        lines = [
            f"📊 Rebalance Signals for {self.rebalance_date}",
            f"   Universe: {len(self.universe_tickers)} stocks",
            f"   Benchmark: {self.benchmark}",
            "",
        ]
        
        for horizon in self.horizons:
            hs = self.horizon_signals[horizon]
            top = hs.get_top_n(3)
            high_conf = len(hs.get_high_confidence())
            
            lines.append(f"📈 {horizon}-Day Horizon:")
            lines.append(f"   High confidence signals: {high_conf}/{len(hs)}")
            lines.append(f"   Top 3: {', '.join(s.ticker for s in top)}")
            lines.append("")
        
        return "\n".join(lines)


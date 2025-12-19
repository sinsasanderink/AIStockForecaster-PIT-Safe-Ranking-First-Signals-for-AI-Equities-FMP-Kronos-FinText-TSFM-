"""
Cross-Sectional Ranking Logic
=============================

Implements the ranking and categorization of stocks based on their signals.
This module transforms individual stock signals into actionable ranked lists.

Key Outputs:
- Ranked stock lists (Top buys / Neutral / Avoid)
- Confidence-stratified buckets
- Cross-sectional statistics
"""

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Dict, List, Optional, Tuple
from enum import Enum
import numpy as np

from .signals import StockSignal, HorizonSignals, RebalanceSignals


class RankingCategory(Enum):
    """Category for stock ranking."""
    TOP_BUY = "top_buy"        # Strong buy signal
    BUY = "buy"                # Moderate buy signal
    NEUTRAL = "neutral"        # No clear signal
    AVOID = "avoid"            # Moderate avoid signal
    STRONG_AVOID = "strong_avoid"  # Strong avoid signal


class ConfidenceBucket(Enum):
    """Confidence level buckets."""
    HIGH = "high"      # Confidence >= 0.7
    MEDIUM = "medium"  # 0.4 <= Confidence < 0.7
    LOW = "low"        # Confidence < 0.4


@dataclass(frozen=True)
class RankedStock:
    """
    A stock with its ranking information.
    
    Attributes:
        ticker: Stock ticker symbol
        rank: Position in the ranking (1 = best)
        percentile: Percentile rank (0-100, 100 = best)
        category: Ranking category (top_buy, buy, neutral, avoid, strong_avoid)
        confidence_bucket: Confidence level bucket
        signal: The underlying StockSignal
    """
    ticker: str
    rank: int
    percentile: float
    category: RankingCategory
    confidence_bucket: ConfidenceBucket
    signal: StockSignal
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "ticker": self.ticker,
            "rank": self.rank,
            "percentile": self.percentile,
            "category": self.category.value,
            "confidence_bucket": self.confidence_bucket.value,
            "expected_excess_return": self.signal.expected_excess_return,
            "alpha_rank_score": self.signal.alpha_rank_score,
            "confidence_score": self.signal.confidence_score,
        }


@dataclass
class CrossSectionalRanking:
    """
    Complete cross-sectional ranking for a single horizon.
    
    Organizes all stocks into ranked lists and confidence buckets.
    
    Attributes:
        rebalance_date: Date of the ranking
        horizon_days: Forecast horizon
        benchmark: Benchmark used
        ranked_stocks: List of all stocks sorted by rank
        by_category: Stocks grouped by category
        by_confidence: Stocks grouped by confidence bucket
    """
    rebalance_date: date
    horizon_days: int
    benchmark: str
    ranked_stocks: List[RankedStock]
    by_category: Dict[RankingCategory, List[RankedStock]] = field(default_factory=dict)
    by_confidence: Dict[ConfidenceBucket, List[RankedStock]] = field(default_factory=dict)
    generated_at: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self):
        """Organize stocks by category and confidence if not provided."""
        if not self.by_category:
            self.by_category = {cat: [] for cat in RankingCategory}
            for stock in self.ranked_stocks:
                self.by_category[stock.category].append(stock)
        
        if not self.by_confidence:
            self.by_confidence = {bucket: [] for bucket in ConfidenceBucket}
            for stock in self.ranked_stocks:
                self.by_confidence[stock.confidence_bucket].append(stock)
    
    def __len__(self) -> int:
        return len(self.ranked_stocks)
    
    # -------------------------------------------------------------------------
    # Accessors
    # -------------------------------------------------------------------------
    
    def get_top_n(self, n: int = 10) -> List[RankedStock]:
        """Get top N ranked stocks."""
        return self.ranked_stocks[:n]
    
    def get_bottom_n(self, n: int = 10) -> List[RankedStock]:
        """Get bottom N ranked stocks."""
        return self.ranked_stocks[-n:]
    
    def get_by_category(self, category: RankingCategory) -> List[RankedStock]:
        """Get all stocks in a specific category."""
        return self.by_category.get(category, [])
    
    def get_by_confidence(self, bucket: ConfidenceBucket) -> List[RankedStock]:
        """Get all stocks in a specific confidence bucket."""
        return self.by_confidence.get(bucket, [])
    
    def get_high_confidence_buys(self) -> List[RankedStock]:
        """Get high-confidence buy signals (most actionable)."""
        return [
            s for s in self.ranked_stocks
            if s.confidence_bucket == ConfidenceBucket.HIGH
            and s.category in (RankingCategory.TOP_BUY, RankingCategory.BUY)
        ]
    
    def get_ticker_rank(self, ticker: str) -> Optional[RankedStock]:
        """Get ranking information for a specific ticker."""
        for stock in self.ranked_stocks:
            if stock.ticker == ticker:
                return stock
        return None
    
    # -------------------------------------------------------------------------
    # Statistics
    # -------------------------------------------------------------------------
    
    @property
    def category_counts(self) -> Dict[str, int]:
        """Count of stocks in each category."""
        return {cat.value: len(stocks) for cat, stocks in self.by_category.items()}
    
    @property
    def confidence_counts(self) -> Dict[str, int]:
        """Count of stocks in each confidence bucket."""
        return {bucket.value: len(stocks) for bucket, stocks in self.by_confidence.items()}
    
    @property
    def mean_expected_return(self) -> float:
        """Average expected excess return across all stocks."""
        returns = [s.signal.expected_excess_return for s in self.ranked_stocks]
        return float(np.mean(returns)) if returns else 0.0
    
    @property
    def return_spread(self) -> float:
        """Spread between top and bottom quintile expected returns."""
        n = len(self.ranked_stocks)
        if n < 5:
            return 0.0
        
        quintile_size = n // 5
        top_quintile = self.ranked_stocks[:quintile_size]
        bottom_quintile = self.ranked_stocks[-quintile_size:]
        
        top_mean = np.mean([s.signal.expected_excess_return for s in top_quintile])
        bottom_mean = np.mean([s.signal.expected_excess_return for s in bottom_quintile])
        
        return float(top_mean - bottom_mean)
    
    # -------------------------------------------------------------------------
    # Display
    # -------------------------------------------------------------------------
    
    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            f"📊 Cross-Sectional Ranking ({self.horizon_days}-Day Horizon)",
            f"   Date: {self.rebalance_date}",
            f"   Universe: {len(self.ranked_stocks)} stocks",
            f"   Benchmark: {self.benchmark}",
            "",
            "   Category Distribution:",
        ]
        
        for cat, count in self.category_counts.items():
            lines.append(f"     {cat}: {count}")
        
        lines.append("")
        lines.append("   Confidence Distribution:")
        for bucket, count in self.confidence_counts.items():
            lines.append(f"     {bucket}: {count}")
        
        lines.append("")
        lines.append(f"   Top-Bottom Quintile Spread: {self.return_spread:.2%}")
        
        # Top 5 buys
        top_buys = self.get_top_n(5)
        if top_buys:
            lines.append("")
            lines.append("   🟢 Top 5 Buys:")
            for i, stock in enumerate(top_buys, 1):
                conf = "★" if stock.confidence_bucket == ConfidenceBucket.HIGH else "○"
                lines.append(
                    f"     {i}. {stock.ticker} "
                    f"({stock.signal.expected_excess_return:+.1%}) {conf}"
                )
        
        # Top 5 avoids
        bottom = self.get_bottom_n(5)
        if bottom:
            lines.append("")
            lines.append("   🔴 Top 5 Avoids:")
            for i, stock in enumerate(reversed(bottom), 1):
                lines.append(
                    f"     {i}. {stock.ticker} "
                    f"({stock.signal.expected_excess_return:+.1%})"
                )
        
        return "\n".join(lines)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "rebalance_date": self.rebalance_date.isoformat(),
            "horizon_days": self.horizon_days,
            "benchmark": self.benchmark,
            "total_stocks": len(self.ranked_stocks),
            "category_counts": self.category_counts,
            "confidence_counts": self.confidence_counts,
            "return_spread": self.return_spread,
            "ranked_stocks": [s.to_dict() for s in self.ranked_stocks],
        }


def _assign_category(
    signal: StockSignal,
    percentile: float,
    top_threshold: float = 90,
    buy_threshold: float = 70,
    avoid_threshold: float = 30,
    strong_avoid_threshold: float = 10,
) -> RankingCategory:
    """
    Assign a ranking category based on percentile and signal characteristics.
    
    Args:
        signal: The stock signal
        percentile: Percentile rank (0-100, higher = better)
        top_threshold: Percentile above which stock is TOP_BUY
        buy_threshold: Percentile above which stock is BUY
        avoid_threshold: Percentile below which stock is AVOID
        strong_avoid_threshold: Percentile below which stock is STRONG_AVOID
    
    Returns:
        RankingCategory for the stock
    """
    if percentile >= top_threshold:
        return RankingCategory.TOP_BUY
    elif percentile >= buy_threshold:
        return RankingCategory.BUY
    elif percentile <= strong_avoid_threshold:
        return RankingCategory.STRONG_AVOID
    elif percentile <= avoid_threshold:
        return RankingCategory.AVOID
    else:
        return RankingCategory.NEUTRAL


def _assign_confidence_bucket(
    confidence_score: float,
    high_threshold: float = 0.7,
    medium_threshold: float = 0.4,
) -> ConfidenceBucket:
    """
    Assign a confidence bucket based on confidence score.
    
    Args:
        confidence_score: Calibrated confidence score (0-1)
        high_threshold: Score above which confidence is HIGH
        medium_threshold: Score above which confidence is MEDIUM
    
    Returns:
        ConfidenceBucket for the stock
    """
    if confidence_score >= high_threshold:
        return ConfidenceBucket.HIGH
    elif confidence_score >= medium_threshold:
        return ConfidenceBucket.MEDIUM
    else:
        return ConfidenceBucket.LOW


def create_ranking_from_signals(
    horizon_signals: HorizonSignals,
    top_threshold: float = 90,
    buy_threshold: float = 70,
    avoid_threshold: float = 30,
    strong_avoid_threshold: float = 10,
    high_confidence_threshold: float = 0.7,
    medium_confidence_threshold: float = 0.4,
) -> CrossSectionalRanking:
    """
    Create a cross-sectional ranking from horizon signals.
    
    This is the main function for converting raw signals into
    actionable ranked lists with categories and confidence buckets.
    
    Args:
        horizon_signals: Signals for all stocks at a specific horizon
        top_threshold: Percentile threshold for TOP_BUY category
        buy_threshold: Percentile threshold for BUY category
        avoid_threshold: Percentile threshold for AVOID category
        strong_avoid_threshold: Percentile threshold for STRONG_AVOID category
        high_confidence_threshold: Score threshold for HIGH confidence
        medium_confidence_threshold: Score threshold for MEDIUM confidence
    
    Returns:
        CrossSectionalRanking with all stocks ranked and categorized
    """
    # Sort signals by alpha rank score (descending)
    sorted_signals = sorted(
        horizon_signals.signals.values(),
        key=lambda s: s.alpha_rank_score,
        reverse=True
    )
    
    n_stocks = len(sorted_signals)
    ranked_stocks = []
    
    for rank, signal in enumerate(sorted_signals, 1):
        # Calculate percentile (100 = best, 0 = worst)
        percentile = 100 * (n_stocks - rank) / max(n_stocks - 1, 1)
        
        # Assign category and confidence bucket
        category = _assign_category(
            signal, percentile,
            top_threshold, buy_threshold,
            avoid_threshold, strong_avoid_threshold
        )
        confidence_bucket = _assign_confidence_bucket(
            signal.confidence_score,
            high_confidence_threshold,
            medium_confidence_threshold
        )
        
        ranked_stock = RankedStock(
            ticker=signal.ticker,
            rank=rank,
            percentile=percentile,
            category=category,
            confidence_bucket=confidence_bucket,
            signal=signal
        )
        ranked_stocks.append(ranked_stock)
    
    return CrossSectionalRanking(
        rebalance_date=horizon_signals.rebalance_date,
        horizon_days=horizon_signals.horizon_days,
        benchmark=horizon_signals.benchmark,
        ranked_stocks=ranked_stocks,
    )


def create_all_rankings(
    rebalance_signals: RebalanceSignals,
    **kwargs
) -> Dict[int, CrossSectionalRanking]:
    """
    Create rankings for all horizons from rebalance signals.
    
    Args:
        rebalance_signals: Complete signals for all horizons
        **kwargs: Additional arguments passed to create_ranking_from_signals
    
    Returns:
        Dictionary mapping horizon to CrossSectionalRanking
    """
    return {
        horizon: create_ranking_from_signals(signals, **kwargs)
        for horizon, signals in rebalance_signals.horizon_signals.items()
    }


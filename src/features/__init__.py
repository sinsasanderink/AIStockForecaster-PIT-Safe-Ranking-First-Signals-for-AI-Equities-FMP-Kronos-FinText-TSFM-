"""
Feature Engineering Module
===========================

Section 5: Feature Engineering (Bias-Safe)

This module provides PIT-safe feature computation for the AI Stock Forecaster.

Submodules:
- labels: Forward excess return labels (5.1) ✅
- price_features: Momentum, volatility, drawdown (5.2) ✅
- fundamental_features: Relative ratios vs sector (5.3) ✅
- time_decay: Sample weighting for training (5.1-5.3) ✅
- event_features: Earnings, filings, calendars (5.4) ✅
- regime_features: VIX, market trend, macro (5.5) - TODO
- missingness: "Known at time T" masks (5.6) - TODO
- hygiene: Standardization, correlation, VIF (5.7) - TODO
- neutralization: Sector/beta/market neutral IC (5.8) - TODO
- feature_store: DuckDB storage for features - TODO

CRITICAL PIT RULES:
1. All features must filter by observed_at <= asof
2. Labels mature at T+H close (filter by asof >= maturity_date)
3. No future information leakage
4. Cross-sectional standardization within universe at time T

TIME DECAY POLICY (for training, not feature computation):
- Recent observations matter more for AI stocks
- Use exponential decay with horizon-specific half-lives
- Recommended half-lives: 2.5y (20d), 3.5y (60d), 4.5y (90d)
- Apply sample_weight during model training (Section 6, 11)
"""

from .labels import (
    LabelGenerator,
    ForwardReturn,
    HORIZONS,
    DEFAULT_BENCHMARK,
)

from .price_features import (
    PriceFeatureGenerator,
    PriceFeatures,
    MOMENTUM_WINDOWS,
    cross_sectional_rank,
    cross_sectional_zscore,
)

from .fundamental_features import (
    FundamentalFeatureGenerator,
    FundamentalFeatures,
    winsorize,
    sector_neutralize,
)

from .time_decay import (
    compute_time_decay_weights,
    get_half_life_for_horizon,
    compute_effective_sample_size,
    summarize_weights,
    DEFAULT_HALF_LIFE_DAYS,
    DEFAULT_HALF_LIVES,
)

from .event_features import (
    EventFeatureGenerator,
    EventFeatures,
    PEAD_WINDOW_DAYS,
    get_event_feature_generator,
)

__all__ = [
    # Labels (5.1)
    "LabelGenerator",
    "ForwardReturn",
    "HORIZONS",
    "DEFAULT_BENCHMARK",
    # Price Features (5.2)
    "PriceFeatureGenerator",
    "PriceFeatures",
    "MOMENTUM_WINDOWS",
    "cross_sectional_rank",
    "cross_sectional_zscore",
    # Fundamental Features (5.3)
    "FundamentalFeatureGenerator",
    "FundamentalFeatures",
    "winsorize",
    "sector_neutralize",
    # Time Decay (for training)
    "compute_time_decay_weights",
    "get_half_life_for_horizon",
    "compute_effective_sample_size",
    "summarize_weights",
    "DEFAULT_HALF_LIFE_DAYS",
    "DEFAULT_HALF_LIVES",
    # Event Features (5.4)
    "EventFeatureGenerator",
    "EventFeatures",
    "PEAD_WINDOW_DAYS",
    "get_event_feature_generator",
]


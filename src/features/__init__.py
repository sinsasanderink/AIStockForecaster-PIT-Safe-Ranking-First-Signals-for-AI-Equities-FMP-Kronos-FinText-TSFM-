"""
Feature Engineering Module
===========================

Section 5: Feature Engineering (Bias-Safe)

This module provides PIT-safe feature computation for the AI Stock Forecaster.

Submodules:
- labels: Forward excess return labels (5.1)
- price_features: Momentum, volatility, drawdown (5.2) - TODO
- fundamental_features: Relative ratios vs sector (5.3) - TODO
- event_features: Earnings, filings, calendars (5.4) - TODO
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
"""

from .labels import (
    LabelGenerator,
    ForwardReturn,
    HORIZONS,
    DEFAULT_BENCHMARK,
)

__all__ = [
    # Labels (5.1)
    "LabelGenerator",
    "ForwardReturn",
    "HORIZONS",
    "DEFAULT_BENCHMARK",
]


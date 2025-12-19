"""
System Outputs Module (Section 1)
=================================

This module defines the signal-only output structures for the AI Stock Forecaster.
At each rebalance date, the system produces ranked stock recommendations and 
return distributions - NOT trades.

Key Outputs:
- Per-stock: Expected excess returns, distributions, confidence scores
- Cross-sectional: Ranked lists (buy/neutral/avoid), confidence buckets
"""

from .signals import (
    StockSignal,
    ReturnDistribution,
    SignalDriver,
    RebalanceSignals,
    HorizonSignals,
    LiquidityFlag,
    SignalSource,
)

from .rankings import (
    RankingCategory,
    ConfidenceBucket,
    RankedStock,
    CrossSectionalRanking,
    create_ranking_from_signals,
)

from .reports import (
    SignalReport,
    generate_signal_report,
    export_signals_to_csv,
    export_signals_to_json,
)

__all__ = [
    # Signals
    "StockSignal",
    "ReturnDistribution", 
    "SignalDriver",
    "RebalanceSignals",
    "HorizonSignals",
    "LiquidityFlag",
    "SignalSource",
    # Rankings
    "RankingCategory",
    "ConfidenceBucket",
    "RankedStock",
    "CrossSectionalRanking",
    "create_ranking_from_signals",
    # Reports
    "SignalReport",
    "generate_signal_report",
    "export_signals_to_csv",
    "export_signals_to_json",
]


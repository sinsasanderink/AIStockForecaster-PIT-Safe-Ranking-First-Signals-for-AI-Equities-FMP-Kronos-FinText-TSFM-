"""
Models Module
=============

Chapter 8+: Time Series Foundation Models and Fusion

This module contains model adapters and inference wrappers for:
- Kronos (K-line/OHLCV price dynamics)
- FinText-TSFM (return structure prediction)
- Fusion models (combining TSFM + tabular features)

All models must:
1. Respect PIT discipline (no future data leakage)
2. Output scores compatible with EvaluationRow format
3. Use deterministic settings for reproducibility
"""

from .kronos_adapter import KronosAdapter, kronos_scoring_function
from .fintext_adapter import FinTextAdapter, fintext_scoring_function
from .finbert_scorer import FinBERTScorer

__all__ = [
    "KronosAdapter",
    "kronos_scoring_function",
    "FinTextAdapter",
    "fintext_scoring_function",
    "FinBERTScorer",
]


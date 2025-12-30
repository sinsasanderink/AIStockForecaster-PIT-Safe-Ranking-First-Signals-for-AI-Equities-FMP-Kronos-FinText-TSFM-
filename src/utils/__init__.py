"""
Shared Utility Functions
========================

Common utilities used across the codebase.
"""

from src.utils.env import (
    load_repo_dotenv,
    resolve_fmp_key,
    get_repo_root,
)

from src.utils.price_validation import (
    detect_split_discontinuities,
    validate_price_series_consistency,
    SplitDiscontinuityError,
)

__all__ = [
    # Environment
    "load_repo_dotenv",
    "resolve_fmp_key",
    "get_repo_root",
    # Price validation
    "detect_split_discontinuities",
    "validate_price_series_consistency",
    "SplitDiscontinuityError",
]


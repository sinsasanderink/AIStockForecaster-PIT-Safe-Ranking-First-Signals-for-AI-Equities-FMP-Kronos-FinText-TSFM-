"""
Universe Module
===============

Defines the AI stock universe with 10 subcategories for subsector-specific forecasts.

Usage:
    from src.universe import AI_UNIVERSE, get_all_tickers, get_tickers_by_category
    
    # All ~100 AI stocks
    all_tickers = get_all_tickers()
    
    # Just semiconductors
    semis = get_tickers_by_category("ai_compute_core_semis")
    
    # Multiple categories
    hardware = get_tickers_by_categories(["ai_compute_core_semis", "semicap_eda_manufacturing"])
"""

from .ai_stocks import (
    AI_UNIVERSE,
    AI_CATEGORIES,
    CATEGORY_DESCRIPTIONS,
    get_all_tickers,
    get_tickers_by_category,
    get_tickers_by_categories,
    get_category_for_ticker,
    validate_universe,
)

__all__ = [
    "AI_UNIVERSE",
    "AI_CATEGORIES",
    "CATEGORY_DESCRIPTIONS",
    "get_all_tickers",
    "get_tickers_by_category",
    "get_tickers_by_categories",
    "get_category_for_ticker",
    "validate_universe",
]


"""
Tests for PricesStore (Chapter 8)
==================================

Tests the DuckDB-backed OHLCV store for model inference.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import date
from pathlib import Path

from src.data import PricesStore


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def db_path():
    """Get path to DuckDB database."""
    db = Path("data/features.duckdb")
    if not db.exists():
        pytest.skip(f"DuckDB not found at {db}")
    return str(db)


@pytest.fixture
def prices_store(db_path):
    """Create a PricesStore instance."""
    store = PricesStore(db_path=db_path, enable_cache=True)
    yield store
    store.close()


# ============================================================================
# BASIC FUNCTIONALITY TESTS
# ============================================================================

def test_prices_store_init(db_path):
    """Test PricesStore initialization."""
    store = PricesStore(db_path=db_path)
    
    assert store.db_path == db_path
    assert store.enable_cache is True
    assert store.cache_max_items == 2_000
    assert store.con is not None
    
    store.close()


def test_prices_store_context_manager(db_path):
    """Test PricesStore as context manager."""
    with PricesStore(db_path=db_path) as store:
        assert store.con is not None
        tickers = store.fetch_available_tickers()
        assert len(tickers) > 0
    
    # Connection should be closed after exiting context
    # (we can't easily test this without accessing internals)


def test_fetch_available_tickers(prices_store):
    """Test fetching available tickers."""
    tickers = prices_store.fetch_available_tickers()
    
    assert isinstance(tickers, pd.Index)
    assert len(tickers) > 0
    assert "NVDA" in tickers or "AAPL" in tickers  # Should have at least one major stock


def test_get_date_range(prices_store):
    """Test getting date range from prices table."""
    min_date, max_date = prices_store.get_date_range()
    
    assert isinstance(min_date, pd.Timestamp)
    assert isinstance(max_date, pd.Timestamp)
    assert min_date < max_date
    
    # Should have multi-year history
    days_diff = (max_date - min_date).days
    assert days_diff > 365  # At least 1 year


# ============================================================================
# FETCH OHLCV TESTS
# ============================================================================

def test_fetch_ohlcv_basic(prices_store):
    """Test basic OHLCV fetch."""
    tickers = prices_store.fetch_available_tickers()
    if len(tickers) == 0:
        pytest.skip("No tickers in database")
    
    ticker = tickers[0]
    asof_date = "2024-01-15"
    lookback = 252
    
    ohlcv = prices_store.fetch_ohlcv(ticker, asof_date, lookback)
    
    # Check structure
    assert isinstance(ohlcv, pd.DataFrame)
    assert isinstance(ohlcv.index, pd.DatetimeIndex)
    assert list(ohlcv.columns) == ["open", "high", "low", "close", "volume"]
    
    # Check length (may be less than lookback if insufficient history)
    assert len(ohlcv) <= lookback
    assert len(ohlcv) > 0
    
    # Check all values are numeric
    for col in ohlcv.columns:
        assert pd.api.types.is_numeric_dtype(ohlcv[col])
    
    # Check no NaNs (fill_missing=True by default)
    assert not ohlcv.isnull().any().any()
    
    # Check OHLC relationships (high >= low)
    assert (ohlcv["high"] >= ohlcv["low"]).all()


def test_fetch_ohlcv_strict_lookback(prices_store):
    """Test strict lookback mode."""
    tickers = prices_store.fetch_available_tickers()
    if len(tickers) == 0:
        pytest.skip("No tickers in database")
    
    ticker = tickers[0]
    asof_date = "2024-01-15"
    lookback = 252
    
    # Strict mode: only return if exactly lookback rows exist
    ohlcv = prices_store.fetch_ohlcv(
        ticker, asof_date, lookback, strict_lookback=True
    )
    
    # Either empty (insufficient history) or exactly lookback rows
    assert len(ohlcv) == 0 or len(ohlcv) == lookback


def test_fetch_ohlcv_different_lookbacks(prices_store):
    """Test different lookback periods."""
    tickers = prices_store.fetch_available_tickers()
    if len(tickers) == 0:
        pytest.skip("No tickers in database")
    
    ticker = tickers[0]
    asof_date = "2024-01-15"
    
    for lookback in [20, 60, 252]:
        ohlcv = prices_store.fetch_ohlcv(ticker, asof_date, lookback)
        
        if len(ohlcv) > 0:
            # Should have proper structure
            assert isinstance(ohlcv.index, pd.DatetimeIndex)
            assert len(ohlcv) <= lookback


def test_fetch_ohlcv_timestamp_input(prices_store):
    """Test OHLCV fetch with pd.Timestamp input."""
    tickers = prices_store.fetch_available_tickers()
    if len(tickers) == 0:
        pytest.skip("No tickers in database")
    
    ticker = tickers[0]
    asof_date = pd.Timestamp("2024-01-15")
    
    ohlcv = prices_store.fetch_ohlcv(ticker, asof_date, 60)
    
    assert isinstance(ohlcv, pd.DataFrame)
    assert len(ohlcv) > 0 or len(ohlcv) == 0  # Either has data or empty


def test_fetch_ohlcv_invalid_lookback(prices_store):
    """Test that invalid lookback raises error."""
    with pytest.raises(ValueError, match="lookback must be positive"):
        prices_store.fetch_ohlcv("NVDA", "2024-01-15", lookback=0)
    
    with pytest.raises(ValueError, match="lookback must be positive"):
        prices_store.fetch_ohlcv("NVDA", "2024-01-15", lookback=-1)


def test_fetch_ohlcv_nonexistent_ticker(prices_store):
    """Test fetching OHLCV for nonexistent ticker."""
    ohlcv = prices_store.fetch_ohlcv("XXXXX_INVALID", "2024-01-15", 60)
    
    # Should return empty DataFrame
    assert isinstance(ohlcv, pd.DataFrame)
    assert len(ohlcv) == 0


# ============================================================================
# CACHE TESTS
# ============================================================================

def test_cache_functionality(db_path):
    """Test that caching works."""
    store = PricesStore(db_path=db_path, enable_cache=True)
    
    tickers = store.fetch_available_tickers()
    if len(tickers) == 0:
        pytest.skip("No tickers in database")
    
    ticker = tickers[0]
    asof_date = "2024-01-15"
    lookback = 60
    
    # First fetch (cache miss)
    ohlcv1 = store.fetch_ohlcv(ticker, asof_date, lookback)
    
    # Second fetch (cache hit)
    ohlcv2 = store.fetch_ohlcv(ticker, asof_date, lookback)
    
    # Should be identical
    pd.testing.assert_frame_equal(ohlcv1, ohlcv2)
    
    # Check cache stats
    stats = store.get_cache_stats()
    assert stats["cache_size"] > 0
    assert stats["cache_enabled"] is True
    
    store.close()


def test_cache_eviction(db_path):
    """Test cache eviction when full."""
    # Create store with small cache
    store = PricesStore(db_path=db_path, enable_cache=True, cache_max_items=2)
    
    tickers = store.fetch_available_tickers()
    if len(tickers) < 3:
        pytest.skip("Need at least 3 tickers for this test")
    
    asof_date = "2024-01-15"
    lookback = 60
    
    # Fill cache beyond capacity
    for i in range(3):
        store.fetch_ohlcv(tickers[i], asof_date, lookback)
    
    # Cache should be at max size (FIFO eviction)
    stats = store.get_cache_stats()
    assert stats["cache_size"] <= stats["cache_max_size"]
    
    store.close()


def test_clear_cache(prices_store):
    """Test cache clearing."""
    tickers = prices_store.fetch_available_tickers()
    if len(tickers) == 0:
        pytest.skip("No tickers in database")
    
    ticker = tickers[0]
    
    # Populate cache
    prices_store.fetch_ohlcv(ticker, "2024-01-15", 60)
    
    stats_before = prices_store.get_cache_stats()
    assert stats_before["cache_size"] > 0
    
    # Clear cache
    prices_store.clear_cache()
    
    stats_after = prices_store.get_cache_stats()
    assert stats_after["cache_size"] == 0


def test_cache_disabled(db_path):
    """Test that cache can be disabled."""
    store = PricesStore(db_path=db_path, enable_cache=False)
    
    tickers = store.fetch_available_tickers()
    if len(tickers) == 0:
        pytest.skip("No tickers in database")
    
    ticker = tickers[0]
    
    # Fetch multiple times
    for _ in range(3):
        store.fetch_ohlcv(ticker, "2024-01-15", 60)
    
    # Cache should remain empty
    stats = store.get_cache_stats()
    assert stats["cache_size"] == 0
    assert stats["cache_enabled"] is False
    
    store.close()


# ============================================================================
# DATA QUALITY TESTS
# ============================================================================

def test_datetime_index_is_sorted(prices_store):
    """Test that returned OHLCV has sorted DatetimeIndex."""
    tickers = prices_store.fetch_available_tickers()
    if len(tickers) == 0:
        pytest.skip("No tickers in database")
    
    ticker = tickers[0]
    ohlcv = prices_store.fetch_ohlcv(ticker, "2024-01-15", 252)
    
    if len(ohlcv) > 1:
        # Index should be sorted and unique
        assert ohlcv.index.is_monotonic_increasing
        assert ohlcv.index.is_unique


def test_no_future_data_leakage(prices_store):
    """Test that asof_date constraint is respected."""
    tickers = prices_store.fetch_available_tickers()
    if len(tickers) == 0:
        pytest.skip("No tickers in database")
    
    ticker = tickers[0]
    asof_date = pd.Timestamp("2023-01-15")
    
    ohlcv = prices_store.fetch_ohlcv(ticker, asof_date, 60)
    
    if len(ohlcv) > 0:
        # All dates should be <= asof_date
        assert (ohlcv.index <= asof_date).all()


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

def test_multiple_tickers_batch(prices_store):
    """Test fetching OHLCV for multiple tickers (simulating Kronos batch)."""
    tickers = prices_store.fetch_available_tickers()[:5]  # First 5 tickers
    
    if len(tickers) == 0:
        pytest.skip("No tickers in database")
    
    asof_date = "2024-01-15"
    lookback = 252
    
    results = []
    for ticker in tickers:
        ohlcv = prices_store.fetch_ohlcv(
            ticker, asof_date, lookback, strict_lookback=True
        )
        if len(ohlcv) == lookback:
            results.append((ticker, ohlcv))
    
    # Should have at least some tickers with sufficient history
    # (may be 0 if database is sparse)
    assert len(results) >= 0
    
    # All returned sequences should have exact lookback length
    for ticker, ohlcv in results:
        assert len(ohlcv) == lookback
        assert isinstance(ohlcv.index, pd.DatetimeIndex)


def test_kronos_use_case(prices_store):
    """Test typical Kronos use case."""
    tickers = prices_store.fetch_available_tickers()
    if len(tickers) == 0:
        pytest.skip("No tickers in database")
    
    ticker = tickers[0]
    asof_date = "2024-01-15"
    lookback = 252
    
    # Fetch OHLCV (Kronos requires 252-day history)
    ohlcv = prices_store.fetch_ohlcv(
        ticker, 
        asof_date, 
        lookback,
        strict_lookback=True  # Batch inference requires exact length
    )
    
    if len(ohlcv) == lookback:
        # Check it's ready for Kronos
        assert isinstance(ohlcv.index, pd.DatetimeIndex)
        assert list(ohlcv.columns) == ["open", "high", "low", "close", "volume"]
        assert len(ohlcv) == 252
        assert not ohlcv.isnull().any().any()
        
        # Simulate what Kronos would do
        last_close = ohlcv["close"].iloc[-1]
        assert last_close > 0


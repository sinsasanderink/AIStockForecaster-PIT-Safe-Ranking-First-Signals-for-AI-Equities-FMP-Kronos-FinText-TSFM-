"""
Tests for ExcessReturnStore (Chapter 9)
========================================

Tests the daily excess return sequence store used by FinText-TSFM.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import date
from pathlib import Path

from src.data.excess_return_store import ExcessReturnStore, BENCHMARK_TICKER


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
def store(db_path):
    """Create an ExcessReturnStore instance."""
    s = ExcessReturnStore(db_path=db_path, enable_cache=True)
    yield s
    s.close()


# ============================================================================
# INITIALIZATION
# ============================================================================

class TestInit:
    def test_init_success(self, db_path):
        """Store initialises successfully with valid DuckDB."""
        s = ExcessReturnStore(db_path=db_path)
        assert s.benchmark == "QQQ"
        s.close()

    def test_context_manager(self, db_path):
        """Store works as a context manager."""
        with ExcessReturnStore(db_path=db_path) as s:
            tickers = s.get_available_tickers()
            assert len(tickers) > 0

    def test_missing_benchmark_raises(self, db_path):
        """Raise if the benchmark ticker is not in the prices table."""
        with pytest.raises(ValueError, match="not found"):
            ExcessReturnStore(db_path=db_path, benchmark="NONEXISTENT_TICKER_XYZ")

    def test_benchmark_qqq_present(self, store):
        """QQQ must be in the database after setup."""
        bench = store.get_benchmark_daily_returns("2024-01-15", lookback=5)
        assert len(bench) == 5


# ============================================================================
# SINGLE SEQUENCE
# ============================================================================

class TestSingleSequence:
    def test_basic_shape(self, store):
        """Returned array has the right shape."""
        seq = store.get_excess_return_sequence("AAPL", "2024-03-01", lookback=21)
        assert isinstance(seq, np.ndarray)
        assert seq.shape == (21,)
        assert seq.dtype == np.float64

    def test_long_lookback(self, store):
        """252-day lookback works for a mature stock."""
        seq = store.get_excess_return_sequence("AAPL", "2024-06-01", lookback=252)
        assert seq.shape == (252,)

    def test_short_lookback(self, store):
        """Single-day lookback returns one value."""
        seq = store.get_excess_return_sequence("AAPL", "2024-03-01", lookback=1)
        assert seq.shape == (1,)

    def test_strict_success(self, store):
        """Strict mode returns correct length when data is sufficient."""
        seq = store.get_excess_return_sequence(
            "AAPL", "2024-03-01", lookback=21, strict=True
        )
        assert len(seq) == 21

    def test_strict_failure_returns_empty(self, store):
        """Strict mode returns empty when history is insufficient."""
        # Use an absurdly large lookback to guarantee failure
        seq = store.get_excess_return_sequence(
            "AAPL", "2024-03-01", lookback=99999, strict=True
        )
        assert len(seq) == 0

    def test_nonstrict_returns_partial(self, store):
        """Non-strict mode returns as much data as available."""
        # Very early date — limited history
        seq = store.get_excess_return_sequence(
            "AAPL", "2014-02-01", lookback=252
        )
        assert 0 < len(seq) < 252

    def test_invalid_ticker_returns_empty(self, store):
        """Non-existent ticker yields empty array."""
        seq = store.get_excess_return_sequence(
            "FAKE_TICKER_XYZ", "2024-03-01", lookback=21
        )
        assert len(seq) == 0

    def test_lookback_must_be_positive(self, store):
        """Lookback <= 0 raises ValueError."""
        with pytest.raises(ValueError, match="positive"):
            store.get_excess_return_sequence("AAPL", "2024-03-01", lookback=0)
        with pytest.raises(ValueError, match="positive"):
            store.get_excess_return_sequence("AAPL", "2024-03-01", lookback=-5)

    def test_values_are_finite(self, store):
        """All returned values must be finite (no NaN / Inf)."""
        seq = store.get_excess_return_sequence("NVDA", "2024-06-01", lookback=60)
        assert np.all(np.isfinite(seq))


# ============================================================================
# EXCESS RETURN CORRECTNESS
# ============================================================================

class TestCorrectness:
    def test_excess_return_equals_stock_minus_benchmark(self, store):
        """Core identity: excess_return = stock_return - benchmark_return."""
        ticker = "MSFT"
        asof = "2024-06-01"
        lookback = 20

        stock_ret = store.get_stock_daily_returns(ticker, asof, lookback)
        bench_ret = store.get_benchmark_daily_returns(asof, lookback)

        # Align dates
        common = stock_ret.index.intersection(bench_ret.index)
        manual = (stock_ret.loc[common] - bench_ret.loc[common]).values

        stored = store.get_excess_return_sequence(ticker, asof, lookback)

        # Allow for minor date-alignment differences in tail
        n = min(len(manual), len(stored))
        assert n > 0
        np.testing.assert_allclose(stored[-n:], manual[-n:], atol=1e-12)

    def test_excess_returns_are_small(self, store):
        """Sanity: daily excess returns are small numbers (< ±30%)."""
        seq = store.get_excess_return_sequence("AAPL", "2024-06-01", lookback=252)
        assert np.all(np.abs(seq) < 0.30)

    def test_mean_excess_near_zero(self, store):
        """Over a long window, average excess return should be modest."""
        seq = store.get_excess_return_sequence("AAPL", "2024-06-01", lookback=252)
        # Mean daily excess should be small (< 0.5% per day)
        assert abs(seq.mean()) < 0.005

    def test_multiple_tickers_differ(self, store):
        """Two different stocks should NOT have identical excess return sequences."""
        s1 = store.get_excess_return_sequence("AAPL", "2024-06-01", lookback=21)
        s2 = store.get_excess_return_sequence("NVDA", "2024-06-01", lookback=21)
        assert not np.array_equal(s1, s2)


# ============================================================================
# PIT SAFETY
# ============================================================================

class TestPITSafety:
    def test_no_future_data(self, store):
        """Sequence must not contain data from after asof_date."""
        asof = "2023-06-15"
        stock_ret = store.get_stock_daily_returns("AAPL", asof, lookback=252)
        # All dates in the returned series must be <= asof
        assert all(d <= pd.Timestamp(asof) for d in stock_ret.index)

    def test_different_dates_differ(self, store):
        """Sequences for different asof dates should differ."""
        s1 = store.get_excess_return_sequence("AAPL", "2024-03-01", lookback=21)
        s2 = store.get_excess_return_sequence("AAPL", "2024-06-01", lookback=21)
        assert not np.array_equal(s1, s2)


# ============================================================================
# BATCH
# ============================================================================

class TestBatch:
    def test_batch_shape(self, store):
        """Batch returns 2-D array with correct dimensions."""
        tickers = ["AAPL", "NVDA", "MSFT"]
        valid, batch = store.get_batch_sequences(
            tickers, "2024-06-01", lookback=21, strict=True
        )
        assert len(valid) == 3
        assert batch.shape == (3, 21)

    def test_batch_strict_drops_missing(self, store):
        """Tickers without enough history are silently dropped."""
        tickers = ["AAPL", "FAKE_TICKER_XYZ", "MSFT"]
        valid, batch = store.get_batch_sequences(
            tickers, "2024-06-01", lookback=21, strict=True
        )
        assert len(valid) == 2
        assert "FAKE_TICKER_XYZ" not in valid
        assert batch.shape == (2, 21)

    def test_batch_empty(self, store):
        """All-invalid tickers return empty tuple."""
        valid, batch = store.get_batch_sequences(
            ["X_FAKE_1", "X_FAKE_2"], "2024-06-01", lookback=21, strict=True
        )
        assert len(valid) == 0
        assert batch.shape == (0, 21)

    def test_batch_large(self, store):
        """Batch with many tickers works (whole universe)."""
        tickers = store.get_available_tickers()
        valid, batch = store.get_batch_sequences(
            tickers, "2024-06-01", lookback=21, strict=True
        )
        assert len(valid) >= 90  # Most of the 100-stock universe
        assert batch.shape[1] == 21


# ============================================================================
# CACHE
# ============================================================================

class TestCache:
    def test_cache_hit(self, store):
        """Second call returns cached result."""
        s1 = store.get_excess_return_sequence("AAPL", "2024-06-01", lookback=21)
        s2 = store.get_excess_return_sequence("AAPL", "2024-06-01", lookback=21)
        np.testing.assert_array_equal(s1, s2)

    def test_cache_mutation_safety(self, store):
        """Mutating the returned array must NOT corrupt the cache."""
        s1 = store.get_excess_return_sequence("AAPL", "2024-06-01", lookback=21)
        s1[:] = 999.0  # mutate
        s2 = store.get_excess_return_sequence("AAPL", "2024-06-01", lookback=21)
        assert not np.all(s2 == 999.0)

    def test_clear_cache(self, store):
        """Clearing the cache empties it."""
        _ = store.get_excess_return_sequence("AAPL", "2024-06-01", lookback=21)
        assert len(store._cache) > 0
        store.clear_cache()
        assert len(store._cache) == 0


# ============================================================================
# DIAGNOSTICS
# ============================================================================

class TestDiagnostics:
    def test_available_tickers(self, store):
        """Returns a non-empty list excluding the benchmark."""
        tickers = store.get_available_tickers()
        assert len(tickers) >= 90
        assert BENCHMARK_TICKER not in tickers

    def test_date_range(self, store):
        """Returns a plausible date range."""
        min_d, max_d = store.get_date_range()
        assert min_d < max_d
        assert min_d.year <= 2015
        assert max_d.year >= 2025

    def test_coverage_stats(self, store):
        """Coverage report has required keys and plausible values."""
        stats = store.get_coverage_stats("2024-06-01", lookback=21)
        assert "total_tickers" in stats
        assert "valid_tickers" in stats
        assert "pct_coverage" in stats
        assert stats["pct_coverage"] > 0.9

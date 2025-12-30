"""
Tests for Qlib Adapter and Parity (Chapter 6.6)

CRITICAL: These tests ensure:
1. Format conversion is correct (timezone-aware datetime, proper index)
2. No duplicate index entries
3. IC/RankIC parity with our manual calculations
4. Index alignment pitfalls are handled
"""

import pytest
import pandas as pd
import numpy as np
from datetime import date, datetime, timezone
from pathlib import Path
import tempfile
import shutil

from src.evaluation.qlib_adapter import (
    is_qlib_available,
    to_qlib_format,
    from_qlib_format,
    validate_qlib_frame,
    check_ic_parity,
)
from src.evaluation.definitions import get_market_close_utc
from src.evaluation.metrics import compute_rankic_per_date


# ============================================================================
# TEST FIXTURES
# ============================================================================

@pytest.fixture
def sample_eval_rows():
    """Create sample evaluation rows for testing."""
    np.random.seed(42)
    
    data = []
    dates = [date(2023, 1, 1), date(2023, 2, 1), date(2023, 3, 1)]
    tickers = ["AAPL", "MSFT", "GOOGL", "NVDA", "META"]
    
    for d in dates:
        for i, ticker in enumerate(tickers):
            data.append({
                "as_of_date": d,
                "ticker": ticker,
                "stable_id": f"STABLE_{ticker}",
                "horizon": 20,
                "fold_id": "fold_01",
                "score": np.random.randn(),
                "excess_return": np.random.randn() * 0.05
            })
    
    return pd.DataFrame(data)


@pytest.fixture
def perfect_positive_data():
    """Create data where score perfectly predicts return (RankIC should be ~1)."""
    data = []
    d = date(2023, 1, 1)
    
    for i in range(20):
        score = i / 20.0
        data.append({
            "as_of_date": d,
            "ticker": f"STOCK_{i}",
            "stable_id": f"STABLE_{i}",
            "horizon": 20,
            "fold_id": "fold_01",
            "score": score,
            "excess_return": score  # Perfect positive correlation
        })
    
    return pd.DataFrame(data)


@pytest.fixture
def perfect_negative_data():
    """Create data where score perfectly negatively predicts return (RankIC should be ~-1)."""
    data = []
    d = date(2023, 1, 1)
    
    for i in range(20):
        score = i / 20.0
        data.append({
            "as_of_date": d,
            "ticker": f"STOCK_{i}",
            "stable_id": f"STABLE_{i}",
            "horizon": 20,
            "fold_id": "fold_01",
            "score": score,
            "excess_return": -score  # Perfect negative correlation
        })
    
    return pd.DataFrame(data)


@pytest.fixture
def random_data():
    """Create random data (RankIC should be ~0)."""
    np.random.seed(42)
    data = []
    d = date(2023, 1, 1)
    
    for i in range(100):  # More samples for reliable stats
        data.append({
            "as_of_date": d,
            "ticker": f"STOCK_{i}",
            "stable_id": f"STABLE_{i}",
            "horizon": 20,
            "fold_id": "fold_01",
            "score": np.random.randn(),
            "excess_return": np.random.randn() * 0.05
        })
    
    return pd.DataFrame(data)


@pytest.fixture
def temp_output_dir():
    """Create temporary output directory."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir)


# ============================================================================
# TEST FORMAT CONVERSION
# ============================================================================

class TestFormatConversion:
    """Test conversion between our format and Qlib format."""
    
    def test_to_qlib_format_basic(self, sample_eval_rows):
        """Test basic conversion to Qlib format."""
        qlib_df = to_qlib_format(sample_eval_rows)
        
        # Check structure
        assert isinstance(qlib_df.index, pd.MultiIndex)
        assert qlib_df.index.names == ["datetime", "instrument"]
        assert "score" in qlib_df.columns
        assert "label" in qlib_df.columns
    
    def test_datetime_is_tz_aware(self, sample_eval_rows):
        """Test that datetime is timezone-aware (UTC)."""
        qlib_df = to_qlib_format(sample_eval_rows)
        
        dt_level = qlib_df.index.get_level_values("datetime")
        assert dt_level.tz is not None
        assert str(dt_level.tz) == "UTC"
    
    def test_datetime_is_market_close(self, sample_eval_rows):
        """Test that datetime corresponds to market close."""
        qlib_df = to_qlib_format(sample_eval_rows)
        
        dt_level = qlib_df.index.get_level_values("datetime")
        
        # Check first date
        first_dt = dt_level[0]
        expected_utc = get_market_close_utc(date(2023, 1, 1))
        
        # Hour should be either 21 (winter) or 20 (summer) UTC
        assert first_dt.hour in [20, 21]
        assert first_dt.minute == 0
    
    def test_uses_stable_id_by_default(self, sample_eval_rows):
        """Test that stable_id is used as instrument by default."""
        qlib_df = to_qlib_format(sample_eval_rows)
        
        inst_level = qlib_df.index.get_level_values("instrument")
        
        # All instruments should be STABLE_* format
        assert all(inst.startswith("STABLE_") for inst in inst_level)
    
    def test_can_use_ticker_as_instrument(self, sample_eval_rows):
        """Test using ticker instead of stable_id."""
        qlib_df = to_qlib_format(sample_eval_rows, use_stable_id=False)
        
        inst_level = qlib_df.index.get_level_values("instrument")
        
        # Should be raw ticker symbols
        assert "AAPL" in inst_level.unique()
    
    def test_preserves_row_count(self, sample_eval_rows):
        """Test that conversion preserves row count."""
        qlib_df = to_qlib_format(sample_eval_rows)
        
        assert len(qlib_df) == len(sample_eval_rows)
    
    def test_round_trip_conversion(self, sample_eval_rows):
        """Test that round-trip conversion preserves data."""
        qlib_df = to_qlib_format(sample_eval_rows)
        back_df = from_qlib_format(qlib_df)
        
        # Should have same number of rows
        assert len(back_df) == len(sample_eval_rows)
        
        # Score values should be preserved
        assert np.allclose(
            sorted(back_df["score"].values),
            sorted(sample_eval_rows["score"].values)
        )


# ============================================================================
# TEST VALIDATION
# ============================================================================

class TestValidation:
    """Test Qlib frame validation."""
    
    def test_valid_frame_passes(self, sample_eval_rows):
        """Valid frame should pass validation."""
        qlib_df = to_qlib_format(sample_eval_rows)
        is_valid, msg = validate_qlib_frame(qlib_df)
        
        assert is_valid, f"Validation failed: {msg}"
    
    def test_detects_duplicate_index(self, sample_eval_rows):
        """Should detect duplicate index entries."""
        qlib_df = to_qlib_format(sample_eval_rows)
        
        # Manually introduce duplicate
        qlib_df_dup = pd.concat([qlib_df, qlib_df.iloc[[0]]])
        
        is_valid, msg = validate_qlib_frame(qlib_df_dup)
        
        assert not is_valid
        assert "duplicate" in msg.lower()
    
    def test_detects_missing_score(self, sample_eval_rows):
        """Should detect missing score column."""
        qlib_df = to_qlib_format(sample_eval_rows)
        qlib_df = qlib_df.drop(columns=["score"])
        
        is_valid, msg = validate_qlib_frame(qlib_df)
        
        assert not is_valid
        assert "score" in msg.lower()
    
    def test_detects_non_multiindex(self, sample_eval_rows):
        """Should detect non-MultiIndex DataFrame."""
        qlib_df = to_qlib_format(sample_eval_rows)
        flat_df = qlib_df.reset_index()
        
        is_valid, msg = validate_qlib_frame(flat_df)
        
        assert not is_valid
        assert "multiindex" in msg.lower()


# ============================================================================
# TEST IC PARITY
# ============================================================================

class TestICParity:
    """Test IC parity between our implementation and Qlib's."""
    
    def test_perfect_positive_ic(self, perfect_positive_data):
        """Perfect positive correlation should give IC ≈ 1."""
        # Our IC
        our_ic = compute_rankic_per_date(perfect_positive_data)
        
        # Check parity
        is_parity, qlib_ic, msg = check_ic_parity(
            perfect_positive_data,
            our_ic,
            tolerance=0.001
        )
        
        # Both should be close to 1
        assert our_ic > 0.99, f"Our IC should be ~1, got {our_ic}"
        
        # Should pass parity
        if is_qlib_available():
            assert is_parity, f"Parity check failed: {msg}"
    
    def test_perfect_negative_ic(self, perfect_negative_data):
        """Perfect negative correlation should give IC ≈ -1."""
        # Our IC
        our_ic = compute_rankic_per_date(perfect_negative_data)
        
        # Check parity
        is_parity, qlib_ic, msg = check_ic_parity(
            perfect_negative_data,
            our_ic,
            tolerance=0.001
        )
        
        # Both should be close to -1
        assert our_ic < -0.99, f"Our IC should be ~-1, got {our_ic}"
        
        # Should pass parity
        if is_qlib_available():
            assert is_parity, f"Parity check failed: {msg}"
    
    def test_random_ic_near_zero(self, random_data):
        """Random data should give IC ≈ 0 (within tolerance)."""
        # Our IC
        our_ic = compute_rankic_per_date(random_data)
        
        # Should be close to zero (within ~0.2 for random data)
        assert abs(our_ic) < 0.25, f"Random IC should be ~0, got {our_ic}"
    
    def test_parity_tolerance(self, sample_eval_rows):
        """Test parity check with tolerance."""
        our_ic = compute_rankic_per_date(sample_eval_rows)
        
        # Should pass with 0.001 tolerance
        is_parity, _, _ = check_ic_parity(
            sample_eval_rows,
            our_ic,
            tolerance=0.001
        )
        
        # Qlib uses same Spearman correlation, so should always pass
        if is_qlib_available():
            assert is_parity


# ============================================================================
# TEST INDEX ALIGNMENT PITFALLS
# ============================================================================

class TestIndexAlignmentPitfalls:
    """Test handling of common index alignment issues."""
    
    def test_instrument_strings_preserved(self, sample_eval_rows):
        """Instrument strings should be preserved exactly."""
        qlib_df = to_qlib_format(sample_eval_rows)
        
        inst_level = qlib_df.index.get_level_values("instrument")
        original_ids = set(sample_eval_rows["stable_id"])
        qlib_ids = set(inst_level)
        
        assert original_ids == qlib_ids
    
    def test_no_nat_in_datetime(self, sample_eval_rows):
        """No NaT values should be in datetime index."""
        qlib_df = to_qlib_format(sample_eval_rows)
        
        dt_level = qlib_df.index.get_level_values("datetime")
        assert not dt_level.isna().any()
    
    def test_no_empty_instruments(self, sample_eval_rows):
        """No empty instruments should be in index."""
        qlib_df = to_qlib_format(sample_eval_rows)
        
        inst_level = qlib_df.index.get_level_values("instrument")
        assert not (inst_level == "").any()
        assert not inst_level.isna().any()
    
    def test_handles_missing_values(self):
        """Should handle rows with missing values."""
        data = pd.DataFrame({
            "as_of_date": [date(2023, 1, 1), date(2023, 1, 1)],
            "ticker": ["A", "B"],
            "stable_id": ["S_A", "S_B"],
            "score": [0.5, np.nan],  # One missing score
            "excess_return": [0.1, 0.2]
        })
        
        # Should still convert (NaN preserved in DataFrame)
        qlib_df = to_qlib_format(data)
        
        # Score column should have NaN
        assert qlib_df["score"].isna().sum() == 1


# ============================================================================
# TEST DETERMINISM
# ============================================================================

class TestDeterminism:
    """Test that conversion is deterministic."""
    
    def test_same_input_same_output(self, sample_eval_rows):
        """Same input should produce identical output."""
        qlib_df1 = to_qlib_format(sample_eval_rows)
        qlib_df2 = to_qlib_format(sample_eval_rows)
        
        pd.testing.assert_frame_equal(qlib_df1, qlib_df2)
    
    def test_shuffled_input_same_output_sorted(self, sample_eval_rows):
        """Shuffled input should produce identical output after sorting."""
        shuffled = sample_eval_rows.sample(frac=1.0, random_state=42)
        
        qlib_df1 = to_qlib_format(sample_eval_rows)
        qlib_df2 = to_qlib_format(shuffled)
        
        # Sort both
        qlib_df1_sorted = qlib_df1.sort_index()
        qlib_df2_sorted = qlib_df2.sort_index()
        
        pd.testing.assert_frame_equal(qlib_df1_sorted, qlib_df2_sorted)


# ============================================================================
# TEST QLIB AVAILABILITY
# ============================================================================

class TestQlibAvailability:
    """Test Qlib availability detection."""
    
    def test_availability_check_returns_bool(self):
        """is_qlib_available should return bool."""
        result = is_qlib_available()
        assert isinstance(result, bool)
    
    def test_parity_check_works_without_qlib(self, sample_eval_rows):
        """Parity check should not fail if Qlib not installed."""
        our_ic = compute_rankic_per_date(sample_eval_rows)
        
        # Should return gracefully even without Qlib
        is_parity, qlib_ic, msg = check_ic_parity(
            sample_eval_rows,
            our_ic,
            tolerance=0.001
        )
        
        # Should have some result (check for bool-like value)
        assert is_parity in [True, False]
        assert isinstance(msg, str)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


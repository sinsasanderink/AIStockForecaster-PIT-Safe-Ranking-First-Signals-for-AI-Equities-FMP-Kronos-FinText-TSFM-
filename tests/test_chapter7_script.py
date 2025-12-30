"""
Test Chapter 7 Execution Script

Smoke test that runs the Chapter 7.5 script in SMOKE_MODE using synthetic data.
Does NOT require DuckDB or live data (fast for CI).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import pandas as pd
import numpy as np
from datetime import date, timedelta
import tempfile
import shutil


@pytest.fixture
def temp_output_dir():
    """Create temporary output directory."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    # Cleanup
    if temp_dir.exists():
        shutil.rmtree(temp_dir)


@pytest.fixture
def synthetic_features_for_script():
    """Create minimal synthetic features for script testing."""
    np.random.seed(42)
    
    # Create data for ~2 years (enough for walk-forward in SMOKE_MODE)
    # SMOKE_MODE needs enough history for initial training window
    dates = pd.date_range("2016-01-01", periods=500, freq='D')
    tickers = [f"TICK{i:03d}" for i in range(10)]
    
    rows = []
    for d in dates:
        for ticker in tickers:
            rows.append({
                "date": d.date(),
                "ticker": ticker,
                "stable_id": f"stable_{ticker}",
                "mom_1m": np.random.randn(),
                "mom_3m": np.random.randn(),
                "mom_6m": np.random.randn(),
                "mom_12m": np.random.randn(),
                "vol_20d": abs(np.random.randn() * 0.2),
                "vol_60d": abs(np.random.randn() * 0.2),
                "max_drawdown_60d": -abs(np.random.randn() * 0.1),
                "adv_20d": abs(np.random.randn() * 1e6),
                "excess_return_20d": np.random.randn() * 0.05,
                "excess_return_60d": np.random.randn() * 0.08,
                "excess_return_90d": np.random.randn() * 0.10,
            })
    
    return pd.DataFrame(rows)


def test_chapter7_script_imports():
    """
    Test that Chapter 7.5 script can be imported and has expected functions.
    
    This is a lightweight smoke test that doesn't require running the full pipeline.
    """
    sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
    
    # Test that script can be imported
    import run_chapter7_tabular_lgb
    
    # Test that key functions exist
    assert hasattr(run_chapter7_tabular_lgb, "compute_baseline_summary")
    assert hasattr(run_chapter7_tabular_lgb, "compute_data_hash")
    assert hasattr(run_chapter7_tabular_lgb, "write_baseline_reference")
    assert hasattr(run_chapter7_tabular_lgb, "load_frozen_baseline_floor")
    assert hasattr(run_chapter7_tabular_lgb, "get_git_commit_hash")
    assert hasattr(run_chapter7_tabular_lgb, "main")
    
    # Test that script has proper docstring
    assert run_chapter7_tabular_lgb.__doc__ is not None
    assert "Chapter 7.5" in run_chapter7_tabular_lgb.__doc__


def test_compute_baseline_summary():
    """Test that baseline summary computation works."""
    import tempfile
    import json
    
    # Create a temporary directory with mock outputs
    # Note: compute_baseline_summary expects experiment dir at: output_dir/baseline_tabular_lgb_{cadence}/
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create the expected nested directory structure
        experiment_dir = temp_path / "baseline_tabular_lgb_monthly"
        experiment_dir.mkdir(parents=True)
        
        # Create mock fold_summaries.csv
        fold_summaries = pd.DataFrame({
            "fold_id": ["fold_01", "fold_01", "fold_01"],
            "horizon": [20, 60, 90],
            "rankic_median": [0.05, 0.06, 0.04],
            "rankic_iqr": [0.1, 0.12, 0.08],
        })
        fold_summaries.to_csv(experiment_dir / "fold_summaries.csv", index=False)
        
        # Create mock cost_overlays.csv
        cost_overlays = pd.DataFrame({
            "scenario": ["base_slippage"] * 3,
            "horizon": [20, 60, 90],
            "alpha_survives": [True, False, True],
            "net_avg_er": [0.01, -0.005, 0.008],  # Updated column name
        })
        cost_overlays.to_csv(experiment_dir / "cost_overlays.csv", index=False)
        
        # Create mock churn_series.csv
        churn_series = pd.DataFrame({
            "horizon": [20, 20, 60, 60, 90, 90],
            "k": [10, 10, 10, 10, 10, 10],
            "churn": [0.2, 0.3, 0.25, 0.28, 0.15, 0.18],
        })
        churn_series.to_csv(experiment_dir / "churn_series.csv", index=False)
        
        # Import and test the function
        sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
        from run_chapter7_tabular_lgb import compute_baseline_summary
        
        summary = compute_baseline_summary(temp_path, "monthly")
        
        # Validate structure
        assert summary["cadence"] == "monthly"
        assert summary["n_folds"] == 1
        assert "horizons" in summary
        assert 20 in summary["horizons"]
        assert 60 in summary["horizons"]
        assert 90 in summary["horizons"]
        
        # Validate metrics
        assert summary["horizons"][20]["median_rankic"] == 0.05
        assert summary["horizons"][20]["churn_top10_median"] == 0.25  # median of [0.2, 0.3]


def test_data_hash_deterministic():
    """Test that data hash is deterministic."""
    sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
    from run_chapter7_tabular_lgb import compute_data_hash
    
    # Create test dataframe
    df = pd.DataFrame({
        "date": [date(2023, 1, 1), date(2023, 1, 2)],
        "ticker": ["A", "B"],
        "value": [1.0, 2.0],
    })
    
    # Compute hash twice
    hash1 = compute_data_hash(df)
    hash2 = compute_data_hash(df)
    
    assert hash1 == hash2, "Hash should be deterministic"
    assert len(hash1) == 16, "Hash should be 16 characters"
    
    # Different data should give different hash
    df2 = pd.DataFrame({
        "date": [date(2023, 1, 1), date(2023, 1, 3)],  # Different date
        "ticker": ["A", "B"],
        "value": [1.0, 2.0],
    })
    
    hash3 = compute_data_hash(df2)
    assert hash3 != hash1, "Different data should give different hash"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


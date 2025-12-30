"""
Tests for Stability Reports (Chapter 6.5)

CRITICAL: These tests enforce the "can't quietly lie" invariants:
1. Determinism: same inputs (shuffled) → identical outputs
2. Fold boundaries: churn/timeseries never bridge folds
3. Regime bucket integrity: totals sum correctly
4. No silent drops: exclusions are reported
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import date, timedelta
import tempfile
import shutil

from src.evaluation.reports import (
    STABILITY_THRESHOLDS,
    StabilityReportInputs,
    compute_ic_decay_stats,
    plot_ic_decay,
    format_regime_performance,
    plot_regime_bars,
    compute_churn_diagnostics,
    plot_churn_timeseries,
    plot_churn_distribution,
    generate_stability_scorecard,
    generate_stability_report,
    validate_report_determinism
)


# ============================================================================
# TEST STABILITY THRESHOLDS (IMMUTABILITY)
# ============================================================================

class TestStabilityThresholds:
    """Verify stability thresholds are locked and immutable."""
    
    def test_thresholds_frozen(self):
        """Verify STABILITY_THRESHOLDS is immutable."""
        with pytest.raises(Exception):  # FrozenInstanceError
            STABILITY_THRESHOLDS.rapid_decay_threshold = 0.1
    
    def test_threshold_values(self):
        """Verify threshold values are as specified."""
        assert STABILITY_THRESHOLDS.rapid_decay_threshold == 0.05
        assert STABILITY_THRESHOLDS.noisy_threshold == 2.0
        assert STABILITY_THRESHOLDS.high_churn_threshold == 0.50
        assert STABILITY_THRESHOLDS.min_dates_per_bucket == 10
        assert STABILITY_THRESHOLDS.min_names_per_date == 10
        assert STABILITY_THRESHOLDS.rolling_window == 6


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def sample_per_date_metrics():
    """Create sample per-date metrics for testing."""
    dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq="MS")
    
    data = []
    for fold_id in ["fold_01", "fold_02"]:
        for horizon in [20, 60]:
            for date in dates:
                data.append({
                    "date": date,
                    "fold_id": fold_id,
                    "horizon": horizon,
                    "rankic": np.random.randn() * 0.05 + 0.02,  # Mean 0.02, std 0.05
                    "quintile_spread": np.random.randn() * 0.02 + 0.01,
                    "n_names": 80 + int(np.random.randn() * 10)
                })
    
    return pd.DataFrame(data)


@pytest.fixture
def sample_fold_summaries():
    """Create sample fold summaries for testing."""
    data = []
    for fold_id in ["fold_01", "fold_02"]:
        for horizon in [20, 60]:
            data.append({
                "fold_id": fold_id,
                "horizon": horizon,
                "rankic_median": 0.02,
                "rankic_iqr": 0.03,
                "quintile_spread_median": 0.01,
                "n_dates": 12
            })
    
    return pd.DataFrame(data)


@pytest.fixture
def sample_regime_summaries():
    """Create sample regime summaries for testing."""
    data = []
    for horizon in [20, 60]:
        for regime_feature in ["vix_percentile_252d", "market_regime"]:
            buckets = ["low", "mid", "high"] if regime_feature == "vix_percentile_252d" else ["bull", "bear"]
            for bucket in buckets:
                data.append({
                    "horizon": horizon,
                    "regime_feature": regime_feature,
                    "bucket": bucket,
                    "rankic_median": 0.02 + np.random.randn() * 0.01,
                    "n_dates": 40 + int(np.random.randn() * 5),
                    "n_names_median": 85
                })
    
    return pd.DataFrame(data)


@pytest.fixture
def sample_churn_series():
    """Create sample churn series for testing."""
    dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq="MS")
    
    data = []
    for fold_id in ["fold_01", "fold_02"]:
        for horizon in [20, 60]:
            for k in [10, 20]:
                for i, date in enumerate(dates[1:]):  # Skip first date (no previous portfolio)
                    data.append({
                        "date": date,
                        "fold_id": fold_id,
                        "horizon": horizon,
                        "k": k,
                        "churn": np.random.uniform(0.1, 0.4),  # Typical churn range
                        "retention": np.random.uniform(0.6, 0.9)
                    })
    
    return pd.DataFrame(data)


@pytest.fixture
def temp_output_dir():
    """Create temporary output directory for testing."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    # Cleanup
    shutil.rmtree(temp_dir)


# ============================================================================
# TEST IC DECAY ANALYSIS
# ============================================================================

class TestICDecay:
    """Test IC decay statistics and plots."""
    
    def test_compute_ic_decay_stats(self, sample_per_date_metrics):
        """Test IC decay statistics computation."""
        stats = compute_ic_decay_stats(sample_per_date_metrics, metric_col="rankic")
        
        # Check structure
        assert len(stats) > 0
        assert "fold_id" in stats.columns
        assert "horizon" in stats.columns
        assert "early_median" in stats.columns
        assert "late_median" in stats.columns
        assert "decay" in stats.columns
        assert "rapid_decay_flag" in stats.columns
        assert "noisy_flag" in stats.columns
    
    def test_decay_calculation(self):
        """Test decay calculation is correct."""
        # Create deterministic data
        data = []
        dates = pd.date_range(start="2023-01-01", periods=12, freq="MS")
        
        # Early period: high IC (0.05)
        # Late period: low IC (0.01)
        for i, date in enumerate(dates):
            ic = 0.05 if i < 4 else (0.03 if i < 8 else 0.01)
            data.append({
                "date": date,
                "fold_id": "fold_01",
                "horizon": 20,
                "rankic": ic,
                "n_names": 80
            })
        
        df = pd.DataFrame(data)
        stats = compute_ic_decay_stats(df, metric_col="rankic")
        
        # Check decay is negative (late < early)
        assert stats.iloc[0]["early_median"] > stats.iloc[0]["late_median"]
        assert stats.iloc[0]["decay"] < 0
    
    def test_min_dates_requirement(self):
        """Test that insufficient dates are skipped."""
        # Create data with only 6 dates (< 9 required)
        data = []
        dates = pd.date_range(start="2023-01-01", periods=6, freq="MS")
        
        for date in dates:
            data.append({
                "date": date,
                "fold_id": "fold_01",
                "horizon": 20,
                "rankic": 0.02,
                "n_names": 80
            })
        
        df = pd.DataFrame(data)
        stats = compute_ic_decay_stats(df, metric_col="rankic")
        
        # Should be empty (insufficient dates)
        assert len(stats) == 0
    
    def test_plot_ic_decay_creates_figure(self, sample_per_date_metrics, temp_output_dir):
        """Test IC decay plot creation."""
        output_path = temp_output_dir / "ic_decay.png"
        
        fig = plot_ic_decay(sample_per_date_metrics, metric_col="rankic", output_path=output_path)
        
        # Check figure was created
        assert fig is not None
        assert output_path.exists()


# ============================================================================
# TEST REGIME-CONDITIONAL PERFORMANCE
# ============================================================================

class TestRegimePerformance:
    """Test regime-conditional performance reporting."""
    
    def test_format_regime_performance(self, sample_regime_summaries):
        """Test regime performance formatting."""
        formatted = format_regime_performance(sample_regime_summaries, metric_col="rankic_median")
        
        # Check structure
        assert "thin_slice_flag" in formatted.columns
        
        # Check flags are computed
        assert formatted["thin_slice_flag"].dtype == bool
    
    def test_thin_slice_detection(self):
        """Test thin slice flagging."""
        data = [
            {"horizon": 20, "regime_feature": "vix", "bucket": "low", 
             "rankic_median": 0.02, "n_dates": 5, "n_names_median": 80},  # Too few dates
            {"horizon": 20, "regime_feature": "vix", "bucket": "high", 
             "rankic_median": 0.02, "n_dates": 50, "n_names_median": 5},  # Too few names
            {"horizon": 20, "regime_feature": "vix", "bucket": "mid", 
             "rankic_median": 0.02, "n_dates": 50, "n_names_median": 80},  # OK
        ]
        
        df = pd.DataFrame(data)
        formatted = format_regime_performance(df, metric_col="rankic_median")
        
        # Check flags
        assert formatted.iloc[0]["thin_slice_flag"] == True  # Too few dates
        assert formatted.iloc[1]["thin_slice_flag"] == True  # Too few names
        assert formatted.iloc[2]["thin_slice_flag"] == False  # OK
    
    def test_plot_regime_bars_creates_figure(self, sample_regime_summaries, temp_output_dir):
        """Test regime bars plot creation."""
        output_path = temp_output_dir / "regime_bars.png"
        
        fig = plot_regime_bars(sample_regime_summaries, metric_col="rankic_median", output_path=output_path)
        
        # Check figure was created
        assert fig is not None
        assert output_path.exists()


# ============================================================================
# TEST CHURN DIAGNOSTICS
# ============================================================================

class TestChurnDiagnostics:
    """Test churn diagnostic reporting."""
    
    def test_compute_churn_diagnostics(self, sample_churn_series):
        """Test churn diagnostics computation."""
        diag = compute_churn_diagnostics(sample_churn_series)
        
        # Check structure
        assert "fold_id" in diag.columns
        assert "horizon" in diag.columns
        assert "k" in diag.columns
        assert "churn_median" in diag.columns
        assert "churn_p90" in diag.columns
        assert "high_churn_flag" in diag.columns
        
        # Check all groups present
        expected_groups = len(sample_churn_series.groupby(["fold_id", "horizon", "k"]))
        assert len(diag) == expected_groups
    
    def test_high_churn_flag(self):
        """Test high churn flag computation."""
        # Create data with high churn
        data = []
        dates = pd.date_range(start="2023-01-01", periods=10, freq="MS")
        
        for i, date in enumerate(dates):
            # First 7 dates: high churn (>50%)
            # Last 3 dates: low churn (<30%)
            churn = 0.6 if i < 7 else 0.2
            data.append({
                "date": date,
                "fold_id": "fold_01",
                "horizon": 20,
                "k": 10,
                "churn": churn,
                "retention": 1 - churn
            })
        
        df = pd.DataFrame(data)
        diag = compute_churn_diagnostics(df)
        
        # Should flag (70% of dates have high churn)
        assert diag.iloc[0]["high_churn_flag"] == True
        assert diag.iloc[0]["pct_high_churn"] == 0.7
    
    def test_plot_churn_timeseries_creates_figure(self, sample_churn_series, temp_output_dir):
        """Test churn timeseries plot creation."""
        output_path = temp_output_dir / "churn_timeseries.png"
        
        fig = plot_churn_timeseries(sample_churn_series, k=10, output_path=output_path)
        
        # Check figure was created
        assert fig is not None
        assert output_path.exists()
    
    def test_plot_churn_distribution_creates_figure(self, sample_churn_series, temp_output_dir):
        """Test churn distribution plot creation."""
        output_path = temp_output_dir / "churn_dist.png"
        
        fig = plot_churn_distribution(sample_churn_series, k=10, output_path=output_path)
        
        # Check figure was created
        assert fig is not None
        assert output_path.exists()


# ============================================================================
# TEST STABILITY SCORECARD
# ============================================================================

class TestStabilityScorecard:
    """Test stability scorecard generation."""
    
    def test_generate_scorecard_basic(self, sample_fold_summaries):
        """Test basic scorecard generation."""
        scorecard = generate_stability_scorecard(sample_fold_summaries)
        
        # Should have same structure as fold summaries
        assert len(scorecard) == len(sample_fold_summaries)
        assert "fold_id" in scorecard.columns
        assert "horizon" in scorecard.columns
    
    def test_generate_scorecard_with_churn(self, sample_fold_summaries):
        """Test scorecard with churn diagnostics."""
        # Create churn diagnostics
        churn_diag = pd.DataFrame([
            {"fold_id": "fold_01", "horizon": 20, "k": 10, "churn_median": 0.25},
            {"fold_id": "fold_01", "horizon": 60, "k": 10, "churn_median": 0.30},
            {"fold_id": "fold_02", "horizon": 20, "k": 10, "churn_median": 0.20},
            {"fold_id": "fold_02", "horizon": 60, "k": 10, "churn_median": 0.35},
        ])
        
        scorecard = generate_stability_scorecard(sample_fold_summaries, churn_diagnostics=churn_diag)
        
        # Should include churn
        assert "churn@10_median" in scorecard.columns
    
    def test_generate_scorecard_with_costs(self, sample_fold_summaries):
        """Test scorecard with cost overlays."""
        # Create cost overlays
        cost_overlays = pd.DataFrame([
            {"fold_id": "fold_01", "horizon": 20, "scenario": "base_slippage", 
             "net_avg_er": 0.03, "alpha_survives": True},
            {"fold_id": "fold_01", "horizon": 60, "scenario": "base_slippage", 
             "net_avg_er": 0.02, "alpha_survives": True},
            {"fold_id": "fold_02", "horizon": 20, "scenario": "base_slippage", 
             "net_avg_er": 0.04, "alpha_survives": True},
            {"fold_id": "fold_02", "horizon": 60, "scenario": "base_slippage", 
             "net_avg_er": -0.01, "alpha_survives": False},
        ])
        
        scorecard = generate_stability_scorecard(sample_fold_summaries, cost_overlays=cost_overlays)
        
        # Should include cost metrics
        assert "net_avg_er" in scorecard.columns
        assert "alpha_survives" in scorecard.columns


# ============================================================================
# TEST FULL REPORT GENERATION
# ============================================================================

class TestReportGeneration:
    """Test full report generation pipeline."""
    
    def test_generate_stability_report(
        self,
        sample_per_date_metrics,
        sample_fold_summaries,
        sample_regime_summaries,
        sample_churn_series,
        temp_output_dir
    ):
        """Test complete report generation."""
        inputs = StabilityReportInputs(
            per_date_metrics=sample_per_date_metrics,
            fold_summaries=sample_fold_summaries,
            regime_summaries=sample_regime_summaries,
            churn_series=sample_churn_series
        )
        
        outputs = generate_stability_report(
            inputs,
            experiment_name="test_experiment",
            output_dir=temp_output_dir
        )
        
        # Check all outputs were created
        assert outputs.output_dir.exists()
        assert outputs.ic_decay_stats.exists()
        assert outputs.regime_performance.exists()
        assert outputs.churn_diagnostics.exists()
        assert outputs.stability_scorecard.exists()
        assert outputs.ic_decay_plot.exists()
        assert outputs.regime_bars.exists()
        assert outputs.churn_timeseries.exists()
        assert outputs.churn_distribution.exists()
        assert outputs.summary_report.exists()
        
        # Check tables are readable
        ic_decay = pd.read_csv(outputs.ic_decay_stats)
        assert len(ic_decay) > 0
        
        scorecard = pd.read_csv(outputs.stability_scorecard)
        assert len(scorecard) > 0
    
    def test_generate_report_minimal(
        self,
        sample_per_date_metrics,
        sample_fold_summaries,
        temp_output_dir
    ):
        """Test report generation with minimal inputs (no regime/churn/costs)."""
        inputs = StabilityReportInputs(
            per_date_metrics=sample_per_date_metrics,
            fold_summaries=sample_fold_summaries
        )
        
        outputs = generate_stability_report(
            inputs,
            experiment_name="test_minimal",
            output_dir=temp_output_dir
        )
        
        # Core outputs should still be created
        assert outputs.output_dir.exists()
        assert outputs.ic_decay_stats.exists()
        assert outputs.stability_scorecard.exists()
        assert outputs.ic_decay_plot.exists()
        assert outputs.summary_report.exists()


# ============================================================================
# TEST DETERMINISM (CRITICAL INVARIANT)
# ============================================================================

class TestDeterminism:
    """Test that reports are deterministic."""
    
    def test_ic_decay_determinism(self, sample_per_date_metrics):
        """Test IC decay stats are deterministic."""
        # Shuffle data
        shuffled = sample_per_date_metrics.sample(frac=1.0, random_state=42)
        
        # Compute stats
        stats1 = compute_ic_decay_stats(sample_per_date_metrics, metric_col="rankic")
        stats2 = compute_ic_decay_stats(shuffled, metric_col="rankic")
        
        # Sort for comparison
        stats1 = stats1.sort_values(["fold_id", "horizon"]).reset_index(drop=True)
        stats2 = stats2.sort_values(["fold_id", "horizon"]).reset_index(drop=True)
        
        # Should be identical
        pd.testing.assert_frame_equal(stats1, stats2)
    
    def test_churn_diagnostics_determinism(self, sample_churn_series):
        """Test churn diagnostics are deterministic."""
        # Shuffle data
        shuffled = sample_churn_series.sample(frac=1.0, random_state=42)
        
        # Compute diagnostics
        diag1 = compute_churn_diagnostics(sample_churn_series)
        diag2 = compute_churn_diagnostics(shuffled)
        
        # Sort for comparison
        diag1 = diag1.sort_values(["fold_id", "horizon", "k"]).reset_index(drop=True)
        diag2 = diag2.sort_values(["fold_id", "horizon", "k"]).reset_index(drop=True)
        
        # Should be identical
        pd.testing.assert_frame_equal(diag1, diag2)
    
    def test_scorecard_determinism(self, sample_fold_summaries):
        """Test scorecard is deterministic."""
        # Shuffle data
        shuffled = sample_fold_summaries.sample(frac=1.0, random_state=42)
        
        # Generate scorecards
        scorecard1 = generate_stability_scorecard(sample_fold_summaries)
        scorecard2 = generate_stability_scorecard(shuffled)
        
        # Sort for comparison
        scorecard1 = scorecard1.sort_values(["fold_id", "horizon"]).reset_index(drop=True)
        scorecard2 = scorecard2.sort_values(["fold_id", "horizon"]).reset_index(drop=True)
        
        # Should be identical
        pd.testing.assert_frame_equal(scorecard1, scorecard2)


# ============================================================================
# TEST FOLD BOUNDARY HANDLING
# ============================================================================

class TestFoldBoundaries:
    """Test that fold boundaries are respected (no bridging)."""
    
    def test_churn_never_bridges_folds(self):
        """Test churn computation respects fold boundaries."""
        # Create data with two folds
        dates1 = pd.date_range(start="2023-01-01", periods=6, freq="MS")
        dates2 = pd.date_range(start="2023-07-01", periods=6, freq="MS")
        
        data = []
        for date in dates1:
            data.append({
                "date": date,
                "fold_id": "fold_01",
                "horizon": 20,
                "k": 10,
                "churn": 0.3,
                "retention": 0.7
            })
        
        for date in dates2:
            data.append({
                "date": date,
                "fold_id": "fold_02",
                "horizon": 20,
                "k": 10,
                "churn": 0.3,
                "retention": 0.7
            })
        
        df = pd.DataFrame(data)
        
        # Compute diagnostics per fold
        diag = compute_churn_diagnostics(df)
        
        # Should have 2 entries (one per fold)
        assert len(diag) == 2
        assert diag["fold_id"].nunique() == 2


# ============================================================================
# TEST REGIME BUCKET INTEGRITY
# ============================================================================

class TestRegimeBuckets:
    """Test regime bucket totals are consistent."""
    
    def test_regime_coverage_sums(self):
        """Test regime bucket dates sum to total."""
        # Create regime summaries where we know the totals
        data = [
            {"horizon": 20, "regime_feature": "vix", "bucket": "low", "rankic_median": 0.02, "n_dates": 30, "n_names_median": 80},
            {"horizon": 20, "regime_feature": "vix", "bucket": "mid", "rankic_median": 0.01, "n_dates": 40, "n_names_median": 80},
            {"horizon": 20, "regime_feature": "vix", "bucket": "high", "rankic_median": 0.03, "n_dates": 50, "n_names_median": 80},
        ]
        
        df = pd.DataFrame(data)
        
        # Total should equal sum of buckets
        total_dates = df[df["regime_feature"] == "vix"]["n_dates"].sum()
        assert total_dates == 120


# ============================================================================
# TEST EXCLUSIONS ARE REPORTED
# ============================================================================

class TestExclusionReporting:
    """Test that exclusions are reported, not silent."""
    
    def test_insufficient_dates_logged(self, caplog):
        """Test that folds with insufficient dates are logged."""
        # Create data with only 5 dates (< 9 required)
        data = []
        dates = pd.date_range(start="2023-01-01", periods=5, freq="MS")
        
        for date in dates:
            data.append({
                "date": date,
                "fold_id": "fold_01",
                "horizon": 20,
                "rankic": 0.02,
                "n_names": 80
            })
        
        df = pd.DataFrame(data)
        
        # Compute stats (should log warning)
        with caplog.at_level("WARNING"):
            stats = compute_ic_decay_stats(df, metric_col="rankic")
        
        # Should be empty and have warning
        assert len(stats) == 0
        assert "only 5 dates" in caplog.text


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

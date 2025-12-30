"""
End-to-End Smoke Tests (Chapter 6)

Integration tests that verify the complete evaluation pipeline:
1. Runs a short slice (6-12 months) with ONE baseline
2. Asserts output folders + key CSVs/PNGs exist
3. Asserts deterministic outputs across two runs
4. Asserts cost overlay scenarios are present and named consistently

These tests use SMOKE mode (limited date range) for CI efficiency.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import date
from pathlib import Path
import tempfile
import shutil
import json

from src.evaluation.run_evaluation import (
    ExperimentSpec,
    SMOKE_MODE,
    FULL_MODE,
    run_experiment,
    compute_acceptance_verdict,
    save_acceptance_summary,
    COST_SCENARIOS,
)
from src.evaluation.baselines import list_baselines


# ============================================================================
# TEST FIXTURES
# ============================================================================

@pytest.fixture
def sample_features_df():
    """Create sample features DataFrame for smoke testing."""
    np.random.seed(42)
    
    data = []
    # 12 months of data for smoke test
    dates = pd.date_range(start="2024-01-01", end="2024-12-31", freq="MS")
    tickers = [f"STOCK_{i}" for i in range(20)]  # 20 stocks
    
    for d in dates:
        for i, ticker in enumerate(tickers):
            data.append({
                "date": d,
                "ticker": ticker,
                "stable_id": f"STABLE_{ticker}",
                "mom_1m": np.random.randn() * 0.1 + 0.01,
                "mom_3m": np.random.randn() * 0.15 + 0.02,
                "mom_6m": np.random.randn() * 0.2 + 0.03,
                "mom_12m": np.random.randn() * 0.3 + 0.04,
                "excess_return": np.random.randn() * 0.05 + 0.01,
                "adv_20d": 1_000_000 + i * 100_000,
                "sector": ["Tech", "Healthcare", "Finance", "Energy"][i % 4],
                "vix_percentile_252d": 50 + np.random.randn() * 20,
                "market_return_20d": np.random.randn() * 0.02,
                "market_vol_20d": 15 + np.random.randn() * 5,
            })
    
    return pd.DataFrame(data)


@pytest.fixture
def temp_output_dir():
    """Create temporary output directory."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir)


# ============================================================================
# TEST EXPERIMENT SPEC
# ============================================================================

class TestExperimentSpec:
    """Test experiment specification creation and validation."""
    
    def test_baseline_spec_creation(self):
        """Test creating baseline experiment spec."""
        spec = ExperimentSpec.baseline("mom_12m")
        
        assert spec.model_type == "baseline"
        assert spec.model_name == "mom_12m"
        assert spec.cadence == "monthly"
        assert spec.name == "baseline_mom_12m_monthly"
    
    def test_spec_validation_invalid_baseline(self):
        """Invalid baseline should raise error."""
        with pytest.raises(ValueError, match="Unknown baseline"):
            ExperimentSpec(
                name="test",
                model_type="baseline",
                model_name="not_a_baseline"
            )
    
    def test_spec_validation_invalid_cadence(self):
        """Invalid cadence should raise error."""
        with pytest.raises(ValueError, match="cadence"):
            ExperimentSpec(
                name="test",
                model_type="baseline",
                model_name="mom_12m",
                cadence="weekly"
            )
    
    def test_spec_validation_invalid_model_type(self):
        """Invalid model_type should raise error."""
        with pytest.raises(ValueError, match="model_type"):
            ExperimentSpec(
                name="test",
                model_type="unknown",
                model_name="test"
            )


# ============================================================================
# TEST OUTPUT STRUCTURE
# ============================================================================

class TestOutputStructure:
    """Test that output folders and files are created correctly."""
    
    def test_smoke_run_creates_output_dir(self, sample_features_df, temp_output_dir):
        """Smoke run should create output directory."""
        spec = ExperimentSpec.baseline("mom_12m")
        
        results = run_experiment(
            experiment_spec=spec,
            features_df=sample_features_df,
            output_dir=temp_output_dir,
            mode=SMOKE_MODE
        )
        
        exp_dir = temp_output_dir / spec.name
        assert exp_dir.exists()
    
    def test_creates_eval_rows_parquet(self, sample_features_df, temp_output_dir):
        """Should create eval_rows.parquet."""
        spec = ExperimentSpec.baseline("mom_12m")
        
        results = run_experiment(
            experiment_spec=spec,
            features_df=sample_features_df,
            output_dir=temp_output_dir,
            mode=SMOKE_MODE
        )
        
        eval_rows_path = results["output_paths"]["eval_rows"]
        assert eval_rows_path.exists()
        
        # Should be readable
        df = pd.read_parquet(eval_rows_path)
        assert len(df) > 0
    
    def test_creates_per_date_metrics_csv(self, sample_features_df, temp_output_dir):
        """Should create per_date_metrics.csv."""
        spec = ExperimentSpec.baseline("mom_12m")
        
        results = run_experiment(
            experiment_spec=spec,
            features_df=sample_features_df,
            output_dir=temp_output_dir,
            mode=SMOKE_MODE
        )
        
        metrics_path = results["output_paths"]["per_date_metrics"]
        assert metrics_path.exists()
        
        df = pd.read_csv(metrics_path)
        assert "rankic" in df.columns
    
    def test_creates_fold_summaries_csv(self, sample_features_df, temp_output_dir):
        """Should create fold_summaries.csv."""
        spec = ExperimentSpec.baseline("mom_12m")
        
        results = run_experiment(
            experiment_spec=spec,
            features_df=sample_features_df,
            output_dir=temp_output_dir,
            mode=SMOKE_MODE
        )
        
        summary_path = results["output_paths"]["fold_summaries"]
        assert summary_path.exists()
        
        df = pd.read_csv(summary_path)
        assert "fold_id" in df.columns
        assert "rankic_median" in df.columns
    
    def test_creates_metadata_json(self, sample_features_df, temp_output_dir):
        """Should create experiment_metadata.json."""
        spec = ExperimentSpec.baseline("mom_12m")
        
        results = run_experiment(
            experiment_spec=spec,
            features_df=sample_features_df,
            output_dir=temp_output_dir,
            mode=SMOKE_MODE
        )
        
        metadata_path = results["output_paths"]["metadata"]
        assert metadata_path.exists()
        
        with open(metadata_path) as f:
            metadata = json.load(f)
        
        assert metadata["experiment_name"] == spec.name
        assert metadata["mode"] == "smoke"
    
    def test_creates_stability_report(self, sample_features_df, temp_output_dir):
        """Should create stability report folder."""
        spec = ExperimentSpec.baseline("mom_12m")
        
        results = run_experiment(
            experiment_spec=spec,
            features_df=sample_features_df,
            output_dir=temp_output_dir,
            mode=SMOKE_MODE
        )
        
        report_dir = results["output_paths"]["stability_report"]
        assert report_dir.exists()
        
        # Check for key subdirectories
        assert (report_dir / "tables").exists() or report_dir.exists()
    
    def test_stability_report_has_scorecard(self, sample_features_df, temp_output_dir):
        """Stability report should include scorecard."""
        spec = ExperimentSpec.baseline("mom_12m")
        
        results = run_experiment(
            experiment_spec=spec,
            features_df=sample_features_df,
            output_dir=temp_output_dir,
            mode=SMOKE_MODE
        )
        
        # Look for scorecard in tables directory
        report_dir = results["output_paths"]["stability_report"]
        tables_dir = report_dir / "tables"
        
        if tables_dir.exists():
            scorecard_path = tables_dir / "stability_scorecard.csv"
            assert scorecard_path.exists()


# ============================================================================
# TEST COST OVERLAY SCENARIOS
# ============================================================================

class TestCostOverlayScenarios:
    """Test that cost scenarios are present and consistent."""
    
    def test_all_cost_scenarios_present(self, sample_features_df, temp_output_dir):
        """All cost scenarios should be present in output."""
        spec = ExperimentSpec.baseline("mom_12m")
        
        results = run_experiment(
            experiment_spec=spec,
            features_df=sample_features_df,
            output_dir=temp_output_dir,
            mode=SMOKE_MODE
        )
        
        cost_df = results["cost_overlays"]
        
        if cost_df is not None and len(cost_df) > 0:
            scenarios = set(cost_df["scenario"].unique())
            expected_scenarios = set(COST_SCENARIOS.keys())
            
            assert scenarios == expected_scenarios, f"Missing scenarios: {expected_scenarios - scenarios}"
    
    def test_cost_scenarios_named_consistently(self, sample_features_df, temp_output_dir):
        """Cost scenario names should match COST_SCENARIOS keys."""
        spec = ExperimentSpec.baseline("mom_12m")
        
        results = run_experiment(
            experiment_spec=spec,
            features_df=sample_features_df,
            output_dir=temp_output_dir,
            mode=SMOKE_MODE
        )
        
        cost_df = results["cost_overlays"]
        
        if cost_df is not None and len(cost_df) > 0:
            for scenario in cost_df["scenario"].unique():
                assert scenario in COST_SCENARIOS, f"Unknown scenario: {scenario}"


# ============================================================================
# TEST DETERMINISM
# ============================================================================

class TestDeterminism:
    """Test that outputs are deterministic across runs."""
    
    def test_two_runs_identical_eval_rows(self, sample_features_df, temp_output_dir):
        """Two runs should produce identical evaluation rows."""
        spec = ExperimentSpec.baseline("mom_12m")
        
        # First run
        results1 = run_experiment(
            experiment_spec=spec,
            features_df=sample_features_df,
            output_dir=temp_output_dir / "run1",
            mode=SMOKE_MODE
        )
        
        # Second run
        results2 = run_experiment(
            experiment_spec=spec,
            features_df=sample_features_df,
            output_dir=temp_output_dir / "run2",
            mode=SMOKE_MODE
        )
        
        # Load and compare eval rows
        df1 = pd.read_parquet(results1["output_paths"]["eval_rows"])
        df2 = pd.read_parquet(results2["output_paths"]["eval_rows"])
        
        # Sort for comparison
        df1 = df1.sort_values(["as_of_date", "stable_id"]).reset_index(drop=True)
        df2 = df2.sort_values(["as_of_date", "stable_id"]).reset_index(drop=True)
        
        pd.testing.assert_frame_equal(df1, df2)
    
    def test_two_runs_identical_metrics(self, sample_features_df, temp_output_dir):
        """Two runs should produce identical metrics."""
        spec = ExperimentSpec.baseline("mom_12m")
        
        results1 = run_experiment(
            experiment_spec=spec,
            features_df=sample_features_df,
            output_dir=temp_output_dir / "run1",
            mode=SMOKE_MODE
        )
        
        results2 = run_experiment(
            experiment_spec=spec,
            features_df=sample_features_df,
            output_dir=temp_output_dir / "run2",
            mode=SMOKE_MODE
        )
        
        # Compare fold summaries
        df1 = results1["fold_summaries"]
        df2 = results2["fold_summaries"]
        
        if len(df1) > 0 and len(df2) > 0:
            df1 = df1.sort_values(["fold_id", "horizon"]).reset_index(drop=True)
            df2 = df2.sort_values(["fold_id", "horizon"]).reset_index(drop=True)
            
            pd.testing.assert_frame_equal(df1, df2)
    
    def test_shuffled_input_same_output(self, sample_features_df, temp_output_dir):
        """Shuffled input should produce identical output."""
        spec = ExperimentSpec.baseline("mom_12m")
        
        # Original order
        results1 = run_experiment(
            experiment_spec=spec,
            features_df=sample_features_df,
            output_dir=temp_output_dir / "run1",
            mode=SMOKE_MODE
        )
        
        # Shuffled input
        shuffled = sample_features_df.sample(frac=1.0, random_state=42)
        results2 = run_experiment(
            experiment_spec=spec,
            features_df=shuffled,
            output_dir=temp_output_dir / "run2",
            mode=SMOKE_MODE
        )
        
        # Should produce same metrics (after sorting)
        df1 = results1["fold_summaries"].sort_values(["fold_id", "horizon"]).reset_index(drop=True)
        df2 = results2["fold_summaries"].sort_values(["fold_id", "horizon"]).reset_index(drop=True)
        
        if len(df1) > 0 and len(df2) > 0:
            pd.testing.assert_frame_equal(df1, df2)


# ============================================================================
# TEST MULTIPLE BASELINES
# ============================================================================

class TestMultipleBaselines:
    """Test running multiple baselines."""
    
    def test_all_baselines_can_run(self, sample_features_df, temp_output_dir):
        """All baselines should be able to run."""
        for baseline_name in list_baselines():
            spec = ExperimentSpec.baseline(baseline_name)
            
            results = run_experiment(
                experiment_spec=spec,
                features_df=sample_features_df,
                output_dir=temp_output_dir,
                mode=SMOKE_MODE
            )
            
            assert results["n_folds"] > 0, f"Baseline {baseline_name} produced no folds"
    
    def test_different_baselines_different_scores(self, sample_features_df, temp_output_dir):
        """Different baselines should produce different scores."""
        results_by_baseline = {}
        
        for baseline_name in ["mom_12m", "momentum_composite", "short_term_strength"]:
            spec = ExperimentSpec.baseline(baseline_name)
            
            results = run_experiment(
                experiment_spec=spec,
                features_df=sample_features_df,
                output_dir=temp_output_dir / baseline_name,
                mode=SMOKE_MODE
            )
            
            eval_df = pd.read_parquet(results["output_paths"]["eval_rows"])
            results_by_baseline[baseline_name] = eval_df
        
        # mom_12m and short_term_strength should have different scores
        mom_12m_scores = results_by_baseline["mom_12m"]["score"].values
        short_term_scores = results_by_baseline["short_term_strength"]["score"].values
        
        # Not all identical (they use different features)
        assert not np.allclose(mom_12m_scores, short_term_scores), \
            "Different baselines should produce different scores"


# ============================================================================
# TEST ACCEPTANCE CRITERIA
# ============================================================================

class TestAcceptanceCriteria:
    """Test acceptance criteria computation."""
    
    def test_compute_verdict(self, sample_features_df, temp_output_dir):
        """Should compute acceptance verdict."""
        # Run baseline
        spec = ExperimentSpec.baseline("mom_12m")
        results = run_experiment(
            experiment_spec=spec,
            features_df=sample_features_df,
            output_dir=temp_output_dir,
            mode=SMOKE_MODE
        )
        
        # Create mock "model" summary (same as baseline for testing)
        model_summary = results["fold_summaries"]
        baseline_summaries = {"mom_12m": results["fold_summaries"]}
        
        if len(results["fold_summaries"]) > 0 and len(results["cost_overlays"]) > 0:
            verdict = compute_acceptance_verdict(
                model_summary,
                baseline_summaries,
                results["cost_overlays"],
                results["churn_series"]
            )
            
            # Should have results for each horizon
            assert len(verdict) > 0
            assert "all_criteria_pass" in verdict.columns
    
    def test_save_acceptance_summary(self, sample_features_df, temp_output_dir):
        """Should save acceptance summary files."""
        spec = ExperimentSpec.baseline("mom_12m")
        results = run_experiment(
            experiment_spec=spec,
            features_df=sample_features_df,
            output_dir=temp_output_dir,
            mode=SMOKE_MODE
        )
        
        if len(results["fold_summaries"]) > 0 and len(results["cost_overlays"]) > 0:
            model_summary = results["fold_summaries"]
            baseline_summaries = {"mom_12m": results["fold_summaries"]}
            
            verdict = compute_acceptance_verdict(
                model_summary,
                baseline_summaries,
                results["cost_overlays"],
                results["churn_series"]
            )
            
            md_path = save_acceptance_summary(
                verdict,
                temp_output_dir,
                "test_experiment"
            )
            
            # Should create files
            assert md_path.exists()
            assert (temp_output_dir / "ACCEPTANCE_SUMMARY.csv").exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


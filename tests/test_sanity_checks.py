"""
Tests for Pre-Implementation Sanity Checks (Chapter 6.0.1)

These tests verify that the IC parity check and experiment naming convention work correctly.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import date

from src.evaluation.sanity_checks import (
    verify_ic_parity,
    validate_experiment_name,
    ExperimentNameBuilder,
    run_sanity_checks,
)


class TestICParityCheck:
    """Test Manual IC vs Qlib IC parity verification."""
    
    def test_ic_computation_basic(self):
        """Test basic IC computation without Qlib comparison."""
        # Create synthetic predictions and labels with positive correlation
        np.random.seed(42)
        n_dates = 10
        n_stocks = 50
        
        data = []
        for i in range(n_dates):
            date_val = date(2023, 1, 1 + i)
            for j in range(n_stocks):
                true_alpha = np.random.randn()
                data.append({
                    "date": date_val,
                    "ticker": f"STOCK_{j}",
                    "prediction": true_alpha + np.random.randn() * 0.5,
                    "label": true_alpha + np.random.randn() * 0.3,
                })
        
        df = pd.DataFrame(data)
        predictions = df[["date", "ticker", "prediction"]]
        labels = df[["date", "ticker", "label"]]
        
        result = verify_ic_parity(predictions, labels, qlib_ic=None)
        
        assert "manual_ic" in result
        assert "details" in result
        assert result["details"]["n_dates"] == n_dates
        assert result["details"]["n_observations"] == n_dates * n_stocks
        # With positive correlation, IC should be positive
        assert result["manual_ic"] > 0
    
    def test_ic_parity_passes_when_close(self):
        """Test that IC parity check passes when ICs are close."""
        # Create data with known IC
        predictions = pd.DataFrame({
            "date": [date(2023, 1, 1)] * 3,
            "ticker": ["A", "B", "C"],
            "prediction": [1.0, 2.0, 3.0]
        })
        labels = pd.DataFrame({
            "date": [date(2023, 1, 1)] * 3,
            "ticker": ["A", "B", "C"],
            "label": [1.0, 2.0, 3.0]  # Perfect correlation
        })
        
        # Manual IC should be 1.0 (perfect rank correlation)
        result = verify_ic_parity(predictions, labels, qlib_ic=1.0, tolerance=0.001)
        
        assert result["match"] == True
        assert result["diff"] <= 0.001
        assert abs(result["manual_ic"] - 1.0) < 0.01
    
    def test_ic_parity_fails_when_different(self):
        """Test that IC parity check fails when ICs differ."""
        predictions = pd.DataFrame({
            "date": [date(2023, 1, 1)] * 3,
            "ticker": ["A", "B", "C"],
            "prediction": [1.0, 2.0, 3.0]
        })
        labels = pd.DataFrame({
            "date": [date(2023, 1, 1)] * 3,
            "ticker": ["A", "B", "C"],
            "label": [1.0, 2.0, 3.0]
        })
        
        # Provide a very different Qlib IC
        with pytest.raises(ValueError, match="IC PARITY CHECK FAILED"):
            verify_ic_parity(predictions, labels, qlib_ic=0.5, tolerance=0.001)
    
    def test_ic_handles_multiple_dates(self):
        """Test IC computation across multiple dates."""
        predictions = pd.DataFrame({
            "date": [date(2023, 1, 1)] * 3 + [date(2023, 1, 2)] * 3,
            "ticker": ["A", "B", "C"] * 2,
            "prediction": [1.0, 2.0, 3.0, 3.0, 2.0, 1.0]
        })
        labels = pd.DataFrame({
            "date": [date(2023, 1, 1)] * 3 + [date(2023, 1, 2)] * 3,
            "ticker": ["A", "B", "C"] * 2,
            "label": [1.0, 2.0, 3.0, 3.0, 2.0, 1.0]
        })
        
        result = verify_ic_parity(predictions, labels, qlib_ic=None)
        
        assert result["details"]["n_dates"] == 2
        assert result["details"]["n_observations"] == 6
    
    def test_ic_rejects_empty_data(self):
        """Test that IC computation fails on empty data."""
        predictions = pd.DataFrame(columns=["date", "ticker", "prediction"])
        labels = pd.DataFrame(columns=["date", "ticker", "label"])
        
        with pytest.raises(ValueError, match="No matching predictions and labels"):
            verify_ic_parity(predictions, labels)


class TestExperimentNameBuilder:
    """Test experiment name builder and validation."""
    
    def test_build_valid_experiment_name(self):
        """Test building a valid experiment name."""
        builder = ExperimentNameBuilder(
            horizon=20,
            model="kronos_v0",
            label_version="v2",
            fold_id="fold_01"
        )
        
        name = builder.build()
        expected = "ai_forecaster/horizon=20/model=kronos_v0/labels=v2_totalreturn/fold=fold_01"
        assert name == expected
    
    def test_builder_normalizes_label_version(self):
        """Test that label version is normalized."""
        builder = ExperimentNameBuilder(
            horizon=60,
            model="baseline_mom12m",
            label_version="v1",  # Should become v1_priceonly
            fold_id="fold_02"
        )
        
        assert builder.label_version == "v1_priceonly"
        name = builder.build()
        assert "labels=v1_priceonly" in name
    
    def test_builder_rejects_invalid_horizon(self):
        """Test that invalid horizons are rejected."""
        with pytest.raises(ValueError, match="horizon must be one of"):
            ExperimentNameBuilder(
                horizon=30,  # Not in [20, 60, 90]
                model="test",
                label_version="v2",
                fold_id="fold_01"
            )
    
    def test_builder_rejects_invalid_model_name(self):
        """Test that invalid model names are rejected."""
        with pytest.raises(ValueError, match="model name must be lowercase"):
            ExperimentNameBuilder(
                horizon=20,
                model="Kronos-V0",  # Uppercase and hyphen not allowed
                label_version="v2",
                fold_id="fold_01"
            )
    
    def test_builder_to_dict(self):
        """Test exporting builder as dictionary."""
        builder = ExperimentNameBuilder(
            horizon=90,
            model="tabular_lgb",
            label_version="v2_totalreturn",
            fold_id="fold_03"
        )
        
        d = builder.to_dict()
        assert d["horizon"] == 90
        assert d["model"] == "tabular_lgb"
        assert d["label_version"] == "v2_totalreturn"
        assert d["fold_id"] == "fold_03"
        assert "experiment_name" in d


class TestExperimentNameValidation:
    """Test experiment name validation and parsing."""
    
    def test_validate_correct_name(self):
        """Test validation of correct experiment name."""
        name = "ai_forecaster/horizon=20/model=kronos_v0/labels=v2_totalreturn/fold=fold_01"
        components = validate_experiment_name(name)
        
        assert components["horizon"] == "20"
        assert components["model"] == "kronos_v0"
        assert components["labels"] == "v2_totalreturn"
        assert components["fold"] == "fold_01"
    
    def test_validate_rejects_wrong_format(self):
        """Test that wrong format is rejected."""
        bad_names = [
            "wrong/format",
            "ai_forecaster/horizon=20/model=test",  # Missing components
            "ai_forecaster/model=test/horizon=20/labels=v2/fold=01",  # Wrong order
            "wrong_prefix/horizon=20/model=test/labels=v2/fold=01",
        ]
        
        for name in bad_names:
            with pytest.raises(ValueError, match="does not match convention"):
                validate_experiment_name(name)
    
    def test_validate_rejects_invalid_horizon(self):
        """Test that invalid horizon values are rejected."""
        name = "ai_forecaster/horizon=45/model=test/labels=v2/fold=01"
        with pytest.raises(ValueError, match="Invalid horizon"):
            validate_experiment_name(name)
    
    def test_validate_accepts_various_model_names(self):
        """Test that various valid model names are accepted."""
        valid_models = [
            "kronos_v0",
            "fintext_v1",
            "baseline_mom12m",
            "tabular_lgb",
            "test123",
        ]
        
        for model in valid_models:
            name = f"ai_forecaster/horizon=20/model={model}/labels=v2/fold=01"
            components = validate_experiment_name(name)
            assert components["model"] == model


class TestRunSanityChecks:
    """Test the combined sanity check runner."""
    
    def test_run_all_checks_success(self):
        """Test running all sanity checks successfully."""
        # Create synthetic data
        predictions = pd.DataFrame({
            "date": [date(2023, 1, 1)] * 3,
            "ticker": ["A", "B", "C"],
            "prediction": [1.0, 2.0, 3.0]
        })
        labels = pd.DataFrame({
            "date": [date(2023, 1, 1)] * 3,
            "ticker": ["A", "B", "C"],
            "label": [1.0, 2.0, 3.0]
        })
        
        exp_name = "ai_forecaster/horizon=20/model=test/labels=v2/fold=01"
        
        results = run_sanity_checks(
            predictions=predictions,
            labels=labels,
            qlib_ic=1.0,
            experiment_name=exp_name
        )
        
        assert "ic_parity" in results
        assert "experiment_name" in results
        assert results["ic_parity"]["match"] == True
        assert results["experiment_name"]["model"] == "test"
    
    def test_run_checks_without_experiment_name(self):
        """Test running checks without experiment name validation."""
        predictions = pd.DataFrame({
            "date": [date(2023, 1, 1)] * 3,
            "ticker": ["A", "B", "C"],
            "prediction": [1.0, 2.0, 3.0]
        })
        labels = pd.DataFrame({
            "date": [date(2023, 1, 1)] * 3,
            "ticker": ["A", "B", "C"],
            "label": [1.0, 2.0, 3.0]
        })
        
        results = run_sanity_checks(
            predictions=predictions,
            labels=labels,
            qlib_ic=None,
            experiment_name=None
        )
        
        assert "ic_parity" in results
        assert "experiment_name" not in results


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


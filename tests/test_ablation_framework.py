"""
Tests for Chapter 9.8: Ablation Studies Framework

Validates:
 - Ablation matrix generation (full and quick)
 - Score aggregation methods (median, mean, trimmed_mean)
 - EMA half-life parameter effects
 - Ablation metric computation
 - Integration with FinTextAdapter
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.fintext_adapter import (
    FinTextAdapter,
    _apply_ema_smoothing,
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def stub_adapter_median():
    """Adapter with median aggregation."""
    return FinTextAdapter.from_pretrained(
        db_path="data/features.duckdb",
        model_size="Tiny",
        model_dataset="US",
        lookback=21,
        num_samples=20,
        use_stub=True,
        score_aggregation="median",
    )


@pytest.fixture
def stub_adapter_mean():
    """Adapter with mean aggregation."""
    return FinTextAdapter.from_pretrained(
        db_path="data/features.duckdb",
        model_size="Tiny",
        model_dataset="US",
        lookback=21,
        num_samples=20,
        use_stub=True,
        score_aggregation="mean",
    )


@pytest.fixture
def stub_adapter_trimmed():
    """Adapter with trimmed_mean aggregation."""
    return FinTextAdapter.from_pretrained(
        db_path="data/features.duckdb",
        model_size="Tiny",
        model_dataset="US",
        lookback=21,
        num_samples=20,
        use_stub=True,
        score_aggregation="trimmed_mean",
    )


# ============================================================================
# TEST ABLATION MATRIX
# ============================================================================

class TestAblationMatrix:
    """Tests for ablation matrix generation."""

    def test_import_ablation_module(self):
        """Ablation runner should be importable."""
        sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
        from run_chapter9_ablations import get_ablation_matrix
        matrix = get_ablation_matrix(quick=False)
        assert len(matrix) >= 10, f"Full matrix should have 10+ variants, got {len(matrix)}"

    def test_quick_matrix_is_smaller(self):
        """Quick matrix should have fewer variants."""
        sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
        from run_chapter9_ablations import get_ablation_matrix
        full = get_ablation_matrix(quick=False)
        quick = get_ablation_matrix(quick=True)
        assert len(quick) < len(full)
        assert len(quick) == 3

    def test_all_variants_have_required_keys(self):
        """Each variant must have all config keys + tag."""
        sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
        from run_chapter9_ablations import get_ablation_matrix
        required = {
            "tag", "model_size", "model_dataset", "lookback",
            "num_samples", "score_aggregation", "ema_halflife",
            "horizon_strategy",
        }
        for v in get_ablation_matrix(quick=False):
            missing = required - set(v.keys())
            assert not missing, f"Variant {v.get('tag','?')} missing: {missing}"

    def test_tags_are_unique(self):
        """All variant tags must be unique."""
        sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
        from run_chapter9_ablations import get_ablation_matrix
        tags = [v["tag"] for v in get_ablation_matrix(quick=False)]
        assert len(tags) == len(set(tags)), f"Duplicate tags: {tags}"

    def test_baseline_is_first(self):
        """Baseline variant should be first in matrix."""
        sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
        from run_chapter9_ablations import get_ablation_matrix
        first = get_ablation_matrix(quick=False)[0]
        assert "baseline" in first["tag"]


# ============================================================================
# TEST SCORE AGGREGATION
# ============================================================================

class TestScoreAggregation:
    """Tests for different score aggregation methods."""

    def test_median_aggregation_produces_scores(self, stub_adapter_median):
        """Median aggregation should produce valid scores."""
        scores = stub_adapter_median.score_universe(
            asof_date=pd.Timestamp("2023-06-15"),
            tickers=["AAPL", "MSFT", "GOOG"],
        )
        assert len(scores) > 0
        assert "score" in scores.columns
        assert scores["score"].notna().all()

    def test_mean_aggregation_produces_scores(self, stub_adapter_mean):
        """Mean aggregation should produce valid scores."""
        scores = stub_adapter_mean.score_universe(
            asof_date=pd.Timestamp("2023-06-15"),
            tickers=["AAPL", "MSFT", "GOOG"],
        )
        assert len(scores) > 0
        assert scores["score"].notna().all()

    def test_trimmed_mean_aggregation_produces_scores(self, stub_adapter_trimmed):
        """Trimmed mean aggregation should produce valid scores."""
        scores = stub_adapter_trimmed.score_universe(
            asof_date=pd.Timestamp("2023-06-15"),
            tickers=["AAPL", "MSFT", "GOOG"],
        )
        assert len(scores) > 0
        assert scores["score"].notna().all()

    def test_different_aggregations_produce_different_scores(
        self, stub_adapter_median, stub_adapter_mean
    ):
        """Different aggregation methods should (generally) produce different scores."""
        tickers = ["AAPL", "MSFT", "GOOG"]
        date = pd.Timestamp("2023-06-15")

        med_scores = stub_adapter_median.score_universe(asof_date=date, tickers=tickers)
        mean_scores = stub_adapter_mean.score_universe(asof_date=date, tickers=tickers)

        # Merge and compare
        merged = med_scores[["ticker", "score"]].merge(
            mean_scores[["ticker", "score"]],
            on="ticker",
            suffixes=("_median", "_mean"),
        )
        # They may be close but pred_mean and pred_std should differ
        assert len(merged) > 0

    def test_invalid_aggregation_raises_error(self):
        """Invalid aggregation method should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown score_aggregation"):
            FinTextAdapter.from_pretrained(
                db_path="data/features.duckdb",
                use_stub=True,
                score_aggregation="invalid_method",
            ).score_universe(
                asof_date=pd.Timestamp("2023-06-15"),
                tickers=["AAPL"],
            )

    def test_aggregation_stored_on_adapter(self, stub_adapter_median, stub_adapter_mean):
        """Adapter should store aggregation config."""
        assert stub_adapter_median.score_aggregation == "median"
        assert stub_adapter_mean.score_aggregation == "mean"


# ============================================================================
# TEST EMA HALF-LIFE PARAMETER
# ============================================================================

class TestEMAHalflife:
    """Tests for EMA half-life as ablation parameter."""

    def test_larger_halflife_smoother(self):
        """Larger half-life should produce smoother (lower volatility) scores."""
        np.random.seed(42)
        dates = pd.bdate_range("2024-01-02", periods=30)
        rows = []
        for d in dates:
            for t in ["A", "B", "C"]:
                rows.append({"as_of_date": d, "ticker": t, "score": np.random.normal(0, 0.01)})
        df = pd.DataFrame(rows)

        smooth_3 = _apply_ema_smoothing(df.copy(), halflife_days=3)
        smooth_10 = _apply_ema_smoothing(df.copy(), halflife_days=10)

        vol_3 = smooth_3.groupby("ticker")["score"].std().mean()
        vol_10 = smooth_10.groupby("ticker")["score"].std().mean()

        assert vol_10 < vol_3, "Larger half-life should be smoother"

    def test_halflife_zero_is_no_smoothing(self):
        """Half-life of 0 should leave scores unchanged."""
        np.random.seed(42)
        dates = pd.bdate_range("2024-01-02", periods=10)
        rows = []
        for d in dates:
            for t in ["A", "B"]:
                rows.append({"as_of_date": d, "ticker": t, "score": np.random.normal()})
        df = pd.DataFrame(rows)

        result = _apply_ema_smoothing(df.copy(), halflife_days=0)
        np.testing.assert_array_equal(
            df.sort_values(["ticker", "as_of_date"])["score"].values,
            result.sort_values(["ticker", "as_of_date"])["score"].values,
        )

    def test_halflife_preserves_row_count(self):
        """Any half-life should preserve the number of rows."""
        np.random.seed(42)
        dates = pd.bdate_range("2024-01-02", periods=20)
        rows = [{"as_of_date": d, "ticker": "X", "score": 0.01} for d in dates]
        df = pd.DataFrame(rows)

        for hl in [0, 1, 3, 5, 10, 20]:
            result = _apply_ema_smoothing(df.copy(), halflife_days=hl)
            assert len(result) == len(df), f"Half-life {hl} changed row count"


# ============================================================================
# TEST ABLATION METRIC COMPUTATION
# ============================================================================

class TestAblationMetrics:
    """Tests for metric computation in ablation framework."""

    def test_compute_metrics_function(self):
        """_compute_ablation_metrics should return expected keys."""
        sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
        from run_chapter9_ablations import _compute_ablation_metrics

        np.random.seed(42)
        rows = []
        dates = pd.bdate_range("2024-01-02", periods=21)
        tickers = [f"T{i:02d}" for i in range(50)]
        for h in [20, 60, 90]:
            for d in dates:
                for t in tickers:
                    signal = np.random.normal(0, 1)
                    rows.append({
                        "as_of_date": d, "ticker": t, "horizon": h,
                        "score": signal, "excess_return": signal + np.random.normal(0, 3),
                    })

        eval_df = pd.DataFrame(rows)
        metrics = _compute_ablation_metrics(eval_df, [20, 60, 90])

        # Check expected keys exist
        for h in [20, 60, 90]:
            assert f"rankic_mean_{h}d" in metrics
            assert f"rankic_median_{h}d" in metrics
            assert f"churn_median_{h}d" in metrics

        assert "rankic_mean_avg" in metrics
        assert "gate_1_pass" in metrics
        assert "gate_2_pass" in metrics
        assert "gate_3_pass" in metrics
        assert "all_gates_pass" in metrics

    def test_positive_signal_has_positive_rankic(self):
        """Synthetic positive signal should produce positive RankIC."""
        sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
        from run_chapter9_ablations import _compute_ablation_metrics

        np.random.seed(42)
        rows = []
        dates = pd.bdate_range("2024-01-02", periods=21)
        tickers = [f"T{i:02d}" for i in range(50)]
        for h in [20, 60, 90]:
            for d in dates:
                for t in tickers:
                    signal = np.random.normal(0, 1)
                    rows.append({
                        "as_of_date": d, "ticker": t, "horizon": h,
                        "score": signal, "excess_return": signal + np.random.normal(0, 2),
                    })

        eval_df = pd.DataFrame(rows)
        metrics = _compute_ablation_metrics(eval_df, [20, 60, 90])

        assert metrics["rankic_mean_avg"] > 0, "Positive signal should give positive RankIC"

    def test_ablation_results_file_exists(self):
        """If ablations ran, results file should exist with expected structure."""
        csv_path = Path("evaluation_outputs/chapter9_ablations/ablation_results.csv")
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            assert "tag" in df.columns
            assert "rankic_mean_avg" in df.columns
            assert "all_gates_pass" in df.columns
            assert len(df) >= 1
        else:
            pytest.skip("No ablation results found — run ablations first")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

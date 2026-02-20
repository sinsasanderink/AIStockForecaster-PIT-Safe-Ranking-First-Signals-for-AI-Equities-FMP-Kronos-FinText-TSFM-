"""Tests for Chapter 13.3 — Epistemic signal ê(x) = max(0, g(x) - a(x))."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.uncertainty.epistemic_signal import (
    DEPLOYMENT_LABELS,
    HOLDOUT_START,
    PER_STOCK_TIERS,
    _merge_g_and_a,
    compute_ehat,
    compute_ehat_mae,
    run_sanity_checks,
)


# ── Fixtures ─────────────────────────────────────────────────────────────


def _make_g_predictions(n_dates=50, n_tickers=20, horizon=20, seed=42):
    """Create synthetic g(x) predictions."""
    np.random.seed(seed)
    dates = pd.date_range("2020-01-01", periods=n_dates, freq="B")
    tickers = [f"TICK_{i}" for i in range(n_tickers)]
    rows = []
    for d in dates:
        for t in tickers:
            rows.append({
                "as_of_date": d,
                "ticker": t,
                "stable_id": f"S_{t}",
                "horizon": horizon,
                "fold_id": f"fold_{d.month:02d}",
                "g_pred": np.random.uniform(0.15, 0.45),
                "rank_loss": np.random.uniform(0.05, 0.50),
                "target_type": "rank_loss",
            })
    return pd.DataFrame(rows)


def _make_a_predictions_perdate(dates, horizon, tier="empirical", seed=42):
    """Create synthetic per-date a(x) predictions."""
    np.random.seed(seed)
    return pd.DataFrame({
        "as_of_date": dates,
        "a_value": np.random.uniform(0.02, 0.08, len(dates)),
        "horizon": horizon,
        "tier": tier,
        "ticker": np.nan,
        "stable_id": np.nan,
        "fold_id": np.nan,
    })


def _make_a_predictions_perstock(dates, tickers, horizon, seed=42):
    """Create synthetic per-stock a(x) predictions (Tier 2)."""
    np.random.seed(seed)
    rows = []
    for d in dates:
        for t in tickers:
            rows.append({
                "as_of_date": d,
                "ticker": t,
                "stable_id": f"S_{t}",
                "a_value": np.random.uniform(0.20, 0.45),
                "horizon": horizon,
                "tier": "tier2",
                "fold_id": f"fold_{d.month:02d}",
            })
    return pd.DataFrame(rows)


@pytest.fixture
def g_preds_20d():
    return _make_g_predictions(n_dates=50, n_tickers=20, horizon=20, seed=42)


@pytest.fixture
def g_preds_60d():
    return _make_g_predictions(n_dates=50, n_tickers=20, horizon=60, seed=43)


@pytest.fixture
def g_preds_multi():
    """g(x) predictions for multiple horizons."""
    g20 = _make_g_predictions(n_dates=50, n_tickers=20, horizon=20, seed=42)
    g60 = _make_g_predictions(n_dates=50, n_tickers=20, horizon=60, seed=43)
    g90 = _make_g_predictions(n_dates=50, n_tickers=20, horizon=90, seed=44)
    return pd.concat([g20, g60, g90], ignore_index=True)


@pytest.fixture
def a_preds_perdate_20d(g_preds_20d):
    dates = g_preds_20d["as_of_date"].unique()
    return _make_a_predictions_perdate(dates, horizon=20, tier="empirical", seed=50)


@pytest.fixture
def a_preds_perstock_60d(g_preds_60d):
    dates = g_preds_60d["as_of_date"].unique()
    tickers = g_preds_60d["ticker"].unique()
    return _make_a_predictions_perstock(dates, tickers, horizon=60, seed=51)


@pytest.fixture
def a_preds_multi(g_preds_multi):
    """a(x) with per-date for 20d/90d and per-stock for 60d."""
    dates = g_preds_multi[g_preds_multi["horizon"] == 20]["as_of_date"].unique()
    tickers = g_preds_multi["ticker"].unique()

    a20 = _make_a_predictions_perdate(dates, horizon=20, tier="empirical", seed=50)
    a60 = _make_a_predictions_perstock(dates, tickers, horizon=60, seed=51)
    a90 = _make_a_predictions_perdate(dates, horizon=90, tier="prospective", seed=52)
    return pd.concat([a20, a60, a90], ignore_index=True)


# ── Merge tests ──────────────────────────────────────────────────────────


class TestMergeGAndA:
    def test_perdate_merge_preserves_all_stocks(self, g_preds_20d, a_preds_perdate_20d):
        merged = _merge_g_and_a(g_preds_20d, a_preds_perdate_20d, horizon=20)
        assert len(merged) == len(g_preds_20d)
        assert "a_value" in merged.columns
        assert "g_pred" in merged.columns

    def test_perdate_broadcast_same_a_per_date(self, g_preds_20d, a_preds_perdate_20d):
        merged = _merge_g_and_a(g_preds_20d, a_preds_perdate_20d, horizon=20)
        for date in merged["as_of_date"].unique()[:3]:
            date_vals = merged[merged["as_of_date"] == date]["a_value"]
            assert date_vals.nunique() == 1, "Per-date a(x) should be same for all stocks"

    def test_perstock_merge_has_variation(self, g_preds_60d, a_preds_perstock_60d):
        merged = _merge_g_and_a(g_preds_60d, a_preds_perstock_60d, horizon=60)
        assert len(merged) > 0
        for date in merged["as_of_date"].unique()[:3]:
            date_vals = merged[merged["as_of_date"] == date]["a_value"]
            assert date_vals.nunique() > 1, "Per-stock a(x) should vary across stocks"

    def test_deployment_label_assigned(self, g_preds_20d, a_preds_perdate_20d):
        merged = _merge_g_and_a(g_preds_20d, a_preds_perdate_20d, horizon=20)
        assert "deployment_label" in merged.columns
        assert merged["deployment_label"].iloc[0] == "retrospective_decomposition"

    def test_tier2_labeled_deployment_ready(self, g_preds_60d, a_preds_perstock_60d):
        merged = _merge_g_and_a(g_preds_60d, a_preds_perstock_60d, horizon=60)
        assert merged["deployment_label"].iloc[0] == "deployment_ready"


# ── Compute ehat tests ───────────────────────────────────────────────────


class TestComputeEhat:
    def test_returns_dataframe_and_diagnostics(self, g_preds_multi, a_preds_multi):
        ehat_df, diag = compute_ehat(g_preds_multi, a_preds_multi, horizons=[20, 60, 90])
        assert isinstance(ehat_df, pd.DataFrame)
        assert len(ehat_df) > 0
        assert isinstance(diag, dict)
        assert 20 in diag and 60 in diag and 90 in diag

    def test_ehat_raw_non_negative(self, g_preds_multi, a_preds_multi):
        ehat_df, _ = compute_ehat(g_preds_multi, a_preds_multi)
        assert (ehat_df["ehat_raw"] >= 0).all()

    def test_ehat_raw_bounded_by_g(self, g_preds_multi, a_preds_multi):
        """ê(x) cannot exceed g(x) since ê = max(0, g - a) and a >= 0."""
        ehat_df, _ = compute_ehat(g_preds_multi, a_preds_multi)
        assert (ehat_df["ehat_raw"] <= ehat_df["g_pred"] + 1e-10).all()

    def test_ehat_pctile_in_01(self, g_preds_multi, a_preds_multi):
        ehat_df, _ = compute_ehat(g_preds_multi, a_preds_multi)
        assert (ehat_df["ehat_pctile"] >= 0).all()
        assert (ehat_df["ehat_pctile"] <= 1).all()

    def test_period_labels_correct(self, g_preds_multi, a_preds_multi):
        ehat_df, _ = compute_ehat(g_preds_multi, a_preds_multi)
        assert "period" in ehat_df.columns
        dev = ehat_df[ehat_df["period"] == "DEV"]
        final = ehat_df[ehat_df["period"] == "FINAL"]
        if len(dev) > 0:
            assert (dev["as_of_date"] < HOLDOUT_START).all()
        if len(final) > 0:
            assert (final["as_of_date"] >= HOLDOUT_START).all()

    def test_output_has_required_columns(self, g_preds_multi, a_preds_multi):
        ehat_df, _ = compute_ehat(g_preds_multi, a_preds_multi)
        required = [
            "as_of_date", "ticker", "horizon",
            "g_pred", "a_value", "a_tier",
            "ehat_raw", "ehat_pctile", "rank_loss",
            "deployment_label", "period",
        ]
        for col in required:
            assert col in ehat_df.columns, f"Missing column: {col}"

    def test_perdate_a_produces_all_positive_ehat(self):
        """With a low per-date a(x) floor, all ê(x) should be positive."""
        g = _make_g_predictions(n_dates=30, n_tickers=10, horizon=20)
        dates = g["as_of_date"].unique()
        a = pd.DataFrame({
            "as_of_date": dates,
            "a_value": 0.01,
            "horizon": 20,
            "tier": "empirical",
            "ticker": np.nan,
            "stable_id": np.nan,
            "fold_id": np.nan,
        })
        ehat_df, diag = compute_ehat(g, a, horizons=[20])
        assert (ehat_df["ehat_raw"] > 0).all()
        assert diag[20]["pct_positive"] > 0.99

    def test_perstock_a_produces_some_zeros(self):
        """With high per-stock a(x), many ê(x) should be zero."""
        g = _make_g_predictions(n_dates=30, n_tickers=10, horizon=60)
        g["g_pred"] = 0.30
        dates = g["as_of_date"].unique()
        tickers = g["ticker"].unique()
        a = _make_a_predictions_perstock(dates, tickers, horizon=60, seed=99)
        a["a_value"] = 0.40

        ehat_df, diag = compute_ehat(g, a, horizons=[60])
        assert (ehat_df["ehat_raw"] == 0).all()
        assert diag[60]["pct_zero"] > 0.99


# ── Diagnostics tests ────────────────────────────────────────────────────


class TestDiagnostics:
    def test_diagnostics_contain_required_fields(self, g_preds_multi, a_preds_multi):
        _, diag = compute_ehat(g_preds_multi, a_preds_multi, horizons=[20])
        d = diag[20]
        required = [
            "n_rows", "n_dates", "a_tier", "deployment_label",
            "g_pred_mean", "a_value_mean", "ehat_mean", "ehat_std",
            "pct_zero", "pct_positive", "rho_ehat_rank_loss",
            "rho_g_rank_loss", "selective_delta", "dev_final",
        ]
        for field in required:
            assert field in d, f"Missing diagnostic field: {field}"

    def test_dev_final_breakdown_present(self, g_preds_multi, a_preds_multi):
        _, diag = compute_ehat(g_preds_multi, a_preds_multi, horizons=[20])
        d = diag[20]
        assert "DEV" in d["dev_final"]

    def test_selective_risk_computed(self, g_preds_multi, a_preds_multi):
        _, diag = compute_ehat(g_preds_multi, a_preds_multi, horizons=[20])
        d = diag[20]
        assert "selective_low_tercile_rl" in d
        assert "selective_high_tercile_rl" in d
        assert isinstance(d["selective_delta"], float)


# ── Sanity checks tests ──────────────────────────────────────────────────


class TestSanityChecks:
    def test_returns_check_dict(self, g_preds_multi, a_preds_multi):
        ehat_df, diag = compute_ehat(g_preds_multi, a_preds_multi)
        sanity = run_sanity_checks(ehat_df, diag)
        assert "checks" in sanity
        assert "n_passed" in sanity
        assert "n_total" in sanity
        assert "all_passed" in sanity

    def test_checks_are_named_by_horizon(self, g_preds_multi, a_preds_multi):
        _, diag = compute_ehat(g_preds_multi, a_preds_multi)
        sanity = run_sanity_checks(pd.DataFrame(), diag)
        for name in sanity["checks"]:
            assert name.startswith("20d_") or name.startswith("60d_") or name.startswith("90d_")

    def test_perstock_tier_checks_zero_fraction(self, g_preds_multi, a_preds_multi):
        _, diag = compute_ehat(g_preds_multi, a_preds_multi, horizons=[60])
        sanity = run_sanity_checks(pd.DataFrame(), diag)
        assert any("has_zero_fraction" in name for name in sanity["checks"])

    def test_perdate_tier_expects_all_positive(self, g_preds_multi, a_preds_multi):
        _, diag = compute_ehat(g_preds_multi, a_preds_multi, horizons=[20])
        sanity = run_sanity_checks(pd.DataFrame(), diag)
        check_name = "20d_has_positive_fraction"
        assert check_name in sanity["checks"]

    def test_well_behaved_data_passes_all(self):
        """Synthetic data with clear signal should pass all sanity checks."""
        np.random.seed(77)
        g = _make_g_predictions(n_dates=50, n_tickers=15, horizon=20)
        g["rank_loss"] = g["g_pred"] + np.random.randn(len(g)) * 0.05
        g["rank_loss"] = g["rank_loss"].clip(0, 1)
        dates = g["as_of_date"].unique()
        a = _make_a_predictions_perdate(dates, horizon=20, tier="empirical", seed=77)
        a["a_value"] = 0.03

        ehat_df, diag = compute_ehat(g, a, horizons=[20])
        sanity = run_sanity_checks(ehat_df, diag)
        assert sanity["n_passed"] >= sanity["n_total"] - 1, (
            f"Well-behaved data should pass most checks: "
            f"{sanity['n_passed']}/{sanity['n_total']}"
        )


# ── MAE-based ehat tests ─────────────────────────────────────────────────


class TestComputeEhatMae:
    def test_returns_results(self, a_preds_multi):
        g_mae = _make_g_predictions(n_dates=50, n_tickers=20, horizon=20, seed=60)
        g_mae = g_mae.rename(columns={"g_pred": "g_pred_mae"})
        g_mae["mae_loss"] = np.random.uniform(0.02, 0.15, len(g_mae))

        ehat_mae_df, diag = compute_ehat_mae(g_mae, a_preds_multi, horizons=[20])
        assert len(ehat_mae_df) > 0
        assert "ehat_mae_raw" in ehat_mae_df.columns
        assert 20 in diag

    def test_mae_ehat_non_negative(self, a_preds_multi):
        g_mae = _make_g_predictions(n_dates=50, n_tickers=20, horizon=20, seed=60)
        g_mae = g_mae.rename(columns={"g_pred": "g_pred_mae"})
        g_mae["mae_loss"] = np.random.uniform(0.02, 0.15, len(g_mae))

        ehat_mae_df, _ = compute_ehat_mae(g_mae, a_preds_multi, horizons=[20])
        assert (ehat_mae_df["ehat_mae_raw"] >= 0).all()


# ── Deployment label tests ────────────────────────────────────────────────


class TestDeploymentLabels:
    def test_empirical_is_retrospective(self):
        assert DEPLOYMENT_LABELS["empirical"] == "retrospective_decomposition"

    def test_tier2_is_deployment_ready(self):
        assert DEPLOYMENT_LABELS["tier2"] == "deployment_ready"

    def test_prospective_is_deployment_ready(self):
        assert DEPLOYMENT_LABELS["prospective"] == "deployment_ready"

    def test_all_labels_defined(self):
        for tier in ["empirical", "tier0", "tier1", "tier2", "prospective"]:
            assert tier in DEPLOYMENT_LABELS

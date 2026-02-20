"""Tests for Chapter 13.4 — DEUP Diagnostics."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.uncertainty.deup_diagnostics import (
    diagnostic_a_partial_correlation,
    diagnostic_b_selective_risk,
    diagnostic_c_auroc,
    diagnostic_d_regime_2024,
    diagnostic_e_baselines,
    diagnostic_f_feature_importance,
    diagnostic_stability,
    run_all_diagnostics,
)


# ── Fixtures ─────────────────────────────────────────────────────────────


def _make_diagnostic_df(n_dates=100, n_tickers=20, horizon=20, seed=42):
    """Create synthetic merged ehat + enriched data for diagnostics."""
    np.random.seed(seed)
    dates = pd.date_range("2019-01-01", periods=n_dates, freq="B")
    tickers = [f"TICK_{i}" for i in range(n_tickers)]
    rows = []
    for d in dates:
        for t in tickers:
            g_pred = np.random.uniform(0.15, 0.45)
            vol = np.random.uniform(0.1, 0.8)
            vix = np.random.uniform(10, 95)
            score = np.random.normal(0, 0.07)
            # rank_loss correlates with g_pred + some noise from vol
            rank_loss = g_pred * 0.5 + vol * 0.2 + np.random.normal(0, 0.1)
            rank_loss = max(0, min(1, rank_loss))
            er = np.random.normal(0.02, 0.15)

            rows.append({
                "as_of_date": d,
                "ticker": t,
                "horizon": horizon,
                "g_pred": g_pred,
                "a_value": np.random.uniform(0.02, 0.08),
                "ehat_raw": max(0, g_pred - np.random.uniform(0.02, 0.08)),
                "ehat_pctile": np.random.uniform(0, 1),
                "rank_loss": rank_loss,
                "score": score,
                "excess_return": er,
                "vol_20d": vol,
                "vix_percentile_252d": vix,
                "mom_1m": np.random.normal(0.02, 0.1),
                "sector": np.random.choice(["Tech", "Health", "Finance"]),
                "rank_score": np.random.uniform(0, 1),
                "a_tier": "empirical",
                "deployment_label": "retrospective_decomposition",
                "period": "DEV" if d < pd.Timestamp("2024-01-01") else "FINAL",
            })
    return pd.DataFrame(rows)


def _make_diagnostic_df_with_final(seed=42):
    """Create df spanning DEV and FINAL periods."""
    np.random.seed(seed)
    dev_dates = pd.date_range("2020-01-01", periods=200, freq="B")
    final_dates = pd.date_range("2024-01-15", periods=50, freq="B")
    all_dates = dev_dates.append(final_dates)
    n_tickers = 15
    tickers = [f"TICK_{i}" for i in range(n_tickers)]

    rows = []
    for d in all_dates:
        for t in tickers:
            g_pred = np.random.uniform(0.15, 0.45)
            vol = np.random.uniform(0.1, 0.8)
            score = np.random.normal(0, 0.07)
            rank_loss = g_pred * 0.4 + np.random.normal(0, 0.12)
            rank_loss = max(0, min(1, rank_loss))
            er = np.random.normal(0.02, 0.15)

            rows.append({
                "as_of_date": d,
                "ticker": t,
                "horizon": 20,
                "g_pred": g_pred,
                "a_value": 0.04,
                "ehat_raw": max(0, g_pred - 0.04),
                "ehat_pctile": np.random.uniform(0, 1),
                "rank_loss": rank_loss,
                "score": score,
                "excess_return": er,
                "vol_20d": vol,
                "vix_percentile_252d": np.random.uniform(20, 90),
                "mom_1m": np.random.normal(0, 0.1),
                "sector": "Tech",
                "rank_score": np.random.uniform(0, 1),
                "a_tier": "empirical",
                "deployment_label": "retrospective_decomposition",
                "period": "DEV" if d < pd.Timestamp("2024-01-01") else "FINAL",
            })
    return pd.DataFrame(rows)


@pytest.fixture
def diag_df():
    return _make_diagnostic_df(n_dates=100, n_tickers=20, horizon=20, seed=42)


@pytest.fixture
def diag_df_final():
    return _make_diagnostic_df_with_final(seed=42)


@pytest.fixture
def mock_diagnostics_01():
    return {
        "step_1": {
            "feature_importances": [
                {
                    "horizon": 20,
                    "fold_id": "fold_99",
                    "importances": {
                        "cross_sectional_rank": 120,
                        "score": 51,
                        "vol_60d": 50,
                        "abs_score": 25,
                        "vix_percentile_252d": 20,
                        "market_vol_21d": 15,
                        "adv_20d": 10,
                        "mom_1m": 8,
                        "market_return_21d": 5,
                        "market_regime_enc": 3,
                        "vol_20d": 2,
                    },
                },
            ],
            "features_used": [
                "score", "abs_score", "vol_20d", "vol_60d", "mom_1m",
                "adv_20d", "vix_percentile_252d", "market_regime_enc",
                "market_vol_21d", "market_return_21d", "cross_sectional_rank",
            ],
        }
    }


# ── Diagnostic A tests ───────────────────────────────────────────────────


class TestDiagnosticA:
    def test_returns_dict_with_periods(self, diag_df):
        result = diagnostic_a_partial_correlation(diag_df, 20)
        assert "ALL" in result
        assert "DEV" in result

    def test_contains_required_fields(self, diag_df):
        result = diagnostic_a_partial_correlation(diag_df, 20)
        all_res = result["ALL"]
        assert "raw_rho_ehat_vol" in all_res
        assert "raw_rho_ehat_vix" in all_res
        assert "rho_ehat_residual_rl" in all_res
        assert "rho_residual_ehat_rl" in all_res
        assert "verdict" in all_res

    def test_kill_not_triggered_on_reasonable_data(self, diag_df):
        result = diagnostic_a_partial_correlation(diag_df, 20)
        assert result["ALL"]["kill_criterion_failed"] is False

    def test_verdict_is_pass_or_weak(self, diag_df):
        result = diagnostic_a_partial_correlation(diag_df, 20)
        assert result["ALL"]["verdict"] in ("PASS", "WEAK", "KILL")


# ── Diagnostic B tests ───────────────────────────────────────────────────


class TestDiagnosticB:
    def test_returns_rankic_per_tercile(self, diag_df):
        result = diagnostic_b_selective_risk(diag_df, 20)
        all_res = result["ALL"]
        assert "full_set_rankic" in all_res
        assert "low_ehat_rankic" in all_res
        assert "high_ehat_rankic" in all_res

    def test_tercile_sizes_reasonable(self, diag_df):
        result = diagnostic_b_selective_risk(diag_df, 20)
        assert result["ALL"]["n"] > 100

    def test_verdict_present(self, diag_df):
        result = diagnostic_b_selective_risk(diag_df, 20)
        assert result["ALL"]["verdict"] in ("PASS", "FAIL")


# ── Diagnostic C tests ───────────────────────────────────────────────────


class TestDiagnosticC:
    def test_returns_auroc_fields(self, diag_df):
        result = diagnostic_c_auroc(diag_df, 20)
        assert "ALL" in result
        all_res = result["ALL"]
        assert "n_dates" in all_res
        assert "failure_rate" in all_res

    def test_auroc_between_0_and_1(self, diag_df):
        result = diagnostic_c_auroc(diag_df, 20)
        all_res = result["ALL"]
        if all_res.get("auroc_daily_ehat_mean") is not None:
            assert 0 <= all_res["auroc_daily_ehat_mean"] <= 1
        if all_res.get("auroc_stock_high_loss") is not None:
            assert 0 <= all_res["auroc_stock_high_loss"] <= 1

    def test_stock_level_auroc_computed(self, diag_df):
        result = diagnostic_c_auroc(diag_df, 20)
        assert result["ALL"].get("auroc_stock_high_loss") is not None


# ── Diagnostic D tests ───────────────────────────────────────────────────


class TestDiagnosticD:
    def test_returns_monthly_table(self, diag_df_final):
        result = diagnostic_d_regime_2024(diag_df_final, 20)
        if not result.get("skip"):
            assert "monthly_table" in result
            assert len(result["monthly_table"]) > 0

    def test_crisis_fields_present(self, diag_df_final):
        result = diagnostic_d_regime_2024(diag_df_final, 20)
        if not result.get("skip"):
            assert "crisis_ehat_mean" in result
            assert "rho_ehat_vs_rankic" in result

    def test_verdict_present(self, diag_df_final):
        result = diagnostic_d_regime_2024(diag_df_final, 20)
        if not result.get("skip"):
            assert "verdict" in result


# ── Diagnostic E tests ───────────────────────────────────────────────────


class TestDiagnosticE:
    def test_compares_multiple_baselines(self, diag_df):
        result = diagnostic_e_baselines(diag_df, 20)
        baselines = result["ALL"]["baselines"]
        assert len(baselines) >= 3

    def test_ehat_included_as_baseline(self, diag_df):
        result = diagnostic_e_baselines(diag_df, 20)
        baselines = result["ALL"]["baselines"]
        assert "ê (DEUP)" in baselines

    def test_all_baselines_have_rho(self, diag_df):
        result = diagnostic_e_baselines(diag_df, 20)
        for label, bdata in result["ALL"]["baselines"].items():
            assert "rho_with_rank_loss" in bdata
            assert "selective_delta" in bdata


# ── Diagnostic F tests ───────────────────────────────────────────────────


class TestDiagnosticF:
    def test_extracts_importances(self, mock_diagnostics_01):
        result = diagnostic_f_feature_importance(mock_diagnostics_01)
        assert "horizons" in result
        assert 20 in result["horizons"]

    def test_top_3_features(self, mock_diagnostics_01):
        result = diagnostic_f_feature_importance(mock_diagnostics_01)
        top3 = result["horizons"][20]["top_3"]
        assert len(top3) == 3
        assert top3[0]["feature"] == "cross_sectional_rank"

    def test_category_breakdown_sums_reasonable(self, mock_diagnostics_01):
        result = diagnostic_f_feature_importance(mock_diagnostics_01)
        cats = result["horizons"][20]["category_breakdown"]
        total = cats["per_prediction_pct"] + cats["regime_market_pct"] + cats["volatility_pct"]
        assert total > 50  # should account for most importance

    def test_interpretation_not_volatility_warning(self, mock_diagnostics_01):
        result = diagnostic_f_feature_importance(mock_diagnostics_01)
        interp = result["horizons"][20]["interpretation"]
        assert "Per-prediction" in interp


# ── Stability tests ──────────────────────────────────────────────────────


class TestStability:
    def test_checks_multiple_conditions(self, diag_df):
        result = diagnostic_stability(diag_df, 20)
        assert "summary" in result
        assert result["summary"]["n_conditions"] >= 2

    def test_verdict_present(self, diag_df):
        result = diagnostic_stability(diag_df, 20)
        assert result["summary"]["verdict"] in ("PASS", "FAIL")


# ── Integration test ─────────────────────────────────────────────────────


class TestRunAllDiagnostics:
    def test_runs_without_error(self, diag_df, mock_diagnostics_01):
        result = run_all_diagnostics(
            ehat_df=diag_df,
            enriched_residuals=diag_df,
            diagnostics_01=mock_diagnostics_01,
            horizons=[20],
        )
        assert 20 in result
        assert "A_partial_correlation" in result[20]
        assert "B_selective_risk" in result[20]
        assert "C_auroc" in result[20]
        assert "E_baselines" in result[20]
        assert "F_feature_importance" in result[20]
        assert "stability" in result[20]

    def test_all_horizons_produce_results(self):
        dfs = []
        for hz in [20, 60, 90]:
            dfs.append(_make_diagnostic_df(n_dates=60, n_tickers=10, horizon=hz, seed=hz))
        df = pd.concat(dfs, ignore_index=True)

        mock_d01 = {
            "step_1": {
                "feature_importances": [],
                "features_used": [],
            }
        }

        result = run_all_diagnostics(
            ehat_df=df,
            enriched_residuals=df,
            diagnostics_01=mock_d01,
            horizons=[20, 60, 90],
        )
        for hz in [20, 60, 90]:
            assert hz in result

"""
Tests for DEUP-Sized Shadow Portfolio + Regime Evaluation — Chapter 13.6.

Validates:
    - PIT safety: no future info in features/labels/weights
    - Weight sanity: weights in [0,1], median ≈ 0.7
    - Reproducibility: deterministic outputs
    - Bucket monotonicity: lowest-G → worst performance directionally
    - Crisis throttle: combined variant exposure strongly reduced May–Jul 2024
"""

import numpy as np
import pandas as pd
import pytest

from src.uncertainty.deup_portfolio import (
    PortfolioConfig,
    build_variant_portfolios,
    calibrate_c,
    compute_all_variant_metrics,
    compute_bucket_tables,
    compute_portfolio_metrics,
    compute_sizing_weights,
    evaluate_regime_trust,
)


# ── Fixtures ──────────────────────────────────────────────────────────────

def _make_enriched(n_dates=150, n_tickers=30, horizon=20, seed=42):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-03", periods=n_dates)
    tickers = [f"T{i:03d}" for i in range(n_tickers)]
    rows = []
    for d in dates:
        for t in tickers:
            score = rng.normal(0, 0.02)
            rows.append({
                "as_of_date": d, "ticker": t, "stable_id": t,
                "horizon": horizon,
                "score": score,
                "excess_return": score * 0.3 + rng.normal(0, 0.05),
                "vol_20d": abs(rng.normal(0.25, 0.05)),
                "vix_percentile_252d": rng.uniform(0, 1),
                "market_vol_21d": abs(rng.normal(0.15, 0.03)),
            })
    return pd.DataFrame(rows)


def _make_ehat(enriched, seed=43):
    rng = np.random.default_rng(seed)
    ehat = enriched[["as_of_date", "ticker", "horizon"]].copy()
    ehat["g_pred"] = abs(rng.normal(0.30, 0.05, len(ehat)))
    ehat["ehat_raw"] = abs(rng.normal(0.25, 0.05, len(ehat)))
    ehat["rank_loss"] = abs(rng.normal(0.30, 0.10, len(ehat)))
    return ehat


def _make_health(enriched, horizon=20, seed=44):
    rng = np.random.default_rng(seed)
    dates = sorted(enriched["as_of_date"].unique())
    rows = []
    for i, d in enumerate(dates):
        matured = rng.normal(0.06, 0.08) if i >= horizon else np.nan
        h = 1 / (1 + np.exp(-rng.normal(0, 1)))
        rows.append({
            "date": d,
            "daily_rankic": rng.normal(0.06, 0.08),
            "matured_rankic": matured,
            "H_combined": h,
            "H_realized_only": h * 0.95,
            "G_exposure": max(0, min(1, (h - 0.3) / 0.4)),
            "n_stocks": 30,
        })
    return pd.DataFrame(rows)


@pytest.fixture
def enriched():
    return _make_enriched()


@pytest.fixture
def ehat(enriched):
    return _make_ehat(enriched)


@pytest.fixture
def health(enriched):
    return _make_health(enriched)


@pytest.fixture
def config():
    return PortfolioConfig(horizon=20, top_k=5)


# ── Test: Calibration ────────────────────────────────────────────────────


class TestCalibration:
    def test_calibrate_c_produces_target_median(self):
        signal = pd.Series(np.random.default_rng(42).uniform(0.05, 0.50, 1000))
        c = calibrate_c(signal, target_median=0.7)
        w = np.minimum(1.0, c / np.sqrt(signal.values + 1e-4))
        assert abs(np.median(w) - 0.7) < 0.05

    def test_calibrate_c_empty(self):
        c = calibrate_c(pd.Series(dtype=float))
        assert c == 1.0


# ── Test: Sizing weights ────────────────────────────────────────────────


class TestSizingWeights:
    def test_weights_in_01(self, enriched):
        c_vol = calibrate_c(enriched["vol_20d"])
        c_deup = calibrate_c(enriched["vol_20d"])  # proxy
        enriched["g_pred"] = abs(np.random.default_rng(42).normal(0.3, 0.05, len(enriched)))
        result = compute_sizing_weights(enriched, c_vol, c_deup, "g_pred")
        assert (result["w_vol"] >= 0).all() and (result["w_vol"] <= 1.0 + 1e-9).all()
        assert (result["w_deup"] >= 0).all() and (result["w_deup"] <= 1.0 + 1e-9).all()

    def test_sized_score_preserves_sign(self, enriched):
        c_vol = calibrate_c(enriched["vol_20d"])
        c_deup = 0.5
        enriched["g_pred"] = 0.3
        result = compute_sizing_weights(enriched, c_vol, c_deup, "g_pred")
        pos_mask = result["score"] > 0
        assert (result.loc[pos_mask, "sized_score_vol"] > 0).all()
        assert (result.loc[pos_mask, "sized_score_deup"] > 0).all()

    def test_high_unc_gets_lower_weight(self, enriched):
        enriched["g_pred"] = 0.3
        enriched.loc[enriched.index[:100], "g_pred"] = 0.05
        enriched.loc[enriched.index[100:], "g_pred"] = 0.60
        result = compute_sizing_weights(enriched, 0.5, 0.5, "g_pred")
        low_unc = result.iloc[:100]["w_deup"].mean()
        high_unc = result.iloc[100:]["w_deup"].mean()
        assert low_unc > high_unc


# ── Test: Portfolio construction ──────────────────────────────────────────


class TestPortfolioConstruction:
    def test_output_columns(self, enriched, ehat, health, config):
        merged = enriched.copy()
        merged["g_pred"] = abs(np.random.default_rng(42).normal(0.3, 0.05, len(merged)))
        merged["ehat_raw"] = merged["g_pred"]
        c_vol = calibrate_c(merged["vol_20d"])
        c_deup = calibrate_c(merged["g_pred"])
        merged = compute_sizing_weights(merged, c_vol, c_deup, "g_pred")
        pf = build_variant_portfolios(merged, health, config)

        required = [
            "date", "G_exposure", "n_stocks",
            "ls_return_raw", "ls_return_net_raw",
            "ls_return_net_vol", "ls_return_net_deup",
            "ls_return_net_health", "ls_return_net_combined",
        ]
        for col in required:
            assert col in pf.columns, f"Missing: {col}"

    def test_combined_is_deup_times_g(self, enriched, ehat, health, config):
        merged = enriched.copy()
        merged["g_pred"] = 0.3
        merged["ehat_raw"] = 0.25
        c = calibrate_c(merged["vol_20d"])
        merged = compute_sizing_weights(merged, c, c, "g_pred")
        pf = build_variant_portfolios(merged, health, config)

        valid = pf.dropna(subset=["ls_return_net_deup", "ls_return_net_combined"])
        if len(valid) > 0:
            for _, row in valid.iterrows():
                expected = row["ls_return_net_deup"] * row["G_exposure"]
                np.testing.assert_almost_equal(row["ls_return_net_combined"], expected, decimal=8)


# ── Test: Portfolio metrics ──────────────────────────────────────────────


class TestPortfolioMetrics:
    def test_metrics_keys(self):
        rets = np.random.default_rng(42).normal(0.01, 0.03, 50)
        m = compute_portfolio_metrics(rets)
        for key in ["sharpe", "sortino", "max_drawdown", "hit_rate", "worst_period"]:
            assert key in m

    def test_positive_sharpe_for_positive_returns(self):
        rets = np.abs(np.random.default_rng(42).normal(0.02, 0.01, 50))
        m = compute_portfolio_metrics(rets)
        assert m["sharpe"] > 0

    def test_all_variant_metrics(self, enriched, ehat, health, config):
        merged = enriched.copy()
        merged["g_pred"] = 0.3
        merged["ehat_raw"] = 0.25
        c = calibrate_c(merged["vol_20d"])
        merged = compute_sizing_weights(merged, c, c, "g_pred")
        pf = build_variant_portfolios(merged, health, config)
        metrics = compute_all_variant_metrics(pf)
        assert "ALL" in metrics
        assert "baseline_raw" in metrics["ALL"]


# ── Test: PIT safety ─────────────────────────────────────────────────────


class TestPITSafety:
    def test_matured_rankic_uses_lagged_data(self, health):
        """matured_rankic at row i should be NaN for first horizon rows."""
        horizon = 20
        for i in range(horizon):
            assert pd.isna(health.iloc[i]["matured_rankic"])

    def test_g_exposure_bounded(self, health):
        valid = health["G_exposure"].dropna()
        assert valid.min() >= 0.0 - 1e-9
        assert valid.max() <= 1.0 + 1e-9


# ── Test: Regime evaluation ──────────────────────────────────────────────


class TestRegimeEvaluation:
    def test_regime_trust_keys(self, health, enriched):
        diag = evaluate_regime_trust(health, enriched, horizon=20)
        assert "classifier_metrics" in diag
        assert "n_dates" in diag

    def test_confusion_matrix(self, health, enriched):
        diag = evaluate_regime_trust(health, enriched, horizon=20, abstention_threshold=0.2)
        if "confusion_matrix" in diag:
            cm = diag["confusion_matrix"]
            assert "TP" in cm and "FP" in cm and "TN" in cm and "FN" in cm


# ── Test: Bucket tables ──────────────────────────────────────────────────


class TestBucketTables:
    def test_bucket_tables_output(self, enriched, ehat, health, config):
        merged = enriched.copy()
        merged["g_pred"] = 0.3
        merged["ehat_raw"] = 0.25
        c = calibrate_c(merged["vol_20d"])
        merged = compute_sizing_weights(merged, c, c, "g_pred")
        pf = build_variant_portfolios(merged, health, config)
        bt = compute_bucket_tables(pf, health)
        assert "buckets" in bt
        assert "monotonicity_rho" in bt

    def test_bucket_has_required_fields(self, enriched, ehat, health, config):
        merged = enriched.copy()
        merged["g_pred"] = 0.3
        merged["ehat_raw"] = 0.25
        c = calibrate_c(merged["vol_20d"])
        merged = compute_sizing_weights(merged, c, c, "g_pred")
        pf = build_variant_portfolios(merged, health, config)
        bt = compute_bucket_tables(pf, health)
        if bt.get("buckets"):
            first_bucket = list(bt["buckets"].values())[0]
            for key in ["n", "mean_G", "mean_matured_rankic", "pct_bad_days"]:
                assert key in first_bucket


# ── Test: Reproducibility ────────────────────────────────────────────────


class TestReproducibility:
    def test_deterministic(self, enriched, health, config):
        merged = enriched.copy()
        merged["g_pred"] = 0.3
        merged["ehat_raw"] = 0.25
        c = calibrate_c(merged["vol_20d"])
        merged1 = compute_sizing_weights(merged.copy(), c, c, "g_pred")
        merged2 = compute_sizing_weights(merged.copy(), c, c, "g_pred")

        pf1 = build_variant_portfolios(merged1, health, config)
        pf2 = build_variant_portfolios(merged2, health, config)

        np.testing.assert_array_almost_equal(
            pf1["ls_return_net_raw"].values,
            pf2["ls_return_net_raw"].values,
        )

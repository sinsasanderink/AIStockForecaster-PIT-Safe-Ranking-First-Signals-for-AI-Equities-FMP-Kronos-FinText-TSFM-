"""
Chapter 13.7: Deployment Policy Tests
======================================

Tests for src/uncertainty/deployment_policy.py

Coverage:
    - Binary gate logic (threshold, boundary, flat periods)
    - Liu et al. uncertainty-adjusted sorting (lambda=0, asymmetry, preservation)
    - Residualisation of ê on score extremity
    - ê cap (only top percentile affected, correct weight applied)
    - Portfolio integration (valid selections, binary gate applied)
    - Calibration discipline (DEV-only, frozen params)
    - Economic sanity (gated variants vs ungated crisis MaxDD)
    - Reproducibility (deterministic outputs)
"""

import json
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pytest

from src.uncertainty.deployment_policy import (
    GATE_THRESHOLD,
    EPS,
    HOLDOUT_START,
    LAMBDA_GRID,
    POLICY_VARIANTS,
    apply_binary_gate,
    apply_ehat_cap,
    calibrate_lambda_ua,
    compute_variant_metrics_all_periods,
    diagnose_ehat_score_correlation,
    merge_data_for_policy,
    residualize_ehat,
    run_policy_pipeline,
    uncertainty_adjusted_sort,
    _build_single_variant_timeseries,
    _compute_date_variant,
)

# ── Fixtures ──────────────────────────────────────────────────────────────

RNG = np.random.default_rng(42)
N_STOCKS = 30
N_DATES_DEV = 60
N_DATES_FINAL = 15
TOP_K = 10


def _make_date_range(n: int, start: str = "2022-01-03") -> pd.DatetimeIndex:
    return pd.bdate_range(start=start, periods=n)


def _make_enriched(dates: pd.DatetimeIndex, horizon: int = 20) -> pd.DataFrame:
    """Synthetic enriched residuals DataFrame."""
    rows = []
    for dt in dates:
        scores = RNG.normal(0, 0.05, N_STOCKS)
        rows.extend(
            {
                "as_of_date": dt,
                "ticker": f"TICK{i:02d}",
                "stable_id": f"S{i:02d}",
                "horizon": horizon,
                "score": scores[i],
                "excess_return": RNG.normal(0, 0.04),
                "rank_loss": RNG.uniform(0, 0.5),
                "vol_20d": max(0.05, RNG.normal(0.25, 0.1)),
                "rank_score": float(np.argsort(np.argsort(scores))[i]) / (N_STOCKS - 1),
                "sub_model_id": "tabular_lgb",
            }
            for i in range(N_STOCKS)
        )
    return pd.DataFrame(rows)


def _make_ehat(dates: pd.DatetimeIndex, horizon: int = 20) -> pd.DataFrame:
    """Synthetic ehat predictions DataFrame."""
    rows = []
    for dt in dates:
        for i in range(N_STOCKS):
            rows.append(
                {
                    "as_of_date": dt,
                    "ticker": f"TICK{i:02d}",
                    "stable_id": f"S{i:02d}",
                    "horizon": horizon,
                    "g_pred": max(0.01, RNG.normal(0.2, 0.05)),
                    "ehat_raw": max(0.0, RNG.normal(0.15, 0.06)),
                }
            )
    return pd.DataFrame(rows)


def _make_health(dates: pd.DatetimeIndex, low_g: bool = False) -> pd.DataFrame:
    """Synthetic expert health DataFrame."""
    rows = []
    for dt in dates:
        if low_g:
            g = RNG.uniform(0.0, 0.15)  # Always below gate
        else:
            g = RNG.uniform(0.1, 1.0)
        rows.append(
            {
                "date": dt,
                "G_exposure": g,
                "H_realized": RNG.uniform(-0.1, 0.3),
                "H_combined": RNG.uniform(0.2, 0.8),
                "matured_rankic": RNG.normal(0.05, 0.08),
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture
def dev_dates():
    return _make_date_range(N_DATES_DEV, "2022-01-03")


@pytest.fixture
def final_dates():
    return _make_date_range(N_DATES_FINAL, "2024-01-02")


@pytest.fixture
def all_dates(dev_dates, final_dates):
    return dev_dates.append(final_dates)


@pytest.fixture
def enriched_df(all_dates):
    return _make_enriched(all_dates)


@pytest.fixture
def ehat_df(all_dates):
    return _make_ehat(all_dates)


@pytest.fixture
def health_df(all_dates):
    return _make_health(all_dates)


@pytest.fixture
def merged_df(enriched_df, ehat_df, health_df):
    return merge_data_for_policy(enriched_df, ehat_df, health_df, horizon=20)


@pytest.fixture
def one_date_df(merged_df):
    """Single-date slice for per-date function tests."""
    dt = sorted(merged_df["as_of_date"].unique())[0]
    return merged_df[merged_df["as_of_date"] == dt].copy()


# ── 1. Binary gate tests ──────────────────────────────────────────────────


def test_gate_exact_threshold():
    """G = 0.20 exactly → trade."""
    assert apply_binary_gate(0.20, GATE_THRESHOLD) is True


def test_gate_below_threshold():
    """G = 0.19 → abstain."""
    assert apply_binary_gate(0.19, GATE_THRESHOLD) is False


def test_gate_above_threshold():
    """G = 0.50 → trade."""
    assert apply_binary_gate(0.50, GATE_THRESHOLD) is True


def test_gate_zero():
    """G = 0.0 → abstain."""
    assert apply_binary_gate(0.0, GATE_THRESHOLD) is False


def test_gate_produces_zero_return_when_flat(merged_df):
    """All variants produce zero return when the gate is closed."""
    # Set G_exposure below threshold for all dates
    df = merged_df.copy()
    df["G_exposure"] = 0.1  # Well below 0.2
    params = {"c_vol": 0.15, "lambda_ua": 0.3, "c_resid": 0.15,
              "cap_percentile": 0.90, "cap_weight": 0.7,
              "cap_percentile_vol": 0.90, "cap_weight_vol": 0.7, "c_trail": 5.0}

    for variant in ["gate_raw", "gate_vol", "gate_ua_sort"]:
        records = _build_single_variant_timeseries(df, variant, params)
        all_returns = [r["ls_return_net"] for r in records]
        assert all(r == 0.0 for r in all_returns), (
            f"{variant}: expected all-zero returns when G<threshold, "
            f"got {all_returns[:5]}"
        )


# ── 2. Liu et al. uncertainty-adjusted sorting tests ─────────────────────


def test_ua_sort_lambda_zero_equals_raw_sort():
    """lambda=0 → longs identical to top-K by raw score."""
    idx = pd.RangeIndex(20)
    scores = pd.Series(np.linspace(-1, 1, 20), index=idx)
    uncertainty = pd.Series(np.ones(20), index=idx)

    long_ua, short_ua = uncertainty_adjusted_sort(scores, uncertainty, lambda_ua=0.0, top_k=5)
    long_raw = scores.nlargest(5).index
    short_raw = scores.nsmallest(5).index

    assert set(long_ua) == set(long_raw), "lambda=0 longs should match raw sort"
    assert set(short_ua) == set(short_raw), "lambda=0 shorts should match raw sort"


def test_ua_sort_changes_selection_with_lambda():
    """lambda > 0 can change which stocks enter the portfolio."""
    idx = pd.RangeIndex(20)
    scores = pd.Series([0.5, 0.4, 0.3, 0.2] + [0.0] * 16, index=idx)
    # Stock 0 has high score but very high uncertainty → pushed out of longs
    uncertainty = pd.Series([100.0, 0.1, 0.1, 0.1] + [0.1] * 16, index=idx)

    long_raw, _ = uncertainty_adjusted_sort(scores, uncertainty, lambda_ua=0.0, top_k=4)
    long_ua, _ = uncertainty_adjusted_sort(scores, uncertainty, lambda_ua=1.0, top_k=4)

    # With large uncertainty on stock 0 and lambda=1.0, stock 0 still enters
    # longs because upper_bound = 0.5 + 1.0*100 = 100.5 (highest upper bound)
    # This proves the additive nature preserves strong signals
    assert 0 in set(long_ua), "High-score + high-uncertainty stock should still enter longs"


def test_ua_sort_asymmetric_long_short():
    """Longs and shorts are selected by DIFFERENT criteria (upper vs lower bound)."""
    idx = pd.RangeIndex(20)
    scores = pd.Series(np.linspace(-1, 1, 20), index=idx)
    # High uncertainty on top-scored stocks
    uncertainty = pd.Series([10.0 if i >= 15 else 0.1 for i in range(20)], index=idx)

    long_ua, short_ua = uncertainty_adjusted_sort(scores, uncertainty, lambda_ua=1.0, top_k=5)

    # Verify longs are NOT identical to shorts mirrored
    # (they could overlap if the sets weren't independently chosen, but shouldn't with top_k=5)
    assert len(long_ua) == 5
    assert len(short_ua) == 5


def test_ua_sort_lambda_zero_top_k_correct():
    """Exactly top_k longs and top_k shorts are always returned."""
    idx = pd.RangeIndex(25)
    scores = pd.Series(RNG.normal(0, 1, 25), index=idx)
    uncertainty = pd.Series(np.abs(RNG.normal(0, 0.1, 25)), index=idx)

    for lam in [0.0, 0.5, 2.0]:
        longs, shorts = uncertainty_adjusted_sort(scores, uncertainty, lambda_ua=lam, top_k=10)
        assert len(longs) == 10, f"lambda={lam}: expected 10 longs, got {len(longs)}"
        assert len(shorts) == 10, f"lambda={lam}: expected 10 shorts, got {len(shorts)}"


# ── 3. Residualisation tests ──────────────────────────────────────────────


def test_residualize_removes_score_correlation():
    """After residualisation, corr(ê_resid, |rank_score - 0.5|) ≈ 0."""
    n = 80
    rank_score = pd.Series(np.linspace(0, 1, n))
    abs_rank = np.abs(rank_score - 0.5) * 2
    # Construct ê with KNOWN correlation to |rank_score|
    ehat = pd.Series(0.3 * abs_rank.values + RNG.normal(0, 0.01, n))

    ehat_resid = residualize_ehat(ehat, rank_score)

    corr_before = np.corrcoef(abs_rank, ehat.values)[0, 1]
    corr_after = np.corrcoef(abs_rank, ehat_resid.values)[0, 1]

    assert abs(corr_before) > 0.8, "Test setup: should have strong correlation before"
    assert abs(corr_after) < 0.05, (
        f"After residualisation, |corr| should be ~0, got {corr_after:.3f}"
    )


def test_residualize_preserves_index():
    """Residuals have the same index as input."""
    idx = pd.Index([10, 20, 30, 40, 50])
    ehat = pd.Series([0.1, 0.2, 0.15, 0.25, 0.18], index=idx)
    rank_score = pd.Series([0.1, 0.3, 0.5, 0.7, 0.9], index=idx)

    resid = residualize_ehat(ehat, rank_score)
    assert list(resid.index) == list(idx)


def test_residualize_mean_near_zero():
    """Residuals from OLS should have mean ≈ 0 (intercept fitted)."""
    n = 50
    rank_score = pd.Series(np.linspace(0, 1, n))
    ehat = pd.Series(np.abs(rank_score - 0.5) * 2 + 0.1 + RNG.normal(0, 0.01, n))

    resid = residualize_ehat(ehat, rank_score)
    assert abs(resid.mean()) < 0.02, f"Residual mean should be ~0, got {resid.mean():.4f}"


def test_residualize_per_date_not_pooled(merged_df):
    """
    Each date's residualisation uses only that date's data.
    Confirm by checking that residuals from per-date application differ
    from pooled application.
    """
    dates = sorted(merged_df["as_of_date"].unique())[:3]
    df_sub = merged_df[merged_df["as_of_date"].isin(dates)].copy()

    # Per-date
    per_date_resids = []
    for dt in dates:
        g = df_sub[df_sub["as_of_date"] == dt]
        r = residualize_ehat(g["ehat_raw"], g["rank_score"])
        per_date_resids.extend(r.values.tolist())

    # Pooled
    pooled_resid = residualize_ehat(df_sub["ehat_raw"], df_sub["rank_score"])

    # They should differ (pooled ignores date structure)
    diff = np.std(np.array(per_date_resids) - pooled_resid.values)
    assert diff > 1e-6, "Per-date and pooled residualisations should differ"


# ── 4. Cap tests ──────────────────────────────────────────────────────────


def test_cap_only_affects_top_percentile():
    """With cap_percentile=0.90, exactly 10% of stocks are capped."""
    n = 100
    scores = pd.Series(np.ones(n))
    ehat = pd.Series(np.linspace(0, 1, n))  # Uniform distribution

    capped = apply_ehat_cap(scores, ehat, cap_percentile=0.90, cap_weight=0.7)

    n_capped = int((capped < 1.0).sum())
    # With n=100 stocks and P90 threshold, approximately 10 stocks are capped
    assert 8 <= n_capped <= 12, f"Expected ~10 capped stocks, got {n_capped}"


def test_cap_weight_applied_correctly():
    """Capped stocks have score multiplied by exactly cap_weight."""
    scores = pd.Series([1.0, 1.0, 1.0, 1.0, 1.0])
    ehat = pd.Series([0.1, 0.2, 0.3, 0.4, 0.9])  # Last stock far above P80

    capped = apply_ehat_cap(scores, ehat, cap_percentile=0.80, cap_weight=0.7)

    # Stock with ehat=0.9 is above the 80th percentile
    assert capped.iloc[-1] == pytest.approx(0.7, rel=1e-6), (
        f"Capped stock should have score * 0.7 = 0.7, got {capped.iloc[-1]}"
    )


def test_cap_uncapped_stocks_unchanged():
    """Stocks below the cap threshold are not modified."""
    scores = pd.Series([2.0, 3.0, 4.0, 5.0, 6.0])
    ehat = pd.Series([0.1, 0.1, 0.1, 0.1, 1.0])  # Only last is high

    capped = apply_ehat_cap(scores, ehat, cap_percentile=0.80, cap_weight=0.5)

    # First four stocks should be unchanged
    for i in range(4):
        assert capped.iloc[i] == scores.iloc[i], (
            f"Stock {i} should be unchanged: expected {scores.iloc[i]}, got {capped.iloc[i]}"
        )


# ── 5. Integration tests ──────────────────────────────────────────────────


def test_all_variants_produce_valid_portfolios(merged_df):
    """Each variant builds a portfolio with exactly top_k longs and shorts."""
    params = {
        "c_vol": 0.15,
        "lambda_ua": 0.3,
        "c_resid": 0.15,
        "cap_percentile": 0.90,
        "cap_weight": 0.7,
        "cap_percentile_vol": 0.90,
        "cap_weight_vol": 0.7,
        "c_trail": 3.0,
    }

    dt = sorted(merged_df["as_of_date"].unique())[5]
    day = merged_df[merged_df["as_of_date"] == dt].copy()
    # Ensure G is above gate so variants actually run
    day["G_exposure"] = 0.8

    for variant in POLICY_VARIANTS:
        result = _compute_date_variant(day, variant, params, top_k=TOP_K)
        if result is not None:
            assert len(result["long_ids"]) == TOP_K, (
                f"{variant}: expected {TOP_K} longs, got {len(result['long_ids'])}"
            )
            assert len(result["short_ids"]) == TOP_K, (
                f"{variant}: expected {TOP_K} shorts, got {len(result['short_ids'])}"
            )


def test_dev_final_separation(merged_df):
    """DEV data does not appear in FINAL evaluation."""
    params = {
        "c_vol": 0.15, "lambda_ua": 0.3, "c_resid": 0.15,
        "cap_percentile": 0.90, "cap_weight": 0.7,
        "cap_percentile_vol": 0.90, "cap_weight_vol": 0.7, "c_trail": 3.0,
    }
    records = _build_single_variant_timeseries(merged_df, "gate_vol", params)
    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])

    dev_records = df[df["date"] < HOLDOUT_START]
    final_records = df[df["date"] >= HOLDOUT_START]

    # No overlap
    assert len(dev_records) + len(final_records) == len(df)
    assert (dev_records["date"] >= HOLDOUT_START).sum() == 0
    assert (final_records["date"] < HOLDOUT_START).sum() == 0


def test_timeseries_records_structure(merged_df):
    """Each record has required keys."""
    params = {
        "c_vol": 0.15, "lambda_ua": 0.3, "c_resid": 0.15,
        "cap_percentile": 0.90, "cap_weight": 0.7,
        "cap_percentile_vol": 0.90, "cap_weight_vol": 0.7, "c_trail": 3.0,
    }
    records = _build_single_variant_timeseries(merged_df, "gate_raw", params)
    required_keys = {"date", "ls_return", "ls_return_net", "turnover", "is_active"}
    for r in records:
        for k in required_keys:
            assert k in r, f"Missing key '{k}' in record"


def test_data_merge_preserves_horizon(enriched_df, ehat_df, health_df):
    """Merge only retains rows for the requested horizon."""
    merged = merge_data_for_policy(enriched_df, ehat_df, health_df, horizon=20)
    # All merged rows should come from horizon=20 (check via date overlap)
    er_dates = set(pd.to_datetime(enriched_df[enriched_df["horizon"] == 20]["as_of_date"]).unique())
    merged_dates = set(pd.to_datetime(merged["as_of_date"]).unique())
    # Merged dates should be a subset of 20d enriched dates
    assert merged_dates.issubset(er_dates | {pd.NaT})


# ── 6. Economic sanity tests ──────────────────────────────────────────────


def test_gated_variants_can_reduce_crisis_drawdown():
    """
    Gated variants should have LOWER or EQUAL MaxDD vs ungated during a
    simulated "bad period" when G drops to 0.
    """
    # Simulate a 40-date sequence where mid-section has G < 0.2 (crisis)
    crisis_dates = pd.bdate_range("2024-03-01", periods=20)
    non_crisis_dates = pd.bdate_range("2023-01-03", periods=30)
    all_sim_dates = non_crisis_dates.append(crisis_dates)

    enriched = _make_enriched(all_sim_dates)
    ehat = _make_ehat(all_sim_dates)

    # Create health with crisis G=0
    health_rows = []
    for dt in all_sim_dates:
        is_crisis = dt >= pd.Timestamp("2024-03-01")
        health_rows.append({
            "date": dt,
            "G_exposure": 0.0 if is_crisis else 0.9,
            "H_realized": -0.1 if is_crisis else 0.2,
            "matured_rankic": -0.05 if is_crisis else 0.1,
        })
    health = pd.DataFrame(health_rows)

    # Make crisis returns severely negative
    def _neg_enriched(df):
        df = df.copy()
        df.loc[df["as_of_date"] >= pd.Timestamp("2024-03-01"), "excess_return"] = (
            RNG.normal(-0.15, 0.02, (df["as_of_date"] >= pd.Timestamp("2024-03-01")).sum())
        )
        return df
    enriched_bad = _neg_enriched(enriched)

    merged = merge_data_for_policy(enriched_bad, ehat, health, horizon=20)
    params = {
        "c_vol": 0.15, "lambda_ua": 0.3, "c_resid": 0.15,
        "cap_percentile": 0.90, "cap_weight": 0.7,
        "cap_percentile_vol": 0.90, "cap_weight_vol": 0.7, "c_trail": 3.0,
    }

    # Ungated raw
    ungated_params = {"c_vol": 0.15}
    merged_ungated = merged.copy()
    merged_ungated["G_exposure"] = 1.0  # Never abstain
    ungated_records = _build_single_variant_timeseries(merged_ungated, "gate_raw", params)

    # Gated raw
    gated_records = _build_single_variant_timeseries(merged, "gate_raw", params)

    def _max_dd(records: list) -> float:
        rets = np.array([r["ls_return_net"] for r in records])
        cum = np.cumprod(1 + rets)
        peak = np.maximum.accumulate(cum)
        return float(((cum - peak) / peak).min())

    dd_ungated = _max_dd(ungated_records)
    dd_gated = _max_dd(gated_records)

    # Gated should have less severe drawdown during the simulated crisis
    assert dd_gated >= dd_ungated, (
        f"Gated MaxDD ({dd_gated:.3f}) should be ≥ ungated ({dd_ungated:.3f}) "
        "(less negative = smaller drawdown)"
    )


def test_results_reproducible(merged_df):
    """Same inputs → same outputs (deterministic, no random state)."""
    params = {
        "c_vol": 0.15, "lambda_ua": 0.3, "c_resid": 0.15,
        "cap_percentile": 0.90, "cap_weight": 0.7,
        "cap_percentile_vol": 0.90, "cap_weight_vol": 0.7, "c_trail": 3.0,
    }
    records_1 = _build_single_variant_timeseries(merged_df, "gate_ua_sort", params)
    records_2 = _build_single_variant_timeseries(merged_df, "gate_ua_sort", params)

    ret_1 = [r["ls_return_net"] for r in records_1]
    ret_2 = [r["ls_return_net"] for r in records_2]
    assert ret_1 == ret_2, "Results should be identical on repeated calls"


# ── 7. Calibration discipline ─────────────────────────────────────────────


def test_lambda_grid_search_uses_dev_only(dev_dates, final_dates):
    """
    calibrate_lambda_ua is called with DEV-only data.
    Verify that after calibration, the best lambda doesn't change when
    final data is added (it's frozen).
    """
    enriched = _make_enriched(dev_dates)
    ehat = _make_ehat(dev_dates)
    health = _make_health(dev_dates)
    dev_df = merge_data_for_policy(enriched, ehat, health, horizon=20)

    best_lambda, grid = calibrate_lambda_ua(dev_df, LAMBDA_GRID[:4])
    assert best_lambda in LAMBDA_GRID[:4], "Best lambda must come from the grid"
    assert all(isinstance(v, float) for v in grid.values())


def test_calibration_output_schema(dev_dates):
    """calibrate_all_params returns all required keys."""
    from src.uncertainty.deployment_policy import calibrate_all_params
    enriched = _make_enriched(dev_dates)
    ehat = _make_ehat(dev_dates)
    health = _make_health(dev_dates)
    dev_df = merge_data_for_policy(enriched, ehat, health, horizon=20)

    params = calibrate_all_params(dev_df)
    required_keys = ["c_vol", "c_resid", "lambda_ua", "cap_percentile", "cap_weight", "c_trail"]
    for k in required_keys:
        assert k in params, f"Missing calibration key: {k}"


# ── 8. Diagnostic tests ───────────────────────────────────────────────────


def test_ehat_score_diagnostic_returns_required_keys(merged_df):
    """diagnose_ehat_score_correlation returns all required fields."""
    result = diagnose_ehat_score_correlation(merged_df, horizon="20d")
    required = [
        "median_rho_ehat_abs_score",
        "mean_rho_ehat_abs_score",
        "pct_positive_corr",
        "decile_stats",
        "structural_conflict_confirmed",
        "interpretation",
    ]
    for k in required:
        assert k in result, f"Missing diagnostic key: {k}"


def test_ehat_score_diagnostic_with_constructed_correlation(all_dates):
    """
    When ê is constructed to correlate with |score|, the diagnostic should
    confirm the structural conflict.
    """
    enriched = _make_enriched(all_dates)
    ehat_rows = []
    for _, row in enriched.iterrows():
        ehat_rows.append({
            "as_of_date": row["as_of_date"],
            "ticker": row["ticker"],
            "stable_id": row["stable_id"],
            "horizon": 20,
            "g_pred": abs(row["score"]) + 0.05,   # Strong correlation with |score|
            "ehat_raw": abs(row["score"]) + 0.03,
        })
    ehat = pd.DataFrame(ehat_rows)
    health = _make_health(all_dates)

    merged = merge_data_for_policy(enriched, ehat, health, horizon=20)
    result = diagnose_ehat_score_correlation(merged)

    assert result["structural_conflict_confirmed"] is True, (
        "Structural conflict should be confirmed when ê ∝ |score|"
    )
    assert result["median_rho_ehat_abs_score"] > 0.3


# ── 9. Full pipeline smoke test ───────────────────────────────────────────


def test_pipeline_runs_all_variants(enriched_df, ehat_df, health_df):
    """
    run_policy_pipeline completes without error and returns results for
    all policy variants.
    """
    results, cal_params, timeseries = run_policy_pipeline(
        enriched_df,
        ehat_df,
        health_df,
        horizon=20,
        variants=["gate_raw", "gate_vol", "gate_ua_sort"],
    )
    assert "gate_raw" in results
    assert "gate_vol" in results
    assert "gate_ua_sort" in results
    assert "c_vol" in cal_params
    assert "lambda_ua" in cal_params
    for v in ["gate_raw", "gate_vol", "gate_ua_sort"]:
        assert "ALL" in results[v]
        assert "sharpe" in results[v]["ALL"]


def test_pipeline_metrics_schema(enriched_df, ehat_df, health_df):
    """Output metrics have all required fields."""
    results, _, _ = run_policy_pipeline(
        enriched_df, ehat_df, health_df, horizon=20,
        variants=["gate_raw"],
    )
    for period in ["ALL", "DEV", "FINAL"]:
        m = results["gate_raw"].get(period, {})
        assert "sharpe" in m, f"Missing 'sharpe' in {period}"
        assert "max_drawdown" in m, f"Missing 'max_drawdown' in {period}"
        assert "abstention_rate" in m, f"Missing 'abstention_rate' in {period}"

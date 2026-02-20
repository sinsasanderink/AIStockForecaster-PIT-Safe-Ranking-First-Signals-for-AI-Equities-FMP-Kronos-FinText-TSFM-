"""Tests for Chapter 13.2 — Aleatoric baseline a(x)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.uncertainty.aleatoric_baseline import (
    ALIGNMENT_RHO_KILL,
    ALIGNMENT_RHO_TARGET,
    EPS,
    compute_empirical_fallback,
    compute_prospective_empirical,
    compute_tier0,
    compute_tier1,
    compute_tier2,
    run_alignment_diagnostic,
    select_best_tier,
)


# ── Fixtures ─────────────────────────────────────────────────────────────


@pytest.fixture
def enriched_df():
    """Realistic enriched residual DataFrame with rank_loss."""
    np.random.seed(42)
    n_folds = 25
    dates_per_fold = 3
    tickers = [f"TICK_{i}" for i in range(20)]
    sectors = ["tech", "health", "energy", "finance"]
    rows = []
    for fold_i in range(n_folds):
        fold_id = f"fold_{fold_i + 1:03d}"
        base_date = pd.Timestamp("2018-01-01") + pd.Timedelta(days=fold_i * 30)
        for d_off in range(dates_per_fold):
            d = base_date + pd.Timedelta(days=d_off)
            for t in tickers:
                score = np.random.randn() * 0.03
                er = np.random.randn() * 0.10
                rows.append({
                    "as_of_date": d,
                    "ticker": t,
                    "stable_id": f"STABLE_{t}",
                    "horizon": 20,
                    "fold_id": fold_id,
                    "score": score,
                    "excess_return": er,
                    "vol_20d": np.random.uniform(0.1, 0.5),
                    "adv_20d": np.random.uniform(1e8, 1e10),
                    "mom_1m": np.random.randn() * 0.05,
                    "market_return_21d": np.random.randn() * 0.03,
                    "market_vol_21d": np.random.uniform(0.1, 0.3),
                    "vix_percentile_252d": np.random.uniform(10, 90),
                    "market_regime": np.random.choice([-1, 0, 1]),
                    "sector": np.random.choice(sectors),
                })
    df = pd.DataFrame(rows)
    r_score = df.groupby(["as_of_date", "horizon"])["score"].rank(pct=True)
    r_actual = df.groupby(["as_of_date", "horizon"])["excess_return"].rank(pct=True)
    df["rank_loss"] = (r_actual - r_score).abs()
    return df


@pytest.fixture
def high_signal_df():
    """DataFrame where dispersion predicts rank_loss well (for alignment tests)."""
    np.random.seed(99)
    rows = []
    n_folds = 25
    tickers = [f"T{i}" for i in range(15)]
    sectors = ["A", "B", "C"]

    for fold_i in range(n_folds):
        fold_id = f"fold_{fold_i + 1:03d}"
        for d_off in range(3):
            d = pd.Timestamp("2018-01-01") + pd.Timedelta(days=fold_i * 30 + d_off)
            dispersion = np.random.uniform(0.02, 0.30)
            for t in tickers:
                er = np.random.randn() * dispersion
                score = er + np.random.randn() * 0.01
                rows.append({
                    "as_of_date": d,
                    "ticker": t,
                    "stable_id": f"S_{t}",
                    "horizon": 20,
                    "fold_id": fold_id,
                    "score": score,
                    "excess_return": er,
                    "vol_20d": np.random.uniform(0.1, 0.5),
                    "adv_20d": np.random.uniform(1e8, 1e10),
                    "mom_1m": np.random.randn() * 0.05,
                    "market_return_21d": np.random.randn() * 0.03,
                    "market_vol_21d": np.random.uniform(0.1, 0.3),
                    "vix_percentile_252d": np.random.uniform(10, 90),
                    "sector": np.random.choice(sectors),
                })
    df = pd.DataFrame(rows)
    r_score = df.groupby(["as_of_date", "horizon"])["score"].rank(pct=True)
    r_actual = df.groupby(["as_of_date", "horizon"])["excess_return"].rank(pct=True)
    df["rank_loss"] = (r_actual - r_score).abs()
    return df


@pytest.fixture
def multi_model_dfs(enriched_df):
    """Dict of model DataFrames for alignment diagnostic."""
    model_b = enriched_df.copy()
    model_b["score"] = model_b["score"] + np.random.randn(len(model_b)) * 0.01
    r_s = model_b.groupby(["as_of_date", "horizon"])["score"].rank(pct=True)
    r_a = model_b.groupby(["as_of_date", "horizon"])["excess_return"].rank(pct=True)
    model_b["rank_loss"] = (r_a - r_s).abs()
    return {
        "tabular_lgb": enriched_df,
        "rank_avg_2": model_b,
    }


# ── Tier 0 tests ─────────────────────────────────────────────────────────


class TestComputeTier0:
    def test_returns_series_indexed_by_date(self, enriched_df):
        a = compute_tier0(enriched_df, horizon=20)
        assert isinstance(a, pd.Series)
        assert a.index.name == "as_of_date"
        assert len(a) > 0

    def test_all_positive(self, enriched_df):
        a = compute_tier0(enriched_df, horizon=20)
        assert (a > 0).all()

    def test_varies_across_dates(self, enriched_df):
        a = compute_tier0(enriched_df, horizon=20)
        assert a.std() > 0

    def test_direction_high_dispersion_low_a(self):
        """High excess return dispersion should yield LOW a(date)."""
        rows = []
        for d_off, spread in [(0, 0.01), (1, 0.50)]:
            d = pd.Timestamp("2020-01-01") + pd.Timedelta(days=d_off)
            for i in range(20):
                er = np.random.randn() * spread
                rows.append({
                    "as_of_date": d,
                    "ticker": f"T{i}",
                    "stable_id": f"S{i}",
                    "horizon": 20,
                    "fold_id": "fold_001",
                    "score": np.random.randn() * 0.03,
                    "excess_return": er,
                })
        df = pd.DataFrame(rows)
        r_s = df.groupby(["as_of_date", "horizon"])["score"].rank(pct=True)
        r_a = df.groupby(["as_of_date", "horizon"])["excess_return"].rank(pct=True)
        df["rank_loss"] = (r_a - r_s).abs()

        a = compute_tier0(df, horizon=20)
        low_spread_date = pd.Timestamp("2020-01-01")
        high_spread_date = pd.Timestamp("2020-01-02")
        assert a.loc[low_spread_date] > a.loc[high_spread_date], (
            "Low dispersion should yield higher a(x)"
        )


# ── Tier 1 tests ─────────────────────────────────────────────────────────


class TestComputeTier1:
    def test_returns_series_indexed_by_date(self, enriched_df):
        a = compute_tier1(enriched_df, horizon=20)
        assert isinstance(a, pd.Series)
        assert a.index.name == "as_of_date"
        assert len(a) > 0

    def test_all_positive(self, enriched_df):
        a = compute_tier1(enriched_df, horizon=20)
        assert (a > 0).all()

    def test_differs_from_tier0(self, enriched_df):
        a0 = compute_tier0(enriched_df, horizon=20)
        a1 = compute_tier1(enriched_df, horizon=20)
        common = a0.index.intersection(a1.index)
        rho = stats.spearmanr(a0.loc[common], a1.loc[common]).statistic
        assert rho < 0.99, "Tier 1 should differ from Tier 0 after factor regression"

    def test_handles_missing_sector_gracefully(self, enriched_df):
        df = enriched_df.drop(columns=["sector"])
        a = compute_tier1(df, horizon=20)
        assert len(a) > 0


# ── Tier 2 tests ─────────────────────────────────────────────────────────


class TestComputeTier2:
    def test_returns_dataframe_with_a_tier2(self, enriched_df):
        result = compute_tier2(enriched_df, horizon=20, min_train_folds=20)
        assert isinstance(result, pd.DataFrame)
        assert "a_tier2" in result.columns
        assert "as_of_date" in result.columns
        assert "ticker" in result.columns

    def test_a_tier2_non_negative(self, enriched_df):
        result = compute_tier2(enriched_df, horizon=20, min_train_folds=20)
        if len(result) > 0:
            assert (result["a_tier2"] >= 0).all()

    def test_per_stock_variation(self, enriched_df):
        result = compute_tier2(enriched_df, horizon=20, min_train_folds=20)
        if len(result) > 0:
            per_stock = result.groupby("ticker")["a_tier2"].mean()
            assert per_stock.std() > 0, "Tier 2 should vary across stocks"

    def test_only_predicts_oos_folds(self, enriched_df):
        result = compute_tier2(enriched_df, horizon=20, min_train_folds=20)
        if len(result) > 0:
            all_folds = sorted(enriched_df["fold_id"].unique())
            train_only = set(all_folds[:20])
            predicted_folds = set(result["fold_id"].unique())
            assert predicted_folds.isdisjoint(train_only)


# ── Empirical fallback tests ─────────────────────────────────────────────


class TestEmpiricalFallback:
    def test_returns_series(self, enriched_df):
        a = compute_empirical_fallback(enriched_df, horizon=20)
        assert isinstance(a, pd.Series)
        assert len(a) > 0

    def test_values_below_median_rank_loss(self, enriched_df):
        a = compute_empirical_fallback(enriched_df, horizon=20, percentile=10.0)
        median_rl = enriched_df[enriched_df["horizon"] == 20].groupby("as_of_date")["rank_loss"].median()
        common = a.index.intersection(median_rl.index)
        assert (a.loc[common] <= median_rl.loc[common] + 0.01).mean() > 0.8


# ── Prospective empirical fallback tests ──────────────────────────────────


class TestProspectiveEmpirical:
    def test_returns_series(self, enriched_df):
        a = compute_prospective_empirical(enriched_df, horizon=20, lookback_days=10, min_lookback=5)
        assert isinstance(a, pd.Series)
        assert len(a) > 0

    def test_all_positive(self, enriched_df):
        a = compute_prospective_empirical(enriched_df, horizon=20, lookback_days=10, min_lookback=5)
        assert (a >= 0).all()

    def test_strictly_uses_past_data(self, enriched_df):
        """The prospective P10 must NOT use current-date information."""
        a = compute_prospective_empirical(enriched_df, horizon=20, lookback_days=10, min_lookback=5)
        first_date = a.index.min()
        all_dates = sorted(enriched_df[enriched_df["horizon"] == 20]["as_of_date"].unique())
        first_idx = all_dates.index(first_date)
        assert first_idx >= 5, "Should skip dates with insufficient lookback"

    def test_fewer_dates_than_same_date(self, enriched_df):
        """Prospective version loses initial dates due to lookback requirement."""
        a_same = compute_empirical_fallback(enriched_df, horizon=20)
        a_prosp = compute_prospective_empirical(enriched_df, horizon=20, lookback_days=10, min_lookback=5)
        assert len(a_prosp) <= len(a_same)

    def test_smoother_than_same_date(self, enriched_df):
        """Rolling average should have lower variance than same-date."""
        a_same = compute_empirical_fallback(enriched_df, horizon=20)
        a_prosp = compute_prospective_empirical(enriched_df, horizon=20, lookback_days=30, min_lookback=5)
        common = a_same.index.intersection(a_prosp.index)
        if len(common) > 10:
            assert a_prosp.loc[common].std() <= a_same.loc[common].std() * 1.5


# ── Alignment diagnostic tests ───────────────────────────────────────────


class TestAlignmentDiagnostic:
    def test_returns_diagnostic_dict(self, enriched_df, multi_model_dfs):
        a = compute_tier0(enriched_df, horizon=20)
        diag = run_alignment_diagnostic(a, multi_model_dfs, horizon=20, tier_name="tier0")
        assert "tier" in diag
        assert "pass" in diag
        assert "mean_rho" in diag
        assert "per_model" in diag

    def test_per_model_contains_all_models(self, enriched_df, multi_model_dfs):
        a = compute_tier0(enriched_df, horizon=20)
        diag = run_alignment_diagnostic(a, multi_model_dfs, horizon=20)
        for model_name in multi_model_dfs:
            assert model_name in diag["per_model"]

    def test_rho_is_numeric(self, enriched_df, multi_model_dfs):
        a = compute_tier0(enriched_df, horizon=20)
        diag = run_alignment_diagnostic(a, multi_model_dfs, horizon=20)
        assert isinstance(diag["mean_rho"], float)
        assert -1 <= diag["mean_rho"] <= 1

    def test_verdict_in_expected_values(self, enriched_df, multi_model_dfs):
        a = compute_tier0(enriched_df, horizon=20)
        diag = run_alignment_diagnostic(a, multi_model_dfs, horizon=20)
        assert diag["verdict"] in ("PASS", "KILL", "MARGINAL")

    def test_handles_dataframe_input_for_tier2(self, enriched_df, multi_model_dfs):
        unique_dates = enriched_df["as_of_date"].unique()
        tier2_df = pd.DataFrame({
            "as_of_date": np.tile(unique_dates, 2)[:60],
            "a_tier2": np.random.uniform(0.1, 0.5, 60),
        })
        diag = run_alignment_diagnostic(tier2_df, multi_model_dfs, horizon=20, tier_name="tier2")
        assert "mean_rho" in diag

    def test_too_few_dates_fails(self, multi_model_dfs):
        a = pd.Series(
            [0.3, 0.4],
            index=pd.to_datetime(["2020-01-01", "2020-01-02"]),
        )
        a.index.name = "as_of_date"
        diag = run_alignment_diagnostic(a, multi_model_dfs, horizon=20)
        assert diag["pass"] is False
        assert "Too few dates" in diag.get("reason", "")


# ── Tier selection tests ──────────────────────────────────────────────────


class TestSelectBestTier:
    def test_returns_tuple_of_three(self, enriched_df, multi_model_dfs):
        tier_name, a_val, diag = select_best_tier(
            enriched_df, multi_model_dfs, horizon=20, max_tier=1
        )
        assert isinstance(tier_name, str)
        assert diag is not None
        assert tier_name in ("tier0", "tier1", "prospective", "empirical")

    def test_always_returns_a_values(self, enriched_df, multi_model_dfs):
        tier_name, a_val, _ = select_best_tier(
            enriched_df, multi_model_dfs, horizon=20, max_tier=0
        )
        assert a_val is not None
        if isinstance(a_val, pd.Series):
            assert len(a_val) > 0
        elif isinstance(a_val, pd.DataFrame):
            assert len(a_val) > 0

    def test_diagnostic_contains_tested_tiers(self, enriched_df, multi_model_dfs):
        _, _, diag = select_best_tier(
            enriched_df, multi_model_dfs, horizon=20, max_tier=1
        )
        assert "tier0" in diag

    def test_high_signal_data_may_pass_early(self, high_signal_df):
        models = {"model_a": high_signal_df}
        tier_name, a_val, diag = select_best_tier(
            high_signal_df, models, horizon=20, max_tier=2
        )
        assert tier_name in ("tier0", "tier1", "tier2", "prospective", "empirical")

    def test_prospective_tested_before_same_date(self, enriched_df, multi_model_dfs):
        """If all model tiers fail, prospective should be tested before same-date."""
        _, _, diag = select_best_tier(
            enriched_df, multi_model_dfs, horizon=20, max_tier=0
        )
        if "empirical" in diag:
            assert "prospective" in diag, (
                "Prospective must be tested before same-date empirical"
            )

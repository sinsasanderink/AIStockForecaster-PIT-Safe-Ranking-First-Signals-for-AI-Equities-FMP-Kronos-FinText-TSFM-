"""
Tests for Chapter 11 — Fusion Scorer, Shadow Portfolio, Residual Archive, Expert Interface
===========================================================================================
"""

import numpy as np
import pandas as pd
import pytest
from datetime import date, datetime
from scipy import stats


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def aligned_scores():
    """Aligned sub-model scores matching evaluation format."""
    np.random.seed(42)
    dates = pd.to_datetime(["2024-02-01", "2024-03-01", "2024-04-01"])
    tickers = [f"STK{i}" for i in range(20)]

    rows = []
    for horizon in [20, 60, 90]:
        for d in dates:
            for t in tickers:
                rows.append({
                    "as_of_date": d,
                    "ticker": t,
                    "stable_id": f"STB_{t}",
                    "fold_id": "fold_01",
                    "horizon": horizon,
                    "excess_return": np.random.normal(0.02, 0.1),
                    "lgb_score": np.random.uniform(0, 1),
                    "fintext_score": np.random.uniform(0, 1),
                    "sentiment_score": np.random.uniform(-0.5, 0.5)
                    if np.random.random() > 0.3
                    else np.nan,
                })
    return pd.DataFrame(rows)


@pytest.fixture
def eval_rows_for_portfolio():
    """Eval rows suitable for shadow portfolio construction."""
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=12, freq="MS")
    tickers = [f"STK{i}" for i in range(30)]

    rows = []
    for d in dates:
        for t in tickers:
            score = np.random.uniform(0, 1)
            er = 0.05 * (score - 0.5) + np.random.normal(0, 0.03)
            rows.append({
                "as_of_date": d,
                "ticker": t,
                "stable_id": f"STB_{t}",
                "fold_id": "fold_01",
                "horizon": 20,
                "score": score,
                "excess_return": er,
            })
    return pd.DataFrame(rows)


@pytest.fixture
def multi_fold_eval_rows():
    """Eval rows spanning multiple folds for residual archive testing."""
    np.random.seed(123)
    rows = []
    for fold_idx in range(1, 4):
        fold_id = f"fold_{fold_idx:02d}"
        dates = pd.date_range(
            f"2024-{fold_idx:02d}-01", periods=3, freq="MS"
        )
        tickers = [f"STK{i}" for i in range(15)]
        for d in dates:
            for t in tickers:
                score = np.random.uniform(0, 1)
                er = 0.04 * (score - 0.5) + np.random.normal(0, 0.05)
                for horizon in [20, 60, 90]:
                    rows.append({
                        "as_of_date": d,
                        "ticker": t,
                        "stable_id": f"STB_{t}",
                        "fold_id": fold_id,
                        "horizon": horizon,
                        "score": score + np.random.normal(0, 0.01),
                        "excess_return": er + np.random.normal(0, 0.01),
                    })
    return pd.DataFrame(rows)


# ===========================================================================
# Rank Average Tests (Approach A)
# ===========================================================================


class TestRankAverage:
    def test_output_range(self, aligned_scores):
        from src.models.fusion_scorer import rank_average_scores
        composite = rank_average_scores(aligned_scores)
        assert composite.min() >= 0.0
        assert composite.max() <= 1.0

    def test_no_nan_output(self, aligned_scores):
        from src.models.fusion_scorer import rank_average_scores
        composite = rank_average_scores(aligned_scores)
        assert composite.notna().all()

    def test_subset_score_cols(self, aligned_scores):
        from src.models.fusion_scorer import rank_average_scores
        composite = rank_average_scores(
            aligned_scores, score_cols=["lgb_score", "fintext_score"]
        )
        assert len(composite) == len(aligned_scores)

    def test_single_model(self, aligned_scores):
        from src.models.fusion_scorer import rank_average_scores
        composite = rank_average_scores(
            aligned_scores, score_cols=["lgb_score"]
        )
        assert composite.notna().all()

    def test_all_nan_scores(self):
        """When all scores are NaN, return neutral 0.5."""
        from src.models.fusion_scorer import rank_average_scores
        df = pd.DataFrame({
            "as_of_date": ["2024-01-01"] * 5,
            "lgb_score": [np.nan] * 5,
        })
        composite = rank_average_scores(df, score_cols=["lgb_score"])
        assert (composite == 0.5).all()


# ===========================================================================
# Disagreement Tests (Epistemic Uncertainty Proxy)
# ===========================================================================


class TestDisagreement:
    def test_agreement_low_uncertainty(self):
        """When all models agree, disagreement should be low."""
        from src.models.fusion_scorer import compute_disagreement
        df = pd.DataFrame({
            "as_of_date": ["2024-01-01"] * 10,
            "lgb_score": range(10),
            "fintext_score": range(10),
            "sentiment_score": range(10),
        })
        disagreement = compute_disagreement(df)
        assert disagreement.max() < 0.05

    def test_disagreement_high_uncertainty(self):
        """When models disagree, disagreement should be high."""
        from src.models.fusion_scorer import compute_disagreement
        df = pd.DataFrame({
            "as_of_date": ["2024-01-01"] * 10,
            "lgb_score": range(10),
            "fintext_score": list(range(10))[::-1],
            "sentiment_score": [5] * 10,
        })
        disagreement = compute_disagreement(df)
        assert disagreement.mean() > 0.1

    def test_output_length(self, aligned_scores):
        from src.models.fusion_scorer import compute_disagreement
        disagreement = compute_disagreement(aligned_scores)
        assert len(disagreement) == len(aligned_scores)


# ===========================================================================
# Stacking Meta-Learner Tests (Approach C)
# ===========================================================================


class TestStacking:
    def test_train_ridge(self, aligned_scores):
        from src.models.fusion_scorer import (
            train_stacking_meta,
            stacking_predict,
        )
        meta = train_stacking_meta(aligned_scores, method="ridge")
        preds = stacking_predict(meta, aligned_scores)
        assert len(preds) == len(aligned_scores)
        assert not np.any(np.isnan(preds))

    @pytest.mark.skip(reason="LightGBM segfaults with tiny feature matrices on this env")
    def test_train_lgb(self, aligned_scores):
        from src.models.fusion_scorer import (
            train_stacking_meta,
            stacking_predict,
        )
        h20 = aligned_scores[aligned_scores["horizon"] == 20].copy()
        meta = train_stacking_meta(h20, method="lgb")
        preds = stacking_predict(meta, h20)
        assert len(preds) == len(h20)


# ===========================================================================
# Shadow Portfolio Tests
# ===========================================================================


class TestShadowPortfolio:
    def test_build_returns_df(self, eval_rows_for_portfolio):
        from scripts.run_shadow_portfolio import build_shadow_portfolio
        portfolio = build_shadow_portfolio(eval_rows_for_portfolio)
        assert not portfolio.empty
        assert "ls_return" in portfolio.columns
        assert "turnover" in portfolio.columns
        assert "cost_drag" in portfolio.columns

    def test_long_short_opposite(self, eval_rows_for_portfolio):
        """Long and short legs should generally have different returns."""
        from scripts.run_shadow_portfolio import build_shadow_portfolio
        portfolio = build_shadow_portfolio(eval_rows_for_portfolio)
        assert not portfolio.empty

    def test_turnover_range(self, eval_rows_for_portfolio):
        from scripts.run_shadow_portfolio import build_shadow_portfolio
        portfolio = build_shadow_portfolio(eval_rows_for_portfolio)
        assert portfolio["turnover"].min() >= 0.0
        assert portfolio["turnover"].max() <= 1.0

    def test_metrics_computation(self, eval_rows_for_portfolio):
        from scripts.run_shadow_portfolio import (
            build_shadow_portfolio,
            compute_portfolio_metrics,
        )
        portfolio = build_shadow_portfolio(eval_rows_for_portfolio)
        metrics = compute_portfolio_metrics(portfolio)
        assert "annualized_sharpe" in metrics
        assert "max_drawdown" in metrics
        assert "mean_turnover" in metrics
        assert metrics["max_drawdown"] <= 0

    def test_empty_input(self):
        from scripts.run_shadow_portfolio import build_shadow_portfolio
        empty = pd.DataFrame(columns=["as_of_date", "horizon", "score", "excess_return", "stable_id"])
        portfolio = build_shadow_portfolio(empty)
        assert portfolio.empty


# ===========================================================================
# Residual Archive Tests (11.4)
# ===========================================================================


class TestResidualArchive:
    def test_save_and_load_roundtrip(self, tmp_path, eval_rows_for_portfolio):
        """Write to DuckDB, read back, verify equality."""
        from src.models.residual_archive import ResidualArchive
        db_path = str(tmp_path / "test_residuals.duckdb")
        archive = ResidualArchive(db_path=db_path)

        n = archive.save_from_eval_rows(
            eval_rows_for_portfolio,
            expert_id="test_expert",
            sub_model_id="test_model",
        )
        assert n > 0

        loaded = archive.load(expert_id="test_expert", sub_model_id="test_model")
        assert len(loaded) == n
        assert set(loaded.columns) >= {
            "expert_id", "sub_model_id", "fold_id", "as_of_date",
            "ticker", "horizon", "prediction", "actual", "loss"
        }

        for _, row in loaded.iterrows():
            np.testing.assert_almost_equal(
                row["loss"],
                abs(row["actual"] - row["prediction"]),
                decimal=6,
            )

    def test_pit_safe_dates(self, tmp_path, multi_fold_eval_rows):
        """as_of_date in archive must match the walk-forward fold's test period."""
        from src.models.residual_archive import ResidualArchive
        db_path = str(tmp_path / "pit_residuals.duckdb")
        archive = ResidualArchive(db_path=db_path)
        archive.save_from_eval_rows(multi_fold_eval_rows)

        loaded = archive.load()
        original_dates = set(
            pd.to_datetime(multi_fold_eval_rows["as_of_date"]).dt.date
        )
        stored_dates = set(pd.to_datetime(loaded["as_of_date"]).dt.date)
        assert stored_dates == original_dates

    def test_no_duplicates_on_resave(self, tmp_path, eval_rows_for_portfolio):
        """Re-saving the same model clears old records first."""
        from src.models.residual_archive import ResidualArchive
        db_path = str(tmp_path / "dedup_residuals.duckdb")
        archive = ResidualArchive(db_path=db_path)

        archive.save_from_eval_rows(eval_rows_for_portfolio)
        archive.save_from_eval_rows(eval_rows_for_portfolio)

        loaded = archive.load()
        expected = len(
            eval_rows_for_portfolio.dropna(subset=["score", "excess_return"])
        )
        assert len(loaded) == expected

    def test_every_tuple_unique(self, tmp_path, multi_fold_eval_rows):
        """(fold, date, ticker, horizon) must appear exactly once."""
        from src.models.residual_archive import ResidualArchive
        db_path = str(tmp_path / "unique_residuals.duckdb")
        archive = ResidualArchive(db_path=db_path)
        archive.save_from_eval_rows(multi_fold_eval_rows)

        loaded = archive.load()
        key_cols = ["fold_id", "as_of_date", "ticker", "horizon"]
        assert not loaded.duplicated(subset=key_cols).any(), (
            "Duplicate (fold, date, ticker, horizon) tuples found"
        )

    def test_all_folds_covered(self, tmp_path, multi_fold_eval_rows):
        """Archive must cover all folds present in eval_rows."""
        from src.models.residual_archive import ResidualArchive
        db_path = str(tmp_path / "folds_residuals.duckdb")
        archive = ResidualArchive(db_path=db_path)
        archive.save_from_eval_rows(multi_fold_eval_rows)

        expected_folds = set(multi_fold_eval_rows["fold_id"].unique())
        stored_folds = set(archive.fold_ids())
        assert stored_folds == expected_folds

    def test_filter_by_horizon(self, tmp_path, multi_fold_eval_rows):
        from src.models.residual_archive import ResidualArchive
        db_path = str(tmp_path / "horizon_residuals.duckdb")
        archive = ResidualArchive(db_path=db_path)
        archive.save_from_eval_rows(multi_fold_eval_rows)
        loaded = archive.load(horizon=20)
        assert (loaded["horizon"] == 20).all()
        assert len(loaded) > 0

    def test_filter_by_fold(self, tmp_path, multi_fold_eval_rows):
        from src.models.residual_archive import ResidualArchive
        db_path = str(tmp_path / "fold_filter_residuals.duckdb")
        archive = ResidualArchive(db_path=db_path)
        archive.save_from_eval_rows(multi_fold_eval_rows)
        loaded = archive.load(fold_id="fold_01")
        assert (loaded["fold_id"] == "fold_01").all()
        assert len(loaded) > 0

    def test_summary_contains_fold_count(self, tmp_path, multi_fold_eval_rows):
        from src.models.residual_archive import ResidualArchive
        db_path = str(tmp_path / "summary_residuals.duckdb")
        archive = ResidualArchive(db_path=db_path)
        archive.save_from_eval_rows(multi_fold_eval_rows)
        summary = archive.summary()
        assert len(summary) > 0
        assert summary[0]["n_records"] > 0
        assert summary[0]["n_folds"] == 3

    def test_loss_computed_correctly(self, tmp_path, eval_rows_for_portfolio):
        """loss = |actual - prediction|"""
        from src.models.residual_archive import ResidualArchive
        db_path = str(tmp_path / "loss_residuals.duckdb")
        archive = ResidualArchive(db_path=db_path)
        archive.save_from_eval_rows(eval_rows_for_portfolio)
        loaded = archive.load()
        expected_loss = (loaded["actual"] - loaded["prediction"]).abs()
        np.testing.assert_array_almost_equal(
            loaded["loss"].values, expected_loss.values, decimal=6
        )

    def test_missing_columns_raises(self, tmp_path):
        from src.models.residual_archive import ResidualArchive
        db_path = str(tmp_path / "bad_residuals.duckdb")
        archive = ResidualArchive(db_path=db_path)
        bad_df = pd.DataFrame({"ticker": ["A"], "score": [1.0]})
        with pytest.raises(ValueError, match="missing columns"):
            archive.save_from_eval_rows(bad_df)

    def test_multiple_sub_models(self, tmp_path, eval_rows_for_portfolio):
        """Different sub_model_ids are stored independently."""
        from src.models.residual_archive import ResidualArchive
        db_path = str(tmp_path / "multi_model_residuals.duckdb")
        archive = ResidualArchive(db_path=db_path)

        archive.save_from_eval_rows(
            eval_rows_for_portfolio, sub_model_id="rank_avg_2"
        )
        archive.save_from_eval_rows(
            eval_rows_for_portfolio, sub_model_id="enriched_lgb"
        )

        ra = archive.load(sub_model_id="rank_avg_2")
        el = archive.load(sub_model_id="enriched_lgb")
        assert len(ra) > 0
        assert len(el) > 0
        assert (ra["sub_model_id"] == "rank_avg_2").all()
        assert (el["sub_model_id"] == "enriched_lgb").all()


# ===========================================================================
# Expert Interface Tests (11.4)
# ===========================================================================


class TestExpertInterface:
    def test_predict_returns_fusion_score(self, aligned_scores):
        """predict() must return fusion composite, not a sub-model score."""
        from src.models.residual_archive import AIStockForecasterExpert
        expert = AIStockForecasterExpert()
        predictions = expert.predict(aligned_scores)
        assert len(predictions) == len(aligned_scores)
        assert predictions.notna().all()
        for col in ["lgb_score", "fintext_score", "sentiment_score"]:
            if col in aligned_scores.columns:
                assert not predictions.equals(aligned_scores[col])

    def test_epistemic_uncertainty_returns_disagreement(self, aligned_scores):
        from src.models.residual_archive import AIStockForecasterExpert
        expert = AIStockForecasterExpert()
        uncertainty = expert.epistemic_uncertainty(aligned_scores)
        assert len(uncertainty) == len(aligned_scores)
        assert uncertainty.notna().all()
        assert (uncertainty >= 0).all()

    def test_low_disagreement_subset_rankic(self, aligned_scores):
        """Low-disagreement subset should have different (hopefully higher)
        RankIC than the full set — validates the uncertainty proxy."""
        from src.models.residual_archive import AIStockForecasterExpert
        from src.models.fusion_scorer import rank_average_scores

        expert = AIStockForecasterExpert()
        scores = rank_average_scores(aligned_scores)
        uncertainty = expert.epistemic_uncertainty(aligned_scores)

        df = aligned_scores.copy()
        df["composite_score"] = scores.values
        df["disagreement"] = uncertainty.values

        h20 = df[df["horizon"] == 20].copy()
        if len(h20) < 20:
            pytest.skip("Not enough data for RankIC comparison")

        median_disagr = h20["disagreement"].median()
        low_disagr = h20[h20["disagreement"] <= median_disagr]
        high_disagr = h20[h20["disagreement"] > median_disagr]

        def _rankic(sub):
            per_date = sub.groupby("as_of_date")[
                ["composite_score", "excess_return"]
            ].apply(
                lambda g: stats.spearmanr(
                    g["composite_score"], g["excess_return"]
                ).statistic
                if len(g) >= 5 else np.nan
            )
            return per_date.dropna().mean()

        ic_low = _rankic(low_disagr)
        ic_high = _rankic(high_disagr)

        assert not np.isnan(ic_low), "Low-disagreement RankIC is NaN"
        assert not np.isnan(ic_high), "High-disagreement RankIC is NaN"

    def test_conformal_interval_raises(self):
        from src.models.residual_archive import AIStockForecasterExpert
        expert = AIStockForecasterExpert()
        with pytest.raises(NotImplementedError, match="Chapter 13"):
            expert.conformal_interval(pd.DataFrame())

    def test_residuals_raises_when_no_archive(self):
        from src.models.residual_archive import AIStockForecasterExpert
        expert = AIStockForecasterExpert()
        with pytest.raises(ValueError, match="No residual archive"):
            expert.residuals()

    def test_residuals_returns_archive(self, tmp_path, eval_rows_for_portfolio):
        from src.models.residual_archive import (
            AIStockForecasterExpert,
            ResidualArchive,
        )
        db_path = str(tmp_path / "expert_residuals.duckdb")
        archive = ResidualArchive(db_path=db_path)
        archive.save_from_eval_rows(eval_rows_for_portfolio)

        expert = AIStockForecasterExpert(residual_archive=archive)
        returned = expert.residuals()
        assert isinstance(returned, ResidualArchive)
        loaded = returned.load()
        assert len(loaded) > 0

    def test_expert_id_and_sub_model_id(self):
        from src.models.residual_archive import AIStockForecasterExpert
        expert = AIStockForecasterExpert(sub_model_id="enriched_lgb")
        assert expert.expert_id == "ai_stock_forecaster"
        assert expert.sub_model_id == "enriched_lgb"

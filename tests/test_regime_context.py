"""
Tests for Chapter 12.4 — regime_context.parquet validation.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

CONTEXT_PATH = Path("data/regime_context.parquet")


@pytest.mark.skipif(not CONTEXT_PATH.exists(), reason="regime_context.parquet not built")
class TestRegimeContext:

    @pytest.fixture(autouse=True)
    def load_data(self):
        self.df = pd.read_parquet(CONTEXT_PATH)

    def test_required_columns_present(self):
        required_stock = {"date", "stable_id", "ticker", "vol_20d", "vol_60d", "vol_of_vol", "mom_1m", "sector"}
        required_regime = {"vix_percentile_252d", "market_regime", "market_vol_21d"}
        assert required_stock.issubset(set(self.df.columns))
        assert required_regime.issubset(set(self.df.columns))

    def test_row_count(self):
        assert len(self.df) > 100_000

    def test_date_range(self):
        dates = pd.to_datetime(self.df["date"])
        assert dates.min().year <= 2016
        assert dates.max().year >= 2025

    def test_stock_count(self):
        assert self.df["stable_id"].nunique() >= 50

    def test_vol_20d_coverage(self):
        assert self.df["vol_20d"].notna().mean() > 0.95

    def test_vol_20d_realistic_range(self):
        """Annualized vol should be roughly 5%-300%."""
        vol = self.df["vol_20d"].dropna()
        assert vol.median() > 0.05
        assert vol.median() < 3.0
        assert vol.min() > 0

    def test_vix_percentile_coverage(self):
        assert self.df["vix_percentile_252d"].notna().mean() > 0.95

    def test_vix_percentile_range(self):
        vix = self.df["vix_percentile_252d"].dropna()
        assert vix.min() >= 0
        assert vix.max() <= 100

    def test_market_regime_values(self):
        valid = {-1, 0, 1}
        actual = set(self.df["market_regime"].dropna().unique())
        assert actual.issubset(valid)

    def test_no_duplicate_date_stock_pairs(self):
        dups = self.df.duplicated(subset=["date", "stable_id"], keep=False).sum()
        assert dups == 0, f"Found {dups} duplicate (date, stable_id) pairs"

    def test_joinable_to_eval_rows(self):
        """regime_context should be joinable to eval_rows by (date, stable_id)."""
        eval_path = Path(
            "evaluation_outputs/chapter7_tabular_lgb_real/monthly/"
            "baseline_tabular_lgb_monthly/eval_rows.parquet"
        )
        if not eval_path.exists():
            pytest.skip("eval_rows not available")
        er = pd.read_parquet(eval_path)
        er_dates = set(pd.to_datetime(er["as_of_date"]).dt.date)
        ctx_dates = set(pd.to_datetime(self.df["date"]).dt.date)
        overlap = len(er_dates & ctx_dates)
        assert overlap / len(er_dates) > 0.95, (
            f"Only {overlap}/{len(er_dates)} eval dates found in regime_context"
        )

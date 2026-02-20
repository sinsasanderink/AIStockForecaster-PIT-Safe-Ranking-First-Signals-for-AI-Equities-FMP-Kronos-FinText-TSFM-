"""
Residual Archive & Expert Interface — Chapter 11.4
=====================================================

Stores per-fold, per-date, per-ticker residuals from the fusion model
in a format compatible with the multi-expert DEUP system described in
UQ_EXPERT_SELECTION_REFERENCE.md.

Also provides the ExpertInterface scaffold that this project must
satisfy to participate in the broader trading system.
"""

import logging
from dataclasses import dataclass, asdict
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ResidualRecord:
    """
    Single residual record from walk-forward evaluation.

    Matches the format required by the DEUP error predictor g(x).
    """
    expert_id: str
    sub_model_id: str
    fold_id: str
    as_of_date: date
    ticker: str
    horizon: int
    prediction: float
    actual: float
    loss: float


class ResidualArchive:
    """
    Stores and retrieves walk-forward residuals for DEUP training.

    Backed by DuckDB for efficient querying.  A unique constraint on
    (expert_id, sub_model_id, fold_id, as_of_date, ticker, horizon)
    guarantees no duplicate records can exist.
    """

    def __init__(self, db_path: str = "data/residuals.duckdb"):
        self._db_path = db_path
        self._init_db()

    def _init_db(self):
        import duckdb

        conn = duckdb.connect(self._db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS residuals (
                expert_id      VARCHAR,
                sub_model_id   VARCHAR,
                fold_id        VARCHAR,
                as_of_date     DATE,
                ticker         VARCHAR,
                horizon        INTEGER,
                prediction     DOUBLE,
                actual         DOUBLE,
                loss           DOUBLE,
                UNIQUE (expert_id, sub_model_id, fold_id,
                        as_of_date, ticker, horizon)
            )
        """)
        conn.close()

    def save_from_eval_rows(
        self,
        eval_rows: pd.DataFrame,
        expert_id: str = "ai_stock_forecaster",
        sub_model_id: str = "fusion_v1",
    ) -> int:
        """
        Convert evaluation rows to residual records and store.

        Clears previous records for the same (expert_id, sub_model_id)
        before inserting, preventing duplicates across re-runs.

        Args:
            eval_rows: DataFrame with as_of_date, ticker, fold_id, horizon,
                       score (prediction), excess_return (actual)
            expert_id: Expert identifier
            sub_model_id: Sub-model identifier

        Returns:
            Number of records stored
        """
        import duckdb

        required = {"as_of_date", "ticker", "fold_id", "horizon",
                     "score", "excess_return"}
        missing = required - set(eval_rows.columns)
        if missing:
            raise ValueError(f"eval_rows missing columns: {missing}")

        records = eval_rows[list(required)].copy()
        records = records.dropna(subset=["score", "excess_return"])

        if records.empty:
            logger.warning("No valid records to store (all NaN)")
            return 0

        records["expert_id"] = expert_id
        records["sub_model_id"] = sub_model_id
        records["prediction"] = records["score"]
        records["actual"] = records["excess_return"]
        records["loss"] = (records["actual"] - records["prediction"]).abs()

        records["as_of_date"] = pd.to_datetime(records["as_of_date"]).dt.date

        store_df = records[[
            "expert_id", "sub_model_id", "fold_id", "as_of_date",
            "ticker", "horizon", "prediction", "actual", "loss"
        ]]

        conn = duckdb.connect(self._db_path)
        conn.execute(
            "DELETE FROM residuals WHERE expert_id = ? AND sub_model_id = ?",
            [expert_id, sub_model_id],
        )
        conn.execute("INSERT INTO residuals SELECT * FROM store_df")
        conn.close()

        logger.info(
            f"Stored {len(store_df):,} residuals for "
            f"{expert_id}/{sub_model_id}"
        )
        return len(store_df)

    def load(
        self,
        expert_id: str = "ai_stock_forecaster",
        sub_model_id: Optional[str] = None,
        horizon: Optional[int] = None,
        fold_id: Optional[str] = None,
    ) -> pd.DataFrame:
        """Load residuals with optional filtering."""
        import duckdb

        conn = duckdb.connect(self._db_path, read_only=True)
        query = "SELECT * FROM residuals WHERE expert_id = ?"
        params: list = [expert_id]

        if sub_model_id is not None:
            query += " AND sub_model_id = ?"
            params.append(sub_model_id)
        if horizon is not None:
            query += " AND horizon = ?"
            params.append(horizon)
        if fold_id is not None:
            query += " AND fold_id = ?"
            params.append(fold_id)

        df = conn.execute(query, params).df()
        conn.close()
        return df

    def fold_ids(
        self,
        expert_id: str = "ai_stock_forecaster",
        sub_model_id: Optional[str] = None,
    ) -> List[str]:
        """Return distinct fold_ids stored for this expert/model."""
        import duckdb

        conn = duckdb.connect(self._db_path, read_only=True)
        query = "SELECT DISTINCT fold_id FROM residuals WHERE expert_id = ?"
        params: list = [expert_id]
        if sub_model_id is not None:
            query += " AND sub_model_id = ?"
            params.append(sub_model_id)
        query += " ORDER BY fold_id"
        folds = conn.execute(query, params).df()["fold_id"].tolist()
        conn.close()
        return folds

    def summary(self) -> List[Dict]:
        """Return summary statistics of stored residuals."""
        import duckdb

        conn = duckdb.connect(self._db_path, read_only=True)
        try:
            df = conn.execute("""
                SELECT expert_id, sub_model_id,
                       COUNT(*) as n_records,
                       COUNT(DISTINCT fold_id) as n_folds,
                       COUNT(DISTINCT as_of_date) as n_dates,
                       COUNT(DISTINCT ticker) as n_tickers,
                       AVG(loss) as mean_loss,
                       MEDIAN(loss) as median_loss
                FROM residuals
                GROUP BY expert_id, sub_model_id
            """).df()
        except Exception:
            df = pd.DataFrame()
        conn.close()
        return df.to_dict("records") if not df.empty else []


class AIStockForecasterExpert:
    """
    Expert interface scaffold for the multi-expert trading system.

    Implements the contract from UQ_EXPERT_SELECTION_REFERENCE.md:
    - predict(): Fusion model ranking score
    - epistemic_uncertainty(): Sub-model disagreement proxy
    - conformal_interval(): Placeholder for Chapter 13
    - residuals(): Access to walk-forward residual archive

    The system-level UCB controller only needs predict() and
    epistemic_uncertainty() to make decisions.
    """

    EXPERT_ID = "ai_stock_forecaster"

    def __init__(
        self,
        residual_archive: Optional[ResidualArchive] = None,
        sub_model_id: str = "fusion_v1",
    ):
        self._archive = residual_archive
        self._sub_model_id = sub_model_id

    @property
    def expert_id(self) -> str:
        return self.EXPERT_ID

    @property
    def sub_model_id(self) -> str:
        return self._sub_model_id

    def predict(self, scores_df: pd.DataFrame) -> pd.Series:
        """
        Produce ranking scores for a universe of stocks.

        Uses rank-average fusion across available sub-model score columns.
        Returns Series of composite scores (higher = better).
        """
        from src.models.fusion_scorer import rank_average_scores
        return rank_average_scores(scores_df)

    def epistemic_uncertainty(
        self, scores_df: pd.DataFrame
    ) -> pd.Series:
        """
        Compute epistemic uncertainty as sub-model disagreement.

        When LGB, FinText, and Sentiment agree on a stock's rank,
        uncertainty is low. When they disagree, uncertainty is high
        (likely OOD or regime shift).

        This is a KDE-free interim proxy. Full DEUP (with g(x) error
        predictor trained on residuals) replaces this in Chapter 13.

        Returns Series of uncertainty values (higher = more uncertain).
        """
        from src.models.fusion_scorer import compute_disagreement
        return compute_disagreement(scores_df)

    def conformal_interval(
        self, features: pd.DataFrame, alpha: float = 0.10
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Placeholder — implemented in Chapter 13.

        Will use rolling conformal prediction on walk-forward residuals.
        """
        raise NotImplementedError(
            "Conformal intervals are implemented in Chapter 13"
        )

    def residuals(self) -> ResidualArchive:
        """Access walk-forward residual archive for DEUP training."""
        if self._archive is None:
            raise ValueError("No residual archive loaded")
        return self._archive

"""
FinText-TSFM Adapter for Chapter 9
====================================

Adapter for FinText Chronos foundation models (finance-pre-trained on daily
excess returns).

**Critical Design Decisions:**
1. Uses ExcessReturnStore (DuckDB) for input sequences, NOT features_df
2. Year-aware model loading — PIT-safe by construction
3. Batch inference via ChronosPipeline for efficiency
4. Score = median of predicted excess return distribution
5. Deterministic: fixed random seed for reproducibility

**Input:** Daily excess return sequences  (stock_return - QQQ_return)
**Output:** Predicted next-period excess return (ranking score)

**References:**
- Paper: Re(Visiting) TSFMs in Finance (Rahimikia et al., 2025)
- Models: https://huggingface.co/FinText
- Code: https://github.com/DeepIntoStreams/TSFM_Finance
"""

from __future__ import annotations

import gc
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from src.data.excess_return_store import ExcessReturnStore

logger = logging.getLogger(__name__)

# Year range for FinText models on HuggingFace
FINTEXT_MIN_YEAR = 2000
FINTEXT_MAX_YEAR = 2023

# Try to import Chronos
try:
    from chronos import ChronosPipeline

    CHRONOS_AVAILABLE = True
except ImportError:
    logger.warning(
        "chronos-forecasting not available. "
        "Install via: pip install chronos-forecasting"
    )
    CHRONOS_AVAILABLE = False
    ChronosPipeline = None  # type: ignore[assignment,misc]


# ============================================================================
# STUB PREDICTOR (testing without real model)
# ============================================================================

class StubChronosPredictor:
    """
    Stub that mimics ChronosPipeline.predict() for testing.

    Returns deterministic predictions: small positive excess returns
    proportional to the mean of the input sequence.
    """

    def predict(
        self,
        context: torch.Tensor,
        prediction_length: int = 1,
        num_samples: int = 20,
        limit_prediction_length: bool = False,
    ) -> torch.Tensor:
        """
        Returns:
            Tensor of shape ``(batch, num_samples, prediction_length)``.
        """
        batch_size = context.shape[0]
        # Deterministic: small signal based on input mean
        means = context.mean(dim=-1, keepdim=True)  # (batch, 1)
        base = means * 0.05  # damped signal
        noise = torch.randn(batch_size, num_samples, prediction_length) * 0.001
        return base.unsqueeze(-1).expand(-1, num_samples, prediction_length) + noise


# ============================================================================
# FINTEXT ADAPTER
# ============================================================================

@dataclass
class FinTextAdapter:
    """
    Adapter for FinText-TSFM (Chronos) foundation model inference.

    **Key Properties:**
    - PIT-safe: year-aware model selection ensures no future leakage
    - Uses ExcessReturnStore for daily excess return inputs
    - Batch inference for efficiency
    - Caches loaded models to avoid repeated downloads
    - Supports multiple horizon strategies for multi-day predictions

    Attributes:
        excess_return_store: Store for daily excess return sequences.
        model_family: HuggingFace model family prefix.
        model_size: Model size variant (``"Tiny"``, ``"Mini"``, ``"Small"``).
        model_dataset: Dataset variant (``"US"``, ``"Global"``, ``"Augmented"``).
        lookback: Number of trading days for input context window.
        num_samples: Number of distribution samples per prediction.
        device: Device for inference (``"cpu"`` or ``"cuda"``).
        use_stub: Use stub predictor instead of real model.
        horizon_strategy: Multi-day prediction strategy:
            - ``"single_step"`` (default): Always predict 1 day ahead, use for all horizons
            - ``"autoregressive"``: Predict H steps ahead autoregressively
            - ``"cumulative"``: Predict H steps and sum daily returns
    """

    excess_return_store: ExcessReturnStore
    model_family: str = "FinText/Chronos"
    model_size: str = "Small"
    model_dataset: str = "US"
    lookback: int = 21
    num_samples: int = 20
    device: str = "cpu"
    use_stub: bool = False
    horizon_strategy: str = "single_step"
    score_aggregation: str = "median"  # "median", "mean", or "trimmed_mean"

    # Internal: cached model pipelines keyed by model_id
    _model_cache: Dict[str, object] = field(
        init=False, repr=False, default_factory=dict
    )

    # ------------------------------------------------------------------
    # factory
    # ------------------------------------------------------------------
    @classmethod
    def from_pretrained(
        cls,
        db_path: str = "data/features.duckdb",
        model_size: str = "Small",
        model_dataset: str = "US",
        lookback: int = 21,
        num_samples: int = 20,
        device: str = "cpu",
        use_stub: bool = False,
        horizon_strategy: str = "single_step",
        score_aggregation: str = "median",
    ) -> "FinTextAdapter":
        """
        Create a FinTextAdapter with an ExcessReturnStore.

        Args:
            db_path: Path to DuckDB database.
            model_size: ``"Tiny"`` (8M), ``"Mini"`` (20M), ``"Small"`` (46M).
            model_dataset: ``"US"``, ``"Global"``, ``"Augmented"``.
            lookback: Context window in trading days (default 21).
            num_samples: Samples from predictive distribution (default 20).
            device: ``"cpu"`` or ``"cuda"``.
            use_stub: Use stub predictor (no model download).
            horizon_strategy: ``"single_step"``, ``"autoregressive"``, or ``"cumulative"``.
            score_aggregation: ``"median"``, ``"mean"``, or ``"trimmed_mean"``.

        Returns:
            Initialised FinTextAdapter.
        """
        store = ExcessReturnStore(db_path=db_path, enable_cache=True)

        adapter = cls(
            excess_return_store=store,
            model_size=model_size,
            model_dataset=model_dataset,
            lookback=lookback,
            num_samples=num_samples,
            device=device,
            use_stub=use_stub,
            horizon_strategy=horizon_strategy,
            score_aggregation=score_aggregation,
        )
        logger.info(
            f"FinTextAdapter created: size={model_size}, dataset={model_dataset}, "
            f"lookback={lookback}, samples={num_samples}, stub={use_stub}, "
            f"strategy={horizon_strategy}, aggregation={score_aggregation}"
        )
        return adapter

    # ------------------------------------------------------------------
    # model selection (PIT-safe)
    # ------------------------------------------------------------------
    def get_model_id(self, asof_date: pd.Timestamp) -> str:
        """
        Determine which year-specific model to use for a given date.

        Rule: for as-of date in year Y, use model trained through Y-1.
        This guarantees no data after the as-of date was used in training.

        Examples:
            - asof 2024-03-01 → ``FinText/Chronos_Small_2023_US``
            - asof 2020-06-15 → ``FinText/Chronos_Small_2019_US``
            - asof 2016-01-04 → ``FinText/Chronos_Small_2015_US``
        """
        year = asof_date.year - 1
        year = max(year, FINTEXT_MIN_YEAR)
        year = min(year, FINTEXT_MAX_YEAR)
        return f"{self.model_family}_{self.model_size}_{year}_{self.model_dataset}"

    # ------------------------------------------------------------------
    # model loading
    # ------------------------------------------------------------------
    def _get_pipeline(self, model_id: str) -> object:
        """Load or retrieve a cached ChronosPipeline."""
        if model_id in self._model_cache:
            return self._model_cache[model_id]

        if self.use_stub:
            logger.info(f"Using StubChronosPredictor for {model_id}")
            pipeline = StubChronosPredictor()
        else:
            if not CHRONOS_AVAILABLE:
                raise ImportError(
                    "chronos-forecasting not installed. "
                    "Run: pip install chronos-forecasting"
                )
            logger.info(f"Loading model: {model_id} ...")
            start = time.time()
            pipeline = ChronosPipeline.from_pretrained(
                model_id,
                device_map=self.device,
                dtype=torch.float32,
            )
            elapsed = time.time() - start
            logger.info(f"Model loaded in {elapsed:.1f}s: {model_id}")

        self._model_cache[model_id] = pipeline
        return pipeline

    # ------------------------------------------------------------------
    # core scoring
    # ------------------------------------------------------------------
    def score_universe(
        self,
        asof_date: pd.Timestamp,
        tickers: List[str],
        *,
        horizon: int = 1,
        verbose: bool = False,
    ) -> pd.DataFrame:
        """
        Score all tickers for one as-of date.

        Steps:
            1. Determine the PIT-safe model year
            2. Fetch excess return sequences for all tickers
            3. Run batch inference through Chronos
            4. Return score based on horizon_strategy

        Args:
            asof_date: Evaluation date.
            tickers: Stock symbols to score.
            horizon: Forecast horizon in trading days (used for autoregressive/cumulative strategies).
            verbose: Log progress details.

        Returns:
            DataFrame with columns ``[ticker, score, pred_mean, pred_std]``.
            Only includes tickers with sufficient history.
        """
        asof_ts = pd.Timestamp(asof_date)

        # 1. Select model
        model_id = self.get_model_id(asof_ts)
        pipeline = self._get_pipeline(model_id)

        if verbose:
            logger.info(
                f"Scoring {len(tickers)} tickers @ {asof_ts.date()} "
                f"(model={model_id}, lookback={self.lookback}, "
                f"strategy={self.horizon_strategy}, horizon={horizon}d)"
            )

        # 2. Fetch excess return sequences
        valid_tickers, sequences = self.excess_return_store.get_batch_sequences(
            tickers, asof_ts, lookback=self.lookback, strict=True
        )

        if len(valid_tickers) == 0:
            if verbose:
                logger.warning(f"No tickers with sufficient history @ {asof_ts.date()}")
            return pd.DataFrame(columns=["ticker", "score", "pred_mean", "pred_std"])

        if verbose:
            logger.info(
                f"  {len(valid_tickers)}/{len(tickers)} tickers have "
                f"{self.lookback}-day history"
            )

        # 3. Determine prediction length based on strategy
        if self.horizon_strategy == "single_step":
            prediction_length = 1
        elif self.horizon_strategy in ("autoregressive", "cumulative"):
            prediction_length = horizon
        else:
            raise ValueError(
                f"Unknown horizon_strategy: {self.horizon_strategy}. "
                f"Must be 'single_step', 'autoregressive', or 'cumulative'."
            )

        # 4. Run inference
        input_tensor = torch.tensor(sequences, dtype=torch.float32)

        start = time.time()
        with torch.inference_mode():
            # output shape: (batch, num_samples, prediction_length)
            samples = pipeline.predict(
                input_tensor,
                prediction_length=prediction_length,
                num_samples=self.num_samples,
                limit_prediction_length=False,
            )
        elapsed = time.time() - start

        if verbose:
            logger.info(
                f"  Inference: {elapsed:.1f}s for {len(valid_tickers)} tickers "
                f"(pred_length={prediction_length})"
            )

        # 5. Compute scores from samples based on strategy
        samples_np = samples.numpy()  # (batch, num_samples, prediction_length)

        scores = []
        for i, ticker in enumerate(valid_tickers):
            ticker_samples = samples_np[i]  # (num_samples, prediction_length)

            if self.horizon_strategy == "single_step":
                final_samples = ticker_samples[:, 0]
            elif self.horizon_strategy == "autoregressive":
                final_samples = ticker_samples[:, -1]
            elif self.horizon_strategy == "cumulative":
                final_samples = ticker_samples.sum(axis=1)
            else:
                raise ValueError(f"Unknown horizon_strategy: {self.horizon_strategy}")

            score = float(self._aggregate(final_samples))
            pred_mean = float(np.mean(final_samples))
            pred_std = float(np.std(final_samples))

            scores.append(
                {
                    "ticker": ticker,
                    "score": score,
                    "pred_mean": pred_mean,
                    "pred_std": pred_std,
                }
            )

        gc.collect()
        return pd.DataFrame(scores)

    # ------------------------------------------------------------------
    # score aggregation
    # ------------------------------------------------------------------
    def _aggregate(self, samples: np.ndarray) -> float:
        """Aggregate distribution samples into a single score."""
        if self.score_aggregation == "median":
            return float(np.median(samples))
        elif self.score_aggregation == "mean":
            return float(np.mean(samples))
        elif self.score_aggregation == "trimmed_mean":
            from scipy.stats import trim_mean
            return float(trim_mean(samples, proportiontocut=0.1))
        else:
            raise ValueError(
                f"Unknown score_aggregation: {self.score_aggregation}. "
                f"Must be 'median', 'mean', or 'trimmed_mean'."
            )

    # ------------------------------------------------------------------
    # cleanup
    # ------------------------------------------------------------------
    def close(self) -> None:
        """Release resources."""
        self._model_cache.clear()
        self.excess_return_store.close()
        gc.collect()

    def __enter__(self) -> "FinTextAdapter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


# ============================================================================
# SCORING FUNCTION FOR run_experiment() INTEGRATION
# ============================================================================

_fintext_adapter: Optional[FinTextAdapter] = None


def initialize_fintext_adapter(
    db_path: str = "data/features.duckdb",
    model_size: str = "Small",
    model_dataset: str = "US",
    lookback: int = 21,
    num_samples: int = 20,
    device: str = "cpu",
    use_stub: bool = False,
    horizon_strategy: str = "single_step",
    score_aggregation: str = "median",
) -> FinTextAdapter:
    """
    Initialise the global FinText adapter (call once before evaluation).
    """
    global _fintext_adapter

    if _fintext_adapter is not None:
        logger.info("FinText adapter already initialised")
        return _fintext_adapter

    logger.info("Initialising FinText adapter …")
    _fintext_adapter = FinTextAdapter.from_pretrained(
        db_path=db_path,
        model_size=model_size,
        model_dataset=model_dataset,
        lookback=lookback,
        num_samples=num_samples,
        device=device,
        use_stub=use_stub,
        horizon_strategy=horizon_strategy,
        score_aggregation=score_aggregation,
    )
    return _fintext_adapter


def _apply_ema_smoothing(
    result: pd.DataFrame,
    halflife_days: int = 5,
) -> pd.DataFrame:
    """
    Apply exponential moving average (EMA) smoothing to scores.

    This reduces day-to-day ranking churn by blending today's raw prediction
    with recent history.  The half-life controls the decay — a 5-day half-life
    means the previous week's predictions still carry ~50% weight.

    Args:
        result: DataFrame with columns ``[as_of_date, ticker, score, ...]``.
        halflife_days: EMA half-life in trading days (default 5).

    Returns:
        DataFrame with ``score`` column replaced by smoothed values.
    """
    if halflife_days <= 0:
        return result

    alpha = 1 - np.exp(-np.log(2) / halflife_days)

    result = result.sort_values(["ticker", "as_of_date"]).copy()
    result["score"] = (
        result.groupby("ticker")["score"]
        .transform(lambda s: s.ewm(alpha=alpha, adjust=False).mean())
    )

    return result


def fintext_scoring_function(
    features_df: pd.DataFrame,
    fold_id: str,
    horizon: int,
) -> pd.DataFrame:
    """
    FinText scoring function compatible with ``run_experiment(scorer_fn=...)``.

    Matches the EvaluationRow contract:
        as_of_date, ticker, stable_id, horizon, fold_id, score, excess_return

    Includes EMA score smoothing (half-life = 5 trading days) to reduce
    day-to-day ranking churn while preserving cross-sectional signal.

    Args:
        features_df: Validation-window features (contains date, ticker,
            stable_id, excess_return columns).
        fold_id: Walk-forward fold identifier.
        horizon: Forecast horizon in trading days (20, 60, 90).

    Returns:
        DataFrame with EvaluationRow-compatible columns.
    """
    if _fintext_adapter is None:
        raise RuntimeError(
            "FinText adapter not initialised. "
            "Call initialize_fintext_adapter() first."
        )

    logger.info(
        f"FinText scoring: fold={fold_id}, horizon={horizon}d, "
        f"rows={len(features_df)}"
    )

    unique_dates = sorted(features_df["date"].unique())
    logger.info(f"  Scoring {len(unique_dates)} unique dates")

    all_scores: List[pd.DataFrame] = []

    for i, asof_date in enumerate(unique_dates):
        date_df = features_df[features_df["date"] == asof_date]
        tickers = date_df["ticker"].unique().tolist()

        logger.info(
            f"  [{i + 1}/{len(unique_dates)}] {asof_date}: {len(tickers)} tickers"
        )

        scores_df = _fintext_adapter.score_universe(
            asof_date=pd.Timestamp(asof_date),
            tickers=tickers,
            horizon=horizon,
            verbose=False,
        )

        if scores_df.empty:
            logger.warning(f"  No scores for {asof_date}")
            continue

        # Determine excess return column
        excess_return_col = f"excess_return_{horizon}d"
        if excess_return_col not in date_df.columns:
            excess_return_col = "excess_return"

        # Merge scores with evaluation metadata
        merge_cols = ["date", "ticker", "stable_id", excess_return_col]
        optional_cols = [
            "adv_20d",
            "adv_60d",
            "sector",
            "vix_percentile",
            "market_return_5d",
            "beta_252d",
        ]
        for col in optional_cols:
            if col in date_df.columns:
                merge_cols.append(col)

        merged = date_df[merge_cols].merge(
            scores_df[["ticker", "score"]], on="ticker", how="inner"
        )

        merged = merged.rename(
            columns={"date": "as_of_date", excess_return_col: "excess_return"}
        )
        merged["fold_id"] = fold_id
        merged["horizon"] = horizon

        all_scores.append(merged)

    if not all_scores:
        raise ValueError(
            f"No scores generated for fold {fold_id}, horizon {horizon}d"
        )

    result = pd.concat(all_scores, ignore_index=True)

    # Apply EMA smoothing to reduce churn (half-life = 5 trading days)
    result = _apply_ema_smoothing(result, halflife_days=5)

    logger.info(f"Generated {len(result)} evaluation rows (EMA smoothed)")
    return result

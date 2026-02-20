"""
Kronos Adapter for Chapter 8
=============================

Adapter for Kronos foundation model (K-line/OHLCV price dynamics prediction).

**Critical Design Decisions:**
1. Uses PricesStore (DuckDB prices table) for OHLCV history, NOT fold-filtered features_df
2. Uses global trading calendar from DuckDB for future timestamps, NO freq="B"
3. Uses batch inference (predict_batch) for efficiency
4. Deterministic inference: T=0.0, top_p=1.0, sample_count=1
5. Score = (pred_close - current_close) / current_close (price return proxy)

**References:**
- Kronos GitHub: https://github.com/shiyu-coder/Kronos
- HF Models: NeoQuasar/Kronos-Tokenizer-base, NeoQuasar/Kronos-base
"""

from __future__ import annotations

# =============================================================================
# CRITICAL: Disable MPS globally BEFORE any other torch imports
# This MUST be at the very top of the file, before importing Kronos
# =============================================================================
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Also disable CUDA to force pure CPU

import logging
import signal
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

# Force CPU as default
torch.set_default_device('cpu')

from src.data import PricesStore, load_global_trading_calendar

logger = logging.getLogger(__name__)

# Try to import Kronos (may not be installed yet)
try:
    from model import Kronos, KronosTokenizer, KronosPredictor
    KRONOS_AVAILABLE = True

except ImportError:
    logger.warning(
        "Kronos not available. Install from https://github.com/shiyu-coder/Kronos"
    )
    KRONOS_AVAILABLE = False
    # Stub classes for type hints
    Kronos = None
    KronosTokenizer = None
    KronosPredictor = None


# =============================================================================
# TIMEOUT UTILITIES
# =============================================================================

class TimeoutException(Exception):
    """Raised when a function times out."""
    pass


@contextmanager
def timeout_handler(seconds: int, ticker_info: str = ""):
    """
    Context manager that raises TimeoutException after `seconds`.
    Works on Unix-like systems (macOS, Linux).
    
    Usage:
        with timeout_handler(30, "AAPL"):
            slow_function()
    """
    def _timeout_signal_handler(signum, frame):
        raise TimeoutException(f"Timeout after {seconds}s processing {ticker_info}")
    
    # Set the signal handler
    old_handler = signal.signal(signal.SIGALRM, _timeout_signal_handler)
    signal.alarm(seconds)
    
    try:
        yield
    finally:
        # Cancel the alarm and restore old handler
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


class StubPredictor:
    """
    Stub predictor for testing without Kronos installed.
    
    Generates deterministic predictions: +2% return for all tickers.
    """
    
    def predict_batch(
        self,
        df_list: List[pd.DataFrame],
        x_timestamp_list: List[pd.DatetimeIndex],
        y_timestamp_list: List[pd.DatetimeIndex],
        pred_len: int,
        T: float = 0.0,
        top_p: float = 1.0,
        sample_count: int = 1,
        verbose: bool = False,
    ) -> List[pd.DataFrame]:
        """
        Stub predict_batch that returns +2% close predictions.
        
        Returns:
            List of predicted OHLCV DataFrames (one per input)
        """
        pred_list = []
        
        for df, x_ts, y_ts in zip(df_list, x_timestamp_list, y_timestamp_list):
            last_close = df["close"].iloc[-1]
            
            # Deterministic +2% return
            pred_close = last_close * 1.02
            
            # Create prediction DataFrame
            pred_df = pd.DataFrame({
                "open": [last_close * 1.01],
                "high": [last_close * 1.03],
                "low": [last_close * 1.00],
                "close": [pred_close],
                "volume": [df["volume"].iloc[-1]],
            })
            
            # Use first future timestamp
            pred_df.index = y_ts[:1] if len(y_ts) > 0 else pd.DatetimeIndex([])
            
            pred_list.append(pred_df)
        
        return pred_list
    
    def predict_single(
        self,
        df: pd.DataFrame,
        x_timestamp: pd.Series,
        y_timestamp: pd.Series,
        pred_len: int,
        T: float = 0.0,
        top_p: float = 1.0,
        sample_count: int = 1,
    ) -> pd.DataFrame:
        """Single-ticker prediction for timeout-safe processing."""
        return self.predict_batch(
            [df], [x_timestamp], [y_timestamp], pred_len, T, top_p, sample_count
        )[0]


class TimeoutSafePredictor:
    """
    Wrapper around KronosPredictor that adds per-ticker timeout protection.
    
    Instead of batch prediction (where one bad ticker blocks all),
    processes tickers one-by-one with timeout protection.
    """
    
    def __init__(self, predictor, timeout_seconds: int = 60):
        """
        Args:
            predictor: The underlying KronosPredictor instance
            timeout_seconds: Max seconds per ticker before skipping
        """
        self._predictor = predictor
        self.timeout_seconds = timeout_seconds
        self.timed_out_tickers = []  # Track which tickers timeout for debugging
    
    def predict_single_safe(
        self,
        ticker: str,
        df: pd.DataFrame,
        x_timestamp: pd.Series,
        y_timestamp: pd.Series,
        pred_len: int,
        T: float = 0.0,
        top_p: float = 1.0,
        sample_count: int = 1,
    ) -> Optional[pd.DataFrame]:
        """
        Predict for a single ticker with timeout protection.
        
        Returns:
            Prediction DataFrame or None if timeout/error
        """
        try:
            with timeout_handler(self.timeout_seconds, ticker):
                start_time = time.time()
                
                # Call predict_batch with single item
                result = self._predictor.predict_batch(
                    df_list=[df],
                    x_timestamp_list=[x_timestamp],
                    y_timestamp_list=[y_timestamp],
                    pred_len=pred_len,
                    T=T,
                    top_p=top_p,
                    sample_count=sample_count,
                    verbose=False,
                )
                
                elapsed = time.time() - start_time
                if elapsed > 30:  # Warn if slow but not timed out
                    logger.warning(f"  ⚠️  {ticker} took {elapsed:.1f}s (slow but completed)")
                
                return result[0] if result else None
                
        except TimeoutException as e:
            logger.error(f"  ⏱️  TIMEOUT: {ticker} exceeded {self.timeout_seconds}s - SKIPPING")
            self.timed_out_tickers.append(ticker)
            return None
        except Exception as e:
            logger.error(f"  ❌ ERROR on {ticker}: {e}")
            return None
    
    def predict_batch_safe(
        self,
        tickers: List[str],
        df_list: List[pd.DataFrame],
        x_timestamp_list: List[pd.Series],
        y_timestamp_list: List[pd.Series],
        pred_len: int,
        T: float = 0.0,
        top_p: float = 1.0,
        sample_count: int = 1,
        verbose: bool = False,
    ) -> Tuple[List[str], List[pd.DataFrame]]:
        """
        Process multiple tickers one-by-one with timeout protection.
        
        Returns:
            Tuple of (successful_tickers, predictions) - only includes tickers that completed
        """
        successful_tickers = []
        predictions = []
        
        for i, (ticker, df, x_ts, y_ts) in enumerate(
            zip(tickers, df_list, x_timestamp_list, y_timestamp_list)
        ):
            if verbose:
                logger.info(f"    Processing {i+1}/{len(tickers)}: {ticker}...")
            
            pred = self.predict_single_safe(
                ticker=ticker,
                df=df,
                x_timestamp=x_ts,
                y_timestamp=y_ts,
                pred_len=pred_len,
                T=T,
                top_p=top_p,
                sample_count=sample_count,
            )
            
            if pred is not None:
                successful_tickers.append(ticker)
                predictions.append(pred)
        
        return successful_tickers, predictions


@dataclass
class KronosAdapter:
    """
    Adapter for Kronos foundation model inference.
    
    **Critical Properties:**
    - PIT-safe: Only uses data available at asof_date
    - Uses PricesStore (DuckDB) for OHLCV, NOT features_df
    - Uses global trading calendar for future timestamps
    - Batch inference for efficiency
    
    **Attributes:**
        prices_store: PricesStore instance for OHLCV fetching
        trading_calendar: Global trading calendar (pd.DatetimeIndex)
        predictor: Kronos predictor instance (if Kronos available)
        lookback: Number of trading days for input sequence (default: 252)
        device: Device for inference ("cpu" or "cuda")
        deterministic: If True, use T=0.0, top_p=1.0, sample_count=1
        batch_size: Max tickers per Kronos call (memory optimization)
        per_ticker_timeout: Seconds before skipping a slow ticker
    """
    
    prices_store: PricesStore
    trading_calendar: pd.DatetimeIndex
    predictor: Optional[object] = None  # KronosPredictor or TimeoutSafePredictor
    lookback: int = 252
    device: str = "cpu"
    deterministic: bool = True
    batch_size: int = 4  # Reduced for debugging - process fewer at once
    per_ticker_timeout: int = 60  # Seconds before skipping a ticker
    
    @classmethod
    def from_pretrained(
        cls,
        db_path: str = "data/features.duckdb",
        tokenizer_id: str = "NeoQuasar/Kronos-Tokenizer-base",
        model_id: str = "NeoQuasar/Kronos-base",
        max_context: int = 512,
        lookback: int = 252,
        device: str = "cpu",
        deterministic: bool = True,
        use_stub: bool = False,
        batch_size: int = 4,
        per_ticker_timeout: int = 60,
    ) -> "KronosAdapter":
        """
        Load Kronos model from HuggingFace and create adapter.
        
        Args:
            db_path: Path to DuckDB database
            tokenizer_id: HuggingFace tokenizer ID
            model_id: HuggingFace model ID
            max_context: Maximum context length for predictor
            lookback: Number of trading days for input sequence
            device: Device for inference
            deterministic: If True, use deterministic inference settings
            use_stub: If True, use StubPredictor (for testing without Kronos)
            batch_size: Max tickers per Kronos call
            per_ticker_timeout: Seconds before skipping a slow ticker
        
        Returns:
            KronosAdapter instance
        """
        # Load trading calendar and prices store
        trading_calendar = load_global_trading_calendar(db_path)
        prices_store = PricesStore(db_path=db_path, enable_cache=True)
        
        # Use stub predictor if requested or if Kronos not available
        if use_stub:
            logger.info("Using StubPredictor (deterministic +2% return)")
            predictor = StubPredictor()
            
            return cls(
                prices_store=prices_store,
                trading_calendar=trading_calendar,
                predictor=predictor,
                lookback=lookback,
                device=device,
                deterministic=deterministic,
                batch_size=batch_size,
                per_ticker_timeout=per_ticker_timeout,
            )
        
        # Try to load real Kronos
        if not KRONOS_AVAILABLE:
            raise ImportError(
                "Kronos not available. Install from https://github.com/shiyu-coder/Kronos "
                "or use use_stub=True for testing"
            )
        
        # FORCE CPU
        device = "cpu"
        
        logger.info(f"Loading Kronos from HuggingFace...")
        logger.info(f"  Tokenizer: {tokenizer_id}")
        logger.info(f"  Model: {model_id}")
        logger.info(f"  Device: {device}")
        logger.info(f"  Per-ticker timeout: {per_ticker_timeout}s")
        
        try:
            # Load tokenizer and model
            tokenizer = KronosTokenizer.from_pretrained(tokenizer_id)
            model = Kronos.from_pretrained(model_id)
            
            # Move to CPU
            model = model.to('cpu')
            
            # CRITICAL: Set to evaluation mode
            model.eval()
            
            # Freeze parameters
            for param in model.parameters():
                param.requires_grad_(False)
            
            # Create base predictor
            base_predictor = KronosPredictor(
                model=model,
                tokenizer=tokenizer,
                max_context=max_context
            )
            
            # Wrap with timeout protection
            predictor = TimeoutSafePredictor(
                predictor=base_predictor,
                timeout_seconds=per_ticker_timeout
            )
            
            logger.info("✓ Kronos loaded successfully")
            logger.info(f"  Model mode: eval (dropout disabled)")
            logger.info(f"  Gradients: disabled (inference-only)")
            logger.info(f"  Timeout protection: {per_ticker_timeout}s per ticker")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load Kronos: {e}")
        
        return cls(
            prices_store=prices_store,
            trading_calendar=trading_calendar,
            predictor=predictor,
            lookback=lookback,
            device=device,
            deterministic=deterministic,
            batch_size=batch_size,
            per_ticker_timeout=per_ticker_timeout,
        )
    
    def get_future_dates(
        self,
        last_x_date: pd.Timestamp,
        horizon: int
    ) -> pd.DatetimeIndex:
        """
        Get future trading dates using global trading calendar.
        """
        last_x_date = pd.Timestamp(last_x_date)
        idx = np.searchsorted(
            self.trading_calendar.values,
            last_x_date.to_datetime64()
        )
        future_dates = self.trading_calendar[idx + 1 : idx + 1 + horizon]
        return pd.DatetimeIndex(future_dates)
    
    def score_universe_batch(
        self,
        asof_date: pd.Timestamp,
        tickers: List[str],
        horizon: int,
        *,
        verbose: bool = False,
    ) -> pd.DataFrame:
        """
        Score multiple tickers for one asof_date using INDIVIDUAL inference with timeout.
        
        **Key Change:** Now processes tickers ONE AT A TIME with timeout protection,
        so one slow/hanging ticker doesn't block the entire batch.
        """
        if self.predictor is None:
            raise RuntimeError("Kronos predictor not loaded. Use from_pretrained() first.")
        
        if verbose:
            logger.info(f"Scoring {len(tickers)} tickers @ {asof_date} (horizon={horizon}d)")
        
        # Prepare batch data
        df_list = []
        x_ts_list = []
        valid_tickers = []
        spot_closes = []
        
        for ticker in tickers:
            # Fetch OHLCV from PricesStore (DuckDB)
            ohlcv = self.prices_store.fetch_ohlcv(
                ticker=ticker,
                asof_date=asof_date,
                lookback=self.lookback,
                strict_lookback=True,
                fill_missing=True
            )
            
            if len(ohlcv) != self.lookback:
                if verbose:
                    logger.debug(f"Skipping {ticker}: insufficient history ({len(ohlcv)} < {self.lookback})")
                continue
            
            df_list.append(ohlcv)
            x_ts_list.append(ohlcv.index)
            valid_tickers.append(ticker)
            spot_closes.append(ohlcv["close"].iloc[-1])
        
        if len(df_list) == 0:
            if verbose:
                logger.warning(f"No tickers with sufficient history @ {asof_date}")
            return pd.DataFrame(columns=["ticker", "score", "pred_close", "spot_close"])
        
        # Get per-ticker future timestamps
        last_x_list = [pd.Timestamp(ts[-1]) for ts in x_ts_list]
        y_ts_list = []
        
        for last_x in last_x_list:
            y_ts = self.get_future_dates(last_x, horizon)
            if len(y_ts) < horizon:
                logger.warning(
                    f"Calendar only has {len(y_ts)} future dates (need {horizon}) from {last_x}."
                )
            y_ts_list.append(y_ts)
        
        # Inference settings
        if self.deterministic:
            temperature = 0.0
            top_p = 1.0
            sample_count = 1
        else:
            temperature = 0.5
            top_p = 0.9
            sample_count = 3
        
        if verbose:
            logger.info(f"Running inference for {len(df_list)} tickers (one-by-one with {self.per_ticker_timeout}s timeout)...")
        
        # Convert timestamps to Series
        def _to_series(ts):
            if isinstance(ts, pd.Series):
                return ts
            elif isinstance(ts, pd.DatetimeIndex):
                return pd.Series(ts.values, index=range(len(ts)))
            else:
                return pd.Series(pd.to_datetime(ts))
        
        x_ts_list_converted = [_to_series(ts) for ts in x_ts_list]
        y_ts_list_converted = [_to_series(ts) for ts in y_ts_list]
        
        # =====================================================================
        # KEY CHANGE: Process ONE TICKER AT A TIME with timeout protection
        # =====================================================================
        
        import gc
        
        successful_tickers = []
        pred_list = []
        successful_spot_closes = []
        successful_last_x = []
        successful_df_lengths = []
        
        for i, (ticker, df, x_ts, y_ts, spot_close, last_x) in enumerate(
            zip(valid_tickers, df_list, x_ts_list_converted, y_ts_list_converted, spot_closes, last_x_list)
        ):
            logger.info(f"  [{i+1}/{len(valid_tickers)}] Processing {ticker}...")
            start_time = time.time()
            
            try:
                with timeout_handler(self.per_ticker_timeout, ticker):
                    with torch.inference_mode():
                        # Check if predictor is TimeoutSafePredictor or regular
                        if hasattr(self.predictor, 'predict_single_safe'):
                            pred = self.predictor.predict_single_safe(
                                ticker=ticker,
                                df=df,
                                x_timestamp=x_ts,
                                y_timestamp=y_ts,
                                pred_len=horizon,
                                T=temperature,
                                top_p=top_p,
                                sample_count=sample_count,
                            )
                        elif hasattr(self.predictor, '_predictor'):
                            # It's TimeoutSafePredictor but use inner predictor
                            pred_result = self.predictor._predictor.predict_batch(
                                df_list=[df],
                                x_timestamp_list=[x_ts],
                                y_timestamp_list=[y_ts],
                                pred_len=horizon,
                                T=temperature,
                                top_p=top_p,
                                sample_count=sample_count,
                                verbose=False,
                            )
                            pred = pred_result[0] if pred_result else None
                        else:
                            # Fallback to predict_batch
                            pred_result = self.predictor.predict_batch(
                                df_list=[df],
                                x_timestamp_list=[x_ts],
                                y_timestamp_list=[y_ts],
                                pred_len=horizon,
                                T=temperature,
                                top_p=top_p,
                                sample_count=sample_count,
                                verbose=False,
                            )
                            pred = pred_result[0] if pred_result else None
                
                elapsed = time.time() - start_time
                
                if pred is not None:
                    successful_tickers.append(ticker)
                    pred_list.append(pred)
                    successful_spot_closes.append(spot_close)
                    successful_last_x.append(last_x)
                    successful_df_lengths.append(len(df))
                    logger.info(f"  [{i+1}/{len(valid_tickers)}] ✓ {ticker} completed in {elapsed:.1f}s")
                else:
                    logger.warning(f"  [{i+1}/{len(valid_tickers)}] ✗ {ticker} returned None")
                    
            except TimeoutException:
                logger.error(f"  [{i+1}/{len(valid_tickers)}] ⏱️ TIMEOUT: {ticker} exceeded {self.per_ticker_timeout}s - SKIPPED")
            except Exception as e:
                logger.error(f"  [{i+1}/{len(valid_tickers)}] ❌ ERROR on {ticker}: {e}")
            
            # Memory cleanup after each ticker
            gc.collect()
        
        logger.info(f"  Completed: {len(successful_tickers)}/{len(valid_tickers)} tickers")
        
        if not pred_list:
            logger.warning(f"No successful predictions @ {asof_date}")
            return pd.DataFrame(columns=["ticker", "score", "pred_close", "spot_close"])
        
        # Compute scores
        scores = []
        for i, (ticker, pred_df, spot_close, last_x, n_hist) in enumerate(
            zip(successful_tickers, pred_list, successful_spot_closes, successful_last_x, successful_df_lengths)
        ):
            try:
                pred_close = pred_df["close"].iloc[-1]
                score = (pred_close - spot_close) / spot_close
                
                scores.append({
                    "ticker": ticker,
                    "score": score,
                    "pred_close": pred_close,
                    "spot_close": spot_close,
                    "last_x_date": last_x,
                    "n_history": n_hist,
                })
            except Exception as e:
                logger.warning(f"Failed to compute score for {ticker}: {e}")
                continue
        
        return pd.DataFrame(scores)


# ============================================================================
# SCORING FUNCTION FOR run_experiment() INTEGRATION
# ============================================================================

_kronos_adapter: Optional[KronosAdapter] = None


def initialize_kronos_adapter(
    db_path: str = "data/features.duckdb",
    tokenizer_id: str = "NeoQuasar/Kronos-Tokenizer-base",
    model_id: str = "NeoQuasar/Kronos-base",
    max_context: int = 512,
    lookback: int = 252,
    device: str = "cpu",
    deterministic: bool = True,
    per_ticker_timeout: int = 60,
) -> KronosAdapter:
    """
    Initialize Kronos adapter (call once at start of evaluation).
    """
    global _kronos_adapter
    
    if _kronos_adapter is not None:
        logger.info("Kronos adapter already initialized")
        return _kronos_adapter
    
    logger.info("Initializing Kronos adapter...")
    _kronos_adapter = KronosAdapter.from_pretrained(
        db_path=db_path,
        tokenizer_id=tokenizer_id,
        model_id=model_id,
        max_context=max_context,
        lookback=lookback,
        device=device,
        deterministic=deterministic,
        per_ticker_timeout=per_ticker_timeout,
    )
    logger.info("✓ Kronos adapter initialized")
    
    return _kronos_adapter


def kronos_scoring_function(
    features_df: pd.DataFrame,
    fold_id: str,
    horizon: int,
) -> pd.DataFrame:
    """
    Kronos scoring function for run_experiment() integration.
    """
    if _kronos_adapter is None:
        raise RuntimeError(
            "Kronos adapter not initialized. Call initialize_kronos_adapter() first."
        )
    
    logger.info(f"Kronos scoring: fold={fold_id}, horizon={horizon}d, rows={len(features_df)}")
    
    unique_dates = sorted(features_df["date"].unique())
    logger.info(f"  Scoring {len(unique_dates)} unique dates")
    
    all_scores = []
    
    for i, asof_date in enumerate(unique_dates):
        date_df = features_df[features_df["date"] == asof_date]
        tickers = date_df["ticker"].unique().tolist()
        
        logger.info(f"  [{i+1}/{len(unique_dates)}] {asof_date}: {len(tickers)} tickers")
        
        scores_df = _kronos_adapter.score_universe_batch(
            asof_date=pd.Timestamp(asof_date),
            tickers=tickers,
            horizon=horizon,
            verbose=True,  # Enable verbose for debugging
        )
        
        if scores_df.empty:
            logger.warning(f"  No scores for {asof_date} (insufficient history or all timed out)")
            continue
        
        excess_return_col = f"excess_return_{horizon}d"
        if excess_return_col not in date_df.columns:
            excess_return_col = "excess_return"
        
        merge_cols = ["date", "ticker", "stable_id", excess_return_col]
        
        optional_cols = ["adv_20d", "adv_60d", "sector", "vix_percentile", "market_return_5d", "beta_252d"]
        for col in optional_cols:
            if col in date_df.columns:
                merge_cols.append(col)
        
        merged = date_df[merge_cols].merge(
            scores_df[["ticker", "score"]],
            on="ticker",
            how="inner"
        )
        
        merged = merged.rename(columns={
            "date": "as_of_date",
            excess_return_col: "excess_return",
        })
        
        merged["fold_id"] = fold_id
        merged["horizon"] = horizon
        
        all_scores.append(merged)
    
    if not all_scores:
        raise ValueError(f"No scores generated for fold {fold_id}, horizon {horizon}d")
    
    result = pd.concat(all_scores, ignore_index=True)
    logger.info(f"✓ Generated {len(result)} evaluation rows")
    
    return result
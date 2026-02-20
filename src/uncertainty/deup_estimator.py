"""
DEUP Error Predictor g(x) — Chapter 13.1
==========================================

Trains a secondary LightGBM model that predicts how wrong the primary
model's RANKING will be at a given input x.

g(x) is trained walk-forward on held-out residuals:
  - For fold k (k >= min_train_folds): train on folds 1..k-1, predict fold k
  - Target: rank_loss = |rank_pct(ER) - rank_pct(score)| per stock per date
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

G_FEATURES = [
    "score",
    "abs_score",
    "vol_20d",
    "vol_60d",
    "mom_1m",
    "adv_20d",
    "vix_percentile_252d",
    "market_regime_enc",
    "market_vol_21d",
    "market_return_21d",
    "cross_sectional_rank",
]

G_PARAMS = {
    "objective": "regression",
    "metric": "mae",
    "n_estimators": 50,
    "max_depth": 3,
    "num_leaves": 8,
    "min_child_samples": 50,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "verbose": -1,
    "n_jobs": -1,
    "random_state": 42,
}


def prepare_g_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features for g(x) from the enriched residual DataFrame."""
    out = df.copy()

    if "score" in out.columns:
        out["abs_score"] = out["score"].abs()

    if "score" in out.columns:
        out["cross_sectional_rank"] = out.groupby("as_of_date")["score"].rank(pct=True)

    if "market_regime" in out.columns:
        regime_map = {"bull": 1, "1": 1, "neutral": 0, "0": 0, "bear": -1, "-1": -1}
        out["market_regime_enc"] = (
            out["market_regime"].astype(str).map(regime_map).fillna(0).astype(float)
        )
    elif "market_regime_enc" not in out.columns:
        out["market_regime_enc"] = 0.0

    return out


def _available_features(df: pd.DataFrame) -> List[str]:
    """Return the subset of G_FEATURES actually present and non-null."""
    present = []
    for f in G_FEATURES:
        if f in df.columns and df[f].notna().mean() > 0.5:
            present.append(f)
    return present


def train_g_walk_forward(
    enriched: pd.DataFrame,
    target_col: str = "rank_loss",
    min_train_folds: int = 20,
    horizons: Optional[List[int]] = None,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Walk-forward training of g(x) on enriched residuals.

    For each horizon, for each fold k >= min_train_folds:
      - Train g(x) on all rows from folds < k
      - Predict fold k

    Returns:
        predictions_df: DataFrame with columns [as_of_date, ticker, horizon,
                        fold_id, g_pred, rank_loss, ...]
        diagnostics: dict with per-horizon summary stats
    """
    if horizons is None:
        horizons = sorted(enriched["horizon"].unique())

    enriched = prepare_g_features(enriched)
    feats = _available_features(enriched)
    logger.info(f"g(x) features ({len(feats)}): {feats}")

    if not feats:
        raise ValueError("No g(x) features available after filtering")

    all_folds = sorted(enriched["fold_id"].unique())
    predict_folds = all_folds[min_train_folds:]
    logger.info(
        f"Walk-forward: {len(all_folds)} total folds, "
        f"predicting folds {predict_folds[0]}..{predict_folds[-1]} "
        f"({len(predict_folds)} folds)"
    )

    results = []
    diagnostics = {}
    feature_importances_all = []

    for hz in horizons:
        hz_data = enriched[enriched["horizon"] == hz].copy()
        hz_preds = []

        for fold_idx, fold_id in enumerate(predict_folds):
            train_folds = set(all_folds[:min_train_folds + fold_idx])
            train = hz_data[hz_data["fold_id"].isin(train_folds)]
            test = hz_data[hz_data["fold_id"] == fold_id]

            if len(test) == 0:
                continue

            X_train = train[feats].fillna(0)
            y_train = train[target_col].fillna(0)
            X_test = test[feats].fillna(0)

            model = lgb.LGBMRegressor(**G_PARAMS)
            model.fit(X_train, y_train)

            preds = model.predict(X_test)
            preds = np.clip(preds, 0, None)

            out = test[["as_of_date", "ticker", "stable_id", "horizon",
                         "fold_id", target_col]].copy()
            out["g_pred"] = preds
            hz_preds.append(out)

            if fold_idx == len(predict_folds) - 1:
                fi = pd.Series(
                    model.feature_importances_, index=feats
                ).sort_values(ascending=False)
                feature_importances_all.append(
                    {"horizon": hz, "fold_id": fold_id, "importances": fi.to_dict()}
                )

        if hz_preds:
            hz_df = pd.concat(hz_preds, ignore_index=True)
            rho = stats.spearmanr(
                hz_df["g_pred"], hz_df[target_col]
            ).statistic
            diagnostics[hz] = {
                "n_rows": len(hz_df),
                "n_folds": hz_df["fold_id"].nunique(),
                "g_mean": float(hz_df["g_pred"].mean()),
                "g_std": float(hz_df["g_pred"].std()),
                "target_mean": float(hz_df[target_col].mean()),
                "spearman_rho": float(rho),
                "pct_positive": float((hz_df["g_pred"] > 0).mean()),
            }
            results.append(hz_df)
            logger.info(
                f"  {hz}d: {len(hz_df):,} predictions, "
                f"rho(g, {target_col})={rho:.4f}"
            )

    predictions_df = pd.concat(results, ignore_index=True) if results else pd.DataFrame()
    diagnostics["features"] = feats
    diagnostics["feature_importances"] = feature_importances_all
    return predictions_df, diagnostics

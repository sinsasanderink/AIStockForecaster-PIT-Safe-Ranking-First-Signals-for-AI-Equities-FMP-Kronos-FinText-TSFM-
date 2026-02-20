"""
Fusion Scorer — Chapter 11.2
==============================

Three fusion approaches for combining sub-model scores:

Approach A: Rank-Average Fusion (parameter-free)
    Rank-normalize each sub-model score per date, take mean of available ranks.

Approach B: Enriched Tabular (feature-level)
    LGB trained on 13 tabular + 9 sentiment = 22 features.

Approach C: Learned Score Stacking
    Meta-learner (Ridge or small LGB) trained on rank-normalized sub-model scores.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

SCORE_MODELS = ["lgb_score", "fintext_score", "sentiment_score"]


# ---------------------------------------------------------------------------
# Approach A: Rank-Average Fusion
# ---------------------------------------------------------------------------


def rank_average_scores(
    eval_df: pd.DataFrame,
    score_cols: List[str] = None,
) -> pd.Series:
    """
    Rank-normalize each sub-model score cross-sectionally per date,
    then average available ranks. Parameter-free, cannot overfit.

    Args:
        eval_df: DataFrame with as_of_date + score columns
        score_cols: Which score columns to use (default: all 3)

    Returns:
        Series of composite scores in [0, 1]
    """
    if score_cols is None:
        score_cols = [c for c in SCORE_MODELS if c in eval_df.columns]

    composite = pd.Series(0.5, index=eval_df.index)

    for asof_date, idx in eval_df.groupby("as_of_date").groups.items():
        date_df = eval_df.loc[idx]
        rank_sum = pd.Series(0.0, index=idx)
        rank_count = pd.Series(0, index=idx)

        for col in score_cols:
            valid = date_df[col].notna()
            if valid.sum() < 3:
                continue
            ranks = date_df.loc[valid.values, col].rank(pct=True)
            rank_sum.loc[ranks.index] += ranks
            rank_count.loc[ranks.index] += 1

        has_ranks = rank_count > 0
        composite.loc[has_ranks[has_ranks].index] = (
            rank_sum[has_ranks] / rank_count[has_ranks]
        )

    return composite


# ---------------------------------------------------------------------------
# Approach B: Enriched Tabular Model
# ---------------------------------------------------------------------------

DEFAULT_TABULAR_FEATURES = [
    "mom_1m", "mom_3m", "mom_6m", "mom_12m",
    "vol_20d", "vol_60d", "vol_of_vol",
    "max_drawdown_60d",
    "adv_20d", "adv_60d",
    "rel_strength_1m", "rel_strength_3m",
    "beta_252d",
]

SENTIMENT_FEATURES = [
    "filing_sentiment_latest",
    "filing_sentiment_change",
    "filing_sentiment_90d",
    "news_sentiment_7d",
    "news_sentiment_30d",
    "news_sentiment_momentum",
    "news_volume_30d",
    "sentiment_zscore",
    "sentiment_vs_momentum",
]


def train_enriched_lgb(
    train_df: pd.DataFrame,
    features: List[str],
    target_col: str,
    objective: str = "regression",
    time_decay_halflife: int = 252,
) -> "lightgbm.LGBMModel":
    """
    Train LightGBM on enriched feature set (tabular + sentiment).

    Args:
        train_df: Training data with feature columns + target
        features: Feature column names
        target_col: Target column name (excess_return)
        objective: 'regression' or 'lambdarank'
        time_decay_halflife: Sample weight half-life in trading days

    Returns:
        Trained LGB model
    """
    import lightgbm as lgb

    available_features = [f for f in features if f in train_df.columns]
    if not available_features:
        raise ValueError("No available features found for enriched LGB training")

    X = train_df[available_features].copy()
    # Normalize numeric matrix for LightGBM stability.
    X = X.replace([np.inf, -np.inf], np.nan).astype(np.float32)
    y = train_df[target_col].values

    valid = ~np.isnan(y)
    X = X.loc[valid]
    y = y[valid]
    if len(X) < 100:
        raise ValueError(f"Insufficient training rows after filtering: {len(X)}")

    # Time-decay sample weights
    weights = None
    if "date" in train_df.columns and time_decay_halflife > 0:
        dates = pd.to_datetime(train_df.loc[valid, "date"])
        max_date = dates.max()
        days_ago = (max_date - dates).dt.days.values
        weights = np.exp(-np.log(2) * days_ago / time_decay_halflife)

    if objective == "lambdarank":
        # LambdaRank expects integer relevance labels. Convert continuous
        # excess returns into cross-sectional relevance buckets per date.
        date_series = pd.to_datetime(train_df.loc[valid, "date"]).reset_index(drop=True)
        y_series = pd.Series(y).reset_index(drop=True)
        rel = np.zeros(len(y_series), dtype=np.int32)
        for d, idx in y_series.groupby(date_series).groups.items():
            vals = y_series.loc[idx]
            if len(vals) < 5:
                rel[idx] = 2  # neutral relevance for tiny groups
                continue
            pct = vals.rank(pct=True)
            # 5 relevance levels: 0..4
            rel[idx] = np.minimum((pct * 5).astype(int).to_numpy(), 4)
        y_rank = rel.astype(np.int32)

        model = lgb.LGBMRanker(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=5,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42,
            n_jobs=1,
            force_col_wise=True,
            verbosity=-1,
        )
        groups = train_df.loc[valid].groupby("date").size().values
        model.fit(X, y_rank, group=groups, sample_weight=weights)
    else:
        model = lgb.LGBMRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=5,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42,
            n_jobs=1,
            force_col_wise=True,
            verbosity=-1,
        )
        model.fit(X, y, sample_weight=weights)

    return model


def make_enriched_lgb_scorer(
    full_features_df: pd.DataFrame,
    sentiment_gen=None,
    features: List[str] = None,
    objective: str = "regression",
    time_decay_halflife: int = 252,
):
    """
    Factory that creates an enriched LGB scoring function (Approach B).

    Captures the full features_df to derive training data per fold.
    The scorer uses expanding-window training: all data before the
    validation period becomes the training set.

    Args:
        full_features_df: Complete features DataFrame (all dates)
        sentiment_gen: Optional SentimentFeatureGenerator for enrichment
        features: Feature columns to use
        objective: 'regression' or 'lambdarank'
        time_decay_halflife: Sample weight half-life

    Returns:
        Scoring function compatible with run_experiment(scorer_fn=...)
    """
    if features is None:
        features = DEFAULT_TABULAR_FEATURES + SENTIMENT_FEATURES

    _enriched_cache = {}

    def _enrich_fold(df: pd.DataFrame) -> pd.DataFrame:
        """Enrich with sentiment, caching to avoid redundant work."""
        if sentiment_gen is None:
            return df
        has_sentiment = any(c in df.columns for c in SENTIMENT_FEATURES)
        if has_sentiment:
            return df
        return sentiment_gen.enrich_features_df(df)

    def scorer(
        features_df: pd.DataFrame,
        fold_id: str,
        horizon: int,
    ) -> pd.DataFrame:
        er_col = f"excess_return_{horizon}d"
        if er_col not in features_df.columns:
            er_col = "excess_return"

        # Determine training period: everything strictly before validation
        val_dates = pd.to_datetime(features_df["date"])
        val_start = val_dates.min()

        all_dates = pd.to_datetime(full_features_df["date"])
        train_mask = all_dates < val_start
        train_df = full_features_df.loc[train_mask].copy()

        if len(train_df) < 100:
            logger.warning(
                f"Fold {fold_id}: only {len(train_df)} training rows, "
                "skipping enriched LGB"
            )
            return pd.DataFrame()

        # Enrich train and val with sentiment
        train_df = _enrich_fold(train_df)
        val_df = _enrich_fold(features_df)

        # Train
        model = train_enriched_lgb(
            train_df, features, er_col,
            objective=objective,
            time_decay_halflife=time_decay_halflife,
        )

        # Predict
        available = [f for f in features if f in val_df.columns]
        X_val = (
            val_df[available]
            .replace([np.inf, -np.inf], np.nan)
            .astype(np.float32)
        )
        raw_scores = model.predict(X_val)

        results = []
        for i, (idx, row) in enumerate(val_df.iterrows()):
            er_val = row.get(er_col)
            if pd.isna(er_val):
                continue
            results.append({
                "as_of_date": row["date"],
                "ticker": row["ticker"],
                "stable_id": row["stable_id"],
                "fold_id": fold_id,
                "horizon": horizon,
                "score": float(raw_scores[i]),
                "excess_return": float(er_val),
            })
        return pd.DataFrame(results)

    return scorer


# ---------------------------------------------------------------------------
# Approach B (alternative): Enriched XGBoost Model
# ---------------------------------------------------------------------------

def train_enriched_xgb(
    train_df: pd.DataFrame,
    features: List[str],
    target_col: str,
    time_decay_halflife: int = 252,
    xgb_params: Optional[Dict] = None,
):
    """
    Train XGBoost regressor on enriched feature set (tabular + sentiment).

    Uses hist tree method and relies on native missing-value handling.
    """
    from xgboost import XGBRegressor

    available_features = [f for f in features if f in train_df.columns]
    if not available_features:
        raise ValueError("No available features found for enriched XGB training")

    X = (
        train_df[available_features]
        .replace([np.inf, -np.inf], np.nan)
        .astype(np.float32)
    )
    y = train_df[target_col].values
    valid = ~np.isnan(y)
    X = X.loc[valid]
    y = y[valid]
    if len(X) < 100:
        raise ValueError(f"Insufficient training rows after filtering: {len(X)}")

    weights = None
    if "date" in train_df.columns and time_decay_halflife > 0:
        dates = pd.to_datetime(train_df.loc[valid, "date"])
        max_date = dates.max()
        days_ago = (max_date - dates).dt.days.values
        weights = np.exp(-np.log(2) * days_ago / time_decay_halflife)

    params = {
        "n_estimators": 500,
        "max_depth": 7,
        "learning_rate": 0.03,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "reg_alpha": 0.2,
        "reg_lambda": 1.0,
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "random_state": 42,
        "n_jobs": 1,
    }
    if xgb_params:
        params.update(xgb_params)

    model = XGBRegressor(**params)
    model.fit(X, y, sample_weight=weights)
    return model


def make_enriched_xgb_scorer(
    full_features_df: pd.DataFrame,
    sentiment_gen=None,
    features: List[str] = None,
    time_decay_halflife: int = 252,
    xgb_params: Optional[Dict] = None,
):
    """
    Factory for enriched XGBoost scoring function.
    """
    if features is None:
        features = DEFAULT_TABULAR_FEATURES + SENTIMENT_FEATURES

    def _enrich_fold(df: pd.DataFrame) -> pd.DataFrame:
        if sentiment_gen is None:
            return df
        has_sentiment = any(c in df.columns for c in SENTIMENT_FEATURES)
        if has_sentiment:
            return df
        return sentiment_gen.enrich_features_df(df)

    def scorer(
        features_df: pd.DataFrame,
        fold_id: str,
        horizon: int,
    ) -> pd.DataFrame:
        er_col = f"excess_return_{horizon}d"
        if er_col not in features_df.columns:
            er_col = "excess_return"

        val_dates = pd.to_datetime(features_df["date"])
        val_start = val_dates.min()

        all_dates = pd.to_datetime(full_features_df["date"])
        train_mask = all_dates < val_start
        train_df = full_features_df.loc[train_mask].copy()
        if len(train_df) < 100:
            return pd.DataFrame()

        train_df = _enrich_fold(train_df)
        val_df = _enrich_fold(features_df)

        model = train_enriched_xgb(
            train_df=train_df,
            features=features,
            target_col=er_col,
            time_decay_halflife=time_decay_halflife,
            xgb_params=xgb_params,
        )

        available = [f for f in features if f in val_df.columns]
        X_val = (
            val_df[available]
            .replace([np.inf, -np.inf], np.nan)
            .astype(np.float32)
        )
        raw_scores = model.predict(X_val)

        results = []
        for i, (_, row) in enumerate(val_df.iterrows()):
            er_val = row.get(er_col)
            if pd.isna(er_val):
                continue
            results.append({
                "as_of_date": row["date"],
                "ticker": row["ticker"],
                "stable_id": row["stable_id"],
                "fold_id": fold_id,
                "horizon": horizon,
                "score": float(raw_scores[i]),
                "excess_return": float(er_val),
            })
        return pd.DataFrame(results)

    return scorer


# ---------------------------------------------------------------------------
# Approach C: Learned Score Stacking
# ---------------------------------------------------------------------------


def train_stacking_meta(
    scores_df: pd.DataFrame,
    score_cols: List[str] = None,
    context_cols: List[str] = None,
    method: str = "ridge",
) -> object:
    """
    Train a meta-learner on rank-normalized sub-model scores.

    Args:
        scores_df: DataFrame with score columns + excess_return
        score_cols: Sub-model score columns to use
        context_cols: Optional context features (vix_regime, etc.)
        method: 'ridge' or 'lgb'

    Returns:
        Trained meta-learner
    """
    if score_cols is None:
        score_cols = [c for c in SCORE_MODELS if c in scores_df.columns]

    # Rank-normalize per date
    rank_df = pd.DataFrame(index=scores_df.index)
    for col in score_cols:
        rank_df[f"{col}_rank"] = scores_df.groupby("as_of_date")[col].rank(
            pct=True
        )

    feature_cols = [f"{c}_rank" for c in score_cols]
    if context_cols:
        for c in context_cols:
            if c in scores_df.columns:
                rank_df[c] = scores_df[c].values
                feature_cols.append(c)

    X = rank_df[feature_cols].fillna(0.5)
    y = scores_df["excess_return"].values

    valid = ~np.isnan(y)
    X = X.loc[valid]
    y = y[valid]

    if method == "ridge":
        from sklearn.linear_model import Ridge

        model = Ridge(alpha=1.0)
        model.fit(X, y)
    else:
        import lightgbm as lgb

        model = lgb.LGBMRegressor(
            n_estimators=30,
            max_depth=3,
            learning_rate=0.05,
            min_child_samples=5,
            min_data_in_leaf=5,
            random_state=42,
            verbose=-1,
        )
        model.fit(X, y)

    model._fusion_feature_cols = feature_cols
    model._fusion_score_cols = score_cols
    return model


def stacking_predict(
    model,
    scores_df: pd.DataFrame,
) -> np.ndarray:
    """Apply trained stacking meta-learner to score DataFrame."""
    score_cols = model._fusion_score_cols
    feature_cols = model._fusion_feature_cols

    rank_df = pd.DataFrame(index=scores_df.index)
    for col in score_cols:
        rank_df[f"{col}_rank"] = scores_df.groupby("as_of_date")[col].rank(
            pct=True
        )

    for c in feature_cols:
        if c not in rank_df.columns and c in scores_df.columns:
            rank_df[c] = scores_df[c].values

    X = rank_df[feature_cols].fillna(0.5)
    return model.predict(X)


# ---------------------------------------------------------------------------
# Sub-model disagreement (epistemic uncertainty proxy)
# ---------------------------------------------------------------------------


def compute_disagreement(
    scores_df: pd.DataFrame,
    score_cols: List[str] = None,
) -> pd.Series:
    """
    Compute sub-model disagreement as epistemic uncertainty proxy.

    Returns std of rank-normalized scores across sub-models.
    Higher disagreement = higher epistemic uncertainty.
    """
    if score_cols is None:
        score_cols = [c for c in SCORE_MODELS if c in scores_df.columns]

    rank_df = pd.DataFrame(index=scores_df.index)
    for col in score_cols:
        rank_df[col] = scores_df.groupby("as_of_date")[col].rank(pct=True)

    return rank_df[score_cols].std(axis=1).fillna(0.5)

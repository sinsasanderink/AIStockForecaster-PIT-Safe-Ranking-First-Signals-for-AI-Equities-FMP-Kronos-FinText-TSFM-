"""
Aleatoric Baseline a(x) — Chapter 13.2
========================================

Estimates irreducible ranking noise. The aleatoric floor captures how
hard it is to rank stocks on a given date / for a given stock.

Fallback chain (stop at first PASS):

    Tier 0:       Inverse IQR of excess returns (per-date)
    Tier 1:       Inverse IQR of factor-residual excess returns (per-date)
    Tier 2:       Heteroscedastic per-stock LGB quantile regression
    Prospective:  Rolling trailing P10 of rank_loss (PIT-safe, deployable)
    Same-date:    Same-date P10 of rank_loss (retrospective only, last resort)

Tier 3 (posterior-predictive simulation) is intentionally excluded: it uses
the same features as Tier 2, so if Tier 2 cannot predict rank_loss IQR from
those features (ρ ≈ 0 at 20d), a simulation wrapper won't create signal
that isn't there. The bottleneck is information, not model complexity.

Each tier must pass an alignment diagnostic (ρ > 0.3 with rank_loss
across ALL models) before being used downstream.
"""

import logging
from typing import Dict, List, Optional, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

ALIGNMENT_RHO_TARGET = 0.3
ALIGNMENT_RHO_KILL = 0.1
EPS = 1e-6


# ── Tier 0: Inverse IQR ──────────────────────────────────────────────────


def compute_tier0(
    df: pd.DataFrame,
    horizon: int,
) -> pd.Series:
    """
    a(date) = c / (IQR(excess_return) + eps)

    Calibrated so median a(date) ≈ median rank_loss.
    Returns a Series indexed by as_of_date.
    """
    hz = df[df["horizon"] == horizon].copy()

    iqr = hz.groupby("as_of_date")["excess_return"].apply(
        lambda x: x.quantile(0.75) - x.quantile(0.25)
    )

    inv_iqr = 1.0 / (iqr + EPS)

    median_rl = hz.groupby("as_of_date")["rank_loss"].mean().median()
    c = median_rl / inv_iqr.median() if inv_iqr.median() > 0 else 1.0
    a = inv_iqr * c

    a.name = "a_tier0"
    return a


# ── Tier 1: Inverse factor-residual IQR ──────────────────────────────────


def compute_tier1(
    df: pd.DataFrame,
    horizon: int,
) -> pd.Series:
    """
    For each date: regress cross-sectional excess returns on sector dummies
    + market_return_21d + mom_1m, take residuals, compute inverse IQR.

    Measures how much unexplained cross-sectional noise remains after
    removing factors any reasonable model could capture.
    """
    hz = df[df["horizon"] == horizon].copy()

    has_sector = "sector" in hz.columns and hz["sector"].notna().mean() > 0.5
    has_mom = "mom_1m" in hz.columns and hz["mom_1m"].notna().mean() > 0.5
    has_mkt = "market_return_21d" in hz.columns and hz["market_return_21d"].notna().mean() > 0.5

    results = {}

    for date, group in hz.groupby("as_of_date"):
        if len(group) < 10:
            continue

        y = group["excess_return"].values
        parts = []

        if has_sector:
            sector_dummies = pd.get_dummies(group["sector"], drop_first=True, dtype=float)
            parts.append(sector_dummies.values)

        if has_mom:
            m = group["mom_1m"].fillna(0).values.reshape(-1, 1)
            parts.append(m)

        if has_mkt:
            mkt = group["market_return_21d"].fillna(0).values.reshape(-1, 1)
            parts.append(mkt)

        if not parts:
            residuals = y - y.mean()
        else:
            X = np.hstack(parts)
            X = np.column_stack([np.ones(len(X)), X])
            try:
                beta = np.linalg.lstsq(X, y, rcond=None)[0]
                predicted = X @ beta
                residuals = y - predicted
            except np.linalg.LinAlgError:
                residuals = y - y.mean()

        q75 = np.percentile(residuals, 75)
        q25 = np.percentile(residuals, 25)
        iqr_resid = q75 - q25
        results[date] = iqr_resid

    iqr_series = pd.Series(results, dtype=float)
    iqr_series.index.name = "as_of_date"

    inv_iqr = 1.0 / (iqr_series + EPS)

    median_rl = hz.groupby("as_of_date")["rank_loss"].mean().median()
    c = median_rl / inv_iqr.median() if inv_iqr.median() > 0 else 1.0
    a = inv_iqr * c

    a.name = "a_tier1"
    return a


# ── Tier 2: Heteroscedastic per-stock LGB quantile regression ────────────


TIER2_FEATURES = ["vol_20d", "adv_20d", "market_vol_21d", "vix_percentile_252d", "mom_1m"]

TIER2_PARAMS_BASE = {
    "n_estimators": 30,
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


def compute_tier2(
    df: pd.DataFrame,
    horizon: int,
    min_train_folds: int = 20,
) -> pd.DataFrame:
    """
    Walk-forward LGB quantile regression: predict per-stock rank_loss Q25
    and Q75.  a(stock, date) = Q75_pred - Q25_pred.

    Unlike Tier 0/1, this produces per-stock (not just per-date) estimates.
    No inversion needed: high predicted IQR = hard to rank = high a.

    Returns a DataFrame with [as_of_date, ticker, stable_id, a_tier2].
    """
    hz = df[df["horizon"] == horizon].copy()

    feats = [f for f in TIER2_FEATURES if f in hz.columns and hz[f].notna().mean() > 0.3]
    if not feats:
        raise ValueError("No features available for Tier 2 quantile regression")

    if "sector" in hz.columns and hz["sector"].notna().mean() > 0.5:
        hz["sector_enc"] = hz["sector"].astype("category").cat.codes.astype(float)
        feats = feats + ["sector_enc"]

    all_folds = sorted(hz["fold_id"].unique())
    predict_folds = all_folds[min_train_folds:]

    logger.info(
        f"Tier 2: {len(feats)} features, "
        f"{len(predict_folds)} prediction folds (starting fold {predict_folds[0]})"
    )

    results = []

    for fold_idx, fold_id in enumerate(predict_folds):
        train_folds = set(all_folds[: min_train_folds + fold_idx])
        train = hz[hz["fold_id"].isin(train_folds)]
        test = hz[hz["fold_id"] == fold_id]

        if len(test) == 0 or len(train) < 100:
            continue

        X_train = train[feats].fillna(0)
        y_train = train["rank_loss"].fillna(0)
        X_test = test[feats].fillna(0)

        params_q25 = {**TIER2_PARAMS_BASE, "objective": "quantile", "alpha": 0.25}
        params_q75 = {**TIER2_PARAMS_BASE, "objective": "quantile", "alpha": 0.75}

        model_q25 = lgb.LGBMRegressor(**params_q25)
        model_q75 = lgb.LGBMRegressor(**params_q75)

        model_q25.fit(X_train, y_train)
        model_q75.fit(X_train, y_train)

        pred_q25 = model_q25.predict(X_test)
        pred_q75 = model_q75.predict(X_test)

        a_vals = np.clip(pred_q75 - pred_q25, 0, None)

        out = test[["as_of_date", "ticker", "stable_id", "fold_id"]].copy()
        out["a_tier2"] = a_vals
        out["q25_pred"] = pred_q25
        out["q75_pred"] = pred_q75
        results.append(out)

    if not results:
        return pd.DataFrame(columns=["as_of_date", "ticker", "stable_id", "fold_id", "a_tier2"])

    result_df = pd.concat(results, ignore_index=True)
    logger.info(
        f"Tier 2: {len(result_df):,} predictions, "
        f"a_tier2 mean={result_df['a_tier2'].mean():.4f}, "
        f"std={result_df['a_tier2'].std():.4f}"
    )
    return result_df


# ── Empirical fallbacks ───────────────────────────────────────────────────


def compute_empirical_fallback(
    df: pd.DataFrame,
    horizon: int,
    percentile: float = 10.0,
) -> pd.Series:
    """
    a(date) = 10th percentile of rank_loss on that date.

    Uses same-date cross-sectional information — not available at inference
    time (retrospective only). Prefer compute_prospective_empirical for
    deployment-ready estimates.
    """
    hz = df[df["horizon"] == horizon]
    a = hz.groupby("as_of_date")["rank_loss"].quantile(percentile / 100.0)
    a.name = "a_empirical"
    return a


def compute_prospective_empirical(
    df: pd.DataFrame,
    horizon: int,
    lookback_days: int = 60,
    min_lookback: int = 20,
    percentile: float = 10.0,
) -> pd.Series:
    """
    a(date) = P10 of rank_loss over the trailing `lookback_days` trading days.

    Strictly PIT-safe: uses only data BEFORE the current date. Deployable
    in production. Works best when ranking difficulty is persistent
    (60d/90d horizons where autocorrelation is high).
    """
    hz = df[df["horizon"] == horizon].copy()
    all_dates = sorted(hz["as_of_date"].unique())

    results = {}
    for i, d in enumerate(all_dates):
        lookback_dates = all_dates[max(0, i - lookback_days):i]
        if len(lookback_dates) < min_lookback:
            continue
        past = hz[hz["as_of_date"].isin(lookback_dates)]
        results[d] = past["rank_loss"].quantile(percentile / 100.0)

    a = pd.Series(results, dtype=float)
    a.index.name = "as_of_date"
    a.name = "a_prospective"
    return a


# ── Alignment diagnostic ─────────────────────────────────────────────────


def run_alignment_diagnostic(
    a_series: pd.Series,
    model_dfs: Dict[str, pd.DataFrame],
    horizon: int,
    tier_name: str = "tier0",
) -> Dict:
    """
    Validate that a(x) captures real ranking difficulty.

    For per-date tiers (0, 1, empirical): a_series indexed by as_of_date.
    For per-stock tiers (2): a_series is a DataFrame with a_tier2 column,
    aggregated to per-date mean before diagnostic.

    Checks:
        1. Bin dates into quintiles by a(date)
        2. For each quintile, compute median rank_loss for ALL models
        3. Check monotonicity + rho > 0.3

    Returns diagnostic dict with pass/fail decision.
    """
    if isinstance(a_series, pd.DataFrame):
        a_daily = a_series.groupby("as_of_date")[f"a_{tier_name}"].mean()
    else:
        a_daily = a_series.copy()

    a_daily = a_daily.dropna()

    if len(a_daily) < 20:
        return {
            "tier": tier_name,
            "pass": False,
            "reason": f"Too few dates ({len(a_daily)})",
            "rho": 0.0,
        }

    try:
        quintile_labels = pd.qcut(a_daily, 5, labels=False, duplicates="drop")
    except ValueError:
        quintile_labels = pd.cut(a_daily.rank(pct=True), 5, labels=False)

    quintile_df = pd.DataFrame({
        "a_value": a_daily,
        "quintile": quintile_labels,
    })

    per_model_results = {}

    for model_name, mdf in model_dfs.items():
        hz_data = mdf[mdf["horizon"] == horizon].copy()
        daily_rl = hz_data.groupby("as_of_date")["rank_loss"].mean()

        common_dates = a_daily.index.intersection(daily_rl.index)
        if len(common_dates) < 20:
            per_model_results[model_name] = {
                "rho": 0.0,
                "quintile_medians": [],
                "monotonic": False,
            }
            continue

        a_common = a_daily.loc[common_dates]
        rl_common = daily_rl.loc[common_dates]

        rho_val = stats.spearmanr(a_common, rl_common).statistic

        merged = quintile_df.loc[common_dates].copy()
        merged["rank_loss"] = rl_common

        quintile_medians = merged.groupby("quintile")["rank_loss"].median().values.tolist()

        diffs = np.diff(quintile_medians)
        monotonic = bool(np.all(diffs >= -0.005))

        per_model_results[model_name] = {
            "rho": float(rho_val),
            "quintile_medians": quintile_medians,
            "monotonic": monotonic,
        }

    all_rhos = [r["rho"] for r in per_model_results.values()]
    mean_rho = float(np.mean(all_rhos)) if all_rhos else 0.0
    all_monotonic = all(r["monotonic"] for r in per_model_results.values())

    passes = mean_rho >= ALIGNMENT_RHO_TARGET
    killed = mean_rho < ALIGNMENT_RHO_KILL

    if passes:
        verdict = "PASS"
    elif killed:
        verdict = "KILL"
    else:
        verdict = "MARGINAL"

    return {
        "tier": tier_name,
        "pass": passes,
        "verdict": verdict,
        "mean_rho": mean_rho,
        "all_monotonic": all_monotonic,
        "per_model": per_model_results,
        "n_dates": len(a_daily),
        "a_stats": {
            "mean": float(a_daily.mean()),
            "median": float(a_daily.median()),
            "std": float(a_daily.std()),
            "min": float(a_daily.min()),
            "max": float(a_daily.max()),
        },
    }


# ── Tier selection orchestrator ───────────────────────────────────────────


def select_best_tier(
    enriched_lgb: pd.DataFrame,
    model_dfs: Dict[str, pd.DataFrame],
    horizon: int,
    max_tier: int = 2,
    min_train_folds: int = 20,
) -> Tuple[str, pd.Series | pd.DataFrame, Dict]:
    """
    Run tiers in priority order, stop at first that passes alignment.

    Priority: Tier 0 → Tier 1 → Tier 2 → Prospective empirical → Same-date empirical

    Prospective empirical is preferred over same-date because it's PIT-safe
    and deployment-ready. Same-date is last resort (retrospective only).

    Returns (tier_name, a_values, diagnostic_report).
    """
    diagnostics_all = {}

    # Tier 0
    logger.info(f"\n--- Tier 0: Inverse IQR dispersion ({horizon}d) ---")
    a0 = compute_tier0(enriched_lgb, horizon)
    diag0 = run_alignment_diagnostic(a0, model_dfs, horizon, "tier0")
    diagnostics_all["tier0"] = diag0
    logger.info(
        f"  Tier 0: mean_rho={diag0['mean_rho']:.4f}, "
        f"monotonic={diag0['all_monotonic']}, verdict={diag0['verdict']}"
    )
    if diag0["pass"]:
        return "tier0", a0, diagnostics_all

    # Tier 1
    if max_tier >= 1:
        logger.info(f"\n--- Tier 1: Inverse factor-residual IQR ({horizon}d) ---")
        a1 = compute_tier1(enriched_lgb, horizon)
        diag1 = run_alignment_diagnostic(a1, model_dfs, horizon, "tier1")
        diagnostics_all["tier1"] = diag1
        logger.info(
            f"  Tier 1: mean_rho={diag1['mean_rho']:.4f}, "
            f"monotonic={diag1['all_monotonic']}, verdict={diag1['verdict']}"
        )
        if diag1["pass"]:
            return "tier1", a1, diagnostics_all

    # Tier 2
    if max_tier >= 2:
        logger.info(f"\n--- Tier 2: Heteroscedastic per-stock noise ({horizon}d) ---")
        a2_df = compute_tier2(enriched_lgb, horizon, min_train_folds=min_train_folds)
        diag2 = run_alignment_diagnostic(a2_df, model_dfs, horizon, "tier2")
        diagnostics_all["tier2"] = diag2
        logger.info(
            f"  Tier 2: mean_rho={diag2['mean_rho']:.4f}, "
            f"monotonic={diag2['all_monotonic']}, verdict={diag2['verdict']}"
        )
        if diag2["pass"]:
            return "tier2", a2_df, diagnostics_all

    # Prospective empirical (PIT-safe, deployment-ready)
    logger.info(f"\n--- Prospective empirical: rolling P10 ({horizon}d) ---")
    a_prosp = compute_prospective_empirical(enriched_lgb, horizon)
    diag_prosp = run_alignment_diagnostic(a_prosp, model_dfs, horizon, "prospective")
    diagnostics_all["prospective"] = diag_prosp
    logger.info(
        f"  Prospective: mean_rho={diag_prosp['mean_rho']:.4f}, "
        f"monotonic={diag_prosp.get('all_monotonic', False)}, "
        f"verdict={diag_prosp['verdict']}"
    )
    if diag_prosp["pass"]:
        return "prospective", a_prosp, diagnostics_all

    # Same-date empirical (retrospective only — last resort)
    logger.warning(
        "  All prospective tiers failed. Falling back to same-date "
        "empirical (retrospective only, not deployment-ready)."
    )
    a_fb = compute_empirical_fallback(enriched_lgb, horizon)
    diag_fb = run_alignment_diagnostic(a_fb, model_dfs, horizon, "empirical")
    diagnostics_all["empirical"] = diag_fb
    return "empirical", a_fb, diagnostics_all

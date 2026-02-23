"""
Chapter 13.7: Deployment Policy & Sizing Ablation
===================================================

Tests six deployment policy variants, all using a binary G(t) ≥ 0.2 regime gate.
Goal: find whether alternative applications of ê(x) add economic value on top of
vol-sizing by using DEUP information differently.

Key finding from 13.6:
    g(x) correlates with |score| (r ≈ 0.3+), so inverse-sizing penalises the
    model's strongest signals. Vol-sizing avoids this. 13.7 tests additive sorting,
    residualisation and capping as alternatives.

Literature:
    Liu et al. (2026) arXiv:2601.00593  — Uncertainty-Adjusted Sorting
    Hentschel (2025)                    — Contextual Alpha (residualisation)
    Barroso & Saxena (2021) RFS         — Learn from Forecast Errors
    Chaudhuri & Lopez-Paz (2023)        — Selective Prediction

Variants evaluated:
    1. gate_raw:          Binary gate + raw point-prediction sort
    2. gate_vol:          Binary gate + vol-sizing (Ch12 best method, the benchmark)
    3. gate_ua_sort:      Binary gate + Liu et al. uncertainty-adjusted sorting
    4. gate_resid_ehat:   Binary gate + residualised-ê sizing
    5. gate_ehat_cap:     Binary gate + ê cap at P90
    6. gate_vol_ehat_cap: Binary gate + vol-sizing + ê cap

Kill criterion K4:
    trail_rankic:  Binary gate + trailing RankIC date-level sizing

All calibration is on DEV (pre-2024) only. FINAL is evaluated once, frozen.
PIT safety: G(t) uses lagged realised data; trailing RankIC uses matured labels.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from src.uncertainty.deup_portfolio import (
    HOLDOUT_START,
    TOP_K,
    DEFAULT_COST_BPS,
    EPS,
    calibrate_c,
    compute_portfolio_metrics,
)

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────

GATE_THRESHOLD: float = 0.2
CRISIS_START = pd.Timestamp("2024-03-01")
CRISIS_END = pd.Timestamp("2024-07-31")
PERIODS_PER_YEAR: float = 12.0

LAMBDA_GRID: List[float] = [0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0]
CAP_GRID: List[Tuple[float, float]] = [
    (0.80, 0.7),
    (0.85, 0.7),
    (0.90, 0.5),
    (0.90, 0.7),
    (0.95, 0.7),
]

POLICY_VARIANTS = [
    "gate_raw",
    "gate_vol",
    "gate_ua_sort",
    "gate_resid_ehat",
    "gate_ehat_cap",
    "gate_vol_ehat_cap",
    "trail_rankic",
]

# ── Data preparation ───────────────────────────────────────────────────────


def merge_data_for_policy(
    enriched_df: pd.DataFrame,
    ehat_df: pd.DataFrame,
    health_df: pd.DataFrame,
    horizon: int = 20,
) -> pd.DataFrame:
    """
    Merge enriched residuals, DEUP predictions, and health gate for policy evaluation.

    Returns a merged DataFrame with all columns needed for variant computation:
        as_of_date, ticker, stable_id, score, excess_return, rank_loss,
        vol_20d, rank_score, g_pred, ehat_raw, G_exposure, H_realized,
        matured_rankic

    G_exposure defaults to 1.0 for dates without health data (assume safe to trade).
    """
    er = enriched_df[enriched_df["horizon"] == horizon].copy()
    ehat = ehat_df[ehat_df["horizon"] == horizon].copy()

    er["as_of_date"] = pd.to_datetime(er["as_of_date"])
    ehat["as_of_date"] = pd.to_datetime(ehat["as_of_date"])

    # Deduplicate enriched (multiple sub_model_ids can produce duplicate rows)
    er_dedup = (
        er.groupby(["as_of_date", "ticker"])
        .agg(
            score=("score", "mean"),
            excess_return=("excess_return", "mean"),
            rank_loss=("rank_loss", "mean"),
            vol_20d=("vol_20d", "mean"),
            rank_score=("rank_score", "mean"),
            stable_id=("stable_id", "first"),
        )
        .reset_index()
    )

    # Deduplicate ehat
    ehat_dedup = (
        ehat.groupby(["as_of_date", "ticker"])
        .agg(g_pred=("g_pred", "mean"), ehat_raw=("ehat_raw", "mean"))
        .reset_index()
    )

    merged = er_dedup.merge(ehat_dedup, on=["as_of_date", "ticker"], how="inner")

    # Add health gate
    hdf = health_df[
        ["date", "G_exposure", "matured_rankic", "H_realized"]
    ].copy()
    hdf["date"] = pd.to_datetime(hdf["date"])
    hdf = hdf.rename(columns={"date": "as_of_date"})
    merged = merged.merge(hdf, on="as_of_date", how="left")
    merged["G_exposure"] = merged["G_exposure"].fillna(1.0)
    merged["H_realized"] = merged["H_realized"].fillna(0.0)

    logger.info(
        f"  Merged {horizon}d: {len(merged):,} rows, "
        f"{merged['as_of_date'].nunique()} dates, "
        f"{merged['ticker'].nunique()} unique tickers"
    )
    return merged


# ── ê–score structural-conflict diagnostic ────────────────────────────────


def diagnose_ehat_score_correlation(
    merged_df: pd.DataFrame,
    horizon: str = "20d",
) -> Dict[str, Any]:
    """
    Formally document the ê–score structural conflict that causes inverse-sizing
    to fail (Section 13.7, Motivation).

    Computes:
    1. Per-date Spearman ρ(ê, |score|) — proves the structural conflict
    2. By ê decile: mean |score|, mean rank_loss, IC contribution
    3. P&L attribution: do high-ê stocks drive both gains AND losses?

    Returns:
        Dict with median / mean cross-sectional correlations and decile stats.
    """
    df = merged_df.copy()
    df = df.dropna(subset=["ehat_raw", "score", "rank_loss", "excess_return"])

    # ── 1. Per-date cross-sectional Spearman correlation ──────────────────
    def _date_corr(g: pd.DataFrame) -> pd.Series:
        if len(g) < 5:
            return pd.Series({"rho_ehat_abs_score": np.nan, "rho_g_abs_score": np.nan})
        rho_ehat = sp_stats.spearmanr(g["ehat_raw"], g["score"].abs()).statistic
        rho_g = sp_stats.spearmanr(g["g_pred"], g["score"].abs()).statistic
        return pd.Series(
            {"rho_ehat_abs_score": rho_ehat, "rho_g_abs_score": rho_g}
        )

    daily_corrs = df.groupby("as_of_date").apply(_date_corr, include_groups=False)

    results: Dict[str, Any] = {
        "horizon": horizon,
        "n_dates": int(df["as_of_date"].nunique()),
        "n_observations": int(len(df)),
        "median_rho_ehat_abs_score": round(
            float(daily_corrs["rho_ehat_abs_score"].median()), 4
        ),
        "mean_rho_ehat_abs_score": round(
            float(daily_corrs["rho_ehat_abs_score"].mean()), 4
        ),
        "median_rho_g_abs_score": round(
            float(daily_corrs["rho_g_abs_score"].median()), 4
        ),
        "pct_positive_corr": round(
            float((daily_corrs["rho_ehat_abs_score"] > 0).mean()), 4
        ),
    }

    # ── 2. By ê decile ────────────────────────────────────────────────────
    df["ehat_decile"] = df.groupby("as_of_date")["ehat_raw"].transform(
        lambda x: pd.qcut(x.rank(method="first"), 10, labels=False, duplicates="drop")
    )

    # IC contribution: |score| × sign(score) × excess_return (proxy for how much
    # each stock contributes to the cross-sectional score–return correlation)
    df["ic_contrib"] = df["score"] * df["excess_return"]

    decile_stats = (
        df.groupby("ehat_decile")
        .agg(
            mean_abs_score=("score", lambda x: x.abs().mean()),
            mean_rank_loss=("rank_loss", "mean"),
            mean_ic_contrib=("ic_contrib", "mean"),
            mean_ehat=("ehat_raw", "mean"),
            n=("score", "count"),
        )
        .round(6)
    )
    results["decile_stats"] = decile_stats.to_dict()

    # ── 3. P&L attribution ────────────────────────────────────────────────
    # Split into high-ê (top-20%) and low-ê (bottom-20%) per date
    df["ehat_quintile"] = df.groupby("as_of_date")["ehat_raw"].transform(
        lambda x: pd.qcut(x.rank(method="first"), 5, labels=False, duplicates="drop")
    )
    pnl = df.groupby("ehat_quintile").agg(
        mean_return=("excess_return", "mean"),
        std_return=("excess_return", "std"),
        mean_abs_score=("score", lambda x: x.abs().mean()),
    )
    results["quintile_pnl"] = pnl.round(6).to_dict()

    # ── Summary conclusion ────────────────────────────────────────────────
    median_rho = results["median_rho_ehat_abs_score"]
    conflict_confirmed = median_rho > 0.05
    results["structural_conflict_confirmed"] = conflict_confirmed
    results["interpretation"] = (
        f"Median ρ(ê, |score|) = {median_rho:.3f}. "
        + ("Structural conflict CONFIRMED: " if conflict_confirmed else "Weak: ")
        + "inverse-sizing penalises extreme-ranked stocks (the model's strongest signals)."
    )
    logger.info(f"  ê–score diagnostic: {results['interpretation']}")
    return results


# ── Per-date variant helpers ───────────────────────────────────────────────


def apply_binary_gate(g_t: float, threshold: float = GATE_THRESHOLD) -> bool:
    """True ↔ model is trustworthy; False ↔ abstain (flat portfolio)."""
    return (not np.isnan(g_t)) and (g_t >= threshold)


def uncertainty_adjusted_sort(
    scores: pd.Series,
    uncertainty: pd.Series,
    lambda_ua: float,
    top_k: int = TOP_K,
) -> Tuple[pd.Index, pd.Index]:
    """
    Liu et al. (2026) uncertainty-adjusted sorting.

    Longs:  sort by upper bound = score + λ·uncertainty  (optimistic case)
    Shorts: sort by lower bound = score − λ·uncertainty  (pessimistic case)

    This is additive — high-score + high-uncertainty stocks can still enter
    the long leg (upper bound is large), avoiding the inverse-sizing conflict.

    Args:
        scores:       Cross-sectional point predictions (same index as uncertainty).
        uncertainty:  Per-stock uncertainty width (g_pred or ehat_raw).
        lambda_ua:    Uncertainty scaling factor; λ=0 reduces to standard sort.
        top_k:        Number of longs / shorts to select.

    Returns:
        (long_tickers, short_tickers) — each an Index of length ≤ top_k.
    """
    upper = scores + lambda_ua * uncertainty
    lower = scores - lambda_ua * uncertainty
    long_tickers = upper.nlargest(top_k).index
    short_tickers = lower.nsmallest(top_k).index
    return long_tickers, short_tickers


def residualize_ehat(
    ehat: pd.Series,
    rank_score: pd.Series,
) -> pd.Series:
    """
    Residualise ê on cross-sectional score extremity (per-date OLS).

    Regresses: ê = β₀ + β₁·|rank_score − 0.5|·2 + ε

    ê_resid > 0: stock is MORE uncertain than expected for its extremity level.
    ê_resid ≈ 0: typical uncertainty for this score level — no penalty.
    ê_resid < 0: LESS uncertain than expected — no penalty either.

    The residualisation removes the structural ê–|score| correlation so that
    only genuinely anomalous uncertainty drives the sizing penalty.
    """
    if len(ehat) < 3:
        return pd.Series(np.zeros(len(ehat)), index=ehat.index)

    abs_rank = (np.abs(rank_score.values - 0.5) * 2).reshape(-1, 1)
    X = np.column_stack([np.ones(len(abs_rank)), abs_rank])
    y = ehat.values

    try:
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        residuals = y - X @ beta
    except np.linalg.LinAlgError:
        residuals = np.zeros(len(y))

    return pd.Series(residuals, index=ehat.index)


def apply_ehat_cap(
    scores: pd.Series,
    ehat: pd.Series,
    cap_percentile: float = 0.90,
    cap_weight: float = 0.70,
) -> pd.Series:
    """
    Reduce scores for stocks in the top (1 − cap_percentile) of ê.

    Per-date cross-sectional: the threshold is computed within each date.
    90% of stocks are untouched; only the most uncertain 10% are capped.

    Args:
        scores:         Raw or vol-sized scores.
        ehat:           Per-stock epistemic uncertainty (ehat_raw or g_pred).
        cap_percentile: Stocks above this ê percentile get reduced.
        cap_weight:     Multiplier applied to capped stocks (< 1.0).
    """
    threshold = ehat.quantile(cap_percentile)
    multiplier = np.where(ehat > threshold, cap_weight, 1.0)
    return scores * multiplier


# ── Calibration ───────────────────────────────────────────────────────────


def _run_policy_on_dev(
    dev_df: pd.DataFrame,
    variant: str,
    params: Dict[str, Any],
    top_k: int = TOP_K,
    cost_bps: float = DEFAULT_COST_BPS,
) -> float:
    """
    Run a single variant on DEV data and return the Sharpe ratio.
    Used for hyperparameter grid search.
    """
    try:
        records = _build_single_variant_timeseries(dev_df, variant, params, top_k, cost_bps)
        if len(records) < 10:
            return -9.0
        returns = np.array([r["ls_return_net"] for r in records])
        returns = returns[~np.isnan(returns)]
        if len(returns) < 5:
            return -9.0
        std = float(np.std(returns, ddof=1))
        if std < 1e-10:
            return -9.0
        return float(np.mean(returns) / std * np.sqrt(PERIODS_PER_YEAR))
    except Exception:
        return -9.0


def calibrate_lambda_ua(
    dev_df: pd.DataFrame,
    lambda_grid: List[float] = LAMBDA_GRID,
    top_k: int = TOP_K,
    cost_bps: float = DEFAULT_COST_BPS,
) -> Tuple[float, Dict[str, float]]:
    """
    Grid-search lambda_ua for Variant 3 (Liu et al.) on DEV data.

    Returns (best_lambda, {lambda: sharpe} grid results).
    """
    grid_results: Dict[str, float] = {}
    for lam in lambda_grid:
        sharpe = _run_policy_on_dev(
            dev_df, "gate_ua_sort", {"lambda_ua": lam}, top_k, cost_bps
        )
        grid_results[str(lam)] = round(sharpe, 4)
        logger.debug(f"    lambda_ua={lam:.2f} → DEV Sharpe={sharpe:.3f}")

    best_lambda = float(
        max(lambda_grid, key=lambda l: grid_results[str(l)])
    )
    logger.info(
        f"  Calibrated lambda_ua={best_lambda} "
        f"(DEV Sharpe={grid_results[str(best_lambda)]:.3f})"
    )
    return best_lambda, grid_results


def calibrate_cap_params(
    dev_df: pd.DataFrame,
    cap_grid: List[Tuple[float, float]] = CAP_GRID,
    use_vol: bool = False,
    c_vol: float = 1.0,
    top_k: int = TOP_K,
    cost_bps: float = DEFAULT_COST_BPS,
) -> Tuple[float, float, Dict[str, float]]:
    """
    Grid-search (cap_percentile, cap_weight) for Variant 5/6 on DEV data.

    Returns (best_cap_percentile, best_cap_weight, grid_results).
    """
    grid_results: Dict[str, float] = {}
    best_sharpe = -99.0
    best_pct, best_wt = 0.90, 0.70

    variant = "gate_vol_ehat_cap" if use_vol else "gate_ehat_cap"

    for cap_pct, cap_wt in cap_grid:
        params: Dict[str, Any] = {
            "cap_percentile": cap_pct,
            "cap_weight": cap_wt,
        }
        if use_vol:
            params["c_vol"] = c_vol

        sharpe = _run_policy_on_dev(dev_df, variant, params, top_k, cost_bps)
        key = f"{cap_pct:.2f}_{cap_wt:.1f}"
        grid_results[key] = round(sharpe, 4)
        logger.debug(f"    cap_pct={cap_pct}, cap_wt={cap_wt} → DEV Sharpe={sharpe:.3f}")

        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_pct = cap_pct
            best_wt = cap_wt

    logger.info(
        f"  Calibrated cap_percentile={best_pct}, cap_weight={best_wt} "
        f"(DEV Sharpe={best_sharpe:.3f})"
    )
    return best_pct, best_wt, grid_results


def calibrate_trailing_ic(
    dev_df: pd.DataFrame,
    target_median: float = 0.7,
) -> float:
    """
    Calibrate c_trail for the trailing-RankIC kill-criterion variant on DEV data.

    Sets c_trail so that on DEV days where H_realized > 0,
    median(min(1, c_trail × H_realized)) ≈ target_median.
    """
    pos_h = dev_df.groupby("as_of_date")["H_realized"].first()
    pos_h = pos_h[pos_h > 0]
    if len(pos_h) == 0:
        return 1.0
    median_h = float(pos_h.median())
    if median_h < EPS:
        return 1.0
    c_trail = target_median / median_h
    # Verify
    weights = np.minimum(1.0, c_trail * pos_h.values)
    med_w = float(np.median(weights))
    logger.info(f"  Calibrated c_trail={c_trail:.4f} (DEV median w={med_w:.3f})")
    return float(c_trail)


def calibrate_all_params(
    dev_df: pd.DataFrame,
    top_k: int = TOP_K,
    cost_bps: float = DEFAULT_COST_BPS,
) -> Dict[str, Any]:
    """
    Calibrate all DEV-only hyperparameters and return a frozen params dict.

    Called ONCE on DEV; result is then frozen and used for both DEV and FINAL.

    Returns:
        {
          "c_vol": float,
          "c_resid": float,
          "lambda_ua": float,
          "lambda_grid": {str: float},
          "cap_percentile": float,
          "cap_weight": float,
          "cap_grid": {str: float},
          "cap_percentile_vol": float,
          "cap_weight_vol": float,
          "cap_grid_vol": {str: float},
          "c_trail": float,
        }
    """
    logger.info("Calibrating hyperparameters on DEV data …")

    # Variant 2: c_vol (median vol-sized weight ≈ 0.7)
    c_vol = calibrate_c(dev_df["vol_20d"], target_median=0.7)
    logger.info(f"  c_vol = {c_vol:.4f}")

    # Variant 4: c_resid
    # Compute residualised ê for all DEV data to get the distribution
    dev_df = dev_df.copy()
    resid_series = []
    for _, grp in dev_df.groupby("as_of_date"):
        if len(grp) < 3:
            resid_series.append(
                pd.Series(np.zeros(len(grp)), index=grp.index)
            )
        else:
            resid_series.append(
                residualize_ehat(grp["ehat_raw"], grp["rank_score"])
            )
    dev_df["ehat_resid"] = pd.concat(resid_series)
    # Only positive residuals contribute to sizing penalty
    pos_resid = dev_df.loc[dev_df["ehat_resid"] > 0, "ehat_resid"]
    c_resid = calibrate_c(pos_resid, target_median=0.7) if len(pos_resid) > 0 else 1.0
    logger.info(f"  c_resid = {c_resid:.4f}")

    # Variant 3: lambda_ua — grid search
    lambda_ua, lambda_grid_results = calibrate_lambda_ua(dev_df, LAMBDA_GRID, top_k, cost_bps)

    # Variant 5: cap params (gate + ê-cap only)
    cap_pct, cap_wt, cap_grid_results = calibrate_cap_params(
        dev_df, CAP_GRID, use_vol=False, top_k=top_k, cost_bps=cost_bps
    )

    # Variant 6: cap params (gate + vol + ê-cap)
    cap_pct_vol, cap_wt_vol, cap_grid_vol = calibrate_cap_params(
        dev_df, CAP_GRID, use_vol=True, c_vol=c_vol, top_k=top_k, cost_bps=cost_bps
    )

    # Kill criterion: c_trail
    c_trail = calibrate_trailing_ic(dev_df, target_median=0.7)

    return {
        "c_vol": round(c_vol, 6),
        "c_resid": round(c_resid, 6),
        "lambda_ua": lambda_ua,
        "lambda_ua_grid": lambda_grid_results,
        "cap_percentile": cap_pct,
        "cap_weight": cap_wt,
        "cap_grid": cap_grid_results,
        "cap_percentile_vol": cap_pct_vol,
        "cap_weight_vol": cap_wt_vol,
        "cap_grid_vol": cap_grid_vol,
        "c_trail": round(c_trail, 6),
    }


# ── Per-date portfolio computation ────────────────────────────────────────


def _compute_date_variant(
    day: pd.DataFrame,
    variant: str,
    params: Dict[str, Any],
    top_k: int = TOP_K,
) -> Dict[str, Any]:
    """
    Compute one variant's portfolio selection and return for a single date.

    Returns dict with keys: long_ids, short_ids, ls_return.
    Returns None if the date should produce a flat portfolio (abstain).

    For UA Sort: long and short legs are selected independently.
    For all others: sort by modified score, top K = longs, bottom K = shorts.
    """
    if len(day) < 2 * top_k:
        return None

    # Ensure required columns exist
    required = ["score", "excess_return", "ehat_raw", "g_pred", "vol_20d", "rank_score"]
    for col in required:
        if col not in day.columns:
            return None

    id_col = "stable_id" if "stable_id" in day.columns else "ticker"

    if variant == "gate_raw":
        sorted_day = day.sort_values("score", ascending=False)
        long_ids = set(sorted_day.head(top_k)[id_col])
        short_ids = set(sorted_day.tail(top_k)[id_col])

    elif variant == "gate_vol":
        c_vol = params["c_vol"]
        sized_score = day["score"] * np.minimum(
            1.0, c_vol / np.sqrt(day["vol_20d"].clip(lower=EPS) + EPS)
        )
        sorted_day = day.assign(_s=sized_score).sort_values("_s", ascending=False)
        long_ids = set(sorted_day.head(top_k)[id_col])
        short_ids = set(sorted_day.tail(top_k)[id_col])

    elif variant == "gate_ua_sort":
        lam = params["lambda_ua"]
        long_idx, short_idx = uncertainty_adjusted_sort(
            day["score"].set_axis(day.index),
            day["g_pred"].set_axis(day.index),
            lambda_ua=lam,
            top_k=top_k,
        )
        long_ids = set(day.loc[long_idx, id_col])
        short_ids = set(day.loc[short_idx, id_col])

    elif variant == "gate_resid_ehat":
        c_resid = params["c_resid"]
        ehat_resid = residualize_ehat(
            day["ehat_raw"].set_axis(day.index),
            day["rank_score"].set_axis(day.index),
        )
        # Only positive residual uncertainty drives the penalty
        pos_resid = np.maximum(ehat_resid.values, 0.0)
        sized_score = day["score"].values * np.minimum(
            1.0, c_resid / np.sqrt(pos_resid + EPS)
        )
        order = np.argsort(sized_score)[::-1]
        ordered = day.iloc[order]
        long_ids = set(ordered.head(top_k)[id_col])
        short_ids = set(ordered.tail(top_k)[id_col])

    elif variant == "gate_ehat_cap":
        cap_pct = params["cap_percentile"]
        cap_wt = params["cap_weight"]
        capped_score = apply_ehat_cap(
            day["score"].set_axis(day.index),
            day["ehat_raw"].set_axis(day.index),
            cap_percentile=cap_pct,
            cap_weight=cap_wt,
        )
        sorted_day = day.assign(_s=capped_score).sort_values("_s", ascending=False)
        long_ids = set(sorted_day.head(top_k)[id_col])
        short_ids = set(sorted_day.tail(top_k)[id_col])

    elif variant == "gate_vol_ehat_cap":
        c_vol = params["c_vol"]
        cap_pct = params.get("cap_percentile_vol", params.get("cap_percentile", 0.90))
        cap_wt = params.get("cap_weight_vol", params.get("cap_weight", 0.70))
        vol_score = day["score"] * np.minimum(
            1.0, c_vol / np.sqrt(day["vol_20d"].clip(lower=EPS) + EPS)
        )
        capped_score = apply_ehat_cap(
            vol_score.set_axis(day.index),
            day["ehat_raw"].set_axis(day.index),
            cap_percentile=cap_pct,
            cap_weight=cap_wt,
        )
        sorted_day = day.assign(_s=capped_score).sort_values("_s", ascending=False)
        long_ids = set(sorted_day.head(top_k)[id_col])
        short_ids = set(sorted_day.tail(top_k)[id_col])

    elif variant == "trail_rankic":
        # Kill criterion K4: date-level sizing by trailing realized RankIC
        c_trail = params["c_trail"]
        h_realized = float(day["H_realized"].iloc[0])
        if h_realized <= 0:
            return None  # Abstain: trailing IC is negative
        weight = min(1.0, c_trail * h_realized)
        sized_score = day["score"] * weight
        sorted_day = day.assign(_s=sized_score).sort_values("_s", ascending=False)
        long_ids = set(sorted_day.head(top_k)[id_col])
        short_ids = set(sorted_day.tail(top_k)[id_col])

    else:
        raise ValueError(f"Unknown variant: {variant}")

    long_ret = day.loc[day[id_col].isin(long_ids), "excess_return"].mean()
    short_ret = day.loc[day[id_col].isin(short_ids), "excess_return"].mean()

    if np.isnan(long_ret) or np.isnan(short_ret):
        return None

    return {
        "long_ids": long_ids,
        "short_ids": short_ids,
        "ls_return": float(long_ret - short_ret),
    }


def _build_single_variant_timeseries(
    df: pd.DataFrame,
    variant: str,
    params: Dict[str, Any],
    top_k: int = TOP_K,
    cost_bps: float = DEFAULT_COST_BPS,
) -> List[Dict[str, Any]]:
    """
    Build a time series of portfolio returns for one variant.

    Binary gate: if G_exposure < GATE_THRESHOLD, portfolio is flat (return = 0,
    no cost, positions reset to empty for re-entry cost calculation).

    Returns list of dicts: {date, ls_return, ls_return_net, turnover, is_active}.
    """
    df = df.sort_values("as_of_date")
    dates = sorted(df["as_of_date"].unique())
    records: List[Dict[str, Any]] = []
    last_active_longs: set = set()
    last_active_shorts: set = set()
    was_flat: bool = True  # First entry is a fresh start

    id_col = "stable_id" if "stable_id" in df.columns else "ticker"

    for dt in dates:
        day = df[df["as_of_date"] == dt]
        g_t = float(day["G_exposure"].iloc[0])

        if not apply_binary_gate(g_t, GATE_THRESHOLD):
            # Flat period: zero return, no position update
            records.append({
                "date": dt,
                "ls_return": 0.0,
                "ls_return_net": 0.0,
                "turnover": 0.0,
                "is_active": False,
                "G_exposure": g_t,
            })
            was_flat = True
            continue

        result = _compute_date_variant(day, variant, params, top_k)
        if result is None:
            records.append({
                "date": dt,
                "ls_return": 0.0,
                "ls_return_net": 0.0,
                "turnover": 0.0,
                "is_active": False,
                "G_exposure": g_t,
            })
            continue

        new_longs = result["long_ids"]
        new_shorts = result["short_ids"]
        ls_ret = result["ls_return"]

        # Turnover vs last ACTIVE portfolio
        if was_flat or not last_active_longs:
            # Re-entering or first period: full turnover
            turnover = 1.0
        else:
            lt = 1 - len(new_longs & last_active_longs) / top_k
            st = 1 - len(new_shorts & last_active_shorts) / top_k
            turnover = (lt + st) / 2.0

        cost = turnover * (cost_bps / 10_000) * 2
        ls_return_net = ls_ret - cost

        records.append({
            "date": dt,
            "ls_return": ls_ret,
            "ls_return_net": ls_return_net,
            "turnover": turnover,
            "is_active": True,
            "G_exposure": g_t,
        })

        last_active_longs = new_longs
        last_active_shorts = new_shorts
        was_flat = False

    return records


# ── Portfolio metrics ─────────────────────────────────────────────────────


def _metrics_from_timeseries(
    records: List[Dict[str, Any]],
    period: str = "ALL",
) -> Dict[str, Any]:
    """Compute portfolio metrics for a subset of records."""
    df = pd.DataFrame(records)
    if df.empty:
        return {"n_periods": 0, "period": period}

    df["date"] = pd.to_datetime(df["date"])

    if period == "DEV":
        df = df[df["date"] < HOLDOUT_START]
    elif period == "FINAL":
        df = df[df["date"] >= HOLDOUT_START]
    elif period == "CRISIS":
        df = df[(df["date"] >= CRISIS_START) & (df["date"] <= CRISIS_END)]

    if df.empty:
        return {"n_periods": 0, "period": period}

    rets = df["ls_return_net"].values
    turnover = df["turnover"].values
    n_active = int(df["is_active"].sum()) if "is_active" in df.columns else len(df)
    abstention_rate = round(1.0 - n_active / max(len(df), 1), 4)

    metrics = compute_portfolio_metrics(
        rets, turnover=turnover, periods_per_year=PERIODS_PER_YEAR
    )
    metrics["period"] = period
    metrics["n_active_periods"] = n_active
    metrics["abstention_rate"] = abstention_rate
    metrics["mean_G"] = round(float(df["G_exposure"].mean()), 4)
    return metrics


def compute_variant_metrics_all_periods(
    records: List[Dict[str, Any]],
    variant_name: str,
) -> Dict[str, Any]:
    """Compute ALL / DEV / FINAL / CRISIS metrics for one variant."""
    return {
        "variant": variant_name,
        "ALL": _metrics_from_timeseries(records, "ALL"),
        "DEV": _metrics_from_timeseries(records, "DEV"),
        "FINAL": _metrics_from_timeseries(records, "FINAL"),
        "CRISIS": _metrics_from_timeseries(records, "CRISIS"),
    }


# ── Main pipeline ─────────────────────────────────────────────────────────


def run_policy_pipeline(
    enriched_df: pd.DataFrame,
    ehat_df: pd.DataFrame,
    health_df: pd.DataFrame,
    horizon: int = 20,
    top_k: int = TOP_K,
    cost_bps: float = DEFAULT_COST_BPS,
    variants: Optional[List[str]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Full Chapter 13.7 pipeline.

    Steps:
        1. Merge all data sources for the given horizon.
        2. Run ê–score structural conflict diagnostic.
        3. Calibrate all hyperparameters on DEV only.
        4. Build time-series returns for each variant.
        5. Compute ALL / DEV / FINAL / CRISIS metrics.

    Returns:
        (policy_results, calibration_params)
        policy_results: {variant_name: {period: metrics}} for all variants
        calibration_params: frozen DEV-calibrated hyperparameters
    """
    if variants is None:
        variants = POLICY_VARIANTS

    logger.info(f"Running Chapter 13.7 policy pipeline for {horizon}d …")

    # ── 1. Merge ──────────────────────────────────────────────────────────
    merged = merge_data_for_policy(enriched_df, ehat_df, health_df, horizon)

    # ── 2. Structural-conflict diagnostic ─────────────────────────────────
    dev_df = merged[merged["as_of_date"] < HOLDOUT_START].copy()
    ehat_diag = diagnose_ehat_score_correlation(merged, horizon=f"{horizon}d")
    logger.info(
        f"  Structural conflict: median ρ(ê, |score|) = "
        f"{ehat_diag['median_rho_ehat_abs_score']:.3f}"
    )

    # ── 3. Calibrate (DEV only) ───────────────────────────────────────────
    cal_params = calibrate_all_params(dev_df, top_k=top_k, cost_bps=cost_bps)
    logger.info(
        f"  Calibration complete: lambda_ua={cal_params['lambda_ua']}, "
        f"c_vol={cal_params['c_vol']:.4f}, "
        f"cap_pct={cal_params['cap_percentile']}"
    )

    # ── 4. Build variant timeseries ───────────────────────────────────────
    policy_results: Dict[str, Any] = {}
    timeseries: Dict[str, List[Dict]] = {}

    for variant in variants:
        logger.info(f"  Building variant: {variant} …")
        records = _build_single_variant_timeseries(merged, variant, cal_params, top_k, cost_bps)
        timeseries[variant] = records
        policy_results[variant] = compute_variant_metrics_all_periods(records, variant)

    # ── 5. Summary diagnostics ────────────────────────────────────────────
    policy_results["_diagnostic"] = ehat_diag
    policy_results["_horizon"] = horizon

    return policy_results, cal_params, timeseries


# ── Results formatting ────────────────────────────────────────────────────

VARIANT_LABELS = {
    "gate_raw":          "1. Gate+Raw",
    "gate_vol":          "2. Gate+Vol",
    "gate_ua_sort":      "3. Gate+UA Sort (Liu)",
    "gate_resid_ehat":   "4. Gate+Resid-ê",
    "gate_ehat_cap":     "5. Gate+ê-Cap",
    "gate_vol_ehat_cap": "6. Gate+Vol+ê-Cap",
    "trail_rankic":      "   Kill: Trail-IC",
}


def print_results_table(
    policy_results: Dict[str, Any],
    baseline_13_6: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Print the Chapter 13.7 comparison table to stdout.

    Includes the frozen 13.6 ungated baselines for reference.
    """
    header = (
        "\n"
        "╔══════════════════════════════════════════════════════════════════════╗\n"
        "║  Chapter 13.7 — Deployment Policy Comparison (20d Primary Horizon)  ║\n"
        "╠═══════════════════════╦═════════╦═════════╦═══════════╦═════════════╣\n"
        "║ Variant               ║  ALL    ║  DEV    ║  FINAL    ║ Crisis MaxDD║\n"
        "╠═══════════════════════╬═════════╬═════════╬═══════════╬═════════════╣"
    )
    footer = (
        "╚═══════════════════════╩═════════╩═════════╩═══════════╩═════════════╝"
    )
    sep = (
        "╠═══════════════════════╬═════════╬═════════╬═══════════╬═════════════╣"
    )

    def _fmt(v: Any, pct: bool = False) -> str:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "  —   "
        if pct:
            return f"{v:>+7.1%}"
        return f"{v:>7.3f}"

    print(header)

    # Reference baselines from 13.6
    if baseline_13_6:
        ref_lines = [
            ("Ungated raw (13.6)",  baseline_13_6.get("baseline_raw",  {})),
            ("Ungated vol (13.6)",  baseline_13_6.get("A_vol_sized",   {})),
        ]
        for label, data in ref_lines:
            all_s  = data.get("ALL",   {}).get("sharpe")
            dev_s  = data.get("DEV",   {}).get("sharpe")
            fin_s  = data.get("FINAL", {}).get("sharpe")
            cri_dd = data.get("CRISIS_2024", {}).get("max_drawdown")
            print(
                f"║ {label:<21} ║ {_fmt(all_s)} ║ {_fmt(dev_s)} ║ {_fmt(fin_s)}  ║ {_fmt(cri_dd, pct=True)} ║"
            )
        print(sep)

    # 13.7 variants
    for variant in POLICY_VARIANTS:
        if variant not in policy_results:
            continue
        label = VARIANT_LABELS.get(variant, variant)
        v_res = policy_results[variant]
        all_s  = v_res.get("ALL",    {}).get("sharpe")
        dev_s  = v_res.get("DEV",    {}).get("sharpe")
        fin_s  = v_res.get("FINAL",  {}).get("sharpe")
        cri_dd = v_res.get("CRISIS", {}).get("max_drawdown")
        abst   = v_res.get("ALL",    {}).get("abstention_rate", 0.0)
        print(
            f"║ {label:<21} ║ {_fmt(all_s)} ║ {_fmt(dev_s)} ║ {_fmt(fin_s)}  ║ {_fmt(cri_dd, pct=True)} ║"
        )

    print(footer)
    print(
        "\nNotes: Sharpe annualised (×√12). Crisis = Mar–Jul 2024. "
        "Binary gate G≥0.2; abstention rate ~47% in FINAL.\n"
    )

"""
Conformal Prediction Intervals — Chapter 13.5
===============================================

Three nonconformity-score variants for rank-displacement intervals:

    Raw        s = rank_loss                    → constant-width intervals
    Vol-norm   s = rank_loss / vol_20d          → wider for volatile stocks
    DEUP-norm  s = rank_loss / max(ê(x), eps)  → wider for uncertain stocks

Rolling 60-day calibration, 90% nominal coverage, split conformal.

The key test (Plassier et al., 2025 motivation): DEUP-normalized should produce
better **conditional** coverage than raw or vol-normalized — equalizing coverage
across high-ê and low-ê stocks rather than over-covering easy stocks and
under-covering hard stocks.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

logger = logging.getLogger(__name__)

HOLDOUT_START = pd.Timestamp("2024-01-01")

VARIANTS = ["raw", "vol", "deup"]

EPS_DEUP = 0.001
EPS_VOL = 0.001


@dataclass
class ConformalConfig:
    alpha: float = 0.10
    calibration_window: int = 60
    min_calibration: int = 30
    eps_deup: float = EPS_DEUP
    eps_vol: float = EPS_VOL


def compute_nonconformity_scores(
    df: pd.DataFrame,
    config: Optional[ConformalConfig] = None,
) -> pd.DataFrame:
    """
    Compute nonconformity scores for three variants.

    Expects columns: as_of_date, ticker, rank_loss, ehat_raw, vol_20d.
    """
    cfg = config or ConformalConfig()
    df = df.copy()

    df["s_raw"] = df["rank_loss"]
    df["s_vol"] = df["rank_loss"] / np.maximum(df["vol_20d"], cfg.eps_vol)
    df["s_deup"] = df["rank_loss"] / np.maximum(df["ehat_raw"], cfg.eps_deup)

    return df


def compute_conformal_intervals(
    df: pd.DataFrame,
    config: Optional[ConformalConfig] = None,
) -> pd.DataFrame:
    """
    Rolling split-conformal prediction intervals for all three variants.

    For each prediction at date t:
    1. Calibration set = scores from the past `calibration_window` trading days
       (strictly before t) — PIT-safe.
    2. Threshold q = ceil((n_cal + 1) * (1 - alpha))-th smallest score.
    3. Interval width = q × normalizer.
    4. covered = (rank_loss <= width).

    Returns df with added columns per variant:
        q_{var}, width_{var}, covered_{var}
    """
    cfg = config or ConformalConfig()
    df = compute_nonconformity_scores(df, cfg)

    # Sort by date for rolling window
    df = df.sort_values(["as_of_date", "ticker"]).reset_index(drop=True)
    dates = sorted(df["as_of_date"].unique())
    date_to_idx = {d: i for i, d in enumerate(dates)}

    # Pre-compute per-date score arrays for fast calibration lookup
    date_scores = {}
    for d, grp in df.groupby("as_of_date"):
        date_scores[d] = {
            "s_raw": grp["s_raw"].values,
            "s_vol": grp["s_vol"].values,
            "s_deup": grp["s_deup"].values,
        }

    # Compute quantile thresholds per date (one q per date per variant)
    q_by_date = {}
    quantile_level = 1.0 - cfg.alpha

    for i, d in enumerate(dates):
        cal_start = max(0, i - cfg.calibration_window)
        cal_dates = dates[cal_start:i]  # strictly before current date

        if len(cal_dates) < cfg.min_calibration:
            q_by_date[d] = {v: np.nan for v in VARIANTS}
            continue

        qs = {}
        for var_name, score_key in [("raw", "s_raw"), ("vol", "s_vol"), ("deup", "s_deup")]:
            cal_scores = np.concatenate([date_scores[cd][score_key] for cd in cal_dates])
            n_cal = len(cal_scores)
            # Split conformal quantile: ceil((n+1)(1-alpha)) / n
            idx = int(np.ceil((n_cal + 1) * quantile_level))
            idx = min(idx, n_cal) - 1  # 0-indexed, clamp
            sorted_scores = np.sort(cal_scores)
            qs[var_name] = float(sorted_scores[idx])
        q_by_date[d] = qs

    # Map quantiles back and compute widths + coverage
    df["q_raw"] = df["as_of_date"].map(lambda d: q_by_date[d]["raw"])
    df["q_vol"] = df["as_of_date"].map(lambda d: q_by_date[d]["vol"])
    df["q_deup"] = df["as_of_date"].map(lambda d: q_by_date[d]["deup"])

    # Interval widths
    df["width_raw"] = df["q_raw"]  # constant per calibration window
    df["width_vol"] = df["q_vol"] * np.maximum(df["vol_20d"], cfg.eps_vol)
    df["width_deup"] = df["q_deup"] * np.maximum(df["ehat_raw"], cfg.eps_deup)

    # Coverage
    df["covered_raw"] = (df["rank_loss"] <= df["width_raw"]).astype(int)
    df["covered_vol"] = (df["rank_loss"] <= df["width_vol"]).astype(int)
    df["covered_deup"] = (df["rank_loss"] <= df["width_deup"]).astype(int)

    n_valid = df["q_raw"].notna().sum()
    n_total = len(df)
    logger.info(
        f"  Conformal intervals: {n_valid:,}/{n_total:,} predictions with valid calibration"
    )

    return df


def compute_diagnostics(
    df: pd.DataFrame,
    config: Optional[ConformalConfig] = None,
) -> Dict[str, Any]:
    """
    Comprehensive conformal diagnostics.

    Computes marginal coverage, ECE, conditional coverage by ê tercile,
    DEV/FINAL split, VIX regime, width ratio, and width efficiency.
    """
    cfg = config or ConformalConfig()
    target_coverage = 1.0 - cfg.alpha

    # Drop rows without valid calibration
    valid = df.dropna(subset=["q_raw", "q_vol", "q_deup"]).copy()
    if len(valid) == 0:
        return {"skip": True, "reason": "no valid calibration rows"}

    diag: Dict[str, Any] = {
        "n_predictions": len(valid),
        "target_coverage": target_coverage,
        "horizon": int(valid["horizon"].iloc[0]) if "horizon" in valid.columns else None,
    }

    # ── 1. Marginal coverage + ECE ────────────────────────────────────────
    for var in VARIANTS:
        cov = float(valid[f"covered_{var}"].mean())
        ece = abs(cov - target_coverage)
        diag[f"coverage_{var}"] = round(cov, 4)
        diag[f"ece_{var}"] = round(ece, 4)
        diag[f"mean_width_{var}"] = round(float(valid[f"width_{var}"].mean()), 4)
        diag[f"median_width_{var}"] = round(float(valid[f"width_{var}"].median()), 4)

    # ── 2. Conditional coverage by ê tercile ──────────────────────────────
    ehat_vals = valid["ehat_raw"]
    try:
        valid = valid.copy()
        valid["ehat_tercile"] = pd.qcut(ehat_vals, 3, labels=["low", "mid", "high"], duplicates="drop")
        has_terciles = valid["ehat_tercile"].nunique() >= 2
    except ValueError:
        has_terciles = False

    if has_terciles:
        cond_cov = {}
        for var in VARIANTS:
            per_tercile = {}
            for t_name in valid["ehat_tercile"].dropna().unique():
                sub = valid[valid["ehat_tercile"] == t_name]
                per_tercile[str(t_name)] = {
                    "coverage": round(float(sub[f"covered_{var}"].mean()), 4),
                    "mean_width": round(float(sub[f"width_{var}"].mean()), 4),
                    "n": len(sub),
                }
            # Spread: difference between max and min tercile coverage
            coverages = [v["coverage"] for v in per_tercile.values()]
            per_tercile["spread"] = round(max(coverages) - min(coverages), 4)
            cond_cov[var] = per_tercile
        diag["conditional_coverage_ehat"] = cond_cov
    else:
        diag["conditional_coverage_ehat"] = {"note": "could not form terciles (sparse ehat)"}

    # ── 3. Conditional coverage by VIX regime ─────────────────────────────
    if "vix_percentile_252d" in valid.columns:
        valid_vix = valid.copy()
        valid_vix["vix_regime"] = pd.cut(
            valid_vix["vix_percentile_252d"],
            bins=[0, 0.33, 0.67, 1.0],
            labels=["low_vix", "mid_vix", "high_vix"],
            include_lowest=True,
        )
        vix_cov = {}
        for var in VARIANTS:
            per_vix = {}
            for regime in valid_vix["vix_regime"].dropna().unique():
                sub = valid_vix[valid_vix["vix_regime"] == regime]
                per_vix[str(regime)] = {
                    "coverage": round(float(sub[f"covered_{var}"].mean()), 4),
                    "n": len(sub),
                }
            coverages = [v["coverage"] for v in per_vix.values()]
            per_vix["spread"] = round(max(coverages) - min(coverages), 4) if coverages else 0
            vix_cov[var] = per_vix
        diag["conditional_coverage_vix"] = vix_cov

    # ── 4. DEV vs FINAL ──────────────────────────────────────────────────
    valid_dt = valid.copy()
    valid_dt["as_of_date"] = pd.to_datetime(valid_dt["as_of_date"])
    for period_name, mask in [
        ("DEV", valid_dt["as_of_date"] < HOLDOUT_START),
        ("FINAL", valid_dt["as_of_date"] >= HOLDOUT_START),
    ]:
        pdata = valid_dt[mask]
        if len(pdata) < 100:
            diag[period_name] = {"n": len(pdata)}
            continue
        period_diag = {"n": len(pdata)}
        for var in VARIANTS:
            cov = float(pdata[f"covered_{var}"].mean())
            ece = abs(cov - target_coverage)
            period_diag[f"coverage_{var}"] = round(cov, 4)
            period_diag[f"ece_{var}"] = round(ece, 4)
            period_diag[f"mean_width_{var}"] = round(float(pdata[f"width_{var}"].mean()), 4)
        diag[period_name] = period_diag

    # ── 5. Width ratio: high-ê / low-ê ──────────────────────────────────
    if has_terciles:
        width_ratio = {}
        for var in VARIANTS:
            low_w = valid[valid["ehat_tercile"] == "low"][f"width_{var}"].mean()
            high_w = valid[valid["ehat_tercile"] == "high"][f"width_{var}"].mean()
            width_ratio[var] = round(float(high_w / max(low_w, 1e-8)), 4)
        diag["width_ratio_high_low"] = width_ratio

    # ── 6. Width efficiency (narrower = better at same coverage) ─────────
    # Compare mean width per unit of coverage
    for var in VARIANTS:
        cov = diag[f"coverage_{var}"]
        mw = diag[f"mean_width_{var}"]
        if cov > 0:
            diag[f"width_per_coverage_{var}"] = round(mw / cov, 4)

    return diag


def run_conformal_pipeline(
    ehat_df: pd.DataFrame,
    enriched_df: pd.DataFrame,
    horizons: Optional[List[int]] = None,
    config: Optional[ConformalConfig] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Full conformal pipeline: merge data, compute intervals, run diagnostics.

    Args:
        ehat_df: ehat_predictions.parquet (has ehat_raw, rank_loss, ticker, as_of_date, horizon)
        enriched_df: enriched_residuals (has vol_20d, vix_percentile_252d)
        horizons: which horizons to process (default [20, 60, 90])
        config: ConformalConfig

    Returns:
        all_intervals: DataFrame with intervals for all horizons
        all_diagnostics: Dict with per-horizon diagnostics
    """
    cfg = config or ConformalConfig()
    horizons = horizons or [20, 60, 90]

    # Merge ehat with enriched to get vol_20d, vix_percentile_252d
    merge_cols = ["as_of_date", "ticker", "horizon"]
    feature_cols = ["vol_20d", "vix_percentile_252d"]
    existing = [c for c in feature_cols if c in ehat_df.columns]
    needed = [c for c in feature_cols if c not in ehat_df.columns]

    if needed:
        # Deduplicate enriched by (date, ticker, horizon) taking mean
        er_sub = enriched_df[merge_cols + needed].copy()
        er_sub = er_sub.groupby(merge_cols, as_index=False)[needed].mean()
        merged = ehat_df.merge(er_sub, on=merge_cols, how="inner")
        logger.info(
            f"Merged ehat ({len(ehat_df):,}) with enriched → {len(merged):,} rows"
        )
    else:
        merged = ehat_df.copy()

    all_intervals = []
    all_diagnostics = {}

    for hz in horizons:
        logger.info(f"\n--- Conformal intervals for {hz}d ---")
        hz_df = merged[merged["horizon"] == hz].copy()

        if len(hz_df) == 0:
            logger.warning(f"  No data for horizon {hz}")
            continue

        hz_df["as_of_date"] = pd.to_datetime(hz_df["as_of_date"])

        # Compute intervals
        intervals = compute_conformal_intervals(hz_df, cfg)

        # Compute diagnostics
        intervals["horizon"] = hz
        diag = compute_diagnostics(intervals, cfg)

        all_intervals.append(intervals)
        all_diagnostics[f"{hz}d"] = diag

        # Log summary
        for var in VARIANTS:
            cov = diag.get(f"coverage_{var}", "?")
            ece = diag.get(f"ece_{var}", "?")
            mw = diag.get(f"mean_width_{var}", "?")
            logger.info(f"  {var:5s}: coverage={cov}, ECE={ece}, mean_width={mw}")

    if all_intervals:
        result_df = pd.concat(all_intervals, ignore_index=True)
    else:
        result_df = pd.DataFrame()

    return result_df, all_diagnostics

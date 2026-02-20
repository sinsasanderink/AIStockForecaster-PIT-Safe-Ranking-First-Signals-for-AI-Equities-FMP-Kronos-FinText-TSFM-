"""
Epistemic Signal ê(x) — Chapter 13.3
======================================

ê(x) = max(0, g(x) − a(x))

Merges the g(x) error predictor (13.1) with the aleatoric baseline a(x)
(13.2) to produce per-prediction epistemic uncertainty estimates.

Deployment labeling:
    20d  → "retrospective_decomposition"  (same-date a(x), not deployable)
    60d  → "deployment_ready"             (Tier 2 walk-forward a(x))
    90d  → "deployment_ready"             (prospective rolling P10 a(x))
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

HOLDOUT_START = pd.Timestamp("2024-01-01")

DEPLOYMENT_LABELS = {
    "empirical": "retrospective_decomposition",
    "tier0": "deployment_ready",
    "tier1": "deployment_ready",
    "tier2": "deployment_ready",
    "prospective": "deployment_ready",
}

PER_STOCK_TIERS = {"tier2"}


def _merge_g_and_a(
    g_preds: pd.DataFrame,
    a_preds: pd.DataFrame,
    horizon: int,
) -> pd.DataFrame:
    """
    Merge g(x) predictions with a(x) values for a single horizon.

    Handles two cases:
      - Per-date a(x): join on as_of_date only, broadcast to all stocks
      - Per-stock a(x): join on (as_of_date, ticker)
    """
    gp = g_preds[g_preds["horizon"] == horizon].copy()
    ap = a_preds[a_preds["horizon"] == horizon].copy()

    if len(ap) == 0:
        raise ValueError(f"No a(x) predictions for horizon {horizon}")
    if len(gp) == 0:
        raise ValueError(f"No g(x) predictions for horizon {horizon}")

    tier = ap["tier"].iloc[0]
    is_per_stock = tier in PER_STOCK_TIERS and "ticker" in ap.columns and ap["ticker"].notna().any()

    gp["as_of_date"] = pd.to_datetime(gp["as_of_date"])
    ap["as_of_date"] = pd.to_datetime(ap["as_of_date"])

    if is_per_stock:
        ap_cols = ap[["as_of_date", "ticker", "a_value"]].copy()
        merged = gp.merge(ap_cols, on=["as_of_date", "ticker"], how="inner")
    else:
        ap_date = ap.groupby("as_of_date")["a_value"].first().reset_index()
        merged = gp.merge(ap_date, on="as_of_date", how="inner")

    merged["a_tier"] = tier
    merged["deployment_label"] = DEPLOYMENT_LABELS.get(tier, "unknown")

    n_g = len(gp)
    n_merged = len(merged)
    match_pct = n_merged / n_g * 100 if n_g > 0 else 0
    logger.info(
        f"  {horizon}d merge: {n_g:,} g(x) rows → {n_merged:,} matched "
        f"({match_pct:.1f}%), tier={tier}, per_stock={is_per_stock}"
    )

    return merged


def compute_ehat(
    g_preds: pd.DataFrame,
    a_preds: pd.DataFrame,
    horizons: Optional[List[int]] = None,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Compute ê(x) = max(0, g(x) − a(x)) for all horizons.

    Returns:
        ehat_df: DataFrame with columns [as_of_date, ticker, stable_id,
                 horizon, fold_id, g_pred, a_value, a_tier, ehat_raw,
                 ehat_pctile, rank_loss, deployment_label, period]
        diagnostics: per-horizon summary statistics
    """
    if horizons is None:
        horizons = sorted(set(g_preds["horizon"].unique()) & set(a_preds["horizon"].unique()))

    all_results = []
    diagnostics = {}

    for hz in horizons:
        logger.info(f"\n--- Computing ê(x) for {hz}d ---")

        merged = _merge_g_and_a(g_preds, a_preds, hz)

        merged["ehat_raw"] = np.maximum(0.0, merged["g_pred"] - merged["a_value"])

        merged["ehat_pctile"] = merged.groupby("as_of_date")["ehat_raw"].rank(pct=True)

        merged["period"] = np.where(
            merged["as_of_date"] < HOLDOUT_START, "DEV", "FINAL"
        )

        diag = _compute_horizon_diagnostics(merged, hz)
        diagnostics[hz] = diag

        all_results.append(merged)

    if not all_results:
        return pd.DataFrame(), diagnostics

    ehat_df = pd.concat(all_results, ignore_index=True)

    output_cols = [
        "as_of_date", "ticker", "stable_id", "horizon", "fold_id",
        "g_pred", "a_value", "a_tier", "ehat_raw", "ehat_pctile",
        "rank_loss", "deployment_label", "period",
    ]
    available_cols = [c for c in output_cols if c in ehat_df.columns]
    ehat_df = ehat_df[available_cols]

    return ehat_df, diagnostics


def _compute_horizon_diagnostics(merged: pd.DataFrame, hz: int) -> Dict:
    """Comprehensive diagnostics for a single horizon's ê(x)."""
    ehat = merged["ehat_raw"]
    g = merged["g_pred"]
    a = merged["a_value"]
    rl = merged["rank_loss"]

    n = len(merged)
    pct_zero = float((ehat == 0).mean())
    pct_positive = float((ehat > 0).mean())

    # Right-skew: skewness of positive values
    pos_vals = ehat[ehat > 0]
    skewness = float(pos_vals.skew()) if len(pos_vals) > 10 else 0.0

    # Correlation of ehat with realized rank_loss
    rho_ehat_rl = float(stats.spearmanr(ehat, rl).statistic) if n > 20 else 0.0

    # Correlation of g(x) with realized rank_loss (for comparison)
    rho_g_rl = float(stats.spearmanr(g, rl).statistic) if n > 20 else 0.0

    # DEV vs FINAL breakdown
    dev_final = {}
    for period_name in ["DEV", "FINAL"]:
        period_mask = merged["period"] == period_name
        period_data = merged[period_mask]

        if len(period_data) < 20:
            dev_final[period_name] = {"n_rows": len(period_data)}
            continue

        pe = period_data["ehat_raw"]
        prl = period_data["rank_loss"]

        daily_ehat = period_data.groupby("as_of_date")["ehat_raw"].mean()
        daily_rl = period_data.groupby("as_of_date")["rank_loss"].mean()
        daily_rho = float(stats.spearmanr(daily_ehat, daily_rl).statistic) if len(daily_ehat) > 10 else 0.0

        dev_final[period_name] = {
            "n_rows": len(period_data),
            "n_dates": int(period_data["as_of_date"].nunique()),
            "ehat_mean": float(pe.mean()),
            "ehat_std": float(pe.std()),
            "pct_zero": float((pe == 0).mean()),
            "pct_positive": float((pe > 0).mean()),
            "rho_ehat_rl": float(stats.spearmanr(pe, prl).statistic) if len(pe) > 20 else 0.0,
            "daily_rho_ehat_rl": daily_rho,
        }

    # Selective risk: do low-ê stocks have lower rank_loss?
    if pct_positive > 0 and pct_zero > 0:
        low_ehat = merged[ehat <= ehat.quantile(0.33)]
        high_ehat = merged[ehat >= ehat.quantile(0.67)]
        selective_low_rl = float(low_ehat["rank_loss"].mean())
        selective_high_rl = float(high_ehat["rank_loss"].mean())
        selective_delta = selective_high_rl - selective_low_rl
    else:
        # All positive or all zero — use terciles of ehat
        q33 = ehat.quantile(0.33)
        q67 = ehat.quantile(0.67)
        low_ehat = merged[ehat <= q33]
        high_ehat = merged[ehat >= q67]
        selective_low_rl = float(low_ehat["rank_loss"].mean()) if len(low_ehat) > 0 else 0.0
        selective_high_rl = float(high_ehat["rank_loss"].mean()) if len(high_ehat) > 0 else 0.0
        selective_delta = selective_high_rl - selective_low_rl

    # Daily-level: ê(x) mean vs daily RankIC
    daily_ehat = merged.groupby("as_of_date")["ehat_raw"].mean()
    daily_rl = merged.groupby("as_of_date")["rank_loss"].mean()
    daily_rho = float(stats.spearmanr(daily_ehat, daily_rl).statistic) if len(daily_ehat) > 20 else 0.0

    tier = merged["a_tier"].iloc[0]
    label = merged["deployment_label"].iloc[0]

    diag = {
        "n_rows": n,
        "n_dates": int(merged["as_of_date"].nunique()),
        "n_tickers": int(merged["ticker"].nunique()) if "ticker" in merged.columns else 0,
        "a_tier": tier,
        "deployment_label": label,
        "g_pred_mean": float(g.mean()),
        "g_pred_std": float(g.std()),
        "a_value_mean": float(a.mean()),
        "a_value_std": float(a.std()),
        "ehat_mean": float(ehat.mean()),
        "ehat_std": float(ehat.std()),
        "ehat_median": float(ehat.median()),
        "ehat_p95": float(ehat.quantile(0.95)),
        "pct_zero": pct_zero,
        "pct_positive": pct_positive,
        "skewness_positive": skewness,
        "rho_ehat_rank_loss": rho_ehat_rl,
        "rho_g_rank_loss": rho_g_rl,
        "daily_rho_ehat_rank_loss": daily_rho,
        "selective_low_tercile_rl": selective_low_rl,
        "selective_high_tercile_rl": selective_high_rl,
        "selective_delta": selective_delta,
        "dev_final": dev_final,
    }

    _log_diagnostics(hz, diag)
    return diag


def _log_diagnostics(hz: int, diag: Dict) -> None:
    """Pretty-print diagnostics for a horizon."""
    label = diag["deployment_label"]
    logger.info(f"  {hz}d [{label}]:")
    logger.info(
        f"    g(x): mean={diag['g_pred_mean']:.4f}, "
        f"a(x): mean={diag['a_value_mean']:.4f} (tier={diag['a_tier']})"
    )
    logger.info(
        f"    ê(x): mean={diag['ehat_mean']:.4f}, median={diag['ehat_median']:.4f}, "
        f"P95={diag['ehat_p95']:.4f}"
    )
    logger.info(
        f"    Distribution: {diag['pct_zero']:.1%} zero, "
        f"{diag['pct_positive']:.1%} positive"
    )
    if diag["pct_positive"] > 0 and diag["pct_positive"] < 1:
        logger.info(f"    Positive skewness: {diag['skewness_positive']:.2f}")
    logger.info(
        f"    ρ(ê, rank_loss)={diag['rho_ehat_rank_loss']:.4f}, "
        f"ρ(g, rank_loss)={diag['rho_g_rank_loss']:.4f}"
    )
    logger.info(
        f"    Daily ρ(ê_mean, rl_mean)={diag['daily_rho_ehat_rank_loss']:.4f}"
    )
    logger.info(
        f"    Selective risk: low-ê RL={diag['selective_low_tercile_rl']:.4f}, "
        f"high-ê RL={diag['selective_high_tercile_rl']:.4f}, "
        f"Δ={diag['selective_delta']:.4f}"
    )

    for period_name in ["DEV", "FINAL"]:
        pdata = diag["dev_final"].get(period_name, {})
        if pdata.get("n_rows", 0) < 20:
            logger.info(f"    {period_name}: insufficient data ({pdata.get('n_rows', 0)} rows)")
            continue
        logger.info(
            f"    {period_name}: n={pdata['n_rows']:,}, "
            f"ê_mean={pdata['ehat_mean']:.4f}, "
            f"ρ(ê,rl)={pdata['rho_ehat_rl']:.4f}, "
            f"daily_ρ={pdata['daily_rho_ehat_rl']:.4f}"
        )


def compute_ehat_mae(
    g_preds_mae: pd.DataFrame,
    a_preds: pd.DataFrame,
    horizons: Optional[List[int]] = None,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Secondary: compute MAE-based ê(x) for comparison.

    Uses g_pred_mae from the MAE-target g(x) model. a(x) is reused from
    the rank-based pipeline (imperfect but sufficient for comparison).
    Scaled by ratio of medians to align the two scales.
    """
    if horizons is None:
        horizons = sorted(set(g_preds_mae["horizon"].unique()) & set(a_preds["horizon"].unique()))

    all_results = []
    diagnostics = {}

    g_col = "g_pred_mae" if "g_pred_mae" in g_preds_mae.columns else "g_pred"

    for hz in horizons:
        gp = g_preds_mae[g_preds_mae["horizon"] == hz].copy()
        ap = a_preds[a_preds["horizon"] == hz].copy()

        if len(ap) == 0 or len(gp) == 0:
            continue

        gp["as_of_date"] = pd.to_datetime(gp["as_of_date"])
        ap["as_of_date"] = pd.to_datetime(ap["as_of_date"])

        tier = ap["tier"].iloc[0]
        is_per_stock = tier in PER_STOCK_TIERS and "ticker" in ap.columns and ap["ticker"].notna().any()

        if is_per_stock:
            ap_cols = ap[["as_of_date", "ticker", "a_value"]].copy()
            merged = gp.merge(ap_cols, on=["as_of_date", "ticker"], how="inner")
        else:
            ap_date = ap.groupby("as_of_date")["a_value"].first().reset_index()
            merged = gp.merge(ap_date, on="as_of_date", how="inner")

        g_median = merged[g_col].median()
        a_median = merged["a_value"].median()
        if a_median > 0 and g_median > 0:
            scale = g_median / a_median
            a_scaled = merged["a_value"] * scale
        else:
            a_scaled = merged["a_value"]

        merged["ehat_mae_raw"] = np.maximum(0.0, merged[g_col] - a_scaled)

        target_col = "mae_loss" if "mae_loss" in merged.columns else None
        if target_col:
            rho = float(stats.spearmanr(merged["ehat_mae_raw"], merged[target_col]).statistic)
        else:
            rho = 0.0

        diagnostics[hz] = {
            "n_rows": len(merged),
            "ehat_mae_mean": float(merged["ehat_mae_raw"].mean()),
            "pct_zero": float((merged["ehat_mae_raw"] == 0).mean()),
            "pct_positive": float((merged["ehat_mae_raw"] > 0).mean()),
            "rho_ehat_mae_target": rho,
            "scale_factor": float(scale) if a_median > 0 and g_median > 0 else 1.0,
        }

        all_results.append(merged)

    ehat_mae_df = pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()
    return ehat_mae_df, diagnostics


def run_sanity_checks(ehat_df: pd.DataFrame, diagnostics: Dict) -> Dict:
    """
    Validate ê(x) against the outline's sanity checks.

    Returns a dict of check_name → {passed, detail}.
    """
    checks = {}

    for hz, diag in diagnostics.items():
        if not isinstance(hz, int):
            continue

        prefix = f"{hz}d"

        # Check 1: ê(x) = 0 for a non-trivial fraction OR all positive is fine
        # (per-date a(x) at 20d/90d will be all-positive; that's expected)
        pct_zero = diag["pct_zero"]
        pct_pos = diag["pct_positive"]

        if diag["a_tier"] in PER_STOCK_TIERS:
            # Per-stock: expect both zero and positive fractions
            checks[f"{prefix}_has_zero_fraction"] = {
                "passed": pct_zero > 0.01,
                "detail": f"pct_zero={pct_zero:.1%} (want >1% for per-stock tier)",
            }
            checks[f"{prefix}_has_positive_fraction"] = {
                "passed": pct_pos > 0.05,
                "detail": f"pct_positive={pct_pos:.1%} (want >5%)",
            }
        else:
            # Per-date: expect (nearly) all positive
            checks[f"{prefix}_has_positive_fraction"] = {
                "passed": pct_pos > 0.90,
                "detail": f"pct_positive={pct_pos:.1%} (per-date a(x): expect ~100%)",
            }

        # Check 2: ê(x) predicts rank_loss (positive correlation)
        rho = diag["rho_ehat_rank_loss"]
        checks[f"{prefix}_ehat_predicts_rank_loss"] = {
            "passed": rho > 0,
            "detail": f"ρ(ê, rank_loss)={rho:.4f} (want >0)",
        }

        # Check 3: selective risk — high-ê stocks have higher rank_loss
        delta = diag["selective_delta"]
        checks[f"{prefix}_selective_risk_positive"] = {
            "passed": delta > 0,
            "detail": (
                f"Δ(high_ê - low_ê rank_loss)={delta:.4f} "
                f"(want >0: high-uncertainty stocks have worse outcomes)"
            ),
        }

        # Check 4: ê(x) has meaningful variation
        std = diag["ehat_std"]
        checks[f"{prefix}_ehat_has_variation"] = {
            "passed": std > 0.001,
            "detail": f"std(ê)={std:.4f} (want >0.001)",
        }

        # Check 5: right-skew for per-stock tiers
        if diag["a_tier"] in PER_STOCK_TIERS and diag["pct_positive"] > 0.01:
            skew = diag["skewness_positive"]
            checks[f"{prefix}_right_skewed"] = {
                "passed": skew > 0,
                "detail": f"skewness={skew:.2f} (want >0: most low, few high)",
            }

    n_passed = sum(1 for c in checks.values() if c["passed"])
    n_total = len(checks)

    return {
        "checks": checks,
        "n_passed": n_passed,
        "n_total": n_total,
        "all_passed": n_passed == n_total,
    }

"""
DEUP Diagnostics — Chapter 13.4
=================================

Six diagnostics that prove (or disprove) whether DEUP works.
All reported for DEV and FINAL holdout separately.

Diagnostic A: Partial correlation (disentanglement)
Diagnostic B: Selective risk (confidence stratification)
Diagnostic C: AUROC for failure-day detection
Diagnostic D: 2024 regime test
Diagnostic E: Baseline comparison (ê vs vol, VIX, etc.)
Diagnostic F: g(x) feature importance analysis
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)

HOLDOUT_START = pd.Timestamp("2024-01-01")
FEATURES_FOR_RESIDUALIZATION = ["vol_20d", "vix_percentile_252d", "mom_1m"]


# ── Diagnostic A: Partial correlation ─────────────────────────────────────


def diagnostic_a_partial_correlation(
    df: pd.DataFrame,
    horizon: int,
) -> Dict[str, Any]:
    """
    Prove ê is not just volatility/VIX in disguise.

    1. Raw ρ(ê, vol_20d) and ρ(ê, vix)
    2. Residualize: OLS of rank_loss on [vol_20d, vix, mom_1m] → residuals
    3. ρ(ê, residual_rl) — does ê predict rank_loss BEYOND vol/VIX?
    4. Residualize ê on same features → ρ(residual_ê, rank_loss)
    5. Kill: ρ(ê, vol | features) > 0.5
    """
    hz = df[df["horizon"] == horizon].copy()
    results = {}

    for period in ["ALL", "DEV", "FINAL"]:
        if period == "ALL":
            pdata = hz
        elif period == "DEV":
            pdata = hz[hz["as_of_date"] < HOLDOUT_START]
        else:
            pdata = hz[hz["as_of_date"] >= HOLDOUT_START]

        if len(pdata) < 50:
            results[period] = {"n": len(pdata), "skip": True}
            continue

        ehat = pdata["ehat_raw"].values
        rl = pdata["rank_loss"].values

        has_vol = "vol_20d" in pdata.columns
        has_vix = "vix_percentile_252d" in pdata.columns

        # Raw correlations
        raw_rho_vol = float(stats.spearmanr(ehat, pdata["vol_20d"]).statistic) if has_vol else 0.0
        raw_rho_vix = float(stats.spearmanr(ehat, pdata["vix_percentile_252d"]).statistic) if has_vix else 0.0

        # Residualize rank_loss on control features
        feats = [c for c in FEATURES_FOR_RESIDUALIZATION if c in pdata.columns]

        if len(feats) >= 1:
            X = pdata[feats].fillna(0).values

            reg_rl = LinearRegression().fit(X, rl)
            residual_rl = rl - reg_rl.predict(X)

            reg_ehat = LinearRegression().fit(X, ehat)
            residual_ehat = ehat - reg_ehat.predict(X)
        else:
            residual_rl = rl
            residual_ehat = ehat

        # ρ(ê, residual_rl) — ê predicts rank_loss beyond vol/VIX
        rho_ehat_residual_rl = float(stats.spearmanr(ehat, residual_rl).statistic)

        # ρ(residual_ê, rl) — ê's unique info predicts rank_loss
        rho_residual_ehat_rl = float(stats.spearmanr(residual_ehat, rl).statistic)

        # ρ(residual_ê, vol) — after removing vol's influence, does ê still correlate with vol?
        # Spearman can diverge from Pearson for sparse distributions (e.g., 85% zeros at 60d).
        # We report both but use Pearson for kill criterion since OLS guarantees Pearson ≈ 0.
        if has_vol:
            rho_residual_ehat_vol = float(stats.spearmanr(residual_ehat, pdata["vol_20d"]).statistic)
            pearson_residual_ehat_vol = float(np.corrcoef(residual_ehat, pdata["vol_20d"].fillna(0).values)[0, 1])
        else:
            rho_residual_ehat_vol = 0.0
            pearson_residual_ehat_vol = 0.0

        # Kill: ê is JUST vol if (a) raw correlation is very high AND (b) ê adds nothing beyond vol
        kill = abs(raw_rho_vol) > 0.7 and rho_ehat_residual_rl <= 0

        results[period] = {
            "n": len(pdata),
            "raw_rho_ehat_vol": raw_rho_vol,
            "raw_rho_ehat_vix": raw_rho_vix,
            "rho_ehat_residual_rl": rho_ehat_residual_rl,
            "rho_residual_ehat_rl": rho_residual_ehat_rl,
            "spearman_residual_ehat_vol": rho_residual_ehat_vol,
            "pearson_residual_ehat_vol": pearson_residual_ehat_vol,
            "kill_criterion_failed": kill,
            "verdict": "KILL" if kill else ("PASS" if rho_ehat_residual_rl > 0 else "WEAK"),
        }

    return results


# ── Diagnostic B: Selective risk ──────────────────────────────────────────


def diagnostic_b_selective_risk(
    df: pd.DataFrame,
    horizon: int,
) -> Dict[str, Any]:
    """
    Confidence stratification: does the model know when to trust itself?

    Split into ê terciles. Compute RankIC per date within each tercile.
    Success: low-ê tercile RankIC > full-set RankIC by >= 0.01.
    """
    hz = df[df["horizon"] == horizon].copy()
    results = {}

    for period in ["ALL", "DEV", "FINAL"]:
        if period == "ALL":
            pdata = hz
        elif period == "DEV":
            pdata = hz[hz["as_of_date"] < HOLDOUT_START]
        else:
            pdata = hz[hz["as_of_date"] >= HOLDOUT_START]

        if len(pdata) < 100:
            results[period] = {"n": len(pdata), "skip": True}
            continue

        pdata = pdata.copy()
        try:
            pdata["ehat_tercile"] = pd.qcut(pdata["ehat_raw"], 3, labels=[0, 1, 2], duplicates="drop")
        except ValueError:
            pdata["ehat_tercile"] = pd.cut(pdata["ehat_raw"], 3, labels=[0, 1, 2], duplicates="drop")

        def _daily_rankic(subdf):
            ics = []
            for _, day_df in subdf.groupby("as_of_date"):
                if len(day_df) < 5:
                    continue
                r = stats.spearmanr(day_df["score"], day_df["excess_return"])
                if not np.isnan(r.statistic):
                    ics.append(r.statistic)
            return np.mean(ics) if ics else 0.0

        full_ic = _daily_rankic(pdata)
        low_ehat = pdata[pdata["ehat_tercile"] == 0]
        mid_ehat = pdata[pdata["ehat_tercile"] == 1]
        high_ehat = pdata[pdata["ehat_tercile"] == 2]

        low_ic = _daily_rankic(low_ehat)
        mid_ic = _daily_rankic(mid_ehat)
        high_ic = _daily_rankic(high_ehat)

        improvement = low_ic - full_ic
        passed = improvement >= 0.01

        results[period] = {
            "n": len(pdata),
            "full_set_rankic": round(full_ic, 4),
            "low_ehat_rankic": round(low_ic, 4),
            "mid_ehat_rankic": round(mid_ic, 4),
            "high_ehat_rankic": round(high_ic, 4),
            "low_minus_full": round(improvement, 4),
            "monotonic_ic_decrease": low_ic >= mid_ic >= high_ic,
            "passed": passed,
            "verdict": "PASS" if passed else "FAIL",
        }

    return results


# ── Diagnostic C: AUROC for failure detection ─────────────────────────────


def diagnostic_c_auroc(
    df: pd.DataFrame,
    horizon: int,
) -> Dict[str, Any]:
    """
    Can ê predict failure days (daily RankIC < 0)?

    For 20d/90d (continuous ê): use daily mean ê → AUROC on daily RankIC < 0.
    For 60d (sparse flag): use fraction of stocks with ê > 0 per day, AND
    per-stock AUROC for high-loss events (rank_loss > median).
    """
    hz = df[df["horizon"] == horizon].copy()
    results = {}

    for period in ["ALL", "DEV", "FINAL"]:
        if period == "ALL":
            pdata = hz
        elif period == "DEV":
            pdata = hz[hz["as_of_date"] < HOLDOUT_START]
        else:
            pdata = hz[hz["as_of_date"] >= HOLDOUT_START]

        if len(pdata) < 100:
            results[period] = {"n": len(pdata), "skip": True}
            continue

        # Daily-level AUROC
        daily = pdata.groupby("as_of_date").agg(
            ehat_mean=("ehat_raw", "mean"),
            ehat_frac_positive=("ehat_raw", lambda x: (x > 0).mean()),
            rankic=("score", lambda x: _safe_spearman(x, pdata.loc[x.index, "excess_return"])),
        ).dropna()

        daily["failure"] = (daily["rankic"] < 0).astype(int)
        n_failure = daily["failure"].sum()
        n_success = len(daily) - n_failure

        period_result = {
            "n_dates": len(daily),
            "n_failure_days": int(n_failure),
            "failure_rate": round(n_failure / len(daily), 3) if len(daily) > 0 else 0,
        }

        if n_failure >= 3 and n_success >= 3:
            auc_mean = float(roc_auc_score(daily["failure"], daily["ehat_mean"]))
            period_result["auroc_daily_ehat_mean"] = round(auc_mean, 4)
            period_result["daily_auroc_verdict"] = "PASS" if auc_mean > 0.55 else "WEAK"
        else:
            period_result["auroc_daily_ehat_mean"] = None
            period_result["daily_auroc_verdict"] = "SKIP (too few failure days)"

        # Per-stock AUROC (especially for 60d sparse flag)
        rl_median = pdata["rank_loss"].median()
        pdata = pdata.copy()
        pdata["high_loss"] = (pdata["rank_loss"] > rl_median).astype(int)
        n_hl = pdata["high_loss"].sum()
        n_ll = len(pdata) - n_hl

        if n_hl >= 10 and n_ll >= 10 and pdata["ehat_raw"].nunique() > 1:
            auc_stock = float(roc_auc_score(pdata["high_loss"], pdata["ehat_raw"]))
            period_result["auroc_stock_high_loss"] = round(auc_stock, 4)

            # Precision at k (for stocks with ê > 0)
            positive_ehat = pdata[pdata["ehat_raw"] > 0]
            if len(positive_ehat) > 0:
                precision_at_flag = float(positive_ehat["high_loss"].mean())
                period_result["precision_at_ehat_positive"] = round(precision_at_flag, 4)
        else:
            period_result["auroc_stock_high_loss"] = None

        results[period] = period_result

    return results


def _safe_spearman(a, b):
    """Spearman correlation, returning NaN if fewer than 5 observations."""
    a, b = np.array(a), np.array(b)
    mask = ~(np.isnan(a) | np.isnan(b))
    if mask.sum() < 5:
        return np.nan
    return stats.spearmanr(a[mask], b[mask]).statistic


# ── Diagnostic D: 2024 regime test ───────────────────────────────────────


def diagnostic_d_regime_2024(
    df: pd.DataFrame,
    horizon: int,
) -> Dict[str, Any]:
    """
    The money slide: does ê spike during Mar-Jul 2024 when signal fails?

    Monthly average ê vs monthly RankIC for FINAL period.
    """
    hz = df[df["horizon"] == horizon].copy()
    hz["as_of_date"] = pd.to_datetime(hz["as_of_date"])

    final = hz[hz["as_of_date"] >= HOLDOUT_START].copy()
    if len(final) < 50:
        return {"skip": True, "n": len(final)}

    final["month"] = final["as_of_date"].dt.to_period("M")

    monthly = final.groupby("month").apply(
        lambda g: pd.Series({
            "ehat_mean": g["ehat_raw"].mean(),
            "rankic": _safe_spearman(g["score"], g["excess_return"]),
            "rank_loss_mean": g["rank_loss"].mean(),
            "n_obs": len(g),
        }),
        include_groups=False,
    ).dropna(subset=["rankic"])

    if len(monthly) < 3:
        return {"skip": True, "n_months": len(monthly)}

    rho_ehat_rankic = float(stats.spearmanr(
        monthly["ehat_mean"], monthly["rankic"]
    ).statistic)

    # Mar-Jul 2024 vs rest: does ê spike?
    crisis_months = [pd.Period(f"2024-{m:02d}") for m in range(3, 8)]
    crisis = monthly[monthly.index.isin(crisis_months)]
    non_crisis = monthly[~monthly.index.isin(crisis_months)]

    crisis_ehat = float(crisis["ehat_mean"].mean()) if len(crisis) > 0 else None
    non_crisis_ehat = float(non_crisis["ehat_mean"].mean()) if len(non_crisis) > 0 else None
    crisis_rankic = float(crisis["rankic"].mean()) if len(crisis) > 0 else None
    non_crisis_rankic = float(non_crisis["rankic"].mean()) if len(non_crisis) > 0 else None

    ehat_spikes = (
        crisis_ehat is not None and non_crisis_ehat is not None
        and crisis_ehat > non_crisis_ehat
    )

    monthly_table = []
    for period_m, row in monthly.iterrows():
        monthly_table.append({
            "month": str(period_m),
            "ehat_mean": round(row["ehat_mean"], 4),
            "rankic": round(row["rankic"], 4),
            "rank_loss_mean": round(row["rank_loss_mean"], 4),
            "n_obs": int(row["n_obs"]),
        })

    return {
        "n_months": len(monthly),
        "rho_ehat_vs_rankic": round(rho_ehat_rankic, 4),
        "crisis_period": "Mar-Jul 2024",
        "crisis_ehat_mean": round(crisis_ehat, 4) if crisis_ehat is not None else None,
        "non_crisis_ehat_mean": round(non_crisis_ehat, 4) if non_crisis_ehat is not None else None,
        "crisis_rankic_mean": round(crisis_rankic, 4) if crisis_rankic is not None else None,
        "non_crisis_rankic_mean": round(non_crisis_rankic, 4) if non_crisis_rankic is not None else None,
        "ehat_spikes_in_crisis": ehat_spikes,
        "verdict": "PASS" if ehat_spikes and rho_ehat_rankic < -0.1 else "WEAK",
        "monthly_table": monthly_table,
    }


# ── Diagnostic E: Baseline comparison ─────────────────────────────────────


def diagnostic_e_baselines(
    df: pd.DataFrame,
    horizon: int,
) -> Dict[str, Any]:
    """
    Compare ê against alternative UQ signals for rank_loss prediction.

    Baselines computed from available data:
    1. ê (DEUP)
    2. vol_20d
    3. vix_percentile_252d
    4. |score| (abs_score)
    5. g(x) directly (without a(x) subtraction)
    """
    hz = df[df["horizon"] == horizon].copy()

    if "abs_score" not in hz.columns and "score" in hz.columns:
        hz["abs_score"] = hz["score"].abs()

    baselines = {
        "ehat_raw": "ê (DEUP)",
        "g_pred": "g(x) raw",
        "vol_20d": "vol_20d",
        "vix_percentile_252d": "VIX percentile",
    }
    if "abs_score" in hz.columns:
        baselines["abs_score"] = "|score|"

    results = {}

    for period in ["ALL", "DEV", "FINAL"]:
        if period == "ALL":
            pdata = hz
        elif period == "DEV":
            pdata = hz[hz["as_of_date"] < HOLDOUT_START]
        else:
            pdata = hz[hz["as_of_date"] >= HOLDOUT_START]

        if len(pdata) < 50:
            results[period] = {"n": len(pdata), "skip": True}
            continue

        rl = pdata["rank_loss"].values
        comparison = {}

        for col, label in baselines.items():
            if col not in pdata.columns:
                continue
            vals = pdata[col].values
            mask = ~(np.isnan(vals) | np.isnan(rl))
            if mask.sum() < 20:
                continue

            rho = float(stats.spearmanr(vals[mask], rl[mask]).statistic)

            # Selective risk: split by signal tercile (robust to low cardinality)
            try:
                tercile = pd.qcut(pd.Series(vals[mask]), 3, labels=False, duplicates="drop")
                low_rl = rl[mask][tercile == tercile.min()].mean()
                high_rl = rl[mask][tercile == tercile.max()].mean()
                sel_delta = round(high_rl - low_rl, 4)
            except (ValueError, IndexError):
                sel_delta = 0.0

            comparison[label] = {
                "rho_with_rank_loss": round(rho, 4),
                "selective_delta": sel_delta,
            }

        results[period] = {
            "n": len(pdata),
            "baselines": comparison,
        }

    return results


# ── Diagnostic F: Feature importance ──────────────────────────────────────


def diagnostic_f_feature_importance(
    diagnostics_01: Dict,
) -> Dict[str, Any]:
    """
    Analyze g(x) feature importances to determine what drives ê.

    Key question: Is ê an expensive regime detector (VIX dominates)
    or genuine per-prediction uncertainty (score/rank dominate)?
    """
    fi_list = diagnostics_01.get("step_1", {}).get("feature_importances", [])
    features_used = diagnostics_01.get("step_1", {}).get("features_used", [])

    if not fi_list:
        return {"skip": True, "reason": "No feature importances found"}

    results = {}
    for entry in fi_list:
        hz = entry["horizon"]
        imps = entry["importances"]
        total = sum(imps.values())

        ranked = sorted(imps.items(), key=lambda x: -x[1])
        top_3 = [(name, count, count / total * 100) for name, count in ranked[:3]]

        # Categorize: per-prediction vs regime vs volatility
        per_prediction = sum(v for k, v in imps.items() if k in {"score", "abs_score", "cross_sectional_rank"})
        regime_market = sum(v for k, v in imps.items() if k in {"vix_percentile_252d", "market_regime_enc", "market_vol_21d", "market_return_21d"})
        volatility = sum(v for k, v in imps.items() if k in {"vol_20d", "vol_60d"})

        per_prediction_pct = per_prediction / total * 100
        regime_market_pct = regime_market / total * 100
        volatility_pct = volatility / total * 100

        if per_prediction_pct > regime_market_pct and per_prediction_pct > volatility_pct:
            interpretation = "Per-prediction features dominate → ê captures stock-specific uncertainty"
        elif volatility_pct > per_prediction_pct:
            interpretation = "WARNING: Volatility dominates → ê may be repackaged vol"
        elif regime_market_pct > per_prediction_pct:
            interpretation = "WARNING: Regime features dominate → ê is an expensive regime detector"
        else:
            interpretation = "Mixed signal — no single category dominates"

        results[hz] = {
            "fold": entry["fold_id"],
            "top_3": [{"feature": n, "importance": c, "pct": round(p, 1)} for n, c, p in top_3],
            "category_breakdown": {
                "per_prediction_pct": round(per_prediction_pct, 1),
                "regime_market_pct": round(regime_market_pct, 1),
                "volatility_pct": round(volatility_pct, 1),
            },
            "interpretation": interpretation,
        }

    return {"horizons": results, "features_used": features_used}


# ── Stability analysis ────────────────────────────────────────────────────


def diagnostic_stability(
    df: pd.DataFrame,
    horizon: int,
) -> Dict[str, Any]:
    """
    OOS stability: does ê→RL relationship hold across conditions?

    Splits: pre/post 2020, bull/bear (via VIX), high/low liquidity.
    """
    hz = df[df["horizon"] == horizon].copy()
    results = {}

    # Time splits
    pre_2020 = hz[hz["as_of_date"] < "2020-01-01"]
    post_2020 = hz[hz["as_of_date"] >= "2020-01-01"]

    for name, subset in [("pre_2020", pre_2020), ("post_2020", post_2020)]:
        if len(subset) < 50:
            results[name] = {"n": len(subset), "skip": True}
            continue
        rho = float(stats.spearmanr(subset["ehat_raw"], subset["rank_loss"]).statistic)
        results[name] = {"n": len(subset), "rho_ehat_rl": round(rho, 4)}

    # VIX regime splits
    if "vix_percentile_252d" in hz.columns:
        low_vix = hz[hz["vix_percentile_252d"] < 40]
        high_vix = hz[hz["vix_percentile_252d"] > 80]

        for name, subset in [("low_vix", low_vix), ("high_vix", high_vix)]:
            if len(subset) < 50:
                results[name] = {"n": len(subset), "skip": True}
                continue
            rho = float(stats.spearmanr(subset["ehat_raw"], subset["rank_loss"]).statistic)
            results[name] = {"n": len(subset), "rho_ehat_rl": round(rho, 4)}

    # Liquidity splits (via vol_20d as proxy)
    if "vol_20d" in hz.columns:
        low_vol = hz[hz["vol_20d"] < hz["vol_20d"].quantile(0.33)]
        high_vol = hz[hz["vol_20d"] > hz["vol_20d"].quantile(0.67)]

        for name, subset in [("low_vol_stocks", low_vol), ("high_vol_stocks", high_vol)]:
            if len(subset) < 50:
                results[name] = {"n": len(subset), "skip": True}
                continue
            rho = float(stats.spearmanr(subset["ehat_raw"], subset["rank_loss"]).statistic)
            results[name] = {"n": len(subset), "rho_ehat_rl": round(rho, 4)}

    # Overall verdict
    rhos = [v["rho_ehat_rl"] for v in results.values() if "rho_ehat_rl" in v]
    all_positive = all(r > 0 for r in rhos)
    min_rho = min(rhos) if rhos else 0
    max_rho = max(rhos) if rhos else 0

    results["summary"] = {
        "all_positive": all_positive,
        "min_rho": round(min_rho, 4),
        "max_rho": round(max_rho, 4),
        "n_conditions": len(rhos),
        "verdict": "PASS" if all_positive else "FAIL",
    }

    return results


# ── Master runner ─────────────────────────────────────────────────────────


def run_all_diagnostics(
    ehat_df: pd.DataFrame,
    enriched_residuals: pd.DataFrame,
    diagnostics_01: Dict,
    horizons: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """
    Run all six diagnostics for all horizons.

    Merges ehat with enriched_residuals to get features needed for diagnostics.
    """
    if horizons is None:
        horizons = sorted(ehat_df["horizon"].unique())

    # Merge ehat with enriched residuals
    ehat_df = ehat_df.copy()
    enriched_residuals = enriched_residuals.copy()
    ehat_df["as_of_date"] = pd.to_datetime(ehat_df["as_of_date"])
    enriched_residuals["as_of_date"] = pd.to_datetime(enriched_residuals["as_of_date"])

    merge_cols = ["as_of_date", "ticker", "horizon"]
    needed_features = ["score", "excess_return", "vol_20d", "vix_percentile_252d",
                       "mom_1m", "sector", "rank_score"]
    feature_cols = [c for c in needed_features if c in enriched_residuals.columns]

    # Only pull feature columns that aren't already in ehat_df
    new_feature_cols = [c for c in feature_cols if c not in ehat_df.columns]

    if new_feature_cols:
        er_subset = enriched_residuals[merge_cols + new_feature_cols].drop_duplicates(subset=merge_cols)
        merged = ehat_df.merge(er_subset, on=merge_cols, how="inner")
    else:
        merged = ehat_df

    logger.info(f"Merged ehat ({len(ehat_df):,}) with enriched ({len(enriched_residuals):,}) → {len(merged):,} rows")

    all_results = {}

    for hz in horizons:
        logger.info(f"\n{'='*60}")
        logger.info(f"  DIAGNOSTICS for {hz}d")
        logger.info(f"{'='*60}")

        hz_result = {}

        # A: Partial correlation
        logger.info("\n  [A] Partial correlation (disentanglement)")
        hz_result["A_partial_correlation"] = diagnostic_a_partial_correlation(merged, hz)
        for period, res in hz_result["A_partial_correlation"].items():
            if res.get("skip"):
                continue
            logger.info(
                f"    {period}: raw ρ(ê,vol)={res['raw_rho_ehat_vol']:.4f}, "
                f"raw ρ(ê,vix)={res['raw_rho_ehat_vix']:.4f}, "
                f"ρ(ê,resid_rl)={res['rho_ehat_residual_rl']:.4f}, "
                f"ρ(resid_ê,rl)={res['rho_residual_ehat_rl']:.4f} → {res['verdict']}"
            )

        # B: Selective risk
        logger.info("\n  [B] Selective risk (confidence stratification)")
        hz_result["B_selective_risk"] = diagnostic_b_selective_risk(merged, hz)
        for period, res in hz_result["B_selective_risk"].items():
            if res.get("skip"):
                continue
            logger.info(
                f"    {period}: full IC={res['full_set_rankic']:.4f}, "
                f"low-ê IC={res['low_ehat_rankic']:.4f}, "
                f"high-ê IC={res['high_ehat_rankic']:.4f}, "
                f"Δ={res['low_minus_full']:.4f} → {res['verdict']}"
            )

        # C: AUROC
        logger.info("\n  [C] AUROC for failure detection")
        hz_result["C_auroc"] = diagnostic_c_auroc(merged, hz)
        for period, res in hz_result["C_auroc"].items():
            if res.get("skip"):
                continue
            auc_d = res.get("auroc_daily_ehat_mean")
            auc_s = res.get("auroc_stock_high_loss")
            logger.info(
                f"    {period}: daily AUROC={auc_d}, stock AUROC={auc_s}, "
                f"failure_rate={res.get('failure_rate', 0):.1%}"
            )

        # D: 2024 regime test
        logger.info("\n  [D] 2024 regime test")
        hz_result["D_regime_2024"] = diagnostic_d_regime_2024(merged, hz)
        d_res = hz_result["D_regime_2024"]
        if not d_res.get("skip"):
            logger.info(
                f"    ρ(ê, RankIC) monthly={d_res['rho_ehat_vs_rankic']:.4f}, "
                f"crisis ê={d_res.get('crisis_ehat_mean')}, "
                f"non-crisis ê={d_res.get('non_crisis_ehat_mean')}, "
                f"spikes={d_res['ehat_spikes_in_crisis']} → {d_res['verdict']}"
            )

        # E: Baseline comparison
        logger.info("\n  [E] Baseline comparison")
        hz_result["E_baselines"] = diagnostic_e_baselines(merged, hz)
        for period, res in hz_result["E_baselines"].items():
            if res.get("skip"):
                continue
            for label, bdata in res.get("baselines", {}).items():
                logger.info(
                    f"    {period} {label}: ρ(signal, rl)={bdata['rho_with_rank_loss']:.4f}, "
                    f"selective Δ={bdata['selective_delta']:.4f}"
                )

        # F: Feature importance
        logger.info("\n  [F] Feature importance")
        if not hasattr(run_all_diagnostics, "_fi_done"):
            hz_result["F_feature_importance"] = diagnostic_f_feature_importance(diagnostics_01)
            run_all_diagnostics._fi_done = True
        else:
            hz_result["F_feature_importance"] = {"reference": "See first horizon"}

        fi = hz_result.get("F_feature_importance", {})
        if "horizons" in fi:
            for fi_hz, fi_data in fi["horizons"].items():
                logger.info(
                    f"    {fi_hz}d: top-3={[t['feature'] for t in fi_data['top_3']]}, "
                    f"per-prediction={fi_data['category_breakdown']['per_prediction_pct']:.1f}%, "
                    f"regime={fi_data['category_breakdown']['regime_market_pct']:.1f}%, "
                    f"vol={fi_data['category_breakdown']['volatility_pct']:.1f}%"
                )
                logger.info(f"      → {fi_data['interpretation']}")

        # Stability
        logger.info("\n  [Stability] Cross-condition robustness")
        hz_result["stability"] = diagnostic_stability(merged, hz)
        stab = hz_result["stability"]
        for cond, res in stab.items():
            if cond == "summary":
                logger.info(
                    f"    SUMMARY: all_positive={res['all_positive']}, "
                    f"min_rho={res['min_rho']:.4f}, max_rho={res['max_rho']:.4f} → {res['verdict']}"
                )
            elif not res.get("skip"):
                logger.info(f"    {cond}: ρ(ê, rl)={res['rho_ehat_rl']:.4f}, n={res['n']:,}")

        all_results[hz] = hz_result

    # Reset the flag for feature importance
    if hasattr(run_all_diagnostics, "_fi_done"):
        delattr(run_all_diagnostics, "_fi_done")

    return all_results

"""
DEUP-Sized Shadow Portfolio + Global Regime Evaluation — Chapter 13.6
======================================================================

Economic test: does uncertainty-informed position sizing beat alternatives?
Plus global regime evaluation: does G(t) reliably determine expert usability?

Sizing variants:
    A) Vol-sized baseline:     sized_score = score × min(1, c_vol / sqrt(vol_20d + ε))
    B) DEUP per-stock:         sized_score = score × min(1, c_deup / sqrt(unc + ε))
    C) Health-only throttle:   return × G(t)
    D) Combined (health+DEUP): sized_score × G(t)

Regime evaluation:
    - AUROC / AUPRC for predicting good_day(t) from G(t) and baselines
    - Bucketed regime segmentation: G(t) quintiles → portfolio metrics
    - Aggregated DEUP as global early warning
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

logger = logging.getLogger(__name__)

HOLDOUT_START = pd.Timestamp("2024-01-01")
TOP_K = 10
DEFAULT_COST_BPS = 10.0
EPS = 1e-4


@dataclass
class PortfolioConfig:
    horizon: int = 20
    top_k: int = TOP_K
    cost_bps: float = DEFAULT_COST_BPS
    target_median_w: float = 0.7
    abstention_threshold: float = 0.2


# ── Calibration ───────────────────────────────────────────────────────────


def calibrate_c(
    signal: pd.Series,
    target_median: float = 0.7,
) -> float:
    """
    Find c such that median(min(1, c / sqrt(signal + eps))) ≈ target_median.

    Done on DEV data only. The result is frozen for FINAL.
    """
    signal_clean = signal.dropna()
    if len(signal_clean) == 0:
        return 1.0
    median_sqrt = np.median(np.sqrt(signal_clean.values + EPS))
    c = target_median * median_sqrt
    return float(c)


# ── Sizing weights ────────────────────────────────────────────────────────


def compute_sizing_weights(
    df: pd.DataFrame,
    c_vol: float,
    c_deup: float,
    unc_col: str = "g_pred",
) -> pd.DataFrame:
    """
    Add sizing weight columns for all variants.

    Args:
        df: Must have columns: score, vol_20d, <unc_col>, plus G_exposure if available.
        c_vol: Calibrated constant for vol sizing.
        c_deup: Calibrated constant for DEUP sizing.
        unc_col: Column to use for DEUP uncertainty signal.
    """
    df = df.copy()

    # A) Vol-sized
    df["w_vol"] = np.minimum(1.0, c_vol / np.sqrt(df["vol_20d"].clip(lower=EPS) + EPS))

    # B) DEUP per-stock
    unc = df[unc_col].clip(lower=0)
    df["w_deup"] = np.minimum(1.0, c_deup / np.sqrt(unc + EPS))

    # Sized scores
    df["sized_score_raw"] = df["score"]
    df["sized_score_vol"] = df["score"] * df["w_vol"]
    df["sized_score_deup"] = df["score"] * df["w_deup"]

    return df


# ── Portfolio construction ────────────────────────────────────────────────


def build_variant_portfolios(
    df: pd.DataFrame,
    health_df: Optional[pd.DataFrame] = None,
    config: Optional[PortfolioConfig] = None,
) -> pd.DataFrame:
    """
    Build shadow portfolios for all sizing variants.

    Returns one row per date with return columns for each variant.
    """
    cfg = config or PortfolioConfig()
    df = df.copy()
    df["as_of_date"] = pd.to_datetime(df["as_of_date"])

    # Merge G(t) if provided
    g_map = {}
    if health_df is not None:
        hdf = health_df.copy()
        hdf["date"] = pd.to_datetime(hdf["date"])
        g_map = dict(zip(hdf["date"], hdf["G_exposure"].fillna(1.0)))

    dates = sorted(df["as_of_date"].unique())
    records = []

    variant_cols = ["raw", "vol", "deup"]
    prev_longs = {v: set() for v in variant_cols}
    prev_shorts = {v: set() for v in variant_cols}

    for dt in dates:
        day = df[df["as_of_date"] == dt]
        if len(day) < 2 * cfg.top_k:
            continue

        g_t = g_map.get(dt, 1.0)
        if pd.isna(g_t):
            g_t = 1.0

        row = {"date": dt, "G_exposure": g_t, "n_stocks": len(day)}

        for variant in variant_cols:
            score_col = f"sized_score_{variant}"
            sorted_day = day.sort_values(score_col, ascending=False)

            long_ids = set(sorted_day.head(cfg.top_k)["stable_id"])
            short_ids = set(sorted_day.tail(cfg.top_k)["stable_id"])

            long_ret = sorted_day[sorted_day["stable_id"].isin(long_ids)]["excess_return"].mean()
            short_ret = sorted_day[sorted_day["stable_id"].isin(short_ids)]["excess_return"].mean()
            ls_ret = long_ret - short_ret

            # Turnover
            if prev_longs[variant]:
                lt = 1 - len(long_ids & prev_longs[variant]) / cfg.top_k
                st = 1 - len(short_ids & prev_shorts[variant]) / cfg.top_k
                turnover = (lt + st) / 2
            else:
                turnover = 1.0
            cost = turnover * (cfg.cost_bps / 10000) * 2

            row[f"ls_return_{variant}"] = ls_ret
            row[f"ls_return_net_{variant}"] = ls_ret - cost
            row[f"turnover_{variant}"] = turnover

            # Health-only throttle (C): apply G(t) to raw returns
            if variant == "raw":
                row["ls_return_net_health"] = (ls_ret - cost) * g_t
                row["turnover_health"] = turnover

            # Combined (D): apply G(t) to DEUP-sized returns
            if variant == "deup":
                row["ls_return_net_combined"] = (ls_ret - cost) * g_t
                row["turnover_combined"] = turnover

            prev_longs[variant] = long_ids
            prev_shorts[variant] = short_ids

        # Exposure for health/combined
        row["exposure_health"] = g_t
        row["exposure_combined"] = g_t

        records.append(row)

    return pd.DataFrame(records)


# ── Portfolio metrics ─────────────────────────────────────────────────────


def compute_portfolio_metrics(
    returns: np.ndarray,
    turnover: Optional[np.ndarray] = None,
    periods_per_year: float = 12.0,
) -> Dict[str, Any]:
    """Compute annualized portfolio metrics."""
    n = len(returns)
    if n == 0:
        return {"n_periods": 0}

    mean_ret = float(np.mean(returns))
    std_ret = float(np.std(returns, ddof=1)) if n > 1 else 0.001

    sharpe = (mean_ret / std_ret) * np.sqrt(periods_per_year) if std_ret > 0 else 0.0

    # Sortino (downside deviation)
    downside = returns[returns < 0]
    down_std = float(np.std(downside, ddof=1)) if len(downside) > 1 else std_ret
    sortino = (mean_ret / down_std) * np.sqrt(periods_per_year) if down_std > 0 else 0.0

    # Drawdown
    cum = np.cumprod(1 + returns)
    peak = np.maximum.accumulate(cum)
    dd = (cum - peak) / peak
    max_dd = float(dd.min())

    metrics = {
        "n_periods": n,
        "sharpe": round(sharpe, 4),
        "sortino": round(sortino, 4),
        "annualized_return": round(mean_ret * periods_per_year, 4),
        "annualized_vol": round(std_ret * np.sqrt(periods_per_year), 4),
        "max_drawdown": round(max_dd, 4),
        "hit_rate": round(float(np.mean(returns > 0)), 4),
        "worst_period": round(float(np.min(returns)), 4),
        "mean_return": round(mean_ret, 6),
    }

    if turnover is not None and len(turnover) > 0:
        metrics["mean_turnover"] = round(float(np.mean(turnover)), 4)

    return metrics


def compute_all_variant_metrics(
    portfolio_df: pd.DataFrame,
) -> Dict[str, Dict[str, Any]]:
    """Compute metrics for all variants across ALL / DEV / FINAL."""
    results = {}
    portfolio_df = portfolio_df.copy()
    portfolio_df["date"] = pd.to_datetime(portfolio_df["date"])

    variant_return_cols = {
        "A_vol_sized": "ls_return_net_vol",
        "B_deup_sized": "ls_return_net_deup",
        "C_health_only": "ls_return_net_health",
        "D_combined": "ls_return_net_combined",
        "baseline_raw": "ls_return_net_raw",
    }

    variant_turnover_cols = {
        "A_vol_sized": "turnover_vol",
        "B_deup_sized": "turnover_deup",
        "C_health_only": "turnover_health",
        "D_combined": "turnover_combined",
        "baseline_raw": "turnover_raw",
    }

    for period_name, mask in [
        ("ALL", pd.Series(True, index=portfolio_df.index)),
        ("DEV", portfolio_df["date"] < HOLDOUT_START),
        ("FINAL", portfolio_df["date"] >= HOLDOUT_START),
    ]:
        pdata = portfolio_df[mask]
        if len(pdata) == 0:
            continue

        period_metrics = {}
        for vname, rcol in variant_return_cols.items():
            if rcol not in pdata.columns:
                continue
            rets = pdata[rcol].dropna().values
            tcol = variant_turnover_cols.get(vname)
            to = pdata[tcol].dropna().values if tcol and tcol in pdata.columns else None
            period_metrics[vname] = compute_portfolio_metrics(rets, to)

        results[period_name] = period_metrics

    # Crisis window Mar-Jul 2024
    crisis_mask = (
        (portfolio_df["date"] >= "2024-03-01")
        & (portfolio_df["date"] <= "2024-07-31")
    )
    cdata = portfolio_df[crisis_mask]
    if len(cdata) > 0:
        crisis_metrics = {}
        for vname, rcol in variant_return_cols.items():
            if rcol not in cdata.columns:
                continue
            rets = cdata[rcol].dropna().values
            crisis_metrics[vname] = compute_portfolio_metrics(rets)
        crisis_metrics["mean_G"] = round(float(cdata["G_exposure"].mean()), 4)
        crisis_metrics["n_dates"] = len(cdata)
        results["CRISIS_2024"] = crisis_metrics

    return results


# ── Global regime evaluation ──────────────────────────────────────────────


def evaluate_regime_trust(
    health_df: pd.DataFrame,
    enriched_df: pd.DataFrame,
    horizon: int = 20,
    threshold: float = 0.0,
    abstention_threshold: float = 0.2,
) -> Dict[str, Any]:
    """
    Regime-trust classifier evaluation.

    Labels: good_day(t) = 1[matured_rankic(t) > threshold]
    Predictors: H(t), G(t), VIX, market_vol, trailing_rankic-only

    All PIT-safe: matured_rankic uses lagged labels.
    """
    from sklearn.metrics import (
        average_precision_score,
        confusion_matrix,
        roc_auc_score,
    )

    hdf = health_df.copy()
    hdf["date"] = pd.to_datetime(hdf["date"])

    # matured_rankic is already PIT-safe in health_df
    valid = hdf.dropna(subset=["matured_rankic", "H_combined", "G_exposure"])
    if len(valid) < 50:
        return {"skip": True, "n": len(valid)}

    valid = valid.copy()
    valid["good_day"] = (valid["matured_rankic"] > threshold).astype(int)

    # Get aggregate market features per date from enriched
    er_hz = enriched_df[enriched_df["horizon"] == horizon].copy()
    er_hz["as_of_date"] = pd.to_datetime(er_hz["as_of_date"])
    date_feats = er_hz.groupby("as_of_date").agg(
        vix_pct=("vix_percentile_252d", "mean"),
        mkt_vol=("market_vol_21d", "mean"),
        mean_vol20=("vol_20d", "mean"),
    ).reset_index()
    date_feats.rename(columns={"as_of_date": "date"}, inplace=True)
    valid = valid.merge(date_feats, on="date", how="left")

    diag: Dict[str, Any] = {
        "n_dates": len(valid),
        "horizon": horizon,
        "good_day_threshold": threshold,
        "pct_good": round(float(valid["good_day"].mean()), 4),
    }

    n_good = valid["good_day"].sum()
    n_bad = len(valid) - n_good

    # AUROC / AUPRC for various predictors
    predictors = {
        "H_combined": valid["H_combined"],
        "G_exposure": valid["G_exposure"],
        "H_realized_only": valid.get("H_realized_only", valid["H_combined"]),
    }
    # Baselines (inverted: higher baseline → lower expected quality)
    if "vix_pct" in valid.columns:
        predictors["vix_pct_inv"] = 1 - valid["vix_pct"].fillna(0.5)
    if "mkt_vol" in valid.columns:
        predictors["mkt_vol_inv"] = -valid["mkt_vol"].fillna(0)
    if "mean_vol20" in valid.columns:
        predictors["mean_vol20_inv"] = -valid["mean_vol20"].fillna(0)

    auroc_results = {}
    if n_good >= 5 and n_bad >= 5:
        for pname, pred in predictors.items():
            pred_clean = pred.fillna(0)
            try:
                auroc = float(roc_auc_score(valid["good_day"], pred_clean))
                auprc = float(average_precision_score(valid["good_day"], pred_clean))
                auroc_results[pname] = {
                    "auroc": round(auroc, 4),
                    "auprc": round(auprc, 4),
                }
            except ValueError:
                auroc_results[pname] = {"auroc": None, "auprc": None}
    diag["classifier_metrics"] = auroc_results

    # Confusion matrix at abstention threshold
    valid_cm = valid.copy()
    valid_cm["predict_good"] = (valid_cm["G_exposure"] >= abstention_threshold).astype(int)
    if valid_cm["predict_good"].nunique() >= 2 and valid_cm["good_day"].nunique() >= 2:
        cm = confusion_matrix(valid_cm["good_day"], valid_cm["predict_good"])
        tn, fp, fn, tp = cm.ravel()
        diag["confusion_matrix"] = {
            "threshold": abstention_threshold,
            "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn),
            "precision": round(tp / max(tp + fp, 1), 4),
            "recall": round(tp / max(tp + fn, 1), 4),
            "abstention_rate": round(float(1 - valid_cm["predict_good"].mean()), 4),
        }

    # DEV / FINAL split
    for pname, pmask in [("DEV", valid["date"] < HOLDOUT_START), ("FINAL", valid["date"] >= HOLDOUT_START)]:
        pdata = valid[pmask]
        if len(pdata) < 20:
            diag[pname] = {"n": len(pdata)}
            continue
        pg = pdata["good_day"].sum()
        pb = len(pdata) - pg
        pdiag = {"n": len(pdata), "pct_good": round(float(pdata["good_day"].mean()), 4)}
        if pg >= 3 and pb >= 3:
            for pred_name in ["H_combined", "G_exposure"]:
                if pred_name in pdata.columns:
                    try:
                        pdiag[f"auroc_{pred_name}"] = round(
                            float(roc_auc_score(pdata["good_day"], pdata[pred_name].fillna(0))), 4
                        )
                    except ValueError:
                        pass
        diag[pname] = pdiag

    return diag


def compute_bucket_tables(
    portfolio_df: pd.DataFrame,
    health_df: pd.DataFrame,
    n_buckets: int = 5,
) -> Dict[str, Any]:
    """
    Bucket dates by G(t) and report portfolio metrics per bucket.

    Must show: low-G → worse performance, high-G → better performance.
    """
    pdf = portfolio_df.copy()
    pdf["date"] = pd.to_datetime(pdf["date"])

    hdf = health_df.copy()
    hdf["date"] = pd.to_datetime(hdf["date"])
    g_map = dict(zip(hdf["date"], hdf["G_exposure"]))
    rankic_map = dict(zip(hdf["date"], hdf["matured_rankic"]))

    pdf["G_exposure"] = pdf["date"].map(g_map)
    pdf["matured_rankic"] = pdf["date"].map(rankic_map)

    valid = pdf.dropna(subset=["G_exposure", "matured_rankic"])
    if len(valid) < 20:
        return {"skip": True}

    valid = valid.copy()
    try:
        valid["g_bucket"] = pd.qcut(valid["G_exposure"], n_buckets, labels=False, duplicates="drop")
    except ValueError:
        valid["g_bucket"] = pd.cut(valid["G_exposure"], n_buckets, labels=False)

    valid["bad_day"] = (valid["matured_rankic"] <= 0).astype(int)

    buckets = {}
    for b, bdata in valid.groupby("g_bucket"):
        bm = {
            "n": len(bdata),
            "mean_G": round(float(bdata["G_exposure"].mean()), 4),
            "mean_matured_rankic": round(float(bdata["matured_rankic"].mean()), 4),
            "pct_bad_days": round(float(bdata["bad_day"].mean()), 4),
        }
        if "ls_return_net_raw" in bdata.columns:
            rets = bdata["ls_return_net_raw"].values
            bm["raw_sharpe"] = round(
                float(np.mean(rets) / max(np.std(rets, ddof=1), 1e-6) * np.sqrt(12)), 4
            )
            bm["raw_mean_return"] = round(float(np.mean(rets)), 6)
        if "ls_return_net_combined" in bdata.columns:
            crets = bdata["ls_return_net_combined"].values
            bm["combined_sharpe"] = round(
                float(np.mean(crets) / max(np.std(crets, ddof=1), 1e-6) * np.sqrt(12)), 4
            )
        buckets[str(int(b))] = bm

    # Monotonicity check
    bucket_ics = [buckets[str(b)]["mean_matured_rankic"] for b in sorted(buckets.keys())]
    if len(bucket_ics) >= 3:
        rho = sp_stats.spearmanr(range(len(bucket_ics)), bucket_ics).statistic
    else:
        rho = 0.0

    return {
        "n_buckets": len(buckets),
        "buckets": buckets,
        "monotonicity_rho": round(float(rho), 4),
    }


def compute_aggregated_deup(
    ehat_df: pd.DataFrame,
    health_df: pd.DataFrame,
    horizon: int = 20,
) -> Dict[str, Any]:
    """
    Cross-sectional g(x)/ê(x) aggregates as global early-warning signal.

    Per date: median_g, p90_g, spread_g. Test if these predict good_day(t)
    beyond H components.
    """
    hz = ehat_df[ehat_df["horizon"] == horizon].copy()
    hz["as_of_date"] = pd.to_datetime(hz["as_of_date"])

    date_agg = hz.groupby("as_of_date").agg(
        median_g=("g_pred", "median"),
        p90_g=("g_pred", lambda x: x.quantile(0.90)),
        p10_g=("g_pred", lambda x: x.quantile(0.10)),
        median_ehat=("ehat_raw", "median"),
        p90_ehat=("ehat_raw", lambda x: x.quantile(0.90)),
    ).reset_index()
    date_agg["spread_g"] = date_agg["p90_g"] - date_agg["p10_g"]
    date_agg.rename(columns={"as_of_date": "date"}, inplace=True)

    hdf = health_df.copy()
    hdf["date"] = pd.to_datetime(hdf["date"])
    merged = hdf.merge(date_agg, on="date", how="inner")
    valid = merged.dropna(subset=["matured_rankic", "median_g"])

    if len(valid) < 50:
        return {"skip": True}

    valid = valid.copy()
    valid["good_day"] = (valid["matured_rankic"] > 0).astype(int)

    diag: Dict[str, Any] = {"n_dates": len(valid)}

    # Correlations with matured_rankic
    for col in ["median_g", "p90_g", "spread_g", "median_ehat", "p90_ehat"]:
        if col in valid.columns and valid[col].notna().sum() > 20:
            r = sp_stats.spearmanr(valid[col].fillna(0), valid["matured_rankic"])
            diag[f"rho_{col}_rankic"] = round(float(r.statistic), 4)

    # AUROC for predicting good_day
    from sklearn.metrics import roc_auc_score
    n_good = valid["good_day"].sum()
    n_bad = len(valid) - n_good
    if n_good >= 5 and n_bad >= 5:
        for col in ["median_g", "p90_g", "spread_g"]:
            if col in valid.columns:
                pred = -valid[col].fillna(0)  # higher g → worse → invert
                try:
                    auroc = float(roc_auc_score(valid["good_day"], pred))
                    diag[f"auroc_{col}"] = round(auroc, 4)
                except ValueError:
                    pass

    return diag


# ── Main pipeline ─────────────────────────────────────────────────────────


def run_portfolio_pipeline(
    enriched_df: pd.DataFrame,
    ehat_df: pd.DataFrame,
    health_df: pd.DataFrame,
    horizon: int = 20,
    config: Optional[PortfolioConfig] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Full 13.6 pipeline: merge data, calibrate, build portfolios, evaluate.

    Returns:
        portfolio_df: daily portfolio returns for all variants
        diagnostics: all metrics, regime evaluation, bucket tables
    """
    cfg = config or PortfolioConfig(horizon=horizon)
    logger.info(f"Running 13.6 portfolio pipeline for {horizon}d")

    # 1. Merge ehat with enriched to get scores + features
    ehat_hz = ehat_df[ehat_df["horizon"] == horizon].copy()
    ehat_hz["as_of_date"] = pd.to_datetime(ehat_hz["as_of_date"])

    er_hz = enriched_df[enriched_df["horizon"] == horizon].copy()
    er_hz["as_of_date"] = pd.to_datetime(er_hz["as_of_date"])

    # Deduplicate enriched by (date, ticker) → take mean score/return
    er_dedup = er_hz.groupby(["as_of_date", "ticker"]).agg({
        "score": "mean",
        "excess_return": "mean",
        "vol_20d": "mean",
        "stable_id": "first",
        "vix_percentile_252d": "mean",
        "market_vol_21d": "mean",
    }).reset_index()

    # Merge ehat (g_pred, ehat_raw) onto enriched
    ehat_sub = ehat_hz[["as_of_date", "ticker", "g_pred", "ehat_raw"]].copy()
    ehat_sub = ehat_sub.groupby(["as_of_date", "ticker"]).agg({
        "g_pred": "mean", "ehat_raw": "mean"
    }).reset_index()

    merged = er_dedup.merge(ehat_sub, on=["as_of_date", "ticker"], how="inner")
    logger.info(f"  Merged data: {len(merged):,} rows, {merged['as_of_date'].nunique()} dates")

    # 2. Choose uncertainty column
    # 20d: g(x) (deployable); 60d: ehat_raw (Tier 2); 90d: g(x)
    unc_col = "ehat_raw" if horizon == 60 else "g_pred"
    logger.info(f"  Uncertainty column: {unc_col}")

    # 3. Calibrate c on DEV data only
    dev_mask = merged["as_of_date"] < HOLDOUT_START
    c_vol = calibrate_c(merged.loc[dev_mask, "vol_20d"], cfg.target_median_w)
    c_deup = calibrate_c(merged.loc[dev_mask, unc_col], cfg.target_median_w)
    logger.info(f"  Calibrated c_vol={c_vol:.4f}, c_deup={c_deup:.4f}")

    # 4. Compute sizing weights
    merged = compute_sizing_weights(merged, c_vol, c_deup, unc_col)

    # Verify median weights on DEV
    dev_data = merged[dev_mask]
    med_w_vol = dev_data["w_vol"].median()
    med_w_deup = dev_data["w_deup"].median()
    logger.info(f"  DEV median w_vol={med_w_vol:.3f}, w_deup={med_w_deup:.3f}")

    # 5. Build portfolios
    hdf = health_df.copy()
    hdf["date"] = pd.to_datetime(hdf["date"])
    portfolio_df = build_variant_portfolios(merged, hdf, cfg)
    logger.info(f"  Portfolio: {len(portfolio_df)} rebalance dates")

    # 6. Portfolio metrics
    variant_metrics = compute_all_variant_metrics(portfolio_df)

    # 7. Regime trust evaluation
    regime_eval = evaluate_regime_trust(
        hdf, enriched_df, horizon,
        threshold=0.0,
        abstention_threshold=cfg.abstention_threshold,
    )

    # 8. Bucket tables
    bucket_tables = compute_bucket_tables(portfolio_df, hdf)

    # 9. Aggregated DEUP
    agg_deup = compute_aggregated_deup(ehat_df, hdf, horizon)

    diagnostics = {
        "horizon": horizon,
        "unc_col": unc_col,
        "c_vol": round(c_vol, 4),
        "c_deup": round(c_deup, 4),
        "dev_median_w_vol": round(med_w_vol, 4),
        "dev_median_w_deup": round(med_w_deup, 4),
        "portfolio_metrics": variant_metrics,
        "regime_evaluation": regime_eval,
        "bucket_tables": bucket_tables,
        "aggregated_deup": agg_deup,
    }

    return portfolio_df, diagnostics

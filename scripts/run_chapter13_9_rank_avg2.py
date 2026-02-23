"""
Chapter 13.9 — DEUP on Rank Avg 2: Robustness Check
=====================================================
Repeats the key 13.1–13.4 DEUP steps using Rank Avg 2 residuals (rather than
tabular_lgb) to determine whether a more holdout-robust base model also
produces better epistemic uncertainty signals, and whether adopting Rank Avg 2
as the primary base model would improve the combined DEUP system.

Decision Gate
-------------
If RA2 + vol-sizing OR RA2 + ê-sizing Sharpe on FINAL > LGB + ê-sizing FINAL,
adopt Rank Avg 2 as primary. Otherwise, LGB remains primary (DEUP economic value
is already captured by the existing 13.7 deployment recommendation).

Usage
-----
    python -m scripts.run_chapter13_9_rank_avg2

Outputs
-------
    evaluation_outputs/chapter13/g_predictions_rank_avg2.parquet
    evaluation_outputs/chapter13/ehat_predictions_rank_avg2.parquet
    evaluation_outputs/chapter13/chapter13_9_ra2_diagnostics.json
"""

from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

warnings.filterwarnings("ignore", category=FutureWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent.parent
OUT  = ROOT / "evaluation_outputs" / "chapter13"

RA2_RESIDUALS    = OUT / "enriched_residuals_rank_avg_2.parquet"
LGB_RESIDUALS    = OUT / "enriched_residuals_tabular_lgb.parquet"
A_PREDICTIONS    = OUT / "a_predictions.parquet"
LGB_EHAT         = OUT / "ehat_predictions.parquet"
HEALTH_20D       = OUT / "expert_health_lgb_20d.parquet"

G_RA2_PATH       = OUT / "g_predictions_rank_avg2.parquet"
EHAT_RA2_PATH    = OUT / "ehat_predictions_rank_avg2.parquet"
RESULTS_PATH     = OUT / "chapter13_9_ra2_diagnostics.json"

# Portfolio parameters
TOP_K        = 10
HOLDOUT_CUT  = pd.Timestamp("2024-01-01")
COST_BPS     = 10.0
VOL_C        = 0.15   # same calibration constant as 13.6
EHAT_C_RA2   = 0.10   # calibrated on DEV for RA2 ê-sizing


# ---------------------------------------------------------------------------
# Step 1: Train g(x) on Rank Avg 2 residuals (walk-forward)
# ---------------------------------------------------------------------------

def train_g_ra2(horizons: List[int] = [20, 60, 90], min_train_folds: int = 20) -> pd.DataFrame:
    """
    Walk-forward g(x) training on RA2 residuals.
    Re-uses src.uncertainty.deup_estimator.train_g_walk_forward.
    """
    from src.uncertainty.deup_estimator import train_g_walk_forward

    logger.info("Loading Rank Avg 2 residuals...")
    ra2 = pd.read_parquet(RA2_RESIDUALS)
    logger.info(f"  Loaded {len(ra2):,} rows across horizons {sorted(ra2['horizon'].unique())}")

    logger.info("Training g(x) walk-forward on Rank Avg 2 (rank_loss target)...")
    g_preds, diag = train_g_walk_forward(
        ra2,
        target_col="rank_loss",
        min_train_folds=min_train_folds,
        horizons=horizons,
    )

    if g_preds.empty:
        raise RuntimeError("g(x) training produced no predictions — check residuals.")

    g_preds.to_parquet(G_RA2_PATH, index=False)
    logger.info(f"  Saved {len(g_preds):,} g(x) predictions → {G_RA2_PATH.name}")

    for hz in horizons:
        d = diag.get(hz, {})
        logger.info(
            f"  {hz}d: n={d.get('n_rows', 0):,}  "
            f"ρ(g, rank_loss)={d.get('spearman_rho', 0):.4f}  "
            f"g_mean={d.get('g_mean', 0):.4f}"
        )

    return g_preds


# ---------------------------------------------------------------------------
# Step 2: Compute ê(x) for Rank Avg 2
# ---------------------------------------------------------------------------

def compute_ehat_ra2(g_preds: pd.DataFrame, horizons: List[int] = [20, 60, 90]) -> pd.DataFrame:
    """
    Merge g(x) predictions with the date-level a(x) baseline to produce ê(x).

    The a(x) baseline is date-level (one value per as_of_date × horizon).
    ê(x) = max(0, g(x) − a(x))
    """
    logger.info("Loading a(x) baseline...")
    a_preds = pd.read_parquet(A_PREDICTIONS)
    a_preds["as_of_date"] = pd.to_datetime(a_preds["as_of_date"])
    # a_preds is date-level: keep only as_of_date, horizon, a_value
    a_date = a_preds[["as_of_date", "horizon", "a_value"]].drop_duplicates(
        subset=["as_of_date", "horizon"]
    )
    logger.info(f"  a(x) rows: {len(a_date):,}")

    g_preds = g_preds.copy()
    g_preds["as_of_date"] = pd.to_datetime(g_preds["as_of_date"])

    # Merge g(x) with a(x) on date × horizon
    ehat_df = g_preds.merge(a_date, on=["as_of_date", "horizon"], how="left")

    # Fill missing a_value with median a per horizon (fallback)
    for hz in horizons:
        mask = (ehat_df["horizon"] == hz) & ehat_df["a_value"].isna()
        if mask.any():
            fallback = ehat_df.loc[ehat_df["horizon"] == hz, "a_value"].median()
            ehat_df.loc[mask, "a_value"] = fallback
            logger.warning(f"  {hz}d: filled {mask.sum()} missing a_value with median={fallback:.4f}")

    # Compute epistemic uncertainty: ê = max(0, g - a)
    ehat_df["ehat_raw"] = np.maximum(0.0, ehat_df["g_pred"] - ehat_df["a_value"])

    # Cross-sectional percentile rank of ê per date × horizon
    ehat_df["ehat_pctile"] = ehat_df.groupby(["as_of_date", "horizon"])["ehat_raw"].transform(
        lambda x: x.rank(pct=True)
    )

    ehat_df.to_parquet(EHAT_RA2_PATH, index=False)
    logger.info(f"  Saved {len(ehat_df):,} ê(x) predictions → {EHAT_RA2_PATH.name}")
    return ehat_df


# ---------------------------------------------------------------------------
# Step 3: Diagnostics — compare RA2 ê vs LGB ê
# ---------------------------------------------------------------------------

def run_diagnostics(ehat_ra2: pd.DataFrame, ra2_residuals: pd.DataFrame) -> Dict[str, Any]:
    """
    Key diagnostics for Chapter 13.9:

    1. ρ(ê, rank_loss): does RA2 g(x) predict rank_loss as well as LGB?
    2. Quintile monotonicity of ê → rank_loss
    3. DEV vs FINAL comparison (does RA2 ê generalise better?)
    4. Comparison with LGB ê
    """
    logger.info("Running diagnostics...")
    diag: Dict[str, Any] = {}

    # Load LGB ê for comparison
    lgb_ehat = pd.read_parquet(LGB_EHAT)
    lgb_ehat["as_of_date"] = pd.to_datetime(lgb_ehat["as_of_date"])

    for hz in [20, 60, 90]:
        # RA2
        ra2_h = ehat_ra2[ehat_ra2["horizon"] == hz].copy()
        ra2_res_h = ra2_residuals[ra2_residuals["horizon"] == hz].copy()
        ra2_res_h["as_of_date"] = pd.to_datetime(ra2_res_h["as_of_date"])

        # Join ê with residuals to get rank_loss (may already be present from g_preds)
        if "rank_loss" in ra2_h.columns:
            merged = ra2_h.copy()
        else:
            merged = ra2_h.merge(
                ra2_res_h[["as_of_date", "ticker", "stable_id", "fold_id", "rank_loss"]],
                on=["as_of_date", "ticker", "stable_id", "fold_id"],
                how="inner",
            )

        # LGB
        lgb_h = lgb_ehat[lgb_ehat["horizon"] == hz].copy()

        def _rho_by_period(df: pd.DataFrame, label: str) -> Dict[str, float]:
            """Per-period Spearman ρ(ê, rank_loss)."""
            result = {}
            for period, mask in [
                ("ALL",   pd.Series(True, index=df.index)),
                ("DEV",   df["as_of_date"] < HOLDOUT_CUT),
                ("FINAL", df["as_of_date"] >= HOLDOUT_CUT),
            ]:
                sub = df[mask].dropna(subset=["ehat_raw", "rank_loss"])
                if len(sub) > 20:
                    rho = spearmanr(sub["ehat_raw"], sub["rank_loss"]).statistic
                else:
                    rho = float("nan")
                result[period] = float(rho)
            return result

        ra2_rho = _rho_by_period(merged, "RA2")

        # Quintile monotonicity (ALL)
        all_data = merged.dropna(subset=["ehat_raw", "rank_loss"])
        if len(all_data) > 50:
            all_data["ehat_q"] = pd.qcut(all_data["ehat_raw"], 5, labels=False, duplicates="drop")
            quint = all_data.groupby("ehat_q")["rank_loss"].mean()
            mono_count = sum(quint.values[i] <= quint.values[i + 1] for i in range(len(quint) - 1))
            monotone = int(mono_count)
            quint_vals = quint.to_dict()
        else:
            monotone = 0
            quint_vals = {}

        # LGB rho
        lgb_merged = lgb_h.merge(
            ra2_res_h[["as_of_date", "ticker", "stable_id", "fold_id", "rank_loss"]].rename(
                columns={"fold_id": "fold_id_ra2"}
            ),
            left_on=["as_of_date", "ticker", "stable_id"],
            right_on=["as_of_date", "ticker", "stable_id"],
            how="inner",
        ) if not lgb_h.empty else pd.DataFrame()

        lgb_rho = {}
        if not lgb_merged.empty:
            lgb_rho = _rho_by_period(
                lgb_merged.rename(columns={"rank_loss_y": "rank_loss"}) if "rank_loss_y" in lgb_merged.columns
                else lgb_merged,
                "LGB"
            )

        # Better approach: LGB rho from its own enriched residuals
        lgb_res_h = pd.read_parquet(LGB_RESIDUALS)
        lgb_res_h = lgb_res_h[lgb_res_h["horizon"] == hz].copy()
        lgb_res_h["as_of_date"] = pd.to_datetime(lgb_res_h["as_of_date"])
        if "rank_loss" in lgb_h.columns:
            lgb_ehat_h = lgb_h.copy()
        else:
            lgb_ehat_h = lgb_h.merge(
                lgb_res_h[["as_of_date", "ticker", "stable_id", "fold_id", "rank_loss"]],
                on=["as_of_date", "ticker", "stable_id", "fold_id"],
                how="inner",
            )
            # Resolve potential _x/_y suffixes
            if "rank_loss_y" in lgb_ehat_h.columns:
                lgb_ehat_h["rank_loss"] = lgb_ehat_h["rank_loss_y"]
            elif "rank_loss_x" in lgb_ehat_h.columns:
                lgb_ehat_h["rank_loss"] = lgb_ehat_h["rank_loss_x"]

        lgb_rho_direct: Dict[str, float] = {}
        for period, mask in [
            ("ALL",   pd.Series(True, index=lgb_ehat_h.index)),
            ("DEV",   lgb_ehat_h["as_of_date"] < HOLDOUT_CUT),
            ("FINAL", lgb_ehat_h["as_of_date"] >= HOLDOUT_CUT),
        ]:
            sub = lgb_ehat_h[mask].dropna(subset=["ehat_raw", "rank_loss"])
            if len(sub) > 20:
                rho = spearmanr(sub["ehat_raw"], sub["rank_loss"]).statistic
            else:
                rho = float("nan")
            lgb_rho_direct[period] = float(rho)

        diag[str(hz)] = {
            "ra2_rho": ra2_rho,
            "lgb_rho": lgb_rho_direct,
            "ra2_monotone_quintiles": monotone,
            "ra2_quintile_rank_loss": {str(k): float(v) for k, v in quint_vals.items()},
            "ra2_n_rows": len(merged),
            "lgb_n_rows": len(lgb_ehat_h),
        }

        logger.info(
            f"  {hz}d RA2: ρ_ALL={ra2_rho.get('ALL', 0):.4f}  "
            f"ρ_DEV={ra2_rho.get('DEV', 0):.4f}  "
            f"ρ_FINAL={ra2_rho.get('FINAL', 0):.4f}  "
            f"monotone={monotone}/4"
        )
        logger.info(
            f"  {hz}d LGB: ρ_ALL={lgb_rho_direct.get('ALL', 0):.4f}  "
            f"ρ_DEV={lgb_rho_direct.get('DEV', 0):.4f}  "
            f"ρ_FINAL={lgb_rho_direct.get('FINAL', 0):.4f}"
        )

    return diag


# ---------------------------------------------------------------------------
# Step 4: Shadow portfolio comparison
# ---------------------------------------------------------------------------

def _build_portfolio_ts(
    residuals: pd.DataFrame,
    ehat: Optional[pd.DataFrame],
    health: pd.DataFrame,
    variant: str,
    horizon: int = 20,
    top_k: int = TOP_K,
    cost_bps: float = COST_BPS,
    vol_c: float = VOL_C,
    ehat_c: float = EHAT_C_RA2,
    gate_threshold: float = 0.2,
    subsample_days: int = 20,
) -> pd.DataFrame:
    """
    Build a non-overlapping L/S portfolio time series.

    Variants: 'raw', 'vol', 'ehat', 'gate_raw', 'gate_vol', 'gate_ehat'

    Returns DataFrame with columns: date, ls_return, is_active
    """
    res = residuals[residuals["horizon"] == horizon].copy()
    res["as_of_date"] = pd.to_datetime(res["as_of_date"])

    # Sub-sample to non-overlapping rebalance dates (every `subsample_days` trading days)
    all_dates = sorted(res["as_of_date"].unique())
    rebalance_dates = all_dates[::subsample_days]

    health = health.copy()
    health["date"] = pd.to_datetime(health["date"])
    health_dict = health.set_index("date")["G_exposure"].to_dict()

    # Merge ê if needed
    if ehat is not None:
        ehat = ehat[ehat["horizon"] == horizon].copy()
        ehat["as_of_date"] = pd.to_datetime(ehat["as_of_date"])

    records = []
    last_longs: Optional[set] = None
    last_shorts: Optional[set] = None

    for date in rebalance_dates:
        day = res[res["as_of_date"] == date].dropna(subset=["score", "excess_return", "vol_20d"])
        if len(day) < top_k * 2 + 2:
            continue

        # Gate
        g_val = health_dict.get(date, float("nan"))
        use_gate = variant.startswith("gate_")
        is_active = True

        if use_gate:
            if np.isnan(g_val) or g_val < gate_threshold:
                records.append({
                    "date": date,
                    "ls_return": 0.0,
                    "turnover": 0.0,
                    "is_active": False,
                    "n_long": 0,
                    "n_short": 0,
                })
                last_longs = None
                last_shorts = None
                continue

        # Sizing
        base = variant.replace("gate_", "")

        if base == "raw":
            scores = day["score"].copy()
            long_idx  = scores.nlargest(top_k).index
            short_idx = scores.nsmallest(top_k).index

        elif base == "vol":
            vol = day["vol_20d"].clip(lower=1e-6)
            sized = day["score"] * np.minimum(1.0, vol_c / np.sqrt(vol))
            long_idx  = sized.nlargest(top_k).index
            short_idx = sized.nsmallest(top_k).index

        elif base == "ehat":
            if ehat is None:
                continue
            day_ehat = ehat[ehat["as_of_date"] == date][["ticker", "ehat_raw"]]
            day_merged = day.merge(day_ehat, on="ticker", how="left")
            day_merged["ehat_raw"] = day_merged["ehat_raw"].fillna(
                day_merged["ehat_raw"].median()
            )
            unc = day_merged["ehat_raw"].clip(lower=1e-6)
            sized_score = day_merged["score"] * np.minimum(1.0, ehat_c / np.sqrt(unc))
            day = day_merged
            long_idx  = sized_score.nlargest(top_k).index
            short_idx = sized_score.nsmallest(top_k).index

        else:
            raise ValueError(f"Unknown variant base: {base}")

        longs  = set(day.loc[long_idx, "ticker"].values)
        shorts = set(day.loc[short_idx, "ticker"].values)

        # Turnover cost
        if last_longs is None:
            turnover = 1.0
        else:
            churn_long  = len(longs.symmetric_difference(last_longs)) / (2 * top_k)
            churn_short = len(shorts.symmetric_difference(last_shorts)) / (2 * top_k)
            turnover = (churn_long + churn_short) / 2

        cost = turnover * cost_bps / 10000.0

        # Returns
        long_ret  = day.loc[long_idx,  "excess_return"].mean()
        short_ret = day.loc[short_idx, "excess_return"].mean()
        ls_ret    = long_ret - short_ret - cost

        last_longs  = longs
        last_shorts = shorts

        records.append({
            "date":      date,
            "ls_return": float(ls_ret),
            "turnover":  float(turnover),
            "is_active": True,
            "n_long":    len(long_idx),
            "n_short":   len(short_idx),
        })

    return pd.DataFrame(records)


def _metrics(ts: pd.DataFrame, label: str) -> Dict[str, float]:
    """Compute annualised Sharpe, MaxDD, return and active-period metrics."""
    rets = ts["ls_return"].fillna(0.0)
    if len(rets) < 5:
        return {"label": label, "n": 0}

    # Periods per year (non-overlapping 20d windows → ~12/year)
    ppy = 12.0

    mean_r  = rets.mean()
    std_r   = rets.std()
    sharpe  = (mean_r / std_r * np.sqrt(ppy)) if std_r > 1e-10 else 0.0
    ann_ret = mean_r * ppy
    ann_vol = std_r * np.sqrt(ppy)

    # MaxDD
    cum = (1 + rets).cumprod()
    peak = cum.cummax()
    dd   = (cum - peak) / peak.clip(lower=1e-10)
    max_dd = float(dd.min())

    # Crisis period (Mar–Jul 2024)
    if "date" in ts.columns:
        ts["date"] = pd.to_datetime(ts["date"])
        crisis = ts[(ts["date"] >= "2024-03-01") & (ts["date"] <= "2024-07-31")]
        c_rets = crisis["ls_return"].fillna(0.0)
        if len(c_rets) > 2:
            c_cum  = (1 + c_rets).cumprod()
            c_peak = c_cum.cummax()
            crisis_mdd = float(((c_cum - c_peak) / c_peak.clip(lower=1e-10)).min())
        else:
            crisis_mdd = float("nan")
    else:
        crisis_mdd = float("nan")

    # Active stats (for gated variants)
    active = ts[ts["is_active"]] if "is_active" in ts.columns else ts
    abstain_rate = 1.0 - (len(active) / len(ts)) if len(ts) > 0 else float("nan")

    return {
        "label":        label,
        "n":            len(rets),
        "sharpe":       float(sharpe),
        "ann_ret":      float(ann_ret),
        "ann_vol":      float(ann_vol),
        "max_dd":       float(max_dd),
        "crisis_mdd":   float(crisis_mdd),
        "abstain_rate": float(abstain_rate),
    }


def _period_metrics(ts: pd.DataFrame, label: str) -> Dict[str, Any]:
    """Split by ALL / DEV / FINAL and compute metrics for each."""
    if "date" not in ts.columns:
        return {}
    ts = ts.copy()
    ts["date"] = pd.to_datetime(ts["date"])

    def _sub(df: pd.DataFrame) -> Dict[str, float]:
        return _metrics(df.reset_index(drop=True), label)

    return {
        "ALL":   _sub(ts),
        "DEV":   _sub(ts[ts["date"] < HOLDOUT_CUT]),
        "FINAL": _sub(ts[ts["date"] >= HOLDOUT_CUT]),
    }


def run_portfolio_comparison(
    ehat_ra2: pd.DataFrame,
    ra2_residuals: pd.DataFrame,
    health: pd.DataFrame,
    horizon: int = 20,
) -> Dict[str, Any]:
    """Compare RA2 vs LGB portfolios across sizing variants."""
    logger.info(f"Building shadow portfolios (horizon={horizon}d)...")

    lgb_residuals = pd.read_parquet(LGB_RESIDUALS)
    lgb_residuals["as_of_date"] = pd.to_datetime(lgb_residuals["as_of_date"])
    lgb_ehat_df   = pd.read_parquet(LGB_EHAT)
    lgb_ehat_df["as_of_date"] = pd.to_datetime(lgb_ehat_df["as_of_date"])
    ra2_residuals["as_of_date"] = pd.to_datetime(ra2_residuals["as_of_date"])

    results: Dict[str, Any] = {}

    configs = [
        ("lgb_raw",       lgb_residuals, None,          "raw"),
        ("lgb_vol",       lgb_residuals, None,          "vol"),
        ("lgb_ehat",      lgb_residuals, lgb_ehat_df,   "ehat"),
        ("lgb_gate_vol",  lgb_residuals, None,          "gate_vol"),
        ("ra2_raw",       ra2_residuals, None,          "raw"),
        ("ra2_vol",       ra2_residuals, None,          "vol"),
        ("ra2_ehat",      ra2_residuals, ehat_ra2,      "ehat"),
        ("ra2_gate_vol",  ra2_residuals, None,          "gate_vol"),
    ]

    for name, res, ehat, variant in configs:
        logger.info(f"  Building {name} ({variant})...")
        ts = _build_portfolio_ts(
            residuals=res,
            ehat=ehat,
            health=health,
            variant=variant,
            horizon=horizon,
        )
        results[name] = _period_metrics(ts, name)

    return results


# ---------------------------------------------------------------------------
# Step 5: Decision gate
# ---------------------------------------------------------------------------

def make_decision(portfolio_results: Dict[str, Any], diag_20d: Dict[str, Any]) -> Dict[str, Any]:
    """
    Decision gate: should we adopt Rank Avg 2 as primary?

    Criteria:
    1. Primary:  RA2 FINAL Sharpe (best RA2 variant) > LGB FINAL Sharpe (best LGB variant)
    2. Secondary: RA2 ρ(ê, rank_loss) FINAL >= LGB ρ(ê, rank_loss) FINAL
    3. Tertiary:  RA2 DEV Sharpe competitive (within 10% of LGB DEV Sharpe)
    """
    def _best_final_sharpe(prefix: str) -> Tuple[str, float]:
        best_v, best_s = "", -99.0
        for v in [f"{prefix}_vol", f"{prefix}_ehat", f"{prefix}_raw", f"{prefix}_gate_vol"]:
            if v in portfolio_results:
                s = portfolio_results[v].get("FINAL", {}).get("sharpe", float("nan"))
                if not np.isnan(s) and s > best_s:
                    best_s, best_v = s, v
        return best_v, best_s

    ra2_best_v, ra2_final_sharpe = _best_final_sharpe("ra2")
    lgb_best_v, lgb_final_sharpe = _best_final_sharpe("lgb")

    ra2_rho_final = diag_20d.get("ra2_rho", {}).get("FINAL", float("nan"))
    lgb_rho_final = diag_20d.get("lgb_rho", {}).get("FINAL", float("nan"))

    ra2_dev_sharpe = portfolio_results.get("ra2_vol", {}).get("DEV", {}).get("sharpe", float("nan"))
    lgb_dev_sharpe = portfolio_results.get("lgb_vol", {}).get("DEV", {}).get("sharpe", float("nan"))

    criterion_1 = (not np.isnan(ra2_final_sharpe)) and ra2_final_sharpe > lgb_final_sharpe
    criterion_2 = (not np.isnan(ra2_rho_final)) and ra2_rho_final >= lgb_rho_final
    criterion_3 = (
        (not np.isnan(ra2_dev_sharpe)) and (not np.isnan(lgb_dev_sharpe))
        and ra2_dev_sharpe >= lgb_dev_sharpe * 0.90
    )

    adopt_ra2 = criterion_1  # primary criterion
    criteria_met = sum([criterion_1, criterion_2, criterion_3])

    if adopt_ra2:
        recommendation = (
            "ADOPT Rank Avg 2 as primary. RA2 achieves higher FINAL holdout Sharpe "
            f"({ra2_final_sharpe:.3f} vs {lgb_final_sharpe:.3f}). "
            "Recommend repeating 13.5–13.7 with RA2 as the base model."
        )
    else:
        recommendation = (
            "RETAIN tabular_lgb as primary. RA2 does not beat LGB on FINAL holdout Sharpe "
            f"({ra2_final_sharpe:.3f} vs {lgb_final_sharpe:.3f}). "
            "DEUP's economic value is already captured by the 13.7 deployment recommendation "
            "(Binary Gate + Vol-Sizing + ê-Cap). Chapter 13 goal is met with LGB as primary."
        )

    return {
        "adopt_ra2":           adopt_ra2,
        "criteria_met":        criteria_met,
        "criterion_1_primary": criterion_1,
        "criterion_2_rho":     criterion_2,
        "criterion_3_dev":     criterion_3,
        "ra2_best_variant":    ra2_best_v,
        "ra2_final_sharpe":    float(ra2_final_sharpe),
        "lgb_best_variant":    lgb_best_v,
        "lgb_final_sharpe":    float(lgb_final_sharpe),
        "ra2_rho_final":       float(ra2_rho_final) if not np.isnan(ra2_rho_final) else None,
        "lgb_rho_final":       float(lgb_rho_final) if not np.isnan(lgb_rho_final) else None,
        "ra2_dev_sharpe":      float(ra2_dev_sharpe) if not np.isnan(ra2_dev_sharpe) else None,
        "lgb_dev_sharpe":      float(lgb_dev_sharpe) if not np.isnan(lgb_dev_sharpe) else None,
        "recommendation":      recommendation,
    }


# ---------------------------------------------------------------------------
# Results table
# ---------------------------------------------------------------------------

def print_results_table(
    portfolio_results: Dict[str, Any],
    diag: Dict[str, Any],
    decision: Dict[str, Any],
) -> None:
    """Print the full comparison table."""
    print()
    print("╔" + "═" * 100 + "╗")
    print("║{:^100}║".format("Chapter 13.9 — DEUP on Rank Avg 2: Robustness Check"))
    print("╠" + "═" * 100 + "╣")
    print()
    print("  ── ê(x) Signal Quality: ρ(ê, rank_loss) ──")
    print()
    print(f"  {'Horizon':<8}  {'RA2 ρ ALL':>10}  {'RA2 ρ DEV':>10}  {'RA2 ρ FINAL':>12}  {'LGB ρ ALL':>10}  {'LGB ρ DEV':>10}  {'LGB ρ FINAL':>12}")
    print("  " + "─" * 82)
    for hz in [20, 60, 90]:
        d = diag.get(str(hz), {})
        ra2 = d.get("ra2_rho", {})
        lgb = d.get("lgb_rho", {})
        print(
            f"  {hz}d      "
            f"  {ra2.get('ALL', float('nan')):>+10.4f}"
            f"  {ra2.get('DEV', float('nan')):>+10.4f}"
            f"  {ra2.get('FINAL', float('nan')):>+12.4f}"
            f"  {lgb.get('ALL', float('nan')):>+10.4f}"
            f"  {lgb.get('DEV', float('nan')):>+10.4f}"
            f"  {lgb.get('FINAL', float('nan')):>+12.4f}"
        )

    print()
    print("  ── Shadow Portfolio (20d, non-overlapping monthly, top-10 L/S) ──")
    print()
    print(
        f"  {'Variant':<18}  {'ALL Sharpe':>10}  {'DEV Sharpe':>10}  {'FINAL Sharpe':>12}  {'Crisis MaxDD':>13}"
    )
    print("  " + "─" * 68)

    order = ["lgb_raw", "lgb_vol", "lgb_ehat", "lgb_gate_vol",
             "ra2_raw", "ra2_vol", "ra2_ehat", "ra2_gate_vol"]
    sep_printed = False
    for name in order:
        if name == "ra2_raw" and not sep_printed:
            print("  " + "─" * 68)
            sep_printed = True
        r = portfolio_results.get(name, {})
        all_s  = r.get("ALL",   {}).get("sharpe",     float("nan"))
        dev_s  = r.get("DEV",   {}).get("sharpe",     float("nan"))
        fin_s  = r.get("FINAL", {}).get("sharpe",     float("nan"))
        c_mdd  = r.get("FINAL", {}).get("crisis_mdd", float("nan"))
        # Use ALL crisis_mdd instead if FINAL period doesn't capture crisis
        if np.isnan(c_mdd):
            c_mdd = r.get("ALL", {}).get("crisis_mdd", float("nan"))
        print(
            f"  {name:<18}  {all_s:>+10.3f}  {dev_s:>+10.3f}  {fin_s:>+12.3f}  "
            f"{c_mdd:>12.1%}"
        )

    print()
    print("╠" + "═" * 100 + "╣")
    print("║{:^100}║".format("DECISION GATE"))
    print("╠" + "═" * 100 + "╣")
    v = decision
    adopt_str = "✅  ADOPT RA2 AS PRIMARY" if v["adopt_ra2"] else "⚠️  RETAIN tabular_lgb AS PRIMARY"
    print(f"  {adopt_str}")
    print(f"  Criteria met: {v['criteria_met']}/3")
    print(f"  (1) RA2 FINAL Sharpe > LGB FINAL: {'✓' if v['criterion_1_primary'] else '✗'}  ({v['ra2_final_sharpe']:.3f} vs {v['lgb_final_sharpe']:.3f})")
    print(f"  (2) RA2 ρ(ê,rl) FINAL ≥ LGB:     {'✓' if v['criterion_2_rho'] else '✗'}  ({v['ra2_rho_final']} vs {v['lgb_rho_final']})")
    print(f"  (3) RA2 DEV Sharpe ≥ 90% LGB:    {'✓' if v['criterion_3_dev'] else '✗'}  ({v['ra2_dev_sharpe']} vs {v['lgb_dev_sharpe']})")
    print()
    print(f"  Recommendation: {v['recommendation']}")
    print("╚" + "═" * 100 + "╝")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    logger.info("=" * 60)
    logger.info("Chapter 13.9 — DEUP on Rank Avg 2")
    logger.info("=" * 60)

    # Load shared data
    ra2_residuals = pd.read_parquet(RA2_RESIDUALS)
    health        = pd.read_parquet(HEALTH_20D)
    health["date"] = pd.to_datetime(health["date"])

    # ── Step 1: Train g(x) on RA2 ──────────────────────────────────────────
    if G_RA2_PATH.exists():
        logger.info(f"[SKIP] g(x) RA2 already trained — loading from {G_RA2_PATH.name}")
        g_preds = pd.read_parquet(G_RA2_PATH)
    else:
        logger.info("\n[1/5] Training g(x) on Rank Avg 2 residuals...")
        g_preds = train_g_ra2(horizons=[20, 60, 90])

    # ── Step 2: Compute ê(x) for RA2 ───────────────────────────────────────
    if EHAT_RA2_PATH.exists():
        logger.info(f"[SKIP] ê(x) RA2 already computed — loading from {EHAT_RA2_PATH.name}")
        ehat_ra2 = pd.read_parquet(EHAT_RA2_PATH)
    else:
        logger.info("\n[2/5] Computing ê(x) for Rank Avg 2...")
        ehat_ra2 = compute_ehat_ra2(g_preds)

    # ── Step 3: Diagnostics ─────────────────────────────────────────────────
    logger.info("\n[3/5] Running diagnostics (ρ comparison)...")
    diag = run_diagnostics(ehat_ra2, ra2_residuals)

    # ── Step 4: Portfolio comparison ─────────────────────────────────────────
    logger.info("\n[4/5] Building shadow portfolio comparison (20d)...")
    portfolio_results = run_portfolio_comparison(ehat_ra2, ra2_residuals, health, horizon=20)

    # ── Step 5: Decision gate ───────────────────────────────────────────────
    logger.info("\n[5/5] Computing decision gate...")
    decision = make_decision(portfolio_results, diag.get("20", {}))

    # Print table
    print_results_table(portfolio_results, diag, decision)

    # Save results
    output = {
        "chapter": "13.9",
        "title": "DEUP on Rank Avg 2 — Robustness Check",
        "diagnostics_by_horizon": diag,
        "portfolio_comparison": portfolio_results,
        "decision_gate": decision,
    }

    with open(RESULTS_PATH, "w") as f:
        json.dump(output, f, indent=2, default=str)

    logger.info(f"\nResults saved → {RESULTS_PATH.name}")
    logger.info("Chapter 13.9 COMPLETE")


if __name__ == "__main__":
    main()

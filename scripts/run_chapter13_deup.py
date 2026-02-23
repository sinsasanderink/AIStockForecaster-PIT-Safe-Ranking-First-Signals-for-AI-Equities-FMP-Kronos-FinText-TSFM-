#!/usr/bin/env python
"""
Chapter 13: DEUP Uncertainty Quantification
=============================================

13.0 — Populate residual archive and build enriched residuals
13.1 — Train g(x) error predictor walk-forward
13.2 — Aleatoric baseline a(x) with tiered approach + alignment diagnostic
13.3 — Epistemic signal ê(x) = max(0, g(x) - a(x))
13.4 — Diagnostics (partial correlation, AUROC, 2024 test, baselines)
13.4b — Expert health H(t) per-date throttle
13.5  — Conformal prediction intervals (raw, vol-norm, DEUP-norm)
13.6  — DEUP-sized shadow portfolio + global regime evaluation

Usage:
    python -m scripts.run_chapter13_deup --step 0   # residual archive only
    python -m scripts.run_chapter13_deup --step 1   # g(x) training only
    python -m scripts.run_chapter13_deup --step 2   # aleatoric baseline only
    python -m scripts.run_chapter13_deup --step 3   # epistemic signal only
    python -m scripts.run_chapter13_deup --step 4   # diagnostics only
    python -m scripts.run_chapter13_deup --step 5   # expert health only
    python -m scripts.run_chapter13_deup --step 6   # conformal intervals only
    python -m scripts.run_chapter13_deup --step 7   # portfolio + regime eval only
    python -m scripts.run_chapter13_deup             # all steps
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ch13_deup")

OUTPUT_DIR = PROJECT_ROOT / "evaluation_outputs" / "chapter13"
DATA_DIR = PROJECT_ROOT / "data"

EVAL_ROWS_PATHS = {
    "tabular_lgb": (
        PROJECT_ROOT
        / "evaluation_outputs"
        / "chapter7_tabular_lgb_real"
        / "monthly"
        / "baseline_tabular_lgb_monthly"
        / "eval_rows.parquet"
    ),
    "rank_avg_2": (
        PROJECT_ROOT
        / "evaluation_outputs"
        / "chapter11_fusion_full"
        / "rank_avg_2"
        / "ch11_rank_avg_2_full"
        / "eval_rows.parquet"
    ),
    "learned_stacking": (
        PROJECT_ROOT
        / "evaluation_outputs"
        / "chapter11_fusion_full"
        / "learned_stacking"
        / "ch11_learned_stacking_full"
        / "eval_rows.parquet"
    ),
}

REGIME_CONTEXT_PATH = DATA_DIR / "regime_context.parquet"
ENRICHED_PATH = OUTPUT_DIR / "enriched_residuals.parquet"
G_PREDICTIONS_PATH = OUTPUT_DIR / "g_predictions.parquet"
DIAGNOSTICS_01_PATH = OUTPUT_DIR / "diagnostics_13_0_1.json"
DIAGNOSTICS_2_PATH = OUTPUT_DIR / "diagnostics_13_2.json"
DIAGNOSTICS_3_PATH = OUTPUT_DIR / "diagnostics_13_3.json"
DIAGNOSTICS_4_PATH = OUTPUT_DIR / "diagnostics_13_4.json"
DIAGNOSTICS_HEALTH_PATH = OUTPUT_DIR / "expert_health_diagnostics.json"
CONFORMAL_PATH = OUTPUT_DIR / "conformal_predictions.parquet"
DIAGNOSTICS_CONFORMAL_PATH = OUTPUT_DIR / "conformal_diagnostics.json"
PORTFOLIO_METRICS_PATH = OUTPUT_DIR / "chapter13_6_portfolio_metrics.json"
REGIME_EVAL_PATH = OUTPUT_DIR / "chapter13_6_regime_eval.json"
DAILY_TS_PATH = OUTPUT_DIR / "chapter13_6_daily_timeseries.parquet"
POLICY_RESULTS_PATH = OUTPUT_DIR / "chapter13_7_policy_results.json"
POLICY_CALIBRATION_PATH = OUTPUT_DIR / "chapter13_7_calibration.json"
EHAT_SCORE_DIAG_PATH = OUTPUT_DIR / "chapter13_7_ehat_score_diagnostic.json"
A_PREDICTIONS_PATH = OUTPUT_DIR / "a_predictions.parquet"
EHAT_PATH = OUTPUT_DIR / "ehat_predictions.parquet"
EHAT_MAE_PATH = OUTPUT_DIR / "ehat_predictions_mae.parquet"


# ── 13.0: Populate residual archive & build enriched residuals ───────────


def compute_rank_loss(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add rank_loss = |rank_pct(ER) - rank_pct(score)| per (date, horizon).

    Also adds mae_loss for secondary comparison.
    """
    out = df.copy()
    out["mae_loss"] = (out["excess_return"] - out["score"]).abs()

    r_score = out.groupby(["as_of_date", "horizon"])["score"].rank(pct=True)
    r_actual = out.groupby(["as_of_date", "horizon"])["excess_return"].rank(pct=True)
    out["rank_loss"] = (r_actual - r_score).abs()
    out["rank_score"] = r_score
    out["rank_actual"] = r_actual
    return out


def enrich_with_regime_context(
    df: pd.DataFrame, rc: pd.DataFrame
) -> pd.DataFrame:
    """Join regime_context features onto residuals by (date, stable_id)."""
    rc = rc.copy()
    rc = rc.rename(columns={"date": "as_of_date"})
    rc["as_of_date"] = pd.to_datetime(rc["as_of_date"])
    df["as_of_date"] = pd.to_datetime(df["as_of_date"])

    rc_features = [
        "as_of_date", "stable_id",
        "vol_20d", "vol_60d", "vol_of_vol", "mom_1m",
        "vix_percentile_252d", "vix_regime", "market_regime",
        "market_vol_21d", "market_return_5d", "market_return_21d",
        "above_ma_50", "above_ma_200",
    ]
    rc_subset = rc[[c for c in rc_features if c in rc.columns]].copy()

    drop_overlap = [
        c for c in rc_subset.columns
        if c in df.columns and c not in ("as_of_date", "stable_id")
    ]
    if drop_overlap:
        df = df.drop(columns=drop_overlap)

    merged = df.merge(rc_subset, on=["as_of_date", "stable_id"], how="left")
    n_before = len(df)
    n_matched = merged[
        merged[[c for c in rc_subset.columns if c not in ("as_of_date", "stable_id")][0]].notna()
    ].shape[0] if len(rc_subset.columns) > 2 else 0

    logger.info(
        f"Regime join: {n_matched:,}/{n_before:,} rows matched "
        f"({n_matched / n_before * 100:.1f}%)"
    )
    return merged


def step_0_populate_archive():
    """13.0: Load eval_rows, compute rank_loss, enrich with regime context."""
    logger.info("=" * 60)
    logger.info("STEP 13.0: Populate residual archive")
    logger.info("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rc = pd.read_parquet(REGIME_CONTEXT_PATH)
    logger.info(f"Loaded regime_context: {len(rc):,} rows")

    report = {}

    for model_name, eval_path in EVAL_ROWS_PATHS.items():
        if not eval_path.exists():
            logger.warning(f"  {model_name}: eval_rows not found at {eval_path}")
            continue

        logger.info(f"\n--- Processing {model_name} ---")
        er = pd.read_parquet(eval_path)
        logger.info(f"  Loaded: {len(er):,} rows, {er['fold_id'].nunique()} folds")

        er = compute_rank_loss(er)
        logger.info(
            f"  rank_loss: mean={er['rank_loss'].mean():.4f}, "
            f"median={er['rank_loss'].median():.4f}"
        )
        logger.info(
            f"  mae_loss:  mean={er['mae_loss'].mean():.4f}, "
            f"median={er['mae_loss'].median():.4f}"
        )

        er["sub_model_id"] = model_name
        enriched = enrich_with_regime_context(er, rc)

        save_path = OUTPUT_DIR / f"enriched_residuals_{model_name}.parquet"
        enriched.to_parquet(save_path, index=False)
        logger.info(f"  Saved: {save_path} ({len(enriched):,} rows)")

        per_hz = {}
        for hz in sorted(enriched["horizon"].unique()):
            hz_data = enriched[enriched["horizon"] == hz]
            rl = hz_data["rank_loss"]
            ml = hz_data["mae_loss"]

            daily_ric = hz_data.groupby("as_of_date").apply(
                lambda g: stats.spearmanr(g["score"], g["excess_return"]).statistic
                if len(g) > 5 else np.nan,
                include_groups=False,
            )
            daily_rl = hz_data.groupby("as_of_date")["rank_loss"].mean()

            rl_vs_ric = stats.spearmanr(daily_rl.dropna(), daily_ric.dropna()).statistic

            per_hz[int(hz)] = {
                "n_rows": len(hz_data),
                "rank_loss_mean": float(rl.mean()),
                "rank_loss_median": float(rl.median()),
                "mae_loss_mean": float(ml.mean()),
                "mae_loss_median": float(ml.median()),
                "daily_rank_loss_vs_rankic_rho": float(rl_vs_ric),
            }

        report[model_name] = {
            "n_rows": len(enriched),
            "n_folds": int(enriched["fold_id"].nunique()),
            "n_dates": int(enriched["as_of_date"].nunique()),
            "n_tickers": int(enriched["ticker"].nunique()),
            "per_horizon": per_hz,
        }

    logger.info("\n" + "=" * 60)
    logger.info("13.0 SUMMARY")
    logger.info("=" * 60)
    for model, info in report.items():
        logger.info(
            f"  {model}: {info['n_rows']:,} rows, "
            f"{info['n_folds']} folds, {info['n_tickers']} tickers"
        )
        for hz, hinfo in info["per_horizon"].items():
            logger.info(
                f"    {hz}d: rank_loss={hinfo['rank_loss_mean']:.4f}, "
                f"mae={hinfo['mae_loss_mean']:.4f}, "
                f"rl_vs_ric_rho={hinfo['daily_rank_loss_vs_rankic_rho']:.4f}"
            )

    return report


# ── 13.1: Train g(x) error predictor ────────────────────────────────────


def step_1_train_g():
    """13.1: Walk-forward g(x) training on enriched LGB residuals."""
    logger.info("=" * 60)
    logger.info("STEP 13.1: Train g(x) error predictor")
    logger.info("=" * 60)

    from src.uncertainty.deup_estimator import train_g_walk_forward

    enriched_path = OUTPUT_DIR / "enriched_residuals_tabular_lgb.parquet"
    if not enriched_path.exists():
        raise FileNotFoundError(
            f"Run step 0 first: {enriched_path} not found"
        )

    enriched = pd.read_parquet(enriched_path)
    logger.info(f"Loaded enriched residuals: {len(enriched):,} rows")

    logger.info("\n--- Training g(x) on rank_loss target ---")
    g_preds_rank, diag_rank = train_g_walk_forward(
        enriched, target_col="rank_loss", min_train_folds=20, horizons=[20, 60, 90]
    )

    logger.info("\n--- Training g(x) on mae_loss target (secondary) ---")
    g_preds_mae, diag_mae = train_g_walk_forward(
        enriched, target_col="mae_loss", min_train_folds=20, horizons=[20, 60, 90]
    )

    if not g_preds_rank.empty:
        g_preds_rank["target_type"] = "rank_loss"
    if not g_preds_mae.empty:
        g_preds_mae["target_type"] = "mae_loss"
        g_preds_mae = g_preds_mae.rename(columns={"g_pred": "g_pred_mae"})

    if not g_preds_rank.empty:
        g_preds_rank.to_parquet(
            OUTPUT_DIR / "g_predictions_rank.parquet", index=False
        )
        logger.info(
            f"  Saved rank g(x): {len(g_preds_rank):,} predictions"
        )
    if not g_preds_mae.empty:
        g_preds_mae.to_parquet(
            OUTPUT_DIR / "g_predictions_mae.parquet", index=False
        )
        logger.info(
            f"  Saved MAE g(x):  {len(g_preds_mae):,} predictions"
        )

    logger.info("\n" + "=" * 60)
    logger.info("13.1 SUMMARY — g(x) on rank_loss")
    logger.info("=" * 60)
    for hz in [20, 60, 90]:
        d = diag_rank.get(hz, {})
        logger.info(
            f"  {hz}d: n={d.get('n_rows', 0):,}, "
            f"rho(g, rank_loss)={d.get('spearman_rho', 0):.4f}, "
            f"g_mean={d.get('g_mean', 0):.4f}, "
            f"target_mean={d.get('target_mean', 0):.4f}"
        )

    fi = diag_rank.get("feature_importances", [])
    if fi:
        logger.info("\n  Feature importances (last fold):")
        for entry in fi:
            hz = entry["horizon"]
            imp = sorted(
                entry["importances"].items(), key=lambda x: x[1], reverse=True
            )
            top3 = ", ".join(f"{k}={v:.0f}" for k, v in imp[:3])
            logger.info(f"    {hz}d: {top3}")

    logger.info("\n  g(x) on mae_loss (secondary):")
    for hz in [20, 60, 90]:
        d = diag_mae.get(hz, {})
        logger.info(
            f"  {hz}d: rho(g, mae_loss)={d.get('spearman_rho', 0):.4f}"
        )

    return {
        "rank_loss_diagnostics": {
            k: v for k, v in diag_rank.items()
            if k != "feature_importances"
        },
        "mae_loss_diagnostics": {
            k: v for k, v in diag_mae.items()
            if k != "feature_importances"
        },
        "feature_importances": fi,
        "features_used": diag_rank.get("features", []),
    }


# ── 13.2: Aleatoric baseline a(x) ────────────────────────────────────────


def step_2_aleatoric():
    """13.2: Compute aleatoric baseline a(x) using tiered approach."""
    logger.info("=" * 60)
    logger.info("STEP 13.2: Aleatoric baseline a(x)")
    logger.info("=" * 60)

    from src.uncertainty.aleatoric_baseline import select_best_tier

    model_dfs = {}
    for model_name in ["tabular_lgb", "rank_avg_2", "learned_stacking"]:
        path = OUTPUT_DIR / f"enriched_residuals_{model_name}.parquet"
        if path.exists():
            model_dfs[model_name] = pd.read_parquet(path)
            logger.info(f"  Loaded {model_name}: {len(model_dfs[model_name]):,} rows")
        else:
            logger.warning(f"  {model_name}: not found at {path}")

    if not model_dfs:
        raise FileNotFoundError("No enriched residual files found. Run step 0 first.")

    enriched_lgb = model_dfs.get("tabular_lgb")
    if enriched_lgb is None:
        raise FileNotFoundError("tabular_lgb enriched residuals required for Tier 2")

    horizons = [20, 60, 90]
    all_a_results = []
    report = {}

    for hz in horizons:
        logger.info(f"\n{'='*40} {hz}d horizon {'='*40}")

        tier_name, a_values, tier_diagnostics = select_best_tier(
            enriched_lgb,
            model_dfs,
            horizon=hz,
            max_tier=2,
            min_train_folds=20,
        )

        logger.info(f"\n  >>> Selected tier: {tier_name} for {hz}d")

        if isinstance(a_values, pd.DataFrame):
            a_out = a_values[["as_of_date", "ticker", "stable_id", "fold_id", "a_tier2"]].copy()
            a_out = a_out.rename(columns={"a_tier2": "a_value"})
        else:
            a_out = a_values.reset_index()
            a_out.columns = ["as_of_date", "a_value"]

        a_out["horizon"] = hz
        a_out["tier"] = tier_name
        all_a_results.append(a_out)

        winning_diag = tier_diagnostics.get(tier_name, {})
        report[int(hz)] = {
            "selected_tier": tier_name,
            "mean_rho": winning_diag.get("mean_rho", 0),
            "all_monotonic": winning_diag.get("all_monotonic", False),
            "verdict": winning_diag.get("verdict", "UNKNOWN"),
            "a_stats": winning_diag.get("a_stats", {}),
            "all_tiers_tested": {
                t: {
                    "mean_rho": d.get("mean_rho", 0),
                    "verdict": d.get("verdict", "UNKNOWN"),
                    "per_model": {
                        m: {"rho": r.get("rho", 0), "monotonic": r.get("monotonic", False)}
                        for m, r in d.get("per_model", {}).items()
                    },
                }
                for t, d in tier_diagnostics.items()
            },
        }

    if all_a_results:
        a_combined = pd.concat(all_a_results, ignore_index=True)
        a_combined.to_parquet(A_PREDICTIONS_PATH, index=False)
        logger.info(f"\nSaved a(x) predictions: {A_PREDICTIONS_PATH} ({len(a_combined):,} rows)")

    logger.info("\n" + "=" * 60)
    logger.info("13.2 SUMMARY")
    logger.info("=" * 60)
    for hz in horizons:
        r = report.get(hz, {})
        logger.info(
            f"  {hz}d: tier={r.get('selected_tier', '?')}, "
            f"mean_rho={r.get('mean_rho', 0):.4f}, "
            f"monotonic={r.get('all_monotonic', False)}, "
            f"verdict={r.get('verdict', '?')}"
        )
        for t, td in r.get("all_tiers_tested", {}).items():
            logger.info(f"    {t}: rho={td.get('mean_rho', 0):.4f}, verdict={td.get('verdict', '?')}")

    return report


# ── 13.3: Epistemic signal ê(x) ──────────────────────────────────────────


def step_3_epistemic():
    """13.3: Compute ê(x) = max(0, g(x) - a(x)) for all horizons."""
    logger.info("=" * 60)
    logger.info("STEP 13.3: Epistemic signal ê(x)")
    logger.info("=" * 60)

    from src.uncertainty.epistemic_signal import (
        compute_ehat,
        compute_ehat_mae,
        run_sanity_checks,
    )

    g_rank_path = OUTPUT_DIR / "g_predictions_rank.parquet"
    g_mae_path = OUTPUT_DIR / "g_predictions_mae.parquet"
    a_path = A_PREDICTIONS_PATH

    for p, label in [(g_rank_path, "g_rank"), (a_path, "a_preds")]:
        if not p.exists():
            raise FileNotFoundError(f"Run earlier steps first: {p} not found")

    g_preds_rank = pd.read_parquet(g_rank_path)
    a_preds = pd.read_parquet(a_path)
    logger.info(f"Loaded g(x) rank: {len(g_preds_rank):,} rows")
    logger.info(f"Loaded a(x):      {len(a_preds):,} rows")

    # Primary: rank-based ê(x)
    logger.info("\n--- Primary: rank-based ê(x) ---")
    ehat_df, diag_rank = compute_ehat(g_preds_rank, a_preds, horizons=[20, 60, 90])

    if not ehat_df.empty:
        ehat_df.to_parquet(EHAT_PATH, index=False)
        logger.info(f"\nSaved ê(x): {EHAT_PATH} ({len(ehat_df):,} rows)")

    # Secondary: MAE-based ê(x)
    diag_mae = {}
    if g_mae_path.exists():
        logger.info("\n--- Secondary: MAE-based ê(x) ---")
        g_preds_mae = pd.read_parquet(g_mae_path)
        logger.info(f"Loaded g(x) MAE: {len(g_preds_mae):,} rows")
        ehat_mae_df, diag_mae = compute_ehat_mae(g_preds_mae, a_preds, horizons=[20, 60, 90])
        if not ehat_mae_df.empty:
            ehat_mae_df.to_parquet(EHAT_MAE_PATH, index=False)
            logger.info(f"Saved MAE ê(x): {EHAT_MAE_PATH} ({len(ehat_mae_df):,} rows)")

    # Sanity checks
    logger.info("\n--- Sanity checks ---")
    sanity = run_sanity_checks(ehat_df, diag_rank)
    for name, check in sanity["checks"].items():
        status = "✓" if check["passed"] else "✗"
        logger.info(f"  {status} {name}: {check['detail']}")
    logger.info(
        f"\n  Sanity: {sanity['n_passed']}/{sanity['n_total']} passed, "
        f"all_passed={sanity['all_passed']}"
    )

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("13.3 SUMMARY")
    logger.info("=" * 60)
    for hz in [20, 60, 90]:
        d = diag_rank.get(hz, {})
        label = d.get("deployment_label", "?")
        logger.info(
            f"  {hz}d [{label}]: ê_mean={d.get('ehat_mean', 0):.4f}, "
            f"pct_zero={d.get('pct_zero', 0):.1%}, "
            f"ρ(ê,rl)={d.get('rho_ehat_rank_loss', 0):.4f}, "
            f"Δ_selective={d.get('selective_delta', 0):.4f}"
        )
        for period in ["DEV", "FINAL"]:
            pd_data = d.get("dev_final", {}).get(period, {})
            if pd_data.get("n_rows", 0) >= 20:
                logger.info(
                    f"    {period}: n={pd_data['n_rows']:,}, "
                    f"ê_mean={pd_data.get('ehat_mean', 0):.4f}, "
                    f"ρ(ê,rl)={pd_data.get('rho_ehat_rl', 0):.4f}"
                )

    report = {
        "rank_diagnostics": {str(k): v for k, v in diag_rank.items()},
        "mae_diagnostics": {str(k): v for k, v in diag_mae.items()},
        "sanity_checks": sanity,
    }
    return report


# ── 13.4: Diagnostics ────────────────────────────────────────────────────


def step_4_diagnostics():
    """13.4: Run all DEUP diagnostics (DEV and FINAL separately)."""
    logger.info("=" * 60)
    logger.info("STEP 13.4: DEUP Diagnostics")
    logger.info("=" * 60)

    from src.uncertainty.deup_diagnostics import run_all_diagnostics

    ehat_path = EHAT_PATH
    er_path = OUTPUT_DIR / "enriched_residuals_tabular_lgb.parquet"
    d01_path = DIAGNOSTICS_01_PATH

    for p, label in [(ehat_path, "ehat"), (er_path, "enriched_residuals"), (d01_path, "diagnostics_01")]:
        if not p.exists():
            raise FileNotFoundError(f"Run earlier steps first: {p} not found")

    ehat_df = pd.read_parquet(ehat_path)
    enriched_residuals = pd.read_parquet(er_path)
    with open(d01_path) as f:
        diagnostics_01 = json.load(f)

    logger.info(f"Loaded ê(x): {len(ehat_df):,} rows")
    logger.info(f"Loaded enriched residuals: {len(enriched_residuals):,} rows")

    report = run_all_diagnostics(
        ehat_df=ehat_df,
        enriched_residuals=enriched_residuals,
        diagnostics_01=diagnostics_01,
        horizons=[20, 60, 90],
    )

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("13.4 DIAGNOSTIC SUMMARY")
    logger.info("=" * 60)
    for hz in [20, 60, 90]:
        hz_res = report.get(hz, {})
        a_res = hz_res.get("A_partial_correlation", {}).get("ALL", {})
        b_res = hz_res.get("B_selective_risk", {}).get("ALL", {})
        stab = hz_res.get("stability", {}).get("summary", {})
        d_res = hz_res.get("D_regime_2024", {})
        logger.info(f"\n  {hz}d:")
        if not a_res.get("skip"):
            logger.info(f"    A (disentanglement): {a_res.get('verdict', '?')}")
        if not b_res.get("skip"):
            logger.info(f"    B (selective risk):   {b_res.get('verdict', '?')}")
        logger.info(f"    D (2024 test):        {d_res.get('verdict', '?')}")
        logger.info(f"    Stability:            {stab.get('verdict', '?')}")

    return {str(k): v for k, v in report.items()}


# ── 13.4b: Expert health ─────────────────────────────────────────────────


def step_5_expert_health():
    """13.4b: Compute per-date expert health H(t) for all models."""
    logger.info("=" * 60)
    logger.info("STEP 13.4b: Expert Health H(t)")
    logger.info("=" * 60)

    from src.uncertainty.expert_health import ExpertHealthEstimator, HealthConfig

    er_lgb_path = OUTPUT_DIR / "enriched_residuals_tabular_lgb.parquet"
    er_ra2_path = OUTPUT_DIR / "enriched_residuals_rank_avg_2.parquet"
    er_ls_path = OUTPUT_DIR / "enriched_residuals_learned_stacking.parquet"

    if not er_lgb_path.exists():
        raise FileNotFoundError(f"Run step 0 first: {er_lgb_path}")

    er_lgb = pd.read_parquet(er_lgb_path)
    logger.info(f"Loaded LGB enriched: {len(er_lgb):,} rows")

    other_models = {}
    for name, path in [("rank_avg_2", er_ra2_path), ("learned_stacking", er_ls_path)]:
        if path.exists():
            other_models[name] = pd.read_parquet(path)
            logger.info(f"Loaded {name}: {len(other_models[name]):,} rows")

    all_diagnostics = {}

    for horizon in [20, 60, 90]:
        logger.info(f"\n--- Expert health for {horizon}d ---")
        config = HealthConfig(horizon=horizon)
        estimator = ExpertHealthEstimator(config)

        health_df, diag = estimator.compute(er_lgb, other_models if other_models else None)

        out_path = OUTPUT_DIR / f"expert_health_lgb_{horizon}d.parquet"
        health_df.to_parquet(out_path, index=False)
        logger.info(f"  Saved: {out_path} ({len(health_df)} dates)")

        all_diagnostics[f"{horizon}d"] = diag

        # Log key results
        crisis = diag.get("crisis_2024", {})
        if not crisis.get("skip"):
            logger.info(
                f"  Crisis 2024: H={crisis.get('crisis_mean_H', '?')}, "
                f"G={crisis.get('crisis_mean_G', '?')}, "
                f"drops={crisis.get('H_drops_in_crisis', '?')}"
            )
        logger.info(
            f"  ρ(H, RankIC)={diag.get('rho_H_rankic', '?')}, "
            f"AUROC={diag.get('auroc_bad_day', '?')}"
        )

    return all_diagnostics


# ── 13.5: Conformal intervals ────────────────────────────────────────────


def step_6_conformal():
    """13.5: Conformal prediction intervals (raw, vol-norm, DEUP-norm)."""
    logger.info("=" * 60)
    logger.info("STEP 13.5: Conformal Intervals")
    logger.info("=" * 60)

    from src.uncertainty.conformal_intervals import run_conformal_pipeline

    ehat_path = OUTPUT_DIR / "ehat_predictions.parquet"
    er_lgb_path = OUTPUT_DIR / "enriched_residuals_tabular_lgb.parquet"

    if not ehat_path.exists():
        raise FileNotFoundError(f"Run step 3 first: {ehat_path}")
    if not er_lgb_path.exists():
        raise FileNotFoundError(f"Run step 0 first: {er_lgb_path}")

    ehat_df = pd.read_parquet(ehat_path)
    enriched_df = pd.read_parquet(er_lgb_path)
    logger.info(f"Loaded ehat: {len(ehat_df):,} rows, enriched: {len(enriched_df):,} rows")

    intervals_df, diagnostics = run_conformal_pipeline(ehat_df, enriched_df)

    intervals_df.to_parquet(CONFORMAL_PATH, index=False)
    logger.info(f"Saved conformal predictions: {CONFORMAL_PATH} ({len(intervals_df):,} rows)")

    return diagnostics


# ── 13.6: Portfolio + regime evaluation ──────────────────────────────────


def step_7_portfolio():
    """13.6: DEUP-sized shadow portfolio + global regime evaluation."""
    logger.info("=" * 60)
    logger.info("STEP 13.6: DEUP-Sized Shadow Portfolio + Regime Evaluation")
    logger.info("=" * 60)

    from src.uncertainty.deup_portfolio import PortfolioConfig, run_portfolio_pipeline

    er_path = OUTPUT_DIR / "enriched_residuals_tabular_lgb.parquet"
    ehat_path = OUTPUT_DIR / "ehat_predictions.parquet"

    if not er_path.exists():
        raise FileNotFoundError(f"Run step 0 first: {er_path}")
    if not ehat_path.exists():
        raise FileNotFoundError(f"Run step 3 first: {ehat_path}")

    enriched = pd.read_parquet(er_path)
    ehat = pd.read_parquet(ehat_path)
    logger.info(f"Loaded enriched: {len(enriched):,}, ehat: {len(ehat):,}")

    all_portfolio_metrics = {}
    all_regime_eval = {}
    all_ts = []

    for horizon in [20, 60, 90]:
        health_path = OUTPUT_DIR / f"expert_health_lgb_{horizon}d.parquet"
        if not health_path.exists():
            logger.warning(f"  Skipping {horizon}d: {health_path} not found")
            continue

        health = pd.read_parquet(health_path)
        logger.info(f"\n--- {horizon}d portfolio ---")

        cfg = PortfolioConfig(horizon=horizon)
        portfolio_df, diag = run_portfolio_pipeline(enriched, ehat, health, horizon, cfg)

        portfolio_df["horizon"] = horizon
        all_ts.append(portfolio_df)

        all_portfolio_metrics[f"{horizon}d"] = diag["portfolio_metrics"]
        all_regime_eval[f"{horizon}d"] = {
            "regime_evaluation": diag["regime_evaluation"],
            "bucket_tables": diag["bucket_tables"],
            "aggregated_deup": diag["aggregated_deup"],
            "calibration": {
                "c_vol": diag["c_vol"],
                "c_deup": diag["c_deup"],
                "unc_col": diag["unc_col"],
                "dev_median_w_vol": diag["dev_median_w_vol"],
                "dev_median_w_deup": diag["dev_median_w_deup"],
            },
        }

        # Log key results
        pm = diag["portfolio_metrics"]
        for period in ["ALL", "FINAL", "CRISIS_2024"]:
            if period in pm:
                for vname in ["baseline_raw", "A_vol_sized", "B_deup_sized", "D_combined"]:
                    if vname in pm[period]:
                        sh = pm[period][vname].get("sharpe", "?")
                        logger.info(f"  {period} {vname}: Sharpe={sh}")

    # Save
    if all_ts:
        ts_df = pd.concat(all_ts, ignore_index=True)
        ts_df.to_parquet(DAILY_TS_PATH, index=False)
        logger.info(f"\nSaved daily timeseries: {DAILY_TS_PATH}")

    return all_portfolio_metrics, all_regime_eval


# ── Step 8: Deployment policy ablation (Chapter 13.7) ────────────────────


def step_8_policy() -> dict:
    """
    Chapter 13.7 — Deployment Policy & Sizing Ablation.

    Runs 6 binary-gate policy variants + kill-criterion trailing-IC variant
    on the 20d primary horizon. Produces:
        chapter13_7_policy_results.json   — ALL/DEV/FINAL/CRISIS metrics
        chapter13_7_calibration.json      — frozen DEV-calibrated params
        chapter13_7_ehat_score_diagnostic.json — structural conflict evidence
    """
    from src.uncertainty.deployment_policy import (
        run_policy_pipeline,
        print_results_table,
    )

    logger.info("\n=== Step 8: Chapter 13.7 Deployment Policy Ablation ===")

    # ── Load all data ──────────────────────────────────────────────────────
    enriched_path = OUTPUT_DIR / "enriched_residuals_tabular_lgb.parquet"
    ehat_path = OUTPUT_DIR / "ehat_predictions.parquet"
    health_path = OUTPUT_DIR / "expert_health_lgb_20d.parquet"

    for p in [enriched_path, ehat_path, health_path]:
        if not p.exists():
            logger.error(f"Required file missing: {p}")
            raise FileNotFoundError(f"Missing: {p}")

    enriched_df = pd.read_parquet(enriched_path)
    ehat_df = pd.read_parquet(ehat_path)
    health_df = pd.read_parquet(health_path)

    logger.info(
        f"  Loaded enriched: {len(enriched_df):,} rows, "
        f"ehat: {len(ehat_df):,} rows, "
        f"health: {len(health_df):,} rows"
    )

    # ── Run pipeline (20d primary horizon) ────────────────────────────────
    policy_results, cal_params, timeseries = run_policy_pipeline(
        enriched_df, ehat_df, health_df, horizon=20
    )

    # ── Save diagnostic ───────────────────────────────────────────────────
    diag = policy_results.pop("_diagnostic", {})
    policy_results.pop("_horizon", None)

    with open(EHAT_SCORE_DIAG_PATH, "w") as f:
        json.dump(diag, f, indent=2, default=str)
    logger.info(f"  ê–score diagnostic saved: {EHAT_SCORE_DIAG_PATH}")

    # ── Save calibration ──────────────────────────────────────────────────
    with open(POLICY_CALIBRATION_PATH, "w") as f:
        json.dump(cal_params, f, indent=2, default=str)
    logger.info(f"  Calibration saved: {POLICY_CALIBRATION_PATH}")

    # ── Save policy results ───────────────────────────────────────────────
    with open(POLICY_RESULTS_PATH, "w") as f:
        json.dump(policy_results, f, indent=2, default=str)
    logger.info(f"  Policy results saved: {POLICY_RESULTS_PATH}")

    # ── Load 13.6 baselines for comparison table ───────────────────────────
    # 13.6 json layout: {"20d": {"ALL": {"baseline_raw": {...}}, "DEV": ..., "FINAL": ..., "CRISIS_2024": ...}}
    # Need to reshape to:  {"baseline_raw": {"ALL": {...}, "DEV": {...}, "FINAL": {...}, "CRISIS_2024": {...}}}
    baseline_13_6 = None
    if PORTFOLIO_METRICS_PATH.exists():
        with open(PORTFOLIO_METRICS_PATH) as f:
            raw = json.load(f)
        hz = raw.get("20d", {})
        # hz maps period_name → {variant_name → metrics_dict}
        baseline_13_6 = {}
        for period_name, variants_dict in hz.items():
            if not isinstance(variants_dict, dict):
                continue
            for vname, metrics in variants_dict.items():
                if not isinstance(metrics, dict):
                    continue
                if vname not in baseline_13_6:
                    baseline_13_6[vname] = {}
                baseline_13_6[vname][period_name] = metrics

    # ── Print comparison table ────────────────────────────────────────────
    print_results_table(policy_results, baseline_13_6=baseline_13_6)

    # ── Log headline ──────────────────────────────────────────────────────
    for variant in ["gate_vol", "gate_ua_sort", "gate_vol_ehat_cap"]:
        if variant in policy_results:
            fin_sh = policy_results[variant].get("FINAL", {}).get("sharpe", "?")
            logger.info(f"  {variant} FINAL Sharpe = {fin_sh}")

    return policy_results


# ── Main ─────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Chapter 13 DEUP")
    parser.add_argument(
        "--step", type=int, default=None,
        help="Run specific step (0–8). Default: run all.",
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.step is None or args.step == 0:
        report_0 = step_0_populate_archive()
        diag_01 = {"step_0": report_0}
        if args.step is None:
            report_1 = step_1_train_g()
            diag_01["step_1"] = report_1
        with open(DIAGNOSTICS_01_PATH, "w") as f:
            json.dump(diag_01, f, indent=2, default=str)
        logger.info(f"\nDiagnostics 0/1 saved to {DIAGNOSTICS_01_PATH}")

    elif args.step == 1:
        report_1 = step_1_train_g()
        diag_01 = {"step_1": report_1}
        with open(DIAGNOSTICS_01_PATH, "w") as f:
            json.dump(diag_01, f, indent=2, default=str)
        logger.info(f"\nDiagnostics 0/1 saved to {DIAGNOSTICS_01_PATH}")

    if args.step is None or args.step == 2:
        report_2 = step_2_aleatoric()
        with open(DIAGNOSTICS_2_PATH, "w") as f:
            json.dump(report_2, f, indent=2, default=str)
        logger.info(f"\nDiagnostics 13.2 saved to {DIAGNOSTICS_2_PATH}")

    if args.step is None or args.step == 3:
        report_3 = step_3_epistemic()
        with open(DIAGNOSTICS_3_PATH, "w") as f:
            json.dump(report_3, f, indent=2, default=str)
        logger.info(f"\nDiagnostics 13.3 saved to {DIAGNOSTICS_3_PATH}")

    if args.step is None or args.step == 4:
        report_4 = step_4_diagnostics()
        with open(DIAGNOSTICS_4_PATH, "w") as f:
            json.dump(report_4, f, indent=2, default=str)
        logger.info(f"\nDiagnostics 13.4 saved to {DIAGNOSTICS_4_PATH}")

    if args.step is None or args.step == 5:
        report_5 = step_5_expert_health()
        with open(DIAGNOSTICS_HEALTH_PATH, "w") as f:
            json.dump(report_5, f, indent=2, default=str)
        logger.info(f"\nExpert health diagnostics saved to {DIAGNOSTICS_HEALTH_PATH}")

    if args.step is None or args.step == 6:
        report_6 = step_6_conformal()
        with open(DIAGNOSTICS_CONFORMAL_PATH, "w") as f:
            json.dump(report_6, f, indent=2, default=str)
        logger.info(f"\nConformal diagnostics saved to {DIAGNOSTICS_CONFORMAL_PATH}")

    if args.step is None or args.step == 7:
        pm, re = step_7_portfolio()
        with open(PORTFOLIO_METRICS_PATH, "w") as f:
            json.dump(pm, f, indent=2, default=str)
        with open(REGIME_EVAL_PATH, "w") as f:
            json.dump(re, f, indent=2, default=str)
        logger.info(f"\nPortfolio metrics saved to {PORTFOLIO_METRICS_PATH}")
        logger.info(f"Regime eval saved to {REGIME_EVAL_PATH}")

    if args.step is None or args.step == 8:
        step_8_policy()
        logger.info(
            f"\n13.7 results saved to:\n"
            f"  {POLICY_RESULTS_PATH}\n"
            f"  {POLICY_CALIBRATION_PATH}\n"
            f"  {EHAT_SCORE_DIAG_PATH}"
        )


if __name__ == "__main__":
    main()

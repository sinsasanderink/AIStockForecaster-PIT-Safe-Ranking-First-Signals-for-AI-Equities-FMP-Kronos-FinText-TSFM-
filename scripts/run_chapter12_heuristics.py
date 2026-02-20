#!/usr/bin/env python
"""
Chapter 12.3 — Regime-Aware Heuristic Baselines
=================================================

Implements two heuristic score-adjustment strategies on top of LGB baseline
eval_rows, then evaluates them on signal metrics + shadow portfolio.

These serve as ablation baselines for Chapter 13's DEUP-based approach.

Approach A — Volatility-Scaled Ranking:
    sized_score = score × min(1, c / vol_20d)
    Penalises stocks with above-median realized volatility in the ranking,
    promoting lower-volatility stocks into the top-10.

Approach B — Regime-Blended Ensemble:
    w_lgb = 1 - α × sigmoid((vix_pctile - threshold) / τ)
    blended_score = w_lgb × lgb_rank + (1 - w_lgb) × mom_rank
    In high-VIX: down-weights LGB ranking in favour of a simpler momentum signal.

Usage:
    python scripts/run_chapter12_heuristics.py \\
        --eval-path evaluation_outputs/chapter7_tabular_lgb_real/monthly/baseline_tabular_lgb_monthly/eval_rows.parquet \\
        --db-path data/features.duckdb \\
        --output-dir evaluation_outputs/chapter12/regime_heuristic
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

HORIZONS = [20, 60, 90]
PERIODS_PER_YEAR = 12

# Approach A defaults
VOL_SCALE_C = 0.25  # ~median vol_20d; stocks below this get multiplier ~1

# Approach B defaults
BLEND_ALPHA = 0.5       # max weight shift toward momentum
BLEND_THRESHOLD = 67.0  # VIX percentile centre of sigmoid
BLEND_TAU = 10.0        # sigmoid steepness


# ────────────────────────────────────────────────────────────────────
# Data loading
# ────────────────────────────────────────────────────────────────────

def load_features_for_join(db_path: str) -> pd.DataFrame:
    """Load vol_20d and mom_1m from features table, keyed by (date, stable_id)."""
    import duckdb

    conn = duckdb.connect(db_path, read_only=True)
    df = conn.execute(
        "SELECT date, stable_id, vol_20d, mom_1m FROM features"
    ).df()
    conn.close()
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df


def join_features_to_eval_rows(
    eval_rows: pd.DataFrame, feat_df: pd.DataFrame
) -> pd.DataFrame:
    """Left-join vol_20d and mom_1m onto eval_rows by (as_of_date, stable_id)."""
    er = eval_rows.copy()
    er["date_key"] = pd.to_datetime(er["as_of_date"]).dt.date

    feat_df = feat_df.rename(columns={"date": "date_key"})
    merged = er.merge(
        feat_df[["date_key", "stable_id", "vol_20d", "mom_1m"]],
        on=["date_key", "stable_id"],
        how="left",
    )
    logger.info(
        "  Feature join: vol_20d matched %.1f%%, mom_1m matched %.1f%%",
        100 * merged["vol_20d"].notna().mean(),
        100 * merged["mom_1m"].notna().mean(),
    )
    return merged


# ────────────────────────────────────────────────────────────────────
# Approach A: Volatility-Scaled Ranking
# ────────────────────────────────────────────────────────────────────

def apply_vol_sizing(
    eval_rows: pd.DataFrame, c: float = VOL_SCALE_C
) -> pd.DataFrame:
    """
    Scale scores by inverse realized volatility.

    sized_score = score × min(1, c / vol_20d)

    Stocks with vol_20d below c are unaffected; those above get penalised.
    Stocks with missing vol_20d keep their original score.
    """
    er = eval_rows.copy()
    vol = er["vol_20d"].copy()
    vol = vol.clip(lower=0.01)  # avoid div by zero
    scale = np.minimum(1.0, c / vol)
    scale = scale.fillna(1.0)
    er["score"] = er["score"] * scale
    return er


# ────────────────────────────────────────────────────────────────────
# Approach B: Regime-Blended Ensemble
# ────────────────────────────────────────────────────────────────────

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def apply_regime_blending(
    eval_rows: pd.DataFrame,
    alpha: float = BLEND_ALPHA,
    threshold: float = BLEND_THRESHOLD,
    tau: float = BLEND_TAU,
) -> pd.DataFrame:
    """
    Blend LGB scores with momentum-based ranks, weighted by VIX regime.

    w_lgb = 1 - α × sigmoid((vix_pctile - threshold) / τ)
    blended_score = w_lgb × lgb_rank + (1 - w_lgb) × mom_rank

    Both lgb_rank and mom_rank are cross-sectional percentile ranks (0–1)
    computed per (date, horizon) group to ensure comparable scales.
    """
    er = eval_rows.copy()

    # Cross-sectional percentile ranks per date+horizon
    er["lgb_rank"] = er.groupby(["as_of_date", "horizon"])["score"].rank(pct=True)

    mom = er["mom_1m"].copy()
    mom = mom.fillna(0.0)
    er["mom_rank"] = er.groupby(["as_of_date", "horizon"])[
        "mom_1m"
    ].rank(pct=True, na_option="bottom")

    # Blending weight from VIX percentile
    vix_pctile = er["vix_percentile_252d"].fillna(50.0)
    w_lgb = 1.0 - alpha * sigmoid((vix_pctile.values - threshold) / tau)

    er["score"] = w_lgb * er["lgb_rank"].values + (1.0 - w_lgb) * er["mom_rank"].values

    er.drop(columns=["lgb_rank", "mom_rank"], inplace=True)
    return er


# ────────────────────────────────────────────────────────────────────
# Metrics (reused from evaluate_fusion_gates logic)
# ────────────────────────────────────────────────────────────────────

def compute_metrics(eval_rows: pd.DataFrame) -> Dict[int, dict]:
    """Compute per-horizon RankIC, IC stability, churn, cost survival."""
    metrics: Dict[int, dict] = {}

    for horizon in HORIZONS:
        h_rows = eval_rows[eval_rows["horizon"] == horizon].copy()
        if h_rows.empty:
            continue

        per_date_ic = h_rows.groupby("as_of_date")[["score", "excess_return"]].apply(
            lambda g: stats.spearmanr(g["score"], g["excess_return"]).statistic
            if len(g) >= 5
            else np.nan
        )
        per_date_ic = per_date_ic.dropna()

        ic_mean = float(per_date_ic.mean()) if len(per_date_ic) else np.nan
        ic_median = float(per_date_ic.median()) if len(per_date_ic) else np.nan
        ic_std = float(per_date_ic.std()) if len(per_date_ic) > 1 else np.nan
        ic_stability = (
            ic_mean / ic_std
            if not np.isnan(ic_mean) and not np.isnan(ic_std) and ic_std > 0
            else np.nan
        )

        # Cost survival
        fold_ids = h_rows["fold_id"].unique() if "fold_id" in h_rows.columns else []
        fold_positive = 0
        fold_total = 0
        for fid in fold_ids:
            fold_h = h_rows[h_rows["fold_id"] == fid]
            if fold_h.empty:
                continue
            top10_er = (
                fold_h.groupby("as_of_date", group_keys=False)
                .apply(
                    lambda g: g.nlargest(10, "score")["excess_return"].mean()
                    if len(g) >= 10
                    else np.nan,
                    include_groups=False,
                )
                .dropna()
            )
            if len(top10_er) == 0:
                continue
            fold_total += 1
            if top10_er.median() > 0:
                fold_positive += 1
        cost_survival = fold_positive / fold_total if fold_total > 0 else np.nan

        # Churn
        dates_sorted = sorted(h_rows["as_of_date"].unique())
        churns = []
        for i in range(1, len(dates_sorted)):
            prev_top = set(
                h_rows[h_rows["as_of_date"] == dates_sorted[i - 1]]
                .nlargest(10, "score")["stable_id"]
            )
            curr_top = set(
                h_rows[h_rows["as_of_date"] == dates_sorted[i]]
                .nlargest(10, "score")["stable_id"]
            )
            if prev_top and curr_top:
                churns.append(1.0 - len(prev_top & curr_top) / 10.0)

        metrics[horizon] = {
            "mean_rankic": ic_mean,
            "median_rankic": ic_median,
            "ic_stability": ic_stability,
            "pct_positive": float(np.mean(per_date_ic > 0)) if len(per_date_ic) else np.nan,
            "cost_survival": cost_survival,
            "median_churn": float(np.median(churns)) if churns else np.nan,
            "mean_churn": float(np.mean(churns)) if churns else np.nan,
            "n_dates": int(len(per_date_ic)),
        }

    return metrics


# ────────────────────────────────────────────────────────────────────
# Shadow portfolio (simplified from run_shadow_portfolio.py)
# ────────────────────────────────────────────────────────────────────

TOP_K = 10
COST_BPS = 10


def build_shadow_portfolio(
    eval_rows: pd.DataFrame, horizon: int = 20
) -> pd.DataFrame:
    """Build monthly L/S portfolio and return non-overlapping monthly returns."""
    h_rows = eval_rows[eval_rows["horizon"] == horizon].copy()
    if h_rows.empty:
        return pd.DataFrame()

    h_rows["as_of_date"] = pd.to_datetime(h_rows["as_of_date"])
    dates = sorted(h_rows["as_of_date"].unique())

    results = []
    prev_long = set()
    prev_short = set()

    for dt in dates:
        day_df = h_rows[h_rows["as_of_date"] == dt].sort_values(
            "score", ascending=False
        )
        if len(day_df) < 2 * TOP_K:
            continue

        long_ids = set(day_df.head(TOP_K)["stable_id"])
        short_ids = set(day_df.tail(TOP_K)["stable_id"])

        long_ret = day_df[day_df["stable_id"].isin(long_ids)]["excess_return"].mean()
        short_ret = day_df[day_df["stable_id"].isin(short_ids)]["excess_return"].mean()
        ls_return = long_ret - short_ret

        if prev_long:
            long_turnover = 1 - len(long_ids & prev_long) / TOP_K
            short_turnover = 1 - len(short_ids & prev_short) / TOP_K
            turnover = (long_turnover + short_turnover) / 2
        else:
            turnover = 1.0

        cost_drag = turnover * (COST_BPS / 10000) * 2
        results.append({
            "date": dt,
            "ls_return_net": ls_return - cost_drag,
            "turnover": turnover,
        })

        prev_long = long_ids
        prev_short = short_ids

    df = pd.DataFrame(results)
    if df.empty:
        return df

    # Subsample to non-overlapping monthly (first day per calendar month)
    df["ym"] = df["date"].dt.to_period("M")
    monthly = df.groupby("ym").first().reset_index(drop=True)
    return monthly


def compute_portfolio_metrics(monthly_df: pd.DataFrame) -> dict:
    """Annualized portfolio metrics from non-overlapping monthly returns."""
    if monthly_df.empty or len(monthly_df) < 5:
        return {
            "ann_sharpe": np.nan, "ann_return": np.nan, "ann_vol": np.nan,
            "max_drawdown": np.nan, "hit_rate": np.nan, "n_months": 0,
        }

    ret = monthly_df["ls_return_net"]
    mean_r = ret.mean()
    std_r = ret.std(ddof=1)
    ann_ret = mean_r * PERIODS_PER_YEAR
    ann_vol = std_r * np.sqrt(PERIODS_PER_YEAR)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0

    cum = (1 + ret).cumprod()
    max_dd = float(((cum - cum.cummax()) / cum.cummax()).min())

    return {
        "ann_sharpe": float(sharpe),
        "ann_return": float(ann_ret),
        "ann_vol": float(ann_vol),
        "max_drawdown": float(max_dd),
        "hit_rate": float((ret > 0).mean()),
        "n_months": int(len(ret)),
    }


# ────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────

def run_heuristic_evaluation(
    eval_rows_path: str,
    db_path: str,
    output_dir: Path,
):
    """Run both heuristic approaches and produce comparison."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load base data
    logger.info("Loading LGB baseline eval_rows from %s", eval_rows_path)
    er_base = pd.read_parquet(eval_rows_path)
    logger.info("  %d rows, %d dates, %d horizons", len(er_base), er_base["as_of_date"].nunique(), er_base["horizon"].nunique())

    logger.info("Loading features from %s", db_path)
    feat_df = load_features_for_join(db_path)
    logger.info("  %d feature rows loaded", len(feat_df))

    er_enriched = join_features_to_eval_rows(er_base, feat_df)

    # ── Baseline ──
    logger.info("\n=== LGB Baseline (original) ===")
    base_metrics = compute_metrics(er_base)
    base_portfolio = build_shadow_portfolio(er_base)
    base_port_metrics = compute_portfolio_metrics(base_portfolio)

    # ── Approach A: Vol-sized ──
    logger.info("\n=== Approach A: Volatility-Scaled (c=%.2f) ===", VOL_SCALE_C)
    er_vol = apply_vol_sizing(er_enriched.copy(), c=VOL_SCALE_C)
    vol_metrics = compute_metrics(er_vol)
    vol_portfolio = build_shadow_portfolio(er_vol)
    vol_port_metrics = compute_portfolio_metrics(vol_portfolio)

    # ── Approach B: Regime-blended ──
    logger.info("\n=== Approach B: Regime-Blended (α=%.2f, τ=%.1f) ===", BLEND_ALPHA, BLEND_TAU)
    er_blend = apply_regime_blending(er_enriched.copy())
    blend_metrics = compute_metrics(er_blend)
    blend_portfolio = build_shadow_portfolio(er_blend)
    blend_port_metrics = compute_portfolio_metrics(blend_portfolio)

    # ── Comparison table ──
    all_results = {
        "LGB_baseline": {"signal": base_metrics, "portfolio": base_port_metrics},
        "Vol_Sized": {"signal": vol_metrics, "portfolio": vol_port_metrics},
        "Regime_Blended": {"signal": blend_metrics, "portfolio": blend_port_metrics},
    }

    # Save
    json_path = output_dir / "heuristic_comparison.json"
    with json_path.open("w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info("Saved to %s", json_path)

    # Save modified eval_rows
    for name, er_df in [("vol_sized", er_vol), ("regime_blended", er_blend)]:
        p = output_dir / name / "eval_rows.parquet"
        p.parent.mkdir(parents=True, exist_ok=True)
        cols_to_save = [c for c in er_base.columns if c in er_df.columns]
        er_df[cols_to_save].to_parquet(p, index=False)
        logger.info("Saved %s eval_rows to %s", name, p)

    # Print comparison
    print("\n" + "=" * 100)
    print("CHAPTER 12.3 — REGIME-AWARE HEURISTIC BASELINE COMPARISON")
    print("=" * 100)

    models = ["LGB_baseline", "Vol_Sized", "Regime_Blended"]

    # Signal metrics table
    print("\n--- Signal Metrics (FULL mode, 109 folds) ---\n")
    print(
        f"  {'Model':<20} {'Hz':>4}"
        f" {'MeanIC':>8} {'MedIC':>8} {'IC_Stab':>8}"
        f" {'CostSurv':>9} {'Churn':>7} {'%Pos':>6}"
    )
    print(f"  {'-' * 80}")

    for model_name in models:
        sig = all_results[model_name]["signal"]
        for h in HORIZONS:
            m = sig.get(h, {})
            print(
                f"  {model_name:<20} {h:>3}d"
                f" {m.get('mean_rankic', 0):>8.4f}"
                f" {m.get('median_rankic', 0):>8.4f}"
                f" {m.get('ic_stability', 0):>8.3f}"
                f" {m.get('cost_survival', 0):>8.1%}"
                f" {m.get('median_churn', 0):>7.2f}"
                f" {m.get('pct_positive', 0):>5.1%}"
            )

    # Portfolio metrics table
    print("\n--- Shadow Portfolio (20d, non-overlapping monthly, annualized ×12/×√12) ---\n")
    print(
        f"  {'Model':<20}"
        f" {'Sharpe':>8} {'Return':>8} {'Vol':>8}"
        f" {'MaxDD':>8} {'HitRate':>8} {'N_mo':>6}"
    )
    print(f"  {'-' * 56}")

    for model_name in models:
        p = all_results[model_name]["portfolio"]
        print(
            f"  {model_name:<20}"
            f" {p.get('ann_sharpe', 0):>8.2f}"
            f" {p.get('ann_return', 0):>7.1%}"
            f" {p.get('ann_vol', 0):>7.1%}"
            f" {p.get('max_drawdown', 0):>7.1%}"
            f" {p.get('hit_rate', 0):>7.1%}"
            f" {p.get('n_months', 0):>6d}"
        )

    # Gate assessment
    print("\n--- Gate Assessment ---")
    b_sharpe = base_port_metrics.get("ann_sharpe", 0)
    b_dd = base_port_metrics.get("max_drawdown", 0)

    for heur, name in [(vol_port_metrics, "Vol_Sized"), (blend_port_metrics, "Regime_Blended")]:
        h_sharpe = heur.get("ann_sharpe", 0)
        h_dd = heur.get("max_drawdown", 0)
        sharpe_better = h_sharpe > b_sharpe
        dd_better = h_dd > b_dd  # less negative = better
        print(
            f"  {name}: Sharpe {'IMPROVED' if sharpe_better else 'WORSE'}"
            f" ({b_sharpe:.2f} → {h_sharpe:.2f}),"
            f" MaxDD {'IMPROVED' if dd_better else 'WORSE'}"
            f" ({b_dd:.1%} → {h_dd:.1%})"
        )

    print("\n" + "=" * 100)
    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Chapter 12.3 — Regime-Aware Heuristic Baselines"
    )
    parser.add_argument(
        "--eval-path",
        default="evaluation_outputs/chapter7_tabular_lgb_real/monthly/baseline_tabular_lgb_monthly/eval_rows.parquet",
        help="Path to LGB baseline eval_rows.parquet",
    )
    parser.add_argument(
        "--db-path",
        default="data/features.duckdb",
        help="Path to DuckDB features database",
    )
    parser.add_argument(
        "--output-dir",
        default="evaluation_outputs/chapter12/regime_heuristic",
        help="Output directory",
    )
    args = parser.parse_args()
    run_heuristic_evaluation(args.eval_path, args.db_path, Path(args.output_dir))


if __name__ == "__main__":
    main()

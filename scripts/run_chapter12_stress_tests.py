#!/usr/bin/env python
"""
Chapter 12.2 — Regime Stress Tests (Shadow Portfolio)
=====================================================

Annotates shadow-portfolio returns with regime labels and computes
per-regime Sharpe, return, drawdown, and hit rate.

IMPORTANT: shadow_portfolio_returns.csv contains one row per trading day,
but each row's ls_return is the 20-day FORWARD excess return of the
portfolio formed on that date.  Consecutive rows overlap by 19/20 days.

To avoid inflated Sharpe estimates we subsample to the first trading day
per calendar month, yielding ~109 non-overlapping monthly observations.
Annualization uses periods_per_year=12 (monthly frequency).

Outputs:
  - evaluation_outputs/chapter12/regime_stress_report.json
  - evaluation_outputs/chapter12/regime_shadow_metrics.csv
  - evaluation_outputs/chapter12/worst_drawdowns.csv
  - evaluation_outputs/chapter12/rolling_sharpe.csv
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

PERIODS_PER_YEAR = 12  # monthly frequency after subsampling
ROLLING_WINDOW = 12    # 12-month rolling Sharpe (in monthly periods)


def load_regime_lookup(db_path: str = "data/features.duckdb") -> pd.DataFrame:
    """Load regime data keyed by date."""
    import duckdb

    conn = duckdb.connect(db_path, read_only=True)
    df = conn.execute("SELECT * FROM regime").df()
    conn.close()

    df["date"] = pd.to_datetime(df["date"]).dt.date
    market_map = {-1: "bear", 0: "neutral", 1: "bull"}
    df["market_regime_label"] = df["market_regime"].map(market_map)

    vix_pctile = df["vix_percentile_252d"]
    df["vix_bucket"] = pd.cut(
        vix_pctile,
        bins=[-np.inf, 33, 67, np.inf],
        labels=["low", "mid", "high"],
    )
    return df.set_index("date")


def load_returns(path: str) -> pd.DataFrame:
    """Load shadow_portfolio_returns.csv and parse dates."""
    df = pd.read_csv(path, parse_dates=["date"])
    df["date_key"] = df["date"].dt.date
    return df


def subsample_monthly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Subsample to the first trading day per calendar month.

    This converts overlapping 20-day forward returns into non-overlapping
    monthly observations suitable for standard annualization.
    """
    df = df.copy()
    df["ym"] = df["date"].dt.to_period("M")
    monthly = df.groupby("ym").first().reset_index(drop=True)
    return monthly


def annotate_with_regime(
    returns_df: pd.DataFrame, regime_lookup: pd.DataFrame
) -> pd.DataFrame:
    """Join regime labels onto return series."""
    df = returns_df.copy()
    for col in ["vix_bucket", "market_regime_label", "vix_percentile_252d", "vix_level"]:
        if col in regime_lookup.columns:
            df[col] = df["date_key"].map(regime_lookup[col])
    return df


def compute_portfolio_metrics(
    returns: pd.Series, label: str = "", min_obs: int = 5
) -> dict:
    """
    Annualized Sharpe, return, vol, max DD, hit rate from monthly returns.

    Annualizes with PERIODS_PER_YEAR=12 (monthly frequency).
    """
    if len(returns) < min_obs:
        return {
            "label": label,
            "n_months": len(returns),
            "ann_sharpe": np.nan,
            "ann_return": np.nan,
            "ann_vol": np.nan,
            "max_drawdown": np.nan,
            "hit_rate": np.nan,
        }

    mean_r = returns.mean()
    std_r = returns.std(ddof=1)
    ann_ret = mean_r * PERIODS_PER_YEAR
    ann_vol = std_r * np.sqrt(PERIODS_PER_YEAR)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0

    cum = (1 + returns).cumprod()
    running_max = cum.cummax()
    drawdown = (cum - running_max) / running_max
    max_dd = float(drawdown.min())

    hit = float((returns > 0).mean())

    return {
        "label": label,
        "n_months": int(len(returns)),
        "ann_sharpe": float(sharpe),
        "ann_return": float(ann_ret),
        "ann_vol": float(ann_vol),
        "max_drawdown": float(max_dd),
        "hit_rate": float(hit),
    }


def find_worst_drawdowns(
    monthly_df: pd.DataFrame, n: int = 5
) -> pd.DataFrame:
    """Find the N worst drawdown episodes with start/trough/recovery dates."""
    cum = (1 + monthly_df["ls_return_net"]).cumprod()
    running_max = cum.cummax()
    drawdown = (cum - running_max) / running_max

    episodes = []
    in_dd = False
    start_idx = 0

    for i in range(len(drawdown)):
        if drawdown.iloc[i] < 0 and not in_dd:
            in_dd = True
            start_idx = i - 1 if i > 0 else 0
        elif drawdown.iloc[i] >= 0 and in_dd:
            in_dd = False
            trough_idx = drawdown.iloc[start_idx : i + 1].idxmin()
            episodes.append({
                "start_date": monthly_df["date"].iloc[start_idx],
                "trough_date": monthly_df["date"].iloc[trough_idx],
                "end_date": monthly_df["date"].iloc[i],
                "max_dd": float(drawdown.iloc[trough_idx]),
                "duration_months": i - start_idx,
            })

    if in_dd:
        trough_idx = drawdown.iloc[start_idx:].idxmin()
        episodes.append({
            "start_date": monthly_df["date"].iloc[start_idx],
            "trough_date": monthly_df["date"].iloc[trough_idx],
            "end_date": monthly_df["date"].iloc[-1],
            "max_dd": float(drawdown.iloc[trough_idx]),
            "duration_months": len(drawdown) - start_idx,
        })

    ep_df = pd.DataFrame(episodes).sort_values("max_dd").head(n).reset_index(drop=True)
    return ep_df


def compute_rolling_sharpe(
    returns: pd.Series, window: int = ROLLING_WINDOW
) -> pd.Series:
    """Rolling annualized Sharpe from monthly returns."""
    rolling_mean = returns.rolling(window).mean() * PERIODS_PER_YEAR
    rolling_std = returns.rolling(window).std(ddof=1) * np.sqrt(PERIODS_PER_YEAR)
    return (rolling_mean / rolling_std).dropna()


def run_stress_tests(
    returns_paths: Dict[str, str],
    regime_lookup: pd.DataFrame,
    output_dir: Path,
):
    """Run full regime stress test suite on non-overlapping monthly returns."""
    all_regime_metrics = []
    all_rolling = []
    all_drawdowns = {}

    for model_name, ret_path in returns_paths.items():
        logger.info("Processing %s ...", model_name)
        raw_df = load_returns(ret_path)
        logger.info("  Raw rows: %d (overlapping 20d returns)", len(raw_df))

        monthly_df = subsample_monthly(raw_df)
        monthly_df = annotate_with_regime(monthly_df, regime_lookup)

        matched = monthly_df["vix_bucket"].notna().sum()
        logger.info(
            "  Monthly (non-overlapping): %d months, %d matched to regime (%.0f%%)",
            len(monthly_df), matched, 100 * matched / max(len(monthly_df), 1),
        )

        # Overall metrics
        overall = compute_portfolio_metrics(monthly_df["ls_return_net"], "ALL")
        overall["model"] = model_name
        overall["regime_axis"] = "overall"
        overall["bucket"] = "ALL"
        all_regime_metrics.append(overall)

        # Per VIX bucket
        for bucket in ["low", "mid", "high"]:
            mask = monthly_df["vix_bucket"] == bucket
            sub = monthly_df.loc[mask, "ls_return_net"]
            m = compute_portfolio_metrics(sub, bucket)
            m["model"] = model_name
            m["regime_axis"] = "vix_bucket"
            m["bucket"] = bucket
            all_regime_metrics.append(m)

        # Per market regime
        for regime in ["bull", "neutral", "bear"]:
            mask = monthly_df["market_regime_label"] == regime
            sub = monthly_df.loc[mask, "ls_return_net"]
            m = compute_portfolio_metrics(sub, regime)
            m["model"] = model_name
            m["regime_axis"] = "market_regime"
            m["bucket"] = regime
            all_regime_metrics.append(m)

        # Rolling 12-month Sharpe (on monthly returns)
        rolling_s = compute_rolling_sharpe(monthly_df["ls_return_net"])
        if len(rolling_s) > 0:
            rs_df = pd.DataFrame({
                "date": monthly_df["date"].iloc[ROLLING_WINDOW - 1:].values[:len(rolling_s)],
                "rolling_sharpe": rolling_s.values,
                "model": model_name,
            })
            if "vix_bucket" in monthly_df.columns:
                rs_df["vix_bucket"] = monthly_df["vix_bucket"].iloc[
                    ROLLING_WINDOW - 1:
                ].values[:len(rolling_s)]
                rs_df["market_regime"] = monthly_df["market_regime_label"].iloc[
                    ROLLING_WINDOW - 1:
                ].values[:len(rolling_s)]
            all_rolling.append(rs_df)

        # Worst drawdowns (on non-overlapping monthly equity curve)
        dd_df = find_worst_drawdowns(monthly_df, n=5)
        if not dd_df.empty:
            dd_df["trough_date_key"] = pd.to_datetime(dd_df["trough_date"]).dt.date
            for col in ["vix_bucket", "market_regime_label"]:
                dd_df[col] = dd_df["trough_date_key"].map(
                    regime_lookup[col] if col in regime_lookup.columns else {}
                )
            dd_df["model"] = model_name
        all_drawdowns[model_name] = dd_df

    # Assemble outputs
    metrics_df = pd.DataFrame(all_regime_metrics)
    col_order = [
        "model", "regime_axis", "bucket", "n_months",
        "ann_sharpe", "ann_return", "ann_vol", "max_drawdown", "hit_rate",
    ]
    metrics_df = metrics_df[[c for c in col_order if c in metrics_df.columns]]

    rolling_df = pd.concat(all_rolling, ignore_index=True) if all_rolling else pd.DataFrame()
    dd_combined = pd.concat(all_drawdowns.values(), ignore_index=True)

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "regime_shadow_metrics.csv"
    metrics_df.to_csv(csv_path, index=False, float_format="%.4f")
    logger.info("Saved regime shadow metrics to %s", csv_path)

    rolling_path = output_dir / "rolling_sharpe.csv"
    rolling_df.to_csv(rolling_path, index=False, float_format="%.4f")
    logger.info("Saved rolling Sharpe to %s (%d rows)", rolling_path, len(rolling_df))

    dd_path = output_dir / "worst_drawdowns.csv"
    dd_combined.to_csv(dd_path, index=False, float_format="%.4f")
    logger.info("Saved worst drawdowns to %s", dd_path)

    report = {
        "methodology": {
            "return_frequency": "monthly (non-overlapping, first trading day per month)",
            "annualization": "×12 for return, ×√12 for vol",
            "note": "Raw CSV contains overlapping 20d forward returns; subsampled to avoid Sharpe inflation",
        },
        "regime_metrics": metrics_df.to_dict(orient="records"),
        "rolling_sharpe_summary": {},
        "worst_drawdowns": {},
    }
    for model_name in returns_paths:
        if not rolling_df.empty:
            rs = rolling_df[rolling_df["model"] == model_name]["rolling_sharpe"]
            report["rolling_sharpe_summary"][model_name] = {
                "mean": float(rs.mean()),
                "min": float(rs.min()),
                "max": float(rs.max()),
                "pct_negative": float((rs < 0).mean()),
            }
        dd = all_drawdowns.get(model_name)
        if dd is not None and not dd.empty:
            report["worst_drawdowns"][model_name] = dd.to_dict(orient="records")

    json_path = output_dir / "regime_stress_report.json"
    with json_path.open("w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info("Saved stress report to %s", json_path)

    # Print summary
    print("\n" + "=" * 90)
    print("CHAPTER 12.2 — REGIME STRESS TESTS (SHADOW PORTFOLIO)")
    print("  Non-overlapping monthly returns, annualized ×12 / ×√12")
    print("=" * 90)

    for model_name in returns_paths:
        mdf = metrics_df[metrics_df["model"] == model_name]
        print(f"\n--- {model_name} ---")
        print(
            f"  {'Axis':<16} {'Bucket':<10} {'N':>4}"
            f" {'Sharpe':>8} {'Return':>8} {'Vol':>8}"
            f" {'MaxDD':>8} {'HitRate':>8}"
        )
        print(f"  {'-' * 72}")
        for _, row in mdf.iterrows():
            print(
                f"  {row['regime_axis']:<16} {row['bucket']:<10}"
                f" {row['n_months']:>4}"
                f" {row['ann_sharpe']:>8.2f}"
                f" {row['ann_return']:>7.1%}"
                f" {row['ann_vol']:>7.1%}"
                f" {row['max_drawdown']:>7.1%}"
                f" {row['hit_rate']:>7.1%}"
            )

    print("\n--- Worst Drawdowns (monthly equity curve) ---")
    for model_name in returns_paths:
        dd = all_drawdowns.get(model_name)
        if dd is not None and not dd.empty:
            print(f"\n  {model_name}:")
            for _, row in dd.iterrows():
                vix_b = row.get("vix_bucket", "?")
                mkt_r = row.get("market_regime_label", "?")
                print(
                    f"    {row['max_dd']:>7.1%} |"
                    f" {str(row['trough_date'])[:10]}"
                    f" | {row['duration_months']:>3}mo"
                    f" | VIX={vix_b}, Mkt={mkt_r}"
                )

    if not rolling_df.empty:
        print("\n--- Rolling 12-month Sharpe Summary ---")
        for model_name in returns_paths:
            s = report["rolling_sharpe_summary"].get(model_name, {})
            print(
                f"  {model_name}: mean={s.get('mean', 0):.2f},"
                f" min={s.get('min', 0):.2f},"
                f" max={s.get('max', 0):.2f},"
                f" %negative={s.get('pct_negative', 0):.1%}"
            )

    print("\n" + "=" * 90)


def main():
    parser = argparse.ArgumentParser(
        description="Chapter 12.2 — Regime Stress Tests (Shadow Portfolio)"
    )
    parser.add_argument(
        "--returns-paths",
        nargs="+",
        required=True,
        help="name=path pairs to shadow_portfolio_returns.csv files",
    )
    parser.add_argument(
        "--db-path",
        default="data/features.duckdb",
        help="Path to DuckDB features database",
    )
    parser.add_argument(
        "--output-dir",
        default="evaluation_outputs/chapter12",
        help="Output directory",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    regime_lookup = load_regime_lookup(args.db_path)
    logger.info("Loaded regime lookup: %d dates", len(regime_lookup))

    paths = {}
    for pair in args.returns_paths:
        if "=" not in pair:
            raise ValueError(f"Expected name=path, got: {pair}")
        name, path = pair.split("=", 1)
        paths[name] = path

    run_stress_tests(paths, regime_lookup, output_dir)


if __name__ == "__main__":
    main()

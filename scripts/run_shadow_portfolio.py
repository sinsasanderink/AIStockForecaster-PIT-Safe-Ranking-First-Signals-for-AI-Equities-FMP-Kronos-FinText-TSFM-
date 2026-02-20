#!/usr/bin/env python
"""
Chapter 11.5 — Shadow Portfolio Report
=========================================

Translates cross-sectional ranking scores into a simple, frozen
dollar-neutral long/short portfolio and computes institutional-grade
metrics: Sharpe, IR, drawdown, turnover, cost drag.

The mapping is deliberately simple (no optimization) to serve as an
evaluation-only sanity check on signal quality.

Usage:
    python scripts/run_shadow_portfolio.py \\
        --eval-dir evaluation_outputs/chapter11_fusion_smoke/rank_avg_3/ch11_rank_avg_3_smoke
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

TRADING_DAYS_PER_YEAR = 252
DEFAULT_COST_BPS = 10
TOP_K = 10


def build_shadow_portfolio(
    eval_rows: pd.DataFrame,
    horizon: int = 20,
    top_k: int = TOP_K,
    cost_bps: float = DEFAULT_COST_BPS,
) -> pd.DataFrame:
    """
    Build a dollar-neutral long/short shadow portfolio from eval rows.

    Frozen mapping:
    - Long: top-K stocks by score, equal-weight (+50% gross)
    - Short: bottom-K stocks by score, equal-weight (-50% gross)
    - Monthly rebalance (first trading day)
    - 20d horizon returns used (avoids overlapping-hold complexity)

    Returns DataFrame with one row per rebalance date:
        date, long_return, short_return, ls_return, turnover, cost_drag
    """
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
        if len(day_df) < 2 * top_k:
            continue

        long_ids = set(day_df.head(top_k)["stable_id"])
        short_ids = set(day_df.tail(top_k)["stable_id"])

        long_ret = day_df[day_df["stable_id"].isin(long_ids)][
            "excess_return"
        ].mean()
        short_ret = day_df[day_df["stable_id"].isin(short_ids)][
            "excess_return"
        ].mean()
        ls_return = long_ret - short_ret

        # Turnover
        if prev_long:
            long_turnover = 1 - len(long_ids & prev_long) / top_k
            short_turnover = 1 - len(short_ids & prev_short) / top_k
            turnover = (long_turnover + short_turnover) / 2
        else:
            turnover = 1.0  # Full build on first date

        cost_drag = turnover * (cost_bps / 10000) * 2  # Both legs

        results.append({
            "date": dt,
            "long_return": long_ret,
            "short_return": short_ret,
            "ls_return": ls_return,
            "ls_return_net": ls_return - cost_drag,
            "turnover": turnover,
            "cost_drag": cost_drag,
            "n_long": len(long_ids),
            "n_short": len(short_ids),
        })

        prev_long = long_ids
        prev_short = short_ids

    return pd.DataFrame(results)


def compute_portfolio_metrics(portfolio_df: pd.DataFrame) -> dict:
    """Compute institutional-grade metrics from shadow portfolio returns."""
    if portfolio_df.empty:
        return {}

    ret = portfolio_df["ls_return_net"].values
    n = len(ret)

    # Annualization: monthly returns → annual
    periods_per_year = 12  # Monthly rebalance

    mean_ret = np.mean(ret)
    std_ret = np.std(ret, ddof=1) if n > 1 else 0.001

    sharpe = (mean_ret / std_ret) * np.sqrt(periods_per_year) if std_ret > 0 else 0.0

    # Cumulative returns
    cum_ret = np.cumprod(1 + ret)
    peak = np.maximum.accumulate(cum_ret)
    drawdowns = (cum_ret - peak) / peak
    max_drawdown = float(drawdowns.min())
    worst_month = float(np.min(ret))

    # Turnover
    mean_turnover = float(portfolio_df["turnover"].mean())
    total_cost_drag = float(portfolio_df["cost_drag"].sum())

    # Rolling 12-month Sharpe (if enough data)
    rolling_sharpe = []
    if n >= 12:
        for i in range(12, n + 1):
            window = ret[i - 12 : i]
            ws = np.std(window, ddof=1)
            if ws > 0:
                rolling_sharpe.append(
                    np.mean(window) / ws * np.sqrt(12)
                )

    metrics = {
        "n_periods": n,
        "annualized_sharpe": float(sharpe),
        "annualized_return": float(mean_ret * periods_per_year),
        "annualized_volatility": float(std_ret * np.sqrt(periods_per_year)),
        "max_drawdown": max_drawdown,
        "worst_month": worst_month,
        "mean_turnover": mean_turnover,
        "total_cost_drag_bps": total_cost_drag * 10000,
        "hit_rate": float(np.mean(ret > 0)),
    }

    if rolling_sharpe:
        metrics["rolling_12m_sharpe_mean"] = float(np.mean(rolling_sharpe))
        metrics["rolling_12m_sharpe_min"] = float(np.min(rolling_sharpe))
        metrics["rolling_12m_sharpe_max"] = float(np.max(rolling_sharpe))

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Chapter 11.5: Shadow Portfolio Report"
    )
    parser.add_argument(
        "--eval-dir",
        required=True,
        help="Path to evaluation output directory with eval_rows.parquet",
    )
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--cost-bps", type=float, default=DEFAULT_COST_BPS)
    parser.add_argument(
        "--compare-dirs",
        nargs="*",
        default=[],
        help="Additional eval dirs to compare (e.g., LGB baseline)",
    )
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    logger.info("=" * 60)
    logger.info("CHAPTER 11.5: SHADOW PORTFOLIO REPORT")
    logger.info("=" * 60)
    logger.info(f"Eval dir: {eval_dir}")
    logger.info(f"Horizon: {args.horizon}d, Top-K: {args.top_k}")

    # Build primary portfolio
    eval_rows = pd.read_parquet(eval_dir / "eval_rows.parquet")
    logger.info(f"Loaded {len(eval_rows):,} eval rows")

    portfolio = build_shadow_portfolio(
        eval_rows, horizon=args.horizon, top_k=args.top_k,
        cost_bps=args.cost_bps,
    )
    metrics = compute_portfolio_metrics(portfolio)

    logger.info("\n--- Shadow Portfolio Metrics ---")
    for k, v in metrics.items():
        if isinstance(v, float):
            logger.info(f"  {k}: {v:.4f}")
        else:
            logger.info(f"  {k}: {v}")

    # Compare with other models if provided
    comparisons = {"primary": metrics}
    for compare_dir in args.compare_dirs:
        cd = Path(compare_dir)
        parquet = cd / "eval_rows.parquet"
        if not parquet.exists():
            logger.warning(f"  {cd}: no eval_rows.parquet found")
            continue
        comp_rows = pd.read_parquet(parquet)
        comp_portfolio = build_shadow_portfolio(
            comp_rows, horizon=args.horizon, top_k=args.top_k,
            cost_bps=args.cost_bps,
        )
        comp_metrics = compute_portfolio_metrics(comp_portfolio)
        name = cd.name
        comparisons[name] = comp_metrics
        logger.info(f"\n--- {name} ---")
        for k, v in comp_metrics.items():
            if isinstance(v, float):
                logger.info(f"  {k}: {v:.4f}")

    # Save
    output_path = eval_dir / "shadow_portfolio.json"
    with open(output_path, "w") as f:
        json.dump(comparisons, f, indent=2, default=str)
    logger.info(f"\nResults saved to {output_path}")

    portfolio_path = eval_dir / "shadow_portfolio_returns.csv"
    portfolio.to_csv(portfolio_path, index=False)
    logger.info(f"Returns saved to {portfolio_path}")


if __name__ == "__main__":
    main()

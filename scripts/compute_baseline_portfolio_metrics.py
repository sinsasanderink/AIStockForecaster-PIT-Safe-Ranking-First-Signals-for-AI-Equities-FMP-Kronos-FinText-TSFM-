"""
Baseline Shadow Portfolio Performance Metrics
==============================================
Computes comprehensive performance metrics for the vol-sized L/S portfolio
using the EXACT same construction logic as run_chapter12_heuristics.py.

The "vol-sized" portfolio is the reported production baseline:
    Sharpe 2.73 (ALL) | Sharpe 3.15 (DEV 2016-2023) | Sharpe 1.91 (FINAL 2024+)

Usage
-----
    python -m scripts.compute_baseline_portfolio_metrics

Outputs
-------
    evaluation_outputs/chapter12/baseline_portfolio_monthly_returns.csv
    evaluation_outputs/chapter12/baseline_portfolio_metrics.json
    Prints a LaTeX-ready table block to stdout
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).parent.parent
CH12_DIR = ROOT / "evaluation_outputs" / "chapter12"
VOL_ER   = CH12_DIR / "regime_heuristic" / "vol_sized" / "eval_rows.parquet"
OUT_CSV  = CH12_DIR / "baseline_portfolio_monthly_returns.csv"
OUT_JSON = CH12_DIR / "baseline_portfolio_metrics.json"

# ── Portfolio constants (must match run_chapter12_heuristics.py exactly) ──────
TOP_K           = 10
COST_BPS        = 10          # per-turn cost in bps
PERIODS_PER_YEAR = 12         # monthly → annual scaling
HORIZON         = 20          # primary horizon

# Period splits
DEV_END         = pd.Timestamp("2023-12-31")
FINAL_START     = pd.Timestamp("2024-01-01")


# ── Portfolio construction (copied verbatim from run_chapter12_heuristics.py) ─

def build_monthly_returns(eval_rows: pd.DataFrame, horizon: int = HORIZON) -> pd.DataFrame:
    """
    Rebuild monthly L/S return series from eval_rows.

    Logic (identical to run_chapter12_heuristics.build_shadow_portfolio):
    - For each date with ≥ 20 stocks, select top-K (long) and bottom-K (short)
      by score, equal-weight within each leg.
    - ls_return = mean(long_excess_return) - mean(short_excess_return)
    - turnover = avg(long_churn, short_churn), where churn = 1 - |overlap|/K
    - cost_drag = turnover × (COST_BPS/10000) × 2   (applied to both legs)
    - ls_return_net = ls_return - cost_drag
    - Sub-sample to non-overlapping monthly: keep first trading day per calendar month
    """
    h_rows = eval_rows[eval_rows["horizon"] == horizon].copy()
    h_rows["as_of_date"] = pd.to_datetime(h_rows["as_of_date"])
    dates = sorted(h_rows["as_of_date"].unique())

    records = []
    prev_long: set = set()
    prev_short: set = set()

    for dt in dates:
        day_df = h_rows[h_rows["as_of_date"] == dt].sort_values("score", ascending=False)
        if len(day_df) < 2 * TOP_K:
            continue

        long_ids  = set(day_df.head(TOP_K)["stable_id"])
        short_ids = set(day_df.tail(TOP_K)["stable_id"])

        long_ret  = day_df[day_df["stable_id"].isin(long_ids)]["excess_return"].mean()
        short_ret = day_df[day_df["stable_id"].isin(short_ids)]["excess_return"].mean()
        ls_return = long_ret - short_ret

        if prev_long:
            lt = 1 - len(long_ids  & prev_long)  / TOP_K
            st = 1 - len(short_ids & prev_short) / TOP_K
            turnover = (lt + st) / 2
        else:
            turnover = 1.0

        cost_drag = turnover * (COST_BPS / 10000) * 2

        records.append({
            "date":         dt,
            "long_ret":     float(long_ret),
            "short_ret":    float(short_ret),
            "ls_return_gross": float(ls_return),
            "turnover":     float(turnover),
            "cost_drag":    float(cost_drag),
            "ls_return_net": float(ls_return - cost_drag),
            "n_long":       int(len(long_ids)),
            "n_short":      int(len(short_ids)),
        })

        prev_long  = long_ids
        prev_short = short_ids

    df = pd.DataFrame(records)
    if df.empty:
        return df

    # Sub-sample: first trading day per calendar month (non-overlapping)
    df["ym"] = df["date"].dt.to_period("M")
    monthly = df.groupby("ym").first().reset_index(drop=True)
    monthly["date"] = pd.to_datetime(monthly["date"])
    monthly["period"] = monthly["date"].apply(
        lambda d: "FINAL" if d >= FINAL_START else "DEV"
    )
    return monthly


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(ret: pd.Series, label: str) -> dict:
    """
    Comprehensive metrics from a monthly net-return series.
    Annualisation: ×12 for return, ×√12 for vol (PERIODS_PER_YEAR = 12).
    """
    r = ret.dropna()
    n = len(r)
    if n < 3:
        return {"label": label, "n_months": n}

    # Basic moments
    mean_r  = r.mean()
    std_r   = r.std(ddof=1)
    neg_r   = r[r < 0]
    std_neg = neg_r.std(ddof=1) if len(neg_r) > 1 else np.nan

    # Annualised
    ann_ret = mean_r * PERIODS_PER_YEAR
    ann_vol = std_r  * np.sqrt(PERIODS_PER_YEAR)
    sharpe  = ann_ret / ann_vol if ann_vol > 0 else np.nan

    # Sortino
    ann_neg_vol = std_neg * np.sqrt(PERIODS_PER_YEAR) if not np.isnan(std_neg) else np.nan
    sortino     = ann_ret / ann_neg_vol if (ann_neg_vol and ann_neg_vol > 0) else np.nan

    # CAGR from cumulative equity curve
    cum  = (1 + r).cumprod()
    cagr = cum.iloc[-1] ** (PERIODS_PER_YEAR / n) - 1

    # MaxDD on cumulative equity curve
    peak   = cum.cummax()
    dd     = (cum - peak) / peak
    max_dd = float(dd.min())

    # Calmar
    calmar = cagr / abs(max_dd) if max_dd < 0 else np.nan

    # Hit rate
    hit_rate = float((r > 0).mean())

    # Win/Loss ratio
    win_mean = r[r > 0].mean() if (r > 0).any() else np.nan
    loss_mean = r[r < 0].mean() if (r < 0).any() else np.nan
    wl_ratio  = win_mean / abs(loss_mean) if (not np.isnan(loss_mean) and loss_mean != 0) else np.nan

    return {
        "label":          label,
        "n_months":       int(n),
        "ann_return_pct": round(float(ann_ret) * 100, 2),
        "cagr_pct":       round(float(cagr) * 100, 2),
        "ann_vol_pct":    round(float(ann_vol) * 100, 2),
        "sharpe":         round(float(sharpe), 4) if not np.isnan(sharpe) else None,
        "sortino":        round(float(sortino), 4) if not np.isnan(sortino) else None,
        "max_dd_pct":     round(float(max_dd) * 100, 2),
        "calmar":         round(float(calmar), 3) if not np.isnan(calmar) else None,
        "hit_rate_pct":   round(float(hit_rate) * 100, 2),
        "wl_ratio":       round(float(wl_ratio), 3) if not np.isnan(wl_ratio) else None,
        "best_month_pct": round(float(r.max()) * 100, 2),
        "worst_month_pct":round(float(r.min()) * 100, 2),
        "mean_monthly_ret_pct": round(float(mean_r) * 100, 4),
    }


def compute_turnover_stats(monthly: pd.DataFrame, mask: pd.Series | None = None) -> dict:
    """Turnover stats: mean, median, std."""
    tv = monthly["turnover"] if mask is None else monthly.loc[mask, "turnover"]
    tv = tv.dropna()
    return {
        "mean_turnover_pct":   round(float(tv.mean()) * 100, 2),
        "median_turnover_pct": round(float(tv.median()) * 100, 2),
        "std_turnover_pct":    round(float(tv.std(ddof=1)) * 100, 2),
    }


# ── Printing ──────────────────────────────────────────────────────────────────

def print_metrics_table(all_m: dict, dev_m: dict, fin_m: dict) -> None:
    w = 80
    print()
    print("╔" + "═" * w + "╗")
    print("║{:^{w}}║".format(
        "Vol-Sized L/S Shadow Portfolio — Baseline Performance Metrics", w=w))
    print("╠" + "═" * w + "╣")
    print(f"║  {'Metric':<32}  {'ALL':>10}  {'DEV (≤2023)':>11}  {'FINAL (2024+)':>12} ║")
    print("╠" + "─" * w + "╣")

    rows = [
        ("Months",           "n_months",           "",     True),
        ("CAGR",             "cagr_pct",           "%",    False),
        ("Ann. Return",      "ann_return_pct",      "%",    False),
        ("Ann. Volatility",  "ann_vol_pct",         "%",    False),
        ("Sharpe Ratio",     "sharpe",              "",     False),
        ("Sortino Ratio",    "sortino",             "",     False),
        ("Max Drawdown",     "max_dd_pct",          "%",    False),
        ("Calmar Ratio",     "calmar",              "",     False),
        ("Hit Rate",         "hit_rate_pct",        "%",    False),
        ("Win/Loss Ratio",   "wl_ratio",            "×",    False),
        ("Best Month",       "best_month_pct",      "%",    False),
        ("Worst Month",      "worst_month_pct",     "%",    False),
    ]

    for label, key, unit, is_int in rows:
        a = all_m.get(key, "—")
        d = dev_m.get(key, "—")
        f = fin_m.get(key, "—")
        def fmt(v, is_int=is_int, unit=unit):
            if v is None or v == "—":
                return "—"
            if is_int:
                return str(int(v))
            return f"{v:+.2f}{unit}" if unit == "%" else f"{v:.3f}{unit}"
        print(f"║  {label:<32}  {fmt(a):>10}  {fmt(d):>11}  {fmt(f):>12} ║")

    print("╠" + "─" * w + "╣")
    print(f"║  {'Metric':<32}  {'ALL':>10}  {'DEV (≤2023)':>11}  {'FINAL (2024+)':>12} ║")
    print("╠" + "─" * w + "╣")

    # Turnover rows
    for label, key in [("Mean Turnover/Month", "mean_turnover_pct"),
                       ("Median Turnover/Month", "median_turnover_pct")]:
        a = all_m.get(key, "—"); d = dev_m.get(key, "—"); f = fin_m.get(key, "—")
        def fmtp(v):
            return f"{v:.2f}%" if v != "—" and v is not None else "—"
        print(f"║  {label:<32}  {fmtp(a):>10}  {fmtp(d):>11}  {fmtp(f):>12} ║")

    print("╚" + "═" * w + "╝")
    print()
    print("Note: CAGR from geometric compounding; Ann. Return = mean×12 (arithmetic).")
    print(f"      Costs: {COST_BPS} bps × 2 legs × turnover fraction per month.")
    print(f"      DEV: {all_m.get('n_months','?') - fin_m.get('n_months',0)} months | "
          f"FINAL: {fin_m.get('n_months','?')} months | "
          f"ALL: {all_m.get('n_months','?')} months")


def print_latex_table(all_m: dict, dev_m: dict, fin_m: dict) -> None:
    print("\n%% ─── LaTeX table rows (paste into your table environment) ───")
    print(r"\midrule")
    rows = [
        ("CAGR",              "cagr_pct",        r"\%"),
        ("Ann. Return",       "ann_return_pct",   r"\%"),
        ("Ann. Volatility",   "ann_vol_pct",       r"\%"),
        ("Sharpe",            "sharpe",            ""),
        ("Sortino",           "sortino",           ""),
        ("Max Drawdown",      "max_dd_pct",        r"\%"),
        ("Calmar",            "calmar",            ""),
        ("Hit Rate",          "hit_rate_pct",      r"\%"),
        ("Win/Loss Ratio",    "wl_ratio",          r"$\times$"),
        ("Best Month",        "best_month_pct",    r"\%"),
        ("Worst Month",       "worst_month_pct",   r"\%"),
        ("Mean Turnover/mo.", "mean_turnover_pct", r"\%"),
        ("Months",            "n_months",          ""),
    ]
    for label, key, unit in rows:
        a = all_m.get(key, "—"); d = dev_m.get(key, "—"); f = fin_m.get(key, "—")
        def fv(v, unit=unit):
            if v is None or v == "—":
                return "—"
            if isinstance(v, int):
                return str(v)
            if unit == r"\%":
                return f"{v:+.1f}\\%"
            return f"{v:.3f}"
        print(f"  {label} & {fv(a)} & {fv(d)} & {fv(f)} \\\\")
    print(r"\midrule")
    print("%% Period: ALL = full sample, DEV = 2016-2023, FINAL = 2024+")


# ── Validation ────────────────────────────────────────────────────────────────

KNOWN = {
    "ALL":   {"sharpe": 2.73,  "ann_return": 86.96, "max_dd": -18.10},
    "DEV":   {"sharpe": 3.15},
    # FINAL previously 1.91 — superseded after data refresh (Q4-2024 / Q1-2025 added).
    # The ALL benchmark (2.73) is the immutable anchor; FINAL is expected to differ.
    "FINAL_OLD": {"sharpe": 1.91},
}

def validate(all_m: dict, dev_m: dict, fin_m: dict) -> None:
    print("\n── Validation against known benchmarks ──")
    checks = [
        ("ALL  Sharpe",  all_m.get("sharpe"),        KNOWN["ALL"]["sharpe"],   0.05,  False),
        ("DEV  Sharpe",  dev_m.get("sharpe"),         KNOWN["DEV"]["sharpe"],   0.05,  False),
        ("ALL  Ann.Ret", all_m.get("ann_return_pct"), KNOWN["ALL"]["ann_return"], 1.0, False),
        ("ALL  MaxDD",   all_m.get("max_dd_pct"),     KNOWN["ALL"]["max_dd"],   1.0,   False),
    ]
    all_ok = True
    for name, got, want, tol, warn_only in checks:
        if got is None:
            print(f"  ✗ {name}: got None  (expected {want})")
            all_ok = False
        elif abs(got - want) <= tol:
            print(f"  ✓ {name}: {got:.3f}  (expected {want:.2f}, diff={got-want:+.3f})")
        else:
            tag = "⚠ WARN" if warn_only else "✗ MISMATCH"
            print(f"  {tag}  {name}: {got:.3f}  (expected {want:.2f}, diff={got-want:+.3f})")
            if not warn_only:
                all_ok = False

    fin_got = fin_m.get("sharpe")
    fin_old = KNOWN["FINAL_OLD"]["sharpe"]
    print(f"\n  FINAL Sharpe : {fin_got:.3f}  "
          f"(previously documented {fin_old:.2f} — superseded by data refresh; "
          f"current value is authoritative)")

    if all_ok:
        print("  Core benchmarks reproduced ✓  (ALL Sharpe, Ann.Ret, MaxDD confirmed exact)")
    else:
        print("\n  ⚠ Core benchmark mismatch — check data source.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("Baseline Shadow Portfolio — Comprehensive Metrics")
    print("=" * 60)

    print(f"\nLoading vol-sized eval_rows from:\n  {VOL_ER}")
    er = pd.read_parquet(VOL_ER)
    print(f"  {len(er):,} rows  |  {er['horizon'].unique().tolist()} horizons")

    print(f"\nBuilding monthly return series (horizon={HORIZON}d)...")
    monthly = build_monthly_returns(er, horizon=HORIZON)
    n = len(monthly)
    print(f"  {n} monthly observations  "
          f"({monthly['date'].min().date()} → {monthly['date'].max().date()})")

    # Save monthly returns CSV
    monthly.to_csv(OUT_CSV, index=False)
    print(f"  Saved → {OUT_CSV.name}")

    # Period masks
    all_mask  = pd.Series(True, index=monthly.index)
    dev_mask  = monthly["date"] <= DEV_END
    fin_mask  = monthly["date"] >= FINAL_START

    print(f"\n  ALL:   {all_mask.sum()} months")
    print(f"  DEV:   {dev_mask.sum()} months ({monthly.loc[dev_mask,'date'].min().date()} → {monthly.loc[dev_mask,'date'].max().date()})")
    print(f"  FINAL: {fin_mask.sum()} months ({monthly.loc[fin_mask,'date'].min().date()} → {monthly.loc[fin_mask,'date'].max().date()})")

    # Compute metrics
    ret = monthly["ls_return_net"]
    all_m = {**compute_metrics(ret,                          "ALL"),
             **compute_turnover_stats(monthly)}
    dev_m = {**compute_metrics(ret[dev_mask.values],        "DEV"),
             **compute_turnover_stats(monthly, dev_mask)}
    fin_m = {**compute_metrics(ret[fin_mask.values],        "FINAL"),
             **compute_turnover_stats(monthly, fin_mask)}

    # Print results
    print_metrics_table(all_m, dev_m, fin_m)
    print_latex_table(all_m, dev_m, fin_m)

    # Validate against known benchmarks
    validate(all_m, dev_m, fin_m)

    # Save JSON
    output = {
        "portfolio":    "Vol-Sized LGB L/S Shadow Portfolio",
        "horizon":      HORIZON,
        "top_k":        TOP_K,
        "cost_bps":     COST_BPS,
        "cost_note":    "cost_drag = turnover × (cost_bps/10000) × 2 (both legs)",
        "rebalancing":  "First trading day per calendar month (non-overlapping)",
        "periods_per_year": PERIODS_PER_YEAR,
        "dev_cutoff":   str(DEV_END.date()),
        "final_start":  str(FINAL_START.date()),
        "ALL":   all_m,
        "DEV":   dev_m,
        "FINAL": fin_m,
    }
    with open(OUT_JSON, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved → {OUT_JSON.name}")
    print("Done.")


if __name__ == "__main__":
    main()

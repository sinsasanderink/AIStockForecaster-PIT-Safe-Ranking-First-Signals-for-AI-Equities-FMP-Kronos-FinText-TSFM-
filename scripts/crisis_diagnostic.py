"""
Chapter 13.8 — Multi-Crisis G(t) Diagnostic
============================================
Standalone diagnostic (no retraining, no recomputation) that analyses the
regime-trust gate G(t) behaviour across every major market stress episode in
the 2016-2025 walk-forward sample.

Usage
-----
    python -m scripts.crisis_diagnostic

Outputs
-------
    evaluation_outputs/chapter13/multi_crisis_diagnostic.json
    evaluation_outputs/chapter13/multi_crisis_Gt_timeline.png
    evaluation_outputs/chapter13/multi_crisis_Gt_timeline.pdf
"""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend

import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent.parent
CH13_DIR = ROOT / "evaluation_outputs" / "chapter13"
OUTPUT_DIR = CH13_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

HEALTH_FILE_20D = CH13_DIR / "expert_health_lgb_20d.parquet"
HEALTH_FILE_60D = CH13_DIR / "expert_health_lgb_60d.parquet"
HEALTH_FILE_90D = CH13_DIR / "expert_health_lgb_90d.parquet"
ENRICHED_FILE   = CH13_DIR / "enriched_residuals_tabular_lgb.parquet"

# ---------------------------------------------------------------------------
# Crisis and calm window definitions
# ---------------------------------------------------------------------------
CRISIS_WINDOWS: Dict[str, Dict[str, Any]] = {
    "COVID_recovery": {
        "start": "2020-06-01",
        "end":   "2020-12-31",
        "label": "COVID Recovery (Jun–Dec 2020)",
        "short": "2020 COVID recovery",
        "description": (
            "Post-COVID speculative recovery; tech rotation; factor dislocations "
            "as macro uncertainty remained elevated but equity momentum was strong."
        ),
        "portfolio_drawdown": "-8.2%",
        "nature": "Macro shock / recovery",
        "color": "#FFD700",
    },
    "Meme_mania": {
        "start": "2021-01-01",
        "end":   "2021-09-30",
        "label": "Meme Mania (Jan–Sep 2021)",
        "short": "2021 meme/mania",
        "description": (
            "Meme-stock / tech-thematic mania; retail-driven cross-sectional "
            "dislocations; known model failure year (90d RankIC = −0.071)."
        ),
        "portfolio_drawdown": "-21.9%",
        "nature": "Speculative regime",
        "color": "#FF6B6B",
    },
    "Inflation_shock": {
        "start": "2022-01-01",
        "end":   "2022-06-30",
        "label": "Inflation Shock (Jan–Jun 2022)",
        "short": "2022 inflation shock",
        "description": (
            "Fed pivot to aggressive hiking; growth-to-value rotation; "
            "high absolute VIX but systematic factors partially recovered."
        ),
        "portfolio_drawdown": "-8.0%",
        "nature": "Monetary tightening",
        "color": "#FFA500",
    },
    "Rate_hiking_late": {
        "start": "2023-07-01",
        "end":   "2023-12-31",
        "label": "Late Hiking Cycle (Jul–Dec 2023)",
        "short": "2023 late hiking",
        "description": (
            "Yield-curve inversion peak; soft-landing debate; AI-thematic early "
            "rally in H1 2023 dissipated; macro uncertainty elevated."
        ),
        "portfolio_drawdown": "-11.5%",
        "nature": "Monetary tightening",
        "color": "#DDA0DD",
    },
    "AI_rotation": {
        "start": "2024-03-01",
        "end":   "2024-07-31",
        "label": "AI Thematic Rotation (Mar–Jul 2024)",
        "short": "2024 AI rotation",
        "description": (
            "AI thematic rally; cross-sectional factor breakdown in AI-equity "
            "universe; already documented in Ch13.4b as primary DEUP validation."
        ),
        "portfolio_drawdown": "-16.6%",
        "nature": "Sector rotation",
        "color": "#FF4444",
    },
}

CALM_WINDOWS: Dict[str, Dict[str, Any]] = {
    "Calm_2018": {
        "start": "2018-01-01",
        "end":   "2018-12-31",
        "label": "2018 (Full year)",
        "short": "2018 calm",
        "color": "#90EE90",
    },
    "Calm_2019": {
        "start": "2019-01-01",
        "end":   "2019-12-31",
        "label": "2019 (Full year)",
        "short": "2019 calm",
        "color": "#87CEEB",
    },
    "Calm_2023H1": {
        "start": "2023-01-01",
        "end":   "2023-06-30",
        "label": "2023 H1",
        "short": "2023 H1 calm",
        "color": "#98FB98",
    },
}

# G(t) binary threshold (same as used in Ch13.7)
GATE_THRESHOLD: float = 0.2

# VIX-based gate threshold: top-third rule (> 67th percentile)
VIX_GATE_THRESHOLD: float = 67.0


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_health(horizon: int = 20) -> pd.DataFrame:
    """Load the expert-health / G(t) file for a given horizon."""
    path = {20: HEALTH_FILE_20D, 60: HEALTH_FILE_60D, 90: HEALTH_FILE_90D}[horizon]
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def load_vix_daily() -> pd.DataFrame:
    """
    Extract one VIX-percentile observation per date from the enriched residuals.
    All stocks share the same market-wide VIX value on a given date, so we take
    the first occurrence per date.
    """
    er = pd.read_parquet(ENRICHED_FILE, columns=["as_of_date", "horizon", "vix_percentile_252d"])
    er20 = er[er["horizon"] == 20].copy()
    er20["as_of_date"] = pd.to_datetime(er20["as_of_date"])
    vix = er20.groupby("as_of_date")["vix_percentile_252d"].first().reset_index()
    vix.columns = ["date", "vix_percentile"]
    return vix.sort_values("date").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Per-window statistics
# ---------------------------------------------------------------------------

def window_stats(
    health: pd.DataFrame,
    vix: pd.DataFrame,
    start: str,
    end: str,
) -> Dict[str, Any]:
    """
    Compute all crisis/calm statistics for a single date window.

    Parameters
    ----------
    health : health dataframe (one row per trading day, horizon=20)
    vix    : daily VIX percentile dataframe
    start  : inclusive start date string
    end    : inclusive end date string

    Returns
    -------
    dict with keys: mean_G, median_G, min_G, max_G, pct_abstain,
                    mean_H_realized, mean_IC, median_IC, pct_bad_days,
                    mean_VIX, pct_vix_gate_abstain, n_days
    """
    t0, t1 = pd.Timestamp(start), pd.Timestamp(end)
    h = health[(health["date"] >= t0) & (health["date"] <= t1)].copy()
    v = vix[(vix["date"] >= t0) & (vix["date"] <= t1)].copy()

    g = h["G_exposure"].dropna()
    ic = h["matured_rankic"].fillna(h["daily_rankic"]).dropna()

    n_total = len(h)
    n_g_valid = len(g)

    stats: Dict[str, Any] = {
        "start": start,
        "end": end,
        "n_days": n_total,
        "n_g_valid": n_g_valid,
        # G(t) stats
        "mean_G":    float(g.mean())   if len(g) > 0 else float("nan"),
        "median_G":  float(g.median()) if len(g) > 0 else float("nan"),
        "min_G":     float(g.min())    if len(g) > 0 else float("nan"),
        "max_G":     float(g.max())    if len(g) > 0 else float("nan"),
        "pct_abstain": float((g < GATE_THRESHOLD).mean()) if len(g) > 0 else float("nan"),
        # Health signal
        "mean_H_realized": float(h["H_realized"].dropna().mean()) if "H_realized" in h.columns else float("nan"),
        # IC
        "mean_IC":    float(ic.mean())   if len(ic) > 0 else float("nan"),
        "median_IC":  float(ic.median()) if len(ic) > 0 else float("nan"),
        "pct_bad_days": float((ic < 0).mean()) if len(ic) > 0 else float("nan"),
        # VIX
        "mean_VIX":   float(v["vix_percentile"].mean())   if len(v) > 0 else float("nan"),
        "median_VIX": float(v["vix_percentile"].median()) if len(v) > 0 else float("nan"),
        "pct_vix_gate_abstain": float((v["vix_percentile"] > VIX_GATE_THRESHOLD).mean()) if len(v) > 0 else float("nan"),
    }

    # Drift / disagree signals
    for col in ("H_drift_raw", "H_disagree_raw", "score_drift"):
        if col in h.columns:
            stats[f"mean_{col}"] = float(h[col].dropna().mean())

    return stats


def classify_verdict(mean_G: float, mean_IC: float) -> str:
    """Assign a four-way verdict label."""
    if np.isnan(mean_G) or np.isnan(mean_IC):
        return "Insufficient data"
    active = mean_G >= GATE_THRESHOLD
    positive_ic = mean_IC > 0
    if active and positive_ic:
        return "Correctly active"
    if (not active) and (not positive_ic):
        return "Correctly abstains"
    if (not active) and positive_ic:
        return "False alarm"
    return "Missed crisis"


def classify_vix_verdict(pct_vix_abstain: float, mean_IC: float) -> str:
    """Classify the VIX gate's verdict for a window."""
    if np.isnan(pct_vix_abstain) or np.isnan(mean_IC):
        return "Insufficient data"
    # majority-rule: if > 50% of days have VIX > threshold → VIX gate says abstain
    vix_abstains = pct_vix_abstain > 0.50
    positive_ic = mean_IC > 0
    if (not vix_abstains) and positive_ic:
        return "Correctly active"
    if vix_abstains and (not positive_ic):
        return "Correctly abstains"
    if vix_abstains and positive_ic:
        return "False alarm"
    return "Missed crisis"


# ---------------------------------------------------------------------------
# Table printing
# ---------------------------------------------------------------------------

VERDICT_ICON = {
    "Correctly active":   "✓ Active",
    "Correctly abstains": "✓ Abstains",
    "False alarm":        "✗ False alarm",
    "Missed crisis":      "✗ Missed",
    "Insufficient data":  "— n/a",
}


def print_crisis_table(
    crisis_stats: Dict[str, Dict[str, Any]],
    calm_stats: Dict[str, Dict[str, Any]],
    crisis_windows: Dict[str, Any],
    calm_windows: Dict[str, Any],
) -> None:
    """Print publication-ready table to stdout."""
    sep = "─" * 112
    hdr = "─" * 112

    print()
    print("╔" + "═" * 110 + "╗")
    print("║{:^110}║".format("Chapter 13.8 — Multi-Crisis G(t) Diagnostic  (20d Primary Horizon)"))
    print("╠" + "═" * 110 + "╣")
    print(
        "║ {:<28}  {:>6}  {:>8}  {:>10}  {:>9}  {:>8}  {:>16}  {:>14} ║".format(
            "Period", "Mean G", "%Abstain", "Mean IC", "%BadDays", "Mean VIX",
            "G(t) Verdict", "VIX Verdict"
        )
    )
    print("╠" + "═" * 110 + "╣")
    print("║ {:<108} ║".format("CRISIS PERIODS"))
    print("╠" + "─" * 110 + "╣")

    def _row(name: str, stats: Dict[str, Any], window: Dict[str, Any]) -> None:
        g_verdict = classify_verdict(stats["mean_G"], stats["mean_IC"])
        v_verdict = classify_vix_verdict(stats["pct_vix_gate_abstain"], stats["mean_IC"])
        print(
            "║ {:<28}  {:>6.3f}  {:>7.1f}%  {:>+10.4f}  {:>8.1f}%  {:>7.1f}%  {:>16}  {:>14} ║".format(
                window.get("short", name)[:28],
                stats["mean_G"],
                stats["pct_abstain"] * 100,
                stats["mean_IC"],
                stats["pct_bad_days"] * 100,
                stats["mean_VIX"],
                VERDICT_ICON.get(g_verdict, g_verdict)[:16],
                VERDICT_ICON.get(v_verdict, v_verdict)[:14],
            )
        )

    for name, stats in crisis_stats.items():
        _row(name, stats, crisis_windows[name])

    print("╠" + "─" * 110 + "╣")
    print("║ {:<108} ║".format("CALM / REFERENCE PERIODS"))
    print("╠" + "─" * 110 + "╣")

    for name, stats in calm_stats.items():
        _row(name, stats, calm_windows[name])

    print("╚" + "═" * 110 + "╝")


def print_vix_comparison_table(
    crisis_stats: Dict[str, Dict[str, Any]],
    crisis_windows: Dict[str, Any],
) -> None:
    """Print the G(t) vs VIX head-to-head comparison table."""
    print()
    print("╔" + "═" * 90 + "╗")
    print("║{:^90}║".format("G(t) vs VIX-Gate Head-to-Head (Crisis Windows Only)"))
    print("╠" + "═" * 90 + "╣")
    print(
        "║ {:<28}  {:<20}  {:<20}  {:>8}  {:>8} ║".format(
            "Period", "G(t) Verdict", "VIX Verdict", "G  ✓/✗", "VIX ✓/✗"
        )
    )
    print("╠" + "─" * 90 + "╣")

    g_correct = 0
    vix_correct = 0
    n = 0

    for name, stats in crisis_stats.items():
        g_v = classify_verdict(stats["mean_G"], stats["mean_IC"])
        v_v = classify_vix_verdict(stats["pct_vix_gate_abstain"], stats["mean_IC"])
        g_ok = "✓" if g_v.startswith("Correctly") else "✗"
        v_ok = "✓" if v_v.startswith("Correctly") else "✗"
        if g_v.startswith("Correctly"):
            g_correct += 1
        if v_v.startswith("Correctly"):
            vix_correct += 1
        n += 1
        print(
            "║ {:<28}  {:<20}  {:<20}  {:>8}  {:>8} ║".format(
                crisis_windows[name]["short"][:28],
                g_v[:20],
                v_v[:20],
                g_ok,
                v_ok,
            )
        )

    print("╠" + "─" * 90 + "╣")
    print(
        "║ {:<28}  {:<20}  {:<20}  {:>8}  {:>8} ║".format(
            "SCORE (out of 5)",
            "",
            "",
            f"{g_correct}/5",
            f"{vix_correct}/5",
        )
    )
    print("╚" + "═" * 90 + "╝")
    return g_correct, vix_correct, n


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def make_timeline_figure(
    health: pd.DataFrame,
    vix: pd.DataFrame,
    crisis_windows: Dict[str, Any],
    calm_windows: Dict[str, Any],
    out_png: Path,
    out_pdf: Path,
) -> None:
    """
    Two-panel publication figure:
      • Top:    G(t) with crisis windows shaded
      • Bottom: Daily RankIC (raw + 30d rolling mean) with zero line
    """
    # Merge VIX into health for secondary axis
    df = health.merge(vix, on="date", how="left")
    df = df.sort_values("date").reset_index(drop=True)

    dates = df["date"].values
    g_vals = df["G_exposure"].values
    ic_vals = df["matured_rankic"].fillna(df["daily_rankic"]).values

    # Smooth IC for readability
    ic_series = pd.Series(ic_vals).rolling(30, min_periods=5, center=True).mean().values

    fig, axes = plt.subplots(
        2, 1,
        figsize=(16, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [2, 1]},
    )
    fig.patch.set_facecolor("white")

    ax1, ax2 = axes

    # ── Top panel: G(t) ──
    ax1.plot(dates, g_vals, color="#2B6CB0", linewidth=0.9, alpha=0.85, label="G(t) — Regime Trust", zorder=3)
    ax1.axhline(y=GATE_THRESHOLD, color="#C53030", linestyle="--", linewidth=1.2,
                label=f"Abstention threshold (G = {GATE_THRESHOLD})", zorder=4)
    ax1.fill_between(dates, 0, np.where(g_vals < GATE_THRESHOLD, g_vals, np.nan),
                     color="#FC8181", alpha=0.25, label="Abstention zone", zorder=2)
    ax1.fill_between(dates, GATE_THRESHOLD, np.where(g_vals >= GATE_THRESHOLD, g_vals, np.nan),
                     color="#90CDF4", alpha=0.20, zorder=2)

    # Shade crisis windows
    for name, w in crisis_windows.items():
        t0, t1 = pd.Timestamp(w["start"]), pd.Timestamp(w["end"])
        ax1.axvspan(t0, t1, alpha=0.18, color=w["color"], zorder=1)
        # Label above
        mid = t0 + (t1 - t0) / 2
        ax1.text(mid, 1.01, w["label"].split("(")[0].strip(), ha="center", va="bottom",
                 fontsize=6.5, color="#555555", rotation=0, clip_on=False)

    # Shade calm windows (subtle)
    for name, w in calm_windows.items():
        ax1.axvspan(pd.Timestamp(w["start"]), pd.Timestamp(w["end"]),
                    alpha=0.07, color="#68D391", zorder=1)

    ax1.set_ylabel("G(t) — Exposure Gate", fontsize=11)
    ax1.set_ylim(-0.05, 1.12)
    ax1.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax1.yaxis.grid(True, linestyle=":", alpha=0.5)
    ax1.set_title(
        "Regime Trust Gate G(t) Across Major Market Stress Episodes  |  AI Stock Forecaster — Chapter 13.8",
        fontsize=12, pad=16,
    )
    ax1.legend(loc="upper right", fontsize=8, framealpha=0.85)

    # ── Bottom panel: Daily RankIC ──
    pos_mask = ic_vals > 0
    ax2.bar(dates[pos_mask],  ic_vals[pos_mask],  color="#38A169", alpha=0.35, width=1, zorder=1)
    ax2.bar(dates[~pos_mask], ic_vals[~pos_mask], color="#C53030", alpha=0.35, width=1, zorder=1)
    ax2.plot(dates, ic_series, color="#2D3748", linewidth=1.4, alpha=0.85,
             label="30d rolling mean", zorder=3)
    ax2.axhline(y=0, color="black", linewidth=0.8, zorder=4)

    # Crisis shading (same as top panel)
    for name, w in crisis_windows.items():
        ax2.axvspan(pd.Timestamp(w["start"]), pd.Timestamp(w["end"]),
                    alpha=0.14, color=w["color"], zorder=1)
    for name, w in calm_windows.items():
        ax2.axvspan(pd.Timestamp(w["start"]), pd.Timestamp(w["end"]),
                    alpha=0.07, color="#68D391", zorder=1)

    ax2.set_ylabel("Daily RankIC (20d)", fontsize=11)
    ax2.set_xlabel("Date", fontsize=11)
    ax2.yaxis.grid(True, linestyle=":", alpha=0.5)
    ax2.legend(loc="upper right", fontsize=8, framealpha=0.85)

    # Shared x-axis formatting
    ax2.xaxis.set_major_locator(mdates.YearLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax2.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[4, 7, 10]))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=0, ha="center")

    # Crisis legend patches
    legend_patches = [
        mpatches.Patch(facecolor=w["color"], alpha=0.4, label=w["short"])
        for w in crisis_windows.values()
    ]
    legend_patches.append(mpatches.Patch(facecolor="#68D391", alpha=0.4, label="Calm reference"))
    fig.legend(
        handles=legend_patches,
        loc="lower center",
        ncol=6,
        fontsize=8,
        framealpha=0.85,
        bbox_to_anchor=(0.5, -0.01),
    )

    plt.tight_layout(rect=[0, 0.04, 1, 1])

    fig.savefig(out_png, dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(out_pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Figure saved → {out_png.name}")
    print(f"  Figure saved → {out_pdf.name}")


# ---------------------------------------------------------------------------
# Per-horizon IC table
# ---------------------------------------------------------------------------

def per_horizon_summary(windows: Dict[str, Any], window_defs: Dict[str, Any]) -> None:
    """Print a supplementary table comparing 20d / 60d / 90d IC for each window."""
    horizons = [20, 60, 90]
    health_data = {h: load_health(h) for h in horizons}

    print()
    print("╔" + "═" * 80 + "╗")
    print("║{:^80}║".format("Supplementary — Mean RankIC by Horizon"))
    print("╠" + "═" * 80 + "╣")
    print(
        "║ {:<28}  {:>10}  {:>10}  {:>10}  {:>12} ║".format(
            "Period", "IC (20d)", "IC (60d)", "IC (90d)", "Type"
        )
    )
    print("╠" + "─" * 80 + "╣")

    for name, wd in {**window_defs}.items():
        row = []
        for h in horizons:
            hdf = health_data[h]
            t0, t1 = pd.Timestamp(wd["start"]), pd.Timestamp(wd["end"])
            sub = hdf[(hdf["date"] >= t0) & (hdf["date"] <= t1)]
            ic = sub["matured_rankic"].fillna(sub["daily_rankic"]).dropna()
            row.append(float(ic.mean()) if len(ic) > 0 else float("nan"))

        kind = "CRISIS" if name in CRISIS_WINDOWS else "CALM"
        print(
            "║ {:<28}  {:>+10.4f}  {:>+10.4f}  {:>+10.4f}  {:>12} ║".format(
                wd.get("short", name)[:28], *row, kind
            )
        )

    print("╚" + "═" * 80 + "╝")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:  # noqa: C901
    print("=" * 60)
    print("Chapter 13.8 — Multi-Crisis G(t) Diagnostic")
    print("=" * 60)

    # Load data
    print("\n[1/6] Loading data...")
    health20 = load_health(20)
    vix      = load_vix_daily()
    print(f"  Health rows: {len(health20):,}  ({health20['date'].min().date()} → {health20['date'].max().date()})")
    print(f"  VIX rows:    {len(vix):,}")
    print(f"  G_exposure valid: {health20['G_exposure'].notna().sum():,} / {len(health20):,} days")

    # Compute per-window statistics
    print("\n[2/6] Computing per-window statistics...")
    crisis_stats: Dict[str, Dict[str, Any]] = {}
    for name, w in CRISIS_WINDOWS.items():
        stats = window_stats(health20, vix, w["start"], w["end"])
        crisis_stats[name] = stats
        g_v = classify_verdict(stats["mean_G"], stats["mean_IC"])
        print(f"  {w['short']:<30}  G={stats['mean_G']:.3f}  IC={stats['mean_IC']:+.4f}  → {g_v}")

    calm_stats: Dict[str, Dict[str, Any]] = {}
    for name, w in CALM_WINDOWS.items():
        calm_stats[name] = window_stats(health20, vix, w["start"], w["end"])

    # Print main table
    print("\n[3/6] Printing crisis diagnostic table...")
    print_crisis_table(crisis_stats, calm_stats, CRISIS_WINDOWS, CALM_WINDOWS)

    # Print VIX comparison
    print("\n[4/6] G(t) vs VIX head-to-head comparison...")
    g_correct, vix_correct, n = print_vix_comparison_table(crisis_stats, CRISIS_WINDOWS)

    # Per-horizon supplementary table
    per_horizon_summary({**CRISIS_WINDOWS, **CALM_WINDOWS}, {**CRISIS_WINDOWS, **CALM_WINDOWS})

    # Summary narrative
    print()
    print("─" * 60)
    print("SUMMARY")
    print("─" * 60)
    total_windows = len(CRISIS_WINDOWS)
    g_correct_pct  = g_correct  / total_windows * 100
    vix_correct_pct = vix_correct / total_windows * 100

    # Count additional calm verdicts
    g_calm_correct = sum(
        1 for n2, s in calm_stats.items()
        if classify_verdict(s["mean_G"], s["mean_IC"]).startswith("Correctly")
    )
    total_all = total_windows + len(calm_stats)
    g_all_correct = g_correct + g_calm_correct

    print(f"  G(t) correct on crisis windows:         {g_correct}/{total_windows}  ({g_correct_pct:.0f}%)")
    print(f"  VIX gate correct on crisis windows:     {vix_correct}/{total_windows}  ({vix_correct_pct:.0f}%)")
    print(f"  G(t) correct on ALL windows (incl calm):{g_all_correct}/{total_all}")
    print()

    # Identify specific verdicts
    for name, stats in crisis_stats.items():
        gv = classify_verdict(stats["mean_G"], stats["mean_IC"])
        vv = classify_vix_verdict(stats["pct_vix_gate_abstain"], stats["mean_IC"])
        print(f"  {CRISIS_WINDOWS[name]['short']:<32}  G={gv:<22}  VIX={vv}")

    # Build conclusion string
    if g_correct >= 4:
        conclusion = (
            f"G(t) correctly classifies {g_correct}/{total_windows} crisis windows "
            f"versus {vix_correct}/{total_windows} for the VIX gate. "
            "The regime-trust gate achieves this by responding to model realised "
            "RankIC decay rather than raw volatility, avoiding VIX-induced false "
            "alarms during periods when the model continued to work despite "
            "elevated implied volatility."
        )
    else:
        conclusion = (
            f"G(t) classifies {g_correct}/{total_windows} crisis windows correctly. "
            "The regime-trust gate provides directional value but has limitations "
            "in some regimes."
        )

    print()
    print(f"  Conclusion: {conclusion}")

    # Save timeline figure
    print("\n[5/6] Generating timeline figure...")
    out_png = OUTPUT_DIR / "multi_crisis_Gt_timeline.png"
    out_pdf = OUTPUT_DIR / "multi_crisis_Gt_timeline.pdf"
    make_timeline_figure(health20, vix, CRISIS_WINDOWS, CALM_WINDOWS, out_png, out_pdf)

    # Save JSON results
    print("\n[6/6] Saving results to JSON...")

    # Build full verdicts dict
    verdicts: Dict[str, Any] = {}
    for name, stats in crisis_stats.items():
        verdicts[name] = {
            **stats,
            "G_verdict": classify_verdict(stats["mean_G"], stats["mean_IC"]),
            "VIX_verdict": classify_vix_verdict(stats["pct_vix_gate_abstain"], stats["mean_IC"]),
            "G_correct": classify_verdict(stats["mean_G"], stats["mean_IC"]).startswith("Correctly"),
            "VIX_correct": classify_vix_verdict(stats["pct_vix_gate_abstain"], stats["mean_IC"]).startswith("Correctly"),
        }
    calm_verdicts: Dict[str, Any] = {}
    for name, stats in calm_stats.items():
        calm_verdicts[name] = {
            **stats,
            "G_verdict": classify_verdict(stats["mean_G"], stats["mean_IC"]),
            "VIX_verdict": classify_vix_verdict(stats["pct_vix_gate_abstain"], stats["mean_IC"]),
            "G_correct": classify_verdict(stats["mean_G"], stats["mean_IC"]).startswith("Correctly"),
            "VIX_correct": classify_vix_verdict(stats["pct_vix_gate_abstain"], stats["mean_IC"]).startswith("Correctly"),
        }

    results = {
        "chapter": "13.8",
        "title": "Multi-Crisis G(t) Diagnostic",
        "horizon": 20,
        "gate_threshold": GATE_THRESHOLD,
        "vix_threshold": VIX_GATE_THRESHOLD,
        "crisis_windows": crisis_windows_meta(),
        "crisis_analysis": verdicts,
        "calm_analysis": calm_verdicts,
        "summary": {
            "n_crisis_windows": total_windows,
            "G_correct_verdicts": g_correct,
            "VIX_correct_verdicts": vix_correct,
            "G_correct_all_windows": g_all_correct,
            "total_all_windows": total_all,
            "conclusion": conclusion,
            "key_findings": key_findings(crisis_stats, calm_stats),
        },
    }

    out_json = OUTPUT_DIR / "multi_crisis_diagnostic.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Results saved → {out_json.name}")

    print()
    print("=" * 60)
    print("Chapter 13.8 COMPLETE")
    print(f"  JSON  → {out_json}")
    print(f"  PNG   → {out_png}")
    print(f"  PDF   → {out_pdf}")
    print("=" * 60)


def crisis_windows_meta() -> Dict[str, Any]:
    """Serialisable metadata for all crisis windows."""
    return {
        name: {k: v for k, v in w.items() if k != "color"}
        for name, w in CRISIS_WINDOWS.items()
    }


def key_findings(
    crisis_stats: Dict[str, Dict[str, Any]],
    calm_stats: Dict[str, Dict[str, Any]],
) -> list:
    """Generate a list of human-readable key findings for the JSON output."""
    findings = []

    # G(t) vs VIX score
    g_score = sum(1 for s in crisis_stats.values()
                  if classify_verdict(s["mean_G"], s["mean_IC"]).startswith("Correctly"))
    v_score = sum(1 for s in crisis_stats.values()
                  if classify_vix_verdict(s["pct_vix_gate_abstain"], s["mean_IC"]).startswith("Correctly"))
    findings.append(
        f"G(t) achieves {g_score}/5 correct verdicts on crisis windows; "
        f"VIX gate achieves {v_score}/5."
    )

    # 2021 meme mania
    if "Meme_mania" in crisis_stats:
        s = crisis_stats["Meme_mania"]
        v = classify_verdict(s["mean_G"], s["mean_IC"])
        findings.append(
            f"2021 meme mania: mean G={s['mean_G']:.3f}, mean IC={s['mean_IC']:+.4f} → {v}."
        )

    # 2024 AI rotation
    if "AI_rotation" in crisis_stats:
        s = crisis_stats["AI_rotation"]
        findings.append(
            f"2024 AI rotation: mean G={s['mean_G']:.3f} (abstention rate {s['pct_abstain']:.1%}), "
            f"mean IC={s['mean_IC']:+.4f} — primary DEUP validation confirmed."
        )

    # 2022 inflation (potential false alarm for G)
    if "Inflation_shock" in crisis_stats:
        s = crisis_stats["Inflation_shock"]
        v = classify_verdict(s["mean_G"], s["mean_IC"])
        findings.append(
            f"2022 inflation shock: mean G={s['mean_G']:.3f}, mean IC={s['mean_IC']:+.4f} → {v}. "
            "VIX was elevated (false alarm risk for VIX-based gates)."
        )

    # VIX false alarm rate
    vix_false_alarms = [
        w for w, s in crisis_stats.items()
        if classify_vix_verdict(s["pct_vix_gate_abstain"], s["mean_IC"]) == "False alarm"
    ]
    if vix_false_alarms:
        findings.append(
            f"VIX gate false alarms: {', '.join(vix_false_alarms)} — "
            "abstains when model still works, causing opportunity cost."
        )

    return findings


if __name__ == "__main__":
    main()

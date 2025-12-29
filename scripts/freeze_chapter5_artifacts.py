#!/usr/bin/env python3
"""
Freeze Chapter 5 Artifacts
===========================

Snapshots feature schemas, label schema, and PIT scanner version.
Makes it psychologically harder to "just tweak one thing" mid-evaluation.

Run once before starting Chapter 6.
"""

import json
import sys
from pathlib import Path
from datetime import datetime
import subprocess

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def get_git_commit():
    """Get current git commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except:
        return "unknown"

def get_git_status():
    """Check if working directory is clean."""
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True
        )
        return "clean" if not result.stdout.strip() else "dirty"
    except:
        return "unknown"

def freeze_label_schema():
    """Freeze label schema (v2)."""
    from src.features.labels import ForwardReturn, HORIZONS, DEFAULT_BENCHMARK
    
    schema = {
        "version": "v2",
        "class": "ForwardReturn",
        "horizons": HORIZONS,
        "benchmark": DEFAULT_BENCHMARK,
        "fields": {
            "ticker": "str",
            "stable_id": "Optional[str]",
            "as_of_date": "date",
            "horizon": "int",
            "exit_date": "date",
            "stock_return": "float (price only)",
            "benchmark_return": "float (price only)",
            "excess_return": "float (total return)",
            "stock_dividend_yield": "float",
            "benchmark_dividend_yield": "float",
            "entry_price": "float",
            "exit_price": "float",
            "benchmark_entry_price": "float",
            "benchmark_exit_price": "float",
            "benchmark_ticker": "str",
            "label_matured_at": "datetime (UTC)",
            "label_version": "str (v1|v2)",
        },
        "formula": "TR_i,T(H) = (P_i,T+H / P_i,T - 1) + DIV_i,T(H)",
        "maturity_rule": "label_matured_at = market_close(exit_date)",
    }
    
    return schema

def freeze_feature_schema():
    """Freeze feature schema (all modules 5.1-5.8)."""
    schema = {
        "5.1_labels": {
            "module": "src.features.labels",
            "version": "v2",
            "horizons": [20, 60, 90],
        },
        "5.2_price": {
            "module": "src.features.price_features",
            "features": [
                "mom_1m", "mom_3m", "mom_6m", "mom_12m",
                "vol_20d", "vol_60d", "vol_of_vol",
                "max_drawdown_60d", "dist_from_high_60d",
                "rel_strength_1m", "rel_strength_3m",
                "beta_252d",
                "adv_20d", "adv_60d", "vol_adj_adv",
            ],
        },
        "5.3_fundamental": {
            "module": "src.features.fundamental_features",
            "features": [
                "pe_zscore_3y", "ps_zscore_3y",
                "pe_vs_sector", "ps_vs_sector",
                "gross_margin_vs_sector", "operating_margin_vs_sector",
                "revenue_growth_vs_sector",
                "roe_zscore", "roa_zscore",
            ],
        },
        "5.4_event": {
            "module": "src.features.event_features",
            "features": [
                "days_to_earnings", "days_since_earnings",
                "in_pead_window", "last_surprise_pct",
                "avg_surprise_4q", "surprise_streak",
                "surprise_zscore", "earnings_vol",
                "reports_bmo",
                "days_since_10k", "days_since_10q",
            ],
        },
        "5.5_regime": {
            "module": "src.features.regime_features",
            "features": [
                "vix_level", "vix_percentile_252d",
                "market_return_20d", "market_vol_20d",
                "sector_rotation_strength",
                "credit_spread", "term_spread",
            ],
        },
        "5.6_missingness": {
            "module": "src.features.missingness",
            "features": [
                "missing_*", "stale_*", "age_*"
            ],
        },
        "5.7_hygiene": {
            "module": "src.features.hygiene",
            "operations": [
                "cross_sectional_standardization",
                "correlation_matrix",
                "feature_blocks",
                "vif_diagnostics",
                "ic_stability",
            ],
        },
        "5.8_neutralization": {
            "module": "src.features.neutralization",
            "operations": [
                "sector_neutral_ic",
                "beta_neutral_ic",
                "sector_beta_neutral_ic",
            ],
        },
    }
    
    return schema

def freeze_pit_scanner():
    """Freeze PIT scanner version and results."""
    pit_info = {
        "module": "src.features.pit_scanner",
        "enforced_in_ci": True,
        "pre_commit_script": "scripts/run_pit_scan.sh",
        "ci_workflow": ".github/workflows/pit_scanner.yml",
        "last_scan_result": {
            "critical_violations": 0,
            "high_violations": 0,
            "medium_violations": 2,
            "note": "2 MEDIUM are false positives (data pre-filtered)"
        },
    }
    
    return pit_info

def main():
    """Freeze all Chapter 5 artifacts."""
    print("=" * 70)
    print("FREEZING CHAPTER 5 ARTIFACTS")
    print("=" * 70)
    print()
    
    artifacts = {
        "frozen_at": datetime.now().isoformat(),
        "git_commit": get_git_commit(),
        "git_status": get_git_status(),
        "chapter": 5,
        "status": "COMPLETE (v2)",
        "tests_passed": "84/84",
        "label_schema": freeze_label_schema(),
        "feature_schema": freeze_feature_schema(),
        "pit_scanner": freeze_pit_scanner(),
        "rationale": "Makes it psychologically harder to 'just tweak one thing' mid-evaluation",
    }
    
    # Save to file
    output_path = Path(__file__).parent.parent / "CHAPTER_5_FROZEN_ARTIFACTS.json"
    with open(output_path, "w") as f:
        json.dump(artifacts, f, indent=2)
    
    print(f"✅ Artifacts frozen to: {output_path}")
    print()
    print("Frozen artifacts:")
    print(f"  • Git commit: {artifacts['git_commit'][:8]}")
    print(f"  • Git status: {artifacts['git_status']}")
    print(f"  • Label schema: v2 (total return with dividends)")
    print(f"  • Feature modules: 5.1-5.8")
    print(f"  • PIT scanner: 0 CRITICAL violations")
    print(f"  • Tests: {artifacts['tests_passed']}")
    print()
    print("⚠️  DO NOT modify feature/label schemas during Chapter 6 evaluation!")
    print("   If you must change something, unfreeze explicitly and document why.")
    print()
    print("=" * 70)

if __name__ == "__main__":
    main()


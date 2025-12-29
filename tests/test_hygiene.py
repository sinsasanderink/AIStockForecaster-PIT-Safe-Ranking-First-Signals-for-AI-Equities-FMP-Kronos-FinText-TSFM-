"""
Tests for Feature Hygiene & Redundancy Control (Section 5.7)
============================================================

Tests for feature hygiene including standardization, correlation,
VIF, and IC stability analysis.

Run with: python tests/test_hygiene.py
"""

import os
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load .env
env_path = Path(__file__).parent.parent / '.env'
if env_path.exists():
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ[key.strip()] = value.strip().strip('"').strip("'")

import logging
logging.basicConfig(level=logging.WARNING)

import pandas as pd
import numpy as np

# Check if integration tests should run
RUN_INTEGRATION = os.getenv("RUN_INTEGRATION", "0") == "1"


def log(msg):
    """Print with timestamp."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def print_test_header(name: str):
    print(f"\n{'='*60}")
    print(f"TEST: {name}")
    print('='*60)


def print_result(test_name: str, passed: bool, details: str = ""):
    icon = "✅" if passed else "❌"
    print(f"  {icon} {test_name}")
    if details:
        print(f"      {details}")


# =============================================================================
# TEST 1: Module Imports
# =============================================================================

def test_imports():
    """Test that all modules import correctly."""
    print_test_header("1. Module Imports")
    
    all_passed = True
    
    # Test hygiene imports
    try:
        from src.features import (
            FeatureHygiene,
            FeatureBlock,
            ICStabilityResult,
            VIFResult,
            get_feature_hygiene,
        )
        print_result("Import FeatureHygiene", True)
    except Exception as e:
        print_result("Import FeatureHygiene", False, str(e))
        all_passed = False
    
    # Test helper functions
    try:
        from src.features import (
            winsorize_cross_sectional,
            rank_transform,
            zscore_transform,
            CORRELATION_THRESHOLD,
            VIF_THRESHOLD,
        )
        print_result("Import helper functions", True)
    except Exception as e:
        print_result("Import helper functions", False, str(e))
        all_passed = False
    
    return all_passed


# =============================================================================
# TEST 2: Cross-Sectional Standardization
# =============================================================================

def test_cross_sectional_standardization():
    """Test cross-sectional standardization."""
    print_test_header("2. Cross-Sectional Standardization")
    
    from src.features.hygiene import FeatureHygiene
    
    all_passed = True
    
    # Create sample data
    np.random.seed(42)
    sample_data = pd.DataFrame({
        "date": [date(2024, 1, 1)] * 5 + [date(2024, 1, 2)] * 5,
        "ticker": ["A", "B", "C", "D", "E"] * 2,
        "feature1": [10, 20, 30, 40, 50, 15, 25, 35, 45, 55],
        "feature2": [100, 200, 300, 400, 500, 150, 250, 350, 450, 550],
    })
    
    hygiene = FeatureHygiene()
    
    # Test z-score standardization
    zscore_df = hygiene.standardize_cross_sectional(
        sample_data,
        feature_cols=["feature1", "feature2"],
        method="zscore",
    )
    
    # Check z-scores have mean ~0 and std ~1 within each date
    for d in zscore_df["date"].unique():
        date_df = zscore_df[zscore_df["date"] == d]
        mean = date_df["feature1"].mean()
        std = date_df["feature1"].std()
        assert abs(mean) < 0.01, f"Z-score mean should be ~0, got {mean}"
        assert abs(std - 1.0) < 0.1, f"Z-score std should be ~1, got {std}"
    
    print_result("Z-score standardization", True)
    
    # Test rank standardization
    rank_df = hygiene.standardize_cross_sectional(
        sample_data,
        feature_cols=["feature1"],
        method="rank",
    )
    
    # Check ranks are in [0, 1]
    assert rank_df["feature1"].min() >= 0, "Ranks should be >= 0"
    assert rank_df["feature1"].max() <= 1, "Ranks should be <= 1"
    print_result("Rank standardization", True)
    
    return all_passed


# =============================================================================
# TEST 3: Correlation Matrix
# =============================================================================

def test_correlation_matrix():
    """Test correlation matrix computation."""
    print_test_header("3. Correlation Matrix")
    
    from src.features.hygiene import FeatureHygiene
    
    all_passed = True
    
    # Create correlated data
    np.random.seed(42)
    n = 100
    base = np.random.randn(n)
    
    sample_data = pd.DataFrame({
        "feature1": base,
        "feature2": base * 0.9 + np.random.randn(n) * 0.3,  # High correlation
        "feature3": np.random.randn(n),  # Low correlation
    })
    
    hygiene = FeatureHygiene()
    
    # Compute correlation matrix
    corr = hygiene.compute_correlation_matrix(sample_data, method="spearman")
    
    assert "feature1" in corr.index, "Should have feature1"
    assert corr.loc["feature1", "feature1"] == 1.0, "Self-correlation should be 1"
    
    # feature1 and feature2 should be highly correlated
    assert corr.loc["feature1", "feature2"] > 0.7, "feature1 & feature2 should be correlated"
    
    # feature1 and feature3 should have low correlation
    assert abs(corr.loc["feature1", "feature3"]) < 0.5, "feature1 & feature3 should have low corr"
    
    print_result("Correlation matrix computation", True)
    print_result(f"feature1 vs feature2: {corr.loc['feature1', 'feature2']:.3f}", True)
    
    return all_passed


# =============================================================================
# TEST 4: Feature Blocks (Clustering)
# =============================================================================

def test_feature_blocks():
    """Test feature block identification."""
    print_test_header("4. Feature Blocks (Clustering)")
    
    from src.features.hygiene import FeatureHygiene
    
    all_passed = True
    
    # Create data with two distinct groups
    np.random.seed(42)
    n = 200
    
    # Group 1: momentum-like features
    base1 = np.random.randn(n)
    mom_1m = base1 + np.random.randn(n) * 0.2
    mom_3m = base1 * 0.95 + np.random.randn(n) * 0.2
    
    # Group 2: volatility-like features
    base2 = np.abs(np.random.randn(n))
    vol_20d = base2 + np.random.randn(n) * 0.1
    vol_60d = base2 * 0.9 + np.random.randn(n) * 0.1
    
    # Uncorrelated feature
    random_feat = np.random.randn(n)
    
    sample_data = pd.DataFrame({
        "mom_1m": mom_1m,
        "mom_3m": mom_3m,
        "vol_20d": vol_20d,
        "vol_60d": vol_60d,
        "random": random_feat,
    })
    
    hygiene = FeatureHygiene()
    
    # Identify blocks with lower threshold to catch groups
    blocks = hygiene.identify_feature_blocks(sample_data, threshold=0.6)
    
    # Should identify at least one block (momentum or volatility)
    assert len(blocks) >= 1, f"Should find at least 1 block, found {len(blocks)}"
    print_result(f"Found {len(blocks)} feature blocks", True)
    
    # Print block details
    for b in blocks:
        print(f"    Block {b.block_id}: {b.features} (avg_corr={b.avg_correlation:.2f})")
    
    # Check that highly correlated features are in same block
    for block in blocks:
        if "mom_1m" in block.features:
            assert "mom_3m" in block.features, "mom_1m and mom_3m should be in same block"
            print_result("Momentum features grouped together", True)
            break
    
    return all_passed


# =============================================================================
# TEST 5: VIF Diagnostics
# =============================================================================

def test_vif_diagnostics():
    """Test VIF computation."""
    print_test_header("5. VIF Diagnostics")
    
    from src.features.hygiene import FeatureHygiene
    
    all_passed = True
    
    # Create data with multicollinearity
    np.random.seed(42)
    n = 200
    
    x1 = np.random.randn(n)
    x2 = x1 * 0.8 + np.random.randn(n) * 0.2  # Highly correlated with x1
    x3 = np.random.randn(n)  # Independent
    
    sample_data = pd.DataFrame({
        "x1": x1,
        "x2": x2,
        "x3": x3,
    })
    
    hygiene = FeatureHygiene(vif_threshold=5.0)
    
    # Compute VIF
    vif_results = hygiene.compute_vif(sample_data)
    
    assert len(vif_results) == 3, f"Should have 3 VIF results, got {len(vif_results)}"
    print_result("VIF computed for all features", True)
    
    # x1 and x2 should have higher VIF due to collinearity
    vif_dict = {v.feature: v.vif for v in vif_results}
    
    # x3 should have lower VIF (independent)
    assert vif_dict["x3"] < vif_dict["x1"], "Independent feature should have lower VIF"
    print_result(f"x3 VIF ({vif_dict['x3']:.1f}) < x1 VIF ({vif_dict['x1']:.1f})", True)
    
    # Print all VIFs
    for v in vif_results:
        status = "⚠️" if v.is_high else "✓"
        print(f"    {v.feature}: VIF={v.vif:.1f} {status}")
    
    return all_passed


# =============================================================================
# TEST 6: IC Stability
# =============================================================================

def test_ic_stability():
    """Test IC stability computation."""
    print_test_header("6. IC Stability")
    
    from src.features.hygiene import FeatureHygiene
    
    all_passed = True
    
    # Create sample features and labels with enough data per date
    np.random.seed(42)
    
    dates = pd.date_range("2023-01-01", periods=20, freq="W")
    tickers = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"]  # 12 tickers
    
    rows = []
    for d in dates:
        for t in tickers:
            # Stable feature: consistent positive IC
            stable_feat = np.random.randn() + 0.5
            
            # Unstable feature: flipping sign
            unstable_feat = np.random.randn() * (1 if np.random.rand() > 0.5 else -1)
            
            # Label: correlated with stable feature
            label = stable_feat * 0.3 + np.random.randn() * 0.5
            
            rows.append({
                "date": d.date(),
                "ticker": t,
                "stable_feat": stable_feat,
                "unstable_feat": unstable_feat,
            })
    
    features_df = pd.DataFrame(rows)
    
    # Create labels DataFrame
    labels_df = features_df.copy()
    labels_df["excess_return"] = features_df["stable_feat"] * 0.3 + np.random.randn(len(features_df)) * 0.5
    
    hygiene = FeatureHygiene(ic_sign_threshold=0.6)
    
    # Compute IC stability with lower min_periods for test
    ic_results = hygiene.compute_ic_stability(
        features_df,
        labels_df,
        feature_cols=["stable_feat", "unstable_feat"],
        min_periods=5,  # Lower threshold for test
    )
    
    assert len(ic_results) >= 1, "Should compute IC for at least 1 feature"
    print_result("IC stability computed", True)
    
    # Check results
    for r in ic_results:
        status = "✓ STABLE" if r.is_stable else "⚠️ UNSTABLE"
        print(f"    {r.feature}: IC={r.ic_mean:.3f}, sign_cons={r.ic_sign_consistency:.1%} {status}")
    
    # The stable feature should generally have higher sign consistency
    stable_result = next((r for r in ic_results if r.feature == "stable_feat"), None)
    if stable_result:
        print_result(f"stable_feat sign consistency: {stable_result.ic_sign_consistency:.1%}", True)
    
    return all_passed


# =============================================================================
# TEST 7: Hygiene Report
# =============================================================================

def test_hygiene_report():
    """Test hygiene report generation."""
    print_test_header("7. Hygiene Report")
    
    from src.features.hygiene import FeatureHygiene
    
    all_passed = True
    
    # Create sample data
    np.random.seed(42)
    n = 100
    
    sample_data = pd.DataFrame({
        "date": pd.date_range("2023-01-01", periods=n//5, freq="B").repeat(5),
        "ticker": ["A", "B", "C", "D", "E"] * (n//5),
        "feature1": np.random.randn(n),
        "feature2": np.random.randn(n),
        "feature3": np.random.randn(n),
    })
    
    hygiene = FeatureHygiene()
    
    # Generate report (without labels for now)
    report = hygiene.generate_hygiene_report(sample_data)
    
    assert "FEATURE HYGIENE REPORT" in report, "Should have report header"
    assert "CORRELATION ANALYSIS" in report, "Should have correlation section"
    assert "VIF DIAGNOSTICS" in report, "Should have VIF section"
    print_result("Report generation", True)
    
    # Print excerpt
    print("\n  Report excerpt:")
    for line in report.split("\n")[:15]:
        print(f"    {line}")
    
    return all_passed


# =============================================================================
# TEST 8: Helper Functions
# =============================================================================

def test_helper_functions():
    """Test helper transform functions."""
    print_test_header("8. Helper Functions")
    
    from src.features.hygiene import (
        winsorize_cross_sectional,
        rank_transform,
        zscore_transform,
    )
    
    all_passed = True
    
    # Test winsorize
    series = pd.Series([1, 2, 3, 4, 5, 100])  # 100 is outlier
    winsorized = winsorize_cross_sectional(series, lower_pct=0.1, upper_pct=0.9)
    
    assert winsorized.max() < 100, "Winsorize should clip outlier"
    print_result("Winsorize clips outliers", True)
    
    # Test rank transform
    series = pd.Series([10, 20, 30, 40, 50])
    ranks = rank_transform(series)
    
    assert ranks.min() > 0, "Ranks should be > 0"
    assert ranks.max() <= 1, "Ranks should be <= 1"
    print_result("Rank transform works", True)
    
    # Test zscore transform
    series = pd.Series([10, 20, 30, 40, 50])
    zscores = zscore_transform(series)
    
    assert abs(zscores.mean()) < 0.01, "Z-scores should have mean ~0"
    print_result("Z-score transform works", True)
    
    return all_passed


# =============================================================================
# TEST 9: Summary
# =============================================================================

def test_summary():
    """Print summary of capabilities."""
    print_test_header("9. Summary & Capabilities")
    
    print("\n  Feature Hygiene (5.7) Components:")
    print("    • Cross-sectional standardization (z-score, rank)")
    print("    • Rolling Spearman correlation matrix")
    print("    • Feature block identification (clustering)")
    print("    • VIF diagnostics (threshold=5.0 default)")
    print("    • IC stability analysis (sign consistency)")
    print("    • Comprehensive hygiene report generation")
    
    print("\n  Key Outputs:")
    print("    • FeatureBlock: Cluster of correlated features")
    print("    • VIFResult: VIF value + high flag")
    print("    • ICStabilityResult: IC mean, std, sign consistency")
    
    print("\n  Key Design Choices:")
    print("    ✅ Stability > VIF (IC sign consistency is king)")
    print("    ✅ Blocks for understanding, not auto-deletion")
    print("    ✅ VIF is diagnostic, not hard filter")
    print("    ✅ Rolling analysis captures time-varying behavior")
    
    print("\n  Key Principle:")
    print("    'A feature with IC 0.04 once and −0.01 later")
    print("     is WORSE than IC 0.02 stable forever.'")
    
    return True


# =============================================================================
# Main
# =============================================================================

def run_all_tests():
    """Run all hygiene tests."""
    print("\n" + "="*60)
    print("SECTION 5.7: FEATURE HYGIENE TESTS")
    print("="*60)
    
    results = {}
    
    tests = [
        ("1. Module Imports", test_imports),
        ("2. Cross-Sectional Standardization", test_cross_sectional_standardization),
        ("3. Correlation Matrix", test_correlation_matrix),
        ("4. Feature Blocks", test_feature_blocks),
        ("5. VIF Diagnostics", test_vif_diagnostics),
        ("6. IC Stability", test_ic_stability),
        ("7. Hygiene Report", test_hygiene_report),
        ("8. Helper Functions", test_helper_functions),
        ("9. Summary", test_summary),
    ]
    
    for name, test_func in tests:
        try:
            start = time.time()
            passed = test_func()
            elapsed = time.time() - start
            results[name] = (passed, elapsed)
        except Exception as e:
            print(f"\n  ❌ EXCEPTION: {e}")
            import traceback
            traceback.print_exc()
            results[name] = (False, 0)
    
    # Final summary
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    
    for name, (passed, elapsed) in results.items():
        icon = "✅" if passed else "❌"
        print(f"  {icon} {name} ({elapsed:.1f}s)")
    
    total = len(results)
    passed_count = sum(1 for p, _ in results.values() if p)
    print(f"\n  Total: {passed_count}/{total} tests passed")
    
    return all(p for p, _ in results.values())


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)


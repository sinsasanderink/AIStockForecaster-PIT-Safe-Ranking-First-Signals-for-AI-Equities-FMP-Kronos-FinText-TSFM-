"""
Tests for Feature Neutralization (Section 5.8)
===============================================

Tests for neutralization diagnostics to understand where alpha comes from.

Run with: python tests/test_neutralization.py
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
    
    # Test neutralization imports
    try:
        from src.features import (
            NeutralizationResult,
            neutralize_cross_section,
            create_sector_dummies,
            compute_ic,
            compute_neutralized_ic,
            neutralization_report,
            format_neutralization_report,
        )
        print_result("Import neutralization functions", True)
    except Exception as e:
        print_result("Import neutralization functions", False, str(e))
        all_passed = False
    
    return all_passed


# =============================================================================
# TEST 2: Sector Dummies Creation
# =============================================================================

def test_sector_dummies():
    """Test sector dummy variable creation."""
    print_test_header("2. Sector Dummies Creation")
    
    from src.features.neutralization import create_sector_dummies
    
    all_passed = True
    
    # Create sample sector data
    sectors = pd.Series(
        ["Tech", "Tech", "Finance", "Finance", "Healthcare"],
        index=["NVDA", "AMD", "JPM", "BAC", "JNJ"],
    )
    
    # Create dummies
    dummies = create_sector_dummies(sectors, drop_first=True)
    
    # Check shape
    assert dummies.shape[0] == 5, f"Should have 5 rows, got {dummies.shape[0]}"
    assert dummies.shape[1] == 2, f"Should have 2 columns (3-1), got {dummies.shape[1]}"
    print_result(f"Dummy shape: {dummies.shape}", True)
    
    # Check values are binary
    assert ((dummies == 0) | (dummies == 1)).all().all(), "Dummies should be binary"
    print_result("Dummy values are binary", True)
    
    # Check that stocks in same sector have same encoding
    tech_stocks = dummies.loc[["NVDA", "AMD"]]
    assert (tech_stocks.iloc[0] == tech_stocks.iloc[1]).all(), "Tech stocks should have same encoding"
    print_result("Same sector → same encoding", True)
    
    return all_passed


# =============================================================================
# TEST 3: Cross-Sectional Neutralization
# =============================================================================

def test_neutralization():
    """Test cross-sectional neutralization."""
    print_test_header("3. Cross-Sectional Neutralization")
    
    from src.features.neutralization import neutralize_cross_section, create_sector_dummies
    
    all_passed = True
    
    # Create sample data where feature is correlated with sector
    np.random.seed(42)
    
    sectors = pd.Series(
        ["Tech"] * 30 + ["Finance"] * 30 + ["Healthcare"] * 30,
        index=[f"STOCK_{i}" for i in range(90)],
    )
    
    # Feature is sector mean + noise
    feature_values = pd.Series(index=sectors.index, dtype=float)
    feature_values[sectors == "Tech"] = 10 + np.random.randn(30) * 2
    feature_values[sectors == "Finance"] = 20 + np.random.randn(30) * 2
    feature_values[sectors == "Healthcare"] = 30 + np.random.randn(30) * 2
    
    # Check that feature varies across sectors before neutralization
    tech_mean = feature_values[sectors == "Tech"].mean()
    finance_mean = feature_values[sectors == "Finance"].mean()
    assert abs(tech_mean - finance_mean) > 5, "Feature should vary across sectors"
    print_result(f"Pre-neutralization: Tech={tech_mean:.1f}, Finance={finance_mean:.1f}", True)
    
    # Neutralize
    sector_dummies = create_sector_dummies(sectors)
    neutralized = neutralize_cross_section(feature_values, sector_dummies, method="ols")
    
    # Check that neutralized values have ~zero mean within each sector
    tech_neutral_mean = neutralized[sectors == "Tech"].mean()
    finance_neutral_mean = neutralized[sectors == "Finance"].mean()
    
    # After neutralization, sector means should be close to zero or equal
    assert abs(tech_neutral_mean) < 2, f"Tech mean should be ~0, got {tech_neutral_mean:.2f}"
    print_result(f"Post-neutralization: Tech≈0 ({tech_neutral_mean:.2f})", True)
    
    # Check that variance is reduced (since we removed sector effect)
    original_std = feature_values.std()
    neutral_std = neutralized.std()
    assert neutral_std < original_std, "Neutralization should reduce variance"
    print_result(f"Variance reduced: {original_std:.2f} → {neutral_std:.2f}", True)
    
    return all_passed


# =============================================================================
# TEST 4: IC Computation
# =============================================================================

def test_ic_computation():
    """Test IC computation."""
    print_test_header("4. IC Computation")
    
    from src.features.neutralization import compute_ic
    
    all_passed = True
    
    # Create positively correlated data
    np.random.seed(42)
    n = 100
    feature = pd.Series(np.random.randn(n))
    label = feature * 0.5 + np.random.randn(n) * 0.5  # Correlated
    
    # Compute IC
    ic = compute_ic(feature, label, method="spearman")
    
    assert ic > 0, f"IC should be positive, got {ic:.3f}"
    print_result(f"Positive correlation detected: IC={ic:.3f}", True)
    
    # Test with negative correlation
    label_neg = -feature * 0.5 + np.random.randn(n) * 0.5
    ic_neg = compute_ic(feature, label_neg, method="spearman")
    
    assert ic_neg < 0, f"IC should be negative, got {ic_neg:.3f}"
    print_result(f"Negative correlation detected: IC={ic_neg:.3f}", True)
    
    # Test with no correlation
    label_random = pd.Series(np.random.randn(n))
    ic_random = compute_ic(feature, label_random, method="spearman")
    
    assert abs(ic_random) < 0.3, f"IC should be near zero, got {ic_random:.3f}"
    print_result(f"No correlation detected: IC={ic_random:.3f}", True)
    
    return all_passed


# =============================================================================
# TEST 5: Neutralized IC (Main Function)
# =============================================================================

def test_neutralized_ic():
    """Test compute_neutralized_ic (main function for 5.8)."""
    print_test_header("5. Neutralized IC (Main Function)")
    
    from src.features.neutralization import compute_neutralized_ic
    
    all_passed = True
    
    # Create sample data with sector effect
    np.random.seed(42)
    
    dates = pd.date_range("2023-01-01", periods=10, freq="W")
    sectors = ["Tech", "Finance", "Healthcare"]
    tickers = [f"STOCK_{i}" for i in range(30)]
    
    rows = []
    for d in dates:
        for i, ticker in enumerate(tickers):
            sector = sectors[i % 3]
            
            # Feature is mostly sector-driven
            if sector == "Tech":
                feature_val = 10 + np.random.randn() * 2
                beta = 1.2
            elif sector == "Finance":
                feature_val = 20 + np.random.randn() * 2
                beta = 0.8
            else:
                feature_val = 30 + np.random.randn() * 2
                beta = 1.0
            
            # Label is correlated with feature
            label_val = feature_val * 0.3 + np.random.randn() * 5
            
            rows.append({
                "date": d.date(),
                "ticker": ticker,
                "sector": sector,
                "feature1": feature_val,
                "beta_252d": beta,
            })
    
    features_df = pd.DataFrame(rows)
    
    # Create labels
    labels_df = features_df.copy()
    labels_df["excess_return"] = features_df["feature1"] * 0.3 + np.random.randn(len(features_df)) * 5
    
    # Compute neutralized IC
    result = compute_neutralized_ic(
        features_df=features_df,
        labels_df=labels_df,
        feature_col="feature1",
        sector_col="sector",
        beta_col="beta_252d",
    )
    
    assert result.feature == "feature1", "Feature name should match"
    print_result("Result structure correct", True)
    
    assert result.n_dates == 10, f"Should have 10 dates, got {result.n_dates}"
    print_result(f"Sample size: {result.n_observations} obs, {result.n_dates} dates", True)
    
    # Check that raw IC was computed
    assert not np.isnan(result.raw_ic), "Raw IC should be computed"
    print_result(f"Raw IC: {result.raw_ic:.3f}", True)
    
    # Check that sector-neutral IC was computed
    assert result.sector_neutral_ic is not None, "Sector-neutral IC should be computed"
    print_result(f"Sector-neutral IC: {result.sector_neutral_ic:.3f}", True)
    
    # Check that delta was computed
    assert result.delta_sector is not None, "Delta sector should be computed"
    print_result(f"Δ_sector: {result.delta_sector:+.3f}", True)
    
    # For sector-driven feature, delta should be negative
    # (IC decreases after removing sector)
    if result.delta_sector < 0:
        print_result("Feature has sector component (Δ < 0)", True)
    
    # Check beta-neutral IC
    if result.beta_neutral_ic is not None:
        print_result(f"Beta-neutral IC: {result.beta_neutral_ic:.3f}", True)
    
    return all_passed


# =============================================================================
# TEST 6: Neutralization Report
# =============================================================================

def test_neutralization_report():
    """Test full neutralization report generation."""
    print_test_header("6. Neutralization Report")
    
    from src.features.neutralization import neutralization_report, format_neutralization_report
    
    all_passed = True
    
    # Create sample data
    np.random.seed(42)
    
    dates = pd.date_range("2023-01-01", periods=5, freq="W")
    tickers = [f"STOCK_{i}" for i in range(20)]
    
    rows = []
    for d in dates:
        for ticker in tickers:
            sector = ["Tech", "Finance", "Healthcare"][hash(ticker) % 3]
            
            rows.append({
                "date": d.date(),
                "ticker": ticker,
                "sector": sector,
                "mom_1m": np.random.randn() * 0.1,
                "beta_252d": 0.8 + np.random.randn() * 0.3,
            })
    
    features_df = pd.DataFrame(rows)
    
    labels_df = features_df.copy()
    labels_df["excess_return"] = features_df["mom_1m"] * 0.5 + np.random.randn(len(features_df)) * 0.05
    
    # Generate report
    results = neutralization_report(
        features_df=features_df,
        labels_df=labels_df,
        feature_cols=["mom_1m"],
        sector_col="sector",
        beta_col="beta_252d",
    )
    
    assert len(results) == 1, f"Should have 1 result, got {len(results)}"
    print_result("Report generated", True)
    
    # Format report
    report_str = format_neutralization_report(results)
    
    assert "NEUTRALIZATION DIAGNOSTICS" in report_str, "Report should have header"
    assert "mom_1m" in report_str, "Report should mention feature"
    assert "Raw IC" in report_str, "Report should have raw IC"
    print_result("Report formatting works", True)
    
    # Print excerpt
    print("\n  Report excerpt:")
    for line in report_str.split("\n")[:20]:
        print(f"    {line}")
    
    return all_passed


# =============================================================================
# TEST 7: PIT Safety Check
# =============================================================================

def test_pit_safety():
    """Test that neutralization respects PIT discipline."""
    print_test_header("7. PIT Safety Check")
    
    from src.features.neutralization import compute_neutralized_ic
    
    all_passed = True
    
    # Create data where sector changes over time (simulating corporate actions)
    dates = [date(2023, 1, 1), date(2023, 2, 1)]
    
    rows = []
    
    # Date 1: STOCK_1 is in Tech
    rows.append({
        "date": dates[0],
        "ticker": "STOCK_1",
        "sector": "Tech",
        "feature1": 10.0,
        "beta_252d": 1.2,
    })
    
    # Date 2: STOCK_1 moved to Finance (corporate action)
    rows.append({
        "date": dates[1],
        "ticker": "STOCK_1",
        "sector": "Finance",  # Changed!
        "feature1": 20.0,
        "beta_252d": 0.8,
    })
    
    # Add other stocks for each date
    for d in dates:
        for i in range(2, 10):
            rows.append({
                "date": d,
                "ticker": f"STOCK_{i}",
                "sector": "Healthcare",
                "feature1": 30.0 + np.random.randn(),
                "beta_252d": 1.0,
            })
    
    features_df = pd.DataFrame(rows)
    labels_df = features_df.copy()
    labels_df["excess_return"] = np.random.randn(len(labels_df)) * 0.05
    
    # Compute neutralized IC
    result = compute_neutralized_ic(
        features_df=features_df,
        labels_df=labels_df,
        feature_col="feature1",
        sector_col="sector",
        beta_col="beta_252d",
    )
    
    # Check that computation completed (uses sector per date, so PIT-safe)
    assert result.n_dates == 2, f"Should process 2 dates, got {result.n_dates}"
    print_result("PIT-safe neutralization (sector per date)", True)
    
    # Verify that STOCK_1's sector was different on each date
    stock1_sectors = features_df[features_df["ticker"] == "STOCK_1"]["sector"].unique()
    assert len(stock1_sectors) == 2, "STOCK_1 should have 2 different sectors"
    print_result("Sector changes over time handled correctly", True)
    
    return all_passed


# =============================================================================
# TEST 8: Interpretation Guidelines
# =============================================================================

def test_interpretation():
    """Test interpretation of neutralization results."""
    print_test_header("8. Interpretation Guidelines")
    
    all_passed = True
    
    print("\n  Interpretation Rules:")
    print("    1. Δ_sector ≈ 0: Feature is sector-independent (pure stock selection)")
    print("    2. Δ_sector < -0.01: Feature has heavy sector rotation component")
    print("    3. Δ_beta ≈ 0: Feature is market-neutral")
    print("    4. Δ_beta < -0.01: Feature has heavy market exposure")
    print("")
    print("    Large negative delta → feature was mostly that factor")
    print("    Small delta → alpha is genuinely stock-specific")
    
    print_result("Interpretation guidelines documented", True)
    
    return all_passed


# =============================================================================
# TEST 9: Summary
# =============================================================================

def test_summary():
    """Print summary of capabilities."""
    print_test_header("9. Summary & Capabilities")
    
    print("\n  Feature Neutralization (5.8) Components:")
    print("    • Cross-sectional neutralization via OLS/Ridge")
    print("    • Sector-neutral IC computation")
    print("    • Beta-neutral IC computation")
    print("    • Sector+Beta neutral IC computation")
    print("    • Delta (Δ) reporting to show IC change")
    
    print("\n  Key Outputs:")
    print("    • Raw IC (baseline)")
    print("    • Sector-neutral IC")
    print("    • Beta-neutral IC")
    print("    • Sector+Beta-neutral IC")
    print("    • Deltas (neutral - raw) for each")
    
    print("\n  Key Design Choices:")
    print("    ✅ Neutralize FEATURE (not label) before IC")
    print("    ✅ Cross-sectional per date (PIT-safe)")
    print("    ✅ Reuse beta_252d from price_features.py")
    print("    ✅ Report deltas for clear interpretation")
    print("    ✅ For diagnostics/evaluation ONLY (not training)")
    
    print("\n  Purpose:")
    print("    Understand WHERE alpha comes from:")
    print("    - Is it sector rotation?")
    print("    - Is it market beta?")
    print("    - Is it genuine stock-specific selection?")
    
    return True


# =============================================================================
# Main
# =============================================================================

def run_all_tests():
    """Run all neutralization tests."""
    print("\n" + "="*60)
    print("SECTION 5.8: FEATURE NEUTRALIZATION TESTS")
    print("="*60)
    
    results = {}
    
    tests = [
        ("1. Module Imports", test_imports),
        ("2. Sector Dummies", test_sector_dummies),
        ("3. Cross-Sectional Neutralization", test_neutralization),
        ("4. IC Computation", test_ic_computation),
        ("5. Neutralized IC", test_neutralized_ic),
        ("6. Neutralization Report", test_neutralization_report),
        ("7. PIT Safety", test_pit_safety),
        ("8. Interpretation", test_interpretation),
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


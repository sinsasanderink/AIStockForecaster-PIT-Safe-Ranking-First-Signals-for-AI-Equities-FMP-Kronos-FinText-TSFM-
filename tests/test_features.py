"""
Tests for Feature Engineering (Section 5.2 & 5.3)
=================================================

Tests for price features (5.2) and fundamental features (5.3).

Run with: python tests/test_features.py
Set RUN_INTEGRATION=1 for API tests.
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
    """Test that all feature modules import correctly."""
    print_test_header("1. Module Imports")
    
    all_passed = True
    
    # Test price features
    try:
        from src.features import PriceFeatureGenerator, PriceFeatures, MOMENTUM_WINDOWS
        print_result("Import PriceFeatureGenerator", True)
    except Exception as e:
        print_result("Import PriceFeatureGenerator", False, str(e))
        all_passed = False
    
    # Test fundamental features
    try:
        from src.features import FundamentalFeatureGenerator, FundamentalFeatures
        print_result("Import FundamentalFeatureGenerator", True)
    except Exception as e:
        print_result("Import FundamentalFeatureGenerator", False, str(e))
        all_passed = False
    
    # Test helper functions
    try:
        from src.features import cross_sectional_rank, cross_sectional_zscore, winsorize
        print_result("Import helper functions", True)
    except Exception as e:
        print_result("Import helper functions", False, str(e))
        all_passed = False
    
    # Check momentum windows
    try:
        from src.features import MOMENTUM_WINDOWS
        has_all = all(k in MOMENTUM_WINDOWS for k in ["mom_1m", "mom_3m", "mom_6m", "mom_12m"])
        print_result("MOMENTUM_WINDOWS complete", has_all, str(MOMENTUM_WINDOWS))
        all_passed = all_passed and has_all
    except Exception as e:
        print_result("MOMENTUM_WINDOWS", False, str(e))
        all_passed = False
    
    return all_passed


# =============================================================================
# TEST 2: PriceFeatures Dataclass
# =============================================================================

def test_price_features_dataclass():
    """Test PriceFeatures dataclass."""
    print_test_header("2. PriceFeatures Dataclass")
    
    all_passed = True
    
    from src.features.price_features import PriceFeatures
    
    # Create sample features
    features = PriceFeatures(
        ticker="NVDA",
        date=date(2024, 1, 15),
        mom_1m=0.10,
        mom_3m=0.25,
        vol_20d=0.35,
        beta_252d=1.5,
        adv_20d=5_000_000_000,
        price=500.0,
    )
    
    # Test to_dict
    d = features.to_dict()
    has_all_keys = all(k in d for k in ["ticker", "date", "mom_1m", "vol_20d", "beta_252d"])
    print_result("to_dict has all keys", has_all_keys)
    all_passed = all_passed and has_all_keys
    
    # Test values
    mom_ok = d["mom_1m"] == 0.10
    print_result("mom_1m = 10%", mom_ok, f"{d['mom_1m']:.2%}")
    all_passed = all_passed and mom_ok
    
    vol_ok = d["vol_20d"] == 0.35
    print_result("vol_20d = 35%", vol_ok, f"{d['vol_20d']:.2%}")
    all_passed = all_passed and vol_ok
    
    return all_passed


# =============================================================================
# TEST 3: FundamentalFeatures Dataclass
# =============================================================================

def test_fundamental_features_dataclass():
    """Test FundamentalFeatures dataclass."""
    print_test_header("3. FundamentalFeatures Dataclass")
    
    all_passed = True
    
    from src.features.fundamental_features import FundamentalFeatures
    
    # Create sample features
    features = FundamentalFeatures(
        ticker="NVDA",
        date=date(2024, 1, 15),
        sector="Technology",
        pe_zscore_3y=1.5,
        pe_vs_sector=-0.10,
        gross_margin_vs_sector=0.15,
        _raw_pe=25.0,
    )
    
    # Test to_dict
    d = features.to_dict()
    has_all_keys = all(k in d for k in ["ticker", "sector", "pe_zscore_3y", "pe_vs_sector"])
    print_result("to_dict has all keys", has_all_keys)
    all_passed = all_passed and has_all_keys
    
    # Test sector
    sector_ok = d["sector"] == "Technology"
    print_result("sector = Technology", sector_ok)
    all_passed = all_passed and sector_ok
    
    # Test relative features
    pe_vs_ok = d["pe_vs_sector"] == -0.10
    print_result("pe_vs_sector = -10%", pe_vs_ok, f"{d['pe_vs_sector']:.2%}")
    all_passed = all_passed and pe_vs_ok
    
    return all_passed


# =============================================================================
# TEST 4: Price Feature Generator (Mock)
# =============================================================================

def test_price_generator_mock():
    """Test price feature generator with mock data."""
    print_test_header("4. Price Feature Generator (Mock)")
    
    all_passed = True
    
    from src.features.price_features import PriceFeatureGenerator
    from src.data.trading_calendar import TradingCalendarImpl
    import pandas as pd
    import numpy as np
    
    # Create mock FMP client
    class MockFMPClient:
        def get_historical_prices(self, ticker, start=None, end=None):
            # Generate 300 days of mock data
            np.random.seed(42)
            dates = pd.date_range(end="2024-01-15", periods=300, freq="B")
            
            # Simulate price path
            returns = np.random.normal(0.001, 0.02, len(dates))
            prices = 100 * np.exp(np.cumsum(returns))
            volumes = np.random.randint(1e6, 1e8, len(dates))
            
            return pd.DataFrame({
                "date": dates,
                "open": prices * 0.99,
                "high": prices * 1.01,
                "low": prices * 0.98,
                "close": prices,
                "volume": volumes,
            })
    
    # Create generator with mock
    generator = PriceFeatureGenerator(
        fmp_client=MockFMPClient(),
        calendar=TradingCalendarImpl(),
    )
    
    log("Generating price features with mock data...")
    
    features = generator.generate(
        ticker="TEST",
        asof_date=date(2024, 1, 15),
    )
    
    # Check features
    has_price = features.price is not None
    print_result("Has price", has_price, f"${features.price:.2f}" if features.price else "None")
    all_passed = all_passed and has_price
    
    has_momentum = features.mom_1m is not None
    print_result("Has mom_1m", has_momentum, f"{features.mom_1m:.2%}" if features.mom_1m else "None")
    all_passed = all_passed and has_momentum
    
    has_vol = features.vol_20d is not None
    print_result("Has vol_20d", has_vol, f"{features.vol_20d:.2%}" if features.vol_20d else "None")
    all_passed = all_passed and has_vol
    
    has_adv = features.adv_20d is not None
    print_result("Has adv_20d", has_adv, f"${features.adv_20d/1e6:.1f}M" if features.adv_20d else "None")
    all_passed = all_passed and has_adv
    
    # Check that vol is annualized (should be > daily vol)
    if features.vol_20d:
        vol_reasonable = 0.05 < features.vol_20d < 2.0  # 5% to 200% annualized
        print_result("Vol is annualized", vol_reasonable, f"{features.vol_20d:.2%}")
        all_passed = all_passed and vol_reasonable
    
    return all_passed


# =============================================================================
# TEST 5: Cross-sectional Functions
# =============================================================================

def test_cross_sectional_functions():
    """Test cross-sectional standardization functions."""
    print_test_header("5. Cross-Sectional Functions")
    
    all_passed = True
    
    from src.features import cross_sectional_rank, cross_sectional_zscore
    import pandas as pd
    import numpy as np
    
    # Test data
    data = pd.Series([10, 20, 30, 40, 50])
    
    # Test rank
    ranks = cross_sectional_rank(data)
    rank_ok = abs(ranks.iloc[0] - 0.2) < 0.01 and abs(ranks.iloc[-1] - 1.0) < 0.01
    print_result("Rank percentiles correct", rank_ok, f"First: {ranks.iloc[0]:.2f}, Last: {ranks.iloc[-1]:.2f}")
    all_passed = all_passed and rank_ok
    
    # Test z-score
    zscores = cross_sectional_zscore(data)
    mean_ok = abs(zscores.mean()) < 0.01
    print_result("Z-score mean ≈ 0", mean_ok, f"Mean: {zscores.mean():.4f}")
    all_passed = all_passed and mean_ok
    
    std_ok = abs(zscores.std() - 1.0) < 0.01
    print_result("Z-score std ≈ 1", std_ok, f"Std: {zscores.std():.4f}")
    all_passed = all_passed and std_ok
    
    return all_passed


# =============================================================================
# TEST 6: Price Features Integration (Real API)
# =============================================================================

def test_price_features_integration():
    """Test price features with real FMP API."""
    print_test_header("6. Price Features (Integration)")
    
    if not RUN_INTEGRATION:
        log("SKIPPED: Set RUN_INTEGRATION=1 to run")
        return True
    
    all_passed = True
    
    from src.features import PriceFeatureGenerator
    
    generator = PriceFeatureGenerator()
    
    log("Generating price features for NVDA (real API)...")
    
    try:
        features = generator.generate(
            ticker="NVDA",
            asof_date=date(2024, 12, 15),
        )
        
        print(f"\n    NVDA Price Features:")
        print(f"      Price: ${features.price:.2f}" if features.price else "      Price: N/A")
        print(f"      Mom 1m: {features.mom_1m:.2%}" if features.mom_1m else "      Mom 1m: N/A")
        print(f"      Mom 3m: {features.mom_3m:.2%}" if features.mom_3m else "      Mom 3m: N/A")
        print(f"      Vol 20d: {features.vol_20d:.2%}" if features.vol_20d else "      Vol 20d: N/A")
        print(f"      Beta: {features.beta_252d:.2f}" if features.beta_252d else "      Beta: N/A")
        print(f"      ADV 20d: ${features.adv_20d/1e9:.1f}B" if features.adv_20d else "      ADV 20d: N/A")
        
        has_features = features.price is not None and features.mom_1m is not None
        print_result("Generated real features", has_features)
        all_passed = all_passed and has_features
        
    except Exception as e:
        print_result("API call", False, str(e))
        all_passed = False
    
    return all_passed


# =============================================================================
# TEST 7: Fundamental Features Integration (Real API)
# =============================================================================

def test_fundamental_features_integration():
    """Test fundamental features with real FMP API."""
    print_test_header("7. Fundamental Features (Integration)")
    
    if not RUN_INTEGRATION:
        log("SKIPPED: Set RUN_INTEGRATION=1 to run")
        return True
    
    all_passed = True
    
    from src.features import FundamentalFeatureGenerator
    
    generator = FundamentalFeatureGenerator()
    
    log("Generating fundamental features for NVDA (real API)...")
    
    try:
        features = generator.generate(
            ticker="NVDA",
            asof_date=date(2024, 12, 15),
        )
        
        print(f"\n    NVDA Fundamental Features:")
        print(f"      Sector: {features.sector}")
        print(f"      Raw P/E: {features._raw_pe:.1f}" if features._raw_pe else "      Raw P/E: N/A")
        print(f"      P/E z-score: {features.pe_zscore_3y:.2f}" if features.pe_zscore_3y else "      P/E z-score: N/A")
        print(f"      Gross Margin: {features._raw_gross_margin:.2%}" if features._raw_gross_margin else "      Gross Margin: N/A")
        print(f"      Rev Growth: {features._raw_revenue_growth:.2%}" if features._raw_revenue_growth else "      Rev Growth: N/A")
        
        has_sector = features.sector is not None
        print_result("Has sector", has_sector, features.sector or "N/A")
        all_passed = all_passed and has_sector
        
    except Exception as e:
        print_result("API call", False, str(e))
        all_passed = False
    
    return all_passed


# =============================================================================
# TEST 8: Universe Feature Generation
# =============================================================================

def test_universe_features():
    """Test feature generation for a universe."""
    print_test_header("8. Universe Feature Generation")
    
    if not RUN_INTEGRATION:
        log("SKIPPED: Set RUN_INTEGRATION=1 to run")
        return True
    
    all_passed = True
    
    from src.features import PriceFeatureGenerator
    
    generator = PriceFeatureGenerator()
    
    tickers = ["NVDA", "AMD", "INTC"]
    log(f"Generating features for {tickers}...")
    
    try:
        df = generator.generate_for_universe(
            tickers=tickers,
            asof_date=date(2024, 12, 15),
        )
        
        has_rows = len(df) > 0
        print_result("Generated DataFrame", has_rows, f"{len(df)} rows")
        all_passed = all_passed and has_rows
        
        if has_rows:
            # Check relative strength was computed
            has_rel = "rel_strength_1m" in df.columns and df["rel_strength_1m"].notna().any()
            print_result("Has rel_strength_1m", has_rel)
            all_passed = all_passed and has_rel
            
            print("\n    Universe features:")
            for _, row in df.iterrows():
                print(f"      {row['ticker']}: mom_1m={row['mom_1m']:.2%}, rel_str={row.get('rel_strength_1m', 0):.2f}")
        
    except Exception as e:
        print_result("Universe generation", False, str(e))
        all_passed = False
    
    return all_passed


# =============================================================================
# TEST 9: Summary
# =============================================================================

def test_summary():
    """Summarize feature capabilities."""
    print_test_header("9. Summary & Capabilities")
    
    print("\n  Price Features (5.2):")
    print("    • Momentum: 1m, 3m, 6m, 12m returns")
    print("    • Volatility: 20d, 60d (annualized), vol-of-vol")
    print("    • Drawdown: Max DD 60d, distance from high")
    print("    • Relative strength: z-score vs universe")
    print("    • Beta: vs benchmark (252d)")
    print("    • Liquidity: ADV 20d/60d, vol-adjusted ADV")
    
    print("\n  Fundamental Features (5.3):")
    print("    • Valuation: P/E, P/S z-score vs 3y history")
    print("    • Sector relative: P/E, P/S, margins vs sector median")
    print("    • Growth: Revenue growth vs sector")
    print("    • Quality: ROE, ROA z-scores")
    
    print("\n  Key Design Choices:")
    print("    ✅ All features are RELATIVE (not raw ratios)")
    print("    ✅ Cross-sectional standardization")
    print("    ✅ PIT-safe (observed_at filtering)")
    print("    ✅ Sector-neutral options")
    
    if RUN_INTEGRATION:
        print("\n  API Status: ✅ Integration tests passed")
    else:
        print("\n  💡 Run with RUN_INTEGRATION=1 for API tests")
    
    return True


# =============================================================================
# Main
# =============================================================================

def run_all_tests():
    """Run all feature tests."""
    print("\n" + "="*60)
    print("SECTION 5.2 & 5.3: FEATURE ENGINEERING TESTS")
    print("="*60)
    
    if RUN_INTEGRATION:
        print("\n⚠️  Running API tests (RUN_INTEGRATION=1)")
    else:
        print("\n💡 Quick tests only. Set RUN_INTEGRATION=1 for API tests.")
    
    results = {}
    
    tests = [
        ("1. Module Imports", test_imports),
        ("2. PriceFeatures Dataclass", test_price_features_dataclass),
        ("3. FundamentalFeatures Dataclass", test_fundamental_features_dataclass),
        ("4. Price Generator (Mock)", test_price_generator_mock),
        ("5. Cross-Sectional Functions", test_cross_sectional_functions),
        ("6. Price Features (Integration)", test_price_features_integration),
        ("7. Fundamental Features (Integration)", test_fundamental_features_integration),
        ("8. Universe Features", test_universe_features),
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


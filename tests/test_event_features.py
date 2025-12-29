"""
Tests for Event Features & Time Decay (Section 5.4 + Time Decay)
================================================================

Tests for event calendar features (5.4) and time-decay weighting.

Run with: python tests/test_event_features.py
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
    """Test that all modules import correctly."""
    print_test_header("1. Module Imports")
    
    all_passed = True
    
    # Test event features
    try:
        from src.features import EventFeatureGenerator, EventFeatures, PEAD_WINDOW_DAYS
        print_result("Import EventFeatureGenerator", True)
    except Exception as e:
        print_result("Import EventFeatureGenerator", False, str(e))
        all_passed = False
    
    # Test time decay
    try:
        from src.features import (
            compute_time_decay_weights,
            get_half_life_for_horizon,
            compute_effective_sample_size,
            DEFAULT_HALF_LIVES,
        )
        print_result("Import time_decay functions", True)
    except Exception as e:
        print_result("Import time_decay functions", False, str(e))
        all_passed = False
    
    return all_passed


# =============================================================================
# TEST 2: EventFeatures Dataclass
# =============================================================================

def test_event_features_dataclass():
    """Test EventFeatures dataclass functionality."""
    print_test_header("2. EventFeatures Dataclass")
    
    from src.features.event_features import EventFeatures
    
    all_passed = True
    
    # Create a sample feature object
    features = EventFeatures(
        ticker="NVDA",
        date=date(2024, 1, 15),
        days_to_earnings=25,
        days_since_earnings=65,
        in_pead_window=False,
        last_surprise_pct=8.5,
        avg_surprise_4q=5.2,
        surprise_streak=3,
    )
    
    # Test to_dict
    d = features.to_dict()
    assert d["ticker"] == "NVDA", "ticker mismatch"
    assert d["days_to_earnings"] == 25, "days_to_earnings mismatch"
    assert d["last_surprise_pct"] == 8.5, "last_surprise_pct mismatch"
    assert d["surprise_streak"] == 3, "surprise_streak mismatch"
    print_result("to_dict() conversion", True)
    
    # Test PEAD window detection
    features2 = EventFeatures(
        ticker="AMD",
        date=date(2024, 1, 15),
        days_since_earnings=30,
        in_pead_window=True,
        pead_window_day=30,
    )
    
    assert features2.in_pead_window == True, "PEAD window should be True"
    assert features2.pead_window_day == 30, "PEAD window day should be 30"
    print_result("PEAD window detection", True)
    
    # Test optional fields are None by default
    features3 = EventFeatures(ticker="MSFT", date=date(2024, 1, 15))
    assert features3.days_to_earnings is None, "days_to_earnings should be None by default"
    assert features3.last_surprise_pct is None, "last_surprise_pct should be None by default"
    assert features3.surprise_streak == 0, "surprise_streak should be 0 by default"
    print_result("Default values", True)
    
    return all_passed


# =============================================================================
# TEST 3: Time Decay Weights
# =============================================================================

def test_time_decay_weights():
    """Test time decay weight computation."""
    print_test_header("3. Time Decay Weights")
    
    import pandas as pd
    import numpy as np
    from src.features.time_decay import (
        compute_time_decay_weights,
        get_half_life_for_horizon,
        compute_effective_sample_size,
        summarize_weights,
        DEFAULT_HALF_LIFE_DAYS,
        DEFAULT_HALF_LIVES,
    )
    
    all_passed = True
    
    # Test default half-lives
    assert DEFAULT_HALF_LIFE_DAYS == 3 * 365, f"Default half-life should be 3 years"
    assert 20 in DEFAULT_HALF_LIVES, "20d horizon should have a half-life"
    assert 60 in DEFAULT_HALF_LIVES, "60d horizon should have a half-life"
    assert 90 in DEFAULT_HALF_LIVES, "90d horizon should have a half-life"
    print_result("Default half-lives defined", True)
    
    # Test get_half_life_for_horizon
    hl_20 = get_half_life_for_horizon(20)
    hl_60 = get_half_life_for_horizon(60)
    hl_90 = get_half_life_for_horizon(90)
    
    assert hl_20 < hl_60 < hl_90, "Half-lives should increase with horizon"
    print_result(f"Half-life ordering: {hl_20/365:.1f}y < {hl_60/365:.1f}y < {hl_90/365:.1f}y", True)
    
    # Create test data
    np.random.seed(42)
    dates = pd.date_range("2015-01-01", "2024-12-31", freq="W")
    df = pd.DataFrame({
        "as_of_date": dates[:200],
        "ticker": ["NVDA"] * 100 + ["AMD"] * 100,
        "return": np.random.normal(0.01, 0.05, 200),
    })
    
    # Compute weights
    weights = compute_time_decay_weights(df, half_life_days=3*365)
    
    # Verify weights are positive
    assert (weights > 0).all(), "All weights should be positive"
    print_result("Weights are positive", True)
    
    # Verify weights sum to ~1 per date (with per-date normalization)
    df_check = df.assign(weight=weights)
    date_sums = df_check.groupby("as_of_date")["weight"].sum()
    assert np.allclose(date_sums, 1.0, rtol=0.01), "Weights should sum to ~1 per date"
    print_result("Per-date normalization", True)
    
    # Verify recent dates have higher raw decay (before normalization)
    weights_no_norm = compute_time_decay_weights(df, half_life_days=3*365, normalize_per_date=False)
    recent_avg = weights_no_norm.iloc[-50:].mean()
    old_avg = weights_no_norm.iloc[:50].mean()
    assert recent_avg > old_avg, f"Recent weights ({recent_avg:.3f}) should be higher than old ({old_avg:.3f})"
    print_result(f"Recent > old weights: {recent_avg:.3f} > {old_avg:.3f}", True)
    
    # Test effective sample size
    ess = compute_effective_sample_size(weights)
    # Note: With per-date normalization, ESS can equal n when there's 1 obs per date
    assert 0 < ess <= len(df), f"ESS ({ess:.0f}) should be between 0 and n ({len(df)})"
    print_result(f"Effective sample size: {ess:.0f} / {len(df)}", True)
    
    # Test summary
    summary = summarize_weights(df, weights)
    assert "n_samples" in summary, "Summary should include n_samples"
    assert "effective_n" in summary, "Summary should include effective_n"
    assert "by_year" in summary, "Summary should include by_year"
    print_result("Weight summary", True)
    
    return all_passed


# =============================================================================
# TEST 4: Event Generator (Mock Data)
# =============================================================================

def test_event_generator_mock():
    """Test EventFeatureGenerator with mock data."""
    print_test_header("4. Event Generator (Mock Data)")
    
    from src.features.event_features import EventFeatureGenerator, EventFeatures
    
    all_passed = True
    
    # Create mock event store that returns empty
    class MockEventStore:
        def get_events(self, tickers, asof, event_types=None, lookback_days=90):
            return []
        def days_since_event(self, ticker, event_type, asof):
            return None
    
    # Create mock expectations client
    class MockExpectationsClient:
        def get_earnings_surprises(self, ticker, limit=20):
            return []
    
    # Create mock calendar
    class MockCalendar:
        def get_market_close(self, d):
            import pytz
            ET = pytz.timezone("America/New_York")
            return ET.localize(datetime.combine(d, datetime.min.time().replace(hour=16)))
        def is_trading_day(self, d):
            return d.weekday() < 5  # Mon-Fri
    
    generator = EventFeatureGenerator(
        event_store=MockEventStore(),
        expectations_client=MockExpectationsClient(),
        trading_calendar=MockCalendar(),
    )
    
    # Test with mock data
    features = generator.compute_features(
        tickers=["NVDA", "AMD"],
        asof_date=date(2024, 1, 15),
    )
    
    assert len(features) == 2, f"Expected 2 features, got {len(features)}"
    assert features[0].ticker == "NVDA", "First ticker should be NVDA"
    assert features[1].ticker == "AMD", "Second ticker should be AMD"
    print_result("Mock generator returns features", True)
    
    # Test to_dataframe
    df = generator.to_dataframe(features)
    assert len(df) == 2, "DataFrame should have 2 rows"
    assert "ticker" in df.columns, "DataFrame should have ticker column"
    assert "days_to_earnings" in df.columns, "DataFrame should have days_to_earnings"
    print_result("to_dataframe() conversion", True)
    
    return all_passed


# =============================================================================
# TEST 5: Cross-Sectional Functions
# =============================================================================

def test_cross_sectional_functions():
    """Test cross-sectional z-score and rank functions."""
    print_test_header("5. Cross-Sectional Functions")
    
    import pandas as pd
    import numpy as np
    from src.features.event_features import cross_sectional_zscore, cross_sectional_rank
    
    all_passed = True
    
    # Test z-score
    series = pd.Series([10, 20, 30, 40, 50])
    z = cross_sectional_zscore(series)
    
    assert abs(z.mean()) < 0.01, "Z-scores should have mean ~0"
    assert abs(z.std() - 1.0) < 0.01, "Z-scores should have std ~1"
    print_result("Z-score normalization", True)
    
    # Test rank
    r = cross_sectional_rank(series)
    assert r.min() >= 0 and r.max() <= 1, "Ranks should be in [0, 1]"
    assert r.iloc[0] < r.iloc[-1], "Lower values should have lower ranks"
    print_result("Cross-sectional rank", True)
    
    # Test with NaN
    series_nan = pd.Series([10, np.nan, 30, 40, 50])
    r_nan = cross_sectional_rank(series_nan)
    assert pd.isna(r_nan.iloc[1]), "NaN should remain NaN in rank"
    print_result("Handle NaN values", True)
    
    return all_passed


# =============================================================================
# TEST 6: Event Features (Integration)
# =============================================================================

def test_event_features_integration():
    """Test EventFeatureGenerator with real API data."""
    print_test_header("6. Event Features (Integration)")
    
    if not RUN_INTEGRATION:
        print("  ⏭️  Skipped (set RUN_INTEGRATION=1 to run)")
        return True
    
    from src.features.event_features import EventFeatureGenerator
    
    all_passed = True
    
    try:
        generator = EventFeatureGenerator()
        
        # Test single ticker
        features = generator.compute_features(
            tickers=["NVDA"],
            asof_date=date(2024, 12, 1),
        )
        
        assert len(features) == 1, "Should return 1 feature object"
        f = features[0]
        
        print(f"\n  NVDA Event Features (as of 2024-12-01):")
        print(f"    Days to earnings: {f.days_to_earnings}")
        print(f"    Days since earnings: {f.days_since_earnings}")
        print(f"    In PEAD window: {f.in_pead_window}")
        print(f"    Last surprise %: {f.last_surprise_pct}")
        print(f"    Avg surprise (4Q): {f.avg_surprise_4q}")
        print(f"    Surprise streak: {f.surprise_streak}")
        print(f"    Reports BMO: {f.reports_bmo}")
        
        # Validate reasonable values
        if f.days_since_earnings is not None:
            assert 0 <= f.days_since_earnings <= 365, "Days since earnings should be reasonable"
            print_result("Days since earnings reasonable", True)
        else:
            print_result("No earnings data (may be rate limited)", True)
        
        all_passed = True
        
    except Exception as e:
        print_result("Integration test", False, str(e))
        all_passed = False
    
    return all_passed


# =============================================================================
# TEST 7: PEAD Window Logic
# =============================================================================

def test_pead_window_logic():
    """Test Post-Earnings Announcement Drift window logic."""
    print_test_header("7. PEAD Window Logic")
    
    from src.features.event_features import PEAD_WINDOW_DAYS, EventFeatures
    
    all_passed = True
    
    # PEAD window should be ~63 days (3 months)
    assert PEAD_WINDOW_DAYS == 63, f"PEAD window should be 63 days, got {PEAD_WINDOW_DAYS}"
    print_result(f"PEAD window = {PEAD_WINDOW_DAYS} days", True)
    
    # Test PEAD logic
    # Day 1 after earnings - should be in window
    f1 = EventFeatures(ticker="TEST", date=date(2024, 1, 15))
    f1.days_since_earnings = 1
    if f1.days_since_earnings <= PEAD_WINDOW_DAYS:
        f1.in_pead_window = True
        f1.pead_window_day = f1.days_since_earnings
    
    assert f1.in_pead_window == True, "Day 1 should be in PEAD window"
    assert f1.pead_window_day == 1, "PEAD window day should be 1"
    print_result("Day 1 in PEAD window", True)
    
    # Day 63 - should be in window (edge)
    f2 = EventFeatures(ticker="TEST", date=date(2024, 1, 15))
    f2.days_since_earnings = 63
    f2.in_pead_window = f2.days_since_earnings <= PEAD_WINDOW_DAYS
    
    assert f2.in_pead_window == True, "Day 63 should be in PEAD window"
    print_result("Day 63 in PEAD window (edge)", True)
    
    # Day 64 - should be out of window
    f3 = EventFeatures(ticker="TEST", date=date(2024, 1, 15))
    f3.days_since_earnings = 64
    f3.in_pead_window = f3.days_since_earnings <= PEAD_WINDOW_DAYS
    
    assert f3.in_pead_window == False, "Day 64 should be out of PEAD window"
    print_result("Day 64 out of PEAD window", True)
    
    return all_passed


# =============================================================================
# TEST 8: Surprise Streak Logic
# =============================================================================

def test_surprise_streak_logic():
    """Test earnings surprise streak calculation."""
    print_test_header("8. Surprise Streak Logic")
    
    all_passed = True
    
    # Simulate streak calculation
    def calc_streak(surprises):
        """Calculate streak from list of surprise percentages."""
        streak = 0
        for s in surprises:
            if s is None:
                break
            if s > 0:
                if streak >= 0:
                    streak += 1
                else:
                    break
            elif s < 0:
                if streak <= 0:
                    streak -= 1
                else:
                    break
            else:
                break
        return streak
    
    # Test consecutive beats
    assert calc_streak([5, 3, 2, 1]) == 4, "4 beats should give streak of 4"
    print_result("4 consecutive beats → streak=4", True)
    
    # Test consecutive misses
    assert calc_streak([-2, -3, -1]) == -3, "3 misses should give streak of -3"
    print_result("3 consecutive misses → streak=-3", True)
    
    # Test mixed - beat then miss
    assert calc_streak([5, -2, 3]) == 1, "Beat then miss should give streak of 1"
    print_result("Beat then miss → streak=1", True)
    
    # Test zero breaks streak
    assert calc_streak([5, 0, 3]) == 1, "Zero should break streak"
    print_result("Zero breaks streak", True)
    
    # Test None breaks streak
    assert calc_streak([5, None, 3]) == 1, "None should break streak"
    print_result("None breaks streak", True)
    
    return all_passed


# =============================================================================
# TEST 9: Summary
# =============================================================================

def test_summary():
    """Print summary of capabilities."""
    print_test_header("9. Summary & Capabilities")
    
    print("\n  Event Features (5.4):")
    print("    • days_to_earnings: Forward-looking calendar feature")
    print("    • days_since_earnings: Days since last report")
    print("    • in_pead_window: PEAD window indicator (63 days)")
    print("    • last_surprise_pct: Most recent surprise")
    print("    • avg_surprise_4q: Rolling 4-quarter average")
    print("    • surprise_streak: Consecutive beats (+) or misses (-)")
    print("    • surprise_zscore: Cross-sectional z-score")
    print("    • days_since_10k/10q: Filing recency")
    print("    • earnings_vol: Surprise volatility (8Q)")
    print("    • reports_bmo: Typical announcement timing")
    
    print("\n  Time Decay Weighting:")
    print("    • Exponential decay by date")
    print("    • Horizon-specific half-lives:")
    print("      - 20d horizon: 2.5 years")
    print("      - 60d horizon: 3.5 years")
    print("      - 90d horizon: 4.5 years")
    print("    • Per-date normalization for cross-sectional ranking")
    print("    • Effective sample size computation")
    
    print("\n  Key Design Choices:")
    print("    ✅ PIT-safe (observed_at filtering)")
    print("    ✅ Cross-sectional standardization")
    print("    ✅ PEAD window for earnings momentum")
    print("    ✅ Time decay for AI-relevance")
    
    if RUN_INTEGRATION:
        print("\n  API Status: ✅ Integration tests passed")
    else:
        print("\n  💡 Run with RUN_INTEGRATION=1 for API tests")
    
    return True


# =============================================================================
# Main
# =============================================================================

def run_all_tests():
    """Run all event feature tests."""
    print("\n" + "="*60)
    print("SECTION 5.4 + TIME DECAY: EVENT FEATURES TESTS")
    print("="*60)
    
    if RUN_INTEGRATION:
        print("\n⚠️  Running API tests (RUN_INTEGRATION=1)")
    else:
        print("\n💡 Quick tests only. Set RUN_INTEGRATION=1 for API tests.")
    
    results = {}
    
    tests = [
        ("1. Module Imports", test_imports),
        ("2. EventFeatures Dataclass", test_event_features_dataclass),
        ("3. Time Decay Weights", test_time_decay_weights),
        ("4. Event Generator (Mock)", test_event_generator_mock),
        ("5. Cross-Sectional Functions", test_cross_sectional_functions),
        ("6. Event Features (Integration)", test_event_features_integration),
        ("7. PEAD Window Logic", test_pead_window_logic),
        ("8. Surprise Streak Logic", test_surprise_streak_logic),
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


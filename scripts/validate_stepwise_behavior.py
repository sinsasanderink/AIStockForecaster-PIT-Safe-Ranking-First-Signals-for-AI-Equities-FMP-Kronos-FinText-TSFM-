#!/usr/bin/env python3
"""
Validate stepwise behavior of fundamental features (Batch 5).

Checks:
1. Raw TTM metrics must be piecewise-constant between filings
2. days_since_10q/10k should reset (detected via drops, not absolute min)
3. Z-scores should have low change rates (stepwise per-ticker)
4. No suspicious identical patterns across tickers
5. Filing counts sanity (10-K ~1/year, 10-Q ~3-4/year)
"""

import duckdb
import pandas as pd
import numpy as np
from pathlib import Path

DB_PATH = "data/features.duckdb"

# Test parameters
TICKERS = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "AVAV", "PLTR"]
START = "2023-01-01"
END = "2024-12-31"

# Minimum non-null values required to judge a feature's behavior
MIN_COVERAGE = 50

RAW_COLS = ["gross_margin_ttm", "operating_margin_ttm", "revenue_growth_yoy", "roe_raw"]
ZSCORE_COLS = ["gross_margin_vs_sector", "operating_margin_vs_sector", "revenue_growth_vs_sector", "roe_zscore"]
EVENT_COLS = ["days_since_10q", "days_since_10k", "days_since_earnings"]


def compute_change_rate(values: np.ndarray) -> tuple:
    """
    Compute change rate for a time series.
    Uses np.isclose() to avoid counting float noise as changes.
    
    Returns: (n_changes, n_non_nan, change_rate)
    """
    non_nan_mask = ~pd.isna(values)
    non_nan_values = np.asarray(values)[non_nan_mask]
    
    if len(non_nan_values) <= 1:
        return 0, len(non_nan_values), np.nan
    
    # Use np.isclose to avoid float noise being counted as changes
    diffs = np.diff(non_nan_values.astype(float))
    changes = ~np.isclose(diffs, 0.0, atol=1e-12, rtol=1e-9)
    n_changes = int(changes.sum())
    change_rate = n_changes / (len(non_nan_values) - 1)
    
    return n_changes, len(non_nan_values), change_rate


def test_raw_ttm_piecewise_constant():
    """Test that raw TTM metrics are piecewise-constant between filings."""
    
    print("=" * 70)
    print("TEST 1: Raw TTM Piecewise-Constant (should be <5% change rate)")
    print("=" * 70)
    print()
    
    con = duckdb.connect(DB_PATH, read_only=True)
    
    # Check if raw columns exist
    cols_info = con.execute("PRAGMA table_info(features)").df()
    existing_raw_cols = [c for c in RAW_COLS if c in cols_info["name"].values]
    
    if not existing_raw_cols:
        print("⚠️  No raw TTM columns found in schema")
        print(f"   Expected: {RAW_COLS}")
        print("   This build may not have raw TTM columns yet.")
        con.close()
        return None  # Inconclusive, not a fail
    
    query = f"""
        SELECT date, ticker, {', '.join(existing_raw_cols)}
        FROM features
        WHERE ticker IN ({','.join(f"'{t}'" for t in TICKERS)})
          AND date >= '{START}' AND date <= '{END}'
        ORDER BY ticker, date
    """
    
    df = con.execute(query).df()
    con.close()
    
    if df.empty:
        print("❌ No data returned")
        return False
    
    results = []
    skipped = 0
    
    for ticker in TICKERS:
        ticker_df = df[df["ticker"] == ticker].sort_values("date")
        
        for col in existing_raw_cols:
            n_changes, n_non_nan, change_rate = compute_change_rate(ticker_df[col].values)
            
            # Skip if insufficient coverage
            if n_non_nan < MIN_COVERAGE:
                skipped += 1
                continue
            
            results.append({
                "ticker": ticker,
                "feature": col,
                "n_days": len(ticker_df),
                "non_nan": n_non_nan,
                "n_changes": n_changes,
                "change_rate": change_rate,
            })
    
    if not results:
        print(f"⚠️  All ticker/feature combinations had <{MIN_COVERAGE} non-null values")
        return None
    
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    print()
    
    if skipped > 0:
        print(f"   (Skipped {skipped} combinations with <{MIN_COVERAGE} non-null values)")
        print()
    
    # Pass if max change rate < 5% (quarterly = ~4 changes/year out of ~252 days = 1.6%)
    max_rate = results_df["change_rate"].max()
    
    if max_rate > 0.05:
        print(f"❌ FAIL: Max change rate {max_rate:.4f} > 0.05")
        print("   Raw TTM should only change ~4 times per year (quarterly)")
        return False
    else:
        print(f"✓ PASS: Max change rate {max_rate:.4f} <= 0.05")
        return True


def test_days_since_resets_by_drops():
    """
    Test that days_since_10q/10k resets by detecting negative jumps (drops).
    
    This is more robust than checking for min=0, because:
    - Filing may happen on weekend (next trading day has days_since=2-3)
    - Non-trading days aren't in features_df
    """
    
    print()
    print("=" * 70)
    print("TEST 2: days_since Resets (detected via drops, not absolute min)")
    print("=" * 70)
    print()
    
    con = duckdb.connect(DB_PATH, read_only=True)
    
    query = f"""
        SELECT date, ticker, days_since_10q, days_since_10k
        FROM features
        WHERE ticker IN ({','.join(f"'{t}'" for t in TICKERS)})
          AND date >= '{START}' AND date <= '{END}'
        ORDER BY ticker, date
    """
    df = con.execute(query).df()
    con.close()
    
    if df.empty:
        print("❌ No data returned")
        return False
    
    passed = True
    summary_rows = []
    
    for ticker in TICKERS:
        tdf = df[df["ticker"] == ticker].sort_values("date")
        
        for col in ["days_since_10q", "days_since_10k"]:
            s = tdf[col].dropna().astype(float)
            if len(s) < 10:
                summary_rows.append({
                    "ticker": ticker, "column": col, 
                    "min": None, "max": None, "drops": None, "status": "SPARSE"
                })
                continue
            
            diffs = s.diff().dropna()
            n_drops = int((diffs < 0).sum())
            min_val = s.min()
            max_val = s.max()
            
            # Expectations:
            # - 10-K: 1-2 drops in 2 years (annual filing)
            # - 10-Q: 6-8 drops in 2 years (quarterly filings, ~4/year)
            if col == "days_since_10k":
                expected_min_drops = 1
                status = "✓ OK" if n_drops >= expected_min_drops else "⚠️  LOW"
            else:
                expected_min_drops = 4
                status = "✓ OK" if n_drops >= expected_min_drops else "⚠️  LOW"
            
            if n_drops < expected_min_drops:
                passed = False
            
            summary_rows.append({
                "ticker": ticker, "column": col,
                "min": min_val, "max": max_val, "drops": n_drops, "status": status
            })
    
    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False))
    print()
    
    print("Interpretation:")
    print("  - 'drops' = number of times the series decreased (reset)")
    print("  - 10-Q should have ~4 drops/year (quarterly)")
    print("  - 10-K should have ~1 drop/year (annual)")
    print()
    
    if passed:
        print("✓ PASS: days_since series show expected reset drops")
    else:
        print("❌ FAIL: Some days_since series have too few drops")
        print("   Check if filings data is being downloaded correctly")
    
    return passed


def test_zscore_stepwise():
    """Test that z-scores are stepwise per-ticker (not daily drift)."""
    
    print()
    print("=" * 70)
    print("TEST 3: Z-Score Stepwise Behavior (should be <20% change rate)")
    print("=" * 70)
    print()
    
    con = duckdb.connect(DB_PATH, read_only=True)
    
    # Check if z-score columns exist
    cols_info = con.execute("PRAGMA table_info(features)").df()
    existing_zscore_cols = [c for c in ZSCORE_COLS if c in cols_info["name"].values]
    
    if not existing_zscore_cols:
        print("⚠️  No z-score columns found in schema")
        con.close()
        return None
    
    query = f"""
        SELECT date, ticker, {', '.join(existing_zscore_cols)}
        FROM features
        WHERE ticker IN ({','.join(f"'{t}'" for t in TICKERS)})
          AND date >= '{START}' AND date <= '{END}'
        ORDER BY ticker, date
    """
    
    df = con.execute(query).df()
    con.close()
    
    if df.empty:
        print("❌ No data returned")
        return False
    
    results = []
    skipped = 0
    
    for ticker in TICKERS:
        ticker_df = df[df["ticker"] == ticker].sort_values("date")
        
        for col in existing_zscore_cols:
            n_changes, n_non_nan, change_rate = compute_change_rate(ticker_df[col].values)
            
            # Skip if insufficient coverage
            if n_non_nan < MIN_COVERAGE:
                skipped += 1
                continue
            
            results.append({
                "ticker": ticker,
                "feature": col,
                "n_days": len(ticker_df),
                "non_nan": n_non_nan,
                "n_changes": n_changes,
                "change_rate": change_rate,
            })
    
    if not results:
        print(f"⚠️  All ticker/feature combinations had <{MIN_COVERAGE} non-null values")
        return None
    
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    print()
    
    if skipped > 0:
        print(f"   (Skipped {skipped} combinations with <{MIN_COVERAGE} non-null values)")
        print()
    
    # Z-scores can change more often than raw (due to cross-section), but should NOT be 50%+
    # Allow up to 20% (still stepwise, ~50 changes/year = monthly)
    max_rate = results_df["change_rate"].max()
    
    if max_rate > 0.20:
        print(f"❌ FAIL: Max change rate {max_rate:.4f} > 0.20")
        print("   Z-scores should be stepwise per-ticker (forward-fill)")
        return False
    else:
        print(f"✓ PASS: Max change rate {max_rate:.4f} <= 0.20")
        return True


def test_no_identical_patterns():
    """Test that different tickers don't have identical change counts."""
    
    print()
    print("=" * 70)
    print("TEST 4: No Identical Change Patterns (cross-section bug check)")
    print("=" * 70)
    print()
    
    con = duckdb.connect(DB_PATH, read_only=True)
    
    # Get available columns
    cols_info = con.execute("PRAGMA table_info(features)").df()
    all_cols = [c for c in (RAW_COLS + ZSCORE_COLS) if c in cols_info["name"].values]
    
    if not all_cols:
        print("⚠️  No fundamental columns found in schema")
        con.close()
        return None
    
    query = f"""
        SELECT date, ticker, {', '.join(all_cols)}
        FROM features
        WHERE ticker IN ({','.join(f"'{t}'" for t in TICKERS)})
          AND date >= '{START}' AND date <= '{END}'
        ORDER BY ticker, date
    """
    
    df = con.execute(query).df()
    con.close()
    
    if df.empty:
        print("❌ No data returned")
        return False
    
    passed = True
    
    for col in all_cols:
        change_counts = {}
        
        for ticker in TICKERS:
            ticker_df = df[df["ticker"] == ticker].sort_values("date")
            n_changes, n_non_nan, _ = compute_change_rate(ticker_df[col].values)
            
            # Skip sparse data
            if n_non_nan < MIN_COVERAGE:
                continue
            
            if n_changes not in change_counts:
                change_counts[n_changes] = []
            change_counts[n_changes].append(ticker)
        
        if not change_counts:
            print(f"{col}: No tickers with sufficient coverage")
            continue
        
        print(f"{col}:")
        for count, tickers_list in sorted(change_counts.items()):
            print(f"  {count} changes: {', '.join(tickers_list)}")
        
        # Check if >3 tickers have identical counts
        max_identical = max(len(t) for t in change_counts.values())
        if max_identical > 3:
            print(f"  ⚠️  WARNING: {max_identical} tickers have identical change counts")
            passed = False
        
        print()
    
    if passed:
        print("✓ PASS: No suspicious identical patterns")
    else:
        print("⚠️  WARNING: Some identical patterns detected (may indicate bug)")
    
    return passed


def test_filing_counts():
    """Verify 10-K/10-Q filing counts are reasonable."""
    
    print()
    print("=" * 70)
    print("TEST 5: Filing Counts Sanity (10-K ~1/year, 10-Q ~3-4/year)")
    print("=" * 70)
    print()
    
    con = duckdb.connect(DB_PATH, read_only=True)
    
    # Count resets (drops) as proxy for filing events
    query = f"""
        SELECT date, ticker, days_since_10q, days_since_10k
        FROM features
        WHERE ticker IN ('AAPL', 'MSFT')
          AND date >= '{START}' AND date <= '{END}'
        ORDER BY ticker, date
    """
    df = con.execute(query).df()
    con.close()
    
    results = []
    
    for ticker in ["AAPL", "MSFT"]:
        tdf = df[df["ticker"] == ticker].sort_values("date")
        
       # Count 10-Q resets
        s_10q = tdf["days_since_10q"].dropna().astype(float)
        n_10q_drops = int((s_10q.diff() < 0).sum()) if len(s_10q) > 1 else 0
        
        # Count 10-K resets
        s_10k = tdf["days_since_10k"].dropna().astype(float)
        n_10k_drops = int((s_10k.diff() < 0).sum()) if len(s_10k) > 1 else 0
        
        results.append({
            "ticker": ticker,
            "10q_resets": n_10q_drops,
            "10k_resets": n_10k_drops,
            "expected_10q": "6-8 (2 years)",
            "expected_10k": "1-2 (2 years)",
        })
    
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    print()
    
    # Check sanity
    passed = True
    for _, row in results_df.iterrows():
        if row["10q_resets"] < 4:
            print(f"⚠️  {row['ticker']}: Only {row['10q_resets']} 10-Q resets (expected 6-8 for 2 years)")
            passed = False
        if row["10k_resets"] < 1:
            print(f"⚠️  {row['ticker']}: Only {row['10k_resets']} 10-K resets (expected 1-2 for 2 years)")
            passed = False
    
    if passed:
        print("✓ PASS: Filing counts look reasonable")
    else:
        print("❌ FAIL: Filing counts are off (check FMP data download)")
    
    return passed


def show_sample_filing_alignment():
    """Show sample data to verify changes align with filing dates."""
    
    print()
    print("=" * 70)
    print("INFO: Sample Filing Alignment (AAPL)")
    print("=" * 70)
    print()
    
    con = duckdb.connect(DB_PATH, read_only=True)
    
    # Check if raw columns exist
    cols_info = con.execute("PRAGMA table_info(features)").df()
    has_raw = "gross_margin_ttm" in cols_info["name"].values
    
    if has_raw:
        select_cols = "days_since_10q, days_since_10k, gross_margin_ttm, gross_margin_vs_sector"
    else:
        select_cols = "days_since_10q, days_since_10k, gross_margin_vs_sector"
    
    query = f"""
        SELECT date, {select_cols}
        FROM features
        WHERE ticker = 'AAPL'
          AND date >= '2023-01-01' AND date <= '2023-06-30'
          AND days_since_10q <= 10
        ORDER BY date
        LIMIT 30
    """
    
    df = con.execute(query).df()
    con.close()
    
    if df.empty:
        print("No data with days_since_10q <= 10")
        print("(This might indicate reset issue or sparse filings data)")
    else:
        print("Rows where days_since_10q <= 10 (near filing dates):")
        print(df.to_string(index=False))
    
    print()


def main():
    db_path = Path(DB_PATH)
    
    if not db_path.exists():
        print(f"❌ DuckDB not found at {DB_PATH}")
        print("Run: python scripts/build_features_duckdb.py --auto-normalize-splits")
        return 1
    
    print()
    print("=" * 70)
    print("BATCH 5 FUNDAMENTAL FEATURES VALIDATION")
    print("=" * 70)
    print()
    print(f"Database: {DB_PATH}")
    print(f"Tickers: {', '.join(TICKERS)}")
    print(f"Date range: {START} to {END}")
    print(f"Min coverage: {MIN_COVERAGE} non-null values per ticker/feature")
    print()
    
    results = []
    
    try:
        result = test_raw_ttm_piecewise_constant()
        if result is not None:
            results.append(("Raw TTM Piecewise", result))
        
        results.append(("days_since Drop Resets", test_days_since_resets_by_drops()))
        
        result = test_zscore_stepwise()
        if result is not None:
            results.append(("Z-Score Stepwise", result))
        
        result = test_no_identical_patterns()
        if result is not None:
            results.append(("No Identical Patterns", result))
        
        results.append(("Filing Counts", test_filing_counts()))
        
        show_sample_filing_alignment()
        
    except Exception as e:
        print(f"\n❌ Validation failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Summary
    print()
    print("=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    
    all_pass = True
    for name, passed in results:
        status = "✓ PASS" if passed else "❌ FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_pass = False
    
    print()
    
    if all_pass:
        print("✅ ALL VALIDATIONS PASSED")
        return 0
    else:
        print("❌ SOME VALIDATIONS FAILED - see details above")
        return 1


if __name__ == "__main__":
    exit(main())

# Batch 4 Regime/Macro Features - Validation Complete

**Date:** 2026-01-02
**Status:** ✅ PASSED ALL PIT SNIFF TESTS

## Summary

Added 12 regime/macro features to the features table. All features use backward-looking windows to prevent leakage.

## Features Added (12 total)

| Feature | Description | Coverage | Notes |
|---------|-------------|----------|-------|
| `vix_level` | Current VIX proxy level | 100% | VIXY ETF |
| `vix_percentile` | 252-day rolling percentile | 100% | 0-100 range |
| `vix_change_5d` | 5-day VIX change | 100% | Backward-looking |
| `vix_regime` | 0=low, 1=normal, 2=elevated, 3=extreme | 100% | Based on percentile |
| `market_return_5d` | 5-day SPY return | 100% | Backward-looking |
| `market_return_21d` | 21-day SPY return | 100% | Backward-looking |
| `market_return_63d` | 63-day SPY return | 100% | Backward-looking |
| `market_vol_21d` | 21-day annualized volatility | 100% | Backward-looking |
| `market_regime` | 1=bull, 0=neutral, -1=bear | 100% | Based on return+vol |
| `above_ma_50` | 1 if SPY > 50-day MA | 100% | Binary |
| `above_ma_200` | 1 if SPY > 200-day MA | 100% | Binary |
| `ma_50_200_cross` | 1=golden cross, -1=death cross | 100% | Binary |

## PIT Sniff Test Results

### Test 1: Per-date consistency ✅
All regime features have exactly 1 unique value per date (same for all tickers).

### Test 2: Market returns ✅
```
market_return_5d:  min=-0.180, max=0.174, mean=0.0026
market_return_21d: min=-0.331, max=0.252, mean=0.0107
market_return_63d: min=-0.305, max=0.393, mean=0.0299
```
No extreme values (|return| > 100%).

### Test 3: VIX features ✅
- `vix_level`: 40.2 - 31,616 (VIXY range is correct)
- `vix_percentile`: 0.4 - 100.0 (in valid 0-100 range)

### Test 4: Technical indicators ✅
- `above_ma_50`: [0, 1] only
- `above_ma_200`: [0, 1] only
- `ma_50_200_cross`: [-1, 1] only

### Test 5: Market regime distribution ✅
```
 1 (bull):    83,390 rows (41%)
 0 (neutral): 73,773 rows (37%)
-1 (bear):    44,144 rows (22%)
```

## Test Results

- **Unit Tests:** 429 passed ✅
- **Smoke Test:** Chapter 7 tabular_lgb runs end-to-end ✅
- **PIT Sniff Tests:** All passed ✅

## Risks Mitigated

1. **Calendar alignment:** ✅ SPY/VIXY on same trading calendar as equities
2. **Backward-looking windows:** ✅ All returns use `closes[i] / closes[i-N] - 1`
3. **Merge semantics:** ✅ Left join on date, verified 1 unique value per date

## DuckDB Schema Changes

- **Features table:** Added 12 regime columns
- **Regime table:** Expanded from 4 to 16 columns

## Deferred Items

**Sector Rotation Features (3 features):** `tech_vs_staples`, `tech_vs_utilities`, `risk_on_indicator`
- **Reason:** Requires sector ETF price data (XLK, XLP, XLU) not currently in the data pipeline
- **Impact:** Low - these are enhancement features, not core
- **Resolution:** Add when sector ETF data is wired into build_features_duckdb.py

## Next Steps

- **Option A:** Start Chapter 8 (Kronos) - regime features now available for diagnostics
- **Option B:** Continue to Batch 5 (Fundamentals) - FMP Premium available


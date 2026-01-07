# Feature Store Gap: Summary & Action Plan

**Date:** December 30, 2025  
**Status:** Documented, Ready for Chapter 8

---

## Executive Summary

**Finding:** Feature code is complete (50 features), but DuckDB pipeline only wires 7 features.

**Impact:** 
- ✅ Chapter 7 frozen baseline unaffected (uses 13 features, subset of 50)
- ⚠️ Chapter 8+ models need expanded feature store
- ✅ FMP Premium available for fundamentals

**Action:** Expand `scripts/build_features_duckdb.py` before starting Chapter 8.

---

## Current State

### What's Implemented (Code)

| Category | Features | Location | Status |
|----------|----------|----------|--------|
| Price/Volume | 14 | `src/features/price_features.py` | ✅ Code complete |
| Fundamentals | 7 | `src/features/fundamental_features.py` | ✅ Code complete |
| Events/Earnings | 12 | `src/features/event_features.py` | ✅ Code complete |
| Regime/Macro | 15 | `src/features/regime_features.py` | ✅ Code complete |
| Missingness | 2 | `src/features/missingness.py` | ✅ Code complete |
| **Total** | **50** | `src/features/` | ✅ **All implemented** |

### What's in DuckDB (Data)

| Feature | Type | In DuckDB |
|---------|------|-----------|
| `mom_1m`, `mom_3m`, `mom_6m`, `mom_12m` | Momentum | ✅ |
| `adv_20d` | Liquidity | ✅ |
| `vol_20d`, `vol_60d` | Volatility | ✅ |
| **All others** | Various | ❌ **Missing (43 features)** |

---

## What Needs to Happen

### Priority 1: Price/Volume (7 missing features) - REQUIRED

**Missing:**
- `vol_of_vol` (volatility of volatility)
- `max_drawdown_60d`, `dist_from_high_60d` (drawdown features)
- `rel_strength_1m`, `rel_strength_3m` (relative strength vs universe)
- `beta_252d` (beta vs QQQ)

**Why:** Core price signals, always available, no API dependencies

**Implementation:**
```python
# In scripts/build_features_duckdb.py, add:
from src.features.price_features import (
    compute_volatility_features,
    compute_drawdown_features,
    compute_relative_strength,
    compute_beta
)

# Call these functions and add columns to features_df
```

---

### Priority 2: Events/Earnings (12 features) - CRITICAL

**Missing:**
- `days_to_earnings`, `days_since_earnings` (earnings timing)
- `in_pead_window`, `pead_window_day` (post-earnings drift)
- `last_surprise_pct`, `avg_surprise_4q`, `surprise_streak`, `surprise_zscore`, `earnings_vol` (surprise history)
- `days_since_10k`, `days_since_10q`, `reports_bmo` (filing recency)

**Why:** Critical for earnings gap microstructure issue (monthly rebalancing + earnings on 2nd-5th)

**Implementation:**
```python
# In scripts/build_features_duckdb.py, add:
from src.features.event_features import (
    compute_earnings_features,
    compute_filing_features
)

# Requires AlphaVantage API for earnings calendar
# Requires EventStore for earnings surprises
```

---

### Priority 3: Regime/Macro (15 features) - Optional for Ch8

**Missing:**
- VIX features: `vix_level`, `vix_percentile`, `vix_change_5d`, `vix_regime`
- Market features: `market_return_5d/21d/63d`, `market_vol_21d`, `market_regime`
- Technical: `above_ma_50`, `above_ma_200`, `ma_50_200_cross`
- Sector rotation: `tech_vs_staples`, `tech_vs_utilities`, `risk_on_indicator`

**Why:** Useful for regime-aware ensembling (Chapter 12), not critical for Kronos

**Implementation:**
```python
# In scripts/build_features_duckdb.py, add:
from src.features.regime_features import (
    compute_vix_features,
    compute_market_regime,
    compute_sector_rotation
)

# Requires FMP API for VIX and sector ETF data
```

---

### Priority 4: Fundamentals (7 features) - Optional

**Missing:**
- `pe_zscore_3y`, `pe_vs_sector`, `ps_vs_sector`
- `gross_margin_vs_sector`, `revenue_growth_vs_sector`
- `roe_zscore`, `roa_zscore`

**Why:** Value signal, you have FMP Premium so this is possible

**Implementation:**
```python
# In scripts/build_features_duckdb.py, add:
from src.features.fundamental_features import (
    compute_valuation_features,
    compute_quality_features
)

# Requires FMP Premium for income statement, balance sheet, cash flow
```

---

### Priority 5: Missingness (2 features) - Diagnostic

**Missing:**
- `coverage_pct` (overall feature coverage 0-1)
- `is_new_stock` (< 1 year of history)

**Why:** Diagnostic signal, not predictive

**Implementation:**
```python
# In scripts/build_features_duckdb.py, add:
from src.features.missingness import compute_coverage_features

# Computed from existing features
```

---

## Why Chapter 7 Baseline is Unaffected

| Component | Frozen? | Location |
|-----------|---------|----------|
| **Baseline metrics** | ✅ Yes | `evaluation_outputs/chapter7_tabular_lgb_full/` |
| **Baseline code** | ✅ Yes | `DEFAULT_TABULAR_FEATURES` (13 features) |
| **DuckDB data** | ❌ No | `data/features.duckdb` (gitignored) |

**Key insight:** The frozen baseline only reads the 13 features it needs. Expanding DuckDB to 50 features doesn't change what the baseline reads.

---

## Implementation Plan

### Step 1: Expand Build Script (Pre-Chapter-8)

**File:** `scripts/build_features_duckdb.py`

**Changes:**
1. Import all feature generators from `src/features/`
2. Call each generator with appropriate inputs
3. Merge results into `features_df`
4. Update DuckDB schema to include all 50 columns

**Estimated Time:** 2-4 hours (mostly testing)

### Step 2: Rebuild DuckDB

```bash
# Backup existing (optional)
mv data/features.duckdb data/features.duckdb.backup

# Rebuild with expanded features
python scripts/build_features_duckdb.py --auto-normalize-splits

# Verify
python -c "
import duckdb
conn = duckdb.connect('data/features.duckdb', read_only=True)
result = conn.execute('DESCRIBE features').fetchall()
print(f'Total columns: {len(result)}')
print('Columns:', [r[0] for r in result])
"
```

### Step 3: Verify Frozen Baseline Still Works

```bash
# Load frozen baseline and verify it still reads correctly
python -c "
from src.evaluation.data_loader import load_features_for_evaluation
from src.evaluation import DEFAULT_TABULAR_FEATURES
from datetime import date

features_df, _ = load_features_for_evaluation(
    source='duckdb',
    db_path='data/features.duckdb',
    eval_start=date(2016, 1, 1),
    eval_end=date(2025, 6, 30),
    horizons=[20, 60, 90],
)

# Check that frozen baseline features are present
for feat in DEFAULT_TABULAR_FEATURES:
    assert feat in features_df.columns, f'Missing: {feat}'

print('✓ Frozen baseline features present')
print(f'Total columns available: {len(features_df.columns)}')
"
```

---

## Documentation Updates

| File | Section | Status |
|------|---------|--------|
| `ROADMAP.md` | Pre-Chapter-8 Feature Expansion | ✅ Added |
| `PROJECT_DOCUMENTATION.md` | Chapter 5 Feature Store Gap | ✅ Added |
| `AI_Stock_Forecaster_*.ipynb` | Chapter 8.0 Prerequisites | ✅ Added |
| `FEATURE_STORE_GAP_SUMMARY.md` | This file | ✅ Created |

---

## Key Takeaways

1. **No urgency:** Chapter 7 is frozen and complete. Feature expansion is a pre-Chapter-8 task.

2. **Clean separation:** Frozen baseline (13 features) and extended features (50 features) are separate code paths.

3. **FMP Premium available:** You can add fundamentals without API quota issues.

4. **Priority order matters:** Price → Events → Regime → Fundamentals → Missingness

5. **Earnings gap issue:** Events/earnings features (Priority 2) are critical for handling the microstructure issue you identified.

---

## Next Steps

**When you're ready to start Chapter 8:**

1. Expand `scripts/build_features_duckdb.py` (Priority 1-2 at minimum)
2. Rebuild DuckDB with `--auto-normalize-splits`
3. Verify frozen baseline still works
4. Start Kronos integration with full feature set available

**Estimated total time:** 1 day (mostly feature wiring + testing)


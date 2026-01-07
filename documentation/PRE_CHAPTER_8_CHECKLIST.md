# Pre-Chapter-8 Checklist: Feature Store Expansion

**Created:** December 30, 2025  
**Updated:** January 7, 2026  
**Status:** ✅ COMPLETE

---

## Executive Summary

**Final State (January 7, 2026):**
- ✅ Feature code: 52 features implemented and wired into DuckDB
- ✅ Chapter 7 baseline: Frozen using 13 features
- ✅ DuckDB pipeline: All features wired into `scripts/build_features_duckdb.py`
- ✅ Chapter 6 REAL closure: Baseline floor frozen

**COMPLETE:** All features wired into DuckDB. Ready for Chapter 8.

**Impact:** Chapter 8+ models (Kronos, FinText, Fusion) need extended features for:
- Earnings gap mitigation (event features)
- Regime-aware ensembling (regime features)
- Tabular context fusion (fundamentals)

---

## What's in DuckDB Now (7 features)

```python
CURRENT_DUCKDB_FEATURES = [
    "mom_1m", "mom_3m", "mom_6m", "mom_12m",  # Momentum
    "adv_20d",                                  # Liquidity
    "vol_20d", "vol_60d",                       # Volatility
]
```

**Source:** `scripts/build_features_duckdb.py` lines 554-567

---

## What Needs to Be Added (43 features)

### Priority 1: Price/Volume (7 missing)

**Already in code:** `src/features/price_features.py`

| Feature | Purpose | Difficulty |
|---------|---------|------------|
| `vol_of_vol` | Volatility regime detection | Easy |
| `max_drawdown_60d` | Drawdown risk | Easy |
| `dist_from_high_60d` | Distance from high | Easy |
| `rel_strength_1m` | Cross-sectional z-score vs universe | Medium |
| `rel_strength_3m` | Cross-sectional z-score vs universe | Medium |
| `beta_252d` | Rolling beta vs QQQ | Medium |
| `adv_60d` | 60-day average dollar volume | Easy |

**Wiring:** Add to `compute_features()` in build script, update DuckDB schema

---

### Priority 2: Events/Earnings (12 missing) ⚠️ CRITICAL

**Already in code:** `src/features/event_features.py`

| Feature | Purpose | Difficulty |
|---------|---------|------------|
| `days_to_earnings` | Earnings proximity (gap mitigation) | Medium |
| `days_since_earnings` | PEAD window tracking | Medium |
| `in_pead_window` | Boolean: in 63-day drift period | Easy |
| `pead_window_day` | Which day of PEAD (1-63) | Easy |
| `last_surprise_pct` | Most recent surprise | Medium |
| `avg_surprise_4q` | 4Q rolling average | Easy |
| `surprise_streak` | Consecutive beats/misses | Easy |
| `surprise_zscore` | Cross-sectional surprise | Easy |
| `earnings_vol` | Surprise volatility (8Q) | Easy |
| `days_since_10k` | Annual filing recency | Easy |
| `days_since_10q` | Quarterly filing recency | Easy |
| `reports_bmo` | BMO vs AMC timing | Easy |

**Why critical:** Monthly rebalancing + early-month earnings = gap moves within horizon. These features let models avoid/predict earnings gaps.

**API Requirements:**
- AlphaVantage (earnings calendar) - you have this ✅
- FMP (earnings history) - you have Premium ✅

---

### Priority 3: Regime Features (15 missing)

**Already in code:** `src/features/regime_features.py`

| Feature | Purpose | Difficulty |
|---------|---------|------------|
| `vix_level` | Raw VIX | Easy |
| `vix_percentile` | 2-year VIX percentile | Easy |
| `vix_change_5d` | 5-day VIX change | Easy |
| `vix_regime` | low/normal/elevated/high | Easy |
| `market_return_5d/21d/63d` | SPY returns | Easy |
| `market_vol_21d` | Market volatility | Easy |
| `market_regime` | bull/bear/neutral | Easy |
| `above_ma_50/200` | Price vs MA | Easy |
| `ma_50_200_cross` | Golden/death cross | Easy |
| `tech_vs_staples` | XLK vs XLP | Medium |
| `tech_vs_utilities` | XLK vs XLU | Medium |
| `risk_on_indicator` | Composite risk-on/off | Medium |

**Needed for:** Chapter 12 (Regime-Aware Ensembling)

**API Requirements:**
- FMP index data (SPY, VIX, XLK, XLP, XLU) - you have Premium ✅

---

### Priority 4: Fundamentals (7 missing)

**Already in code:** `src/features/fundamental_features.py`

| Feature | Purpose | Difficulty |
|---------|---------|------------|
| `pe_zscore_3y` | P/E vs own history | Medium |
| `pe_vs_sector` | P/E vs sector median | Medium |
| `ps_vs_sector` | P/S vs sector median | Medium |
| `gross_margin_vs_sector` | Margins vs sector | Medium |
| `revenue_growth_vs_sector` | Growth vs sector | Medium |
| `roe_zscore` | Quality z-score | Easy |
| `roa_zscore` | Quality z-score | Easy |

**Needed for:** Chapter 11 (Fusion with tabular context)

**API Requirements:**
- FMP Premium (quarterly fundamentals) - you have this ✅

---

### Priority 5: Missingness (2 missing)

**Already in code:** `src/features/missingness.py`

| Feature | Purpose | Difficulty |
|---------|---------|------------|
| `coverage_pct` | Overall data availability | Easy |
| `is_new_stock` | < 1 year of history | Easy |

**Purpose:** Diagnostic, not predictive

---

## Implementation Plan

### Step 1: Update DuckDB Schema

**File:** `scripts/build_features_duckdb.py` (lines 554-567)

**Current schema:**
```python
CREATE TABLE features (
    date DATE,
    ticker VARCHAR,
    stable_id VARCHAR,
    mom_1m DOUBLE,
    mom_3m DOUBLE,
    mom_6m DOUBLE,
    mom_12m DOUBLE,
    adv_20d DOUBLE,
    vol_20d DOUBLE,
    vol_60d DOUBLE,
    PRIMARY KEY (date, ticker)
)
```

**Expanded schema (50 columns):**
```python
CREATE TABLE features (
    date DATE,
    ticker VARCHAR,
    stable_id VARCHAR,
    
    -- Momentum (4)
    mom_1m DOUBLE, mom_3m DOUBLE, mom_6m DOUBLE, mom_12m DOUBLE,
    
    -- Volatility (3)
    vol_20d DOUBLE, vol_60d DOUBLE, vol_of_vol DOUBLE,
    
    -- Drawdown (2)
    max_drawdown_60d DOUBLE, dist_from_high_60d DOUBLE,
    
    -- Liquidity (2)
    adv_20d DOUBLE, adv_60d DOUBLE,
    
    -- Relative Strength (2)
    rel_strength_1m DOUBLE, rel_strength_3m DOUBLE,
    
    -- Beta (1)
    beta_252d DOUBLE,
    
    -- Fundamentals (7)
    pe_zscore_3y DOUBLE, pe_vs_sector DOUBLE, ps_vs_sector DOUBLE,
    gross_margin_vs_sector DOUBLE, revenue_growth_vs_sector DOUBLE,
    roe_zscore DOUBLE, roa_zscore DOUBLE,
    
    -- Events/Earnings (12)
    days_to_earnings DOUBLE, days_since_earnings DOUBLE,
    in_pead_window BOOLEAN, pead_window_day INTEGER,
    last_surprise_pct DOUBLE, avg_surprise_4q DOUBLE,
    surprise_streak INTEGER, surprise_zscore DOUBLE, earnings_vol DOUBLE,
    days_since_10k DOUBLE, days_since_10q DOUBLE, reports_bmo BOOLEAN,
    
    -- Regime (15)
    vix_level DOUBLE, vix_percentile DOUBLE, vix_change_5d DOUBLE, vix_regime VARCHAR,
    market_return_5d DOUBLE, market_return_21d DOUBLE, market_return_63d DOUBLE,
    market_vol_21d DOUBLE, market_regime VARCHAR,
    above_ma_50 BOOLEAN, above_ma_200 BOOLEAN, ma_50_200_cross DOUBLE,
    tech_vs_staples DOUBLE, tech_vs_utilities DOUBLE, risk_on_indicator DOUBLE,
    
    -- Missingness (2)
    coverage_pct DOUBLE, is_new_stock BOOLEAN,
    
    PRIMARY KEY (date, ticker)
)
```

---

### Step 2: Wire Feature Generators

**File:** `scripts/build_features_duckdb.py`

**Current:** Only calls momentum + volume generators

**Needed:** Call all feature generators from `src/features/`:

```python
from src.features.price_features import (
    compute_momentum_features,
    compute_volatility_features,
    compute_drawdown_features,
    compute_relative_strength_features,
    compute_beta_features,
    compute_liquidity_features,
)
from src.features.fundamental_features import compute_fundamental_features
from src.features.event_features import compute_event_features
from src.features.regime_features import compute_regime_features
from src.features.missingness import compute_missingness_features

def compute_all_features(prices_df, fundamentals_df, events_df, regime_df):
    """Compute all 50 features."""
    features = []
    
    # Price/Volume (14 features)
    features.append(compute_momentum_features(prices_df))
    features.append(compute_volatility_features(prices_df))
    features.append(compute_drawdown_features(prices_df))
    features.append(compute_liquidity_features(prices_df))
    features.append(compute_relative_strength_features(prices_df))
    features.append(compute_beta_features(prices_df, benchmark_df))
    
    # Fundamentals (7 features) - requires FMP Premium
    features.append(compute_fundamental_features(fundamentals_df, prices_df))
    
    # Events (12 features)
    features.append(compute_event_features(events_df, prices_df))
    
    # Regime (15 features)
    features.append(compute_regime_features(regime_df, prices_df))
    
    # Missingness (2 features)
    features.append(compute_missingness_features(features))
    
    # Merge all
    return merge_features(features)
```

---

### Step 3: Rebuild DuckDB

```bash
# Backup current DuckDB (optional)
cp data/features.duckdb data/features_7feat_backup.duckdb

# Rebuild with extended features
python scripts/build_features_duckdb.py \
  --start-date 2014-01-01 \
  --end-date 2025-06-30 \
  --auto-normalize-splits

# Verify schema
python -c "
import duckdb
conn = duckdb.connect('data/features.duckdb', read_only=True)
result = conn.execute('DESCRIBE features').fetchall()
print(f'Features table has {len(result)} columns')
for col_name, col_type, *_ in result:
    print(f'  {col_name}: {col_type}')
conn.close()
"
```

---

### Step 4: Verify Backward Compatibility

**Critical:** Chapter 7 frozen baseline must still work with expanded DuckDB.

```bash
# Run Chapter 7 baseline with expanded DuckDB
python scripts/run_chapter7_tabular_lgb.py --smoke

# Should complete without errors
# Baseline uses only its frozen 13 features (subset of 50)
```

---

## Acceptance Criteria

| Criterion | Threshold | How to Check |
|-----------|-----------|--------------|
| All features in DuckDB | 50 columns | `DESCRIBE features` |
| Non-null coverage | > 90% per feature | Check `DATA_MANIFEST.json` |
| Chapter 7 baseline works | Smoke test passes | Run `--smoke` |
| Build time | < 30 min | Time the rebuild |
| DuckDB size | < 500 MB | Check file size |

---

## API Quota Estimates

| Feature Category | API Calls | Quota Used |
|------------------|-----------|------------|
| Price/Volume | 0 | Already cached |
| Fundamentals | ~100 tickers × 4 quarters × 10 years = 4,000 | FMP Premium: 100K/day ✅ |
| Events/Earnings | ~100 tickers × 10 years = 1,000 | AlphaVantage: 500/day ✅ |
| Regime | ~10 indices × 10 years = 100 | FMP Premium: 100K/day ✅ |
| **Total** | **~5,100 calls** | **Well within quota** |

---

## Why This Matters for Chapter 8+

### Chapter 8: Kronos
- **Needs:** OHLCV only (no tabular features required)
- **But:** Extended features enable better diagnostics and ablation studies

### Chapter 9: FinText
- **Needs:** Historical returns only
- **But:** Extended features enable context-aware fusion

### Chapter 10: Context Features (Tabular)
- **Needs:** ALL 50 features (this is the tabular model chapter)
- **Critical:** Can't proceed without expanded feature store

### Chapter 11: Fusion
- **Needs:** ALL 50 features for tabular context branch
- **Critical:** Fusion combines Kronos + FinText + tabular context

### Chapter 12: Regime-Aware Ensembling
- **Needs:** Regime features (15) for regime detection
- **Critical:** Can't implement regime-aware weighting without regime features

---

## Recommended Approach

### Option A: Expand Now (Recommended)

**Pros:**
- One-time effort
- Ready for all future chapters
- Enables earnings gap diagnostics immediately

**Cons:**
- ~30 min rebuild time
- Uses API quota (but well within limits)

**Command:**
```bash
python scripts/build_features_duckdb.py --auto-normalize-splits
```

### Option B: Incremental Expansion

**Phase 1 (before Kronos):**
- Add price/volume features (7) - enables better diagnostics

**Phase 2 (before FinText):**
- Add event features (12) - enables earnings gap mitigation

**Phase 3 (before Fusion):**
- Add regime + fundamentals (22) - enables full fusion

**Pros:**
- Spreads out API usage
- Can test incrementally

**Cons:**
- Multiple rebuilds (more time)
- Risk of forgetting features

---

## Files to Modify

1. **`scripts/build_features_duckdb.py`**:
   - Update schema (lines 554-567)
   - Wire feature generators (add calls to `src/features/*`)
   - Update progress logging

2. **`src/features/` (no changes needed)**:
   - All feature generators already exist
   - All tests passing (84/84)

3. **`src/evaluation/baselines.py` (no changes needed)**:
   - `DEFAULT_TABULAR_FEATURES` stays frozen (13 features)
   - `EXTENDED_TABULAR_FEATURES` already added (50 features)

---

## Verification Checklist

After expanding DuckDB:

- [ ] Run `python -c "import duckdb; conn = duckdb.connect('data/features.duckdb'); print(len(conn.execute('DESCRIBE features').fetchall()))"`
  - Should print: `50`

- [ ] Run `python scripts/run_chapter7_tabular_lgb.py --smoke`
  - Should complete without errors
  - Baseline uses only its 13 features

- [ ] Check `data/DATA_MANIFEST.json`
  - Should show coverage stats for all 50 features
  - Non-null coverage should be > 90% for most features

- [ ] Run full test suite: `pytest tests/ -q`
  - Should still show 429/429 passing

---

## Next Steps

1. **Expand feature store** (this checklist)
2. **Run earnings gap diagnostic** (see `EARNINGS_GAP_ANALYSIS.md`)
3. **Start Chapter 8** (Kronos integration)

---

**Status:** ✅ COMPLETE (January 7, 2026)  
**Result:** DuckDB expanded to 52 columns, 201K rows  
**Verification:** All tests passing (429/429), Chapter 6 baseline floor frozen


# Feature Wiring TODO Checklist

**Status:** 🔄 IN PROGRESS  
**Current:** 7 features → **Target:** 50 features  
**Approach:** Incremental batches (test after each batch)

---

## Batch 1: Price/Volume Extensions (7 new) ⏳ NEXT
**Difficulty:** EASY (no new data sources)  
**Time:** 20-30 min

### Features to Add:
- [ ] `vol_of_vol` - Volatility of volatility
- [ ] `max_drawdown_60d` - Maximum drawdown over 60d
- [ ] `dist_from_high_60d` - Distance from 60d high
- [ ] `adv_60d` - 60-day ADV  
- [ ] `rel_strength_1m` - 1m return z-score vs universe
- [ ] `rel_strength_3m` - 3m return z-score vs universe
- [ ] `beta_252d` - Beta vs QQQ

### Implementation:
**File:** `scripts/build_features_duckdb.py`
**Location:** Inside `compute_momentum_features()` function (line ~294-337)

**Add after vol_60d computation (line ~321):**
```python
# Vol of vol (volatility of volatility)
vol_of_vol = None
if i >= VOL_WINDOW_60D:
    # Compute rolling 20d vol over 60d window
    vol_series = []
    for j in range(i - VOL_WINDOW_60D + VOL_WINDOW_20D, i + 1, VOL_WINDOW_20D):
        if j >= VOL_WINDOW_20D:
            rets = np.diff(np.log(closes[j-VOL_WINDOW_20D:j+1]))
            vol_series.append(np.std(rets) * np.sqrt(252))
    if len(vol_series) >= 2:
        vol_of_vol = float(np.std(vol_series))

# Max drawdown (60d)
max_drawdown_60d = None
if i >= VOL_WINDOW_60D:
    window_prices = closes[i-VOL_WINDOW_60D:i+1]
    running_max = np.maximum.accumulate(window_prices)
    drawdowns = (window_prices - running_max) / running_max
    max_drawdown_60d = float(np.min(drawdowns))

# Distance from high (60d)
dist_from_high_60d = None
if i >= VOL_WINDOW_60D:
    window_high = np.max(closes[i-VOL_WINDOW_60D:i+1])
    dist_from_high_60d = float((closes[i] - window_high) / window_high)

# ADV 60d
adv_60d = None
if volumes is not None and i >= VOL_WINDOW_60D:
    dollar_volumes = closes[i-VOL_WINDOW_60D:i] * volumes[i-VOL_WINDOW_60D:i]
    adv_60d = float(np.mean(dollar_volumes))
```

**Note:** `rel_strength_*` and `beta_252d` require cross-sectional data, so we'll add these in a second pass after the main loop.

### Schema Update:
**File:** `scripts/build_features_duckdb.py`
**Location:** Line ~554-567 (CREATE TABLE features)

**Add to schema:**
```sql
vol_of_vol DOUBLE,
max_drawdown_60d DOUBLE,
dist_from_high_60d DOUBLE,
adv_60d DOUBLE,
rel_strength_1m DOUBLE,
rel_strength_3m DOUBLE,
beta_252d DOUBLE,
```

### Test Command:
```bash
python scripts/build_features_duckdb.py --auto-normalize-splits
python - <<'PY'
import duckdb
conn = duckdb.connect("data/features.duckdb", read_only=True)
cols = conn.execute("DESCRIBE features").fetchall()
print(f"Columns: {len(cols)}")
for c in cols:
    print(f"  {c[0]}")
conn.close()
PY
```

Expected: 17 columns (3 meta + 7 old + 7 new)

---

## Batch 2: Missingness Features (2 new) ⏳ AFTER BATCH 1
**Difficulty:** EASY  
**Time:** 10 min

### Features to Add:
- [ ] `coverage_pct` - % of features non-null
- [ ] `is_new_stock` - Boolean: <252 trading days

### Implementation:
Add after all features computed, before returning DataFrame:

```python
# Compute missingness features
def add_missingness_features(features_df):
    # Count non-null features (exclude meta columns)
    feature_cols = [c for c in features_df.columns if c not in ['date', 'ticker', 'stable_id', 'coverage_pct', 'is_new_stock']]
    features_df['coverage_pct'] = features_df[feature_cols].notna().sum(axis=1) / len(feature_cols)
    
    # New stock indicator (has < 252 rows)
    stock_lengths = features_df.groupby('ticker').size()
    new_stocks = stock_lengths[stock_lengths < 252].index
    features_df['is_new_stock'] = features_df['ticker'].isin(new_stocks)
    
    return features_df

features_df = add_missingness_features(features_df)
```

### Schema Update:
```sql
coverage_pct DOUBLE,
is_new_stock BOOLEAN,
```

### Test:
Same as Batch 1, expect 19 columns.

---

## Batch 3: Event/Earnings Features (12 new) ⚠️ REQUIRES EVENT STORE
**Difficulty:** HARD (requires new data pipeline)  
**Time:** 60-90 min

### Prerequisites:
Need to build EventStore with:
- Earnings calendar (AlphaVantage)
- Earnings history (FMP)  
- SEC filings (FMP or sec-api)

### Features:
- [ ] `days_to_earnings`
- [ ] `days_since_earnings`
- [ ] `in_pead_window`
- [ ] `pead_window_day`
- [ ] `last_surprise_pct`
- [ ] `avg_surprise_4q`
- [ ] `surprise_streak`
- [ ] `surprise_zscore`
- [ ] `earnings_vol`
- [ ] `days_since_10k`
- [ ] `days_since_10q`
- [ ] `reports_bmo`

### Implementation Strategy:
Option A: Add to build script
Option B: Create separate `scripts/build_event_store.py` and merge

**Recommended:** Option B (cleaner separation)

### Steps:
1. Create `scripts/build_event_store.py`
2. Download earnings calendar + history from APIs
3. Compute event features
4. Save to `data/event_features.parquet`
5. Merge into build script via left join

---

## Batch 4: Regime Features (15 new) ⏳ REQUIRES INDEX DATA
**Difficulty:** MEDIUM (requires downloading indexes)  
**Time:** 30-45 min

### Prerequisites:
Download from FMP:
- SPY (already have for regime_df)
- VIX or ^VIX
- XLK (tech)
- XLP (staples)
- XLU (utilities)

### Features:
- [ ] `vix_level`
- [ ] `vix_percentile`
- [ ] `vix_change_5d`
- [ ] `vix_regime`
- [ ] `market_return_5d`
- [ ] `market_return_21d`
- [ ] `market_return_63d`
- [ ] `market_vol_21d`
- [ ] `market_regime`
- [ ] `above_ma_50`
- [ ] `above_ma_200`
- [ ] `ma_50_200_cross`
- [ ] `tech_vs_staples`
- [ ] `tech_vs_utilities`
- [ ] `risk_on_indicator`

### Implementation:
Expand `compute_regime_features()` function.

Most of these can be computed from SPY + VIX + sector ETFs.

---

## Batch 5: Fundamental Features (7 new) ⚠️ REQUIRES QUARTERLY DATA
**Difficulty:** HARD (requires new data source)  
**Time:** 45-60 min

### Prerequisites:
Need to download from FMP:
- Quarterly income statements
- Quarterly balance sheets
- Quarterly cash flow

### Features:
- [ ] `pe_zscore_3y`
- [ ] `pe_vs_sector`
- [ ] `ps_vs_sector`
- [ ] `gross_margin_vs_sector`
- [ ] `operating_margin_vs_sector`
- [ ] `revenue_growth_vs_sector`
- [ ] `roe_zscore`

### Implementation:
Similar to events: separate script or module.

---

## Progress Tracking

### Current Status:
- [x] Batch 0: Original 7 features (DONE)
- [ ] Batch 1: Price/volume extensions (7 new) → 14 total
- [ ] Batch 2: Missingness (2 new) → 16 total
- [ ] Batch 3: Events (12 new) → 28 total ⚠️ HARD
- [ ] Batch 4: Regime (15 new) → 43 total
- [ ] Batch 5: Fundamentals (7 new) → 50 total ⚠️ HARD

### Recommended Order:
1. **Start:** Batch 1 (easy wins, no new data)
2. **Next:** Batch 2 (trivial, works with any feature set)
3. **Parallel:** Batch 4 (medium, unlocks regime diagnostics)
4. **Later:** Batch 3 + 5 (hard, but critical for Chapter 8+)

### Fallback Strategy:
If Batch 3 or 5 is too complex:
- Document as "TODO: requires event/fundamental store"
- Proceed with Batches 1 + 2 + 4 (34 features total)
- Defer events + fundamentals to dedicated feature expansion sprint

---

## Next Action

**Immediate:** Execute Batch 1 (price/volume extensions)
- Low risk (same data source)
- Big value (7 features, 30 min work)
- Tests existing infrastructure

**Command to start:**
```bash
# Open build script
code scripts/build_features_duckdb.py
# Follow Batch 1 instructions above
```

---

**Ready to start Batch 1?**


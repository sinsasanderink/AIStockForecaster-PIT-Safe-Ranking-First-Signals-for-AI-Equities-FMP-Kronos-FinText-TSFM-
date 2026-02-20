# Batch 5: Bug Fixes (January 6, 2026)

## Summary

**Five** bugs were identified and fixed in the Batch 5 fundamental features implementation:

| Bug | Impact | Fix |
|-----|--------|-----|
| `days_since_10q/10k` never resets | Min was 3, not 0 | Changed `<` to `<=` in filing date comparison |
| 10-K vs 10-Q misclassification | Used `period_end.month` heuristic | Download both quarterly + annual income statements separately |
| Raw TTM metrics not persisted | Can't verify piecewise-constant | Added 4 new columns to DuckDB schema |
| Z-scores recomputed daily | ~50% change rate (should be <5%) | Per-ticker stepwise with 90-day lookback |
| Z-score index mismatch | Fragile positional indexing | Key z-scores by filing_date, not list index |

---

## Bug 1: `days_since_10q/10k` Never Resets

### Evidence
```
ticker  min_10q  max_10q  reset_count_10q
AAPL      3.0    189.0            0
```
- Minimum value was 3, not 0
- `count <= 1` was 0 (never reset)

### Root Cause
```python
# OLD (BUGGY):
past_filings = ticker_filings[ticker_filings["filing_date"] < asof_date]

# Filing on 2023-01-15, asof_date = 2023-01-15:
# filing_date < asof_date → False, not included!
# So days_since starts at 1 minimum (day after filing)
```

### Fix
```python
# NEW (CORRECT):
past_filings = ticker_filings[ticker_filings["filing_date"] <= asof_date]

# Filing on 2023-01-15, asof_date = 2023-01-15:
# filing_date <= asof_date → True, included!
# So days_since = 0 on filing day
```

### File Changed
`scripts/build_features_duckdb.py`, line ~724

---

## Bug 2: 10-K vs 10-Q Misclassification

### Evidence
```python
# OLD (BUGGY):
all_filings.append({
    ...
    "filing_type": "10-Q",  # HARDCODED for ALL filings!
})

# Later, tried to infer from period_end.month:
period_month = last_filing["period_end"].month
is_annual = period_month in [12, 1, 2]  # WRONG! Apple ends in Sep, MSFT in Jun
```

### Root Cause
- All filings were hardcoded as "10-Q"
- Tried to infer 10-K from `period_end.month in [12, 1, 2]`
- This is wrong because many companies have non-December fiscal year ends:
  - Apple: September
  - Microsoft: June
  - Walmart: January

### Fix
Download both quarterly and annual income statements separately:

```python
# Get QUARTERLY income statements (10-Q filings)
quarterly_data = client._request("income-statement", 
                                 {"symbol": ticker, "period": "quarter", "limit": 60})
for stmt in quarterly_data:
    all_filings.append({..., "filing_type": "10-Q"})

# Get ANNUAL income statements (10-K filings)
annual_data = client._request("income-statement", 
                              {"symbol": ticker, "period": "annual", "limit": 20})
for stmt in annual_data:
    all_filings.append({..., "filing_type": "10-K"})
```

Then use actual `filing_type` field:
```python
quarterly_filings = past_filings[past_filings["filing_type"] == "10-Q"]
annual_filings = past_filings[past_filings["filing_type"] == "10-K"]
```

### File Changed
`scripts/build_features_duckdb.py`:
- Lines ~255-325: `download_earnings_data()` - download both quarterly and annual
- Lines ~738-760: `compute_event_features()` - use actual filing_type

---

## Bug 3: Raw TTM Metrics Not Persisted

### Evidence
```sql
PRAGMA table_info(features);
-- Only z-score columns existed:
-- gross_margin_vs_sector, operating_margin_vs_sector, etc.
-- No raw TTM columns
```

### Impact
- Couldn't verify that raw TTM is piecewise-constant between filings
- Debugging was impossible without raw values

### Fix
Added 4 new columns to DuckDB schema:

| Column | Description |
|--------|-------------|
| `gross_margin_ttm` | TTM gross profit / TTM revenue |
| `operating_margin_ttm` | TTM operating income / TTM revenue |
| `revenue_growth_yoy` | (TTM revenue / TTM revenue 4Q ago) - 1 |
| `roe_raw` | TTM net income / avg equity (4Q) |

### File Changed
`scripts/build_features_duckdb.py`:
- Lines ~1654-1668: CREATE TABLE schema
- Lines ~1684-1691: features_cols list

---

## Bug 3: Z-Scores Recomputed Daily (50% Change Rate)

### Evidence
```
ticker  feature                    n_changes  change_rate
AAPL    gross_margin_vs_sector         264     0.525896
MSFT    gross_margin_vs_sector         264     0.525896
NVDA    gross_margin_vs_sector         264     0.525896
AI      gross_margin_vs_sector         264     0.525896
```
- AAPL/MSFT/NVDA/AI all had **identical** 264 changes out of 502 days
- Change rate ~52% (should be ~2-5% for quarterly filings)
- Identical counts strongly suggest cross-sectional bug

### Root Cause
```python
# OLD (BUGGY): Compute z-scores for EVERY date
for date_val in features_df["date"].unique():  # 2,690 dates!
    date_data = raw_df[raw_df["date"] == date_val]
    
    for sector in sectors:
        sector_data = date_data[sector_data["sector"] == sector]
        mean, std = sector_data["gross_margin"].mean(), .std()
        
        # ALL tickers in sector get updated on this date
        for ticker in sector_tickers:
            features_df[...] = (value - mean) / std

# Problem: When ANY tech ticker files, ALL tech tickers' z-scores change
# Because the sector mean/std changes daily as new filings come in
```

### Fix
```python
# NEW (CORRECT): Compute z-scores at FILING DATES, forward-fill

# Step 1: Compute raw TTM at each ticker's filing dates only
for ticker in tickers:
    for filing_date in ticker_filing_dates:
        metrics = compute_ttm(ticker, filing_date)
        ticker_filings_data[ticker].append(metrics)

# Step 2: At each ticker's filing, compute z-score using 90-day lookback
for ticker, filings_list in ticker_filings_data.items():
    for metrics in filings_list:
        filing_date = metrics["filing_date"]
        
        # Get cross-section: other tickers' most recent filing within 90 days
        sector_cross_section = []
        for other_ticker in sector_tickers:
            recent = get_most_recent_filing(other_ticker, filing_date, lookback=90)
            sector_cross_section.append(recent)
        
        # Compute z-score using this stable cross-section
        mean, std = sector_cross_section.mean(), .std()
        zscore = (metrics["gross_margin"] - mean) / std
        ticker_zscore_data[ticker].append((filing_date, zscore))

# Step 3: Forward-fill into features_df
for ticker in tickers:
    for row_date in ticker_dates:
        # Find most recent filing <= row_date
        recent_filing = get_most_recent_filing(ticker, row_date)
        features_df[row_date] = recent_filing["zscore"]
```

### Key Changes
1. **Per-ticker stepwise**: Z-scores only change when THAT ticker files, not when other tickers file
2. **90-day lookback**: Cross-section uses other tickers' most recent filings within 90 days (not that exact date's cross-section)
3. **Forward-fill**: Z-scores constant between filings, matching expected quarterly behavior

### Expected Results After Fix

| Metric | Old (Buggy) | New (Fixed) |
|--------|-------------|-------------|
| Raw TTM change rate | N/A (not stored) | ~2-4% |
| Z-score change rate | ~50% | ~5-20% |
| Identical change counts | 4 tickers same | All different |
| `days_since_10q` min | 3 | 0 |

### File Changed
`scripts/build_features_duckdb.py`:
- Lines 445-646: Completely rewrote `compute_fundamental_features()`

---

## Bug 5: Z-Score Index Mismatch (Fragile Positional Indexing)

### Root Cause
```python
# OLD (FRAGILE):
filing_idx = np.where(valid_mask)[0][-1]
...
if filing_idx < len(zscore_list):
    zs = zscore_list[filing_idx]  # POSITIONAL INDEX - can mismatch!
```

This assumes `zscore_list` has exact positional alignment with `filings_list`.
If there's any inconsistency (skipped entries, different filtering), indices would mismatch.

### Fix
Key by `filing_date` instead of positional index:

```python
# NEW (ROBUST):
# Build dicts keyed by filing_date
filings_by_date = {f["filing_date"]: f for f in filings_list}
zscore_by_date = {z["filing_date"]: z for z in zscore_list}

filing_dates = np.array(sorted(filings_by_date.keys()))

# Find most recent filing <= row_date
matched_filing_date = filing_dates[valid_mask][-1]

# Lookup by filing_date, not index
raw = filings_by_date.get(matched_filing_date)
zs = zscore_by_date.get(matched_filing_date)
```

### File Changed
`scripts/build_features_duckdb.py`:
- Lines ~670-725: Forward-fill section now uses dict lookup by filing_date

---

## Validation

After rebuild, run:
```bash
python scripts/validate_stepwise_behavior.py
```

Expected output:
- ✓ Raw TTM Piecewise: change_rate < 5%
- ✓ days_since Reset: min = 0 or 1
- ✓ Z-Score Stepwise: change_rate < 20%
- ✓ No Identical Patterns: tickers have different change counts

---

## Column Summary (After Fix)

### New DuckDB Schema (52 columns)

**Raw TTM (4 new):**
- `gross_margin_ttm` - piecewise-constant
- `operating_margin_ttm` - piecewise-constant
- `revenue_growth_yoy` - piecewise-constant
- `roe_raw` - piecewise-constant

**Z-Scores (4, now stepwise per-ticker):**
- `gross_margin_vs_sector` - stepwise per-ticker
- `operating_margin_vs_sector` - stepwise per-ticker
- `revenue_growth_vs_sector` - stepwise per-ticker
- `roe_zscore` - stepwise per-ticker

---

## Rebuild Command

```bash
cd /Users/ursinasanderink/Downloads/AI\ Stock\ Forecast
python scripts/build_features_duckdb.py --auto-normalize-splits
```

ETA: ~50-60 minutes (API-limited)


# Batch 5: Fundamentals - Implementation Plan

**Status:** 🟡 PHASE 1 IMPLEMENTED, BUILDING
**Risk Level:** HIGH (PIT leakage concerns)
**FMP Premium:** ✅ Available

---

## Critical PIT Issues Found

### Issue 1: Price Leakage (CRITICAL)
**Location:** `src/features/fundamental_features.py` lines 304-306
```python
if price is None and profile:
    price = profile.get("price")  # ❌ This is TODAY's price!
```
**Impact:** P/E, P/S ratios use current price for historical dates = lookahead leakage

**Fix Required:**
- Use historical close from OHLCV data (already in features_df)
- Pass `price` parameter from `build_features_duckdb.py` using as-of close

### Issue 2: Shares Outstanding Leakage (CRITICAL)
**Location:** `src/features/fundamental_features.py` lines 330-332
```python
shares = profile["sharesOutstanding"]  # ❌ This is CURRENT shares!
```
**Impact:** Market cap calculation uses current shares for historical dates

**Fix Required:**
- Extract shares from most recent quarterly/annual filing with observed_at <= asof
- FMP income statements have `weightedAverageShsOutDiluted`
- Forward-fill shares until next filing becomes available

### Issue 3: Sector Assignment (LOW RISK)
**Location:** Lines 277-278
```python
profile = self._get_profile(ticker)
sector = profile.get("sector", "Unknown")
```
**Impact:** Sector can change over time (reclassifications)

**Mitigation:** Document as approximation; sector changes are rare for our universe

---

## Implementation Strategy

### Step 1: Create PIT-Safe Fundamental Computation (in build_features_duckdb.py)
Instead of using `FundamentalFeatureGenerator` directly, integrate fundamentals into the existing build script with proper PIT controls:

1. **Download income statements** (already done for event features - limit=60)
2. **Use as-of price** from features_df (already computed)
3. **Extract shares** from filings (weightedAverageShsOutDiluted)
4. **Compute ratios** with PIT-safe inputs

### Step 2: Features to Add (7 total)
| Feature | Source | PIT-Safe Method |
|---------|--------|-----------------|
| `pe_zscore_3y` | Price / TTM EPS | as-of close ÷ TTM EPS from filings |
| `pe_vs_sector` | P/E vs sector median | Cross-sectional on each date |
| `ps_vs_sector` | P/S vs sector median | mcap = as-of close × filing shares |
| `gross_margin_vs_sector` | Margin vs sector | From filing grossProfit/revenue |
| `operating_margin_vs_sector` | Op margin vs sector | From filing opIncome/revenue |
| `revenue_growth_vs_sector` | YoY growth vs sector | From filings |
| `roe_zscore` | ROE z-score | From filing equity/income |

### Step 3: Validation Gates
1. **PIT Invariance Test:** Pick 2018-06-01, verify features don't change when today's price changes
2. **As-of Monotonicity:** Features should be step-function (constant until next filing)
3. **Coverage Ramp:** Expect low coverage in 2016, increasing over time
4. **Sector Median Sanity:** No NaN-heavy sectors, stable distributions

---

## Detailed TODO List

### Phase 1: Filings-Only Features ✅ IMPLEMENTED
- [x] Add function `compute_fundamental_features()` - uses TTM metrics
- [x] Add function `download_fundamental_data()` - downloads income + balance sheet
- [x] Compute TTM gross margin, operating margin from filing data
- [x] Compute YoY revenue growth (TTM now vs TTM 4 quarters ago)
- [x] Compute ROE (TTM net income / avg equity)
- [x] Cross-sectional z-scores per date, per sector (min 5 tickers)
- [x] Add sector from profile (documented as static, not PIT)
- [x] Update DuckDB schema (5 new columns)
- [x] Update features_cols list for INSERT

**Implementation Details:**
- TTM = sum of last 4 quarters where filing_date <= asof_date
- Forward-fill: features stay constant until next filing arrives
- Sector z-scores: minimum 5 tickers per sector, else NaN
- ROE z-score: universe-wide (not sector-specific)

### Phase 2: Valuation Features (P/E, P/S) - PENDING
- [ ] Compute P/E using: as-of close ÷ TTM EPS
- [ ] Compute P/S using: (as-of close × filing shares) ÷ TTM revenue
- [ ] Extract shares from filings (`weightedAverageShsOutDiluted`)
- [ ] Forward-fill shares until next filing
- [ ] Add pe_zscore_3y, pe_vs_sector, ps_vs_sector columns

### Phase 3: Validation - IN PROGRESS
- [ ] Build completes successfully
- [ ] PIT invariance test (historical date, today's price shouldn't matter)
- [ ] As-of monotonicity test (step-function behavior)
- [ ] Coverage ramp test (2016 < 2020 < 2024)
- [ ] Sector median stability test
- [ ] Unit tests + smoke test

### Phase 4: Documentation
- [ ] Create BATCH5_VALIDATION_COMPLETE.md
- [ ] Update ROADMAP.md

---

## Risk Mitigation

1. **Don't use `FundamentalFeatureGenerator` directly** - it has PIT issues
2. **Build fundamentals into `build_features_duckdb.py`** like events/regime
3. **Use existing filings_df** (already downloaded with limit=60)
4. **Use existing features_df close** as PIT-safe price
5. **Test with known historical dates** before trusting results

---

## Go/No-Go Criteria

| Check | Requirement |
|-------|-------------|
| Unit tests | 429+ passed |
| Smoke test | Chapter 7 runs |
| PIT invariance | Features stable when today's price changes |
| Coverage | <50% in 2016, >80% in 2024 (expected ramp) |
| Sector sanity | No sector with >50% NaN on any date |

---

## Time Estimate

- Phase 1: 30 min (PIT-safe function)
- Phase 2: 45 min (7 features)
- Phase 3: 30 min (validation tests)
- Phase 4: 15 min (documentation)
- **Total:** ~2 hours

---

## Alternative: Conservative Approach

If PIT-safe implementation is too complex:

1. **Skip P/E, P/S** (require market data)
2. **Only compute margin/growth features** (100% from filings, no price needed)
3. Add: `gross_margin_vs_sector`, `revenue_growth_vs_sector`, `roe_zscore`

This reduces scope but avoids all price-related leakage risks.


# AI Stock Forecaster - Project Documentation

## Overview

Building a **signal-only, PIT-safe, ranking-first** AI stock forecasting system that answers:

> "Which AI stocks are most attractive to buy today, on a risk-adjusted basis, over the next 20/60/90 trading days?"

---

## Table of Contents

1. [Current Status](#current-status)
2. [Completed Work](#completed-work)
3. [Key Achievements](#key-achievements)
4. [Issues & Solutions](#issues--solutions)
5. [Data Sources](#data-sources)
6. [PIT Timestamp Convention](#pit-timestamp-convention)
7. [Test Results](#test-results)
8. [What Needs To Be Done](#what-needs-to-be-done)
9. [Notes & Clarifications](#notes--clarifications)
10. [API Keys & Configuration](#api-keys--configuration)
11. [Project Structure](#project-structure)

---

## Current Status

| Section | Status | Notes |
|---------|--------|-------|
| 1. System Outputs | ✅ Complete | Signals, rankings, reports with distribution quantiles |
| 2. CLI & Pipelines | ✅ Complete | Commands: download-data, build-universe, score, make-report |
| 3. Data Infrastructure | ✅ Complete | FMP, Alpha Vantage, SEC EDGAR, Event Store |
| 4. Survivorship-Safe Universe | ✅ Complete | Polygon symbol master + UniverseBuilder with stable_id |
| 5. Feature Engineering | 🟡 In Progress | 5.1-5.4 ✅, 5.5 Regime ✅, 5.6 Missingness ✅ |
| 6. Evaluation Framework | 🔲 Pending | Walk-forward, purging/embargo, ranking metrics |
| 7-13. Models & Production | 🔲 Pending | Kronos, FinText, baselines, deployment |

**Section 4 is COMPLETE.** Ready to proceed to Section 5 (Feature Engineering).

### Section 5 Readiness Assessment

**Infrastructure in place (✅):**
| Component | Module | Status |
|-----------|--------|--------|
| Price data | `FMPClient.get_historical_prices()` | ✅ Split-adjusted, PIT-safe |
| Fundamentals | `FMPClient.get_income_statement()` etc. | ✅ With fillingDate for PIT |
| Volume/ADV | `DuckDBPITStore.get_avg_volume()` | ✅ Computed from OHLCV |
| Events | `EventStore` | ✅ EARNINGS, FILING, SENTIMENT, etc. |
| Earnings data | `AlphaVantageClient` + `ExpectationsClient` | ✅ With BMO/AMC timing |
| Regime data | `FMPClient.get_index_historical()` | ✅ For SPY/VIX |
| Security master | `SecurityMaster` | ✅ Stable IDs, ticker changes |
| Universe | `UniverseBuilder` | ✅ FULL survivorship with Polygon |

**Section 5 blockers (⚠️ None critical):**
- Rate limits: Use caching (already implemented in FMP, AV, Polygon clients)
- Options/positioning data: Stubs exist, require paid APIs if needed
- Multi-year backtest data: May require careful API call management

**Key considerations for Section 5:**
1. **Cache universe snapshots** by rebalance date to avoid repeated Polygon calls
2. **Batch API requests** where possible (FMP supports batch profiles)
3. **PIT discipline**: All features must use `observed_at <= asof` filtering
4. **Relative features preferred**: P/E vs sector median, not raw P/E
5. **Missingness masks**: Explicit indicators for "known at time T"

---

## Completed Work

### Section 1: System Outputs (✅ Complete)

**Files:**
- `src/outputs/signals.py` - StockSignal, ReturnDistribution, SignalDriver
- `src/outputs/reports.py` - SignalReport, CSV/JSON export
- `src/outputs/rankings.py` - CrossSectionalRanking, RankedStock

**Key Features:**
- Distribution quantiles (P5, P25, P50, P75, P95)
- Prob(Outperform) calculation
- Liquidity flags (ADV-based)
- Key drivers for interpretability
- Confidence buckets (calibrated uncertainty)

### Section 2: CLI & Pipelines (✅ Complete)

**Files:**
- `src/cli.py` - Command-line interface
- `src/pipelines/data_pipeline.py` - Data download orchestration
- `src/pipelines/universe_pipeline.py` - Universe construction
- `src/pipelines/scoring_pipeline.py` - Signal scoring

**Commands:**
```bash
python -m src.cli download-data --tickers NVDA,AMD --start 2024-01-01
python -m src.cli build-universe --asof 2024-06-15
python -m src.cli list-universe
python -m src.cli score --asof 2024-06-15 --horizon 20
```

### Section 3: Data Infrastructure (✅ Complete)

#### 3.1 Data Sources

| Source | Client | Status | Notes |
|--------|--------|--------|-------|
| **FMP** | `FMPClient` | ✅ Working | OHLCV (split-adjusted), fundamentals, profiles |
| **Alpha Vantage** | `AlphaVantageClient` | ✅ Implemented | Earnings calendar (date-only) |
| **SEC EDGAR** | `SECEdgarClient` | ✅ Implemented | Filing timestamps (GOLD STANDARD) |
| **Event Store** | `EventStore` | ✅ NEW | Earnings, filings, news, **sentiment** events |

**FMP Free Tier - What's Available:**
- ✅ Historical Prices (OHLCV) - **already split-adjusted** via `/stable/historical-price-eod/full`
- ✅ Quote (15-min delay)
- ✅ Profile (sector, industry, mcap)
- ✅ Income Statement (with filingDate!)
- ✅ Balance Sheet (with filingDate)
- ✅ Cash Flow (with filingDate)
- ✅ Ratios TTM
- ✅ Enterprise Value
- ❌ Earnings Calendar (paid only)
- ❌ Key Metrics endpoint (returns error on free tier)

#### 3.2 PIT Store (`DuckDBPITStore`)

**Schema:**
```sql
-- Prices with PIT timestamps
-- Note: close and adj_close are identical (FMP /full is split-adjusted)
prices (ticker, date, open, high, low, close, adj_close, volume, observed_at TIMESTAMPTZ)

-- Fundamentals supporting revisions
fundamentals (ticker, period_end, statement_type, field, value, filing_date, observed_at TIMESTAMPTZ)

-- Market snapshots for computed values
market_snapshots (ticker, date, market_cap, shares_outstanding, avg_volume_20d, observed_at TIMESTAMPTZ)
```

**Column Convention:**
- `close` = split-adjusted close (from FMP `/full` endpoint)
- `adj_close` = same as `close` (populated for compatibility)
- Both can be used interchangeably for returns/features

**Key Features:**
- All timestamps in UTC
- All queries filter by `observed_at <= asof`
- Window functions for per-ticker calculations (fixed get_avg_volume bug)
- Supports revisions (same period_end, different observed_at)

#### 3.3 Event Store (NEW)

**File:** `src/data/event_store.py`

A clean abstraction for discrete events with PIT-safe queries:

```python
from src.data import EventStore, Event, EventType, EventTiming

store = EventStore()

# Store earnings event with exact SEC timestamp
store.store_event(Event(
    ticker="NVDA",
    event_type=EventType.EARNINGS,
    event_date=date(2024, 11, 20),
    observed_at=datetime(2024, 11, 20, 21, 5, tzinfo=UTC),  # From SEC 8-K
    source="sec_8k",
    payload={"eps_actual": 0.81, "eps_estimate": 0.75},
    timing=EventTiming.AMC,
))

# Query events (PIT-safe)
events = store.get_events(["NVDA"], asof=asof_datetime)

# Get sentiment score (for feature engineering)
sentiment = store.get_sentiment_score("NVDA", asof, lookback_days=7)

# Days since last earnings
days = store.days_since_event("NVDA", EventType.EARNINGS, asof)
```

**Event Types:**
- `EARNINGS` - Quarterly earnings releases
- `FILING` - SEC filings (10-K, 10-Q, 8-K)
- `NEWS` - News articles
- `SENTIMENT` - Sentiment scores from text analysis
- `DIVIDEND`, `SPLIT`, `GUIDANCE`, `ANALYST`

**Why This Matters:**
- Sentiment integration becomes trivial (just add events)
- PIT rules are automatically enforced
- Easy to add new event types

#### 3.4 Trading Calendar

**Features:**
- NYSE holidays (Christmas, Thanksgiving, etc.)
- 4pm ET cutoff for daily data
- DST handling (winter: 21:00 UTC, summer: 20:00 UTC)
- Rebalance date generation (monthly, quarterly)

#### 3.5 Data Audits

**Checks:**
- PIT violation scanner (future_price, future_fundamental)
- Fundamental filing date validation
- Cutoff boundary tests (15:59 violation, 16:00 valid)
- Timezone consistency validation

### Section 4: Survivorship-Safe Universe (✅ Complete)

#### 4.1 Architecture

**Symbol Master:** Polygon.io Reference Tickers API
- `date=` parameter: retrieve tickers active on historical dates
- `active=false`: includes delisted securities
- CIK available for stable identifier mapping

**Price Vendor:** FMP (existing)
- OHLCV for tickers identified by Polygon
- Fundamentals and metadata

**Files:**
- `src/data/polygon_client.py` - PolygonClient for symbol master queries
- `src/data/universe_builder.py` - UniverseBuilder with survivorship status
- `tests/test_chapter4_universe.py` - Comprehensive tests

#### 4.2 Universe Construction

```python
from src.data.universe_builder import UniverseBuilder

builder = UniverseBuilder()
snapshot = builder.build(
    asof_date=date(2024, 6, 15),
    min_price=5.0,           # Price filter
    min_adv=1_000_000,       # $1M ADV filter
    max_constituents=100,     # Top N by mcap
    ai_filter=True,          # AI relevance filter
    use_polygon=True,        # Use Polygon for candidates
    skip_enrichment=False,   # Enrich with FMP data
)
```

#### 4.3 Survivorship Status

| Status | Meaning | When Used |
|--------|---------|-----------|
| FULL | Polygon candidates + verified delisted coverage | Production backtests |
| PARTIAL | ai_stocks.py fallback, may miss delistings | Development/unit tests |
| UNKNOWN | Not verified | Never use for backtests |

**⚠️ CRITICAL:** `SurvivorshipStatus.FULL` is ONLY possible when:
- `use_polygon=True` AND
- Candidates come from Polygon's "as-of date T" query

Unit tests may use `ai_stocks.py` for speed (PARTIAL); production builds MUST use Polygon (FULL).

#### 4.4 Key Principles

1. **ai_stocks.py is label-only**: Used for AI-relevance tagging, NEVER as candidate universe
2. **stable_id is primary identity**: CIK/FIGI survives ticker changes
3. **All filtering as-of-T**: No future information leakage
4. **Rate limit awareness**: Cache universe snapshots by rebalance date to avoid API hammering

#### 4.5 Test Results

```
CHAPTER 4 TEST RESULTS (7/7 passed):
  ✅ 1. Polygon API Access - CIK available
  ✅ 2. Universe Construction - 100 candidates from ai_stocks.py (PARTIAL - for speed)
  ✅ 3. Polygon Universe - Historical date queries work (FULL capable)
  ✅ 4. Stable ID Consistency - Reproducible
  ✅ 5. AI Stocks Integration - 10 categories, 100 tickers (label-only)
  ✅ 6. Delisted Tickers - delisted_utc available (FULL capable)
  ✅ 7. Summary - All capabilities confirmed
```

**Note:** Test 2 uses ai_stocks.py for speed with `skip_enrichment=True`. 
This results in PARTIAL status. Production FULL builds use Polygon.

**Polygon Free Tier Assessment:** ✅ SUFFICIENT for FULL survivorship
- Historical date queries work
- Delisted tickers accessible with timestamps
- CIK available for stable IDs

---

## Key Achievements

### 1. Split-Adjusted Prices Work Out of the Box

**Verified:** FMP's `/stable/historical-price-eod/full` endpoint IS split-adjusted.

Test result for NVDA's 10-for-1 split (June 10, 2024):
```
2024-06-07: FULL close = 120.89 vs RAW adjClose = 1208.9 → ratio = 10.0
2024-06-10: ratio = 1.0 (post-split, prices align)
```

**You do NOT need FMP paid tier for split handling.**

### 2. SEC EDGAR Integration (Gold Standard)

**Example NVDA 10-Q Filing:**
- Filed: 2025-11-19
- Accepted: 2025-11-19 16:36:17 UTC (exact second!)

This is the most accurate PIT timestamp available.

### 3. All Gate Tests Pass (5/5)

| Gate Test | What It Validates |
|-----------|-------------------|
| PIT Replay Invariance | Same query = identical results |
| As-Of Boundaries | 1-second precision cutoffs |
| SEC Filing Timestamps | Exact acceptance times |
| Corporate Action Integrity | No split artifacts |
| Universe Reproducibility | Deterministic builds |

### 4. Event Store for Sentiment

The new `EventStore` abstraction makes sentiment integration clean:
- Store news/earnings with observed_at timestamps
- Query sentiment scores with automatic PIT filtering
- No risk of lookahead bias contaminating features

---

## Issues & Solutions

### Issue 1: PIT Timestamp Inconsistency
**Problem:** Code used `date + 1 day` but cutoff was 4pm ET
**Solution:** Changed to `get_market_close_utc(date)` returning exact market close in UTC

### Issue 2: get_avg_volume() LIMIT Bug
**Problem:** `LIMIT N` applied globally, not per ticker
**Solution:** Window function with `PARTITION BY ticker ORDER BY date DESC`

### Issue 3: FMP Key Metrics Endpoint Error
**Problem:** Endpoint returns error on free tier
**Solution:** Compute metrics from income/balance/cashflow ourselves

### Issue 4: Alpha Vantage No BMO/AMC Timing
**Problem:** Earnings calendar provides dates but not pre/post market timing
**Solution:** 
1. Conservative default: assume AMC, available next market open
2. For precise timing: use SEC 8-K acceptance timestamps

### Issue 5: FMP Column Name Typo
**Problem:** FMP uses `filingDate` (single l), code checked `fillingDate` (double l)
**Solution:** Check for both spellings

### Issue 6: SEC CIK URL Host Header Mismatch
**Problem:** SEC CIK mapping is at www.sec.gov, but client had Host: data.sec.gov
**Solution:** Use different headers for www.sec.gov vs data.sec.gov endpoints

### Issue 7: Misunderstanding About adj_close
**Problem:** Assumed FMP free tier lacked split-adjusted prices
**Solution:** Verified `/stable/historical-price-eod/full` IS split-adjusted. No paid tier needed.

**Convention Adopted:**
- FMP `/full` endpoint returns `close` which IS already split-adjusted
- We populate `adj_close = close` in FMPClient for schema consistency
- Downstream code can use either `close` or `adj_close` interchangeably

---

## Data Sources

### Primary: FMP (Financial Modeling Prep)

**Rate Limits (Free Tier):**
- 250 requests/day
- 5 requests/minute

**Best For:**
- Historical prices (OHLCV) - **split-adjusted by default**
- Quarterly fundamentals with filing dates
- Company profiles

**Pricing Endpoints:**
| Endpoint | Result |
|----------|--------|
| `/stable/historical-price-eod/full` | Split-adjusted prices ✅ |
| `/stable/historical-price-eod/non-split-adjusted` | Raw prices (pre-split scale) |

### Secondary: Alpha Vantage

**Rate Limits (Free Tier):**
- 25 requests/day (very limited!)
- 5 requests/minute

**Best For:**
- Earnings calendar (supplement to FMP)

**Limitation:** No BMO/AMC timing information

### Gold Standard: SEC EDGAR

**Rate Limits:**
- 10 requests/second (recommended < 8)
- No daily limit

**Best For:**
- **PIT-accurate filing timestamps** (acceptanceDateTime to the second)
- 8-K filings for exact earnings release times
- XBRL fundamentals

**No API Key Required** - just need User-Agent header

---

## PIT Timestamp Convention

All timestamps stored in **UTC**.

| Data Type | observed_at Rule |
|-----------|-----------------|
| Prices | Market close (4pm ET → 20:00 or 21:00 UTC) |
| Fundamentals (FMP) | filing_date + next market open (conservative) |
| Fundamentals (SEC) | acceptanceDateTime (exact, gold standard) |
| Earnings (Alpha Vantage) | Assume AMC → next market open |
| Earnings (SEC 8-K) | acceptanceDateTime (exact) |
| News/Sentiment | Publication timestamp |

**Cutoff Boundaries:**
```
15:59 ET → VIOLATION (data not yet available)
16:00 ET → VALID (exactly at market close)
16:01 ET → VALID (after market close)
```

---

## Test Results

### Section 3 Unit Tests (No API Calls)

```
============================================================
SUMMARY
============================================================
  ✓ PASS: timestamp_functions       (DST winter/summer handling)
  ✓ PASS: observed_at_filtering     (PIT query boundaries)
  ✓ PASS: avg_volume_per_ticker     (window function fix)
  ✓ PASS: fundamental_pit           (filing date validation)
  ✓ PASS: violation_types           (PRICE vs FUNDAMENTAL)
  ✓ PASS: cutoff_boundaries         (15:59/16:00/16:01)
  ✓ PASS: calendar_holidays         (NYSE holidays)
  ✓ PASS: rebalance_dates           (EOM trading days)

  Total: 8/8 tests passed
```

### Section 4 Gate Tests

```
============================================================
SECTION 4 GATE TESTS
============================================================
  ✓ PASS: 1_replay_invariance      (Same query = same result)
  ✓ PASS: 2a_asof_boundaries       (Price/fundamental timing)
  ✓ PASS: 2b_sec_boundaries        (SEC exact timestamps)
  ✓ PASS: 3_corp_actions           (Split/dividend integrity)
  ✓ PASS: 4_universe_repro         (Deterministic universe build)

  Total: 5/5 gate tests passed
  ✓ GO: Ready for Section 4
```

### Event Store Tests

```
Testing EventStore...
  ✓ Stored 3 events
  ✓ Retrieved 2 NVDA events
  ✓ PIT boundary test passed (event not visible before observed_at)
  ✓ Sentiment score: 0.85
  ✓ Days since earnings: 2
All EventStore tests passed! ✓
```

**Run Tests:**
```bash
# Unit tests only (no API)
python tests/test_section3.py

# With integration tests (uses API quota)
RUN_INTEGRATION=1 python tests/test_section3.py

# Gate tests for Section 4
RUN_INTEGRATION=1 python tests/test_section4_gates.py

# Event store tests
python -c "from src.data.event_store import test_event_store; test_event_store()"
```

---

## Chapter 3 Extensions (COMPLETE)

### New Data Clients Implemented

| Client | Status | Free Tier? | Data Available |
|--------|--------|------------|----------------|
| `ExpectationsClient` | ✅ Complete | ✅ AV | Earnings surprises with BMO/AMC |
| `PositioningClient` | 🔲 Stubs | ❌ | Short interest, 13F, ETF flows |
| `OptionsClient` | 🔲 Stubs | ❌ | IV surfaces, implied moves |
| `SecurityMaster` | ✅ Complete | ✅ | Ticker changes, delistings |

### New EventTypes Added

```python
# Tier 1 - Forward-looking expectations
ESTIMATE_SNAPSHOT    # Consensus estimate at a point in time
ESTIMATE_REVISION    # Analyst estimate revision
GUIDANCE             # Company forward guidance
ANALYST_ACTION       # Rating change / price target update

# Tier 2 - Options & Positioning
OPTIONS_SNAPSHOT     # EOD options chain / IV surface
SHORT_INTEREST       # Short interest + days to cover
BORROW_COST          # Stock borrow fee rate
ETF_FLOW             # ETF inflow/outflow
INSTITUTIONAL_13F    # 13F institutional holdings

# Tier 0 - Survivorship
SECURITY_MASTER      # Ticker changes, delistings, mergers
DELISTING            # Delisting event with terminal price
```

### Earnings Surprises with PIT Timing

Alpha Vantage provides BMO/AMC timing (critical for PIT):

```python
from src.data import get_expectations_client

client = get_expectations_client()
surprises = client.get_earnings_surprises("NVDA")

# Example output:
# reported_date: 2024-11-19
# report_time: "post-market"
# observed_at: 2024-11-19 21:05:00 UTC (after 4pm ET)
# surprise: 4.84%
```

### Chapter 3 Extension Test Results

```
============================================================
CHAPTER 3 EXTENSIONS - TEST SUITE
============================================================
  ✓ PASS: 1_event_types       (17 event types)
  ✓ PASS: 2_earnings_pit      (BMO/AMC handling)
  ✓ PASS: 3_security_master   (ticker changes, delistings)
  ✓ PASS: 4_positioning_pit   (13F uses filing_date!)
  ✓ PASS: 5_options_pit       (IV at market close)
  ✓ PASS: 6_eventstore        (PIT-safe storage)

  Total: 6/6 tests passed
```

---

## What Needs To Be Done (Future Chapters)

---

### ⚠️ CRITICAL NOTES FOR CHAPTER 4 (URGENT)

Before or during Chapter 4 implementation, ensure these items are addressed:

#### 1. Stable IDs Must Be First-Class (IMPLEMENTED ✅)

**Problem:** Ticker changes (FB→META, TWTR→delisted) break historical universe replay if you key on tickers.

**Solution Applied:**
- `TickerMetadata` now includes `stable_id` field
- `UniverseResult` includes `stable_ids` list (canonical for replay)
- Helper methods: `get_ticker_for_stable_id()`, `get_stable_id_for_ticker()`
- All universe construction must key on stable IDs end-to-end

**Enforcement:**
```python
# BAD: using ticker as identity
historical_universe = [ticker for ticker in get_all_tickers()]

# GOOD: using stable_id as identity
historical_universe = [(stable_id, get_ticker_for_stable_id(stable_id, asof_date)) 
                       for stable_id in get_all_stable_ids()]
```

#### 2. Conservative Earnings Time Handling (VERIFIED ✅)

**Problem:** If BMO/AMC timing is missing, you might leak information.

**Solution Verified in `ExpectationsClient._get_observed_at_from_report()`:**
```python
if report_time == "pre-market":
    # BMO: available at market open (9:30 ET)
elif report_time == "post-market":
    # AMC: available at market close + buffer (4:05 ET)
else:
    # UNKNOWN: conservative → next market open
    next_open = calendar.get_next_trading_day(reported_date)
```

#### 3. Survivorship Status Labels (IMPLEMENTED ✅)

**Problem:** Until a real survivorship-bias-free feed is used, backtest results may be optimistic.

**Solution Applied:**
- `SurvivorshipStatus` class with `FULL`, `PARTIAL`, `UNKNOWN`
- `UniverseResult.survivorship_status` defaults to `PARTIAL`
- Summary output shows warning: "⚠️ PARTIAL SURVIVORSHIP"

**Rule:** Don't trust backtest results until survivorship_status = FULL.

---

### Chapter 4: Survivorship-Safe Dynamic Universe (✅ COMPLETE)

**Data Sources:**
- ✅ Polygon.io for symbol master (universe-as-of-T)
- ✅ FMP for prices/fundamentals (existing)

**Completed Tasks:**
- [x] Build historical universe reconstruction using stable IDs
- [x] Implement top-N by market cap as-of T with liquidity/price thresholds
- [x] PolygonClient for "active on date T" queries
- [x] UniverseBuilder with survivorship_status tracking
- [x] Comprehensive tests (7/7 passed)

**Polygon Free Tier:** ✅ SUFFICIENT
- Historical date queries work
- Delisted tickers with timestamps
- CIK for stable IDs

**Files Created:**
- `src/data/polygon_client.py`
- `src/data/universe_builder.py`
- `tests/test_chapter4_universe.py`

**Acceptance Criteria Met:**
- ✅ Constituents vary through time (via Polygon historical queries)
- ✅ Includes delisted names (active=false)
- ✅ Reproducible for any historical date

### Chapter 5: Feature Engineering (Bias-Safe) — 🔲 NEXT

#### Infrastructure Available (from Chapters 3-4) ✅
| Component | Module | What It Provides |
|-----------|--------|------------------|
| Prices | `FMPClient.get_historical_prices()` | Split-adjusted OHLCV with `observed_at` |
| Fundamentals | `FMPClient.get_income_statement()` etc. | With `fillingDate` for PIT |
| Volume/ADV | `DuckDBPITStore.get_avg_volume()` | Computed from OHLCV |
| Events | `EventStore` | EARNINGS, FILING, SENTIMENT with PIT |
| Earnings | `AlphaVantageClient` + `ExpectationsClient` | BMO/AMC timing, surprises |
| Regime/VIX | `FMPClient.get_index_historical()` | SPY, VIX for regime detection |
| Universe | `UniverseBuilder` | FULL survivorship via Polygon |
| ID Mapping | `SecurityMaster` | Stable IDs, ticker changes |
| Calendar | `TradingCalendarImpl` | NYSE holidays, cutoffs |
| Caching | All clients | `data/cache/*` directories |

#### API Keys Available ✅
- `FMP_KEYS` - **PREMIUM** - 30 years data, QQQ, all endpoints
- `POLYGON_KEYS` - Symbol master, universe (free tier: 5/min)
- `ALPHAVANTAGE_KEYS` - Earnings calendar (free tier: 25/day)

---

#### Chapter 5 Detailed TODO

**5.1 Targets (Labels) — DEFINITION LOCKED**

**Return Definition (v1):**
- **Split-adjusted close price return** (close-to-close)
- Dividends NOT included in v1 (documented limitation, TODO for v2)
- Source: FMP `/stable/historical-price-eod/full` (already split-adjusted)

**Label Formula:**
```
y_i,T(H) = (P_i,T+H / P_i,T - 1) - (P_b,T+H / P_b,T - 1)

where:
  P_i,T    = stock i split-adjusted close on date T
  P_i,T+H  = stock i split-adjusted close on date T+H trading days
  P_b,T    = benchmark (QQQ) close on date T
  H        = horizon in TRADING DAYS (20, 60, 90)
```

**Label Alignment (matches cutoff policy):**
- Entry: price(T close) — 4:00pm ET cutoff
- Exit: price(T+H close) — H trading days forward
- Benchmark: same dates as stock
- Calendar: Use `TradingCalendarImpl` for trading day arithmetic

**Label Availability Rule (PIT-safe):**
- Labels are future-looking, so NO `observed_at` in traditional sense
- Labels mature at T+H close
- During training/eval: filter by `asof >= T+H close`
- Labels table supports walk-forward + purging/embargo from day 1

**Storage:**
- Same DuckDB pattern as features
- Keys: `stable_id`, `ticker`, `date`, `horizon`
- Values: `excess_return`, `stock_return`, `benchmark_return`
- Metadata: `label_matured_at`, `benchmark_ticker`

**Implementation Tasks:**
- [x] Lock return definition (split-adjusted close, no dividends v1)
- [x] Implement forward excess return calculation vs benchmark
- [x] Create label generator for 20/60/90 trading day horizons
- [x] Store labels in DuckDB with maturity timestamps
- [x] Add tests for label correctness and PIT safety (8/8 tests pass)

**Note:** FMP Premium available - QQQ and 30 years of data accessible.

**5.2 Price & Volume Features ✅ COMPLETE**

| Feature | Description | Status |
|---------|-------------|--------|
| `mom_1m`, `mom_3m`, `mom_6m`, `mom_12m` | Returns over 21/63/126/252 trading days | ✅ |
| `vol_20d`, `vol_60d` | Annualized volatility | ✅ |
| `vol_of_vol` | Volatility of rolling volatility | ✅ |
| `max_drawdown_60d` | Maximum drawdown over 60 days | ✅ |
| `dist_from_high_60d` | Distance from 60-day high | ✅ |
| `rel_strength_1m`, `rel_strength_3m` | Z-score vs universe | ✅ |
| `beta_252d` | Beta vs benchmark (QQQ) | ✅ |
| `adv_20d`, `adv_60d` | Average daily dollar volume | ✅ |
| `vol_adj_adv` | ADV / volatility | ✅ |

**Files:** `src/features/price_features.py`
**Tests:** 9/9 passed in `tests/test_features.py`

**5.3 Fundamental Features (Relative, Normalized) ✅ COMPLETE**

| Feature | Description | Status |
|---------|-------------|--------|
| `pe_zscore_3y` | P/E vs own 3-year history | ✅ |
| `ps_zscore_3y` | P/S vs own 3-year history | ✅ |
| `pe_vs_sector` | P/E relative to sector median | ✅ |
| `ps_vs_sector` | P/S relative to sector median | ✅ |
| `gross_margin_vs_sector` | Gross margin vs sector | ✅ |
| `operating_margin_vs_sector` | Operating margin vs sector | ✅ |
| `revenue_growth_vs_sector` | Revenue growth vs sector | ✅ |
| `roe_zscore`, `roa_zscore` | Quality metrics z-scored | ✅ |

**Key Design:** Raw ratios avoided - all features are RELATIVE
**Files:** `src/features/fundamental_features.py`
**Tests:** 9/9 passed in `tests/test_features.py`

---

### Time-Decay Sample Weighting (Training Policy) ✅ COMPLETE

**Why Time Decay Matters for AI Stocks:**
- AI stocks' business models have changed dramatically over time
- The "AI regime" (2020+) is fundamentally different from earlier eras
- Market microstructure evolves (HFT, retail flow, etc.)
- Recent observations are more relevant for forward predictions
- Many AI stocks didn't exist 15+ years ago - that's OK, don't fill missing years

**Policy:**
- Use exponentially-decayed sample weights by date during training
- Half-life determines how quickly old data becomes less relevant
- Per-date normalization ensures each date contributes equally

**Recommended Half-Lives:**
| Horizon | Half-Life | Rationale |
|---------|-----------|-----------|
| 20d     | 2.5 years | Short-term patterns change faster |
| 60d     | 3.5 years | Medium-term patterns |
| 90d     | 4.5 years | Longer-term trends more stable |

**Formula:**
```
w(t) = 2^(-Δ(t) / half_life)
where Δ(t) = days between t and training_end_date

Example (3-year half-life):
- 3 years old → weight = 0.50
- 6 years old → weight = 0.25
- 9 years old → weight = 0.125
```

**Key Rules:**
1. Use survivorship-safe universe as-of each date (from Chapter 4)
2. Compute weights per row, not per stock (young stocks get fewer rows, higher weight)
3. Normalize within each date for cross-sectional ranking

**Where This Applies:**
- Section 6: Evaluation framework (walk-forward training)
- Section 11: Fusion model training
- NOT during feature computation (features use fixed lookback windows)

**Implementation:**
- **File:** `src/features/time_decay.py`
- **Functions:** `compute_time_decay_weights()`, `get_half_life_for_horizon()`
- **Usage:** `model.fit(X, y, sample_weight=weights)`

**Note on 30-Year Data:**
- FMP Premium provides 30 years of data
- Don't use a fixed "30-year window" for all stocks
- Use what exists, let time-decay handle relevance
- Effective sample will naturally concentrate in the last 10-15 years for AI stocks

---

**5.4 Event & Calendar Features ✅ COMPLETE**

| Feature | Description | Status |
|---------|-------------|--------|
| `days_to_earnings` | Days until next expected earnings | ✅ |
| `days_since_earnings` | Days since last earnings report | ✅ |
| `in_pead_window` | Post-earnings drift window indicator (63 days) | ✅ |
| `pead_window_day` | Which day of PEAD window (1-63) | ✅ |
| `last_surprise_pct` | Most recent earnings surprise % | ✅ |
| `avg_surprise_4q` | Rolling 4-quarter average surprise | ✅ |
| `surprise_streak` | Consecutive beats (+) or misses (-) | ✅ |
| `surprise_zscore` | Cross-sectional z-score of surprise | ✅ |
| `earnings_vol` | Std dev of surprises (8Q) | ✅ |
| `days_since_10k` | Days since last annual report | ✅ |
| `days_since_10q` | Days since last quarterly report | ✅ |
| `reports_bmo` | Typical BMO vs AMC timing | ✅ |

**Files:** `src/features/event_features.py`
**Tests:** 9/9 passed in `tests/test_event_features.py`

**5.5 Regime & Macro Features ✅ COMPLETE**

| Feature | Description | Status |
|---------|-------------|--------|
| `vix_level` | Raw VIX level | ✅ |
| `vix_percentile` | VIX percentile over 2-year window | ✅ |
| `vix_change_5d` | 5-day VIX change | ✅ |
| `vix_regime` | low/normal/elevated/high classification | ✅ |
| `market_return_5d` | 5-day SPY return | ✅ |
| `market_return_21d` | 21-day (~1 month) SPY return | ✅ |
| `market_return_63d` | 63-day (~3 month) SPY return | ✅ |
| `market_vol_21d` | 21-day realized volatility | ✅ |
| `market_regime` | bull/bear/neutral classification | ✅ |
| `above_ma_50` | Price > 50-day MA | ✅ |
| `above_ma_200` | Price > 200-day MA | ✅ |
| `ma_50_200_cross` | (MA50 - MA200) / MA200 | ✅ |
| `tech_vs_staples` | XLK vs XLP relative strength | ✅ |
| `tech_vs_utilities` | XLK vs XLU relative strength | ✅ |
| `risk_on_indicator` | Composite risk-on/off signal | ✅ |

**Key:** Market-level features common to all stocks in universe.
**Files:** `src/features/regime_features.py`
**Tests:** 10/10 passed in `tests/test_regime_missingness.py`

**5.6 Missingness Masks ✅ COMPLETE**

| Feature | Description | Status |
|---------|-------------|--------|
| `coverage_pct` | Overall feature coverage (0-1) | ✅ |
| `price_coverage` | Price feature category coverage | ✅ |
| `fundamental_coverage` | Fundamental feature coverage | ✅ |
| `event_coverage` | Event feature coverage | ✅ |
| `regime_coverage` | Regime feature coverage | ✅ |
| `has_price_data` | Boolean price availability flag | ✅ |
| `has_fundamental_data` | Boolean fundamental availability | ✅ |
| `has_earnings_data` | Boolean earnings availability | ✅ |
| `is_new_stock` | < 252 days of history | ✅ |
| `fundamental_staleness_days` | Days since last fundamental update | ✅ |
| `{feature}_available` | Per-feature availability mask | ✅ |

**Key Philosophy:** Missingness is a SIGNAL, not just noise to impute.
**Files:** `src/features/missingness.py`
**Tests:** 10/10 passed in `tests/test_regime_missingness.py`

**Coverage Report Generation:**
```python
from src.features.missingness import MissingnessTracker
tracker = MissingnessTracker()
print(tracker.generate_coverage_report(features_df))
```

**5.7 Feature Hygiene & Redundancy Control (NEW)**
- [ ] Cross-sectional z-score/rank standardization
- [ ] Rolling Spearman correlation matrix
- [ ] Feature clustering & block aggregation (don't drop singles)
- [ ] VIF diagnostics for tabular features (diagnostic, not hard filter)
- [ ] Rolling IC stability checks (more important than VIF)
- [ ] Sign consistency analysis across time

> **Principle**: A feature with IC 0.04 once and −0.01 later is worse than IC 0.02 stable forever.

**5.8 Feature Neutralization (Evaluation-Only, Optional) (NEW)**
- [ ] Sector-neutral IC computation
- [ ] Beta-neutral IC computation
- [ ] Market-neutral IC computation

*Used for diagnostics to reveal where alpha comes from — not for training.*

**Optional Paid-Tier Features:**
- [ ] Expectations: Revisions momentum, estimate dispersion, guidance
- [ ] Options: IV level/skew, implied move, IV percentile (requires paid data)
- [ ] Positioning: Short interest, ETF flows, 13F shifts (requires paid data)

---

#### Testing & Validation Requirements
- [ ] Unit tests for each feature block
- [ ] PIT violation scanner on all features
- [ ] Univariate IC ≥ 0.03 check for strong signals
- [ ] IC stability across ≥70% of rolling windows
- [ ] Feature coverage > 95% (post-masking)
- [ ] Redundancy documented: correlation matrix, feature blocks

#### Rate Limit Strategy
1. Cache universe snapshots by rebalance date (Polygon: 5/min)
2. Batch FMP requests where possible (profiles, quotes)
3. Use Alpha Vantage sparingly (25/day limit)
4. Store computed features in DuckDB for reuse

#### Success Criteria
- > 95% feature completeness (post-masking)
- Strong univariate signals show IC ≳ 0.03
- No feature introduces PIT violations
- IC sign consistent across ≥70% of rolling windows
- Redundancy understood: feature blocks documented

### Chapter 6: Evaluation Realism

- [ ] Re-run walk-forward once universe is survivorship-safe
- [ ] Confirm performance doesn't depend on survivorship bias
- [ ] Add diagnostics for signals that break under constraints (borrow/short crowding)
- [ ] Purging & embargo (gap between train/test)
- [ ] Ranking metrics (top-N hit rate, rank correlation)
- [ ] **Apply time-decay weighting** during training (`src/features/time_decay.py`)
- [ ] Use horizon-specific half-lives: 2.5y (20d), 3.5y (60d), 4.5y (90d)
- [ ] Per-date normalization for cross-sectional ranking loss

### Chapter 11/12: Fusion + Regime-Aware Ensembling

- [ ] Add expectations/options/positioning blocks into fusion
- [ ] Add regime-aware weighting (earnings windows vs normal regimes)
- [ ] Check redundancy (correlation with Kronos/FinText; keep only additive blocks)
- [ ] **Apply time-decay weighting** during fusion training
- [ ] Keep decay inside each regime the same way (weights by date, normalized by date)

### Chapter 13/14: Confidence + Monitoring

- [ ] Use options/dispersion for confidence stratification
- [ ] Monitoring alerts for estimates/options feed gaps
- [ ] Drift detection for positioning metrics

---

## Notes & Clarifications

### When Would FMP Paid Tier Be Recommended?

**Only if you need:**

| Need | Free Tier | Paid Tier |
|------|-----------|-----------|
| Split-adjusted prices | ✅ Already have | Same |
| Rate limits (250/day) | May hit limits with 100+ tickers | Higher limits |
| History (5+ years) | May need more | Longer history |
| Earnings calendar | ❌ Use SEC/AV instead | ✅ Available |
| Key Metrics endpoint | ❌ Compute yourself | ✅ Available |
| **Analyst estimates** | ❌ | ✅ Required |
| **Short interest** | ❌ | ✅ Required |
| **13F holdings** | ❌ (use SEC parsing) | ✅ Easier |
| **Options data** | ❌ | ❌ (use CBOE/OptionMetrics) |
| **Survivorship data** | ❌ | ✅ Available |
| Survivorship Bias Free | Check if available | ✅ Documented |

**Verdict:** Free tier is sufficient for development. Consider upgrading only if:
1. You're hitting rate limits frequently
2. You need multi-year backtests with many tickers
3. You want convenience endpoints

### Sentiment Is NOT Optional

Changed in notebook from "optional" to "Core Feature".

**Why sentiment matters for AI stocks:**
- Earnings surprises drive significant price moves
- News flow affects investor sentiment directly
- AI sector is particularly news-sensitive (new model releases, partnerships, etc.)

**How it's implemented:**
- `EventStore` with `SENTIMENT` event type
- PIT-safe queries (observed_at = publication time)
- Ready for FinBERT or similar NLP models

### FMP Price Data Clarification

- `/stable/historical-price-eod/full` = split-adjusted (use this)
- `/stable/historical-price-eod/non-split-adjusted` = raw pre-split prices

The "full" endpoint doesn't have a separate `adj_close` column because the `close` IS already adjusted. This is intentional, not a limitation.

---

## API Keys & Configuration

**Required in `.env`:**
```env
# Financial Modeling Prep (required for main data)
FMP_KEYS=your_fmp_api_key

# Alpha Vantage (optional - for earnings calendar)
ALPHAVANTAGE_KEYS=your_alphavantage_key

# SEC EDGAR (optional - for precise filing timestamps)
SEC_CONTACT_EMAIL=your_email@domain.com
```

**Get API Keys:**
- FMP: https://financialmodelingprep.com/developer/docs/pricing
- Alpha Vantage: https://www.alphavantage.co/support/#api-key
- SEC: No key needed, just provide contact email

---

## Project Structure

```
AI Stock Forecast/
├── src/
│   ├── cli.py                    # Command-line interface
│   ├── config.py                 # Configuration
│   ├── interfaces.py             # Protocols (PITStore, TradingCalendar)
│   ├── data/
│   │   ├── fmp_client.py         # FMP API client
│   │   ├── alphavantage_client.py # Alpha Vantage client
│   │   ├── sec_edgar_client.py   # SEC EDGAR client
│   │   ├── pit_store.py          # DuckDB PIT store
│   │   ├── trading_calendar.py   # NYSE calendar
│   │   ├── event_store.py        # Event store (17 event types)
│   │   ├── expectations_client.py # Earnings surprises, estimates (Ch.3 ext)
│   │   ├── positioning_client.py  # Short interest, 13F, ETF flows (stubs)
│   │   ├── options_client.py      # IV surfaces, implied moves (stubs)
│   │   └── security_master.py     # Identifier mapping, delistings
│   ├── outputs/
│   │   ├── signals.py            # Signal dataclasses
│   │   ├── reports.py            # Report generation
│   │   └── rankings.py           # Cross-sectional ranking
│   ├── universe/
│   │   └── ai_stocks.py          # 100 AI stocks × 10 categories
│   ├── pipelines/
│   │   ├── data_pipeline.py      # Data download
│   │   ├── universe_pipeline.py  # Universe construction
│   │   └── scoring_pipeline.py   # Signal scoring
│   └── audits/
│       ├── pit_scanner.py        # PIT violation detection
│       ├── survivorship_audit.py # Survivorship bias checks
│       └── corp_action_checks.py # Corporate action validation
├── tests/
│   ├── test_signals.py           # Signal unit tests
│   ├── test_reports.py           # Report unit tests
│   ├── test_section3.py          # Data infrastructure tests
│   ├── test_section4_gates.py    # Gate tests for Section 4
│   └── test_chapter3_extensions.py # Chapter 3 extension tests
├── data/                         # Downloaded data (gitignored)
├── outputs/                      # Generated reports (gitignored)
├── requirements/
│   ├── base.txt                  # Core dependencies
│   ├── dev.txt                   # Development tools
│   ├── ml.txt                    # ML libraries
│   └── research.txt              # Jupyter, plotting
├── PROJECT_DOCUMENTATION.md      # This file
└── PROJECT_STRUCTURE.md          # Architecture details
```

---

## Changelog

| Date | Changes |
|------|---------|
| Dec 2025 | Section 3 complete, Event Store added, Gate tests added |
| Dec 2025 | Fixed PIT timestamp bugs, SEC client Host header |
| Dec 2025 | Clarified FMP split-adjusted prices (no paid tier needed) |
| Dec 2025 | Made sentiment NOT optional in notebook |

---

*Last Updated: December 2025*

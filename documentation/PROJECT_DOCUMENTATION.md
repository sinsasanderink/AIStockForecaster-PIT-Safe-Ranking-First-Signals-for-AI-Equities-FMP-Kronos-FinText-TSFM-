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
| 5. Feature Engineering | ✅ **COMPLETE (v2)** | **5.1-5.8 + v2 labels (dividends) + PIT scanner enforced. 84/84 tests passed.** |
| 6. Evaluation Framework | 🔲 **NEXT** | Walk-forward, purging/embargo, ranking metrics |
| 7-13. Models & Production | 🔲 Pending | Kronos, FinText, baselines, deployment |

**Chapter 5 is COMPLETE (v2).** Ready to proceed to Chapter 6 (Evaluation Framework).

**Key Completions:**
- ✅ v2 Labels: Total return with dividends (DEFAULT), v1 available for comparison
- ✅ PIT Scanner: Automated, enforced in CI, 0 CRITICAL violations
- ✅ All feature modules: 5.1-5.8 complete and tested
- ✅ 84/84 tests passed (was 81/81, +3 for PIT scanner, dividend tests, smoke test update)

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

## Qlib Integration (Shadow Evaluator for Chapter 6+)

**Integration Philosophy: "Optional Evaluator, Not Replacement"**

[Qlib](https://github.com/microsoft/qlib) (Microsoft's AI-oriented quantitative investment platform) will be integrated starting Chapter 6 as a **shadow evaluator and benchmark harness**, NOT as a replacement for our core infrastructure.

### What Qlib Will NOT Replace (Chapters 1-5 Remain Intact)

❌ **DO NOT use Qlib for:**
- Data ingestion / PIT store / DuckDB backend
- Universe construction / survivorship-safe snapshots
- Feature engineering / PIT discipline / stable IDs
- Label generation / dividend handling / maturity rules
- Security master / ticker change handling

**Rationale:** We've built institutional-grade PIT discipline, survivorship handling, and feature engineering. Swapping this would be **high-risk / low-reward** and could lose the exact guarantees we've established.

### Where Qlib Saves Time (Chapter 6 Onwards)

✅ **DO use Qlib for:**

**1. Chapter 6.3 Metrics + 6.5 Reporting (Biggest Win)**

Qlib provides built-in standardized factor evaluation:
- **IC Analysis:** Information Coefficient (Pearson & Spearman), monthly IC, IC by regime
- **Quintile Analysis:** Group returns, top-bottom spread, long-short distribution
- **Prediction Diagnostics:** Autocorrelation, prediction distribution, rank-label scatter
- **Portfolio Metrics:** Cumulative return, drawdown, Sharpe/IR, turnover

**Qlib Reports:** [Reference](https://qlib.readthedocs.io/en/latest/component/report.html)

![Qlib IC Analysis](https://qlib.readthedocs.io/en/latest/_images/analysis_model_ic.png)
![Qlib Cumulative Returns](https://qlib.readthedocs.io/en/latest/_images/analysis_position_cumulative_return.png)

**Integration:**
```python
# Our system generates predictions
predictions_df = our_model.predict(features_df)  # (date, ticker, pred)
labels_df = our_label_generator.get_labels(...)  # (date, ticker, label)

# Hand to Qlib for evaluation
qlib_df = pd.merge(predictions_df, labels_df, on=["date", "ticker"])
qlib_df["group"] = sector_map  # Optional: sector/liquidity buckets

# Generate standardized reports
from qlib.contrib.evaluate import backtest_daily
reports = backtest_daily(prediction=qlib_df, ...)
```

**Benefit:** Instead of writing custom IC plot / quintile analysis / churn diagnostic code, leverage Qlib's mature reporting stack.

**2. Chapter 6.4 Cost Realism (Second Opinion)**

Qlib's backtest engine outputs:
- Excess return **without cost**
- Excess return **with cost** (configurable transaction costs)
- Risk metrics: IR, max drawdown, turnover

**Integration:**
```python
# Use Qlib's backtest as validation
qlib_backtest = qlib.backtest.backtest(
    strategy=top_k_strategy,
    costs={"buy": 0.002, "sell": 0.002},  # 20 bps round-trip
)
```

**Benefit:** Provides independent validation of "does alpha survive?" with standardized cost models.

**3. Chapter 6.X Experiment Tracking**

Qlib's `Recorder` system logs:
- Model artifacts (weights, configs)
- Evaluation metrics (IC, backtest returns)
- Experiment metadata (walk-forward fold ID, hyperparameters)

**Integration:**
```python
from qlib.workflow import R

with R.start(experiment_name="walk_forward_fold_1"):
    # Train model
    model.fit(X_train, y_train)
    # Log metrics
    R.log_metrics({"ic": ic, "rankic": rankic, "quintile_spread": spread})
    # Save artifacts
    R.save_objects(**{"model.pkl": model})
```

**Benefit:** Clean experiment management across walk-forward folds, model variants (v1 vs v2 labels), and hyperparameter sweeps.

**4. Chapter 7 Baseline Harness (Optional)**

Qlib provides ready-made baselines:
- LightGBM, XGBoost, CatBoost
- Deep models: LSTM, GRU, Transformer, TFT, TabNet
- Quant-specific: HIST, TRA, ALSTM, DDG-DA

**Integration:**
```python
# Run LightGBM baseline with our features
qlib.init(provider_uri="~/.qlib/qlib_data")
qrun benchmarks/LightGBM/workflow_config_lightgbm.yaml
```

**Benefit:** Fast baseline IC comparisons without writing training loops. Our models must beat these.

### Integration Pattern (Narrow & Safe)

**Data Flow:**
```
┌─────────────────────────────────────┐
│ Our System (Source of Truth)       │
│ - Universe snapshots (stable_id)   │
│ - Features (PIT-safe, 5.1-5.8)     │
│ - Labels (v2 total return)         │
│ - Predictions (our models)         │
└─────────────────┬───────────────────┘
                  │
                  │ DataFrame: (date, ticker, pred, label, group)
                  ↓
┌─────────────────────────────────────┐
│ Qlib (Evaluation & Reporting)      │
│ - IC analysis (monthly, regime)    │
│ - Quintile spread & hit rate       │
│ - Backtest (cost-inclusive)        │
│ - Experiment tracking (Recorder)   │
└─────────────────────────────────────┘
```

**Key Principle:** Qlib receives **predictions + labels**, NOT raw data. This avoids forcing our PIT/survivorship logic into Qlib's data handlers.

### Implementation Checklist (Chapter 6)

- [ ] Install Qlib: `pip install pyqlib`
- [ ] Create adapter: `our_predictions_to_qlib_format()`
- [ ] Run first evaluation report with Qlib
- [ ] Compare Qlib's IC with our manual IC calculation (sanity check)
- [ ] Set up Recorder for walk-forward experiment tracking
- [ ] Optional: Run LightGBM baseline via Qlib for comparison
- [ ] Document fallback: If Qlib reporting breaks, we can still compute IC manually

### What to Watch (Gotchas)

⚠️ **Don't let Qlib own the data format:** If you try to plug DuckDB/PIT store into Qlib's `DataProvider`, you'll lose weeks. Keep it narrow: predictions + labels only.

⚠️ **Version pinning:** Qlib is actively developed. Pin version once it works:
```bash
pip install pyqlib==0.9.7  # or whatever version works
```

⚠️ **Qlib's data expectations:** Qlib expects `(instrument, datetime)` multiindex. Our adapter must handle this.

### References

- **Qlib Documentation:** https://qlib.readthedocs.io/en/latest/
- **GitHub:** https://github.com/microsoft/qlib
- **Evaluation & Reporting:** https://qlib.readthedocs.io/en/latest/component/report.html
- **Recorder (Experiment Tracking):** https://qlib.readthedocs.io/en/latest/component/recorder.html
- **Backtest:** https://qlib.readthedocs.io/en/latest/component/strategy.html

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

### Chapter 5: Feature Engineering (Bias-Safe) ✅ COMPLETE

**Status:** Feature code: 50 features implemented, 84/84 tests passing  
**⚠️ Note:** Feature pipeline to DuckDB is partial (7 of 50 features). Full pipeline expansion is a pre-Chapter-8 task.

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

**5.1 Targets (Labels) — DEFINITION LOCKED ✅ COMPLETE (v2)**

**Return Definition (v2 - DEFAULT):**
- **Total return (price + dividends)** for BOTH stock and benchmark
- Split-adjusted close price + dividend yield
- Source: FMP `/stable/historical-price-eod/full` (prices) + `/historical-price-eod/stock_dividend` (dividends)

**Label Formula (v2):**
```
y_i,T(H) = TR_i,T(H) - TR_b,T(H)

where:
  TR_i,T(H) = (P_i,T+H / P_i,T - 1) + DIV_i,T(H)
  TR_b,T(H) = (P_b,T+H / P_b,T - 1) + DIV_b,T(H)
  
  DIV_i,T(H) = sum(dividends with ex-date in (T, T+H]) / P_i,T
  
  P_i,T    = stock i split-adjusted close on date T
  P_i,T+H  = stock i split-adjusted close on date T+H trading days
  H        = horizon in TRADING DAYS (20, 60, 90)
```

**Why v2 Matters:**
- Ranking fairness: mature dividend payers (MSFT ~0.8% yield) vs growth stocks
- For 90d horizon: ~0.2% dividend impact affects relative ranking
- Avoids systematic bias in performance attribution
- Consistency: total return for BOTH stock AND benchmark (no distortion)

**Legacy v1 (price only):**
- Available via `label_version='v1'` for comparison/debugging
- Formula: `y = (P_T+H / P_T - 1) - (P_b,T+H / P_b,T - 1)` (no dividends)
- NOT recommended for production use

**Label Alignment (matches cutoff policy):**
- Entry: price(T close) — 4:00pm ET cutoff
- Exit: price(T+H close) — H trading days forward
- Dividends: ex-dates in (T, T+H] (exclusive of entry, inclusive of exit)
- Benchmark: same dates as stock
- Calendar: Use `TradingCalendarImpl` for trading day arithmetic

**Label Availability Rule (PIT-safe):**
- Labels mature at T+H close
- Dividends use ex-date (conservative - avoids forward-looking bias)
- During training/eval: filter by `asof >= T+H close`
- Labels table supports walk-forward + purging/embargo from day 1

**Benchmark Handling (HARD POLICY):**
- **Preferred:** Stock TR vs Benchmark TR (total return for both)
- **Fallback:** Stock TR vs Benchmark price return (if benchmark dividends unavailable)
- **Monitoring Rule:**
  - Log every fallback occurrence with ticker, date, horizon
  - Count fallback rate: `n_fallback / n_total_labels`
  - **Threshold:** Fallback must be <1% of labels
  - **Action if threshold exceeded:** Re-run with alternative benchmark dividend source
- **Rationale:** Even graceful fallback can create regime-dependent bias (ETF distributions matter in some windows)
- **Implementation:** `LabelGenerator._calculate_dividend_yield()` logs fallback, `_log_label_composition()` reports fallback rate

**Storage:**
- Same DuckDB pattern as features
- Keys: `stable_id`, `ticker`, `date`, `horizon`
- Values: `excess_return`, `stock_return`, `benchmark_return`, `stock_dividend_yield`, `benchmark_dividend_yield`
- Metadata: `label_matured_at`, `benchmark_ticker`, `label_version`

**Implementation Tasks:**
- [x] Lock return definition (v2: total return with dividends, DEFAULT)
- [x] Implement forward excess return calculation vs benchmark
- [x] Add dividend fetching and caching for both stock and benchmark
- [x] Create label generator for 20/60/90 trading day horizons
- [x] Store labels in DuckDB with maturity timestamps and version flag
- [x] Add tests for label correctness and PIT safety (9/9 tests pass, including v1 vs v2)
- [x] Backward compatibility: v1 available via flag

**Files:** `src/features/labels.py`
**Tests:** 9/9 passed in `tests/test_labels.py` (including v1 vs v2 comparison)
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

**⚠️ Pipeline Status:** Code implemented ✅, DuckDB wiring pending ⏳ (pre-Chapter-8 task)

---

**⚠️ MICROSTRUCTURE CAVEAT: Earnings Gap Timing**

Since we rebalance on the **1st of each month**, stocks with earnings on the **2nd-5th** will experience gap moves within our forward horizon. This creates two scenarios:

1. **Signal (Good):** If our model learns to predict surprise direction via `last_surprise_pct`, `surprise_streak`, etc., earnings gaps are exploitable alpha.
2. **Noise (Bad):** If our model does NOT predict surprise direction, earnings gaps are lottery tickets that contaminate our IC.

**Current Status:**
- ⚠️ **Chapter 7 frozen baseline (`tabular_lgb`) does NOT include earnings features**
- The frozen baseline uses only: momentum, volatility, drawdown, liquidity, relative strength, beta
- Earnings features are implemented but will be added to models starting **Chapter 8+**

**Future Mitigation (Chapter 8+):**
- Add earnings features (`days_to_earnings`, `surprise_streak`, etc.) to model feature list
- IC stratification by `days_to_earnings` (0-5d vs 6-21d vs 22-90d)
- Earnings-adjusted labels (replace gaps with market return, measure IC delta)
- Feature importance analysis for earnings features
- See `EARNINGS_GAP_ANALYSIS.md` for full diagnostic protocol

**Acceptance Threshold:**
- If earnings contamination > 0.01 IC, implement mitigation (filtering or adjustment)
- If contamination < 0.01 IC, document as "handled via features"

---

**⚠️ FEATURE STORE GAP (Pre-Chapter-8 Task)**

**Status Summary:**
- ✅ **Feature code implemented:** All 50 features exist in `src/features/`
- ⚠️ **DuckDB pipeline partial:** Only 7 features wired into `scripts/build_features_duckdb.py`
- ✅ **Chapter 7 frozen baseline unaffected:** Uses only 13 features (subset of 50)

**What's in DuckDB now:** `mom_1m`, `mom_3m`, `mom_6m`, `mom_12m`, `adv_20d`, `vol_20d`, `vol_60d`

**What needs to be added (43 features):**
- Price/Volume: 7 more (vol_of_vol, drawdown, rel_strength, beta)
- Events/Earnings: 12 features (days_to_earnings, surprise_streak, etc.)
- Regime/Macro: 15 features (VIX, market regime, sector rotation)
- Fundamentals: 7 features (P/E, P/S, margins - requires FMP Premium)
- Missingness: 2 features (coverage_pct, is_new_stock)

**Action Required Before Chapter 8:**
Expand `scripts/build_features_duckdb.py` to call all feature generators from `src/features/`. See ROADMAP.md "Pre-Chapter-8 Feature Expansion" section for priority order.

**Available Now for Models:**
- `DEFAULT_TABULAR_FEATURES` (13 features) - Chapter 7 frozen baseline
- `EXTENDED_TABULAR_FEATURES` (50 features) - Chapter 8+ models (once DuckDB expanded)

---

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
**⚠️ Pipeline Status:** Code implemented ✅, DuckDB wiring pending ⏳ (pre-Chapter-8 task)

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

**5.7 Feature Hygiene & Redundancy Control ✅ COMPLETE**

| Component | Status | Description |
|-----------|--------|-------------|
| Cross-sectional standardization | ✅ | z-score and rank transforms per date |
| Rolling Spearman correlation | ✅ | Correlation matrix computation |
| Feature clustering | ✅ | Hierarchical clustering to identify blocks |
| VIF diagnostics | ✅ | Variance Inflation Factor (diagnostic, not filter) |
| IC stability analysis | ✅ | Rolling IC with sign consistency tracking |
| Hygiene report generation | ✅ | Comprehensive report output |

**Key Classes:**
- `FeatureHygiene`: Main hygiene analysis class
- `FeatureBlock`: Cluster of correlated features
- `ICStabilityResult`: IC mean, std, sign consistency
- `VIFResult`: VIF value + high flag

**Key Principle:** A feature with IC 0.04 once and −0.01 later is WORSE than IC 0.02 stable forever.

**Usage:**
```python
from src.features.hygiene import FeatureHygiene

hygiene = FeatureHygiene()

# Standardize features
std_df = hygiene.standardize_cross_sectional(features_df, method="zscore")

# Identify correlated feature blocks
blocks = hygiene.identify_feature_blocks(features_df, threshold=0.7)

# VIF diagnostics
vif_results = hygiene.compute_vif(features_df)

# IC stability (requires labels)
ic_results = hygiene.compute_ic_stability(features_df, labels_df)

# Full report
report = hygiene.generate_hygiene_report(features_df, labels_df=labels_df)
print(report)
```

**Files:** `src/features/hygiene.py`
**Tests:** 9/9 passed in `tests/test_hygiene.py`

**5.8 Feature Neutralization ✅ COMPLETE**

| Component | Status | Description |
|-----------|--------|-------------|
| Cross-sectional neutralization | ✅ | Remove exposures via OLS/Ridge |
| Sector-neutral IC | ✅ | IC after removing sector effects |
| Beta-neutral IC | ✅ | IC after removing market beta |
| Sector+Beta neutral IC | ✅ | IC after removing both |
| Delta (Δ) reporting | ✅ | neutral_IC - raw_IC for interpretation |

**Purpose:** For diagnostics ONLY (not training). Reveals WHERE alpha comes from:
- Large negative Δ_sector → feature was mostly sector rotation
- Large negative Δ_beta → feature was mostly market exposure  
- Small Δ → alpha is genuinely stock-specific

**Design Choices:**
- Neutralize FEATURE (not label) before computing IC
- Cross-sectional per date (PIT-safe, sectors as-of date T)
- Reuses beta_252d from price_features.py (consistent definition)
- Market-neutral = beta-neutral (linear market exposure removed)

**Usage:**
```python
from src.features.neutralization import compute_neutralized_ic, neutralization_report

# Single feature
result = compute_neutralized_ic(
    features_df=features_df,
    labels_df=labels_df,
    feature_col="mom_1m",
    sector_col="sector",
    beta_col="beta_252d",
)

print(f"Raw IC: {result.raw_ic:.3f}")
print(f"Sector-neutral IC: {result.sector_neutral_ic:.3f} (Δ={result.delta_sector:+.3f})")
print(f"Beta-neutral IC: {result.beta_neutral_ic:.3f} (Δ={result.delta_beta:+.3f})")

# Full report for multiple features
results = neutralization_report(
    features_df=features_df,
    labels_df=labels_df,
    feature_cols=["mom_1m", "mom_3m", "pe_vs_sector"],
)
```

**Files:** `src/features/neutralization.py`
**Tests:** 9/9 passed in `tests/test_neutralization.py`

**Optional Paid-Tier Features:**
- [ ] Expectations: Revisions momentum, estimate dispersion, guidance
- [ ] Options: IV level/skew, implied move, IV percentile (requires paid data)
- [ ] Positioning: Short interest, ETF flows, 13F shifts (requires paid data)

---

#### Testing & Validation Requirements
- [x] Unit tests for each feature block (5.1-5.8 all have tests, 84/84 passed)
- [x] **PIT violation scanner on all features** (src/features/pit_scanner.py, 0 CRITICAL violations, enforced in CI)
- [x] **Univariate IC ≥ 0.03 check - IMPLEMENTED** (FeatureHygiene.compute_ic_stability ready, **to be executed in Chapter 6**)
- [x] **IC stability tool - IMPLEMENTED** (FeatureHygiene.compute_ic_stability ready, **to be executed in Chapter 6**)
- [x] Feature coverage > 95% (post-masking) - IMPLEMENTED (MissingnessTracker.compute_coverage_stats)
- [x] Redundancy documentation - IMPLEMENTED (FeatureHygiene.identify_feature_blocks)

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

#### Feature Pipeline Status

**Code Implementation:** ✅ COMPLETE (50 features, 84/84 tests passing)

**DuckDB Pipeline:** ⚠️ PARTIAL (7 of 50 features wired)

| Category | Features | In Code | In DuckDB | Status |
|----------|----------|---------|-----------|--------|
| Price/Volume | 14 | ✅ | 7 | ⚠️ 7 missing |
| Fundamentals | 7 | ✅ | 0 | ⚠️ Needs wiring + FMP Premium |
| Events/Earnings | 12 | ✅ | 0 | ⚠️ Needs wiring (critical for earnings gaps) |
| Regime/Macro | 15 | ✅ | 0 | ⚠️ Needs wiring |
| Missingness | 2 | ✅ | 0 | ⚠️ Needs wiring |
| **Total** | **50** | **50** | **7** | **43 pending** |

**Pre-Chapter-8 Task:** Expand `scripts/build_features_duckdb.py` to wire all feature generators. See ROADMAP.md "Pre-Chapter-8: Feature Store Expansion" for detailed checklist.

### Chapter 6: Evaluation Realism ✅ CLOSED & FROZEN

---

## 🔒 FREEZE STATUS

**Status:** CLOSED & FROZEN (December 30, 2025)  
**Tests:** 413/413 passing  
**Commits:** `18bad8a` (code fixes) + `7e6fa3a` (artifacts freeze)  
**Reference Doc:** `CHAPTER_6_FREEZE.md`  
**Baseline Floor:** Best RankIC per horizon (20d: 0.0283, 60d: 0.0392, 90d: 0.0169)

**Frozen Artifacts (tracked in git):**
- `evaluation_outputs/chapter6_closure_real/` - Baseline reference artifacts (IMMUTABLE)
- `BASELINE_FLOOR.json` - Metrics to beat for Chapter 7+
- `BASELINE_REFERENCE.md` - Usage instructions
- `CLOSURE_MANIFEST.json` - Commit hash, data hash, environment

**What This Means:**
- ✅ Chapter 6 evaluation infrastructure is COMPLETE and may not be modified
- ✅ Baseline reference is FROZEN and is the immutable comparison anchor
- ✅ Chapter 7+ models must use this frozen pipeline and beat the frozen baseline floor
- ⚠️ Any changes to evaluation definitions require a new version and re-freeze

---

## DEFINITION LOCK — SINGLE SOURCE OF TRUTH ✅ IMPLEMENTED

**Canonical File:** `src/evaluation/definitions.py`

All evaluation code MUST reference this file for time conventions, not define their own.
This prevents silent interpretation drift that can invalidate results.

### Time Conventions (LOCKED)

| Parameter | Value | Unit | Rationale |
|-----------|-------|------|-----------|
| **Horizons** | 20, 60, 90 | TRADING DAYS | Not calendar days |
| **Embargo** | 90 | TRADING DAYS | Must be ≥ max horizon |
| **Rebalance** | 1st of month/quarter | Calendar day | First trading day approximation |
| **Pricing** | Close-to-close | - | Labels use closing prices |
| **Market Close** | 4:00 PM ET | Time | UTC conversion done in code |

### Maturity Rule (LOCKED)

**Canonical:** `label_matured_at <= cutoff_close_utc`

- Cutoff is market close time (4 PM ET) converted to UTC
- Code uses `get_market_close_utc(date)` helper
- Naive datetimes are REJECTED (must be timezone-aware)

### Purging Rules (LOCKED)

| Rule | Applies To | Condition |
|------|------------|-----------|
| Train purge | Training labels | T + H (trading days) > train_end |
| Val purge | Validation labels | T - H (trading days) < train_end |
| Granularity | All | Per-row-per-horizon (NOT global) |

### End-of-Sample Eligibility (LOCKED)

- **Rule:** `all_horizons_must_be_valid`
- **Valid label:** exists + matured + required fields populated
- **Partial horizons:** NOT ALLOWED in Chapter 6

### Implementation Status

| Component | File | Tests | Status |
|-----------|------|-------|--------|
| Definitions | `src/evaluation/definitions.py` | 40 tests | ✅ |
| Walk-Forward | `src/evaluation/walk_forward.py` | 25 tests | ✅ |
| Sanity Checks | `src/evaluation/sanity_checks.py` | 16 tests | ✅ |
| Metrics | `src/evaluation/metrics.py` | 30 tests | ✅ |
| Costs | `src/evaluation/costs.py` | 28 tests | ✅ |

**Total:** 139/139 tests passing (includes 16 tests for sanity_checks not in evaluation suite)

---

## 6.0 Pre-Implementation Sanity Checks

**BEFORE writing any Chapter 6 code, complete these two validation steps:**

### ✅ Sanity Check 1: Manual IC vs Qlib IC Parity Test

**Purpose:** Ensure adapter/indexing is correct before generating hundreds of evaluation reports.

**Test Protocol:**
```python
# One fold, one horizon, same predictions
fold = "2023-Q1"
horizon = 20

# Manual RankIC calculation
manual_rankic = df.groupby("date").apply(
    lambda x: spearmanr(x["prediction"], x["label"])[0]
).median()

# Qlib RankIC calculation
qlib_df = adapter.to_qlib_format(predictions, labels)
qlib_rankic = qlib.evaluate(qlib_df)["IC"].median()

# STOP if they don't match
assert abs(manual_rankic - qlib_rankic) < 0.001, "IC mismatch - fix adapter!"
```

**Acceptance:** Manual and Qlib RankIC must agree to 3 decimal places.

**If they don't match → STOP immediately:**
- Check MultiIndex formatting (datetime, instrument)
- Check date alignment (T vs T+H)
- Check for missing data handling differences
- Check for sign flips (prediction vs label)

### ✅ Sanity Check 2: Experiment Naming Convention

**Purpose:** Prevent chaos when Recorder usage explodes across hundreds of experiments.

**Convention (LOCK THIS IN NOW):**
```
exp = ai_forecaster/
      horizon={20,60,90}/
      model={kronos_v0, fintext_v0, tabular_lgb, baseline_mom12m}/
      labels={v1_priceonly, v2_totalreturn}/
      fold={01, 02, ..., 40}/
```

**Example paths:**
```
ai_forecaster/horizon=20/model=kronos_v0/labels=v2/fold=03
ai_forecaster/horizon=60/model=baseline_mom12m/labels=v2/fold=12
ai_forecaster/horizon=90/model=tabular_lgb/labels=v2/fold=25
```

**Why this matters:**
- Enables bulk queries: "all Kronos results for horizon=20"
- Prevents accidental overwrites
- Makes debugging obvious (path tells you what broke)
- Enables automated comparison scripts

**Implementation:**
```python
from qlib.workflow import R

exp_name = f"ai_forecaster/horizon={horizon}/model={model_name}/labels={label_version}/fold={fold_id:02d}"
with R.start(experiment_name=exp_name):
    recorder.save_objects(predictions=pred_df)
```

---

## 6.1 Evaluation Strategy (Lock Before Coding)

### Rebalance Cadence

**Primary: Monthly**
- Cleanest match to horizons (20d ≈ 1 month)
- Enough samples for statistical significance (~110 points over 9.5 years)
- Overlapping holdings for 60/90d are realistic (not a bug)
- Monthly is evaluation of ranking signals, not execution simulation

**Secondary: Quarterly (robustness check only)**
- Report as supplementary slice
- Fewer points → noisier, easier to fool yourself
- Use to confirm results aren't regime-specific artifacts

**Rebalance date:** First trading day of each month (NYSE calendar)

**Rationale:**
- 20d horizon + monthly rebalance = one horizon per rebalance (natural)
- 60/90d horizons + monthly rebalance = overlapping positions (realistic for Chapter 6)
- Quarterly would give too few observations (<40 points)

### Evaluation Date Range

**Principles:**
1. **PIT-safe end date:** Last `as_of_date` with `label_matured_at` available for 90d horizon
2. **Regime diversity:** Include multiple market regimes (pre-COVID, COVID, 2022 drawdown, 2023-25 AI rally)

**Locked Parameters:**
```python
EVAL_START = "2016-01-01"  # Earliest date with reliable fundamentals + universe snapshots
EVAL_END = "2025-06-30"    # Conservative: guarantees 90d label maturity

# Dynamic alternative (if needed):
# EVAL_END = max_label_matured_at - pd.Timedelta(days=90)  # computed from labels table
```

**Why 2016-01-01?**
- FMP fundamentals become more reliable
- Universe snapshots (stable_id) have sufficient coverage
- Captures ~9.5 years = enough for regime diversity

**Why 2025-06-30?**
- Guarantees all 90d labels are mature (no future leakage)
- Includes 2023-25 AI rally (critical for AI stock forecaster)
- Conservative: leaves buffer for label maturity

**Result:** ~110 monthly rebalance points, multiple regimes, zero PIT violations.

### Baselines (Models to Beat)

**3 baselines to beat:**

**Baseline A: `mom_12m` (primary naive baseline)**
- Rank stocks by 12-month momentum feature
- Standard factor baseline (embarrassing if we can't beat this)
- Already in feature set (`src/features/price_features.py`)
- **Gate:** If Kronos/FinText can't beat this, something is wrong

**Baseline B: `momentum_composite` (stronger but transparent)**
- Equal-weight rank average: `(mom_1m + mom_3m + mom_6m + mom_12m) / 4`
- Often materially stronger than single-horizon momentum
- Still no ML, still transparent
- **Gate:** Realistic bar for "is ML worth it?"

**Baseline C: `short_term_strength` (diagnostic baseline)**
- Rank by `mom_1m` or `rel_strength_1m`
- Often weak in mean-reversion regimes
- **Purpose:** Diagnostic for horizon sensitivity
  - If it beats you in certain windows → evaluation is telling you something
  - If you can't beat it → may be penalizing longer-horizon signals

**+ 1 sanity baseline (not a target):**
- **`naive_random`**: Deterministic random scores (sanity check: RankIC ≈ 0, confirms no systematic bias in evaluation pipeline)

**Critical Guardrail:**
All baselines run through **identical pipeline:**
- Same universe snapshots (stable_id)
- Same missingness handling
- Same neutralization setting (raw or sector/beta-neutral)
- Same cost diagnostic treatment (6.4)
- Same purging/embargo
- Same walk-forward splits

**Why these 3?**
- Cover naive (A), strong (B), diagnostic (C)
- All use existing features (no new dependencies)
- Prevent "baseline shopping" (temptation to add weak baselines)

### Top-K Definition (Churn & Hit Rate)

**Primary: Top-10**
- Matches "forecaster" product narrative (shortlist)
- Churn is interpretable ("did the list change?")
- Stable enough for month-to-month comparison
- Aligns with concentrated portfolio construction

**Secondary: Top-20 (fixed K)**
- Robustness check for universe size changes
- Simpler than percentage-based (no dynamic computation)
- Still interpretable

**Alternative Secondary: Top-10% (percentage-based)**
- More robust to universe size changes over time
- Use this if universe size varies significantly (e.g., 80-150 stocks)

**Locked Definitions:**

**Churn (overlap-based):**
```python
# Jaccard similarity between consecutive Top-K
churn = 1 - len(set(top_k_t) & set(top_k_t_minus_1)) / len(set(top_k_t) | set(top_k_t_minus_1))

# Or simpler: % retained
retention = len(set(top_k_t) & set(top_k_t_minus_1)) / K
churn = 1 - retention
```

**Hit Rate (excess return):**
```python
# Definition: % of Top-K portfolios with excess return > 0 over horizon
hit = (top_k_portfolio_return_t_to_t_plus_H - benchmark_return_t_to_t_plus_H) > 0
hit_rate = hits / total_rebalances
```

**Alternative:** Top-K beats equal-weight universe (use if preferred for interpretability)

**Target Metrics:**
- Churn: < 30% month-over-month (exploitable without excessive turnover)
- Hit Rate: > 55% (better than coin flip, realistic for noisy markets)

---

## 6.2 Qlib Integration (Shadow Evaluator)

**Qlib Integration (Shadow Evaluator):**
- [x] **Qlib as optional evaluator** (NOT replacing Chapters 1-5 infrastructure)
- Our system = source of truth (universe, PIT, features, labels)
- Qlib receives: predictions + realized labels + optional groups (sector, liquidity)
- Qlib generates: standardized factor evaluation + backtest-style reports
- **Benefits:**
  - 6.3 Metrics: IC/RankIC, monthly IC, quintile analysis, autocorrelation (built-in plots)
  - 6.5 Reporting: Cumulative returns, group analysis, long-short distribution
  - 6.4 Cost realism: Backtest analytics as second opinion ("does alpha survive?")
  - Experiment tracking: Recorder system for walk-forward folds
- **Reference:** [Qlib Documentation](https://qlib.readthedocs.io/en/latest/), [GitHub](https://github.com/microsoft/qlib)

---

## 6.1.2 Implementation Status ✅ PHASE 0, 1, 1.5 COMPLETE

**Implemented Files:**
- `src/evaluation/__init__.py` - Module initialization with all exports
- `src/evaluation/definitions.py` - **CANONICAL DEFINITIONS (single source of truth)**
- `src/evaluation/walk_forward.py` - **WalkForwardSplitter with ENFORCED purging & embargo**
- `src/evaluation/sanity_checks.py` - IC parity test & experiment naming
- `tests/test_definitions.py` - 40 tests for canonical definitions
- `tests/test_walk_forward.py` - 25 tests verifying purging and embargo
- `tests/test_sanity_checks.py` - 16 tests verifying sanity checks

**Total: 65/65 tests passing**

### ✅ Phase 1.5: Definition Lock (ENFORCED)

All time conventions are now locked in `src/evaluation/definitions.py`:

```python
from src.evaluation import (
    TIME_CONVENTIONS,     # horizons, embargo, pricing, maturity rule
    EMBARGO_RULES,        # per-row-per-horizon purging rules
    ELIGIBILITY_RULES,    # all-horizons-valid requirement
    trading_days_to_calendar_days,  # CONSERVATIVE conversion
    get_market_close_utc,           # UTC datetime for maturity checks
)

# All of these are FROZEN (cannot be modified at runtime)
assert TIME_CONVENTIONS.embargo_trading_days == 90  # TRADING DAYS
assert EMBARGO_RULES.purging_granularity == "per_row_per_horizon"
assert ELIGIBILITY_RULES.allow_partial_horizons == False
```

### ✅ WalkForwardSplitter (ENFORCED, not just documented)

```python
from src.evaluation import WalkForwardSplitter

splitter = WalkForwardSplitter(
    eval_start=date(2016, 1, 1),
    eval_end=date(2025, 6, 30),
    rebalance_freq="monthly",
    embargo_trading_days=90,  # TRADING DAYS (raises ValueError if < 90)
)

folds = splitter.generate_folds(
    labels_df, 
    horizons=[20, 60, 90],
    require_all_horizons=True  # End-of-sample eligibility enforced
)

for fold in folds:
    train_df = fold.filter_labels(labels_df, split="train")
    val_df = fold.filter_labels(labels_df, split="val")
    # Guarantees:
    # ✅ Embargo: 90+ TRADING DAYS between train_end and val_start
    # ✅ Purging: Per-row-per-horizon (not global)
    # ✅ Maturity: label_matured_at <= cutoff_close_utc
    # ✅ Eligibility: All horizons valid (no partial)
```

**CRITICAL Anti-Leakage Enforcement:**

1. **Embargo (TRADING DAYS, not calendar):**
   - `embargo_trading_days` parameter explicit
   - Converted conservatively to calendar days: `trading_days * 1.45 + 1`
   - Raises `ValueError` if < 90

2. **Purging (Per-Row-Per-Horizon):**
   - Each (date, ticker, horizon) evaluated independently
   - Train rule: Purge if T + H > train_end
   - Val rule: Purge if T - H < train_end
   - NOT a global rule that "happens to work"

3. **Maturity (UTC Market Close):**
   - Uses `get_market_close_utc(date)` for cutoff
   - Naive datetimes REJECTED
   - Timezone-aware comparison enforced

4. **End-of-Sample (All Horizons Valid):**
   - `require_all_horizons=True` by default
   - Dates missing any horizon are EXCLUDED
   - Prevents silent horizon dropout

### ✅ Sanity Checks (6.0)
```python
from src.evaluation import verify_ic_parity, ExperimentNameBuilder

# Sanity Check 1: IC Parity
result = verify_ic_parity(predictions, labels, qlib_ic=0.045, tolerance=0.001)
# Raises ValueError if |manual_ic - qlib_ic| > tolerance

# Sanity Check 2: Experiment Naming
builder = ExperimentNameBuilder(
    horizon=20,
    model="kronos_v0",
    label_version="v2",
    fold_id="fold_01"
)
exp_name = builder.build()
# Returns: "ai_forecaster/horizon=20/model=kronos_v0/labels=v2_totalreturn/fold=fold_01"
```

**Test Results:**
- ✅ 40 definition tests pass (constants, frozen dataclasses, UTC conversions)
- ✅ 25 walk-forward tests pass (embargo, purging, maturity, eligibility)
- ✅ 16 sanity check tests pass (IC parity, experiment naming)

---

## 6.3 Metrics Implementation ✅ COMPLETE

### Canonical Evaluation Data Contract

All models and baselines MUST produce standardized `EvaluationRow` format:

**Required Fields:**
- `as_of_date`, `ticker`, `stable_id`, `horizon`, `fold_id`
- `score` (higher = better)
- `excess_return` (v2 total return vs benchmark)

**Rules (LOCKED):**
1. Score direction: Higher = better
2. Duplicates: NOT ALLOWED per (as_of_date, stable_id, horizon)
3. Missing score/return: Row DROPPED (logged)
4. Tie-breaking: Deterministic via stable_id sorting
5. Minimum cross-section: 10 stocks per date

### Core Metrics Implemented

| Metric | Formula | Aggregation |
|--------|---------|-------------|
| **RankIC** | Spearman(rank(score), rank(return)) | Per-date → Median |
| **Quintile Spread** | mean(top 20%) - mean(bottom 20%) | Per-date → Median |
| **Top-K Hit Rate** | % with excess_return > 0 | Per-date → Median |
| **Top-K Avg ER** | mean(excess_return of Top-K) | Per-date → Median |
| **Churn** | 1 - (overlap / K) | Consecutive dates → Median |

### Churn (ENFORCED Rules)

- Uses `stable_id` (not ticker) to avoid rename noise
- Computed only on consecutive dates within fold
- NOT across fold boundaries
- Top-10 (primary), Top-20 (secondary)
- Target: < 30%

### Regime Slicing (LOCKED Definitions)

Using existing regime features only:

```python
REGIME_DEFINITIONS = {
    "vix_percentile_252d": {
        "low": <= 33, "mid": (33, 67], "high": > 67
    },
    "market_regime": {
        "bull": market_return_20d > 0,
        "bear": market_return_20d <= 0
    }
}
```

### Implementation Status

| Component | Lines | Tests | Status |
|-----------|-------|-------|--------|
| `src/evaluation/metrics.py` | 606 | 30 | ✅ |

**Test Coverage:** 30/30 passing
- Tie-breaking deterministic ✅
- Perfect correlation → RankIC ≈ 1 ✅
- Random scores → RankIC ≈ 0 ✅
- Churn matches hand-calculated ✅
- Regime slicing preserves counts ✅

---

## 6.4 Cost Realism ✅ COMPLETE

**Purpose:** Diagnostic overlay to answer "does alpha survive trading costs?"

### Philosophy

- Costs affect portfolio metrics (Top-K, spreads), NOT RankIC
- RankIC measures ranking skill; costs measure implementability
- Use sensitivity bands (low/base/high), not one "true" value
- Answer "does it survive?" without pretending we know exact slippage

### Locked Trading Assumptions

```python
from src.evaluation import TRADING_ASSUMPTIONS

# Portfolio Definition
K = 10 (primary), 20 (secondary)  # Top-K equal-weight, long-only
Rebalance: on as-of close (first trading day of month/quarter)
Turnover: weight changes between consecutive rebalances (stable_id)

# AUM Assumptions (for ADV scaling)
AUM = $1M (primary), $10M (secondary)

# Cost Model
Base Cost: 20 bps round-trip (10 bps entry + 10 bps exit)
  └─ Always applied to liquid large caps

Slippage: c * sqrt(participation_rate)
  └─ participation_rate = trade_value / adv_dollars
  └─ c = {5 (low), 10 (base), 20 (high)}
  └─ Floor: 0 bps, Cap: 500 bps

Missing ADV: 100 bps penalty (conservative)
```

### Cost Model Function

**Square-root impact model** (standard in market microstructure):

```python
slippage_bps = c * sqrt(trade_value / adv_dollars)
total_cost_bps = base_cost_bps + slippage_bps
```

**Properties:**
- Monotone: Higher participation → higher cost
- Concave: Diminishing marginal impact
- Empirically validated
- Simple and transparent

### Net-of-Cost Metrics

```python
from src.evaluation import compute_portfolio_costs, compute_net_metrics

# Compute costs for one rebalance date
costs = compute_portfolio_costs(
    df=eval_df,  # EvaluationRow format
    k=10,
    aum=1_000_000,
    slippage_coef=10.0,  # Base scenario
    prev_portfolio=None  # For turnover tracking
)

# Net metrics
net = compute_net_metrics(
    gross_metrics={"avg_excess_return": 0.05},
    cost_pct=costs["total_cost_pct"]
)

# Result
net["gross_avg_er"]   # 5.0%
net["cost_pct"]        # 0.2% (20 bps)
net["net_avg_er"]      # 4.8%
net["alpha_survives"]  # True (net > 0)
```

### Sensitivity Analysis

Run each fold/horizon under 4 scenarios:

| Scenario | Base Cost | Slippage Coef | Description |
|----------|-----------|---------------|-------------|
| base_only | 20 bps | 0 | Base cost only (no slippage) |
| low_slippage | 20 bps | 5 | Optimistic slippage |
| base_slippage | 20 bps | 10 | Expected slippage |
| high_slippage | 20 bps | 20 | Conservative slippage |

**Report per horizon:**
- % folds with net Top-K avg ER > 0
- Median net Top-K avg ER
- Where it breaks (small ADV names, specific regimes)

### Enforced Invariants (Tests)

✅ Zero turnover → minimal costs
✅ Lower ADV → higher costs (monotonicity)
✅ Higher AUM → higher costs (monotonicity)
✅ Deterministic results (same inputs → same outputs)
✅ ADV missing → conservative penalty

### Implementation Status

| Component | Lines | Tests | Status |
|-----------|-------|-------|--------|
| `src/evaluation/costs.py` | 514 | 28 | ✅ |

**Test Coverage:** 28/28 passing
- Frozen trading assumptions ✅
- Square-root impact law ✅
- Monotonicity (ADV, AUM) ✅
- Turnover calculation ✅
- Net metrics computation ✅
- Determinism ✅

**Total Chapter 6 Tests:** 123/123 passing (40 definitions + 25 walk-forward + 30 metrics + 28 costs)

---

## 6.5 Stability Reports ✅ COMPLETE

**Purpose:** Pure consumer of 6.3 metrics + 6.4 costs → deterministic reporting artifacts.

**Philosophy:** NO new APIs, NO new features, NO new modeling. Just clean rendering.

### Locked Reporting Parameters

```python
from src.evaluation import STABILITY_THRESHOLDS

# IC Decay Thresholds
rapid_decay_threshold = 0.05  # Late IC drops > 5% vs early
noisy_threshold = 2.0  # IQR / median > 2.0

# Churn Thresholds
high_churn_threshold = 0.50  # >50% turnover = unstable

# Coverage Requirements
min_dates_per_bucket = 10  # Regime slices
min_names_per_date = 10  # Cross-section minimum

# Rolling Window
rolling_window = 6  # 6-month for monthly cadence
```

### Report Contract

**Inputs (must already exist):**
- Per-date RankIC series (from `evaluate_fold`)
- Per-date quintile spread series
- Per-date Top-K metrics + churn
- Regime-sliced summaries
- Cost overlays (turnover, gross vs net, sensitivity bands)

**Outputs (deterministic artifacts):**
- `tables/` (CSV): ic_decay_stats, regime_performance, churn_diagnostics, stability_scorecard
- `figures/` (PNG): ic_decay, regime_bars, churn_timeseries, churn_distribution
- `REPORT_SUMMARY.md`: Human-readable summary

### Components Implemented

**A) IC Decay Analysis**
```python
from src.evaluation import compute_ic_decay_stats, plot_ic_decay

# Compute early vs late stats
stats = compute_ic_decay_stats(per_date_metrics, metric_col="rankic")

# Results: early_median, late_median, decay, rapid_decay_flag, noisy_flag
```

**B) Regime-Conditional Performance**
```python
from src.evaluation import format_regime_performance, plot_regime_bars

# Format with coverage stats
formatted = format_regime_performance(regime_summaries)

# Add thin_slice_flag for insufficient coverage
```

**C) Churn Diagnostics**
```python
from src.evaluation import compute_churn_diagnostics

# Summary per fold/horizon/k
diag = compute_churn_diagnostics(churn_series)

# Results: churn_median, churn_p90, pct_high_churn, high_churn_flag
```

**D) Stability Scorecard**
```python
from src.evaluation import generate_stability_scorecard

# One-screen summary
scorecard = generate_stability_scorecard(
    fold_summaries,
    churn_diagnostics=churn_diag,
    cost_overlays=cost_overlays
)

# The thing you paste into your Chapter 6 writeup
```

**E) Full Report Generation**
```python
from src.evaluation import generate_stability_report, StabilityReportInputs

inputs = StabilityReportInputs(
    per_date_metrics=per_date_df,
    fold_summaries=fold_summary_df,
    regime_summaries=regime_df,
    churn_series=churn_df,
    cost_overlays=cost_df
)

outputs = generate_stability_report(
    inputs,
    experiment_name="kronos_v0_h20_m",
    output_dir=Path("reports")
)

# Outputs: tables/, figures/, REPORT_SUMMARY.md
```

### Enforced Invariants (Tests)

✅ **Determinism**: Same inputs (shuffled) → identical outputs
✅ **Fold boundaries**: Churn never bridges folds
✅ **Regime bucket integrity**: Totals sum correctly
✅ **No silent drops**: Exclusions are logged
✅ **Thin slice detection**: Flags insufficient coverage
✅ **Rapid decay detection**: Flags early vs late drops
✅ **High churn detection**: Flags unstable rankings

### Implementation Status

| Component | Lines | Tests | Status |
|-----------|-------|-------|--------|
| `src/evaluation/reports.py` | 768 | 24 | ✅ |

**Test Coverage:** 24/24 passing
- Frozen thresholds ✅
- IC decay statistics ✅
- Regime performance formatting ✅
- Churn diagnostics ✅
- Stability scorecard ✅
- Full report generation ✅
- Determinism enforcement ✅
- Fold boundary handling ✅
- Regime integrity checks ✅
- Exclusion logging ✅

**Total Chapter 6 Tests:** 147/147 passing (40 definitions + 25 walk-forward + 30 metrics + 28 costs + 24 reports)

---

## 6.6 Walk-Forward Tasks

**Phase 0, 1, 1.5, 2, 4, 5, 6: ✅ IMPLEMENTATION COMPLETE**

### Implementation Status (Code + Tests)
- [x] **Complete 6.0 sanity checks** (IC parity, experiment naming) ✅
- [x] **Implement walk-forward splitter** (purging & embargo ENFORCED) ✅
- [x] **Definition lock** (canonical time conventions) ✅
- [x] **Implement metrics** (RankIC, churn, hit rate, regime slicing) ✅
- [x] **Implement cost realism** (trading costs, slippage, net metrics) ✅
- [x] **Implement stability reports** (IC decay, regime performance, churn diagnostics) ✅
- [x] **Implement baselines** (mom_12m, momentum_composite, short_term_strength, naive_random) ✅
- [x] **Create end-to-end runner** (run_evaluation.py) ✅
- [x] **Create Qlib adapter** (shadow evaluator) ✅
- [x] **Write comprehensive tests** (269 evaluation tests, all passing) ✅

**Total Tests:** 413/413 passing (100%)

### Execution Status ✅ CLOSED & FROZEN
- [x] **Execute FULL_MODE baseline run** (2016-2025, monthly + quarterly, all 4 baselines) ✅
- [x] **Freeze baseline reference** (commit hash + output directory) ✅
- [x] **Produce baseline floor summary** (best baseline per horizon) ✅
- [x] **Verify sanity baseline** (naive_random RankIC ≈ 0) ✅

### Real Data Infrastructure ✅
- [x] **DuckDB feature store** (`data/features.duckdb`) - build from FMP
- [x] **Data loader** supports synthetic and DuckDB modes
- [x] **Closure script** defaults to real data, fails loudly if missing
- [x] **24 new tests** for DuckDB loader (no live FMP calls)

### How to Build Real Data Feature Store
```bash
# Set FMP API key (Premium required)
export FMP_KEYS="your_fmp_premium_api_key"

# Build DuckDB (downloads data, computes features/labels)
python scripts/build_features_duckdb.py

# Run real-data Chapter 6 closure
python scripts/run_chapter6_closure.py
# Outputs: evaluation_outputs/chapter6_closure_real/
```

### Why FULL_MODE Matters
The acceptance criteria (RankIC lift, net-positive folds, churn, regime robustness) can only be verified with actual numbers from a full run. The FULL_MODE baseline run becomes the **immovable reference** for all Chapter 7+ model comparisons.

```bash
# To execute FULL_MODE baseline run:
python -m src.evaluation.run_evaluation --mode full --output-dir evaluation_outputs

# Or programmatically:
for baseline in list_baselines():
    spec = ExperimentSpec.baseline(baseline, cadence="monthly")
    run_experiment(spec, features_df, Path("evaluation_outputs"), FULL_MODE)
```

### Chapter 6 Closure Checklist ✅ COMPLETE

- [x] FULL_MODE run completes without errors (REAL DuckDB data)
- [x] All 4 baselines produce outputs for all 3 horizons (mom_12m, momentum_composite, short_term_strength, naive_random)
- [x] stability_scorecard.csv generated for each baseline (8 runs: 4 baselines x 2 cadences)
- [x] cost_overlays.csv includes all 4 scenarios (base_only, low/base/high_slippage)
- [x] BASELINE_REFERENCE.md documents baseline floor metrics + usage instructions
- [x] Commit hash frozen as reference point (18bad8a + 7e6fa3a)
- [x] Outputs directory tracked in git (evaluation_outputs/chapter6_closure_real/)
- [x] Data snapshot frozen (data_hash: 5723d4c88b8ecba1..., 192,307 rows, 2016-2025)
- [x] Sanity check passed (naive_random RankIC ≈ 0)
- [x] Qlib shadow evaluation completed and verified
- [x] Critical bugs fixed (per-horizon metrics, baseline floor paths, fold date alignment, Qlib inputs)

**Freeze Date:** December 30, 2025  
**See:** `CHAPTER_6_FREEZE.md` for complete details

---

## Chapter 7: Baselines to Beat

**Status:** ✅ COMPLETE + FROZEN (tag: `chapter7-tabular-lgb-freeze`)

### 7.1-7.2: Factor Baselines ✅ COMPLETE (Frozen in Chapter 6)

The factor baselines are already implemented and frozen as part of Chapter 6:
- `mom_12m`: 12-month momentum (primary naive baseline)
- `momentum_composite`: Equal-weight average of mom_1m/3m/6m/12m
- `short_term_strength`: 1-month momentum (diagnostic)
- `naive_random`: Deterministic random (sanity check)

**Frozen Baseline Floor:** See `evaluation_outputs/chapter6_closure_real/BASELINE_FLOOR.json`

### 7.3: Tabular ML Baseline ✅ COMPLETE

**Implementation:** `src/evaluation/baselines.py` (ML baselines section)

**Baseline:** `tabular_lgb`
- **Model:** LightGBM Regressor (predicts excess return; higher = better)
- **Training:** Per-fold using walk-forward splits (purging/embargo/maturity)
- **Time Decay:** Exponential sample weighting (half-life = 252 trading days)
- **Features:** Momentum (1m/3m/6m/12m), volatility (20d/60d), drawdown, liquidity (ADV), relative strength, beta
- **Horizon-Specific:** Separate models trained for 20/60/90d horizons
- **Deterministic:** Fixed `random_state=42`, no hyperparameter tuning in baseline

**Hyperparameters (Fixed):**
```python
n_estimators=100
learning_rate=0.05
max_depth=5
num_leaves=31
min_child_samples=20
```

**Usage:**
```python
from src.evaluation import ExperimentSpec, run_experiment, FULL_MODE
from src.evaluation.data_loader import load_features_for_evaluation

# 1) LOADER: source + date range + horizons (cadence NOT here)
features_df, metadata = load_features_for_evaluation(
    source="duckdb",
    db_path="data/features.duckdb",
    eval_start=date(2016, 1, 1),
    eval_end=date(2025, 6, 30),
    horizons=[20, 60, 90],
)

# 2) RUNNER: cadence belongs to ExperimentSpec
spec = ExperimentSpec.baseline("tabular_lgb", cadence="monthly")
results = run_experiment(spec, features_df, output_dir, FULL_MODE)
```

> **Note:** `cadence` controls walk-forward fold frequency (monthly vs quarterly) and belongs to `ExperimentSpec`, not the loader. The loader only retrieves data; fold generation is the runner's responsibility.

**Tests:** `tests/test_ml_baselines.py` (13 tests)
- Registration and listing
- Time-decay weighting
- Training and prediction
- Determinism (same inputs → same outputs)
- Integration with evaluation pipeline
- Guardrails (train/val split enforcement)

**Total Tests:** 426/426 passing (413 existing + 13 new)

### 7.4: Baseline Gates ✅ IMPLEMENTED

**Factor Gate:** `median_RankIC(best_factor) ≥ 0.02`  
**ML Gate:** `median_RankIC(tabular_lgb) ≥ 0.05`  
**TSFM Rule (Chapters 8-12):** Must beat tuned ML baseline

**Implementation:** `src/evaluation/run_evaluation.py` (`compute_acceptance_verdict`)

**Acceptance Criteria (vs Frozen Baseline Floor):**
1. **RankIC Lift:** Median RankIC ≥ baseline + 0.02
2. **Net-of-Cost:** % positive folds ≥ baseline + 10pp (relative gate; frozen floor: 5.8%/25.1%/40.1%)
3. **Churn:** Top-10 median churn < 0.30
4. **Regime Robustness:** No fold with negative median RankIC

### 7.5: FULL_MODE Execution Script ✅ COMPLETE

**Implementation:** `scripts/run_chapter7_tabular_lgb.py`

**What It Does:**
- Loads REAL data from DuckDB feature store
- Runs `tabular_lgb` baseline for monthly + quarterly cadences (FULL_MODE)
- Produces cost overlays, stability reports, churn metrics
- Compares vs frozen Chapter 6 baseline floor
- Writes `BASELINE_REFERENCE.md` with performance summary
- Does NOT modify Chapter 6 frozen artifacts

**Usage:**
```bash
# Build DuckDB feature store (if not already built)
python scripts/build_features_duckdb.py --auto-normalize-splits

# Run tabular_lgb baseline (FULL MODE)
# This will take significant time (trains model per fold per horizon)
python scripts/run_chapter7_tabular_lgb.py

# Smoke test (quick validation, 1 fold per cadence)
python scripts/run_chapter7_tabular_lgb.py --smoke
```

**Output Directory (FROZEN):**
```
evaluation_outputs/chapter7_tabular_lgb_full/  # FROZEN artifacts
├── monthly/
│   ├── eval_rows.parquet
│   ├── fold_summaries.csv
│   ├── cost_overlays.csv
│   ├── churn_series.csv
│   └── stability_report/
├── quarterly/
│   └── (same structure)
├── BASELINE_REFERENCE.md
├── BASELINE_FLOOR.json  # ML baseline floor for Chapter 8+
├── CLOSURE_MANIFEST.json
└── DATA_MANIFEST.json
```

**Tests:** `tests/test_chapter7_script.py` (3 tests, smoke mode)

### 7.6: ML Baseline Freeze ✅ FROZEN

**Freeze Tag:** `chapter7-tabular-lgb-freeze`  
**Commit:** `a264fd2a`  
**Data Hash:** `f3899b37cb9f34f1`

**Frozen ML Baseline Floor:**

| Horizon | tabular_lgb (Monthly) | Lift vs Factor Floor |
|---------|----------------------|----------------------|
| 20d | 0.1009 | **+0.0726** (+256%) |
| 60d | 0.1275 | **+0.0883** (+225%) |
| 90d | 0.1808 | **+0.1639** (+970%) |

**Gate Policy for Chapter 8+ (TSFM Models):**
- **RankIC:** Must beat `tabular_lgb` + 0.02
- **Cost Survival:** % positive folds ≥ `tabular_lgb` + 10pp
- **Churn:** Median Top-10 churn < 0.30

**Frozen Artifacts Location:** `evaluation_outputs/chapter7_tabular_lgb_full/`

---

## Chapter 7 Summary

**Status:** ✅ COMPLETE + FROZEN

**Deliverables:**
1. ✅ 7.1-7.2: Factor baselines (frozen in Chapter 6)
2. ✅ 7.3: `tabular_lgb` ML baseline implemented and tested
3. ✅ 7.4: Gating policy implemented (relative to frozen floor)
4. ✅ 7.5: FULL_MODE execution script executed
5. ✅ 7.6: ML baseline frozen with git tag

**Frozen Artifacts:**
- `evaluation_outputs/chapter6_closure_real/` - Factor baselines (immutable)
- `evaluation_outputs/chapter7_tabular_lgb_full/` - ML baseline (immutable)

**Total Tests:** 429/429 passing

**Next Chapter:** Chapter 8 (Kronos TSFM) - models must beat ML baseline floor

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

**Required in `.env`** (in repository root):
```env
# Financial Modeling Prep (required for main data)
# Premium key required for 30-year history and QQQ benchmark
FMP_KEYS=your_fmp_api_key

# Alpha Vantage (optional - for earnings calendar)
ALPHAVANTAGE_KEYS=your_alphavantage_key

# SEC EDGAR (optional - for precise filing timestamps)
SEC_CONTACT_EMAIL=your_email@domain.com
```

**Key Loading (auto-loaded from `.env`):**

Scripts automatically load `.env` from the repository root. No need to manually export variables.

**Key Precedence (highest to lowest):**
1. CLI argument: `--api-key YOUR_KEY`
2. Environment variable: `FMP_KEYS` (comma-separated, uses first)
3. Environment variable: `FMP_API_KEY` (fallback)
4. `.env` file in repo root (auto-loaded)

**Security:**
- Keys are NEVER logged
- Only the source used is logged (e.g., "Using FMP_KEYS")

**Get API Keys:**
- FMP: https://financialmodelingprep.com/developer/docs/pricing (Premium required)
- Alpha Vantage: https://www.alphavantage.co/support/#api-key
- SEC: No key needed, just provide contact email

### How to Build `data/features.duckdb`

The DuckDB feature store is required for real-data Chapter 6 closure:

```bash
# Option 1: Using .env (recommended)
# Put your key in .env file, then run:
python scripts/build_features_duckdb.py

# Option 2: Using CLI argument
python scripts/build_features_duckdb.py --api-key YOUR_FMP_KEY

# Option 3: Using environment variable
export FMP_KEYS='your_key'
python scripts/build_features_duckdb.py

# Dry run (validate setup without downloading)
python scripts/build_features_duckdb.py --dry-run
```

**Output:**
- `data/features.duckdb` - DuckDB feature store (NOT committed to git)
- `data/DATA_MANIFEST.json` - Metadata (row counts, date range, data hash)
- `data/cache/fmp/` - Cached API responses

### How to Run Real Chapter 6 Closure

After building the DuckDB feature store:

```bash
# Run with real data (default if data/features.duckdb exists)
python scripts/run_chapter6_closure.py

# Smoke test (quick verification)
python scripts/run_chapter6_closure.py --smoke

# Synthetic data only (for pipeline testing)
python scripts/run_chapter6_closure.py --mode synthetic
```

**Output directories:**
- Real data: `evaluation_outputs/chapter6_closure_real/`
- Synthetic data: `evaluation_outputs/chapter6_closure_synth/`

**Key artifacts:**
- `BASELINE_FLOOR.json` - Frozen baseline performance metrics
- `BASELINE_REFERENCE.md` - Human-readable baseline reference
- `CLOSURE_MANIFEST.json` - Commit hash, data hash, environment

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

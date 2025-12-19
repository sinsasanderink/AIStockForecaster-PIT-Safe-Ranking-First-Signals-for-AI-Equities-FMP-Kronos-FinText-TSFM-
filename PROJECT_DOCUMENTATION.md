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
| 4. Survivorship-Safe Universe | 🔲 Next | Use PIT store + FMP Survivorship API for dynamic universe |
| 5. Feature Engineering | 🔲 Pending | Price/volume, fundamentals, events, regime, **sentiment** |
| 6. Evaluation Framework | 🔲 Pending | Walk-forward, purging/embargo, ranking metrics |
| 7-13. Models & Production | 🔲 Pending | Kronos, FinText, baselines, deployment |

**Section 3 is COMPLETE and fully tested.** Ready to proceed to Section 4.

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
prices (ticker, date, open, high, low, close, volume, observed_at TIMESTAMPTZ)

-- Fundamentals supporting revisions
fundamentals (ticker, period_end, statement_type, field, value, filing_date, observed_at TIMESTAMPTZ)

-- Market snapshots for computed values
market_snapshots (ticker, date, market_cap, shares_outstanding, avg_volume_20d, observed_at TIMESTAMPTZ)
```

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
**Solution:** Verified `/stable/historical-price-eod/full` IS split-adjusted. No paid tier needed for this.

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

## What Needs To Be Done

### Section 4: Survivorship-Safe Dynamic Universe

**Key Resource:** FMP has a "Survivorship Bias Free API" endpoint (legacy v4) specifically for this.
See: [FMP Survivorship Bias Free](https://site.financialmodelingprep.com/developer/docs)

**Tasks:**
- [ ] Integrate FMP Survivorship Bias Free endpoint
- [ ] Replace placeholder universe with real PIT queries
- [ ] Implement universe reconstruction for any historical date
- [ ] Add delisted stock tracking
- [ ] Survivorship audit automation

### Section 5: Feature Engineering

**Core Features (ALL are important):**
- [ ] Price & volume (momentum, volatility, relative strength)
- [ ] Fundamentals (growth, margins, valuation ratios)
- [ ] Events (earnings surprise, days since filing)
- [ ] Regime (VIX, market breadth, sector rotation)
- [ ] **Sentiment** (now integrated via EventStore - NOT optional!)

**Sentiment Implementation:**
```python
# In feature builder
sentiment_7d = event_store.get_sentiment_score(ticker, asof, lookback_days=7)
news_volume = event_store.count_events(ticker, EventType.NEWS, asof, lookback_days=7)
days_since_earnings = event_store.days_since_event(ticker, EventType.EARNINGS, asof)
```

### Section 6: Evaluation Framework

- [ ] Walk-forward validation
- [ ] Purging & embargo (gap between train/test)
- [ ] Ranking metrics (top-N hit rate, rank correlation)
- [ ] PIT audit integration in backtest loop

### Sections 7-13: Models & Production

- [ ] Kronos integration
- [ ] FinText/TSFM integration
- [ ] ML baselines (GBDT)
- [ ] Fusion model
- [ ] Calibration
- [ ] Production deployment

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
│   │   └── event_store.py        # Event store (NEW)
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
│   └── test_section4_gates.py    # Gate tests for Section 4
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

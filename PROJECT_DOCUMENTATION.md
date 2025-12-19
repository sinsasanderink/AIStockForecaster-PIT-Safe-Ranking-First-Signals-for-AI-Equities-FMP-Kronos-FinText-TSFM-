# AI Stock Forecaster - Project Documentation

## Overview

Building a **signal-only, PIT-safe, ranking-first** AI stock forecasting system that answers:

> "Which AI stocks are most attractive to buy today, on a risk-adjusted basis, over the next 20/60/90 trading days?"

---

## Table of Contents

1. [Current Status](#current-status)
2. [Completed Work](#completed-work)
3. [Issues & Solutions](#issues--solutions)
4. [Data Sources](#data-sources)
5. [PIT Timestamp Convention](#pit-timestamp-convention)
6. [Test Results](#test-results)
7. [Next Steps](#next-steps)
8. [API Keys & Configuration](#api-keys--configuration)

---

## Current Status

| Section | Status | Notes |
|---------|--------|-------|
| 1. System Outputs | ✅ Complete | Signals, rankings, reports with distribution quantiles |
| 2. CLI & Pipelines | ✅ Complete | Commands: download-data, build-universe, score, make-report |
| 3. Data Infrastructure | ✅ Complete | FMP, Alpha Vantage, SEC EDGAR clients + PIT store |
| 4. Survivorship-Safe Universe | 🔲 Pending | Use PIT store for dynamic universe |
| 5. Feature Engineering | 🔲 Pending | Price/volume, fundamentals, events, regime |
| 6. Evaluation Framework | 🔲 Pending | Walk-forward, purging/embargo, ranking metrics |
| 7-13. Models & Production | 🔲 Pending | Kronos, FinText, baselines, deployment |

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
| **FMP** | `FMPClient` | ✅ Working | OHLCV, fundamentals (with filingDate), profiles |
| **Alpha Vantage** | `AlphaVantageClient` | ✅ Implemented | Earnings calendar (date-only, no BMO/AMC) |
| **SEC EDGAR** | `SECEdgarClient` | ✅ Implemented | Filing timestamps (GOLD STANDARD for PIT) |

**FMP Free Tier Availability:**
- ✅ Historical Prices (OHLCV)
- ✅ Quote (15-min delay)
- ✅ Profile (sector, industry, mcap)
- ✅ Income Statement (with filingDate!)
- ✅ Balance Sheet (with filingDate)
- ✅ Cash Flow (with filingDate)
- ✅ Ratios TTM
- ✅ Enterprise Value
- ❌ Earnings Calendar (paid only)
- ❌ Key Metrics (endpoint error on free tier)

**Key Metrics Workaround:**
Since FMP Key Metrics endpoint errors on free tier, we compute metrics ourselves:
- Margins from income statement
- Leverage from balance sheet
- FCF from cash flow
- Valuation ratios from price + fundamentals

#### 3.2 PIT Store (`DuckDBPITStore`)

**Schema:**
```sql
-- Prices with PIT timestamps
prices (ticker, date, open, high, low, close, adj_close, volume, observed_at TIMESTAMPTZ)

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

#### 3.3 Trading Calendar

**Features:**
- NYSE holidays (Christmas, Thanksgiving, etc.)
- 4pm ET cutoff for daily data
- DST handling (winter: 21:00 UTC, summer: 20:00 UTC)
- Rebalance date generation (monthly, quarterly)

#### 3.4 Data Audits

**Checks:**
- PIT violation scanner (future_price, future_fundamental)
- Fundamental filing date validation
- Cutoff boundary tests (15:59 violation, 16:00 valid)
- Timezone consistency validation

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
**Solution:** Check for both spellings: `filingDate` or `fillingDate`

---

## Data Sources

### Primary: FMP (Financial Modeling Prep)

**Rate Limits (Free Tier):**
- 250 requests/day
- 5 requests/minute

**Best For:**
- Historical prices (OHLCV)
- Quarterly fundamentals with filing dates
- Company profiles

### Secondary: Alpha Vantage

**Rate Limits (Free Tier):**
- 25 requests/day (very limited!)
- 5 requests/minute

**Best For:**
- Earnings calendar (supplement to FMP)
- Company overview

**Limitation:** No BMO/AMC timing information

### Gold Standard: SEC EDGAR

**Rate Limits:**
- 10 requests/second (recommended < 8)
- No daily limit

**Best For:**
- **PIT-accurate filing timestamps** (acceptanceDateTime)
- 8-K filings for exact earnings release times
- XBRL fundamentals

**No API Key Required** - just need User-Agent header

---

## PIT Timestamp Convention

All timestamps stored in **UTC**.

| Data Type | observed_at Rule |
|-----------|-----------------|
| Prices | Market close (4pm ET → 20:00 or 21:00 UTC depending on DST) |
| Fundamentals (FMP) | filing_date + next market open (conservative) |
| Fundamentals (SEC) | acceptanceDateTime (exact, gold standard) |
| Earnings (Alpha Vantage) | Assume AMC → next market open |
| Earnings (SEC 8-K) | acceptanceDateTime (exact) |

**Cutoff Boundaries:**
```
15:59 ET → VIOLATION (data not yet available)
16:00 ET → VALID (exactly at market close)
16:01 ET → VALID (after market close)
```

---

## Test Results

### Unit Tests (No API Calls)

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

### Integration Tests (With API)

```
  ✓ PASS: Historical prices have observed_at
  ✓ PASS: observed_at has timezone (UTC)
  ✓ PASS: observed_at hour is market close (20 or 21 UTC)
  ✓ PASS: Income statement has filingDate
  ✓ PASS: Filing date example: Period 2025-10-26, Filed 2025-11-19
  ✓ PASS: get_price returns values
  ✓ PASS: get_avg_volume returns value
  ✓ PASS: PIT validation passes (no issues)
```

**Run Tests:**
```bash
# Unit tests only (no API)
python tests/test_section3.py

# With integration tests (uses API quota)
RUN_INTEGRATION=1 python tests/test_section3.py
```

---

## Next Steps

### Section 4: Survivorship-Safe Dynamic Universe

- [ ] Replace placeholder universe with real PIT queries
- [ ] Implement universe reconstruction for any historical date
- [ ] Add delisted stock tracking
- [ ] Survivorship audit automation

### Section 5: Feature Engineering

- [ ] Price & volume features (momentum, volatility, relative strength)
- [ ] Fundamental features (growth, margins, valuation)
- [ ] Event features (earnings surprise, days since filing)
- [ ] Regime features (VIX, market breadth, sector rotation)

### Section 6: Evaluation Framework

- [ ] Walk-forward validation
- [ ] Purging & embargo (gap between train/test)
- [ ] Ranking metrics (top-N hit rate, rank correlation)
- [ ] PIT audit integration in backtest loop

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
│   │   └── trading_calendar.py   # NYSE calendar
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
│   └── test_section3.py          # Data infrastructure tests
├── data/                         # Downloaded data (gitignored)
├── outputs/                      # Generated reports (gitignored)
├── requirements/
│   ├── base.txt                  # Core dependencies
│   ├── dev.txt                   # Development tools
│   ├── ml.txt                    # ML libraries
│   └── research.txt              # Jupyter, plotting
└── PROJECT_DOCUMENTATION.md      # This file
```

---

## Git History

```
79b026f Section 3 fixes: Critical PIT bugs and timestamp consistency
d238759 Section 3: Data & Point-in-Time Infrastructure
3053f1d Initial commit: AI Stock Forecaster foundation
```

---

*Last Updated: December 2025*


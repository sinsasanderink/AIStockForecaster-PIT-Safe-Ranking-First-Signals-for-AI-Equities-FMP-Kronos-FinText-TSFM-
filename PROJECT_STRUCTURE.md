# AI Stock Forecaster - Project Structure & Implementation Plan

## Overview

This project builds a **decision-support forecasting model** combining:
- **Kronos** - Foundation model for financial candlestick (K-line) price dynamics
- **FinText-TSFM** - Time series foundation models for return structure prediction
- **Context Features** - Tabular models with fundamentals and macro regime data

**Core Question Answered:**
> Which AI stocks are most attractive to buy today, on a risk-adjusted basis, over the next 20 / 60 / 90 trading days?

---

## Folder Structure

```
AI_Stock_Forecast/
│
├── .env                          # API keys (FMP_KEYS) - gitignored
├── .gitignore                    # Git ignore rules
├── requirements.txt              # Full dependencies (references layer files)
├── PROJECT_STRUCTURE.md          # This file
│
├── requirements/                 # Layered dependency management
│   ├── base.txt                 # Core runtime (pandas, duckdb, pyarrow, etc.)
│   ├── ml.txt                   # + Tabular ML (lightgbm, catboost, sklearn)
│   ├── gpu.txt                  # + Deep learning (torch, transformers)
│   ├── dev.txt                  # + Testing/linting (pytest, ruff, black)
│   └── research.txt             # + Notebooks/viz (jupyter, plotly)
│
├── src/                          # Main source code
│   ├── __init__.py
│   ├── __main__.py              # Entry point for python -m src
│   ├── config.py                # Configuration & environment management
│   ├── cli.py                   # Command-line interface ⭐ NEW
│   │
│   ├── outputs/                  # Section 1: System Outputs (Signal-Only)
│   │   ├── __init__.py
│   │   ├── signals.py           # StockSignal, ReturnDistribution, LiquidityFlag
│   │   ├── rankings.py          # Cross-sectional ranking logic
│   │   └── reports.py           # Report generation & export
│   │
│   ├── pipelines/               # ⭐ NEW: Reproducible workflow orchestration
│   │   ├── __init__.py
│   │   ├── data_pipeline.py     # Download & store market data
│   │   ├── universe_pipeline.py # Build survivorship-safe universe
│   │   ├── scoring_pipeline.py  # Generate signals for a date
│   │   └── report_pipeline.py   # Create reports
│   │
│   ├── audits/                  # ⭐ NEW: Data integrity validation
│   │   ├── __init__.py
│   │   ├── pit_scanner.py       # Point-in-time violation detection
│   │   ├── survivorship_audit.py # Survivorship bias checks
│   │   └── corp_action_checks.py # Split/dividend validation
│   │
│   ├── data/                     # Sections 3-4: Data Infrastructure ✅ COMPLETE
│   │   ├── __init__.py
│   │   ├── fmp_client.py         # FMP API client (split-adjusted OHLCV)
│   │   ├── alphavantage_client.py # Alpha Vantage (earnings calendar)
│   │   ├── sec_edgar_client.py   # SEC EDGAR (gold standard timestamps)
│   │   ├── polygon_client.py     # Polygon symbol master (Ch4) ⭐ NEW
│   │   ├── pit_store.py          # Point-in-time data storage (DuckDB)
│   │   ├── event_store.py        # Event store (earnings, filings, sentiment)
│   │   ├── security_master.py    # Stable IDs, ticker changes (Ch4) ⭐ NEW
│   │   ├── universe_builder.py   # Survivorship-safe universe (Ch4) ⭐ NEW
│   │   ├── trading_calendar.py   # NYSE calendar, cutoffs, holidays
│   │   ├── expectations_client.py # Earnings surprises, estimates
│   │   ├── options_client.py     # Options data (stub for paid APIs)
│   │   └── positioning_client.py # Short interest, 13F (stub for paid APIs)
│   │
│   ├── universe/                 # AI stock definitions (label-only)
│   │   ├── __init__.py
│   │   └── ai_stocks.py          # 100 AI stocks x 10 categories (tagging only)
│   │
│   ├── features/                 # Section 5: Feature Engineering (IN PROGRESS)
│   │   ├── __init__.py
│   │   ├── labels.py             # 5.1 Forward excess returns ✅
│   │   ├── price_features.py     # 5.2 Momentum, volatility, drawdown ✅
│   │   ├── fundamental_features.py # 5.3 Relative ratios vs sector ✅
│   │   ├── time_decay.py         # Sample weighting for training ✅
│   │   ├── event_features.py     # 5.4 Earnings, filings, calendars ✅
│   │   ├── regime_features.py    # 5.5 VIX, market trend, sector rotation ✅
│   │   ├── missingness.py        # 5.6 Coverage tracking, availability masks ✅
│   │   ├── regime_features.py    # 5.5 VIX, market trend, macro
│   │   ├── missingness.py        # 5.6 "Known at time T" masks
│   │   ├── hygiene.py            # 5.7 Standardization, correlation, VIF
│   │   ├── neutralization.py     # 5.8 Sector/beta/market neutral IC
│   │   └── feature_store.py      # DuckDB storage for computed features
│   │
│   ├── models/                   # Sections 7-12: Models (TODO)
│   │   ├── baselines/            # Section 7: Baseline models
│   │   ├── kronos/               # Section 8: Kronos module
│   │   ├── fintext/              # Section 9: FinText-TSFM module
│   │   ├── fusion/               # Section 11: Fusion model
│   │   └── ensemble/             # Section 12: Regime-aware ensembling
│   │
│   ├── evaluation/               # Section 6: Evaluation Framework (TODO)
│   │   └── ...
│   │
│   ├── calibration/              # Section 13: Calibration (TODO)
│   │   └── ...
│   │
│   └── monitoring/               # Section 14: Monitoring (TODO)
│       └── ...
│
├── examples/                     # Working examples
│   └── outputs_demo.py          # Demo of Section 1 outputs
│
├── data/                         # Data storage (gitignored)
│   ├── raw/
│   ├── processed/
│   └── cache/
│
├── models/                       # Model checkpoints (gitignored)
├── outputs/                      # Generated outputs (gitignored)
├── notebooks/                    # Jupyter notebooks
└── tests/                        # Unit tests
```

---

## CLI Commands

The system exposes reproducible commands via `python -m src.cli`:

```bash
# Data Operations
python -m src.cli download-data --start 2020-01-01 --end 2024-01-01
python -m src.cli download-data --tickers NVDA,AMD,MSFT --start 2023-01-01 --end 2024-01-01

# Universe Construction
python -m src.cli build-universe --asof 2024-01-15 --max-size 50

# Scoring & Signals
python -m src.cli score --asof 2024-01-15
python -m src.cli make-report --asof 2024-01-15 --formats text,csv,json

# Full Pipeline
python -m src.cli run --asof 2024-01-15

# Audits
python -m src.cli audit-pit --start 2023-01-01 --end 2024-01-01
python -m src.cli audit-survivorship --start 2023-01-01 --end 2024-01-01
```

---

## AI Stock Universe

**100 stocks across 10 subcategories** defined in `src/universe/ai_stocks.py`:

| Category | Count | Description |
|----------|-------|-------------|
| `ai_compute_core_semis` | 18 | GPUs, CPUs, ASICs (NVDA, AMD, AVGO, QCOM...) |
| `semicap_eda_manufacturing` | 12 | Foundries, equipment, EDA (TSM, ASML, AMAT, CDNS...) |
| `networking_optics_dc_hw` | 10 | Switches, optics, servers (ANET, CSCO, SMCI...) |
| `datacenter_power_cooling_reits` | 8 | Power, cooling, REITs (VRT, ETN, EQIX...) |
| `cloud_platforms_model_owners` | 8 | Hyperscalers (MSFT, AMZN, GOOGL, META...) |
| `data_platforms_enterprise_sw` | 18 | Enterprise AI (PLTR, SNOW, CRM, ADBE...) |
| `cybersecurity` | 8 | AI-native security (CRWD, PANW, ZS...) |
| `robotics_industrial_ai` | 6 | Industrial automation (ISRG, ABB, HON...) |
| `autonomy_defense_ai` | 6 | Autonomy, defense (MBLY, LMT, RTX...) |
| `ai_apps_ads_misc` | 6 | AI products (APP, TTD, UBER, AI...) |

**Usage:**
```python
from src.universe import get_all_tickers, get_tickers_by_category, AI_UNIVERSE

# All 100 AI stocks
all_tickers = get_all_tickers()

# Just semiconductors (30 stocks)
semis = get_tickers_by_categories(["ai_compute_core_semis", "semicap_eda_manufacturing"])

# Single category
cyber = get_tickers_by_category("cybersecurity")
```

**CLI:**
```bash
python -m src.cli list-universe                           # Show all categories
python -m src.cli list-universe --category cybersecurity  # Show one category
python -m src.cli build-universe --asof 2024-01-15 --categories ai_compute_core_semis,cybersecurity
```

---

## Signal Output Format

Each stock signal includes **full distribution metrics** (not just a point estimate):

```
🟢 NVDA (20d horizon)

  Expected Excess Return: +15.5% vs QQQ
  Return Range (P5/P50/P95): [-44.0% / +15.5% / +75.1%]
  Prob(Outperform):  67%

  Alpha Rank Score:  1.510
  Confidence:        0.85 (HIGH)
  Avg Daily Volume:  50.0M

  Key Drivers:
    • momentum_12m: +0.062
    • revenue_growth: +0.047
```

**Decision-support metrics shown:**
- Expected excess return (mean or median)
- P5/P50/P95 range (full uncertainty)
- Prob(outperform) - probability of beating benchmark
- Confidence bucket (calibrated uncertainty)
- Liquidity flag (capacity warnings)
- Key drivers (explainability)

---

## Implementation Phases

### ✅ Phase 1a: Foundation (Complete)
- [x] Project structure
- [x] Section 1: System outputs with distribution metrics
- [x] CLI for reproducible runs
- [x] Audits module (PIT, survivorship, corp actions)
- [x] Requirements split into layers

### 🔲 Phase 1b: Data Infrastructure (Next)
- [ ] FMP API client
- [ ] DuckDB-based PIT store
- [ ] Trading calendar integration
- [ ] Universe construction from real data

### 🔲 Phase 2: Features
- [ ] Price & volume features
- [ ] Fundamental features (relative, normalized)
- [ ] Event & calendar features
- [ ] Regime features

### 🔲 Phase 3: Baselines
- [ ] Evaluation framework (walk-forward)
- [ ] Naive baselines
- [ ] Factor baselines (momentum, low-vol, quality)
- [ ] Tabular ML (LightGBM/CatBoost)

### 🔲 Phase 4: Foundation Models
- [ ] Kronos integration (price dynamics)
- [ ] FinText integration (return distributions)

### 🔲 Phase 5: Advanced
- [ ] Fusion model (ranking-first objectives)
- [ ] Regime-aware ensemble
- [ ] Calibration & confidence
- [ ] Monitoring

---

## Key Design Decisions

### Data Philosophy
- **Point-in-Time (PIT) Correctness**: All features respect `observed_at` timestamps
- **Conservative Lag Rules**: FMP data uses filing_date + 1 day lag when observed_at unavailable
- **Survivorship Safe**: Universe includes historical losers, not just current winners
- **Cutoff Policy**: 4:00pm ET daily cutoff for feature availability

### Model Philosophy
- **Ranking > Regression**: Focus on relative stock ordering, not exact price prediction
- **Multiple Weak Signals**: Combine Kronos + FinText + tabular context
- **Economic Validity**: Signals must survive transaction costs (20bps round-trip)
- **Distribution-Aware**: Report full uncertainty, not just point estimates

### Storage
- **DuckDB**: PIT-safe storage with fast replay for backtesting
- **Parquet**: Efficient columnar storage for large datasets
- **Trading Calendars**: Correct handling of trading days and cutoffs

---

## External Dependencies

### Data Sources
- **Financial Modeling Prep (FMP)**: Market data, fundamentals, events

### Foundation Models
- **Kronos**: [github.com/shiyu-coder/Kronos](https://github.com/shiyu-coder/Kronos)
  - Inference: `pip install git+https://github.com/shiyu-coder/Kronos.git`
  - Fine-tuning: Also needs `pip install pyqlib`
  - Max context: 512 tokens for small/base models
  
- **FinText**: [huggingface.co/FinText](https://huggingface.co/FinText)
  - Loaded via transformers
  - Year-specific checkpoints to reduce pretraining leakage

---

## Success Criteria

| Metric | Threshold |
|--------|-----------|
| Median Walk-Forward RankIC | ≥ baseline + 0.02 |
| Net-of-Cost Positive Folds | ≥ 70% |
| Top-10 Monthly Churn | < 30% |
| PIT Violations | < 0.1% |
| Feature Completeness | > 95% |

---

## Notes for Colab Integration

Since you have the Colab plugin installed:
- Heavy model inference (Kronos, FinText) can be offloaded to Colab GPUs
- Data preprocessing can run locally
- Use the `notebooks/` directory for interactive experiments
- Models can be cached in Google Drive for persistence

---

## Folder Distinction

**Important**: Keep these two `outputs/` paths strictly separate:
- `src/outputs/` = **Code** + schemas (Python modules)
- `outputs/` = **Generated files** only (CSV, JSON, Parquet)

---

## Deferred Items (TODOs by Section)

Items that are noted in code but require later sections to implement.

### Section 3: Data Infrastructure
**Files to implement:**
- `src/data/fmp_client.py` - FMP API client
- `src/data/pit_store.py` - DuckDB-based PIT store (implements `PITStore` protocol)

**Code TODOs:**
- `src/pipelines/universe_pipeline.py`: Replace `StubPITStore` with real `DuckDBPITStore`
- `src/pipelines/data_pipeline.py`: Implement actual FMP data fetching
- `src/interfaces.py`: `PITStore` and `TradingCalendar` have stubs; need real implementations

### Section 4: Survivorship-Safe Dynamic Universe ✅ COMPLETE
**Implemented:**
- `src/data/polygon_client.py`: Symbol master for universe-as-of-T
- `src/data/universe_builder.py`: UniverseBuilder with SurvivorshipStatus
- `src/data/security_master.py`: Stable IDs, ticker changes
- `tests/test_chapter4_universe.py`: Comprehensive tests (7/7 passed)

### Section 5: Feature Engineering 🟡 IN PROGRESS
**Files created:**
- `src/features/labels.py` - Forward excess returns (5.1) ✅
- `src/features/price_features.py` - Momentum, volatility, drawdown (5.2) ✅
- `src/features/fundamental_features.py` - Relative ratios vs sector (5.3) ✅
- `src/features/time_decay.py` - Sample weighting for training ✅
- `src/features/event_features.py` - Earnings, filings, calendars (5.4) ✅
- `src/features/regime_features.py` - VIX, market trend, sector rotation (5.5) ✅
- `src/features/missingness.py` - Coverage tracking, availability masks (5.6) ✅

**Files to create:**
- `src/features/missingness.py` - "Known at time T" masks (5.6)
- `src/features/hygiene.py` - Standardization, correlation, VIF (5.7)
- `src/features/neutralization.py` - Sector/beta/market neutral IC (5.8)
- `src/features/feature_store.py` - DuckDB storage for features

**Key requirements:**
- All features must use `observed_at <= asof` filtering
- Cross-sectional standardization (z-score or rank)
- VIF diagnostics (as diagnostic, not hard filter)
- IC stability checks (more important than VIF)
- Missingness as first-class feature

### Section 6: Evaluation Framework
**Code TODOs:**
- Implement walk-forward validation
- Add purging/embargo for overlapping labels

### Section 8: Kronos Module
**Code TODOs:**
- `src/outputs/signals.py` line ~83: `prob_positive` uses normal approximation
  - When Kronos provides samples, switch to empirical: `(samples > 0).mean()`
  - Use `ReturnDistribution.from_samples()` for sample-based distributions

### Section 9: FinText-TSFM Module
**Code TODOs:**
- Same as Section 8: switch `prob_outperform` to empirical when quantiles available
- Use year-specific checkpoints to prevent pretraining leakage

### Section 13: Calibration & Confidence
**Notes:**
- Current `prob_outperform` is normal-approximation based
- Add quantile calibration to verify predicted distributions match realized

---

## Testing Strategy

### Current Tests (`tests/`)
- `test_signals.py`: Unit tests for signal data structures
- `test_reports.py`: Unit tests for reports and CSV export

### Tests to Add

**Unit Tests (Fast, No Network):**
- [ ] `test_interfaces.py`: Stub implementations
- [ ] `test_universe_pipeline.py`: Deterministic with stub PIT store

**Integration Tests:**
- [ ] PIT store with DuckDB (Section 3)
- [ ] End-to-end pipeline smoke test

**Invariant Tests:**
- [ ] "No row with observed_at > cutoff is visible at asof T"
- [ ] "Universe at T is reproducible and stored as snapshot"
- [ ] "Forward-fill only starts after observed_at"

---

## Code Conventions

### Signal Consistency
- `expected_excess_return` MUST equal `return_distribution.mean` (enforced in `__post_init__`)
- Use `StockSignal.create()` factory to auto-set from distribution
- P50 (median) may differ from mean for skewed distributions

### Dependencies
- `scipy.stats` is used for `prob_positive` (in `requirements/base.txt`)
- A no-scipy fallback exists: `ReturnDistribution._prob_positive_approx()`

### Pipeline Dependencies
- Pipelines accept interface types (`PITStore`, `TradingCalendar`, `Config`)
- Default to stubs for testing; inject real implementations in production
- This makes pipelines deterministic and testable without network access

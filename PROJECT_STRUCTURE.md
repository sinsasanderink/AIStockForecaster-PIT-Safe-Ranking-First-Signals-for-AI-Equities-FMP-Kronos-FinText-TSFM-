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
│   ├── research.txt             # + Notebooks/viz (jupyter, plotly)
│   └── qlib.txt                 # + Qlib (optional, for Chapter 6+ evaluation)
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
│   ├── utils/                    # Shared utility functions
│   │   ├── __init__.py
│   │   ├── env.py                # Environment loading (.env auto-load, API key resolution)
│   │   └── price_validation.py   # Split-discontinuity detection and validation
│   │
│   ├── features/                 # Section 5: Feature Engineering (IN PROGRESS)
│   │   ├── __init__.py
│   │   ├── labels.py             # 5.1 Forward excess returns ✅
│   │   ├── price_features.py     # 5.2 Momentum, volatility, drawdown ✅
│   │   ├── fundamental_features.py # 5.3 Relative ratios vs sector ✅
│   │   ├── time_decay.py         # Sample weighting for training ✅
│   │   ├── event_features.py     # 5.4 Earnings, filings, calendars ✅
│   │   ├── regime_features.py    # 5.5 VIX, market trend, sector rotation, macro ✅
│   │   ├── missingness.py        # 5.6 Coverage tracking, "known at time T" masks ✅
│   │   ├── hygiene.py            # 5.7 Standardization, correlation, VIF, IC stability ✅
│   │   ├── neutralization.py     # 5.8 Sector/beta/market neutral IC ✅
│   │   └── feature_store.py      # DuckDB storage for computed features
│   │
│   ├── models/                   # Sections 7-12: Models (TODO)
│   │   ├── baselines/            # Section 7: Baseline models
│   │   ├── kronos/               # Section 8: Kronos module
│   │   ├── fintext/              # Section 9: FinText-TSFM module
│   │   ├── fusion/               # Section 11: Fusion model
│   │   └── ensemble/             # Section 12: Regime-aware ensembling
│   │
│   ├── evaluation/               # Section 6: Evaluation Framework ✅ FROZEN
│   │   ├── __init__.py
│   │   ├── definitions.py        # Canonical time conventions (frozen)
│   │   ├── walk_forward.py       # Expanding window + purging/embargo/maturity
│   │   ├── sanity_checks.py      # IC parity + experiment naming
│   │   ├── metrics.py            # EvaluationRow contract + RankIC/churn/regime
│   │   ├── costs.py              # Cost overlay (base + ADV-scaled slippage)
│   │   ├── reports.py            # Stability reports (IC decay, regime, churn)
│   │   ├── baselines.py          # 4 baselines (mom_12m, composite, short_term, naive)
│   │   ├── run_evaluation.py     # End-to-end orchestrator (SMOKE/FULL modes)
│   │   ├── qlib_adapter.py       # Qlib shadow evaluator (IC parity)
│   │   └── data_loader.py        # Deterministic loading (synthetic + DuckDB)
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
│   ├── cache/
│   ├── features.duckdb           # REAL feature store (gitignored)
│   └── DATA_MANIFEST.json        # Data snapshot metadata (gitignored)
│
├── evaluation_outputs/           # Evaluation artifacts (mostly gitignored)
│   ├── chapter6_closure_synth/   # Synthetic baseline runs (gitignored)
│   └── chapter6_closure_real/    # ✅ TRACKED: Frozen baseline reference (exception in .gitignore)
│       ├── BASELINE_FLOOR.json
│       ├── BASELINE_REFERENCE.md
│       ├── CLOSURE_MANIFEST.json
│       └── baseline_*/           # Full stability reports + figures per baseline
│
├── models/                       # Model checkpoints (gitignored)
├── outputs/                      # Generated outputs (gitignored)
├── notebooks/                    # Jupyter notebooks
└── tests/                        # Unit tests (413/413 passing)
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
| Net-of-Cost Positive Folds | ≥ frozen baseline (per horizon) + 10pp |
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

**Files created (5.7-5.8):**
- `src/features/hygiene.py` - Feature hygiene & redundancy control ✅
- `src/features/neutralization.py` - Sector/beta/market neutral IC ✅

**Files to create (future):**
- `src/features/feature_store.py` - DuckDB storage for features (deferred to Chapter 6)

**Key requirements:**
- All features must use `observed_at <= asof` filtering
- Cross-sectional standardization (z-score or rank)
- VIF diagnostics (as diagnostic, not hard filter)
- IC stability checks (more important than VIF)
- Missingness as first-class feature

### Section 6: Evaluation Framework ✅ FROZEN

**Status:** CLOSED & FROZEN (December 30, 2025)  
**Tests:** 413/413 passing  
**Commits:** `18bad8a` (code fixes) + `7e6fa3a` (artifacts freeze)  
**Reference Doc:** `CHAPTER_6_FREEZE.md`

**What Was Frozen:**
- Evaluation infrastructure (`src/evaluation/*.py`) - 10 modules, fully tested
- Baseline reference artifacts (`evaluation_outputs/chapter6_closure_real/`) - tracked in git
- Baseline floor: Best RankIC per horizon (20d: 0.0283, 60d: 0.0392, 90d: 0.0169)
- Data snapshot: 192,307 rows (2016-2025), `data_hash: 5723d4c88b8ecba1...`

**Frozen Baselines:**
- `mom_12m` - Naive momentum baseline
- `momentum_composite` - Stronger momentum composite
- `short_term_strength` - Short-term diagnostic
- `naive_random` - Sanity check (RankIC ≈ 0)

**Key Files:**
- `scripts/build_features_duckdb.py` - Build DuckDB feature store from FMP
- `scripts/run_chapter6_closure.py` - Run baseline reference freeze
- `src/evaluation/data_loader.py` - Deterministic data loading (synthetic + DuckDB)
- `tests/test_duckdb_loader.py` - 24 tests for DuckDB loading

**Output Directories:**
- `evaluation_outputs/chapter6_closure_real/` - REAL data outputs (from DuckDB)
- `evaluation_outputs/chapter6_closure_synth/` - SYNTHETIC data outputs (for testing)

**Build Real Data:**
```bash
export FMP_KEYS="your_api_key"
python scripts/build_features_duckdb.py
python scripts/run_chapter6_closure.py
```

**Implementation Status:**

| Phase | Status | Files |
|-------|--------|-------|
| **Phase 0: Sanity Checks** | ✅ COMPLETE | `src/evaluation/sanity_checks.py` |
| **Phase 1: Walk-Forward** | ✅ COMPLETE | `src/evaluation/walk_forward.py` |
| **Phase 1.5: Definition Lock** | ✅ COMPLETE | `src/evaluation/definitions.py` |
| **Phase 2: Metrics** | ✅ COMPLETE | `src/evaluation/metrics.py` |
| **Phase 3: Baselines** | ✅ COMPLETE | `src/evaluation/baselines.py` |
| **Phase 4: Cost Realism** | ✅ COMPLETE | `src/evaluation/costs.py` |
| **Phase 5: Stability Reports** | ✅ COMPLETE | `src/evaluation/reports.py` |
| **Phase 6: Qlib + Runner** | ✅ COMPLETE | `src/evaluation/qlib_adapter.py`, `src/evaluation/run_evaluation.py` |

**✅ Implemented (Phase 0, 1, 1.5, 2, 4, 5, 6):**

**`src/evaluation/definitions.py`** - **CANONICAL DEFINITIONS (Single Source of Truth)**
- `TIME_CONVENTIONS`: Horizons (20/60/90 TRADING DAYS), embargo, pricing, maturity
- `EMBARGO_RULES`: Per-row-per-horizon purging rules
- `ELIGIBILITY_RULES`: All-horizons-valid requirement
- `trading_days_to_calendar_days()`: CONSERVATIVE conversion (not naive multiplication)
- `get_market_close_utc()`: UTC datetime for maturity checks
- `is_label_mature()`: Canonical maturity comparison
- All dataclasses are FROZEN (cannot be modified)

**`src/evaluation/walk_forward.py`** - Walk-forward splitter with **ENFORCED** purging & embargo
- `WalkForwardSplitter`: Generates folds with expanding window
- `WalkForwardFold`: Represents a single train/val split
- **CRITICAL**: Purging & embargo are HARD CONSTRAINTS (raises ValueError if violated)
- **Embargo**: Minimum 90 trading days between train_end and val_start
- **Purging**: Overlapping label windows automatically removed
- **Maturity**: Only mature labels (label_matured_at <= cutoff) used

**`src/evaluation/sanity_checks.py`** - Pre-implementation validation
- `verify_ic_parity()`: Manual vs Qlib IC comparison (tolerance: 0.001)
- `ExperimentNameBuilder`: Standardized experiment naming
- `validate_experiment_name()`: Parse and validate naming convention
- `run_sanity_checks()`: Combined checker for all pre-checks

**`src/evaluation/metrics.py`** - Ranking-first metrics with locked definitions
- `EvaluationRow`: Canonical data contract (all models/baselines must produce this)
- `compute_rankic_per_date()`: Spearman correlation per date
- `compute_quintile_spread_per_date()`: Top 20% - bottom 20% spread
- `compute_topk_metrics_per_date()`: Hit rate & avg excess return for Top-K
- `compute_churn()`: Retention/churn using stable_id, consecutive dates only
- `assign_regime_bucket()`: VIX/market regime bucketing (locked definitions)
- `evaluate_fold()`: Complete fold evaluation with all metrics
- `REGIME_DEFINITIONS`: Locked regime bucket definitions

**`src/evaluation/costs.py`** - Cost realism (diagnostic overlay)
- `TRADING_ASSUMPTIONS`: Frozen dataclass with portfolio, AUM, cost parameters
- `compute_participation_rate()`: Trade value / ADV with cap at 100%
- `compute_slippage_bps()`: Square-root impact model (c * sqrt(participation))
- `compute_trade_cost()`: Base cost + slippage with ADV missing penalty
- `compute_turnover()`: Weight changes between consecutive rebalances
- `compute_portfolio_costs()`: Full portfolio cost computation for one rebalance
- `compute_net_metrics()`: Net-of-cost metrics (gross ER - costs)
- `run_cost_sensitivity()`: Sensitivity analysis (low/base/high scenarios)
- **Enforced Invariants**: ADV monotonicity, AUM monotonicity, determinism

**`src/evaluation/reports.py`** - Stability reports (pure consumer)
- `STABILITY_THRESHOLDS`: Frozen thresholds for flags (decay, noise, churn)
- `compute_ic_decay_stats()`: Early vs late performance statistics
- `plot_ic_decay()`: IC time series with rolling mean
- `format_regime_performance()`: Regime performance with coverage stats
- `plot_regime_bars()`: Regime-conditional performance bars
- `compute_churn_diagnostics()`: Churn summary per fold/horizon/k
- `plot_churn_timeseries()`: Churn over time
- `plot_churn_distribution()`: Churn histogram
- `generate_stability_scorecard()`: One-screen summary
- `generate_stability_report()`: Complete report with tables + figures + summary
- **Enforced Invariants**: Determinism, fold boundaries, regime integrity, no silent drops

**`src/evaluation/baselines.py`** - Phase 3 baselines
- `BASELINE_REGISTRY`: Frozen registry of 3 baselines
- `generate_baseline_scores()`: Generate EvaluationRow format scores for a baseline
- `run_all_baselines()`: Run all baselines at once
- `list_baselines()`: List available baseline names
- **Baselines**: mom_12m, momentum_composite, short_term_strength

**`src/evaluation/run_evaluation.py`** - End-to-end runner
- `ExperimentSpec`: Experiment specification (baseline or model)
- `SMOKE_MODE`, `FULL_MODE`: Evaluation modes
- `run_experiment()`: Main orchestrator (walk-forward → metrics → costs → reports)
- `compute_acceptance_verdict()`: Compute pass/fail per criterion
- `save_acceptance_summary()`: Write ACCEPTANCE_SUMMARY.md

**`src/evaluation/qlib_adapter.py`** - Qlib shadow evaluator
- `to_qlib_format()`: Convert EvaluationRow to Qlib MultiIndex format
- `validate_qlib_frame()`: Validate Qlib frame for pitfalls
- `run_qlib_shadow_evaluation()`: Run shadow IC analysis
- `check_ic_parity()`: Verify IC computation matches
- **Note**: Qlib is SHADOW EVALUATOR only

**Tests:**
- `tests/test_definitions.py`: 40 tests (all pass) ✅
- `tests/test_walk_forward.py`: 25 tests (all pass) ✅
- `tests/test_sanity_checks.py`: 16 tests (all pass) ✅
- `tests/test_metrics.py`: 30 tests (all pass) ✅
- `tests/test_costs.py`: 28 tests (all pass) ✅
- `tests/test_reports.py`: 24 tests (all pass) ✅
- `tests/test_baselines.py`: 39 tests (all pass) ✅
- `tests/test_qlib_parity.py`: 21 tests (all pass) ✅
- `tests/test_end_to_end_smoke.py`: 22 tests (all pass) ✅
- `tests/test_duckdb_loader.py`: 24 tests (all pass) ✅
- `tests/test_env_utils.py`: 16 tests (all pass) ✅
- `tests/test_price_validation.py`: 18 tests (all pass) ✅
- **Total**: 413/413 tests passing (all suites)

**Key Validation Results:**
- ✅ Embargo < 90 days is REJECTED (raises ValueError)
- ✅ Overlapping train/val dates are REJECTED (raises ValueError)
- ✅ Purged labels are REMOVED from both train and val splits
- ✅ Immature labels are DROPPED (PIT-safe)
- ✅ Expanding window verified (train_start always equals eval_start)
- ✅ IC parity check works (manual vs Qlib comparison)
- ✅ Experiment naming validated (format enforcement)

---

**Locked Evaluation Parameters:**

| Parameter | Value | Implementation |
|-----------|-------|----------------|
| **Rebalance (Primary)** | Monthly (1st trading day) | `WalkForwardSplitter(rebalance_freq="monthly")` |
| **Rebalance (Secondary)** | Quarterly | `WalkForwardSplitter(rebalance_freq="quarterly")` |
| **Date Range** | 2016-01-01 to 2025-06-30 | `eval_start`, `eval_end` parameters |
| **Embargo** | 90 trading days | `embargo_days=90` (HARD minimum) |
| **Window** | Expanding (not rolling) | `train_start` always equals `eval_start` |

**Baselines (Models to Beat - TODO Phase 3):**
1. **`mom_12m`**: 12-month momentum (naive baseline)
2. **`momentum_composite`**: `(mom_1m + mom_3m + mom_6m + mom_12m) / 4` (strong but transparent)
3. **`short_term_strength`**: `mom_1m` or `rel_strength_1m` (diagnostic for horizon sensitivity)

**Top-K Metrics (TODO Phase 2):**
- **Primary**: Top-10 (matches product narrative)
- **Secondary**: Top-20 (robustness check)
- **Churn**: Jaccard or % retained (target < 30%)
- **Hit Rate**: % with excess return > 0 (target > 55%)

**Acceptance Criteria:**
- ✅ Median walk-forward RankIC > baseline by ≥ 0.02
- ✅ Net-of-cost improvement: % positive folds ≥ baseline + 10pp (relative; frozen floor: 5.8%-40.1%)
- ✅ Top-10 churn < 30% month-over-month
- ✅ Performance degrades gracefully under regime shifts
- ✅ NO PIT violations (enforced by scanner)

**Remaining TODOs:**
- [x] Phase 2: Implement metrics (Top-K churn, hit rate, regime slicing) ✅
- [x] Phase 4: Implement cost realism (trading costs, slippage, net metrics) ✅
- [x] Phase 5: Implement stability reports (IC decay, regime performance, churn diagnostics) ✅
- [x] Phase 3: Implement 3 baselines through identical pipeline ✅
- [x] Phase 3: Create `src/evaluation/baselines.py` ✅
- [x] Phase 6: Create `src/evaluation/qlib_adapter.py` for Qlib integration ✅
- [x] Phase 6: Create `src/evaluation/run_evaluation.py` for end-to-end runner ✅
- [ ] Execute FULL mode evaluation (2016-2025) with actual features
- [ ] Run acceptance criteria on model vs baselines

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

---

## Qlib Integration (Chapter 6+)

**Purpose:** Microsoft's [Qlib](https://github.com/microsoft/qlib) serves as an **optional shadow evaluator** for standardized reporting and experiment tracking, starting in Chapter 6.

**Integration Philosophy:**
- **NOT a replacement:** Chapters 1-5 infrastructure (PIT, survivorship, features, labels) remain untouched
- **Narrow integration:** Qlib receives predictions + labels, NOT raw data/features
- **Shadow evaluator:** Qlib generates standardized factor evaluation reports for validation

### What Qlib Provides

**1. Chapter 6 Evaluation & Reporting:**
- IC/RankIC analysis with built-in plots (monthly IC, regime IC, autocorrelation)
- Quintile analysis (group returns, top-bottom spread, long-short distribution)
- Backtest engine with configurable transaction costs
- Portfolio metrics (cumulative return, Sharpe/IR, max drawdown, turnover)

**2. Experiment Tracking:**
- Recorder system for managing walk-forward folds, model variants, hyperparameter sweeps
- Artifact storage (models, configs, predictions)
- Metric logging (IC, backtest returns, cost-adjusted performance)

**3. Baseline Harness (Chapter 7):**
- Ready-made implementations: LightGBM, XGBoost, CatBoost, LSTM, Transformer, etc.
- Standardized evaluation across baselines

### Data Flow

```
┌──────────────────────────────────────┐
│ Our System (Source of Truth)        │
│ - Universe (stable_id, survivorship)│
│ - Features (PIT-safe, 5.1-5.8)      │
│ - Labels (v2 total return)          │
│ - Predictions (our models)          │
└──────────────┬───────────────────────┘
               │
               │ DataFrame: (date, ticker, pred, label, group)
               ↓
┌──────────────────────────────────────┐
│ Qlib (Evaluation & Reporting)       │
│ - IC analysis (monthly, regime)     │
│ - Quintile spread & hit rate        │
│ - Backtest (cost-inclusive)         │
│ - Experiment tracking (Recorder)    │
└──────────────────────────────────────┘
```

### Installation

```bash
pip install pyqlib  # Add to requirements/research.txt
```

**Version pinning recommended:**
```bash
pip install pyqlib==0.9.7  # or latest stable
```

### Adapter Pattern

```python
# src/evaluation/qlib_adapter.py (Chapter 6)
def our_predictions_to_qlib_format(predictions_df, labels_df):
    """
    Convert our predictions to Qlib's expected format.
    
    Qlib expects:
    - MultiIndex: (instrument, datetime)
    - Columns: score (prediction), label (realized return)
    """
    qlib_df = pd.merge(predictions_df, labels_df, on=["date", "ticker"])
    qlib_df = qlib_df.rename(columns={"ticker": "instrument", "pred": "score"})
    qlib_df = qlib_df.set_index(["instrument", "date"])
    return qlib_df

# Usage in Chapter 6
from qlib.contrib.evaluate import backtest_daily
qlib_df = our_predictions_to_qlib_format(predictions, labels)
reports = backtest_daily(prediction=qlib_df, ...)
```

### What NOT to Do

❌ **Don't plug DuckDB into Qlib's DataProvider:** This is where migrations get expensive and you risk losing PIT guarantees.

❌ **Don't force our features into Qlib's handlers:** Our feature engineering (PIT discipline, missingness, neutralization) is institutional-grade. Keep it.

❌ **Don't let Qlib own the universe:** Our survivorship-safe universe construction with stable IDs is a core differentiator.

### References

- **Qlib GitHub:** https://github.com/microsoft/qlib
- **Qlib Documentation:** https://qlib.readthedocs.io/en/latest/
- **Evaluation & Reporting:** https://qlib.readthedocs.io/en/latest/component/report.html
- **Recorder (Experiment Tracking):** https://qlib.readthedocs.io/en/latest/component/recorder.html
- **Backtest Engine:** https://qlib.readthedocs.io/en/latest/component/strategy.html

---

# Thesis Dataset Export

This folder contains the complete dataset used for the AI Stock Forecast project, exported on 2026-01-21.

## Files Overview

### Main Modeling Datasets (Full Size)
| File | Size | Rows | Description |
|------|------|------|-------------|
| `modeling_dataset_20d.csv` | 132 MB | 201,307 | Features + 20-day forward returns |
| `modeling_dataset_60d.csv` | 132 MB | 201,307 | Features + 60-day forward returns |
| `modeling_dataset_90d.csv` | 132 MB | 201,307 | Features + 90-day forward returns |

### Supporting Files
| File | Size | Description |
|------|------|-------------|
| `regime_data.csv` | 475 KB | Daily market regime indicators (2,386 days) |
| `data_dictionary.csv` | 3.6 KB | Column definitions for all 56 features |
| `dataset_summary.txt` | 5.3 KB | Metadata, sources, and usage guide |
| `summary_statistics.csv` | 5.9 KB | Descriptive statistics for all numeric columns |

## Quick Start

### Load Dataset in Python

```python
import pandas as pd

# Load main dataset (20-day horizon)
df = pd.read_csv('datasets/modeling_dataset_20d.csv', 
                 parse_dates=['date', 'label_matured_at'])

# Load regime data
regime = pd.read_csv('datasets/regime_data.csv', 
                     parse_dates=['date'])

print(f"Dataset shape: {df.shape}")
print(f"Date range: {df['date'].min()} to {df['date'].max()}")
print(f"Unique tickers: {df['ticker'].nunique()}")
```

### Basic Analysis

```python
# Check target distribution
print(df['target_excess_return_20d'].describe())

# Top features by ticker
top_tickers = df['ticker'].value_counts().head(10)
print(top_tickers)

# Check missing values
missing = df.isnull().sum().sort_values(ascending=False)
print(missing[missing > 0])
```

## Dataset Structure

### Dimensions
- **201,307 observations** (100 tickers × ~2,386 trading days)
- **54 columns** in modeling datasets (52 features + target + metadata)
- **Date range:** 2016-01-04 to 2025-06-30
- **Universe:** 100 AI and technology stocks

### Feature Categories

| Category | Count | Examples |
|----------|-------|----------|
| **Momentum** | 4 | `mom_1m`, `mom_3m`, `mom_6m`, `mom_12m` |
| **Liquidity** | 2 | `adv_20d`, `adv_60d` |
| **Volatility** | 3 | `vol_20d`, `vol_60d`, `vol_of_vol` |
| **Drawdown** | 2 | `max_drawdown_60d`, `dist_from_high_60d` |
| **Relative Strength** | 2 | `rel_strength_1m`, `rel_strength_3m` |
| **Beta** | 1 | `beta_252d` |
| **Earnings** | 12 | `days_to_earnings`, `last_surprise_pct`, `in_pead_window`, etc. |
| **Regime** | 12 | `vix_level`, `market_return_21d`, `market_regime`, etc. |
| **Fundamentals** | 9 | `gross_margin_ttm`, `revenue_growth_yoy`, `roe_zscore`, etc. |
| **Missingness** | 2 | `coverage_pct`, `is_new_stock` |

### Target Variable

```
target_excess_return_20d = (stock_return_20d - QQQ_return_20d)
```

- **Mean:** 0.0068 (0.68% average excess return)
- **Std:** 0.1034 (10.34% volatility)
- **Range:** -65.71% to +341.70%

## Data Quality

### ✅ Point-in-Time Safety
- All features are **PIT-safe** (no look-ahead bias)
- Earnings data filtered by **announcement datetime**
- SEC filings filtered by **acceptance datetime**
- Prices are **split-adjusted**
- Labels include **maturation timestamp** for validation

### 📊 Completeness
- **192,307** non-null targets (95.5% coverage)
- Missing values are intentional (e.g., IPOs lack 12M momentum)
- Fundamentals forward-filled between quarterly updates

## Usage Guidelines

### 1. Time-Series Splits
Always use the `date` column for chronological splitting:

```python
train_cutoff = '2023-01-01'
train = df[df['date'] < train_cutoff]
test = df[df['date'] >= train_cutoff]
```

### 2. Feature Engineering
Consider:
- **Winsorizing** extreme values (volatility, returns)
- **Forward-filling** missing fundamentals
- **Rank transformation** for skewed features
- **Interaction terms** (momentum × volatility, etc.)

### 3. Evaluation Metrics
Recommended metrics for stock ranking:
- **RankIC (Spearman correlation)** - Primary metric
- **Quintile spread** - Top quintile vs bottom quintile
- **Hit rate @10** - % of top-10 picks that outperform
- **Cost survival** - % positive after transaction costs

### 4. Transaction Costs
Include realistic costs in backtests:
- **10 bps** (0.001) per trade typical for institutional
- Monthly rebalancing: ~0.20 churn → ~0.02% monthly cost

## Benchmark Results

These are the published results from the LightGBM baseline (Chapter 7):

| Horizon | RankIC | Cost Survival | Churn |
|---------|--------|---------------|-------|
| 20d | 0.1009 | 6.4% | 0.20 |
| 60d | 0.1275 | 45.9% | 0.20 |
| 90d | 0.1808 | 56.9% | 0.20 |

Your model should aim to beat these benchmarks!

## Data Sources

- **Prices:** Financial Modeling Prep API (split-adjusted OHLCV)
- **Earnings:** FMP earnings calendar + Alpha Vantage surprises
- **Filings:** SEC EDGAR (10-K, 10-Q timestamps)
- **Fundamentals:** FMP income statements (TTM)
- **Regime:** FMP historical prices (SPY, VIX ETF)
- **Benchmark:** QQQ (Nasdaq-100 ETF)

## Sample Tickers

The dataset includes 100 AI and technology stocks:

**Top holdings:**
- AAPL, MSFT, NVDA, GOOGL, META, AMZN, TSLA
- AMD, INTC, QCOM, AVGO, MU, AMAT
- CRM, NOW, PANW, CRWD, ZS, DDOG
- ORCL, ADBE, SNOW, PLTR, U, PATH

## Citation

If you use this dataset in your thesis or publication, please cite:

```
AI Stock Forecast Dataset (2016-2025)
100 AI and Technology Stocks with 52 Point-in-Time Features
Data Sources: Financial Modeling Prep, SEC EDGAR, Alpha Vantage
Generated: 2026-01-21
```

## Questions?

See `dataset_summary.txt` for complete documentation including:
- Detailed feature descriptions
- Data collection methodology
- Point-in-time validation approach
- Recommended modeling workflow

For column-level documentation, see `data_dictionary.csv`.

---

**Total Dataset Size:** ~400 MB uncompressed (CSV format)  
**Recommended Tools:** pandas, scikit-learn, LightGBM, PyTorch  
**Python Version:** 3.8+


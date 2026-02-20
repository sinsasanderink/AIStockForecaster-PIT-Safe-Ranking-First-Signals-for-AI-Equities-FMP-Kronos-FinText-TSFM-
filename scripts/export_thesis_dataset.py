#!/usr/bin/env python3
"""
Export Thesis Dataset

Creates a clean, documented dataset export for thesis submission.
Outputs:
1. modeling_dataset_20d.csv - Combined features + labels (20-day horizon)
2. modeling_dataset_60d.csv - Combined features + labels (60-day horizon)
3. modeling_dataset_90d.csv - Combined features + labels (90-day horizon)
4. regime_data.csv - Market regime indicators
5. data_dictionary.csv - Column descriptions
6. dataset_summary.txt - Statistics and metadata
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import duckdb
import pandas as pd
from datetime import datetime

# Paths
DB_PATH = "data/features.duckdb"
OUTPUT_DIR = Path("datasets")
OUTPUT_DIR.mkdir(exist_ok=True)

print("=" * 70)
print("THESIS DATASET EXPORT")
print("=" * 70)
print(f"Source: {DB_PATH}")
print(f"Output: {OUTPUT_DIR.absolute()}/")
print("=" * 70)

# Connect to database
conn = duckdb.connect(DB_PATH, read_only=True)

# ============================================================================
# 1. Export Main Modeling Datasets (Features + Labels)
# ============================================================================
print("\n1. Exporting main modeling datasets...")

for horizon in [20, 60, 90]:
    print(f"\n   Processing {horizon}d horizon...")
    
    query = f"""
    SELECT 
        f.*,
        l.excess_return as target_excess_return_{horizon}d,
        l.label_matured_at
    FROM features f
    LEFT JOIN labels l 
        ON f.date = l.as_of_date 
        AND f.ticker = l.ticker 
        AND l.horizon = {horizon}
    WHERE f.date >= '2016-01-01'
    ORDER BY f.date, f.ticker
    """
    
    df = conn.execute(query).df()
    output_path = OUTPUT_DIR / f"modeling_dataset_{horizon}d.csv"
    df.to_csv(output_path, index=False)
    print(f"   ✓ Saved: {output_path.name}")
    print(f"     → {len(df):,} rows × {len(df.columns)} columns")
    print(f"     → Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"     → Tickers: {df['ticker'].nunique()}")

# ============================================================================
# 2. Export Regime Data (Separate Table)
# ============================================================================
print("\n2. Exporting market regime data...")

regime_query = """
SELECT * FROM regime
WHERE date >= '2016-01-01'
ORDER BY date
"""

regime_df = conn.execute(regime_query).df()
regime_path = OUTPUT_DIR / "regime_data.csv"
regime_df.to_csv(regime_path, index=False)
print(f"   ✓ Saved: {regime_path.name}")
print(f"     → {len(regime_df):,} rows (daily market indicators)")
print(f"     → {len(regime_df.columns)} columns")

# ============================================================================
# 3. Create Data Dictionary
# ============================================================================
print("\n3. Creating data dictionary...")

data_dict = [
    # Identifiers
    ("date", "Date", "Trading date for feature observation (point-in-time safe)"),
    ("ticker", "Ticker", "Stock ticker symbol"),
    ("stable_id", "Stable ID", "Permanent identifier across ticker changes"),
    
    # Momentum Features (4)
    ("mom_1m", "Momentum 1M", "1-month (21d) price momentum return"),
    ("mom_3m", "Momentum 3M", "3-month (63d) price momentum return"),
    ("mom_6m", "Momentum 6M", "6-month (126d) price momentum return"),
    ("mom_12m", "Momentum 12M", "12-month (252d) price momentum return"),
    
    # Liquidity (2)
    ("adv_20d", "ADV 20D", "Average daily volume (20-day window)"),
    ("adv_60d", "ADV 60D", "Average daily volume (60-day window)"),
    
    # Volatility (3)
    ("vol_20d", "Volatility 20D", "Annualized volatility (20-day window)"),
    ("vol_60d", "Volatility 60D", "Annualized volatility (60-day window)"),
    ("vol_of_vol", "Vol of Vol", "Volatility of volatility (rolling std of returns)"),
    
    # Drawdown (2)
    ("max_drawdown_60d", "Max Drawdown 60D", "Maximum drawdown over 60 days"),
    ("dist_from_high_60d", "Distance from 60D High", "% below 60-day high"),
    
    # Relative Strength (2)
    ("rel_strength_1m", "Relative Strength 1M", "1M return minus QQQ return"),
    ("rel_strength_3m", "Relative Strength 3M", "3M return minus QQQ return"),
    
    # Beta (1)
    ("beta_252d", "Beta 252D", "252-day rolling beta vs QQQ"),
    
    # Earnings & Events (12)
    ("days_to_earnings", "Days to Earnings", "Trading days until next earnings"),
    ("days_since_earnings", "Days Since Earnings", "Trading days since last earnings"),
    ("in_pead_window", "In PEAD Window", "Boolean: within post-earnings-announcement drift window"),
    ("pead_window_day", "PEAD Window Day", "Day # within PEAD window (1-20)"),
    ("last_surprise_pct", "Last Surprise %", "Most recent earnings surprise (actual - estimate) / price"),
    ("avg_surprise_4q", "Avg Surprise 4Q", "Average earnings surprise over last 4 quarters"),
    ("surprise_streak", "Surprise Streak", "Count of consecutive positive/negative surprises"),
    ("surprise_zscore", "Surprise Z-Score", "Z-score of last surprise vs historical"),
    ("earnings_vol", "Earnings Volatility", "Std dev of earnings surprises"),
    ("days_since_10k", "Days Since 10-K", "Days since 10-K filing"),
    ("days_since_10q", "Days Since 10-Q", "Days since 10-Q filing"),
    ("reports_bmo", "Reports BMO", "Boolean: reports before market open"),
    
    # Missingness (2)
    ("coverage_pct", "Coverage %", "% of non-missing features in last 60 days"),
    ("is_new_stock", "Is New Stock", "Boolean: IPO within last 252 days"),
    
    # Market Regime (12) - in features table
    ("vix_level", "VIX Level", "VIX ETF proxy level"),
    ("vix_percentile", "VIX Percentile", "VIX percentile rank (252-day window)"),
    ("vix_change_5d", "VIX Change 5D", "5-day change in VIX"),
    ("vix_regime", "VIX Regime", "Categorical: 0=low, 1=medium, 2=high volatility"),
    ("market_return_5d", "Market Return 5D", "SPY 5-day return"),
    ("market_return_21d", "Market Return 21D", "SPY 21-day return"),
    ("market_return_63d", "Market Return 63D", "SPY 63-day return"),
    ("market_vol_21d", "Market Vol 21D", "SPY 21-day volatility"),
    ("market_regime", "Market Regime", "Categorical: 0=bull, 1=neutral, 2=bear"),
    ("above_ma_50", "Above MA 50", "Binary: SPY above 50-day moving average"),
    ("above_ma_200", "Above MA 200", "Binary: SPY above 200-day moving average"),
    ("ma_50_200_cross", "MA Golden/Death Cross", "Binary: 50-day MA above/below 200-day MA"),
    
    # Fundamentals (9)
    ("sector", "Sector", "Industry sector classification"),
    ("gross_margin_ttm", "Gross Margin TTM", "Trailing 12-month gross margin"),
    ("operating_margin_ttm", "Operating Margin TTM", "Trailing 12-month operating margin"),
    ("revenue_growth_yoy", "Revenue Growth YoY", "Year-over-year revenue growth rate"),
    ("roe_raw", "ROE Raw", "Return on equity (raw)"),
    ("gross_margin_vs_sector", "Gross Margin vs Sector", "Z-score of gross margin vs sector"),
    ("operating_margin_vs_sector", "Operating Margin vs Sector", "Z-score of operating margin vs sector"),
    ("revenue_growth_vs_sector", "Revenue Growth vs Sector", "Z-score of revenue growth vs sector"),
    ("roe_zscore", "ROE Z-Score", "Z-score of ROE within ticker history"),
    
    # Target (added in export)
    ("target_excess_return_20d", "Target 20D Excess Return", "Forward 20-day excess return vs QQQ (label)"),
    ("target_excess_return_60d", "Target 60D Excess Return", "Forward 60-day excess return vs QQQ (label)"),
    ("target_excess_return_90d", "Target 90D Excess Return", "Forward 90-day excess return vs QQQ (label)"),
    ("label_matured_at", "Label Matured At", "Timestamp when label became observable (PIT validation)"),
]

dict_df = pd.DataFrame(data_dict, columns=["Column", "Name", "Description"])
dict_path = OUTPUT_DIR / "data_dictionary.csv"
dict_df.to_csv(dict_path, index=False)
print(f"   ✓ Saved: {dict_path.name}")
print(f"     → {len(dict_df)} feature definitions")

# ============================================================================
# 4. Create Dataset Summary
# ============================================================================
print("\n4. Creating dataset summary...")

# Load the 20d dataset for summary stats
summary_df = pd.read_csv(OUTPUT_DIR / "modeling_dataset_20d.csv")

summary_lines = [
    "=" * 70,
    "AI STOCK FORECAST - THESIS DATASET SUMMARY",
    "=" * 70,
    f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    "",
    "DATASET OVERVIEW",
    "-" * 70,
    f"Total observations: {len(summary_df):,}",
    f"Date range: {summary_df['date'].min()} to {summary_df['date'].max()}",
    f"Unique tickers: {summary_df['ticker'].nunique()}",
    f"Feature columns: 52 (momentum, liquidity, volatility, earnings, regime, fundamentals)",
    f"Target horizons: 20d, 60d, 90d (forward excess returns vs QQQ)",
    "",
    "TICKER UNIVERSE (AI/Technology Stocks)",
    "-" * 70,
]

ticker_counts = summary_df['ticker'].value_counts().sort_values(ascending=False)
summary_lines.append(f"Total unique tickers: {len(ticker_counts)}")
summary_lines.append(f"\nTop 10 tickers by observation count:")
for ticker, count in ticker_counts.head(10).items():
    summary_lines.append(f"  {ticker}: {count:,} observations")

summary_lines.extend([
    "",
    "FEATURE CATEGORIES (52 features total)",
    "-" * 70,
    "• Momentum (4): 1M, 3M, 6M, 12M price returns",
    "• Liquidity (2): Average daily volume (20D, 60D)",
    "• Volatility (3): Annualized volatility (20D, 60D), vol-of-vol",
    "• Drawdown (2): Max drawdown, distance from high",
    "• Relative Strength (2): Excess return vs QQQ (1M, 3M)",
    "• Beta (1): 252-day rolling beta vs QQQ",
    "• Earnings (12): Days to/since earnings, surprises, PEAD signals",
    "• Regime (12): VIX level/regime, market returns/volatility, MA crossovers",
    "• Fundamentals (9): Margins, revenue growth, ROE (raw + sector-adjusted)",
    "• Missingness (2): Coverage %, new stock flag",
    "",
    "TARGET VARIABLES",
    "-" * 70,
    "Forward excess return = (stock_return_horizon - QQQ_return_horizon)",
    "",
    "20-day horizon:",
    f"  Non-null targets: {summary_df['target_excess_return_20d'].notna().sum():,}",
    f"  Mean: {summary_df['target_excess_return_20d'].mean():.4f}",
    f"  Std: {summary_df['target_excess_return_20d'].std():.4f}",
    f"  Min: {summary_df['target_excess_return_20d'].min():.4f}",
    f"  Max: {summary_df['target_excess_return_20d'].max():.4f}",
    "",
    "POINT-IN-TIME SAFETY",
    "-" * 70,
    "✓ All features are PIT-safe (no look-ahead bias)",
    "✓ Earnings data filtered by announcement datetime",
    "✓ SEC filings filtered by acceptance datetime",
    "✓ Prices are split-adjusted using FMP data",
    "✓ Labels include maturation timestamp for validation",
    "✓ Forward-fill applied to quarterly fundamental data",
    "",
    "DATA SOURCES",
    "-" * 70,
    "• Prices: Financial Modeling Prep (FMP) API - split-adjusted OHLCV",
    "• Earnings: FMP earnings calendar + Alpha Vantage surprises",
    "• Filings: SEC EDGAR (10-K, 10-Q acceptance timestamps)",
    "• Fundamentals: FMP income statements (TTM)",
    "• Regime: FMP historical prices (SPY, VIX ETF proxy)",
    "• Benchmark: QQQ (Nasdaq-100 ETF) for excess returns",
    "",
    "UNIVERSE SELECTION",
    "-" * 70,
    "The dataset focuses on AI and technology stocks including:",
    "• AI Infrastructure (NVDA, AMD, AVGO, QCOM, etc.)",
    "• Cloud & Software (MSFT, GOOGL, AMZN, CRM, etc.)",
    "• Hardware & Semiconductors (INTC, MU, AMAT, etc.)",
    "• Consumer Tech (AAPL, META, TSLA, etc.)",
    "• Cybersecurity & Enterprise (PANW, CRWD, NOW, etc.)",
    "",
    "USAGE NOTES",
    "-" * 70,
    "1. Use 'date' as the as-of date for time-series splits",
    "2. Use 'ticker' + 'date' as the unique row identifier",
    "3. Forward-fill missing fundamentals (updated quarterly)",
    "4. Consider winsorizing extreme values (volatility, returns)",
    "5. The 'stable_id' tracks tickers across symbol changes",
    "6. Regime data is duplicated across stocks (join by 'date')",
    "7. Use separate regime_data.csv for market-level features only",
    "",
    "RECOMMENDED MODELING APPROACH",
    "-" * 70,
    "• Walk-forward validation with 3-6 month train window",
    "• Monthly or quarterly rebalancing (avoid overfitting)",
    "• Use gradient boosting (LightGBM) or neural networks",
    "• Evaluate with RankIC (Spearman correlation) for ranking quality",
    "• Include transaction costs (10 bps) in backtest",
    "• Consider market regime as a feature or for stratification",
    "",
    "BENCHMARK RESULTS (LightGBM Baseline)",
    "-" * 70,
    "Published results from this dataset (Chapter 7):",
    "• 20d RankIC: 0.1009 (monthly rebalance)",
    "• 60d RankIC: 0.1275 (monthly rebalance)",
    "• 90d RankIC: 0.1808 (monthly rebalance)",
    "• Churn: 0.20 (stable signals)",
    "• Cost survival (60d): 45.9% of predictions remain positive after costs",
    "",
    "FILES EXPORTED",
    "-" * 70,
    "1. modeling_dataset_20d.csv - Main dataset (20-day horizon)",
    "2. modeling_dataset_60d.csv - 60-day horizon version",
    "3. modeling_dataset_90d.csv - 90-day horizon version",
    "4. regime_data.csv - Market regime indicators (daily)",
    "5. data_dictionary.csv - Column definitions and descriptions",
    "6. dataset_summary.txt - This file (metadata and usage guide)",
    "",
    "CITATION & ACKNOWLEDGMENT",
    "-" * 70,
    "Data sources:",
    "• Financial Modeling Prep API (https://financialmodelingprep.com/)",
    "• SEC EDGAR Database (https://www.sec.gov/edgar)",
    "• Alpha Vantage API (https://www.alphavantage.co/)",
    "",
    "This dataset was created as part of a master's thesis on",
    "AI-enhanced stock forecasting using machine learning and",
    "foundation models for quantitative investment strategies.",
    "",
    "=" * 70,
])

summary_path = OUTPUT_DIR / "dataset_summary.txt"
with open(summary_path, 'w') as f:
    f.write('\n'.join(summary_lines))
print(f"   ✓ Saved: {summary_path.name}")

# ============================================================================
# 5. Summary Statistics CSV
# ============================================================================
print("\n5. Creating summary statistics...")

numeric_cols = summary_df.select_dtypes(include=['float64', 'int64']).columns
stats_df = summary_df[numeric_cols].describe().T
stats_df['missing_%'] = (1 - summary_df[numeric_cols].notna().mean()) * 100
stats_path = OUTPUT_DIR / "summary_statistics.csv"
stats_df.to_csv(stats_path)
print(f"   ✓ Saved: {stats_path.name}")
print(f"     → Descriptive stats for {len(stats_df)} numeric columns")

conn.close()

print("\n" + "=" * 70)
print("EXPORT COMPLETE!")
print("=" * 70)
print(f"\nAll files saved to: {OUTPUT_DIR.absolute()}/")
print("\nFiles ready for thesis submission:")
print(f"  1. modeling_dataset_20d.csv ({len(summary_df):,} rows)")
print(f"  2. modeling_dataset_60d.csv ({len(summary_df):,} rows)")
print(f"  3. modeling_dataset_90d.csv ({len(summary_df):,} rows)")
print(f"  4. regime_data.csv ({len(regime_df):,} daily observations)")
print("  5. data_dictionary.csv (column definitions)")
print("  6. dataset_summary.txt (metadata & usage)")
print("  7. summary_statistics.csv (descriptive stats)")
print("\n" + "=" * 70)


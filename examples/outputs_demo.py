"""
Section 1: System Outputs Demo
==============================

This example demonstrates how to use the signal output structures
defined in Section 1 of the AI Stock Forecaster.

Run with: python examples/outputs_demo.py
"""

import sys
from pathlib import Path
from datetime import date, datetime
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from outputs import (
    # Signal classes
    StockSignal,
    ReturnDistribution,
    SignalDriver,
    HorizonSignals,
    RebalanceSignals,
    LiquidityFlag,
    SignalSource,
    # Ranking classes
    create_ranking_from_signals,
    RankingCategory,
    ConfidenceBucket,
    # Report classes
    generate_signal_report,
    export_signals_to_csv,
    export_signals_to_json,
)


def create_mock_signal(
    ticker: str,
    rebalance_date: date,
    horizon: int,
    expected_return: float,
    confidence: float,
    avg_volume: float = 5_000_000,
) -> StockSignal:
    """Create a mock signal for demonstration."""
    
    # Generate mock return distribution based on expected return
    mean = expected_return
    std = abs(expected_return) * 2 + 0.05  # Higher expected return = higher volatility
    
    distribution = ReturnDistribution(
        percentile_5=mean - 1.65 * std,
        percentile_25=mean - 0.67 * std,
        percentile_50=mean,
        percentile_75=mean + 0.67 * std,
        percentile_95=mean + 1.65 * std,
        mean=mean,
        std=std,
    )
    
    # Create some mock drivers
    drivers = []
    if expected_return > 0:
        drivers.append(SignalDriver(
            feature_name="momentum_12m",
            category="price",
            contribution=expected_return * 0.4,
            description="Strong 12-month price momentum"
        ))
        drivers.append(SignalDriver(
            feature_name="revenue_growth",
            category="fundamental",
            contribution=expected_return * 0.3,
            description="Above-sector revenue growth"
        ))
    else:
        drivers.append(SignalDriver(
            feature_name="volatility",
            category="price",
            contribution=expected_return * 0.5,
            description="Elevated volatility vs peers"
        ))
    
    # Alpha rank score (higher is better)
    alpha_score = expected_return * 10 + np.random.normal(0, 0.1)
    
    # Liquidity flag based on volume
    if avg_volume < 500_000:
        liq_flag = LiquidityFlag.LOW_VOLUME
    else:
        liq_flag = LiquidityFlag.OK
    
    return StockSignal(
        ticker=ticker,
        rebalance_date=rebalance_date,
        horizon_days=horizon,
        benchmark="QQQ",
        expected_excess_return=expected_return,
        return_distribution=distribution,
        alpha_rank_score=alpha_score,
        confidence_score=confidence,
        liquidity_flag=liq_flag,
        avg_daily_volume=avg_volume,
        key_drivers=tuple(drivers),
        source=SignalSource.FUSION,
    )


def main():
    """Demonstrate the outputs module."""
    
    print("=" * 70)
    print("AI STOCK FORECASTER - SECTION 1: SYSTEM OUTPUTS DEMO")
    print("=" * 70)
    print()
    
    # Parameters
    rebalance_date = date(2024, 1, 15)
    horizons = [20, 60, 90]
    benchmark = "QQQ"
    
    # Use a subset of real AI universe for demo
    # Full universe: 100 stocks across 10 categories
    # See src/universe/ai_stocks.py for all tickers
    universe = [
        # AI Compute & Semiconductors
        ("NVDA", 0.15, 0.85, 50_000_000),   # Strong buy, high confidence
        ("AMD", 0.10, 0.75, 30_000_000),    # Buy, high confidence
        ("AVGO", 0.08, 0.80, 5_000_000),    # Buy
        ("INTC", -0.10, 0.65, 30_000_000),  # Strong avoid
        # Cloud & Hyperscalers  
        ("MSFT", 0.08, 0.80, 25_000_000),   # Buy
        ("GOOGL", 0.05, 0.65, 20_000_000),  # Moderate buy
        ("AMZN", 0.03, 0.60, 15_000_000),   # Neutral
        ("META", 0.02, 0.55, 12_000_000),   # Neutral
        # Enterprise Software
        ("CRM", -0.02, 0.50, 5_000_000),    # Neutral
        ("PLTR", 0.12, 0.70, 30_000_000),   # Buy (AI platform)
        # Cybersecurity
        ("CRWD", 0.09, 0.72, 4_000_000),    # Buy
        ("PANW", 0.06, 0.68, 3_000_000),    # Moderate buy
    ]
    
    # -------------------------------------------------------------------------
    # Step 1: Create individual signals
    # -------------------------------------------------------------------------
    print("📊 Step 1: Creating individual stock signals...")
    print()
    
    # Create signals for all horizons
    horizon_signals_dict = {}
    
    for horizon in horizons:
        signals = {}
        for ticker, base_return, confidence, avg_vol in universe:
            # Adjust return based on horizon (longer horizon = more uncertainty)
            horizon_factor = 1 + (horizon - 20) / 100
            adj_return = base_return * horizon_factor + np.random.normal(0, 0.01)
            adj_confidence = confidence - (horizon - 20) / 500  # Slightly lower for longer horizons
            
            signal = create_mock_signal(
                ticker=ticker,
                rebalance_date=rebalance_date,
                horizon=horizon,
                expected_return=adj_return,
                confidence=max(0.3, adj_confidence),
                avg_volume=avg_vol,
            )
            signals[ticker] = signal
        
        horizon_signals_dict[horizon] = HorizonSignals(
            horizon_days=horizon,
            signals=signals,
            rebalance_date=rebalance_date,
            benchmark=benchmark,
        )
    
    # Show a sample signal
    sample_signal = horizon_signals_dict[20].signals["NVDA"]
    print("Sample signal for NVDA (20-day horizon):")
    print(sample_signal.summary())
    print()
    
    # -------------------------------------------------------------------------
    # Step 2: Create cross-sectional rankings
    # -------------------------------------------------------------------------
    print("-" * 70)
    print("📈 Step 2: Creating cross-sectional rankings...")
    print()
    
    ranking_20d = create_ranking_from_signals(horizon_signals_dict[20])
    print(ranking_20d.summary())
    print()
    
    # -------------------------------------------------------------------------
    # Step 3: Create full rebalance signals
    # -------------------------------------------------------------------------
    print("-" * 70)
    print("📋 Step 3: Creating complete rebalance signals...")
    print()
    
    rebalance_signals = RebalanceSignals(
        rebalance_date=rebalance_date,
        horizon_signals=horizon_signals_dict,
        universe_tickers=[t for t, _, _, _ in universe],
        benchmark=benchmark,
        metadata={"demo": True, "version": "0.1.0"},
    )
    
    print(rebalance_signals.summary())
    
    # -------------------------------------------------------------------------
    # Step 4: Generate report
    # -------------------------------------------------------------------------
    print("-" * 70)
    print("📄 Step 4: Generating signal report...")
    print()
    
    report = generate_signal_report(rebalance_signals)
    
    # Print text report (truncated for demo)
    text_report = report.to_text()
    lines = text_report.split("\n")
    print("\n".join(lines[:60]))  # First 60 lines
    print("\n... (truncated for demo) ...\n")
    
    # -------------------------------------------------------------------------
    # Step 5: Export to files
    # -------------------------------------------------------------------------
    print("-" * 70)
    print("💾 Step 5: Exporting signals...")
    print()
    
    # Export to CSV (just show the content, don't write file in demo)
    csv_content = export_signals_to_csv(ranking_20d)
    print("CSV Export (20-day horizon):")
    print(csv_content[:500])
    print("...\n")
    
    # Export to JSON
    json_content = export_signals_to_json(report)
    print("JSON Export (summary):")
    print(json_content[:800])
    print("...\n")
    
    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)
    print()
    print("Section 1: System Outputs provides:")
    print("  ✅ StockSignal - Per-stock signal with distribution & drivers")
    print("  ✅ ReturnDistribution - Full uncertainty quantification (P5/P50/P95)")
    print("  ✅ Prob(Outperform) - Probability of beating benchmark")
    print("  ✅ LiquidityFlag - Capacity/liquidity warnings")
    print("  ✅ CrossSectionalRanking - Ranked lists with categories")
    print("  ✅ SignalReport - Comprehensive exportable reports")
    print("  ✅ CSV/JSON export - Machine-readable outputs")
    print()
    print("CLI commands now available:")
    print("  python -m src.cli build-universe --asof 2024-01-15")
    print("  python -m src.cli score --asof 2024-01-15")
    print("  python -m src.cli make-report --asof 2024-01-15")
    print("  python -m src.cli audit-pit --start 2023-01-01 --end 2024-01-01")
    print()
    print("Next: Section 3 (Data Infrastructure) to fetch real FMP data")


if __name__ == "__main__":
    main()


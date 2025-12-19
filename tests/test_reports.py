"""
Unit Tests for reports.py
=========================

Tests for report generation and export, ensuring:
- CSV export includes all required columns including percentiles
- Text report contains distribution ranges and prob_outperform
- Serialization is correct
"""

import pytest
from datetime import date
import sys
from pathlib import Path
import io

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from outputs.signals import (
    ReturnDistribution,
    StockSignal,
    SignalDriver,
    LiquidityFlag,
    SignalSource,
    HorizonSignals,
    RebalanceSignals,
)
from outputs.rankings import (
    create_ranking_from_signals,
    RankingCategory,
    ConfidenceBucket,
)
from outputs.reports import (
    SignalReport,
    generate_signal_report,
    export_signals_to_csv,
    export_signals_to_json,
)


def create_test_signal(
    ticker: str,
    expected_return: float,
    confidence: float,
    adv: float = 5_000_000,
) -> StockSignal:
    """Helper to create test signals."""
    std = abs(expected_return) * 2 + 0.05
    dist = ReturnDistribution(
        percentile_5=expected_return - 1.65 * std,
        percentile_25=expected_return - 0.67 * std,
        percentile_50=expected_return,
        percentile_75=expected_return + 0.67 * std,
        percentile_95=expected_return + 1.65 * std,
        mean=expected_return,
        std=std,
    )
    
    liq_flag = LiquidityFlag.LOW_VOLUME if adv < 500_000 else LiquidityFlag.OK
    
    return StockSignal.create(
        ticker=ticker,
        rebalance_date=date(2024, 1, 15),
        horizon_days=20,
        benchmark="QQQ",
        return_distribution=dist,
        alpha_rank_score=expected_return * 10,
        confidence_score=confidence,
        liquidity_flag=liq_flag,
        avg_daily_volume=adv,
    )


@pytest.fixture
def sample_horizon_signals():
    """Create sample horizon signals for testing."""
    signals = {
        "NVDA": create_test_signal("NVDA", 0.15, 0.85, 50_000_000),
        "AMD": create_test_signal("AMD", 0.10, 0.75, 30_000_000),
        "MSFT": create_test_signal("MSFT", 0.08, 0.80, 25_000_000),
        "GOOGL": create_test_signal("GOOGL", 0.05, 0.65, 20_000_000),
        "INTC": create_test_signal("INTC", -0.10, 0.60, 400_000),  # Low liquidity
    }
    
    return HorizonSignals(
        horizon_days=20,
        signals=signals,
        rebalance_date=date(2024, 1, 15),
        benchmark="QQQ",
    )


@pytest.fixture
def sample_rebalance_signals(sample_horizon_signals):
    """Create sample rebalance signals for all horizons."""
    # Create signals for all horizons
    horizon_signals = {20: sample_horizon_signals}
    
    # Add 60-day horizon
    signals_60 = {
        ticker: create_test_signal(
            ticker, 
            sig.expected_excess_return * 1.2,  # Slightly different
            sig.confidence_score - 0.05,
            sig.avg_daily_volume or 1_000_000,
        )
        for ticker, sig in sample_horizon_signals.signals.items()
    }
    # Need to update horizon_days for 60-day signals
    for ticker in signals_60:
        old = signals_60[ticker]
        signals_60[ticker] = StockSignal.create(
            ticker=ticker,
            rebalance_date=date(2024, 1, 15),
            horizon_days=60,
            benchmark="QQQ",
            return_distribution=old.return_distribution,
            alpha_rank_score=old.alpha_rank_score,
            confidence_score=old.confidence_score,
            liquidity_flag=old.liquidity_flag,
            avg_daily_volume=old.avg_daily_volume,
        )
    
    horizon_signals[60] = HorizonSignals(
        horizon_days=60,
        signals=signals_60,
        rebalance_date=date(2024, 1, 15),
        benchmark="QQQ",
    )
    
    return RebalanceSignals(
        rebalance_date=date(2024, 1, 15),
        horizon_signals=horizon_signals,
        universe_tickers=list(sample_horizon_signals.signals.keys()),
        benchmark="QQQ",
    )


class TestExportSignalsToCSV:
    """Tests for CSV export functionality."""
    
    def test_csv_includes_all_required_columns(self, sample_horizon_signals):
        """Test that CSV has all expected columns including percentiles."""
        ranking = create_ranking_from_signals(sample_horizon_signals)
        csv_content = export_signals_to_csv(ranking)
        
        # Parse header
        lines = csv_content.strip().split('\n')
        header = lines[0].split(',')
        
        # Required columns
        required_columns = [
            'rank',
            'ticker',
            'expected_excess_return',
            'return_p5',
            'return_p25',
            'return_p50',
            'return_p75',
            'return_p95',
            'return_std',
            'prob_outperform',
            'confidence_score',
            'liquidity_flag',
        ]
        
        for col in required_columns:
            assert col in header, f"Missing column: {col}"
    
    def test_csv_percentiles_are_numeric(self, sample_horizon_signals):
        """Test that percentile values are valid numbers."""
        ranking = create_ranking_from_signals(sample_horizon_signals)
        csv_content = export_signals_to_csv(ranking)
        
        lines = csv_content.strip().split('\n')
        header = lines[0].split(',')
        
        # Find column indices
        p5_idx = header.index('return_p5')
        p50_idx = header.index('return_p50')
        p95_idx = header.index('return_p95')
        
        # Check first data row
        data_row = lines[1].split(',')
        
        p5 = float(data_row[p5_idx])
        p50 = float(data_row[p50_idx])
        p95 = float(data_row[p95_idx])
        
        # P5 < P50 < P95
        assert p5 < p50 < p95
    
    def test_csv_includes_prob_outperform(self, sample_horizon_signals):
        """Test that prob_outperform is included and valid."""
        ranking = create_ranking_from_signals(sample_horizon_signals)
        csv_content = export_signals_to_csv(ranking)
        
        lines = csv_content.strip().split('\n')
        header = lines[0].split(',')
        prob_idx = header.index('prob_outperform')
        
        # Check all rows have valid probabilities
        for line in lines[1:]:
            values = line.split(',')
            prob = float(values[prob_idx])
            assert 0 <= prob <= 1
    
    def test_csv_includes_liquidity_flag(self, sample_horizon_signals):
        """Test that liquidity flags are included."""
        ranking = create_ranking_from_signals(sample_horizon_signals)
        csv_content = export_signals_to_csv(ranking)
        
        assert 'liquidity_flag' in csv_content
        # INTC has low liquidity
        assert 'low_volume' in csv_content


class TestSignalReportToText:
    """Tests for text report generation."""
    
    def test_to_text_contains_header(self, sample_rebalance_signals):
        """Test that text report has proper header."""
        report = generate_signal_report(sample_rebalance_signals)
        text = report.to_text()
        
        assert "AI STOCK FORECASTER" in text
        assert "SIGNAL REPORT" in text
        assert "2024-01-15" in text
    
    def test_to_text_contains_top_10_table(self, sample_rebalance_signals):
        """Test that text report contains TOP 10 BUY SIGNALS table."""
        report = generate_signal_report(sample_rebalance_signals)
        text = report.to_text()
        
        assert "TOP 10 BUY SIGNALS" in text
    
    def test_to_text_contains_distribution_range(self, sample_rebalance_signals):
        """Test that text report shows P5/P50/P95 range."""
        report = generate_signal_report(sample_rebalance_signals)
        text = report.to_text()
        
        # Should contain range indicators
        assert "P5/P50/P95" in text or "Range" in text
    
    def test_to_text_contains_prob_outperform(self, sample_rebalance_signals):
        """Test that text report shows probability of outperformance."""
        report = generate_signal_report(sample_rebalance_signals)
        text = report.to_text()
        
        # Should have P(>0) column header or similar
        assert "P(>0)" in text or "Prob" in text
    
    def test_to_text_contains_legend(self, sample_rebalance_signals):
        """Test that text report includes legend explaining abbreviations."""
        report = generate_signal_report(sample_rebalance_signals)
        text = report.to_text()
        
        assert "LEGEND" in text
        assert "E[r]" in text or "Expected" in text
    
    def test_to_text_shows_liquidity_warnings(self, sample_rebalance_signals):
        """Test that liquidity warnings appear in report."""
        report = generate_signal_report(sample_rebalance_signals)
        text = report.to_text()
        
        # INTC should have a liquidity warning
        # The ⚠ symbol or "Liq" column should indicate issues
        assert "Liq" in text or "liquidity" in text.lower()


class TestSignalReportSerialization:
    """Tests for report serialization."""
    
    def test_to_dict_structure(self, sample_rebalance_signals):
        """Test that to_dict returns correct structure."""
        report = generate_signal_report(sample_rebalance_signals)
        d = report.to_dict()
        
        assert "metadata" in d
        assert "summary_stats" in d
        assert "rankings" in d
        
        assert d["metadata"]["benchmark"] == "QQQ"
        assert d["metadata"]["universe_size"] == 5
    
    def test_json_export(self, sample_rebalance_signals):
        """Test JSON export produces valid JSON."""
        import json
        
        report = generate_signal_report(sample_rebalance_signals)
        json_str = export_signals_to_json(report)
        
        # Should be valid JSON
        parsed = json.loads(json_str)
        
        assert "metadata" in parsed
        assert "rankings" in parsed


class TestRankingCreation:
    """Tests for ranking creation from signals."""
    
    def test_ranking_order(self, sample_horizon_signals):
        """Test that stocks are ranked by alpha_rank_score."""
        ranking = create_ranking_from_signals(sample_horizon_signals)
        
        scores = [s.signal.alpha_rank_score for s in ranking.ranked_stocks]
        assert scores == sorted(scores, reverse=True)
    
    def test_category_assignment(self, sample_horizon_signals):
        """Test that categories are assigned based on percentiles."""
        ranking = create_ranking_from_signals(sample_horizon_signals)
        
        # Top stock should be top_buy or buy
        top_stock = ranking.ranked_stocks[0]
        assert top_stock.category in (RankingCategory.TOP_BUY, RankingCategory.BUY)
        
        # Bottom stock should be avoid or strong_avoid
        bottom_stock = ranking.ranked_stocks[-1]
        assert bottom_stock.category in (RankingCategory.AVOID, RankingCategory.STRONG_AVOID)
    
    def test_confidence_bucket_assignment(self, sample_horizon_signals):
        """Test that confidence buckets are correctly assigned."""
        ranking = create_ranking_from_signals(sample_horizon_signals)
        
        for stock in ranking.ranked_stocks:
            if stock.signal.confidence_score >= 0.7:
                assert stock.confidence_bucket == ConfidenceBucket.HIGH
            elif stock.signal.confidence_score >= 0.4:
                assert stock.confidence_bucket == ConfidenceBucket.MEDIUM
            else:
                assert stock.confidence_bucket == ConfidenceBucket.LOW


class TestPipelineInvariants:
    """Tests for pipeline-level invariants (PIT-safe, survivorship)."""
    
    def test_universe_result_metadata_stored(self):
        """Test that universe construction stores ticker metadata."""
        from pipelines.universe_pipeline import run_universe_construction, TickerMetadata
        
        result = run_universe_construction(
            asof_date=date(2024, 1, 15),
            max_size=10,
        )
        
        # Should have metadata for each ticker
        assert len(result.ticker_metadata) > 0
        
        # Each metadata should have required fields
        for ticker, meta in result.ticker_metadata.items():
            assert isinstance(meta, TickerMetadata)
            assert meta.ticker == ticker
            # Should have price/volume/mcap data
            assert meta.asof_price is not None or not meta.passed_filters
    
    def test_universe_records_exclusion_reasons(self):
        """Test that excluded tickers have reasons recorded."""
        from pipelines.universe_pipeline import run_universe_construction
        
        result = run_universe_construction(
            asof_date=date(2024, 1, 15),
            max_size=10,
            min_price=5.0,
        )
        
        # Check excluded tickers have reasons
        excluded = result.get_excluded_metadata()
        for ticker, meta in excluded.items():
            assert meta.exclusion_reason is not None
            assert not meta.passed_filters


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


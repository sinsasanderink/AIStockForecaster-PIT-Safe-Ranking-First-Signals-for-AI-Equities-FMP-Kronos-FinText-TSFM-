"""
Signal Report Generation
========================

Creates human-readable reports and exports signals in various formats.
Supports CSV, JSON, and formatted text reports.
"""

import json
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional, Union
import csv
import io

from .signals import StockSignal, HorizonSignals, RebalanceSignals
from .rankings import (
    CrossSectionalRanking, 
    RankedStock, 
    RankingCategory,
    ConfidenceBucket,
    create_ranking_from_signals,
    create_all_rankings,
)


@dataclass
class SignalReport:
    """
    Comprehensive report of signals for a rebalance date.
    
    Combines signals, rankings, and summary statistics into
    a single exportable report.
    """
    rebalance_date: date
    benchmark: str
    universe_size: int
    horizons: List[int]
    rankings: Dict[int, CrossSectionalRanking]
    generated_at: datetime = field(default_factory=datetime.utcnow)
    
    # Summary statistics computed at report generation
    stats: Dict = field(default_factory=dict)
    
    def __post_init__(self):
        """Compute summary statistics."""
        if not self.stats:
            self.stats = self._compute_stats()
    
    def _compute_stats(self) -> Dict:
        """Compute summary statistics across all horizons."""
        stats = {
            "rebalance_date": self.rebalance_date.isoformat(),
            "benchmark": self.benchmark,
            "universe_size": self.universe_size,
            "horizons": {},
        }
        
        for horizon, ranking in self.rankings.items():
            horizon_stats = {
                "total_stocks": len(ranking),
                "category_distribution": ranking.category_counts,
                "confidence_distribution": ranking.confidence_counts,
                "return_spread": ranking.return_spread,
                "mean_expected_return": ranking.mean_expected_return,
                "high_confidence_buys": len(ranking.get_high_confidence_buys()),
            }
            
            # Top 10 tickers
            top_10 = ranking.get_top_n(10)
            horizon_stats["top_10"] = [
                {
                    "ticker": s.ticker,
                    "expected_return": s.signal.expected_excess_return,
                    "confidence": s.signal.confidence_score,
                }
                for s in top_10
            ]
            
            stats["horizons"][horizon] = horizon_stats
        
        return stats
    
    def to_text(self) -> str:
        """
        Generate formatted text report with full distribution metrics.
        
        Shows for each stock:
        - Expected return (mean)
        - P5/P50/P95 range
        - Prob(outperform)
        - Confidence bucket
        - Liquidity flag
        """
        lines = [
            "=" * 80,
            "AI STOCK FORECASTER - SIGNAL REPORT",
            "=" * 80,
            "",
            f"Rebalance Date: {self.rebalance_date}",
            f"Generated At:   {self.generated_at.strftime('%Y-%m-%d %H:%M:%S UTC')}",
            f"Benchmark:      {self.benchmark}",
            f"Universe Size:  {self.universe_size} stocks",
            "",
        ]
        
        for horizon in sorted(self.horizons):
            ranking = self.rankings[horizon]
            
            lines.extend([
                "-" * 80,
                f"  {horizon}-DAY HORIZON",
                "-" * 80,
                "",
            ])
            
            # Category breakdown
            lines.append("  CATEGORY BREAKDOWN:")
            for cat, count in ranking.category_counts.items():
                pct = 100 * count / len(ranking) if ranking else 0
                lines.append(f"    {cat:15s}: {count:3d} ({pct:5.1f}%)")
            
            lines.append("")
            
            # Confidence breakdown
            lines.append("  CONFIDENCE BREAKDOWN:")
            for bucket, count in ranking.confidence_counts.items():
                pct = 100 * count / len(ranking) if ranking else 0
                lines.append(f"    {bucket:15s}: {count:3d} ({pct:5.1f}%)")
            
            lines.extend([
                "",
                f"  Return Spread (Top-Bottom Quintile): {ranking.return_spread:+.2%}",
                "",
            ])
            
            # Top 10 buys with full distribution info
            lines.append("  TOP 10 BUY SIGNALS (with distribution):")
            lines.append("  " + "-" * 76)
            lines.append(
                f"  {'Rk':<3} {'Ticker':<7} {'E[r]':>7} "
                f"{'P5/P50/P95 Range':<28} {'P(>0)':>6} {'Conf':>5} {'Liq':<4}"
            )
            lines.append("  " + "-" * 76)
            
            for stock in ranking.get_top_n(10):
                sig = stock.signal
                dist = sig.return_distribution
                
                # Format distribution range
                range_str = dist.format_range("p5_p50_p95")
                
                # Prob outperform
                prob_str = f"{sig.prob_outperform:.0%}"
                
                # Confidence indicator
                conf_str = f"{sig.confidence_score:.2f}"
                
                # Liquidity flag
                liq_str = "⚠" if sig.has_liquidity_concern else "OK"
                
                lines.append(
                    f"  {stock.rank:<3d} {stock.ticker:<7s} "
                    f"{sig.expected_excess_return:+6.1%} "
                    f"{range_str:<28s} "
                    f"{prob_str:>6s} "
                    f"{conf_str:>5s} "
                    f"{liq_str:<4s}"
                )
            
            lines.append("")
            
            # High confidence buys with details
            hc_buys = ranking.get_high_confidence_buys()
            if hc_buys:
                lines.append(f"  ★ HIGH-CONFIDENCE BUY SIGNALS ({len(hc_buys)} stocks):")
                lines.append("")
                for stock in hc_buys[:5]:
                    sig = stock.signal
                    dist = sig.return_distribution
                    lines.append(f"    {stock.ticker}:")
                    lines.append(f"      Expected:  {sig.expected_excess_return:+.1%}")
                    lines.append(f"      Range:     {dist.format_range()}")
                    lines.append(f"      P(>0):     {sig.prob_outperform:.0%}")
                    lines.append(f"      Confidence: {sig.confidence_score:.2f}")
                    if sig.key_drivers:
                        top_driver = sig.key_drivers[0]
                        lines.append(f"      Top driver: {top_driver.feature_name}")
                    lines.append("")
                if len(hc_buys) > 5:
                    lines.append(f"    ... and {len(hc_buys) - 5} more high-confidence signals")
            
            lines.append("")
            
            # Bottom 5 avoids with distribution
            lines.append("  ⚠ TOP 5 AVOID SIGNALS:")
            for stock in ranking.get_bottom_n(5):
                sig = stock.signal
                dist = sig.return_distribution
                lines.append(
                    f"    {stock.ticker}: {sig.expected_excess_return:+.1%} "
                    f"Range: {dist.format_range()} P(>0): {sig.prob_outperform:.0%}"
                )
            
            lines.extend(["", ""])
        
        lines.extend([
            "=" * 80,
            "LEGEND:",
            "  E[r] = Expected excess return vs benchmark",
            "  P5/P50/P95 = 5th/50th/95th percentile of return distribution",
            "  P(>0) = Probability of outperforming benchmark",
            "  Conf = Calibrated confidence score (0-1)",
            "  Liq = Liquidity flag (OK or ⚠ warning)",
            "  ★ = High confidence signal",
            "=" * 80,
            "END OF REPORT",
            "=" * 80,
        ])
        
        return "\n".join(lines)
    
    def to_dict(self) -> Dict:
        """Convert full report to dictionary."""
        return {
            "metadata": {
                "rebalance_date": self.rebalance_date.isoformat(),
                "generated_at": self.generated_at.isoformat(),
                "benchmark": self.benchmark,
                "universe_size": self.universe_size,
            },
            "summary_stats": self.stats,
            "rankings": {
                str(horizon): ranking.to_dict()
                for horizon, ranking in self.rankings.items()
            },
        }


def generate_signal_report(
    rebalance_signals: RebalanceSignals,
    **ranking_kwargs
) -> SignalReport:
    """
    Generate a complete signal report from rebalance signals.
    
    Args:
        rebalance_signals: Complete signals for all horizons
        **ranking_kwargs: Arguments passed to ranking creation
    
    Returns:
        SignalReport with rankings and statistics
    """
    rankings = create_all_rankings(rebalance_signals, **ranking_kwargs)
    
    return SignalReport(
        rebalance_date=rebalance_signals.rebalance_date,
        benchmark=rebalance_signals.benchmark,
        universe_size=len(rebalance_signals.universe_tickers),
        horizons=rebalance_signals.horizons,
        rankings=rankings,
    )


def export_signals_to_csv(
    ranking: CrossSectionalRanking,
    output_path: Optional[Union[str, Path]] = None,
) -> str:
    """
    Export ranking to CSV format with full distribution metrics.
    
    Includes:
    - Expected return
    - P5/P25/P50/P75/P95 percentiles
    - Prob(outperform)
    - Confidence score
    - Liquidity flag
    
    Args:
        ranking: The ranking to export
        output_path: Optional path to write file (if None, returns string)
    
    Returns:
        CSV content as string
    """
    output = io.StringIO()
    
    fieldnames = [
        "rank",
        "ticker",
        "percentile",
        "category",
        "confidence_bucket",
        "expected_excess_return",
        "return_p5",
        "return_p25",
        "return_p50",
        "return_p75",
        "return_p95",
        "return_std",
        "prob_outperform",
        "alpha_rank_score",
        "confidence_score",
        "liquidity_flag",
        "avg_daily_volume",
    ]
    
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    
    for stock in ranking.ranked_stocks:
        sig = stock.signal
        dist = sig.return_distribution
        writer.writerow({
            "rank": stock.rank,
            "ticker": stock.ticker,
            "percentile": f"{stock.percentile:.1f}",
            "category": stock.category.value,
            "confidence_bucket": stock.confidence_bucket.value,
            "expected_excess_return": f"{sig.expected_excess_return:.4f}",
            "return_p5": f"{dist.percentile_5:.4f}",
            "return_p25": f"{dist.percentile_25:.4f}",
            "return_p50": f"{dist.percentile_50:.4f}",
            "return_p75": f"{dist.percentile_75:.4f}",
            "return_p95": f"{dist.percentile_95:.4f}",
            "return_std": f"{dist.std:.4f}",
            "prob_outperform": f"{sig.prob_outperform:.4f}",
            "alpha_rank_score": f"{sig.alpha_rank_score:.4f}",
            "confidence_score": f"{sig.confidence_score:.4f}",
            "liquidity_flag": sig.liquidity_flag.value,
            "avg_daily_volume": sig.avg_daily_volume if sig.avg_daily_volume else "",
        })
    
    csv_content = output.getvalue()
    
    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(csv_content)
    
    return csv_content


def export_signals_to_json(
    report: SignalReport,
    output_path: Optional[Union[str, Path]] = None,
    indent: int = 2,
) -> str:
    """
    Export full report to JSON format.
    
    Args:
        report: The report to export
        output_path: Optional path to write file (if None, returns string)
        indent: JSON indentation level
    
    Returns:
        JSON content as string
    """
    json_content = json.dumps(report.to_dict(), indent=indent, default=str)
    
    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json_content)
    
    return json_content


def export_all_horizons_to_csv(
    report: SignalReport,
    output_dir: Union[str, Path],
    prefix: str = "signals",
) -> List[Path]:
    """
    Export all horizons to separate CSV files.
    
    Args:
        report: The report containing all horizons
        output_dir: Directory to write files
        prefix: Filename prefix
    
    Returns:
        List of paths to created files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    created_files = []
    date_str = report.rebalance_date.isoformat()
    
    for horizon, ranking in report.rankings.items():
        filename = f"{prefix}_{date_str}_{horizon}d.csv"
        filepath = output_dir / filename
        export_signals_to_csv(ranking, filepath)
        created_files.append(filepath)
    
    # Also export summary JSON
    summary_path = output_dir / f"{prefix}_{date_str}_summary.json"
    export_signals_to_json(report, summary_path)
    created_files.append(summary_path)
    
    return created_files


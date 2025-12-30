"""
Tests for Cost Realism (Chapter 6.4)

CRITICAL: These tests enforce the "can't quietly lie" invariants:
1. Zero turnover → minimal costs
2. Lower ADV → higher costs (monotonicity)
3. Higher AUM → higher costs (monotonicity)
4. Deterministic results
5. ADV missing → conservative penalty
"""

import pytest
import pandas as pd
import numpy as np
from datetime import date

from src.evaluation.costs import (
    TRADING_ASSUMPTIONS,
    compute_participation_rate,
    compute_slippage_bps,
    compute_trade_cost,
    compute_turnover,
    compute_portfolio_costs,
    compute_net_metrics,
    validate_cost_monotonicity,
    validate_aum_monotonicity
)


# ============================================================================
# TEST TRADING ASSUMPTIONS (IMMUTABILITY)
# ============================================================================

class TestTradingAssumptions:
    """Verify trading assumptions are locked and immutable."""
    
    def test_assumptions_frozen(self):
        """Verify TRADING_ASSUMPTIONS is immutable."""
        with pytest.raises(Exception):  # FrozenInstanceError
            TRADING_ASSUMPTIONS.base_cost_bps = 50.0
    
    def test_k_values(self):
        """Verify K values are as specified."""
        assert TRADING_ASSUMPTIONS.k_primary == 10
        assert TRADING_ASSUMPTIONS.k_secondary == 20
    
    def test_base_cost(self):
        """Verify base cost is 20 bps."""
        assert TRADING_ASSUMPTIONS.base_cost_bps == 20.0
        assert TRADING_ASSUMPTIONS.entry_cost_bps == 10.0
        assert TRADING_ASSUMPTIONS.exit_cost_bps == 10.0
    
    def test_slippage_coefficients(self):
        """Verify slippage coefficients are ordered."""
        assert TRADING_ASSUMPTIONS.slippage_coef_low < TRADING_ASSUMPTIONS.slippage_coef_base
        assert TRADING_ASSUMPTIONS.slippage_coef_base < TRADING_ASSUMPTIONS.slippage_coef_high
    
    def test_aum_values(self):
        """Verify AUM values for diagnostics."""
        assert TRADING_ASSUMPTIONS.aum_primary == 1_000_000.0
        assert TRADING_ASSUMPTIONS.aum_secondary == 10_000_000.0


# ============================================================================
# TEST COST MODEL FUNCTIONS
# ============================================================================

class TestCostModel:
    """Test core cost calculation functions."""
    
    def test_participation_rate_normal(self):
        """Test participation rate calculation."""
        trade_value = 100_000  # $100K
        adv = 1_000_000  # $1M ADV
        
        participation = compute_participation_rate(trade_value, adv)
        assert participation == 0.1  # 10% of ADV
    
    def test_participation_rate_high(self):
        """Test participation rate is capped at 100%."""
        trade_value = 2_000_000  # $2M
        adv = 1_000_000  # $1M ADV
        
        participation = compute_participation_rate(trade_value, adv)
        assert participation == 1.0  # Capped at 100%
    
    def test_participation_rate_missing_adv(self):
        """Test participation rate with missing ADV."""
        trade_value = 100_000
        adv = np.nan
        
        participation = compute_participation_rate(trade_value, adv)
        assert participation == 1.0  # Conservative: assume high participation
    
    def test_slippage_square_root(self):
        """Test slippage follows square-root law."""
        coef = 10.0
        
        # 4x participation → 2x slippage (sqrt relationship)
        slip_1 = compute_slippage_bps(0.01, coef)
        slip_4 = compute_slippage_bps(0.04, coef)
        
        assert slip_4 / slip_1 == pytest.approx(2.0, rel=0.01)
    
    def test_slippage_floor_and_cap(self):
        """Test slippage floor and cap are enforced."""
        coef = 100.0  # High coefficient
        
        # Very low participation → floor
        slip_low = compute_slippage_bps(0.0001, coef, floor_bps=1.0, cap_bps=100.0)
        assert slip_low == 1.0
        
        # Very high participation → cap
        slip_high = compute_slippage_bps(1.0, coef, floor_bps=1.0, cap_bps=100.0)
        assert slip_high == 100.0
    
    def test_trade_cost_breakdown(self):
        """Test trade cost is base + slippage."""
        trade_value = 100_000  # $100K
        adv = 1_000_000  # $1M ADV
        base_cost = 20.0
        slippage_coef = 10.0
        
        cost = compute_trade_cost(
            trade_value,
            adv,
            base_cost_bps=base_cost,
            slippage_coef=slippage_coef
        )
        
        # Should have base + slippage
        assert cost["base_cost_bps"] == base_cost
        assert cost["slippage_bps"] > 0
        assert cost["total_cost_bps"] == cost["base_cost_bps"] + cost["slippage_bps"]
        assert not cost["adv_missing"]
    
    def test_trade_cost_missing_adv(self):
        """Test trade cost with missing ADV applies penalty."""
        trade_value = 100_000
        adv = np.nan
        
        cost = compute_trade_cost(trade_value, adv)
        
        assert cost["adv_missing"]
        assert cost["total_cost_bps"] == TRADING_ASSUMPTIONS.adv_missing_penalty_bps
        assert pd.isna(cost["participation_rate"])


# ============================================================================
# TEST MONOTONICITY INVARIANTS
# ============================================================================

class TestMonotonicity:
    """Test cost monotonicity invariants (CRITICAL for "can't lie")."""
    
    def test_lower_adv_higher_cost(self):
        """Lower ADV → higher cost (monotonicity)."""
        trade_value = 100_000
        adv_low = 500_000
        adv_high = 2_000_000
        slippage_coef = 10.0
        
        cost_low_adv = compute_trade_cost(trade_value, adv_low, slippage_coef=slippage_coef)
        cost_high_adv = compute_trade_cost(trade_value, adv_high, slippage_coef=slippage_coef)
        
        # Lower ADV → higher cost
        assert cost_low_adv["total_cost_bps"] > cost_high_adv["total_cost_bps"]
        
        # Validate using helper
        assert validate_cost_monotonicity(trade_value, adv_low, adv_high, slippage_coef)
    
    def test_higher_aum_higher_cost(self):
        """Higher AUM → higher cost (monotonicity)."""
        adv = 1_000_000
        trade_value_low = 50_000
        trade_value_high = 200_000
        slippage_coef = 10.0
        
        cost_low = compute_trade_cost(trade_value_low, adv, slippage_coef=slippage_coef)
        cost_high = compute_trade_cost(trade_value_high, adv, slippage_coef=slippage_coef)
        
        # Higher trade value → higher cost
        assert cost_high["total_cost_bps"] > cost_low["total_cost_bps"]
        
        # Validate using helper
        assert validate_aum_monotonicity(trade_value_low, trade_value_high, adv, slippage_coef)
    
    def test_zero_trade_minimal_cost(self):
        """Zero trade value → minimal cost."""
        adv = 1_000_000
        slippage_coef = 10.0
        
        cost = compute_trade_cost(0.0, adv, slippage_coef=slippage_coef)
        
        # Should only have base cost (no slippage from zero participation)
        assert cost["participation_rate"] == 0.0
        assert cost["slippage_bps"] == 0.0
        assert cost["total_cost_bps"] == TRADING_ASSUMPTIONS.base_cost_bps


# ============================================================================
# TEST TURNOVER CALCULATION
# ============================================================================

class TestTurnover:
    """Test turnover calculation."""
    
    def test_zero_turnover(self):
        """Identical portfolios → zero turnover."""
        weights = pd.Series([0.5, 0.3, 0.2], index=["A", "B", "C"])
        
        turnover = compute_turnover(weights, weights)
        assert turnover == 0.0
    
    def test_full_turnover(self):
        """Complete replacement → 100% turnover."""
        weights_old = pd.Series([0.5, 0.5], index=["A", "B"])
        weights_new = pd.Series([0.5, 0.5], index=["C", "D"])
        
        turnover = compute_turnover(weights_old, weights_new)
        assert turnover == 1.0
    
    def test_partial_turnover(self):
        """Partial replacement → intermediate turnover."""
        weights_old = pd.Series([0.5, 0.5], index=["A", "B"])
        weights_new = pd.Series([0.5, 0.5], index=["A", "C"])
        
        turnover = compute_turnover(weights_old, weights_new)
        # A stays (0.5 → 0.5): 0 change
        # B exits (0.5 → 0): 0.5 change
        # C enters (0 → 0.5): 0.5 change
        # Total change = 1.0, turnover = 1.0 / 2 = 0.5
        assert turnover == 0.5
    
    def test_rebalance_turnover(self):
        """Rebalancing existing holdings → turnover from weight changes."""
        weights_old = pd.Series([0.6, 0.4], index=["A", "B"])
        weights_new = pd.Series([0.5, 0.5], index=["A", "B"])
        
        turnover = compute_turnover(weights_old, weights_new)
        # A: 0.6 → 0.5 = -0.1
        # B: 0.4 → 0.5 = +0.1
        # Total change = 0.2, turnover = 0.2 / 2 = 0.1
        assert turnover == pytest.approx(0.1, rel=0.001)


# ============================================================================
# TEST PORTFOLIO COST COMPUTATION
# ============================================================================

class TestPortfolioCosts:
    """Test portfolio-level cost computation."""
    
    def test_single_date_portfolio(self):
        """Test cost computation for single date portfolio."""
        # Create sample data
        df = pd.DataFrame({
            "stable_id": ["A", "B", "C", "D", "E"],
            "score": [1.0, 0.8, 0.6, 0.4, 0.2],
            "adv_20d": [1_000_000, 2_000_000, 500_000, 3_000_000, 1_500_000],
            "excess_return": [0.05, 0.03, 0.02, 0.01, -0.01]
        })
        
        k = 3
        aum = 1_000_000
        
        result = compute_portfolio_costs(
            df,
            k=k,
            aum=aum,
            slippage_coef=10.0
        )
        
        assert result["success"]
        assert result["n_stocks"] == k
        assert result["turnover"] == 1.0  # First rebalance = full deployment
        assert result["total_cost_pct"] > 0
        assert result["avg_cost_bps"] >= TRADING_ASSUMPTIONS.base_cost_bps
        assert len(result["portfolio_weights"]) == k
    
    def test_insufficient_stocks(self):
        """Test portfolio when insufficient stocks available."""
        df = pd.DataFrame({
            "stable_id": ["A", "B"],
            "score": [1.0, 0.8],
            "adv_20d": [1_000_000, 2_000_000]
        })
        
        k = 5  # Want 5, only have 2
        aum = 1_000_000
        
        result = compute_portfolio_costs(df, k=k, aum=aum)
        
        assert not result["success"]
        assert result["reason"] == "insufficient_stocks"
    
    def test_zero_turnover_minimal_cost(self):
        """Test that zero turnover results in minimal cost."""
        df = pd.DataFrame({
            "stable_id": ["A", "B", "C"],
            "score": [1.0, 0.8, 0.6],
            "adv_20d": [1_000_000, 2_000_000, 3_000_000]
        })
        
        k = 3
        aum = 1_000_000
        
        # First call: get initial portfolio
        result1 = compute_portfolio_costs(df, k=k, aum=aum, slippage_coef=10.0)
        
        # Second call: same portfolio (no changes)
        result2 = compute_portfolio_costs(
            df,
            k=k,
            aum=aum,
            slippage_coef=10.0,
            prev_portfolio=result1["portfolio_weights"]
        )
        
        # Zero turnover → zero cost (since no trades)
        assert result2["turnover"] == 0.0
        assert result2["total_cost_pct"] == 0.0


# ============================================================================
# TEST NET METRICS COMPUTATION
# ============================================================================

class TestNetMetrics:
    """Test net-of-cost metrics computation."""
    
    def test_net_avg_er(self):
        """Test net average excess return calculation."""
        gross_metrics = {
            "avg_excess_return": 0.05,  # 5% gross
            "hit_rate": 0.60
        }
        
        cost_pct = 0.002  # 20 bps = 0.2%
        
        net = compute_net_metrics(gross_metrics, cost_pct)
        
        assert net["gross_avg_er"] == 0.05
        assert net["cost_pct"] == 0.002
        assert net["net_avg_er"] == pytest.approx(0.048, rel=0.001)
        assert net["alpha_survives"]  # Still positive after costs
    
    def test_alpha_does_not_survive(self):
        """Test alpha that doesn't survive costs."""
        gross_metrics = {
            "avg_excess_return": 0.001,  # Only 10 bps gross
        }
        
        cost_pct = 0.002  # 20 bps cost
        
        net = compute_net_metrics(gross_metrics, cost_pct)
        
        assert net["net_avg_er"] < 0
        assert not net["alpha_survives"]


# ============================================================================
# TEST DETERMINISM
# ============================================================================

class TestDeterminism:
    """Test that cost calculations are deterministic."""
    
    def test_same_inputs_same_outputs(self):
        """Same inputs → same outputs."""
        trade_value = 100_000
        adv = 1_000_000
        slippage_coef = 10.0
        
        cost1 = compute_trade_cost(trade_value, adv, slippage_coef=slippage_coef)
        cost2 = compute_trade_cost(trade_value, adv, slippage_coef=slippage_coef)
        
        assert cost1 == cost2
    
    def test_portfolio_determinism(self):
        """Portfolio costs are deterministic."""
        df = pd.DataFrame({
            "stable_id": ["A", "B", "C", "D", "E"],
            "score": [1.0, 0.8, 0.6, 0.4, 0.2],
            "adv_20d": [1_000_000] * 5
        })
        
        result1 = compute_portfolio_costs(df, k=3, aum=1_000_000, slippage_coef=10.0)
        result2 = compute_portfolio_costs(df, k=3, aum=1_000_000, slippage_coef=10.0)
        
        assert result1["total_cost_pct"] == result2["total_cost_pct"]
        assert result1["avg_cost_bps"] == result2["avg_cost_bps"]


# ============================================================================
# TEST EDGE CASES
# ============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_all_missing_adv(self):
        """Test portfolio with all missing ADV."""
        df = pd.DataFrame({
            "stable_id": ["A", "B", "C"],
            "score": [1.0, 0.8, 0.6],
            "adv_20d": [np.nan, np.nan, np.nan]
        })
        
        result = compute_portfolio_costs(df, k=3, aum=1_000_000)
        
        assert result["success"]
        assert result["n_missing_adv"] == 3
        # Should apply penalty to all trades
        assert result["avg_cost_bps"] >= TRADING_ASSUMPTIONS.adv_missing_penalty_bps
    
    def test_very_small_adv(self):
        """Test with very small ADV (high participation)."""
        df = pd.DataFrame({
            "stable_id": ["A", "B", "C"],
            "score": [1.0, 0.8, 0.6],
            "adv_20d": [1000, 2000, 3000]  # Very small ADV
        })
        
        k = 3
        aum = 1_000_000  # Large AUM relative to ADV
        
        # Use high slippage coefficient to test high-cost scenario
        result = compute_portfolio_costs(df, k=k, aum=aum, slippage_coef=50.0)
        
        # Should have very high costs due to high participation
        # With participation ≈ 1.0 and coef=50, slippage = 50 bps + base 20 = 70 bps
        assert result["avg_cost_bps"] > 50.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


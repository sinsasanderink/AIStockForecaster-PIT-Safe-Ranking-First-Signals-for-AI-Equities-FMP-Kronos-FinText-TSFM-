"""
Cost Realism (Chapter 6.4)

Implements cost overlay for "does alpha survive?" diagnostic.

CRITICAL: Costs affect portfolio metrics (Top-K avg ER, spreads), NOT RankIC.
RankIC measures ranking skill; costs measure implementability.

PHILOSOPHY: Use sensitivity bands (low/base/high), not one "true" value.
This answers "does it survive?" without pretending we know exact slippage.

LOCKED TRADING ASSUMPTIONS: See TRADING_ASSUMPTIONS for canonical parameters.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# LOCKED TRADING ASSUMPTIONS
# ============================================================================

@dataclass(frozen=True)
class TradingAssumptions:
    """
    LOCKED TRADING ASSUMPTIONS — SINGLE SOURCE OF TRUTH
    
    These define how we model costs for diagnostic purposes.
    
    Portfolio Definition:
        - Long-only Top-K equal-weight
        - K = 10 (primary), 20 (secondary)
        - No shorts, no leverage
    
    Rebalance Timing:
        - On as-of close (consistent with label entry price)
        - First trading day of month (or quarter)
    
    Turnover Definition:
        - Derived from weight changes between consecutive rebalance dates
        - Uses stable_id (not ticker)
        - Does NOT cross fold boundaries
    
    AUM Assumption:
        - Fixed notionals for diagnostics
        - $1M (primary), $10M (secondary)
        - Gives meaning to ADV scaling
    
    Position Sizing:
        - Equal weight (1/K per stock)
        - No volatility targeting (avoid complexity)
        - No leverage
    """
    
    # Portfolio parameters
    k_primary: int = 10
    k_secondary: int = 20
    
    # AUM for diagnostics (USD)
    aum_primary: float = 1_000_000.0  # $1M
    aum_secondary: float = 10_000_000.0  # $10M
    
    # Base trading cost (bps round-trip)
    base_cost_bps: float = 20.0  # 20 bps round-trip for liquid large caps
    
    # Cost split (entry + exit)
    entry_cost_bps: float = 10.0  # Half of round-trip
    exit_cost_bps: float = 10.0   # Half of round-trip
    
    # Slippage model parameters (square-root impact)
    # slippage_bps = c * sqrt(participation_rate)
    # where participation_rate = trade_value / adv_dollars
    slippage_coef_low: float = 5.0     # Low scenario
    slippage_coef_base: float = 10.0   # Base scenario
    slippage_coef_high: float = 20.0   # High scenario
    
    # Slippage caps (prevent absurd tails)
    slippage_floor_bps: float = 0.0    # Minimum slippage
    slippage_cap_bps: float = 500.0    # Maximum slippage (5%)
    
    # ADV missing value handling
    adv_missing_penalty_bps: float = 100.0  # Conservative penalty if ADV unknown


# Singleton instance
TRADING_ASSUMPTIONS = TradingAssumptions()


# ============================================================================
# COST MODEL FUNCTIONS
# ============================================================================

def compute_participation_rate(
    trade_value: float,
    adv_dollars: float,
    min_adv: float = 1000.0
) -> float:
    """
    Compute participation rate for slippage calculation.
    
    Participation rate = trade_value / adv_dollars
    
    Args:
        trade_value: Dollar value of trade
        adv_dollars: Average daily volume in dollars
        min_adv: Minimum ADV to avoid division by zero
        
    Returns:
        Participation rate (capped at reasonable levels)
    """
    if pd.isna(adv_dollars) or adv_dollars < min_adv:
        # Conservative: assume high participation if ADV unknown
        return 1.0
    
    participation = trade_value / adv_dollars
    
    # Cap at 100% (can't trade more than daily volume in one day)
    return min(participation, 1.0)


def compute_slippage_bps(
    participation_rate: float,
    slippage_coef: float,
    floor_bps: float = TRADING_ASSUMPTIONS.slippage_floor_bps,
    cap_bps: float = TRADING_ASSUMPTIONS.slippage_cap_bps
) -> float:
    """
    Compute slippage in basis points using square-root impact model.
    
    slippage_bps = c * sqrt(participation_rate)
    
    Square-root model is standard in market microstructure literature:
    - Monotone (higher participation → higher cost)
    - Concave (diminishing marginal impact)
    - Empirically validated
    
    Args:
        participation_rate: Trade value / ADV
        slippage_coef: Coefficient (low/base/high scenario)
        floor_bps: Minimum slippage
        cap_bps: Maximum slippage
        
    Returns:
        Slippage in basis points
    """
    if participation_rate <= 0:
        return floor_bps
    
    # Square-root impact
    slippage = slippage_coef * np.sqrt(participation_rate)
    
    # Apply floor and cap
    slippage = max(floor_bps, min(cap_bps, slippage))
    
    return slippage


def compute_trade_cost(
    trade_value: float,
    adv_dollars: float,
    base_cost_bps: float = TRADING_ASSUMPTIONS.base_cost_bps,
    slippage_coef: float = TRADING_ASSUMPTIONS.slippage_coef_base,
    adv_missing_penalty: float = TRADING_ASSUMPTIONS.adv_missing_penalty_bps
) -> Dict[str, float]:
    """
    Compute total trading cost for a single trade.
    
    Total cost = base_cost + slippage
    
    Args:
        trade_value: Dollar value of trade
        adv_dollars: Average daily volume in dollars
        base_cost_bps: Base cost in bps (always applied)
        slippage_coef: Slippage coefficient for sensitivity
        adv_missing_penalty: Penalty if ADV is missing
        
    Returns:
        Dictionary with cost breakdown
    """
    # Handle missing ADV
    if pd.isna(adv_dollars):
        return {
            "base_cost_bps": base_cost_bps,
            "slippage_bps": adv_missing_penalty - base_cost_bps,  # Extra penalty
            "total_cost_bps": adv_missing_penalty,
            "participation_rate": np.nan,
            "adv_missing": True
        }
    
    # Compute participation rate
    participation = compute_participation_rate(trade_value, adv_dollars)
    
    # Compute slippage
    slippage = compute_slippage_bps(participation, slippage_coef)
    
    # Total cost
    total_cost = base_cost_bps + slippage
    
    return {
        "base_cost_bps": base_cost_bps,
        "slippage_bps": slippage,
        "total_cost_bps": total_cost,
        "participation_rate": participation,
        "adv_missing": False
    }


# ============================================================================
# TURNOVER CALCULATION
# ============================================================================

def compute_turnover(
    weights_prev: pd.Series,
    weights_curr: pd.Series
) -> float:
    """
    Compute turnover between two portfolio weights.
    
    Turnover = sum(|w_new - w_old|) / 2
    
    This is the standard definition: fraction of portfolio that changes.
    Divide by 2 because buys and sells are counted separately.
    
    Args:
        weights_prev: Previous portfolio weights (indexed by stable_id)
        weights_curr: Current portfolio weights (indexed by stable_id)
        
    Returns:
        Turnover fraction (0 = no change, 1 = complete replacement)
    """
    # Align indices (fill missing with 0)
    all_ids = weights_prev.index.union(weights_curr.index)
    w_prev = weights_prev.reindex(all_ids, fill_value=0.0)
    w_curr = weights_curr.reindex(all_ids, fill_value=0.0)
    
    # Turnover = sum of absolute weight changes / 2
    turnover = (w_curr - w_prev).abs().sum() / 2.0
    
    return turnover


# ============================================================================
# NET-OF-COST PORTFOLIO METRICS
# ============================================================================

def compute_portfolio_costs(
    df: pd.DataFrame,
    k: int,
    aum: float,
    score_col: str = "score",
    adv_col: str = "adv_20d",
    stable_id_col: str = "stable_id",
    slippage_coef: float = TRADING_ASSUMPTIONS.slippage_coef_base,
    prev_portfolio: Optional[pd.Series] = None
) -> Dict[str, any]:
    """
    Compute costs for a Top-K equal-weight portfolio at one rebalance date.
    
    Args:
        df: DataFrame for single date with score, adv, stable_id
        k: Top-K size
        aum: Portfolio AUM in dollars
        score_col: Column name for scores
        adv_col: Column name for ADV in dollars
        stable_id_col: Column name for stable IDs
        slippage_coef: Slippage coefficient
        prev_portfolio: Previous portfolio weights (optional, for turnover)
        
    Returns:
        Dictionary with costs and portfolio details
    """
    # Rank and select Top-K
    df = df.copy()
    df = df.dropna(subset=[score_col])
    
    if len(df) < k:
        logger.warning(f"Only {len(df)} stocks available, need {k}")
        return {
            "success": False,
            "n_stocks": len(df),
            "reason": "insufficient_stocks"
        }
    
    # Rank with deterministic tie-breaking (use existing function)
    from .metrics import rank_with_ties
    df["rank"] = rank_with_ties(df[[score_col, stable_id_col]], score_col)
    
    # Select Top-K
    top_k = df[df["rank"] <= k].copy()
    
    # Equal-weight portfolio
    target_weight = 1.0 / k
    top_k["target_weight"] = target_weight
    top_k["target_value"] = aum * target_weight
    
    # Current portfolio weights (as Series indexed by stable_id)
    curr_portfolio = pd.Series(
        target_weight,
        index=top_k[stable_id_col]
    )
    
    # Compute turnover
    if prev_portfolio is not None:
        turnover = compute_turnover(prev_portfolio, curr_portfolio)
    else:
        # First rebalance: full deployment
        turnover = 1.0
    
    # Compute trade costs for each stock
    costs = []
    for _, row in top_k.iterrows():
        # Determine trade value
        prev_weight = prev_portfolio.get(row[stable_id_col], 0.0) if prev_portfolio is not None else 0.0
        trade_weight = abs(target_weight - prev_weight)
        trade_value = aum * trade_weight
        
        # Get ADV
        adv_value = row.get(adv_col, np.nan)
        
        # Compute cost
        cost = compute_trade_cost(
            trade_value=trade_value,
            adv_dollars=adv_value,
            slippage_coef=slippage_coef
        )
        
        costs.append({
            stable_id_col: row[stable_id_col],
            "trade_value": trade_value,
            "adv_dollars": adv_value,
            **cost
        })
    
    costs_df = pd.DataFrame(costs)
    
    # Aggregate costs
    total_cost_dollars = (costs_df["total_cost_bps"] * costs_df["trade_value"] / 10000).sum()
    total_cost_pct = total_cost_dollars / aum
    
    # Average cost in bps (weighted by trade value)
    if costs_df["trade_value"].sum() > 0:
        avg_cost_bps = (
            costs_df["total_cost_bps"] * costs_df["trade_value"]
        ).sum() / costs_df["trade_value"].sum()
    else:
        avg_cost_bps = 0.0
    
    return {
        "success": True,
        "n_stocks": k,
        "turnover": turnover,
        "total_cost_dollars": total_cost_dollars,
        "total_cost_pct": total_cost_pct,
        "avg_cost_bps": avg_cost_bps,
        "n_missing_adv": costs_df["adv_missing"].sum(),
        "portfolio_weights": curr_portfolio,
        "cost_details": costs_df
    }


def compute_net_metrics(
    gross_metrics: Dict[str, float],
    cost_pct: float
) -> Dict[str, float]:
    """
    Compute net-of-cost metrics from gross metrics.
    
    Net excess return = gross excess return - costs
    
    Args:
        gross_metrics: Dictionary with gross metrics (e.g., avg_excess_return)
        cost_pct: Total cost as percentage (e.g., 0.002 for 20 bps)
        
    Returns:
        Dictionary with net metrics
    """
    net = {}
    
    # Net average excess return
    if "avg_excess_return" in gross_metrics:
        net["gross_avg_er"] = gross_metrics["avg_excess_return"]
        net["cost_pct"] = cost_pct
        net["net_avg_er"] = gross_metrics["avg_excess_return"] - cost_pct
        net["alpha_survives"] = net["net_avg_er"] > 0
    
    # Adjust hit rate (conservative: count as miss if net < 0)
    if "hit_rate" in gross_metrics and "n_positive" in gross_metrics:
        net["gross_hit_rate"] = gross_metrics["hit_rate"]
        # Can't easily adjust hit rate without per-stock returns
        # So we leave it as gross (hit rate is about % positive, costs are systematic)
    
    return net


# ============================================================================
# SENSITIVITY ANALYSIS
# ============================================================================

def run_cost_sensitivity(
    eval_df: pd.DataFrame,
    fold_id: str,
    horizon: int,
    k: int = 10,
    aum: float = TRADING_ASSUMPTIONS.aum_primary,
    scenarios: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Run cost sensitivity analysis across low/base/high scenarios.
    
    Args:
        eval_df: DataFrame in EvaluationRow format
        fold_id: Fold identifier
        horizon: Horizon in trading days
        k: Top-K size
        aum: Portfolio AUM
        scenarios: List of scenarios to run (default: all)
        
    Returns:
        DataFrame with sensitivity results
    """
    if scenarios is None:
        scenarios = ["base_only", "low_slippage", "base_slippage", "high_slippage"]
    
    # Map scenarios to slippage coefficients
    scenario_coefs = {
        "base_only": 0.0,  # Only base cost, no slippage
        "low_slippage": TRADING_ASSUMPTIONS.slippage_coef_low,
        "base_slippage": TRADING_ASSUMPTIONS.slippage_coef_base,
        "high_slippage": TRADING_ASSUMPTIONS.slippage_coef_high
    }
    
    # Filter for fold and horizon
    df = eval_df[
        (eval_df["fold_id"] == fold_id) & 
        (eval_df["horizon"] == horizon)
    ].copy()
    
    results = []
    
    for scenario in scenarios:
        slippage_coef = scenario_coefs[scenario]
        
        # Run evaluation with this cost scenario
        # (This is a simplified version; full version would integrate with evaluate_fold)
        
        # For now, just report the scenario setup
        results.append({
            "scenario": scenario,
            "slippage_coef": slippage_coef,
            "base_cost_bps": TRADING_ASSUMPTIONS.base_cost_bps,
            "k": k,
            "aum": aum
        })
    
    return pd.DataFrame(results)


# ============================================================================
# VALIDATION HELPERS
# ============================================================================

def validate_cost_monotonicity(
    trade_value: float,
    adv_low: float,
    adv_high: float,
    slippage_coef: float
) -> bool:
    """
    Validate that costs are monotonic in ADV (lower ADV → higher cost).
    
    Args:
        trade_value: Trade value in dollars
        adv_low: Low ADV
        adv_high: High ADV
        slippage_coef: Slippage coefficient
        
    Returns:
        True if monotonic, False otherwise
    """
    cost_low_adv = compute_trade_cost(trade_value, adv_low, slippage_coef=slippage_coef)
    cost_high_adv = compute_trade_cost(trade_value, adv_high, slippage_coef=slippage_coef)
    
    # Lower ADV should have higher cost
    return cost_low_adv["total_cost_bps"] >= cost_high_adv["total_cost_bps"]


def validate_aum_monotonicity(
    trade_value_low: float,
    trade_value_high: float,
    adv: float,
    slippage_coef: float
) -> bool:
    """
    Validate that costs are monotonic in AUM (higher AUM → higher cost).
    
    Args:
        trade_value_low: Low trade value
        trade_value_high: High trade value
        adv: ADV in dollars
        slippage_coef: Slippage coefficient
        
    Returns:
        True if monotonic, False otherwise
    """
    cost_low = compute_trade_cost(trade_value_low, adv, slippage_coef=slippage_coef)
    cost_high = compute_trade_cost(trade_value_high, adv, slippage_coef=slippage_coef)
    
    # Higher trade value should have higher participation and thus higher cost
    return cost_high["total_cost_bps"] >= cost_low["total_cost_bps"]


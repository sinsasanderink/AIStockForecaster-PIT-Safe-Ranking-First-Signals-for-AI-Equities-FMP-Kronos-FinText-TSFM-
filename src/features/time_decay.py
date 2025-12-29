"""
Time-Decay Sample Weighting
===========================

Exponential time-decay weights for training data.

WHY TIME DECAY MATTERS FOR AI STOCKS:
- AI stocks' business models have changed dramatically over time
- Market microstructure evolves (HFT, retail flow, etc.)
- The "AI regime" (2020+) is fundamentally different from earlier eras
- Recent observations are more relevant for forward predictions

POLICY:
- Use exponentially-decayed sample weights by date
- Half-life determines how quickly old data becomes less relevant
- Per-date normalization ensures each date contributes equally

RECOMMENDED HALF-LIVES:
| Horizon | Half-Life | Rationale |
|---------|-----------|-----------|
| 20d     | 2-3 years | Short-term patterns change faster |
| 60d     | 3-4 years | Medium-term patterns |
| 90d     | 4-5 years | Longer-term trends more stable |

DEFAULT: 3-year half-life is a good starting point.

FORMULA:
    w(t) = 2^(-Δ(t) / half_life)
    
    where Δ(t) = days between t and training end date

NORMALIZATION:
- Normalize weights within each date (cross-sectional)
- This prevents older eras from dominating due to more stocks/dates

HANDLING "STOCK DIDN'T EXIST":
- Don't fill missing years - use survivorship-safe universe
- A young stock gets fewer rows, but recent rows have high weight
- This is exactly the correct behavior

USAGE:
    from src.features.time_decay import compute_time_decay_weights
    
    # After building training DataFrame
    weights = compute_time_decay_weights(
        df=train_df,
        date_col="as_of_date",
        half_life_days=3 * 365,  # 3 years
    )
    
    # Pass to model training
    model.fit(X, y, sample_weight=weights)

WHERE THIS APPLIES:
- Section 6: Evaluation framework (walk-forward training)
- Section 11: Fusion model training
- NOT during feature computation (features use fixed lookback windows)
"""

import logging
from datetime import date, datetime
from typing import Optional, Union

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# Default half-lives by horizon (in days)
DEFAULT_HALF_LIVES = {
    20: 2.5 * 365,   # 2.5 years for 20-day horizon
    60: 3.5 * 365,   # 3.5 years for 60-day horizon
    90: 4.5 * 365,   # 4.5 years for 90-day horizon
}

# General default
DEFAULT_HALF_LIFE_DAYS = 3 * 365  # 3 years


def compute_time_decay_weights(
    df: pd.DataFrame,
    date_col: str = "as_of_date",
    half_life_days: int = DEFAULT_HALF_LIFE_DAYS,
    normalize_per_date: bool = True,
    train_end: Optional[date] = None,
) -> pd.Series:
    """
    Compute exponential time-decay weights with per-date normalization.
    
    Args:
        df: DataFrame with date column
        date_col: Name of the date column
        half_life_days: Half-life in days (default: 3 years)
        normalize_per_date: If True, normalize so each date sums to 1
        train_end: End of training window (default: max date in df)
    
    Returns:
        Series of weights per row (aligned with df index)
    
    Example:
        With half_life=3 years:
        - Data 3 years old → weight = 0.50
        - Data 6 years old → weight = 0.25
        - Data 9 years old → weight = 0.125
    """
    # Convert to datetime
    dates = pd.to_datetime(df[date_col])
    
    # Determine training end date
    if train_end is None:
        train_end_dt = dates.max()
    else:
        train_end_dt = pd.Timestamp(train_end)
    
    # Compute age in days (clip to >= 0)
    age_days = (train_end_dt - dates).dt.days.clip(lower=0)
    
    # Exponential decay: w(t) = 2^(-age / half_life)
    w_date = 2.0 ** (-age_days / float(half_life_days))
    
    if normalize_per_date:
        # Normalize within each date so each date contributes equally
        # This keeps cross-sectional ranking training well-behaved
        df_with_w = df.assign(_w=w_date)
        grp_sum = df_with_w.groupby(date_col)["_w"].transform("sum")
        w_row = w_date / grp_sum
    else:
        w_row = w_date
    
    return w_row.astype(float)


def get_half_life_for_horizon(horizon_days: int) -> int:
    """
    Get recommended half-life for a given prediction horizon.
    
    Args:
        horizon_days: Prediction horizon in trading days
    
    Returns:
        Recommended half-life in calendar days
    """
    if horizon_days in DEFAULT_HALF_LIVES:
        return int(DEFAULT_HALF_LIVES[horizon_days])
    
    # Interpolate for other horizons
    if horizon_days < 20:
        return int(2 * 365)  # 2 years
    elif horizon_days < 60:
        # Linear interpolation between 20d and 60d
        ratio = (horizon_days - 20) / (60 - 20)
        return int(2.5 * 365 + ratio * (3.5 - 2.5) * 365)
    elif horizon_days < 90:
        # Linear interpolation between 60d and 90d
        ratio = (horizon_days - 60) / (90 - 60)
        return int(3.5 * 365 + ratio * (4.5 - 3.5) * 365)
    else:
        return int(5 * 365)  # 5 years for very long horizons


def compute_effective_sample_size(weights: pd.Series) -> float:
    """
    Compute effective sample size given weights.
    
    ESS = (sum(w))^2 / sum(w^2)
    
    This tells you how many "equivalent" unweighted samples you have.
    If ESS is much smaller than n, your effective sample is concentrated
    in recent data (which is the intended behavior).
    """
    w_sum = weights.sum()
    w_sq_sum = (weights ** 2).sum()
    
    if w_sq_sum == 0:
        return 0.0
    
    return (w_sum ** 2) / w_sq_sum


def summarize_weights(
    df: pd.DataFrame,
    weights: pd.Series,
    date_col: str = "as_of_date",
) -> dict:
    """
    Summarize weight distribution for diagnostics.
    
    Returns dict with:
    - n_samples: Total number of samples
    - effective_n: Effective sample size
    - weight_ratio: ESS / n (lower = more concentrated in recent)
    - median_weight_date: Date at which cumulative weight = 0.5
    - by_year: Weight sum by year
    """
    dates = pd.to_datetime(df[date_col])
    
    n = len(df)
    ess = compute_effective_sample_size(weights)
    
    # Find median weight date
    df_sorted = pd.DataFrame({
        "date": dates,
        "weight": weights,
    }).sort_values("date", ascending=False)
    df_sorted["cum_weight"] = df_sorted["weight"].cumsum() / df_sorted["weight"].sum()
    median_idx = (df_sorted["cum_weight"] >= 0.5).idxmax()
    median_date = df_sorted.loc[median_idx, "date"]
    
    # Weight by year
    by_year = df.assign(
        year=dates.dt.year,
        weight=weights,
    ).groupby("year")["weight"].sum()
    
    return {
        "n_samples": n,
        "effective_n": ess,
        "weight_ratio": ess / n if n > 0 else 0,
        "median_weight_date": median_date,
        "by_year": by_year.to_dict(),
    }


# =============================================================================
# CLI/Demo
# =============================================================================

if __name__ == "__main__":
    import sys
    
    print("=" * 60)
    print("TIME DECAY WEIGHTING DEMO")
    print("=" * 60)
    
    # Create sample data spanning 10 years
    np.random.seed(42)
    dates = pd.date_range("2015-01-01", "2024-12-31", freq="B")
    n = len(dates)
    
    df = pd.DataFrame({
        "as_of_date": np.random.choice(dates, size=1000),
        "ticker": np.random.choice(["NVDA", "AMD", "INTC"], size=1000),
        "return": np.random.normal(0.01, 0.05, 1000),
    })
    
    # Compute weights
    weights = compute_time_decay_weights(df, half_life_days=3*365)
    
    # Summarize
    summary = summarize_weights(df, weights)
    
    print(f"\nSample data: {summary['n_samples']} rows")
    print(f"Effective sample size: {summary['effective_n']:.0f}")
    print(f"Weight concentration: {summary['weight_ratio']:.1%}")
    print(f"Median weight date: {summary['median_weight_date'].date()}")
    
    print("\nWeight by year:")
    for year, weight in sorted(summary["by_year"].items()):
        bar = "█" * int(weight * 50)
        print(f"  {year}: {bar} ({weight:.3f})")
    
    print("\n" + "=" * 60)


# Chapter 13.7 — Deployment Policy & Sizing Ablation

## Cursor Knowledge Document

**Purpose:** Complete reference for implementing Section 13.7 of the AI Stock Forecaster.
This document contains all context, research findings, implementation specs, and success
criteria needed to implement the deployment policy variants and ablation study.

**Read this BEFORE writing any code for 13.7.**

---

## 1. PROJECT CONTEXT

### What Is The AI Stock Forecaster?

A research-grade cross-sectional equity ranking system for ~100 AI-exposed U.S. stocks.
The primary model is **LightGBM (LGB)** trained to predict excess returns vs QQQ at
20/60/90 trading-day horizons. Evaluation uses **RankIC** (Spearman correlation between
predicted scores and realized returns, cross-sectionally per date).

### Key System Properties

- **Primary model:** LightGBM (tabular features: momentum, vol, fundamentals, regime)
- **Primary horizon:** 20d (confirmed in holdout, FINAL Sharpe 2.34)
- **Universe:** ~84 AI stocks per date (dynamic, survivorship-safe)
- **Evaluation:** Walk-forward, 109 folds, expanding window, 90-day embargo
- **Holdout:** DEV (2016–2023, 95 months) / FINAL (2024+, 14 months)
- **Shadow portfolio:** Top-10 long, bottom-10 short, equal-weight legs, monthly rebalance, 10bps cost
- **ALL calibration done on DEV only, then frozen for FINAL**

### What Chapters 13.0–13.6 Built

| Section | What It Does | Key Result |
|---------|-------------|------------|
| 13.0 | Residual archive — walk-forward prediction errors for 3 models | 591K LGB rows, rank_loss target |
| 13.1 | g(x) error predictor — LGB regression predicting rank displacement | ρ(g, rank_loss) = 0.192 at 20d |
| 13.2 | a(x) aleatoric baseline — irreducible ranking noise | Tiers 0/1 KILLED (ρ≈0), Tier 2 PASS at 60d |
| 13.3 | ê(x) = max(0, g−a) epistemic signal | Quintile monotonicity ρ=1.0 in FINAL |
| 13.4 | Diagnostics A–F | ê ≠ vol (survives residualization), dominates 3–10× |
| 13.4b | H(t) expert health / G(t) exposure gate | G(t)→0 by Apr 2024, AUROC 0.72 |
| 13.5 | DEUP-normalized conformal intervals | 96× better conditional coverage (gap 20.2% → 0.21%) |
| 13.6 | Shadow portfolio + regime evaluation | Vol-sized wins; DEUP sizing FAILS; regime gate WORKS |

### The Core Problem 13.7 Must Solve

**DEUP inverse-sizing fails because it penalizes the model's strongest signals.**

In cross-sectional ranking:
- g(x) is highest at extreme cross-sectional ranks (cross_sectional_rank is #1 feature)
- Extreme ranks carry the strongest signal (the whole point of top-K/bottom-K)
- w = 1/sqrt(g(x)) systematically de-levers the best ideas
- Vol-sizing avoids this because vol measures stock risk, not prediction quality

**13.6 portfolio results (20d, the numbers to beat):**

| Variant | ALL Sharpe | FINAL Sharpe | Crisis MaxDD |
|---------|:---------:|:----------:|:----------:|
| Raw baseline | 1.14 | 1.37 | −44.1% |
| Vol-sized (A) | **1.18** | **1.68** | −47.3% |
| DEUP-sized (B) | 1.13 | 1.35 | −44.4% |
| Health-only (C) | 0.67 | −0.34 | **−17.5%** |
| Combined (D) | 0.69 | −0.34 | **−18.0%** |

**Key insight from 13.6:** Continuous G(t) throttling destroys recoveries. G(t) should
be binary gate (trade if G≥0.2, else flat). Vol-sizing is the best per-stock method.

**13.7 goal:** Find a combined system (binary gate + smart per-stock sizing) that beats
vol-only on FINAL, using uncertainty information that vol-sizing misses.

---

## 2. THE SIX POLICY VARIANTS TO TEST

All variants use the binary gate: trade only when G(t) ≥ 0.2, otherwise flat (no exposure).
Within trading days, they differ in per-stock sizing/selection:

### Variant 1: Binary Gate + Raw Scores
```python
if G_t >= 0.2:
    # Standard portfolio: sort by raw score, top-10 long, bottom-10 short, equal weight
    portfolio = equal_weight_top_bottom_10(scores)
else:
    portfolio = flat()  # No exposure
```
**Baseline.** Tests whether the gate alone adds value.

### Variant 2: Binary Gate + Vol-Sizing
```python
if G_t >= 0.2:
    sized_score = score * min(1.0, c_vol / sqrt(vol_20d + eps))
    portfolio = equal_weight_top_bottom_10(sized_score)
else:
    portfolio = flat()
```
**This is the Ch12 heuristic + binary gate.** The benchmark to beat.
- `c_vol` calibrated on DEV: set so median weight ≈ 0.7
- `eps = 1e-8`

### Variant 3: Binary Gate + Uncertainty-Adjusted Sorting (Liu et al. 2026)

**THIS IS THE PRIMARY CANDIDATE — DIRECTLY FROM LITERATURE**

**Paper:** Liu, Luo, Wang, Zhang. "Uncertainty-Adjusted Sorting for Asset Pricing
with Machine Learning." arXiv:2601.00593, January 2026.

**Core idea:** Instead of sorting by point predictions alone, sort by uncertainty-adjusted
bounds. For longs, use the upper bound (optimistic case); for shorts, use the lower bound
(pessimistic case).

```python
if G_t >= 0.2:
    # For long leg: sort by upper bound = score + lambda * uncertainty_width
    # For short leg: sort by lower bound = score - lambda * uncertainty_width
    upper_bound = score + lambda_ua * q_hat  # q_hat = uncertainty width
    lower_bound = score - lambda_ua * q_hat
    
    long_candidates = top_10_by(upper_bound)   # Stocks with best optimistic case
    short_candidates = bottom_10_by(lower_bound)  # Stocks with worst pessimistic case
    
    portfolio = equal_weight(long_candidates, short_candidates)
else:
    portfolio = flat()
```

**Key details from the paper:**
- `q_hat_i,t(alpha)` = estimate of the α-quantile of absolute residuals in a rolling window
- Prediction interval: `[score - q_hat, score + q_hat]`
- Longs sorted by upper bound `score + q_hat`; shorts sorted by lower bound `score - q_hat`
- Tested at alpha = 1%, 5%, 10% (increasingly aggressive uncertainty adjustment)
- **Results:** Sharpe ratio improvements across most ML models. NN1 Sharpe: 1.48 → 1.86.
  PCR Sharpe: 1.22 → 1.56. Gains mainly from reduced volatility.
- **Critical finding:** "gains persist even when bounds are built from partial or
  misspecified uncertainty information" — even residual variance alone helps
- **Critical finding:** "improvements are driven by asset-level rather than time or
  aggregate predictive uncertainty" — per-stock uncertainty is what matters

**How to adapt for our system:**
- Our `q_hat` can be: (a) g(x) raw error prediction, (b) ê(x) epistemic signal,
  (c) conformal interval half-width, or (d) rolling residual quantile
- Start with g(x) since it's the most informative per-stock signal (ρ=0.192 with rank_loss)
- `lambda_ua` is calibrated on DEV to maximize FINAL Sharpe
- Test lambda_ua ∈ {0.1, 0.3, 0.5, 1.0, 2.0} on DEV, pick best, freeze for FINAL

**Why this should work for us:**
- It's ADDITIVE, not multiplicative. A stock with score=0.05 and high g(x) loses some
  rank but doesn't get killed. Preserves strong signals.
- It uses uncertainty to RE-SORT, not to scale weights. Different stocks enter/exit the
  top-10/bottom-10 based on uncertainty, but within the portfolio it's still equal weight.
- The paper validates on U.S. equity cross-section with ML models — very close to our setup.

### Variant 4: Binary Gate + Residualized-ê Sizing

**Literature:** Hentschel (2025) "Contextual Alpha"; Barroso & Saxena (2021)
"Lest We Forget" (Review of Financial Studies).

**Core idea:** The problem with raw ê for sizing is corr(ê, |score|) > 0. Residualize
ê on score magnitude to get "excess uncertainty" — uncertainty BEYOND what's expected
for a stock's score level.

```python
if G_t >= 0.2:
    # Step 1: Residualize ê on score magnitude (cross-sectional, per date)
    # Regress: ê = beta_0 + beta_1 * |score_rank| + epsilon
    # ê_resid = epsilon (residual)
    for each date t:
        X = |cross_sectional_rank|  # or |score|
        y = ehat  # or g(x) at 20d/90d
        beta = OLS(X, y)
        ehat_resid = y - beta @ X  # Residuals
    
    # Step 2: Size using residualized uncertainty
    # ê_resid > 0: MORE uncertain than expected → penalize
    # ê_resid < 0: LESS uncertain than expected → don't penalize
    sized_score = score * min(1.0, c_resid / sqrt(max(ehat_resid, 0) + eps))
    
    portfolio = equal_weight_top_bottom_10(sized_score)
else:
    portfolio = flat()
```

**Why this should work:**
- Kills the structural conflict: a top-ranked stock with *typical* ê for its rank
  gets ê_resid ≈ 0 (no penalty). A top-ranked stock with *unusually high* ê gets
  ê_resid > 0 (penalized — something specific to THIS stock is wrong).
- Preserves strong signals while catching genuine anomalies.

**Calibration:** `c_resid` on DEV, median weight ≈ 0.7.

### Variant 5: Binary Gate + ê-Cap at P90

**Literature:** Selective prediction / hybrid abstention (Chaudhuri & Lopez-Paz 2023;
Rabanser 2025). "Apply a reject threshold for extremely uncertain cases and apply
continuous sizing within the accepted set."

**Core idea:** Don't use ê for continuous sizing. Just cap the most uncertain stocks.

```python
if G_t >= 0.2:
    # Compute ê percentile cross-sectionally per date
    ehat_pct = percentile_rank(ehat, per_date=True)
    
    # Cap top-decile uncertainty stocks
    weight_multiplier = np.where(ehat_pct > 0.90, 0.7, 1.0)
    sized_score = score * weight_multiplier
    
    portfolio = equal_weight_top_bottom_10(sized_score)
else:
    portfolio = flat()
```

**Why this might work:**
- Prevents catastrophic single-name blowups from the most unpredictable stocks
- Doesn't touch the other 90% — minimal interference with signal
- Simple, robust, no calibration beyond the cap level

**Variants to test:** cap_threshold ∈ {P80, P85, P90, P95}, cap_weight ∈ {0.5, 0.7}

### Variant 6: Binary Gate + Vol-Sizing + ê-Cap

**Combination of the Ch12 best method + DEUP tail-risk guard.**

```python
if G_t >= 0.2:
    # Vol-size first
    vol_sized = score * min(1.0, c_vol / sqrt(vol_20d + eps))
    
    # Then cap high-ê stocks
    ehat_pct = percentile_rank(ehat, per_date=True)
    weight_multiplier = np.where(ehat_pct > 0.90, 0.7, 1.0)
    final_score = vol_sized * weight_multiplier
    
    portfolio = equal_weight_top_bottom_10(final_score)
else:
    portfolio = flat()
```

**Why this is important:** If this beats pure vol-sizing, it proves DEUP adds
incremental value ON TOP of the existing best method.

---

## 3. RESEARCH LITERATURE SYNTHESIS

### 3.1 Liu et al. (2026) — The Primary Reference

**Paper:** "Uncertainty-Adjusted Sorting for Asset Pricing with Machine Learning"
arXiv:2601.00593

**Setup:** U.S. equity panel, monthly rebalanced, 10 ML models (OLS, Ridge, Lasso,
PCR, PLS, Random Forest, GBT, NN1, NN2, NN3), decile-sorted long-short portfolios.

**Method:**
- Point prediction: μ̂_i,t = ML model forecast for stock i at time t
- Uncertainty width: q̂_i,t(α) = estimate of α-quantile of |residuals| in rolling window
- Upper bound: μ̂_i,t + q̂_i,t(α) — used to sort LONGS
- Lower bound: μ̂_i,t − q̂_i,t(α) — used to sort SHORTS
- α ∈ {1%, 5%, 10%} — smaller α = less aggressive uncertainty adjustment

**Results:**
- Sharpe improvements for most models, especially flexible ones (neural nets, GBT)
- NN1 Sharpe: 1.48 → 1.86 (at 5% quantile level)
- PCR Sharpe: 1.22 → 1.56
- Gains mainly from REDUCED VOLATILITY, not higher returns
- Gains persist with partial/misspecified uncertainty (even just residual variance helps)
- Asset-level uncertainty drives gains, NOT time-level or aggregate uncertainty

**Critical insight for us:** The paper uses rolling residual quantiles as uncertainty.
We have something BETTER: g(x), a dedicated error predictor trained walk-forward with
ρ = 0.192 correlation to actual rank displacement. Our uncertainty estimate is more
informative than their rolling residual quantile.

**Adaptation for our system:**
- Their q̂_i,t(α) → our g(x) or ê(x) or conformal half-width
- Their μ̂_i,t → our LGB score
- Their decile sort → our top-10/bottom-10 selection
- Their monthly rebalance → same (our 20d horizon ≈ monthly)
- Key difference: they use equal-weight within deciles; we do too

### 3.2 Hentschel (2025) — Contextual Alpha / Residualization

**Paper:** "Contextual Alpha: Emphasizing Forecasts Where They Work Best"

**Key technique:** Residualize uncertainty on signal strength before using it for sizing.
```
u_resid = u - beta_0 - beta_1 * |signal|
```
Then use u_resid for sizing: `w_i ∝ s_i / (σ_i * (1 + λ * u_resid_i))`

**Relevance:** Directly addresses our corr(ê, |score|) > 0 problem. After residualization,
uncertainty penalizes only stocks that are MORE uncertain than expected for their signal level.

### 3.3 Barroso & Saxena (2021) — Learn from Forecast Errors

**Paper:** "Lest We Forget: Learn from Out-of-Sample Forecast Errors"
Review of Financial Studies.

**Key finding:** Learning from and calibrating forecast errors improves optimized
portfolio outcomes out-of-sample. Signal-correlated error correction raises realized
performance.

**Relevance:** Validates the principle of using walk-forward residuals (which is exactly
what our g(x) error predictor does) to improve portfolio construction.

### 3.4 Spears, Zohren & Roberts (2020/2021) — Deep Bayesian Investment Sizing

**Paper:** "Investment Sizing with Deep Learning Prediction Uncertainties for
High-Frequency Eurodollar Futures Trading" (JFDS 2021)

**Key technique:** Scale position size by `w = c / (eps + sigma_pred)` where sigma_pred
is posterior predictive standard deviation from Bayesian deep learning.

**Key finding:** Gating trades when variance exceeds threshold reduces unnecessary exposure.
Scaling by uncertainty improves Sharpe ratios empirically.

**Relevance:** This is the `w = c/sqrt(unc + eps)` template our 13.6 DEUP sizing uses.
The paper validates the approach for futures but uses return-level uncertainty. Our
problem is that rank-displacement uncertainty correlates with signal strength, which
their return-level setup doesn't face.

### 3.5 Garlappi, Uppal & Wang (2007) — Multi-Prior Ambiguity Aversion

**Paper:** "Portfolio Selection with Parameter and Model Uncertainty: A Multi-Prior
Approach" (Review of Financial Studies)

**Key result:** Multi-prior minimax yields closed-form portfolios equivalent to shrinkage
between mean-variance and minimum-variance. More stable weights, better OOS Sharpe.

**Relevance:** Theoretical backing for uncertainty-based portfolio shrinkage. Our binary
gate + vol-sizing is operationally a simplified version of their ambiguity-averse portfolio.

### 3.6 Selective Prediction Literature

**Key papers:** Chaudhuri & Lopez-Paz (2023), Franc et al. (2021), Rabanser (2025)

**Core framework:**
- Selective risk = E[loss | accepted]. Coverage = P(accept).
- Chow-style rule: reject when expected prediction loss > rejection cost
- Hybrid recommendation: "Apply reject threshold for extremely uncertain cases AND
  continuous sizing within the accepted set"

**Relevance:** Formalizes our binary G(t) gate as a Chow-style reject option. The
ê-cap (Variant 5) is a secondary selective prediction layer within the accepted set.
Our system already implements the hybrid recommendation.

### 3.7 Regime-Switching Portfolio Allocation

**Key papers:** Oprisor & Kwon (2020), Pun et al. (2023), Nystrup et al. (2017)

**Key techniques:**
- Regime-conditioned optimization: solve allocation under regime-specific μ, Σ
- Exposure multiplier α_s by regime (smaller in crisis)
- Position-level stop logic conditional on regime
- Regime-dependent CVaR constraints

**Relevance:** Our binary G(t) gate is a special case of regime-dependent exposure
multiplier (α_s ∈ {0, 1}). The literature supports both binary and continuous approaches;
our 13.6 finding that continuous throttling destroys recoveries is consistent with
Nystrup et al.'s finding that too-frequent switching hurts.

### 3.8 Prediction Extremity and Error Magnitude

**Finding from literature review:** There is NO universal U-shaped relationship between
prediction extremity and error magnitude in ranking models. Our finding that extreme
cross-sectional ranks have higher ê is an empirical contribution specific to our
cross-sectional equity ranking system.

**Implication:** The structural conflict (high score → high ê → inverse sizing penalizes
best ideas) is a property of our specific system, not a known universal law. This
strengthens the novelty of finding solutions (Liu et al. sorting, residualization, capping)
that address it.

### 3.9 Conformal Prediction for Portfolios

**Key papers:** Alonso (2025), Kato (2024), Yeh et al. (2024)

**Key techniques:**
- Conformal intervals as uncertainty inputs to portfolio optimization
- End-to-end conformal calibration (CRO) that trains uncertainty jointly with optimizer
- Conformal sets as constraint/scenario sets in robust optimization

**Relevance:** Supports using conformal interval width as a sizing input (our Variant
using conformal width), though the literature primarily uses intervals as constraints
rather than direct sizing signals. Our DEUP-normalized conformal width is a novel
uncertainty input that none of these papers test.

---

## 4. EXISTING CODE AND DATA

### Available Data Files

```
evaluation_outputs/chapter13/
├── residual_archive.parquet        # 591K rows: date, ticker, score, rank_loss, features
├── ehat_predictions.parquet        # g(x) per stock per date (walk-forward)
├── aleatoric_baseline.parquet      # a(x) per date/stock
├── epistemic_signal.parquet        # ê(x) = max(0, g-a)
├── expert_health.parquet           # H(t), G(t) per date
├── conformal_predictions.parquet   # 495K rows: intervals for 3 variants
├── chapter13_6_portfolio_metrics.json  # Existing sizing variant results
├── chapter13_6_regime_eval.json    # AUROC, confusion matrix, bucket tables
└── chapter13_6_daily_timeseries.parquet # Date-level returns, G(t)
```

### Available Source Modules

```
src/uncertainty/
├── deup_estimator.py       # g(x) error predictor
├── aleatoric_baseline.py   # a(x) computation
├── epistemic_signal.py     # ê(x) = max(0, g-a)
├── expert_health.py        # H(t), G(t) computation
├── conformal_intervals.py  # Conformal prediction pipeline
└── deup_portfolio.py       # 13.6 portfolio sizing (EXTEND THIS)
```

### Key Function Signatures (from deup_portfolio.py)

The 13.6 module already has sizing infrastructure. 13.7 extends it with new variants.

```python
# Existing sizing from 13.6:
def compute_sized_score(score, vol_20d, ehat, c_vol, c_deup, variant, eps=1e-8):
    """Compute sized score for one of the existing variants (A/B/C/D)"""
    ...

def build_shadow_portfolio(sized_scores, top_k=10, cost_bps=10):
    """Build equal-weight top-K long / bottom-K short portfolio"""
    ...

def evaluate_regime_trust(daily_returns, G_t, threshold=0.2):
    """AUROC, confusion matrix, bucket analysis for regime gate"""
    ...
```

### Key Data Columns in residual_archive.parquet

```
date, ticker, fold_id, score, excess_return_20d, rank_loss_20d,
vol_20d, vix, cross_sectional_rank, mom_1m, mom_3m, mom_12m,
adv_20d, market_cap, sector, ...
```

### Key Data Columns in ehat_predictions.parquet

```
date, ticker, horizon, g_x, a_x, ehat, ehat_percentile
```

### Key Data Columns in expert_health.parquet

```
date, horizon, H_realized, H_drift, H_disagree, H_combined, G_t
```

---

## 5. IMPLEMENTATION SPECIFICATION

### 5.1 New File: `src/uncertainty/deployment_policy.py`

This is the main implementation file for 13.7.

```python
"""
Chapter 13.7: Deployment Policy Variants

Six sizing/selection variants, all using binary G(t) gate.
Calibration on DEV only; FINAL evaluated once.
"""

import numpy as np
import pandas as pd
from typing import Literal

GATE_THRESHOLD = 0.2
EPS = 1e-8

PolicyVariant = Literal[
    "gate_raw",           # Variant 1
    "gate_vol",           # Variant 2
    "gate_ua_sort",       # Variant 3 (Liu et al.)
    "gate_resid_ehat",    # Variant 4
    "gate_ehat_cap",      # Variant 5
    "gate_vol_ehat_cap",  # Variant 6
]


def apply_binary_gate(G_t: float, threshold: float = GATE_THRESHOLD) -> bool:
    """Binary abstention gate. Returns True if model is trustworthy."""
    return G_t >= threshold


def uncertainty_adjusted_sort(
    scores: pd.Series,
    uncertainty: pd.Series,  # g(x) or ê(x) or conformal width
    lambda_ua: float,
) -> tuple[pd.Index, pd.Index]:
    """
    Liu et al. (2026) uncertainty-adjusted sorting.
    
    Longs: sort by upper bound = score + lambda * uncertainty
    Shorts: sort by lower bound = score - lambda * uncertainty
    
    Returns (long_tickers, short_tickers) each of length top_k.
    """
    upper = scores + lambda_ua * uncertainty
    lower = scores - lambda_ua * uncertainty
    
    long_tickers = upper.nlargest(10).index
    short_tickers = lower.nsmallest(10).index
    
    return long_tickers, short_tickers


def residualize_ehat(
    ehat: pd.Series,
    score_rank: pd.Series,
) -> pd.Series:
    """
    Residualize ê on cross-sectional score magnitude.
    
    Regress ê = beta_0 + beta_1 * |score_rank| + epsilon
    Return epsilon (residual uncertainty).
    
    ê_resid > 0: MORE uncertain than expected for this score level
    ê_resid ≈ 0: typical uncertainty for this score level
    ê_resid < 0: LESS uncertain than expected
    """
    X = np.abs(score_rank.values).reshape(-1, 1)
    X = np.column_stack([np.ones(len(X)), X])  # Add intercept
    y = ehat.values
    
    # OLS
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    residuals = y - X @ beta
    
    return pd.Series(residuals, index=ehat.index)


def apply_ehat_cap(
    scores: pd.Series,
    ehat: pd.Series,
    cap_percentile: float = 0.90,
    cap_weight: float = 0.70,
) -> pd.Series:
    """
    Cap scores for stocks with ê above the percentile threshold.
    
    Stocks in top (1-cap_percentile) of ê get multiplied by cap_weight.
    All others untouched.
    """
    threshold = ehat.quantile(cap_percentile)
    multiplier = np.where(ehat > threshold, cap_weight, 1.0)
    return scores * multiplier
```

### 5.2 Calibration Protocol

**ALL calibration on DEV only. FINAL is evaluated ONCE.**

```python
# Hyperparameters to calibrate on DEV:

# Variant 2: c_vol (already calibrated in 13.6)
# Set so median vol-sized weight ≈ 0.7

# Variant 3: lambda_ua
# Grid search on DEV: lambda_ua ∈ [0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0]
# Select lambda_ua that maximizes DEV Sharpe (or minimizes DEV volatility)
# Use g(x) as uncertainty input (best per-stock predictor)

# Variant 4: c_resid
# Set so median sized weight ≈ 0.7 (same approach as c_vol)

# Variant 5: cap_percentile, cap_weight
# Test: cap_percentile ∈ {0.80, 0.85, 0.90, 0.95}
# Test: cap_weight ∈ {0.5, 0.7}
# Select combo that maximizes DEV Sharpe

# Variant 6: c_vol (from Variant 2) + cap params (from Variant 5)
```

### 5.3 Evaluation Metrics

For each variant, compute on ALL / DEV / FINAL:

```python
metrics = {
    "sharpe": annualized_sharpe(monthly_returns),
    "sortino": annualized_sortino(monthly_returns),
    "ann_return": annualized_return(monthly_returns),
    "ann_vol": annualized_volatility(monthly_returns),
    "max_dd": max_drawdown(cumulative_returns),
    "hit_rate": fraction_positive_months,
    "calmar": ann_return / abs(max_dd),
    "avg_turnover": mean_monthly_turnover,
    "crisis_sharpe": sharpe(returns[mar_2024:jul_2024]),
    "crisis_max_dd": max_dd(returns[mar_2024:jul_2024]),
    "n_trading_months": months_where_G >= 0.2,
    "abstention_rate": months_where_G < 0.2 / total_months,
}
```

### 5.4 Test Specification

```python
# tests/test_deployment_policy.py

# 1. Binary gate tests
def test_gate_threshold_exact():
    """G=0.2 exactly → trade"""
def test_gate_below_threshold():
    """G=0.19 → flat"""
def test_gate_above_threshold():
    """G=0.5 → trade"""

# 2. Liu et al. sorting tests
def test_ua_sort_basic():
    """Uncertainty changes which stocks enter top-10"""
def test_ua_sort_lambda_zero():
    """lambda=0 → identical to raw sort"""
def test_ua_sort_preserves_strong_signals():
    """High-score + high-uncertainty stock can still make top-10"""

# 3. Residualization tests
def test_residualize_removes_score_correlation():
    """corr(ehat_resid, |score|) ≈ 0 after residualization"""
def test_residualize_preserves_mean():
    """mean(ehat_resid) ≈ 0"""

# 4. Cap tests
def test_cap_only_affects_top_percentile():
    """90% of stocks unchanged, 10% reduced"""
def test_cap_weight_applied_correctly():
    """Capped stock score = original * 0.7"""

# 5. Integration tests
def test_all_variants_produce_valid_portfolios():
    """Each variant: 10 longs, 10 shorts, weights sum correctly"""
def test_dev_final_separation():
    """No DEV data in FINAL evaluation"""
def test_calibration_uses_dev_only():
    """Lambda, c values from DEV grid search"""

# 6. Economic tests
def test_gate_reduces_crisis_drawdown():
    """Any gated variant has lower crisis MaxDD than ungated"""
def test_variants_match_13_6_baselines():
    """Variant 1 (gate+raw) matches 13.6 raw + binary gate logic"""

# Target: ~20 tests
```

### 5.5 Runner Script

```python
# scripts/run_chapter13_deup.py --step 8  (or new file)

# 1. Load all data
# 2. Calibrate hyperparameters on DEV
# 3. Run all 6 variants on ALL/DEV/FINAL
# 4. Compute metrics
# 5. Save results to evaluation_outputs/chapter13/chapter13_7_policy_results.json
# 6. Print comparison table
```

---

## 6. SUCCESS CRITERIA FOR 13.7

### Primary Success Criteria

| Criterion | Target | How to Evaluate |
|-----------|--------|-----------------|
| **Combined system beats vol-only on FINAL** | FINAL Sharpe > 1.68 | At least one of Variants 3–6 |
| **Gate + any sizing > raw ungated** | ALL Sharpe improvement | All gated variants vs raw baseline |
| **Crisis MaxDD reduced** | < −44.1% (raw) | Any gated variant |
| **ê adds incremental value** | Gate+vol+ê > gate+vol | Variant 6 vs Variant 2 |

### Kill Criterion K4 (From Original Outline)

Test whether trailing RankIC sizing works as an alternative:
```python
trailing_ic_sizing = 1 - trailing_60d_RankIC
# If ê-sized ≤ vol AND ≤ trailing_ic: DEUP per-stock sizing has no use case
```

If K4 triggers (ê can't beat vol OR trailing IC), the conclusion is:
> "DEUP's value is architectural (regime gate + calibrated intervals), not per-stock
> inverse-sizing. The combined system (gate + vol) is the deployment recommendation."

This is an HONEST finding, not a failure. Document it clearly.

### Reframed Success Criteria (Updated from Original Outline)

| Original | Updated |
|----------|---------|
| "ê spikes in 2024" | "System detects 2024 crisis (H(t)/G(t) → 0)" — ALREADY PASSED in 13.4b |
| "ê-sized > ALL baselines" | "Combined system (gate + best sizing) > vol-only on FINAL" |
| "Low-ê RankIC > full-set" | "ê quintile monotonicity ρ = 1.0 for rank_loss" — ALREADY PASSED in 13.3 |
| "AUROC (ê predicts bad days) > 0.60" | "Regime AUROC > 0.65" — ALREADY PASSED at 0.72 |

---

## 7. DIAGNOSTIC: WHY ê CORRELATES WITH |SCORE|

Before implementing Variants 3–6, run this diagnostic to formally document the
structural conflict. Save to `evaluation_outputs/chapter13/ehat_score_correlation.json`.

```python
# Compute per-date cross-sectional statistics
for each date in DEV:
    corr_ehat_abs_score = spearman(ehat, |score|)
    corr_g_abs_score = spearman(g_x, |score|)
    
    # By ê decile:
    for decile in 1..10:
        mean_abs_score[decile] = mean(|score| where ehat in decile)
        mean_rank_loss[decile] = mean(rank_loss where ehat in decile)
        mean_ic_contribution[decile] = ...  # How much does this decile contribute to IC?

# Expected finding:
# - corr(ê, |score|) > 0 (positive, confirming structural conflict)
# - Top ê decile has highest |score| AND highest rank_loss
# - Top ê decile contributes disproportionately to P&L (both gains AND losses)
# - This formally proves: inverse sizing clips the stocks that drive both
#   the best and worst months, reducing portfolio convexity
```

---

## 8. HEADLINE RESULTS TABLE (TO FILL IN)

After running all variants:

```
╔══════════════════════════════════════════════════════════════════════╗
║  Chapter 13.7 — Deployment Policy Comparison (20d Primary Horizon) ║
╠═════════════════╦═════════╦═════════╦═══════════╦════════╦═════════╣
║ Variant         ║ ALL     ║ DEV     ║ FINAL     ║ Crisis ║ Crisis  ║
║                 ║ Sharpe  ║ Sharpe  ║ Sharpe    ║ Sharpe ║ MaxDD   ║
╠═════════════════╬═════════╬═════════╬═══════════╬════════╬═════════╣
║ Ungated raw     ║ 1.14    ║ 1.16    ║ 1.37      ║ -0.42  ║ -44.1%  ║
║ Ungated vol     ║ 1.18    ║ 1.17    ║ 1.68      ║ -0.75  ║ -47.3%  ║
╠═════════════════╬═════════╬═════════╬═══════════╬════════╬═════════╣
║ 1. Gate+Raw     ║         ║         ║           ║        ║         ║
║ 2. Gate+Vol     ║         ║         ║           ║        ║         ║
║ 3. Gate+UA Sort ║         ║         ║           ║        ║         ║
║ 4. Gate+Resid-ê ║         ║         ║           ║        ║         ║
║ 5. Gate+ê-Cap   ║         ║         ║           ║        ║         ║
║ 6. Gate+Vol+Cap ║         ║         ║           ║        ║         ║
╚═════════════════╩═════════╩═════════╩═══════════╩════════╩═════════╝
```

---

## 9. DOCUMENTATION TEMPLATE FOR CHAPTER_13.MD

After results are in, add this section to CHAPTER_13.md:

```markdown
## 13.7 Deployment Policy & Sizing Ablation

### Motivation

Section 13.6 established that:
1. **G(t) binary gate works** (AUROC 0.72, precision 80%)
2. **Vol-sizing beats DEUP inverse-sizing** (FINAL Sharpe 1.68 vs 1.35)
3. **The structural conflict:** g(x) correlates with |score|, so inverse-sizing
   penalizes the model's strongest signals

Section 13.7 tests whether alternative applications of ê(x) — that avoid the
inverse-sizing conflict — can add value ON TOP of vol-sizing.

### Literature Basis

[Liu et al. (2026)] proposed uncertainty-adjusted sorting: sort longs by
score + λ·uncertainty and shorts by score − λ·uncertainty. This is additive
(preserves strong signals) rather than multiplicative (penalizes them).
Tested on U.S. equities with ML models; Sharpe improvements of 0.25–0.38
across most models, mainly from reduced volatility.

[Hentschel (2025)] proposed residualizing uncertainty on signal magnitude
before using it for sizing, removing the structural correlation between
prediction extremity and error magnitude.

### Results

[FILL IN AFTER RUNNING]

### Deployment Recommendation

[FILL IN — expected to be one of:]
- "Gate(G≥0.2) + Vol-sizing + ê-cap at P90" if Variant 6 wins
- "Gate(G≥0.2) + Uncertainty-adjusted sorting" if Variant 3 wins
- "Gate(G≥0.2) + Vol-sizing" if nothing beats Variant 2 (honest finding)
```

---

## 10. CITATIONS

```
# Core — must cite in 13.7
Liu, Y., Luo, Y., Wang, Z., & Zhang, X. (2026). Uncertainty-Adjusted Sorting
  for Asset Pricing with Machine Learning. arXiv:2601.00593.

# Supporting — cite where relevant
Hentschel, L. (2025). Contextual Alpha: Emphasizing Forecasts Where They Work Best.

Barroso, P. & Saxena, K. (2021). Lest We Forget: Learn from Out-of-Sample Forecast
  Errors When Optimizing Portfolios. Review of Financial Studies.

Spears, T., Zohren, S., & Roberts, S. J. (2021). Investment Sizing with Deep Learning
  Prediction Uncertainties for HF Eurodollar Futures Trading. JFDS 3(1).

Garlappi, L., Uppal, R., & Wang, T. (2007). Portfolio Selection with Parameter and
  Model Uncertainty: A Multi-Prior Approach. Review of Financial Studies 20(1).

Chaudhuri, K. & Lopez-Paz, D. (2023). Unified Uncertainty Calibration. arXiv:2310.01202.

Rabanser, S. (2025). Uncertainty-Driven Reliability: Selective Prediction and
  Trustworthy Deployment in Modern ML. arXiv:2508.07556.

# Already cited in 13.0–13.6
Lahlou, S. et al. (2023). DEUP: Direct Epistemic Uncertainty Prediction. TMLR.
Kotelevskii, N. & Panov, M. (2025). From Risk to Uncertainty. ICLR.
Plassier, V. et al. (2025). Conformal prediction with DEUP. ICLR.
```

---

## 11. IMPLEMENTATION ORDER

1. **Run the ê-score correlation diagnostic** (Section 7). Document the structural conflict.
2. **Implement `deployment_policy.py`** with all 6 variants.
3. **Write tests** (~20 tests in `test_deployment_policy.py`).
4. **Calibrate on DEV** (grid search for lambda_ua, c_resid, cap params).
5. **Run all variants** on ALL/DEV/FINAL. Fill in the results table.
6. **Clear Kill Criterion K4** (trailing RankIC comparison).
7. **Update CHAPTER_13.md** with 13.7 section.
8. **Update RESULTS.md** headline table.
9. **Update README.md** project status (Ch13: 85% → 95%).
10. **Run full test suite** (`pytest -q`). Verify all 154 + new tests pass.

**Estimated time:** 4–6 hours.

---

## 12. WHAT HAPPENS AFTER 13.7

| Step | Task | Time |
|------|------|------|
| 13.8 | Skip DEUP on Rank Avg 2 (not needed — 13.6 already shows regime trust generalizes) | — |
| 13.9 | Freeze: wire real ê(x) into AIStockForecasterExpert interface, git tag | 2 hours |
| Ch16 | Factor attribution: Fama-French 5-factor regression on shadow portfolio | 3–4 hours |
| Ch18 | Live deployment pipeline (separate repo, ntfy notifications) | 1–2 days |

**After 13.7, the research project is effectively complete.** Ch16 is a quick
validation step. Ch14/15/17 are optional extensions.

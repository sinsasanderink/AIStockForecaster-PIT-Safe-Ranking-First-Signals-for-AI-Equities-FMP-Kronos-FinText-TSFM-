# Kronos Root Cause Analysis: The Normalization Issue

**Date:** January 9, 2026  
**Status:** ✅ **ROOT CAUSE IDENTIFIED**  
**Issue:** Mean-reversion bias caused by de-normalization, NOT a pipeline bug

---

## Executive Summary

We found the **root cause** of Kronos's systematic negative predictions:

**Kronos normalizes input using the 252-day historical mean, then de-normalizes output using the same mean.** For stocks that have trended UP, predictions regress toward the historical mean, creating systematic negative bias.

**This is NOT a pipeline bug - it's how Kronos is designed to work.**

---

## The Evidence

### Prediction vs Historical Mean

| Ticker | 252d Mean | Spot Close | Pred Close | Mean/Spot | Pred/Mean |
|--------|-----------|------------|------------|-----------|-----------|
| **NVDA** | **$39.85** | **$63.03** | **$46.62** | **63%** | **117%** |
| **AMD** | **$110.54** | **$170.48** | **$124.40** | **65%** | **113%** |
| **META** | **$281.18** | **$394.78** | **$319.08** | **71%** | **113%** |
| AAPL | $176.79 | $186.86 | $180.89 | 95% | 102% |
| INTC | $35.34 | $43.36 | $39.38 | 81% | 111% |
| CSCO | $51.01 | $50.18 | $52.71 | 102% | 103% |

**Key Insight:** Predictions are ~100-117% of the **historical mean**, not the spot price!

---

## How Kronos Normalization Works

From `Kronos/model/kronos.py`, lines 544-556:

```python
# NORMALIZE input using historical mean/std
x_mean, x_std = np.mean(x, axis=0), np.std(x, axis=0)
x = (x - x_mean) / (x_std + 1e-5)
x = np.clip(x, -5, 5)

# ... model predicts in NORMALIZED space ...

# DE-NORMALIZE output back to price space
preds = preds * (x_std + 1e-5) + x_mean  # <-- Uses SAME mean!
```

**The Problem:**

1. For NVDA (trending UP strongly):
   - 252-day mean close = $39.85
   - Current spot close = $63.03
   - If Kronos predicts "slight increase in normalized space" (+10%)
   - De-normalized: $39.85 × 1.10 = $43.84
   - Score = ($43.84 - $63.03) / $63.03 = **-30.5%** (appears bearish!)

2. For CSCO (flat):
   - 252-day mean close = $51.01
   - Current spot close = $50.18
   - If Kronos predicts same +10% in normalized space
   - De-normalized: $51.01 × 1.10 = $56.11
   - Score = ($56.11 - $50.18) / $50.18 = **+11.8%** (appears bullish!)

**Same model behavior, completely different scores!**

---

## Why This Happens

```
NVDA Price History (252 days before Feb 1, 2024):
  Feb 2023: ~$20
  May 2023: ~$30 (AI rally starts)
  Aug 2023: ~$45
  Nov 2023: ~$50
  Feb 2024: ~$63 (current)
  
  Mean of these: ~$40 (well below current!)
```

Kronos's internal logic:
- "This stock is now at +1.5 std above its historical mean"
- "I predict it will be at +0.5 std in 20 days" (slight mean reversion)
- De-normalized: ~$46 (which is BELOW current $63)

---

## Is This a Bug?

**NO.** This is intentional design:

1. **Kronos was trained on normalized data** - it MUST normalize
2. The normalization is per-sequence, using that sequence's statistics
3. This naturally creates mean-reversion predictions for trending stocks
4. It's appropriate for **short-term intraday** (their main use case: 5-min bars)
5. It's problematic for **cross-sectional daily ranking** (our use case)

---

## Why Rankings Are Still Inverted

Even if normalization creates absolute bias, shouldn't **rankings** be preserved?

**Unfortunately, no.** The bias is **ticker-dependent**:

| Ticker | Trend Strength | Bias Magnitude |
|--------|---------------|----------------|
| NVDA | Very strong up | Very strong negative |
| AMD | Strong up | Strong negative |
| CSCO | Flat | Near zero |
| INTC | Weak up | Moderate negative |

Stocks with **stronger trends** get **more negative bias**. But in Feb 2024, the strongest trending stocks (NVDA, AMD, META) were also the **best performers!**

So the ranking gets inverted:
- Kronos says: "NVDA most negative, CSCO most positive"
- Reality was: "NVDA best, CSCO worst"
- RankIC = **negative**

---

## Potential Fixes (None are great)

### Option A: Shorter Normalization Window ❓

```python
# Use 20-day mean instead of 252-day
x_mean = np.mean(x[-20:], axis=0)
x_std = np.std(x[-20:], axis=0)
```

**Pros:** Less bias from long-term trends  
**Cons:** Model was trained with full-window normalization; may break

### Option B: Different Score Calculation ❓

```python
# Score relative to normalized prediction, not de-normalized
# But this changes the interpretation entirely
```

**Cons:** Unclear what the "right" interpretation would be

### Option C: Cross-Sectional Z-Scoring ❓

```python
# Z-score Kronos predictions cross-sectionally
scores_zscore = (scores - scores.mean()) / scores.std()
```

**Pros:** Removes absolute bias  
**Cons:** Still won't fix inverted rankings

### Option D: Accept the Limitation ✅

Kronos is designed for:
- **Intraday** predictions (5-min bars)
- **Single-ticker** forecasting
- **Short horizons** where normalization doesn't create as much bias

Our use case (daily cross-sectional ranking) doesn't match.

---

## What Cross-Sectional Ranking Would Require

For cross-sectional ranking to work, we'd need predictions that:
1. Are comparable across tickers
2. Don't have ticker-dependent bias
3. Capture **relative** future performance

Kronos's normalization makes predictions **non-comparable** across tickers because each ticker's bias depends on its own historical trend.

---

## Conclusion

| Question | Answer |
|----------|--------|
| Is there a pipeline bug? | **NO** ✅ |
| Is the score calculation correct? | **YES** ✅ |
| Why are predictions negative? | **Normalization bias** |
| Can we fix it? | **Not easily** |
| Is Kronos useful for our task? | **No** ❌ |

**Root Cause:** Kronos's per-sequence normalization creates ticker-dependent bias that inverts rankings for trending stocks.

**Not a bug in our code - a fundamental mismatch between Kronos's design and our use case.**

---

## Recommendations

### 1. Document the Finding ✅

This is valuable research! We now understand:
- Why Kronos predictions look bearish
- Why rankings are inverted
- How the normalization affects cross-sectional use

### 2. Proceed to Chapter 9 (FinText)

Text-based signals don't have this normalization issue and may be more suitable for cross-sectional ranking.

### 3. Consider Alternative Time Series Models

If we want a price-based foundation model for cross-sectional ranking, look for models that:
- Use returns instead of normalized prices
- Have cross-sectional training
- Don't rely on per-sequence normalization

### 4. Possible Ensemble Use

Despite poor standalone performance, Kronos captures **something** (mean-reversion signal). In regimes where mean-reversion IS the dominant pattern (bear markets, high-VIX), it might help in an ensemble.

---

## Appendix: Kronos Normalization Code

From `Kronos/model/kronos.py`:

```python
# predict() method, lines 544-556
x_mean, x_std = np.mean(x, axis=0), np.std(x, axis=0)  # Per-feature mean/std over full history

x = (x - x_mean) / (x_std + 1e-5)  # Normalize to ~N(0,1)
x = np.clip(x, -self.clip, self.clip)  # Clip to [-5, 5]

# ... model prediction in normalized space ...

preds = preds * (x_std + 1e-5) + x_mean  # De-normalize using SAME statistics
```

**This is the source of the bias.** The model is working correctly; it's just not designed for our use case.


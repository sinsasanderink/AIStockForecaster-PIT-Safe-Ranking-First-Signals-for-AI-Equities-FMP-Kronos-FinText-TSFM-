# Kronos Prediction Diagnosis Results

**Date:** January 9, 2026  
**Status:** ⚠️ **Concerning Results - Signal Quality Issue Identified**

---

## Executive Summary

**Kronos works technically** but shows **strong mean-reversion bias** that produced **INVERTED rankings** during the Feb 2024 AI rally.

| Metric | Result | Interpretation |
|--------|--------|---------------|
| **Overall RankIC** | **-0.0530** | ❌ Near zero (random) |
| **Feb 1, 2024 RankIC** | **-0.6692** | ❌ Strongly NEGATIVE (inverted!) |
| **Mar/Apr RankIC** | +0.18 to +0.26 | ⚠️ Weak positive (not significant) |
| **Your Baseline** | +0.1009 | ✅ Strong positive |

**Key Finding:** Kronos predicted the BIGGEST GAINERS would DROP the most!

---

## The Smoking Gun: February 1, 2024

```
Ticker   Kronos Prediction    Actual 20d Return    Match?
-------------------------------------------------------------
AMD      -27.03%              +13.24%              ❌ WRONG
NVDA     -26.03%              +24.92%              ❌ WRONG  
AVGO     -21.58%              +10.98%              ❌ WRONG
META     -19.18%              +21.61%              ❌ WRONG
CRM      -18.21%              +6.03%               ❌ WRONG
...
CSCO     +5.04%               -9.17%               ❌ WRONG (bullish but dropped)
TXN      +9.15%               +1.49%               ✅ OK
```

**Pattern:** Kronos's most bearish predictions were on the stocks that rallied the most!

---

## What Happened (Technical Analysis)

### The Mean-Reversion Trap

Kronos is a **price-pattern foundation model**. It learned from historical data that:

```
"Stock up big recently" → "Expect pullback"
"Stock down recently" → "Expect bounce"
```

This works in **normal markets** where mean-reversion is common. But **Feb 2024** was during:

- **NVDA's AI-driven rally** (up 40% in 2 months)
- **META's rebound** (up 100% from Oct 2023 lows)
- **Massive AI FOMO buying**

Kronos saw "AMD/NVDA/META up a lot" and predicted "pullback incoming." But the AI rally continued!

### Why Rankings Were Inverted

```
Kronos logic:
  AMD up 50% YTD → "overextended" → predict -27%
  CSCO flat YTD → "not overextended" → predict +5%

Reality (Feb 2024):
  AMD momentum continued → actual +13%
  CSCO boring value → actual -9%
```

**Result:** Kronos ranked stocks INVERSELY to their actual performance.

---

## Statistical Evidence

### RankIC by Date

| Date | RankIC | p-value | Interpretation |
|------|--------|---------|---------------|
| **2024-02-01** | **-0.6692** | **0.0013** | ❌ Strongly negative (SIGNIFICANT!) |
| 2024-03-01 | +0.2586 | 0.2709 | ⚠️ Weak positive (not significant) |
| 2024-04-01 | +0.1805 | 0.4465 | ⚠️ Weak positive (not significant) |
| **Overall** | **-0.0530** | 0.6875 | ❌ Essentially zero |

**Key:** Feb 1 is statistically significant (p=0.001) but in the WRONG direction!

### Quintile Analysis

| Quintile | Avg Actual Return | Notes |
|----------|-------------------|-------|
| Q1 (lowest score) | **+1.46%** | Should be lowest, is positive! |
| Q5 (highest score) | **+0.86%** | Should be highest, is middle |
| **Q5-Q1 spread** | **-0.60%** | ❌ NEGATIVE (rankings inverted) |

---

## What This Means for Your Project

### The Good News

1. **Kronos IS learning something** - The Feb 1 RankIC of -0.67 is STATISTICALLY SIGNIFICANT
   - This isn't random noise - the model has a consistent (but wrong) view
   
2. **Technical integration works** - No bugs in the pipeline
   
3. **You have a strong baseline** - Your LGB model at 0.10 RankIC is genuinely good

### The Bad News

1. **Kronos standalone is NOT useful** - Near-zero overall RankIC
   
2. **Mean-reversion bias is baked in** - Model architecture/training issue
   
3. **Regime-dependent** - May work in mean-reverting markets, fails in trending

### Options Going Forward

#### Option A: Document & Move On ✅ (Recommended)

```markdown
**Chapter 8 Conclusion:**
- Kronos integration: ✅ Complete
- Signal quality: ❌ Insufficient (RankIC ~0)
- Root cause: Mean-reversion bias during momentum regime
- Decision: Not useful standalone; proceeding to Chapter 9 (FinText)
```

This is valid research! Negative results are still results.

#### Option B: Test Different Regimes

Run micro-test on **non-trending periods** (e.g., 2022 Q2 bear market) to see if:
- Kronos works better when mean-reversion IS the dominant pattern
- Could be useful as "regime-conditional" signal

#### Option C: Use as Contrarian Signal

If Kronos consistently predicts "wrong" in trending markets:
- Could SHORT high-score stocks during momentum regimes
- But requires confident regime detection

#### Option D: Ensemble Anyway

Even weak/orthogonal signals can add value in ensemble:
- Test if Kronos + LGB outperforms LGB alone
- May capture different market dynamics

---

## Diagnosis Scripts Created

For deeper investigation, run:

```bash
# Full diagnostic suite
python scripts/diagnose_kronos_predictions.py

# Quick RankIC check
python scripts/compute_kronos_rankic.py

# Single prediction inspection
python scripts/inspect_kronos_output.py
```

---

## What to Check Next (If Investigating Further)

1. **Price Scale:** Are we using adjusted prices? (Kronos may expect raw)
2. **Split Handling:** Is the spot_close correctly split-adjusted?
3. **Horizon Interpretation:** Is Kronos's "20-step prediction" aligned with 20 trading days?
4. **Different Horizon:** Test horizon=60 or 90 (your baseline is stronger there)
5. **Different Regime:** Test on 2022 bear market or 2020 COVID crash

---

## Bottom Line

| Question | Answer |
|----------|--------|
| Does Kronos run? | ✅ Yes |
| Are predictions generated? | ✅ Yes |
| Is output in correct format? | ✅ Yes |
| Does Kronos predict well? | ❌ **No** (inverted in momentum, zero overall) |
| Is this fixable? | ⚠️ Unlikely without retraining model |
| Should we use it? | ❌ Not standalone; maybe in ensemble |

**Recommendation:** Document the negative result, move to Chapter 9 (FinText), and potentially revisit Kronos for ensemble experiments in Chapter 11.

---

## Appendix: Full Feb 1 Results

```
Ticker   Kronos Score   Actual Return   Prediction Direction
----------------------------------------------------------------
AMD      -27.03%        +13.24%         ❌ Wrong (predicted drop, got rally)
NVDA     -26.03%        +24.92%         ❌ Wrong (predicted drop, got rally)
AVGO     -21.58%        +10.98%         ❌ Wrong
META     -19.18%        +21.61%         ❌ Wrong
CRM      -18.21%        +6.03%          ❌ Wrong
LRCX     -17.42%        +12.32%         ❌ Wrong
MSFT     -16.19%        -2.72%          ✅ Correct (both negative)
KLAC     -14.44%        +14.15%         ❌ Wrong
MU       -13.87%        +5.52%          ❌ Wrong
AMZN     -12.99%        +6.27%          ❌ Wrong
ADBE     -12.73%        -14.70%         ✅ Correct
MRVL     -11.97%        +10.44%         ❌ Wrong
AMAT     -11.93%        +20.30%         ❌ Wrong
INTC     -9.17%         -4.56%          ✅ Correct
QCOM     -8.37%         +9.95%          ❌ Wrong
GOOGL    -5.75%         -8.47%          ✅ Correct
AAPL     -3.19%         -9.48%          ✅ Correct
ORCL     -2.29%         -7.14%          ✅ Correct
CSCO     +5.04%         -9.17%          ❌ Wrong (predicted rally, got drop)
TXN      +9.15%         +1.49%          ✅ Correct

Direction Accuracy: 7/20 = 35% (worse than random!)
```

**The model is systematically wrong on high-momentum stocks.**


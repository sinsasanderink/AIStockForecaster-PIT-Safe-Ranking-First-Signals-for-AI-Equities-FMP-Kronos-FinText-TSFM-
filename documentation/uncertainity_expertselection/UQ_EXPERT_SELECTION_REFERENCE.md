# Uncertainty Quantification & Expert Selection — Complete Reference

> **Purpose:** Context document. Contains all theory, research papers, architecture decisions, and implementation guidance for the UQ layer of the trading system.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Uncertainty Theory — From Basics to DEUP](#2-uncertainty-theory)
3. [DEUP — Salem Lahlou's Method (Core)](#3-deup)
4. [Aleatoric Uncertainty Estimation](#4-aleatoric-uncertainty)
5. [Conformal Prediction](#5-conformal-prediction)
6. [KDE & OOD Detection](#6-kde-and-ood-detection)
7. [Contextual Bandits & Expert Selection](#7-contextual-bandits)
8. [Architecture: What Belongs Where](#8-architecture)
9. [Research Papers & Literature](#9-research-papers)
10. [Literature Gap — Why This Is Novel](#10-literature-gap)
11. [Diagnostics & Ablations](#11-diagnostics)
12. [Success Criteria](#12-success-criteria)
13. [Implementation Guide](#13-implementation)
14. [Open Questions & Risks](#14-open-questions)

---

## 1. System Overview

### The Big Picture

We are building a **multi-strategy trading system** where multiple independent trading strategies ("experts") each produce return predictions, and a **meta-controller** decides which expert to trust at any given time and how much capital to allocate.

The key innovation is using **DEUP-based uncertainty quantification** to make these decisions, rather than simple heuristics or ensemble disagreement.

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────┐
│  TRADING SYSTEM (Meta-Controller)                       │
│                                                         │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐   │
│  │Expert 1 │  │Expert 2 │  │Expert 3 │  │Expert N │   │
│  │AI Stock │  │Mean Rev │  │Momentum │  │Earnings │   │
│  │Fcaster  │  │Strategy │  │Strategy │  │Drift    │   │
│  │         │  │         │  │         │  │         │   │
│  │ g_1(x)  │  │ g_2(x)  │  │ g_3(x)  │  │ g_N(x)  │   │
│  │ ê_1(x)  │  │ ê_2(x)  │  │ ê_3(x)  │  │ ê_N(x)  │   │
│  │conformal│  │conformal│  │conformal│  │conformal│   │
│  │intervals│  │intervals│  │intervals│  │intervals│   │
│  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘   │
│       │             │             │             │        │
│       └─────────────┴──────┬──────┴─────────────┘        │
│                            │                             │
│  SHARED:        ┌──────────┴──────────┐                  │
│  - a(x)         │  System-Level UCB   │                  │
│    (aleatoric)  │  Expert Selection   │                  │
│  - Residual     │  Position Sizing    │                  │
│    Archive      │  Monitoring         │                  │
│                 └─────────────────────┘                  │
└─────────────────────────────────────────────────────────┘
```

### Key Decisions

- **DEUP is per-expert.** Each trading strategy trains its own error predictor g_i(x) because each has different failure modes.
- **Aleatoric baseline is shared.** Market noise is a property of the asset, not the model.
- **Expert selection (UCB) is system-level.** It compares ê_i across strategies.
- **Conformal intervals are per-expert.** Each strategy's residuals have different distributions.

---

## 2. Uncertainty Theory

### 2.1 Aleatoric vs Epistemic Uncertainty

**Aleatoric (market noise):** Inherent randomness. Even a perfect model can't predict through earnings surprises, macro shocks, or microstructure noise. **Irreducible.**

*Analogy:* Rolling a fair die. You know everything about the die. You still can't predict the outcome.

*In finance:* Daily price moves have inherent volatility. NVDA moves ±3% on a normal day — no model eliminates this.

**Epistemic (model ignorance):** The model hasn't seen enough data, is the wrong type, or the regime changed. **Reducible** with better data or a different model.

*Analogy:* Someone hands you a weighted die you've never seen. Your uncertainty includes both "dice are random" AND "I don't know this die." Rolling it 1000 times would reduce the second part.

*In finance:* Your model trained on bull market data enters a crisis. It has never seen conditions like this.

### 2.2 Why the Distinction Drives Different Actions

| Situation | Type | Correct Response |
|-----------|------|-----------------|
| Market is volatile today | Aleatoric ↑ | Trade smaller (reduce position size) |
| Model has never seen this regime | Epistemic ↑ | Switch to a different expert / adapt |
| Both at once | Both ↑ | Trade smaller AND consider switching |

**Conflating them leads to wrong decisions.** You might switch experts when you should just trade smaller, or vice versa. Every experienced trader intuitively knows this distinction — they just detect the second by losing money first. DEUP tries to detect it before the loss.

### 2.3 The Law of Total Variance (Formal Decomposition)

```
Var[y | x, D] = E_θ[Var[y | x, θ]]  +  Var_θ[E[y | x, θ]]
                ─────────────────────     ─────────────────────
                    ALEATORIC                  EPISTEMIC
```

- **Left side:** Total uncertainty about y given input x and data D.
- **First term (Aleatoric):** Average noise level across all possible models θ. Irreducible.
- **Second term (Epistemic):** How much different models disagree about the expected value. Reducible with more data.

**Numeric example:**
```
Model 1 (θ₁): predicts mean=5%, noise std=2%
Model 2 (θ₂): predicts mean=3%, noise std=2%
Model 3 (θ₃): predicts mean=7%, noise std=2%

Aleatoric = average of variances = (4 + 4 + 4)/3 = 4
Epistemic = variance of means = Var(5, 3, 7) = 8/3 ≈ 2.67
Total = 6.67
```

### 2.4 Why Variance Decomposition Is Insufficient

The variance decomposition assumes your model class is correct — it just has parameter uncertainty. But what if the entire model class is wrong?

**Model misspecification:** Using a linear model for a nonlinear problem. Even with infinite data, the posterior collapses to a point (Var_θ → 0) but the model is still wrong. The variance decomposition reports zero epistemic uncertainty, but the model is fundamentally broken.

This is DEUP's key insight: epistemic uncertainty should be measured as **excess risk** (how much worse than theoretically possible), not **posterior variance** (how spread out are the weights).

### 2.5 Why Ensembles Fail

Ensembles capture parameter uncertainty by training multiple models and measuring disagreement. Limitations:

1. **Shared blindness:** All members trained on the same data. If market enters an unseen regime, they all confidently agree on wrong predictions. Disagreement is low, but error is high.
2. **Cost:** N experts × M ensemble members = N×M models. Prohibitive.
3. **No misspecification detection:** Ensemble variance measures "how spread are compatible weights," not "is this model class wrong here."
4. **No aleatoric separation by default:** Raw disagreement mixes both uncertainty types.

**DEUP's advantage:** Single-model epistemic estimation that captures misspecification. Train one expert + one uncertainty predictor, not 5-10 ensemble copies.

---

## 3. DEUP — Salem Lahlou's Method

### 3.1 Paper Details

**Citation:** Jain, M., Lahlou, S., Nekoei, H., Butoi, V.I., Bertin, P., Rector-Brooks, J., Korablyov, M., & Bengio, Y. (2023). "DEUP: Direct Epistemic Uncertainty Prediction." *Transactions on Machine Learning Research (TMLR)*, 2023.

**ArXiv:** 2102.08501

**Authors:** Moksh Jain, **Salem Lahlou** (co-first author), + 6 others including **Yoshua Bengio**

**Original test domains:** Sequential model optimization (synthetic), RL (MuJoCo), image classification (CIFAR-10 with ResNet-18), drug synergy prediction. **NEVER tested on finance.**

### 3.2 Core Concept

DEUP reframes epistemic uncertainty as **excess risk** — the gap between your predictor's generalization error and the Bayes-optimal predictor's error.

### 3.3 Mathematical Formulation

**Population risk:**
```
R(f) = E[l(Y, f(X))]          — average loss of predictor f
R*    = inf_f R(f)             — Bayes risk (best possible)
```

**Per-input decomposition:**
```
L_f(x) = E[l(Y, f(X)) | X=x]    — expected loss at input x
L*(x)  = inf_g E[l(Y, g(X)) | X=x] — Bayes pointwise loss (aleatoric)
```

**Epistemic uncertainty = excess risk:**
```
ê(x) = max(0, g(x) - a(x))
```

Where:
- `g(x)`: predicts generalization error of model at input x
- `a(x)`: estimates irreducible noise (Bayes pointwise loss) at input x
- `max(0, ...)`: ensures non-negative epistemic uncertainty

### 3.4 Why DEUP Captures Misspecification

Even when all ensemble members agree (posterior variance → 0), if the model class is wrong for the current regime, g(x) stays high while a(x) stays moderate → ê(x) reveals the model is failing. This is exactly the failure mode ensembles miss.

**Concrete trading examples:**

**Scenario A: Normal market day**
```
g(x) = 3%    (model expects ~3% error)
a(x) = 2.5%  (irreducible noise is ~2.5%)
ê(x) = 0.5%  → LOW epistemic uncertainty
→ Trust the model. Almost all error is from market noise.
```

**Scenario B: Regime shift (COVID crash)**
```
g(x) = 12%   (model expects very high error)
a(x) = 4%    (volatile, but noise is only 4%)
ê(x) = 8%    → HIGH epistemic uncertainty
→ Model is lost. Switch expert or reduce exposure.
```

**Scenario C: High volatility, model is fine**
```
g(x) = 6%    (moderate error expected)
a(x) = 5.5%  (world is very noisy)
ê(x) = 0.5%  → LOW epistemic uncertainty
→ Trade smaller (noise), but keep the expert (model is fine).
```

### 3.5 DEUP Algorithm (6 Steps)

**Step 1:** Train primary predictor f on training data (already done — e.g., tabular_lgb with RankIC 0.18).

**Step 2:** Get out-of-sample errors using walk-forward evaluation:
```
For each stock on each date in validation:
    loss = |actual_return - predicted_return|
→ Dataset of (features, loss) pairs
```

**Step 3:** Train error predictor g(x) — a secondary model that predicts how wrong the primary model will be:
```python
g = LightGBM()
g.fit(features, observed_losses)
```
Uses same features as main model + auxiliary DEUP features:
- Log-density estimate (k-NN distance or PCA-reduced KDE)
- Ensemble variance (if available)
- Binary in-training-set flag

**Step 4:** Estimate aleatoric baseline a(x) using financial-native methods (see Section 4).

**Step 5:** Compute epistemic signal:
```python
ê(x) = max(0, g(x) - a(x))
```

**Step 6:** Use for decisions (expert selection, position sizing, adaptation triggers).

### 3.6 Cross-Validation Bootstrap (DEUP Algorithm 1)

For the initial training set (before interactive updates):
1. Split training data into K folds
2. Train expert on K-1 folds, get held-out losses on remaining fold
3. Use these as initial targets for g_i
4. This avoids needing a separate hold-out set for the error predictor

### 3.7 Non-Stationarity Handling

DEUP was designed for interactive learning (RL, active learning) where data distribution changes. The paper handles this via:
- Retraining g(x) on rolling windows of recent held-out losses
- Adding density features that capture distribution shift
- Using the in-training-set binary flag to distinguish familiar vs unfamiliar inputs

This maps naturally to financial walk-forward evaluation where the training window slides forward over time.

---

## 4. Aleatoric Uncertainty Estimation

The aleatoric baseline a(x) estimates "how much error would ANY model make here due to irreducible market noise." This is a property of the **asset and market conditions**, not the model. One a(x) is shared across all experts.

### 4.1 Realized Volatility (Primary)

```
a(x) = σ_realized(x) = std(returns over rolling 20-day window)
```

Simple, fast, directly interpretable. "How noisy is this stock right now."

### 4.2 GARCH(1,1) (Secondary)

```
σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}
```

Models time-varying volatility with persistence. Better at capturing volatility clustering (high-vol days follow high-vol days). Standard in quantitative finance.

- ω: long-run variance weight
- α: reaction to recent shocks (typically 0.05-0.15)
- β: persistence of volatility (typically 0.80-0.95)
- α + β < 1 for stationarity

### 4.3 VIX Level (Tertiary — Market-Wide Proxy)

Captures systematic noise level across all stocks. Useful as a regime-level aleatoric floor. Less granular (doesn't differentiate between stocks) but captures macro uncertainty.

### 4.4 Calibration

Fit a(x) to predict the minimum achievable loss across all experts at each point — this approximates L*(x):

```
a(x) ≈ min_i(loss_i(x))  averaged over a local neighborhood
```

Or: heteroscedastic regression from features to min-expert-loss.

---

## 5. Conformal Prediction

### 5.1 Core Idea

Turn any point prediction into a prediction **interval** with **finite-sample, distribution-free** coverage guarantees.

### 5.2 Split Conformal Algorithm

1. Train model f on training data
2. Compute nonconformity scores on calibration set: `s_j = |y_j - f(x_j)|`
3. Sort scores, find threshold q at the `⌈(n_cal+1)(1-α)⌉`-th smallest
4. Prediction interval: `[f(x*) - q, f(x*) + q]`

**Guarantee:** `P[Y ∈ interval] ≥ 1-α` for exchangeable data. No distributional assumptions.

### 5.3 Problem: Non-Stationarity

The guarantee assumes exchangeability (future looks statistically like past). Financial data violates this. Last month's calibration errors may not represent next month's.

### 5.4 Rolling Conformal (Our Solution)

Replace fixed calibration set with a rolling window:
- Use most recent 60 trading days of out-of-sample residuals as calibration
- Recompute q at each rebalance date
- Intervals auto-widen after high-error periods, tighten during stable periods

```
Month 1 (calm):    q = 2% → tight intervals
Month 2 (crisis):  q = 8% → wide intervals
Month 3 (recovery): q = 3% → intervals tighten
```

### 5.5 DEUP-Normalized Conformal Scores (Novel Contribution)

Instead of raw residuals, normalize by epistemic uncertainty:

```
s_j = |y_j - f(x_j)| / ê(x_j)
```

Resulting interval:
```
[f(x*) - q · ê(x*),  f(x*) + q · ê(x*)]
```

Width adapts per-stock, per-date:
- High ê → wide intervals (model is uncertain)
- Low ê → tight intervals (model is confident)

This approximates **conditional coverage**, which standard conformal cannot guarantee.

### 5.6 Adaptive Conformal Inference (ACI)

Reference: Gibbs & Candès (2021). Adjusts α online based on whether recent predictions were covered. If coverage drops, α decreases (intervals widen).

### 5.7 Connection to Maxim Panov's Work

Maxim's research on conditional conformal prediction shows perfect conditional coverage is impossible without strong assumptions. Our DEUP-normalized scores are a practical attempt to improve conditional adaptivity. His federated conformal prediction work (NeurIPS/ICML workshops) addresses label shift between agents — analogous to our regime shift problem.

---

## 6. KDE and OOD Detection

### 6.1 What KDE Is

Kernel Density Estimation — a nonparametric density estimator. Place a kernel K (typically Gaussian) centered on each training point x_i:

```
p̂(x) = (1/nh^d) Σ K((x - x_i)/h)
```

- n: number of training points
- h: bandwidth (smoothing parameter)
- d: dimensionality
- K: kernel function (usually Gaussian bell curve)

### 6.2 Why It Works for OOD Detection

If p̂(x_new) is low, the new input lies in a region rarely seen during training → model is extrapolating → high epistemic uncertainty.

### 6.3 The Curse of Dimensionality

KDE fails in high dimensions (50+ features). In high-D, everything is far from everything else, and density estimates become unreliable.

**Solution:** Operate in a learned latent space:
- PCA-reduced feature space (e.g., top 10 components)
- Model's internal representations
- Or use k-NN distance as a simpler proxy (distance to k-th nearest training neighbor)

### 6.4 Connection to DEUP

DEUP uses density features (log-density, k-NN distance) as auxiliary inputs to g(x). Low density signals "the model hasn't seen conditions like this" which helps g(x) predict higher generalization error for unfamiliar inputs.

### 6.5 Connection to Panov's NeurIPS 2022 Paper

Kotelevskii et al. (NeurIPS 2022), "Nonparametric Uncertainty Quantification for Single Deterministic Neural Network" — uses density estimation in the model's representation space for UQ without ensembles. Directly relevant to our density-based OOD detection approach.

---

## 7. Contextual Bandits & Expert Selection

### 7.1 The Multi-Armed Bandit Problem

5 slot machines with unknown payouts. Do you exploit (keep playing the best one) or explore (try others that might be better)?

### 7.2 Contextual Bandits (Our Setting)

Add context: at each round, observe market conditions. Different experts may be better in different contexts.

- "Machines" = expert trading strategies
- "Context" = market features (volatility, regime, sector rotation)
- "Payout" = trading profit/loss
- "Choose a machine" = pick which expert to follow

### 7.3 UCB Expert Selection Rule

```
UCB_i(x_t) = r̂_i(x_t) + β_t · √ê_i(x_t) - λ_switch · 1{i ≠ i_{t-1}}
```

- `r̂_i(x_t)`: Expected reward from expert i given current conditions
- `β_t · √ê_i(x_t)`: Exploration bonus — higher when epistemic uncertainty is high (maybe this expert is better than we think)
- `λ_switch · 1{i ≠ i_{t-1}}`: Switching cost penalty (turnover, transaction costs)

Pick the expert with the highest UCB score.

### 7.4 Why Per-Expert DEUP Makes UCB Work

UCB needs to **compare** uncertainty across experts. This only works if each ê_i captures that expert's **specific** failure mode:

- If ê_momentum spikes but ê_meanrev is low → UCB switches to mean reversion
- If all ê_i spike simultaneously → system recognizes "no expert is reliable" → reduce overall exposure
- A single system-wide ê couldn't make these comparisons

### 7.5 Uncertainty-Gated Position Sizing

After expert selection, size position inversely proportional to epistemic uncertainty:

```
w_t = min(1, c / √(ê_{i_t}(x_t) + ε))
```

Different from volatility-based sizing — responds to **model failure risk**, not market noise.

**Optional dual-gated sizing** (combine epistemic and aleatoric):
```
w_t = min(1, c / √(ê(x_t) + ε)) · min(1, d / √(a(x_t) + ε))
```
- First term: reduce when model is uncertain (epistemic)
- Second term: reduce when market is noisy (aleatoric)
- Distinct actions for distinct uncertainty types — the core thesis

### 7.6 Dynamic Regret

Standard bandits measure regret against the single best expert. Our setting needs **dynamic regret** against a comparator that can also switch (up to S* times):

```
R_T ≤ Õ(√(NV_T(T+d))) + switching costs + ε_gate·T + execution costs
```

Where V_T is the total variation of the best-expert sequence.

---

## 8. Architecture: What Belongs Where

### 8.1 Per-Expert (Inside Each Trading Strategy)

| Component | Why Per-Expert | Implementation |
|-----------|---------------|----------------|
| g_i(x) error predictor | Each model has different failure modes | `deup_estimator.py` |
| ê_i(x) = max(0, g_i - a) | Epistemic uncertainty is model-specific | `deup_estimator.py` |
| Conformal intervals | Each model's residuals have different distributions | `conformal_intervals.py` |
| Calibration diagnostics | Per-model ECE, coverage checks | `calibration.py` |
| Diagnostic ablations | Per-model selective risk, partial correlation | `diagnostics.py` |
| Residual archive | Each model's held-out losses (training data for g_i) | `residual_archive.py` |

### 8.2 Shared Infrastructure

| Component | Why Shared | Implementation |
|-----------|-----------|----------------|
| a(x) aleatoric baseline | Property of asset, not model. NVDA's noise is the same regardless of which strategy predicts it | `aleatoric_baseline.py` |
| Residual archive format | Common storage so any new strategy plugs in identically | `residual_archive.py` |

### 8.3 System-Level (Meta-Controller)

| Component | Why System-Level | Implementation |
|-----------|-----------------|----------------|
| UCB expert selection | Compares ê_i across strategies | `ucb_selector.py` |
| Position sizing | Capital allocation after expert selection | `position_sizer.py` |
| Cross-strategy monitoring | Switching frequency, correlation, dynamic regret | `cross_strategy_monitor.py` |
| System-level ablations | UCB with DEUP vs VIX, vs static expert, vs regime heuristic | `system_ablations.py` |

### 8.4 What Each Expert Must Provide to the System

Each expert strategy exposes this interface:

```python
class ExpertInterface:
    def predict(self, features) -> float:
        """Return prediction (ranking score or expected return)"""
    
    def epistemic_uncertainty(self, features) -> float:
        """Return ê_i(x) — DEUP epistemic uncertainty"""
    
    def conformal_interval(self, features, alpha=0.10) -> tuple[float, float]:
        """Return (lower, upper) prediction interval"""
    
    def residuals(self) -> ResidualArchive:
        """Access to held-out loss history for diagnostics"""
```

The system-level controller only needs `predict()` and `epistemic_uncertainty()` to make decisions.

### 8.5 Why This Split Matters

**Modularity:** When you build a new strategy (e.g., earnings drift model), you just:
1. Save residuals in the standard archive format
2. Train its own g_i(x) using `deup_estimator.py`
3. Reuse the shared `aleatoric_baseline.py`
4. Run `diagnostics.py` to validate

The same code handles any expert. You don't retrain the system-level controller.

**Correct information flow:** Each expert's failure modes are private information (only its residuals reveal when it fails). But the decision of which to trust is a comparative operation that needs all private signals simultaneously.

---

## 9. Research Papers & Literature

### 9.1 Core Papers (Our Methods)

**DEUP:**
- Jain, M., Lahlou, S., et al. "DEUP: Direct Epistemic Uncertainty Prediction." *TMLR*, 2023. (arXiv: 2102.08501)
- **Key contribution:** Epistemic uncertainty as excess risk, not posterior variance. Captures model misspecification.
- **Test domains:** Sequential model optimization, RL (MuJoCo), CIFAR-10, drug synergy. **No finance.**

**Panov's Nonparametric UQ:**
- Kotelevskii, A., Artemenkov, K., et al. "Nonparametric Uncertainty Quantification for Single Deterministic Neural Network." *NeurIPS*, 2022.
- **Key contribution:** UQ without ensembles using density estimation in learned representation space.
- **Test domains:** MNIST, SVHN, CIFAR, ImageNet, text classification. **No finance.**

**Panov's Conformal Prediction:**
- Plassier, V., Makni, M., Rubashevskii, A., Moulines, E., & Panov, M. "Conformal Prediction for Federated Uncertainty Quantification Under Label Shift." *ICML/Proceedings of ML Research*.
- **Key contribution:** Federated conformal prediction with importance weighting for label shift.
- **Test domains:** Federated learning benchmarks. **No trading systems.**

### 9.2 Salem Lahlou — Research Profile

- **Position:** Assistant Professor, MBZUAI (Mohamed bin Zayed University of Artificial Intelligence)
- **PhD:** Uncertainty modeling, Université de Montréal
- **Key works:** DEUP (TMLR 2023), GFlowNet Foundations (JMLR 2023, with Yoshua Bengio), BabyAI (ICLR 2019)
- **Current focus:** AI for science tools, uncertainty quantification, interactive learning, sample-efficient RL, generative models (GFlowNets), language model reasoning
- **Recent:** FinChain (2025) — LLM reasoning on financial math (compound interest, DCF). Not UQ on market data.
- **Connection to Maxim:** Co-organizes Bayesian ML workshop ("Rethinking the Role of Bayesianism in the Age of Modern AI," October 2025) with Maxim Panov.

### 9.3 Maxim Panov — Research Profile

- **Position:** Assistant Professor, MBZUAI
- **Previous:** Research scientist at DATADVANCE (pSeven library, used by Airbus, Porsche, Toyota), Skolkovo Institute
- **Key works:** Nonparametric UQ (NeurIPS 2022), Conformal Prediction for Federated UQ, UQ for LLMs benchmark, Embedded Ensembles (AISTATS 2022)
- **Current focus:** UQ for ML predictions, Bayesian approaches, conformal prediction, OOD detection, UQ for LLMs
- **Awards:** Best Paper Runner-up at UAI 2023

### 9.4 Finance UQ Papers (Existing Literature)

**Chauhan, Alberg & Lipton (2020) — "Uncertainty-Aware Lookahead Factor Models"**
- Most rigorous finance UQ with real backtest to date
- Methods: MC Dropout + heteroscedastic regression (basic by current standards)
- Results: Sharpe 0.84 vs 0.52, industry-grade simulator
- Limitation: No excess-risk decomposition, no conformal intervals, no formal epistemic/aleatoric separation
- Venue: ICML workshop (not top-tier)

**Zhang et al. (Oct 2025) — "Trading with the Devil"**
- Extends CAPM to separate foundation-model risk: epistemic (pretrained) vs aleatoric (fine-tuning)
- Methods: MC Dropout for disentanglement
- Tests: US equities, crypto with TSFMs
- Focus: Pricing foundation-model risk (pretrained market line), not trading decisions
- Still uses MC Dropout, not DEUP

**PACIS 2025 RL Trading Paper**
- SHAP-weighted reconstruction + MC Dropout in RL agent
- Tests: 5 US indices, better recession performance
- Methods: MC Dropout + heuristics (not frontier UQ)

**VMD Dual-Uncertainty (2025)**
- Most explicit aleatoric/epistemic separation on S&P 500, FTSE 100
- Methods: Concrete Dropout (epistemic), WGAN (aleatoric), VMD for signal stabilization
- Limitation: 2016-2018 generation methods, not recent theoretical advances

**Kato (2024/2025) — Conformal Predictive Portfolio Selection**
- Uses conformal prediction for portfolio returns on real data
- Limitation: No epistemic/aleatoric decomposition, treats uncertainty as monolithic

### 9.5 UQ Survey Papers

**Blasco et al. (2024):** Explicitly states: "Very few authors analyze both epistemic and aleatoric uncertainty, and none analyze in depth how to decouple them" in financial asset prediction.

**Eggen et al. (2025):** Confirms the same gap from a different angle.

**Mucsányi et al. (2024):** Argued true aleatoric/epistemic disentanglement "may be impossible" because they're inherently correlated. Others (2025 multi-modal paper) pushed back: correlation is a data property, not fundamental impossibility. Active theoretical debate.

### 9.6 Panov Group — Recent Papers (2024-2025, directly relevant)

**Kotelevskii & Panov (ICLR 2025) — "From Risk to Uncertainty: A Generalized Framework for Uncertainty Quantification"**
- Formalizes Total Risk = Bayes Risk (aleatoric) + Excess Risk (epistemic) using strictly proper scoring rules
- Bayes risk = -G(η), excess risk = Bregman divergence D_G(η ∥ η̂)
- Explicitly mentions DEUP (Lahlou et al., 2023) as an alternative when Bayesian inference is intractable
- **Our connection:** Our DEUP formula ê = g(x) - a(x) is the empirical instantiation of this framework for ranking loss. We follow DEUP because LightGBM doesn't admit tractable Bayesian posterior inference.

**Plassier, Makni, Rubashevskii, Moulines & Panov (ICLR 2025) — "Probabilistic Conformal Prediction"**
- Shows standard conformal provides only marginal coverage, not conditional coverage
- CP² method achieves approximate conditional coverage by modeling conditional quantile of conformity score
- **Our connection:** Our DEUP-normalized conformal scores (13.5) approximate conditional validity by scaling nonconformity scores by predicted epistemic uncertainty — same motivation as CP².

**Fishkov, Kotelevskii & Panov (2025) — "UQ for Regression using Proper Scoring Rules"**
- Extends the classification framework to regression
- Under squared error: aleatoric = predictive variance, epistemic = variance of predictive mean across ensemble
- **Our connection:** Our ranking problem sits between classification and regression. The analog: aleatoric = ranking noise from cross-sectional dispersion, epistemic = excess ranking error from model failure.

### 9.7 Additional Key References

- **Adaptive Conformal Inference:** Gibbs & Candès (2021) — online α adjustment for non-stationarity
- **Deep Ensembles:** Lakshminarayanan et al. (2017) — the standard ensemble baseline
- **MC Dropout:** Gal & Ghahramani (2016) — dropout as approximate Bayesian inference
- **DUQ:** van Amersfoort et al. (2020) — deterministic uncertainty quantification
- **UPER (2024/2025):** Uses DEUP-style decomposition for RL replay prioritization — showed that dividing epistemic by aleatoric (information gain) outperforms raw epistemic for exploration
- **URL Pretrained Uncertainties (ICML 2025):** Kirchhof et al. — pretrained uncertainty modules that transfer zero-shot. Found learned uncertainties capture aleatoric uncertainty disentangled from epistemic. Solved gradient conflict in feed-forward uncertainty modules.

---

## 10. Literature Gap — Why This Is Novel

### 10.1 The Critical Gap

**Neither DEUP nor Panov's nonparametric UQ has ever been tested on real financial trading systems with point-in-time safe evaluation.**

The intersection of "sophisticated UQ methods" and "real finance with proper evaluation" is **empty**:

- **Finance UQ papers:** Use MC Dropout or ensembles (2016-era methods) without proper PIT-safe evaluation
- **Sophisticated UQ papers:** Test on CIFAR, ImageNet, MNIST, drug synergy. Never finance.
- **Survey papers:** Explicitly confirm this gap

### 10.2 What Makes Our Setup Unique

The AI Stock Forecaster has infrastructure that the UQ-for-finance literature lacks:
- 109 walk-forward folds with frozen baselines → real, PIT-safe held-out losses to train g(x)
- 483 passing tests → evaluation rigor
- Survivorship-bias-free universe construction
- Multiple sub-models (tabular_lgb, FinText, Kronos) with different architectures → structural diversity for DEUP evaluation
- RankIC of 0.10-0.18 across horizons → confirmed signal to protect/improve with UQ

### 10.3 Why This Is Publishable Either Way

If DEUP works brilliantly on finance → novel positive result in unexplored domain.
If DEUP fails informatively → diagnostic ablations explain why, still a contribution ("here's what happens when you try DEUP on non-stationary financial data with X challenges").

The ablation framework (Section 11) is designed to make it publishable regardless of outcome.

---

## 11. Diagnostics & Ablations

### 11.1 Proving Disentanglement

**Ablation A:** Replace ê(x) with realized volatility
- Use RV as uncertainty signal instead of DEUP
- Expected: Partially works (correlates with total error) but misses regime-specific model failures

**Ablation B:** Replace ê(x) with VIX
- Use VIX as uncertainty signal
- Expected: Captures market-wide stress but not stock-specific or model-specific failure

**Ablation C:** Full DEUP ê(x)
- Expected: Outperforms A and B on selective risk and OOD detection

### 11.2 Partial Correlation Test (Key Diagnostic)

```
ρ(ê_i, realized_vol | features) ≈ 0
```

If DEUP works correctly, ê should have near-zero partial correlation with realized volatility after conditioning on features. Proves ê captures something beyond repackaged volatility.

### 11.3 Selective Risk Curves

- Rank predictions by ê(x) from low to high
- Plot cumulative loss as predictions are added from most confident to least
- If ê works: loss is low for low-ê predictions and high for high-ê predictions
- Report: selective risk at 80% coverage (reject top 20% highest ê)

### 11.4 Cross-Strategy Ablations (System Level)

- **UCB with DEUP** vs **Static best-expert** (always use historically best strategy)
- **UCB with DEUP** vs **Regime-based switching** (heuristic rules)
- **UCB with DEUP** vs **UCB with VIX** (replace all ê_i with VIX)
- **UCB with DEUP** vs **Equal weight** (uniform allocation)

### 11.5 Position Sizing Ablations

- **ê-sizing** vs **vol-sizing** (use realized vol instead of ê)
- **Dual-gated** vs **ê-only** vs **vol-only**
- Expected: Dual-gated > ê-only > vol-only

---

## 12. Success Criteria

### 12.1 Per-Expert Criteria

| Metric | Target | What It Proves |
|--------|--------|----------------|
| ρ(ê_i, RV \| features) | ≈ 0 | ê is not repackaged volatility |
| AUROC (OOD detection via ê_i) | > 0.70 | ê identifies regime shifts |
| Selective risk at 80% | < full-set risk | Rejecting high-ê reduces error |
| ECE (calibration error) | < 0.05 | Conformal intervals well-calibrated |
| Coverage (90% nominal) | 85–95% | Rolling conformal maintains validity |
| Low-ê bucket RankIC | > full-set RankIC | Expert knows when to trust itself |

### 12.2 System-Level Criteria

| Metric | Target | What It Proves |
|--------|--------|----------------|
| UCB Sharpe > static best-expert | Positive spread | Dynamic selection adds value |
| UCB with DEUP > UCB with VIX | Positive spread | Per-expert ê beats market-wide proxy |
| Dynamic regret | Sublinear in T | UCB is learning |
| Switching frequency | 2-6 per year | Reasonable, not erratic |
| Dual-gated Sharpe > vol-only Sharpe | Positive spread | Decomposition has economic value |
| Cross-expert ê correlation | < 0.7 | Experts have genuinely different failure modes |

---

## 13. Implementation Guide

### 13.1 Directory Structure

```
src/uncertainty/
├── deup_estimator.py        # Core: g_i(x) - a(x) = ê_i(x), per expert
├── aleatoric_baseline.py    # GARCH, realized vol, VIX proxy (SHARED)
├── conformal_intervals.py   # Rolling adaptive conformal with ê normalization (per expert)
├── diagnostics.py           # Ablations, partial correlation, selective risk (per expert)
├── calibration.py           # ECE, quantile calibration, confidence stratification
└── residual_archive.py      # Stores per-fold held-out losses — shared format

src/system/
├── expert_registry.py       # Register experts, their DEUP models, reward histories
├── ucb_selector.py          # Contextual bandit with UCB rule
├── position_sizer.py        # Epistemic + aleatoric dual-gated sizing
├── cross_strategy_monitor.py # Switching frequency, correlation, dynamic regret
└── system_outputs.py        # System-level reporting and reasoning
```

### 13.2 Residual Archive Format

Every expert saves residuals in this format:

```python
@dataclass
class ResidualRecord:
    expert_id: str           # e.g., "ai_stock_forecaster", "mean_reversion"
    sub_model_id: str        # e.g., "tabular_lgb", "fintext" (within an expert)
    fold_id: int             # walk-forward fold number
    as_of_date: date         # prediction date (PIT-safe)
    ticker: str              # stock symbol
    features: dict           # input features at prediction time
    prediction: float        # model's output
    actual: float            # realized return
    loss: float              # |actual - prediction|
    horizon: str             # "5d", "20d", "60d", "90d"
```

### 13.3 Adding a New Expert

When a new trading strategy is built:

1. **Implement the strategy** with its own training, evaluation, walk-forward pipeline
2. **Save residuals** using `ResidualArchive` in the standard format
3. **Train g_i(x)** using `DEUPEstimator(expert_id="new_strategy").fit(residual_archive)`
4. **Reuse** `AleatonicBaseline` (shared, already fitted)
5. **Run** `Diagnostics(expert_id="new_strategy").run_all()` to validate ê_i
6. **Register** with `ExpertRegistry` to enter the UCB pool

Same code path for every expert. The only per-expert work is training g_i on that expert's specific residuals.

### 13.4 Key Implementation Notes

**g(x) training data size:** ~100 stocks × 109 folds × 3 horizons ≈ 30,000+ training points per expert. Sufficient for LightGBM.

**Rolling window for g(x):** Retrain g(x) periodically (e.g., every 20 folds) to capture evolving failure patterns. The relationship between features and model failure may itself be non-stationary.

**ê calibration across experts:** ê values from different experts may be on different scales (each g_i trained independently). Normalize via:
- Percentile rank within each expert's own history
- Or learn per-expert β_i in the UCB formula

**PIT safety:** All residuals must come from true out-of-sample predictions. Never use in-sample losses to train g(x). The walk-forward evaluation infrastructure guarantees this.

---

## 14. Open Questions & Risks

### 14.1 Core Gamble

Can g(x) (secondary model predicting primary model's error) learn anything useful in financial data?

In DEUP's original experiments (images, drugs, RL), the input→error relationship is relatively stable. In finance, the relationship between features and model failure is itself non-stationary. The error predictor might be as lost as the primary model during genuine regime shifts.

**Implicit assumption:** The pattern of failure (which features correlate with high error) is somewhat stable even if specific failures change. In finance, this might hold for structural reasons (model fails more during earnings, regime transitions, low liquidity) — but it's an empirical question.

### 14.2 KDE/Density-Based OOD: Weakest Link

With 50+ features all changing over time, the entire distribution is drifting. "Am I in-distribution?" becomes slippery. Must work in compressed latent space. The curse of dimensionality is real. This is known to be hard.

### 14.3 Narrow Universe Risk (AI Stock Forecaster Specifically)

~100 AI stocks is a narrow, correlated sector. Aleatoric noise is highly correlated (all move with NVDA, macro AI sentiment). GARCH on individual stocks might not capture this — might need factor-level aleatoric model.

Cross-sectional rankings with 100 stocks have limited granularity for confidence stratification (only ~33 per tercile).

**Mitigation:** AI Stock Forecaster is one expert in a broader system. A separate broad-market strategy (S&P 500) would complement it.

### 14.4 Disentanglement May Be Impossible

Mucsányi et al. (2024) argued true aleatoric/epistemic disentanglement "may be impossible" because they're inherently correlated. This is an active theoretical debate. Our ablation framework (partial correlation test, selective risk curves) is designed to measure how well the disentanglement works empirically, regardless of theoretical limits.

### 14.5 Circularity in Error Prediction

DEUP needs historical out-of-sample losses to train g(x). If the model and market conditions change fundamentally, past failure patterns may not predict future failures. The cross-validation bootstrap (Algorithm 1) partially addresses this but doesn't eliminate the risk.

### 14.6 Expert Selection with Few Strategies

UCB expert selection has only ~109 independent switching decisions across the backtest (one per rebalance). With 2-3 strategies, the dynamic regret evaluation is thin. Meaningful UCB evaluation requires either more rebalance dates or more strategies.

### 14.7 Kill Criteria

Build with clear kill criteria, not assumed success:
- If ρ(ê, RV | features) > 0.5: ê is just volatility, DEUP isn't working
- If selective risk at 80% ≈ full risk: ê has no discriminative power
- If conformal coverage < 70% or > 99%: rolling conformal is broken
- If UCB Sharpe ≤ static best-expert: expert selection adds no value

Even if DEUP disappoints, the infrastructure (residual archive, conformal intervals, diagnostics framework) is useful regardless. Expert switching has value even with simpler uncertainty proxies (ensemble disagreement, rolling error rate).

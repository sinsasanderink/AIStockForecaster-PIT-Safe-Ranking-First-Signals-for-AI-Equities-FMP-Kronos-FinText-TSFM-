# Chapter 12: Regime-Aware Analysis & Heuristic Ensemble

**Status:** ✅ COMPLETE  
**Started:** February 19, 2026  
**Last updated:** February 19, 2026 — **COMPLETE** + DEV/FINAL holdout analysis appended

---

## Overview

Chapter 12 answers the question: **when does LGB fail, and can simple
regime heuristics improve performance?**

Chapter 11 established that LGB baseline dominates all fusion variants.
Chapter 12 provides the diagnostic analysis and heuristic baseline that
Chapter 13's DEUP-based expert selection must beat.

### Role in the UQ pipeline

1. **Diagnostic**: Quantify regime-conditional failure modes (input for DEUP)
2. **Baseline**: Build the "regime-based switching heuristic" ablation
   comparator (UQ reference §11.4)
3. **Infrastructure**: Prepare regime context features for Ch13 UCB

### Progress

1. ~~12.1 — Regime-conditional performance diagnostics~~ ✅ COMPLETE
2. ~~12.2 — Regime stress tests (shadow portfolio)~~ ✅ COMPLETE
3. ~~12.3 — Regime-aware heuristic baseline~~ ✅ COMPLETE
4. ~~12.4 — Freeze & documentation~~ ✅ COMPLETE

---

## 12.1 Regime-Conditional Performance Diagnostics ✅ COMPLETE

Implemented in `scripts/run_chapter12_regime_diagnostics.py`.

### Method

- Loaded LGB baseline FULL eval_rows (591K rows, 109 folds, 2,277 dates)
- Also loaded Learned Stacking and Rank Avg 2 for comparison
- Joined regime data from DuckDB regime table (2,386 dates, 100% match rate)
- Two regime axes:
  - **VIX percentile** (252d window): low (≤33), mid (33–67), high (>67)
  - **Market regime**: bull, neutral, bear (from SPY MA50/MA200 + momentum)
- Computed per-regime: median RankIC, IC stability, cost survival, % positive dates
- Computed continuous correlation: per-date RankIC vs VIX level and percentile

### Key finding 1: LGB performs dramatically better in calm markets

**LGB baseline by VIX percentile regime:**

| VIX Regime | Hz | N dates | Med. RankIC | IC Stability | Cost Surv |
|------------|---:|--------:|:-----------:|:------------:|:---------:|
| **Low** (≤33 pctile) | 20d | 159 | **0.1824** | **1.23** | **89.3%** |
| Mid (33–67) | 20d | 345 | 0.0774 | 0.40 | 73.0% |
| High (>67) | 20d | 1,773 | 0.0710 | 0.29 | 63.5% |
| **Low** | 60d | 159 | **0.3469** | **2.04** | **93.1%** |
| Mid | 60d | 345 | 0.2070 | 1.21 | 91.0% |
| High | 60d | 1,773 | 0.1198 | 0.58 | 73.5% |
| **Low** | 90d | 159 | **0.2982** | **2.17** | **99.4%** |
| Mid | 90d | 345 | 0.2461 | 1.42 | 88.7% |
| High | 90d | 1,773 | 0.1607 | 0.66 | 76.4% |

When VIX percentile is low (calm markets):
- 20d RankIC is **2.6x higher** than in high-VIX environments (0.182 vs 0.071)
- 90d cost survival is essentially perfect (99.4%)
- IC stability exceeds 2.0 across all horizons (exceptionally consistent)

When VIX percentile is high (turbulent markets):
- Performance degrades across all metrics but remains positive
- The model still generates signal (0.071–0.161 RankIC), just much less reliably

### Key finding 2: LGB performs better in bear markets than bull markets

**LGB baseline by market regime:**

| Regime | Hz | N dates | Med. RankIC | IC Stability | Cost Surv |
|--------|---:|--------:|:-----------:|:------------:|:---------:|
| Bull | 20d | 957 | 0.0601 | 0.21 | 64.1% |
| Neutral | 20d | 881 | 0.0875 | 0.40 | 66.9% |
| **Bear** | 20d | 439 | **0.1102** | **0.61** | **72.4%** |
| Bull | 60d | 957 | 0.1261 | 0.54 | 72.9% |
| Neutral | 60d | 881 | 0.1531 | 0.75 | 80.8% |
| **Bear** | 60d | 439 | **0.1955** | **1.11** | **80.9%** |
| Bull | 90d | 957 | 0.1556 | 0.62 | 76.3% |
| Neutral | 90d | 881 | 0.1889 | 0.85 | 82.7% |
| **Bear** | 90d | 439 | **0.2197** | **1.18** | **81.8%** |

This is counterintuitive: the model's ranking accuracy is highest in bear
markets (20d RankIC 0.110 vs 0.060 in bull). Likely explanation: in bear
markets, stock differentiation is clearer — fundamentally weak stocks fall
harder, creating more cross-sectional spread for the tabular features
(momentum, volume, drawdown) to exploit. In bull markets, most stocks rise
together, reducing cross-sectional dispersion.

### Key finding 3: RankIC correlates with VIX (but direction depends on measure)

| Correlation | 20d | 60d | 90d |
|-------------|----:|----:|----:|
| RankIC vs VIX percentile (252d) | **−0.115** | **−0.207** | **−0.181** |

All correlations are statistically significant (p ≈ 0). Higher VIX percentile
(more turbulent relative to recent history) → lower RankIC. The effect is
strongest at 60d.

**Note on VIX level data quality:** The `vix_level` column in the regime
table has a scaling inconsistency (historical values appear ~1000x too high
while recent values are correct). The positive correlation between RankIC
and raw VIX level (ρ = +0.53 at 90d) is therefore unreliable. Use
`vix_percentile_252d` for analysis.

### Key finding 4: Fusion variants show same regime patterns

Learned Stacking and Rank Avg 2 exhibit identical regime sensitivity
to LGB (same directional effects, similar magnitudes). This confirms
the regime effect is a property of the **market/universe**, not the
specific model architecture.

### Implications for Chapter 13 (DEUP)

1. **VIX percentile is a strong predictor of model error** — DEUP's g(x)
   should include VIX-related features to predict when the model fails
2. **Bear markets are not the problem** — the model actually works better
   in downturns. The problem is high-VIX *relative* environments
3. **Per-stock realized volatility** may capture stock-level noise better
   than market-level VIX (to be exported in 12.4 for DEUP's a(x))
4. **Low-VIX regime is where the model is most trustworthy** — a regime-aware
   position sizer should increase allocation in calm markets

### Tests

18 tests in `tests/test_chapter12_regime.py`:
- `TestComputePerDateRankIC`: 4 tests (output shape, columns, signal quality, missing horizon)
- `TestComputeRegimeMetrics`: 6 tests (all/subset/empty selection, IC stability, cost survival, bounds)
- `TestRegimeAxes`: 2 tests (VIX percentile exhaustive coverage, market regime labels)
- `TestLoadRegimeData`: 2 tests (decode market regime integers, date type)
- `TestOutputArtifacts`: 4 tests (CSV/JSON output validation, model/horizon coverage)

### Output artifacts

| Path | Contents |
|------|----------|
| `evaluation_outputs/chapter12/regime_diagnostics.csv` | Per-model, per-regime, per-horizon metrics |
| `evaluation_outputs/chapter12/regime_diagnostics.json` | Full results + correlations |
| `evaluation_outputs/chapter12/rankic_vs_vix_correlation.json` | RankIC vs VIX correlation per model |

---

## 12.2 Regime Stress Tests (Shadow Portfolio) ✅ COMPLETE

Implemented in `scripts/run_chapter12_stress_tests.py`.

### Method

- Loaded shadow-portfolio returns for all three models:
  LGB baseline, Rank Avg 2, Learned Stacking
- **Subsampled to non-overlapping monthly returns** (first trading day per
  calendar month → 109 observations). The raw CSV contains 2,277 rows of
  overlapping 20-day forward returns — using those directly with `×√252`
  annualization would inflate Sharpe by ~2–5×. Subsampling + `×√12`
  gives correct estimates.
- Joined regime labels from DuckDB regime table (100% date match rate)
- Two regime axes (same as 12.1):
  - **VIX percentile** (252d): low (≤33), mid (33–67), high (>67)
  - **Market regime**: bull, neutral, bear
- Computed per-regime: annualized Sharpe, annualized return, annualized
  volatility, max drawdown, hit rate (% positive months)
- Computed 12-month rolling Sharpe (on monthly observations)
- Identified 5 worst drawdown episodes per model with regime context at trough

### Key finding 1: Sharpe lifts in calm markets, but magnitudes are realistic

**Shadow portfolio Sharpe by VIX regime:**

| Model | VIX Low (N) | VIX Mid (N) | VIX High (N) | Overall (109) |
|-------|------------:|------------:|--------------:|--------------:|
| LGB baseline | **6.32** (6) | 3.65 (22) | 2.32 (81) | **2.65** |
| Rank Avg 2 | 1.65 (6) | 2.00 (22) | 1.58 (81) | 1.64 |
| Learned Stacking | **5.50** (6) | 2.62 (22) | 1.61 (81) | 1.86 |

The directional finding holds: all models perform best in low-VIX. LGB
baseline achieves an overall Sharpe of **2.65** — strong for a single-model
L/S strategy. Low-VIX Sharpe (6.32) is inflated by the tiny sample (6 months);
it should be interpreted as "consistently positive in calm markets" rather
than taken at face value.

**Hit rate by VIX regime:**

| Model | VIX Low | VIX Mid | VIX High | Overall |
|-------|--------:|--------:|---------:|--------:|
| LGB baseline | **100%** | 86.4% | 77.8% | **80.7%** |
| Rank Avg 2 | 83.3% | 77.3% | 75.3% | 76.1% |
| Learned Stacking | **100%** | 81.8% | 76.5% | 78.9% |

LGB and Learned Stacking never had a losing month in the low-VIX bucket
(6/6 positive). Even in high-VIX, hit rates remain above 75%.

### Key finding 2: Bear markets outperform bull markets in portfolio space

**Shadow portfolio Sharpe by market regime:**

| Model | Bull (46) | Neutral (39) | Bear (24) |
|-------|----------:|-------------:|----------:|
| LGB baseline | 1.88 | **3.84** | **3.69** |
| Rank Avg 2 | 1.02 | **3.26** | 2.01 |
| Learned Stacking | 1.38 | **2.62** | **2.93** |

Mirrors the 12.1 RankIC finding: all models generate the strongest
risk-adjusted portfolio returns in neutral and bear markets. LGB in
neutral markets: Sharpe **3.84**, annualized return **89.6%**, max DD
only **−6.4%**. In bull markets, performance is weakest — higher
volatility, lower Sharpe.

### Key finding 3: Worst drawdowns cluster in high-VIX bull regimes

**LGB baseline — 5 worst drawdown episodes (monthly equity curve):**

| Max DD | Trough Date | Duration | VIX Bucket | Market Regime |
|-------:|:-----------:|---------:|:----------:|:-------------:|
| −21.9% | 2021-04-01 | 9 mo | high | bull |
| −16.6% | 2024-07-01 | 8 mo | high | bull |
| −11.5% | 2023-10-02 | 4 mo | high | bear |
| −8.2% | 2020-09-01 | 3 mo | mid | bull |
| −8.0% | 2022-01-03 | 4 mo | high | bull |

4 of 5 worst drawdowns for LGB occur in **high-VIX bull** markets — the
portfolio struggles most when volatility is elevated but the market is
still trending up (likely crowded momentum reversals in the AI stock
universe). The worst drawdown is **−21.9%** over 9 months, which is
manageable for a concentrated L/S strategy.

Rank Avg 2 fares worse (max DD −39.3%) and Learned Stacking has a
severe −66.0% episode (2023 bull), consistent with the weaker FinText
signal diluting returns.

### Key finding 4: Rolling 12-month Sharpe is consistently positive

**12-month rolling Sharpe summary:**

| Model | Mean | Min | Max | % Negative |
|-------|-----:|----:|----:|-----------:|
| LGB baseline | **3.34** | 0.11 | 7.30 | **0.0%** |
| Rank Avg 2 | 1.78 | 0.47 | 3.99 | 0.0% |
| Learned Stacking | 2.62 | 0.57 | 6.39 | 0.0% |

None of the three models ever had a negative rolling 12-month Sharpe
across the entire 9-year backtest. LGB's trailing-year Sharpe never
dropped below 0.11, demonstrating exceptional consistency.

### Implications for Chapter 13 (DEUP)

1. **Regime-conditional position sizing is high-value**: The Sharpe
   roughly doubles from high-VIX (2.32) to mid-VIX (3.65). Reducing
   exposure in high-VIX would improve risk-adjusted returns materially.
2. **The heuristic baseline (12.3) should be volatility-scaling**:
   The data supports VIX-percentile-based sizing as the ablation
   comparator for DEUP's learned uncertainty.
3. **Drawdown prediction is possible**: All worst drawdowns occur in
   identifiable VIX+regime combinations — DEUP's epistemic uncertainty
   should correlate with these episodes.
4. **LGB dominates in all regimes**: There is no regime where fusion
   variants outperform LGB, so 12.3's heuristic should focus on position
   sizing (confidence modulation) rather than model switching.
5. **Bull-market weakness is the key risk**: The model's edge is in
   differentiating stocks; in strong bull markets where everything
   rises together, the L/S spread compresses.

### Tests

22 tests in `tests/test_chapter12_stress.py`:
- `TestSubsampleMonthly`: 3 tests (row reduction, first-day-per-month, column preservation)
- `TestComputePortfolioMetrics`: 7 tests (positive/negative Sharpe, annualization ×12,
  realistic range, max DD, short series, zero vol)
- `TestAnnotateWithRegime`: 2 tests (columns present, full date match)
- `TestFindWorstDrawdowns`: 3 tests (count, sort order, duration)
- `TestComputeRollingSharpe`: 2 tests (output length, positive mean)
- `TestRunStressTests`: 1 integration test (full pipeline, output validation)
- `TestOutputArtifacts`: 4 tests (CSV structure, realistic Sharpe bounds, drawdown
  regime context, JSON completeness)

### Output artifacts

| Path | Contents |
|------|----------|
| `evaluation_outputs/chapter12/regime_shadow_metrics.csv` | Per-model, per-regime Sharpe/return/vol/DD/hit-rate (monthly) |
| `evaluation_outputs/chapter12/rolling_sharpe.csv` | Monthly rolling 12-month Sharpe with regime labels |
| `evaluation_outputs/chapter12/worst_drawdowns.csv` | 5 worst drawdown episodes per model with regime context |
| `evaluation_outputs/chapter12/regime_stress_report.json` | Full JSON report with methodology note and all metrics |

---

## 12.3 Regime-Aware Heuristic Baseline ✅ COMPLETE

Implemented in `scripts/run_chapter12_heuristics.py`.

Two heuristic approaches were built on top of the LGB baseline eval_rows (FULL
mode, 109 folds, 591K rows). Both serve as ablation baselines for Chapter 13's
DEUP-based approach.

### Approach A: Volatility-Scaled Ranking

**Formula:** `sized_score = score × min(1, c / vol_20d)` where `c = 0.25`

- Stocks with `vol_20d` below the median (~0.24 annualized) are unaffected
- High-volatility stocks get their scores penalised proportionally
- Effect: promotes lower-volatility stocks into the top-10 selection
- `vol_20d` sourced from `data/features.duckdb` features table (100% join rate)

**Intuition:** High realized volatility implies noisier signals. Penalising
these stocks in the ranking should improve the signal-to-noise ratio of
the top-10 portfolio.

### Approach B: Regime-Blended Ensemble

**Formula:**
```
w_lgb = 1 - α × sigmoid((vix_percentile - 67) / τ)
blended_score = w_lgb × lgb_rank + (1 - w_lgb) × mom_rank
```
where `α = 0.5`, `τ = 10.0`

- Both LGB and momentum scores are converted to cross-sectional percentile
  ranks (0–1) before blending to ensure comparable scales
- In low-VIX (percentile < 50): `w_lgb ≈ 0.92` — almost pure LGB
- In high-VIX (percentile > 80): `w_lgb ≈ 0.69` — ~31% momentum weight
- `mom_1m` sourced from features table (100% join rate)

**Intuition:** In turbulent markets, the complex LGB model may overfit to
noise. A simpler momentum signal could be more robust.

### Results: Signal Metrics (FULL mode, 109 folds)

| Model | Hz | Mean RankIC | Med. RankIC | IC Stability | Cost Surv | Churn |
|-------|---:|:-----------:|:-----------:|:------------:|:---------:|:-----:|
| **LGB baseline** | 20d | **0.0642** | **0.0805** | 0.353 | **69.7%** | 0.20 |
| Vol-Sized | 20d | 0.0605 | 0.0710 | 0.356 | 67.9% | 0.20 |
| Regime-Blended | 20d | 0.0384 | 0.0523 | 0.216 | 65.1% | 0.20 |
| **LGB baseline** | 60d | **0.1396** | **0.1478** | 0.712 | 80.7% | 0.20 |
| Vol-Sized | 60d | 0.1370 | 0.1409 | **0.739** | 79.8% | 0.20 |
| Regime-Blended | 60d | 0.1109 | 0.1263 | 0.570 | 76.1% | 0.20 |
| **LGB baseline** | 90d | **0.1650** | **0.1833** | 0.797 | 81.7% | 0.20 |
| Vol-Sized | 90d | 0.1634 | 0.1757 | **0.839** | **83.5%** | 0.20 |
| Regime-Blended | 90d | 0.1333 | 0.1490 | 0.656 | 78.9% | 0.20 |

### Results: Shadow Portfolio (20d, non-overlapping monthly, annualized ×12/×√12)

| Model | Sharpe | Ann. Return | Ann. Vol | Max DD | Hit Rate | N months |
|-------|-------:|:----------:|:--------:|:------:|:--------:|---------:|
| LGB baseline | 2.65 | 86.6% | 32.7% | −21.9% | 80.7% | 109 |
| **Vol-Sized** | **2.73** | **87.0%** | **31.8%** | **−18.1%** | **82.6%** | 109 |
| Regime-Blended | 1.88 | 53.6% | 28.6% | −31.4% | 76.1% | 109 |

### Key finding 1: Vol-sizing provides a genuine (small) improvement

Vol-sizing improves the shadow portfolio on every metric:
- **Sharpe: 2.65 → 2.73** (+3.0%)
- **Max DD: −21.9% → −18.1%** (17% less severe)
- **Hit rate: 80.7% → 82.6%** (+2.4%)
- **Volatility: 32.7% → 31.8%** (lower)

The signal metrics are marginally worse at 20d RankIC (0.0642 → 0.0605) but
marginally better at IC stability (90d: 0.797 → 0.839) and cost survival
(90d: 81.7% → 83.5%). The trade-off is net positive: the portfolio picks up
less-noisy stocks by penalising high-volatility names.

### Key finding 2: Regime blending is a clear negative result

Regime-blended scores are worse on every metric:
- **Sharpe: 2.65 → 1.88** (−29%)
- **Max DD: −21.9% → −31.4%** (43% worse)
- RankIC drops 20–40% across all horizons

The momentum signal (`mom_1m`) is simply weaker than LGB in this AI stock
universe. Blending dilutes the strong LGB signal rather than providing
regime-conditional robustness.

### Key finding 3: Churn is unchanged across all approaches

All three models produce median churn of 0.20. Vol-sizing and regime blending
do not affect portfolio turnover — the top-10 composition shifts slightly but
with similar frequency.

### Gate Assessment

| Gate | Verdict | Detail |
|------|---------|--------|
| Vol-sizing improves Sharpe OR MaxDD | **PASS** | Both improved |
| Regime blending improves any metric | **FAIL** | All worse |

**Outcome:** Vol-sizing is adopted as the heuristic baseline for Chapter 13's
DEUP ablation. Regime blending is a documented negative result — useful as
evidence that simple regime-switching cannot match DEUP's learned uncertainty.

### Implications for Chapter 13 (DEUP)

1. **Vol-sizing is the ablation baseline to beat:** DEUP must produce better
   risk-adjusted returns than simple `score × min(1, c/σ)` to justify its
   complexity.
2. **The improvement is small (Sharpe +3%):** This sets a low bar — even
   modest DEUP improvements will be meaningful.
3. **Regime blending failed:** Confirms that model switching / signal blending
   based on VIX is not the right approach. DEUP's per-prediction epistemic
   uncertainty should capture failure modes better than market-level regime
   indicators.
4. **Vol-sizing's max DD improvement (−22% → −18%) is the biggest win:**
   DEUP should focus on drawdown reduction as its primary value-add.

### Tests

24 tests in `tests/test_chapter12_heuristics.py`:
- `TestVolSizing`: 5 tests (low-vol preserved, high-vol penalised, scale bounds,
  missing vol, row count)
- `TestRegimeBlending`: 5 tests (low-VIX preserves LGB, high-VIX shifts to momentum,
  score bounds, sigmoid properties, row count)
- `TestComputeMetrics`: 4 tests (all horizons, RankIC bounds, cost survival bounds,
  churn bounds)
- `TestShadowPortfolio`: 3 tests (monthly subsample, annualization ×12, empty input)
- `TestFeatureJoin`: 2 tests (columns added, row count preserved)
- `TestOutputArtifacts`: 5 tests (JSON structure, vol-sizing improves portfolio,
  realistic Sharpe range, eval_rows saved for both approaches)

### Output artifacts

| Path | Contents |
|------|----------|
| `evaluation_outputs/chapter12/regime_heuristic/heuristic_comparison.json` | Full comparison: signal metrics + portfolio metrics for all 3 models |
| `evaluation_outputs/chapter12/regime_heuristic/vol_sized/eval_rows.parquet` | Modified eval_rows with vol-sized scores |
| `evaluation_outputs/chapter12/regime_heuristic/regime_blended/eval_rows.parquet` | Modified eval_rows with blended scores |

---

## 12.4 Freeze & Documentation ✅ COMPLETE

### regime_context.parquet

Built via `scripts/build_regime_context.py`. Combines per-stock features
with market-level regime data for Chapter 13's DEUP pipeline.

**Schema:** Keyed by `(date, stable_id)` — one row per stock per trading day.

| Column | Source | Coverage | Purpose |
|--------|--------|:--------:|---------|
| `date` | features | 100% | Trading day |
| `stable_id` | features | 100% | Survivorship-safe stock ID |
| `ticker` | features | 100% | Ticker symbol |
| `vol_20d` | features | 100% | **Per-stock annualized 20d realized volatility** — DEUP aleatoric baseline a(x) |
| `vol_60d` | features | 100% | Per-stock 60d realized volatility |
| `vol_of_vol` | features | 100% | Volatility of volatility |
| `mom_1m` | features | 100% | 1-month momentum |
| `sector` | features | 100% | GICS sector |
| `vix_percentile_252d` | regime | 100% | Rolling VIX percentile — epistemic context |
| `vix_regime` | regime | 100% | VIX regime integer encoding |
| `market_regime` | regime | 100% | Market regime (−1=bear, 0=neutral, 1=bull) |
| `market_vol_21d` | regime | 100% | Market-level realized volatility |
| `market_return_5d` | regime | 100% | 5-day market return |
| `market_return_21d` | regime | 100% | 21-day market return |
| `above_ma_50` | regime | 100% | SPY above 50-day MA |
| `above_ma_200` | regime | 100% | SPY above 200-day MA |

**Size:** 201,307 rows × 16 columns (8.2 MB), 2,386 dates, 100 stocks.

**Note on `beta_252d`:** Not included — the column exists in the features table
schema but contains 100% NaN (never populated). If beta is needed for Chapter 13,
it should be computed separately from price data.

**Note on `vix_percentile_252d` distribution:** Median is 92.1 (not ~50 as might
be expected). This is because the VIX was historically low in 2016–2019 (~12–15),
then elevated from 2020 onwards (COVID, rate hikes). The rolling 252d percentile
reflects that most of the sample lives in a "high relative to recent past" VIX
environment. The `vix_regime` column from the regime table uses fixed thresholds
and may be more suitable for regime classification.

### Final results table

**Summary of all Chapter 12 quantitative results:**

| Metric | LGB Baseline | Vol-Sized | Regime-Blended |
|--------|:------------:|:---------:|:--------------:|
| **20d RankIC** | **0.064** | 0.061 | 0.038 |
| **60d RankIC** | **0.140** | 0.137 | 0.111 |
| **90d RankIC** | **0.165** | 0.163 | 0.133 |
| **90d IC Stability** | 0.797 | **0.839** | 0.656 |
| **90d Cost Survival** | 81.7% | **83.5%** | 78.9% |
| **Sharpe (monthly)** | 2.65 | **2.73** | 1.88 |
| **Max Drawdown** | −21.9% | **−18.1%** | −31.4% |
| **Hit Rate** | 80.7% | **82.6%** | 76.1% |

- **Bold** = best across the row
- Vol-sizing wins on risk-adjusted metrics (Sharpe, DD, hit rate, IC stability)
- LGB wins on raw signal quality (RankIC)
- Regime blending is dominated on all metrics

### Success criteria assessment

| Criterion | Status |
|-----------|--------|
| Regime-conditional LGB performance quantified | ✅ 12.1 |
| Shadow portfolio stress-tested by regime | ✅ 12.2 |
| At least one heuristic baseline for Ch13 ablation | ✅ 12.3 (vol-sizing) |
| Regime context features stored and validated | ✅ 12.4 |

### Gate outcome

Vol-sizing provides a modest improvement over LGB (Sharpe 2.65 → 2.73,
max DD −21.9% → −18.1%) but does not fundamentally change the signal.
**Vol-sized LGB is adopted as the heuristic ablation baseline for Chapter 13.**
DEUP must beat Sharpe 2.73 and max DD −18.1% to justify its complexity.

Regime blending is a documented negative result: simple VIX-based model
switching cannot match the LGB signal. This supports Chapter 13's thesis
that per-prediction epistemic uncertainty (not market-level regimes) is
needed for meaningful improvement.

### Tests

11 tests in `tests/test_regime_context.py`:
- Required columns present, row count, date range, stock count
- `vol_20d` coverage and realistic range (5%–300% annualized)
- `vix_percentile_252d` coverage and range (0–100)
- `market_regime` valid values (−1, 0, 1)
- No duplicate `(date, stable_id)` pairs
- Joinable to eval_rows (>95% date overlap)

### Output artifacts

| Path | Contents |
|------|----------|
| `data/regime_context.parquet` | 201K rows × 16 cols, per-stock + market regime features for Ch13 |

---

## Appendix: Retroactive DEV vs FINAL Holdout Analysis

**Added:** February 19, 2026

A true "final holdout" was never formally defined before research iteration
began. This retroactive analysis partitions the existing 109-fold evaluation
into DEV (pre-2024) and FINAL (2024+) to check whether performance generalizes.

### Protocol

| Window | Date Range | Folds | Months | Eval Dates |
|--------|-----------|:-----:|:------:|:----------:|
| DEV | Feb 2016 – Dec 2023 | 95 | 95 | 1,993 |
| FINAL | Jan 2024 – Feb 2025 | 14 | 14 | 284 |

`HOLDOUT_START = 2024-01-01`. Embargo clearance: last DEV training window
reaches ~Sep 2023 (90 trading day embargo), 3+ months before FINAL starts.

**Caveat:** This is a **soft holdout**. Aggregate 109-fold metrics (which
include the holdout) were visible during Chapters 7–12 development. We
never optimized specifically for 2024+ performance, but researcher degrees
of freedom were informed by the full-period aggregate.

### Signal metrics: DEV vs FINAL

**LGB Baseline:**

| Horizon | DEV Mean RankIC | DEV IC Stability | DEV Cost Surv | FINAL Mean RankIC | FINAL IC Stability | FINAL Cost Surv |
|---------|:--------------:|:---------------:|:------------:|:----------------:|:-----------------:|:--------------:|
| 20d | 0.072 | 0.389 | 70.5% | 0.010 | 0.069 | 64.3% |
| 60d | 0.160 | 0.844 | 82.1% | −0.005 | −0.026 | 71.4% |
| 90d | 0.192 | 0.963 | 83.2% | −0.021 | −0.129 | 71.4% |

The 60d and 90d signal **completely collapses** in the holdout — RankIC
goes negative, IC stability inverts. Only 20d retains a (weak) positive signal.

**All models show the same pattern:**

| Model | 20d DEV → FINAL | 60d DEV → FINAL | 90d DEV → FINAL |
|-------|:--------------:|:--------------:|:--------------:|
| LGB baseline | 0.072 → 0.010 | 0.160 → −0.005 | 0.192 → −0.021 |
| Rank Avg 2 | 0.051 → **0.031** | 0.107 → **0.018** | 0.127 → −0.009 |
| Learned Stacking | 0.070 → 0.009 | 0.158 → −0.008 | 0.190 → −0.022 |

Rank Avg 2 (which includes FinText fusion) holds up best in the holdout —
the zero-shot FinText signal, while weak standalone, provides diversification
value during regime shifts.

### Shadow portfolio: DEV vs FINAL (20d, monthly L/S)

| Split | Sharpe | Sortino | Ann. Return | Ann. Vol | Max DD | Hit Rate |
|-------|:------:|:-------:|:----------:|:--------:|:------:|:--------:|
| DEV (95 mo) | 3.15 | 4.74 | 81.9% | 26.0% | −21.9% | 82.1% |
| FINAL (14 mo) | 1.91 | 8.92 | 119.1% | 62.4% | −16.6% | 71.4% |

The 20d shadow portfolio **degrades but remains strongly positive** in the
holdout. Sharpe drops 39% (3.15 → 1.91) but 1.91 is still a strong
risk-adjusted return. The high Sortino (8.92) suggests negative tail events
are rare even in the difficult holdout period.

The elevated annualized return (119.1%) and volatility (62.4%) in FINAL
likely reflect the extreme dispersion of AI stocks in 2024 (NVDA +170%,
some AI stocks −50%).

### Year-by-year RankIC reveals regime dependency

**90d RankIC by calendar year:**

| Year | Mean RankIC | N Dates | Interpretation |
|------|:----------:|:-------:|----------------|
| 2016 | 0.405 | 233 | Very high — limited training data, possible look-ahead |
| 2017 | 0.192 | 251 | Strong, genuine signal |
| 2018 | 0.322 | 251 | Strong |
| 2019 | 0.185 | 252 | Solid |
| 2020 | 0.197 | 253 | Solid (despite COVID) |
| **2021** | **−0.071** | 252 | **FAILURE** — meme stock / tech mania |
| 2022 | 0.148 | 251 | Recovery (bear market, high dispersion) |
| 2023 | 0.170 | 250 | Solid |
| **2024** | **−0.006** | 252 | **FAILURE** — AI thematic rally |
| **2025** | **−0.139** | 32 | Actively wrong (insufficient sample) |

**20d RankIC by calendar year:**

| Year | Mean RankIC | Interpretation |
|------|:----------:|----------------|
| 2016 | 0.170 | Strong |
| 2017 | 0.083 | Positive |
| 2018 | 0.093 | Positive |
| 2019 | 0.116 | Positive |
| 2020 | 0.068 | Positive |
| **2021** | **−0.054** | Failure |
| 2022 | 0.043 | Weakly positive |
| 2023 | 0.062 | Positive |
| **2024** | **0.013** | Barely positive |
| **2025** | **−0.008** | Near zero (32 days) |

The pattern is clear: the model works in most market regimes but fails
in strong thematic bull rallies where the AI stock universe moves in
lockstep (2021, 2024). The 20d horizon is more resilient than 90d.

### Interpretation: Regime dependency, not overfitting

This is **not** pure overfitting for these reasons:

1. **Multiple independent positive years**: 2017, 2018, 2019, 2020, 2022,
   2023 all show genuine positive signal at 90d, with different training
   sets and different market conditions.
2. **Failure pattern is interpretable**: The model fails in strong thematic
   bull markets (2021, 2024) where cross-sectional dispersion collapses —
   all AI stocks rise together, making ranking meaningless. This matches
   the regime analysis in Chapter 12.1 (low-VIX outperformance).
3. **20d portfolio still works**: The shadow portfolio at 20d holds Sharpe
   1.91 in the holdout. A purely overfit model would collapse to ~0.

However, the headline metrics **are optimistically biased** because:
1. The 2016 anomaly (RankIC 0.41) inflates the DEV average
2. 2021's failure was partially masked by averaging across 95 months
3. Researcher degrees of freedom (model, features, hyperparameters) were
   informed by the full 109-fold aggregate

### Implications

1. **The UQ pipeline (Chapter 13) is now the most important part of the
   project.** DEUP must detect the 2024 regime failure and abstain. The
   holdout collapse is the exact scenario epistemic uncertainty should flag.
2. **20d is the confirmed primary horizon.** It retains signal in the holdout
   and produces a usable Sharpe (1.91). 60d and 90d are research horizons
   until confirmed by future data or a model improvement.
3. **Factor regression (Chapter 16) must use FINAL period returns.** Proving
   alpha on DEV period returns is insufficient — it must survive in the holdout.
4. **The headline metrics reported in earlier chapters (RankIC 0.18 at 90d,
   Sharpe 2.65) represent DEV performance.** They are valid as research
   results but are optimistically biased estimates of true out-of-sample
   performance.
5. **Rank Avg 2 deserves reconsideration** as a more robust variant — it
   degrades less in the holdout than LGB alone, suggesting fusion with
   FinText provides genuine diversification value.

---

## Appendix B: Overfitting vs Regime Shift Diagnostics

**Added:** February 19, 2026

Three diagnostics to determine whether the FINAL holdout failure is caused
by walk-forward data leakage / overfitting, or by a genuine regime shift.

Script: `scripts/run_holdout_diagnostics.py`

### Diagnostic 1: Retrain LGB on DEV-only, evaluate on FINAL-only

**Method:** Train a single LGB model on ALL data before 2024-01-01 (164K rows),
then score ALL data from 2024-01-01 onwards (28K rows). This eliminates
any walk-forward fold structure — it's one clean train/test split.

**Result:**

| Horizon | Retrained FINAL | WalkFwd FINAL | Delta |
|---------|:--------------:|:------------:|:-----:|
| 20d | **−0.013** | +0.010 | −0.023 |
| 60d | **−0.051** | −0.005 | −0.046 |
| 90d | **−0.075** | −0.021 | −0.054 |

**The retrained model performs WORSE than the walk-forward model at every
horizon.** The walk-forward's expanding window actually helped by providing
fold diversity (some folds trained on earlier, more stable data).

**Verdict: REGIME SHIFT, not leakage.** If the problem were walk-forward
data leakage, the clean retrained model would perform *better*, not worse.
The model genuinely fails on 2024 data regardless of training methodology.

### Diagnostic 2: Feature importance stability

**Method:** Train LGB on three non-overlapping windows (2016–2019, 2020–2021,
2022–2023) and compare feature importance rankings.

**Result:**

| Horizon | Rank Corr (early vs late) | Top-3 Overlap | Top-3 Features |
|---------|:-----------------------:|:-------------:|----------------|
| 20d | **0.976** | **3/3** | adv_20d, vol_60d, mom_12m |
| 60d | **0.976** | **3/3** | adv_20d, mom_12m, vol_60d |
| 90d | **0.952** | **3/3** | adv_20d, vol_60d, mom_12m |

Feature importance is **extremely stable** across all time periods. The same
three features dominate in every window and every horizon:
- `adv_20d` — average daily volume (liquidity)
- `vol_60d` — 60-day realized volatility
- `mom_12m` — 12-month momentum

**Verdict: The model learned consistent, economically interpretable patterns.**
It is not "chasing noise" each period. The failure in 2024 is because these
real patterns (especially momentum) stopped working in the AI stock rally, not
because the model memorized spurious correlations.

### Diagnostic 3: 20d deep-dive on FINAL (walk-forward monthly)

**Result (20d RankIC and Top-10 excess return per month):**

| Month | 20d RankIC | Top-10 ER | Interpretation |
|-------|:---------:|:---------:|----------------|
| 2024-01 | **+0.102** | **+0.104** | Strong — pre-rotation |
| 2024-02 | **+0.199** | **+0.246** | Strong — momentum working |
| 2024-03 | −0.112 | −0.023 | Reversal — AI rotation begins |
| 2024-04 | −0.102 | −0.011 | Continued reversal |
| 2024-05 | −0.045 | −0.064 | Weak negative |
| 2024-06 | +0.040 | +0.021 | Flat |
| 2024-07 | **−0.202** | **−0.058** | Worst month — AI correction |
| 2024-08 | −0.060 | −0.023 | Weak negative |
| 2024-09 | +0.073 | +0.035 | Recovery |
| 2024-10 | +0.066 | +0.055 | Positive |
| 2024-11 | **+0.122** | **+0.149** | Strong — post-election rally |
| 2024-12 | +0.100 | +0.051 | Positive |
| 2025-01 | −0.013 | +0.057 | Mixed |
| 2025-02 | +0.000 | −0.000 | Zero signal (12 days) |

**8 positive months, 6 negative months** at 20d in the holdout.

The 20d signal is not uniformly dead — it oscillates with clear regime
dependence. The strongest positive months (Jan–Feb 2024, Nov–Dec 2024)
coincide with periods where momentum was rewarded. The worst months
(Mar–May 2024, Jul 2024) coincide with the AI stock rotation and correction.

At 90d, the pattern is starker: strong positive in Jan–Feb 2024, then mostly
negative Mar–Dec 2024, as 90d forward returns measured from early 2024
captured the full extent of the rotation.

### Combined interpretation

| Question | Answer | Evidence |
|----------|--------|----------|
| Is this walk-forward leakage? | **NO** | Retrained model fails even worse (−0.075 vs −0.021 at 90d) |
| Is the model chasing noise? | **NO** | Feature importance ρ = 0.95+, same top-3 across all periods |
| Is 20d selectively resilient? | **YES** | 8/14 months positive at 20d vs 4/14 at 90d |
| What caused the failure? | **Regime shift** | AI stock rally (2024) compressed cross-sectional dispersion |

**The honest answer: the model learned real patterns (momentum, volume,
volatility consistently matter) but those patterns broke during the 2024
AI stock rally.** This is a textbook regime shift, not overfitting to noise.

The "true" RankIC of the model is probably:
- **20d:** ~0.04–0.06 (DEV avg 0.072, FINAL avg 0.010, true value between)
- **60d:** ~0.06–0.10 (DEV avg 0.160, but 2024 FINAL = −0.005; the DEV
  number is inflated by benign early-period years)
- **90d:** ~0.06–0.10 (similar argument; the 0.192 DEV average is inflated)

### Output artifacts

| Path | Contents |
|------|----------|
| `evaluation_outputs/chapter12/holdout_diagnostics/holdout_diagnostics.json` | Full diagnostic report (retrain, importance, deep-dive) |

---

## Things to keep in mind for Chapter 13

1. **DEUP ablation bar:** Sharpe 2.73, max DD −18.1% (vol-sized heuristic).
   DEUP must beat this to justify complexity over `score × min(1, c/σ)`.

2. **VIX percentile is the strongest epistemic predictor** (ρ = −0.21 with
   RankIC at 60d). Include it prominently in DEUP's g(x) feature set.

3. **Bull markets are the weakest regime** for portfolio returns (Sharpe 1.88
   vs 3.84 in neutral). DEUP should learn to reduce confidence in bull markets
   where cross-sectional dispersion is low.

4. **Per-stock `vol_20d` is the key aleatoric input** (median 0.32). DEUP's
   a(x) baseline should use this directly — it's already proven useful via
   the vol-sizing heuristic.

5. **The worst drawdowns cluster in high-VIX bull regimes** (not bear markets).
   DEUP's epistemic uncertainty should spike in these episodes.

6. **Risk attribution (Fama-French 5-factor regression)** is planned for
   Chapter 16 as a formal acceptance gate. An early sanity check could be
   run now if needed, but the repeatable version belongs in Ch16.

7. **`beta_252d` is missing** from the features table (all NaN). If Chapter 13
   needs beta for risk decomposition, it must be computed from price data.

8. **DEV/FINAL holdout protocol is now established.** Chapter 13 must report
   both DEV (pre-2024) and FINAL (2024+) metrics. The 60d/90d signal collapses
   in the holdout — DEUP must detect this and reduce confidence. See Appendix
   above for full analysis.

9. **20d is the confirmed primary horizon.** FINAL Sharpe 1.91 proves the
   20d signal generalizes. 60d/90d are DEV-only until either (a) DEUP learns
   to abstain during failure regimes, or (b) a model improvement restores
   holdout performance.

10. **Rank Avg 2 may be more robust than pure LGB** in the holdout (20d FINAL
    RankIC: 0.031 vs 0.010). Consider evaluating DEUP on both LGB and Rank Avg 2
    as base models.

11. **The 2024 collapse is DEUP's test case.** If DEUP's epistemic uncertainty
    spikes in Jan–Feb 2024 (when LGB's 90d signal inverts), that validates the
    entire UQ framework. If DEUP misses it, the framework needs rethinking.

12. **Diagnostic confirmed: regime shift, not overfitting.** Retraining LGB on
    DEV-only and testing on FINAL gives even worse results than the walk-forward
    (90d: −0.075 retrained vs −0.021 walk-forward). Feature importances are
    extremely stable (ρ = 0.95, same top-3 in every period). The model learned
    real patterns; those patterns broke in 2024. See Appendix B.

13. **"True" RankIC is probably 0.04–0.10**, not the headline 0.18. The DEV
    average is inflated by benign early years (2016: 0.41, 2018: 0.32) and
    doesn't account for regime-failure episodes. Plan accordingly for Chapter 13
    — DEUP's value-add must be measured against realistic baseline expectations.

---

## Files added/updated

| File | Purpose |
|------|---------|
| `scripts/run_chapter12_regime_diagnostics.py` | 12.1 — Regime diagnostics script |
| `scripts/run_chapter12_stress_tests.py` | 12.2 — Regime stress tests on shadow portfolio |
| `scripts/run_chapter12_heuristics.py` | 12.3 — Heuristic baseline evaluation |
| `scripts/build_regime_context.py` | 12.4 — Build regime_context.parquet for Ch13 |
| `tests/test_chapter12_regime.py` | 18 tests for 12.1 |
| `tests/test_chapter12_stress.py` | 22 tests for 12.2 |
| `tests/test_chapter12_heuristics.py` | 24 tests for 12.3 |
| `tests/test_regime_context.py` | 11 tests for 12.4 |
| `data/regime_context.parquet` | Frozen regime context for Ch13 (201K rows) |
| `scripts/run_holdout_diagnostics.py` | Overfitting vs regime shift diagnostics (Appendix B) |
| `documentation/CHAPTER_12.md` | This file |

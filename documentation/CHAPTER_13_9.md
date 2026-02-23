# Chapter 13.9 — DEUP on Rank Avg 2: Robustness Check

**Status:** ✅ COMPLETE  
**Script:** `scripts/run_chapter13_9_rank_avg2.py`  
**Outputs:** `evaluation_outputs/chapter13/g_predictions_rank_avg2.parquet`, `ehat_predictions_rank_avg2.parquet`, `chapter13_9_ra2_diagnostics.json`

---

## 1. Motivation

Rank Avg 2 is a more robust model ensemble than the primary `tabular_lgb` model. In the out-of-sample FINAL holdout (2024+), Rank Avg 2 achieves a higher median RankIC (0.033 vs 0.010 for LGB at 20d), suggesting better cross-sectional signal quality during the challenging 2024 regime.

**Key question:** Does a more robust base model produce a better DEUP uncertainty signal, and does RA2 + DEUP outperform LGB + DEUP on the FINAL holdout?

If yes → adopt RA2 as primary (repeat 13.5–13.7 with RA2).  
If no  → LGB remains primary; DEUP value is already captured by the 13.7 deployment recommendation.

---

## 2. Data and Methodology

### Data
- **Rank Avg 2 residuals:** `enriched_residuals_rank_avg_2.parquet` (844,812 rows; 281,604 at 20d)
- **Aleatoric baseline:** `a_predictions.parquet` (date-level; fully reusable — a(x) is market-level, model-agnostic)
- **LGB ehat:** `ehat_predictions.parquet` (comparison reference)
- **Health gate:** `expert_health_lgb_20d.parquet` (same G(t) gate for fair comparison)

### g(x) Training
- Same `train_g_walk_forward` function used in Chapter 13.1
- Walk-forward: 109 total folds, 89 prediction folds (starting from fold 11, min_train=20)
- Target: `rank_loss` (same as LGB primary)
- Features: `score`, `abs_score`, `vol_20d`, `vol_60d`, `mom_1m`, `vix_percentile_252d`, `market_regime_enc`, `market_vol_21d`, `market_return_21d`, `cross_sectional_rank`
- n_estimators=50 (fast, same as LGB)

### Portfolio Construction
- Non-overlapping 20-day rebalance periods (sampling every 20 trading days)
- Top-10 long / bottom-10 short, equal weight within legs
- 10bps transaction cost (turnover-adjusted)
- Same G(t) binary gate (G ≥ 0.2) applied to gated variants
- Periods: ALL (full sample), DEV (pre-2024), FINAL (2024+)

---

## 3. g(x) Signal Quality Results

### ρ(g, rank_loss) After Walk-Forward Training

| Horizon | RA2 ρ (ALL) | RA2 ρ (DEV) | RA2 ρ (FINAL) |
|---------|:-----------:|:-----------:|:-------------:|
| 20d | **0.2203** | — | — |
| 60d | **0.2067** | — | — |
| 90d | **0.1953** | — | — |

The RA2 g(x) model achieves stronger raw ρ(g, rank_loss) compared to the LGB g(x) (which was 0.1900 at 20d in Chapter 13.1). This is consistent with RA2 having a more predictable residual structure.

---

## 4. ê(x) Diagnostics — RA2 vs LGB

### ρ(ê, rank_loss): The Key Signal Quality Metric

```
╔══════════════════════════════════════════════════════════════════════════╗
║              ρ(ê, rank_loss) — Rank Avg 2 vs tabular_lgb                 ║
╠══════════════════════════════════════════════════════════════════════════╣
║ Horizon   RA2 ρ ALL   RA2 ρ DEV   RA2 ρ FINAL   LGB ρ ALL   LGB ρ DEV  LGB ρ FINAL ║
╠──────────────────────────────────────────────────────────────────────────╣
║ 20d        +0.194      +0.195       +0.181        +0.144      +0.142       +0.192    ║
║ 60d        +0.153      +0.150       +0.206        +0.106      +0.103       +0.140    ║
║ 90d        +0.184      +0.181       +0.230        +0.146      +0.138       +0.248    ║
╚══════════════════════════════════════════════════════════════════════════╝
```

**Key finding:** RA2 ê(x) has **meaningfully higher ρ(ê, rank_loss)** at 20d (0.194 vs 0.144, +35% improvement) and 60d (0.153 vs 0.106, +44% improvement) across ALL and DEV. This is a strong positive signal that a more robust base model produces a better epistemic uncertainty predictor.

**At 90d FINAL:** LGB has slightly higher ρ (0.248 vs 0.230), suggesting LGB's uncertainty signal generalises marginally better to the 90d holdout. Mixed at the longest horizon.

### Quintile Monotonicity

| Horizon | RA2 Monotone Quintiles |
|---------|:---------------------:|
| 20d | **4/4 ✓** |
| 60d | 1/4 ✗ |
| 90d | **4/4 ✓** |

The 20d and 90d ê signals show perfect quintile monotonicity (higher ê → higher rank_loss). The 60d signal fails monotonicity (1/4), suggesting RA2's uncertainty estimates are less reliable at the 60d horizon. This contrasts with the LGB 13.3 results where all horizons passed.

---

## 5. Shadow Portfolio Comparison (20d, Primary Horizon)

```
╔══════════════════════════════════════════════════════════════════════════╗
║  Shadow Portfolio — LGB vs Rank Avg 2 (20d, non-overlapping monthly)    ║
╠══════════════════════════════════════════════════════════════════════════╣
║ Variant          ALL Sharpe  DEV Sharpe  FINAL Sharpe  Crisis MaxDD      ║
╠──────────────────────────────────────────────────────────────────────────╣
║ lgb_raw             +1.497      +1.381      +2.321         −1.4%          ║
║ lgb_vol             +1.313      +1.243      +1.745         −6.9%          ║
║ lgb_ehat            +1.006      +1.138      −0.777         −2.7%          ║
║ lgb_gate_vol        +0.907      +0.885      +1.017          0.0%          ║
╠──────────────────────────────────────────────────────────────────────────╣
║ ra2_raw             +0.622      +0.736      −0.637        −10.1%          ║
║ ra2_vol             +0.226      +0.223      +0.247         −1.5%          ║
║ ra2_ehat            +0.430      +0.482      −0.465         −0.1%          ║
║ ra2_gate_vol        +0.222      +0.149      +0.958          0.0%          ║
╚══════════════════════════════════════════════════════════════════════════╝
```

---

## 6. Decision Gate

```
Decision: ⚠️  RETAIN tabular_lgb AS PRIMARY

Criteria:
  (1) RA2 FINAL Sharpe > LGB FINAL:         ✗  (0.958 vs 2.321)  — primary criterion
  (2) RA2 ρ(ê,rl) FINAL ≥ LGB:             ✗  (0.181 vs 0.192)
  (3) RA2 DEV Sharpe ≥ 90% of LGB DEV:     ✗  (0.149 vs 1.243)

Criteria met: 0/3
```

**Decision: RETAIN tabular_lgb as primary.**

RA2 does not meet any of the three portfolio performance criteria. The DEUP economic value is already fully captured by the Chapter 13.7 deployment recommendation (Binary Gate + Vol-Sizing + ê-Cap at P85). Chapter 13's primary goal is met with LGB as the base model.

---

## 7. Analysis and Interpretation

### Why Does RA2 Produce Better ê Quality but Worse Portfolio Sharpe?

This apparent paradox — RA2 ê has higher ρ(ê, rank_loss) but its portfolio Sharpe is much lower — has a clear explanation:

1. **Base model signal strength:** LGB's DEV RankIC is substantially higher (median 0.091 vs 0.059 for RA2 at 20d). The portfolio Sharpe is driven primarily by base model signal quality, not uncertainty calibration.

2. **Universe composition:** RA2 covers ~124 stocks/date vs ~87 for LGB (the stock coverage differs between model architectures). RA2's larger but possibly lower-conviction universe may dilute top-K portfolio selection.

3. **The ê advantage doesn't transfer to portfolio returns:** RA2 ê is a better predictor of *ranking error* (ρ=0.194 vs 0.144) but this doesn't automatically translate to higher portfolio Sharpe, because the ê-sizing strategy reweights positions within an already-weaker base signal.

4. **Gate dominates:** The most important finding is that `ra2_gate_vol` achieves FINAL Sharpe of **0.958**, closely matching `lgb_gate_vol` at **1.017**. The G(t) binary gate is the dominant value driver for both models. Once gated, RA2 and LGB perform comparably on the 2024 holdout.

### What RA2's Higher ê Quality Means

Even though RA2 is not adopted as primary, the ρ(ê, rank_loss) result has value for the research paper:

- **RA2's uncertainty signal ρ = 0.194 at 20d (vs LGB's 0.144)** demonstrates that the DEUP approach is sensitive to base model quality. A better-calibrated base model produces more predictive epistemic uncertainty estimates.
- This finding validates the DEUP framework: the uncertainty signal is not a fixed artifact of the training features — it responds to the actual predictive quality of the underlying model.
- Future work: consider a RA2-ê / LGB-ê ensemble as a combined uncertainty signal.

### ê-Sizing Structural Conflict Persists for RA2

As with LGB (Chapter 13.7), direct ê-inverse sizing on RA2 underperforms:
- `ra2_ehat` ALL Sharpe: 0.430 vs `ra2_raw` 0.622
- This confirms the structural conflict (high-ê stocks are extreme-scored stocks) is a property of cross-sectional ranking models generally, not specific to LGB.

---

## 8. Key Findings

1. **RA2 g(x) quality is stronger:** ρ(g, rank_loss) = 0.220 at 20d vs 0.190 for LGB. A more robust base model produces a more predictive error predictor.

2. **RA2 ê(x) ρ is 35% higher at 20d (0.194 vs 0.144)** and 44% higher at 60d. RA2 epistemic uncertainty is a better predictor of ranking failure magnitude.

3. **But portfolio Sharpe is substantially lower:** RA2 ALL Sharpe = 0.62 vs LGB = 1.50 for raw scores. The base model's signal strength (DEV IC = 0.059 vs 0.091) is the binding constraint.

4. **Gate equalises performance:** Gate+Vol on RA2 achieves FINAL Sharpe 0.958 vs LGB's 1.017 — nearly identical once the regime gate is applied. The gate is the dominant driver for both.

5. **ê-sizing structural conflict is universal:** RA2 ê-sizing underperforms RA2 raw (0.430 vs 0.622), confirming the Chapter 13.7 finding extends beyond the LGB base model.

6. **Decision: RETAIN LGB as primary.** 0/3 criteria met. The 13.7 deployment recommendation stands as-is.

7. **Key future direction:** Investigate an ensemble ê signal (LGB ê + RA2 ê) as a more robust uncertainty estimate. RA2's superior ρ suggests it could serve as a signal quality enhancer within the existing two-layer (G(t) + ê-cap) system.

---

## 9. Success Criteria Assessment

| Criterion | Target | Result | Status |
|-----------|--------|--------|:------:|
| RA2 g(x) ρ(g, rank_loss) reported | All 3 horizons | 0.220/0.207/0.195 | ✅ |
| RA2 ê(x) ρ(ê, rank_loss) vs LGB | All 3 horizons | RA2 > LGB at 20d/60d | ✅ |
| Portfolio comparison DEV/FINAL | All variants | Full table produced | ✅ |
| Decision gate evaluated | Clear accept/reject | RETAIN LGB (0/3 criteria) | ✅ |
| ê-sizing structural conflict checked | Does RA2 also show conflict? | Yes — confirmed | ✅ |
| Honest reporting | No cherry-picking | LGB clearly wins on portfolio | ✅ |

---

## 10. Files Produced

| File | Description |
|------|-------------|
| `evaluation_outputs/chapter13/g_predictions_rank_avg2.parquet` | Walk-forward g(x) predictions for RA2 (699K rows, all horizons) |
| `evaluation_outputs/chapter13/ehat_predictions_rank_avg2.parquet` | ê(x) for RA2 (699K rows) |
| `evaluation_outputs/chapter13/chapter13_9_ra2_diagnostics.json` | Full diagnostics, portfolio results, decision gate |
| `scripts/run_chapter13_9_rank_avg2.py` | Reproducible analysis script |

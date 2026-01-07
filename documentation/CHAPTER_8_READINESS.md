# Chapter 8: Kronos Integration - Readiness Check

**Date:** January 7, 2026  
**Status:** 🟢 READY TO START

---

## Prerequisites Status

| Requirement | Status | Verification |
|-------------|--------|--------------|
| **Chapter 6 Frozen** | ✅ COMPLETE | Baseline floor frozen at `evaluation_outputs/chapter6_closure_real/` |
| **Chapter 7 Frozen** | ✅ COMPLETE | ML baseline: 0.1009/0.1275/0.1808 (20d/60d/90d) |
| **DuckDB Feature Store** | ✅ COMPLETE | 52 columns, 201,307 rows |
| **OHLCV Data** | ✅ COMPLETE | 100 tickers, 2016-2025, split-adjusted |
| **Tests Passing** | ✅ COMPLETE | 429/429 tests pass |
| **Batch 5 Fundamentals** | ✅ COMPLETE | Stepwise behavior validated |

---

## Frozen Baseline Floor (REAL Data)

**Source:** `evaluation_outputs/chapter6_closure_real/BASELINE_FLOOR.json`

### Factor Baselines (Gate 1: RankIC ≥ 0.02)

| Horizon | Best Factor | Median RankIC | Kronos Target |
|---------|-------------|---------------|---------------|
| 20d | `mom_12m_monthly` | 0.0283 | Must beat 0.02 |
| 60d | `momentum_composite_monthly` | 0.0392 | Must beat 0.02 |
| 90d | `momentum_composite_monthly` | 0.0169 | Must beat 0.02 |

### ML Baseline (Gate 2: RankIC ≥ 0.05)

| Horizon | tabular_lgb | Kronos Target |
|---------|-------------|---------------|
| 20d | 0.1009 | ≥ 0.08 (approach) |
| 60d | 0.1275 | ≥ 0.10 (approach) |
| 90d | 0.1808 | ≥ 0.15 (approach) |

---

## Data Available for Kronos

### OHLCV (Primary Input)
- **Source:** DuckDB `features` table (via `prices_df` from FMP)
- **Columns:** date, ticker, open, high, low, close, volume
- **Rows:** ~200K daily bars
- **Lookback:** Up to 252 trading days per sequence
- **Split-adjusted:** ✅ Yes (with auto-normalize)

### Additional Context (Optional)
- **Events:** days_to_earnings, days_since_earnings, in_pead_window
- **Regime:** vix_level, vix_regime, market_regime
- **Fundamentals:** sector, margin z-scores (if fusion needed)

---

## What Kronos Needs

### Model Components (To Implement)
1. **`src/models/kronos_adapter.py`** - Model wrapper
   - Load pre-trained Kronos weights
   - ReVIN normalization (rolling mean/std)
   - Horizon-specific prediction heads

2. **`src/pipelines/kronos_pipeline.py`** - Inference pipeline
   - Prepare OHLCV sequences (252-day lookback)
   - Batch inference for all tickers per date
   - Cache embeddings (optional, for fusion)

3. **`scripts/run_chapter8_kronos.py`** - Evaluation script
   - Use frozen Chapter 6 walk-forward splitter
   - Generate Kronos scores per fold
   - Compare vs tabular_lgb baseline

### Model Source
**Options:**
1. **HuggingFace:** Check for "kronos" or "k-line" models
2. **TimesNet/PatchTST:** Alternative TSFM models
3. **Custom:** Train horizon-specific heads on our data

---

## Acceptance Criteria

### Gate 1: Zero-Shot Performance
- [ ] Kronos RankIC ≥ 0.02 (factor baseline)
- [ ] Independent signal (correlation with mom_12m < 0.5)

### Gate 2: ML Comparison
- [ ] Kronos RankIC ≥ 0.05 (ML gate)
- [ ] Approach tabular_lgb: ≥ 0.08/0.10/0.15 (20d/60d/90d)

### Gate 3: Practical Viability
- [ ] Churn ≤ 0.30
- [ ] Cost survival ≥ 30% positive folds
- [ ] Stable across VIX regimes

---

## Files to Create

```
src/models/
├── __init__.py
├── kronos_adapter.py      # Model wrapper + inference
├── kronos_normalizer.py   # ReVIN normalization
└── kronos_heads.py        # Horizon-specific heads

scripts/
├── run_chapter8_kronos.py # Main evaluation script
└── test_kronos_smoke.py   # Quick sanity check

tests/
└── test_kronos_adapter.py # Unit tests
```

---

## Commands to Start

```bash
# Step 1: Create model directory
mkdir -p src/models
touch src/models/__init__.py

# Step 2: Verify DuckDB is ready
python -c "
import duckdb
con = duckdb.connect('data/features.duckdb', read_only=True)
print('Features:', con.execute('SELECT COUNT(*) FROM features').fetchone()[0])
print('Labels:', con.execute('SELECT COUNT(*) FROM labels').fetchone()[0])
con.close()
"

# Step 3: Verify tests pass
pytest tests/ -q --tb=no

# Step 4: Start implementing Kronos adapter
# (see CHAPTER_8_PLAN.md for details)
```

---

## Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Kronos model not available | Low | High | Use PatchTST or TimesNet |
| Poor zero-shot IC | Medium | Medium | Fine-tune horizon heads |
| Price normalization issues | Low | High | Use ReVIN (learnable) |
| Inference too slow | Low | Medium | Batch processing, cache |

---

## Next Steps

1. **Investigate Kronos model source** (HuggingFace, papers)
2. **Create `src/models/` directory structure**
3. **Implement `KronosAdapter` skeleton**
4. **Run zero-shot on single date to verify pipeline**
5. **If zero-shot RankIC > 0.01, proceed to FULL evaluation**

---

## References

- `CHAPTER_8_PLAN.md` - Detailed implementation plan
- `PRE_CHAPTER_8_CHECKLIST.md` - Feature store expansion (COMPLETE)
- `evaluation_outputs/chapter6_closure_real/BASELINE_FLOOR.json` - Target to beat
- `evaluation_outputs/chapter7_tabular_lgb_full/` - ML baseline to approach

---

**Bottom Line:** All prerequisites are met. Ready to implement Kronos adapter and run evaluation.


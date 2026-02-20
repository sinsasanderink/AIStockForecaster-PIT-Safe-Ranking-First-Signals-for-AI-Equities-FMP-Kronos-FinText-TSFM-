# Chapter 8: Your Next Actions

**Date:** January 8, 2026  
**Current Status:** Phase 3 COMPLETE ✅ (98% overall)  
**Next Phase:** Phase 4 (Comparison & Freeze)

---

## ✅ What's Been Completed

### Phase 1: Data Plumbing ✅
- Global trading calendar from DuckDB
- `PricesStore` for OHLCV access
- Tests passing (32 tests)

### Phase 2: Kronos Adapter ✅
- Batch-first `KronosAdapter`
- Per-ticker future timestamps (robustness fix)
- StubPredictor for testing
- Tests passing (20 tests: 19 passed, 1 skipped)

### Phase 3: Evaluation Integration ✅
- Walk-forward evaluation runner (`scripts/run_chapter8_kronos.py`)
- Kronos scoring function (matches `run_experiment()` contract)
- Leak tripwires (shuffle, lag)
- Momentum correlation check
- SMOKE test verified (19,110 eval rows, 3 folds)

---

## 🎯 Your Next Steps (Phase 4)

### Step 1: Run FULL Evaluation

You have two options:

**Option A: FULL with Stub (recommended for structure validation)**
```bash
cd /Users/ursinasanderink/Downloads/AI\ Stock\ Forecast
python scripts/run_chapter8_kronos.py --mode full --stub
```
- Runtime: ~2-4 hours
- Uses deterministic +2% predictor
- Validates full pipeline structure
- RankIC will be near-zero (expected for stub)

**Option B: FULL with Real Kronos (requires installation)**
```bash
# First, install Kronos (if not already done)
git clone https://github.com/shiyu-coder/Kronos
cd Kronos
pip install -r requirements.txt

# Then run FULL evaluation
cd /Users/ursinasanderink/Downloads/AI\ Stock\ Forecast
python scripts/run_chapter8_kronos.py --mode full

# Or with GPU:
python scripts/run_chapter8_kronos.py --mode full --device cuda
```
- Runtime: ~2-4 hours (faster on GPU)
- Real Kronos predictions
- RankIC should be meaningful
- Models download from HuggingFace on first run (~500MB)

---

### Step 2: Review FULL Results

After FULL completes, review the outputs:

```bash
# Check output directory
ls -lh evaluation_outputs/chapter8_kronos_full*/chapter8_kronos_full/

# Read summary report
cat evaluation_outputs/chapter8_kronos_full*/chapter8_kronos_full/REPORT_SUMMARY.md

# Check fold summaries
head -n 20 evaluation_outputs/chapter8_kronos_full*/chapter8_kronos_full/fold_summaries.csv

# Check leak tripwires
cat evaluation_outputs/chapter8_kronos_full*/chapter8_kronos_full/leak_tripwires.json
```

**Key metrics to check:**
- Median RankIC per horizon (20d/60d/90d)
- Leak tripwires: shuffle ≈ 0? lag collapsed?
- Churn: < 30%?
- Cost survival: positive?

---

### Step 3: Compare vs Frozen Baselines

**Manually (for now):**

1. Load frozen baseline floor:
   ```python
   import json
   from pathlib import Path
   
   baseline_floor = json.loads(
       Path("evaluation_outputs/chapter6_closure_real/BASELINE_FLOOR.json").read_text()
   )
   print("Factor baseline floor:")
   print("  20d:", baseline_floor["best_monthly"]["20d"])  # 0.0283
   print("  60d:", baseline_floor["best_monthly"]["60d"])  # 0.0392
   print("  90d:", baseline_floor["best_monthly"]["90d"])  # 0.0169
   ```

2. Load Ch7 LGB baseline:
   ```python
   import pandas as pd
   
   lgb_df = pd.read_csv(
       "evaluation_outputs/chapter7_tabular_lgb_full/fold_summaries.csv"
   )
   print("\nLGB baseline:")
   for h in [20, 60, 90]:
       median_ic = lgb_df[lgb_df["horizon"] == h]["rankic_median"].median()
       print(f"  {h}d: {median_ic:.4f}")
   # Expected: 0.1009 / 0.1275 / 0.1808
   ```

3. Load Kronos results:
   ```python
   kronos_df = pd.read_csv(
       "evaluation_outputs/chapter8_kronos_full*/chapter8_kronos_full/fold_summaries.csv"
   )
   print("\nKronos:")
   for h in [20, 60, 90]:
       median_ic = kronos_df[kronos_df["horizon"] == h]["rankic_median"].median()
       print(f"  {h}d: {median_ic:.4f}")
   ```

**Optional: Create comparison script (recommended for Phase 4)**
- Create `scripts/compare_kronos_vs_baselines.py`
- Automate the above comparison
- Generate markdown table

---

### Step 4: Evaluate Against Gates

**Gate 1 (Factor Baseline):**
- ✅ RankIC ≥ 0.02 for ≥2 horizons?
- ✅ Signal not redundant with momentum (corr < 0.5)?
- ✅ Churn ≤ 30%?

**Gate 2 (ML Baseline):**
- ✅ Any horizon RankIC ≥ 0.05?
- OR ✅ Within 0.03 of LGB baseline?

**Gate 3 (Practical):**
- ✅ Cost survival positive?
- ✅ Stable across regimes?

---

### Step 5: Decide Next Action

**If Kronos passes Gate 2:**
1. Write `documentation/CHAPTER_8_FREEZE.md`
2. Commit all artifacts
3. Tag release:
   ```bash
   git add .
   git commit -m "Chapter 8: Kronos frozen (Phase 4 complete)"
   git tag v0.8.0-kronos-frozen
   git push origin main --tags
   ```
4. Proceed to **Chapter 9: FinText-TSFM**

**If Kronos passes Gate 1 but not Gate 2:**
1. Consider fine-tuning:
   - Use Qlib for data preparation
   - Fine-tune on AI stock universe
   - Re-evaluate
2. Document fine-tuning plan in `documentation/CHAPTER_8_FINE_TUNE_PLAN.md`

**If Kronos fails Gate 1:**
1. Debug:
   - Check leak tripwires (shuffle/lag)
   - Review PricesStore coverage
   - Check for NaN scores
2. Iterate on adapter/scoring function
3. Re-run evaluation

---

## 📋 Quick Reference

### Commands

```bash
# SMOKE test (already done)
python scripts/run_chapter8_kronos.py --mode smoke --stub

# FULL test (next step)
python scripts/run_chapter8_kronos.py --mode full --stub

# FULL with real Kronos
python scripts/run_chapter8_kronos.py --mode full --device cuda

# Check results
ls -lh evaluation_outputs/chapter8_kronos_*/
cat evaluation_outputs/chapter8_kronos_full*/chapter8_kronos_full/REPORT_SUMMARY.md
```

### Success Criteria

| Gate | Metric | Threshold |
|------|--------|-----------|
| Gate 1 | RankIC | ≥ 0.02 for ≥2 horizons |
| Gate 1 | Momentum corr | < 0.5 |
| Gate 1 | Churn | ≤ 30% |
| Gate 2 | RankIC (any) | ≥ 0.05 |
| Gate 2 | vs LGB | Within 0.03 |
| Gate 3 | Cost survival | Positive |
| Gate 3 | Regime stability | No catastrophic regime failures |

### Baselines to Beat

| Horizon | Ch6 Factor Floor | Ch7 LGB Baseline |
|---------|------------------|------------------|
| 20d | 0.0283 | 0.1009 |
| 60d | 0.0392 | 0.1275 |
| 90d | 0.0169 | 0.1808 |

---

## 💡 Tips

1. **Runtime:** FULL evaluation takes ~2-4 hours. Consider running overnight or on GPU.

2. **Stub vs Real:** If you don't have Kronos installed yet, run FULL with `--stub` first to validate pipeline structure. Then install Kronos and re-run.

3. **Monitoring:** The script logs progress every 10 dates. Watch for:
   - Tickers scored per date (~98 expected)
   - Any warnings about insufficient history
   - Batch inference timing

4. **Debugging:** If RankIC is surprisingly low/high:
   - Check `leak_tripwires.json` (shuffle should be ~0, lag should collapse)
   - Check `per_date_metrics.csv` for anomalies
   - Review `evaluation_outputs/.../REPORT_SUMMARY.md`

5. **Storage:** FULL evaluation outputs are ~50-100MB. Ensure you have space.

---

## 📚 Documentation

**Key Documents:**
- `PHASE_3_COMPLETE_SUMMARY.md` - Phase 3 summary
- `documentation/CHAPTER_8_PHASE3_COMPLETE.md` - Detailed Phase 3 report
- `documentation/CHAPTER_8_IMPLEMENTATION_PLAN.md` - Original plan
- `documentation/CHAPTER_8_CRITICAL_FIXES.md` - All fixes applied
- `documentation/ROADMAP.md` - Overall project status
- `outline.ipynb` - Chapter 8 section (updated with institutional metrics)

**Need Help?**
- Review `scripts/run_chapter8_kronos.py --help`
- Check test files in `tests/test_kronos_*.py`
- Review SMOKE outputs in `evaluation_outputs/chapter8_kronos_smoke_stub/`

---

## ✨ Summary

You're 98% done with Chapter 8! Only Phase 4 (FULL evaluation + comparison) remains.

**Recommended next action:**
```bash
python scripts/run_chapter8_kronos.py --mode full --stub
```

Then review results, compare to baselines, and decide whether to freeze or iterate.

**Good luck! 🚀**


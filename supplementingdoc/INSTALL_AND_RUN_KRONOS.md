# How to Install and Run Kronos (Real Model)

**Date:** January 8, 2026  
**Purpose:** Complete Chapter 8 evaluation with real Kronos foundation model

---

## Quick Start (3 Steps)

```bash
# Step 1: Install Kronos
./INSTALL_KRONOS.sh

# Step 2: Set PYTHONPATH (choose one)
# Option A: Temporary (current session only)
export PYTHONPATH="$(pwd)/Kronos:$PYTHONPATH"

# Option B: Permanent (add to shell profile)
echo 'export PYTHONPATH="'$(pwd)'/Kronos:$PYTHONPATH"' >> ~/.bash_profile
source ~/.bash_profile

# Step 3: Run evaluation
./run_kronos_full.sh
```

---

## Detailed Instructions

### Step 1: Install Kronos

Run the installation script:

```bash
cd /Users/ursinasanderink/Downloads/AI\ Stock\ Forecast
./INSTALL_KRONOS.sh
```

**What this does:**
- Clones the Kronos repository from GitHub
- Installs required dependencies (torch, transformers, datasets, etc.)
- Shows you how to set PYTHONPATH

**Expected output:**
```
==========================================
Installing Kronos Foundation Model
==========================================

Step 1: Cloning Kronos repository...
Cloning into 'Kronos'...

Step 2: Installing Kronos dependencies...
Successfully installed ...

Step 3: Adding Kronos to Python path...
Kronos installed at: /Users/ursinasanderink/Downloads/AI Stock Forecast/Kronos
...
```

**Time:** ~2-5 minutes (depending on network speed)

---

### Step 2: Set PYTHONPATH

Kronos needs to be in your Python path. Choose one option:

**Option A: Temporary (lasts until you close the terminal)**
```bash
export PYTHONPATH="$(pwd)/Kronos:$PYTHONPATH"
```

**Option B: Permanent (persists across sessions)**
```bash
# Add to bash profile
echo 'export PYTHONPATH="'$(pwd)'/Kronos:$PYTHONPATH"' >> ~/.bash_profile
source ~/.bash_profile

# OR if using zsh (macOS Catalina+)
echo 'export PYTHONPATH="'$(pwd)'/Kronos:$PYTHONPATH"' >> ~/.zshrc
source ~/.zshrc
```

**Verify it worked:**
```bash
echo $PYTHONPATH
# Should show: /Users/ursinasanderink/Downloads/AI Stock Forecast/Kronos:...
```

---

### Step 3: Verify Installation (Recommended)

Before running the full evaluation, verify Kronos is working:

```bash
python scripts/test_kronos_installation.py
```

**Expected output:**
```
======================================================================
KRONOS INSTALLATION TEST
======================================================================

Test 1: Checking if Kronos can be imported...
✓ SUCCESS: Kronos imports successfully

Test 2: Loading Kronos tokenizer...
✓ SUCCESS: Tokenizer loaded from HuggingFace

Test 3: Loading Kronos model...
✓ SUCCESS: Model loaded from HuggingFace

Test 4: Creating Kronos predictor...
✓ SUCCESS: Predictor created

Test 5: Checking if KronosAdapter can be imported...
✓ SUCCESS: KronosAdapter imported

Test 6: Initializing KronosAdapter...
✓ SUCCESS: KronosAdapter initialized

======================================================================
✓ ALL TESTS PASSED
======================================================================

Kronos is ready to use!
```

**Note:** First run will download models from HuggingFace (~600MB total):
- Tokenizer: ~100MB
- Model: ~500MB

**If test fails:**
- Check PYTHONPATH is set correctly
- Ensure you have internet connection
- Verify sufficient disk space (~1GB free)

---

### Step 4: Run FULL Evaluation

**Option A: Using the wrapper script (recommended)**
```bash
./run_kronos_full.sh
```

**Option B: Direct command**
```bash
# Make sure PYTHONPATH is set first!
export PYTHONPATH="$(pwd)/Kronos:$PYTHONPATH"
python scripts/run_chapter8_kronos.py --mode full
```

**With GPU (if available):**
```bash
./run_kronos_full.sh --device cuda
```

**Expected runtime:** 2-4 hours (109 folds × 3 horizons)

**Progress indicators:**
- You'll see log messages every 10 dates
- Example: `[10/21] 2024-03-14: Scored 98 tickers`
- Fold completion: `✓ Fold fold_01, horizon 20d: 2058 total scores`

---

## Troubleshooting

### Issue 1: "Kronos not available" error

**Symptom:**
```
Kronos not available. Install from https://github.com/shiyu-coder/Kronos
ERROR - Kronos not installed.
```

**Solution:**
```bash
# Check if PYTHONPATH is set
echo $PYTHONPATH

# If not set or missing Kronos:
export PYTHONPATH="$(pwd)/Kronos:$PYTHONPATH"

# Verify it worked
python -c "from model import Kronos; print('SUCCESS')"
```

---

### Issue 2: "No module named 'model'" error

**Symptom:**
```python
ImportError: No module named 'model'
```

**Solution:**
```bash
# Check if Kronos directory exists
ls -l Kronos/

# If missing, install:
./INSTALL_KRONOS.sh

# Set PYTHONPATH
export PYTHONPATH="$(pwd)/Kronos:$PYTHONPATH"
```

---

### Issue 3: HuggingFace download fails

**Symptom:**
```
ConnectionError: Couldn't reach server at 'https://huggingface.co/...'
```

**Solution:**
- Check internet connection
- Try again (downloads can timeout)
- Set HuggingFace cache directory:
  ```bash
  export HF_HOME="$HOME/.cache/huggingface"
  ```

---

### Issue 4: Out of memory

**Symptom:**
```
RuntimeError: CUDA out of memory
# OR
MemoryError: Unable to allocate array
```

**Solution:**
```bash
# Use CPU instead of GPU
./run_kronos_full.sh --device cpu

# OR reduce batch size (requires code modification)
```

---

### Issue 5: Script doesn't complete

**Symptom:**
- Script runs for hours but doesn't finish
- No progress messages

**Solution:**
- Check terminal logs for errors
- Verify database exists: `ls -lh data/features.duckdb`
- Try SMOKE mode first to validate:
  ```bash
  python scripts/run_chapter8_kronos.py --mode smoke
  ```

---

## What Happens During Evaluation

1. **Initialization (~1 minute)**
   - Loads features from DuckDB
   - Loads Kronos model (if not cached)
   - Initializes trading calendar

2. **Fold Processing (~2-4 hours)**
   - 109 folds (monthly rebalance, 2016-2024)
   - Each fold: 3 horizons (20d, 60d, 90d)
   - Each date: ~98 tickers scored via batch inference

3. **Metrics Computation (~5 minutes)**
   - RankIC per date
   - Quintile spread
   - Churn analysis
   - Cost overlays

4. **Leak Tripwires (~2 minutes)**
   - Shuffle-within-date test
   - +1 trading-day lag test

5. **Output Generation (~1 minute)**
   - Stability reports
   - Summary markdown
   - All results saved to `evaluation_outputs/chapter8_kronos_full/`

---

## Expected Output

**Directory structure:**
```
evaluation_outputs/chapter8_kronos_full/chapter8_kronos_full/
├── eval_rows.parquet          # ~700,000+ rows
├── per_date_metrics.csv       # ~7,000+ rows
├── fold_summaries.csv         # 327 rows (109 folds × 3 horizons)
├── churn_series.csv           # Churn over time
├── cost_overlays.csv          # Net-of-cost metrics
├── leak_tripwires.json        # Negative control results
├── REPORT_SUMMARY.md          # Main report
└── chapter8_kronos_full/
    ├── tables/
    │   ├── ic_decay_stats.csv
    │   ├── churn_diagnostics.csv
    │   └── stability_scorecard.csv
    └── figures/
        ├── ic_decay.png
        ├── churn_timeseries.png
        └── churn_distribution.png
```

**Key file to review:**
```bash
cat evaluation_outputs/chapter8_kronos_full/chapter8_kronos_full/REPORT_SUMMARY.md
```

---

## Next Steps After Evaluation

1. **Review results:**
   ```bash
   cat evaluation_outputs/chapter8_kronos_full/chapter8_kronos_full/REPORT_SUMMARY.md
   head -n 20 evaluation_outputs/chapter8_kronos_full/chapter8_kronos_full/fold_summaries.csv
   ```

2. **Check RankIC:**
   - Look for median RankIC per horizon
   - Compare to frozen baselines:
     - Ch6 factor floor: 0.0283/0.0392/0.0169
     - Ch7 LGB: 0.1009/0.1275/0.1808

3. **Check leak tripwires:**
   ```bash
   cat evaluation_outputs/chapter8_kronos_full/chapter8_kronos_full/leak_tripwires.json
   ```
   - Shuffle RankIC should be ≈ 0
   - Lag RankIC should collapse

4. **Evaluate gates:**
   - Gate 1: RankIC ≥ 0.02 for ≥2 horizons?
   - Gate 2: RankIC ≥ 0.05 or within 0.03 of LGB?
   - Gate 3: Churn ≤ 30%, cost survival OK?

5. **Proceed to Phase 4:**
   - See `CHAPTER_8_NEXT_ACTIONS.md`
   - Create comparison script
   - Freeze if passing gates

---

## Alternative: Run with Stub First

If you want to validate the pipeline structure before waiting 2-4 hours:

```bash
# Run FULL with stub (fast, ~30 minutes)
python scripts/run_chapter8_kronos.py --mode full --stub

# Review structure
ls -lh evaluation_outputs/chapter8_kronos_full_stub/

# Then run with real Kronos
./run_kronos_full.sh
```

---

## Summary

**Installation:**
```bash
./INSTALL_KRONOS.sh
export PYTHONPATH="$(pwd)/Kronos:$PYTHONPATH"
python scripts/test_kronos_installation.py
```

**Run evaluation:**
```bash
./run_kronos_full.sh
```

**Review results:**
```bash
cat evaluation_outputs/chapter8_kronos_full/chapter8_kronos_full/REPORT_SUMMARY.md
```

**That's it!** You're now running the real Kronos foundation model. 🚀


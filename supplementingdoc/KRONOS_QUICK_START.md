# Kronos Quick Start

## 3 Commands to Run Real Kronos

```bash
# 1. Install
./INSTALL_KRONOS.sh

# 2. Set path (pick one line)
export PYTHONPATH="$(pwd)/Kronos:$PYTHONPATH"  # Temporary
echo 'export PYTHONPATH="'$(pwd)'/Kronos:$PYTHONPATH"' >> ~/.bash_profile && source ~/.bash_profile  # Permanent

# 3. Run
./run_kronos_full.sh
```

---

## Verify Installation (Optional but Recommended)

```bash
python scripts/test_kronos_installation.py
```

Expected: `✓ ALL TESTS PASSED`

---

## Common Issues

**"Kronos not available"**
```bash
export PYTHONPATH="$(pwd)/Kronos:$PYTHONPATH"
```

**"No module named 'model'"**
```bash
./INSTALL_KRONOS.sh
export PYTHONPATH="$(pwd)/Kronos:$PYTHONPATH"
```

**Downloads fail**
- Check internet connection
- Wait and retry (HuggingFace can timeout)

---

## After Evaluation Completes

```bash
# View results
cat evaluation_outputs/chapter8_kronos_full/chapter8_kronos_full/REPORT_SUMMARY.md

# Check fold summaries
head evaluation_outputs/chapter8_kronos_full/chapter8_kronos_full/fold_summaries.csv

# Compare to baselines (manual for now)
# Ch6: 0.0283/0.0392/0.0169
# Ch7: 0.1009/0.1275/0.1808
```

---

## Runtime

- **SMOKE** (3 folds): ~3 minutes
- **FULL** (109 folds): **2-4 hours**
- First run: +5-10 minutes (HuggingFace downloads)

---

## Full Documentation

See `INSTALL_AND_RUN_KRONOS.md` for detailed instructions and troubleshooting.


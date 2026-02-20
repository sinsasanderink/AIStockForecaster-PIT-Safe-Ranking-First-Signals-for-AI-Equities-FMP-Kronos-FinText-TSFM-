# Chapter 8: Kronos Integration Plan

**Status:** 🟢 READY TO START  
**Prerequisites:** ✅ ALL MET

---

## Prerequisites Check

| Requirement | Status | Details |
|-------------|--------|---------|
| Chapter 6 Frozen | ✅ | Baseline floor: 0.0283/0.0392/0.0169 (20d/60d/90d) |
| Chapter 7 Frozen | ✅ | ML baseline: 0.1009/0.1275/0.1808 (tabular_lgb) |
| DuckDB Feature Store | ✅ | 52 columns, 201K rows, 2016-2025 |
| OHLCV Data | ✅ | 100 tickers, split-adjusted, PIT-safe |
| Tests Passing | ✅ | 429/429 tests pass |
| Fundamentals | ✅ | Batch 5 Phase 1 complete |

---

## What is Kronos?

**Kronos** is the first open-source foundation model for financial K-lines (candlesticks), trained on 45+ global exchanges. [GitHub](https://github.com/shiyu-coder/Kronos) | [Paper (arXiv:2508.02739)](https://arxiv.org/html/2508.02739v1)

**Architecture:**
- **Tokenizer:** VQ-VAE-based hierarchical tokenization of OHLCV sequences
- **Predictor:** Decoder-only Transformer (GPT-like architecture)
- **Pre-training:** 45+ exchanges, multiple asset classes, billions of tokens
- **Input:** OHLCV sequences (lookback configurable, default 252 days)
- **Output:** Future OHLCV predictions (autoregressive generation)

**Key Properties:**
- Available on HuggingFace: `shiyu-coder/Kronos-tokenizer`, `shiyu-coder/Kronos-predictor`
- Supports multiple sampling for distribution estimation
- Can extract embeddings for fusion (Chapter 11)
- Includes fine-tuning infrastructure via Qlib

---

## Chapter 8 Deliverables

### 1. Kronos Model Adapter (Priority 1)
```python
# src/models/kronos_adapter.py
class KronosAdapter:
    """Adapter to run Kronos on our OHLCV sequences."""
    
    def __init__(self, model_path: str, device: str = "cpu"):
        self.model = load_kronos_model(model_path)
        self.normalizer = ReVINNormalizer()  # Rolling normalization
    
    def prepare_sequence(self, ohlcv_df: pd.DataFrame, lookback: int = 252) -> Tensor:
        """Convert OHLCV DataFrame to model input tensor."""
        # Normalize prices (rolling mean/std)
        # Handle missing data
        # Return tensor of shape (seq_len, 5)
    
    def forward(self, sequences: Tensor, horizon: int) -> Tensor:
        """Run inference, return predictions for horizon."""
        # Extract embeddings
        # Apply horizon-specific head
        # Return ranking scores
    
    def get_embeddings(self, sequences: Tensor) -> Tensor:
        """Extract intermediate embeddings for fusion."""
```

### 2. Inference Pipeline (Priority 2)
```python
# src/pipelines/kronos_pipeline.py
def run_kronos_inference(
    asof_date: date,
    tickers: List[str],
    horizons: List[int] = [20, 60, 90],
    lookback: int = 252,
    cache_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Run Kronos inference for all tickers as of a date.
    
    Returns DataFrame with columns:
    - date, ticker, horizon
    - kronos_score (raw model output)
    - kronos_rank (cross-sectional rank)
    - kronos_embedding (optional, for fusion)
    """
```

### 3. Evaluation Integration (Priority 3)
```python
# scripts/run_chapter8_kronos.py
def run_kronos_evaluation(mode: str = "SMOKE"):
    """
    Run Kronos through the frozen Chapter 6 evaluation pipeline.
    
    1. Load features from DuckDB
    2. Generate Kronos scores for each fold's dates
    3. Compute RankIC, churn, cost survival
    4. Compare vs tabular_lgb baseline (0.1009/0.1275/0.1808)
    """
```

### 4. FULL_MODE Execution + Comparison
```bash
# Run Kronos evaluation
python scripts/run_chapter8_kronos.py --mode FULL

# Output: evaluation_outputs/chapter8_kronos_full/
# - kronos_monthly/
# - kronos_quarterly/
# - KRONOS_REFERENCE.md (comparison vs baselines)
```

---

## Acceptance Criteria

### Gate 1: Zero-Shot Performance
- [ ] Kronos (zero-shot) median RankIC ≥ 0.02 (factor baseline)
- [ ] Kronos (zero-shot) provides independent signal (correlation with mom_12m < 0.5)

### Gate 2: ML Comparison
- [ ] Kronos median RankIC ≥ 0.05 (ML gate)
- [ ] Must approach or beat `tabular_lgb`:
  - 20d: ≥ 0.08 (vs 0.1009 baseline)
  - 60d: ≥ 0.10 (vs 0.1275 baseline)
  - 90d: ≥ 0.15 (vs 0.1808 baseline)

### Gate 3: Practical Viability
- [ ] Churn ≤ 0.30
- [ ] Cost survival ≥ 30% positive folds (60d/90d)
- [ ] Stable across VIX regimes (no catastrophic collapse)

---

## Implementation Strategy

### Phase 1: Zero-Shot (Week 1)
1. Set up Kronos environment (PyTorch, model weights)
2. Implement `KronosAdapter` with ReVIN normalization
3. Run zero-shot inference on test dates
4. Compute RankIC vs labels
5. **Decision point:** If zero-shot RankIC < 0.01, investigate model/data issues

### Phase 2: Evaluation Integration (Week 2)
1. Integrate with walk-forward pipeline
2. Run SMOKE mode (1 fold) to verify pipeline
3. Run FULL mode (all folds)
4. Generate stability reports
5. Compare vs frozen baselines

### Phase 3: Fine-tuning (Optional, Week 3)
1. If zero-shot underperforms:
   - Fine-tune horizon-specific heads
   - Use time-decay sample weighting
   - Per-fold training (no future leakage)
2. Verify fine-tuning improves IC by ≥ 0.01

---

## Technical Requirements

### Model Setup
```bash
# Dependencies already installed:
# - torch 2.0.1 ✅
# - transformers 4.57.1 ✅
# - einops 0.8.1 ✅
# - qlib 0.9.7 ✅ (for fine-tuning)

# Clone Kronos repo for reference
git clone https://github.com/shiyu-coder/Kronos.git /tmp/kronos_reference

# Models auto-download from HuggingFace on first use:
# - shiyu-coder/Kronos-tokenizer
# - shiyu-coder/Kronos-predictor
```

### Data Requirements
- OHLCV: Already in DuckDB (prices table)
- Lookback: 252 trading days (1 year)
- Sequence shape: (252, 5) per stock-date

### Compute Requirements
- CPU: Sufficient for inference (~100 stocks × ~2000 dates)
- GPU: Optional, ~5x speedup
- Memory: ~8GB for batch inference

---

## Known Risks & Mitigations

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Kronos pre-training leak | Medium | High | Use year-specific checkpoints, verify OOS dates |
| Poor zero-shot IC | Medium | Medium | Fine-tune horizon heads |
| Price normalization issues | Low | High | Use ReVIN (learnable normalization) |
| Inference too slow | Low | Medium | Batch processing, caching |
| Model not available | Low | High | Use alternative (e.g., PatchTST, TimesNet) |

---

## Optional Enhancements (Post-Chapter 8)

### Event Conditioning (Chapter 8.5)
- Add `days_to_earnings`, `days_since_earnings` as conditioning
- Test if earnings-aware Kronos improves IC
- Ablation: Kronos-only vs Kronos+events

### Fundamental Ablation (Chapter 10)
- Add fundamental features to Kronos embeddings
- Test if context improves predictions
- Compare: Kronos-only vs Kronos+fundamentals

---

## Files to Create

```
src/models/
├── __init__.py
├── kronos_adapter.py      # Model adapter
├── kronos_normalizer.py   # ReVIN normalization
└── kronos_heads.py        # Horizon-specific prediction heads

scripts/
├── run_chapter8_kronos.py # Main evaluation script
└── test_kronos_smoke.py   # Quick sanity check

tests/
└── test_kronos_adapter.py # Unit tests for adapter
```

---

## Commands to Run

```bash
# Step 1: Verify prerequisites
pytest tests/ -q --tb=no  # Should be 429 passed

# Step 2: Run Kronos smoke test
python scripts/run_chapter8_kronos.py --mode SMOKE

# Step 3: Run full evaluation
python scripts/run_chapter8_kronos.py --mode FULL

# Step 4: Compare vs baselines
python -c "
from pathlib import Path
import json

ch7 = json.loads(Path('evaluation_outputs/chapter7_tabular_lgb_full/BASELINE_FLOOR.json').read_text())
ch8 = json.loads(Path('evaluation_outputs/chapter8_kronos_full/fold_summaries.json').read_text())

print('Kronos vs tabular_lgb:')
for h in [20, 60, 90]:
    lgb = ch7['horizons'][str(h)]['median_rankic']
    kronos = ch8['horizons'][str(h)]['median_rankic']
    print(f'  {h}d: Kronos={kronos:.4f} vs LGB={lgb:.4f} (delta={kronos-lgb:+.4f})')
"
```

---

## Timeline

| Week | Focus | Deliverable |
|------|-------|-------------|
| 1 | Model setup + zero-shot | `KronosAdapter`, zero-shot RankIC |
| 2 | Evaluation integration | FULL mode results, comparison |
| 3 | Fine-tuning (if needed) | Improved IC, stability reports |
| 4 | Documentation + freeze | `CHAPTER_8_FREEZE.md` |

---

## Decision Points

1. **After zero-shot:** If RankIC < 0.01, investigate data/model issues before proceeding
2. **After FULL mode:** If Kronos significantly underperforms LGB, consider:
   - Fine-tuning
   - Alternative models (PatchTST, TimesNet)
   - Skip to Chapter 9 (FinText)
3. **Before freeze:** Verify cost survival and regime stability

---

## Next Action

**Start with:**
```bash
# Create model directory and adapter skeleton
mkdir -p src/models
touch src/models/__init__.py
touch src/models/kronos_adapter.py
```

Then implement `KronosAdapter` and run zero-shot on a single date to verify pipeline.


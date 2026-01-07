# Earnings Gap Microstructure Analysis

**Issue Raised:** December 30, 2025  
**Status:** Potential blind spot requiring diagnostic analysis

---

## The Problem

### Timeline Example
```
Jan 31 (Month-end): Generate predictions, rebalance portfolio
Feb 1:  Market opens with our new positions
Feb 2:  Stock A reports earnings → 15% gap move
...
Feb 28: Measure 20d forward return (includes the Feb 2 gap)
```

**The Question:** Is the Feb 2 earnings gap:
1. **Signal** (our model predicted the surprise direction via earnings features), OR
2. **Noise** (we got lucky/unlucky with timing)?

---

## Current Mitigation (Chapter 5.4)

We **DO** have earnings event features in the model:

| Feature | Purpose |
|---------|---------|
| `days_to_earnings` | Model knows earnings are imminent |
| `days_since_earnings` | Model knows we're in PEAD window |
| `last_surprise_pct` | Model sees historical surprise direction |
| `avg_surprise_4q` | Model sees consistency of beats/misses |
| `surprise_streak` | Model sees momentum in surprises |
| `in_pead_window` | Model knows if we're in 63-day drift period |
| `reports_bmo` | Model knows typical announcement timing |

**Theory:** If the model learns that "stocks with earnings in 2 days are unpredictable", it will down-rank them, avoiding the lottery.

**Risk:** If the model does NOT learn this effectively, earnings gaps become noise in our labels.

---

## Diagnostic Analysis Required

### 1. IC Stratification by Days-to-Earnings

**Question:** Does our RankIC drop for stocks with imminent earnings?

**Test:**
```python
# Stratify eval_rows by days_to_earnings
buckets = {
    "imminent": (0, 5),      # Earnings within 5 days
    "near": (6, 21),         # Earnings within 1 month
    "distant": (22, 90),     # Earnings 1-3 months out
    "no_earnings": (None, None)  # No scheduled earnings
}

for bucket_name, (min_days, max_days) in buckets.items():
    bucket_rows = filter_by_days_to_earnings(eval_rows, min_days, max_days)
    bucket_ic = compute_rankic(bucket_rows)
    print(f"{bucket_name}: IC = {bucket_ic:.4f}, N = {len(bucket_rows)}")
```

**Expected Result (if we're handling it well):**
- IC should be relatively stable across buckets
- OR IC should be lower for "imminent" but we should be ranking fewer stocks in that bucket (model avoiding them)

**Bad Result (if it's noise):**
- IC is significantly lower for "imminent" bucket
- Model is NOT avoiding imminent-earnings stocks

---

### 2. Earnings Gap Attribution

**Question:** What % of our forward return variance is explained by earnings gaps?

**Test:**
```python
# For each stock-date with an earnings event within the 20d horizon:
earnings_contribution = (close[earnings_date+1] / close[earnings_date-1]) - 1
ex_earnings_return = total_return - earnings_contribution

# Compare:
ic_full = spearmanr(predictions, total_return)
ic_ex_earnings = spearmanr(predictions, ex_earnings_return)

print(f"IC (full): {ic_full:.4f}")
print(f"IC (ex-earnings): {ic_ex_earnings:.4f}")
print(f"Earnings contamination: {ic_full - ic_ex_earnings:.4f}")
```

**Interpretation:**
- If `ic_ex_earnings > ic_full`: Earnings gaps are adding noise
- If `ic_ex_earnings ≈ ic_full`: We're already handling it (or earnings are small)
- If `ic_ex_earnings < ic_full`: We're successfully predicting earnings gap direction

---

### 3. Feature Importance Check

**Question:** Are earnings timing features actually being used?

**Test:**
```python
# For tabular_lgb baseline:
feature_importance = model.feature_importances_
earnings_features = [
    "days_to_earnings", "days_since_earnings", "last_surprise_pct",
    "avg_surprise_4q", "surprise_streak", "in_pead_window"
]

for feat in earnings_features:
    importance = feature_importance[feat]
    print(f"{feat}: {importance:.4f}")
```

**Expected Result:**
- `days_to_earnings` and `last_surprise_pct` should have non-trivial importance
- If importance is near-zero, the model is ignoring these signals

---

## Potential Solutions (if diagnostics show a problem)

### Option A: Earnings-Aware Label Filtering (Conservative)

**Approach:** Exclude or down-weight labels where earnings occur within the horizon

```python
def filter_earnings_contaminated_labels(labels_df, events_df, horizon_days=20):
    """
    Remove labels where an earnings event occurs within the forward horizon.
    """
    contaminated = labels_df.merge(
        events_df[events_df["event_type"] == "EARNINGS"],
        on=["ticker", "as_of_date"],
        how="inner"
    )
    contaminated = contaminated[
        contaminated["days_to_earnings"] <= horizon_days
    ]
    
    clean_labels = labels_df[~labels_df.index.isin(contaminated.index)]
    return clean_labels
```

**Pros:**
- Removes noise from unpredictable earnings gaps
- Makes IC more representative of "pure alpha"

**Cons:**
- Loses a lot of data (earnings are frequent for AI stocks)
- Throws away potential signal (if we CAN predict surprise direction)

---

### Option B: Earnings-Adjusted Labels (Robustness Check)

**Approach:** Replace earnings gap with market return, measure IC on adjusted labels

```python
def create_earnings_adjusted_labels(labels_df, prices_df, events_df, benchmark_df):
    """
    For labels with earnings gaps, replace gap with benchmark return.
    """
    adjusted_labels = labels_df.copy()
    
    for idx, row in labels_df.iterrows():
        # Check if earnings occurred in forward horizon
        earnings_events = get_earnings_in_horizon(
            row["ticker"], row["as_of_date"], row["horizon"]
        )
        
        if earnings_events:
            for event_date in earnings_events:
                # Replace (event_date-1 to event_date+1) return with benchmark
                stock_gap = compute_gap_return(prices_df, row["ticker"], event_date)
                benchmark_gap = compute_gap_return(benchmark_df, "QQQ", event_date)
                
                # Adjust label
                adjusted_labels.loc[idx, "excess_return"] -= (stock_gap - benchmark_gap)
    
    return adjusted_labels
```

**Usage:**
- Run evaluation with both regular and earnings-adjusted labels
- Compare IC: if IC improves with adjustment, earnings gaps were noise
- If IC stays the same, we're already robust to earnings timing

**Pros:**
- Keeps all data
- Provides diagnostic information
- Can run alongside main evaluation

**Cons:**
- More complex labeling pipeline
- Requires precise earnings date-time tracking

---

### Option C: Ensemble with Earnings-Aware Submodel (Advanced)

**Approach:** Train two submodels:
1. **Main model:** All features, all labels
2. **Earnings-immune model:** Excludes stocks with imminent earnings (<5 days)

Then ensemble based on `days_to_earnings`:
```python
if days_to_earnings < 5:
    weight_main = 0.3
    weight_immune = 0.7
else:
    weight_main = 0.8
    weight_immune = 0.2

final_score = weight_main * main_pred + weight_immune * immune_pred
```

**Pros:**
- Keeps all data for main model (learns earnings patterns)
- Reduces exposure to unpredictable gaps via earnings-immune model

**Cons:**
- More complex training pipeline
- Requires careful hyperparameter tuning

---

## Recommendation

### Phase 1: Diagnostic (Immediate)
Add to `src/evaluation/diagnostics.py`:
```python
def earnings_gap_analysis(
    eval_rows: pd.DataFrame,
    events_df: pd.DataFrame,
    output_dir: Path
) -> Dict[str, float]:
    """
    Analyze impact of earnings gaps on IC.
    
    Returns:
        - ic_by_days_to_earnings: IC stratified by proximity to earnings
        - earnings_contamination: IC difference with/without gaps
        - feature_importance: Importance of earnings features
    """
    # Implementation TBD
    pass
```

Run this as part of Chapter 7 baseline evaluation to quantify the issue.

### Phase 2: Mitigation (If Needed)
Based on Phase 1 results:
- **If IC drops >0.01 for imminent-earnings stocks AND model isn't avoiding them:**
  - Implement Option B (earnings-adjusted labels) as robustness check
  - Consider Option A (filtering) for high-frequency trading use cases

- **If IC is stable OR model successfully avoids imminent-earnings stocks:**
  - No action needed, document as "handled via features"

---

## Current Status

**Chapter 5:** ✅ Earnings features implemented  
**Chapter 6:** ✅ Evaluation pipeline supports filtering/adjustment  
**Chapter 7:** ✅ ML baseline uses earnings features  
**Diagnostics:** ⏳ TODO (add earnings_gap_analysis to evaluation reports)

**Next Action:** Add diagnostic code to `src/evaluation/diagnostics.py` and run on Chapter 7 frozen artifacts.

---

## Related Issues

### Similar Microstructure Concerns

1. **Corporate Actions (Splits, Spinoffs):**
   - Handled via FMP split-adjusted prices
   - Extreme returns (>50%) flagged in tests

2. **Halts and Delisting:**
   - Handled via survivorship-safe universe (stable_id)
   - Missing data explicitly modeled

3. **Month-End Positioning:**
   - Rebalancing on 1st avoids month-end window dressing
   - But could still be affected by rebalancing flows

4. **Intraday Timing:**
   - All predictions use prior-day close
   - Labels use close-to-close returns (no intraday leakage)

---

## Documentation Updates

**Files to Update:**
1. `PROJECT_DOCUMENTATION.md` (Chapter 5.4: Add earnings gap caveat)
2. `CHAPTER_7_FREEZE.md` (Add to "Known Limitations" section)
3. `src/evaluation/diagnostics.py` (Add earnings_gap_analysis function)

**Acceptance Criteria:**
- Diagnostic runs on all Chapter 7+ evaluations
- If contamination > 0.01 IC, implement mitigation
- Document findings in BASELINE_REFERENCE.md


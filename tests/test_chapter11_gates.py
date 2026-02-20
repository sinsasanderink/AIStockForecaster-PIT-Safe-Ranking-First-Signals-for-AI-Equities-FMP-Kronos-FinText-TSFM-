import numpy as np
import pandas as pd

from scripts.evaluate_fusion_gates import compute_metrics, evaluate_gates


def _make_eval_rows(seed: int, signal_strength: float) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=8, freq="MS")
    tickers = [f"T{i:02d}" for i in range(30)]

    rows = []
    fold_idx = 0
    for horizon in (20, 60, 90):
        for i, d in enumerate(dates):
            fold_id = f"fold_{(i // 3) + 1:02d}"
            for t in tickers:
                raw = rng.normal(0, 1)
                score = raw + rng.normal(0, 0.1)
                ret = signal_strength * score + rng.normal(0, 0.5)
                rows.append(
                    {
                        "as_of_date": d,
                        "ticker": t,
                        "stable_id": t,
                        "fold_id": fold_id,
                        "horizon": horizon,
                        "score": score,
                        "excess_return": ret,
                    }
                )
    return pd.DataFrame(rows)


def test_compute_metrics_shape():
    eval_rows = _make_eval_rows(seed=1, signal_strength=0.2)
    metrics = compute_metrics(eval_rows)
    assert set(metrics.keys()) == {20, 60, 90}
    assert "mean_rankic" in metrics[20]
    assert "median_churn" in metrics[20]


def test_compute_metrics_has_ic_stability_and_cost_survival():
    eval_rows = _make_eval_rows(seed=10, signal_strength=0.3)
    metrics = compute_metrics(eval_rows)
    for h in (20, 60, 90):
        assert "ic_stability" in metrics[h], f"Missing ic_stability for {h}d"
        assert "cost_survival" in metrics[h], f"Missing cost_survival for {h}d"
        assert not np.isnan(metrics[h]["ic_stability"])
        assert not np.isnan(metrics[h]["cost_survival"])
        assert 0 <= metrics[h]["cost_survival"] <= 1.0


def test_evaluate_gates_pass_when_fusion_is_better():
    fusion = compute_metrics(_make_eval_rows(seed=2, signal_strength=0.35))
    lgb = compute_metrics(_make_eval_rows(seed=3, signal_strength=0.20))
    fintext = compute_metrics(_make_eval_rows(seed=4, signal_strength=0.10))
    sentiment = compute_metrics(_make_eval_rows(seed=5, signal_strength=0.03))

    gates = evaluate_gates(
        fusion_metrics=fusion,
        lgb_metrics=lgb,
        single_model_metrics={"lgb": lgb, "fintext": fintext, "sentiment": sentiment},
    )
    assert gates["gate_1_factor"]["pass"] is True
    assert gates["gate_2_ml"]["pass"] is True
    assert isinstance(gates["gate_3_practical"]["pass"], bool)
    assert isinstance(gates["gate_4_fusion_specific"]["pass"], bool)


def test_evaluate_gates_fail_on_weak_fusion():
    fusion = compute_metrics(_make_eval_rows(seed=6, signal_strength=0.01))
    lgb = compute_metrics(_make_eval_rows(seed=7, signal_strength=0.22))
    fintext = compute_metrics(_make_eval_rows(seed=8, signal_strength=0.12))
    sentiment = compute_metrics(_make_eval_rows(seed=9, signal_strength=0.02))

    gates = evaluate_gates(
        fusion_metrics=fusion,
        lgb_metrics=lgb,
        single_model_metrics={"lgb": lgb, "fintext": fintext, "sentiment": sentiment},
    )
    assert gates["gate_4_fusion_specific"]["pass"] is False

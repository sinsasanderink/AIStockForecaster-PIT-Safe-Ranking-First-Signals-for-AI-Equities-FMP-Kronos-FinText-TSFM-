"""
Uncertainty Quantification — Chapter 13
=========================================

DEUP-based epistemic uncertainty estimation for the AI Stock Forecaster.

Modules:
    deup_estimator     — g(x) error predictor (13.1)
    aleatoric_baseline — a(x) aleatoric noise estimation (13.2)
    epistemic_signal   — ê(x) = max(0, g(x) - a(x)) decomposition (13.3)
    deup_diagnostics   — Diagnostics A-F + stability (13.4)
    expert_health      — Per-date expert health H(t) + gating (13.4b)
    conformal_intervals — DEUP-normalized conformal prediction intervals (13.5)
    deup_portfolio      — DEUP-sized shadow portfolio + regime evaluation (13.6)
"""

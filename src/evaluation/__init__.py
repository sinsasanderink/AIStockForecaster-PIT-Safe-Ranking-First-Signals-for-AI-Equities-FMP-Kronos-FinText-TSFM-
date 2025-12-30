"""
Evaluation Framework (Chapter 6)

Implements walk-forward validation with strict PIT discipline:
- Expanding window (not rolling)
- Purging of overlapping label windows (per-row-per-horizon)
- Embargo period between train and validation (90 TRADING DAYS)
- Label maturity enforcement (label_matured_at <= cutoff_utc)
- End-of-sample eligibility (all horizons must be valid)
- Sanity checks (IC parity, experiment naming)
- Cost realism (trading costs, slippage, net-of-cost metrics)
- Stability reports (IC decay, regime performance, churn diagnostics)

CANONICAL DEFINITIONS: See definitions.py for time conventions and locked parameters.
"""

from .walk_forward import WalkForwardSplitter, WalkForwardFold
from .sanity_checks import (
    verify_ic_parity,
    validate_experiment_name,
    ExperimentNameBuilder,
)
from .metrics import (
    EvaluationRow,
    compute_rankic_per_date,
    compute_quintile_spread_per_date,
    compute_topk_metrics_per_date,
    compute_churn,
    evaluate_fold,
    evaluate_with_regime_slicing,
    REGIME_DEFINITIONS,
)
from .costs import (
    TRADING_ASSUMPTIONS,
    compute_participation_rate,
    compute_slippage_bps,
    compute_trade_cost,
    compute_turnover,
    compute_portfolio_costs,
    compute_net_metrics,
    run_cost_sensitivity,
    validate_cost_monotonicity,
    validate_aum_monotonicity,
)
from .reports import (
    STABILITY_THRESHOLDS,
    StabilityReportInputs,
    StabilityReportOutputs,
    compute_ic_decay_stats,
    plot_ic_decay,
    format_regime_performance,
    plot_regime_bars,
    compute_churn_diagnostics,
    plot_churn_timeseries,
    plot_churn_distribution,
    generate_stability_scorecard,
    generate_stability_report,
    validate_report_determinism,
)
from .baselines import (
    BASELINE_REGISTRY,
    BASELINE_MOM_12M,
    BASELINE_MOMENTUM_COMPOSITE,
    BASELINE_SHORT_TERM_STRENGTH,
    BASELINE_TABULAR_LGB,
    BASELINE_NAIVE_RANDOM,
    FACTOR_BASELINES,
    ML_BASELINES,
    SANITY_BASELINES,
    generate_baseline_scores,
    generate_ml_baseline_scores,
    run_all_baselines,
    list_baselines,
)
from .run_evaluation import (
    ExperimentSpec,
    SMOKE_MODE,
    FULL_MODE,
    COST_SCENARIOS,
    run_experiment,
    compute_acceptance_verdict,
    save_acceptance_summary,
)
from .qlib_adapter import (
    is_qlib_available,
    to_qlib_format,
    from_qlib_format,
    validate_qlib_frame,
    run_qlib_shadow_evaluation,
    check_ic_parity,
)
from .definitions import (
    # Time Conventions
    TIME_CONVENTIONS,
    HORIZONS_TRADING_DAYS,
    TRADING_DAYS_PER_YEAR,
    CALENDAR_TO_TRADING_RATIO,
    # Evaluation Range
    EVALUATION_RANGE,
    # Rules
    EMBARGO_RULES,
    ELIGIBILITY_RULES,
    # Helpers
    get_market_close_utc,
    is_label_mature,
    trading_days_to_calendar_days,
    calendar_days_to_trading_days,
    validate_horizon,
    validate_embargo,
    # Documentation
    DEFINITION_LOCK_SUMMARY,
)
from .data_loader import (
    DataLoaderConfig,
    SYNTHETIC_CONFIG,
    SMOKE_CONFIG,
    DEFAULT_DUCKDB_PATH,
    generate_synthetic_features,
    load_features_for_evaluation,
    load_features_from_duckdb,
    check_duckdb_available,
    validate_features_for_evaluation,
    compute_data_hash,
    generate_data_manifest,
)

__all__ = [
    # Walk-Forward
    "WalkForwardSplitter",
    "WalkForwardFold",
    # Sanity Checks
    "verify_ic_parity",
    "validate_experiment_name",
    "ExperimentNameBuilder",
    # Metrics
    "EvaluationRow",
    "compute_rankic_per_date",
    "compute_quintile_spread_per_date",
    "compute_topk_metrics_per_date",
    "compute_churn",
    "evaluate_fold",
    "evaluate_with_regime_slicing",
    "REGIME_DEFINITIONS",
    # Costs
    "TRADING_ASSUMPTIONS",
    "compute_participation_rate",
    "compute_slippage_bps",
    "compute_trade_cost",
    "compute_turnover",
    "compute_portfolio_costs",
    "compute_net_metrics",
    "run_cost_sensitivity",
    "validate_cost_monotonicity",
    "validate_aum_monotonicity",
    # Reports
    "STABILITY_THRESHOLDS",
    "StabilityReportInputs",
    "StabilityReportOutputs",
    "compute_ic_decay_stats",
    "plot_ic_decay",
    "format_regime_performance",
    "plot_regime_bars",
    "compute_churn_diagnostics",
    "plot_churn_timeseries",
    "plot_churn_distribution",
    "generate_stability_scorecard",
    "generate_stability_report",
    "validate_report_determinism",
    # Baselines
    "BASELINE_REGISTRY",
    "BASELINE_MOM_12M",
    "BASELINE_MOMENTUM_COMPOSITE",
    "BASELINE_SHORT_TERM_STRENGTH",
    "BASELINE_TABULAR_LGB",
    "BASELINE_NAIVE_RANDOM",
    "FACTOR_BASELINES",
    "ML_BASELINES",
    "SANITY_BASELINES",
    "generate_baseline_scores",
    "generate_ml_baseline_scores",
    "run_all_baselines",
    "list_baselines",
    # Run Evaluation
    "ExperimentSpec",
    "SMOKE_MODE",
    "FULL_MODE",
    "COST_SCENARIOS",
    "run_experiment",
    "compute_acceptance_verdict",
    "save_acceptance_summary",
    # Qlib Adapter
    "is_qlib_available",
    "to_qlib_format",
    "from_qlib_format",
    "validate_qlib_frame",
    "run_qlib_shadow_evaluation",
    "check_ic_parity",
    # Definitions (Canonical)
    "TIME_CONVENTIONS",
    "HORIZONS_TRADING_DAYS",
    "TRADING_DAYS_PER_YEAR",
    "CALENDAR_TO_TRADING_RATIO",
    "EVALUATION_RANGE",
    "EMBARGO_RULES",
    "ELIGIBILITY_RULES",
    # Helpers
    "get_market_close_utc",
    "is_label_mature",
    "trading_days_to_calendar_days",
    "calendar_days_to_trading_days",
    "validate_horizon",
    "validate_embargo",
    "DEFINITION_LOCK_SUMMARY",
    # Data Loader
    "DataLoaderConfig",
    "SYNTHETIC_CONFIG",
    "SMOKE_CONFIG",
    "DEFAULT_DUCKDB_PATH",
    "generate_synthetic_features",
    "load_features_for_evaluation",
    "load_features_from_duckdb",
    "check_duckdb_available",
    "validate_features_for_evaluation",
    "compute_data_hash",
    "generate_data_manifest",
]

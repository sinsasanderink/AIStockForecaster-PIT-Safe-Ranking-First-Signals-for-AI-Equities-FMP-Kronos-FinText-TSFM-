"""
Pipelines Module
================

Pure functions orchestrating the major steps of the forecasting system.
Each pipeline is a reproducible, auditable workflow.

Pipelines:
- data_pipeline: Download and store market data with PIT safety
- universe_pipeline: Build survivorship-safe dynamic universe
- feature_pipeline: Engineer and validate features
- training_pipeline: Train/fine-tune models with walk-forward
- scoring_pipeline: Generate signals for a given as-of date
- report_pipeline: Create human/machine-readable outputs
"""

from .data_pipeline import (
    run_data_download,
    run_incremental_update,
)

from .universe_pipeline import (
    run_universe_construction,
)

from .scoring_pipeline import (
    run_scoring,
)

from .report_pipeline import (
    run_report_generation,
)

__all__ = [
    "run_data_download",
    "run_incremental_update",
    "run_universe_construction",
    "run_scoring",
    "run_report_generation",
]


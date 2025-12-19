"""
Configuration Management
========================

Centralized configuration for the AI Stock Forecaster.
Loads environment variables and defines project-wide settings.
"""

import os
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass, field
from datetime import time
import pytz

from dotenv import load_dotenv

# Load environment variables from .env file
PROJECT_ROOT = Path(__file__).parent.parent
load_dotenv(PROJECT_ROOT / ".env")


@dataclass
class APIConfig:
    """API configuration settings."""
    fmp_api_key: str = field(default_factory=lambda: os.getenv("FMP_KEYS", ""))
    
    def __post_init__(self):
        if not self.fmp_api_key:
            raise ValueError(
                "FMP_KEYS not found in environment. "
                "Please set it in your .env file."
            )


@dataclass
class DataConfig:
    """Data-related configuration."""
    # Paths
    raw_data_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "raw")
    processed_data_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "processed")
    cache_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "cache")
    pit_store_path: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "pit_store.duckdb")
    
    # Point-in-time settings
    cutoff_time: time = field(default_factory=lambda: time(16, 0))  # 4:00 PM ET
    cutoff_timezone: str = "America/New_York"
    
    # Conservative lag rules for FMP data (when observed_at is unavailable)
    # These are conservative estimates for when data becomes publicly available
    fundamental_lag_days: int = 1  # Earnings usually available next day after filing
    price_lag_days: int = 0  # EOD prices available same day after close
    
    # Data quality thresholds
    min_price: float = 5.0  # Minimum stock price
    min_avg_volume: int = 100_000  # Minimum average daily volume
    min_market_cap: float = 1e9  # $1B minimum market cap
    
    def __post_init__(self):
        # Create directories if they don't exist
        for dir_path in [self.raw_data_dir, self.processed_data_dir, self.cache_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)


@dataclass
class UniverseConfig:
    """Universe construction settings."""
    # Use the seed AI universe (100 stocks across 10 categories)
    # See src/universe/ai_stocks.py for full definitions
    use_seed_universe: bool = True
    
    # If not using seed, these filters apply
    ai_sectors: List[str] = field(default_factory=lambda: [
        "Technology",
        "Communication Services",
    ])
    
    ai_industries: List[str] = field(default_factory=lambda: [
        "Semiconductors",
        "Software—Infrastructure", 
        "Software—Application",
        "Information Technology Services",
        "Computer Hardware",
        "Electronic Components",
        "Scientific & Technical Instruments",
        "Communication Equipment",
    ])
    
    # Universe size (when filtering, not using seed)
    max_universe_size: int = 100  # All 100 AI stocks by default
    
    # Subset of categories to include (None = all 10 categories)
    # Options: ai_compute_core_semis, semicap_eda_manufacturing, 
    # networking_optics_dc_hw, datacenter_power_cooling_reits,
    # cloud_platforms_model_owners, data_platforms_enterprise_sw,
    # cybersecurity, robotics_industrial_ai, autonomy_defense_ai, ai_apps_ads_misc
    categories: Optional[List[str]] = None  # None = all categories
    
    # Benchmarks
    primary_benchmark: str = "QQQ"
    secondary_benchmarks: List[str] = field(default_factory=lambda: ["XLK", "SMH", "SOXX"])


@dataclass
class HorizonConfig:
    """Forecast horizon settings."""
    # Trading days for each horizon
    short_horizon: int = 20   # ~1 month
    medium_horizon: int = 60  # ~3 months
    long_horizon: int = 90    # ~4.5 months
    
    @property
    def all_horizons(self) -> List[int]:
        return [self.short_horizon, self.medium_horizon, self.long_horizon]


@dataclass
class ModelConfig:
    """Model-related configuration."""
    # Model paths
    models_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "models")
    kronos_checkpoint: str = "Doradx/Kronos"  # HuggingFace model ID
    fintext_checkpoint: str = "FinText/Chronos-Bolt-Base"  # HuggingFace model ID
    
    # Kronos settings
    kronos_max_context: int = 512  # Max context length for small/base
    
    # Training settings
    batch_size: int = 32
    learning_rate: float = 1e-4
    
    # Inference settings
    device: str = "cuda"  # or "cpu"
    
    def __post_init__(self):
        self.models_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class EvaluationConfig:
    """Evaluation and validation settings."""
    # Walk-forward settings
    min_train_years: int = 2
    validation_months: int = 6
    
    # Cost modeling
    base_transaction_cost_bps: float = 20.0  # 20 bps round-trip
    
    # Success thresholds
    min_rank_ic: float = 0.02  # Minimum RankIC for baselines
    min_ml_ic: float = 0.05   # Minimum IC for ML models
    min_improvement: float = 0.02  # Required IC improvement over baseline
    
    # Turnover limits
    max_monthly_churn: float = 0.30  # 30% max ranking churn


@dataclass
class OutputConfig:
    """Output and logging settings."""
    outputs_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "outputs")
    signals_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "outputs" / "signals")
    reports_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "outputs" / "reports")
    logs_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "outputs" / "logs")
    
    # Signal output settings
    confidence_buckets: int = 3  # High, Medium, Low
    top_n_display: int = 10  # Number of top/bottom stocks to highlight
    
    def __post_init__(self):
        for dir_path in [self.outputs_dir, self.signals_dir, self.reports_dir, self.logs_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)


@dataclass
class Config:
    """Master configuration aggregating all settings."""
    api: APIConfig = field(default_factory=APIConfig)
    data: DataConfig = field(default_factory=DataConfig)
    universe: UniverseConfig = field(default_factory=UniverseConfig)
    horizons: HorizonConfig = field(default_factory=HorizonConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


def get_config() -> Config:
    """Get the global configuration instance."""
    return Config()


# Convenience function to get timezone-aware cutoff
def get_cutoff_datetime(date, config: Optional[DataConfig] = None):
    """Get the cutoff datetime for a given date."""
    if config is None:
        config = DataConfig()
    
    tz = pytz.timezone(config.cutoff_timezone)
    from datetime import datetime
    return tz.localize(datetime.combine(date, config.cutoff_time))


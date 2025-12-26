"""Configuration module for ATM project."""

from nq.config.loader import Config, DataIngestorTaskConfig, DatabaseConfig, SourceConfig, load_config

__all__ = ["Config", "DatabaseConfig", "SourceConfig", "DataIngestorTaskConfig", "load_config"]

# Backward compatibility alias
CollectorTaskConfig = DataIngestorTaskConfig


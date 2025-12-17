"""
Configuration loader for ATM project.

Loads configuration from YAML files and environment variables.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field


class DatabaseConfig(BaseModel):
    """Database configuration."""

    model_config = ConfigDict(populate_by_name=True)

    host: str = Field(default="localhost", description="Database host")
    port: int = Field(default=5432, description="Database port")
    user: str = Field(default="quant", description="Database user")
    password: str = Field(default="", description="Database password")
    database: str = Field(default="quant_db", description="Database name")
    schema_name: str = Field(default="quant", alias="schema", description="Database schema")

    @property
    def schema(self) -> str:
        """Get schema name (for backward compatibility)."""
        return self.schema_name


class SourceConfig(BaseModel):
    """Data source configuration."""

    type: str = Field(..., description="Source type (http, file, etc.)")
    url: Optional[str] = Field(default=None, description="Source URL")
    params: Dict[str, Any] = Field(default_factory=dict, description="Source parameters")
    headers: Dict[str, str] = Field(default_factory=dict, description="HTTP headers")
    timeout: int = Field(default=30, description="Request timeout in seconds")
    retry_count: int = Field(default=3, description="Number of retries")
    retry_delay: int = Field(default=1, description="Delay between retries in seconds")


class DataIngestorTaskConfig(BaseModel):
    """Data ingestor task configuration."""

    name: str = Field(..., description="Task name")
    enabled: bool = Field(default=True, description="Whether task is enabled")
    schedule: Optional[str] = Field(default=None, description="Cron schedule expression")
    source: SourceConfig = Field(..., description="Data source configuration")
    table: str = Field(..., description="Target database table")
    batch_size: int = Field(default=1000, description="Batch size for data insertion")
    on_conflict: str = Field(default="update", description="Conflict resolution strategy")


class Config(BaseModel):
    """Main configuration model."""

    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    tasks: list[DataIngestorTaskConfig] = Field(default_factory=list, description="Data ingestor tasks")


def load_config(config_path: Optional[str] = None) -> Config:
    """
    Load configuration from YAML file and environment variables.

    Args:
        config_path: Path to configuration file. If None, searches for config.yaml
                     in project root and config directory.

    Returns:
        Config object with loaded configuration.

    Raises:
        FileNotFoundError: If configuration file is not found.
        ValueError: If configuration is invalid.
    """
    if config_path is None:
        # Try to find config file in common locations
        project_root = Path(__file__).parent.parent.parent.parent
        possible_paths = [
            project_root / "config" / "config.yaml",
            project_root / "config.yaml",
            Path("config.yaml"),
        ]

        config_path = None
        for path in possible_paths:
            if path.exists():
                config_path = str(path)
                break

        if config_path is None:
            raise FileNotFoundError("Configuration file not found")

    with open(config_path, "r", encoding="utf-8") as f:
        config_data = yaml.safe_load(f) or {}

    # Override with environment variables
    config_data = _override_with_env(config_data)

    try:
        return Config(**config_data)
    except Exception as e:
        raise ValueError(f"Invalid configuration: {e}") from e


def _override_with_env(config_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Override configuration with environment variables.

    Environment variables format:
    - ATM_DATABASE_HOST
    - ATM_DATABASE_PORT
    - ATM_DATABASE_USER
    - ATM_DATABASE_PASSWORD
    - ATM_DATABASE_NAME

    Args:
        config_data: Configuration dictionary.

    Returns:
        Updated configuration dictionary.
    """
    env_mappings = {
        "ATM_DATABASE_HOST": ("database", "host"),
        "ATM_DATABASE_PORT": ("database", "port"),
        "ATM_DATABASE_USER": ("database", "user"),
        "ATM_DATABASE_PASSWORD": ("database", "password"),
        "ATM_DATABASE_NAME": ("database", "database"),
        "ATM_DATABASE_SCHEMA": ("database", "schema"),
    }

    for env_var, (section, key) in env_mappings.items():
        value = os.getenv(env_var)
        if value is not None:
            if section not in config_data:
                config_data[section] = {}
            if key == "port":
                config_data[section][key] = int(value)
            else:
                config_data[section][key] = value

    return config_data


"""
Base backtest interface.

Provides abstract base class for all backtest implementations.
"""

import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class BacktestResult(BaseModel):
    """Backtest result model."""

    ts_code: str = Field(..., description="Stock code")
    backtest_date: datetime = Field(..., description="Date when backtest was performed")
    start_date: datetime = Field(..., description="Start date of backtest period")
    end_date: datetime = Field(..., description="End date of backtest period")
    predictions: pd.DataFrame = Field(..., description="Predicted data")
    actuals: pd.DataFrame = Field(..., description="Actual data for comparison")
    metrics: Dict[str, Any] = Field(default_factory=dict, description="Backtest metrics")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    class Config:
        """Pydantic configuration."""

        arbitrary_types_allowed = True


class BaseBacktester(ABC):
    """
    Abstract base class for all backtesters.

    All backtesters should inherit from this class and implement the required methods.
    """

    def __init__(
        self,
        db_config,
        schema: str = "quant",
    ):
        """
        Initialize backtester.

        Args:
            db_config: Database configuration.
            schema: Database schema name.
        """
        self.db_config = db_config
        self.schema = schema
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def run(
        self,
        ts_code: str,
        start_date: datetime,
        end_date: datetime,
        **kwargs,
    ) -> BacktestResult:
        """
        Run backtest.

        Args:
            ts_code: Stock code in Tushare format (e.g., 000001.SZ).
            start_date: Start date of backtest period (inclusive).
            end_date: End date of backtest period (inclusive).
            **kwargs: Additional backtest parameters.

        Returns:
            BacktestResult containing predictions, actuals, and metrics.
        """
        pass

    def validate_inputs(
        self,
        ts_code: str,
        start_date: datetime,
        end_date: datetime,
    ) -> None:
        """
        Validate input parameters.

        Args:
            ts_code: Stock code.
            start_date: Start date.
            end_date: End date.

        Raises:
            ValueError: If inputs are invalid.
        """
        if not ts_code or not isinstance(ts_code, str):
            raise ValueError("ts_code must be a non-empty string")

        if start_date >= end_date:
            raise ValueError("start_date must be before end_date")

    def get_info(self) -> dict:
        """
        Get backtester information.

        Returns:
            Dictionary containing backtester information.
        """
        return {
            "class": self.__class__.__name__,
            "schema": self.schema,
        }


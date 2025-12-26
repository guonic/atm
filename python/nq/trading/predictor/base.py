"""
Base predictor interface for trading predictions.

Provides abstract base class for all prediction models.
"""

import logging
from abc import ABC, abstractmethod
from datetime import date, datetime
from typing import Optional

import pandas as pd
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class PredictionResult(BaseModel):
    """Prediction result model."""

    ts_code: str = Field(..., description="Stock code")
    historical_data: pd.DataFrame = Field(..., description="Historical data used for prediction")
    predicted_data: pd.DataFrame = Field(..., description="Predicted future data")
    prediction_date: datetime = Field(..., description="Date when prediction was made")
    lookback_days: int = Field(..., description="Number of historical days used")
    pred_len: int = Field(..., description="Number of days predicted")
    start_date: Optional[datetime] = Field(None, description="Start date of historical data range")
    end_date: Optional[datetime] = Field(None, description="End date of historical data range (cutoff date)")
    metadata: dict = Field(default_factory=dict, description="Additional metadata")

    class Config:
        """Pydantic configuration."""

        arbitrary_types_allowed = True


class BasePredictor(ABC):
    """
    Abstract base class for all predictors.

    All predictors should inherit from this class and implement the required methods.
    """

    def __init__(
        self,
        db_config,
        schema: str = "quant",
    ):
        """
        Initialize predictor.

        Args:
            db_config: Database configuration.
            schema: Database schema name.
        """
        self.db_config = db_config
        self.schema = schema
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def predict(
        self,
        ts_code: str,
        pred_len: int,
        lookback: Optional[int] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        **kwargs,
    ) -> PredictionResult:
        """
        Predict future stock prices.

        Args:
            ts_code: Stock code in Tushare format (e.g., 000001.SZ).
            pred_len: Number of trading days to predict.
            lookback: Number of historical days to use. If None, use default.
                     Only used when start_date and end_date are not specified.
            start_date: Start date for historical data (inclusive). If specified,
                       loads data from this date. If None, uses latest data.
            end_date: End date for historical data (inclusive). If specified,
                     loads data up to this date. If None, uses latest data.
                     For backtesting, this is the cutoff date before prediction period.
            **kwargs: Additional prediction parameters.

        Returns:
            PredictionResult containing historical and predicted data.
        """
        pass

    @abstractmethod
    def load_model(self, **kwargs):
        """
        Load prediction model.

        Args:
            **kwargs: Model loading parameters.
        """
        pass

    def validate_inputs(
        self,
        ts_code: str,
        pred_len: int,
        lookback: Optional[int] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> None:
        """
        Validate input parameters.

        Args:
            ts_code: Stock code.
            pred_len: Prediction length.
            lookback: Lookback length.
            start_date: Start date for historical data.
            end_date: End date for historical data.

        Raises:
            ValueError: If inputs are invalid.
        """
        if not ts_code or not isinstance(ts_code, str):
            raise ValueError("ts_code must be a non-empty string")

        if pred_len <= 0:
            raise ValueError("pred_len must be positive")

        if lookback is not None and lookback <= 0:
            raise ValueError("lookback must be positive if provided")

        if start_date is not None and end_date is not None:
            if start_date >= end_date:
                raise ValueError("start_date must be before end_date")

    def get_info(self) -> dict:
        """
        Get predictor information.

        Returns:
            Dictionary containing predictor information.
        """
        return {
            "class": self.__class__.__name__,
            "schema": self.schema,
        }


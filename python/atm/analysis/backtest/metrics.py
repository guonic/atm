"""
Backtest metrics calculation.

Provides functions to calculate various backtest metrics.
"""

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class BacktestMetrics:
    """Backtest metrics calculator."""

    @staticmethod
    def calculate_price_metrics(
        predicted: pd.Series,
        actual: pd.Series,
        price_col: str = "close",
    ) -> Dict[str, float]:
        """
        Calculate price prediction metrics.

        Args:
            predicted: DataFrame with predicted prices.
            actual: DataFrame with actual prices.
            price_col: Column name for price (default: 'close').

        Returns:
            Dictionary containing metrics.
        """
        if isinstance(predicted, pd.DataFrame):
            pred_values = predicted[price_col].values
        else:
            pred_values = predicted.values

        if isinstance(actual, pd.DataFrame):
            actual_values = actual[price_col].values
        else:
            actual_values = actual.values

        # Remove NaN values
        mask = ~(np.isnan(pred_values) | np.isnan(actual_values))
        pred_values = pred_values[mask]
        actual_values = actual_values[mask]

        if len(pred_values) == 0:
            logger.warning("No valid data for metrics calculation")
            return {}

        # Calculate metrics
        mae = np.mean(np.abs(pred_values - actual_values))
        mse = np.mean((pred_values - actual_values) ** 2)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((actual_values - pred_values) / (actual_values + 1e-8))) * 100

        # Direction accuracy (up/down prediction)
        if len(pred_values) > 1:
            pred_direction = np.diff(pred_values) > 0
            actual_direction = np.diff(actual_values) > 0
            direction_accuracy = np.mean(pred_direction == actual_direction) * 100
        else:
            direction_accuracy = 0.0

        # Correlation
        correlation = np.corrcoef(pred_values, actual_values)[0, 1] if len(pred_values) > 1 else 0.0

        return {
            "mae": float(mae),
            "mse": float(mse),
            "rmse": float(rmse),
            "mape": float(mape),
            "direction_accuracy": float(direction_accuracy),
            "correlation": float(correlation),
            "num_samples": int(len(pred_values)),
        }

    @staticmethod
    def calculate_return_metrics(
        predicted: pd.Series,
        actual: pd.Series,
        price_col: str = "close",
    ) -> Dict[str, float]:
        """
        Calculate return prediction metrics.

        Args:
            predicted: DataFrame with predicted prices.
            actual: DataFrame with actual prices.
            price_col: Column name for price (default: 'close').

        Returns:
            Dictionary containing return metrics.
        """
        if isinstance(predicted, pd.DataFrame):
            pred_prices = predicted[price_col].values
        else:
            pred_prices = predicted.values

        if isinstance(actual, pd.DataFrame):
            actual_prices = actual[price_col].values
        else:
            actual_prices = actual.values

        # Calculate returns
        if len(pred_prices) > 1:
            pred_returns = np.diff(pred_prices) / pred_prices[:-1] * 100
            actual_returns = np.diff(actual_prices) / actual_prices[:-1] * 100

            # Remove NaN values
            mask = ~(np.isnan(pred_returns) | np.isnan(actual_returns))
            pred_returns = pred_returns[mask]
            actual_returns = actual_returns[mask]

            if len(pred_returns) == 0:
                return {}

            mae_return = np.mean(np.abs(pred_returns - actual_returns))
            rmse_return = np.sqrt(np.mean((pred_returns - actual_returns) ** 2))
            return_correlation = (
                np.corrcoef(pred_returns, actual_returns)[0, 1] if len(pred_returns) > 1 else 0.0
            )

            return {
                "return_mae": float(mae_return),
                "return_rmse": float(rmse_return),
                "return_correlation": float(return_correlation),
            }
        else:
            return {}

    @staticmethod
    def calculate_all_metrics(
        predicted: pd.DataFrame,
        actual: pd.DataFrame,
        price_col: str = "close",
    ) -> Dict[str, float]:
        """
        Calculate all available metrics.

        Args:
            predicted: DataFrame with predicted data.
            actual: DataFrame with actual data.
            price_col: Column name for price (default: 'close').

        Returns:
            Dictionary containing all metrics.
        """
        metrics = {}

        # Price metrics
        price_metrics = BacktestMetrics.calculate_price_metrics(
            predicted, actual, price_col=price_col
        )
        metrics.update(price_metrics)

        # Return metrics
        return_metrics = BacktestMetrics.calculate_return_metrics(
            predicted, actual, price_col=price_col
        )
        metrics.update(return_metrics)

        return metrics


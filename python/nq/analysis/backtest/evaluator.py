"""
Backtest evaluator.

Provides evaluation and comparison functionality for backtest results.
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .base import BacktestResult
from .metrics import BacktestMetrics

logger = logging.getLogger(__name__)


class BacktestEvaluator:
    """Evaluator for backtest results."""

    @staticmethod
    def evaluate(result: BacktestResult) -> Dict[str, any]:
        """
        Evaluate a single backtest result.

        Args:
            result: BacktestResult to evaluate.

        Returns:
            Dictionary containing evaluation summary.
        """
        metrics = result.metrics

        # Calculate additional evaluation metrics
        evaluation = {
            "ts_code": result.ts_code,
            "period": {
                "start": result.start_date.isoformat(),
                "end": result.end_date.isoformat(),
            },
            "metrics": metrics,
            "metadata": result.metadata,
        }

        # Add performance rating
        if "mape" in metrics:
            mape = metrics["mape"]
            if mape < 5:
                evaluation["performance_rating"] = "Excellent"
            elif mape < 10:
                evaluation["performance_rating"] = "Good"
            elif mape < 20:
                evaluation["performance_rating"] = "Fair"
            else:
                evaluation["performance_rating"] = "Poor"

        if "direction_accuracy" in metrics:
            direction_acc = metrics["direction_accuracy"]
            if direction_acc > 70:
                evaluation["direction_rating"] = "Excellent"
            elif direction_acc > 60:
                evaluation["direction_rating"] = "Good"
            elif direction_acc > 50:
                evaluation["direction_rating"] = "Fair"
            else:
                evaluation["direction_rating"] = "Poor"

        return evaluation

    @staticmethod
    def compare(results: List[BacktestResult]) -> Dict[str, any]:
        """
        Compare multiple backtest results.

        Args:
            results: List of BacktestResult to compare.

        Returns:
            Dictionary containing comparison summary.
        """
        if not results:
            return {}

        comparison = {
            "num_results": len(results),
            "results": [],
        }

        # Evaluate each result
        for result in results:
            evaluation = BacktestEvaluator.evaluate(result)
            comparison["results"].append(evaluation)

        # Aggregate metrics
        if results:
            all_mae = [r.metrics.get("mae", 0) for r in results if "mae" in r.metrics]
            all_rmse = [r.metrics.get("rmse", 0) for r in results if "rmse" in r.metrics]
            all_direction_acc = [
                r.metrics.get("direction_accuracy", 0)
                for r in results
                if "direction_accuracy" in r.metrics
            ]

            aggregate_metrics = {}

            if all_mae:
                aggregate_metrics.update({
                    "avg_mae": float(np.mean(all_mae)),
                    "min_mae": float(np.min(all_mae)),
                    "max_mae": float(np.max(all_mae)),
                })

            if all_rmse:
                aggregate_metrics.update({
                    "avg_rmse": float(np.mean(all_rmse)),
                    "min_rmse": float(np.min(all_rmse)),
                    "max_rmse": float(np.max(all_rmse)),
                })

            if all_direction_acc:
                aggregate_metrics.update({
                    "avg_direction_accuracy": float(np.mean(all_direction_acc)),
                    "min_direction_accuracy": float(np.min(all_direction_acc)),
                    "max_direction_accuracy": float(np.max(all_direction_acc)),
                })

            if aggregate_metrics:
                comparison["aggregate_metrics"] = aggregate_metrics

        return comparison


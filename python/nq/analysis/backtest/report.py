"""
Backtest report generation.

Provides functionality to generate backtest reports.
"""

import logging
from typing import Dict, Optional

import pandas as pd

from .base import BacktestResult
from .evaluator import BacktestEvaluator

logger = logging.getLogger(__name__)


class BacktestReport:
    """Backtest report generator."""

    @staticmethod
    def generate_text_report(result: BacktestResult) -> str:
        """
        Generate text report for backtest result.

        Args:
            result: BacktestResult to generate report for.

        Returns:
            Formatted text report.
        """
        evaluation = BacktestEvaluator.evaluate(result)
        metrics = result.metrics

        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append(f"Backtest Report: {result.ts_code}")
        report_lines.append("=" * 80)
        report_lines.append("")
        report_lines.append("Period:")
        report_lines.append(f"  Start: {result.start_date.date()}")
        report_lines.append(f"  End: {result.end_date.date()}")
        report_lines.append(f"  Backtest Date: {result.backtest_date.strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        report_lines.append("Metrics:")
        report_lines.append(f"  MAE (Mean Absolute Error): {metrics.get('mae', 0):.2f}")
        report_lines.append(f"  RMSE (Root Mean Squared Error): {metrics.get('rmse', 0):.2f}")
        report_lines.append(f"  MAPE (Mean Absolute Percentage Error): {metrics.get('mape', 0):.2f}%")
        report_lines.append(
            f"  Direction Accuracy: {metrics.get('direction_accuracy', 0):.2f}%"
        )
        report_lines.append(f"  Correlation: {metrics.get('correlation', 0):.4f}")
        report_lines.append("")

        if "return_mae" in metrics:
            report_lines.append("Return Metrics:")
            report_lines.append(f"  Return MAE: {metrics.get('return_mae', 0):.4f}%")
            report_lines.append(f"  Return RMSE: {metrics.get('return_rmse', 0):.4f}%")
            report_lines.append(
                f"  Return Correlation: {metrics.get('return_correlation', 0):.4f}"
            )
            report_lines.append("")

        if "performance_rating" in evaluation:
            report_lines.append(f"Performance Rating: {evaluation['performance_rating']}")
        if "direction_rating" in evaluation:
            report_lines.append(f"Direction Rating: {evaluation['direction_rating']}")
        report_lines.append("")

        report_lines.append("Metadata:")
        for key, value in result.metadata.items():
            report_lines.append(f"  {key}: {value}")
        report_lines.append("")

        report_lines.append("=" * 80)

        return "\n".join(report_lines)

    @staticmethod
    def print_report(result: BacktestResult) -> None:
        """
        Print backtest report to logger.

        Args:
            result: BacktestResult to print report for.
        """
        report = BacktestReport.generate_text_report(result)
        logger.info("\n" + report)

    @staticmethod
    def save_report(result: BacktestResult, filepath: str) -> None:
        """
        Save backtest report to file.

        Args:
            result: BacktestResult to save report for.
            filepath: Path to save report file.
        """
        report = BacktestReport.generate_text_report(result)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(report)
        logger.info(f"Backtest report saved to: {filepath}")

    @staticmethod
    def generate_summary_dataframe(results: list[BacktestResult]) -> pd.DataFrame:
        """
        Generate summary DataFrame from multiple backtest results.

        Args:
            results: List of BacktestResult.

        Returns:
            DataFrame with summary metrics.
        """
        summary_data = []

        for result in results:
            row = {
                "ts_code": result.ts_code,
                "start_date": result.start_date.date(),
                "end_date": result.end_date.date(),
                "mae": result.metrics.get("mae", None),
                "rmse": result.metrics.get("rmse", None),
                "mape": result.metrics.get("mape", None),
                "direction_accuracy": result.metrics.get("direction_accuracy", None),
                "correlation": result.metrics.get("correlation", None),
                "num_samples": result.metrics.get("num_samples", None),
            }
            summary_data.append(row)

        return pd.DataFrame(summary_data)


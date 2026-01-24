"""
Statistics for Teapot pattern recognition signals.

Computes win rate, average returns, and other statistics.
"""

import logging
from typing import Dict, List

import polars as pl

logger = logging.getLogger(__name__)


class TeapotStatistics:
    """
    Statistics calculator for Teapot signals.

    Computes win rate, average returns, Sharpe ratio, etc.
    """

    def __init__(self):
        """Initialize statistics calculator."""
        pass

    def compute_basic_stats(
        self, evaluation_results: pl.DataFrame
    ) -> Dict:
        """
        Compute basic statistics.

        Args:
            evaluation_results: Evaluation results DataFrame.

        Returns:
            Dictionary with statistics.
        """
        if evaluation_results.is_empty():
            return {}

        stats = {
            "total_signals": len(evaluation_results),
        }

        # Compute win rates and average returns for each horizon
        for col in evaluation_results.columns:
            if col.startswith("return_t"):
                returns = evaluation_results[col].drop_nulls()
                if not returns.is_empty():
                    horizon = col.replace("return_t", "")
                    stats[f"win_rate_t{horizon}"] = (
                        (returns > 0).sum() / len(returns)
                    )
                    stats[f"avg_return_t{horizon}"] = returns.mean()
                    stats[f"std_return_t{horizon}"] = returns.std()
                    stats[f"sharpe_ratio_t{horizon}"] = (
                        returns.mean() / returns.std()
                        if returns.std() > 0
                        else 0
                    )
                    stats[f"max_return_t{horizon}"] = returns.max()
                    stats[f"min_return_t{horizon}"] = returns.min()

        # Compute max drawdown
        for col in evaluation_results.columns:
            if col.startswith("max_drawdown_t"):
                drawdowns = evaluation_results[col].drop_nulls()
                if not drawdowns.is_empty():
                    horizon = col.replace("max_drawdown_t", "")
                    stats[f"max_drawdown_t{horizon}"] = drawdowns.min()
                    stats[f"avg_drawdown_t{horizon}"] = drawdowns.mean()

        # Compute profit/loss ratio
        if "return_t20" in evaluation_results.columns:
            returns = evaluation_results["return_t20"].drop_nulls()
            if not returns.is_empty():
                profits = returns.filter(returns > 0)
                losses = returns.filter(returns < 0)
                if not losses.is_empty() and not profits.is_empty():
                    stats["profit_loss_ratio"] = abs(
                        profits.mean() / losses.mean()
                    )

        return stats

    def compute_by_period(
        self, evaluation_results: pl.DataFrame, period: str = "year"
    ) -> pl.DataFrame:
        """
        Compute statistics by period.

        Args:
            evaluation_results: Evaluation results DataFrame.
            period: Period type ("year" or "month").

        Returns:
            DataFrame with statistics by period.
        """
        if evaluation_results.is_empty():
            return pl.DataFrame()

        # Extract period from signal_date
        if period == "year":
            evaluation_results = evaluation_results.with_columns(
                pl.col("signal_date").str.slice(0, 4).alias("period")
            )
        elif period == "month":
            evaluation_results = evaluation_results.with_columns(
                pl.col("signal_date").str.slice(0, 7).alias("period")
            )

        # Group by period and compute statistics
        stats_by_period = []
        for period_val in evaluation_results["period"].unique():
            period_data = evaluation_results.filter(
                pl.col("period") == period_val
            )
            stats = self.compute_basic_stats(period_data)
            stats["period"] = period_val
            stats_by_period.append(stats)

        return pl.DataFrame(stats_by_period)

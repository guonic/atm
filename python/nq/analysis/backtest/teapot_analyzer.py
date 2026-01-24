"""
Analyzer for Teapot backtest results.

Analyzes backtest performance and generates reports.
"""

import logging
from typing import Dict, List

import pandas as pd

from nq.analysis.backtest.teapot_backtester import BacktestResult

logger = logging.getLogger(__name__)


class TeapotAnalyzer:
    """
    Analyzer for Teapot backtest results.

    Computes performance metrics and generates analysis reports.
    """

    def __init__(self):
        """Initialize analyzer."""
        pass

    def analyze_backtest(self, backtest_result: BacktestResult) -> Dict:
        """
        Analyze single backtest result.

        Args:
            backtest_result: BacktestResult object.

        Returns:
            Dictionary with analysis results.
        """
        metrics = backtest_result.metrics.copy()

        # Compute additional metrics
        if not backtest_result.returns.empty:
            returns = backtest_result.returns
            metrics["volatility"] = returns.std() * (252 ** 0.5)  # Annualized
            metrics["sharpe_ratio"] = (
                returns.mean() * 252 / metrics["volatility"]
                if metrics["volatility"] > 0
                else 0
            )

        # Compute profit factor
        if not backtest_result.trades.empty():
            profits = backtest_result.trades[
                backtest_result.trades["pnl"] > 0
            ]["pnl"].sum()
            losses = abs(
                backtest_result.trades[backtest_result.trades["pnl"] < 0][
                    "pnl"
                ].sum()
            )
            metrics["profit_factor"] = (
                profits / losses if losses > 0 else 0
            )

        return metrics

    def compare_backtests(
        self,
        backtest_results: List[BacktestResult],
        labels: List[str],
    ) -> pd.DataFrame:
        """
        Compare multiple backtest results.

        Args:
            backtest_results: List of BacktestResult objects.
            labels: List of labels for each backtest.

        Returns:
            DataFrame with comparison metrics.
        """
        comparison_data = []
        for result, label in zip(backtest_results, labels):
            analysis = self.analyze_backtest(result)
            analysis["label"] = label
            comparison_data.append(analysis)

        return pd.DataFrame(comparison_data)

    def analyze_by_market_regime(
        self,
        backtest_results: List[BacktestResult],
        market_regimes: List[str],
    ) -> Dict:
        """
        Analyze backtests by market regime.

        Args:
            backtest_results: List of BacktestResult objects.
            market_regimes: List of market regime labels.

        Returns:
            Dictionary with analysis by regime.
        """
        regime_analysis = {}
        for result, regime in zip(backtest_results, market_regimes):
            analysis = self.analyze_backtest(result)
            regime_analysis[regime] = analysis

        return regime_analysis

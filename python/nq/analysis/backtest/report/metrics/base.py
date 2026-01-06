"""
Base metric calculator.
"""

from typing import List

from ..models import BacktestData, MetricResult


class BaseMetricCalculator:
    """Base class for metric calculators."""
    
    def calculate(self, data: BacktestData) -> MetricResult:
        """
        Calculate metric.
        
        Args:
            data: Backtest data.
        
        Returns:
            Metric result.
        
        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError
    
    def get_dependencies(self) -> List[str]:
        """
        Get dependencies on other metrics.
        
        Returns:
            List of metric names this metric depends on.
        """
        return []


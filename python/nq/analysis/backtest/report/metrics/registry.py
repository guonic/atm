"""
Metric registry for backtest report system.
"""

import logging
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class MetricDefinition:
    """Metric definition."""
    
    def __init__(
        self,
        category: str,
        name: str,
        calculator: Callable,
        description: Optional[str] = None,
    ):
        """
        Initialize metric definition.
        
        Args:
            category: Metric category (portfolio, trading, turnover, risk, model)
            name: Metric name
            calculator: Calculator function or class
            description: Metric description
        """
        self.category = category
        self.name = name
        self.calculator = calculator
        self.description = description or name
    
    def get_key(self) -> str:
        """Get metric key (category.name)."""
        return f"{self.category}.{self.name}"


class MetricRegistry:
    """Metric registry for managing metrics."""
    
    _metrics: Dict[str, MetricDefinition] = {}
    
    @classmethod
    def register(
        cls,
        category: str,
        name: str,
        description: Optional[str] = None,
    ) -> Callable:
        """
        Register a metric.
        
        Usage:
            @MetricRegistry.register(category="portfolio", name="total_return")
            class TotalReturnCalculator(BaseMetricCalculator):
                ...
        
        Args:
            category: Metric category
            name: Metric name
            description: Metric description
        
        Returns:
            Decorator function
        """
        def decorator(calculator: Callable) -> Callable:
            key = f"{category}.{name}"
            if key in cls._metrics:
                logger.warning(f"Metric {key} already registered, overwriting")
            
            cls._metrics[key] = MetricDefinition(
                category=category,
                name=name,
                calculator=calculator,
                description=description,
            )
            logger.debug(f"Registered metric: {key}")
            return calculator
        
        return decorator
    
    @classmethod
    def get_metric(cls, category: str, name: str) -> Optional[MetricDefinition]:
        """
        Get metric definition.
        
        Args:
            category: Metric category
            name: Metric name
        
        Returns:
            Metric definition or None if not found
        """
        key = f"{category}.{name}"
        return cls._metrics.get(key)
    
    @classmethod
    def list_metrics(cls, category: Optional[str] = None) -> List[MetricDefinition]:
        """
        List metrics.
        
        Args:
            category: Filter by category (optional)
        
        Returns:
            List of metric definitions
        """
        if category:
            return [m for m in cls._metrics.values() if m.category == category]
        return list(cls._metrics.values())
    
    @classmethod
    def get_calculator(cls, category: str, name: str) -> Optional[Callable]:
        """
        Get calculator for metric.
        
        Args:
            category: Metric category
            name: Metric name
        
        Returns:
            Calculator class or function
        """
        metric = cls.get_metric(category, name)
        if metric:
            return metric.calculator
        return None
    
    @classmethod
    def clear(cls) -> None:
        """Clear all registered metrics (mainly for testing)."""
        cls._metrics.clear()


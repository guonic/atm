"""Metrics module for backtest report system."""

from .base import BaseMetricCalculator
from .registry import MetricRegistry

# Import all metric calculators to register them
from . import portfolio  # noqa: F401
from . import trading  # noqa: F401
from . import turnover  # noqa: F401

__all__ = ["BaseMetricCalculator", "MetricRegistry"]


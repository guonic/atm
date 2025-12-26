"""
OpenTelemetry helper functions for common observability patterns.

Provides decorators and context managers for easy integration.
"""

import functools
import logging
from contextlib import contextmanager
from typing import Callable, Optional

from nq.utils.otel import get_config, get_meter, get_tracer

logger = logging.getLogger(__name__)


def trace_function(
    name: Optional[str] = None,
    attributes: Optional[dict] = None,
    enabled: Optional[bool] = None,
):
    """
    Decorator to trace a function execution.

    Args:
        name: Span name (defaults to function name).
        attributes: Additional span attributes.
        enabled: Override global OpenTelemetry enabled setting.

    Example:
        @trace_function(attributes={"component": "data_ingestor"})
        def sync_stock_data():
            ...
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            config = get_config()
            if enabled is False or (enabled is None and not config.is_enabled()):
                return func(*args, **kwargs)

            tracer = get_tracer(func.__module__)
            if tracer is None:
                return func(*args, **kwargs)

            span_name = name or f"{func.__module__}.{func.__name__}"
            span_attributes = attributes or {}

            with tracer.start_as_current_span(span_name, attributes=span_attributes) as span:
                try:
                    result = func(*args, **kwargs)
                    span.set_attribute("function.result", "success")
                    return result
                except Exception as e:
                    span.set_attribute("function.result", "error")
                    span.set_attribute("error", True)
                    span.set_attribute("error.message", str(e))
                    span.record_exception(e)
                    raise

        return wrapper

    return decorator


@contextmanager
def trace_span(
    name: str,
    attributes: Optional[dict] = None,
    enabled: Optional[bool] = None,
):
    """
    Context manager for creating a trace span.

    Args:
        name: Span name.
        attributes: Additional span attributes.
        enabled: Override global OpenTelemetry enabled setting.

    Example:
        with trace_span("data_sync", attributes={"stock_code": "000001.SZ"}):
            sync_stock_data()
    """
    config = get_config()
    if enabled is False or (enabled is None and not config.is_enabled()):
        yield
        return

    tracer = get_tracer(__name__)
    if tracer is None:
        yield
        return

    span_attributes = attributes or {}
    with tracer.start_as_current_span(name, attributes=span_attributes) as span:
        try:
            yield span
        except Exception as e:
            span.set_attribute("error", True)
            span.set_attribute("error.message", str(e))
            span.record_exception(e)
            raise


class MetricsCounter:
    """Counter metric helper."""

    def __init__(self, name: str, description: str = "", unit: str = "1"):
        """
        Initialize a counter metric.

        Args:
            name: Metric name.
            description: Metric description.
            unit: Metric unit.
        """
        self.name = name
        self.description = description
        self.unit = unit
        self._counter = None
        self._initialize()

    def _initialize(self):
        """Initialize the counter."""
        meter = get_meter(__name__)
        if meter is None:
            return

        try:
            self._counter = meter.create_counter(
                name=self.name, description=self.description, unit=self.unit
            )
        except Exception as e:
            logger.warning(f"Failed to create counter {self.name}: {e}")

    def add(self, value: float = 1.0, attributes: Optional[dict] = None):
        """
        Add to the counter.

        Args:
            value: Value to add.
            attributes: Metric attributes.
        """
        if self._counter is None:
            return

        try:
            self._counter.add(value, attributes=attributes or {})
        except Exception as e:
            logger.debug(f"Failed to record counter {self.name}: {e}")


class MetricsHistogram:
    """Histogram metric helper."""

    def __init__(self, name: str, description: str = "", unit: str = "ms"):
        """
        Initialize a histogram metric.

        Args:
            name: Metric name.
            description: Metric description.
            unit: Metric unit.
        """
        self.name = name
        self.description = description
        self.unit = unit
        self._histogram = None
        self._initialize()

    def _initialize(self):
        """Initialize the histogram."""
        meter = get_meter(__name__)
        if meter is None:
            return

        try:
            self._histogram = meter.create_histogram(
                name=self.name, description=self.description, unit=self.unit
            )
        except Exception as e:
            logger.warning(f"Failed to create histogram {self.name}: {e}")

    def record(self, value: float, attributes: Optional[dict] = None):
        """
        Record a value.

        Args:
            value: Value to record.
            attributes: Metric attributes.
        """
        if self._histogram is None:
            return

        try:
            self._histogram.record(value, attributes=attributes or {})
        except Exception as e:
            logger.debug(f"Failed to record histogram {self.name}: {e}")


def record_duration(metric_name: str, attributes: Optional[dict] = None):
    """
    Decorator to record function execution duration.

    Args:
        metric_name: Metric name for duration.
        attributes: Additional metric attributes.

    Example:
        @record_duration("data_sync.duration", attributes={"component": "ingestor"})
        def sync_data():
            ...
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            import time

            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration_ms = (time.time() - start_time) * 1000

                histogram = MetricsHistogram(metric_name, unit="ms")
                histogram.record(duration_ms, attributes=attributes)

                return result
            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000

                histogram = MetricsHistogram(metric_name, unit="ms")
                error_attrs = (attributes or {}).copy()
                error_attrs["error"] = True
                histogram.record(duration_ms, attributes=error_attrs)

                raise

        return wrapper

    return decorator


"""
Example: Using OpenTelemetry in ATM project.

Demonstrates how to use OpenTelemetry for logs, metrics, and traces.
"""

import logging
import os
import time

from atm.utils.otel import initialize_otel, get_tracer, get_meter
from atm.utils.otel_helpers import (
    MetricsCounter,
    MetricsHistogram,
    record_duration,
    trace_function,
    trace_span,
)
from atm.utils.otel_logger import setup_logger

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@trace_function(attributes={"component": "example", "operation": "demo"})
@record_duration("example.duration", attributes={"component": "example"})
def example_function():
    """Example function with tracing and metrics."""
    logger.info("Executing example function")
    time.sleep(0.1)  # Simulate work
    return "success"


def main():
    """Run OpenTelemetry example."""
    # Initialize OpenTelemetry
    # Can be controlled via environment variables:
    #   export ATM_OTEL_ENABLED=true
    #   export ATM_OTEL_ENDPOINT=localhost:4317
    #   export ATM_SERVICE_NAME=atm-example

    initialize_otel(
        service_name="atm-example",
        service_version="0.1.0",
        endpoint=os.getenv("ATM_OTEL_ENDPOINT", "localhost:4317"),
        enabled=os.getenv("ATM_OTEL_ENABLED", "false").lower() == "true",
    )

    # Setup logger (format stays the same)
    logger = setup_logger("example")
    logger.info("OpenTelemetry example started")

    # Example 1: Using decorators
    logger.info("Example 1: Using decorators")
    result = example_function()
    logger.info(f"Result: {result}")

    # Example 2: Using context manager
    logger.info("Example 2: Using context manager")
    with trace_span("manual_operation", attributes={"type": "manual"}):
        logger.info("Inside trace span")
        time.sleep(0.05)

    # Example 3: Manual tracing
    logger.info("Example 3: Manual tracing")
    tracer = get_tracer(__name__)
    if tracer:
        with tracer.start_as_current_span("manual_trace") as span:
            span.set_attribute("custom.attribute", "value")
            logger.info("Inside manual trace")
            time.sleep(0.05)

    # Example 4: Metrics - Counter
    logger.info("Example 4: Metrics - Counter")
    counter = MetricsCounter("example.operations.total", description="Total operations")
    counter.add(1, attributes={"status": "success"})
    counter.add(1, attributes={"status": "error"})

    # Example 5: Metrics - Histogram
    logger.info("Example 5: Metrics - Histogram")
    histogram = MetricsHistogram("example.duration", description="Operation duration", unit="ms")
    histogram.record(150.5, attributes={"operation": "data_sync"})
    histogram.record(200.3, attributes={"operation": "data_sync"})

    # Example 6: Using meter directly
    logger.info("Example 6: Using meter directly")
    meter = get_meter(__name__)
    if meter:
        counter = meter.create_counter("example.custom.counter", description="Custom counter")
        counter.add(1, attributes={"source": "example"})

    logger.info("OpenTelemetry example completed")


if __name__ == "__main__":
    main()


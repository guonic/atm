"""Utility modules for ATM project."""

from atm.utils.logger import setup_logger

# OpenTelemetry utilities (optional)
try:
    from atm.utils.otel import (
        get_config,
        get_meter,
        get_tracer,
        initialize_otel,
        shutdown_otel,
    )
    from atm.utils.otel_helpers import (
        MetricsCounter,
        MetricsHistogram,
        record_duration,
        trace_function,
        trace_span,
    )
    from atm.utils.otel_logger import setup_logger as setup_logger_with_otel

    __all__ = [
        "setup_logger",
        "setup_logger_with_otel",
        "initialize_otel",
        "shutdown_otel",
        "get_config",
        "get_tracer",
        "get_meter",
        "trace_function",
        "trace_span",
        "record_duration",
        "MetricsCounter",
        "MetricsHistogram",
    ]
except ImportError:
    __all__ = ["setup_logger"]



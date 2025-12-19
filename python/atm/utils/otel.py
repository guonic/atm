"""
OpenTelemetry integration for ATM project.

Provides unified observability (logs, metrics, traces) across Python, Go, and C++ modules.
All modules follow OpenTelemetry standards and can be optionally enabled.
"""

import logging
import os
from typing import Optional

from opentelemetry import trace, metrics
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader, ConsoleMetricExporter
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.sdk.resources import Resource, SERVICE_NAME, SERVICE_VERSION
from opentelemetry.instrumentation.logging import LoggingInstrumentor

logger = logging.getLogger(__name__)


class OpenTelemetryConfig:
    """OpenTelemetry configuration."""

    def __init__(self):
        """Initialize configuration from environment variables."""
        self.enabled = os.getenv("ATM_OTEL_ENABLED", "false").lower() == "true"
        self.endpoint = os.getenv("ATM_OTEL_ENDPOINT", "http://localhost:4317")
        self.service_name = os.getenv("ATM_SERVICE_NAME", "atm")
        self.service_version = os.getenv("ATM_SERVICE_VERSION", "0.1.0")
        self.environment = os.getenv("ATM_ENVIRONMENT", "development")
        self.trace_sampling_ratio = float(os.getenv("ATM_OTEL_TRACE_SAMPLING", "1.0"))
        self.logs_enabled = os.getenv("ATM_OTEL_LOGS_ENABLED", "true").lower() == "true"
        self.metrics_enabled = os.getenv("ATM_OTEL_METRICS_ENABLED", "true").lower() == "true"
        self.traces_enabled = os.getenv("ATM_OTEL_TRACES_ENABLED", "true").lower() == "true"

    def is_enabled(self) -> bool:
        """Check if OpenTelemetry is enabled."""
        return self.enabled


_global_config: Optional[OpenTelemetryConfig] = None
_tracer_provider: Optional[TracerProvider] = None
_meter_provider: Optional[MeterProvider] = None
_initialized = False


def get_config() -> OpenTelemetryConfig:
    """Get global OpenTelemetry configuration."""
    global _global_config
    if _global_config is None:
        _global_config = OpenTelemetryConfig()
    return _global_config


def initialize_otel(
    service_name: Optional[str] = None,
    service_version: Optional[str] = None,
    endpoint: Optional[str] = None,
    enabled: Optional[bool] = None,
) -> bool:
    """
    Initialize OpenTelemetry.

    Args:
        service_name: Service name (overrides environment variable).
        service_version: Service version (overrides environment variable).
        endpoint: OpenTelemetry Collector endpoint (overrides environment variable).
        enabled: Whether to enable OpenTelemetry (overrides environment variable).

    Returns:
        True if initialization was successful, False otherwise.
    """
    global _tracer_provider, _meter_provider, _initialized

    if _initialized:
        logger.warning("OpenTelemetry already initialized")
        return True

    config = get_config()

    # Override with parameters if provided
    if enabled is not None:
        config.enabled = enabled
    if service_name is not None:
        config.service_name = service_name
    if service_version is not None:
        config.service_version = service_version
    if endpoint is not None:
        config.endpoint = endpoint

    if not config.is_enabled():
        logger.info("OpenTelemetry is disabled")
        return False

    try:
        # Create resource with service information
        resource = Resource.create(
            {
                SERVICE_NAME: config.service_name,
                SERVICE_VERSION: config.service_version,
                "environment": config.environment,
                "deployment.environment": config.environment,
            }
        )

        # Initialize Tracer Provider (Traces)
        if config.traces_enabled:
            _tracer_provider = TracerProvider(resource=resource)

            # Add span processor
            if config.endpoint:
                try:
                    otlp_exporter = OTLPSpanExporter(endpoint=config.endpoint)
                    _tracer_provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
                    logger.info(f"OpenTelemetry traces enabled, exporting to {config.endpoint}")
                except Exception as e:
                    logger.warning(f"Failed to create OTLP exporter, using console: {e}")
                    console_exporter = ConsoleSpanExporter()
                    _tracer_provider.add_span_processor(BatchSpanProcessor(console_exporter))
            else:
                console_exporter = ConsoleSpanExporter()
                _tracer_provider.add_span_processor(BatchSpanProcessor(console_exporter))

            trace.set_tracer_provider(_tracer_provider)

        # Initialize Meter Provider (Metrics)
        if config.metrics_enabled:
            if config.endpoint:
                try:
                    metric_exporter = OTLPMetricExporter(endpoint=config.endpoint)
                    metric_reader = PeriodicExportingMetricReader(
                        metric_exporter, export_interval_millis=5000
                    )
                    _meter_provider = MeterProvider(
                        resource=resource, metric_readers=[metric_reader]
                    )
                    logger.info(f"OpenTelemetry metrics enabled, exporting to {config.endpoint}")
                except Exception as e:
                    logger.warning(f"Failed to create OTLP metric exporter, using console: {e}")
                    console_exporter = ConsoleMetricExporter()
                    metric_reader = PeriodicExportingMetricReader(
                        console_exporter, export_interval_millis=5000
                    )
                    _meter_provider = MeterProvider(
                        resource=resource, metric_readers=[metric_reader]
                    )
            else:
                console_exporter = ConsoleMetricExporter()
                metric_reader = PeriodicExportingMetricReader(
                    console_exporter, export_interval_millis=5000
                )
                _meter_provider = MeterProvider(
                    resource=resource, metric_readers=[metric_reader]
                )

            metrics.set_meter_provider(_meter_provider)

        # Instrument logging (Logs)
        if config.logs_enabled:
            # This adds trace context to logs but doesn't change log format
            LoggingInstrumentor().instrument(set_logging_format=False)
            logger.info("OpenTelemetry logging instrumentation enabled")

        _initialized = True
        logger.info(
            f"OpenTelemetry initialized: service={config.service_name}, "
            f"version={config.service_version}, endpoint={config.endpoint}"
        )
        return True

    except Exception as e:
        logger.error(f"Failed to initialize OpenTelemetry: {e}", exc_info=True)
        return False


def get_tracer(name: str):
    """
    Get a tracer instance.

    Args:
        name: Tracer name (usually module name).

    Returns:
        Tracer instance or None if OpenTelemetry is not enabled.
    """
    config = get_config()
    if not config.is_enabled() or not config.traces_enabled:
        return None

    try:
        return trace.get_tracer(name)
    except Exception as e:
        logger.warning(f"Failed to get tracer: {e}")
        return None


def get_meter(name: str):
    """
    Get a meter instance for metrics.

    Args:
        name: Meter name (usually module name).

    Returns:
        Meter instance or None if OpenTelemetry is not enabled.
    """
    config = get_config()
    if not config.is_enabled() or not config.metrics_enabled:
        return None

    try:
        return metrics.get_meter(name)
    except Exception as e:
        logger.warning(f"Failed to get meter: {e}")
        return None


def shutdown_otel():
    """Shutdown OpenTelemetry."""
    global _tracer_provider, _meter_provider, _initialized

    if not _initialized:
        return

    try:
        if _tracer_provider:
            _tracer_provider.shutdown()
        if _meter_provider:
            _meter_provider.shutdown()
        _initialized = False
        logger.info("OpenTelemetry shut down")
    except Exception as e:
        logger.warning(f"Error shutting down OpenTelemetry: {e}")


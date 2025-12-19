// Package otel provides OpenTelemetry integration for ATM project.
// Supports logs, metrics, and traces following OpenTelemetry standards.
package otel

import (
	"context"
	"fmt"
	"os"
	"strconv"
	"time"

	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracegrpc"
	"go.opentelemetry.io/otel/exporters/otlp/otlpmetric/otlpmetricgrpc"
	"go.opentelemetry.io/otel/exporters/stdout/stdoutmetric"
	"go.opentelemetry.io/otel/exporters/stdout/stdouttrace"
	"go.opentelemetry.io/otel/propagation"
	"go.opentelemetry.io/otel/sdk/metric"
	"go.opentelemetry.io/otel/sdk/resource"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	semconv "go.opentelemetry.io/otel/semconv/v1.21.0"
	"go.opentelemetry.io/otel/trace"
)

// Config holds OpenTelemetry configuration.
type Config struct {
	Enabled           bool
	Endpoint          string
	ServiceName       string
	ServiceVersion    string
	Environment       string
	TraceSampling     float64
	LogsEnabled       bool
	MetricsEnabled    bool
	TracesEnabled     bool
	MetricsInterval   time.Duration
}

// LoadConfig loads configuration from environment variables.
func LoadConfig() *Config {
	enabled, _ := strconv.ParseBool(getEnv("ATM_OTEL_ENABLED", "false"))
	traceSampling, _ := strconv.ParseFloat(getEnv("ATM_OTEL_TRACE_SAMPLING", "1.0"), 64)
	logsEnabled, _ := strconv.ParseBool(getEnv("ATM_OTEL_LOGS_ENABLED", "true"))
	metricsEnabled, _ := strconv.ParseBool(getEnv("ATM_OTEL_METRICS_ENABLED", "true"))
	tracesEnabled, _ := strconv.ParseBool(getEnv("ATM_OTEL_TRACES_ENABLED", "true"))
	metricsInterval, _ := time.ParseDuration(getEnv("ATM_OTEL_METRICS_INTERVAL", "5s"))

	return &Config{
		Enabled:         enabled,
		Endpoint:        getEnv("ATM_OTEL_ENDPOINT", "localhost:4317"),
		ServiceName:     getEnv("ATM_SERVICE_NAME", "atm"),
		ServiceVersion:  getEnv("ATM_SERVICE_VERSION", "0.1.0"),
		Environment:     getEnv("ATM_ENVIRONMENT", "development"),
		TraceSampling:   traceSampling,
		LogsEnabled:     logsEnabled,
		MetricsEnabled:  metricsEnabled,
		TracesEnabled:   tracesEnabled,
		MetricsInterval: metricsInterval,
	}
}

func getEnv(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}

var (
	globalConfig     *Config
	tracerProvider   *sdktrace.TracerProvider
	meterProvider    *metric.MeterProvider
	initialized      bool
)

// Initialize initializes OpenTelemetry.
func Initialize(ctx context.Context, config *Config) error {
	if initialized {
		return fmt.Errorf("OpenTelemetry already initialized")
	}

	if config == nil {
		config = LoadConfig()
	}
	globalConfig = config

	if !config.Enabled {
		return nil
	}

	// Create resource
	res, err := resource.New(ctx,
		resource.WithAttributes(
			semconv.ServiceNameKey.String(config.ServiceName),
			semconv.ServiceVersionKey.String(config.ServiceVersion),
			semconv.DeploymentEnvironmentKey.String(config.Environment),
		),
	)
	if err != nil {
		return fmt.Errorf("failed to create resource: %w", err)
	}

	// Initialize Tracer Provider
	if config.TracesEnabled {
		var traceExporter sdktrace.SpanExporter
		if config.Endpoint != "" {
			exporter, err := otlptracegrpc.New(ctx,
				otlptracegrpc.WithEndpoint(config.Endpoint),
				otlptracegrpc.WithInsecure(),
			)
			if err != nil {
				// Fallback to console exporter
				exporter, _ = stdouttrace.New()
			}
			traceExporter = exporter
		} else {
			traceExporter, _ = stdouttrace.New()
		}

		tracerProvider = sdktrace.NewTracerProvider(
			sdktrace.WithBatcher(traceExporter),
			sdktrace.WithResource(res),
			sdktrace.WithSampler(sdktrace.TraceIDRatioBased(config.TraceSampling)),
		)
		otel.SetTracerProvider(tracerProvider)
		otel.SetTextMapPropagator(propagation.NewCompositeTextMapPropagator(
			propagation.TraceContext{},
			propagation.Baggage{},
		))
	}

	// Initialize Meter Provider
	if config.MetricsEnabled {
		var metricExporter metric.Exporter
		if config.Endpoint != "" {
			exporter, err := otlpmetricgrpc.New(ctx,
				otlpmetricgrpc.WithEndpoint(config.Endpoint),
				otlpmetricgrpc.WithInsecure(),
			)
			if err != nil {
				// Fallback to console exporter
				exporter, _ = stdoutmetric.New()
			}
			metricExporter = exporter
		} else {
			metricExporter, _ = stdoutmetric.New()
		}

		meterProvider = metric.NewMeterProvider(
			metric.WithResource(res),
			metric.WithReader(metric.NewPeriodicReader(metricExporter,
				metric.WithInterval(config.MetricsInterval))),
		)
		otel.SetMeterProvider(meterProvider)
	}

	initialized = true
	return nil
}

// GetTracer returns a tracer instance.
func GetTracer(name string) trace.Tracer {
	if !initialized || globalConfig == nil || !globalConfig.TracesEnabled {
		return trace.NewNoopTracerProvider().Tracer(name)
	}
	return otel.Tracer(name)
}

// GetMeter returns a meter instance.
func GetMeter(name string) metric.Meter {
	if !initialized || globalConfig == nil || !globalConfig.MetricsEnabled {
		return metric.NewNoopMeterProvider().Meter(name)
	}
	return otel.Meter(name)
}

// Shutdown shuts down OpenTelemetry.
func Shutdown(ctx context.Context) error {
	if !initialized {
		return nil
	}

	var errs []error
	if tracerProvider != nil {
		if err := tracerProvider.Shutdown(ctx); err != nil {
			errs = append(errs, fmt.Errorf("tracer provider shutdown: %w", err))
		}
	}
	if meterProvider != nil {
		if err := meterProvider.Shutdown(ctx); err != nil {
			errs = append(errs, fmt.Errorf("meter provider shutdown: %w", err))
		}
	}

	initialized = false
	if len(errs) > 0 {
		return fmt.Errorf("shutdown errors: %v", errs)
	}
	return nil
}

// IsEnabled returns whether OpenTelemetry is enabled.
func IsEnabled() bool {
	return initialized && globalConfig != nil && globalConfig.Enabled
}


# OpenTelemetry Integration for Go

## Installation

Add OpenTelemetry dependencies to `go.mod`:

```bash
go get go.opentelemetry.io/otel@latest
go get go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracegrpc@latest
go get go.opentelemetry.io/otel/exporters/otlp/otlpmetric/otlpmetricgrpc@latest
go get go.opentelemetry.io/otel/sdk@latest
go get go.opentelemetry.io/otel/semconv/v1.21.0@latest
```

## Usage

See [docs/OPENTELEMETRY_INTEGRATION.md](../../../docs/OPENTELEMETRY_INTEGRATION.md) for detailed usage examples.


#!/bin/bash
# Setup OpenTelemetry dependencies for all languages

set -e

echo "Setting up OpenTelemetry dependencies..."

# Python
echo "Installing Python OpenTelemetry dependencies..."
cd python
pip install -e ".[otel]" || echo "Warning: Failed to install Python OpenTelemetry dependencies"
cd ..

# Golang
echo "Installing Go OpenTelemetry dependencies..."
cd go
go get go.opentelemetry.io/otel@latest || echo "Warning: Failed to install Go OpenTelemetry dependencies"
go get go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracegrpc@latest
go get go.opentelemetry.io/otel/exporters/otlp/otlpmetric/otlpmetricgrpc@latest
go get go.opentelemetry.io/otel/exporters/stdout/stdouttrace@latest
go get go.opentelemetry.io/otel/exporters/stdout/stdoutmetric@latest
go get go.opentelemetry.io/otel/sdk@latest
go get go.opentelemetry.io/otel/semconv/v1.21.0@latest
go mod tidy
cd ..

# C++
echo "C++ OpenTelemetry dependencies need to be installed manually."
echo "See: https://opentelemetry.io/docs/instrumentation/cpp/"
echo "The current implementation uses stub classes that compile without OpenTelemetry SDK."

echo "Done!"


// Package otel provides helper functions for OpenTelemetry.
package otel

import (
	"context"
	"time"

	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/metric"
	"go.opentelemetry.io/otel/trace"
)

// TraceFunc traces a function execution.
func TraceFunc(ctx context.Context, name string, fn func(context.Context) error, attrs ...attribute.KeyValue) error {
	if !IsEnabled() {
		return fn(ctx)
	}

	tracer := GetTracer("nexusquant")
	ctx, span := tracer.Start(ctx, name, trace.WithAttributes(attrs...))
	defer span.End()

	err := fn(ctx)
	if err != nil {
		span.RecordError(err)
		span.SetAttributes(attribute.Bool("error", true))
		span.SetAttributes(attribute.String("error.message", err.Error()))
	}
	return err
}

// RecordDuration records function execution duration.
func RecordDuration(ctx context.Context, metricName string, fn func(context.Context) error, attrs ...attribute.KeyValue) error {
	start := time.Now()
	err := fn(ctx)
	duration := time.Since(start)

	if IsEnabled() {
		meter := GetMeter("nexusquant")
		histogram, createErr := meter.Float64Histogram(metricName, metric.WithDescription("Function execution duration"))
		if createErr == nil && histogram != nil {
			histogram.Record(ctx, float64(duration.Milliseconds()), metric.WithAttributes(attrs...))
		}
	}

	return err
}

// Counter is a counter metric helper.
type Counter struct {
	counter metric.Int64Counter
	name    string
}

// NewCounter creates a new counter.
func NewCounter(name, description string) *Counter {
	if !IsEnabled() {
		return &Counter{}
	}

	meter := GetMeter("nexusquant")
	counter, _ := meter.Int64Counter(name, metric.WithDescription(description))
	return &Counter{
		counter: counter,
		name:    name,
	}
}

// Add adds to the counter.
func (c *Counter) Add(ctx context.Context, value int64, attrs ...attribute.KeyValue) {
	if c.counter != nil {
		c.counter.Add(ctx, value, metric.WithAttributes(attrs...))
	}
}

// Histogram is a histogram metric helper.
type Histogram struct {
	histogram metric.Float64Histogram
	name      string
}

// NewHistogram creates a new histogram.
func NewHistogram(name, description, unit string) *Histogram {
	if !IsEnabled() {
		return &Histogram{}
	}

	meter := GetMeter("nexusquant")
	histogram, _ := meter.Float64Histogram(name, metric.WithDescription(description), metric.WithUnit(unit))
	return &Histogram{
		histogram: histogram,
		name:      name,
	}
}

// Record records a value.
func (h *Histogram) Record(ctx context.Context, value float64, attrs ...attribute.KeyValue) {
	if h.histogram != nil {
		h.histogram.Record(ctx, value, metric.WithAttributes(attrs...))
	}
}

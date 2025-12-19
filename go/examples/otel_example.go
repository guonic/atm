package main

import (
	"context"
	"fmt"
	"log"
	"time"

	"github.com/atm/atm/pkg/utils/otel"
	"go.opentelemetry.io/otel/attribute"
)

func main() {
	ctx := context.Background()

	// Initialize OpenTelemetry
	// Can be controlled via environment variables:
	//   export ATM_OTEL_ENABLED=true
	//   export ATM_OTEL_ENDPOINT=localhost:4317
	//   export ATM_SERVICE_NAME=atm-example

	if err := otel.Initialize(ctx, nil); err != nil {
		log.Fatalf("Failed to initialize OpenTelemetry: %v", err)
	}
	defer otel.Shutdown(ctx)

	// Example 1: Using TraceFunc helper
	err := otel.TraceFunc(ctx, "example.operation", func(ctx context.Context) error {
		fmt.Println("Executing operation...")
		time.Sleep(100 * time.Millisecond)
		return nil
	}, attribute.String("component", "example"))
	if err != nil {
		log.Printf("Operation failed: %v", err)
	}

	// Example 2: Manual tracing
	tracer := otel.GetTracer("example")
	ctx, span := tracer.Start(ctx, "manual.operation", attribute.String("type", "manual"))
	defer span.End()

	fmt.Println("Inside manual trace")
	time.Sleep(50 * time.Millisecond)

	// Example 3: Metrics - Counter
	counter := otel.NewCounter("example.operations.total", "Total operations")
	counter.Add(ctx, 1, attribute.String("status", "success"))
	counter.Add(ctx, 1, attribute.String("status", "error"))

	// Example 4: Metrics - Histogram
	histogram := otel.NewHistogram("example.duration", "Operation duration", "ms")
	histogram.Record(ctx, 150.5, attribute.String("operation", "data_sync"))
	histogram.Record(ctx, 200.3, attribute.String("operation", "data_sync"))

	// Example 5: Record duration
	err = otel.RecordDuration(ctx, "example.duration", func(ctx context.Context) error {
		time.Sleep(100 * time.Millisecond)
		return nil
	}, attribute.String("component", "example"))
	if err != nil {
		log.Printf("Duration recording failed: %v", err)
	}

	fmt.Println("OpenTelemetry example completed")
}


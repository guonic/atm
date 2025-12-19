// +build !otel

package otel

import (
	"context"
	"testing"
)

// Test that code compiles without OpenTelemetry dependencies
// This test file is only compiled when otel build tag is not set
func TestCompilation(t *testing.T) {
	ctx := context.Background()
	config := LoadConfig()
	
	// Test that functions can be called without OpenTelemetry
	_ = Initialize(ctx, config)
	_ = IsEnabled()
	_ = GetTracer("test")
	_ = GetMeter("test")
	_ = Shutdown(ctx)
}


// Example: Using OpenTelemetry in C++
#include "atm/utils/otel.hpp"
#include <iostream>

int main() {
    // Load configuration from environment variables
    auto config = atm::utils::otel::LoadConfig();
    
    // Initialize OpenTelemetry
    if (!atm::utils::otel::Initialize(config)) {
        std::cout << "OpenTelemetry initialization failed or disabled" << std::endl;
        return 1;
    }
    
    // Example 1: Using tracer
    auto tracer = atm::utils::otel::GetTracer("example");
    if (tracer) {
        auto span = tracer->StartSpan("example_operation", {
            {"component", "example"},
            {"operation", "test"}
        });
        
        if (span) {
            span->SetAttribute("status", "success");
            span->End();
        }
    }
    
    // Example 2: Using meter
    auto meter = atm::utils::otel::GetMeter("example");
    if (meter) {
        auto counter = meter->CreateCounter("example.operations", "Total operations");
        if (counter) {
            counter->Add(1, {{"status", "success"}});
        }
        
        auto histogram = meter->CreateHistogram("example.duration", "Operation duration", "ms");
        if (histogram) {
            histogram->Record(150.5, {{"operation", "test"}});
        }
    }
    
    // Example 3: Using RAII helper
    {
        auto tracer2 = atm::utils::otel::GetTracer("example");
        if (tracer2) {
            auto span = tracer2->StartSpan("raii_example");
            atm::utils::otel::SpanScope span_scope(std::move(span));
            
            if (span_scope.get()) {
                span_scope->SetAttribute("key", "value");
                // Span automatically ends when scope exits
            }
        }
    }
    
    // Shutdown
    atm::utils::otel::Shutdown();
    
    std::cout << "OpenTelemetry example completed" << std::endl;
    return 0;
}


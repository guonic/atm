#pragma once

#include <memory>
#include <string>
#include <unordered_map>

namespace nq {
namespace utils {
namespace otel {

/**
 * OpenTelemetry configuration.
 */
struct Config {
    bool enabled = false;
    std::string endpoint = "localhost:4317";
    std::string service_name = "nexusquant";
    std::string service_version = "0.1.0";
    std::string environment = "development";
    double trace_sampling = 1.0;
    bool logs_enabled = true;
    bool metrics_enabled = true;
    bool traces_enabled = true;
};

/**
 * Load configuration from environment variables.
 */
Config LoadConfig();

/**
 * Initialize OpenTelemetry.
 * Returns true on success, false otherwise.
 */
bool Initialize(const Config& config = LoadConfig());

/**
 * Shutdown OpenTelemetry.
 */
void Shutdown();

/**
 * Check if OpenTelemetry is enabled.
 */
bool IsEnabled();

/**
 * Tracer interface for creating spans.
 */
class Tracer {
public:
    virtual ~Tracer() = default;
    
    /**
     * Start a new span.
     */
    virtual std::unique_ptr<class Span> StartSpan(
        const std::string& name,
        const std::unordered_map<std::string, std::string>& attributes = {}
    ) = 0;
};

/**
 * Span interface for tracing operations.
 */
class Span {
public:
    virtual ~Span() = default;
    
    /**
     * Set span attribute.
     */
    virtual void SetAttribute(const std::string& key, const std::string& value) = 0;
    
    /**
     * Set span attribute (numeric).
     */
    virtual void SetAttribute(const std::string& key, double value) = 0;
    
    /**
     * Record an exception.
     */
    virtual void RecordException(const std::string& message) = 0;
    
    /**
     * End the span.
     */
    virtual void End() = 0;
};

/**
 * Get a tracer instance.
 */
std::unique_ptr<Tracer> GetTracer(const std::string& name);

/**
 * Meter interface for metrics.
 */
class Meter {
public:
    virtual ~Meter() = default;
    
    /**
     * Create a counter metric.
     */
    virtual std::unique_ptr<class Counter> CreateCounter(
        const std::string& name,
        const std::string& description = ""
    ) = 0;
    
    /**
     * Create a histogram metric.
     */
    virtual std::unique_ptr<class Histogram> CreateHistogram(
        const std::string& name,
        const std::string& description = "",
        const std::string& unit = "ms"
    ) = 0;
};

/**
 * Counter metric interface.
 */
class Counter {
public:
    virtual ~Counter() = default;
    
    /**
     * Add to the counter.
     */
    virtual void Add(double value, const std::unordered_map<std::string, std::string>& attributes = {}) = 0;
};

/**
 * Histogram metric interface.
 */
class Histogram {
public:
    virtual ~Histogram() = default;
    
    /**
     * Record a value.
     */
    virtual void Record(double value, const std::unordered_map<std::string, std::string>& attributes = {}) = 0;
};

/**
 * Get a meter instance.
 */
std::unique_ptr<Meter> GetMeter(const std::string& name);

/**
 * RAII helper for automatic span management.
 */
class SpanScope {
public:
    SpanScope(std::unique_ptr<Span> span) : span_(std::move(span)) {}
    ~SpanScope() { if (span_) span_->End(); }
    
    Span* operator->() { return span_.get(); }
    Span* get() { return span_.get(); }
    
private:
    std::unique_ptr<Span> span_;
};

} // namespace otel
} // namespace utils
} // namespace nq


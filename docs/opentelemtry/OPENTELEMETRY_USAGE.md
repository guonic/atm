# OpenTelemetry 使用指南

## 在功能代码中启用 OpenTelemetry

本文档说明如何在各个功能模块中酌情启用 OpenTelemetry。

## 原则

1. **可选启用**：通过环境变量控制，默认关闭
2. **不影响现有功能**：不启用时，代码行为完全一致
3. **性能友好**：使用采样和异步导出
4. **符合标准**：日志、指标、追踪都符合 OpenTelemetry 标准

## Python 模块集成示例

### 数据同步服务

```python
# python/tools/dataingestor/service/stock_ingestor_service.py
from nq.utils.otel import get_tracer
from nq.utils.otel_helpers import trace_span, MetricsCounter

class StockIngestorService:
    def __init__(self):
        self.tracer = get_tracer(__name__)
        self.sync_counter = MetricsCounter("data.sync.total", "Total sync operations")
        self.error_counter = MetricsCounter("data.sync.errors", "Sync errors")
    
    def ingest_stock_basic(self, ...):
        # 使用 trace_span 追踪整个同步过程
        with trace_span("stock_basic.ingest", attributes={
            "exchange": exchange,
            "list_status": list_status,
        }) as span:
            try:
                # 原有代码
                ...
                self.sync_counter.add(1, {"status": "success"})
            except Exception as e:
                self.error_counter.add(1, {"error_type": type(e).__name__})
                span.record_exception(e)
                raise
```

### API 服务

```python
# python/atm/api/rest/main.py
from nq.utils.otel import initialize_otel, get_tracer
from nq.utils.otel_helpers import trace_function

# 在应用启动时初始化
initialize_otel()

@trace_function(attributes={"component": "api", "endpoint": "/api/v1/orders"})
def create_order(request):
    tracer = get_tracer(__name__)
    with tracer.start_as_current_span("order.create") as span:
        span.set_attribute("order.type", request.order_type)
        span.set_attribute("order.symbol", request.symbol)
        # 原有代码
        ...
```

### 交易策略

```python
# python/atm/trading/strategy/backtrader_strategy.py
from nq.utils.otel_helpers import record_duration, trace_span

class BacktraderStrategy:
    @record_duration("strategy.backtest.duration", attributes={"strategy": "sma_cross"})
    def run(self):
        with trace_span("strategy.backtest", attributes={
            "strategy_name": self.name,
            "initial_cash": self.initial_cash,
        }):
            # 原有代码
            ...
```

## Golang 模块集成示例

### 数据采集服务

```go
// go/cmd/collector/main.go
package main

import (
    "context"
    "github.com/atm/atm/pkg/utils/otel"
    "go.opentelemetry.io/otel/attribute"
)

func main() {
    ctx := context.Background()
    
    // 初始化
    if err := otel.Initialize(ctx, nil); err != nil {
        log.Fatal(err)
    }
    defer otel.Shutdown(ctx)
    
    // 使用
    syncData(ctx)
}

func syncData(ctx context.Context) error {
    return otel.TraceFunc(ctx, "data.sync", func(ctx context.Context) error {
        // 原有代码
        return nil
    }, attribute.String("component", "collector"))
}
```

### 交易执行服务

```go
// go/internal/trading/execution/executor.go
package execution

import (
    "context"
    "github.com/atm/atm/pkg/utils/otel"
    "go.opentelemetry.io/otel/attribute"
)

func ExecuteOrder(ctx context.Context, order *Order) error {
    tracer := otel.GetTracer("atm.trading.execution")
    ctx, span := tracer.Start(ctx, "order.execute", trace.WithAttributes(
        attribute.String("order.id", order.ID),
        attribute.String("order.type", order.Type),
        attribute.String("order.symbol", order.Symbol),
    ))
    defer span.End()
    
    // 原有代码
    ...
    
    // 记录指标
    counter := otel.NewCounter("orders.executed", "Executed orders")
    counter.Add(ctx, 1, attribute.String("status", "success"))
    
    return nil
}
```

## C++ 模块集成示例

### 高性能计算模块

```cpp
// cpp/src/trading/strategy.cpp
#include "nq/utils/otel.hpp"

void ExecuteStrategy(const StrategyConfig& config) {
    if (!nq::utils::otel::IsEnabled()) {
        // 不启用时，直接执行原有逻辑
        ExecuteStrategyInternal(config);
        return;
    }
    
    auto tracer = nq::utils::otel::GetTracer("nexusquant.trading.strategy");
    auto span = tracer->StartSpan("strategy.execute", {
        {"strategy.name", config.name},
        {"strategy.type", config.type},
    });
    
    nq::utils::otel::SpanScope span_scope(std::move(span));
    
    try {
        ExecuteStrategyInternal(config);
        span_scope->SetAttribute("result", "success");
    } catch (const std::exception& e) {
        span_scope->RecordException(e.what());
        span_scope->SetAttribute("result", "error");
        throw;
    }
}
```

## 环境变量配置

### 开发环境

```bash
# .env.development
ATM_OTEL_ENABLED=true
ATM_OTEL_ENDPOINT=localhost:4317
ATM_SERVICE_NAME=atm-dev
ATM_ENVIRONMENT=development
ATM_OTEL_TRACE_SAMPLING=1.0  # 100% 采样用于调试
```

### 生产环境

```bash
# .env.production
ATM_OTEL_ENABLED=true
ATM_OTEL_ENDPOINT=otel-collector:4317
ATM_SERVICE_NAME=atm-prod
ATM_ENVIRONMENT=production
ATM_OTEL_TRACE_SAMPLING=0.1  # 10% 采样降低性能影响
```

### 禁用 OpenTelemetry

```bash
# 不设置或设置为 false
ATM_OTEL_ENABLED=false
# 或不设置环境变量
```

## 最佳实践

### 1. 关键路径追踪

```python
# 追踪关键业务操作
@trace_function(attributes={"component": "trading", "operation": "order_placement"})
def place_order(order):
    ...
```

### 2. 性能指标

```python
# 记录关键操作的耗时
@record_duration("order.execution.duration", attributes={"order_type": "market"})
def execute_order(order):
    ...
```

### 3. 错误追踪

```python
try:
    operation()
except Exception as e:
    span.record_exception(e)
    span.set_attribute("error", True)
    raise
```

### 4. 业务指标

```python
# 记录业务指标
orders_counter = MetricsCounter("orders.total", "Total orders")
orders_counter.add(1, {"status": "filled", "symbol": "000001.SZ"})
```

## 检查清单

在功能代码中集成 OpenTelemetry 时：

- [ ] 使用环境变量控制启用/禁用
- [ ] 不启用时，代码行为完全一致
- [ ] 添加有意义的 span 名称和属性
- [ ] 记录异常和错误
- [ ] 使用采样策略（生产环境）
- [ ] 记录关键业务指标
- [ ] 测试启用和禁用两种情况

## 总结

- ✅ **可选启用**：通过环境变量控制
- ✅ **向后兼容**：不启用时行为一致
- ✅ **符合标准**：日志、指标、追踪都符合 OpenTelemetry 标准
- ✅ **性能友好**：支持采样和异步导出
- ✅ **易于集成**：提供装饰器和辅助函数


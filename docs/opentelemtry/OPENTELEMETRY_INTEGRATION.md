# OpenTelemetry 集成指南

## 概述

NexusQuant 项目已全面集成 OpenTelemetry，支持 Python、Golang、C++ 三语言，提供统一的日志、指标和追踪能力。

## 关键特性

- ✅ **日志格式保持不变**：现有日志格式完全保留
- ✅ **符合 OpenTelemetry 标准**：日志、指标、追踪都符合 OTLP 标准
- ✅ **可选启用**：通过环境变量控制，不影响现有功能
- ✅ **跨语言支持**：Python、Golang、C++ 统一标准
- ✅ **性能友好**：异步导出，采样支持

## 快速开始

### 1. 安装依赖

#### Python
```bash
pip install -e ".[otel]"
```

#### Golang
```bash
cd go
go get go.opentelemetry.io/otel@latest
go get go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracegrpc@latest
go get go.opentelemetry.io/otel/exporters/otlp/otlpmetric/otlpmetricgrpc@latest
go get go.opentelemetry.io/otel/sdk@latest
```

#### C++
```bash
# 需要安装 OpenTelemetry C++ SDK
# 参考: https://opentelemetry.io/docs/instrumentation/cpp/
```

### 2. 配置环境变量

```bash
# 启用 OpenTelemetry
export NQ_OTEL_ENABLED=true

# OpenTelemetry Collector 地址
export NQ_OTEL_ENDPOINT=localhost:4317

# 服务信息
export NQ_SERVICE_NAME=nexusquant-api
export NQ_SERVICE_VERSION=0.1.0
export NQ_ENVIRONMENT=development

# 功能开关（可选）
export NQ_OTEL_LOGS_ENABLED=true
export NQ_OTEL_METRICS_ENABLED=true
export NQ_OTEL_TRACES_ENABLED=true

# 采样率（0.0 到 1.0）
export NQ_OTEL_TRACE_SAMPLING=1.0
```

### 3. 初始化（各语言）

#### Python
```python
from nq.utils.otel import initialize_otel

# 在应用启动时初始化
initialize_otel()

# 或使用自定义配置
initialize_otel(
    service_name="atm-api",
    service_version="0.1.0",
    endpoint="localhost:4317",
    enabled=True
)
```

#### Golang
```go
import "github.com/atm/atm/pkg/utils/otel"

func main() {
    ctx := context.Background()
    
    // 初始化
    if err := otel.Initialize(ctx, nil); err != nil {
        log.Fatal(err)
    }
    defer otel.Shutdown(ctx)
    
    // 使用...
}
```

#### C++
```cpp
#include "nq/utils/otel.hpp"

int main() {
    // 初始化
    auto config = nq::utils::otel::LoadConfig();
    if (!nq::utils::otel::Initialize(config)) {
        return 1;
    }
    
    // 使用...
    
    // 关闭
    nq::utils::otel::Shutdown();
    return 0;
}
```

## 使用示例

### Python

#### 追踪（Traces）
```python
from nq.utils.otel import get_tracer
from nq.utils.otel_helpers import trace_function, trace_span

# 方式 1: 使用装饰器
@trace_function(attributes={"component": "data_ingestor"})
def sync_stock_data():
    # 你的代码
    pass

# 方式 2: 使用上下文管理器
from nq.utils.otel_helpers import trace_span

with trace_span("data_sync", attributes={"stock_code": "000001.SZ"}):
    sync_stock_data()

# 方式 3: 手动创建 span
tracer = get_tracer(__name__)
with tracer.start_as_current_span("operation") as span:
    span.set_attribute("key", "value")
    # 你的代码
```

#### 指标（Metrics）
```python
from nq.utils.otel_helpers import MetricsCounter, MetricsHistogram, record_duration

# 计数器
counter = MetricsCounter("orders.total", description="Total orders")
counter.add(1, attributes={"status": "success"})

# 直方图
histogram = MetricsHistogram("order.duration", unit="ms")
histogram.record(150.5, attributes={"order_type": "market"})

# 自动记录函数执行时间
@record_duration("data_sync.duration", attributes={"component": "ingestor"})
def sync_data():
    # 你的代码
    pass
```

### Golang

#### 追踪（Traces）
```go
import (
    "context"
    "github.com/atm/atm/pkg/utils/otel"
    "go.opentelemetry.io/otel/attribute"
)

// 方式 1: 使用辅助函数
err := otel.TraceFunc(ctx, "sync_stock_data", func(ctx context.Context) error {
    // 你的代码
    return nil
}, attribute.String("component", "data_ingestor"))

// 方式 2: 手动创建 span
tracer := otel.GetTracer("atm")
ctx, span := tracer.Start(ctx, "operation", trace.WithAttributes(
    attribute.String("key", "value"),
))
defer span.End()
// 你的代码
```

#### 指标（Metrics）
```go
import "github.com/atm/atm/pkg/utils/otel"

// 计数器
counter := otel.NewCounter("orders.total", "Total orders")
counter.Add(ctx, 1, attribute.String("status", "success"))

// 直方图
histogram := otel.NewHistogram("order.duration", "Order duration", "ms")
histogram.Record(ctx, 150.5, attribute.String("order_type", "market"))

// 自动记录执行时间
err := otel.RecordDuration(ctx, "data_sync.duration", func(ctx context.Context) error {
    // 你的代码
    return nil
}, attribute.String("component", "ingestor"))
```

### C++

#### 追踪（Traces）
```cpp
#include "nq/utils/otel.hpp"

// 方式 1: 使用 RAII helper
{
    auto tracer = nq::utils::otel::GetTracer("nexusquant");
    auto span = tracer->StartSpan("operation", {{"key", "value"}});
    nq::utils::otel::SpanScope span_scope(std::move(span));
    
    // 你的代码
    span_scope->SetAttribute("status", "success");
}

// 方式 2: 手动管理
auto tracer = nq::utils::otel::GetTracer("nexusquant");
auto span = tracer->StartSpan("operation");
span->SetAttribute("key", "value");
// 你的代码
span->End();
```

#### 指标（Metrics）
```cpp
#include "nq/utils/otel.hpp"

// 计数器
auto meter = nq::utils::otel::GetMeter("nexusquant");
auto counter = meter->CreateCounter("orders.total", "Total orders");
counter->Add(1, {{"status", "success"}});

// 直方图
auto histogram = meter->CreateHistogram("order.duration", "Order duration", "ms");
histogram->Record(150.5, {{"order_type", "market"}});
```

## 日志集成

### Python
```python
from nq.utils.otel_logger import setup_logger

# 日志格式保持不变
logger = setup_logger("my_module")
logger.info("This log format stays the same")
# 输出: 2025-12-19 09:38:54,877 - my_module - INFO - This log format stays the same

# OpenTelemetry 会自动添加 trace context（不影响格式）
```

### Golang
```go
import (
    "go.opentelemetry.io/otel/log"
    "go.opentelemetry.io/otel/log/global"
)

// 使用 OpenTelemetry logger（符合标准）
logger := global.Logger("atm")
logger.Info(ctx, "Operation completed",
    log.String("component", "data_ingestor"),
    log.String("status", "success"),
)
```

## 配置说明

### 环境变量

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `ATM_OTEL_ENABLED` | 是否启用 OpenTelemetry | `false` |
| `ATM_OTEL_ENDPOINT` | Collector 地址 | `localhost:4317` |
| `ATM_SERVICE_NAME` | 服务名称 | `atm` |
| `ATM_SERVICE_VERSION` | 服务版本 | `0.1.0` |
| `ATM_ENVIRONMENT` | 环境 | `development` |
| `ATM_OTEL_LOGS_ENABLED` | 启用日志导出 | `true` |
| `ATM_OTEL_METRICS_ENABLED` | 启用指标导出 | `true` |
| `ATM_OTEL_TRACES_ENABLED` | 启用追踪导出 | `true` |
| `ATM_OTEL_TRACE_SAMPLING` | 追踪采样率 | `1.0` |

## 最佳实践

### 1. 性能关键路径

对于性能敏感的操作，使用采样：

```python
# 只追踪 10% 的请求
export ATM_OTEL_TRACE_SAMPLING=0.1
```

### 2. 选择性启用

```python
# 在 API 服务中启用
if os.getenv("SERVICE_TYPE") == "api":
    initialize_otel()

# 在数据同步任务中不启用（保持轻量）
```

### 3. 结构化属性

```python
# 使用有意义的属性
span.set_attribute("stock.code", "000001.SZ")
span.set_attribute("order.id", order_id)
span.set_attribute("operation.type", "buy")
```

### 4. 错误处理

```python
try:
    # 操作
    pass
except Exception as e:
    span.record_exception(e)
    span.set_attribute("error", True)
    raise
```

## 故障排查

### 检查 OpenTelemetry 是否启用

```python
from nq.utils.otel import get_config
config = get_config()
print(f"Enabled: {config.is_enabled()}")
```

### 查看导出的数据

如果使用 Console Exporter（开发环境），数据会输出到控制台。

### 连接 Collector

确保 OpenTelemetry Collector 正在运行：
```bash
# 使用 Docker 运行 Collector
docker run -p 4317:4317 otel/opentelemetry-collector
```

## 总结

- ✅ 日志格式完全保持不变
- ✅ 符合 OpenTelemetry 标准
- ✅ 跨语言统一标准
- ✅ 可选启用，不影响现有功能
- ✅ 性能友好，支持采样


# OpenTelemetry 日志集成指南

## 概述

OpenTelemetry 的引入**不会改变现有的日志格式**。本文档说明如何在不影响现有日志格式的情况下集成 OpenTelemetry。

## 关键点

### ✅ 不会改变现有日志格式

- **现有格式保持不变**：`%(asctime)s - %(name)s - %(levelname)s - %(message)s`
- **现有日志输出不变**：控制台和文件日志保持原样
- **向后兼容**：不使用 OpenTelemetry 时，行为完全一致

### 📊 OpenTelemetry 日志的作用

OpenTelemetry 日志主要用于：
1. **关联追踪上下文**：将日志与分布式追踪关联
2. **统一收集**：通过 OpenTelemetry Collector 统一收集
3. **结构化导出**：导出到支持 OTLP 的后端（如 Jaeger、Loki）

## 使用方式

### 方式 1：保持现有格式（推荐）

```python
from atm.utils.logger import setup_logger

# 使用原有方式，格式完全不变
logger = setup_logger("my_module")
logger.info("This log format stays the same")
# 输出: 2025-12-19 09:38:54,877 - my_module - INFO - This log format stays the same
```

### 方式 2：启用 OpenTelemetry（可选）

```python
from atm.utils.otel_logger import setup_logger_with_otel

# 启用 OpenTelemetry，但格式仍然不变
logger = setup_logger_with_otel(
    name="my_module",
    enable_otel=True,
    otel_endpoint="http://localhost:4317",  # OpenTelemetry Collector
    service_name="atm-api"
)

logger.info("This log format is still the same")
# 控制台输出: 2025-12-19 09:38:54,877 - my_module - INFO - This log format is still the same
# OpenTelemetry 会额外导出结构化日志（包含 trace context）
```

## 日志格式对比

### 不使用 OpenTelemetry

```
2025-12-19 09:38:54,877 - atm.trading.strategy - INFO - Strategy started
```

### 使用 OpenTelemetry（控制台输出不变）

```
2025-12-19 09:38:54,877 - atm.trading.strategy - INFO - Strategy started
```

**控制台格式完全相同！** OpenTelemetry 只是额外导出到 Collector。

### OpenTelemetry 导出的结构化日志

OpenTelemetry 会导出包含以下信息的结构化日志：
```json
{
  "timestamp": "2025-12-19T09:38:54.877Z",
  "severity": "INFO",
  "body": "Strategy started",
  "attributes": {
    "logger.name": "atm.trading.strategy",
    "service.name": "atm-api",
    "trace_id": "abc123...",
    "span_id": "def456..."
  }
}
```

这些结构化日志**不会影响**控制台输出格式。

## 配置选项

### 环境变量控制

```bash
# 启用 OpenTelemetry
export ATM_ENABLE_OTEL=true
export ATM_OTEL_ENDPOINT=http://localhost:4317
export ATM_SERVICE_NAME=atm-api

# 禁用 OpenTelemetry（默认）
# 不设置这些环境变量即可
```

### 代码中控制

```python
import os
from atm.utils.otel_logger import setup_logger_with_otel

logger = setup_logger_with_otel(
    name="my_module",
    enable_otel=os.getenv("ATM_ENABLE_OTEL", "false").lower() == "true",
    otel_endpoint=os.getenv("ATM_OTEL_ENDPOINT"),
    service_name=os.getenv("ATM_SERVICE_NAME", "atm")
)
```

## 迁移建议

### 阶段 1：保持现状（当前）

```python
# 继续使用原有方式
from atm.utils.logger import setup_logger
logger = setup_logger("my_module")
```

### 阶段 2：选择性启用（推荐）

```python
# 在需要追踪的服务中启用
from atm.utils.otel_logger import setup_logger_with_otel

# API 服务启用
logger = setup_logger_with_otel("atm.api", enable_otel=True)

# 数据同步任务不启用（保持原样）
logger = setup_logger("atm.data.sync")
```

### 阶段 3：全面启用（生产环境）

```python
# 所有服务启用，但格式仍然不变
logger = setup_logger_with_otel("my_module", enable_otel=True)
```

## 性能考虑

- **格式不变**：控制台/文件日志性能无影响
- **额外导出**：OpenTelemetry 导出是异步的，影响极小
- **采样**：可以配置采样策略，只导出部分日志

## 常见问题

### Q: 引入 OpenTelemetry 后，日志格式会变吗？

**A: 不会。** 控制台和文件日志格式完全不变。OpenTelemetry 只是额外导出结构化日志。

### Q: 现有代码需要修改吗？

**A: 不需要。** 继续使用 `setup_logger()` 即可，格式和行为完全一致。

### Q: 如何查看 OpenTelemetry 导出的日志？

**A:** 需要配置 OpenTelemetry Collector 和后端（如 Jaeger、Loki）。控制台日志仍然使用原有格式。

### Q: 可以只启用追踪，不启用日志导出吗？

**A: 可以。** OpenTelemetry 的追踪（Traces）和日志（Logs）是独立的，可以分别启用。

## 总结

- ✅ **日志格式不变**：现有格式完全保留
- ✅ **向后兼容**：不使用 OpenTelemetry 时行为一致
- ✅ **可选启用**：按需启用，不影响现有代码
- ✅ **性能友好**：额外导出是异步的，影响极小


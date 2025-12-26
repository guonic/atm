# AkShare 数据源使用指南

## 概述

AkShare 数据源提供了基于 AkShare 库的数据获取功能。AkShare 是一个 Python 库，用于获取中国股票、期货、基金等金融数据。

**参考文档**: https://akshare.akfamily.xyz/

## 安装

在使用 AkShare 数据源之前，需要先安装 AkShare 库：

```bash
pip install akshare --upgrade
```

## 基本使用

### 1. 创建数据源

```python
from nq.data.source.akshare_source import AkshareSource, AkshareSourceConfig

# 创建配置
config = AkshareSourceConfig(
    api_name="stock_zh_a_spot",  # API 名称
    params={},  # API 参数
)

# 创建数据源
source = AkshareSource(config)
```

### 2. 获取股票实时行情

```python
# 方法 1: 使用便捷方法
source = AkshareSource(AkshareSourceConfig())
for record in source.fetch_stock_spot(market="A"):
    print(record)

# 方法 2: 使用通用 fetch 方法
config = AkshareSourceConfig(api_name="stock_zh_a_spot")
source = AkshareSource(config)
for record in source.fetch():
    print(record)
```

### 3. 获取股票历史K线数据

```python
source = AkshareSource(AkshareSourceConfig())

# 获取日线数据
for record in source.fetch_stock_hist(
    symbol="000001",  # 股票代码（不含后缀）
    period="daily",
    start_date="20240101",
    end_date="20241231",
    adjust="qfq",  # 前复权
):
    print(record)

# 获取周线数据
for record in source.fetch_stock_hist(
    symbol="000001",
    period="weekly",
    start_date="20240101",
    end_date="20241231",
):
    print(record)

# 获取分钟线数据
for record in source.fetch_stock_hist(
    symbol="000001",
    period="5min",  # 支持 1min, 5min, 15min, 30min, 60min
    start_date="20241201",
    end_date="20241231",
):
    print(record)
```

### 4. 获取股票基本信息

```python
source = AkshareSource(AkshareSourceConfig())

# 获取A股代码和名称列表
for record in source.fetch_stock_info():
    print(record)
```

### 5. 获取指数历史数据

```python
source = AkshareSource(AkshareSourceConfig())

# 获取上证指数历史数据
for record in source.fetch_index_hist(
    symbol="sh000001",  # 指数代码
    period="daily",
    start_date="20240101",
    end_date="20241231",
):
    print(record)
```

### 6. 获取基金历史数据

```python
source = AkshareSource(AkshareSourceConfig())

# 获取ETF基金历史数据
for record in source.fetch_fund_hist(
    symbol="000001",  # 基金代码
    period="daily",
    start_date="20240101",
    end_date="20241231",
):
    print(record)
```

## 便捷方法

AkShare 数据源提供了以下便捷方法：

### `fetch_stock_spot(market="A")`
获取股票实时行情数据。

**参数**:
- `market`: 市场类型 ('A' 表示A股, 'B' 表示B股)

### `fetch_stock_hist(symbol, period, start_date, end_date, adjust)`
获取股票历史K线数据。

**参数**:
- `symbol`: 股票代码（不含后缀，如 '000001'）
- `period`: 周期类型 ('daily', 'weekly', 'monthly', '1min', '5min', '15min', '30min', '60min')
- `start_date`: 开始日期 (YYYYMMDD 或 YYYY-MM-DD)
- `end_date`: 结束日期 (YYYYMMDD 或 YYYY-MM-DD)
- `adjust`: 复权类型 ('qfq'=前复权, 'hfq'=后复权, ''=不复权)

### `fetch_stock_info()`
获取股票基本信息（A股代码和名称列表）。

### `fetch_stock_list(market="A")`
获取股票列表。

### `fetch_index_hist(symbol, period, start_date, end_date)`
获取指数历史数据。

**参数**:
- `symbol`: 指数代码（如 'sh000001' 表示上证指数）
- `period`: 周期类型 ('daily', 'weekly', 'monthly')
- `start_date`: 开始日期
- `end_date`: 结束日期

### `fetch_fund_hist(symbol, period, start_date, end_date, adjust)`
获取基金历史数据。

## 通用 fetch 方法

如果便捷方法不满足需求，可以使用通用的 `fetch` 方法调用任何 AkShare API：

```python
config = AkshareSourceConfig(
    api_name="stock_zh_a_hist",  # 任何 AkShare API 名称
    params={
        "symbol": "000001",
        "period": "日K",
        "adjust": "qfq",
        "start_date": "20240101",
        "end_date": "20241231",
    },
)
source = AkshareSource(config)
for record in source.fetch():
    print(record)
```

## 连接测试

```python
source = AkshareSource(AkshareSourceConfig())
if source.test_connection():
    print("AkShare 连接成功")
else:
    print("AkShare 连接失败")
```

## 上下文管理器

数据源支持上下文管理器，可以自动清理资源：

```python
with AkshareSource(AkshareSourceConfig()) as source:
    for record in source.fetch_stock_spot():
        print(record)
# 自动调用 close() 方法
```

## 错误处理

```python
from nq.data.source.akshare_source import AkshareSource, AkshareSourceConfig
from nq.data.source.base import DataFetchError, ConnectionError

try:
    source = AkshareSource(AkshareSourceConfig())
    for record in source.fetch_stock_spot():
        print(record)
except ConnectionError as e:
    print(f"连接错误: {e}")
except DataFetchError as e:
    print(f"数据获取错误: {e}")
except ImportError as e:
    print(f"AkShare 库未安装: {e}")
```

## 注意事项

1. **AkShare 库安装**: 使用前需要安装 AkShare 库，否则会抛出 `ImportError`。

2. **数据源限制**: AkShare 的数据源主要用于学术研究和个人学习，使用时请遵守相关数据源的使用规定。

3. **API 更新**: AkShare 的 API 可能会更新，如果遇到 API 不存在的情况，请检查 AkShare 文档。

4. **日期格式**: 日期参数支持 `YYYYMMDD` 和 `YYYY-MM-DD` 两种格式，内部会自动转换。

5. **数据量**: 某些 API 可能返回大量数据，建议使用迭代器方式处理，避免内存溢出。

## 常见问题

### Q: 如何获取特定股票的实时行情？

```python
source = AkshareSource(AkshareSourceConfig())
for record in source.fetch_stock_spot():
    if record.get("代码") == "000001":
        print(record)
```

### Q: 如何获取复权数据？

在调用 `fetch_stock_hist` 时，设置 `adjust` 参数：
- `adjust="qfq"`: 前复权
- `adjust="hfq"`: 后复权
- `adjust=""`: 不复权

### Q: 如何获取分钟级别的K线数据？

使用 `fetch_stock_hist` 方法，设置 `period` 为分钟级别：
```python
for record in source.fetch_stock_hist(
    symbol="000001",
    period="5min",  # 5分钟K线
    start_date="20241201",
    end_date="20241231",
):
    print(record)
```

### Q: 如何调用 AkShare 的其他 API？

使用通用的 `fetch` 方法，指定 `api_name` 和参数：
```python
config = AkshareSourceConfig(
    api_name="your_akshare_api_name",
    params={"param1": "value1", "param2": "value2"},
)
source = AkshareSource(config)
for record in source.fetch():
    print(record)
```

## 更多信息

- AkShare 官方文档: https://akshare.akfamily.xyz/
- AkShare GitHub: https://github.com/akfamily/akshare


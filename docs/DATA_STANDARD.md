# 数据标准化规范

本文档定义了 ATM 项目中所有数据格式的标准化规范，确保整个项目中使用一致的数据格式。

## 概述

所有数据标准化函数都位于 `nq.utils.data_normalize` 模块中，提供统一的标准化接口。

```python
from nq.utils.data_normalize import (
    normalize_stock_code,
    normalize_index_code,
    normalize_date,
    normalize_date_for_storage,
    normalize_industry_code,
    normalize_qlib_directory_name,
)
```

## 1. 股票代码标准化

### 标准格式

**格式**: `{code}.{market}`

- **code**: 6位数字（例如：`000001`, `600000`）
- **market**: 大写的交易所代码（`SH`, `SZ`, `BJ`）

### 示例

- `000001.SZ` - 深交所股票
- `600000.SH` - 上交所股票
- `430001.BJ` - 北交所股票

### 使用

```python
from nq.utils.data_normalize import normalize_stock_code

# 标准化股票代码
normalized = normalize_stock_code("000001.sz")  # 返回: "000001.SZ"
normalized = normalize_stock_code("sh.000001")  # 返回: "000001.SH"
normalized = normalize_stock_code("000001")     # 返回: "000001.SZ"
normalized = normalize_stock_code("600000")     # 返回: "600000.SH"
normalized = normalize_stock_code("SH000300")   # 返回: "000300.SH"
```

### 支持的输入格式

1. **标准格式**（已标准化）: `000001.SZ`, `600000.SH`
2. **小写格式**: `000001.sz`, `600000.sh`
3. **反向格式**: `sh.000001`, `SZ.600000`
4. **纯数字**: `000001` (自动判断为 SZ), `600000` (自动判断为 SH)
5. **前缀格式**: `SH000300`, `SZ399001`

### 验证

```python
from nq.utils.data_normalize import is_valid_stock_code

is_valid_stock_code("000001.SZ")  # True
is_valid_stock_code("000001.sz")  # False (小写不符合标准)
is_valid_stock_code("000001")     # False (缺少市场代码)
```

### 提取代码和市场

```python
from nq.utils.data_normalize import extract_code_and_market

code, market = extract_code_and_market("000001.SZ")
# code: "000001", market: "SZ"
```

## 2. 指数代码标准化

### 标准格式

指数代码使用与股票代码相同的格式：`{code}.{market}`

### 使用

```python
from nq.utils.data_normalize import normalize_index_code

normalized = normalize_index_code("000300.SH")  # 返回: "000300.SH"
normalized = normalize_index_code("CSI300")     # 返回: "CSI300" (无法解析时返回大写)
```

## 3. 日期标准化

### 显示格式（YYYY-MM-DD）

用于用户界面、日志输出等显示场景。

```python
from nq.utils.data_normalize import normalize_date

normalized = normalize_date("2025-01-02")  # 返回: "2025-01-02"
normalized = normalize_date("20250102")    # 返回: "2025-01-02"
normalized = normalize_date(datetime(2025, 1, 2))  # 返回: "2025-01-02"
```

### 存储格式（YYYYMMDD）

用于数据库存储、Qlib 数据等存储场景。

```python
from nq.utils.data_normalize import normalize_date_for_storage

normalized = normalize_date_for_storage("2025-01-02")  # 返回: "20250102"
normalized = normalize_date_for_storage("20250102")    # 返回: "20250102"
normalized = normalize_date_for_storage(datetime(2025, 1, 2))  # 返回: "20250102"
```

### 支持的输入格式

1. **YYYY-MM-DD**: `2025-01-02`
2. **YYYYMMDD**: `20250102`
3. **datetime 对象**: `datetime(2025, 1, 2)`
4. **其他 pandas 可解析格式**: 自动尝试解析

## 4. 行业代码标准化

### 标准格式

行业代码通常格式为：`{code}.{market}`，例如申万行业代码 `850531.SI`

### 使用

```python
from nq.utils.data_normalize import normalize_industry_code

normalized = normalize_industry_code("850531.SI")  # 返回: "850531.SI"
normalized = normalize_industry_code("850531.si")  # 返回: "850531.SI"
```

## 5. Qlib 目录名标准化

### 说明

Qlib 使用小写格式存储 feature 目录（例如：`000001.sz`），但我们的标准格式是大写（`000001.SZ`）。此函数用于将标准格式转换为 Qlib 兼容的小写格式。

### 使用

```python
from nq.utils.data_normalize import normalize_qlib_directory_name

dir_name = normalize_qlib_directory_name("000001.SZ")  # 返回: "000001.sz"
dir_name = normalize_qlib_directory_name("600000.SH")  # 返回: "600000.sh"
```

## 使用规范

### 1. 数据入口统一转换

**在数据导入入口处统一转换**，而不是在多个地方分散转换：

```python
# ✅ 正确：在数据导入时统一转换
from nq.utils.data_normalize import normalize_stock_code, normalize_date_for_storage

def import_stock_data(ts_code: str, trade_date: str, data: dict):
    normalized_code = normalize_stock_code(ts_code)  # 入口处转换
    normalized_date = normalize_date_for_storage(trade_date)  # 入口处转换
    # 后续所有处理都使用标准化后的数据
    save_to_database(normalized_code, normalized_date, data)
```

### 2. 数据库存储

- **股票代码**: 存储为标准格式（大写，如 `000001.SZ`）
- **日期**: 存储为 YYYYMMDD 格式（如 `20250102`）

### 3. Qlib 数据格式

- **instruments/all.txt**: 使用标准格式（大写，如 `000001.SZ`）
- **features/ 目录**: Qlib 内部使用小写（`000001.sz`），使用 `normalize_qlib_directory_name()` 转换

### 4. 代码中禁止分散转换

**禁止**在代码中分散定义转换函数：

```python
# ❌ 错误：不要定义自己的转换函数
def convert_ts_code_to_qlib_format(ts_code: str) -> str:
    return ts_code.upper()

# ✅ 正确：使用统一工具
from nq.utils.data_normalize import normalize_stock_code
```

### 5. 向后兼容

为了保持向后兼容，提供了以下别名：

```python
from nq.utils.data_normalize import (
    normalize_ts_code,  # 别名: normalize_stock_code
    convert_ts_code_to_qlib_format,  # 别名: normalize_stock_code
)
```

## 扩展指南

### 添加新的标准化函数

1. 在 `nq.utils.data_normalize` 模块中添加新函数
2. 遵循命名规范：`normalize_{data_type}`
3. 添加详细的文档字符串，说明标准格式和示例
4. 在 `nq.utils.__init__.py` 中导出新函数
5. 在本文档中添加相应的规范说明

### 示例：添加新的标准化函数

```python
def normalize_currency_code(code: str) -> str:
    """
    Normalize currency code to standard format: ISO 4217 format.
    
    Standard format: 3-letter uppercase code (e.g., 'USD', 'CNY')
    
    Args:
        code: Currency code in various formats.
    
    Returns:
        Normalized currency code in standard format.
    """
    if not code:
        return ""
    return str(code).strip().upper()[:3]
```

## 迁移指南

### 删除分散的转换函数

项目中所有分散的转换函数都应删除，统一使用 `nq.utils.data_normalize` 中的函数。

### 更新导入

```python
# 旧代码
def convert_ts_code_to_qlib_format(ts_code: str) -> str:
    return ts_code.upper()

# 新代码
from nq.utils.data_normalize import normalize_stock_code
# 或者使用别名
from nq.utils.data_normalize import convert_ts_code_to_qlib_format
```

## 注意事项

1. **大小写一致性**: 标准格式使用大写（`000001.SZ`），但 Qlib 的 features 目录使用小写。使用相应的转换函数处理。
2. **日期格式**: 显示使用 `YYYY-MM-DD`，存储使用 `YYYYMMDD`。
3. **验证**: 使用相应的验证函数（如 `is_valid_stock_code()`）验证数据格式。
4. **错误处理**: 如果输入无法解析，函数会返回原始值的大写版本或空字符串，不会抛出异常。

## 相关文档

- [股票代码标准化详细说明](STOCK_CODE_STANDARD.md)
- [Qlib 数据格式要求](QLIB_DATA_FORMAT.md)

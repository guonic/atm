# 股票代码标准化规范

> **注意**: 本文档已整合到 [数据标准化规范](DATA_STANDARD.md) 中。  
> 请参考 [数据标准化规范](DATA_STANDARD.md) 获取最新的完整信息。

## 快速参考

### 标准格式

**格式**: `{code}.{market}`

- **code**: 6位数字（例如：`000001`, `600000`）
- **market**: 大写的交易所代码（`SH`, `SZ`, `BJ`）

### 使用

```python
from nq.utils.data_normalize import normalize_stock_code

# 标准化股票代码
normalized = normalize_stock_code("000001.sz")  # 返回: "000001.SZ"
normalized = normalize_stock_code("sh.000001")  # 返回: "000001.SH"
normalized = normalize_stock_code("000001")     # 返回: "000001.SZ"
normalized = normalize_stock_code("600000")     # 返回: "600000.SH"
```

## 详细文档

请参考 [数据标准化规范](DATA_STANDARD.md) 获取：
- 完整的标准化规范
- 所有支持的数据类型标准化
- 使用规范和最佳实践
- 迁移指南

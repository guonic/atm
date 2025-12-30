# Qlib 数据格式要求

## 概述

Qlib 对数据格式有严格的要求。本文档说明所有格式要求。

## 1. Calendar 文件格式

### 文件位置
- 路径：`<qlib_dir>/calendars/<freq>.txt`
- 例如：`~/.qlib/qlib_data/cn_data/calendars/day.txt`

### 格式要求
- **格式**：`YYYYMMDD`（8位数字，无分隔符）
- **示例**：`20251225`（表示 2025年12月25日）
- **每行一个日期**，按时间顺序排列

### 示例文件内容
```
20000512
20000515
20000516
...
20251224
20251225
```

### ⚠️ 重要说明
- **Qlib 官方要求**：`YYYYMMDD` 格式（8位数字）
- **`dump_bin.py` 的默认行为**：生成 `YYYY-MM-DD` 格式（带分隔符）
- **解决方案**：`export_qlib.py` 会在转换后自动修复格式，将 `YYYY-MM-DD` 转换为 `YYYYMMDD`

## 2. Instruments 文件格式

### 文件位置
- 路径：`<qlib_dir>/instruments/all.txt`

### 格式要求
- **标准格式**：`<stock_code>\t<start_date>\t<end_date>`
- **日期格式**：`YYYYMMDD`（8位数字）
- **分隔符**：Tab 字符（`\t`）

### 示例文件内容
```
000001.SZ	20000726	20251225
000002.SZ	20000512	20251225
000300.SH	20050104	20251225
```

### 字段说明
- `stock_code`：股票代码，格式为 `000001.SZ` 或 `000300.SH`
- `start_date`：该股票数据的开始日期（YYYYMMDD）
- `end_date`：该股票数据的结束日期（YYYYMMDD）

### ⚠️ 重要说明
- **推荐格式**：带日期范围的完整格式（`code\tstart\tend`）
- **简化格式**：只有股票代码（`code`）也可以，但 Qlib 可能无法正确识别日期范围
- **日期格式**：必须使用 `YYYYMMDD` 格式，不能使用 `YYYY-MM-DD`

## 3. Features（Bin 文件）格式

### 文件位置
- 路径：`<qlib_dir>/features/<stock_code>/<field>.<freq>.bin`
- 例如：`~/.qlib/qlib_data/cn_data/features/000001.sz/close.day.bin`

### 文件命名规则
- 目录名：股票代码（小写，如 `000001.sz`）
- 文件名：`<field>.<freq>.bin`
  - `field`：字段名（如 `open`, `close`, `high`, `low`, `volume`, `factor`）
  - `freq`：频率（如 `day`, `week`, `1h`）

### Bin 文件格式
- **数据类型**：`float32`（小端序）
- **文件结构**：
  - 第一个 `float32`：`date_index`（指向 calendar 中的日期索引位置）
  - 后续 `float32`：该字段的数据值（按日期顺序）

### 示例
```
文件：close.day.bin
内容：
[date_index (float32), close_price_1 (float32), close_price_2 (float32), ...]
```

### 字段要求
- **必需字段**：`open`, `close`, `high`, `low`, `volume`, `factor`
- **factor 字段**：复权因子，默认值为 `1.0`

## 4. CSV 源文件格式（用于转换）

### 文件位置
- 路径：`<csv_dir>/<stock_code>.csv`
- 例如：`~/.qlib/csv_data/cn_data/000001.SZ.csv`

### 格式要求
- **无表头**（no header）
- **列顺序**：`date, open, high, low, close, volume, factor`
- **日期格式**：`YYYY-MM-DD`（如 `2025-12-25`）
- **分隔符**：逗号（`,`）

### 示例文件内容
```
2025-12-25,11.54,11.62,11.52,11.56,547455,1.0
2025-12-24,11.50,11.60,11.48,11.54,523421,1.0
```

### ⚠️ 重要说明
- CSV 文件中的日期格式是 `YYYY-MM-DD`（带分隔符）
- 但转换后的 calendar 文件必须是 `YYYYMMDD`（无分隔符）
- `dump_bin.py` 会自动处理这个转换

## 5. 目录结构

### 完整的 Qlib 数据目录结构
```
<qlib_dir>/
├── calendars/
│   └── day.txt              # 交易日历（YYYYMMDD 格式）
├── instruments/
│   └── all.txt              # 股票列表（code\tstart\tend 格式）
└── features/
    ├── 000001.sz/
    │   ├── open.day.bin
    │   ├── close.day.bin
    │   ├── high.day.bin
    │   ├── low.day.bin
    │   ├── volume.day.bin
    │   └── factor.day.bin
    ├── 000002.sz/
    │   └── ...
    └── ...
```

## 6. 格式转换流程

### 从 CSV 到 Qlib Bin 的转换流程

1. **读取 CSV 文件**
   - CSV 日期格式：`YYYY-MM-DD`
   - 使用 `pd.to_datetime()` 解析

2. **生成 Calendar**
   - `dump_bin.py` 使用 `_format_datetime()` 生成 `YYYY-MM-DD` 格式
   - **问题**：Qlib 需要 `YYYYMMDD` 格式
   - **解决方案**：`export_qlib.py` 在转换后自动修复格式

3. **生成 Instruments**
   - 从 CSV 文件名提取股票代码
   - 从数据中提取日期范围
   - 格式：`code\tstart\tend`（日期为 `YYYYMMDD`）

4. **生成 Bin 文件**
   - 读取 CSV 数据
   - 对齐到 calendar
   - 写入 bin 文件（第一个 float32 是 date_index）

## 7. 常见问题

### Q1: Calendar 文件格式不一致
**问题**：`dump_bin.py` 生成 `YYYY-MM-DD`，但 Qlib 需要 `YYYYMMDD`

**解决方案**：
- `export_qlib.py` 会自动修复格式
- 或使用 `regenerate_instruments_calendar.py` 重新生成

### Q2: Instruments 文件缺少日期范围
**问题**：只有股票代码，没有日期范围

**解决方案**：
- 使用 `regenerate_instruments_calendar.py` 重新生成
- 或重新导出数据

### Q3: Bin 文件中的 date_index 不匹配
**问题**：Bin 文件的 date_index 指向错误的 calendar 位置

**原因**：
- Calendar 文件格式错误（如 `YYYY-MM-DD` 被错误解析）
- Calendar 文件被覆盖或损坏

**解决方案**：
- 修复 `_read_calendars` 方法（已修复）
- 重新生成 calendar 文件

### Q4: 数据日期不在 calendar 中
**问题**：`dump_bin` 警告 "data is not in calendars"

**原因**：
- Calendar 文件格式错误，导致日期解析失败
- CSV 中的日期格式不正确

**解决方案**：
- 确保 calendar 文件是 `YYYYMMDD` 格式
- 确保 CSV 文件中的日期是 `YYYY-MM-DD` 格式

## 8. 最佳实践

1. **始终使用 `YYYYMMDD` 格式**存储 calendar 文件
2. **Instruments 文件使用完整格式**（`code\tstart\tend`）
3. **CSV 文件使用 `YYYY-MM-DD` 格式**（便于阅读和解析）
4. **转换后验证格式**（使用 `verify_exported_data.py`）
5. **定期检查数据覆盖范围**（使用 `check_data_coverage.py`）

## 9. 工具说明

### 数据导出工具
- `export_qlib.py`：导出股票数据到 Qlib 格式
  - 自动处理格式转换
  - 自动修复 calendar 格式

### 数据验证工具
- `verify_exported_data.py`：验证 Qlib 数据完整性
- `check_data_coverage.py`：检查数据覆盖范围

### 数据修复工具
- `regenerate_instruments_calendar.py`：重新生成 instruments 和 calendar 文件

## 10. 总结

| 文件类型 | 格式要求 | 示例 |
|---------|---------|------|
| Calendar | `YYYYMMDD` | `20251225` |
| Instruments | `code\tstart\tend` | `000001.SZ\t20000726\t20251225` |
| CSV 源文件 | `YYYY-MM-DD` | `2025-12-25` |
| Bin 文件 | `float32` 数组 | `[date_index, data1, data2, ...]` |

**关键点**：
- Calendar 和 Instruments 中的日期必须使用 `YYYYMMDD` 格式
- CSV 源文件可以使用 `YYYY-MM-DD` 格式（便于阅读）
- `dump_bin.py` 会自动处理格式转换，但需要后处理修复 calendar 格式


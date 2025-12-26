# Qlib 官方 Scripts 工具指南

## 概述

Qlib 官方仓库的 `scripts` 目录（[GitHub](https://github.com/microsoft/qlib/tree/main/scripts)）包含多个实用工具脚本，用于数据获取、转换和管理。本文档总结了这些工具的功能、使用方法和数据格式要求。

## 主要工具

### 1. get_data.py - 数据下载工具

**功能**: 下载和准备市场数据

**使用方法**:
```bash
python scripts/get_data.py qlib_data \
  --target_dir ~/.qlib/qlib_data/cn_data \
  --region cn
```

**参数说明**:
- `qlib_data`: 数据类型（qlib_data 表示下载 Qlib 格式的数据）
- `--target_dir`: 目标目录，数据将存储在此
- `--region`: 市场区域（cn 表示中国市场）

**输出**:
- 下载的数据会存储在指定的 `target_dir` 目录下
- 数据格式为 Qlib 标准的二进制格式（.bin 文件）

### 2. dump_bin.py - 数据格式转换工具

**功能**: 将原始数据（CSV）转换为 Qlib 可识别的二进制格式（.bin）

**使用方法**:

#### 方法 1: 命令行接口
```bash
python -m qlib.tools.dump_bin dump_all \
  --csv_path ~/.qlib/csv_data/cn_data \
  --qlib_dir ~/.qlib/qlib_data/cn_data \
  --include_fields open,close,high,low,volume \
  --freq day
```

#### 方法 2: Python API
```python
from qlib.tools.dump_bin import dump_all

dump_all(
    csv_path="~/.qlib/csv_data/cn_data",
    qlib_dir="~/.qlib/qlib_data/cn_data",
    include_fields="open,close,high,low,volume",
    freq="day",
    max_workers=4,
)
```

**参数说明**:
- `csv_path`: CSV 数据目录路径
- `qlib_dir`: Qlib 二进制数据输出目录
- `include_fields`: 包含的字段（逗号分隔），如 `open,close,high,low,volume`
- `freq`: 数据频率（`day`, `1min`, `5min` 等）
- `max_workers`: 并行处理的工作线程数（可选）

**CSV 格式要求**:
1. **文件命名**: 每个股票一个 CSV 文件，文件名为股票代码（如 `000001.csv`）
2. **文件格式**: 无表头，数据行格式为：`date,open,high,low,close,volume`
3. **日期格式**: `YYYY-MM-DD`（如 `2023-12-24`）
4. **编码**: UTF-8

**示例 CSV 文件内容**:
```
2023-12-24,10.50,10.80,10.45,10.75,1000000
2023-12-25,10.75,10.90,10.70,10.85,1200000
```

**输出结构**:
```
~/.qlib/qlib_data/cn_data/
├── calendars/
│   └── day.txt          # 交易日历
├── instruments/
│   ├── all.txt          # 所有股票列表
│   ├── csi300.txt       # CSI300 股票列表
│   └── ...
└── features/
    ├── <stock_code>/
    │   └── day.bin       # 二进制数据文件
    └── ...
```

### 3. generate_instruments.py - 股票列表生成工具

**功能**: 生成包含股票列表的 `instruments` 文件，供 Qlib 使用

**使用方法**:
```bash
python scripts/generate_instruments.py \
  --output_dir ~/.qlib/qlib_data/cn_data/instruments \
  --region cn
```

**输出**:
- 生成 `all.txt`、`csi300.txt`、`csi500.txt` 等股票列表文件
- 每个文件包含股票代码列表，每行一个代码

## Qlib 数据目录结构

Qlib 数据目录的标准结构如下：

```
~/.qlib/qlib_data/cn_data/
├── calendars/
│   └── day.txt              # 交易日历，每行一个日期（YYYY-MM-DD）
├── instruments/
│   ├── all.txt              # 所有股票列表，每行一个股票代码
│   ├── csi300.txt           # CSI300 股票列表
│   ├── csi500.txt           # CSI500 股票列表
│   └── ...
└── features/
    ├── <stock_code>/        # 每个股票一个目录
    │   └── day.bin          # 二进制数据文件（float32 数组）
    └── ...
```

### 文件格式说明

#### 1. calendars/day.txt
- **格式**: 每行一个日期，格式为 `YYYY-MM-DD`
- **示例**:
```
2023-12-24
2023-12-25
2023-12-26
```

#### 2. instruments/*.txt
- **格式**: 每行一个股票代码
- **示例** (all.txt):
```
000001
000002
000004
000006
```

#### 3. features/<stock_code>/day.bin
- **格式**: 二进制文件，包含 float32 数组
- **数据结构**: 
  - 每个字段（open, close, high, low, volume）是一个 float32 数组
  - 数组长度等于交易日历的天数
  - 数据按日期顺序排列
  - 缺失数据用 NaN（转换为 0）填充

## 工具集成建议

### 在 ATM 项目中使用

我们的 `export_to_qlib.py` 工具已经实现了与 Qlib 官方工具的集成：

1. **优先使用官方工具**: 自动检测并使用 `qlib.tools.dump_bin`
2. **多方法回退**: 如果官方工具不可用，提供手动转换方法
3. **格式兼容**: 确保 CSV 格式符合 Qlib 要求

### 最佳实践

1. **数据准备**:
   - 确保 CSV 文件格式正确（无表头，日期格式为 YYYY-MM-DD）
   - 股票代码格式统一（如 `000001`，不带后缀）

2. **数据转换**:
   - 优先使用官方 `dump_bin` 工具
   - 如果官方工具不可用，使用手动转换方法

3. **数据验证**:
   - 检查生成的 `calendars/day.txt` 是否正确
   - 检查 `instruments/all.txt` 是否包含所有股票
   - 使用 `check_data.py` 验证数据加载

## 常见问题

### Q1: 为什么 `qlib.tools.dump_bin` 找不到？

**原因**: 某些 qlib 安装版本可能不包含 `tools` 模块，或者模块路径不同。

**解决方案**:
1. 升级 qlib: `pip install --upgrade pyqlib`
2. 检查安装路径: `python -c "import qlib; print(qlib.__file__)"`
3. 使用手动转换方法（`export_to_qlib.py` 已实现）

### Q2: CSV 转换后数据为空或 NaN？

**可能原因**:
1. CSV 文件格式不正确（缺少字段或日期格式错误）
2. 数据日期范围与日历不匹配
3. 股票代码格式不一致

**解决方案**:
1. 检查 CSV 文件格式（无表头，字段顺序：date,open,high,low,close,volume）
2. 确保日期格式为 `YYYY-MM-DD`
3. 使用 `check_data.py` 验证数据

### Q3: 如何生成股票列表文件？

**方法 1**: 使用官方工具
```bash
python scripts/generate_instruments.py --output_dir ~/.qlib/qlib_data/cn_data/instruments
```

**方法 2**: 手动生成
```python
# 从数据库导出股票列表
stocks = get_all_stocks()
with open("all.txt", "w") as f:
    for stock in stocks:
        f.write(f"{stock.code}\n")
```

## 参考资源

- [Qlib GitHub 仓库](https://github.com/microsoft/qlib)
- [Qlib 官方文档](https://qlib.readthedocs.io/)
- [Qlib Scripts 目录](https://github.com/microsoft/qlib/tree/main/scripts)

## 相关文档

- [OFFICIAL_TOOLS.md](./OFFICIAL_TOOLS.md) - 官方工具使用方法
- [README.md](./README.md) - Qlib 数据导出工具说明
- [export_to_qlib.py](./export_to_qlib.py) - 数据导出工具实现


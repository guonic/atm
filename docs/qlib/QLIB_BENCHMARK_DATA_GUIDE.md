# Qlib 基准指数数据制作指南

## 概述

Qlib 回测框架需要基准指数数据才能进行回测。本指南介绍如何收集和导入基准指数数据到 Qlib 格式。

## 快速开始

### 方法 1: 使用项目工具（推荐）

项目提供了专门的工具来导出指数数据：

```bash
# 导出沪深300指数数据
python python/tools/qlib/export_index_to_qlib.py \
    --index_code 000300.SH \
    --start_date 2020-01-01 \
    --end_date 2024-12-31 \
    --qlib_dir ~/.qlib/qlib_data/cn_data \
    --token YOUR_TUSHARE_TOKEN

# 导出多个指数
python python/tools/qlib/export_index_to_qlib.py \
    --index_code 000300.SH 000905.SH 399001.SZ \
    --start_date 2020-01-01 \
    --end_date 2024-12-31 \
    --token YOUR_TUSHARE_TOKEN
```

### 方法 2: 使用 Qlib 官方工具

Qlib 官方提供了数据下载工具，可能包含基准指数数据：

```bash
# 下载 Qlib 官方数据（可能包含基准指数）
python -m qlib.run.get_data qlib_data \
    --target_dir ~/.qlib/qlib_data/cn_data \
    --region cn
```

## 详细步骤

### 步骤 1: 准备 Tushare Token

1. 注册 Tushare Pro 账号：https://tushare.pro/
2. 获取 API Token
3. 设置环境变量：
   ```bash
   export TUSHARE_TOKEN=your_token_here
   ```
   或者在命令中直接指定：
   ```bash
   --token your_token_here
   ```

### 步骤 2: 导出指数数据

使用项目提供的工具：

```bash
python python/tools/qlib/export_index_to_qlib.py \
    --index_code 000300.SH \
    --start_date 2020-01-01 \
    --end_date 2024-12-31 \
    --qlib_dir ~/.qlib/qlib_data/cn_data \
    --token $TUSHARE_TOKEN
```

**常用指数代码**：
- `000300.SH` - 沪深300指数（CSI300）
- `000905.SH` - 中证500指数（CSI500）
- `000001.SH` - 上证指数
- `399001.SZ` - 深证成指
- `399006.SZ` - 创业板指

### 步骤 3: 验证数据

导出完成后，验证数据是否正确：

```python
import qlib
from qlib.data import D

qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region="cn")

# 测试加载基准数据
data = D.features(
    ["000300.SH"],
    ["$close"],
    start_time="2024-01-01",
    end_time="2024-12-31",
    freq="day",
)

print(f"Loaded {len(data)} days of benchmark data")
print(data.head())
```

## 工具说明

### export_index_to_qlib.py

**功能**：
- 从 Tushare 获取指数日线数据
- 转换为 Qlib 标准格式（CSV）
- 转换为 Qlib 二进制格式（.bin）

**参数说明**：
- `--index_code`: 指数代码（Qlib 格式，如 `000300.SH`），可指定多个
- `--start_date`: 开始日期（YYYY-MM-DD）
- `--end_date`: 结束日期（YYYY-MM-DD）
- `--qlib_dir`: Qlib 数据目录（默认：`~/.qlib/qlib_data/cn_data`）
- `--token`: Tushare Pro API Token（或设置 `TUSHARE_TOKEN` 环境变量）
- `--freq`: 数据频率（默认：`day`）

**输出**：
- CSV 文件：`~/.qlib/qlib_data/cn_data/csv_index_data/000300.SH.csv`
- Qlib bin 文件：`~/.qlib/qlib_data/cn_data/features/000300.SH/day.bin`

## 常见问题

### 1. Tushare API 限制

**问题**：`积分不足，无法访问该接口`

**解决**：
- Tushare Pro 需要积分才能访问某些接口
- `index_daily` 接口需要一定积分
- 可以：
  1. 升级 Tushare 账号获取更多积分
  2. 使用 Qlib 官方数据（如果包含基准指数）
  3. 手动准备 CSV 文件（见下方）

### 2. 数据格式问题

**问题**：`Factor field is missing`

**解决**：
- 指数数据通常不需要复权因子，脚本会自动添加 `factor=1.0`
- 如果仍有问题，检查 CSV 文件格式

### 3. 日期范围问题

**问题**：`No data found for index`

**解决**：
- 检查日期范围是否在指数上市日期之后
- 检查 Tushare 是否有该指数的数据
- 尝试使用更近的日期范围

## 手动准备数据（高级）

如果无法使用 Tushare API，可以手动准备 CSV 文件：

### CSV 格式要求

1. **文件命名**：`000300.SH.csv`（指数代码）
2. **文件格式**：无表头，每行一条记录
3. **列顺序**：`date,open,high,low,close,volume,factor`
4. **日期格式**：`YYYY-MM-DD`（如 `2024-01-01`）
5. **编码**：UTF-8

**示例 CSV 内容**：
```
2024-01-01,3500.00,3520.00,3490.00,3510.00,1000000000,1.0
2024-01-02,3510.00,3530.00,3500.00,3520.00,1200000000,1.0
```

### 转换为 Qlib 格式

将 CSV 文件放在临时目录，然后使用 dump_bin 工具：

```bash
# 创建临时目录
mkdir -p /tmp/index_csv
cp 000300.SH.csv /tmp/index_csv/

# 使用 dump_bin 转换
python -m qlib.tools.dump_bin dump_all \
    --csv_path /tmp/index_csv \
    --qlib_dir ~/.qlib/qlib_data/cn_data \
    --include_fields open,close,high,low,volume,factor \
    --freq day
```

## 验证基准数据

导出完成后，验证数据：

```python
import qlib
from qlib.data import D

# 初始化 Qlib
qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region="cn")

# 测试加载基准数据
try:
    data = D.features(
        ["000300.SH"],
        ["$close"],
        start_time="2024-01-01",
        end_time="2024-12-31",
        freq="day",
    )
    print(f"✓ Benchmark data loaded successfully: {len(data)} days")
    print(data.head())
except Exception as e:
    print(f"✗ Failed to load benchmark data: {e}")
```

## 使用基准数据

数据准备好后，可以在回测中使用：

```bash
python python/examples/backtest_structure_expert.py \
    --model_path models/structure_expert_large.pth \
    --start_date 2024-01-01 \
    --end_date 2024-12-31 \
    --benchmark 000300.SH \
    --config_path config/config.yaml
```

## 相关文件

- 导出工具：`python/tools/qlib/export_index_to_qlib.py`
- 股票导出工具：`python/tools/qlib/export_qlib.py`
- 单股票导出：`python/tools/qlib/export_qlib.py --freq day --stocks <stock_code>`
- Qlib 工具指南：`python/tests/qlib/QLIB_SCRIPTS_GUIDE.md`

## 总结

1. **准备 Tushare Token**：注册账号并获取 API Token
2. **导出指数数据**：使用 `export_index_to_qlib.py` 工具
3. **验证数据**：使用 Python 脚本验证数据是否正确
4. **使用基准**：在回测脚本中使用 `--benchmark` 参数

完成以上步骤后，就可以在回测中使用基准指数了！


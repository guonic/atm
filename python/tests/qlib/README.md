# Qlib 测试流程

本目录包含 Qlib 相关的测试脚本和示例。

## 文件说明

### `test_qlib_basic.py`
基本的 Qlib 测试流程，包含以下测试：

1. **Qlib 初始化测试** - 验证 Qlib 数据是否正确加载
2. **日历加载测试** - 测试交易日历的加载
3. **股票列表加载测试** - 测试股票列表的加载
4. **数据加载测试** - 测试基础数据的加载（价格、成交量等）
5. **特征提取测试** - 测试技术指标特征的计算
6. **简单策略测试** - 测试移动平均交叉策略的信号生成

### `run_baseline.py`
Qlib 工作流示例，展示如何使用 Qlib 的完整工作流进行模型训练和预测。

### `QLIB_SCRIPTS_GUIDE.md`
Qlib 脚本使用指南，包含数据导出、转换、验证等工具的详细说明。

## 使用方法

### 基本测试流程

运行基本测试流程：

```bash
# 使用默认 Qlib 数据目录
python python/tests/qlib/test_qlib_basic.py

# 使用自定义 Qlib 数据目录
python python/tests/qlib/test_qlib_basic.py --qlib-dir ~/.qlib/qlib_data/cn_data

# 测试特定股票
python python/tests/qlib/test_qlib_basic.py --instruments 000001.SZ 000002.SZ

# 跳过策略测试
python python/tests/qlib/test_qlib_basic.py --skip-strategy
```

### 参数说明

- `--qlib-dir`: Qlib 数据目录路径（默认: `~/.qlib/qlib_data/cn_data`）
- `--region`: Qlib 区域（默认: `cn`）
- `--instruments`: 要测试的股票代码列表（可选）
- `--skip-strategy`: 跳过策略测试

## 测试内容

### 1. Qlib 初始化
验证 Qlib 数据是否正确导出和配置。

### 2. 日历加载
测试交易日历的加载，包括：
- 总交易日数
- 日期范围
- 特定日期范围的日历

### 3. 股票列表加载
测试股票列表的加载，包括：
- 总股票数
- 股票代码示例

### 4. 数据加载
测试基础数据的加载，包括：
- 价格数据（开盘、收盘、最高、最低）
- 成交量数据
- 复权因子
- 缺失值检查

### 5. 特征提取
测试技术指标特征的计算，包括：
- 移动平均线（5日、20日）
- 标准差
- 收益率
- 其他技术指标

### 6. 简单策略
测试移动平均交叉策略的信号生成，包括：
- 买入信号（短期均线上穿长期均线）
- 卖出信号（短期均线下穿长期均线）

## 输出示例

```
================================================================================
Test 1: Qlib Initialization
================================================================================
✓ Qlib initialized successfully
  Data directory: /Users/guonic/.qlib/qlib_data/cn_data
  Region: cn

================================================================================
Test 2: Calendar Loading
================================================================================
✓ Calendar loaded successfully
  Total trading days: 245
  First date: 2023-01-03
  Last date: 2024-12-26

...

================================================================================
Test Summary
================================================================================
  Qlib Initialization: ✓ PASSED
  Calendar Loading: ✓ PASSED
  Instruments Loading: ✓ PASSED
  Data Loading: ✓ PASSED
  Feature Extraction: ✓ PASSED
  Simple Strategy: ✓ PASSED

Total: 6/6 tests passed
================================================================================
✓ All tests passed!
================================================================================
```

## 前置要求

1. **Qlib 数据已导出**
   - 确保已运行数据导出脚本生成 Qlib 格式数据
   - 数据目录应包含 `calendars/`, `instruments/`, `features/` 目录

2. **Python 依赖**
   - `qlib` - Qlib 库
   - `pandas` - 数据处理
   - `numpy` - 数值计算

3. **数据验证**
   - 运行 `python python/tools/qlib/verify_exported_data.py` 验证数据完整性

## 故障排除

### Qlib 初始化失败
- 检查数据目录是否存在
- 运行验证脚本检查数据完整性
- 确认数据目录路径正确

### 数据加载失败
- 检查 `features/` 目录是否存在
- 确认股票代码格式正确（如 `000001.SZ`）
- 检查日期范围是否在数据范围内

### 特征提取失败
- 确认数据量足够（至少需要计算周期的数据量）
- 检查特征表达式语法是否正确

## 相关文档

- [Qlib 官方文档](https://qlib.readthedocs.io/)
- [Qlib 脚本使用指南](./QLIB_SCRIPTS_GUIDE.md)
- [项目贡献指南](../../../docs/CONTRIBUTING.md)



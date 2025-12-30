# Structure Expert 模型回测指南

## 概述

本指南介绍如何使用训练好的 Structure Expert GNN 模型进行回测。

Structure Expert 是一个基于图神经网络（GNN）的股票预测模型，它利用行业关系构建图结构，通过图注意力机制（GATv2）学习股票之间的关联性。

## 快速开始

### 1. 准备模型文件

确保你已经训练好了 Structure Expert 模型，并保存为 `.pth` 文件：

```bash
# 训练模型（如果还没有训练）
python python/tools/qlib/train/train_structure_expert.py \
    --start-date 2024-01-01 \
    --end-date 2024-12-31 \
    --qlib-dir ~/.qlib/qlib_data/cn_data
```

训练完成后，模型会保存在 `models/structure_expert.pth`（或你指定的路径）。

### 2. 检查模型文件

在回测之前，建议先检查模型文件是否正常：

```bash
python python/tools/qlib/train/check_structure_expert.py \
    --model_path models/structure_expert.pth
```

### 3. 运行回测

基本用法：

```bash
python python/examples/backtest_structure_expert.py \
    --model_path models/structure_expert.pth \
    --start_date 2024-01-01 \
    --end_date 2024-06-30
```

## 详细参数说明

### 必需参数

- `--model_path`: 训练好的模型文件路径（`.pth` 文件）
- `--start_date`: 回测开始日期（格式：YYYY-MM-DD）
- `--end_date`: 回测结束日期（格式：YYYY-MM-DD）

### 可选参数

#### 模型参数

- `--n_feat`: 输入特征数量（默认：158，对应 Alpha158）
- `--n_hidden`: 隐藏层大小（默认：128）
- `--n_heads`: 注意力头数（默认：8）
- `--device`: 运行设备（默认：cuda，如果可用，否则 cpu）

**注意**：这些参数应该与训练时使用的参数一致。如果模型文件正常，脚本会自动推断这些参数。

#### 回测参数

- `--top_k`: 选择前 K 只股票（默认：30）
- `--initial_cash`: 初始资金（默认：1,000,000）
- `--strategy`: 策略类名（默认：TopkDropoutStrategy）
- `--benchmark`: 基准指数代码（如：SH000300）。如果不提供，脚本会尝试自动检测
- `--no_benchmark`: 明确禁用基准对比

#### 数据配置

- `--qlib_dir`: Qlib 数据目录（默认：~/.qlib/qlib_data/cn_data）
- `--region`: Qlib 区域（默认：cn）
- `--config_path`: 配置文件路径（用于数据库配置，加载行业映射）

#### 输出选项

- `--save_results`: 保存回测结果到文件

## 使用示例

### 示例 1：基本回测

```bash
python python/examples/backtest_structure_expert.py \
    --model_path models/structure_expert.pth \
    --start_date 2024-01-01 \
    --end_date 2024-06-30
```

### 示例 2：自定义参数

```bash
python python/examples/backtest_structure_expert.py \
    --model_path models/structure_expert.pth \
    --start_date 2024-01-01 \
    --end_date 2024-06-30 \
    --top_k 50 \
    --initial_cash 2000000 \
    --n_feat 158 \
    --n_hidden 128 \
    --n_heads 8
```

### 示例 3：指定基准指数

```bash
python python/examples/backtest_structure_expert.py \
    --model_path models/structure_expert.pth \
    --start_date 2024-01-01 \
    --end_date 2024-06-30 \
    --benchmark SH000300
```

### 示例 4：保存结果

```bash
python python/examples/backtest_structure_expert.py \
    --model_path models/structure_expert.pth \
    --start_date 2024-01-01 \
    --end_date 2024-06-30 \
    --save_results
```

结果会保存到：`outputs/structure_expert_backtest_2024-01-01_2024-06-30.csv`

### 示例 5：使用数据库配置（推荐）

如果你有数据库配置，可以加载行业映射以获得更好的图结构：

```bash
python python/examples/backtest_structure_expert.py \
    --model_path models/structure_expert.pth \
    --start_date 2024-01-01 \
    --end_date 2024-06-30 \
    --config_path config/config.yaml
```

## 回测流程说明

### 1. 模型加载

脚本会：
- 加载模型权重（`.pth` 文件）
- 初始化模型架构（GATv2Conv + Linear layers）
- 将模型设置为评估模式（`model.eval()`）

### 2. 数据准备

对于每个交易日：
- 加载 Alpha158 特征
- 加载行业映射（如果提供了数据库配置）
- 构建图结构（同行业股票之间建立边）

### 3. 预测生成

- 对每个交易日的图数据进行前向传播
- 获取每只股票的预测分数
- 将预测结果转换为 Qlib 格式的信号

### 4. 策略构建

- 使用 `TopkDropoutStrategy` 策略
- 选择预测分数最高的 K 只股票
- 定期调仓（每天）

### 5. 回测执行

- 使用 Qlib 的回测框架
- 模拟交易执行（考虑交易成本、涨跌停限制等）
- 计算组合收益、夏普比率、最大回撤等指标

### 6. 结果展示

输出包括：
- 组合收益率
- 年化收益率
- 波动率
- 夏普比率
- 最大回撤
- 其他性能指标

## 输出结果解读

### 组合指标

- **Return**: 总收益率
- **Return.annualized**: 年化收益率
- **Volatility**: 波动率（标准差）
- **Sharpe**: 夏普比率（风险调整后收益）
- **Max Drawdown**: 最大回撤

### 基准对比

如果指定了基准指数，会显示：
- 相对基准的超额收益
- 信息比率
- 跟踪误差

## 常见问题

### 1. 模型参数不匹配

**问题**：`RuntimeError: Error(s) in loading state_dict`

**解决**：
- 检查模型参数（`n_feat`, `n_hidden`, `n_heads`）是否与训练时一致
- 使用 `check_structure_expert.py` 检查模型文件，它会自动推断参数

### 2. 没有数据

**问题**：`No data for YYYY-MM-DD, skipping`

**解决**：
- 检查 Qlib 数据是否完整
- 确认日期范围内有交易日
- 检查股票池是否包含数据

### 3. 行业映射缺失

**问题**：`No database config, using empty industry map`

**解决**：
- 提供 `--config_path` 参数指向配置文件
- 确保数据库中有行业成员数据
- 如果没有数据库，模型仍可运行，但图结构会较弱（无边连接）

### 4. 内存不足

**问题**：`CUDA out of memory` 或系统内存不足

**解决**：
- 使用 `--device cpu` 强制使用 CPU
- 减少回测日期范围
- 减少 `top_k` 参数

### 5. 基准指数不存在

**问题**：`The benchmark ['XXX'] does not exist`

**解决**：
- 检查基准代码是否正确（如：SH000300, 000300.SH）
- 使用 `--no_benchmark` 禁用基准对比
- 脚本会自动尝试常见格式，如果都失败会跳过基准

## 性能优化建议

1. **使用 GPU**：如果可用，使用 `--device cuda` 加速推理
2. **批量处理**：脚本已经按日期批量处理，无需额外优化
3. **数据缓存**：Qlib 会自动缓存数据，首次运行可能较慢

## 与训练脚本的关系

回测脚本使用的模型参数应该与训练脚本一致：

| 参数 | 训练脚本 | 回测脚本 |
|------|---------|---------|
| n_feat | 158 (Alpha158) | --n_feat 158 |
| n_hidden | 128 (默认) | --n_hidden 128 |
| n_heads | 8 (默认) | --n_heads 8 |

如果不确定，可以：
1. 查看训练日志
2. 使用 `check_structure_expert.py` 自动推断参数

## 下一步

- 查看 [训练指南](TRAIN_STRUCTURE_EXPERT_GUIDE.md) 了解如何训练模型
- 查看 [模型检查指南](check_structure_expert.py) 了解如何验证模型
- 查看 [Qlib 回测文档](https://qlib.readthedocs.io/) 了解更多回测选项

## 相关文件

- 回测脚本：`python/examples/backtest_structure_expert.py`
- 训练脚本：`python/tools/qlib/train/train_structure_expert.py`
- 模型定义：`python/tools/qlib/train/structure_expert.py`
- 模型检查：`python/tools/qlib/train/check_structure_expert.py`


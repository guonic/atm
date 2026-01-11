# 完整模型回测指南（买入+卖出）

## 文件说明

### `backtest_exit_model.py` - 完整的买入+卖出模型回测脚本 ⭐

**这是集成了买入和卖出模型的完整回测脚本。**

它包含：
- ✅ **买入模型**：Structure Expert GNN 模型（生成选股信号）
- ✅ **卖出模型**：Exit Model（决定何时卖出）
- ✅ 完整的回测流程
- ✅ 结果对比功能

### `backtest_structure_expert_ml_exit.py` - 策略类定义

这是策略类的定义文件，定义了 `MLExitStrategy` 类：
- 继承自 `RefinedTopKStrategy`（买入逻辑）
- 集成了 Exit 模型（卖出逻辑）
- 不是独立的回测脚本，需要被其他脚本调用

## 完整回测流程

### 架构图

```
┌─────────────────────────────────────────────────┐
│     backtest_exit_model.py (完整回测脚本)        │
├─────────────────────────────────────────────────┤
│                                                  │
│  1. 加载 Structure Expert 模型 (买入模型)        │
│     ↓                                            │
│  2. 生成每日预测分数 (选股信号)                  │
│     ↓                                            │
│  3. 加载 Exit 模型 (卖出模型)                    │
│     ↓                                            │
│  4. 创建 MLExitStrategy                          │
│     ├─ 买入：基于 Structure Expert 预测分数      │
│     └─ 卖出：基于 Exit 模型风险概率              │
│     ↓                                            │
│  5. 运行回测                                     │
│     ↓                                            │
│  6. 输出结果对比                                 │
└─────────────────────────────────────────────────┘
```

## 使用方法

### 基本用法

```bash
python python/examples/backtest_exit_model.py \
    --model_path models/structure_expert.pth \
    --exit_model_path models/exit_model.pkl \
    --start_date 2024-01-01 \
    --end_date 2024-12-31
```

### 对比回测（推荐）

对比使用完整模型（买入+卖出）和仅使用买入模型的效果：

```bash
python python/examples/backtest_exit_model.py \
    --model_path models/structure_expert.pth \
    --exit_model_path models/exit_model.pkl \
    --start_date 2024-01-01 \
    --end_date 2024-12-31 \
    --compare
```

### 完整参数示例

```bash
python python/examples/backtest_exit_model.py \
    --model_path models/structure_expert_directional.pth \
    --exit_model_path models/exit_model.pkl \
    --start_date 2024-01-01 \
    --end_date 2024-12-31 \
    --top_k 30 \
    --exit_threshold 0.65 \
    --buffer_ratio 0.15 \
    --initial_cash 1000000 \
    --benchmark SH000300 \
    --compare
```

## 工作流程详解

### 1. 买入逻辑（Structure Expert 模型）

```python
# Structure Expert 模型生成预测分数
predictions = generate_predictions(
    model=structure_expert_model,  # 买入模型
    builder=graph_builder,
    start_date=start_date,
    end_date=end_date,
)

# 策略根据分数选择 Top K 股票
strategy = MLExitStrategy(
    signal=predictions,  # 买入信号（预测分数）
    topk=30,  # 选择前30只股票
)
```

**买入时机**：
- 每日根据 Structure Expert 模型的预测分数
- 选择分数最高的 Top K 股票
- 使用 RefinedTopKStrategy 的缓冲机制降低换手率

### 2. 卖出逻辑（Exit 模型）

```python
# Exit 模型预测退出概率
for symbol in active_positions:
    risk_prob = exit_model.predict_proba(
        daily_df=historical_data,
        entry_price=position.entry_price,
        highest_price_since_entry=position.max_price,
        days_held=position.days_held,
    )
    
    # 如果风险概率超过阈值，卖出
    if risk_prob > exit_threshold:  # 默认 0.65
        close_position(symbol, reason="ML_Exit")
```

**卖出时机**：
- 每日检查每个持仓
- Exit 模型计算风险概率
- 如果风险概率 > 阈值（如 0.65），触发卖出

### 3. 完整交易流程

```
每个交易日：
├─ 1. Structure Expert 模型生成预测分数
├─ 2. 根据分数选择新的买入标的（Top K）
├─ 3. 对于每个持仓：
│   ├─ 计算 Exit 模型特征
│   ├─ 预测退出概率
│   └─ 如果概率 > 阈值 → 卖出
└─ 4. 执行买卖订单
```

## 模型文件说明

### 买入模型（Structure Expert）

- **文件**：`models/structure_expert.pth` 或 `models/structure_expert_directional.pth`
- **作用**：生成每日股票预测分数，用于选股
- **输入**：股票特征（Alpha158）和行业关系图
- **输出**：每只股票的预测分数（score）

### 卖出模型（Exit Model）

- **文件**：`models/exit_model.pkl`
- **作用**：预测持仓是否应该卖出
- **输入**：动量枯竭特征 + 持仓管理特征
- **输出**：退出概率（0-1）

## 参数调优

### 买入参数

- `--top_k`: 持仓数量（默认 30）
- `--buffer_ratio`: 缓冲比例，降低换手率（默认 0.15）

### 卖出参数

- `--exit_threshold`: 退出概率阈值（默认 0.65）
  - 较低值（0.55-0.6）：更频繁卖出，保护盈利
  - 中等值（0.65-0.7）：平衡（推荐）
  - 较高值（0.75-0.8）：较少卖出，可能错过最佳时机

## 结果解读

使用 `--compare` 参数会输出：

```
Baseline (仅买入模型):
  Total Return: 15.23%
  Max Drawdown: -8.45%
  Sharpe Ratio: 1.23

ML Exit Strategy (买入+卖出模型):
  Total Return: 18.67%
  Max Drawdown: -5.12%  ← 改善！
  Sharpe Ratio: 1.45    ← 改善！

Improvement: +3.44%
```

**预期改善**：
- ✅ 最大回撤降低（保护盈利）
- ✅ 盈亏比提高（及时止盈）
- ✅ 总收益率可能提高（避免利润回吐）
- ⚠️ 胜率可能降低（更早卖出）
- ⚠️ 换手率可能提高（更频繁交易）

## 快速开始

### 完整流程

```bash
# 1. 训练买入模型（如果还没有）
python python/tools/qlib/train/train_structure_expert.py \
    --start-date 2023-01-01 --end-date 2023-12-31 \
    --save-model models/structure_expert.pth

# 2. 训练卖出模型（如果还没有）
python python/tools/qlib/train/train_exit_model.py \
    --data outputs/exit_training_data.csv \
    --model models/exit_model.pkl

# 3. 运行完整回测（买入+卖出）
python python/examples/backtest_exit_model.py \
    --model_path models/structure_expert.pth \
    --exit_model_path models/exit_model.pkl \
    --start_date 2024-01-01 \
    --end_date 2024-12-31 \
    --compare
```

## 与其他回测脚本的区别

| 脚本 | 买入模型 | 卖出模型 | 用途 |
|------|---------|---------|------|
| `backtest_structure_expert.py` | ✅ Structure Expert | ❌ 规则-based | 仅测试买入模型 |
| `backtest_exit_model.py` | ✅ Structure Expert | ✅ Exit Model | **完整模型回测** ⭐ |

## 总结

**`backtest_exit_model.py` 是集成了买入和卖出模型的完整回测脚本。**

它同时使用：
- **Structure Expert 模型**：决定买什么（选股）
- **Exit 模型**：决定何时卖（退出时机）

这是最完整的回测方案，能够全面评估买入和卖出模型的综合效果。

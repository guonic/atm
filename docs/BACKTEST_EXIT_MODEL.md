# 退出模型回测指南

## 快速开始

### 1. 准备模型文件

确保你有以下模型文件：
- Structure Expert 模型：`models/structure_expert.pth`（或 `models/structure_expert_directional.pth`）
- 退出模型：`models/exit_model.pkl`
- 特征缩放器：`models/exit_model_scaler.pkl`（自动生成）

### 2. 基本回测命令

```bash
python python/examples/backtest_exit_model.py \
    --model_path models/structure_expert.pth \
    --exit_model_path models/exit_model.pkl \
    --start_date 2024-01-01 \
    --end_date 2024-06-30
```

### 3. 对比回测（推荐）

对比使用退出模型和不使用退出模型的效果：

```bash
python python/examples/backtest_exit_model.py \
    --model_path models/structure_expert.pth \
    --exit_model_path models/exit_model.pkl \
    --start_date 2024-01-01 \
    --end_date 2024-06-30 \
    --compare
```

## 完整示例

### 示例1：基本回测

```bash
python python/examples/backtest_exit_model.py \
    --model_path models/structure_expert_directional.pth \
    --exit_model_path models/exit_model.pkl \
    --start_date 2024-01-01 \
    --end_date 2024-12-31 \
    --top_k 30 \
    --initial_cash 1000000 \
    --benchmark SH000300
```

### 示例2：调整退出阈值

尝试不同的退出阈值，找到最优值：

```bash
# 较低阈值（更频繁卖出）
python python/examples/backtest_exit_model.py \
    --model_path models/structure_expert.pth \
    --exit_model_path models/exit_model.pkl \
    --start_date 2024-01-01 \
    --end_date 2024-12-31 \
    --exit_threshold 0.55

# 中等阈值（默认）
python python/examples/backtest_exit_model.py \
    --model_path models/structure_expert.pth \
    --exit_model_path models/exit_model.pkl \
    --start_date 2024-01-01 \
    --end_date 2024-12-31 \
    --exit_threshold 0.65

# 较高阈值（较少卖出）
python python/examples/backtest_exit_model.py \
    --model_path models/structure_expert.pth \
    --exit_model_path models/exit_model.pkl \
    --start_date 2024-01-01 \
    --end_date 2024-12-31 \
    --exit_threshold 0.75
```

### 示例3：对比分析

```bash
python python/examples/backtest_exit_model.py \
    --model_path models/structure_expert.pth \
    --exit_model_path models/exit_model.pkl \
    --start_date 2024-01-01 \
    --end_date 2024-12-31 \
    --compare \
    --top_k 30 \
    --buffer_ratio 0.15
```

这会输出：
- Baseline（RefinedTopKStrategy）的表现
- ML Exit Strategy 的表现
- 改进幅度对比

## 参数说明

### 必需参数

- `--model_path`: Structure Expert 模型路径（.pth 文件）
- `--exit_model_path`: 退出模型路径（.pkl 文件）
- `--start_date`: 回测开始日期（YYYY-MM-DD）
- `--end_date`: 回测结束日期（YYYY-MM-DD）

### 可选参数

- `--exit_scaler_path`: 特征缩放器路径（默认自动从模型路径生成）
- `--exit_threshold`: 退出概率阈值（默认 0.65）
  - 较低值（0.5-0.6）：更频繁卖出，保护盈利但可能过早退出
  - 中等值（0.65-0.7）：平衡卖出频率和盈利保护
  - 较高值（0.75-0.8）：较少卖出，可能错过最佳退出时机
- `--top_k`: 持仓股票数量（默认 30）
- `--buffer_ratio`: 缓冲比例，用于降低换手率（默认 0.15）
- `--initial_cash`: 初始资金（默认 1,000,000）
- `--benchmark`: 基准指数代码（如 'SH000300' 或 '000300.SH'）
- `--compare`: 是否对比有/无退出模型的效果
- `--qlib_dir`: Qlib 数据目录（默认 ~/.qlib/qlib_data/cn_data）
- `--device`: 计算设备（默认 cuda，如果不可用则使用 cpu）
- `--config`: 配置文件路径（默认 config/config.yaml）

## 回测结果解读

### 关键指标

1. **Total Return（总收益率）**
   - 整个回测期间的总收益率
   - 对比有/无退出模型的差异

2. **Annual Return（年化收益率）**
   - 年化后的收益率
   - 便于跨时间段比较

3. **Sharpe Ratio（夏普比率）**
   - 风险调整后的收益
   - 越高越好，通常 > 1 为良好

4. **Max Drawdown（最大回撤）**
   - 从峰值到谷底的最大跌幅
   - 退出模型应该能降低最大回撤

5. **Win Rate（胜率）**
   - 盈利交易占比
   - 退出模型可能降低胜率但提高盈亏比

### 预期效果

使用退出模型后，你应该看到：

✅ **改善的指标**：
- 最大回撤降低（保护盈利）
- 盈亏比提高（及时止盈）
- 总收益率可能提高（避免利润回吐）

⚠️ **可能的变化**：
- 胜率可能降低（更早卖出）
- 换手率可能提高（更频繁交易）
- 交易次数增加（更多退出信号）

## 调优建议

### 1. 退出阈值调优

通过网格搜索找到最优阈值：

```bash
for threshold in 0.55 0.60 0.65 0.70 0.75; do
    echo "Testing threshold: $threshold"
    python python/examples/backtest_exit_model.py \
        --model_path models/structure_expert.pth \
        --exit_model_path models/exit_model.pkl \
        --start_date 2024-01-01 \
        --end_date 2024-12-31 \
        --exit_threshold $threshold \
        --compare
done
```

### 2. 不同市场环境测试

在不同市场环境下测试退出模型：

```bash
# 牛市
python python/examples/backtest_exit_model.py \
    --model_path models/structure_expert.pth \
    --exit_model_path models/exit_model.pkl \
    --start_date 2020-01-01 \
    --end_date 2021-12-31 \
    --compare

# 熊市
python python/examples/backtest_exit_model.py \
    --model_path models/structure_expert.pth \
    --exit_model_path models/exit_model.pkl \
    --start_date 2022-01-01 \
    --end_date 2022-12-31 \
    --compare

# 震荡市
python python/examples/backtest_exit_model.py \
    --model_path models/structure_expert.pth \
    --exit_model_path models/exit_model.pkl \
    --start_date 2023-01-01 \
    --end_date 2023-12-31 \
    --compare
```

### 3. 与现有策略对比

你也可以在现有的回测脚本中使用退出模型：

```python
# 修改 backtest_structure_expert.py 中的策略创建部分
from examples.backtest_structure_expert_ml_exit import MLExitStrategy

strategy = MLExitStrategy(
    signal=predictions,
    topk=30,
    exit_model_path='models/exit_model.pkl',
    exit_threshold=0.65,
    use_ml_exit=True,
)
```

## 故障排查

### 问题1：模型文件不存在

**错误**：`FileNotFoundError: Model file not found`

**解决**：
```bash
# 检查模型文件是否存在
ls -lh models/exit_model.pkl
ls -lh models/exit_model_scaler.pkl

# 如果不存在，先训练模型
python python/tools/qlib/train/train_exit_model.py \
    --data outputs/exit_training_data.csv \
    --model models/exit_model.pkl
```

### 问题2：退出模型未触发

**现象**：回测结果与不使用退出模型相同

**可能原因**：
- 退出阈值设置过高
- 模型预测概率总是低于阈值
- 策略集成有问题

**解决**：
1. 降低退出阈值（如 0.55）
2. 检查日志中的退出信号
3. 验证模型是否正确加载

### 问题3：性能没有改善

**可能原因**：
- 退出模型训练数据不足
- 特征与当前市场不匹配
- 阈值设置不当

**解决**：
1. 使用更多训练数据重新训练
2. 调整特征工程
3. 尝试不同的阈值

## 进阶使用

### 自定义退出逻辑

你可以在策略中自定义退出逻辑：

```python
class CustomExitStrategy(MLExitStrategy):
    def _check_exit_signal(self, symbol, date, price):
        # 先检查硬性规则（止损、止盈）
        if self._check_stop_loss(symbol, price):
            return True
        
        if self._check_take_profit(symbol, price):
            return True
        
        # 再检查 ML 退出信号
        return super()._check_exit_signal(symbol, date, price)
```

### 多模型集成

可以同时使用多个退出模型：

```python
# 加载多个退出模型
exit_model_1 = ExitModel.load('models/exit_model_v1.pkl')
exit_model_2 = ExitModel.load('models/exit_model_v2.pkl')

# 投票机制
proba_1 = exit_model_1.predict_proba(...)
proba_2 = exit_model_2.predict_proba(...)
avg_proba = (proba_1 + proba_2) / 2

if avg_proba > threshold:
    exit_position()
```

## 相关文档

- [退出模型使用指南](./EXIT_MODEL_USAGE.md) - 详细的使用说明
- [退出策略分析](./EXIT_STRATEGY_ANALYSIS.md) - 策略原理和设计思路
- [退出模型训练文档](./exit_train.md) - 原始设计文档

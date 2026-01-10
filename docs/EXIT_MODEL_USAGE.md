# 退出模型使用指南

## 概述

退出模型（Exit Model）是一个基于机器学习的卖出预测系统，用于在动能耗尽和利润回撤之前及时卖出持仓，保护盈利并避免亏损扩大。

## 架构

### 核心模块

1. **特征工程模块** (`python/nq/analysis/exit/feature_builder.py`)
   - 构建动量枯竭指标
   - 构建持仓管理特征
   - 生成训练标签

2. **退出模型类** (`python/nq/analysis/exit/exit_model.py`)
   - 训练和保存模型
   - 加载模型进行预测
   - 输出退出概率

3. **数据提取脚本** (`python/tools/qlib/extract_exit_training_data.py`)
   - 从回测结果中提取持仓快照
   - 生成训练数据

4. **模型训练脚本** (`python/tools/qlib/train/train_exit_model.py`)
   - 训练退出模型
   - 保存模型和特征缩放器

5. **策略集成** (`python/examples/backtest_structure_expert_ml_exit.py`)
   - MLExitStrategy 类
   - 集成退出模型到回测策略

## 使用流程

### 第一步：提取训练数据

从回测结果中提取持仓期间的逐日快照：

```python
from python.tools.qlib.extract_exit_training_data import extract_from_executed_orders

# 从策略实例获取已执行的订单
executed_orders = strategy_instance.get_executed_orders()

# 提取持仓快照
snapshots_df = extract_from_executed_orders(executed_orders, price_data)

# 保存到 CSV
snapshots_df.to_csv('outputs/exit_training_data.csv', index=False)
```

**数据格式要求**：
- `trade_id`: 交易ID
- `date`: 日期
- `symbol`: 股票代码
- `close`, `high`, `low`, `volume`: OHLCV 数据
- `entry_price`: 买入价格
- `highest_price_since_entry`: 持仓期间最高价
- `days_held`: 持仓天数
- `next_3d_max_loss`: 未来3天最大跌幅（用于标签）

### 第二步：训练模型

使用训练脚本训练模型：

```bash
python python/tools/qlib/train/train_exit_model.py \
    --data outputs/exit_training_data.csv \
    --model models/exit_model.pkl \
    --C 0.1 \
    --class-weight balanced
```

**参数说明**：
- `--data`: 训练数据 CSV 文件路径
- `--model`: 模型保存路径
- `--C`: 正则化强度（默认 0.1）
- `--class-weight`: 类别权重策略（balanced 或 None）
- `--ma-period`: 移动平均周期（默认 5）
- `--volume-ma-period`: 成交量移动平均周期（默认 5）

### 第三步：在策略中使用

#### 方法1：使用 MLExitStrategy

```python
from examples.backtest_structure_expert_ml_exit import MLExitStrategy

# 创建策略实例
strategy = MLExitStrategy(
    signal=signal_df,
    topk=30,
    buffer_ratio=0.15,
    exit_model_path='models/exit_model.pkl',
    exit_threshold=0.65,  # 风险概率阈值
    use_ml_exit=True,
)

# 运行回测
results = run_backtest(strategy, start_date, end_date)
```

#### 方法2：手动集成到现有策略

```python
from nq.analysis.exit import ExitModel

class MyStrategy(RefinedTopKStrategy):
    def __init__(self, ...):
        super().__init__(...)
        # 加载退出模型
        self.exit_model = ExitModel.load(
            model_path='models/exit_model.pkl',
            threshold=0.65,
        )
        self.position_tracker = {}
    
    def _check_exit_signal(self, symbol, date, price):
        """检查退出信号"""
        if symbol not in self.position_tracker:
            return False
        
        pos = self.position_tracker[symbol]
        
        # 获取历史数据
        hist_data = D.features(
            instruments=[symbol],
            fields=["$close", "$high", "$low", "$volume"],
            start_time=(date - pd.Timedelta(days=15)).strftime("%Y-%m-%d"),
            end_time=date.strftime("%Y-%m-%d"),
        )
        
        # 构造特征并预测
        daily_df = pd.DataFrame({
            'close': hist_data[symbol]['$close'],
            'high': hist_data[symbol]['$high'],
            'low': hist_data[symbol]['$low'],
            'volume': hist_data[symbol]['$volume'],
        })
        
        proba = self.exit_model.predict_proba(
            daily_df=daily_df,
            entry_price=pos['entry_price'],
            highest_price_since_entry=pos['highest_price_since_entry'],
            days_held=(date - pos['entry_date']).days,
        )
        
        return proba[-1] > 0.65 if len(proba) > 0 else False
```

## 特征说明

### 1. 动量枯竭指标

- **bias_5**: 价格偏离 5 日均线的程度
  - 公式: `(close - ma5) / ma5`
  - 信号: `bias_5 > 0.1` 且 `vol_ratio < 0.8` → 强卖出信号

- **close_pos**: 收盘价在当日波幅中的位置 (0-1)
  - 公式: `(close - low) / (high - low)`
  - 信号: 连续多日接近 0 → 阴跌信号

- **vol_ratio**: 成交量衰减
  - 公式: `volume / volume_ma5`
  - 信号: `vol_ratio < 0.8` → 缩量上涨，动能衰竭

### 2. 持仓管理指标

- **curr_ret**: 当前收益率
  - 公式: `(close - entry_price) / entry_price`

- **drawdown**: 利润回撤（最重要）
  - 公式: `(highest_price - close) / highest_price`
  - 作用: 保护盈利的核心指标
  - 逻辑: 当收益从 5% 回落到 2% 时，立即卖出

- **days_held**: 持仓天数
  - 作用: 时效性特征，避免持仓过久

## 模型参数调优

### 阈值调整

退出概率阈值（`exit_threshold`）影响卖出频率：

- **较低阈值（0.5-0.6）**: 更频繁卖出，保护盈利但可能过早退出
- **中等阈值（0.65-0.7）**: 平衡卖出频率和盈利保护
- **较高阈值（0.75-0.8）**: 较少卖出，可能错过最佳退出时机

建议通过回测找到最优阈值。

### 正则化强度

`C` 参数控制模型复杂度：

- **较小 C（0.01-0.1）**: 更强的正则化，防止过拟合
- **较大 C（1.0-10.0）**: 更复杂的模型，可能过拟合

默认 0.1 通常效果较好。

## 最佳实践

### 1. 数据质量

- 确保训练数据包含足够的正样本（应卖出）和负样本（应持有）
- 数据时间跨度应覆盖不同市场环境（牛市、熊市、震荡市）
- 避免未来信息泄露（标签应基于未来数据，但特征不能使用未来数据）

### 2. 模型更新

- 定期用新数据重新训练模型（建议每季度或每半年）
- 监控模型在回测和实盘中的表现
- 如果表现下降，考虑调整特征或重新训练

### 3. 特征工程

- 可以根据实际效果添加新特征
- 例如：技术指标（RSI、MACD）、市场情绪指标等
- 注意特征之间的相关性，避免多重共线性

### 4. 集成策略

- ML 退出模型应与规则-based 退出逻辑结合使用
- 例如：止损、止盈等硬性规则 + ML 退出模型
- 设置优先级：硬性规则 > ML 退出模型

## 故障排查

### 问题1：模型预测概率总是很低/很高

**原因**：特征分布与训练时不一致

**解决**：
- 检查特征计算是否正确
- 确保使用相同的特征缩放器
- 检查数据质量

### 问题2：模型总是建议卖出/持有

**原因**：阈值设置不当或模型训练不充分

**解决**：
- 调整退出阈值
- 检查训练数据的标签分布
- 尝试调整 `class_weight` 参数

### 问题3：模型加载失败

**原因**：模型文件路径错误或文件损坏

**解决**：
- 检查模型文件是否存在
- 确认 scaler 文件路径正确
- 重新训练模型

## 示例代码

完整的使用示例请参考：
- `python/examples/backtest_structure_expert_ml_exit.py`
- `python/tools/qlib/train/train_exit_model.py`

## 相关文档

- [退出策略分析](./EXIT_STRATEGY_ANALYSIS.md) - 策略原理和设计思路
- [退出模型训练文档](./exit_train.md) - 原始设计文档

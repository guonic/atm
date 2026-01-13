# 从交易框架回测结果训练 Exit Model

## 概述

新的交易框架（`python/nq/trading/`）的回测结果**完全支持**用于训练 Exit Model。

框架在回测过程中会自动记录：
- 每日持仓快照（包含 entry_price, highest_price_since_entry, days_held 等）
- 订单历史（买入/卖出记录）
- 账户快照（每日净值、持仓数量等）

## 数据流程

```
回测执行 (run_custom_backtest)
    ↓
每日更新持仓状态 (PositionManager.update_positions)
    ↓
保存持仓快照到 Storage (storage.save("snapshot:date:symbol", snapshot))
    ↓
提取训练数据 (extract_from_backtest_results)
    ↓
添加未来标签 (add_future_labels)
    ↓
保存为 CSV (save_training_data)
    ↓
训练 Exit Model (train_exit_model.py)
```

## 使用步骤

### 1. 运行回测

使用新的交易框架运行回测：

```bash
python python/examples/backtest_trading_framework.py \
    --model_path models/structure_expert_directional.pth \
    --exit_model_path models/exit_model.pkl \
    --start_date 2025-07-01 \
    --end_date 2025-08-01 \
    --initial_cash 1000000
```

### 2. 提取训练数据

从回测结果中提取训练数据：

```python
from nq.trading.backtest import run_custom_backtest
from nq.trading.backtest.extract_training_data import extract_from_backtest_results, save_training_data

# 假设你已经运行了回测并保存了结果
# results = run_custom_backtest(...)

# 提取训练数据
training_data = extract_from_backtest_results(
    results=results,
    start_date="2025-07-01",
    end_date="2025-08-01",
    add_future_labels=True,
    future_days=3,
    loss_threshold=-0.03,
)

# 保存为 CSV
save_training_data(training_data, "outputs/exit_training_data.csv")
```

### 3. 训练 Exit Model

使用提取的数据训练模型：

```bash
python python/tools/qlib/train/train_exit_model.py \
    --data outputs/exit_training_data.csv \
    --output models/exit_model.pkl
```

## 数据格式

提取的训练数据包含以下列：

### 必需列（用于特征构建）
- `symbol`: 股票代码
- `date`: 快照日期
- `close`: 收盘价
- `high`: 最高价
- `low`: 最低价
- `volume`: 成交量
- `entry_price`: 买入均价
- `highest_price_since_entry`: 自买入以来的最高价
- `days_held`: 持仓天数

### 标签列（用于训练）
- `label`: 退出标签（1 = 应该卖出, 0 = 继续持有）
  - 标签规则：未来 3 天内最大跌幅超过 3% 或收益变负，则标记为 1

### 特征列（由 ExitFeatureBuilder 生成）
- `bias_5`: 偏离 5 日均线的程度
- `close_pos`: 收盘价在当日波幅中的位置（0-1）
- `vol_ratio`: 成交量相对 5 日均量的比率
- `curr_ret`: 当前收益率
- `drawdown`: 从最高价回撤的幅度
- `days_held`: 持仓天数

## 数据来源

### 持仓快照（Position Snapshots）

在 `PositionManager.update_positions()` 中，每日收盘后会保存快照：

```python
snapshot = {
    'date': date.strftime('%Y-%m-%d'),
    'symbol': symbol,
    'entry_price': position.entry_price,
    'current_price': current_close,
    'amount': position.amount,
    'high_price_since_entry': position.high_price_since_entry,
    'current_return': position.calculate_return(current_close),
    'drawdown': position.calculate_drawdown(current_close),
    'market_value': position.calculate_market_value(current_close),
}
storage.save(f"snapshot:{date.strftime('%Y-%m-%d')}:{symbol}", snapshot)
```

### 订单历史（Order History）

所有订单都保存在 `OrderBook` 中，可以通过 `order_book.orders` 访问。

## 注意事项

1. **存储后端**：确保使用 `MemoryStorage` 或支持 `get_all_keys()` 的存储后端
2. **未来标签**：需要从 Qlib 加载未来数据来计算标签，确保数据范围足够
3. **数据量**：回测时间越长，生成的训练数据越多
4. **标签质量**：未来标签基于未来 3 天的表现，需要确保数据完整性

## 完整示例

```python
#!/usr/bin/env python3
"""Complete example: Run backtest and extract training data."""

from nq.trading.backtest import run_custom_backtest
from nq.trading.strategies import DualModelStrategy
from nq.trading.strategies.buy_models import StructureExpertBuyModel
from nq.trading.strategies.sell_models import MLExitSellModel
from nq.trading.state import Account, PositionManager, OrderBook
from nq.trading.logic import RiskManager, PositionAllocator
from nq.trading.storage import MemoryStorage
from nq.trading.backtest.extract_training_data import (
    extract_from_backtest_results,
    save_training_data,
)

# ... 初始化所有组件 ...

# 运行回测
results = run_custom_backtest(
    strategy=strategy,
    start_date="2025-07-01",
    end_date="2025-08-01",
    initial_cash=1000000.0,
)

# 提取训练数据
training_data = extract_from_backtest_results(
    results=results,
    start_date="2025-07-01",
    end_date="2025-08-01",
    add_future_labels=True,
)

# 保存训练数据
save_training_data(training_data, "outputs/exit_training_data.csv")

print(f"Extracted {len(training_data)} training samples")
print(f"Positive labels: {training_data['label'].sum()}")
```

## 优势

1. **自动记录**：回测过程中自动记录所有持仓快照
2. **完整信息**：包含 entry_price, highest_price_since_entry, days_held 等关键信息
3. **真实场景**：基于实际回测结果，更贴近真实交易场景
4. **易于扩展**：可以轻松添加更多特征或标签规则

## 总结

新的交易框架的回测数据**完全支持**训练 Exit Model。框架在回测过程中自动记录所有必要的持仓信息，只需要使用 `extract_from_backtest_results()` 函数提取数据即可。

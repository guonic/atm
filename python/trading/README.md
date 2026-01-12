# Trading System Framework

模块化交易系统框架，实现买卖逻辑分离，完全独立于 Qlib 的状态管理。

## 架构设计

```
python/trading/
├── interfaces/          # 接口抽象层
│   ├── models.py       # IBuyModel, ISellModel
│   └── storage.py      # IStorageBackend
├── state/              # 状态层（完全独立于 Qlib）
│   ├── account.py      # Account
│   ├── position.py     # Position, PositionManager
│   └── orderbook.py    # Order, OrderBook
├── execution/          # 执行层
│   └── executor.py     # Executor
├── logic/              # 逻辑层
│   ├── risk.py         # RiskManager
│   └── allocation.py   # PositionAllocator
├── strategy/           # 策略层
│   ├── base.py         # DualModelStrategy
│   ├── buy_models/      # 买入模型实现
│   │   └── structure_expert.py
│   └── sell_models/    # 卖出模型实现
│       └── ml_exit.py
├── storage/            # 存储层
│   ├── memory.py       # MemoryStorage
│   └── sql.py          # SQLStorage
└── backtest/           # 回测框架
    └── engine.py       # run_custom_backtest
```

## 核心特性

1. **完全独立于 Qlib 的状态管理**
   - 自定义 `Account`、`Position`、`OrderBook`
   - 不依赖 Qlib 的 Account、Position、Executor

2. **买卖逻辑分离**
   - `IBuyModel`：生成买入信号
   - `ISellModel`：生成卖出信号
   - `DualModelStrategy`：协调买入和卖出

3. **模块化设计**
   - 策略层、逻辑层、状态层、执行层、存储层完全解耦
   - 支持多种存储后端（Memory/Redis/SQL）

4. **保留 Qlib 的数据能力**
   - 使用 `D.features()` 加载市场数据
   - 使用 `D.calendar()` 获取交易日历
   - 不依赖 Qlib 的状态管理

## 使用示例

### 1. 创建买入模型

```python
from trading.strategy.buy_models import StructureExpertBuyModel
from tools.qlib.train.structure_expert import GraphDataBuilder

# 创建 GraphDataBuilder
builder = GraphDataBuilder(industry_map={})

# 创建买入模型
buy_model = StructureExpertBuyModel(
    model_path="models/structure_expert_directional.pth",
    builder=builder,
    device="cpu",
)
```

### 2. 创建卖出模型

```python
from trading.strategy.sell_models import MLExitSellModel
from nq.analysis.exit import ExitModel

# 加载退出模型
exit_model = ExitModel.load("models/exit_model.pkl")

# 创建卖出模型
sell_model = MLExitSellModel(
    exit_model=exit_model,
    threshold=0.65,
)
```

### 3. 创建策略

```python
from trading.strategy import DualModelStrategy
from trading.state import Account, PositionManager, OrderBook
from trading.logic import RiskManager, PositionAllocator
from trading.storage import MemoryStorage

# 初始化存储
storage = MemoryStorage()

# 初始化账户
account = Account(
    account_id="backtest_001",
    available_cash=1000000.0,
    initial_cash=1000000.0,
)

# 初始化状态管理
position_manager = PositionManager(account, storage)
order_book = OrderBook(storage)

# 初始化逻辑层
risk_manager = RiskManager(account, position_manager, storage)
position_allocator = PositionAllocator(target_positions=30)

# 创建策略
strategy = DualModelStrategy(
    buy_model=buy_model,
    sell_model=sell_model,
    position_manager=position_manager,
    order_book=order_book,
    risk_manager=risk_manager,
    position_allocator=position_allocator,
)
```

### 4. 运行回测

```python
from trading.backtest import run_custom_backtest

# 运行回测
results = run_custom_backtest(
    strategy=strategy,
    start_date="2025-07-01",
    end_date="2025-08-01",
    initial_cash=1000000.0,
)

# 查看结果
print(f"Final total value: {results['account'].get_total_value(position_manager):.2f}")
print(f"Final positions: {len(results['positions'])}")
print(f"Total orders: {len(results['orders'])}")
```

## 工作流程

1. **数据加载**：使用 Qlib 的 `D.features()` 加载市场数据
2. **策略生成信号**：
   - `DualModelStrategy.on_bar()` 被调用
   - 卖出模型扫描持仓，生成卖出信号
   - 买入模型扫描市场，生成买入信号
3. **风险检查**：`RiskManager` 检查订单
4. **订单提交**：订单提交到 `OrderBook`
5. **订单执行**：`Executor` 执行订单，更新 `Account` 和 `Position`
6. **状态更新**：每日收盘后更新持仓状态

## 与 Qlib 的集成

- ✅ **保留**：数据加载（`D.features()`）、交易日历（`D.calendar()`）
- ❌ **剥离**：Account、Position、Executor、Order 管理

## 扩展

### 添加新的买入模型

```python
from trading.interfaces.models import IBuyModel

class MyBuyModel(IBuyModel):
    def generate_ranks(self, date, market_data, **kwargs):
        # 实现排名生成逻辑
        return pd.DataFrame(...)
```

### 添加新的卖出模型

```python
from trading.interfaces.models import ISellModel

class MySellModel(ISellModel):
    @property
    def threshold(self):
        return 0.7
    
    def predict_exit(self, position, market_data, date, **kwargs):
        # 实现退出预测逻辑
        return 0.5  # 返回风险概率
```

### 添加新的存储后端

```python
from trading.interfaces.storage import IStorageBackend

class RedisStorage(IStorageBackend):
    def save(self, key, data):
        # 实现 Redis 保存逻辑
        pass
    
    def load(self, key):
        # 实现 Redis 加载逻辑
        pass
```

## 注意事项

1. **StructureExpertBuyModel** 已实现特征加载逻辑，使用 `trading.utils.feature_loader` 模块
2. **SQLStorage** 需要集成 Eidos 数据库（当前为占位实现）
3. **RiskManager** 的涨跌停检查需要从 Qlib 或其他数据源获取实际标志（当前为简化实现）

## 示例脚本

完整的使用示例请参考：`python/examples/backtest_trading_framework.py`

该脚本展示了如何：
- 加载 Structure Expert 和 Exit 模型
- 创建买入和卖出模型
- 初始化交易框架的所有组件
- 运行回测并查看结果

## 相关文档

- 设计文档：`docs/trading/qlib_integration_design.md`
- 交易设计：`docs/trading/trading_design.md`

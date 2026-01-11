这份设计文档旨在构建一个工业级、高可扩展的非对称量化回测与交易系统。系统采用**模块化解耦**设计，确保买卖逻辑分离，并支持多种存储后端。

---

### 1. 总体架构图 (High-Level Architecture)

系统由五个核心层次组成：
*   **策略层 (Strategy)**：双模型驱动（Buy Model / Sell Model）。
*   **逻辑层 (Logic)**：风险准入、仓位分配逻辑。
*   **状态层 (State)**：账号、订单簿、持仓生命周期管理。
*   **执行层 (Execution)**：订单撮合、状态回传。
*   **持久层 (Storage)**：抽象接口，支持 Memory/Redis/SQL。

---

### 2. 标准接口抽象 (`BaseInterface`)

通过 Python 抽象基类（ABC）定义标准接口，确保后端存储可随时切换。

```python
from abc import ABC, abstractmethod

class IStorageBackend(ABC):
    @abstractmethod
    def save(self, key: str, data: dict): pass
    @abstractmethod
    def load(self, key: str) -> dict: pass

class ITradeEntity(ABC):
    @abstractmethod
    def to_dict(self): pass
```

---

### 3. 各功能模块详细设计建议

<details>
<summary><b>3.1 账号与仓位管理 (Position & Account)</b></summary>

**核心职责**：维护资金水位、持仓成本、持仓快照、计算非对称特征（如最高价回撤）。

*   **Account**: 记录 `available_cash`, `frozen_cash`, `total_asset`。
*   **Position**: 记录 `avg_price`, `amount`, `high_price_since_entry`, `entry_time`。
*   **Snapshot**: 每日收盘后将 Position 状态序列化存入数据库，便于复盘。

```python
class PositionManager:
    def __init__(self, storage: IStorageBackend):
        self.storage = storage # 可选 Redis 或内存

    def update_position(self, symbol, current_high, current_close):
        # 更新高点回撤逻辑，这对卖出模型至关重要
        pos = self.get_position(symbol)
        pos.high_price = max(pos.high_price, current_high)
        pos.cur_ret = (current_close - pos.entry_price) / pos.entry_price
        pos.drawdown = (pos.high_price - current_close) / pos.high_price
        self.storage.save(f"pos:{symbol}", pos.to_dict())
```
</details>

<details>
<summary><b>3.2 订单簿管理 (OrderBook)</b></summary>

**核心职责**：管理订单的生命周期（Pending -> Sent -> Filled/Canceled）。

*   **状态机驱动**：支持 `MarketOrder`, `LimitOrder`。
*   **存储解耦**：Redis 后端可支持分布式策略实例，Memory 后端用于极速回测。

```python
class Order:
    order_id: str
    symbol: str
    side: str # BUY/SELL
    status: str # NEW, PARTIAL_FILLED, FILLED, CANCELED
    create_time: datetime
```
</details>

<details>
<summary><b>3.3 风险管理模块 (Risk Control)</b></summary>

**核心职责**：在指令下发执行器前执行“硬拦截”。

*   **准入检查**：停牌检查、涨跌停检查。
*   **阈值检查**：最大持仓个股限制（防止单票暴雷）、每日最大回撤熔断。
*   **审计日志**：所有被 Risk 拒绝的订单必须记录在数据库 `risk_events` 表。

```python
class RiskManager:
    def check_order(self, order, market_data):
        # 1. 检查流动性 (是否跌停/停牌)
        if market_data.is_limit_down(order.symbol):
            self.log_event(order, "REJECT_LIMIT_DOWN")
            return False
        # 2. 检查集中度
        if self.account.get_weight(order.symbol) > 0.2:
            return False
        return True
```
</details>

<details>
<summary><b>3.4 执行器与回调 (Executor)</b></summary>

**核心职责**：撮合订单，并反向通知仓位管理更新状态。

*   **回测模式**：根据下一根 Bar 的 Open/VWAP 成交。
*   **实盘模式**：对接 Broker API。
*   **反馈回路**：`on_order_filled` 触发 `PositionManager.add_position`。

```python
class Executor:
    def execute(self, order):
        # 执行撮合
        fill_info = self.match_engine(order)
        # 关键：反向更新仓位
        if fill_info.status == "FILLED":
            self.pos_manager.on_fill(fill_info)
            self.log_execution(fill_info) # 记录 Execution Log
```
</details>

<details>
<summary><b>3.5 策略分离设计 (Dual-Model Strategy)</b></summary>

**核心职责**：独立运行买入逻辑（生成 Alpha 分数）和卖出逻辑（生成风险评分）。

*   **Buy Model**: 负责寻找 `Positive Skew` 的机会。
*   **Sell Model**: 负责监控持仓特征 `(pnl, drawdown, days_held, vol_spike)` 并给出卖出概率。

```python
class DualModelStrategy:
    def on_bar(self, data):
        # 1. 扫描持仓：Sell Model 运行
        for pos in self.pos_manager.all_positions:
            risk_prob = self.sell_model.predict(pos, data)
            if risk_prob > threshold:
                self.order_bus.submit(pos.symbol, side='SELL')

        # 2. 扫描市场：Buy Model 运行 (仅当有空余仓位)
        if self.pos_manager.has_free_slot():
            top_list = self.buy_model.get_ranks(data)
            self.order_bus.submit(top_list[0], side='BUY')
```
</details>

---

### 4. 数据表结构设计 (Database Schema)

| 表名 | 关键字段 | 用途 |
| :--- | :--- | :--- |
| **`account_snapshots`** | `timestamp, total_value, cash, mdd` | 绘制资产曲线 |
| **`positions_active`** | `symbol, entry_price, high_price, amount` | 实时持仓监控 (Redis/DB) |
| **`orders_history`** | `order_id, symbol, side, price, status` | 交易流水审计 |
| **`risk_events`** | `timestamp, symbol, reason, order_id` | 风控拒绝记录 |
| **`execution_logs`** | `order_id, fill_price, commission, slippage` | 成交详情与滑点分析 |

---

### 5. 系统优势：解决“1.84 盈亏比”失效问题

1.  **数据沉淀**：通过 `positions_active` 持续记录 `high_price`。训练卖出模型时，你可以直接从数据库提取：“为什么某些订单盈利 20% 最后亏损离场？”
2.  **异步风控**：通过 `risk_events` 你可以发现，很多 ML 建议的卖出在跌停时无法转换。这能帮你修正模型，让模型学会**“预判拥挤度，提前撤退”**。
3.  **状态隔离**：即使你修改了买入模型（Alpha），你的卖出模型（Risk-Aversion）依然可以保持稳定运行，实现真正的**买卖分离训练**。

**您需要我针对其中某一个模块（例如：支持 Redis 存储的 PositionManager）给出具体的 Python 类实现逻辑吗？**
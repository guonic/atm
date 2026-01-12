# Qlib 回测系统集成设计方案

## 1. 设计目标

将 `trading_design.md` 中的模块化架构与 Qlib 回测系统集成，实现：
- **彻底剥离 Qlib 的 Account、Position 模块**：使用自定义的状态管理层
- **保留 Qlib 的数据加载能力**：利用 Qlib 的数据接口和特征工程
- **完全控制交易流程**：从排名生成 → 买入信号 → 卖出信号 → 订单执行
- **模块化解耦**：买卖逻辑分离，支持独立训练和优化

---

## 2. 架构设计

### 2.1 整体架构图

```
┌─────────────────────────────────────────────────────────────┐
│                    Qlib 数据层 (保留)                         │
│  - 数据加载 (D.features)                                      │
│  - 特征工程 (Alpha158, 自定义特征)                            │
│  - 交易日历 (D.calendar)                                      │
└─────────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                   策略层 (Strategy Layer)                     │
│  ┌──────────────────┐         ┌──────────────────┐         │
│  │  Buy Model       │         │  Sell Model       │         │
│  │  (Structure      │         │  (Exit Model)    │         │
│  │   Expert)        │         │                   │         │
│  │                  │         │                   │         │
│  │  generate_ranks()│         │  predict_exit()   │         │
│  └──────────────────┘         └──────────────────┘         │
│           │                            │                    │
│           └────────────┬───────────────┘                    │
│                        ▼                                    │
│              DualModelStrategy                              │
│              (协调买入和卖出逻辑)                              │
└─────────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                   逻辑层 (Logic Layer)                        │
│  ┌──────────────────┐         ┌──────────────────┐         │
│  │  RiskManager     │         │  PositionAllocator│         │
│  │  - 停牌检查       │         │  - 仓位分配       │         │
│  │  - 涨跌停检查     │         │  - 资金管理       │         │
│  │  - 集中度检查     │         │  - 换手控制       │         │
│  └──────────────────┘         └──────────────────┘         │
└─────────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                   状态层 (State Layer) - 自定义实现            │
│  ┌──────────────────┐         ┌──────────────────┐         │
│  │  Account         │         │  PositionManager  │         │
│  │  (自定义)         │         │  (自定义)         │         │
│  │  - available_cash│         │  - 持仓跟踪       │         │
│  │  - frozen_cash   │         │  - 成本计算       │         │
│  │  - total_asset   │         │  - 高点回撤       │         │
│  └──────────────────┘         └──────────────────┘         │
│  ┌──────────────────┐                                       │
│  │  OrderBook       │                                       │
│  │  (自定义)         │                                       │
│  │  - 订单状态机     │                                       │
│  │  - 订单历史       │                                       │
│  └──────────────────┘                                       │
└─────────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                   执行层 (Execution Layer) - 自定义实现        │
│  ┌──────────────────┐                                       │
│  │  Executor        │                                       │
│  │  (自定义)         │                                       │
│  │  - 回测撮合       │                                       │
│  │  - 实盘对接       │                                       │
│  │  - 滑点模拟       │                                       │
│  └──────────────────┘                                       │
└─────────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                   持久层 (Storage Layer) - 抽象接口            │
│  ┌──────────────────┐         ┌──────────────────┐         │
│  │  MemoryStorage   │         │  RedisStorage    │         │
│  │  (回测)           │         │  (实盘/分布式)    │         │
│  └──────────────────┘         └──────────────────┘         │
│  ┌──────────────────┐                                       │
│  │  SQLStorage      │                                       │
│  │  (Eidos)         │                                       │
│  └──────────────────┘                                       │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 关键设计原则

1. **Qlib 仅用于数据加载**：不依赖 Qlib 的 Account、Position、Executor
2. **完全自定义状态管理**：实现自己的 Account、Position、OrderBook
3. **策略接口标准化**：定义清晰的 Strategy 接口，支持双模型驱动
4. **存储抽象化**：支持 Memory/Redis/SQL 多种后端

---

## 3. 核心模块设计

### 3.1 策略层 (Strategy Layer)

#### 3.1.1 双模型策略接口

```python
from abc import ABC, abstractmethod
from typing import Dict, List, Optional
import pandas as pd

class IBuyModel(ABC):
    """买入模型接口"""
    
    @abstractmethod
    def generate_ranks(
        self, 
        date: pd.Timestamp, 
        market_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        生成股票排名。
        
        Args:
            date: 当前日期
            market_data: 市场数据 (从 Qlib 加载)
        
        Returns:
            DataFrame with columns: ['symbol', 'score', 'rank']
        """
        pass


class ISellModel(ABC):
    """卖出模型接口"""
    
    @abstractmethod
    def predict_exit(
        self,
        position: 'Position',
        market_data: pd.DataFrame,
        date: pd.Timestamp
    ) -> float:
        """
        预测是否应该退出持仓。
        
        Args:
            position: 持仓信息
            market_data: 市场数据
            date: 当前日期
        
        Returns:
            风险概率 (0-1)，> threshold 则卖出
        """
        pass


class DualModelStrategy:
    """
    双模型策略：协调买入和卖出逻辑。
    
    工作流程：
    1. 每日开盘前：Sell Model 扫描所有持仓，生成卖出信号
    2. 每日开盘后：Buy Model 生成排名，生成买入信号
    3. 通过 OrderBus 提交订单
    """
    
    def __init__(
        self,
        buy_model: IBuyModel,
        sell_model: ISellModel,
        position_manager: 'PositionManager',
        order_bus: 'OrderBus',
        risk_manager: 'RiskManager',
        position_allocator: 'PositionAllocator',
    ):
        self.buy_model = buy_model
        self.sell_model = sell_model
        self.position_manager = position_manager
        self.order_bus = order_bus
        self.risk_manager = risk_manager
        self.position_allocator = position_allocator
    
    def on_bar(self, date: pd.Timestamp, market_data: pd.DataFrame):
        """
        每日交易逻辑（完全独立于 Qlib 的回调）。
        
        Args:
            date: 当前交易日
            market_data: 市场数据（从 Qlib 加载，但不依赖 Qlib 的状态）
        """
        # Step 1: 扫描持仓，Sell Model 运行
        sell_orders = []
        for symbol, position in self.position_manager.all_positions.items():
            risk_prob = self.sell_model.predict_exit(
                position=position,
                market_data=market_data,
                date=date
            )
            
            if risk_prob > self.sell_model.threshold:
                # 生成卖出订单
                order = Order(
                    symbol=symbol,
                    side=OrderSide.SELL,
                    amount=position.amount,
                    order_type=OrderType.MARKET,
                    date=date,
                )
                
                # 风险检查
                if self.risk_manager.check_order(order, market_data):
                    sell_orders.append(order)
                else:
                    self.risk_manager.log_rejection(
                        order=order,
                        reason="RISK_CHECK_FAILED"
                    )
        
        # Step 2: 扫描市场，Buy Model 运行（仅当有空余仓位）
        buy_orders = []
        if self.position_manager.has_free_slot():
            # 生成排名
            ranks = self.buy_model.generate_ranks(
                date=date,
                market_data=market_data
            )
            
            # 选择 top K（排除已有持仓）
            current_symbols = set(self.position_manager.all_positions.keys())
            available_ranks = ranks[~ranks['symbol'].isin(current_symbols)]
            top_k = available_ranks.head(self.position_allocator.target_positions)
            
            # 生成买入订单
            for _, row in top_k.iterrows():
                symbol = row['symbol']
                target_value = self.position_allocator.calculate_position_size(
                    symbol=symbol,
                    account=self.position_manager.account,
                    market_data=market_data.loc[market_data.index == date]
                )
                
                if target_value > 0:
                    order = Order(
                        symbol=symbol,
                        side=OrderSide.BUY,
                        amount=0,  # 由 Executor 根据 target_value 计算
                        target_value=target_value,
                        order_type=OrderType.MARKET,
                        date=date,
                    )
                    
                    # 风险检查
                    if self.risk_manager.check_order(order, market_data):
                        buy_orders.append(order)
                    else:
                        self.risk_manager.log_rejection(
                            order=order,
                            reason="RISK_CHECK_FAILED"
                        )
        
        # Step 3: 提交订单到 OrderBus
        for order in sell_orders + buy_orders:
            self.order_bus.submit(order)
```

#### 3.1.2 Structure Expert 买入模型实现

```python
class StructureExpertBuyModel(IBuyModel):
    """Structure Expert 模型作为买入模型"""
    
    def __init__(
        self,
        model_path: str,
        builder: GraphDataBuilder,
        device: str = "cuda",
    ):
        self.model = load_model(model_path, device)
        self.builder = builder
        self.device = device
    
    def generate_ranks(
        self,
        date: pd.Timestamp,
        market_data: pd.DataFrame
    ) -> pd.DataFrame:
        """生成股票排名"""
        # 使用 Qlib 加载数据（但不依赖 Qlib 的状态）
        graph_data = self.builder.get_daily_graph(
            date=date,
            # ... 其他参数
        )
        
        # 模型预测
        with torch.no_grad():
            predictions = self.model(graph_data)
        
        # 转换为排名 DataFrame
        ranks = pd.DataFrame({
            'symbol': graph_data.symbols,
            'score': predictions.cpu().numpy(),
        })
        ranks = ranks.sort_values('score', ascending=False)
        ranks['rank'] = range(1, len(ranks) + 1)
        
        return ranks
```

#### 3.1.3 ML Exit 卖出模型实现

```python
class MLExitSellModel(ISellModel):
    """ML Exit 模型作为卖出模型"""
    
    def __init__(
        self,
        exit_model: ExitModel,
        threshold: float = 0.65,
    ):
        self.exit_model = exit_model
        self.threshold = threshold
    
    def predict_exit(
        self,
        position: 'Position',
        market_data: pd.DataFrame,
        date: pd.Timestamp
    ) -> float:
        """预测退出概率"""
        # 获取历史数据（从 Qlib 加载，但不依赖 Qlib 的状态）
        hist_data = D.features(
            instruments=[position.symbol],
            fields=["$close", "$high", "$low", "$volume"],
            start_time=(date - pd.Timedelta(days=15)).strftime("%Y-%m-%d"),
            end_time=date.strftime("%Y-%m-%d"),
        )
        
        # 构建特征
        daily_df = pd.DataFrame({
            'close': hist_data[position.symbol]['$close'],
            'high': hist_data[position.symbol]['$high'],
            'low': hist_data[position.symbol]['$low'],
            'volume': hist_data[position.symbol]['$volume'],
        })
        
        # 预测
        proba = self.exit_model.predict_proba(
            daily_df=daily_df,
            entry_price=position.entry_price,
            highest_price_since_entry=position.high_price_since_entry,
            days_held=(date - position.entry_date).days,
        )
        
        return proba[-1] if len(proba) > 0 else 0.0
```

---

### 3.2 状态层 (State Layer) - 自定义实现

#### 3.2.1 Account (账户管理)

```python
from dataclasses import dataclass
from typing import Dict
from datetime import datetime

@dataclass
class Account:
    """账户状态（完全独立于 Qlib）"""
    
    account_id: str
    available_cash: float  # 可用资金
    frozen_cash: float     # 冻结资金（挂单）
    total_asset: float     # 总资产
    initial_cash: float    # 初始资金
    
    def get_total_value(self, position_manager: 'PositionManager') -> float:
        """计算总资产（现金 + 持仓市值）"""
        holdings_value = sum(
            pos.market_value 
            for pos in position_manager.all_positions.values()
        )
        return self.available_cash + self.frozen_cash + holdings_value
    
    def can_afford(self, required_cash: float) -> bool:
        """检查是否有足够资金"""
        return self.available_cash >= required_cash
    
    def freeze_cash(self, amount: float):
        """冻结资金"""
        if self.available_cash >= amount:
            self.available_cash -= amount
            self.frozen_cash += amount
        else:
            raise ValueError(f"Insufficient cash: {self.available_cash} < {amount}")
    
    def unfreeze_cash(self, amount: float):
        """解冻资金"""
        if self.frozen_cash >= amount:
            self.frozen_cash -= amount
            self.available_cash += amount
        else:
            raise ValueError(f"Insufficient frozen cash: {self.frozen_cash} < {amount}")
    
    def deduct_cash(self, amount: float):
        """扣除资金（订单成交后）"""
        if self.frozen_cash >= amount:
            self.frozen_cash -= amount
        else:
            raise ValueError(f"Insufficient frozen cash: {self.frozen_cash} < {amount}")
    
    def add_cash(self, amount: float):
        """增加资金（卖出成交后）"""
        self.available_cash += amount
```

#### 3.2.2 Position (持仓管理)

```python
@dataclass
class Position:
    """持仓信息（完全独立于 Qlib）"""
    
    symbol: str
    entry_date: pd.Timestamp
    entry_price: float
    amount: float
    high_price_since_entry: float  # 自买入以来的最高价
    high_date: pd.Timestamp         # 最高价日期
    
    @property
    def avg_price(self) -> float:
        """平均成本价"""
        return self.entry_price
    
    def update_high_price(self, current_high: float, date: pd.Timestamp):
        """更新最高价"""
        if current_high > self.high_price_since_entry:
            self.high_price_since_entry = current_high
            self.high_date = date
    
    def calculate_return(self, current_price: float) -> float:
        """计算当前收益率"""
        return (current_price - self.entry_price) / self.entry_price
    
    def calculate_drawdown(self, current_price: float) -> float:
        """计算回撤（从最高价）"""
        if self.high_price_since_entry > 0:
            return (self.high_price_since_entry - current_price) / self.high_price_since_entry
        return 0.0
    
    def calculate_market_value(self, current_price: float) -> float:
        """计算市值"""
        return self.amount * current_price


class PositionManager:
    """持仓管理器（完全独立于 Qlib）"""
    
    def __init__(self, account: Account, storage: IStorageBackend):
        self.account = account
        self.storage = storage
        self.positions: Dict[str, Position] = {}
    
    @property
    def all_positions(self) -> Dict[str, Position]:
        """获取所有持仓"""
        return self.positions
    
    def add_position(
        self,
        symbol: str,
        entry_date: pd.Timestamp,
        entry_price: float,
        amount: float,
    ):
        """添加持仓（买入成交后）"""
        if symbol in self.positions:
            # 加仓：更新平均成本
            pos = self.positions[symbol]
            old_value = pos.entry_price * pos.amount
            new_value = entry_price * amount
            pos.amount += amount
            pos.entry_price = (old_value + new_value) / pos.amount
        else:
            # 新建持仓
            self.positions[symbol] = Position(
                symbol=symbol,
                entry_date=entry_date,
                entry_price=entry_price,
                amount=amount,
                high_price_since_entry=entry_price,
                high_date=entry_date,
            )
        
        # 持久化
        self.storage.save(f"pos:{symbol}", self.positions[symbol].to_dict())
    
    def remove_position(self, symbol: str):
        """移除持仓（卖出成交后）"""
        if symbol in self.positions:
            del self.positions[symbol]
            self.storage.delete(f"pos:{symbol}")
    
    def reduce_position(self, symbol: str, amount: float):
        """减少持仓（部分卖出）"""
        if symbol in self.positions:
            pos = self.positions[symbol]
            pos.amount -= amount
            if pos.amount <= 0:
                self.remove_position(symbol)
            else:
                self.storage.save(f"pos:{symbol}", pos.to_dict())
    
    def update_positions(self, date: pd.Timestamp, market_data: pd.DataFrame):
        """更新持仓状态（每日收盘后）"""
        for symbol, position in self.positions.items():
            if symbol in market_data.index:
                current_high = market_data.loc[symbol, '$high']
                current_close = market_data.loc[symbol, '$close']
                
                # 更新最高价
                position.update_high_price(current_high, date)
                
                # 持久化快照
                snapshot = {
                    'date': date,
                    'symbol': symbol,
                    'entry_price': position.entry_price,
                    'current_price': current_close,
                    'amount': position.amount,
                    'high_price_since_entry': position.high_price_since_entry,
                    'current_return': position.calculate_return(current_close),
                    'drawdown': position.calculate_drawdown(current_close),
                    'market_value': position.calculate_market_value(current_close),
                }
                self.storage.save(f"snapshot:{date}:{symbol}", snapshot)
    
    def has_free_slot(self, max_positions: int = 30) -> bool:
        """检查是否有空余仓位"""
        return len(self.positions) < max_positions
```

#### 3.2.3 OrderBook (订单簿管理)

```python
from enum import Enum
from typing import Optional
import uuid

class OrderStatus(Enum):
    NEW = "NEW"
    PENDING = "PENDING"
    PARTIAL_FILLED = "PARTIAL_FILLED"
    FILLED = "FILLED"
    CANCELED = "CANCELED"
    REJECTED = "REJECTED"


class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"


@dataclass
class Order:
    """订单（完全独立于 Qlib）"""
    
    order_id: str
    symbol: str
    side: OrderSide
    amount: float
    order_type: OrderType
    date: pd.Timestamp
    status: OrderStatus = OrderStatus.NEW
    target_value: Optional[float] = None  # 目标金额（买入时使用）
    limit_price: Optional[float] = None   # 限价（限价单使用）
    filled_amount: float = 0.0
    filled_price: Optional[float] = None
    create_time: datetime = None
    
    def __post_init__(self):
        if self.order_id is None:
            self.order_id = str(uuid.uuid4())
        if self.create_time is None:
            self.create_time = datetime.now()


class OrderBook:
    """订单簿（完全独立于 Qlib）"""
    
    def __init__(self, storage: IStorageBackend):
        self.storage = storage
        self.orders: Dict[str, Order] = {}
    
    def submit(self, order: Order):
        """提交订单"""
        order.status = OrderStatus.PENDING
        self.orders[order.order_id] = order
        self.storage.save(f"order:{order.order_id}", order.to_dict())
    
    def update_order(
        self,
        order_id: str,
        status: OrderStatus,
        filled_amount: Optional[float] = None,
        filled_price: Optional[float] = None,
    ):
        """更新订单状态"""
        if order_id in self.orders:
            order = self.orders[order_id]
            order.status = status
            if filled_amount is not None:
                order.filled_amount = filled_amount
            if filled_price is not None:
                order.filled_price = filled_price
            self.storage.save(f"order:{order_id}", order.to_dict())
    
    def get_pending_orders(self) -> List[Order]:
        """获取待处理订单"""
        return [
            order for order in self.orders.values()
            if order.status == OrderStatus.PENDING
        ]
```

---

### 3.3 执行层 (Execution Layer) - 自定义实现

#### 3.3.1 Executor (执行器)

```python
class Executor:
    """订单执行器（完全独立于 Qlib）"""
    
    def __init__(
        self,
        position_manager: PositionManager,
        order_book: OrderBook,
        commission_rate: float = 0.0015,
        slippage_rate: float = 0.0,
    ):
        self.position_manager = position_manager
        self.order_book = order_book
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate
    
    def execute_order(
        self,
        order: Order,
        market_data: pd.DataFrame,
        date: pd.Timestamp
    ) -> Optional['FillInfo']:
        """
        执行订单（回测模式）。
        
        Args:
            order: 订单
            market_data: 市场数据（从 Qlib 加载）
            date: 当前日期
        
        Returns:
            FillInfo 如果成交，否则 None
        """
        if order.symbol not in market_data.index:
            self.order_book.update_order(order.order_id, OrderStatus.REJECTED)
            return None
        
        symbol_data = market_data.loc[order.symbol]
        
        # 确定成交价格（回测模式：使用下一根 Bar 的 Open）
        if order.order_type == OrderType.MARKET:
            # 买入：使用开盘价（考虑滑点）
            if order.side == OrderSide.BUY:
                fill_price = symbol_data['$open'] * (1 + self.slippage_rate)
            else:
                fill_price = symbol_data['$open'] * (1 - self.slippage_rate)
        else:
            # 限价单
            fill_price = order.limit_price
        
        # 计算成交数量
        if order.side == OrderSide.BUY:
            # 买入：根据目标金额或可用资金计算
            if order.target_value:
                available_cash = self.position_manager.account.available_cash
                target_cash = min(order.target_value, available_cash)
                fill_amount = int(target_cash / fill_price / 100) * 100  # 取整到100股
            else:
                fill_amount = order.amount
            
            # 检查资金
            required_cash = fill_amount * fill_price * (1 + self.commission_rate)
            if not self.position_manager.account.can_afford(required_cash):
                fill_amount = int(
                    self.position_manager.account.available_cash 
                    / fill_price / (1 + self.commission_rate) / 100
                ) * 100
                if fill_amount <= 0:
                    self.order_book.update_order(order.order_id, OrderStatus.REJECTED)
                    return None
        else:
            # 卖出：检查持仓
            if order.symbol not in self.position_manager.positions:
                self.order_book.update_order(order.order_id, OrderStatus.REJECTED)
                return None
            
            pos = self.position_manager.positions[order.symbol]
            fill_amount = min(order.amount, pos.amount)
            if fill_amount <= 0:
                self.order_book.update_order(order.order_id, OrderStatus.REJECTED)
                return None
        
        # 计算手续费
        commission = fill_amount * fill_price * self.commission_rate
        
        # 创建成交信息
        fill_info = FillInfo(
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            amount=fill_amount,
            price=fill_price,
            commission=commission,
            date=date,
        )
        
        # 更新订单状态
        if fill_amount == order.amount:
            self.order_book.update_order(
                order.order_id,
                OrderStatus.FILLED,
                filled_amount=fill_amount,
                filled_price=fill_price,
            )
        else:
            self.order_book.update_order(
                order.order_id,
                OrderStatus.PARTIAL_FILLED,
                filled_amount=fill_amount,
                filled_price=fill_price,
            )
        
        # 反向更新仓位和账户
        self._update_position_and_account(fill_info)
        
        return fill_info
    
    def _update_position_and_account(self, fill_info: FillInfo):
        """更新仓位和账户（订单成交后）"""
        if fill_info.side == OrderSide.BUY:
            # 买入成交
            self.position_manager.account.deduct_cash(
                fill_info.amount * fill_info.price + fill_info.commission
            )
            self.position_manager.add_position(
                symbol=fill_info.symbol,
                entry_date=fill_info.date,
                entry_price=fill_info.price,
                amount=fill_info.amount,
            )
        else:
            # 卖出成交
            proceeds = fill_info.amount * fill_info.price - fill_info.commission
            self.position_manager.account.add_cash(proceeds)
            self.position_manager.reduce_position(
                symbol=fill_info.symbol,
                amount=fill_info.amount,
            )
```

---

### 3.4 逻辑层 (Logic Layer)

#### 3.4.1 RiskManager (风险管理)

```python
class RiskManager:
    """风险管理（完全独立于 Qlib）"""
    
    def __init__(
        self,
        account: Account,
        position_manager: PositionManager,
        storage: IStorageBackend,
    ):
        self.account = account
        self.position_manager = position_manager
        self.storage = storage
    
    def check_order(
        self,
        order: Order,
        market_data: pd.DataFrame
    ) -> bool:
        """检查订单是否通过风控"""
        # 1. 检查流动性（停牌、涨跌停）
        if order.symbol not in market_data.index:
            self.log_event(order, "REJECT_NO_DATA")
            return False
        
        symbol_data = market_data.loc[order.symbol]
        
        # 检查跌停（卖出时）
        if order.side == OrderSide.SELL:
            if self._is_limit_down(symbol_data):
                self.log_event(order, "REJECT_LIMIT_DOWN")
                return False
        
        # 检查涨停（买入时）
        if order.side == OrderSide.BUY:
            if self._is_limit_up(symbol_data):
                self.log_event(order, "REJECT_LIMIT_UP")
                return False
        
        # 2. 检查集中度
        if order.side == OrderSide.BUY:
            current_weight = self.position_manager.get_weight(order.symbol)
            if current_weight > 0.2:  # 单票不超过20%
                self.log_event(order, "REJECT_CONCENTRATION")
                return False
        
        return True
    
    def _is_limit_down(self, symbol_data: pd.Series) -> bool:
        """检查是否跌停"""
        # 简化实现：如果 close == low 且 close < prev_close * 0.9，可能是跌停
        # 实际应该从 Qlib 获取涨跌停信息
        return False
    
    def _is_limit_up(self, symbol_data: pd.Series) -> bool:
        """检查是否涨停"""
        return False
    
    def log_event(self, order: Order, reason: str):
        """记录风控事件"""
        event = {
            'timestamp': datetime.now(),
            'order_id': order.order_id,
            'symbol': order.symbol,
            'reason': reason,
        }
        self.storage.save(f"risk_event:{order.order_id}", event)
```

#### 3.4.2 PositionAllocator (仓位分配)

```python
class PositionAllocator:
    """仓位分配器（完全独立于 Qlib）"""
    
    def __init__(
        self,
        target_positions: int = 30,
        equal_weight: bool = True,
    ):
        self.target_positions = target_positions
        self.equal_weight = equal_weight
    
    def calculate_position_size(
        self,
        symbol: str,
        account: Account,
        market_data: pd.DataFrame,
    ) -> float:
        """计算目标仓位金额"""
        if self.equal_weight:
            # 等权重分配
            total_value = account.get_total_value()
            return total_value / self.target_positions
        else:
            # 可以扩展为其他分配策略（如风险平价）
            total_value = account.get_total_value()
            return total_value / self.target_positions
```

---

### 3.5 持久层 (Storage Layer)

#### 3.5.1 存储接口抽象

```python
from abc import ABC, abstractmethod

class IStorageBackend(ABC):
    """存储后端接口"""
    
    @abstractmethod
    def save(self, key: str, data: dict):
        """保存数据"""
        pass
    
    @abstractmethod
    def load(self, key: str) -> Optional[dict]:
        """加载数据"""
        pass
    
    @abstractmethod
    def delete(self, key: str):
        """删除数据"""
        pass


class MemoryStorage(IStorageBackend):
    """内存存储（回测用）"""
    
    def __init__(self):
        self.data: Dict[str, dict] = {}
    
    def save(self, key: str, data: dict):
        self.data[key] = data
    
    def load(self, key: str) -> Optional[dict]:
        return self.data.get(key)
    
    def delete(self, key: str):
        if key in self.data:
            del self.data[key]


class SQLStorage(IStorageBackend):
    """SQL 存储（Eidos 数据库）"""
    
    def __init__(self, db_config):
        self.db_config = db_config
        # 使用 EidosRepo 实现
    
    def save(self, key: str, data: dict):
        # 保存到 Eidos 数据库
        pass
    
    def load(self, key: str) -> Optional[dict]:
        # 从 Eidos 数据库加载
        pass
    
    def delete(self, key: str):
        # 从 Eidos 数据库删除
        pass
```

---

## 4. 与 Qlib 集成的工作流程

### 4.1 数据加载（使用 Qlib）

```python
from qlib.data import D

def load_market_data(
    instruments: List[str],
    start_date: str,
    end_date: str,
    fields: List[str] = ["$close", "$open", "$high", "$low", "$volume"]
) -> pd.DataFrame:
    """
    使用 Qlib 加载市场数据（但不依赖 Qlib 的状态管理）。
    
    Returns:
        DataFrame with MultiIndex (instrument, datetime)
    """
    return D.features(
        instruments=instruments,
        fields=fields,
        start_time=start_date,
        end_time=end_date,
    )
```

### 4.2 回测主循环（完全自定义）

```python
def run_custom_backtest(
    strategy: DualModelStrategy,
    start_date: str,
    end_date: str,
    initial_cash: float = 1000000.0,
    instruments: Optional[List[str]] = None,
):
    """
    运行回测（完全独立于 Qlib 的回测框架）。
    
    工作流程：
    1. 使用 Qlib 加载数据
    2. 使用自定义的状态管理（Account, Position）
    3. 使用自定义的执行器（Executor）
    4. 策略生成订单，执行器撮合
    """
    # 初始化
    storage = MemoryStorage()
    account = Account(
        account_id="backtest_001",
        available_cash=initial_cash,
        frozen_cash=0.0,
        total_asset=initial_cash,
        initial_cash=initial_cash,
    )
    position_manager = PositionManager(account, storage)
    order_book = OrderBook(storage)
    executor = Executor(position_manager, order_book)
    
    # 加载交易日历（使用 Qlib）
    calendar = D.calendar(start_time=start_date, end_time=end_date)
    
    # 加载市场数据（使用 Qlib）
    if instruments is None:
        instruments = D.instruments()
    market_data = load_market_data(instruments, start_date, end_date)
    
    # 回测主循环
    for date in calendar:
        date_ts = pd.Timestamp(date)
        date_str = date_ts.strftime("%Y-%m-%d")
        
        # 获取当日市场数据
        daily_data = market_data.loc[market_data.index.get_level_values(1) == date_ts]
        
        # 策略生成订单
        strategy.on_bar(date_ts, daily_data)
        
        # 执行订单
        pending_orders = order_book.get_pending_orders()
        for order in pending_orders:
            fill_info = executor.execute_order(order, daily_data, date_ts)
            if fill_info:
                logger.info(f"Order filled: {fill_info}")
        
        # 更新持仓状态（收盘后）
        position_manager.update_positions(date_ts, daily_data)
        
        # 记录账户快照
        snapshot = {
            'date': date_str,
            'total_value': account.get_total_value(position_manager),
            'cash': account.available_cash,
            'holdings_value': sum(
                pos.calculate_market_value(daily_data.loc[pos.symbol, '$close'])
                for pos in position_manager.all_positions.values()
            ),
        }
        storage.save(f"snapshot:{date_str}", snapshot)
    
    # 返回回测结果
    return {
        'account': account,
        'positions': position_manager.all_positions,
        'orders': list(order_book.orders.values()),
        'snapshots': [storage.load(f"snapshot:{d}") for d in calendar],
    }
```

---

## 5. 集成优势

### 5.1 完全控制交易流程

- **买入逻辑**：由 Buy Model（Structure Expert）独立生成排名和买入信号
- **卖出逻辑**：由 Sell Model（ML Exit）独立判断退出时机
- **订单执行**：完全自定义的 Executor，不依赖 Qlib 的执行器
- **状态管理**：完全自定义的 Account、Position，支持任意扩展

### 5.2 模块化解耦

- **策略层**：Buy Model 和 Sell Model 可以独立训练和优化
- **状态层**：Account、Position 可以支持多种存储后端（Memory/Redis/SQL）
- **执行层**：Executor 可以支持回测和实盘两种模式

### 5.3 数据沉淀

- **持仓快照**：每日记录持仓状态，便于训练卖出模型
- **订单历史**：完整记录所有订单，便于分析
- **风控事件**：记录所有被拒绝的订单，便于优化模型

---

## 6. 实现优先级

### Phase 1: 核心框架
1. ✅ 定义接口抽象（IBuyModel, ISellModel, IStorageBackend）
2. ✅ 实现状态层（Account, Position, OrderBook）
3. ✅ 实现执行层（Executor）
4. ✅ 实现逻辑层（RiskManager, PositionAllocator）

### Phase 2: 策略集成
1. ✅ 实现 StructureExpertBuyModel
2. ✅ 实现 MLExitSellModel
3. ✅ 实现 DualModelStrategy

### Phase 3: 回测框架
1. ✅ 实现回测主循环（run_custom_backtest）
2. ✅ 集成 Qlib 数据加载
3. ✅ 实现结果输出和 Eidos 集成

### Phase 4: 存储后端
1. ✅ 实现 MemoryStorage（回测用）
2. ✅ 实现 SQLStorage（Eidos 集成）
3. ⏳ 实现 RedisStorage（实盘/分布式用）

---

## 7. 关键注意事项

### 7.1 Qlib 仅用于数据加载

- ✅ 使用 `D.features()` 加载市场数据
- ✅ 使用 `D.calendar()` 获取交易日历
- ❌ **不使用** Qlib 的 `Account`、`Position`、`Executor`

### 7.2 状态管理完全自定义

- ✅ 实现自己的 `Account` 类
- ✅ 实现自己的 `PositionManager` 类
- ✅ 实现自己的 `OrderBook` 类
- ✅ 实现自己的 `Executor` 类

### 7.3 策略接口标准化

- ✅ 定义 `IBuyModel` 和 `ISellModel` 接口
- ✅ 实现 `DualModelStrategy` 协调买入和卖出
- ✅ 支持任意模型实现（Structure Expert, ML Exit, 等）

---

## 8. 总结

这个设计方案实现了：

1. **彻底剥离 Qlib 的状态管理**：使用完全自定义的 Account、Position、OrderBook
2. **保留 Qlib 的数据能力**：继续使用 Qlib 的数据加载和特征工程
3. **完全控制交易流程**：从排名生成 → 买入信号 → 卖出信号 → 订单执行
4. **模块化解耦**：买卖逻辑分离，支持独立训练和优化
5. **存储抽象化**：支持 Memory/Redis/SQL 多种后端

这样就可以实现真正的**买卖分离训练**，同时充分利用 Qlib 的数据能力。

# 为什么会出现 amount=0 的交易

## 问题描述

在回测过程中，有时会出现 `amount=0` 的交易记录，这会导致数据库约束违反错误：
```
psycopg2.errors.CheckViolation: new row for relation "bt_trades" violates check constraint "bt_trades_amount_check"
DETAIL:  Failing row contains (..., amount=0, ...)
```

数据库约束要求 `amount > 0`，但某些情况下 Qlib 的订单执行结果可能包含 `amount=0` 的订单。

## 原因分析

### 1. **订单被拒绝或取消**

Qlib 的 executor 在处理订单时，如果订单因为以下原因被拒绝或取消，仍然可能返回一个 Order 对象，但 `order.amount` 为 0：

- **资金不足**：买入订单时，可用资金不足以购买最小交易单位（100股）
- **持仓不足**：卖出订单时，持仓数量为 0 或不足
- **价格限制**：限价单价格超出涨跌停板范围
- **订单被取消**：策略逻辑或风控规则取消了订单
- **数据缺失**：目标日期的价格数据缺失，导致无法执行

### 2. **部分成交后计算错误**

在某些边缘情况下，订单可能部分成交，但在计算最终 `amount` 时由于：
- 浮点数精度问题
- 四舍五入导致结果为 0
- 计算逻辑错误

### 3. **Qlib 执行器的行为**

Qlib 的执行器（如 `SimulatorExecutor`）在以下情况下可能返回 `amount=0` 的订单：

```python
# Qlib 执行器可能的行为
# 1. 订单被拒绝但仍返回 Order 对象
order = Order(...)
order.amount = 0  # 因为资金不足等原因

# 2. 订单部分成交但剩余量被计算为 0
order.amount = original_amount - filled_amount
# 如果 filled_amount >= original_amount（由于精度问题），amount 可能为 0
```

### 4. **策略逻辑问题**

策略在生成订单时可能计算出 `amount=0`：

```python
# 示例：计算出的交易量过小，被四舍五入为 0
cash_to_use = self.broker.getcash() * 0.01  # 1% 的资金
size = int(cash_to_use / current_price)  # 如果价格很高，可能为 0
if size > 0:
    self.buy(size=size)
else:
    # 如果 size=0，但策略仍然创建了订单，就会导致 amount=0
    pass
```

### 5. **数据转换过程中的问题**

在从 Qlib Order 转换为交易记录时：

```python
# python/examples/backtest_structure_expert.py
order_info = {
    "amount": order.amount,  # 直接使用 Qlib 的 order.amount
    # 如果 order.amount 为 0 或 None，就会导致问题
}
```

## 解决方案

### 1. **在订单提取时过滤**（已实现）

在 `extract_trades_from_backtest_results` 函数中过滤掉 `amount <= 0` 的订单：

```python
# python/nq/analysis/backtest/eidos_structure_expert.py
amount = int(order_info.get("amount", 0))
if amount <= 0:
    logger.warning(f"Skipping order with invalid amount: {amount}")
    continue
```

### 2. **在数据库插入时过滤**（已实现）

在 `batch_insert` 方法中添加双重保险：

```python
# python/nq/repo/eidos_repo.py
filtered_trades = []
for trade in trades_data:
    amount = trade.get("amount", 0)
    if amount is None or amount <= 0:
        skipped_count += 1
        continue
    filtered_trades.append(trade)
```

### 3. **在策略层面避免生成 amount=0 的订单**

在策略代码中，确保在创建订单前检查交易量：

```python
def next(self):
    # 计算交易量
    cash_to_use = self.broker.getcash() * self.p.position_pct
    size = int(cash_to_use / current_price)
    
    # 确保交易量 > 0 才创建订单
    if size > 0:
        self.buy(size=size)
    else:
        logger.debug(f"Skipping order: calculated size={size} is too small")
```

### 4. **增强日志记录**

在 `post_exe_step` 中记录所有订单信息，包括 `amount=0` 的情况：

```python
# python/examples/backtest_structure_expert.py
logger.info(
    f"Raw order from Qlib: stock_id={order.stock_id}, "
    f"amount={order.amount}, ..."
)

if order.amount == 0:
    logger.warning(
        f"⚠️  Order with amount=0 detected: {order.stock_id}, "
        f"direction={order.direction}, price={trade_price}"
    )
```

## 预防措施

### 1. **最小交易量检查**

在策略中设置最小交易量阈值：

```python
MIN_TRADE_AMOUNT = 100  # 最小交易量（股）

def calculate_position_size(self, price: float) -> int:
    cash = self.broker.getcash()
    size = int(cash * self.p.position_pct / price)
    return max(size, MIN_TRADE_AMOUNT) if size > 0 else 0
```

### 2. **资金充足性检查**

在创建买入订单前检查资金是否充足：

```python
def can_afford(self, price: float, size: int) -> bool:
    required_cash = price * size * (1 + self.commission_rate)
    return self.broker.getcash() >= required_cash
```

### 3. **持仓充足性检查**

在创建卖出订单前检查持仓是否充足：

```python
def can_sell(self, size: int) -> bool:
    return self.position.size >= size
```

## 调试建议

如果遇到 `amount=0` 的交易，可以通过以下方式调试：

1. **查看日志**：检查 `post_exe_step` 中的日志，查看哪些订单的 `amount=0`
2. **检查策略逻辑**：确认策略在生成订单时是否正确计算了交易量
3. **检查资金状态**：确认是否有足够的资金或持仓来执行订单
4. **检查数据质量**：确认价格数据是否完整，是否有缺失值

## 相关代码位置

- 订单提取：`python/nq/analysis/backtest/eidos_structure_expert.py::extract_trades_from_backtest_results`
- 订单捕获：`python/examples/backtest_structure_expert.py::BaseCustomStrategy::post_exe_step`
- 数据库插入：`python/nq/repo/eidos_repo.py::EidosTradesRepo::batch_insert`
- 数据库约束：`docs/design/eidos_improvements.md` (line 104: `amount INTEGER NOT NULL CHECK (amount > 0)`)

## 总结

`amount=0` 的交易通常是由于：
1. 订单被拒绝或取消（资金不足、持仓不足等）
2. 计算精度问题导致四舍五入为 0
3. 策略逻辑问题生成无效订单

通过在多个层面添加过滤和检查，可以确保不会将 `amount=0` 的交易插入数据库。如果仍然出现，应该检查策略逻辑和数据质量。

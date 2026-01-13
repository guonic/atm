# Trading Framework

Trading system with dual-model strategy (Buy Model / Sell Model).

This package implements a modular trading system that:
- Separates buy and sell logic (independent model training)
- Uses custom state management (Account, Position, OrderBook)
- Supports multiple storage backends (Memory/Redis/SQL)
- Integrates with Qlib for data loading only (not state management)

## Architecture

- **Strategy Layer**: Buy Model / Sell Model
- **Logic Layer**: Risk Management / Position Allocation
- **State Layer**: Account / Position / OrderBook
- **Execution Layer**: Executor
- **Storage Layer**: Storage Backend (Memory/Redis/SQL)

## Usage

```python
from nq.trading.strategies import DualModelStrategy
from nq.trading.state import Account, PositionManager, OrderBook
from nq.trading.backtest import run_custom_backtest

# Create strategy
strategy = DualModelStrategy(...)

# Run backtest
results = run_custom_backtest(
    strategy=strategy,
    start_date="2025-01-01",
    end_date="2025-12-31",
    initial_cash=1000000.0,
)
```

## Example

See `python/examples/backtest_trading_framework.py` for a complete example.

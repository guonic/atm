"""
Trading system with dual-model strategy (Buy Model / Sell Model).

This package implements a modular trading system that:
- Separates buy and sell logic (independent model training)
- Uses custom state management (Account, Position, OrderBook)
- Supports multiple storage backends (Memory/Redis/SQL)
- Integrates with Qlib for data loading only (not state management)

Architecture:
- Strategy Layer: Buy Model / Sell Model
- Logic Layer: Risk Management / Position Allocation
- State Layer: Account / Position / OrderBook
- Execution Layer: Executor
- Storage Layer: Storage Backend (Memory/Redis/SQL)
"""

__version__ = "0.1.0"

# Import main components for easy access
from .strategies import AsymmetricStrategy
# Keep DualModelStrategy as alias for backward compatibility
from .strategies import AsymmetricStrategy as DualModelStrategy
from .state import Account, PositionManager, OrderBook, Position, Order, OrderSide, OrderType, OrderStatus
from .execution import Executor, FillInfo
from .logic import RiskManager, PositionAllocator
from .storage import MemoryStorage
from .backtest import run_custom_backtest

__all__ = [
    "AsymmetricStrategy",
    "DualModelStrategy",  # Alias for backward compatibility
    "Account",
    "PositionManager",
    "OrderBook",
    "Position",
    "Order",
    "OrderSide",
    "OrderType",
    "OrderStatus",
    "Executor",
    "FillInfo",
    "RiskManager",
    "PositionAllocator",
    "MemoryStorage",
    "run_custom_backtest",
]

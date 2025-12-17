"""Trading strategy module for ATM project."""

from atm.trading.strategy.backtrader_strategy import BacktraderStrategy
from atm.trading.strategy.base import BaseStrategy, StrategyConfig
from atm.trading.strategy.sma_cross_strategy import SMACrossStrategy

__all__ = [
    "BaseStrategy",
    "BacktraderStrategy",
    "StrategyConfig",
    "SMACrossStrategy",
]


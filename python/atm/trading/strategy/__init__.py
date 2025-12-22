"""Trading strategy module for ATM project."""

from atm.trading.strategy.arg_parser import (
    create_strategy_parser,
    parse_date_args,
    parse_strategy_args,
    validate_dates,
)
from atm.trading.strategy.base import BaseStrategy, StrategyConfig
from atm.trading.strategy.atr_strategy import ATRStrategy
from atm.trading.strategy.cci_strategy import CCIStrategy
from atm.trading.strategy.cci_strategy_optimized import CCIStrategyOptimized
from atm.trading.strategy.fibonacci_ma_strategy import FibonacciMAStrategy
from atm.trading.strategy.improved_macd_strategy import ImprovedMACDStrategy
from atm.trading.strategy.strategy_runner import StrategyRunner
from atm.trading.strategy.sma_cross_strategy import SMACrossStrategy

__all__ = [
    "BaseStrategy",
    "StrategyConfig",
    "StrategyRunner",
    "SMACrossStrategy",
    "CCIStrategy",
    "CCIStrategyOptimized",
    "ATRStrategy",
    "FibonacciMAStrategy",
    "ImprovedMACDStrategy",
    "create_strategy_parser",
    "parse_date_args",
    "parse_strategy_args",
    "validate_dates",
]


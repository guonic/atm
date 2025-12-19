"""Trading strategy module for ATM project."""

from atm.trading.strategy.arg_parser import (
    create_strategy_parser,
    parse_date_args,
    parse_strategy_args,
    validate_dates,
)
from atm.trading.strategy.base import BaseStrategy, StrategyConfig
from atm.trading.strategy.strategy_runner import StrategyRunner
from atm.trading.strategy.sma_cross_strategy import SMACrossStrategy

__all__ = [
    "BaseStrategy",
    "StrategyConfig",
    "StrategyRunner",
    "SMACrossStrategy",
    "create_strategy_parser",
    "parse_date_args",
    "parse_strategy_args",
    "validate_dates",
]


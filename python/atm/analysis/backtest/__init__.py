"""Backtest module for ATM project."""

from atm.analysis.backtest.base import BaseBacktester, BacktestResult
from atm.analysis.backtest.batch_evaluator import BatchStrategyEvaluator
from atm.analysis.backtest.common_args import create_base_parser, parse_common_args, validate_dates
from atm.analysis.backtest.evaluator import BacktestEvaluator
from atm.analysis.backtest.evaluation_runner import run_strategy_evaluation
from atm.analysis.backtest.metrics import BacktestMetrics
from atm.analysis.backtest.predictor_backtest import PredictorBacktester
from atm.analysis.backtest.report import BacktestReport
from atm.analysis.backtest.strategy_backtest import StrategyBacktester

__all__ = [
    "BaseBacktester",
    "BacktestResult",
    "BacktestEvaluator",
    "BatchStrategyEvaluator",
    "BacktestMetrics",
    "PredictorBacktester",
    "StrategyBacktester",
    "BacktestReport",
    "create_base_parser",
    "parse_common_args",
    "validate_dates",
    "run_strategy_evaluation",
]


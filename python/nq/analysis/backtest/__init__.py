"""Backtest module for ATM project."""

from nq.analysis.backtest.base import BaseBacktester, BacktestResult
from nq.analysis.backtest.batch_evaluator import BatchStrategyEvaluator
from nq.analysis.backtest.common_args import create_base_parser, parse_common_args, validate_dates
from nq.analysis.backtest.evaluator import BacktestEvaluator
from nq.analysis.backtest.evaluation_runner import run_strategy_evaluation
from nq.analysis.backtest.metrics import BacktestMetrics
from nq.analysis.backtest.predictor_backtest import PredictorBacktester
from nq.analysis.backtest.report import BacktestReport
from nq.analysis.backtest.strategy_backtest import StrategyBacktester

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


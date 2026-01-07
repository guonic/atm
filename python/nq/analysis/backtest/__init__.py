"""Backtest module for ATM project."""

# Use relative imports for same-package modules
from .base import BaseBacktester, BacktestResult
from .batch_evaluator import BatchStrategyEvaluator
from .common_args import create_base_parser, parse_common_args, validate_dates
from .evaluator import BacktestEvaluator
from .evaluation_runner import run_strategy_evaluation
from .metrics import BacktestMetrics
from .predictor_backtest import PredictorBacktester
from .strategy_backtest import StrategyBacktester

# Import BacktestReport from report.py file (not report/ directory)
# Due to naming conflict with report/ directory, use importlib with proper package context
import importlib.util
import sys
import types
from pathlib import Path

_report_file_path = Path(__file__).parent / "report.py"
if _report_file_path.exists():
    # Set up proper package context for relative imports
    # The current module is nq.analysis.backtest, so report.py should be in the same package
    package_name = __name__  # nq.analysis.backtest
    module_name = f"{package_name}.report_file"  # nq.analysis.backtest.report_file
    
    # Ensure current package module is in sys.modules (it should be, but make sure)
    if package_name not in sys.modules:
        sys.modules[package_name] = sys.modules[__name__]
    
    # Ensure parent modules are in sys.modules for relative imports to work
    parent_parts = package_name.split(".")
    for i in range(1, len(parent_parts) + 1):
        parent_name = ".".join(parent_parts[:i])
        if parent_name not in sys.modules:
            # Create a dummy module for parent packages
            parent_module = types.ModuleType(parent_name)
            if i < len(parent_parts):
                # Only set __path__ for packages (not leaf modules)
                parent_module.__path__ = []
            sys.modules[parent_name] = parent_module
    
    spec = importlib.util.spec_from_file_location(module_name, _report_file_path)
    _backtest_report_module = importlib.util.module_from_spec(spec)
    
    # Set __package__ and __name__ to enable relative imports
    _backtest_report_module.__package__ = package_name
    _backtest_report_module.__name__ = module_name
    
    # Add to sys.modules BEFORE exec_module so relative imports can find it
    sys.modules[module_name] = _backtest_report_module
    
    spec.loader.exec_module(_backtest_report_module)
    BacktestReport = _backtest_report_module.BacktestReport
else:
    raise ImportError(f"Could not find report.py file at {_report_file_path}")

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


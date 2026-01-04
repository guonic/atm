#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
backtest_structure_expert.py

Description:
    Backtest script for Structure Expert GNN model.
    This script demonstrates how to:
    1. Load a trained Structure Expert model (.pth file)
    2. Generate predictions for each trading day
    3. Convert predictions to Qlib format signals
    4. Run backtest using Qlib's backtest framework
    5. Display backtest results and metrics

Usage:
    # Basic usage
    python backtest_structure_expert.py \
        --model_path models/structure_expert.pth \
        --start_date 2024-01-01 \
        --end_date 2024-06-30

    # Custom portfolio strategy
    python backtest_structure_expert.py \
        --model_path models/structure_expert.pth \
        --start_date 2024-01-01 \
        --end_date 2024-06-30 \
        --top_k 30 \
        --initial_cash 1000000

    # Use RefinedTopKStrategy to reduce turnover
    python backtest_structure_expert.py \
        --model_path models/structure_expert.pth \
        --start_date 2024-01-01 \
        --end_date 2024-06-30 \
        --strategy RefinedTopKStrategy \
        --buffer_ratio 0.15 \
        --top_k 30

    # Use SimpleLowTurnoverStrategy (simpler, only sell if stock falls out of top N)
    python backtest_structure_expert.py \
        --model_path models/structure_expert.pth \
        --start_date 2024-01-01 \
        --end_date 2024-06-30 \
        --strategy SimpleLowTurnoverStrategy \
        --retain_threshold 30 \
        --top_k 10

Arguments:
    --model_path         Path to trained model file (.pth)
    --start_date         Start date of backtest period (YYYY-MM-DD)
    --end_date           End date of backtest period (YYYY-MM-DD)
    --top_k              Number of top stocks to select (default: 30)
    --strategy           Portfolio strategy class name (default: TopkDropoutStrategy)
                         Options: 'TopkDropoutStrategy', 'RefinedTopKStrategy', 'SimpleLowTurnoverStrategy'
    --buffer_ratio       Buffer ratio for RefinedTopKStrategy (default: 0.15)
                         Higher value reduces turnover but may reduce returns
    --retain_threshold   Retain threshold for SimpleLowTurnoverStrategy (default: 30)
                         Only sell existing holdings if they fall out of top N
    --initial_cash       Initial cash amount (default: 1000000)
    --save_results       Save backtest results to file (default: False)
    --qlib_dir           Qlib data directory (default: ~/.qlib/qlib_data/cn_data)
    --region             Qlib region (default: cn)
    --n_feat             Number of input features (default: 158 for Alpha158)
    --n_hidden           Hidden layer size (default: 128)
    --n_heads            Number of attention heads (default: 8)
    --device             Device to use (default: cuda if available, else cpu)
    --config_path        Path to config file (optional, for database config)

Output:
    - Prints backtest metrics and portfolio performance
    - Optionally saves results to file
"""

import argparse
import importlib.util
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

# Gymnasium compatibility patch for Qlib - MUST be before any qlib imports
# Qlib's RL module imports 'gym', but gym is unmaintained.
# Gymnasium is the maintained drop-in replacement.
# This patch allows Qlib to use gymnasium if installed.
# Note: Using importlib.import_module() to avoid import statements in conditional blocks
_gymnasium_spec = importlib.util.find_spec("gymnasium")
if _gymnasium_spec is not None:
    gym = importlib.import_module("gymnasium")
    sys.modules['gym'] = gym
    spaces = importlib.import_module("gymnasium.spaces")
    sys.modules['gym.spaces'] = spaces
    _gym_patched = True
else:
    # Fallback to gym if gymnasium is not available
    _gym_spec = importlib.util.find_spec("gym")
    if _gym_spec is not None:
        gym = importlib.import_module("gym")
        _gym_patched = False
    else:
        raise ImportError(
            "Neither 'gymnasium' nor 'gym' is installed. "
            "Qlib's backtest module requires one of them. "
            "Please install gymnasium (recommended):\n"
            "  pip install gymnasium\n"
            "Or install gym (deprecated):\n"
            "  pip install gym"
        )

import numpy as np
import pandas as pd
import qlib
import torch
from qlib.backtest import backtest, executor
from qlib.backtest.decision import Order as QlibOrder
from qlib.contrib.data.handler import Alpha158
from qlib.contrib.strategy.signal_strategy import TopkDropoutStrategy
from qlib.data import D

# Matplotlib for visualization (optional dependency)
# Using importlib.import_module() to avoid import statements in conditional blocks
_matplotlib_spec = importlib.util.find_spec("matplotlib")
if _matplotlib_spec is not None:
    matplotlib = importlib.import_module("matplotlib")
    matplotlib.use('Agg')  # Use non-interactive backend
    plt = importlib.import_module("matplotlib.pyplot")
    mdates = importlib.import_module("matplotlib.dates")
    font_manager = importlib.import_module("matplotlib.font_manager")
    HAS_MATPLOTLIB = True
else:
    HAS_MATPLOTLIB = False
    # Set dummy values to avoid NameError when HAS_MATPLOTLIB is False
    matplotlib = None
    plt = None
    mdates = None
    font_manager = None

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nq.config import load_config
from nq.utils.data_normalize import normalize_index_code, normalize_stock_code
from nq.utils.industry import load_industry_label_map, load_industry_map
from nq.utils.model import save_embeddings as save_embeddings_to_file
from nq.analysis.backtest.eidos_integration import EidosBacktestWriter
from nq.analysis.backtest.eidos_structure_expert import save_structure_expert_backtest_to_eidos
from nq.analysis.backtest.qlib_types import QlibBacktestResult

# Import structure expert model using standard package import
from tools.qlib.train.structure_expert import (
    GraphDataBuilder,
    StructureExpertGNN,
    load_structure_expert_model,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Log gymnasium patch status (only once, suppress in worker processes)
# Check if we're in the main process by checking if __name__ is __main__
# In multiprocessing, worker processes have __name__ == '__mp_main__'
_gym_patch_logged = False
if __name__ == "__main__":
    _gym_patch_logged = True
    if _gym_patched:
        logger.info("Patched 'gym' import to use 'gymnasium'.")
    else:
        logger.warning(
            "Using deprecated 'gym' package. "
            "Please install 'gymnasium' (pip install gymnasium) for better compatibility."
        )

# Default settings
DEFAULT_TOP_K = 30
DEFAULT_INITIAL_CASH = 1000000
DEFAULT_QLIB_DIR = "~/.qlib/qlib_data/cn_data"
DEFAULT_N_FEAT = 158  # Alpha158 features
DEFAULT_N_HIDDEN = 128
DEFAULT_N_HEADS = 8


# Use unified stock code normalization
convert_ts_code_to_qlib_format = normalize_stock_code

# Use system library function for benchmark code normalization
# Benchmark codes are index codes, so use normalize_index_code
normalize_benchmark_code = normalize_index_code

# Use system library function for loading Structure Expert model
load_model = load_structure_expert_model


def get_refined_top_k(
    current_holdings: List[str],
    pred_scores: Dict[str, float],
    top_k: int = 10,
    buffer_ratio: float = 0.15,
) -> List[str]:
    """
    Get refined top K stocks with reduced turnover.

    This function implements a buffer mechanism to reduce portfolio turnover by
    keeping existing holdings that are still performing well, even if they're
    not in the absolute top K.

    Args:
        current_holdings: Current holdings list (e.g., ['000001.SZ', ...]).
        pred_scores: Model prediction scores dict {symbol: score}.
        top_k: Target number of holdings.
        buffer_ratio: Buffer ratio. Only swap when new stock ranks significantly
                     higher than old holdings.

    Returns:
        List of selected stock symbols.

    Examples:
        >>> current = ['000001.SZ', '000002.SZ']
        >>> scores = {'000001.SZ': 0.5, '000002.SZ': 0.3, '000003.SZ': 0.8, '000004.SZ': 0.7}
        >>> selected = get_refined_top_k(current, scores, top_k=2, buffer_ratio=0.15)
        >>> print(selected)
        ['000003.SZ', '000001.SZ']  # Keeps 000001.SZ if it's still in top 2.3
    """
    # 1. Sort all stocks by prediction scores
    sorted_scores = sorted(pred_scores.items(), key=lambda x: x[1], reverse=True)
    candidates = [s[0] for s in sorted_scores]

    # Get absolute Top K (hard threshold)
    absolute_top_k = set(candidates[:top_k])

    # 2. If no current holdings, return absolute Top K
    if not current_holdings:
        return list(absolute_top_k)

    # 3. Core logic: Provide "competitive protection" for existing holdings
    # Only swap when new stock enters top TopK * (1 - buffer_ratio) region,
    # or old stock falls out of TopK * (1 + buffer_ratio) region.

    new_holdings = []

    # Check which existing holdings can still compete (still in top ranks)
    # Give existing holdings a buffer zone
    refined_limit = int(top_k * (1 + buffer_ratio))
    keep_holdings = [s for s in current_holdings if s in candidates[:refined_limit]]

    # 4. Fill new holdings
    # First add good existing holdings
    new_holdings.extend(keep_holdings)

    # Fill remaining slots with top-scoring new stocks
    for stock in candidates:
        if len(new_holdings) >= top_k:
            break
        if stock not in new_holdings:
            new_holdings.append(stock)

    return new_holdings


def execution_logic(
    current_portfolio: Dict[str, float],
    pred_scores: Dict[str, float],
    top_k: int = 10,
    retain_threshold: int = 30,
) -> List[str]:
    """
    Simple low turnover execution logic.

    This function implements a simple strategy to reduce portfolio turnover:
    - Only sell existing holdings if they fall out of top N (retain_threshold)
    - Fill empty slots with highest-scoring new stocks

    Args:
        current_portfolio: Current portfolio dict {symbol: weight} or {symbol: amount}.
        pred_scores: Model prediction scores dict {symbol: score}.
        top_k: Target number of holdings.
        retain_threshold: Threshold for retaining existing holdings (default: 30).
                        Only sell if stock falls out of top N.

    Returns:
        List of selected stock symbols.

    Examples:
        >>> current = {'000001.SZ': 0.1, '000002.SZ': 0.1}
        >>> scores = {'000001.SZ': 0.5, '000002.SZ': 0.3, '000003.SZ': 0.8, '000004.SZ': 0.7}
        >>> selected = execution_logic(current, scores, top_k=2, retain_threshold=30)
        >>> print(selected)
        ['000003.SZ', '000001.SZ']  # Keeps 000001.SZ if it's in top 30
    """
    # 1. Sort stocks by prediction scores
    sorted_stocks = sorted(pred_scores.items(), key=lambda x: x[1], reverse=True)
    potential_candidates = [s[0] for s in sorted_stocks]

    # 2. Only sell existing holdings if they fall out of top N
    current_holdings = list(current_portfolio.keys())
    next_holdings = []

    # Check existing holdings
    for stock in current_holdings:
        # If old stock is still in top N, keep it
        if stock in potential_candidates[:retain_threshold]:
            next_holdings.append(stock)

    # 3. Fill empty slots with highest-scoring new stocks
    for stock in potential_candidates:
        if len(next_holdings) >= top_k:
            break
        if stock not in next_holdings:
            next_holdings.append(stock)

    return next_holdings


class BaseCustomStrategy(TopkDropoutStrategy):
    """
    Base class for all custom strategies with order capture functionality.
    
    This class provides a unified interface for capturing executed orders from Qlib executor.
    All custom strategies should inherit from this class to enable order tracking.
    """
    
    def __init__(self, *args, **kwargs):
        """
        Initialize base custom strategy.
        
        Args:
            *args: Positional arguments passed to parent class.
            **kwargs: Keyword arguments passed to parent class.
        """
        super().__init__(*args, **kwargs)
        # Store executed orders from post_exe_step
        self.executed_orders: List[Dict[str, Any]] = []
    
    def post_exe_step(self, execute_result: Optional[list]) -> None:
        """
        Capture executed orders from Qlib executor.
        
        This method is called by Qlib after each execution step to capture order information.
        The execute_result contains a list of tuples: (order, trade_val, trade_cost, trade_price).
        
        Args:
            execute_result: List of tuples, each containing (order, trade_val, trade_cost, trade_price).
                          Each tuple represents an executed order with:
                          - order: qlib.backtest.decision.Order object from Qlib
                          - trade_val: Trade volume (交易量)
                          - trade_cost: Trade cost (成本)
                          - trade_price: Trade price (价格)
        """
        if execute_result is None or len(execute_result) == 0:
            return
        
        # Process each executed order tuple
        logger.debug(f"Processing {len(execute_result)} executed orders")
        for order_tuple in execute_result:
            # Unpack tuple: (order, trade_val, trade_cost, trade_price)
            if len(order_tuple) != 4:
                logger.warning(f"Unexpected execute_result tuple length: {len(order_tuple)}, expected 4")
                continue
            
            order: QlibOrder
            order, trade_val, trade_cost, trade_price = order_tuple
            
            # Debug: Log all order information to understand Qlib's direction values
            logger.debug(
                f"Order details: stock_id={order.stock_id}, "
                f"direction={order.direction} (type={type(order.direction).__name__}), "
                f"amount={order.amount}, price={trade_price}, "
                f"hasattr(direction)={hasattr(order, 'direction')}"
            )
            
            # Extract order information from qlib.backtest.decision.Order
            # Qlib Order object has: stock_id, amount, direction, factor, start_time, end_time, etc.
            # direction must be 1 (Buy) or -1 (Sell) to satisfy database constraint
            # Note: We've already filtered out direction==0 above, so order.direction is either 1 or -1
            order_info = {
                "instrument": order.stock_id,  # Qlib Order uses stock_id attribute
                "amount": order.amount,
                "direction": -1 if order.direction == 0 else 1,  # 1=Buy, -1=Sell (already filtered out 0)
                "factor": order.factor,
                "trade_val": float(trade_val),  # 交易量
                "trade_cost": float(trade_cost),  # 成本
                "trade_price": float(trade_price),  # 价格
                "deal_time": order.end_time,
            }
            
            # Store order information
            self.executed_orders.append(order_info)
            
            # Log order execution
            direction_str = "BUY" if order_info["direction"] == 1 else "SELL"
            date_str = f" on {order_info['deal_time']}" if order_info['deal_time'] else ""
            logger.info(
                f"Order executed{date_str}: {direction_str} {order_info['instrument']}, "
                f"amount={order_info['amount']}, price={order_info['trade_price']:.4f}, "
                f"val={order_info['trade_val']:.2f}, cost={order_info['trade_cost']:.2f}, "
                f"direction={order_info['direction']} (order.direction={order.direction})"
            )
    
    def get_executed_orders(self) -> List[Dict[str, Any]]:
        """
        Get all executed orders captured during backtest.
        
        Returns:
            List of executed order dictionaries, each containing:
            - instrument: Stock symbol
            - amount: Order amount
            - direction: Order direction (1=Buy, -1=Sell)
            - factor: Order factor
            - trade_val: Trade volume
            - trade_cost: Trade cost
            - trade_price: Trade price
        """
        # Log summary of captured orders
        buy_count = sum(1 for o in self.executed_orders if o.get("direction") == 1)
        sell_count = sum(1 for o in self.executed_orders if o.get("direction") == -1)
        other_count = len(self.executed_orders) - buy_count - sell_count
        logger.info(
            f"Total executed orders: {len(self.executed_orders)}, "
            f"BUY: {buy_count}, SELL: {sell_count}, Other: {other_count}"
        )
        if other_count > 0:
            # Log orders with unexpected direction values
            for o in self.executed_orders:
                if o.get("direction") not in [1, -1]:
                    logger.warning(f"Order with unexpected direction: {o}")
        return self.executed_orders.copy()


class RefinedTopKStrategy(BaseCustomStrategy):
    """
    Custom strategy with reduced turnover using buffer mechanism.

    This strategy extends TopkDropoutStrategy with a buffer mechanism to reduce
    portfolio turnover by keeping existing holdings that are still performing well.
    It wraps the signal DataFrame to apply get_refined_top_k logic before passing
    to the base TopkDropoutStrategy.
    """

    def __init__(
        self,
        signal: pd.DataFrame,
        topk: int = 30,
        buffer_ratio: float = 0.15,
        n_drop: int = 5,
    ):
        """
        Initialize RefinedTopKStrategy.

        Args:
            signal: Signal DataFrame with MultiIndex (datetime, instrument) and 'score' column.
            topk: Target number of stocks to hold.
            buffer_ratio: Buffer ratio for reducing turnover (default: 0.15).
            n_drop: Number of bottom stocks to drop (default: 5).
        """
        # Apply refined top K logic to signal before passing to base strategy
        refined_signal = self._apply_refined_top_k(signal, topk, buffer_ratio)

        # Initialize base strategy with refined signal
        super().__init__(signal=refined_signal, topk=topk, n_drop=n_drop)
        self.buffer_ratio = buffer_ratio
        self.original_signal = signal

    def _apply_refined_top_k(
        self,
        signal: pd.DataFrame,
        topk: int,
        buffer_ratio: float,
    ) -> pd.DataFrame:
        """
        Apply refined top K logic to signal DataFrame.

        Args:
            signal: Original signal DataFrame.
            topk: Target number of stocks.
            buffer_ratio: Buffer ratio.

        Returns:
            Refined signal DataFrame with adjusted scores.
        """
        if len(signal) == 0:
            return signal

        # Create a copy to modify
        refined_signal = signal.copy()

        # Group by date and apply refined top K for each date
        dates = signal.index.get_level_values(0).unique()
        current_holdings = []
        total_kept = 0

        for date in dates:
            date_signal = signal.loc[date]

            # Get prediction scores for this date
            if isinstance(date_signal, pd.Series):
                # Single stock case
                pred_scores = {date_signal.name: date_signal.get('score', 0.0)}
            else:
                # Multiple stocks case
                pred_scores = date_signal['score'].to_dict()

            # Apply refined top K selection
            selected_stocks = get_refined_top_k(
                current_holdings=current_holdings,
                pred_scores=pred_scores,
                top_k=topk,
                buffer_ratio=buffer_ratio,
            )

            # Count how many existing holdings were kept
            kept_count = len(set(current_holdings) & set(selected_stocks))
            total_kept += kept_count

            # Update current holdings for next iteration
            current_holdings = selected_stocks

            # Modify signal: set low score for non-selected stocks
            if isinstance(date_signal, pd.Series):
                if date_signal.name not in selected_stocks:
                    refined_signal.loc[(date, date_signal.name), 'score'] = -999.0
            else:
                # Set low score for non-selected stocks
                for stock in date_signal.index:
                    if stock not in selected_stocks:
                        refined_signal.loc[(date, stock), 'score'] = -999.0

        # Log statistics
        avg_kept = total_kept / len(dates) if len(dates) > 0 else 0
        logger.info(
            f"Applied refined top K logic: average {avg_kept:.1f} holdings kept per day "
            f"(out of {topk}) with buffer_ratio={buffer_ratio}"
        )

        return refined_signal


class SimpleLowTurnoverStrategy(BaseCustomStrategy):
    """
    Simple low turnover strategy with retain threshold.

    This strategy extends TopkDropoutStrategy with a simple retain threshold mechanism:
    - Only sell existing holdings if they fall out of top N (retain_threshold)
    - Fill empty slots with highest-scoring new stocks
    - Much simpler than RefinedTopKStrategy, focuses on reducing unnecessary turnover
    """

    def __init__(
        self,
        signal: pd.DataFrame,
        topk: int = 30,
        retain_threshold: int = 30,
        n_drop: int = 5,
    ):
        """
        Initialize SimpleLowTurnoverStrategy.

        Args:
            signal: Signal DataFrame with MultiIndex (datetime, instrument) and 'score' column.
            topk: Target number of stocks to hold.
            retain_threshold: Threshold for retaining existing holdings (default: 30).
                            Only sell if stock falls out of top N.
            n_drop: Number of bottom stocks to drop (default: 5).
        """
        # Apply simple low turnover logic to signal before passing to base strategy
        refined_signal = self._apply_execution_logic(signal, topk, retain_threshold)

        # Initialize base strategy with refined signal
        super().__init__(signal=refined_signal, topk=topk, n_drop=n_drop)
        self.retain_threshold = retain_threshold
        self.original_signal = signal

    def _apply_execution_logic(
        self,
        signal: pd.DataFrame,
        topk: int,
        retain_threshold: int,
    ) -> pd.DataFrame:
        """
        Apply simple low turnover execution logic to signal DataFrame.

        Args:
            signal: Original signal DataFrame.
            topk: Target number of stocks.
            retain_threshold: Threshold for retaining existing holdings.

        Returns:
            Refined signal DataFrame with adjusted scores.
        """
        if len(signal) == 0:
            return signal

        # Create a copy to modify
        refined_signal = signal.copy()

        # Group by date and apply execution logic for each date
        dates = signal.index.get_level_values(0).unique()
        current_portfolio = {}  # Track current portfolio as dict
        total_kept = 0

        for date in dates:
            date_signal = signal.loc[date]

            # Get prediction scores for this date
            if isinstance(date_signal, pd.Series):
                # Single stock case
                pred_scores = {date_signal.name: date_signal.get('score', 0.0)}
            else:
                # Multiple stocks case
                pred_scores = date_signal['score'].to_dict()

            # Apply execution logic
            selected_stocks = execution_logic(
                current_portfolio=current_portfolio,
                pred_scores=pred_scores,
                top_k=topk,
                retain_threshold=retain_threshold,
            )

            # Count how many existing holdings were kept
            kept_count = len(set(current_portfolio.keys()) & set(selected_stocks))
            total_kept += kept_count

            # Update current portfolio for next iteration
            # Convert to dict format for next iteration
            current_portfolio = {stock: 1.0 / len(selected_stocks) for stock in selected_stocks}

            # Modify signal: set low score for non-selected stocks
            if isinstance(date_signal, pd.Series):
                if date_signal.name not in selected_stocks:
                    refined_signal.loc[(date, date_signal.name), 'score'] = -999.0
            else:
                # Set low score for non-selected stocks
                for stock in date_signal.index:
                    if stock not in selected_stocks:
                        refined_signal.loc[(date, stock), 'score'] = -999.0

        # Log statistics
        avg_kept = total_kept / len(dates) if len(dates) > 0 else 0
        logger.info(
            f"Applied simple low turnover logic: average {avg_kept:.1f} holdings kept per day "
            f"(out of {topk}) with retain_threshold={retain_threshold}"
        )

        return refined_signal


def _get_qlib_data_range() -> tuple[Optional[datetime.date], Optional[datetime.date]]:
    """
    Get the date range of available Qlib data.

    Returns:
        Tuple of (start_date, end_date) as date objects, or (None, None) if no data.
    """
    full_calendar = D.calendar()
    if len(full_calendar) == 0:
        return None, None

    data_start_ts = full_calendar[0]
    data_end_ts = full_calendar[-1]

    # Convert Timestamp to date
    if isinstance(data_start_ts, pd.Timestamp):
        data_start_date = data_start_ts.date()
    elif hasattr(data_start_ts, 'date'):
        data_start_date = data_start_ts.date()
    else:
        data_start_date = data_start_ts

    if isinstance(data_end_ts, pd.Timestamp):
        data_end_date = data_end_ts.date()
    elif hasattr(data_end_ts, 'date'):
        data_end_date = data_end_ts.date()
    else:
        data_end_date = data_end_ts

    return data_start_date, data_end_date


def _load_instruments(qlib_dir: Optional[str] = None) -> Optional[List[str]]:
    """
    Load instrument list from Qlib data directory.

    Args:
        qlib_dir: Qlib data directory path.

    Returns:
        List of instrument codes, or None if loading fails.
    """
    try:
        qlib_dir_path = Path(qlib_dir if qlib_dir else '~/.qlib/qlib_data/cn_data').expanduser()
        instruments_file = qlib_dir_path / 'instruments' / 'all.txt'
        if not instruments_file.exists():
            logger.warning(f"Instruments file not found: {instruments_file}")
            return None

        instruments = []
        with open(instruments_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # Parse tab-separated format: stock_code\tstart_date\tend_date
                parts = line.split('\t')
                if len(parts) >= 1:
                    stock_code = parts[0].strip()
                    if stock_code:
                        instruments.append(stock_code)

        logger.debug(f"Loaded {len(instruments)} instruments from file")
        return instruments
    except Exception as e:
        logger.debug(f"Could not load instruments: {e}")
        return None


def _calculate_fit_period(
    lookback_start: datetime.date,
    data_start_date: Optional[datetime.date],
    lookback_start_str: str,
) -> tuple[str, str]:
    """
    Calculate the fit period for Alpha158 feature handler.

    Args:
        lookback_start: Start date of lookback period.
        data_start_date: Start date of available data.
        lookback_start_str: String representation of lookback start date.

    Returns:
        Tuple of (fit_start_str, fit_end_str) date strings.
    """
    fit_end_date = lookback_start - timedelta(days=1)
    fit_start_date = fit_end_date - timedelta(days=365)  # 1 year of data for fitting

    # Ensure fit period is within available data range
    if data_start_date is not None:
        if fit_start_date < data_start_date:
            logger.debug(f"Fit start {fit_start_date} is before data start {data_start_date}, adjusting...")
            fit_start_date = data_start_date
        if fit_end_date < data_start_date:
            logger.warning(
                f"Fit end {fit_end_date} is before data start {data_start_date}, using minimal fit period"
            )
            fit_start_date = data_start_date
            # Use at least 30 days for fit if possible
            fit_calendar = D.calendar(start_time=data_start_date.strftime("%Y-%m-%d"), end_time=lookback_start_str)
            if len(fit_calendar) > 30:
                fit_end_ts = fit_calendar[-30]
                fit_end_date = fit_end_ts.date() if hasattr(fit_end_ts, 'date') else fit_end_ts
            else:
                if len(fit_calendar) > 0:
                    fit_end_ts = fit_calendar[-1]
                    fit_end_date = fit_end_ts.date() if hasattr(fit_end_ts, 'date') else fit_end_ts
                else:
                    fit_end_date = lookback_start

    fit_start_str = fit_start_date.strftime("%Y-%m-%d")
    fit_end_str = fit_end_date.strftime("%Y-%m-%d")
    return fit_start_str, fit_end_str


def _load_features_for_date(
    trade_date: pd.Timestamp,
    lookback_days: int,
    instruments: Optional[List[str]],
    data_start_date: Optional[datetime.date],
) -> Optional[pd.DataFrame]:
    """
    Load Alpha158 features for a specific trading date.

    Args:
        trade_date: Trading date to load features for.
        lookback_days: Number of days to look back for feature calculation.
        instruments: Optional list of instruments to load.
        data_start_date: Start date of available data.

    Returns:
        DataFrame with features, or None if loading fails.
    """
    trade_date_str = trade_date.strftime("%Y-%m-%d")
    lookback_start = trade_date - timedelta(days=lookback_days)
    lookback_start_str = lookback_start.strftime("%Y-%m-%d")

    # Convert lookback_start to date for fit period calculation
    if isinstance(lookback_start, pd.Timestamp):
        lookback_start_date = lookback_start.date()
    elif hasattr(lookback_start, 'date'):
        lookback_start_date = lookback_start.date()
    else:
        lookback_start_date = lookback_start

    fit_start_str, fit_end_str = _calculate_fit_period(lookback_start_date, data_start_date, lookback_start_str)

    logger.debug(f"Alpha158 fit period: {fit_start_str} to {fit_end_str}")
    logger.debug(f"Alpha158 inference period: {lookback_start_str} to {trade_date_str}")

    # Try with lookback first
    try:
        handler_kwargs = {
            "start_time": lookback_start_str,
            "end_time": trade_date_str,
            "fit_start_time": fit_start_str,
            "fit_end_time": fit_end_str,
        }
        if instruments is not None:
            handler_kwargs["instruments"] = instruments

        date_handler = Alpha158(**handler_kwargs)
        date_handler.setup_data()

        try:
            df_x = date_handler.data
            if df_x is None or df_x.empty:
                df_x = date_handler.fetch(col_set="feature")
        except Exception:
            df_x = date_handler.fetch(col_set="feature")
    except Exception as e:
        logger.debug(f"Failed to load with lookback ({lookback_start_str} to {trade_date_str}): {e}")
        # Try without lookback
        logger.debug(f"Trying without lookback (using only {trade_date_str})...")
        handler_kwargs = {
            "start_time": trade_date_str,
            "end_time": trade_date_str,
            "fit_start_time": fit_start_str,
            "fit_end_time": fit_end_str,
        }
        if instruments is not None:
            handler_kwargs["instruments"] = instruments

        date_handler = Alpha158(**handler_kwargs)
        date_handler.setup_data()
        try:
            df_x = date_handler.data
            if df_x is None or df_x.empty:
                df_x = date_handler.fetch(col_set="feature")
        except Exception:
            df_x = date_handler.fetch(col_set="feature")

    if df_x.empty:
        # Diagnostic check
        try:
            instruments_check = D.instruments()
            logger.debug(f"Total instruments available: {len(instruments_check)}")
            if len(instruments_check) > 0:
                sample_stock = instruments_check[0]
                sample_data = D.features(
                    [sample_stock],
                    ["$close"],
                    start_time=trade_date_str,
                    end_time=trade_date_str,
                    freq="day",
                )
                logger.debug(f"Sample stock {sample_stock} data for {trade_date_str}: {sample_data.shape}")
        except Exception as diag_e:
            logger.debug(f"Diagnostic check failed: {diag_e}")

        logger.warning(
            f"No data for {trade_date_str} (with lookback from {lookback_start_str}). "
            f"Alpha158 returned empty DataFrame."
        )
        return None

    # Filter to only the target date
    if isinstance(df_x.index, pd.MultiIndex):
        date_level = df_x.index.get_level_values(0)
        if isinstance(date_level[0], pd.Timestamp):
            df_x = df_x.loc[date_level.date == trade_date.date()]
        else:
            date_strs = pd.to_datetime(date_level).dt.strftime("%Y-%m-%d")
            df_x = df_x.loc[date_strs == trade_date_str]

    if df_x.empty:
        logger.warning(f"No data for target date {trade_date_str} after filtering")
        return None

    # Clean NaN/Inf
    df_x = df_x.fillna(0.0).replace([np.inf, -np.inf], 0.0)
    return df_x


def _process_single_date(
    trade_date: pd.Timestamp,
    model: StructureExpertGNN,
    builder: GraphDataBuilder,
    device_obj: torch.device,
    lookback_days: int,
    instruments: Optional[List[str]],
    data_start_date: Optional[datetime.date],
    save_embeddings: bool,
    embeddings_storage_dir: Optional[str],
    industry_label_map: Optional[Dict[str, str]],
) -> Optional[pd.DataFrame]:
    """
    Process a single trading date to generate predictions.

    Args:
        trade_date: Trading date to process.
        model: Trained Structure Expert model.
        builder: GraphDataBuilder instance.
        device_obj: PyTorch device object.
        lookback_days: Number of days to look back for features.
        instruments: Optional list of instruments.
        data_start_date: Start date of available data.
        save_embeddings: Whether to save embeddings.
        embeddings_storage_dir: Directory to save embeddings.
        industry_label_map: Optional industry label mapping.

    Returns:
        DataFrame with predictions for this date, or None if processing fails.
    """
    trade_date_str = trade_date.strftime("%Y-%m-%d")
    logger.info(f"Processing {trade_date_str}")

    try:
        # Load features
        df_x = _load_features_for_date(trade_date, lookback_days, instruments, data_start_date)
        if df_x is None or df_x.empty:
            return None

        # Build graph
        daily_graph = builder.get_daily_graph(df_x, None)
        if daily_graph.x.shape[0] == 0:
            logger.warning(f"No stocks for {trade_date_str}, skipping")
            return None

        # Run inference
        logger.debug(f"Running inference for {trade_date_str}...")
        with torch.no_grad():
            data = daily_graph.to(device_obj)
            pred, embedding = model(data.x, data.edge_index)
            pred = pred.cpu().numpy().flatten()
            embedding_np = embedding.cpu().numpy()

        # Get stock symbols
        if hasattr(daily_graph, "symbols"):
            symbols = daily_graph.symbols
        else:
            symbols = df_x.index.get_level_values("instrument").unique().tolist()

        symbols_normalized = [normalize_stock_code(s) for s in symbols]

        # Save embeddings if requested
        if save_embeddings:
            save_embeddings_to_file(
                symbols=symbols,
                symbols_normalized=symbols_normalized,
                predictions=pred,
                embeddings=embedding_np,
                trade_date_str=trade_date_str,
                storage_dir=embeddings_storage_dir,
                industry_label_map=industry_label_map,
            )

        # Validate symbol count
        if len(symbols) != len(pred):
            logger.warning(
                f"Symbol count ({len(symbols)}) != prediction count ({len(pred)}) "
                f"for {trade_date_str}, skipping"
            )
            return None

        # Create prediction DataFrame
        pred_df = pd.DataFrame(
            {"score": pred},
            index=pd.MultiIndex.from_product(
                [[trade_date], symbols],
                names=["datetime", "instrument"],
            ),
        )

        logger.debug(f"✓ Successfully generated predictions for {trade_date_str}: {len(symbols)} stocks")
        return pred_df

    except Exception as e:
        logger.error(f"Error processing {trade_date_str}: {e}", exc_info=True)
        return None


def generate_predictions(
    model: StructureExpertGNN,
    builder: GraphDataBuilder,
    start_date: str,
    end_date: str,
    device: str = "cuda",
    save_embeddings: bool = False,
    embeddings_storage_dir: Optional[str] = None,
    industry_label_map: Optional[Dict[str, str]] = None,
    qlib_dir: Optional[str] = None,
) -> pd.DataFrame:
    """
    Generate predictions for all trading days in the date range.

    Args:
        model: Trained Structure Expert model.
        builder: GraphDataBuilder instance.
        start_date: Start date (YYYY-MM-DD).
        end_date: End date (YYYY-MM-DD).
        device: Device to run inference on.
        save_embeddings: Whether to save embeddings.
        embeddings_storage_dir: Directory to save embeddings.
        industry_label_map: Optional industry label mapping.
        qlib_dir: Qlib data directory.

    Returns:
        DataFrame with predictions in Qlib format.
        Index: MultiIndex (datetime, instrument)
        Column: 'score' (prediction score)
    """
    logger.info(f"Generating predictions from {start_date} to {end_date}")
    logger.info("Note: This is inference only, not training. Model weights are frozen.")

    # Get trading calendar
    calendar = D.calendar(start_time=start_date, end_time=end_date)
    if len(calendar) == 0:
        raise ValueError(f"No trading days found between {start_date} and {end_date}")

    logger.info(f"Found {len(calendar)} trading days to process")

    # Get data range and load instruments
    data_start_date, data_end_date = _get_qlib_data_range()
    logger.debug(f"Qlib data range: {data_start_date} to {data_end_date}")

    LOOKBACK_DAYS = 60  # Number of historical days needed for feature calculation
    device_obj = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
    instruments = _load_instruments(qlib_dir)

    # Process each trading day
    all_predictions = []
    skipped_days = []
    successful_days = []

    for i, trade_date in enumerate(calendar):
        logger.info(f"Processing {trade_date.strftime('%Y-%m-%d')} ({i+1}/{len(calendar)})")

        pred_df = _process_single_date(
            trade_date=trade_date,
            model=model,
            builder=builder,
            device_obj=device_obj,
            lookback_days=LOOKBACK_DAYS,
            instruments=instruments,
            data_start_date=data_start_date,
            save_embeddings=save_embeddings,
            embeddings_storage_dir=embeddings_storage_dir,
            industry_label_map=industry_label_map,
        )

        if pred_df is not None:
            all_predictions.append(pred_df)
            successful_days.append(trade_date.strftime("%Y-%m-%d"))
        else:
            skipped_days.append(trade_date.strftime("%Y-%m-%d"))

    # Validate results
    if not all_predictions:
        error_msg = (
            f"No predictions generated for date range {start_date} to {end_date}.\n"
            f"  - Total trading days: {len(calendar)}\n"
            f"  - Successfully processed: {len(successful_days)}\n"
            f"  - Skipped/Failed: {len(skipped_days)}\n"
        )
        if skipped_days:
            error_msg += f"  - Skipped days: {', '.join(skipped_days[:10])}"
            if len(skipped_days) > 10:
                error_msg += f" ... and {len(skipped_days) - 10} more"
            error_msg += "\n"

        error_msg += (
            f"\nPossible reasons:\n"
            f"  1. Date range is in the future (no data available yet)\n"
            f"     → Use historical dates (e.g., 2024-01-01 to 2024-12-31)\n"
            f"  2. Qlib data is incomplete or missing\n"
            f"     → Check data availability: python -c \"import qlib; qlib.init(); from qlib.data import D; print(D.calendar())\"\n"
            f"  3. No stocks have data in the specified date range\n"
            f"     → Check stock list: python -c \"import qlib; qlib.init(); from qlib.data import D; print(D.instruments())\"\n"
        )

        raise ValueError(error_msg)

    # Concatenate all predictions
    predictions = pd.concat(all_predictions)
    logger.info(f"Generated predictions for {len(predictions)} stock-days")

    return predictions


def create_portfolio_strategy(
    predictions: pd.DataFrame,
    strategy_class: type = TopkDropoutStrategy,
    top_k: int = DEFAULT_TOP_K,
    buffer_ratio: Optional[float] = None,
    retain_threshold: Optional[int] = None,
) -> TopkDropoutStrategy:
    """
    Create portfolio strategy from predictions.

    Args:
        predictions: DataFrame with predictions (MultiIndex: datetime, instrument).
        strategy_class: Strategy class to use.
        top_k: Number of top stocks to select.
        buffer_ratio: Buffer ratio for RefinedTopKStrategy (default: None, uses 0.15 if RefinedTopKStrategy).
        retain_threshold: Retain threshold for SimpleLowTurnoverStrategy (default: None, uses 30 if SimpleLowTurnoverStrategy).

    Returns:
        Strategy instance.
    """
    logger.info(f"Creating {strategy_class.__name__} with top_k={top_k}")

    # Create signal DataFrame in Qlib format
    # Qlib expects a DataFrame with MultiIndex (datetime, instrument) and a 'score' column
    signal = predictions.copy()

    # Create strategy config
    if strategy_class == RefinedTopKStrategy:
        # Use RefinedTopKStrategy with buffer mechanism
        if buffer_ratio is None:
            buffer_ratio = 0.15  # Default buffer ratio
        logger.info(f"Using RefinedTopKStrategy with buffer_ratio={buffer_ratio} to reduce turnover")
        strategy_config = {
            "signal": signal,
            "topk": top_k,
            "buffer_ratio": buffer_ratio,
            "n_drop": 5,  # Drop bottom 5 to reduce turnover
        }
    elif strategy_class == SimpleLowTurnoverStrategy:
        # Use SimpleLowTurnoverStrategy with retain threshold
        if retain_threshold is None:
            retain_threshold = 30  # Default retain threshold
        logger.info(
            f"Using SimpleLowTurnoverStrategy with retain_threshold={retain_threshold} "
            f"to reduce turnover (only sell if stock falls out of top {retain_threshold})"
        )
        strategy_config = {
            "signal": signal,
            "topk": top_k,
            "retain_threshold": retain_threshold,
            "n_drop": 5,  # Drop bottom 5 to reduce turnover
        }
    else:
        # Use standard TopkDropoutStrategy
        strategy_config = {
            "signal": signal,
            "topk": top_k,
            "n_drop": 5,  # Drop bottom 5 to reduce turnover
        }

    # Initialize strategy
    strategy = strategy_class(**strategy_config)

    return strategy


def run_backtest(
    strategy: TopkDropoutStrategy,
    start_date: str,
    end_date: str,
    initial_cash: float = DEFAULT_INITIAL_CASH,
    benchmark: Optional[str] = None,
    skip_auto_detect: bool = False,
) -> dict:
    """
    Run backtest using Qlib's backtest framework.

    Args:
        strategy: Portfolio strategy instance.
        start_date: Start date (YYYY-MM-DD).
        end_date: End date (YYYY-MM-DD).
        initial_cash: Initial cash amount.
        benchmark: Optional benchmark code (e.g., 'SH000300').

    Returns:
        Dictionary containing backtest results and metrics.
    """
    logger.info(f"Running backtest from {start_date} to {end_date}")
    logger.info(f"Initial cash: {initial_cash:,.2f}")

    # Normalize benchmark: convert empty string to None
    if benchmark == "":
        benchmark = None

    # Normalize benchmark format if provided (in case not normalized in main)
    if benchmark is not None and benchmark.strip() != "":
        original_benchmark = benchmark
        benchmark = normalize_index_code(benchmark)
        if original_benchmark != benchmark:
            logger.debug(f"Normalized benchmark code: {original_benchmark} -> {benchmark}")

    # IMPORTANT: Qlib appears to require a benchmark for backtesting.
    # Even when --no_benchmark is used, we need to find a valid benchmark code.
    # We'll find one but note that it's auto-selected, not user-specified.
    
    if benchmark is None:
        if skip_auto_detect:
            logger.info(
                "Note: --no_benchmark was specified, but Qlib requires a benchmark. "
                "Automatically finding a valid benchmark to use..."
            )
        else:
            logger.info("No benchmark specified. Auto-detecting a valid benchmark...")
        
        # Try common benchmark formats (in Qlib format: code.EXCHANGE)
        common_benchmarks = ["000300.SH", "399001.SZ", "000001.SH", "000905.SH"]
        for bm in common_benchmarks:
            try:
                # Try to load benchmark data
                test_data = D.features([bm], ["$close"], start_time=start_date, end_time=end_date, freq="day")
                if not test_data.empty:
                    benchmark = bm
                    if skip_auto_detect:
                        logger.info(f"Auto-selected benchmark (required by Qlib): {benchmark}")
                    else:
                        logger.info(f"Auto-detected benchmark: {benchmark}")
                    break
            except Exception as e:
                logger.debug(f"Failed to load benchmark {bm}: {e}")
                continue

        if benchmark is None:
            logger.warning(
                "Could not find a valid benchmark in the data. "
                "Qlib requires a benchmark for backtesting. "
                "Please provide a valid benchmark code using --benchmark option."
            )
    else:
        # Validate provided benchmark (should already be normalized)
        logger.debug(f"Validating benchmark: {benchmark}")
        try:
            test_data = D.features([benchmark], ["$close"], start_time=start_date, end_time=end_date, freq="day")
            if test_data.empty:
                logger.warning(f"Benchmark {benchmark} has no data in the date range. Running without benchmark.")
                benchmark = None
            else:
                logger.info(f"Validated benchmark: {benchmark} (has data)")
        except Exception as e:
            logger.warning(f"Failed to validate benchmark {benchmark}: {e}. Running without benchmark.")
            benchmark = None

    # Create backtest configuration
    backtest_config = {
        "start_time": start_date,
        "end_time": end_date,
        "account": initial_cash,
        "exchange_kwargs": {
            "freq": "day",
            "limit_threshold": 0.095,
            "deal_price": "close",
            "open_cost": 0.0015,
            "close_cost": 0.0015,
            "min_cost": 5,
        },
    }

    # Handle benchmark configuration
    # IMPORTANT: Qlib appears to require a benchmark. If user wants to disable it,
    # we need to find a valid benchmark code first, then handle it in error recovery.
    # If benchmark is None and skip_auto_detect is True, we'll try without benchmark first,
    # but will automatically find one if Qlib requires it (handled in exception handler).
    
    if benchmark is not None and benchmark.strip() != "":
        # Final check: ensure benchmark is in correct format
        final_benchmark = normalize_index_code(benchmark)
        if final_benchmark != benchmark:
            logger.warning(f"Benchmark format corrected: {benchmark} -> {final_benchmark}")
            benchmark = final_benchmark
        
        # Ensure benchmark is in correct format before passing to Qlib
        backtest_config["benchmark"] = benchmark
        logger.info(f"Final benchmark for backtest: {benchmark} (Qlib format: code.EXCHANGE)")
    else:
        # User wants to disable benchmark, but Qlib may require it
        # We'll try without benchmark first, and if it fails, automatically find one
        logger.info("Attempting to run without benchmark (Qlib may require one, will auto-detect if needed)")
        # Do NOT add benchmark key - let Qlib error handler find one if needed

    # Create executor config
    exec_config = {
        "time_per_step": "day",
        "generate_portfolio_metrics": True,
        "verbose": True,
        "track_data": True,
    }

    # Run backtest
    logger.info("Executing backtest...")
    logger.debug(f"Backtest config keys: {list(backtest_config.keys())}")
    if "benchmark" in backtest_config:
        logger.debug(f"Benchmark in config: {backtest_config['benchmark']}")
    else:
        logger.debug("No benchmark key in config")
    
    # Create executor instance to access trade history after backtest
    exec_instance = executor.SimulatorExecutor(**exec_config)
    
    try:
        portfolio_metric, indicator = backtest(
            executor=exec_instance,
            strategy=strategy,
            **backtest_config,
        )
    except ValueError as e:
        # If error is about benchmark, automatically find and use a valid one
        error_msg = str(e)
        
        # Check if this is a benchmark-related error
        # Qlib may show "The benchmark []" or "The benchmark ['SH000300']" etc.
        is_benchmark_error = (
            "benchmark" in error_msg.lower() and 
            ("does not exist" in error_msg.lower() or "provide the right benchmark" in error_msg.lower())
        )
        
        if is_benchmark_error:
            logger.warning(
                f"Qlib requires a valid benchmark. "
                f"Error: {error_msg}. "
                f"Automatically finding a valid benchmark..."
            )
            # Try to find a valid benchmark (in Qlib format: code.EXCHANGE)
            common_benchmarks = ["000300.SH", "399001.SZ", "000001.SH", "000905.SH"]
            found_benchmark = None
            for bm in common_benchmarks:
                try:
                    test_data = D.features([bm], ["$close"], start_time=start_date, end_time=end_date, freq="day")
                    if not test_data.empty:
                        found_benchmark = bm
                        logger.info(f"✓ Found valid benchmark: {found_benchmark}. Using it for backtest.")
                        break
                except Exception as ex:
                    logger.debug(f"Failed to load benchmark {bm}: {ex}")
                    continue
            
            if found_benchmark:
                # Retry with found benchmark
                backtest_config["benchmark"] = found_benchmark
                logger.info("Retrying backtest with auto-detected benchmark...")
                try:
                    portfolio_metric, indicator = backtest(
                        executor=executor.SimulatorExecutor(**exec_config),
                        strategy=strategy,
                        **backtest_config,
                    )
                    logger.info("✓ Backtest completed successfully with auto-detected benchmark")
                except Exception as retry_error:
                    logger.error(f"Backtest still failed after adding benchmark: {retry_error}")
                    raise
            else:
                logger.error(
                    f"❌ Could not find a valid benchmark in the data. "
                    f"Qlib requires a benchmark for backtesting. "
                    f"Please:\n"
                    f"  1. Provide a valid benchmark code using --benchmark option (e.g., --benchmark 000300.SH)\n"
                    f"  2. Or ensure your Qlib data contains benchmark index data\n"
                    f"  3. Original error: {error_msg}"
                )
                raise
        else:
            # Re-raise if it's not a benchmark-related error
            raise
    except Exception as e:
        logger.error(f"Backtest failed: {e}", exc_info=True)
        raise

    # Extract results
    results = {
        "portfolio_metric": portfolio_metric,
        "indicator": indicator,
        "executor": exec_instance,  # Include executor to access trade history if available
    }

    logger.info("Backtest completed successfully")
    return results


class Position:
    """Encapsulates position information from backtest results."""

    def __init__(self, data):
        """
        Initialize Position from dict or object.

        Args:
            data: Position data (dict or object with position attribute).
        """
        self._position_dict = self._normalize_to_dict(data)

    def _normalize_to_dict(self, data) -> dict:
        """Normalize position data to dict format."""
        if data is None:
            return {}

        # If it's already a dict
        if isinstance(data, dict):
            if 'position' in data:
                position = data['position']
            else:
                position = data
        # If it's an object
        elif hasattr(data, 'position'):
            position = data.position
        else:
            position = data

        # Convert position to dict if needed
        if isinstance(position, dict):
            return position
        elif hasattr(position, '__dict__'):
            return position.__dict__
        elif hasattr(position, 'to_dict'):
            return position.to_dict()
        else:
            return {}

    def get_holdings(self) -> Dict[str, tuple]:
        """
        Extract current holdings from position.

        Returns:
            Dict mapping stock code to (price, amount) tuple.
        """
        holdings = {}
        for key, value in self._position_dict.items():
            if key in ['cash', 'now_account_value']:
                continue
            if isinstance(value, dict) and 'price' in value and 'amount' in value:
                price = float(value['price'])
                amount = float(value['amount'])
                if amount > 0:
                    holdings[key] = (price, amount)
        return holdings

    def get_stock_price(self, stock: str) -> Optional[float]:
        """Get price for a specific stock."""
        if stock in self._position_dict:
            stock_info = self._position_dict[stock]
            if isinstance(stock_info, dict) and 'price' in stock_info:
                return float(stock_info['price'])
        return None


class PositionDetails:
    """Encapsulates position details from backtest results."""

    def __init__(self, data: Optional[Dict]):
        """
        Initialize PositionDetails from dict.

        Args:
            data: Position details dict mapping dates to position info.
        """
        self._data = data or {}

    def get_dates(self) -> List[pd.Timestamp]:
        """Get sorted list of dates."""
        return sorted([pd.Timestamp(d) for d in self._data.keys() if isinstance(d, pd.Timestamp)])

    def get_position(self, date: pd.Timestamp) -> Optional[Position]:
        """Get Position object for a specific date."""
        if date not in self._data:
            return None
        return Position(self._data[date])


class PortfolioMetrics:
    """Encapsulates portfolio metrics from backtest results."""

    def __init__(self, data):
        """
        Initialize PortfolioMetrics from various formats.

        Args:
            data: Portfolio metrics (dict, DataFrame, or None).
        """
        self._raw_data = data
        self._metric_df = self._extract_dataframe()
        self._position_details = self._extract_position_details()

    def _extract_dataframe(self) -> Optional[pd.DataFrame]:
        """Extract DataFrame from various formats."""
        if self._raw_data is None:
            return None

        # If it's already a DataFrame
        if isinstance(self._raw_data, pd.DataFrame):
            return self._raw_data

        # If it's a dict, try to extract DataFrame
        if isinstance(self._raw_data, dict):
            for key, value in self._raw_data.items():
                if isinstance(value, tuple) and len(value) >= 1:
                    if isinstance(value[0], pd.DataFrame):
                        return value[0]
                elif isinstance(value, pd.DataFrame):
                    return value

        return None

    def _extract_position_details(self) -> Optional[PositionDetails]:
        """Extract position details from portfolio metrics."""
        if not isinstance(self._raw_data, dict):
            return None

        for key, value in self._raw_data.items():
            if isinstance(value, tuple) and len(value) >= 2:
                if isinstance(value[1], dict):
                    return PositionDetails(value[1])

        return None

    def has_data(self) -> bool:
        """Check if metrics data is available."""
        return self._metric_df is not None and len(self._metric_df) > 0

    def get_dataframe(self) -> Optional[pd.DataFrame]:
        """Get the metrics DataFrame."""
        return self._metric_df

    def get_position_details(self) -> Optional[PositionDetails]:
        """Get position details."""
        return self._position_details


class PerformanceMetrics:
    """Calculates and stores performance metrics."""

    def __init__(self, metric_df: pd.DataFrame):
        """
        Initialize PerformanceMetrics from DataFrame.

        Args:
            metric_df: Metrics DataFrame with time series data.
        """
        self._df = metric_df
        self._final_row = metric_df.iloc[-1]
        self._first_row = metric_df.iloc[0]

    @property
    def total_return(self) -> Optional[float]:
        """Calculate total return."""
        total_return = self._final_row.get('return', None)
        if total_return is None or pd.isna(total_return):
            initial_account = self._first_row.get('account', None)
            final_account = self._final_row.get('account', None)
            if initial_account is not None and final_account is not None and initial_account > 0:
                total_return = (final_account - initial_account) / initial_account
            else:
                total_return = None
        return total_return

    @property
    def annualized_return(self) -> Optional[float]:
        """Calculate annualized return."""
        if self.total_return is None or pd.isna(self.total_return):
            return None
        num_days = len(self._df)
        if num_days == 0:
            return None
        years = num_days / 252.0
        if years > 0:
            return (1 + self.total_return) ** (1.0 / years) - 1
        return None

    @property
    def volatility(self) -> Optional[float]:
        """Calculate annualized volatility."""
        if 'return' not in self._df.columns:
            return None
        returns = self._df['return'].dropna()
        if len(returns) <= 1:
            return None
        daily_returns = returns.diff().dropna()
        if len(daily_returns) == 0:
            return None
        return daily_returns.std() * (252 ** 0.5)

    @property
    def sharpe_ratio(self) -> Optional[float]:
        """Calculate Sharpe ratio."""
        if self.annualized_return is None or self.volatility is None or self.volatility <= 0:
            return None
        # Assume risk-free rate is 0
        return self.annualized_return / self.volatility

    @property
    def max_drawdown(self) -> Optional[float]:
        """Calculate maximum drawdown."""
        if 'account' not in self._df.columns:
            return None
        account_values = self._df['account'].dropna()
        if len(account_values) == 0:
            return None
        running_max = account_values.expanding().max()
        drawdown = (account_values - running_max) / running_max
        return drawdown.min()

    @property
    def initial_account(self) -> Optional[float]:
        """Get initial account value."""
        if 'account' in self._df.columns:
            return self._df['account'].iloc[0]
        return None

    @property
    def final_account(self) -> Optional[float]:
        """Get final account value."""
        if 'account' in self._df.columns:
            return self._df['account'].iloc[-1]
        return None

    @property
    def num_trading_days(self) -> int:
        """Get number of trading days."""
        return len(self._df)

    @property
    def first_date(self):
        """Get first date."""
        if len(self._df) > 0:
            return self._df.index[0]
        return None

    @property
    def last_date(self):
        """Get last date."""
        if len(self._df) > 0:
            return self._df.index[-1]
        return None

    @property
    def total_turnover(self) -> Optional[float]:
        """Get total turnover."""
        if 'total_turnover' in self._df.columns:
            return self._df['total_turnover'].iloc[-1]
        return None

    @property
    def avg_daily_turnover(self) -> Optional[float]:
        """Get average daily turnover."""
        if 'turnover' in self._df.columns:
            return self._df['turnover'].mean()
        return None


class Trade:
    """Represents a single trade."""

    def __init__(
        self,
        buy_date: pd.Timestamp,
        sell_date: pd.Timestamp,
        stock: str,
        buy_price: float,
        sell_price: float,
        pnl: float,
        pnl_pct: float,
    ):
        self.buy_date = buy_date
        self.sell_date = sell_date
        self.stock = stock
        self.buy_price = buy_price
        self.sell_price = sell_price
        self.pnl = pnl
        self.pnl_pct = pnl_pct

    @property
    def holding_period_days(self) -> int:
        """Get holding period in days."""
        return (self.sell_date - self.buy_date).days

    @property
    def is_winning(self) -> bool:
        """Check if trade is winning."""
        return self.pnl > 0


class TradingStatistics:
    """Calculates trading statistics from position details."""

    def __init__(self, position_details: PositionDetails):
        """
        Initialize TradingStatistics from position details.

        Args:
            position_details: PositionDetails object.
        """
        self._trades = self._extract_trades(position_details)

    def _extract_trades(self, position_details: PositionDetails) -> List[Trade]:
        """Extract trades from position details."""
        if position_details is None:
            return []

        trades = []
        holdings = {}  # {stock: (buy_date, buy_price, amount)}
        dates = position_details.get_dates()

        for i, date in enumerate(dates):
            current_position = position_details.get_position(date)
            if current_position is None:
                continue

            current_holdings = current_position.get_holdings()

            if i > 0:
                prev_date = dates[i - 1]
                prev_position = position_details.get_position(prev_date)
                prev_holdings = prev_position.get_holdings() if prev_position else {}

                # Find new positions (bought)
                for stock in current_holdings:
                    if stock not in prev_holdings or prev_holdings[stock][1] == 0:
                        price, amount = current_holdings[stock]
                        holdings[stock] = (date, price, amount)

                # Find closed positions (sold)
                for stock in list(holdings.keys()):
                    buy_date, buy_price, buy_amount = holdings[stock]
                    if stock not in current_holdings:
                        # Completely sold
                        sell_price = prev_position.get_stock_price(stock) if prev_position else buy_price
                        if sell_price is None:
                            sell_price = buy_price
                        pnl = (sell_price - buy_price) * buy_amount
                        pnl_pct = (sell_price - buy_price) / buy_price if buy_price > 0 else 0
                        trades.append(Trade(buy_date, date, stock, buy_price, sell_price, pnl, pnl_pct))
                        del holdings[stock]
                    elif current_holdings[stock][1] < buy_amount:
                        # Partially sold
                        sell_price = current_holdings[stock][0]
                        pnl = (sell_price - buy_price) * buy_amount
                        pnl_pct = (sell_price - buy_price) / buy_price if buy_price > 0 else 0
                        trades.append(Trade(buy_date, date, stock, buy_price, sell_price, pnl, pnl_pct))
                        remaining_amount = current_holdings[stock][1]
                        if remaining_amount > 0:
                            holdings[stock] = (date, sell_price, remaining_amount)
                        else:
                            del holdings[stock]
            else:
                # First day - all are new buys
                for stock, (price, amount) in current_holdings.items():
                    holdings[stock] = (date, price, amount)

        return trades

    @property
    def total_trades(self) -> int:
        """Get total number of trades."""
        return len(self._trades)

    @property
    def winning_trades(self) -> List[Trade]:
        """Get list of winning trades."""
        return [t for t in self._trades if t.is_winning]

    @property
    def losing_trades(self) -> List[Trade]:
        """Get list of losing trades."""
        return [t for t in self._trades if not t.is_winning]

    @property
    def win_rate(self) -> float:
        """Calculate win rate percentage."""
        if self.total_trades == 0:
            return 0.0
        return len(self.winning_trades) / self.total_trades * 100

    @property
    def avg_holding_days(self) -> float:
        """Calculate average holding period in days."""
        if self.total_trades == 0:
            return 0.0
        holding_periods = [t.holding_period_days for t in self._trades]
        return sum(holding_periods) / len(holding_periods)

    @property
    def avg_return_per_trade(self) -> float:
        """Calculate average return per trade percentage."""
        if self.total_trades == 0:
            return 0.0
        return sum([t.pnl_pct for t in self._trades]) / self.total_trades * 100

    @property
    def profit_factor(self) -> float:
        """Calculate profit factor."""
        total_profit = sum([t.pnl for t in self.winning_trades])
        total_loss = abs(sum([t.pnl for t in self.losing_trades]))
        if total_loss > 0:
            return total_profit / total_loss
        elif total_profit > 0:
            return float('inf')
        else:
            return 0.0

    @property
    def avg_win_loss_ratio(self) -> float:
        """Calculate average win/loss ratio."""
        if len(self.winning_trades) == 0 or len(self.losing_trades) == 0:
            return 0.0
        avg_win = sum([t.pnl for t in self.winning_trades]) / len(self.winning_trades)
        avg_loss = abs(sum([t.pnl for t in self.losing_trades]) / len(self.losing_trades))
        if avg_loss != 0:
            return abs(avg_win / avg_loss)
        elif avg_win > 0:
            return float('inf')
        else:
            return 0.0

    @property
    def total_profit(self) -> float:
        """Calculate total profit."""
        return sum([t.pnl for t in self.winning_trades])

    @property
    def total_loss(self) -> float:
        """Calculate total loss."""
        return abs(sum([t.pnl for t in self.losing_trades]))


class Indicators:
    """Encapsulates performance indicators."""

    def __init__(self, data):
        """
        Initialize Indicators from various formats.

        Args:
            data: Indicators data (dict or None).
        """
        self._data = data or {}

    def is_dict(self) -> bool:
        """Check if indicators is a dict."""
        return isinstance(self._data, dict)

    def items(self):
        """Get items iterator."""
        if self.is_dict():
            return self._data.items()
        return []


def print_results(results: dict) -> None:
    """Print backtest results in a readable format."""
    portfolio_metrics = PortfolioMetrics(results.get("portfolio_metric"))
    indicators = Indicators(results.get("indicator"))

    print("\n" + "=" * 80)
    print("Backtest Results")
    print("=" * 80)

    # Portfolio metrics
    if portfolio_metrics.has_data():
        print("\n📊 Portfolio Metrics:")
        
        metric_df = portfolio_metrics.get_dataframe()
        if metric_df is not None:
            perf_metrics = PerformanceMetrics(metric_df)
            
            # Display metrics
            total_return = perf_metrics.total_return
            print(f"  Total Return: {total_return * 100:.2f}%" if total_return is not None and not pd.isna(total_return) else "  Total Return: N/A")
            
            annualized_return = perf_metrics.annualized_return
            print(f"  Annualized Return: {annualized_return * 100:.2f}%" if annualized_return is not None and not pd.isna(annualized_return) else "  Annualized Return: N/A")
            
            volatility = perf_metrics.volatility
            print(f"  Volatility: {volatility * 100:.2f}%" if volatility is not None and not pd.isna(volatility) else "  Volatility: N/A")
            
            sharpe = perf_metrics.sharpe_ratio
            print(f"  Sharpe Ratio: {sharpe:.4f}" if sharpe is not None and not pd.isna(sharpe) else "  Sharpe Ratio: N/A")
            
            max_drawdown = perf_metrics.max_drawdown
            print(f"  Max Drawdown: {max_drawdown * 100:.2f}%" if max_drawdown is not None and not pd.isna(max_drawdown) else "  Max Drawdown: N/A")
            
            # Show additional info
            initial_account = perf_metrics.initial_account
            final_account = perf_metrics.final_account
            if initial_account is not None and final_account is not None:
                print(f"\n  Initial Account Value: {initial_account:,.2f}")
                print(f"  Final Account Value: {final_account:,.2f}")
                print(f"  Net P&L: {final_account - initial_account:,.2f}")
            
            if perf_metrics.num_trading_days > 1:
                print(f"\n  Time Series: {perf_metrics.num_trading_days} trading days")
                print(f"    First date: {perf_metrics.first_date}")
                print(f"    Last date: {perf_metrics.last_date}")
            
            # Extract trading statistics from position details
            position_details = portfolio_metrics.get_position_details()
            
            if position_details:
                trading_stats = TradingStatistics(position_details)
                
                if trading_stats.total_trades > 0:
                    print(f"\n📈 Trading Statistics:")
                    print(f"  Total Trades: {trading_stats.total_trades}")
                    print(f"  Winning Trades: {len(trading_stats.winning_trades)}")
                    print(f"  Losing Trades: {len(trading_stats.losing_trades)}")
                    print(f"  Win Rate: {trading_stats.win_rate:.2f}%")
                    print(f"  Average Holding Period: {trading_stats.avg_holding_days:.1f} days")
                    print(f"  Average Return per Trade: {trading_stats.avg_return_per_trade:.2f}%")
                    profit_factor = trading_stats.profit_factor
                    print(f"  Profit Factor: {profit_factor:.2f}" if profit_factor != float('inf') else "  Profit Factor: ∞")
                    avg_win_loss = trading_stats.avg_win_loss_ratio
                    print(f"  Avg Win / Avg Loss: {avg_win_loss:.2f}" if avg_win_loss != float('inf') else "  Avg Win / Avg Loss: ∞")
                    print(f"  Total Profit: {trading_stats.total_profit:,.2f}")
                    print(f"  Total Loss: {trading_stats.total_loss:,.2f}")
            
            # Show turnover statistics
            total_turnover = perf_metrics.total_turnover
            avg_daily_turnover = perf_metrics.avg_daily_turnover
            if total_turnover is not None:
                print(f"\n💰 Turnover Statistics:")
                print(f"  Total Turnover: {total_turnover:,.2f}")
                if avg_daily_turnover is not None:
                    print(f"  Average Daily Turnover: {avg_daily_turnover:.2%}")
        else:
            print("  ⚠ Could not extract DataFrame from portfolio_metric")
    else:
        print("\n📊 Portfolio Metrics:")
        print("  ⚠ Portfolio metrics is None")
        print("  This may indicate:")
        print("    1. generate_portfolio_metrics was not enabled in executor config")
        print("    2. Backtest period is too short to calculate metrics")
        print("    3. Qlib backtest function did not return portfolio metrics")

    # Indicators
    if indicators.is_dict():
        print("\n📈 Performance Indicators:")
        for key, value in indicators.items():
            if isinstance(value, (int, float)):
                print(f"  {key}: {value:.4f}")
            elif isinstance(value, pd.DataFrame):
                print(f"  {key}: (DataFrame with shape {value.shape})")
                if len(value) <= 10:
                    print(f"    {value}")
            else:
                print(f"  {key}: {value}")

    print("\n" + "=" * 80)


def plot_results(results: dict, output_path: Optional[str] = None) -> None:
    """
    Plot backtest results visualization.
    
    Args:
        results: Backtest results dictionary containing portfolio_metric and indicator.
        output_path: Optional path to save the plot. If None, saves to 'backtest_results.png'.
    """
    if not HAS_MATPLOTLIB:
        logger.warning("matplotlib not available. Skipping visualization.")
        return
    
    portfolio_metric = results.get("portfolio_metric")
    
    if portfolio_metric is None:
        logger.warning("No portfolio_metric available for plotting.")
        return
    
    # Extract DataFrame from portfolio_metric
    metric_df = None
    if isinstance(portfolio_metric, dict):
        for key, value in portfolio_metric.items():
            if isinstance(value, tuple) and len(value) >= 1:
                if isinstance(value[0], pd.DataFrame):
                    metric_df = value[0]
                    break
            elif isinstance(value, pd.DataFrame):
                metric_df = value
                break
    elif isinstance(portfolio_metric, pd.DataFrame):
        metric_df = portfolio_metric
    
    if metric_df is None or len(metric_df) == 0:
        logger.warning("No valid DataFrame found in portfolio_metric for plotting.")
        return
    
    # Set up Chinese font support (if available)
    try:
        # Try to find a Chinese font
        chinese_fonts = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei', 'STHeiti']
        font_found = False
        for font_name in chinese_fonts:
            try:
                plt.rcParams['font.sans-serif'] = [font_name]
                plt.rcParams['axes.unicode_minus'] = False
                font_found = True
                break
            except:
                continue
        if not font_found:
            # Use default font
            plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    except:
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # Convert index to datetime if needed
    if not isinstance(metric_df.index, pd.DatetimeIndex):
        try:
            metric_df.index = pd.to_datetime(metric_df.index)
        except:
            pass
    
    dates = metric_df.index
    
    # 1. Account Value Over Time
    ax1 = fig.add_subplot(gs[0, :])
    if 'account' in metric_df.columns:
        account_values = metric_df['account']
        ax1.plot(dates, account_values, linewidth=2, label='Account Value', color='#2E86AB')
        ax1.axhline(y=account_values.iloc[0], color='gray', linestyle='--', alpha=0.5, label='Initial Value')
        ax1.set_title('Account Value Over Time', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Account Value (CNY)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax1.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(dates)//10)))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 2. Cumulative Return
    ax2 = fig.add_subplot(gs[1, 0])
    if 'return' in metric_df.columns:
        returns = metric_df['return'] * 100  # Convert to percentage
        ax2.plot(dates, returns, linewidth=2, label='Cumulative Return', color='#A23B72')
        ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax2.set_title('Cumulative Return (%)', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Return (%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 3. Drawdown
    ax3 = fig.add_subplot(gs[1, 1])
    if 'account' in metric_df.columns:
        account_values = metric_df['account']
        running_max = account_values.expanding().max()
        drawdown = (account_values - running_max) / running_max * 100
        ax3.fill_between(dates, drawdown, 0, alpha=0.3, color='#F18F01', label='Drawdown')
        ax3.plot(dates, drawdown, linewidth=1.5, color='#F18F01')
        ax3.set_title('Drawdown (%)', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Date')
        ax3.set_ylabel('Drawdown (%)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 4. Daily Returns
    ax4 = fig.add_subplot(gs[2, 0])
    if 'return' in metric_df.columns:
        returns = metric_df['return']
        daily_returns = returns.diff().dropna() * 100  # Convert to percentage
        if len(daily_returns) > 0:
            colors = ['#06A77D' if x >= 0 else '#D00000' for x in daily_returns]
            ax4.bar(daily_returns.index, daily_returns.values, color=colors, alpha=0.7, width=0.8)
            ax4.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
            ax4.set_title('Daily Returns (%)', fontsize=12, fontweight='bold')
            ax4.set_xlabel('Date')
            ax4.set_ylabel('Daily Return (%)')
            ax4.grid(True, alpha=0.3, axis='y')
            ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 5. Turnover
    ax5 = fig.add_subplot(gs[2, 1])
    if 'turnover' in metric_df.columns:
        turnover = metric_df['turnover'] * 100  # Convert to percentage
        ax5.plot(dates, turnover, linewidth=2, label='Daily Turnover', color='#6A4C93', marker='o', markersize=3)
        ax5.set_title('Daily Turnover (%)', fontsize=12, fontweight='bold')
        ax5.set_xlabel('Date')
        ax5.set_ylabel('Turnover (%)')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        ax5.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.setp(ax5.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Add overall title
    fig.suptitle('Backtest Performance Analysis', fontsize=16, fontweight='bold', y=0.995)
    
    # Save or show
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Plot saved to {output_path}")
    else:
        plt.savefig('backtest_results.png', dpi=300, bbox_inches='tight')
        logger.info("Plot saved to backtest_results.png")
    
    plt.close()


def save_results(results: dict, output_path: str) -> None:
    """Save backtest results to file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    portfolio_metric = results["portfolio_metric"]
    indicator = results["indicator"]

    # Combine all metrics
    all_metrics = {}
    if portfolio_metric is not None:
        all_metrics.update(portfolio_metric)
    if indicator is not None:
        all_metrics.update(indicator)

    # Save to CSV
    df = pd.DataFrame([all_metrics])
    df.to_csv(output_path, index=False)
    logger.info(f"Results saved to {output_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Backtest Structure Expert GNN model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Required arguments
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to trained model file (.pth)",
    )
    parser.add_argument(
        "--start_date",
        type=str,
        required=True,
        help="Start date of backtest period (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end_date",
        type=str,
        required=True,
        help="End date of backtest period (YYYY-MM-DD)",
    )

    # Optional arguments
    parser.add_argument(
        "--top_k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"Number of top stocks to select (default: {DEFAULT_TOP_K})",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="TopkDropoutStrategy",
        help="Portfolio strategy class name (default: TopkDropoutStrategy). "
        "Options: 'TopkDropoutStrategy', 'RefinedTopKStrategy', 'SimpleLowTurnoverStrategy'.",
    )
    parser.add_argument(
        "--buffer_ratio",
        type=float,
        default=None,
        help="Buffer ratio for RefinedTopKStrategy (default: 0.15). "
        "Higher value reduces turnover but may reduce returns. "
        "Only used with RefinedTopKStrategy.",
    )
    parser.add_argument(
        "--retain_threshold",
        type=int,
        default=None,
        help="Retain threshold for SimpleLowTurnoverStrategy (default: 30). "
        "Only sell existing holdings if they fall out of top N. "
        "Only used with SimpleLowTurnoverStrategy.",
    )
    parser.add_argument(
        "--initial_cash",
        type=float,
        default=DEFAULT_INITIAL_CASH,
        help=f"Initial cash amount (default: {DEFAULT_INITIAL_CASH})",
    )
    parser.add_argument(
        "--save_results",
        action="store_true",
        help="Save backtest results to file",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate visualization plots for backtest results",
    )
    parser.add_argument(
        "--plot_output",
        type=str,
        default=None,
        help="Path to save the plot image (default: outputs/structure_expert_backtest_<dates>.png)",
    )
    parser.add_argument(
        "--save_embeddings",
        action="store_true",
        help="Save model embeddings and scores for visualization dashboard",
    )
    parser.add_argument(
        "--embeddings_storage_dir",
        type=str,
        default=None,
        help="Directory to save embeddings (default: storage/structure_expert_cache)",
    )
    parser.add_argument(
        "--qlib_dir",
        type=str,
        default=DEFAULT_QLIB_DIR,
        help=f"Qlib data directory (default: {DEFAULT_QLIB_DIR})",
    )
    parser.add_argument(
        "--region",
        type=str,
        default="cn",
        help="Qlib region (default: cn)",
    )
    parser.add_argument(
        "--n_feat",
        type=int,
        default=DEFAULT_N_FEAT,
        help=f"Number of input features (default: {DEFAULT_N_FEAT})",
    )
    parser.add_argument(
        "--n_hidden",
        type=int,
        default=DEFAULT_N_HIDDEN,
        help=f"Hidden layer size (default: {DEFAULT_N_HIDDEN})",
    )
    parser.add_argument(
        "--n_heads",
        type=int,
        default=DEFAULT_N_HEADS,
        help=f"Number of attention heads (default: {DEFAULT_N_HEADS})",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (default: cuda if available, else cpu)",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default=None,
        help="Path to config file (for database config)",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default=None,
        help="Benchmark code (e.g., 'SH000300'). If not provided, will try to auto-detect.",
    )
    parser.add_argument(
        "--no_benchmark",
        action="store_true",
        help="Explicitly disable benchmark comparison",
    )
    parser.add_argument(
        "--enable_eidos",
        action="store_true",
        help="Enable Eidos integration to save backtest results to database",
    )
    parser.add_argument(
        "--eidos_exp_name",
        type=str,
        default=None,
        help="Eidos experiment name (default: auto-generated from parameters)",
    )

    args = parser.parse_args()

    # Handle benchmark
    skip_auto_detect = args.no_benchmark
    if args.no_benchmark:
        benchmark = None
        logger.info("Benchmark comparison disabled by --no_benchmark flag")
    else:
        benchmark = args.benchmark
        # Normalize benchmark format early if provided
        if benchmark is not None and benchmark.strip() != "":
            benchmark = normalize_index_code(benchmark)
            logger.debug(f"Normalized benchmark code in main: {benchmark}")

    # Validate dates
    start_dt = datetime.strptime(args.start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(args.end_date, "%Y-%m-%d")
    today = datetime.now()
    
    if start_dt >= end_dt:
        logger.error(f"Start date ({args.start_date}) must be before end date ({args.end_date})")
        return 1
    
    # Check for future dates - warn but don't error (data might exist in test/simulated scenarios)
    if start_dt > today or end_dt > today:
        logger.warning(
            f"⚠️  Warning: Date range contains future dates.\n"
            f"  Start date: {args.start_date} {'(FUTURE)' if start_dt > today else ''}\n"
            f"  End date: {args.end_date} {'(FUTURE)' if end_dt > today else ''}\n"
            f"  Today: {today.strftime('%Y-%m-%d')}\n"
            f"Proceeding anyway - Qlib data may contain future dates (test/simulated data)."
        )

    # Initialize Qlib
    qlib_dir = Path(args.qlib_dir).expanduser()
    logger.info(f"Initializing Qlib with data directory: {qlib_dir}")
    qlib.init(provider_uri=str(qlib_dir), region=args.region)
    
    # Check Qlib data availability and date range
    try:
        full_calendar = D.calendar()
        if len(full_calendar) == 0:
            logger.error("❌ No trading days found in Qlib data. Please check your Qlib data installation.")
            return 1
        
        data_start = full_calendar[0].date()
        data_end = full_calendar[-1].date()
        
        logger.info(f"Qlib data date range: {data_start} to {data_end} ({len(full_calendar)} trading days)")
        
        # Check if requested dates are within data range
        if start_dt < data_start:
            logger.warning(
                f"⚠️  Start date {args.start_date} is before Qlib data start date ({data_start}). "
                f"Adjusting to {data_start}."
            )
            args.start_date = data_start.strftime("%Y-%m-%d")
            start_dt = data_start
        
        if end_dt > data_end:
            logger.warning(
                f"⚠️  End date {args.end_date} is after Qlib data end date ({data_end}). "
                f"Adjusting to {data_end}."
            )
            args.end_date = data_end.strftime("%Y-%m-%d")
            end_dt = data_end
        
        # Check if date range (with lookback) is available
        LOOKBACK_DAYS = 60  # Same as in generate_predictions
        lookback_start = start_dt - timedelta(days=LOOKBACK_DAYS)
        if lookback_start < data_start:
            logger.warning(
                f"⚠️  Start date {args.start_date} requires lookback to {lookback_start}, "
                f"but Qlib data only starts from {data_start}. "
                f"Some dates may not have enough historical data for Alpha158 features."
            )
        
        # Verify that requested date range has data
        requested_calendar = D.calendar(start_time=args.start_date, end_time=args.end_date)
        if len(requested_calendar) == 0:
            logger.error(
                f"❌ No trading days found in Qlib data for date range {args.start_date} to {args.end_date}.\n"
                f"Available data range: {data_start} to {data_end}\n"
                f"Please adjust your date range to be within the available data."
            )
            return 1
        
        logger.info(f"✓ Found {len(requested_calendar)} trading days in requested range")
        
    except Exception as e:
        logger.warning(f"Could not verify Qlib data availability: {e}")
        logger.warning("Proceeding anyway, but data may not be available for requested dates.")

    # Load database config if provided
    if args.config_path:
        config = load_config(args.config_path)
        db_config = config.database
    else:
        # Try to load from default location
        try:
            config = load_config("config/config.yaml")
            db_config = config.database
        except Exception:
            logger.warning(
                "Could not load database config. "
                "Industry mapping will be limited. "
                "Provide --config_path to specify config file."
            )
            db_config = None

    try:
        # Load model
        model = load_model(
            model_path=args.model_path,
            n_feat=args.n_feat,
            n_hidden=args.n_hidden,
            n_heads=args.n_heads,
            device=args.device,
        )

        # Load industry mapping
        if db_config is not None:
            # Use end_date for industry mapping (most recent)
            end_dt = datetime.strptime(args.end_date, "%Y-%m-%d")
            industry_map = load_industry_map(db_config, target_date=end_dt)
            logger.info(f"Loaded industry mapping: {len(industry_map)} stocks with industry info")
            logger.info(
                f"Note: Stocks not in industry_map will still be processed, "
                f"but won't have industry-based graph edges"
            )
        else:
            logger.warning("No database config, using empty industry map")
            logger.warning("All stocks will be processed, but without industry-based graph edges")
            industry_map = {}

        # Create graph builder
        builder = GraphDataBuilder(industry_map)

        # Load industry labels if saving embeddings
        industry_label_map = {}
        if args.save_embeddings:
            try:
                if db_config:
                    end_dt = datetime.strptime(args.end_date, "%Y-%m-%d")
                    industry_label_map = load_industry_label_map(db_config, target_date=end_dt)
                    logger.info(f"Loaded industry labels: {len(industry_label_map)} stocks")
                else:
                    logger.warning("No database config for industry labels, using empty map")
            except Exception as e:
                logger.warning(f"Failed to load industry labels: {e}")

        # Generate predictions
        predictions = generate_predictions(
            model=model,
            builder=builder,
            start_date=args.start_date,
            end_date=args.end_date,
            device=args.device,
            save_embeddings=args.save_embeddings,
            embeddings_storage_dir=args.embeddings_storage_dir,
            industry_label_map=industry_label_map,
            qlib_dir=str(qlib_dir),
        )

        # Determine strategy class
        strategy_class_name = args.strategy
        if strategy_class_name == "RefinedTopKStrategy":
            strategy_class = RefinedTopKStrategy
        elif strategy_class_name == "SimpleLowTurnoverStrategy":
            strategy_class = SimpleLowTurnoverStrategy
        elif strategy_class_name == "TopkDropoutStrategy":
            strategy_class = TopkDropoutStrategy
        else:
            logger.warning(f"Unknown strategy class: {strategy_class_name}, using TopkDropoutStrategy")
            strategy_class = TopkDropoutStrategy

        # Create portfolio strategy
        strategy = create_portfolio_strategy(
            predictions=predictions,
            strategy_class=strategy_class,
            top_k=args.top_k,
            buffer_ratio=args.buffer_ratio,
            retain_threshold=args.retain_threshold,
        )

        # Run backtest
        results = run_backtest(
            strategy=strategy,
            start_date=args.start_date,
            end_date=args.end_date,
            initial_cash=args.initial_cash,
            benchmark=benchmark,
            skip_auto_detect=skip_auto_detect,
        )


        # Extract portfolio_metric and indicator from results
        portfolio_metric = results.get("portfolio_metric")
        indicator = results.get("indicator")

        # Eidos Integration
        if args.enable_eidos:
            try:
                if db_config is None:
                    logger.warning(
                        "Eidos enabled but no database config found. "
                        "Please provide --config_path to enable Eidos."
                    )
                else:
                    logger.info("Saving backtest results to Eidos...")
                    
                    # Create Eidos writer
                    writer = EidosBacktestWriter(db_config)
                    
                    # Generate experiment name if not provided
                    exp_name = args.eidos_exp_name
                    if exp_name is None:
                        exp_name = (
                            f"Structure Expert - {args.strategy} - "
                            f"TopK{args.top_k} - {args.start_date}_{args.end_date}"
                        )
                    
                    # Create experiment
                    from datetime import date as date_type
                    exp_id = writer.create_experiment_from_backtest(
                        name=exp_name,
                        start_date=date_type.fromisoformat(args.start_date),
                        end_date=date_type.fromisoformat(args.end_date),
                        config={
                            "topk": args.top_k,
                            "strategy": args.strategy,
                            "buffer_ratio": args.buffer_ratio,
                            "retain_threshold": args.retain_threshold,
                            "initial_cash": args.initial_cash,
                            "model_path": args.model_path,
                            "n_feat": args.n_feat,
                            "n_hidden": args.n_hidden,
                            "n_heads": args.n_heads,
                        },
                        model_type="GNN",
                        engine_type="Qlib",
                        strategy_name="Structure Expert",
                    )
                    
                    logger.info(f"Created Eidos experiment: {exp_id}")
                    
                    # Convert Qlib output to standard structure
                    qlib_result = QlibBacktestResult.from_qlib_dict_output(
                        portfolio_metric_dict=portfolio_metric,
                        indicator_dict=indicator,
                    )
                    
                    # Save backtest results
                    counts = save_structure_expert_backtest_to_eidos(
                        exp_id=exp_id,
                        writer=writer,
                        qlib_result=qlib_result,
                        predictions=predictions,
                        initial_cash=args.initial_cash,
                        builder=builder,
                        model=model,
                        device=torch.device(args.device),
                        embeddings_dict=None,  # TODO: Collect embeddings during prediction
                        strategy_instance=strategy,  # Pass strategy instance for order extraction
                    )
                    
                    # Calculate summary metrics for finalization
                    metrics_summary = {}
                    
                    # Use standard structure (reuse the one created above)
                    # qlib_result is already created above, no need to recreate
                    metric_df = qlib_result.portfolio_metrics.metric_df
                    
                    if metric_df is not None and not metric_df.empty:
                        if "return" in metric_df.columns:
                            returns = metric_df["return"]
                            metrics_summary["return"] = float(returns.mean())
                            metrics_summary["sharpe"] = (
                                float(returns.mean() / returns.std() * (252 ** 0.5))
                                if returns.std() > 0
                                else 0.0
                            )
                            # Calculate max drawdown
                            cum_returns = (1 + returns).cumprod()
                            running_max = cum_returns.expanding().max()
                            drawdown = (cum_returns - running_max) / running_max
                            metrics_summary["max_drawdown"] = float(drawdown.min())
                    
                    # Finalize experiment
                    writer.finalize_experiment(exp_id, metrics_summary=metrics_summary)
                    logger.info(f"✓ Eidos integration completed. Experiment ID: {exp_id}")
                    
            except Exception as e:
                logger.error(f"Failed to save to Eidos: {e}", exc_info=True)
                logger.warning("Continuing without Eidos integration...")

        # Print results
        # print_results(results)

        # Plot results if requested
        if args.plot:
            plot_output = args.plot_output if args.plot_output else f"outputs/structure_expert_backtest_{args.start_date}_{args.end_date}.png"
            plot_results(results, plot_output)

        # Save results if requested
        if args.save_results:
            output_path = f"outputs/structure_expert_backtest_{args.start_date}_{args.end_date}.csv"
            save_results(results, output_path)

    except Exception as e:
        logger.error(f"Backtest failed: {e}", exc_info=True)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())


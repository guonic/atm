"""
Backtest engine.

Main backtest loop that coordinates all components.
"""

import logging
from typing import Optional, List, Dict, Any

import pandas as pd
from qlib.data import D

from ..execution import Executor
from ..strategy import DualModelStrategy
from ..utils.data_validation import validate_and_filter_nan

logger = logging.getLogger(__name__)


def _get_position_price(symbol: str, date: pd.Timestamp, market_data: pd.DataFrame) -> float:
    """
    Get position price for a symbol on a specific date.
    
    Handles cases where the date was filtered out due to NaN values.
    In such cases, returns 0.0 (position value will be 0).
    
    Args:
        symbol: Stock symbol.
        date: Date to get price for.
        market_data: Market data DataFrame (may have filtered rows).
    
    Returns:
        Price if available, 0.0 otherwise.
    """
    try:
        if isinstance(market_data.index, pd.MultiIndex):
            symbol_data = market_data.xs(symbol, level=0)
            if date in symbol_data.index:
                price = symbol_data.loc[date, '$close']
                if pd.isna(price):
                    return 0.0
                return float(price)
            else:
                # Date was filtered out (NaN), return 0.0
                return 0.0
        else:
            if symbol in market_data.index:
                price = market_data.loc[symbol, '$close']
                if pd.isna(price):
                    return 0.0
                return float(price)
            else:
                return 0.0
    except (KeyError, IndexError) as e:
        # Date or symbol not found (likely filtered out)
        return 0.0


def run_custom_backtest(
    strategy: DualModelStrategy,
    start_date: str,
    end_date: str,
    initial_cash: float = 1000000.0,
    instruments: Optional[List[str]] = None,
    storage_backend: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Run backtest (completely independent from Qlib's backtest framework).
    
    Workflow:
    1. Use Qlib to load data
    2. Use custom state management (Account, Position)
    3. Use custom executor (Executor)
    4. Strategy generates orders, executor matches them
    
    Args:
        strategy: DualModelStrategy instance (must have order_book, position_manager initialized).
        start_date: Start date (YYYY-MM-DD).
        end_date: End date (YYYY-MM-DD).
        initial_cash: Initial cash amount (ignored if strategy already has account).
        instruments: Optional list of instruments. If None, uses all instruments.
        storage_backend: Optional storage backend. If None, uses strategy's storage.
    
    Returns:
        Dictionary containing backtest results:
        - account: Final account state
        - positions: Final positions
        - orders: All orders
        - snapshots: Account snapshots
    """
    # Use strategy's existing components (don't create new ones!)
    # This ensures orders submitted by strategy are visible to executor
    order_book = strategy.order_book
    position_manager = strategy.position_manager
    account = position_manager.account
    
    # Get storage from strategy's position_manager
    if storage_backend is None:
        storage_backend = position_manager.storage
    
    # Initialize executor (use strategy's order_book and position_manager)
    executor = Executor(position_manager, order_book)
    
    # Load trading calendar (use Qlib)
    calendar = D.calendar(start_time=start_date, end_time=end_date)
    if len(calendar) == 0:
        raise ValueError(f"No trading days found between {start_date} and {end_date}")
    
    logger.info(f"Found {len(calendar)} trading days to process")
    
    # Load instruments (use Qlib)
    # IMPORTANT: Load ALL instruments, not just a subset
    # The buy model (e.g., StructureExpertBuyModel) may generate rankings for stocks
    # that are not in the initial instruments list, so we need to load data for all stocks
    if instruments is None:
        instruments = D.instruments()
        logger.info(f"Using all instruments from Qlib: {len(instruments)} stocks")
    else:
        logger.info(f"Using provided instruments: {len(instruments)} stocks")
    
    # Load market data (use Qlib)
    logger.info("Loading market data from Qlib...")
    market_data = D.features(
        instruments=instruments,
        fields=["$close", "$open", "$high", "$low", "$volume"],
        start_time=start_date,
        end_time=end_date,
    )
    
    if market_data.empty:
        raise ValueError("No market data loaded from Qlib")
    
    # CRITICAL: Validate and filter NaN values before data enters the system
    # This prevents NaN from propagating and causing errors downstream
    required_fields = ["$close", "$open", "$high", "$low", "$volume"]
    market_data, nan_details = validate_and_filter_nan(
        market_data=market_data,
        required_fields=required_fields,
        context="backtest market data"
    )
    
    # Get actual instruments from loaded data (may differ from input if some stocks have no data)
    if isinstance(market_data.index, pd.MultiIndex):
        actual_instruments = market_data.index.get_level_values(0).unique().tolist()
    else:
        actual_instruments = market_data.index.unique().tolist()
    
    logger.info(f"Loaded market data for {len(actual_instruments)} instruments")
    
    # Backtest main loop
    snapshots = []
    for i, date in enumerate(calendar):
        date_ts = pd.Timestamp(date)
        date_str = date_ts.strftime("%Y-%m-%d")
        
        logger.info(f"Processing {date_str} ({i+1}/{len(calendar)})")
        
        # Get daily market data for strategy (filtered by date)
        if isinstance(market_data.index, pd.MultiIndex):
            # MultiIndex: (instrument, datetime)
            daily_data_for_strategy = market_data.loc[market_data.index.get_level_values(1) == date_ts]
        else:
            # Single index: use all data (assuming it's for current date)
            daily_data_for_strategy = market_data
        
        # Strategy generates orders
        strategy.on_bar(date_ts, daily_data_for_strategy)
        
        # Execute pending orders (use full market_data, executor will filter by symbol and date)
        pending_orders = order_book.get_pending_orders()
        total_orders = len(order_book.orders)
        if pending_orders:
            logger.info(
                f"Executing {len(pending_orders)} pending orders on {date_str} "
                f"(total orders in book: {total_orders})"
            )
        elif total_orders > 0:
            # Debug: check order statuses
            status_counts = {}
            for order in order_book.orders.values():
                status = order.status.value
                status_counts[status] = status_counts.get(status, 0) + 1
            logger.warning(
                f"No pending orders on {date_str}, but {total_orders} total orders exist. "
                f"Status breakdown: {status_counts}"
            )
        for order in pending_orders:
            fill_info = executor.execute_order(order, market_data, date_ts)
            if fill_info:
                logger.info(
                    f"Order filled: {fill_info.symbol} {fill_info.side.value} "
                    f"{fill_info.amount:.0f} @ {fill_info.price:.2f}"
                )
            else:
                logger.debug(f"Order not filled: {order.symbol} {order.side.value} (status: {order.status.value})")
        
        # Update positions (after market close)
        # Use full market_data for position updates
        position_manager.update_positions(date_ts, market_data)
        
        # Record account snapshot
        snapshot = {
            'date': date_str,
            'total_value': account.get_total_value(position_manager),
            'cash': account.available_cash,
            'frozen_cash': account.frozen_cash,
            'holdings_value': sum(
                pos.calculate_market_value(
                    _get_position_price(pos.symbol, date_ts, market_data)
                )
                for pos in position_manager.all_positions.values()
            ),
            'position_count': len(position_manager.all_positions),
        }
        storage_backend.save(f"snapshot:{date_str}", snapshot)
        snapshots.append(snapshot)
        
        logger.info(
            f"Snapshot: total_value={snapshot['total_value']:.2f}, "
            f"cash={snapshot['cash']:.2f}, positions={snapshot['position_count']}"
        )
    
    # Collect results
    results = {
        'account': account,
        'positions': position_manager.all_positions,
        'orders': list(order_book.orders.values()),
        'snapshots': snapshots,
        'storage': storage_backend,  # Include storage backend for data extraction
    }
    
    logger.info("Backtest completed successfully")
    
    return results

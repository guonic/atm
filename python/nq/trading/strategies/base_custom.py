"""
Base strategy class for custom backtesting framework.

Provides common functionality for all custom strategies including:
- Data capture and logging
- Order management
- Position tracking
"""

import logging
from typing import Optional, Dict, Any, List, TYPE_CHECKING

import pandas as pd

from ..interfaces.strategy import IStrategy
from ..state import Order, OrderBook, OrderSide, OrderType, OrderStatus
from ..logic import RiskManager, PositionAllocator

if TYPE_CHECKING:
    from ..state import PositionManager, Account

logger = logging.getLogger(__name__)


class BaseCustomStrategy(IStrategy):
    """
    Base class for custom backtesting strategies.
    
    Provides common functionality:
    - Order submission and tracking
    - Risk management integration
    - Position allocation
    - Data capture and logging
    - State management
    
    Subclasses should implement:
    - on_bar(): Main trading logic
    - name: Strategy name property
    """
    
    def __init__(
        self,
        position_manager: 'PositionManager',
        order_book: OrderBook,
        risk_manager: RiskManager,
        position_allocator: PositionAllocator,
        account: 'Account',
    ):
        """
        Initialize base custom strategy.
        
        Args:
            position_manager: PositionManager instance.
            order_book: OrderBook instance.
            risk_manager: RiskManager instance.
            position_allocator: PositionAllocator instance.
            account: Account instance.
        """
        self.position_manager = position_manager
        self.order_book = order_book
        self.risk_manager = risk_manager
        self.position_allocator = position_allocator
        self.account = account
        
        # Data capture for analysis
        self._executed_orders: List[Dict[str, Any]] = []
        self._signals: List[Dict[str, Any]] = []
        self._daily_stats: List[Dict[str, Any]] = []
        
        # Current bar state
        self._current_date: Optional[pd.Timestamp] = None
        self._current_market_data: Optional[pd.DataFrame] = None
    
    @property
    def name(self) -> str:
        """Strategy name - must be overridden by subclasses."""
        return self.__class__.__name__
    
    def on_bar(
        self,
        date: pd.Timestamp,
        market_data: pd.DataFrame,
    ) -> None:
        """
        Called on each trading bar.
        
        Subclasses should override this method to implement trading logic.
        This base implementation provides common setup and logging.
        
        Args:
            date: Current trading date.
            market_data: Market data DataFrame in normalized format.
        """
        self._current_date = date
        self._current_market_data = market_data
        
        logger.debug(f"[{self.name}] Processing bar: {date.strftime('%Y-%m-%d')}")
    
    def submit_order(
        self,
        symbol: str,
        side: OrderSide,
        amount: Optional[float] = None,
        target_value: Optional[float] = None,
        order_type: OrderType = OrderType.MARKET,
        reason: Optional[str] = None,
    ) -> Optional[Order]:
        """
        Submit an order with risk check and logging.
        
        Args:
            symbol: Stock symbol.
            side: Order side (BUY or SELL).
            amount: Order amount (shares). Required if target_value is None.
            target_value: Target order value. If provided, amount will be calculated.
            order_type: Order type (MARKET or LIMIT).
            reason: Reason for the order (for logging).
        
        Returns:
            Order object if submitted successfully, None otherwise.
        """
        if amount is None and target_value is None:
            logger.warning(f"[{self.name}] Cannot submit order: both amount and target_value are None")
            return None
        
        order = Order(
            symbol=symbol,
            side=side,
            amount=amount or 0.0,
            target_value=target_value,
            order_type=order_type,
            date=self._current_date or pd.Timestamp.now(),
        )
        
        # Risk check
        if not self.risk_manager.check_order(order, self._current_market_data):
            logger.debug(f"[{self.name}] Order rejected by risk manager: {symbol} {side.value}")
            return None
        
        # Submit to order book
        try:
            self.order_book.submit(order)
            
            # Log signal
            signal = {
                'date': self._current_date,
                'symbol': symbol,
                'side': side.value,
                'amount': amount,
                'target_value': target_value,
                'reason': reason,
            }
            self._signals.append(signal)
            
            logger.info(
                f"[{self.name}] Order submitted: {symbol} {side.value} "
                f"(amount={amount}, target_value={target_value}, reason={reason})"
            )
            
            return order
        except Exception as e:
            logger.error(f"[{self.name}] Failed to submit order: {e}", exc_info=True)
            return None
    
    def capture_executed_order(
        self,
        order: Order,
        filled_amount: float,
        filled_price: float,
    ) -> None:
        """
        Capture executed order for analysis.
        
        This should be called by the executor after order execution.
        
        Args:
            order: Original order.
            filled_amount: Filled amount.
            filled_price: Filled price.
        """
        order_info = {
            'order_id': order.order_id,
            'date': order.date,
            'symbol': order.symbol,
            'side': order.side.value,
            'amount': filled_amount,
            'price': filled_price,
            'target_value': order.target_value,
            'reason': getattr(order, 'reason', None),
        }
        self._executed_orders.append(order_info)
    
    def capture_daily_stats(
        self,
        stats: Dict[str, Any],
    ) -> None:
        """
        Capture daily statistics for analysis.
        
        Args:
            stats: Dictionary containing daily statistics.
        """
        daily_stat = {
            'date': self._current_date,
            **stats,
        }
        self._daily_stats.append(daily_stat)
    
    def get_executed_orders(self) -> List[Dict[str, Any]]:
        """
        Get all executed orders.
        
        Returns:
            List of executed order dictionaries.
        """
        return self._executed_orders.copy()
    
    def get_signals(self) -> List[Dict[str, Any]]:
        """
        Get all trading signals.
        
        Returns:
            List of signal dictionaries.
        """
        return self._signals.copy()
    
    def get_daily_stats(self) -> List[Dict[str, Any]]:
        """
        Get all daily statistics.
        
        Returns:
            List of daily statistics dictionaries.
        """
        return self._daily_stats.copy()
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get current strategy state.
        
        Returns:
            Dictionary containing strategy state.
        """
        return {
            'name': self.name,
            'current_date': self._current_date.isoformat() if self._current_date else None,
            'position_count': len(self.position_manager.all_positions),
            'pending_orders': len(self.order_book.get_pending_orders()),
            'executed_orders_count': len(self._executed_orders),
            'signals_count': len(self._signals),
        }
    
    def on_backtest_start(self, start_date: pd.Timestamp, end_date: pd.Timestamp) -> None:
        """Called when backtest starts."""
        logger.info(f"[{self.name}] Backtest starting: {start_date.date()} to {end_date.date()}")
        self._executed_orders.clear()
        self._signals.clear()
        self._daily_stats.clear()
    
    def on_backtest_end(self) -> None:
        """Called when backtest ends."""
        logger.info(
            f"[{self.name}] Backtest ended. "
            f"Executed orders: {len(self._executed_orders)}, "
            f"Signals: {len(self._signals)}"
        )

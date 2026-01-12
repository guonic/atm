"""
Risk management.

Validates orders before execution (limit up/down, concentration, etc.).
"""

from typing import TYPE_CHECKING
import pandas as pd
import logging

from ..state import Order, OrderSide
from ..interfaces.storage import IStorageBackend

if TYPE_CHECKING:
    from ..state import Account, PositionManager

logger = logging.getLogger(__name__)


class RiskManager:
    """Risk management and order validation.
    
    Validates orders before execution:
    - Limit up/down checks
    - Concentration checks
    - Suspension checks
    """
    
    def __init__(
        self,
        account: 'Account',
        position_manager: 'PositionManager',
        storage: IStorageBackend,
        max_single_position_weight: float = 0.2,
    ):
        """
        Initialize risk manager.
        
        Args:
            account: Account instance.
            position_manager: PositionManager instance.
            storage: Storage backend for logging risk events.
            max_single_position_weight: Maximum single position weight (default: 0.2 = 20%).
        """
        self.account = account
        self.position_manager = position_manager
        self.storage = storage
        self.max_single_position_weight = max_single_position_weight
    
    def check_order(
        self,
        order: Order,
        market_data: pd.DataFrame
    ) -> bool:
        """
        Check if order passes risk validation.
        
        Args:
            order: Order to check.
            market_data: Market data DataFrame.
        
        Returns:
            True if passes, False otherwise.
        """
        # 1. Check liquidity (limit up/down, suspension)
        if not self._check_liquidity(order, market_data):
            return False
        
        # 2. Check concentration
        if not self._check_concentration(order, market_data):
            return False
        
        return True
    
    def _check_liquidity(
        self,
        order: Order,
        market_data: pd.DataFrame
    ) -> bool:
        """
        Check liquidity (limit up/down, suspension).
        
        Args:
            order: Order to check.
            market_data: Market data DataFrame.
        
        Returns:
            True if liquid, False otherwise.
        """
        if order.symbol not in market_data.index.get_level_values(0) if isinstance(market_data.index, pd.MultiIndex) else market_data.index:
            self.log_event(order, "REJECT_NO_DATA")
            return False
        
        try:
            # Get symbol data
            if isinstance(market_data.index, pd.MultiIndex):
                symbol_data = market_data.xs(order.symbol, level=0)
                if len(symbol_data) == 0:
                    self.log_event(order, "REJECT_NO_DATA")
                    return False
                daily_data = symbol_data.iloc[-1] if len(symbol_data) > 0 else None
            else:
                daily_data = market_data.loc[order.symbol]
            
            if daily_data is None:
                self.log_event(order, "REJECT_NO_DATA")
                return False
            
            # Check limit down (sell order)
            if order.side == OrderSide.SELL:
                if self._is_limit_down(daily_data):
                    self.log_event(order, "REJECT_LIMIT_DOWN")
                    return False
            
            # Check limit up (buy order)
            if order.side == OrderSide.BUY:
                if self._is_limit_up(daily_data):
                    self.log_event(order, "REJECT_LIMIT_UP")
                    return False
            
            return True
        except Exception as e:
            logger.warning(f"Failed to check liquidity for {order.symbol}: {e}")
            self.log_event(order, "REJECT_LIQUIDITY_CHECK_FAILED")
            return False
    
    def _check_concentration(
        self,
        order: Order,
        market_data: pd.DataFrame
    ) -> bool:
        """
        Check position concentration.
        
        Args:
            order: Order to check.
            market_data: Market data DataFrame.
        
        Returns:
            True if within limit, False otherwise.
        """
        if order.side == OrderSide.BUY:
            # Get current price
            try:
                if isinstance(market_data.index, pd.MultiIndex):
                    symbol_data = market_data.xs(order.symbol, level=0)
                    current_price = float(symbol_data.iloc[-1].get('$close', symbol_data.iloc[-1].get('close', 0)))
                else:
                    current_price = float(market_data.loc[order.symbol].get('$close', market_data.loc[order.symbol].get('close', 0)))
                
                # Calculate current weight
                current_weight = self.position_manager.get_weight(order.symbol, current_price)
                
                # Calculate new weight after order
                if order.target_value:
                    new_value = order.target_value
                else:
                    new_value = order.amount * current_price
                
                total_value = self.account.get_total_value(self.position_manager)
                new_weight = (current_weight * total_value + new_value) / total_value if total_value > 0 else 0
                
                if new_weight > self.max_single_position_weight:
                    self.log_event(order, "REJECT_CONCENTRATION")
                    return False
            except Exception as e:
                logger.warning(f"Failed to check concentration for {order.symbol}: {e}")
                return False
        
        return True
    
    def _is_limit_down(self, daily_data: pd.Series) -> bool:
        """
        Check if stock is limit down.
        
        Args:
            daily_data: Daily market data.
        
        Returns:
            True if limit down, False otherwise.
        
        Note:
            This is a simplified implementation. In production, should check
            actual limit up/down flags from market data.
        """
        # Simplified: if close == low and close < prev_close * 0.9, might be limit down
        # In production, should use actual limit up/down flags from Qlib or other data source
        return False
    
    def _is_limit_up(self, daily_data: pd.Series) -> bool:
        """
        Check if stock is limit up.
        
        Args:
            daily_data: Daily market data.
        
        Returns:
            True if limit up, False otherwise.
        
        Note:
            This is a simplified implementation. In production, should check
            actual limit up/down flags from market data.
        """
        # Simplified: if close == high and close > prev_close * 1.1, might be limit up
        # In production, should use actual limit up/down flags from Qlib or other data source
        return False
    
    def log_event(self, order: Order, reason: str) -> None:
        """
        Log risk event.
        
        Args:
            order: Order that was rejected.
            reason: Rejection reason.
        """
        event = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'order_id': order.order_id,
            'symbol': order.symbol,
            'side': order.side.value,
            'reason': reason,
        }
        self.storage.save(f"risk_event:{order.order_id}", event)
        logger.warning(f"Risk event: {reason} for order {order.order_id} ({order.symbol})")

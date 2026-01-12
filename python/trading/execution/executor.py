"""
Order execution engine.

Executes orders and updates position/account state.
"""

from dataclasses import dataclass
from typing import Optional
import pandas as pd
import logging

from ..state import PositionManager, OrderBook, Order, OrderStatus, OrderSide

logger = logging.getLogger(__name__)


@dataclass
class FillInfo:
    """Fill information after order execution."""
    
    order_id: str
    symbol: str
    side: OrderSide
    amount: float
    price: float
    commission: float
    date: pd.Timestamp
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'order_id': self.order_id,
            'symbol': self.symbol,
            'side': self.side.value,
            'amount': self.amount,
            'price': self.price,
            'commission': self.commission,
            'date': self.date.strftime('%Y-%m-%d'),
        }


class Executor:
    """Order execution engine.
    
    Executes orders in backtest mode (using next bar's open price).
    Completely independent from Qlib's executor.
    """
    
    def __init__(
        self,
        position_manager: PositionManager,
        order_book: OrderBook,
        commission_rate: float = 0.0015,
        slippage_rate: float = 0.0,
        min_commission: float = 5.0,
    ):
        """
        Initialize executor.
        
        Args:
            position_manager: PositionManager instance.
            order_book: OrderBook instance.
            commission_rate: Commission rate (default: 0.0015 = 0.15%).
            slippage_rate: Slippage rate (default: 0.0 = no slippage).
            min_commission: Minimum commission (default: 5.0).
        """
        self.position_manager = position_manager
        self.order_book = order_book
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate
        self.min_commission = min_commission
    
    def execute_order(
        self,
        order: Order,
        market_data: pd.DataFrame,
        date: pd.Timestamp
    ) -> Optional[FillInfo]:
        """
        Execute order (backtest mode).
        
        Args:
            order: Order to execute.
            market_data: Market data DataFrame (from Qlib).
            date: Current trading date.
        
        Returns:
            FillInfo if filled, None if rejected.
        """
        # Check if symbol exists in market data
        if isinstance(market_data.index, pd.MultiIndex):
            available_symbols = market_data.index.get_level_values(0).unique()
            if order.symbol not in available_symbols:
                logger.warning(
                    f"Symbol {order.symbol} not found in market data. "
                    f"Available symbols: {len(available_symbols)} (showing first 10: {list(available_symbols[:10])})"
                )
                self.order_book.update_order(order.order_id, OrderStatus.REJECTED)
                return None
        else:
            if order.symbol not in market_data.index:
                logger.warning(f"Symbol {order.symbol} not found in market data")
                self.order_book.update_order(order.order_id, OrderStatus.REJECTED)
                return None
        
        # Get symbol data
        try:
            if isinstance(market_data.index, pd.MultiIndex):
                # MultiIndex: (instrument, datetime)
                # Filter by symbol and date
                symbol_mask = market_data.index.get_level_values(0) == order.symbol
                date_mask = market_data.index.get_level_values(1) == date
                combined_mask = symbol_mask & date_mask
                
                if not combined_mask.any():
                    # Debug: check what dates are available for this symbol
                    symbol_data = market_data.xs(order.symbol, level=0)
                    available_dates = symbol_data.index.get_level_values(0).unique() if isinstance(symbol_data.index, pd.MultiIndex) else symbol_data.index.unique()
                    logger.warning(
                        f"No data for {order.symbol} on {date}. "
                        f"Available dates for this symbol: {len(available_dates)} "
                        f"(first 5: {list(available_dates[:5]) if len(available_dates) > 0 else 'none'})"
                    )
                    self.order_book.update_order(order.order_id, OrderStatus.REJECTED)
                    return None
                
                # Get the row(s) matching symbol and date
                symbol_date_data = market_data.loc[combined_mask]
                if len(symbol_date_data) > 0:
                    # Take first row if multiple (shouldn't happen, but safe)
                    daily_data = symbol_date_data.iloc[0]
                else:
                    logger.warning(f"No data for {order.symbol} on {date}")
                    self.order_book.update_order(order.order_id, OrderStatus.REJECTED)
                    return None
            else:
                # Single index: assume it's symbol index, and data is for current date
                if order.symbol in market_data.index:
                    daily_data = market_data.loc[order.symbol]
                else:
                    logger.warning(f"Symbol {order.symbol} not found in market data")
                    self.order_book.update_order(order.order_id, OrderStatus.REJECTED)
                    return None
        except Exception as e:
            logger.warning(f"Failed to get market data for {order.symbol} on {date}: {e}", exc_info=True)
            self.order_book.update_order(order.order_id, OrderStatus.REJECTED)
            return None
        
        # Determine fill price (backtest mode: use open price with slippage)
        # CRITICAL: Data should have been filtered for NaN in backtest engine
        # If we still get NaN here, it's a bug - raise error instead of handling
        raw_price = daily_data.get('$open') or daily_data.get('open')
        
        if raw_price is None:
            raise ValueError(
                f"CRITICAL BUG: Price field missing for {order.symbol} on {date}. "
                f"Available fields: {list(daily_data.index) if hasattr(daily_data, 'index') else list(daily_data.keys())}"
            )
        
        if pd.isna(raw_price):
            raise ValueError(
                f"CRITICAL BUG: Price is NaN for {order.symbol} on {date}. "
                f"This should have been filtered in backtest engine. "
                f"Raw price value: {raw_price}"
            )
        
        if order.order_type.value == "MARKET":
            if order.side == OrderSide.BUY:
                # Buy: use open price + slippage
                fill_price = float(raw_price) * (1 + self.slippage_rate)
            else:
                # Sell: use open price - slippage
                fill_price = float(raw_price) * (1 - self.slippage_rate)
        else:
            # Limit order
            fill_price = order.limit_price or float(raw_price)
        
        # Check for NaN or invalid price
        if pd.isna(fill_price) or fill_price <= 0:
            logger.warning(f"Invalid fill price for {order.symbol}: {fill_price} (raw_price={raw_price})")
            self.order_book.update_order(order.order_id, OrderStatus.REJECTED)
            return None
        
        # Calculate fill amount
        if order.side == OrderSide.BUY:
            # Buy: calculate based on target_value or available cash (including frozen)
            # For backtest, we use available_cash + frozen_cash as total available
            total_available_cash = (
                self.position_manager.account.available_cash + 
                self.position_manager.account.frozen_cash
            )
            
            if order.target_value:
                # CRITICAL: Check for NaN - this should never happen
                if pd.isna(order.target_value):
                    raise ValueError(
                        f"CRITICAL BUG: order.target_value is NaN for {order.symbol}. "
                        f"This should not happen - check position_allocator.calculate_position_size()"
                    )
                if pd.isna(total_available_cash):
                    raise ValueError(
                        f"CRITICAL BUG: total_available_cash is NaN for {order.symbol}. "
                        f"available_cash={self.position_manager.account.available_cash}, "
                        f"frozen_cash={self.position_manager.account.frozen_cash}"
                    )
                target_cash = min(order.target_value, total_available_cash)
                fill_amount = int(target_cash / fill_price / 100) * 100  # Round to 100 shares
            else:
                fill_amount = order.amount
            
            # Check cash availability (use total available cash)
            required_cash = fill_amount * fill_price * (1 + self.commission_rate)
            if total_available_cash < required_cash:
                # Adjust fill amount based on available cash
                fill_amount = int(
                    total_available_cash
                    / fill_price / (1 + self.commission_rate) / 100
                ) * 100
                if fill_amount <= 0:
                    logger.warning(f"Insufficient cash for {order.symbol}")
                    self.order_book.update_order(order.order_id, OrderStatus.REJECTED)
                    return None
        else:
            # Sell: check position
            if order.symbol not in self.position_manager.positions:
                logger.warning(f"No position for {order.symbol}")
                self.order_book.update_order(order.order_id, OrderStatus.REJECTED)
                return None
            
            pos = self.position_manager.positions[order.symbol]
            fill_amount = min(order.amount, pos.amount)
            if fill_amount <= 0:
                logger.warning(f"Insufficient position for {order.symbol}")
                self.order_book.update_order(order.order_id, OrderStatus.REJECTED)
                return None
        
        # Calculate commission
        commission = max(
            fill_amount * fill_price * self.commission_rate,
            self.min_commission
        )
        
        # Create fill info
        fill_info = FillInfo(
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            amount=fill_amount,
            price=fill_price,
            commission=commission,
            date=date,
        )
        
        # Update order status
        if fill_amount >= order.amount:
            self.order_book.update_order(
                order.order_id,
                OrderStatus.FILLED,
                filled_amount=fill_amount,
                filled_price=fill_price,
            )
        else:
            self.order_book.update_order(
                order.order_id,
                OrderStatus.PARTIAL_FILLED,
                filled_amount=fill_amount,
                filled_price=fill_price,
            )
        
        # Update position and account
        self._update_position_and_account(fill_info)
        
        logger.info(
            f"Order filled: {order.side.value} {order.symbol} "
            f"{fill_amount:.0f} @ {fill_price:.2f}, commission={commission:.2f}"
        )
        
        return fill_info
    
    def _update_position_and_account(self, fill_info: FillInfo) -> None:
        """
        Update position and account after order fill.
        
        Args:
            fill_info: Fill information.
        """
        if fill_info.side == OrderSide.BUY:
            # Buy filled
            total_cost = fill_info.amount * fill_info.price + fill_info.commission
            
            # For backtest: if frozen_cash is insufficient, use available_cash
            # This handles the case where cash wasn't frozen at order submission
            account = self.position_manager.account
            if account.frozen_cash >= total_cost:
                # Use frozen cash if available
                account.deduct_cash(total_cost)
            else:
                # Use available cash directly (backtest mode)
                if account.available_cash < total_cost:
                    raise ValueError(
                        f"Insufficient cash: available={account.available_cash:.2f}, "
                        f"frozen={account.frozen_cash:.2f}, required={total_cost:.2f}"
                    )
                account.available_cash -= total_cost
            
            self.position_manager.add_position(
                symbol=fill_info.symbol,
                entry_date=fill_info.date,
                entry_price=fill_info.price,
                amount=fill_info.amount,
            )
        else:
            # Sell filled
            proceeds = fill_info.amount * fill_info.price - fill_info.commission
            self.position_manager.account.add_cash(proceeds)
            
            self.position_manager.reduce_position(
                symbol=fill_info.symbol,
                amount=fill_info.amount,
            )

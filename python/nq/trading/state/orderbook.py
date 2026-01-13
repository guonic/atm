"""
Order book management.

Manages order lifecycle and order history.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional
from datetime import datetime
import uuid
import pandas as pd
import logging

from ..interfaces.storage import IStorageBackend

logger = logging.getLogger(__name__)


class OrderStatus(Enum):
    """Order status enumeration."""
    NEW = "NEW"
    PENDING = "PENDING"
    PARTIAL_FILLED = "PARTIAL_FILLED"
    FILLED = "FILLED"
    CANCELED = "CANCELED"
    REJECTED = "REJECTED"


class OrderSide(Enum):
    """Order side enumeration."""
    BUY = "BUY"
    SELL = "SELL"


class OrderType(Enum):
    """Order type enumeration."""
    MARKET = "MARKET"
    LIMIT = "LIMIT"


@dataclass
class Order:
    """Order representation.
    
    Completely independent of Qlib's Order class.
    """
    
    symbol: str
    side: OrderSide
    amount: float
    order_type: OrderType
    date: pd.Timestamp
    order_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    status: OrderStatus = OrderStatus.NEW
    target_value: Optional[float] = None  # Target value for buy orders
    limit_price: Optional[float] = None  # Limit price for limit orders
    filled_amount: float = 0.0
    filled_price: Optional[float] = None
    create_time: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'order_id': self.order_id,
            'symbol': self.symbol,
            'side': self.side.value,
            'amount': self.amount,
            'order_type': self.order_type.value,
            'date': self.date.strftime('%Y-%m-%d'),
            'status': self.status.value,
            'target_value': self.target_value,
            'limit_price': self.limit_price,
            'filled_amount': self.filled_amount,
            'filled_price': self.filled_price,
            'create_time': self.create_time.isoformat(),
        }


class OrderBook:
    """Order book management.
    
    Manages order lifecycle and order history.
    Completely independent of Qlib's order management.
    """
    
    def __init__(self, storage: IStorageBackend):
        """
        Initialize order book.
        
        Args:
            storage: Storage backend.
        """
        self.storage = storage
        self.orders: Dict[str, Order] = {}
    
    def submit(self, order: Order) -> None:
        """
        Submit order to order book.
        
        Args:
            order: Order to submit.
        """
        order.status = OrderStatus.PENDING
        self.orders[order.order_id] = order
        try:
            self.storage.save(f"order:{order.order_id}", order.to_dict())
        except Exception as e:
            logger.warning(f"Failed to save order to storage: {e}")
        logger.debug(
            f"Order submitted: {order.order_id}, {order.side.value} {order.symbol}, "
            f"status={order.status.value}, total_orders={len(self.orders)}"
        )
    
    def update_order(
        self,
        order_id: str,
        status: OrderStatus,
        filled_amount: Optional[float] = None,
        filled_price: Optional[float] = None,
    ) -> None:
        """
        Update order status.
        
        Args:
            order_id: Order ID.
            status: New status.
            filled_amount: Filled amount (if filled).
            filled_price: Filled price (if filled).
        """
        if order_id in self.orders:
            order = self.orders[order_id]
            order.status = status
            if filled_amount is not None:
                order.filled_amount = filled_amount
            if filled_price is not None:
                order.filled_price = filled_price
            self.storage.save(f"order:{order_id}", order.to_dict())
            logger.debug(f"Order updated: {order_id}, status={status.value}")
    
    def get_order(self, order_id: str) -> Optional[Order]:
        """
        Get order by ID.
        
        Args:
            order_id: Order ID.
        
        Returns:
            Order if found, None otherwise.
        """
        return self.orders.get(order_id)
    
    def get_pending_orders(self) -> List[Order]:
        """
        Get all pending orders.
        
        Returns:
            List of pending orders.
        """
        return [
            order for order in self.orders.values()
            if order.status == OrderStatus.PENDING
        ]
    
    def get_orders_by_symbol(self, symbol: str) -> List[Order]:
        """
        Get all orders for a symbol.
        
        Args:
            symbol: Stock symbol.
        
        Returns:
            List of orders.
        """
        return [
            order for order in self.orders.values()
            if order.symbol == symbol
        ]
    
    def get_orders_by_status(self, status: OrderStatus) -> List[Order]:
        """
        Get all orders with specific status.
        
        Args:
            status: Order status.
        
        Returns:
            List of orders.
        """
        return [
            order for order in self.orders.values()
            if order.status == status
        ]

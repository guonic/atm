"""
Unit tests for OrderBook.

Test order submission, status management, and retrieval.
"""

import unittest
from datetime import datetime
import pandas as pd

from trading.state import Order, OrderBook, OrderStatus, OrderSide, OrderType
from trading.storage import MemoryStorage


class TestOrderBook(unittest.TestCase):
    """Test OrderBook functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.storage = MemoryStorage()
        self.order_book = OrderBook(self.storage)
    
    def test_submit_order(self):
        """Test order submission."""
        order = Order(
            symbol="000001.SZ",
            side=OrderSide.BUY,
            amount=100,
            order_type=OrderType.MARKET,
            date=pd.Timestamp("2025-07-01"),
        )
        
        # Order should start as NEW
        self.assertEqual(order.status, OrderStatus.NEW)
        
        # Submit order
        self.order_book.submit(order)
        
        # Order should be PENDING after submission
        self.assertEqual(order.status, OrderStatus.PENDING)
        
        # Order should be in order book
        self.assertIn(order.order_id, self.order_book.orders)
        self.assertEqual(len(self.order_book.orders), 1)
        
        # Order should be retrievable
        retrieved_order = self.order_book.get_order(order.order_id)
        self.assertIsNotNone(retrieved_order)
        self.assertEqual(retrieved_order.symbol, "000001.SZ")
        self.assertEqual(retrieved_order.status, OrderStatus.PENDING)
    
    def test_get_pending_orders(self):
        """Test getting pending orders."""
        # Submit multiple orders
        order1 = Order(
            symbol="000001.SZ",
            side=OrderSide.BUY,
            amount=100,
            order_type=OrderType.MARKET,
            date=pd.Timestamp("2025-07-01"),
        )
        order2 = Order(
            symbol="000002.SZ",
            side=OrderSide.BUY,
            amount=200,
            order_type=OrderType.MARKET,
            date=pd.Timestamp("2025-07-01"),
        )
        
        self.order_book.submit(order1)
        self.order_book.submit(order2)
        
        # Get pending orders
        pending = self.order_book.get_pending_orders()
        
        # Should have 2 pending orders
        self.assertEqual(len(pending), 2)
        self.assertIn(order1, pending)
        self.assertIn(order2, pending)
        
        # Update one order to FILLED
        self.order_book.update_order(order1.order_id, OrderStatus.FILLED)
        
        # Should have 1 pending order now
        pending = self.order_book.get_pending_orders()
        self.assertEqual(len(pending), 1)
        self.assertIn(order2, pending)
        self.assertNotIn(order1, pending)
    
    def test_order_status_transition(self):
        """Test order status transitions."""
        order = Order(
            symbol="000001.SZ",
            side=OrderSide.BUY,
            amount=100,
            order_type=OrderType.MARKET,
            date=pd.Timestamp("2025-07-01"),
        )
        
        # Initial status: NEW
        self.assertEqual(order.status, OrderStatus.NEW)
        
        # Submit: NEW -> PENDING
        self.order_book.submit(order)
        self.assertEqual(order.status, OrderStatus.PENDING)
        
        # Update to FILLED: PENDING -> FILLED
        self.order_book.update_order(order.order_id, OrderStatus.FILLED, filled_amount=100, filled_price=10.0)
        self.assertEqual(order.status, OrderStatus.FILLED)
        self.assertEqual(order.filled_amount, 100)
        self.assertEqual(order.filled_price, 10.0)
    
    def test_multiple_orders_same_symbol(self):
        """Test multiple orders for same symbol."""
        order1 = Order(
            symbol="000001.SZ",
            side=OrderSide.BUY,
            amount=100,
            order_type=OrderType.MARKET,
            date=pd.Timestamp("2025-07-01"),
        )
        order2 = Order(
            symbol="000001.SZ",
            side=OrderSide.BUY,
            amount=200,
            order_type=OrderType.MARKET,
            date=pd.Timestamp("2025-07-01"),
        )
        
        self.order_book.submit(order1)
        self.order_book.submit(order2)
        
        # Both orders should be in order book
        self.assertEqual(len(self.order_book.orders), 2)
        
        # Both should be pending
        pending = self.order_book.get_pending_orders()
        self.assertEqual(len(pending), 2)


if __name__ == "__main__":
    unittest.main()

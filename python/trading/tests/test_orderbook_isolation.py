"""
Test to verify OrderBook isolation issue.

This test demonstrates the problem: if strategy uses one OrderBook instance
but backtest engine uses another, orders won't be found.
"""

import unittest
from trading.state import OrderBook, Order, OrderSide, OrderType, OrderStatus
from trading.storage import MemoryStorage
import pandas as pd


class TestOrderBookIsolation(unittest.TestCase):
    """Test OrderBook isolation issue."""
    
    def test_orderbook_isolation_problem(self):
        """Demonstrate the isolation problem."""
        storage1 = MemoryStorage()
        storage2 = MemoryStorage()
        
        # OrderBook instance 1 (used by strategy)
        order_book_1 = OrderBook(storage1)
        
        # OrderBook instance 2 (used by backtest engine)
        order_book_2 = OrderBook(storage2)
        
        # Submit order to order_book_1
        order = Order(
            symbol="000001.SZ",
            side=OrderSide.BUY,
            amount=100,
            order_type=OrderType.MARKET,
            date=pd.Timestamp("2025-07-01"),
        )
        order_book_1.submit(order)
        
        # Check: order is in order_book_1
        self.assertEqual(len(order_book_1.orders), 1)
        self.assertEqual(len(order_book_1.get_pending_orders()), 1)
        
        # Check: order is NOT in order_book_2
        self.assertEqual(len(order_book_2.orders), 0)
        self.assertEqual(len(order_book_2.get_pending_orders()), 0)
        
        # This demonstrates the problem: if strategy uses order_book_1
        # but backtest engine uses order_book_2, orders won't be found!


if __name__ == "__main__":
    unittest.main()

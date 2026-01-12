"""
Complete flow test: Strategy -> OrderBook -> Executor.

This test verifies the complete data flow without requiring Qlib or actual models.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd

from trading.strategy import DualModelStrategy
from trading.strategy.buy_models import IBuyModel
from trading.strategy.sell_models import ISellModel
from trading.state import Account, PositionManager, OrderBook, OrderStatus
from trading.logic import RiskManager, PositionAllocator
from trading.storage import MemoryStorage
from trading.execution import Executor


class SimpleBuyModel(IBuyModel):
    """Simple buy model that returns fixed rankings."""
    
    def __init__(self, symbols=None):
        self.symbols = symbols or ['000001.SZ', '000002.SZ']
    
    def generate_ranks(self, date, market_data, **kwargs):
        """Return fixed rankings."""
        return pd.DataFrame({
            'symbol': self.symbols,
            'score': [0.9, 0.8],
            'rank': [1, 2],
        })


class SimpleSellModel(ISellModel):
    """Simple sell model that never exits."""
    
    @property
    def threshold(self):
        return 0.65
    
    def predict_exit(self, position, market_data, date, **kwargs):
        """Never exit."""
        return 0.3


class TestCompleteFlow(unittest.TestCase):
    """Test complete order flow."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.storage = MemoryStorage()
        self.account = Account(
            account_id="test_001",
            available_cash=1000000.0,
            initial_cash=1000000.0,
        )
        self.position_manager = PositionManager(self.account, self.storage)
        self.order_book = OrderBook(self.storage)
        self.risk_manager = RiskManager(self.account, self.position_manager, self.storage)
        self.position_allocator = PositionAllocator(target_positions=2)
        
        self.buy_model = SimpleBuyModel()
        self.sell_model = SimpleSellModel()
        
        self.strategy = DualModelStrategy(
            buy_model=self.buy_model,
            sell_model=self.sell_model,
            position_manager=self.position_manager,
            order_book=self.order_book,
            risk_manager=self.risk_manager,
            position_allocator=self.position_allocator,
        )
    
    def test_strategy_submits_orders_to_orderbook(self):
        """Test that strategy submits orders to the correct OrderBook instance."""
        # Create mock market data
        market_data = pd.DataFrame({
            '$close': [10.0, 20.0],
            '$open': [10.0, 20.0],
            '$high': [10.5, 20.5],
            '$low': [9.5, 19.5],
            '$volume': [1000000, 2000000],
        }, index=pd.MultiIndex.from_product([
            ['000001.SZ', '000002.SZ'],
            [pd.Timestamp('2025-07-01')]
        ], names=['instrument', 'datetime']))
        
        date = pd.Timestamp('2025-07-01')
        
        # Initial state: no orders
        self.assertEqual(len(self.order_book.orders), 0)
        self.assertEqual(len(self.order_book.get_pending_orders()), 0)
        
        # Strategy generates and submits orders
        self.strategy.on_bar(date, market_data)
        
        # Verify orders were submitted to the correct OrderBook
        # This is the key test: orders should be in strategy.order_book
        self.assertGreater(len(self.strategy.order_book.orders), 0,
                          "No orders in strategy.order_book")
        self.assertGreater(len(self.order_book.orders), 0,
                          "No orders in order_book (should be same instance)")
        
        # Verify they are the same instance
        self.assertIs(self.strategy.order_book, self.order_book,
                     "OrderBook instances are different - this is the bug!")
        
        # Verify orders are PENDING
        pending = self.order_book.get_pending_orders()
        self.assertGreater(len(pending), 0, "No pending orders found")
        
        for order in pending:
            self.assertEqual(order.status, OrderStatus.PENDING)
    
    def test_executor_can_find_orders_from_strategy(self):
        """Test that executor can find orders submitted by strategy."""
        # Create mock market data
        market_data = pd.DataFrame({
            '$close': [10.0, 20.0],
            '$open': [10.0, 20.0],
            '$high': [10.5, 20.5],
            '$low': [9.5, 19.5],
            '$volume': [1000000, 2000000],
        }, index=pd.MultiIndex.from_product([
            ['000001.SZ', '000002.SZ'],
            [pd.Timestamp('2025-07-01')]
        ], names=['instrument', 'datetime']))
        
        date = pd.Timestamp('2025-07-01')
        
        # Strategy generates orders
        self.strategy.on_bar(date, market_data)
        
        # Create executor using the same order_book
        executor = Executor(self.position_manager, self.order_book)
        
        # Executor should be able to find pending orders
        pending = self.order_book.get_pending_orders()
        self.assertGreater(len(pending), 0, "Executor cannot find orders submitted by strategy")
        
        # Try to execute one order
        if len(pending) > 0:
            order = pending[0]
            fill_info = executor.execute_order(order, market_data, date)
            
            # Order should either be filled or rejected (not None without reason)
            if fill_info is None:
                # Check if order was rejected
                self.assertIn(order.status, [OrderStatus.REJECTED, OrderStatus.PENDING],
                             f"Order status is {order.status}, expected REJECTED or still PENDING")


if __name__ == "__main__":
    unittest.main()

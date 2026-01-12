"""
Unit tests for strategy order flow.

Test the complete flow: strategy generates orders -> submits to order book -> executor processes.
"""

import unittest
from unittest.mock import Mock, MagicMock, patch
import pandas as pd

from trading.strategy import DualModelStrategy
from trading.strategy.buy_models import IBuyModel
from trading.strategy.sell_models import ISellModel
from trading.state import Account, PositionManager, OrderBook, OrderStatus
from trading.logic import RiskManager, PositionAllocator
from trading.storage import MemoryStorage


class MockBuyModel(IBuyModel):
    """Mock buy model for testing."""
    
    def generate_ranks(self, date, market_data, **kwargs):
        """Generate mock rankings."""
        return pd.DataFrame({
            'symbol': ['000001.SZ', '000002.SZ', '000003.SZ'],
            'score': [0.9, 0.8, 0.7],
            'rank': [1, 2, 3],
        })


class MockSellModel(ISellModel):
    """Mock sell model for testing."""
    
    @property
    def threshold(self):
        return 0.65
    
    def predict_exit(self, position, market_data, date, **kwargs):
        """Always return low probability (no exit)."""
        return 0.3


class TestStrategyOrderFlow(unittest.TestCase):
    """Test strategy order generation and submission flow."""
    
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
        self.position_allocator = PositionAllocator(target_positions=3)
        
        self.buy_model = MockBuyModel()
        self.sell_model = MockSellModel()
        
        self.strategy = DualModelStrategy(
            buy_model=self.buy_model,
            sell_model=self.sell_model,
            position_manager=self.position_manager,
            order_book=self.order_book,
            risk_manager=self.risk_manager,
            position_allocator=self.position_allocator,
        )
    
    def test_strategy_generates_orders(self):
        """Test that strategy generates and submits orders."""
        # Create mock market data
        market_data = pd.DataFrame({
            '$close': [10.0, 20.0, 30.0],
            '$open': [10.0, 20.0, 30.0],
            '$high': [10.5, 20.5, 30.5],
            '$low': [9.5, 19.5, 29.5],
            '$volume': [1000000, 2000000, 3000000],
        }, index=pd.MultiIndex.from_product([
            ['000001.SZ', '000002.SZ', '000003.SZ'],
            [pd.Timestamp('2025-07-01')]
        ], names=['instrument', 'datetime']))
        
        date = pd.Timestamp('2025-07-01')
        
        # Initial state: no orders
        self.assertEqual(len(self.order_book.orders), 0)
        self.assertEqual(len(self.order_book.get_pending_orders()), 0)
        
        # Strategy generates orders
        self.strategy.on_bar(date, market_data)
        
        # Check that orders were submitted
        # Should have buy orders (no sell orders because no positions)
        all_orders = list(self.order_book.orders.values())
        self.assertGreater(len(all_orders), 0, "No orders were submitted")
        
        # Check order status
        pending_orders = self.order_book.get_pending_orders()
        self.assertGreater(len(pending_orders), 0, "No pending orders found")
        
        # All submitted orders should be PENDING
        for order in all_orders:
            self.assertEqual(order.status, OrderStatus.PENDING, 
                           f"Order {order.order_id} status is {order.status}, expected PENDING")
        
        # Check order details
        buy_orders = [o for o in all_orders if o.side.value == "BUY"]
        self.assertGreater(len(buy_orders), 0, "No buy orders found")
        
        for order in buy_orders:
            self.assertIsNotNone(order.symbol)
            self.assertIsNotNone(order.target_value)
            self.assertGreater(order.target_value, 0)


if __name__ == "__main__":
    unittest.main()

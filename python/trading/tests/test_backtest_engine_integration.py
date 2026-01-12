"""
Integration test for backtest engine.

Tests the complete flow: strategy -> order submission -> order execution.
"""

import unittest
from unittest.mock import Mock, MagicMock, patch
import pandas as pd

from trading.strategy import DualModelStrategy
from trading.strategy.buy_models import IBuyModel
from trading.strategy.sell_models import ISellModel
from trading.state import Account, PositionManager, OrderBook, OrderStatus, OrderSide
from trading.logic import RiskManager, PositionAllocator
from trading.storage import MemoryStorage
from trading.backtest import run_custom_backtest


class MockBuyModel(IBuyModel):
    """Mock buy model for testing."""
    
    def generate_ranks(self, date, market_data, **kwargs):
        """Generate mock rankings."""
        return pd.DataFrame({
            'symbol': ['000001.SZ', '000002.SZ'],
            'score': [0.9, 0.8],
            'rank': [1, 2],
        })


class MockSellModel(ISellModel):
    """Mock sell model for testing."""
    
    @property
    def threshold(self):
        return 0.65
    
    def predict_exit(self, position, market_data, date, **kwargs):
        """Always return low probability (no exit)."""
        return 0.3


class TestBacktestEngineIntegration(unittest.TestCase):
    """Test complete backtest engine integration."""
    
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
    
    @patch('trading.backtest.engine.D')
    def test_order_submission_and_execution(self, mock_D):
        """Test that orders submitted by strategy are visible to executor."""
        # Mock Qlib data
        calendar = [pd.Timestamp('2025-07-01'), pd.Timestamp('2025-07-02')]
        mock_D.calendar.return_value = calendar
        mock_D.instruments.return_value = ['000001.SZ', '000002.SZ']
        
        # Create mock market data
        market_data = pd.DataFrame({
            '$close': [10.0, 20.0, 10.5, 20.5],
            '$open': [10.0, 20.0, 10.5, 20.5],
            '$high': [10.5, 20.5, 11.0, 21.0],
            '$low': [9.5, 19.5, 10.0, 20.0],
            '$volume': [1000000, 2000000, 1100000, 2100000],
        }, index=pd.MultiIndex.from_product([
            ['000001.SZ', '000002.SZ'],
            [pd.Timestamp('2025-07-01'), pd.Timestamp('2025-07-02')]
        ], names=['instrument', 'datetime']))
        
        mock_D.features.return_value = market_data
        
        # Initial state: no orders
        self.assertEqual(len(self.order_book.orders), 0)
        
        # Run backtest
        results = run_custom_backtest(
            strategy=self.strategy,
            start_date="2025-07-01",
            end_date="2025-07-02",
            initial_cash=1000000.0,
        )
        
        # Verify that orders were submitted and executed
        # The key test: orders submitted by strategy should be in results
        all_orders = results['orders']
        self.assertGreater(len(all_orders), 0, "No orders found in results")
        
        # Verify order book is the same instance
        self.assertIs(results.get('order_book', None), self.order_book, 
                     "OrderBook instance mismatch - this is the bug!")
        
        # Verify orders are in the order book
        self.assertEqual(len(self.order_book.orders), len(all_orders),
                        "Order count mismatch between results and order_book")


if __name__ == "__main__":
    unittest.main()

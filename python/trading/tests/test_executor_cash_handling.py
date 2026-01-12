"""
Unit tests for Executor cash handling logic.

Tests the fix for frozen_cash vs available_cash deduction.
"""

import unittest
from unittest.mock import Mock, patch
import pandas as pd

from trading.state import Account, PositionManager, OrderBook, Order, OrderSide, OrderType, OrderStatus
from trading.execution import Executor, FillInfo
from trading.storage import MemoryStorage


class TestExecutorCashHandling(unittest.TestCase):
    """Test Executor cash handling logic."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.storage = MemoryStorage()
        self.account = Account(
            account_id="test_001",
            available_cash=100000.0,
            frozen_cash=0.0,
            initial_cash=100000.0,
        )
        self.position_manager = PositionManager(self.account, self.storage)
        self.order_book = OrderBook(self.storage)
        self.executor = Executor(
            position_manager=self.position_manager,
            order_book=self.order_book,
            commission_rate=0.0015,
            min_commission=5.0,
        )
    
    def test_execute_buy_order_without_frozen_cash(self):
        """Test that buy order execution works when frozen_cash is 0.
        
        This tests the fix: if frozen_cash is insufficient, use available_cash.
        """
        # Create a buy order
        order = Order(
            symbol="000001.SZ",
            side=OrderSide.BUY,
            amount=1000,
            order_type=OrderType.MARKET,
            date=pd.Timestamp("2025-07-01"),
        )
        self.order_book.submit(order)
        
        # Verify initial state: no frozen cash
        self.assertEqual(self.account.frozen_cash, 0.0)
        self.assertEqual(self.account.available_cash, 100000.0)
        
        # Create mock market data
        market_data = pd.DataFrame({
            '$close': [10.0],
            '$open': [10.0],
            '$high': [10.5],
            '$low': [9.5],
            '$volume': [1000000],
        }, index=pd.MultiIndex.from_product([
            ['000001.SZ'],
            [pd.Timestamp('2025-07-01')]
        ], names=['instrument', 'datetime']))
        
        # Execute order
        fill_info = self.executor.execute_order(order, market_data, pd.Timestamp('2025-07-01'))
        
        # Verify order was filled
        self.assertIsNotNone(fill_info, "Order should be filled")
        self.assertEqual(fill_info.symbol, "000001.SZ")
        self.assertEqual(fill_info.amount, 1000)
        
        # Verify cash was deducted from available_cash (not frozen_cash)
        # Expected cost: 1000 * 10.0 * (1 + 0.0015) + commission = 10015 + 5 = 10020
        expected_cost = 1000 * 10.0 * (1 + 0.0015) + 5.0  # price * amount * (1 + commission_rate) + min_commission
        self.assertAlmostEqual(
            self.account.available_cash,
            100000.0 - expected_cost,
            places=2,
            msg="Available cash should be deducted"
        )
        self.assertEqual(self.account.frozen_cash, 0.0, "Frozen cash should remain 0")
        
        # Verify position was created
        self.assertIn("000001.SZ", self.position_manager.all_positions)
        position = self.position_manager.all_positions["000001.SZ"]
        self.assertEqual(position.amount, 1000)
        self.assertEqual(position.entry_price, 10.0)
    
    def test_execute_buy_order_with_frozen_cash(self):
        """Test that buy order execution uses frozen_cash when available."""
        # Create a buy order
        order = Order(
            symbol="000001.SZ",
            side=OrderSide.BUY,
            amount=1000,
            order_type=OrderType.MARKET,
            date=pd.Timestamp("2025-07-01"),
        )
        self.order_book.submit(order)
        
        # Freeze cash for the order (simulating order submission with cash freeze)
        expected_cost = 1000 * 10.0 * (1 + 0.0015) + 5.0
        self.account.freeze_cash(expected_cost)
        
        # Verify frozen cash
        self.assertGreater(self.account.frozen_cash, 0.0)
        frozen_before = self.account.frozen_cash
        available_before = self.account.available_cash
        
        # Create mock market data
        market_data = pd.DataFrame({
            '$close': [10.0],
            '$open': [10.0],
            '$high': [10.5],
            '$low': [9.5],
            '$volume': [1000000],
        }, index=pd.MultiIndex.from_product([
            ['000001.SZ'],
            [pd.Timestamp('2025-07-01')]
        ], names=['instrument', 'datetime']))
        
        # Execute order
        fill_info = self.executor.execute_order(order, market_data, pd.Timestamp('2025-07-01'))
        
        # Verify order was filled
        self.assertIsNotNone(fill_info)
        
        # Verify cash was deducted from frozen_cash
        self.assertAlmostEqual(
            self.account.frozen_cash,
            frozen_before - expected_cost,
            places=2,
            msg="Frozen cash should be deducted"
        )
        self.assertEqual(
            self.account.available_cash,
            available_before,
            msg="Available cash should not change when using frozen cash"
        )
    
    def test_execute_buy_order_with_target_value(self):
        """Test buy order execution with target_value (no frozen cash)."""
        # Create a buy order with target_value
        order = Order(
            symbol="000001.SZ",
            side=OrderSide.BUY,
            amount=0,  # Will be calculated from target_value
            target_value=50000.0,  # Target 50k value
            order_type=OrderType.MARKET,
            date=pd.Timestamp("2025-07-01"),
        )
        self.order_book.submit(order)
        
        # Verify initial state: no frozen cash
        self.assertEqual(self.account.frozen_cash, 0.0)
        initial_cash = self.account.available_cash
        
        # Create mock market data
        market_data = pd.DataFrame({
            '$close': [10.0],
            '$open': [10.0],
            '$high': [10.5],
            '$low': [9.5],
            '$volume': [1000000],
        }, index=pd.MultiIndex.from_product([
            ['000001.SZ'],
            [pd.Timestamp('2025-07-01')]
        ], names=['instrument', 'datetime']))
        
        # Execute order
        fill_info = self.executor.execute_order(order, market_data, pd.Timestamp('2025-07-01'))
        
        # Verify order was filled
        self.assertIsNotNone(fill_info)
        
        # Verify fill amount is calculated from target_value
        # Expected: 50000 / 10.0 / 1.0015 / 100 * 100 = 4900 shares (rounded to 100)
        expected_shares = int(50000.0 / 10.0 / (1 + 0.0015) / 100) * 100
        self.assertEqual(fill_info.amount, expected_shares)
        
        # Verify cash was deducted from available_cash
        expected_cost = expected_shares * 10.0 * (1 + 0.0015) + 5.0
        self.assertAlmostEqual(
            self.account.available_cash,
            initial_cash - expected_cost,
            places=2,
            msg="Available cash should be deducted"
        )
        self.assertEqual(self.account.frozen_cash, 0.0)
    
    def test_cash_availability_check_uses_total_cash(self):
        """Test that cash availability check uses available_cash + frozen_cash."""
        # Set up account with some frozen cash
        self.account.available_cash = 50000.0
        self.account.frozen_cash = 50000.0
        
        # Create a buy order with target_value that requires more than available_cash alone
        order = Order(
            symbol="000001.SZ",
            side=OrderSide.BUY,
            amount=0,
            target_value=80000.0,  # More than available_cash (50k) but less than total (100k)
            order_type=OrderType.MARKET,
            date=pd.Timestamp("2025-07-01"),
        )
        self.order_book.submit(order)
        
        # Create mock market data
        market_data = pd.DataFrame({
            '$close': [10.0],
            '$open': [10.0],
            '$high': [10.5],
            '$low': [9.5],
            '$volume': [1000000],
        }, index=pd.MultiIndex.from_product([
            ['000001.SZ'],
            [pd.Timestamp('2025-07-01')]
        ], names=['instrument', 'datetime']))
        
        # Execute order - should succeed because total cash (100k) > target (80k)
        fill_info = self.executor.execute_order(order, market_data, pd.Timestamp('2025-07-01'))
        
        # Verify order was filled (not rejected)
        self.assertIsNotNone(fill_info, "Order should be filled when total cash is sufficient")
        self.assertNotEqual(order.status, OrderStatus.REJECTED)
    
    def test_insufficient_cash_rejects_order(self):
        """Test that order is rejected when total cash is insufficient."""
        # Set up account with limited cash
        self.account.available_cash = 1000.0
        self.account.frozen_cash = 0.0
        
        # Create a buy order that requires more cash than available
        order = Order(
            symbol="000001.SZ",
            side=OrderSide.BUY,
            amount=0,
            target_value=50000.0,  # Much more than available (1k)
            order_type=OrderType.MARKET,
            date=pd.Timestamp("2025-07-01"),
        )
        self.order_book.submit(order)
        
        # Create mock market data
        market_data = pd.DataFrame({
            '$close': [10.0],
            '$open': [10.0],
            '$high': [10.5],
            '$low': [9.5],
            '$volume': [1000000],
        }, index=pd.MultiIndex.from_product([
            ['000001.SZ'],
            [pd.Timestamp('2025-07-01')]
        ], names=['instrument', 'datetime']))
        
        # Execute order - should be rejected
        fill_info = self.executor.execute_order(order, market_data, pd.Timestamp('2025-07-01'))
        
        # Verify order was rejected
        self.assertIsNone(fill_info, "Order should be rejected when cash is insufficient")
        self.assertEqual(order.status, OrderStatus.REJECTED)
        self.assertEqual(self.account.available_cash, 1000.0, "Cash should not be deducted")
    
    def test_sell_order_adds_cash(self):
        """Test that sell order execution adds cash to available_cash."""
        # First, create a position
        self.position_manager.add_position(
            symbol="000001.SZ",
            entry_date=pd.Timestamp("2025-06-01"),
            entry_price=8.0,
            amount=1000,
        )
        
        initial_cash = self.account.available_cash
        
        # Create a sell order
        order = Order(
            symbol="000001.SZ",
            side=OrderSide.SELL,
            amount=1000,
            order_type=OrderType.MARKET,
            date=pd.Timestamp("2025-07-01"),
        )
        self.order_book.submit(order)
        
        # Create mock market data
        market_data = pd.DataFrame({
            '$close': [10.0],
            '$open': [10.0],
            '$high': [10.5],
            '$low': [9.5],
            '$volume': [1000000],
        }, index=pd.MultiIndex.from_product([
            ['000001.SZ'],
            [pd.Timestamp('2025-07-01')]
        ], names=['instrument', 'datetime']))
        
        # Execute order
        fill_info = self.executor.execute_order(order, market_data, pd.Timestamp('2025-07-01'))
        
        # Verify order was filled
        self.assertIsNotNone(fill_info)
        
        # Verify cash was added to available_cash
        # Expected proceeds: 1000 * 10.0 - commission = 10000 - 5 = 9995
        expected_proceeds = 1000 * 10.0 - 5.0
        self.assertAlmostEqual(
            self.account.available_cash,
            initial_cash + expected_proceeds,
            places=2,
            msg="Available cash should be increased"
        )
        
        # Verify position was reduced
        self.assertNotIn("000001.SZ", self.position_manager.all_positions, "Position should be closed")


if __name__ == "__main__":
    unittest.main()

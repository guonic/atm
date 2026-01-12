"""
Test NaN handling in Executor.execute_order.

Tests various scenarios where NaN values might occur:
1. Market data with NaN prices
2. Orders with NaN target_value
3. Account with NaN cash values
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import unittest
from unittest.mock import Mock, patch
import pandas as pd
import numpy as np

from trading.execution import Executor
from trading.state import Account, PositionManager, OrderBook, Order, OrderSide, OrderType, OrderStatus
from trading.storage import MemoryStorage


class TestExecutorNaNHandling(unittest.TestCase):
    """Test NaN handling in Executor."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.storage = MemoryStorage()
        self.account = Account(
            account_id="test",
            available_cash=100000.0,
            initial_cash=100000.0,
        )
        self.position_manager = PositionManager(self.account, self.storage)
        self.order_book = OrderBook(self.storage)
        self.executor = Executor(self.position_manager, self.order_book)
    
    def test_nan_price_in_market_data(self):
        """Test handling when market data has NaN price."""
        print("\nTest 1: NaN price in market data")
        
        # Create order
        order = Order(
            symbol="000001.SZ",
            side=OrderSide.BUY,
            amount=100,
            order_type=OrderType.MARKET,
            date=pd.Timestamp("2025-07-01"),
            target_value=10000.0,
        )
        self.order_book.submit(order)
        
        # Create market data with NaN price
        market_data = pd.DataFrame({
            '$open': [np.nan],  # NaN price
            '$high': [10.5],
            '$low': [9.5],
            '$close': [10.0],
            '$volume': [1000000],
        }, index=pd.MultiIndex.from_product([
            ["000001.SZ"],
            [pd.Timestamp("2025-07-01")]
        ], names=['instrument', 'datetime']))
        
        date = pd.Timestamp("2025-07-01")
        
        # Execute order - should handle NaN gracefully
        result = self.executor.execute_order(order, market_data, date)
        
        # Should reject order due to invalid price
        self.assertIsNone(result, "Order should be rejected when price is NaN")
        self.assertEqual(order.status, OrderStatus.REJECTED, "Order status should be REJECTED")
        print("  ✓ PASSED: NaN price handled correctly (order rejected)")
    
    def test_nan_target_value_in_order(self):
        """Test handling when order has NaN target_value."""
        print("\nTest 2: NaN target_value in order")
        
        # Create order with NaN target_value
        order = Order(
            symbol="000001.SZ",
            side=OrderSide.BUY,
            amount=100,
            order_type=OrderType.MARKET,
            date=pd.Timestamp("2025-07-01"),
            target_value=np.nan,  # NaN target_value
        )
        self.order_book.submit(order)
        
        # Create valid market data
        market_data = pd.DataFrame({
            '$open': [10.0],
            '$high': [10.5],
            '$low': [9.5],
            '$close': [10.0],
            '$volume': [1000000],
        }, index=pd.MultiIndex.from_product([
            ["000001.SZ"],
            [pd.Timestamp("2025-07-01")]
        ], names=['instrument', 'datetime']))
        
        date = pd.Timestamp("2025-07-01")
        
        # Execute order - should handle NaN gracefully
        try:
            result = self.executor.execute_order(order, market_data, date)
            # Should either reject or handle NaN
            if result is None:
                self.assertEqual(order.status, OrderStatus.REJECTED)
                print("  ✓ PASSED: NaN target_value handled correctly (order rejected)")
            else:
                # If not rejected, should use order.amount instead
                self.assertIsNotNone(result)
                print("  ✓ PASSED: NaN target_value handled correctly (used order.amount)")
        except (ValueError, TypeError) as e:
            if "NaN" in str(e) or "nan" in str(e).lower():
                self.fail(f"NaN not handled gracefully: {e}")
            else:
                raise
    
    def test_nan_cash_in_account(self):
        """Test handling when account has NaN cash values."""
        print("\nTest 3: NaN cash in account")
        
        # Create account with NaN cash
        account = Account(
            account_id="test",
            available_cash=np.nan,  # NaN cash
            initial_cash=100000.0,
        )
        position_manager = PositionManager(account, self.storage)
        executor = Executor(position_manager, self.order_book)
        
        # Create order
        order = Order(
            symbol="000001.SZ",
            side=OrderSide.BUY,
            amount=100,
            order_type=OrderType.MARKET,
            date=pd.Timestamp("2025-07-01"),
            target_value=10000.0,
        )
        self.order_book.submit(order)
        
        # Create valid market data
        market_data = pd.DataFrame({
            '$open': [10.0],
            '$high': [10.5],
            '$low': [9.5],
            '$close': [10.0],
            '$volume': [1000000],
        }, index=pd.MultiIndex.from_product([
            ["000001.SZ"],
            [pd.Timestamp("2025-07-01")]
        ], names=['instrument', 'datetime']))
        
        date = pd.Timestamp("2025-07-01")
        
        # Execute order - should handle NaN gracefully
        try:
            result = executor.execute_order(order, market_data, date)
            # Should reject due to insufficient cash (NaN is treated as 0 or invalid)
            if result is None:
                self.assertEqual(order.status, OrderStatus.REJECTED)
                print("  ✓ PASSED: NaN cash handled correctly (order rejected)")
            else:
                # Should not happen, but if it does, it's a bug
                self.fail("Order should be rejected when account has NaN cash")
        except (ValueError, TypeError) as e:
            if "NaN" in str(e) or "nan" in str(e).lower():
                self.fail(f"NaN not handled gracefully: {e}")
            else:
                raise
    
    def test_missing_price_field(self):
        """Test handling when price field is missing from market data."""
        print("\nTest 4: Missing price field in market data")
        
        # Create order
        order = Order(
            symbol="000001.SZ",
            side=OrderSide.BUY,
            amount=100,
            order_type=OrderType.MARKET,
            date=pd.Timestamp("2025-07-01"),
            target_value=10000.0,
        )
        self.order_book.submit(order)
        
        # Create market data without $open or open field
        market_data = pd.DataFrame({
            '$high': [10.5],
            '$low': [9.5],
            '$close': [10.0],
            '$volume': [1000000],
            # Missing $open and open
        }, index=pd.MultiIndex.from_product([
            ["000001.SZ"],
            [pd.Timestamp("2025-07-01")]
        ], names=['instrument', 'datetime']))
        
        date = pd.Timestamp("2025-07-01")
        
        # Execute order - should use default value (0) and reject
        result = self.executor.execute_order(order, market_data, date)
        
        # Should reject order due to invalid price (0)
        self.assertIsNone(result, "Order should be rejected when price is 0")
        self.assertEqual(order.status, OrderStatus.REJECTED, "Order status should be REJECTED")
        print("  ✓ PASSED: Missing price field handled correctly (order rejected)")
    
    def test_zero_price(self):
        """Test handling when price is 0."""
        print("\nTest 5: Zero price in market data")
        
        # Create order
        order = Order(
            symbol="000001.SZ",
            side=OrderSide.BUY,
            amount=100,
            order_type=OrderType.MARKET,
            date=pd.Timestamp("2025-07-01"),
            target_value=10000.0,
        )
        self.order_book.submit(order)
        
        # Create market data with zero price
        market_data = pd.DataFrame({
            '$open': [0.0],  # Zero price
            '$high': [10.5],
            '$low': [9.5],
            '$close': [10.0],
            '$volume': [1000000],
        }, index=pd.MultiIndex.from_product([
            ["000001.SZ"],
            [pd.Timestamp("2025-07-01")]
        ], names=['instrument', 'datetime']))
        
        date = pd.Timestamp("2025-07-01")
        
        # Execute order - should reject
        result = self.executor.execute_order(order, market_data, date)
        
        # Should reject order due to invalid price (0)
        self.assertIsNone(result, "Order should be rejected when price is 0")
        self.assertEqual(order.status, OrderStatus.REJECTED, "Order status should be REJECTED")
        print("  ✓ PASSED: Zero price handled correctly (order rejected)")
    
    def test_valid_scenario(self):
        """Test that valid scenario still works."""
        print("\nTest 6: Valid scenario (should work)")
        
        # Create order
        order = Order(
            symbol="000001.SZ",
            side=OrderSide.BUY,
            amount=100,
            order_type=OrderType.MARKET,
            date=pd.Timestamp("2025-07-01"),
            target_value=10000.0,
        )
        self.order_book.submit(order)
        
        # Create valid market data
        market_data = pd.DataFrame({
            '$open': [10.0],
            '$high': [10.5],
            '$low': [9.5],
            '$close': [10.0],
            '$volume': [1000000],
        }, index=pd.MultiIndex.from_product([
            ["000001.SZ"],
            [pd.Timestamp("2025-07-01")]
        ], names=['instrument', 'datetime']))
        
        date = pd.Timestamp("2025-07-01")
        
        # Execute order - should succeed
        result = self.executor.execute_order(order, market_data, date)
        
        # Should fill order
        self.assertIsNotNone(result, "Order should be filled with valid data")
        self.assertEqual(order.status, OrderStatus.FILLED, "Order status should be FILLED")
        self.assertGreater(result.amount, 0, "Fill amount should be positive")
        self.assertGreater(result.price, 0, "Fill price should be positive")
        print(f"  ✓ PASSED: Valid scenario works (filled {result.amount} @ {result.price:.2f})")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing Executor NaN Handling")
    print("=" * 60)
    
    suite = unittest.TestLoader().loadTestsFromTestCase(TestExecutorNaNHandling)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "=" * 60)
    print(f"Results: {result.testsRun} tests, {len(result.failures)} failures, {len(result.errors)} errors")
    print("=" * 60)
    
    return len(result.failures) == 0 and len(result.errors) == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

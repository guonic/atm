"""
Test position update fixes.

Tests that:
1. pd variable scope issue is fixed (no "cannot access local variable 'pd'" error)
2. Position updates work correctly with NaN filtering
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import unittest
import pandas as pd
import numpy as np

from trading.state import Position, PositionManager, Account
from trading.storage import MemoryStorage


class TestPositionUpdateFix(unittest.TestCase):
    """Test position update fixes."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.storage = MemoryStorage()
        self.account = Account(
            account_id="test_001",
            available_cash=1000000.0,
            initial_cash=1000000.0,
        )
        self.position_manager = PositionManager(self.account, self.storage)
    
    def test_pd_scope_issue_fixed(self):
        """Test that pd variable scope issue is fixed."""
        print("\nTest 1: pd variable scope issue fixed")
        
        # Create a position
        position = Position(
            symbol="000001.SZ",
            entry_date=pd.Timestamp("2025-07-01"),
            entry_price=10.0,
            amount=1000,
            high_price_since_entry=10.0,
            high_date=pd.Timestamp("2025-07-01"),
        )
        self.position_manager.positions["000001.SZ"] = position
        
        # Create market data with valid prices
        dates = pd.date_range("2025-07-01", periods=5, freq="D")
        market_data = pd.DataFrame({
            ('000001.SZ', '$high'): [10.5, 11.0, 11.5, 12.0, 12.5],
            ('000001.SZ', '$close'): [10.0, 10.5, 11.0, 11.5, 12.0],
        }, index=dates)
        market_data.columns = pd.MultiIndex.from_tuples(market_data.columns)
        market_data = market_data.stack(level=0).swaplevel(0, 1).sort_index()
        market_data.index.names = ['instrument', 'datetime']
        
        # Update positions - should not raise "cannot access local variable 'pd'" error
        try:
            self.position_manager.update_positions(
                date=pd.Timestamp("2025-07-02"),
                market_data=market_data
            )
            print("  ✓ PASSED: No pd scope error")
        except UnboundLocalError as e:
            if "pd" in str(e):
                self.fail(f"pd variable scope issue not fixed: {e}")
            raise
    
    def test_position_update_with_nan_filtered_date(self):
        """Test position update when date is filtered out due to NaN."""
        print("\nTest 2: Position update with NaN filtered date")
        
        # Create a position
        position = Position(
            symbol="000001.SZ",
            entry_date=pd.Timestamp("2025-07-01"),
            entry_price=10.0,
            amount=1000,
            high_price_since_entry=10.0,
            high_date=pd.Timestamp("2025-07-01"),
        )
        self.position_manager.positions["000001.SZ"] = position
        
        # Create market data where 2025-07-02 is missing (filtered out)
        dates = pd.date_range("2025-07-01", periods=5, freq="D")
        market_data = pd.DataFrame({
            ('000001.SZ', '$high'): [10.5, 11.0, 11.5, 12.0, 12.5],
            ('000001.SZ', '$close'): [10.0, 10.5, 11.0, 11.5, 12.0],
        }, index=dates)
        market_data.columns = pd.MultiIndex.from_tuples(market_data.columns)
        market_data = market_data.stack(level=0).swaplevel(0, 1).sort_index()
        market_data.index.names = ['instrument', 'datetime']
        
        # Remove 2025-07-02 (simulating NaN filtering)
        market_data = market_data.drop(pd.Timestamp("2025-07-02"), level=1)
        
        # Update positions for 2025-07-02 - should skip gracefully
        try:
            self.position_manager.update_positions(
                date=pd.Timestamp("2025-07-02"),
                market_data=market_data
            )
            print("  ✓ PASSED: Position update skipped missing date gracefully")
        except Exception as e:
            self.fail(f"Position update failed for missing date: {e}")
    
    def test_position_update_with_nan_prices(self):
        """Test position update when prices are NaN."""
        print("\nTest 3: Position update with NaN prices")
        
        # Create a position
        position = Position(
            symbol="000001.SZ",
            entry_date=pd.Timestamp("2025-07-01"),
            entry_price=10.0,
            amount=1000,
            high_price_since_entry=10.0,
            high_date=pd.Timestamp("2025-07-01"),
        )
        self.position_manager.positions["000001.SZ"] = position
        
        # Create market data with NaN prices
        dates = pd.date_range("2025-07-01", periods=3, freq="D")
        market_data = pd.DataFrame({
            ('000001.SZ', '$high'): [10.5, np.nan, 11.5],
            ('000001.SZ', '$close'): [10.0, np.nan, 11.0],
        }, index=dates)
        market_data.columns = pd.MultiIndex.from_tuples(market_data.columns)
        market_data = market_data.stack(level=0).swaplevel(0, 1).sort_index()
        market_data.index.names = ['instrument', 'datetime']
        
        # Update positions for 2025-07-02 (with NaN) - should skip gracefully
        try:
            self.position_manager.update_positions(
                date=pd.Timestamp("2025-07-02"),
                market_data=market_data
            )
            print("  ✓ PASSED: Position update skipped NaN prices gracefully")
        except Exception as e:
            self.fail(f"Position update failed for NaN prices: {e}")


class TestBacktestEnginePriceLookup(unittest.TestCase):
    """Test backtest engine price lookup fixes."""
    
    def test_get_position_price_with_filtered_date(self):
        """Test _get_position_price when date is filtered out."""
        print("\nTest 4: Backtest engine price lookup with filtered date")
        
        from trading.backtest.engine import _get_position_price
        
        # Create market data where 2025-07-02 is missing (filtered out)
        dates = pd.date_range("2025-07-01", periods=5, freq="D")
        market_data = pd.DataFrame({
            ('000001.SZ', '$close'): [10.0, 10.5, 11.0, 11.5, 12.0],
        }, index=dates)
        market_data.columns = pd.MultiIndex.from_tuples(market_data.columns)
        market_data = market_data.stack(level=0).swaplevel(0, 1).sort_index()
        market_data.index.names = ['instrument', 'datetime']
        
        # Remove 2025-07-02 (simulating NaN filtering)
        market_data = market_data.drop(pd.Timestamp("2025-07-02"), level=1)
        
        # Get price for existing date
        price = _get_position_price("000001.SZ", pd.Timestamp("2025-07-01"), market_data)
        self.assertEqual(price, 10.0)
        print("  ✓ PASSED: Found price for existing date")
        
        # Get price for filtered date - should return 0.0
        price = _get_position_price("000001.SZ", pd.Timestamp("2025-07-02"), market_data)
        self.assertEqual(price, 0.0)
        print("  ✓ PASSED: Returned 0.0 for filtered date")
        
        # Get price for non-existent symbol - should return 0.0
        price = _get_position_price("999999.SZ", pd.Timestamp("2025-07-01"), market_data)
        self.assertEqual(price, 0.0)
        print("  ✓ PASSED: Returned 0.0 for non-existent symbol")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing Position Update Fixes")
    print("=" * 60)
    
    suite = unittest.TestLoader().loadTestsFromTestCase(TestPositionUpdateFix)
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestBacktestEnginePriceLookup))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "=" * 60)
    print(f"Results: {result.testsRun} tests, {len(result.failures)} failures, {len(result.errors)} errors")
    print("=" * 60)
    
    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"  {test}: {traceback}")
    
    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"  {test}: {traceback}")
    
    return len(result.failures) == 0 and len(result.errors) == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

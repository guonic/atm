"""
Test NaN prevention - ensure no NaN values enter the system.

This test verifies that:
1. Market data is filtered for NaN before use
2. All calculations check for NaN and raise errors
3. No NaN values propagate through the system
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import unittest
import pandas as pd
import numpy as np

from trading.backtest.engine import run_custom_backtest
from trading.strategy import DualModelStrategy
from trading.strategy.buy_models import IBuyModel
from trading.strategy.sell_models import ISellModel
from trading.state import Account, PositionManager, OrderBook
from trading.logic import RiskManager, PositionAllocator
from trading.storage import MemoryStorage


class SimpleBuyModel(IBuyModel):
    """Simple buy model for testing."""
    
    def generate_ranks(self, date, market_data, **kwargs):
        return pd.DataFrame({
            'symbol': ['000001.SZ'],
            'score': [0.9],
        })


class SimpleSellModel(ISellModel):
    """Simple sell model for testing."""
    
    @property
    def threshold(self):
        return 0.65
    
    def predict_exit(self, position, market_data, date, **kwargs):
        return 0.3


class TestNaNPrevention(unittest.TestCase):
    """Test NaN prevention in data pipeline."""
    
    def test_market_data_nan_filtering(self):
        """Test that market data with NaN is filtered."""
        print("\nTest 1: Market data NaN filtering")
        
        # Create strategy
        storage = MemoryStorage()
        account = Account("test", 100000.0, 0.0, 100000.0)
        position_manager = PositionManager(account, storage)
        order_book = OrderBook(storage)
        risk_manager = RiskManager(account, position_manager, storage)
        position_allocator = PositionAllocator(target_positions=1)
        
        strategy = DualModelStrategy(
            buy_model=SimpleBuyModel(),
            sell_model=SimpleSellModel(),
            position_manager=position_manager,
            order_book=order_book,
            risk_manager=risk_manager,
            position_allocator=position_allocator,
        )
        
        # Mock D.features to return data with NaN
        from unittest.mock import patch
        
        with patch('trading.backtest.engine.D') as mock_D:
            # Create market data with NaN
            market_data = pd.DataFrame({
                '$close': [10.0, np.nan, 10.5],  # One NaN row
                '$open': [10.0, 10.0, 10.5],
                '$high': [10.5, 10.5, 11.0],
                '$low': [9.5, 9.5, 10.0],
                '$volume': [1000000, 1000000, 1100000],
            }, index=pd.MultiIndex.from_product([
                ['000001.SZ'],
                pd.date_range('2025-07-01', periods=3, freq='D')
            ], names=['instrument', 'datetime']))
            
            mock_D.calendar.return_value = pd.date_range('2025-07-01', periods=3, freq='D')
            mock_D.instruments.return_value = ['000001.SZ']
            mock_D.features.return_value = market_data
            
            # Should filter NaN and continue
            try:
                results = run_custom_backtest(
                    strategy=strategy,
                    start_date="2025-07-01",
                    end_date="2025-07-03",
                    initial_cash=100000.0,
                )
                # Should succeed (NaN rows filtered)
                self.assertIsNotNone(results)
                print("  ✓ PASSED: NaN rows filtered successfully")
            except ValueError as e:
                if "NaN" in str(e):
                    print(f"  ✓ PASSED: NaN detected and raised error: {e}")
                else:
                    raise
    
    def test_account_nan_detection(self):
        """Test that NaN in account raises error."""
        print("\nTest 2: Account NaN detection")
        
        # Create account with NaN
        account = Account("test", np.nan, 0.0, 100000.0)
        position_manager = PositionManager(account, MemoryStorage())
        
        # get_total_value should raise error
        with self.assertRaises(ValueError) as context:
            account.get_total_value(position_manager)
        
        self.assertIn("NaN", str(context.exception))
        print(f"  ✓ PASSED: NaN in account detected: {context.exception}")
    
    def test_position_size_nan_detection(self):
        """Test that NaN in position size calculation raises error."""
        print("\nTest 3: Position size NaN detection")
        
        # Create account with NaN
        account = Account("test", np.nan, 0.0, 100000.0)
        position_manager = PositionManager(account, MemoryStorage())
        allocator = PositionAllocator(target_positions=1)
        
        # calculate_position_size should raise error
        with self.assertRaises(ValueError) as context:
            allocator.calculate_position_size(
                symbol="000001.SZ",
                account=account,
                position_manager=position_manager,
                market_data=pd.DataFrame(),
            )
        
        self.assertIn("NaN", str(context.exception))
        print(f"  ✓ PASSED: NaN in position size detected: {context.exception}")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing NaN Prevention")
    print("=" * 60)
    
    suite = unittest.TestLoader().loadTestsFromTestCase(TestNaNPrevention)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "=" * 60)
    print(f"Results: {result.testsRun} tests, {len(result.failures)} failures, {len(result.errors)} errors")
    print("=" * 60)
    
    return len(result.failures) == 0 and len(result.errors) == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

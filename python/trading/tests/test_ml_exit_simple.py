"""
Simple test script to verify MLExitSellModel data extraction fix.
Run with: python python/trading/tests/test_ml_exit_simple.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from unittest.mock import Mock, patch
import pandas as pd
import numpy as np

from trading.strategy.sell_models.ml_exit import MLExitSellModel
from nq.analysis.exit import ExitModel


def test_single_instrument_format():
    """Test single instrument format (most common case)."""
    print("Test 1: Single instrument format")
    
    # Create mock ExitModel
    mock_exit_model = Mock(spec=ExitModel)
    mock_exit_model.predict_proba = Mock(return_value=np.array([0.7]))
    
    # Create MLExitSellModel
    sell_model = MLExitSellModel(exit_model=mock_exit_model, threshold=0.65)
    
    # Create mock Position
    position = Mock()
    position.symbol = "000001.SZ"
    position.entry_price = 10.0
    position.entry_date = pd.Timestamp("2025-07-01")
    position.high_price_since_entry = 12.0
    position.amount = 1000
    
    # Create mock data in single instrument format (columns are field names)
    dates = pd.date_range("2025-07-01", periods=10, freq="D")
    hist_data = pd.DataFrame({
        '$close': np.linspace(10.0, 12.0, 10),
        '$high': np.linspace(11.0, 13.0, 10),
        '$low': np.linspace(9.0, 11.0, 10),
        '$volume': np.full(10, 1000000),
    }, index=dates)
    
    with patch('trading.strategy.sell_models.ml_exit.D') as mock_D:
        mock_D.features.return_value = hist_data
        
        date = pd.Timestamp("2025-07-10")
        risk_prob = sell_model.predict_exit(
            position=position,
            market_data=None,
            date=date
        )
        
        # Verify
        assert risk_prob == 0.7, f"Expected 0.7, got {risk_prob}"
        assert mock_exit_model.predict_proba.called, "predict_proba should be called"
        
        # Verify daily_df was passed correctly
        call_args = mock_exit_model.predict_proba.call_args[1]
        daily_df = call_args['daily_df']
        assert 'close' in daily_df.columns, "daily_df should have 'close' column"
        assert 'high' in daily_df.columns, "daily_df should have 'high' column"
        assert len(daily_df) == 10, f"Expected 10 rows, got {len(daily_df)}"
        
        print("  ✓ PASSED: Single instrument format works correctly")
        return True


def test_multiindex_columns_format():
    """Test MultiIndex columns format."""
    print("Test 2: MultiIndex columns format")
    
    # Create mock ExitModel
    mock_exit_model = Mock(spec=ExitModel)
    mock_exit_model.predict_proba = Mock(return_value=np.array([0.8]))
    
    # Create MLExitSellModel
    sell_model = MLExitSellModel(exit_model=mock_exit_model, threshold=0.65)
    
    # Create mock Position
    position = Mock()
    position.symbol = "000001.SZ"
    position.entry_price = 10.0
    position.entry_date = pd.Timestamp("2025-07-01")
    position.high_price_since_entry = 12.0
    position.amount = 1000
    
    # Create mock data in MultiIndex columns format
    dates = pd.date_range("2025-07-01", periods=10, freq="D")
    columns = pd.MultiIndex.from_product([
        ["000001.SZ"],
        ["$close", "$high", "$low", "$volume"]
    ])
    hist_data = pd.DataFrame(
        np.random.uniform(10.0, 13.0, (10, 4)),
        index=dates,
        columns=columns
    )
    hist_data[("000001.SZ", "$close")] = np.linspace(10.0, 12.0, 10)
    hist_data[("000001.SZ", "$high")] = np.linspace(11.0, 13.0, 10)
    hist_data[("000001.SZ", "$low")] = np.linspace(9.0, 11.0, 10)
    hist_data[("000001.SZ", "$volume")] = np.full(10, 1000000)
    
    with patch('trading.strategy.sell_models.ml_exit.D') as mock_D:
        mock_D.features.return_value = hist_data
        
        date = pd.Timestamp("2025-07-10")
        risk_prob = sell_model.predict_exit(
            position=position,
            market_data=None,
            date=date
        )
        
        # Verify
        assert risk_prob == 0.8, f"Expected 0.8, got {risk_prob}"
        assert mock_exit_model.predict_proba.called, "predict_proba should be called"
        
        # Verify daily_df was passed correctly
        call_args = mock_exit_model.predict_proba.call_args[1]
        daily_df = call_args['daily_df']
        assert 'close' in daily_df.columns, "daily_df should have 'close' column"
        assert len(daily_df) == 10, f"Expected 10 rows, got {len(daily_df)}"
        
        print("  ✓ PASSED: MultiIndex columns format works correctly")
        return True


def test_empty_data():
    """Test handling of empty data."""
    print("Test 3: Empty data handling")
    
    # Create mock ExitModel
    mock_exit_model = Mock(spec=ExitModel)
    
    # Create MLExitSellModel
    sell_model = MLExitSellModel(exit_model=mock_exit_model, threshold=0.65)
    
    # Create mock Position
    position = Mock()
    position.symbol = "000001.SZ"
    position.entry_price = 10.0
    position.entry_date = pd.Timestamp("2025-07-01")
    position.high_price_since_entry = 12.0
    position.amount = 1000
    
    with patch('trading.strategy.sell_models.ml_exit.D') as mock_D:
        mock_D.features.return_value = pd.DataFrame()
        
        date = pd.Timestamp("2025-07-10")
        risk_prob = sell_model.predict_exit(
            position=position,
            market_data=None,
            date=date
        )
        
        # Verify
        assert risk_prob == 0.0, f"Expected 0.0 for empty data, got {risk_prob}"
        assert not mock_exit_model.predict_proba.called, "predict_proba should not be called for empty data"
        
        print("  ✓ PASSED: Empty data handled correctly")
        return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing MLExitSellModel data extraction fix")
    print("=" * 60)
    print()
    
    tests = [
        test_single_instrument_format,
        test_multiindex_columns_format,
        test_empty_data,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
        print()
    
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

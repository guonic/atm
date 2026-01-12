"""
Test data validation utilities.

Tests that NaN filtering works correctly and provides detailed warnings.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import unittest
import pandas as pd
import numpy as np

from trading.utils.data_validation import validate_and_filter_nan, validate_single_instrument_data


class TestDataValidation(unittest.TestCase):
    """Test data validation functions."""
    
    def test_filter_nan_multiindex(self):
        """Test filtering NaN in MultiIndex DataFrame."""
        print("\nTest 1: Filter NaN in MultiIndex DataFrame")
        
        # Create market data with NaN
        dates = pd.date_range("2025-07-01", periods=5, freq="D")
        instruments = ["000001.SZ", "000002.SZ"]
        
        # Create data with some NaN rows
        data_dict = {}
        for inst in instruments:
            for field in ["$close", "$open", "$high", "$low", "$volume"]:
                data_dict[(inst, field)] = np.random.uniform(10.0, 20.0, 5)
        
        market_data = pd.DataFrame(data_dict, index=dates)
        
        # Add NaN to some rows
        market_data.loc[dates[1], (instruments[0], "$close")] = np.nan
        market_data.loc[dates[2], (instruments[1], "$open")] = np.nan
        market_data.loc[dates[3], (instruments[0], "$high")] = np.nan
        
        # Convert to MultiIndex columns
        market_data.columns = pd.MultiIndex.from_tuples(market_data.columns)
        
        # Convert to MultiIndex index
        market_data = market_data.stack(level=0).swaplevel(0, 1).sort_index()
        market_data.index.names = ['instrument', 'datetime']
        
        required_fields = ["$close", "$open", "$high", "$low", "$volume"]
        filtered_data, nan_details = validate_and_filter_nan(
            market_data=market_data,
            required_fields=required_fields,
            context="test data"
        )
        
        # Should have filtered out rows with NaN
        self.assertLess(len(filtered_data), len(market_data))
        self.assertGreater(len(nan_details), 0)
        print(f"  ✓ PASSED: Filtered {len(market_data) - len(filtered_data)} NaN rows")
    
    def test_filter_nan_single_index(self):
        """Test filtering NaN in single-index DataFrame."""
        print("\nTest 2: Filter NaN in single-index DataFrame")
        
        # Create market data with NaN
        dates = pd.date_range("2025-07-01", periods=5, freq="D")
        market_data = pd.DataFrame({
            '$close': [10.0, np.nan, 12.0, 13.0, 14.0],
            '$open': [10.0, 10.5, np.nan, 13.0, 14.0],
            '$high': [10.5, 11.0, 12.5, np.nan, 14.5],
            '$low': [9.5, 10.0, 11.5, 12.5, 13.5],
            '$volume': [1000000, 1100000, 1200000, 1300000, 1400000],
        }, index=dates)
        
        required_fields = ["$close", "$open", "$high", "$low", "$volume"]
        filtered_data, nan_details = validate_and_filter_nan(
            market_data=market_data,
            required_fields=required_fields,
            context="test data"
        )
        
        # Should have filtered out rows with NaN
        self.assertLess(len(filtered_data), len(market_data))
        self.assertGreater(len(nan_details), 0)
        print(f"  ✓ PASSED: Filtered {len(market_data) - len(filtered_data)} NaN rows")
    
    def test_no_nan(self):
        """Test that valid data passes through unchanged."""
        print("\nTest 3: No NaN in data")
        
        # Create valid market data
        dates = pd.date_range("2025-07-01", periods=5, freq="D")
        market_data = pd.DataFrame({
            '$close': [10.0, 11.0, 12.0, 13.0, 14.0],
            '$open': [10.0, 10.5, 11.5, 12.5, 13.5],
            '$high': [10.5, 11.5, 12.5, 13.5, 14.5],
            '$low': [9.5, 10.0, 11.0, 12.0, 13.0],
            '$volume': [1000000, 1100000, 1200000, 1300000, 1400000],
        }, index=dates)
        
        required_fields = ["$close", "$open", "$high", "$low", "$volume"]
        filtered_data, nan_details = validate_and_filter_nan(
            market_data=market_data,
            required_fields=required_fields,
            context="test data"
        )
        
        # Should pass through unchanged
        self.assertEqual(len(filtered_data), len(market_data))
        self.assertEqual(len(nan_details), 0)
        pd.testing.assert_frame_equal(filtered_data, market_data)
        print("  ✓ PASSED: Valid data passed through unchanged")
    
    def test_all_nan_raises_error(self):
        """Test that all NaN data raises error."""
        print("\nTest 4: All NaN data raises error")
        
        # Create all NaN data
        dates = pd.date_range("2025-07-01", periods=3, freq="D")
        market_data = pd.DataFrame({
            '$close': [np.nan, np.nan, np.nan],
            '$open': [np.nan, np.nan, np.nan],
            '$high': [np.nan, np.nan, np.nan],
            '$low': [np.nan, np.nan, np.nan],
            '$volume': [np.nan, np.nan, np.nan],
        }, index=dates)
        
        required_fields = ["$close", "$open", "$high", "$low", "$volume"]
        
        with self.assertRaises(ValueError) as context:
            validate_and_filter_nan(
                market_data=market_data,
                required_fields=required_fields,
                context="test data"
            )
        
        self.assertIn("All rows", str(context.exception))
        print(f"  ✓ PASSED: All NaN data raises error: {context.exception}")
    
    def test_validate_single_instrument(self):
        """Test single instrument validation."""
        print("\nTest 5: Single instrument validation")
        
        # Valid data
        dates = pd.date_range("2025-07-01", periods=5, freq="D")
        valid_data = pd.DataFrame({
            '$close': [10.0, 11.0, 12.0, 13.0, 14.0],
            '$high': [10.5, 11.5, 12.5, 13.5, 14.5],
            '$low': [9.5, 10.0, 11.0, 12.0, 13.0],
            '$volume': [1000000, 1100000, 1200000, 1300000, 1400000],
        }, index=dates)
        
        required_fields = ['$close', '$high', '$low', '$volume']
        result = validate_single_instrument_data(
            data=valid_data,
            required_fields=required_fields,
            symbol="000001.SZ",
            date=pd.Timestamp("2025-07-01")
        )
        self.assertTrue(result)
        print("  ✓ PASSED: Valid single instrument data")
        
        # Invalid data with NaN
        invalid_data = pd.DataFrame({
            '$close': [10.0, np.nan, 12.0],
            '$high': [10.5, 11.5, 12.5],
            '$low': [9.5, 10.0, 11.0],
            '$volume': [1000000, 1100000, 1200000],
        }, index=dates[:3])
        
        result = validate_single_instrument_data(
            data=invalid_data,
            required_fields=required_fields,
            symbol="000001.SZ",
            date=pd.Timestamp("2025-07-01")
        )
        self.assertFalse(result)
        print("  ✓ PASSED: Invalid single instrument data detected")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing Data Validation Utilities")
    print("=" * 60)
    
    suite = unittest.TestLoader().loadTestsFromTestCase(TestDataValidation)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "=" * 60)
    print(f"Results: {result.testsRun} tests, {len(result.failures)} failures, {len(result.errors)} errors")
    print("=" * 60)
    
    return len(result.failures) == 0 and len(result.errors) == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

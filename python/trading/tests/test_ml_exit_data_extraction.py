"""
Unit test for MLExitSellModel data extraction logic.

Tests that the model correctly extracts OHLCV data from Qlib's D.features()
which can return data in different formats:
1. Single instrument: columns are field names ($close, $high, etc.)
2. Multiple instruments: MultiIndex columns (instrument, field)
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import unittest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np

from trading.strategy.sell_models.ml_exit import MLExitSellModel
from trading.state import Position
from nq.analysis.exit import ExitModel


class TestMLExitDataExtraction(unittest.TestCase):
    """Test MLExitSellModel data extraction from Qlib."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create mock ExitModel
        self.mock_exit_model = Mock(spec=ExitModel)
        self.mock_exit_model.predict_proba = Mock(return_value=np.array([0.3, 0.4, 0.7]))
        
        # Create MLExitSellModel
        self.sell_model = MLExitSellModel(
            exit_model=self.mock_exit_model,
            threshold=0.65
        )
        
        # Create mock Position
        self.position = Mock(spec=Position)
        self.position.symbol = "000001.SZ"
        self.position.entry_price = 10.0
        self.position.entry_date = pd.Timestamp("2025-07-01")
        self.position.high_price_since_entry = 12.0
        self.position.amount = 1000
    
    @patch('trading.strategy.sell_models.ml_exit.D')
    def test_single_instrument_format(self, mock_D):
        """Test data extraction when Qlib returns single instrument format.
        
        Format: columns are field names ($close, $high, etc.), index is datetime.
        This is the most common case for single instrument queries.
        """
        # Create mock data in single instrument format
        dates = pd.date_range("2025-07-01", periods=10, freq="D")
        hist_data = pd.DataFrame({
            '$close': np.random.uniform(10.0, 12.0, 10),
            '$high': np.random.uniform(11.0, 13.0, 10),
            '$low': np.random.uniform(9.0, 11.0, 10),
            '$volume': np.random.uniform(1000000, 2000000, 10),
        }, index=dates)
        
        mock_D.features.return_value = hist_data
        
        # Test predict_exit
        date = pd.Timestamp("2025-07-10")
        risk_prob = self.sell_model.predict_exit(
            position=self.position,
            market_data=None,  # Not used in current implementation
            date=date
        )
        
        # Verify D.features was called correctly
        mock_D.features.assert_called_once()
        call_args = mock_D.features.call_args
        self.assertEqual(call_args[1]['instruments'], ["000001.SZ"])
        self.assertEqual(call_args[1]['fields'], ["$close", "$high", "$low", "$volume"])
        
        # Verify predict_proba was called with correct data
        self.mock_exit_model.predict_proba.assert_called_once()
        proba_call_args = self.mock_exit_model.predict_proba.call_args[1]
        
        # Verify daily_df structure
        daily_df = proba_call_args['daily_df']
        self.assertIsInstance(daily_df, pd.DataFrame)
        self.assertIn('close', daily_df.columns)
        self.assertIn('high', daily_df.columns)
        self.assertIn('low', daily_df.columns)
        self.assertIn('volume', daily_df.columns)
        self.assertEqual(len(daily_df), 10)
        
        # Verify other parameters
        self.assertEqual(proba_call_args['entry_price'], 10.0)
        self.assertEqual(proba_call_args['highest_price_since_entry'], 12.0)
        self.assertEqual(proba_call_args['days_held'], 9)
        
        # Verify return value (should be last probability)
        self.assertEqual(risk_prob, 0.7)
    
    @patch('trading.strategy.sell_models.ml_exit.D')
    def test_multiindex_columns_format(self, mock_D):
        """Test data extraction when Qlib returns MultiIndex columns format.
        
        Format: MultiIndex columns (instrument, field), index is datetime.
        This can happen when querying multiple instruments.
        """
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
        # Set realistic values
        hist_data[("000001.SZ", "$close")] = np.random.uniform(10.0, 12.0, 10)
        hist_data[("000001.SZ", "$high")] = np.random.uniform(11.0, 13.0, 10)
        hist_data[("000001.SZ", "$low")] = np.random.uniform(9.0, 11.0, 10)
        hist_data[("000001.SZ", "$volume")] = np.random.uniform(1000000, 2000000, 10)
        
        mock_D.features.return_value = hist_data
        
        # Test predict_exit
        date = pd.Timestamp("2025-07-10")
        risk_prob = self.sell_model.predict_exit(
            position=self.position,
            market_data=None,
            date=date
        )
        
        # Verify predict_proba was called
        self.mock_exit_model.predict_proba.assert_called_once()
        proba_call_args = self.mock_exit_model.predict_proba.call_args[1]
        
        # Verify daily_df structure
        daily_df = proba_call_args['daily_df']
        self.assertIsInstance(daily_df, pd.DataFrame)
        self.assertIn('close', daily_df.columns)
        self.assertIn('high', daily_df.columns)
        self.assertEqual(len(daily_df), 10)
        
        # Verify return value
        self.assertEqual(risk_prob, 0.7)
    
    @patch('trading.strategy.sell_models.ml_exit.D')
    def test_empty_data(self, mock_D):
        """Test handling of empty data from Qlib."""
        mock_D.features.return_value = pd.DataFrame()
        
        date = pd.Timestamp("2025-07-10")
        risk_prob = self.sell_model.predict_exit(
            position=self.position,
            market_data=None,
            date=date
        )
        
        # Should return 0.0 for empty data
        self.assertEqual(risk_prob, 0.0)
        # predict_proba should not be called
        self.mock_exit_model.predict_proba.assert_not_called()
    
    @patch('trading.strategy.sell_models.ml_exit.D')
    def test_missing_fields(self, mock_D):
        """Test handling of missing fields in data."""
        # Create data with missing fields
        dates = pd.date_range("2025-07-01", periods=10, freq="D")
        hist_data = pd.DataFrame({
            '$close': np.random.uniform(10.0, 12.0, 10),
            # Missing $high, $low, $volume
        }, index=dates)
        
        mock_D.features.return_value = hist_data
        
        date = pd.Timestamp("2025-07-10")
        risk_prob = self.sell_model.predict_exit(
            position=self.position,
            market_data=None,
            date=date
        )
        
        # Should return 0.0 for missing fields
        self.assertEqual(risk_prob, 0.0)
        # predict_proba should not be called
        self.mock_exit_model.predict_proba.assert_not_called()
    
    @patch('trading.strategy.sell_models.ml_exit.D')
    def test_exception_handling(self, mock_D):
        """Test exception handling in predict_exit."""
        # Make D.features raise an exception
        mock_D.features.side_effect = Exception("Qlib error")
        
        date = pd.Timestamp("2025-07-10")
        risk_prob = self.sell_model.predict_exit(
            position=self.position,
            market_data=None,
            date=date
        )
        
        # Should return 0.0 on exception
        self.assertEqual(risk_prob, 0.0)
    
    @patch('trading.strategy.sell_models.ml_exit.D')
    def test_wrong_symbol_in_multiindex(self, mock_D):
        """Test handling when symbol is not in MultiIndex columns."""
        # Create data with different symbol
        dates = pd.date_range("2025-07-01", periods=10, freq="D")
        columns = pd.MultiIndex.from_product([
            ["000002.SZ"],  # Different symbol
            ["$close", "$high", "$low", "$volume"]
        ])
        hist_data = pd.DataFrame(
            np.random.uniform(10.0, 13.0, (10, 4)),
            index=dates,
            columns=columns
        )
        
        mock_D.features.return_value = hist_data
        
        date = pd.Timestamp("2025-07-10")
        risk_prob = self.sell_model.predict_exit(
            position=self.position,
            market_data=None,
            date=date
        )
        
        # Should return 0.0 when symbol not found
        self.assertEqual(risk_prob, 0.0)
        # predict_proba should not be called
        self.mock_exit_model.predict_proba.assert_not_called()
    
    @patch('trading.strategy.sell_models.ml_exit.D')
    def test_exit_signal_generation(self, mock_D):
        """Test that exit signal is generated when risk_prob > threshold."""
        # Create mock data
        dates = pd.date_range("2025-07-01", periods=10, freq="D")
        hist_data = pd.DataFrame({
            '$close': np.random.uniform(10.0, 12.0, 10),
            '$high': np.random.uniform(11.0, 13.0, 10),
            '$low': np.random.uniform(9.0, 11.0, 10),
            '$volume': np.random.uniform(1000000, 2000000, 10),
        }, index=dates)
        
        mock_D.features.return_value = hist_data
        
        # Set predict_proba to return high risk probability
        self.mock_exit_model.predict_proba.return_value = np.array([0.8, 0.85, 0.9])
        
        date = pd.Timestamp("2025-07-10")
        risk_prob = self.sell_model.predict_exit(
            position=self.position,
            market_data=None,
            date=date
        )
        
        # Risk prob should be 0.9 (last value)
        self.assertEqual(risk_prob, 0.9)
        # This is > threshold (0.65), so should trigger exit signal
        self.assertGreater(risk_prob, self.sell_model.threshold)


if __name__ == "__main__":
    unittest.main()

"""
Strategy interface for custom backtesting framework.

Defines the core interface that all strategies must implement.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any

import pandas as pd


class IStrategy(ABC):
    """
    Strategy interface for custom backtesting framework.
    
    All strategies must implement this interface to work with the custom backtest engine.
    """
    
    @abstractmethod
    def on_bar(
        self,
        date: pd.Timestamp,
        market_data: pd.DataFrame,
    ) -> None:
        """
        Called on each trading bar (daily for daily backtest).
        
        This is the main entry point for strategy logic. The strategy should:
        1. Analyze current market data
        2. Check existing positions
        3. Generate buy/sell signals
        4. Submit orders through the order book
        
        Args:
            date: Current trading date.
            market_data: Market data DataFrame in normalized format:
                - Index: MultiIndex (instrument, datetime)
                - Columns: Single-level field names ($open, $close, etc.)
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy name."""
        pass
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get current strategy state (for logging/debugging).
        
        Returns:
            Dictionary containing strategy state information.
        """
        return {}
    
    def on_backtest_start(self, start_date: pd.Timestamp, end_date: pd.Timestamp) -> None:
        """
        Called when backtest starts.
        
        Args:
            start_date: Backtest start date.
            end_date: Backtest end date.
        """
        pass
    
    def on_backtest_end(self) -> None:
        """Called when backtest ends."""
        pass

"""
Trading model interfaces.

Defines abstract interfaces for buy and sell models.
"""

from abc import ABC, abstractmethod

import pandas as pd


class IBuyModel(ABC):
    """Buy model interface.
    
    Responsible for generating stock rankings and buy signals.
    """
    
    @abstractmethod
    def generate_ranks(
        self,
        date: pd.Timestamp,
        market_data: pd.DataFrame,
        **kwargs
    ) -> pd.DataFrame:
        """
        Generate stock rankings.
        
        Args:
            date: Current trading date.
            market_data: Market data DataFrame (from Qlib).
            **kwargs: Additional parameters.
        
        Returns:
            DataFrame with columns: ['symbol', 'score', 'rank']
            Sorted by score (descending).
        """
        pass


class ISellModel(ABC):
    """Sell model interface.
    
    Responsible for predicting exit signals based on position state.
    """
    
    @property
    @abstractmethod
    def threshold(self) -> float:
        """Risk probability threshold for exit signal."""
        pass
    
    @abstractmethod
    def predict_exit(
        self,
        position: 'Position',  # Forward reference
        market_data: pd.DataFrame,
        date: pd.Timestamp,
        **kwargs
    ) -> float:
        """
        Predict exit probability for a position.
        
        Args:
            position: Position object with entry_price, high_price_since_entry, etc.
            market_data: Market data DataFrame (from Qlib).
            date: Current trading date.
            **kwargs: Additional parameters.
        
        Returns:
            Risk probability (0-1). If > threshold, should exit.
        """
        pass

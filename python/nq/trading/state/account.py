"""
Account management.

Manages cash, frozen cash, and total asset tracking.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, Optional
from datetime import datetime

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .position import PositionManager


@dataclass
class Account:
    """Account state management.
    
    Tracks available cash, frozen cash, and total assets.
    Completely independent of Qlib's account management.
    
    Note: Account holds a reference to PositionManager to calculate total asset value.
    The PositionManager should be set after Account creation using set_position_manager().
    """
    
    account_id: str
    available_cash: float  # Available cash
    frozen_cash: float = 0.0  # Frozen cash (pending orders)
    initial_cash: float = 0.0  # Initial cash
    created_at: datetime = field(default_factory=datetime.now)
    position_manager: Optional['PositionManager'] = field(default=None, init=False)
    
    def __post_init__(self):
        """Initialize account."""
        if self.initial_cash == 0.0:
            self.initial_cash = self.available_cash
    
    def set_position_manager(self, position_manager: 'PositionManager') -> None:
        """
        Set position manager reference.
        
        This should be called after PositionManager is created (which requires Account).
        This creates a bidirectional relationship: Account <-> PositionManager.
        
        Args:
            position_manager: PositionManager instance.
        """
        self.position_manager = position_manager
    
    def get_total_value(self, current_prices: Optional[Dict[str, float]] = None) -> float:
        """
        Calculate total asset value (cash + holdings market value).
        
        Args:
            current_prices: Optional dict of {symbol: current_price} for position valuation.
                          If None, PositionManager will use entry_price.
        
        Returns:
            Total asset value.
        
        Raises:
            ValueError: If any component is NaN or if position_manager is not set.
        """
        if self.position_manager is None:
            raise ValueError(
                "CRITICAL BUG: position_manager is not set. "
                "Call account.set_position_manager(position_manager) after creating PositionManager."
            )
        
        # CRITICAL: Check for NaN in cash values
        if pd.isna(self.available_cash) or np.isnan(self.available_cash):
            raise ValueError(
                f"CRITICAL BUG: account.available_cash is NaN: {self.available_cash}"
            )
        if pd.isna(self.frozen_cash) or np.isnan(self.frozen_cash):
            raise ValueError(
                f"CRITICAL BUG: account.frozen_cash is NaN: {self.frozen_cash}"
            )
        
        # Get holdings value from PositionManager
        holdings_value = self.position_manager.get_total_value(current_prices=current_prices)
        
        total = self.available_cash + self.frozen_cash + holdings_value
        if pd.isna(total) or np.isnan(total):
            raise ValueError(
                f"CRITICAL BUG: get_total_value returned NaN. "
                f"available_cash={self.available_cash}, "
                f"frozen_cash={self.frozen_cash}, "
                f"holdings_value={holdings_value}"
            )
        return total
    
    def can_afford(self, required_cash: float) -> bool:
        """
        Check if account has sufficient cash.
        
        Args:
            required_cash: Required cash amount.
        
        Returns:
            True if sufficient, False otherwise.
        """
        return (self.available_cash + self.frozen_cash) >= required_cash
    
    def freeze_cash(self, amount: float) -> None:
        """
        Freeze cash for pending order.
        
        Args:
            amount: Amount to freeze.
        
        Raises:
            ValueError: If insufficient cash.
        """
        if self.available_cash < amount:
            raise ValueError(
                f"Insufficient cash: {self.available_cash:.2f} < {amount:.2f}"
            )
        self.available_cash -= amount
        self.frozen_cash += amount
    
    def unfreeze_cash(self, amount: float) -> None:
        """
        Unfreeze cash (order canceled).
        
        Args:
            amount: Amount to unfreeze.
        
        Raises:
            ValueError: If insufficient frozen cash.
        """
        if self.frozen_cash < amount:
            raise ValueError(
                f"Insufficient frozen cash: {self.frozen_cash:.2f} < {amount:.2f}"
            )
        self.frozen_cash -= amount
        self.available_cash += amount
    
    def deduct_cash(self, amount: float) -> None:
        """
        Deduct cash (order filled).
        
        Args:
            amount: Amount to deduct.
        
        Raises:
            ValueError: If insufficient frozen cash.
        """
        if self.frozen_cash < amount:
            raise ValueError(
                f"Insufficient frozen cash: {self.frozen_cash:.2f} < {amount:.2f}"
            )
        self.frozen_cash -= amount
    
    def add_cash(self, amount: float) -> None:
        """
        Add cash (sell order filled).
        
        Args:
            amount: Amount to add.
        """
        self.available_cash += amount
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'account_id': self.account_id,
            'available_cash': self.available_cash,
            'frozen_cash': self.frozen_cash,
            'initial_cash': self.initial_cash,
            'total_cash': self.available_cash + self.frozen_cash,
            'created_at': self.created_at.isoformat(),
        }

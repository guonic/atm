"""
Account management.

Manages cash, frozen cash, and total asset tracking.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from datetime import datetime

if TYPE_CHECKING:
    from .position import PositionManager


@dataclass
class Account:
    """Account state management.
    
    Tracks available cash, frozen cash, and total assets.
    Completely independent from Qlib's account management.
    """
    
    account_id: str
    available_cash: float  # Available cash
    frozen_cash: float = 0.0  # Frozen cash (pending orders)
    initial_cash: float = 0.0  # Initial cash
    created_at: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Initialize account."""
        if self.initial_cash == 0.0:
            self.initial_cash = self.available_cash
    
    def get_total_value(self, position_manager: 'PositionManager') -> float:
        """
        Calculate total asset value (cash + holdings market value).
        
        Args:
            position_manager: PositionManager instance.
        
        Returns:
            Total asset value.
        
        Raises:
            ValueError: If any component is NaN.
        """
        import numpy as np
        import pandas as pd
        
        # CRITICAL: Check for NaN in cash values
        if pd.isna(self.available_cash) or np.isnan(self.available_cash):
            raise ValueError(
                f"CRITICAL BUG: account.available_cash is NaN: {self.available_cash}"
            )
        if pd.isna(self.frozen_cash) or np.isnan(self.frozen_cash):
            raise ValueError(
                f"CRITICAL BUG: account.frozen_cash is NaN: {self.frozen_cash}"
            )
        
        # Calculate holdings value - ensure no NaN
        holdings_value = 0.0
        for pos in position_manager.all_positions.values():
            pos_value = pos.market_value
            if pd.isna(pos_value) or np.isnan(pos_value):
                raise ValueError(
                    f"CRITICAL BUG: position.market_value is NaN for {pos.symbol}. "
                    f"amount={pos.amount}, entry_price={pos.entry_price}"
                )
            holdings_value += pos_value
        
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
        return self.available_cash >= required_cash
    
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

"""
Position allocation.

Calculates position sizing and allocation strategy.
"""

from typing import TYPE_CHECKING
import pandas as pd

if TYPE_CHECKING:
    from ..state import Account

logger = __import__('logging').getLogger(__name__)


class PositionAllocator:
    """Position allocation and sizing.
    
    Calculates target position size for each stock.
    """
    
    def __init__(
        self,
        target_positions: int = 30,
        equal_weight: bool = True,
    ):
        """
        Initialize position allocator.
        
        Args:
            target_positions: Target number of positions.
            equal_weight: Whether to use equal weight allocation (default: True).
        """
        self.target_positions = target_positions
        self.equal_weight = equal_weight
    
    def calculate_position_size(
        self,
        symbol: str,
        account: 'Account',
        position_manager: 'PositionManager',
        market_data: pd.DataFrame,
    ) -> float:
        """
        Calculate target position value for a symbol.
        
        Args:
            symbol: Stock symbol.
            account: Account instance.
            position_manager: PositionManager instance.
            market_data: Market data DataFrame.
        
        Returns:
            Target position value.
        """
        if self.equal_weight:
            # Equal weight allocation
            total_value = account.get_total_value()
            # CRITICAL: Ensure no NaN values
            if pd.isna(total_value) or total_value <= 0:
                raise ValueError(
                    f"CRITICAL BUG: total_value is NaN or <= 0: {total_value}. "
                    f"available_cash={account.available_cash}, "
                    f"frozen_cash={account.frozen_cash}, "
                    f"positions={len(position_manager.positions)}"
                )
            result = total_value / self.target_positions
            if pd.isna(result):
                raise ValueError(
                    f"CRITICAL BUG: calculate_position_size returned NaN. "
                    f"total_value={total_value}, target_positions={self.target_positions}"
                )
            return result
        else:
            # Can extend to other allocation strategies (risk parity, etc.)
            total_value = account.get_total_value()
            if pd.isna(total_value) or total_value <= 0:
                raise ValueError(
                    f"CRITICAL BUG: total_value is NaN or <= 0: {total_value}"
                )
            result = total_value / self.target_positions
            if pd.isna(result):
                raise ValueError(
                    f"CRITICAL BUG: calculate_position_size returned NaN. "
                    f"total_value={total_value}, target_positions={self.target_positions}"
                )
            return result

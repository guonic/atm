"""
Position management.

Manages individual positions and position lifecycle.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..interfaces.storage import IStorageBackend

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """Individual position tracking.
    
    Tracks entry price, amount, total cost (including commission), high price since entry, etc.
    Completely independent from Qlib's position management.
    
    Note: 
    - entry_price: Average entry price per share (excluding commission)
    - total_cost: Total cost including commission (for accurate cost basis calculation)
    - avg_cost_per_share: Average cost per share including commission (total_cost / amount)
    """
    
    symbol: str
    entry_date: pd.Timestamp
    entry_price: float  # Average entry price per share (excluding commission)
    amount: float
    total_cost: float = 0.0  # Total cost including commission
    high_price_since_entry: float = 0.0
    high_date: Optional[pd.Timestamp] = None
    
    @property
    def avg_price(self) -> float:
        """Average entry price per share (excluding commission)."""
        return self.entry_price
    
    @property
    def avg_cost_per_share(self) -> float:
        """
        Average cost per share including commission.
        
        This is the true cost basis for calculating returns.
        Returns total_cost / amount, or entry_price if total_cost is 0 (backward compatibility).
        """
        if self.amount > 0 and self.total_cost > 0:
            return self.total_cost / self.amount
        return self.entry_price
    
    def empty(self) -> bool:
        """
        Check if position is empty (should be removed).
        
        Returns:
            True if position has no shares (amount <= 0), False otherwise.
        """
        return self.amount <= 0
    
    def update_high_price(self, current_high: float, date: pd.Timestamp) -> None:
        """
        Update the highest price since entry.
        
        Args:
            current_high: Current high price.
            date: Current date.
        """
        if self.high_price_since_entry == 0.0 or current_high > self.high_price_since_entry:
            self.high_price_since_entry = current_high
            self.high_date = date
    
    def apply_transaction(
        self,
        entry_date: pd.Timestamp,
        entry_price: float,
        amount: float,
        commission: float = 0.0,
    ) -> None:
        """
        Apply a transaction to this position (buy or sell).
        
        For buy (amount > 0):
        - Updates average entry price, total cost, and amount.
        - Updates high price if the new entry price is higher.
        
        For sell (amount < 0):
        - Reduces amount and total_cost proportionally.
        - Does NOT update entry_price (cost basis remains unchanged).
        - Does NOT update high_price (selling doesn't affect highest price).
        
        Args:
            entry_date: Entry date of the transaction.
            entry_price: Price per share of the transaction.
            amount: Amount (shares). Positive for buy, negative for sell.
            commission: Commission cost for this transaction (default: 0.0).
        
        Raises:
            ValueError: If any input is NaN or invalid, or if selling more than available.
        """
        # CRITICAL: Validate inputs
        if pd.isna(entry_price) or np.isnan(entry_price):
            raise ValueError(
                f"CRITICAL BUG: entry_price is NaN for {self.symbol}. "
                f"entry_price={entry_price}"
            )
        if pd.isna(amount) or np.isnan(amount) or amount == 0:
            raise ValueError(
                f"CRITICAL BUG: amount is NaN or 0 for {self.symbol}. "
                f"amount={amount}"
            )
        if pd.isna(commission) or np.isnan(commission) or commission < 0:
            raise ValueError(
                f"CRITICAL BUG: commission is NaN or < 0 for {self.symbol}. "
                f"commission={commission}"
            )
        
        if amount > 0:
            # BUY: Add to position
            # Calculate new transaction cost
            total_cost_this_trade = entry_price * amount + commission
            
            # Update amount
            old_amount = self.amount
            self.amount += amount
            
            # Update average entry price (excluding commission)
            # Weighted average: (old_value + new_value) / total_amount
            old_value = self.entry_price * old_amount
            new_value = entry_price * amount
            self.entry_price = (old_value + new_value) / self.amount if self.amount > 0 else entry_price
            
            # Update total cost (including commission)
            self.total_cost += total_cost_this_trade
            
            # Update high price if needed
            if entry_price > self.high_price_since_entry or self.high_price_since_entry == 0.0:
                self.high_price_since_entry = entry_price
                self.high_date = entry_date
            
            logger.debug(
                f"Applied BUY transaction to position {self.symbol}: "
                f"amount={amount:.0f} @ {entry_price:.2f}, "
                f"new_total_amount={self.amount:.0f}, "
                f"new_avg_price={self.entry_price:.2f}, "
                f"new_avg_cost={self.avg_cost_per_share:.2f}, "
                f"new_total_cost={self.total_cost:.2f}"
            )
        else:
            # SELL: Reduce position
            sell_amount = abs(amount)
            
            # Check if we have enough shares
            if sell_amount > self.amount:
                raise ValueError(
                    f"CRITICAL BUG: Trying to sell {sell_amount:.0f} shares but only have {self.amount:.0f} "
                    f"for {self.symbol}"
                )
            
            # Calculate reduction ratio
            reduction_ratio = sell_amount / self.amount
            
            # Reduce amount
            self.amount -= sell_amount
            
            # Reduce total_cost proportionally (maintain cost basis for remaining shares)
            self.total_cost *= (1 - reduction_ratio)
            
            # Note: entry_price remains unchanged (cost basis per share doesn't change)
            # Note: high_price_since_entry remains unchanged (selling doesn't affect highest price)
            
            logger.debug(
                f"Applied SELL transaction to position {self.symbol}: "
                f"amount={sell_amount:.0f} @ {entry_price:.2f}, "
                f"remaining_amount={self.amount:.0f}, "
                f"remaining_cost={self.total_cost:.2f}, "
                f"avg_cost={self.avg_cost_per_share:.2f}"
            )
    
    def calculate_return(self, current_price: float, use_cost_basis: bool = True) -> float:
        """
        Calculate current return.
        
        Args:
            current_price: Current price.
            use_cost_basis: If True, use avg_cost_per_share (including commission) for return calculation.
                          If False, use entry_price (excluding commission). Default: True.
        
        Returns:
            Return ratio (e.g., 0.1 for 10% return).
        """
        cost_basis = self.avg_cost_per_share if use_cost_basis else self.entry_price
        if cost_basis > 0:
            return (current_price - cost_basis) / cost_basis
        return 0.0
    
    def calculate_drawdown(self, current_price: float) -> float:
        """
        Calculate drawdown from the highest price.
        
        Args:
            current_price: Current price.
        
        Returns:
            Drawdown ratio (e.g., 0.05 for 5% drawdown).
        """
        if self.high_price_since_entry > 0:
            return (self.high_price_since_entry - current_price) / self.high_price_since_entry
        return 0.0
    
    def calculate_market_value(self, current_price: Optional[float] = None) -> float:
        """
        Calculate market value.
        
        Args:
            current_price: Current price. If None (e.g., stock suspended), uses entry_price as fallback.
        
        Returns:
            Market value (amount * price). Returns 0.0 if current_price is None and amount is 0.
        
        Raises:
            ValueError: If current_price is NaN (not None), or if result is NaN.
        """
        # Handle None (stock suspended or no data available)
        if current_price is None:
            # Use entry_price as fallback for suspended stocks
            # This is a reasonable approximation when market data is unavailable
            current_price = self.entry_price
            logger.debug(
                f"Using entry_price as fallback for {self.symbol} "
                f"(current_price is None, possibly suspended)"
            )
        
        # CRITICAL: Check for NaN - fail fast (None is handled above)
        if pd.isna(current_price) or np.isnan(current_price):
            raise ValueError(
                f"CRITICAL BUG: current_price is NaN for {self.symbol}. "
                f"This should have been filtered in data validation. "
                f"None is allowed (for suspended stocks), but NaN is not."
            )
        if pd.isna(self.amount) or np.isnan(self.amount):
            raise ValueError(
                f"CRITICAL BUG: position.amount is NaN for {self.symbol}. "
                f"amount={self.amount}"
            )
        
        value = self.amount * current_price
        
        if np.isnan(value):
            raise ValueError(
                f"CRITICAL BUG: calculate_market_value returned NaN for {self.symbol}. "
                f"amount={self.amount}, current_price={current_price}"
            )
        
        return value
    
    @property
    def market_value(self) -> float:
        """
        Market value (placeholder - uses entry_price).
        
        Note: This is a placeholder property. For accurate market value,
        use calculate_market_value(current_price) with actual current price.
        
        Raises:
            ValueError: If any component is NaN.
        """
        # CRITICAL: Check for NaN - fail fast
        if pd.isna(self.amount) or np.isnan(self.amount):
            raise ValueError(
                f"CRITICAL BUG: position.amount is NaN for {self.symbol}. "
                f"amount={self.amount}"
            )
        if pd.isna(self.entry_price) or np.isnan(self.entry_price):
            raise ValueError(
                f"CRITICAL BUG: position.entry_price is NaN for {self.symbol}. "
                f"entry_price={self.entry_price}"
            )
        
        value = self.amount * self.entry_price
        
        if np.isnan(value):
            raise ValueError(
                f"CRITICAL BUG: market_value property returned NaN for {self.symbol}. "
                f"amount={self.amount}, entry_price={self.entry_price}"
            )
        
        return value
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'symbol': self.symbol,
            'entry_date': self.entry_date.strftime('%Y-%m-%d'),
            'entry_price': self.entry_price,
            'amount': self.amount,
            'total_cost': self.total_cost,
            'avg_cost_per_share': self.avg_cost_per_share,
            'high_price_since_entry': self.high_price_since_entry,
            'high_date': self.high_date.strftime('%Y-%m-%d') if self.high_date is not None else None,
        }


class PositionManager:
    """Position lifecycle management.
    
    Manages all positions, tracks entry/exit, updates high prices.
    Completely independent of Qlib's position management.
    """
    
    def __init__(self, account: 'Account', storage: IStorageBackend):
        """
        Initialize position manager.
        
        Args:
            account: Account instance.
            storage: Storage backend.
        """
        self.account = account
        self.storage = storage
        self.positions: Dict[str, Position] = {}
    
    @property
    def all_positions(self) -> Dict[str, Position]:
        """Get all active positions."""
        return self.positions
    
    def add_position(
        self,
        symbol: str,
        entry_date: pd.Timestamp,
        entry_price: float,
        amount: float,
        commission: float = 0.0,
    ) -> None:
        """
        Add position (buy order filled).
        
        If position doesn't exist, creates an empty position first, then applies the new transaction.
        This ensures unified handling through the apply_transaction() method.
        
        Args:
            symbol: Stock symbol.
            entry_date: Entry date.
            entry_price: Entry price per share.
            amount: Position amount (shares).
            commission: Commission cost for this transaction (default: 0.0).
        """
        if symbol not in self.positions:
            # Create empty position (will be populated by merge)
            self.positions[symbol] = Position(
                symbol=symbol,
                entry_date=entry_date,  # Use first transaction date as initial entry_date
                entry_price=0.0,  # Will be set by merge
                amount=0.0,  # Will be set by merge
                total_cost=0.0,  # Will be set by merge
                high_price_since_entry=0.0,  # Will be set by merge
                high_date=None,  # Will be set by merge
            )
        
        # Apply new transaction to position (works for both new and existing)
        self.positions[symbol].apply_transaction(
            entry_date=entry_date,
            entry_price=entry_price,
            amount=amount,
            commission=commission,
        )
        
        # Persist
        self.storage.save(f"pos:{symbol}", self.positions[symbol].to_dict())
    
    def remove_position(self, symbol: str) -> None:
        """
        Remove position (sell order filled, position closed).
        
        Args:
            symbol: Stock symbol.
        """
        if symbol in self.positions:
            del self.positions[symbol]
            self.storage.delete(f"pos:{symbol}")
            logger.debug(f"Position closed: {symbol}")
    
    def reduce_position(
        self,
        symbol: str,
        amount: float,
        sell_date: pd.Timestamp,
        sell_price: float,
        commission: float = 0.0,
    ) -> None:
        """
        Reduce position (sell order filled).
        
        Uses negative apply_transaction to reduce position, then removes if empty.
        
        Args:
            symbol: Stock symbol.
            amount: Amount to reduce (shares).
            sell_date: Sell date.
            sell_price: Sell price per share.
            commission: Commission cost for this transaction (default: 0.0).
        """
        if symbol not in self.positions:
            raise ValueError(
                f"CRITICAL BUG: Trying to reduce position for {symbol} but position doesn't exist"
            )
        
        pos = self.positions[symbol]
        
        # Use negative apply_transaction to reduce position
        pos.apply_transaction(
            entry_date=sell_date,
            entry_price=sell_price,
            amount=-amount,  # Negative for sale
            commission=commission,
        )
        
        # Check if position is empty and remove if so
        if pos.empty():
            self.remove_position(symbol)
        else:
            # Persist updated position
            self.storage.save(f"pos:{symbol}", pos.to_dict())
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """
        Get position by symbol.
        
        Args:
            symbol: Stock symbol.
        
        Returns:
            Position if exists, None otherwise.
        """
        return self.positions.get(symbol)
    
    def get_total_value(self, current_prices: Optional[Dict[str, float]] = None) -> float:
        """
        Calculate total market value of all positions.
        
        Args:
            current_prices: Dict of {symbol: current_price}. Defaults to {}.
                          If symbol not in dict, uses entry_price for that position.
        
        Returns:
            Total market value of all positions.
        
        Raises:
            ValueError: If any position value is NaN.
        """
        # Default to empty dict for simpler code
        if current_prices is None:
            current_prices = {}
        
        total_value = 0.0
        for symbol, position in self.positions.items():
            # Use current_price if provided, otherwise use entry_price
            current_price = current_prices.get(symbol, position.entry_price)
            
            pos_value = position.calculate_market_value(current_price)
            
            # CRITICAL: Check for NaN
            if pd.isna(pos_value) or np.isnan(pos_value):
                raise ValueError(
                    f"CRITICAL BUG: position.market_value is NaN for {symbol}. "
                    f"amount={position.amount}, entry_price={position.entry_price}, "
                    f"current_price={current_price}"
                )
            
            total_value += pos_value
        
        # CRITICAL: Check final result for NaN
        if pd.isna(total_value) or np.isnan(total_value):
            raise ValueError(
                f"CRITICAL BUG: PositionManager.get_total_value returned NaN. "
                f"positions={len(self.positions)}"
            )
        
        return total_value
    
    def get_weight(self, symbol: str, current_price: float) -> float:
        """
        Get position weight in portfolio.
        
        Args:
            symbol: Stock symbol.
            current_price: Current price.
        
        Returns:
            Weight ratio (0-1).
        """
        if symbol not in self.positions:
            return 0.0
        
        pos = self.positions[symbol]
        pos_value = pos.calculate_market_value(current_price)
        
        # Use PositionManager.get_total_value instead of Account.get_total_value
        # to avoid circular dependency
        current_prices = {symbol: current_price}
        total_value = self.get_total_value(current_prices=current_prices)
        
        if total_value > 0:
            return pos_value / total_value
        return 0.0
    
    @staticmethod
    def _get_symbol_daily_data(
        market_data: pd.DataFrame,
        symbol: str,
        date: pd.Timestamp
    ) -> Optional[pd.Series]:
        """
        Extract daily data for a single symbol from market_data DataFrame.
        
        Args:
            market_data: Market data DataFrame (MultiIndex or single index).
            symbol: Stock symbol.
            date: Trading date.
        
        Returns:
            Series with daily data for the symbol, or None if not found.
        """
        from ..utils.market_data import MarketDataFrame
        
        try:
            mdf = MarketDataFrame(market_data)
            return mdf.get_daily_data(symbol, date)
        except (KeyError, IndexError):
            return None
    
    @staticmethod
    def _safe_get_ohlc(
        daily_data: pd.Series,
        fields: Dict[str, List[str]]
    ) -> Optional[Dict[str, float]]:
        """
        Safely extract OHLC data from daily_data Series.
        
        Args:
            daily_data: Series with market data (may have '$high', '$close', etc.).
            fields: Dict mapping field names to list of possible column names.
                   Example: {'high': ['$high', 'high'], 'close': ['$close', 'close']}
        
        Returns:
            Dict with extracted values, or None if any required field is missing/NaN.
        
        Example:
            >>> daily_data = pd.Series({'$high': 10.5, '$close': 10.0})
            >>> result = _safe_get_ohlc(daily_data, {'high': ['$high', 'high'], 'close': ['$close', 'close']})
            >>> result
            {'high': 10.5, 'close': 10.0}
        """
        result = {}
        
        for field_name, possible_names in fields.items():
            value = None
            for name in possible_names:
                if name in daily_data.index:
                    raw_value = daily_data[name]
                    if raw_value is not None and not pd.isna(raw_value):
                        value = float(raw_value)
                        break
            
            if value is None:
                return None
            
            # Validate value
            if value <= 0:
                return None
            
            result[field_name] = value
        
        return result
    
    def update_positions(
        self,
        date: pd.Timestamp,
        market_data: pd.DataFrame
    ) -> None:
        """
        Update all positions (called after market close).
        
        Args:
            date: Current date.
            market_data: Market data DataFrame.
        """
        for symbol, position in self.positions.items():
            try:
                # Step 1: Get symbol's daily data from market_data
                daily_data = self._get_symbol_daily_data(market_data, symbol, date)
                if daily_data is None:
                    continue
                
                # Step 2: Safely extract OHLC data
                ohlc_fields = {
                    'high': ['$high', 'high'],
                    'close': ['$close', 'close'],
                }
                ohlc_data = self._safe_get_ohlc(daily_data, ohlc_fields)
                if ohlc_data is None:
                    logger.warning(
                        f"Skipping position update for {symbol} on {date} due to missing/NaN/invalid OHLC data"
                    )
                    continue
                
                current_high = ohlc_data['high']
                current_close = ohlc_data['close']
                
                # Update high price
                position.update_high_price(current_high, date)
                
                # Persist snapshot
                snapshot = {
                    'date': date.strftime('%Y-%m-%d'),
                    'symbol': symbol,
                    'entry_price': position.entry_price,
                    'current_price': current_close,
                    'amount': position.amount,
                    'high_price_since_entry': position.high_price_since_entry,
                    'current_return': position.calculate_return(current_close),
                    'drawdown': position.calculate_drawdown(current_close),
                    'market_value': position.calculate_market_value(current_close),
                }
                self.storage.save(f"snapshot:{date.strftime('%Y-%m-%d')}:{symbol}", snapshot)
            except Exception as e:
                logger.warning(f"Failed to update position {symbol}: {e}")
    
    def has_free_slot(self, max_positions: int = 30) -> bool:
        """
        Check if there are free position slots.
        
        Args:
            max_positions: Maximum number of positions.
        
        Returns:
            True if has free slot, False otherwise.
        """
        return len(self.positions) < max_positions

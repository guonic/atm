"""
Position management.

Manages individual positions and position lifecycle.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional
import pandas as pd
import logging

from ..interfaces.storage import IStorageBackend

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """Individual position tracking.
    
    Tracks entry price, amount, high price since entry, etc.
    Completely independent from Qlib's position management.
    """
    
    symbol: str
    entry_date: pd.Timestamp
    entry_price: float
    amount: float
    high_price_since_entry: float
    high_date: pd.Timestamp
    
    @property
    def avg_price(self) -> float:
        """Average entry price (for now, same as entry_price)."""
        return self.entry_price
    
    def update_high_price(self, current_high: float, date: pd.Timestamp) -> None:
        """
        Update highest price since entry.
        
        Args:
            current_high: Current high price.
            date: Current date.
        """
        if current_high > self.high_price_since_entry:
            self.high_price_since_entry = current_high
            self.high_date = date
    
    def calculate_return(self, current_price: float) -> float:
        """
        Calculate current return.
        
        Args:
            current_price: Current price.
        
        Returns:
            Return ratio (e.g., 0.1 for 10% return).
        """
        if self.entry_price > 0:
            return (current_price - self.entry_price) / self.entry_price
        return 0.0
    
    def calculate_drawdown(self, current_price: float) -> float:
        """
        Calculate drawdown from highest price.
        
        Args:
            current_price: Current price.
        
        Returns:
            Drawdown ratio (e.g., 0.05 for 5% drawdown).
        """
        if self.high_price_since_entry > 0:
            return (self.high_price_since_entry - current_price) / self.high_price_since_entry
        return 0.0
    
    def calculate_market_value(self, current_price: float) -> float:
        """
        Calculate market value.
        
        Args:
            current_price: Current price.
        
        Returns:
            Market value (amount * price).
        """
        import numpy as np
        import pandas as pd
        # Ensure no NaN values
        if pd.isna(current_price) or np.isnan(current_price):
            return 0.0
        if pd.isna(self.amount) or np.isnan(self.amount):
            return 0.0
        value = self.amount * current_price
        if np.isnan(value):
            return 0.0
        return value
    
    @property
    def market_value(self) -> float:
        """Market value (requires current_price to be set)."""
        # This is a placeholder - actual market value should be calculated with current price
        # Ensure no NaN values
        import numpy as np
        value = self.amount * self.entry_price
        if np.isnan(value):
            return 0.0
        return value
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'symbol': self.symbol,
            'entry_date': self.entry_date.strftime('%Y-%m-%d'),
            'entry_price': self.entry_price,
            'amount': self.amount,
            'high_price_since_entry': self.high_price_since_entry,
            'high_date': self.high_date.strftime('%Y-%m-%d'),
        }


class PositionManager:
    """Position lifecycle management.
    
    Manages all positions, tracks entry/exit, updates high prices.
    Completely independent from Qlib's position management.
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
    ) -> None:
        """
        Add position (buy order filled).
        
        Args:
            symbol: Stock symbol.
            entry_date: Entry date.
            entry_price: Entry price.
            amount: Position amount.
        """
        if symbol in self.positions:
            # Add to existing position (update average cost)
            pos = self.positions[symbol]
            old_value = pos.entry_price * pos.amount
            new_value = entry_price * amount
            pos.amount += amount
            pos.entry_price = (old_value + new_value) / pos.amount if pos.amount > 0 else entry_price
            
            # Update high price if needed
            if entry_price > pos.high_price_since_entry:
                pos.high_price_since_entry = entry_price
                pos.high_date = entry_date
            
            logger.debug(
                f"Added to position: {symbol}, new amount={pos.amount:.0f}, "
                f"avg_price={pos.entry_price:.2f}"
            )
        else:
            # New position
            self.positions[symbol] = Position(
                symbol=symbol,
                entry_date=entry_date,
                entry_price=entry_price,
                amount=amount,
                high_price_since_entry=entry_price,
                high_date=entry_date,
            )
            logger.debug(
                f"New position: {symbol}, amount={amount:.0f}, price={entry_price:.2f}"
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
    
    def reduce_position(self, symbol: str, amount: float) -> None:
        """
        Reduce position (partial sell).
        
        Args:
            symbol: Stock symbol.
            amount: Amount to reduce.
        """
        if symbol in self.positions:
            pos = self.positions[symbol]
            pos.amount -= amount
            if pos.amount <= 0:
                self.remove_position(symbol)
            else:
                self.storage.save(f"pos:{symbol}", pos.to_dict())
                logger.debug(f"Reduced position: {symbol}, remaining={pos.amount:.0f}")
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """
        Get position by symbol.
        
        Args:
            symbol: Stock symbol.
        
        Returns:
            Position if exists, None otherwise.
        """
        return self.positions.get(symbol)
    
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
        total_value = self.account.get_total_value(self)
        
        if total_value > 0:
            return pos_value / total_value
        return 0.0
    
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
                # Get symbol data from market_data
                # market_data should have MultiIndex (instrument, datetime) or single index
                if isinstance(market_data.index, pd.MultiIndex):
                    symbol_data = market_data.xs(symbol, level=0)
                    if date in symbol_data.index:
                        daily_data = symbol_data.loc[date]
                    else:
                        continue
                else:
                    if symbol in market_data.index:
                        daily_data = market_data.loc[symbol]
                    else:
                        continue
                
                # Extract OHLC data - ensure no NaN values
                raw_high = daily_data.get('$high')
                if raw_high is None or pd.isna(raw_high):
                    raw_high = daily_data.get('high')
                if raw_high is None or pd.isna(raw_high):
                    logger.warning(
                        f"Skipping position update for {symbol} on {date} due to NaN prices: "
                        f"high={raw_high}"
                    )
                    continue
                
                raw_close = daily_data.get('$close')
                if raw_close is None or pd.isna(raw_close):
                    raw_close = daily_data.get('close')
                if raw_close is None or pd.isna(raw_close):
                    logger.warning(
                        f"Skipping position update for {symbol} on {date} due to NaN prices: "
                        f"close={raw_close}"
                    )
                    continue
                
                current_high = float(raw_high)
                current_close = float(raw_close)
                
                # Skip if prices are invalid
                if current_high <= 0 or current_close <= 0:
                    logger.warning(f"Invalid prices for {symbol} on {date}: high={current_high}, close={current_close}")
                    continue
                
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

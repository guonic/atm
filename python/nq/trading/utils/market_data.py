"""
Market data wrapper for unified DataFrame access.

This module provides a unified interface for accessing market data,
regardless of whether the DataFrame uses MultiIndex or single-level index/columns.
"""

from typing import Optional, List, Dict, Any, Union
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class MarketDataFrame:
    """
    Wrapper for pandas DataFrame that provides unified access to market data.
    
    Handles both MultiIndex and single-level index/columns transparently.
    
    Common patterns:
    - Index: MultiIndex (instrument, datetime) or single-level (instrument or datetime)
    - Columns: MultiIndex (instrument, field) or single-level (field)
    
    Examples:
        >>> # MultiIndex index, single-level columns
        >>> data = MarketDataFrame(df)  # df.index = MultiIndex([('000001.SZ', '2025-01-01'), ...])
        >>> symbol_data = data.get_symbol_data('000001.SZ')
        >>> daily_data = data.get_daily_data('000001.SZ', pd.Timestamp('2025-01-01'))
        >>> price = data.get_field('000001.SZ', pd.Timestamp('2025-01-01'), '$close')
        
        >>> # Single-level index, MultiIndex columns
        >>> data = MarketDataFrame(df)  # df.columns = MultiIndex([('000001.SZ', '$close'), ...])
        >>> price = data.get_field('000001.SZ', None, '$close')  # date not needed for single index
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize MarketDataFrame wrapper.
        
        Args:
            df: pandas DataFrame with market data.
        """
        self.df = df
        self._is_multiindex_index = isinstance(df.index, pd.MultiIndex)
        self._is_multiindex_columns = isinstance(df.columns, pd.MultiIndex)
        
        # Cache index level names for MultiIndex
        if self._is_multiindex_index:
            self._index_levels = df.index.names
            # Assume standard Qlib format: (instrument, datetime)
            self._instrument_level = 0
            self._datetime_level = 1
        else:
            self._index_levels = None
            self._instrument_level = None
            self._datetime_level = None
    
    @property
    def is_multiindex_index(self) -> bool:
        """Check if index is MultiIndex."""
        return self._is_multiindex_index
    
    @property
    def is_multiindex_columns(self) -> bool:
        """Check if columns is MultiIndex."""
        return self._is_multiindex_columns
    
    def get_symbol_data(self, symbol: str) -> pd.DataFrame:
        """
        Get all data for a specific symbol.
        
        Args:
            symbol: Stock symbol.
        
        Returns:
            DataFrame with data for the symbol.
        """
        if self._is_multiindex_index:
            return self.df.xs(symbol, level=self._instrument_level)
        else:
            if symbol in self.df.index:
                return self.df.loc[[symbol]]
            else:
                return pd.DataFrame()
    
    def get_daily_data(
        self,
        symbol: str,
        date: pd.Timestamp
    ) -> Optional[pd.Series]:
        """
        Get daily data for a specific symbol and date.
        
        Args:
            symbol: Stock symbol.
            date: Trading date.
        
        Returns:
            Series with daily data, or None if not found.
        """
        if self._is_multiindex_index:
            try:
                symbol_data = self.df.xs(symbol, level=self._instrument_level)
                if date in symbol_data.index:
                    return symbol_data.loc[date]
                else:
                    return None
            except (KeyError, IndexError):
                return None
        else:
            # Single index: assume it's symbol index, and data is for current date
            if symbol in self.df.index:
                return self.df.loc[symbol]
            else:
                return None
    
    def get_field(
        self,
        symbol: str,
        date: Optional[pd.Timestamp],
        field: str
    ) -> Optional[float]:
        """
        Get a specific field value for a symbol and date.
        
        Args:
            symbol: Stock symbol.
            date: Trading date (None if single-level index).
            field: Field name (e.g., '$close', '$open').
        
        Returns:
            Field value, or None if not found.
        """
        daily_data = self.get_daily_data(symbol, date) if date is not None else self.get_symbol_data(symbol)
        
        if daily_data is None or daily_data.empty:
            return None
        
        # Handle Series (single row)
        if isinstance(daily_data, pd.Series):
            if self._is_multiindex_columns:
                # MultiIndex columns: find column matching (symbol, field) or (field, ...)
                for col in daily_data.index:
                    if isinstance(col, tuple):
                        # Check if field matches
                        if len(col) >= 2 and col[1] == field:
                            value = daily_data[col]
                            if pd.isna(value):
                                return None
                            return float(value)
                        # Or check if symbol matches and field is in tuple
                        if col[0] == symbol and field in col:
                            value = daily_data[col]
                            if pd.isna(value):
                                return None
                            return float(value)
                return None
            else:
                # Single-level columns
                if field in daily_data.index:
                    value = daily_data[field]
                    if pd.isna(value):
                        return None
                    return float(value)
                return None
        
        # Handle DataFrame (multiple rows)
        if isinstance(daily_data, pd.DataFrame):
            if self._is_multiindex_columns:
                # Try to find column matching field
                for col in daily_data.columns:
                    if isinstance(col, tuple):
                        if len(col) >= 2 and col[1] == field:
                            value = daily_data[col].iloc[0] if len(daily_data) > 0 else None
                            if value is None or pd.isna(value):
                                return None
                            return float(value)
                return None
            else:
                if field in daily_data.columns:
                    value = daily_data[field].iloc[0] if len(daily_data) > 0 else None
                    if value is None or pd.isna(value):
                        return None
                    return float(value)
                return None
        
        return None
    
    def get_all_symbols(self) -> List[str]:
        """
        Get all symbols in the data.
        
        Returns:
            List of symbols.
        """
        if self._is_multiindex_index:
            return self.df.index.get_level_values(self._instrument_level).unique().tolist()
        else:
            return self.df.index.unique().tolist()
    
    def get_all_dates(self, symbol: Optional[str] = None) -> List[pd.Timestamp]:
        """
        Get all dates in the data.
        
        Args:
            symbol: Optional symbol to filter dates.
        
        Returns:
            List of dates.
        """
        if self._is_multiindex_index:
            if symbol:
                symbol_data = self.get_symbol_data(symbol)
                if isinstance(symbol_data.index, pd.MultiIndex):
                    return symbol_data.index.get_level_values(0).unique().tolist()
                else:
                    return symbol_data.index.unique().tolist()
            else:
                return self.df.index.get_level_values(self._datetime_level).unique().tolist()
        else:
            # Single index: assume dates are not in index
            return []
    
    def filter_by_date(self, date: pd.Timestamp) -> pd.DataFrame:
        """
        Filter data by date.
        
        Args:
            date: Trading date.
        
        Returns:
            Filtered DataFrame.
        """
        if self._is_multiindex_index:
            date_mask = self.df.index.get_level_values(self._datetime_level) == date
            return self.df.loc[date_mask]
        else:
            # Single index: return all data (assume it's for current date)
            return self.df
    
    def has_symbol(self, symbol: str) -> bool:
        """
        Check if symbol exists in data.
        
        Args:
            symbol: Stock symbol.
        
        Returns:
            True if symbol exists, False otherwise.
        """
        if self._is_multiindex_index:
            return symbol in self.df.index.get_level_values(self._instrument_level)
        else:
            return symbol in self.df.index
    
    def has_date(self, date: pd.Timestamp, symbol: Optional[str] = None) -> bool:
        """
        Check if date exists in data.
        
        Args:
            date: Trading date.
            symbol: Optional symbol to check.
        
        Returns:
            True if date exists, False otherwise.
        """
        if self._is_multiindex_index:
            if symbol:
                symbol_data = self.get_symbol_data(symbol)
                return date in symbol_data.index
            else:
                return date in self.df.index.get_level_values(self._datetime_level)
        else:
            # Single index: assume dates are not in index
            return False
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        Get underlying DataFrame.
        
        Returns:
            Original pandas DataFrame.
        """
        return self.df
    
    def __getitem__(self, key):
        """Delegate indexing to underlying DataFrame."""
        return self.df[key]
    
    def __len__(self) -> int:
        """Return length of underlying DataFrame."""
        return len(self.df)
    
    @property
    def empty(self) -> bool:
        """Check if DataFrame is empty."""
        return self.df.empty
    
    @property
    def shape(self) -> tuple:
        """Get DataFrame shape."""
        return self.df.shape

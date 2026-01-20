"""
Data normalizer for Qlib DataFrames.

This module normalizes Qlib's inconsistent DataFrame formats into a unified format:
- Index: MultiIndex (instrument, datetime) - ALWAYS
- Columns: Single-level field names (e.g., '$close', '$open') - ALWAYS

This eliminates the need for isinstance checks throughout the codebase.
"""

from typing import Optional, List
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def normalize_qlib_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize Qlib DataFrame to unified format.
    
    Qlib's D.features() returns inconsistent formats:
    1. Single instrument: Index=datetime, Columns=single-level fields
    2. Multiple instruments: Index=datetime, Columns=MultiIndex (instrument, field)
    3. With MultiIndex index: Index=MultiIndex (instrument, datetime), Columns=single-level fields
    
    This function normalizes ALL formats to:
    - Index: MultiIndex (instrument, datetime)
    - Columns: Single-level field names
    
    Args:
        df: Raw DataFrame from Qlib D.features().
    
    Returns:
        Normalized DataFrame with consistent format.
    
    Examples:
        >>> # Input: Single instrument, single-level columns
        >>> # Index: datetime, Columns: ['$close', '$open']
        >>> # Output: Index: MultiIndex([('000001.SZ', '2025-01-01'), ...]), Columns: ['$close', '$open']
        
        >>> # Input: Multiple instruments, MultiIndex columns
        >>> # Index: datetime, Columns: MultiIndex([('000001.SZ', '$close'), ...])
        >>> # Output: Index: MultiIndex([('000001.SZ', '2025-01-01'), ...]), Columns: ['$close', '$open']
    """
    if df.empty:
        return df
    
    # Case 1: MultiIndex index, single-level columns
    if isinstance(df.index, pd.MultiIndex) and not isinstance(df.columns, pd.MultiIndex):
        # Ensure level names are correct and order is (instrument, datetime)
        if len(df.index.levels) >= 2:
            level_names = [n.lower() if n else "" for n in df.index.names]
            
            # Check if first level looks like datetime
            first_level_is_date = False
            try:
                # Check if first element of level 0 is a timestamp or date-like
                first_val = df.index.get_level_values(0)[0]
                if isinstance(first_val, (pd.Timestamp, pd.DatetimeIndex)) or \
                   hasattr(first_val, 'year'):
                    first_level_is_date = True
            except:
                pass
                
            if "datetime" in level_names[0] or "date" in level_names[0] or first_level_is_date:
                # Levels might be swapped: (datetime, instrument)
                # Check if second level is NOT a date
                second_level_is_date = False
                try:
                    second_val = df.index.get_level_values(1)[0]
                    if isinstance(second_val, (pd.Timestamp, pd.DatetimeIndex)) or \
                       hasattr(second_val, 'year'):
                        second_level_is_date = True
                except:
                    pass
                
                if not second_level_is_date or "instrument" in level_names[1] or "symbol" in level_names[1]:
                    # Confirmed swapped: (datetime, instrument) -> (instrument, datetime)
                    df = df.swaplevel(0, 1).sort_index()
        
        # Consistent level names
        df.index.names = ['instrument', 'datetime']
        return df
    
    # Case 2: Single-level index, MultiIndex columns
    # Format: Index=datetime, Columns=MultiIndex(instrument, field) or MultiIndex(field, instrument)
    if not isinstance(df.index, pd.MultiIndex) and isinstance(df.columns, pd.MultiIndex):
        # Set index name if not set
        if df.index.name is None:
            df.index.name = 'datetime'
            
        # Determine which level is instrument
        col_level_names = [n.lower() if n else "" for n in df.columns.names]
        instrument_level = 0
        if "field" in col_level_names[0] or "$" in str(df.columns.get_level_values(0)[0]):
            instrument_level = 1
            
        # Stack instrument level to index
        normalized_df = df.stack(level=instrument_level)
        
        # Now index is (datetime, instrument)
        # Swap to get (instrument, datetime)
        normalized_df = normalized_df.swaplevel(0, 1).sort_index()
        normalized_df.index.names = ['instrument', 'datetime']
        
        # If columns are still MultiIndex (happens if multiple levels remained), flatten them
        if isinstance(normalized_df.columns, pd.MultiIndex):
            normalized_df.columns = normalized_df.columns.get_level_values(-1)
            
        return normalized_df
    
    # Case 3: Single-level index, single-level columns
    # Still requires external instrument info
    if not isinstance(df.index, pd.MultiIndex) and not isinstance(df.columns, pd.MultiIndex):
        raise ValueError(
            "Cannot normalize single-level index/columns without instrument information. "
            "Please use normalize_qlib_dataframe_with_instrument() instead."
        )
    
    # Case 4: MultiIndex index, MultiIndex columns (shouldn't happen, but handle it)
    if isinstance(df.index, pd.MultiIndex) and isinstance(df.columns, pd.MultiIndex):
        # Flatten columns to single level (use field name only)
        df.columns = df.columns.get_level_values(1)
        return df
    
    return df


def normalize_qlib_dataframe_with_instrument(
    df: pd.DataFrame,
    instrument: str
) -> pd.DataFrame:
    """
    Normalize Qlib DataFrame with explicit instrument.
    
    Used when DataFrame has single-level index/columns (single instrument query).
    
    Args:
        df: Raw DataFrame from Qlib D.features().
        instrument: Instrument symbol.
    
    Returns:
        Normalized DataFrame with MultiIndex (instrument, datetime) index.
    """
    if df.empty:
        return df
    
    # Already normalized?
    if isinstance(df.index, pd.MultiIndex):
        return df
    
    # Create MultiIndex index
    new_index = pd.MultiIndex.from_product(
        [[instrument], df.index],
        names=['instrument', 'datetime']
    )
    
    df_normalized = df.copy()
    df_normalized.index = new_index
    
    return df_normalized


def normalize_qlib_features_result(
    df: pd.DataFrame,
    instruments: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Normalize Qlib D.features() result to unified format.
    
    This is the main entry point for normalizing Qlib data.
    It automatically detects the format and normalizes it.
    
    Args:
        df: DataFrame from Qlib D.features().
        instruments: Optional list of instruments (for single instrument queries).
    
    Returns:
        Normalized DataFrame with:
        - Index: MultiIndex (instrument, datetime)
        - Columns: Single-level field names
    """
    if df.empty:
        return df
    
    # Already normalized?
    if isinstance(df.index, pd.MultiIndex) and not isinstance(df.columns, pd.MultiIndex):
        return df
    
    # Case: Single instrument, single-level columns
    if not isinstance(df.index, pd.MultiIndex) and not isinstance(df.columns, pd.MultiIndex):
        if instruments and len(instruments) == 1:
            return normalize_qlib_dataframe_with_instrument(df, instruments[0])
        else:
            raise ValueError(
                "Cannot normalize single instrument data without instrument information. "
                "Please provide instruments parameter."
            )
    
    # Case: MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        return normalize_qlib_dataframe(df)
    
    # Case: MultiIndex index (should already be normalized)
    return df


def validate_normalized_format(
    df: pd.DataFrame,
    context: str = "DataFrame"
) -> None:
    """
    Validate that DataFrame is in normalized format.
    
    Normalized format requirements:
    - Index: MultiIndex (instrument, datetime) - REQUIRED
    - Columns: Single-level field names - REQUIRED (NOT MultiIndex)
    
    Args:
        df: DataFrame to validate.
        context: Context string for error messages (e.g., "market_data").
    
    Raises:
        ValueError: If DataFrame format is not normalized.
    
    Examples:
        >>> # Valid normalized format
        >>> df = pd.DataFrame(
        ...     {'$close': [10.0, 10.2], '$open': [10.1, 10.3]},
        ...     index=pd.MultiIndex.from_tuples(
        ...         [('000001.SZ', pd.Timestamp('2025-01-01')),
        ...          ('000001.SZ', pd.Timestamp('2025-01-02'))],
        ...         names=['instrument', 'datetime']
        ...     )
        ... )
        >>> validate_normalized_format(df)  # No error
        
        >>> # Invalid: Single-level index
        >>> df = pd.DataFrame({'$close': [10.0]}, index=[pd.Timestamp('2025-01-01')])
        >>> validate_normalized_format(df)  # Raises ValueError
        
        >>> # Invalid: MultiIndex columns
        >>> df = pd.DataFrame(
        ...     {('000001.SZ', '$close'): [10.0]},
        ...     index=pd.MultiIndex.from_tuples([('000001.SZ', pd.Timestamp('2025-01-01'))])
        ... )
        >>> validate_normalized_format(df)  # Raises ValueError
    """
    if df.empty:
        # Empty DataFrame is considered valid (no data to validate)
        return
    
    # Check Index: MUST be MultiIndex
    if not isinstance(df.index, pd.MultiIndex):
        raise ValueError(
            f"{context} must have MultiIndex index (instrument, datetime). "
            f"Got: {type(df.index).__name__}. "
            f"Please use normalize_qlib_features_result() to normalize the data."
        )
    
    # Check Index levels: Should have 'instrument' and 'datetime' levels
    if len(df.index.names) < 2:
        raise ValueError(
            f"{context} MultiIndex must have at least 2 levels (instrument, datetime). "
            f"Got: {df.index.names}. "
            f"Please use normalize_qlib_features_result() to normalize the data."
        )
    
    # Check Columns: MUST be single-level (NOT MultiIndex)
    if isinstance(df.columns, pd.MultiIndex):
        raise ValueError(
            f"{context} must have single-level columns. "
            f"Got: MultiIndex columns. "
            f"Please use normalize_qlib_features_result() to normalize the data."
        )

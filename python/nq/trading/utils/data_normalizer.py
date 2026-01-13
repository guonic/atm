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
    
    # Case 1: Already normalized (MultiIndex index, single-level columns)
    if isinstance(df.index, pd.MultiIndex) and not isinstance(df.columns, pd.MultiIndex):
        # Already in target format
        return df
    
    # Case 2: Single-level index, MultiIndex columns
    # Format: Index=datetime, Columns=MultiIndex(instrument, field)
    if not isinstance(df.index, pd.MultiIndex) and isinstance(df.columns, pd.MultiIndex):
        # Extract instrument and field from columns
        instruments = df.columns.get_level_values(0).unique().tolist()
        fields = df.columns.get_level_values(1).unique().tolist()
        
        # Build new DataFrame with MultiIndex (instrument, datetime)
        # Use stack/unstack to reshape more efficiently
        # First, set index name if not set
        if df.index.name is None:
            df.index.name = 'datetime'
        
        # Stack to get (datetime, instrument, field) structure
        stacked = df.stack(level=0)  # Stack instrument level
        stacked.index.names = ['datetime', 'instrument']
        
        # Unstack field level to get fields as columns
        normalized_df = stacked.unstack(level=-1)  # Unstack field level
        
        # Swap levels to get (instrument, datetime) index
        normalized_df = normalized_df.swaplevel(0, 1).sort_index()
        
        # Flatten column MultiIndex to single level (keep only field names)
        normalized_df.columns = normalized_df.columns.get_level_values(-1)
        
        return normalized_df
    
    # Case 3: Single-level index, single-level columns
    # This happens when querying single instrument
    # We need to infer the instrument from context or use a placeholder
    if not isinstance(df.index, pd.MultiIndex) and not isinstance(df.columns, pd.MultiIndex):
        # This is tricky - we don't know the instrument
        # For now, raise an error and require instrument to be passed
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

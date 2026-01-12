"""
Data validation utilities.

Validates and filters market data to ensure no NaN values enter the system.
"""

from typing import List, Tuple, Optional
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def validate_and_filter_nan(
    market_data: pd.DataFrame,
    required_fields: List[str],
    context: str = "market data",
) -> Tuple[pd.DataFrame, List[Tuple[str, str, List[str]]]]:
    """
    Validate market data and filter out rows with NaN values.
    
    This function checks for NaN values in required fields and filters them out,
    while logging detailed warnings about which instruments and dates have NaN.
    
    Args:
        market_data: Market data DataFrame from Qlib.
        required_fields: List of required field names (e.g., ['$close', '$open']).
        context: Context string for logging (e.g., 'backtest', 'training').
    
    Returns:
        Tuple of:
        - Filtered DataFrame (with NaN rows removed)
        - List of (instrument, date, nan_fields) tuples for logging
    
    Raises:
        ValueError: If all rows are filtered out (all data is NaN).
    """
    if market_data.empty:
        return market_data, []
    
    # Detect NaN rows
    if isinstance(market_data.columns, pd.MultiIndex):
        # MultiIndex columns: (instrument, field)
        # Check each required field across all instruments
        nan_mask = pd.Series(False, index=market_data.index)
        for field in required_fields:
            # Get all columns for this field
            field_cols = [col for col in market_data.columns if col[1] == field]
            if field_cols:
                field_data = market_data[field_cols]
                nan_mask = nan_mask | field_data.isna().any(axis=1)
    else:
        # Single level columns
        nan_mask = market_data[required_fields].isna().any(axis=1)
    
    nan_count = nan_mask.sum()
    
    if nan_count == 0:
        # No NaN found, return original data
        return market_data, []
    
    # Collect NaN details for logging
    nan_details = []
    nan_rows = market_data[nan_mask]
    
    # Get unique instruments and dates with NaN
    if isinstance(market_data.index, pd.MultiIndex):
        # MultiIndex: (instrument, datetime)
        nan_instruments = market_data.index[nan_mask].get_level_values(0).unique().tolist()
        nan_dates = market_data.index[nan_mask].get_level_values(1).unique().tolist()
        
        for idx in nan_rows.index:
            instrument = idx[0] if isinstance(idx, tuple) else None
            date = idx[1] if isinstance(idx, tuple) else None
            row = nan_rows.loc[idx]
            
            # Find which fields have NaN
            nan_fields = []
            if isinstance(market_data.columns, pd.MultiIndex):
                for field in required_fields:
                    field_cols = [col for col in market_data.columns if col[1] == field]
                    if field_cols:
                        field_values = row[field_cols]
                        if field_values.isna().any():
                            nan_fields.append(field)
            else:
                for field in required_fields:
                    if pd.isna(row.get(field, default=None)):
                        nan_fields.append(field)
            
            if nan_fields:
                nan_details.append((instrument, str(date), nan_fields))
    else:
        # Single index
        nan_instruments = market_data.index[nan_mask].unique().tolist()
        nan_dates = []
        
        for idx in nan_rows.index:
            row = nan_rows.loc[idx]
            nan_fields = []
            for field in required_fields:
                if pd.isna(row.get(field, default=None)):
                    nan_fields.append(field)
            
            if nan_fields:
                nan_details.append((str(idx), None, nan_fields))
    
    # Log detailed warnings
    logger.warning(
        f"⚠ Found {nan_count} rows with NaN values in {context}. "
        f"Filtering them out to prevent system errors."
    )
    logger.warning(
        f"  Affected instruments ({len(nan_instruments)}): {nan_instruments[:20]}{'...' if len(nan_instruments) > 20 else ''}"
    )
    if nan_dates:
        logger.warning(
            f"  Affected dates ({len(nan_dates)}): {sorted(nan_dates)[:10]}{'...' if len(nan_dates) > 10 else ''}"
        )
    
    # Log first 10 detailed NaN entries
    for i, (instrument, date, nan_fields) in enumerate(nan_details[:10]):
        if date:
            logger.warning(f"  {instrument} on {date}: NaN in {nan_fields}")
        else:
            logger.warning(f"  {instrument}: NaN in {nan_fields}")
    
    if len(nan_details) > 10:
        logger.warning(f"  ... and {len(nan_details) - 10} more NaN entries")
    
    # Filter out NaN rows
    filtered_data = market_data[~nan_mask].copy()
    
    if filtered_data.empty:
        raise ValueError(
            f"CRITICAL: All rows in {context} contain NaN values after filtering. "
            f"Cannot proceed. Please check Qlib data source."
        )
    
    logger.info(
        f"✓ Filtered {context}: removed {nan_count} NaN rows, "
        f"kept {len(filtered_data)} valid rows"
    )
    
    return filtered_data, nan_details


def validate_single_instrument_data(
    data: pd.DataFrame,
    required_fields: List[str],
    symbol: str,
    date: Optional[pd.Timestamp] = None,
) -> bool:
    """
    Validate data for a single instrument.
    
    Args:
        data: Data DataFrame (should have single-level columns with field names).
        required_fields: List of required field names.
        symbol: Stock symbol (for logging).
        date: Optional date (for logging).
    
    Returns:
        True if data is valid (no NaN), False otherwise.
    """
    if data.empty:
        logger.warning(f"No data for {symbol}" + (f" on {date}" if date else ""))
        return False
    
    # Check for NaN in required fields
    for field in required_fields:
        if field not in data.columns:
            logger.warning(
                f"Missing field '{field}' for {symbol}" + (f" on {date}" if date else "")
            )
            return False
        
        if data[field].isna().any():
            nan_count = data[field].isna().sum()
            logger.warning(
                f"NaN values in '{field}' for {symbol}" + (f" on {date}" if date else "") +
                f": {nan_count} NaN values found"
            )
            return False
    
    return True

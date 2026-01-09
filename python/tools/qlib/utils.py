#!/usr/bin/env python3
"""
Utility functions for Qlib data handling.

This module provides common utilities for working with Qlib data handlers,
especially for handling inconsistent API behavior across different versions.
"""

import logging
from datetime import datetime
from typing import TYPE_CHECKING, List, Optional, Tuple

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from qlib.contrib.data.handler import Alpha158

try:
    from qlib.data import D
except ImportError:
    D = None

logger = logging.getLogger(__name__)


def get_handler_data(handler: "Alpha158", col_set: str = "feature") -> pd.DataFrame:
    """
    Get data from Alpha158 handler with fallback mechanism.

    This function attempts to get data using handler.data property first,
    and falls back to handler.fetch() if the property is unavailable or empty.
    This is necessary because Qlib's Alpha158 handler has inconsistent behavior
    across different versions and usage scenarios:
    - In some cases, handler.data is available after setup_data()
    - In other cases, handler.data may be None, empty, or raise exceptions
    - handler.fetch() is the more reliable method but may require data_key

    Args:
        handler: Alpha158 handler instance (must have setup_data() called).
        col_set: Column set to fetch (default: "feature").

    Returns:
        DataFrame containing the requested data. Returns empty DataFrame
        if both methods fail.

    Example:
        >>> handler = Alpha158(start_time="2024-01-01", end_time="2024-01-01")
        >>> handler.setup_data()
        >>> df = get_handler_data(handler, col_set="feature")
    """
    # Try using handler.data property first (may be faster if already loaded)
    try:
        df = handler.data
        if df is not None and not df.empty:
            return df
    except (AttributeError, Exception) as e:
        logger.debug(f"handler.data not available: {e}")

    # Fallback: use fetch method (more reliable)
    try:
        df = handler.fetch(col_set=col_set)
        if df is not None and not df.empty:
            return df
    except Exception as e:
        logger.warning(f"handler.fetch() failed: {e}")

    # If both methods fail, return empty DataFrame
    logger.warning(
        "Both handler.data and handler.fetch() failed, returning empty DataFrame"
    )
    return pd.DataFrame()


def load_next_day_returns(
    current_date: datetime,
    date_range: List[datetime],
    current_idx: int,
    instruments: Optional[List[str]] = None,
) -> Optional[pd.DataFrame]:
    """
    Load next day returns as labels for training.

    This function calculates the next trading day's return as labels for the
    current date. If the current date is the last date in the range, returns None.

    Args:
        current_date: Current training date.
        date_range: List of all dates in the training range.
        current_idx: Index of current_date in date_range.
        instruments: Optional list of instrument codes. If None, uses all available
            instruments from D.instruments().

    Returns:
        DataFrame containing next day returns with MultiIndex (datetime, instrument),
        or None if current_date is the last date in date_range.

    Example:
        >>> date_range = [datetime(2024, 1, 1), datetime(2024, 1, 2), datetime(2024, 1, 3)]
        >>> labels = load_next_day_returns(
        ...     current_date=date_range[0],
        ...     date_range=date_range,
        ...     current_idx=0
        ... )
        >>> # Returns DataFrame with returns from 2024-01-01 to 2024-01-02
    """
    if D is None:
        raise ImportError("qlib.data.D is not available. Please initialize Qlib first.")

    # Check if there's a next date
    if current_idx >= len(date_range) - 1:
        # Last date, no next day return available
        logger.debug(f"No next day return available for last date: {current_date}")
        return None

    next_date = date_range[current_idx + 1]
    current_date_str = current_date.strftime("%Y-%m-%d")
    next_date_str = next_date.strftime("%Y-%m-%d")

    # Get instruments if not provided
    if instruments is None:
        instruments = D.instruments()
        if not isinstance(instruments, list):
            instruments = list(instruments)

    # Calculate next day returns using Qlib expression
    # Ref($close, -1) / $close - 1 means: (next_day_close / current_close) - 1
    df_y = D.features(
        instruments,
        ["Ref($close, -1) / $close - 1"],  # Next day return
        start_time=current_date_str,
        end_time=next_date_str,
    )

    return df_y


def align_and_clean_features_labels(
    df_x: pd.DataFrame,
    df_y: Optional[pd.DataFrame],
    fill_value: float = 0.0,
    log_stats: bool = False,
    context: Optional[str] = None,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Align features and labels DataFrames by common indices and clean them.

    This function performs the following operations:
    1. If labels (df_y) are provided, finds common indices between features and labels
    2. Filters both DataFrames to common indices
    3. Cleans NaN and Inf values in both DataFrames
    4. If labels are not provided, only cleans features DataFrame
    5. Validates that aligned data is not empty

    Args:
        df_x: Features DataFrame.
        df_y: Optional labels DataFrame. If None, only features are cleaned.
        fill_value: Value to use for filling NaN and replacing Inf (default: 0.0).
        log_stats: If True, log statistics about cleaning operations (default: False).
        context: Optional context string for logging (e.g., date string).

    Returns:
        Tuple of (aligned_features, aligned_labels):
            - aligned_features: Cleaned and aligned features DataFrame.
            - aligned_labels: Cleaned and aligned labels DataFrame, or None if df_y was None.

    Example:
        >>> df_x = pd.DataFrame({'a': [1, 2, 3]}, index=['A', 'B', 'C'])
        >>> df_y = pd.DataFrame({'ret': [0.1, 0.2]}, index=['A', 'B'])
        >>> df_x_aligned, df_y_aligned = align_and_clean_features_labels(
        ...     df_x, df_y, context="2024-01-01"
        ... )
        >>> # Both DataFrames are aligned to common indices ['A', 'B'] and cleaned
    """
    # Case 1: Labels are provided
    if df_y is not None and not df_y.empty:
        # Find common indices between features and labels
        common_idx = df_x.index.intersection(df_y.index)

        if len(common_idx) == 0:
            context_str = f" ({context})" if context else ""
            raise ValueError(
                f"No common indices between features and labels{context_str}"
            )

        # Filter to common indices
        df_x_aligned = df_x.loc[common_idx]
        df_y_aligned = df_y.loc[common_idx]

        # Clean aligned data
        context_x = f"aligned features{(' for ' + context) if context else ''}"
        df_x_aligned = clean_dataframe(
            df_x_aligned, fill_value=fill_value, log_stats=log_stats, context=context_x
        )

        context_y = f"aligned labels{(' for ' + context) if context else ''}"
        df_y_aligned = clean_dataframe(
            df_y_aligned, fill_value=fill_value, log_stats=log_stats, context=context_y
        )

        # Validate aligned data
        if df_x_aligned.empty:
            context_str = f" ({context})" if context else ""
            raise ValueError(f"No aligned data after cleaning{context_str}")

        return df_x_aligned, df_y_aligned

    # Case 2: No labels provided, only clean features
    context_x = f"features{(' (no labels) for ' + context) if context else ' (no labels)'}"
    df_x_aligned = clean_dataframe(
        df_x, fill_value=fill_value, log_stats=log_stats, context=context_x
    )

    # Validate cleaned features are not empty (same validation as Case 1)
    if df_x_aligned.empty:
        context_str = f" ({context})" if context else ""
        raise ValueError(
            f"Features DataFrame is empty after cleaning{context_str}. "
            f"This may indicate all data was invalid or removed during cleaning."
        )

    return df_x_aligned, None


def is_valid_number(value: Optional[float]) -> bool:
    """
    Check if a value is a valid number (not None, not NaN, not Inf).

    This function is commonly used to validate loss values, metrics, or other
    numerical results before using them in calculations or logging.

    Args:
        value: Value to check. Can be None, int, float, or other types.

    Returns:
        True if value is a valid number (not None, not NaN, not Inf), False otherwise.

    Example:
        >>> is_valid_number(1.5)  # True
        >>> is_valid_number(None)  # False
        >>> is_valid_number(float('nan'))  # False
        >>> is_valid_number(float('inf'))  # False
        >>> is_valid_number(-float('inf'))  # False
    """
    if value is None:
        return False

    if not isinstance(value, (int, float)):
        return False

    # Check for NaN: NaN != NaN is True
    if value != value:
        return False

    # Check for Inf
    if value == float('inf') or value == float('-inf'):
        return False

    return True


def clean_dataframe(
    df: pd.DataFrame,
    fill_value: float = 0.0,
    inplace: bool = False,
    log_stats: bool = False,
    context: Optional[str] = None,
) -> pd.DataFrame:
    """
    Clean DataFrame by filling NaN values and replacing Inf values.

    This function is commonly used for cleaning feature and label DataFrames
    before training or inference. It fills NaN values with a specified value
    (default: 0.0) and replaces positive/negative infinity with the same value.

    Args:
        df: DataFrame to clean.
        fill_value: Value to use for filling NaN and replacing Inf (default: 0.0).
        inplace: If True, modify DataFrame in place (default: False).
        log_stats: If True, log statistics about cleaning operations (default: False).
        context: Optional context string for logging (e.g., "features", "labels").

    Returns:
        Cleaned DataFrame. If inplace=True, returns the same DataFrame object.

    Example:
        >>> df = pd.DataFrame({'a': [1, np.nan, np.inf], 'b': [2, 3, -np.inf]})
        >>> df_clean = clean_dataframe(df, log_stats=True, context="features")
        >>> # NaN and Inf values are replaced with 0.0
    """
    if df is None or df.empty:
        return df

    # Work on a copy if not inplace
    if not inplace:
        df = df.copy()

    # Count NaN and Inf before cleaning
    nan_count = df.isna().sum().sum()
    inf_count = ((df == np.inf) | (df == -np.inf)).sum().sum()

    # Fill NaN values
    if nan_count > 0:
        df = df.fillna(fill_value)

    # Replace Inf values
    if inf_count > 0:
        df = df.replace([np.inf, -np.inf], fill_value)

    # Log statistics if requested
    if log_stats and (nan_count > 0 or inf_count > 0):
        context_str = f" ({context})" if context else ""
        logger.debug(
            f"Cleaned DataFrame{context_str}: {nan_count} NaN values, "
            f"{inf_count} Inf values replaced with {fill_value}"
        )

    return df


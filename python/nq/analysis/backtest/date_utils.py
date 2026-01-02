"""
Date utility functions for backtest data processing.

All date conversions should use these functions to ensure consistency.
Invalid inputs will cause program to crash.
"""

from datetime import date

import pandas as pd


def to_date(date_val: date | pd.Timestamp | str) -> date:
    """
    Convert date value to standard date object.
    
    Args:
        date_val: Date value (date, Timestamp, or str).
        
    Returns:
        date object.
    """
    return pd.to_datetime(date_val).date()


def normalize_dataframe_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize DataFrame index to ensure date index is datetime type.
    
    Args:
        df: DataFrame with date index.
        
    Returns:
        DataFrame with normalized datetime index.
    """
    df.index = pd.to_datetime(df.index)
    return df

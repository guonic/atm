"""
Feature loading utilities for trading framework.

Extracted from backtest_structure_expert.py to avoid circular dependencies.
"""

from typing import Optional, List
from datetime import datetime, timedelta
import pandas as pd
import logging

from qlib.data import D
from qlib.contrib.data.handler import Alpha158
from tools.qlib.utils import clean_dataframe, get_handler_data

logger = logging.getLogger(__name__)


def get_qlib_data_range() -> tuple[Optional[datetime.date], Optional[datetime.date]]:
    """
    Get the date range of available Qlib data.

    Returns:
        Tuple of (start_date, end_date) as date objects, or (None, None) if no data.
    """
    full_calendar = D.calendar()
    if len(full_calendar) == 0:
        return None, None

    data_start_ts = full_calendar[0]
    data_end_ts = full_calendar[-1]

    # Convert Timestamp to date
    if isinstance(data_start_ts, pd.Timestamp):
        data_start_date = data_start_ts.date()
    elif hasattr(data_start_ts, 'date'):
        data_start_date = data_start_ts.date()
    else:
        data_start_date = data_start_ts

    if isinstance(data_end_ts, pd.Timestamp):
        data_end_date = data_end_ts.date()
    elif hasattr(data_end_ts, 'date'):
        data_end_date = data_end_ts.date()
    else:
        data_end_date = data_end_ts

    return data_start_date, data_end_date


def _calculate_fit_period(
    lookback_start: datetime.date,
    data_start_date: Optional[datetime.date],
    lookback_start_str: str,
) -> tuple[str, str]:
    """
    Calculate the fit period for Alpha158 feature handler.

    Args:
        lookback_start: Start date of lookback period.
        data_start_date: Start date of available data.
        lookback_start_str: String representation of lookback start date.

    Returns:
        Tuple of (fit_start_str, fit_end_str) date strings.
    """
    fit_end_date = lookback_start - timedelta(days=1)
    fit_start_date = fit_end_date - timedelta(days=365)  # 1 year of data for fitting

    # Ensure fit period is within available data range
    if data_start_date is not None:
        if fit_start_date < data_start_date:
            logger.debug(f"Fit start {fit_start_date} is before data start {data_start_date}, adjusting...")
            fit_start_date = data_start_date
        if fit_end_date < data_start_date:
            logger.warning(
                f"Fit end {fit_end_date} is before data start {data_start_date}, using minimal fit period"
            )
            fit_start_date = data_start_date
            # Use at least 30 days for fit if possible
            fit_calendar = D.calendar(start_time=data_start_date.strftime("%Y-%m-%d"), end_time=lookback_start_str)
            if len(fit_calendar) > 30:
                fit_end_ts = fit_calendar[-30]
                fit_end_date = fit_end_ts.date() if hasattr(fit_end_ts, 'date') else fit_end_ts
            else:
                if len(fit_calendar) > 0:
                    fit_end_ts = fit_calendar[-1]
                    fit_end_date = fit_end_ts.date() if hasattr(fit_end_ts, 'date') else fit_end_ts
                else:
                    fit_end_date = lookback_start

    fit_start_str = fit_start_date.strftime("%Y-%m-%d")
    fit_end_str = fit_end_date.strftime("%Y-%m-%d")
    return fit_start_str, fit_end_str


def load_features_for_date(
    trade_date: pd.Timestamp,
    lookback_days: int,
    instruments: Optional[List[str]],
    data_start_date: Optional[datetime.date],
) -> Optional[pd.DataFrame]:
    """
    Load Alpha158 features for a specific trading date.

    Args:
        trade_date: Trading date to load features for.
        lookback_days: Number of days to look back for feature calculation.
        instruments: Optional list of instruments to load.
        data_start_date: Start date of available data.

    Returns:
        DataFrame with features, or None if loading fails.
    """
    trade_date_str = trade_date.strftime("%Y-%m-%d")
    lookback_start = trade_date - timedelta(days=lookback_days)
    lookback_start_str = lookback_start.strftime("%Y-%m-%d")

    # Convert lookback_start to date for fit period calculation
    if isinstance(lookback_start, pd.Timestamp):
        lookback_start_date = lookback_start.date()
    elif hasattr(lookback_start, 'date'):
        lookback_start_date = lookback_start.date()
    else:
        lookback_start_date = lookback_start

    fit_start_str, fit_end_str = _calculate_fit_period(lookback_start_date, data_start_date, lookback_start_str)

    logger.debug(f"Alpha158 fit period: {fit_start_str} to {fit_end_str}")
    logger.debug(f"Alpha158 inference period: {lookback_start_str} to {trade_date_str}")

    # Try with lookback first
    try:
        handler_kwargs = {
            "start_time": lookback_start_str,
            "end_time": trade_date_str,
            "fit_start_time": fit_start_str,
            "fit_end_time": fit_end_str,
        }
        if instruments is not None:
            handler_kwargs["instruments"] = instruments

        date_handler = Alpha158(**handler_kwargs)
        date_handler.setup_data()

        # Get features using unified data retrieval function
        df_x = get_handler_data(date_handler, col_set="feature")
    except Exception as e:
        logger.debug(f"Failed to load with lookback ({lookback_start_str} to {trade_date_str}): {e}")
        # Try without lookback
        logger.debug(f"Trying without lookback (using only {trade_date_str})...")
        handler_kwargs = {
            "start_time": trade_date_str,
            "end_time": trade_date_str,
            "fit_start_time": fit_start_str,
            "fit_end_time": fit_end_str,
        }
        if instruments is not None:
            handler_kwargs["instruments"] = instruments

        date_handler = Alpha158(**handler_kwargs)
        date_handler.setup_data()

        # Get features using unified data retrieval function
        df_x = get_handler_data(date_handler, col_set="feature")

    if df_x.empty:
        # Diagnostic check
        try:
            instruments_check = D.instruments()
            logger.debug(f"Total instruments available: {len(instruments_check)}")
            if len(instruments_check) > 0:
                sample_stock = instruments_check[0]
                sample_data = D.features(
                    [sample_stock],
                    ["$close"],
                    start_time=trade_date_str,
                    end_time=trade_date_str,
                    freq="day",
                )
                logger.debug(f"Sample stock {sample_stock} data for {trade_date_str}: {sample_data.shape}")
        except Exception as diag_e:
            logger.debug(f"Diagnostic check failed: {diag_e}")

        logger.warning(
            f"No data for {trade_date_str} (with lookback from {lookback_start_str}). "
            f"Alpha158 returned empty DataFrame."
        )
        return None

    # Filter to only the target date
    if isinstance(df_x.index, pd.MultiIndex):
        date_level = df_x.index.get_level_values(0)
        if isinstance(date_level[0], pd.Timestamp):
            df_x = df_x.loc[date_level.date == trade_date.date()]
        else:
            date_strs = pd.to_datetime(date_level).dt.strftime("%Y-%m-%d")
            df_x = df_x.loc[date_strs == trade_date_str]

    if df_x.empty:
        logger.warning(f"No data for target date {trade_date_str} after filtering")
        return None

    # Clean NaN/Inf using unified function
    df_x = clean_dataframe(df_x, fill_value=0.0, log_stats=True, context=f"features for {trade_date_str}")
    return df_x

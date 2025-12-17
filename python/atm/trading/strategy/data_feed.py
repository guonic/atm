"""
Data feed adapter for backtrader.

Converts database K-line data to backtrader-compatible format.
"""

import logging
from datetime import datetime
from typing import Optional

import backtrader as bt
import pandas as pd

from atm.config import DatabaseConfig
from atm.repo.kline_repo import StockKlineDayRepo

logger = logging.getLogger(__name__)


class DatabaseDataFeed(bt.feeds.PandasData):
    """
    Backtrader data feed from database.

    Loads K-line data from database and converts it to backtrader format.
    """

    params = (
        ("datetime", None),  # Use index as datetime
        ("open", "open"),
        ("high", "high"),
        ("low", "low"),
        ("close", "close"),
        ("volume", "volume"),
        ("openinterest", -1),  # Not used
        ("db_config", None),  # Database configuration
        ("ts_code", None),  # Stock code
        ("start_date", None),  # Start date
        ("end_date", None),  # End date
        ("schema", "quant"),  # Database schema
    )

    def _load(self):
        """
        Load data from database.

        This method is called by backtrader to load the data.
        """
        # If dataname is already set (DataFrame passed directly), use it
        if self.p.dataname is not None and isinstance(self.p.dataname, pd.DataFrame):
            return super()._load()

        # Otherwise, load from database using params
        db_config = self.p.db_config
        ts_code = self.p.ts_code

        if db_config is None or ts_code is None:
            raise ValueError("db_config and ts_code must be provided as params")

        # Load data from database
        df = self._load_data_from_db(
            db_config=db_config,
            ts_code=ts_code,
            start_date=self.p.start_date,
            end_date=self.p.end_date,
            schema=self.p.schema,
        )

        if df.empty:
            logger.warning(
                f"No data found for {ts_code} "
                f"in date range {self.p.start_date} to {self.p.end_date}"
            )
            # Create empty DataFrame with required columns
            df = pd.DataFrame(
                columns=["datetime", "open", "high", "low", "close", "volume"]
            )
            df.set_index("datetime", inplace=True)

        # Set the DataFrame as dataname
        self.p.dataname = df

        # Call parent's _load method
        return super()._load()

    def _load_data_from_db(
        self,
        db_config: DatabaseConfig,
        ts_code: str,
        start_date: Optional[datetime],
        end_date: Optional[datetime],
        schema: str,
    ) -> pd.DataFrame:
        """
        Load K-line data from database.

        Args:
            db_config: Database configuration.
            ts_code: Stock code.
            start_date: Start date.
            end_date: End date.
            schema: Database schema.

        Returns:
            DataFrame with OHLCV data.
        """
        repo = StockKlineDayRepo(db_config, schema)
        klines = repo.get_by_ts_code(
            ts_code=ts_code,
            start_time=start_date,
            end_time=end_date,
        )

        if not klines:
            return pd.DataFrame()

        # Convert to DataFrame
        data_list = []
        for kline in klines:
            data_list.append(
                {
                    "datetime": kline.trade_date,
                    "open": float(kline.open) if kline.open else None,
                    "high": float(kline.high) if kline.high else None,
                    "low": float(kline.low) if kline.low else None,
                    "close": float(kline.close) if kline.close else None,
                    "volume": int(kline.volume) if kline.volume else 0,
                }
            )

        df = pd.DataFrame(data_list)

        if df.empty:
            return df

        # Set datetime as index and ensure it's DatetimeIndex
        df.set_index("datetime", inplace=True)
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        df.sort_index(inplace=True)

        # Ensure all required columns exist
        required_cols = ["open", "high", "low", "close", "volume"]
        for col in required_cols:
            if col not in df.columns:
                df[col] = 0.0 if col == "volume" else None

        # Fill missing values (forward fill for prices, 0 for volume)
        df["open"] = df["open"].ffill()
        df["high"] = df["high"].ffill()
        df["low"] = df["low"].ffill()
        df["close"] = df["close"].ffill()
        df["volume"] = df["volume"].fillna(0)

        # Drop rows with any remaining NaN values
        df.dropna(inplace=True)

        logger.info(
            f"Loaded {len(df)} bars for {ts_code} from {df.index[0]} to {df.index[-1]}"
        )

        return df


def _load_kline_data(
    db_config: DatabaseConfig,
    ts_code: str,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    schema: str = "quant",
) -> pd.DataFrame:
    """
    Load K-line data from database and convert to DataFrame.

    Args:
        db_config: Database configuration.
        ts_code: Stock code.
        start_date: Start date.
        end_date: End date.
        schema: Database schema.

    Returns:
        DataFrame with OHLCV data.
    """
    repo = StockKlineDayRepo(db_config, schema)
    klines = repo.get_by_ts_code(
        ts_code=ts_code,
        start_time=start_date,
        end_time=end_date,
    )

    if not klines:
        return pd.DataFrame()

    # Convert to DataFrame
    data_list = []
    for kline in klines:
        data_list.append(
            {
                "datetime": kline.trade_date,
                "open": float(kline.open) if kline.open else None,
                "high": float(kline.high) if kline.high else None,
                "low": float(kline.low) if kline.low else None,
                "close": float(kline.close) if kline.close else None,
                "volume": int(kline.volume) if kline.volume else 0,
            }
        )

    df = pd.DataFrame(data_list)

    if df.empty:
        return df

    # Set datetime as index and ensure it's DatetimeIndex
    df.set_index("datetime", inplace=True)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)

    # Ensure all required columns exist
    required_cols = ["open", "high", "low", "close", "volume"]
    for col in required_cols:
        if col not in df.columns:
            df[col] = 0.0 if col == "volume" else None

    # Fill missing values (forward fill for prices, 0 for volume)
    df["open"] = df["open"].ffill()
    df["high"] = df["high"].ffill()
    df["low"] = df["low"].ffill()
    df["close"] = df["close"].ffill()
    df["volume"] = df["volume"].fillna(0)

    # Drop rows with any remaining NaN values
    df.dropna(inplace=True)

    logger.info(
        f"Loaded {len(df)} bars for {ts_code} from {df.index[0]} to {df.index[-1]}"
    )

    return df


def create_data_feed(
    db_config: DatabaseConfig,
    ts_code: str,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    schema: str = "quant",
) -> bt.feeds.PandasData:
    """
    Create a database data feed for backtrader.

    Args:
        db_config: Database configuration.
        ts_code: Stock code (e.g., '000001.SZ').
        start_date: Start date for data (inclusive).
        end_date: End date for data (inclusive).
        schema: Database schema name.

    Returns:
        PandasData instance with loaded data.
    """
    # Load data from database
    df = _load_kline_data(
        db_config=db_config,
        ts_code=ts_code,
        start_date=start_date,
        end_date=end_date,
        schema=schema,
    )

    if df.empty:
        logger.warning(
            f"No data found for {ts_code} in date range {start_date} to {end_date}"
        )
        # Create empty DataFrame with required columns
        df = pd.DataFrame(
            columns=["datetime", "open", "high", "low", "close", "volume"]
        )
        df.set_index("datetime", inplace=True)

    # Create a simple wrapper class that sets dataname before initialization
    # This avoids the issue where backtrader tries to access dataname before it's set
    class SimplePandasData(bt.feeds.PandasData):
        params = (
            ("datetime", None),
            ("open", "open"),
            ("high", "high"),
            ("low", "low"),
            ("close", "close"),
            ("volume", "volume"),
            ("openinterest", -1),
        )

    # Create instance with dataname parameter
    # Backtrader's metabase system should handle this correctly
    feed = SimplePandasData(dataname=df)
    return feed


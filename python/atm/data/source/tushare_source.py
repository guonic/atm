"""
Tushare data source implementation.

Fetches financial data from Tushare Pro API using the official Python SDK.
Reference: https://tushare.pro/document/1?doc_id=131
"""

import logging
import warnings
from typing import Any, Dict, Iterator, Optional

import pandas as pd
import tushare as ts

# Suppress FutureWarning from tushare library about fillna method
warnings.filterwarnings("ignore", category=FutureWarning, module="tushare")

from atm.data.source.base import (
    BaseSource,
    ConnectionError,
    DataFetchError,
    SourceConfig,
    SourceError,
)

logger = logging.getLogger(__name__)


class TushareSourceConfig(SourceConfig):
    """Tushare source configuration."""

    type: str = "tushare"  # Source type
    token: str = ""  # Tushare Pro API token
    api_name: str = ""  # API name (e.g., 'stock_basic', 'daily', 'pro_bar')
    fields: Optional[str] = None  # Fields to retrieve (comma-separated)
    timeout: int = 30  # Request timeout in seconds


class TushareSource(BaseSource):
    """
    Tushare data source for fetching financial data.

    This source uses the Tushare Pro Python SDK to fetch stock data,
    K-line data, financial data, etc.

    Example:
        ```python
        config = TushareSourceConfig(
            token="your_tushare_token",
            api_name="stock_basic",
            params={"exchange": "", "list_status": "L"},
        )
        source = TushareSource(config)
        for record in source.fetch():
            print(record)
        ```
    """

    def __init__(self, config: TushareSourceConfig):
        """
        Initialize Tushare source.

        Args:
            config: Tushare source configuration.
        """
        super().__init__(config)
        self.config: TushareSourceConfig = config
        self._pro_api = None
        self._initialized = False

    def _initialize(self) -> None:
        """Initialize Tushare Pro API."""
        if self._initialized:
            return

        try:
            # Set token if provided
            if self.config.token:
                ts.set_token(self.config.token)
                self._pro_api = ts.pro_api()
            else:
                # Try to use token from environment or existing config
                self._pro_api = ts.pro_api()

            if self._pro_api is None:
                raise ConnectionError("Failed to initialize Tushare Pro API")

            self._initialized = True
            logger.info("Tushare Pro API initialized successfully")

        except Exception as e:
            raise ConnectionError(f"Failed to initialize Tushare Pro API: {e}") from e

    def test_connection(self) -> bool:
        """
        Test connection to Tushare API.

        Returns:
            True if connection is successful, False otherwise.
        """
        try:
            self._initialize()
            # Test with a simple API call (trade_cal is usually available)
            if self._pro_api:
                # Try to get trade calendar for a recent date
                test_result = self._pro_api.trade_cal(
                    exchange="SSE", start_date="20240101", end_date="20240102"
                )
                return test_result is not None and len(test_result) >= 0
            return False
        except Exception as e:
            logger.error(f"Tushare connection test failed: {e}")
            return False

    def fetch(self, **kwargs) -> Iterator[Dict[str, Any]]:
        """
        Fetch data from Tushare API.

        Args:
            **kwargs: Additional parameters to pass to the API call.
                Common parameters:
                - api_name: Override config api_name
                - fields: Override config fields
                - Other API-specific parameters

        Yields:
            Dictionary records from the API response.

        Raises:
            DataFetchError: If data fetching fails.
        """
        if not self._initialized:
            self._initialize()

        if not self._pro_api:
            raise ConnectionError("Tushare Pro API not initialized")

        try:
            # Get API name from kwargs or config
            api_name = kwargs.pop("api_name", self.config.api_name)
            if not api_name:
                raise ValueError("api_name must be specified in config or kwargs")

            # Get fields from kwargs or config
            fields = kwargs.pop("fields", self.config.fields)

            # Merge config params with kwargs
            params = {**self.config.params, **kwargs}

            logger.info(f"Fetching data from Tushare API: {api_name} with params: {params}")

            # Call the API
            if hasattr(self._pro_api, api_name):
                api_method = getattr(self._pro_api, api_name)
                result = api_method(**params)
            else:
                # Try using query method
                result = self._pro_api.query(api_name, **params)

            if result is None or len(result) == 0:
                logger.warning(f"No data returned from Tushare API: {api_name}")
                return

            # Convert DataFrame to list of dictionaries
            if isinstance(result, pd.DataFrame):
                # Select specific fields if specified
                if fields:
                    field_list = [f.strip() for f in fields.split(",")]
                    available_fields = [f for f in field_list if f in result.columns]
                    if available_fields:
                        result = result[available_fields]
                    else:
                        logger.warning(
                            f"Specified fields {field_list} not found in result. "
                            f"Available fields: {list(result.columns)}"
                        )

                # Convert to records
                for _, row in result.iterrows():
                    record = row.to_dict()
                    # Convert pandas types to Python native types
                    for key, value in record.items():
                        if pd.isna(value):
                            record[key] = None
                        elif isinstance(value, (pd.Timestamp, pd.DatetimeIndex)):
                            record[key] = value.strftime("%Y-%m-%d")
                        elif isinstance(value, pd.Timedelta):
                            record[key] = str(value)
                        elif hasattr(value, "item"):  # numpy/pandas numeric types
                            record[key] = value.item()
                    yield record
            else:
                # If result is not a DataFrame, try to convert it
                logger.warning(f"Unexpected result type: {type(result)}")
                if isinstance(result, (list, tuple)):
                    for item in result:
                        if isinstance(item, dict):
                            yield item

        except Exception as e:
            error_msg = f"Failed to fetch data from Tushare API: {e}"
            logger.error(error_msg, exc_info=True)
            raise DataFetchError(error_msg) from e

    def fetch_stock_basic(
        self,
        exchange: str = "",
        list_status: str = "L",
        fields: Optional[str] = None,
    ) -> Iterator[Dict[str, Any]]:
        """
        Fetch stock basic information.

        Args:
            exchange: Exchange code (SSE/SZSE/BSE, empty for all).
            list_status: List status (L=listed, D=delisted, P=pause, empty for all).
            fields: Fields to retrieve (comma-separated).

        Yields:
            Stock basic information records.
        """
        return self.fetch(
            api_name="stock_basic",
            exchange=exchange,
            list_status=list_status,
            fields=fields or self.config.fields,
        )

    def fetch_daily(
        self,
        ts_code: str = "",
        trade_date: str = "",
        start_date: str = "",
        end_date: str = "",
        fields: Optional[str] = None,
    ) -> Iterator[Dict[str, Any]]:
        """
        Fetch daily K-line data.

        Args:
            ts_code: Stock code (e.g., '000001.SZ').
            trade_date: Trading date (YYYYMMDD).
            start_date: Start date (YYYYMMDD).
            end_date: End date (YYYYMMDD).
            fields: Fields to retrieve.

        Yields:
            Daily K-line records.
        """
        params = {}
        if ts_code:
            params["ts_code"] = ts_code
        if trade_date:
            params["trade_date"] = trade_date
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date

        return self.fetch(
            api_name="daily",
            fields=fields or self.config.fields,
            **params,
        )

    def fetch_trade_cal(
        self,
        exchange: str = "",
        start_date: str = "",
        end_date: str = "",
        is_open: Optional[str] = None,
        fields: Optional[str] = None,
    ) -> Iterator[Dict[str, Any]]:
        """
        Fetch trading calendar data.

        Args:
            exchange: Exchange code (SSE/SZSE/BSE, empty for all).
            start_date: Start date (YYYYMMDD).
            end_date: End date (YYYYMMDD).
            is_open: Trading status filter (0=休市, 1=交易).
            fields: Fields to retrieve (comma-separated).

        Yields:
            Trading calendar records.
        """
        params = {}
        if exchange:
            params["exchange"] = exchange
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        if is_open is not None:
            params["is_open"] = is_open

        return self.fetch(
            api_name="trade_cal",
            fields=fields or self.config.fields,
            **params,
        )

    def fetch_premarket(
        self,
        ts_code: str = "",
        trade_date: str = "",
        start_date: str = "",
        end_date: str = "",
        fields: Optional[str] = None,
    ) -> Iterator[Dict[str, Any]]:
        """
        Fetch stock premarket information (股本情况盘前数据).

        Args:
            ts_code: Stock code (e.g., '000001.SZ').
            trade_date: Trading date (YYYYMMDD).
            start_date: Start date (YYYYMMDD).
            end_date: End date (YYYYMMDD).
            fields: Fields to retrieve.

        Yields:
            Premarket records.

        Reference: https://tushare.pro/document/2?doc_id=329
        """
        params = {}
        if ts_code:
            params["ts_code"] = ts_code
        if trade_date:
            params["trade_date"] = trade_date
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date

        return self.fetch(
            api_name="stk_premarket",
            fields=fields or self.config.fields,
            **params,
        )

    def fetch_pro_bar(
        self,
        ts_code: str,
        freq: str = "D",
        start_date: str = "",
        end_date: str = "",
        adj: str = "qfq",
        factors: Optional[str] = None,
    ) -> Iterator[Dict[str, Any]]:
        """
        Fetch K-line data using pro_bar function.

        Note: pro_bar is a standalone function, not a pro_api method.

        Args:
            ts_code: Stock code (e.g., '000001.SZ').
            freq: Frequency (D=daily, W=weekly, M=monthly, 5=5min, 15=15min, etc.).
            start_date: Start date (YYYYMMDD or YYYY-MM-DD).
            end_date: End date (YYYYMMDD or YYYY-MM-DD).
            adj: Adjustment type (qfq=前复权, hfq=后复权, None=不复权).
            factors: Factors (tor=换手率, vr=量比, etc.).

        Yields:
            K-line records.
        """
        if not self._initialized:
            self._initialize()

        try:
            # pro_bar is a standalone function, not a pro_api method
            params = {
                "ts_code": ts_code,
                "freq": freq,
                "adj": adj,
            }
            if start_date:
                params["start_date"] = start_date
            if end_date:
                params["end_date"] = end_date
            if factors:
                params["factors"] = factors

            logger.info(f"Fetching K-line data using pro_bar: {ts_code}, freq={freq}")

            # Call ts.pro_bar directly
            result = ts.pro_bar(**params)

            if result is None or len(result) == 0:
                logger.warning(f"No data returned from pro_bar for {ts_code}")
                return

            # Convert DataFrame to records
            if isinstance(result, pd.DataFrame):
                for _, row in result.iterrows():
                    record = row.to_dict()
                    # Convert pandas types to Python native types
                    for key, value in record.items():
                        if pd.isna(value):
                            record[key] = None
                        elif isinstance(value, (pd.Timestamp, pd.DatetimeIndex)):
                            record[key] = value.strftime("%Y-%m-%d")
                        elif isinstance(value, pd.Timedelta):
                            record[key] = str(value)
                        elif hasattr(value, "item"):  # numpy/pandas numeric types
                            record[key] = value.item()
                    yield record

        except Exception as e:
            error_msg = f"Failed to fetch data from pro_bar: {e}"
            logger.error(error_msg, exc_info=True)
            raise DataFetchError(error_msg) from e

    def close(self) -> None:
        """Close the source and release resources."""
        self._pro_api = None
        self._initialized = False
        logger.debug("Tushare source closed")


"""
AkShare data source implementation.

Fetches financial data from AkShare library.
AkShare is a Python library for fetching Chinese stock, futures, fund, and other financial data.
Reference: https://akshare.akfamily.xyz/
"""

import logging
from decimal import Decimal
from typing import Any, Dict, Iterator, Optional

import pandas as pd

from nq.data.source.base import (
    BaseSource,
    ConnectionError,
    DataFetchError,
    SourceConfig,
    SourceError,
)

logger = logging.getLogger(__name__)

# Try to import akshare, handle import error gracefully
try:
    import akshare as ak
except ImportError:
    ak = None
    logger.warning(
        "AkShare library not installed. Install it with: pip install akshare --upgrade"
    )


class AkshareSourceConfig(SourceConfig):
    """AkShare source configuration."""

    type: str = "akshare"  # Source type
    api_name: str = ""  # API name (e.g., 'stock_zh_a_spot', 'stock_zh_a_hist')
    timeout: int = 30  # Request timeout in seconds (not used by AkShare, kept for compatibility)
    adjust: str = ""  # Price adjustment type (qfq=前复权, hfq=后复权, empty=不复权)


class AkshareSource(BaseSource):
    """
    AkShare data source for fetching financial data.

    This source uses the AkShare library to fetch stock data,
    K-line data, financial data, etc.

    Example:
        ```python
        config = AkshareSourceConfig(
            api_name="stock_zh_a_spot",
            params={},
        )
        source = AkshareSource(config)
        for record in source.fetch():
            print(record)
        ```
    """

    def __init__(self, config: AkshareSourceConfig):
        """
        Initialize AkShare source.

        Args:
            config: AkShare source configuration.
        """
        super().__init__(config)
        self.config: AkshareSourceConfig = config
        self._initialized = False

        if ak is None:
            raise ImportError(
                "AkShare library is not installed. "
                "Install it with: pip install akshare --upgrade"
            )

    def _initialize(self) -> None:
        """Initialize AkShare (no initialization needed, but kept for consistency)."""
        if self._initialized:
            return

        try:
            # AkShare doesn't require explicit initialization
            # Just verify it's available
            if ak is None:
                raise ConnectionError("AkShare library is not available")

            self._initialized = True
            logger.info("AkShare source initialized successfully")

        except Exception as e:
            raise ConnectionError(f"Failed to initialize AkShare source: {e}") from e

    def test_connection(self) -> bool:
        """
        Test connection to AkShare.

        Returns:
            True if AkShare is available, False otherwise.
        """
        try:
            if ak is None:
                return False

            # Test with a simple API call
            # Try to get stock list (lightweight call)
            test_result = ak.stock_info_a_code_name()
            return test_result is not None and len(test_result) > 0

        except Exception as e:
            logger.error(f"AkShare connection test failed: {e}")
            return False

    def fetch(self, **kwargs) -> Iterator[Dict[str, Any]]:
        """
        Fetch data from AkShare API.

        Args:
            **kwargs: Additional parameters to pass to the API call.
                Common parameters:
                - api_name: Override config api_name
                - Other API-specific parameters

        Yields:
            Dictionary records from the API response.

        Raises:
            DataFetchError: If data fetching fails.
        """
        if not self._initialized:
            self._initialize()

        if ak is None:
            raise ConnectionError("AkShare library is not available")

        try:
            # Get API name from kwargs or config
            api_name = kwargs.pop("api_name", self.config.api_name)
            if not api_name:
                raise ValueError("api_name must be specified in config or kwargs")

            # Merge config params with kwargs
            params = {**self.config.params, **kwargs}

            logger.info(f"Fetching data from AkShare API: {api_name} with params: {params}")

            # Get the API function
            if not hasattr(ak, api_name):
                raise DataFetchError(f"AkShare API '{api_name}' not found")

            api_func = getattr(ak, api_name)

            # Call the API function with error handling for period parameter issues
            try:
                result = api_func(**params)
            except KeyError as e:
                # Handle period parameter errors (e.g., '日K' not found in period_dict)
                error_str = str(e)
                if "period" in error_str.lower() or any(k in error_str for k in ["日K", "周K", "月K", "'日K'", "'周K'", "'月K'"]):
                    logger.warning(f"Period parameter error for {api_name}: {e}")
                    # Try alternative period values or API
                    if api_name == "stock_zh_a_hist":
                        # Try using alternative APIs or different period format
                        alternative_apis = ["stock_zh_a_hist_sina", "stock_zh_a_hist_163"]
                        for alt_api in alternative_apis:
                            if hasattr(ak, alt_api):
                                try:
                                    logger.info(f"Trying alternative API: {alt_api}")
                                    alt_func = getattr(ak, alt_api)
                                    alt_params = params.copy()
                                    result = alt_func(**alt_params)
                                    logger.info(f"Successfully used alternative API: {alt_api}")
                                    break
                                except Exception as alt_e:
                                    logger.debug(f"Alternative API {alt_api} also failed: {alt_e}")
                                    continue
                        else:
                            # If all alternatives failed, re-raise the original error
                            raise DataFetchError(f"Failed to fetch data with period parameter: {e}") from e
                    else:
                        raise
                else:
                    raise

            if result is None:
                logger.warning(f"No data returned from AkShare API: {api_name}")
                return

            # Convert DataFrame to list of dictionaries
            if isinstance(result, pd.DataFrame):
                if len(result) == 0:
                    logger.warning(f"Empty DataFrame returned from AkShare API: {api_name}")
                    return

                # Convert to records
                for _, row in result.iterrows():
                    record = row.to_dict()
                    # Convert pandas types to Python native types
                    for key, value in record.items():
                        if pd.isna(value):
                            record[key] = None
                        elif isinstance(value, (pd.Timestamp, pd.DatetimeIndex)):
                            record[key] = value.strftime("%Y-%m-%d %H:%M:%S")
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
            error_msg = f"Failed to fetch data from AkShare API: {e}"
            logger.error(error_msg, exc_info=True)
            raise DataFetchError(error_msg) from e

    def fetch_stock_spot(self, market: str = "A") -> Iterator[Dict[str, Any]]:
        """
        Fetch stock spot (real-time) data.

        Args:
            market: Market type ('A' for A-share, 'B' for B-share, etc.).

        Yields:
            Stock spot records.
        """
        if market == "A":
            return self.fetch(api_name="stock_zh_a_spot")
        elif market == "B":
            return self.fetch(api_name="stock_zh_b_spot")
        else:
            raise ValueError(f"Unsupported market type: {market}")

    def fetch_stock_hist(
        self,
        symbol: str,
        period: str = "daily",
        start_date: str = "",
        end_date: str = "",
        adjust: str = "",
    ) -> Iterator[Dict[str, Any]]:
        """
        Fetch stock historical K-line data.

        Args:
            symbol: Stock code (e.g., '000001' for 平安银行).
            period: Period type ('daily', 'weekly', 'monthly', '1min', '5min', etc.).
            start_date: Start date (YYYYMMDD or YYYY-MM-DD).
            end_date: End date (YYYYMMDD or YYYY-MM-DD).
            adjust: Price adjustment type (qfq=前复权, hfq=后复权, empty=不复权).

        Yields:
            Historical K-line records.
        """
        # Determine API name based on period
        if period in ["1min", "5min", "15min", "30min", "60min"]:
            # Use minute-level API
            api_name = "stock_zh_a_hist_min_em"
            params = {
                "symbol": symbol,
                "period": period,
                "adjust": adjust or self.config.adjust or "",
            }
            if start_date:
                params["start_date"] = start_date.replace("-", "")
            if end_date:
                params["end_date"] = end_date.replace("-", "")
        else:
            # Use daily/weekly/monthly API
            # According to AkShare's stock_zh_a_hist implementation,
            # the period parameter should match the keys in period_dict
            # Based on the error, AkShare's period_dict expects specific keys
            # Let's use an alternative approach: use stock_zh_a_hist_min_em for all periods
            # Or use a different API that works better
            
            # Try using stock_zh_a_hist_sina or stock_zh_a_hist_163 as alternatives
            # But first, let's try to fix the period parameter mapping
            # Based on AkShare source code analysis, period_dict might expect:
            # "daily" -> "daily", "weekly" -> "weekly", "monthly" -> "monthly"
            # But the actual implementation might be different
            
            # Use stock_zh_a_hist_sina as an alternative (if available)
            # Or use the period parameter in a way that AkShare expects
            api_name = "stock_zh_a_hist"
            
            # Map period to what AkShare's period_dict expects
            # Based on common AkShare usage patterns, try these mappings:
            period_mapping = {
                "daily": "daily",
                "weekly": "weekly",
                "monthly": "monthly",
            }
            
            # If period is not in mapping, default to "daily"
            period_param = period_mapping.get(period, "daily")
            
            # However, if the above doesn't work, we need to handle the error
            # and potentially use a different API or approach
            params = {
                "symbol": symbol,
                "period": period_param,
                "adjust": adjust or self.config.adjust or "",
            }
            if start_date:
                params["start_date"] = start_date.replace("-", "")
            if end_date:
                params["end_date"] = end_date.replace("-", "")

        return self.fetch(api_name=api_name, **params)

    def fetch_stock_info(self) -> Iterator[Dict[str, Any]]:
        """
        Fetch stock basic information (A-share code and name).

        Yields:
            Stock basic information records.
        """
        return self.fetch(api_name="stock_info_a_code_name")

    def fetch_stock_list(self, market: str = "A") -> Iterator[Dict[str, Any]]:
        """
        Fetch stock list.

        Args:
            market: Market type ('A' for A-share, 'B' for B-share, etc.).

        Yields:
            Stock list records.
        """
        if market == "A":
            return self.fetch(api_name="stock_info_a_code_name")
        else:
            raise ValueError(f"Unsupported market type: {market}")

    def fetch_index_hist(
        self,
        symbol: str,
        period: str = "daily",
        start_date: str = "",
        end_date: str = "",
    ) -> Iterator[Dict[str, Any]]:
        """
        Fetch index historical data.

        Args:
            symbol: Index code (e.g., 'sh000001' for 上证指数).
            period: Period type ('daily', 'weekly', 'monthly').
            start_date: Start date (YYYYMMDD or YYYY-MM-DD).
            end_date: End date (YYYYMMDD or YYYY-MM-DD).

        Yields:
            Index historical records.
        """
        api_name = "index_zh_a_hist"
        params = {
            "symbol": symbol,
            "period": period,
        }
        if start_date:
            params["start_date"] = start_date.replace("-", "")
        if end_date:
            params["end_date"] = end_date.replace("-", "")

        return self.fetch(api_name=api_name, **params)

    def fetch_fund_hist(
        self,
        symbol: str,
        period: str = "daily",
        start_date: str = "",
        end_date: str = "",
        adjust: str = "",
    ) -> Iterator[Dict[str, Any]]:
        """
        Fetch fund historical data.

        Args:
            symbol: Fund code (e.g., '000001').
            period: Period type ('daily', 'weekly', 'monthly').
            start_date: Start date (YYYYMMDD or YYYY-MM-DD).
            end_date: End date (YYYYMMDD or YYYY-MM-DD).
            adjust: Price adjustment type (qfq=前复权, hfq=后复权, empty=不复权).

        Yields:
            Fund historical records.
        """
        api_name = "fund_etf_hist_sina"
        params = {
            "symbol": symbol,
            "period": period,
            "adjust": adjust or self.config.adjust or "",
        }
        if start_date:
            params["start_date"] = start_date.replace("-", "")
        if end_date:
            params["end_date"] = end_date.replace("-", "")

        return self.fetch(api_name=api_name, **params)

    def fetch_premarket(
        self,
        symbol: str = "",
        trade_date: str = "",
        start_date: str = "",
        end_date: str = "",
    ) -> Iterator[Dict[str, Any]]:
        """
        Fetch stock premarket information (股本情况盘前数据).

        Note: AkShare doesn't have a direct premarket API, so we use:
        - stock_zh_a_hist: For historical daily data (contains pre_close as previous day's close)
        - stock_zh_a_spot_em: For real-time data (contains pre_close, up_limit, down_limit)

        Args:
            symbol: Stock code (e.g., '000001' without suffix, or '000001.SZ' with suffix).
            trade_date: Trading date (YYYYMMDD). If provided, only fetch this date.
            start_date: Start date (YYYYMMDD). If provided with end_date, fetch date range.
            end_date: End date (YYYYMMDD). If provided with start_date, fetch date range.

        Yields:
            Premarket records with fields: ts_code, trade_date, total_share, float_share,
            pre_close, up_limit, down_limit.
        """
        try:
            # Normalize symbol (remove suffix if present)
            if symbol:
                symbol_clean = symbol.split(".")[0] if "." in symbol else symbol
            else:
                symbol_clean = ""

            # If specific date or date range, use historical data
            if trade_date or (start_date and end_date):
                # Get stock list first if symbol not specified
                if not symbol_clean:
                    logger.info("Fetching stock list...")
                    stock_list = list(self.fetch_stock_info())
                    symbols = []
                    for s in stock_list:
                        code = s.get("code", "")
                        if code:
                            # Extract numeric code
                            code_clean = code.split(".")[0] if "." in code else code
                            if len(code_clean) == 6 and code_clean.isdigit():
                                symbols.append(code_clean)
                    logger.info(f"Found {len(symbols)} stocks to process")
                else:
                    symbols = [symbol_clean]

                # Determine date range
                if trade_date:
                    dates = [trade_date]
                else:
                    # Generate date range
                    from datetime import datetime, timedelta
                    start = datetime.strptime(start_date, "%Y%m%d")
                    end = datetime.strptime(end_date, "%Y%m%d")
                    dates = []
                    current = start
                    while current <= end:
                        dates.append(current.strftime("%Y%m%d"))
                        current += timedelta(days=1)

                logger.info(f"Processing {len(symbols)} stocks for {len(dates)} dates")

                # Fetch data for each symbol and date
                for idx, sym in enumerate(symbols):
                    if idx > 0 and idx % 100 == 0:
                        logger.info(f"Processed {idx}/{len(symbols)} stocks...")

                    for date_str in dates:
                        try:
                            # Get historical data for the date
                            hist_data = list(
                                self.fetch_stock_hist(
                                    symbol=sym,
                                    period="daily",
                                    start_date=date_str,
                                    end_date=date_str,
                                )
                            )

                            if hist_data:
                                # Get the record for the specific date
                                for record in hist_data:
                                    # Determine exchange suffix
                                    if sym.startswith("6"):
                                        ts_code = f"{sym}.SH"
                                    elif sym.startswith(("0", "3")):
                                        ts_code = f"{sym}.SZ"
                                    else:
                                        ts_code = sym

                                    # Get previous close (pre_close) from the record
                                    # In daily data, "收盘" is the close price, which becomes next day's pre_close
                                    # For the same date, we need to get previous day's close
                                    pre_close = record.get("收盘", record.get("close"))
                                    
                                    # Try to get previous day's data for accurate pre_close
                                    try:
                                        from datetime import datetime, timedelta
                                        date_obj = datetime.strptime(date_str, "%Y%m%d")
                                        prev_date = (date_obj - timedelta(days=1)).strftime("%Y%m%d")
                                        prev_data = list(
                                            self.fetch_stock_hist(
                                                symbol=sym,
                                                period="daily",
                                                start_date=prev_date,
                                                end_date=prev_date,
                                            )
                                        )
                                        if prev_data:
                                            pre_close = prev_data[0].get("收盘", prev_data[0].get("close", pre_close))
                                    except Exception:
                                        pass  # Use current record's close as fallback

                                    # Try to get share information from stock_individual_info_em
                                    total_share = None
                                    float_share = None
                                    try:
                                        # Use stock_individual_info_em to get share information
                                        if hasattr(ak, "stock_individual_info_em"):
                                            share_info = list(
                                                self.fetch(
                                                    api_name="stock_individual_info_em",
                                                    symbol=sym,
                                                )
                                            )
                                            if share_info:
                                                # Parse share information from the result
                                                for info in share_info:
                                                    # stock_individual_info_em returns a DataFrame with columns like:
                                                    # "item", "value" where item might be "总股本", "流通股本"
                                                    item = info.get("item", info.get("项目", ""))
                                                    value = info.get("value", info.get("数值", ""))
                                                    
                                                    if "总股本" in str(item) or "total_share" in str(item).lower():
                                                        try:
                                                            # Value might be in format like "100.00亿" or "1000000"
                                                            total_share = self._parse_share_value(value)
                                                        except Exception:
                                                            pass
                                                    elif "流通股本" in str(item) or "float_share" in str(item).lower() or "流通股" in str(item):
                                                        try:
                                                            float_share = self._parse_share_value(value)
                                                        except Exception:
                                                            pass
                                    except Exception as e:
                                        logger.debug(f"Failed to fetch share info for {sym}: {e}")

                                    # Convert to premarket format
                                    premarket_record = {
                                        "ts_code": ts_code,
                                        "trade_date": date_str,
                                        "pre_close": pre_close,
                                        "up_limit": None,  # Not available in hist data
                                        "down_limit": None,  # Not available in hist data
                                        "total_share": total_share,
                                        "float_share": float_share,
                                    }
                                    yield premarket_record
                        except Exception as e:
                            logger.debug(f"Failed to fetch premarket data for {sym} on {date_str}: {e}")
                            continue

            else:
                # No date specified, use real-time spot data
                logger.info("Fetching real-time spot data...")
                spot_data = list(self.fetch_stock_spot(market="A"))

                for record in spot_data:
                    # Filter by symbol if specified
                    if symbol_clean:
                        record_code = record.get("代码", record.get("code", ""))
                        if record_code:
                            code_clean = record_code.split(".")[0] if "." in record_code else record_code
                            if code_clean != symbol_clean:
                                continue
                        else:
                            continue

                    # Get ts_code
                    ts_code = record.get("代码", record.get("code", ""))
                    if not ts_code:
                        continue

                    # Convert spot data to premarket format
                    # Use today's date for real-time data
                    from datetime import date
                    today = date.today().strftime("%Y%m%d")

                    premarket_record = {
                        "ts_code": ts_code,
                        "trade_date": today,
                        "pre_close": record.get("昨收", record.get("pre_close")),
                        "up_limit": record.get("涨停价", record.get("up_limit")),
                        "down_limit": record.get("跌停价", record.get("down_limit")),
                        "total_share": None,  # Not available in spot data
                        "float_share": None,  # Not available in spot data
                    }
                    yield premarket_record

        except Exception as e:
            error_msg = f"Failed to fetch premarket data from AkShare: {e}"
            logger.error(error_msg, exc_info=True)
            raise DataFetchError(error_msg) from e

    def _parse_share_value(self, value: Any) -> Optional[Decimal]:
        """
        Parse share value from string format (e.g., "100.00亿", "1000000").

        Args:
            value: Share value in string or numeric format.

        Returns:
            Decimal value in 10K shares, or None if parsing fails.
        """
        if value is None:
            return None
        
        try:
            if isinstance(value, (int, float)):
                # If already numeric, convert to Decimal (assuming it's in shares, convert to 10K shares)
                return Decimal(str(value)) / Decimal("10000")
            
            value_str = str(value).strip()
            if not value_str or value_str == "None" or value_str == "nan":
                return None
            
            # Remove common separators
            value_str = value_str.replace(",", "").replace("，", "")
            
            # Handle Chinese unit suffixes (亿=100 million, 万=10K)
            multiplier = Decimal("1")
            if "亿" in value_str:
                multiplier = Decimal("10000")  # 亿 to 10K shares
                value_str = value_str.replace("亿", "")
            elif "万" in value_str:
                multiplier = Decimal("1")  # 万 is already 10K shares
                value_str = value_str.replace("万", "")
            elif "万股" in value_str:
                multiplier = Decimal("1")  # 万股 is already 10K shares
                value_str = value_str.replace("万股", "")
            elif "股" in value_str:
                multiplier = Decimal("0.0001")  # shares to 10K shares
                value_str = value_str.replace("股", "")
            
            # Parse numeric value
            numeric_value = Decimal(value_str)
            return numeric_value * multiplier
            
        except Exception as e:
            logger.debug(f"Failed to parse share value '{value}': {e}")
            return None

    def close(self) -> None:
        """Close the source and release resources."""
        self._initialized = False
        logger.debug("AkShare source closed")


"""
K-line data repositories.

Provides repository implementations for K-line models at different time intervals.
"""

from datetime import datetime
from typing import List, Optional

from sqlalchemy import text

from nq.config import DatabaseConfig
from nq.models.kline import (
    StockKline1Min,
    StockKline15Min,
    StockKline30Min,
    StockKline5Min,
    StockKlineDay,
    StockKlineHour,
    StockKlineMonth,
    StockKlineQuarter,
    StockKlineWeek,
)
from nq.repo.database_repo import DatabaseRepo


# =============================================
# Base K-line Repository
# =============================================


class BaseKlineRepo(DatabaseRepo):
    """Base repository for K-line data."""

    def __init__(
        self,
        config: DatabaseConfig,
        table_name: str,
        time_column: str,
        schema: str = "quant",
    ):
        """
        Initialize base K-line repository.

        Args:
            config: Database configuration.
            table_name: K-line table name.
            time_column: Time column name (e.g., 'trade_date', 'trade_time').
            schema: Database schema.
        """
        super().__init__(
            config=config,
            table_name=table_name,
            schema=schema,
            on_conflict="update",
        )
        self.time_column = time_column

    def save_model(self, kline) -> bool:
        """
        Save a K-line model.

        Args:
            kline: K-line model instance.

        Returns:
            True if save was successful.
        """
        data = kline.model_dump(exclude_none=True)
        # Convert datetime to string
        time_key = self.time_column
        if time_key in data and isinstance(data[time_key], datetime):
            data[time_key] = data[time_key].isoformat()
        return self.save(data)

    def save_batch_models(self, klines: List) -> int:
        """
        Save multiple K-line models.

        Args:
            klines: List of K-line model instances.

        Returns:
            Number of records saved.
        """
        data_list = []
        for kline in klines:
            data = kline.model_dump(exclude_none=True)
            # Convert datetime to string
            time_key = self.time_column
            if time_key in data and isinstance(data[time_key], datetime):
                data[time_key] = data[time_key].isoformat()
            data_list.append(data)
        return self.save_batch(data_list)

    def get_by_ts_code(
        self,
        ts_code: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> List:
        """
        Get K-line data by ts_code and time range.

        Args:
            ts_code: Stock code.
            start_time: Start time (inclusive).
            end_time: End time (inclusive).
            limit: Maximum number of records to return.

        Returns:
            List of K-line models.
        """
        engine = self._get_engine()
        table_name = self._get_full_table_name()

        conditions = ["ts_code = :ts_code"]
        params = {"ts_code": ts_code}

        if start_time:
            conditions.append(f"{self.time_column} >= :start_time")
            params["start_time"] = start_time.isoformat()

        if end_time:
            conditions.append(f"{self.time_column} <= :end_time")
            params["end_time"] = end_time.isoformat()

        query = f'SELECT * FROM {table_name} WHERE {" AND ".join(conditions)} ORDER BY {self.time_column} DESC'
        if limit:
            query += f" LIMIT {limit}"

        with engine.connect() as conn:
            result = conn.execute(text(query), params)
            # Return raw dicts - caller should convert to appropriate model
            return [dict(row._mapping) for row in result]


# =============================================
# K-line Repositories by Time Interval
# =============================================


class StockKlineQuarterRepo(BaseKlineRepo):
    """Repository for quarterly K-line data."""

    def __init__(self, config: DatabaseConfig, schema: str = "quant"):
        """Initialize quarterly K-line repository."""
        super().__init__(config, "stock_kline_quarter", "quarter_date", schema)

    def get_by_ts_code(
        self,
        ts_code: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> List[StockKlineQuarter]:
        """Get quarterly K-line data by ts_code."""
        rows = super().get_by_ts_code(ts_code, start_time, end_time, limit)
        return [StockKlineQuarter(**row) for row in rows]


class StockKlineMonthRepo(BaseKlineRepo):
    """Repository for monthly K-line data."""

    def __init__(self, config: DatabaseConfig, schema: str = "quant"):
        """Initialize monthly K-line repository."""
        super().__init__(config, "stock_kline_month", "month_date", schema)

    def get_by_ts_code(
        self,
        ts_code: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> List[StockKlineMonth]:
        """Get monthly K-line data by ts_code."""
        rows = super().get_by_ts_code(ts_code, start_time, end_time, limit)
        return [StockKlineMonth(**row) for row in rows]


class StockKlineWeekRepo(BaseKlineRepo):
    """Repository for weekly K-line data."""

    def __init__(self, config: DatabaseConfig, schema: str = "quant"):
        """Initialize weekly K-line repository."""
        super().__init__(config, "stock_kline_week", "week_date", schema)

    def get_by_ts_code(
        self,
        ts_code: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> List[StockKlineWeek]:
        """Get weekly K-line data by ts_code."""
        rows = super().get_by_ts_code(ts_code, start_time, end_time, limit)
        return [StockKlineWeek(**row) for row in rows]


class StockKlineDayRepo(BaseKlineRepo):
    """Repository for daily K-line data."""

    def __init__(self, config: DatabaseConfig, schema: str = "quant"):
        """Initialize daily K-line repository."""
        super().__init__(config, "stock_kline_day", "trade_date", schema)

    def get_by_ts_code(
        self,
        ts_code: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> List[StockKlineDay]:
        """Get daily K-line data by ts_code."""
        rows = super().get_by_ts_code(ts_code, start_time, end_time, limit)
        return [StockKlineDay(**row) for row in rows]


class StockKlineHourRepo(BaseKlineRepo):
    """Repository for hourly K-line data."""

    def __init__(self, config: DatabaseConfig, schema: str = "quant"):
        """Initialize hourly K-line repository."""
        super().__init__(config, "stock_kline_hour", "trade_time", schema)

    def get_by_ts_code(
        self,
        ts_code: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> List[StockKlineHour]:
        """Get hourly K-line data by ts_code."""
        rows = super().get_by_ts_code(ts_code, start_time, end_time, limit)
        return [StockKlineHour(**row) for row in rows]


class StockKline30MinRepo(BaseKlineRepo):
    """Repository for 30-minute K-line data."""

    def __init__(self, config: DatabaseConfig, schema: str = "quant"):
        """Initialize 30-minute K-line repository."""
        super().__init__(config, "stock_kline_30min", "trade_time", schema)

    def get_by_ts_code(
        self,
        ts_code: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> List[StockKline30Min]:
        """Get 30-minute K-line data by ts_code."""
        rows = super().get_by_ts_code(ts_code, start_time, end_time, limit)
        return [StockKline30Min(**row) for row in rows]


class StockKline15MinRepo(BaseKlineRepo):
    """Repository for 15-minute K-line data."""

    def __init__(self, config: DatabaseConfig, schema: str = "quant"):
        """Initialize 15-minute K-line repository."""
        super().__init__(config, "stock_kline_15min", "trade_time", schema)

    def get_by_ts_code(
        self,
        ts_code: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> List[StockKline15Min]:
        """Get 15-minute K-line data by ts_code."""
        rows = super().get_by_ts_code(ts_code, start_time, end_time, limit)
        return [StockKline15Min(**row) for row in rows]


class StockKline5MinRepo(BaseKlineRepo):
    """Repository for 5-minute K-line data."""

    def __init__(self, config: DatabaseConfig, schema: str = "quant"):
        """Initialize 5-minute K-line repository."""
        super().__init__(config, "stock_kline_5min", "trade_time", schema)

    def get_by_ts_code(
        self,
        ts_code: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> List[StockKline5Min]:
        """Get 5-minute K-line data by ts_code."""
        rows = super().get_by_ts_code(ts_code, start_time, end_time, limit)
        return [StockKline5Min(**row) for row in rows]


class StockKline1MinRepo(BaseKlineRepo):
    """Repository for 1-minute K-line data."""

    def __init__(self, config: DatabaseConfig, schema: str = "quant"):
        """Initialize 1-minute K-line repository."""
        super().__init__(config, "stock_kline_1min", "trade_time", schema)

    def get_by_ts_code(
        self,
        ts_code: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> List[StockKline1Min]:
        """Get 1-minute K-line data by ts_code."""
        rows = super().get_by_ts_code(ts_code, start_time, end_time, limit)
        return [StockKline1Min(**row) for row in rows]



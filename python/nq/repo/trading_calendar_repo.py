"""
Trading calendar repository.

Provides repository implementation for trading calendar model.
"""

from datetime import date, datetime
from typing import List, Optional

from sqlalchemy import text

from nq.config import DatabaseConfig
from nq.models.trading_calendar import TradingCalendar
from nq.repo.base import RepoError
from nq.repo.database_repo import DatabaseRepo


class TradingCalendarRepo(DatabaseRepo):
    """Repository for trading calendar information."""

    def __init__(self, config: DatabaseConfig, schema: str = "quant"):
        """Initialize trading calendar repository."""
        super().__init__(
            config=config,
            table_name="trading_calendar",
            schema=schema,
            on_conflict="update",
        )

    def save_model(self, calendar: TradingCalendar) -> bool:
        """
        Save a TradingCalendar model.

        Args:
            calendar: TradingCalendar model instance.

        Returns:
            True if save was successful.
        """
        data = calendar.model_dump(exclude_none=True)
        # Convert date to string for SQL
        if "cal_date" in data and isinstance(data["cal_date"], date):
            data["cal_date"] = data["cal_date"].isoformat()
        if "pretrade_date" in data and isinstance(data["pretrade_date"], date):
            data["pretrade_date"] = data["pretrade_date"].isoformat()
        if "created_at" in data and isinstance(data["created_at"], datetime):
            data["created_at"] = data["created_at"].isoformat()
        if "updated_at" in data and isinstance(data["updated_at"], datetime):
            data["updated_at"] = data["updated_at"].isoformat()

        return self.save(data)

    def save_batch_models(self, calendars: List[TradingCalendar]) -> int:
        """
        Save multiple TradingCalendar models.

        Args:
            calendars: List of TradingCalendar model instances.

        Returns:
            Number of records saved.
        """
        data_list = []
        for calendar in calendars:
            data = calendar.model_dump(exclude_none=True)
            # Convert date/datetime to string
            if "cal_date" in data and isinstance(data["cal_date"], date):
                data["cal_date"] = data["cal_date"].isoformat()
            if "pretrade_date" in data:
                if isinstance(data["pretrade_date"], date):
                    data["pretrade_date"] = data["pretrade_date"].isoformat()
                elif data["pretrade_date"] is None:
                    # Remove None values to avoid SQL parameter binding issues
                    data.pop("pretrade_date", None)
            if "created_at" in data and isinstance(data["created_at"], datetime):
                data["created_at"] = data["created_at"].isoformat()
            if "updated_at" in data and isinstance(data["updated_at"], datetime):
                data["updated_at"] = data["updated_at"].isoformat()
            data_list.append(data)

        return self.save_batch(data_list)

    def get_by_exchange(self, exchange: str) -> List[TradingCalendar]:
        """
        Get trading calendar by exchange.

        Args:
            exchange: Exchange code (SSE/SZSE/BSE).

        Returns:
            List of TradingCalendar models.
        """
        engine = self._get_engine()
        table_name = self._get_full_table_name()

        with engine.connect() as conn:
            result = conn.execute(
                text(f'SELECT * FROM {table_name} WHERE exchange = :exchange ORDER BY cal_date'),
                {"exchange": exchange},
            )
            return [TradingCalendar(**dict(row._mapping)) for row in result]

    def get_by_date_range(
        self,
        exchange: str,
        start_date: date,
        end_date: date,
        is_open: Optional[bool] = None,
    ) -> List[TradingCalendar]:
        """
        Get trading calendar by date range.

        Args:
            exchange: Exchange code (SSE/SZSE/BSE).
            start_date: Start date.
            end_date: End date.
            is_open: Optional filter for trading days (True=交易, False=休市).

        Returns:
            List of TradingCalendar models.
        """
        engine = self._get_engine()
        table_name = self._get_full_table_name()

        conditions = ["exchange = :exchange", "cal_date >= :start_date", "cal_date <= :end_date"]
        params = {
            "exchange": exchange,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
        }

        if is_open is not None:
            conditions.append("is_open = :is_open")
            params["is_open"] = is_open

        where_clause = "WHERE " + " AND ".join(conditions)

        with engine.connect() as conn:
            result = conn.execute(
                text(f'SELECT * FROM {table_name} {where_clause} ORDER BY cal_date'),
                params,
            )
            return [TradingCalendar(**dict(row._mapping)) for row in result]

    def get_trading_days(
        self,
        exchange: str,
        start_date: date,
        end_date: date,
    ) -> List[TradingCalendar]:
        """
        Get trading days (is_open = True) in date range.

        Args:
            exchange: Exchange code (SSE/SZSE/BSE).
            start_date: Start date.
            end_date: End date.

        Returns:
            List of TradingCalendar models for trading days only.
        """
        return self.get_by_date_range(exchange, start_date, end_date, is_open=True)

    def is_trading_day(self, exchange: str, cal_date: date) -> bool:
        """
        Check if a date is a trading day.

        Args:
            exchange: Exchange code (SSE/SZSE/BSE).
            cal_date: Calendar date.

        Returns:
            True if trading day, False otherwise.
        """
        engine = self._get_engine()
        table_name = self._get_full_table_name()

        with engine.connect() as conn:
            result = conn.execute(
                text(
                    f'SELECT is_open FROM {table_name} WHERE exchange = :exchange AND cal_date = :cal_date'
                ),
                {"exchange": exchange, "cal_date": cal_date.isoformat()},
            )
            row = result.fetchone()
            return row[0] if row else False

    def get_previous_trading_day(self, exchange: str, cal_date: date) -> Optional[date]:
        """
        Get previous trading day.

        Args:
            exchange: Exchange code (SSE/SZSE/BSE).
            cal_date: Calendar date.

        Returns:
            Previous trading day if found, None otherwise.
        """
        engine = self._get_engine()
        table_name = self._get_full_table_name()

        with engine.connect() as conn:
            result = conn.execute(
                text(
                    f"""
                    SELECT pretrade_date FROM {table_name} 
                    WHERE exchange = :exchange AND cal_date = :cal_date
                    """
                ),
                {"exchange": exchange, "cal_date": cal_date.isoformat()},
            )
            row = result.fetchone()
            if row and row[0]:
                return row[0]
            return None


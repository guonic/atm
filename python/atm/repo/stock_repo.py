"""
Stock information repositories.

Provides repository implementations for stock-related models.
"""

import logging
from datetime import date, datetime
from typing import List, Optional

from sqlalchemy import text

from atm.config import DatabaseConfig
from atm.models.stock import (
    StockBasic,
    StockClassify,
    StockFinanceBasic,
    StockKlineSyncState,
    StockQuoteSnapshot,
    StockTradeRule,
)
from atm.repo.base import RepoError
from atm.repo.database_repo import DatabaseRepo

logger = logging.getLogger(__name__)


# =============================================
# Stock Basic Repository
# =============================================


class StockBasicRepo(DatabaseRepo):
    """Repository for stock basic information."""

    def __init__(self, config: DatabaseConfig, schema: str = "quant"):
        """Initialize stock basic repository."""
        super().__init__(
            config=config,
            table_name="stock_basic",
            schema=schema,
            on_conflict="update",
        )

    def save_model(self, stock: StockBasic) -> bool:
        """
        Save a StockBasic model.

        Args:
            stock: StockBasic model instance.

        Returns:
            True if save was successful.
        """
        data = stock.model_dump(exclude_none=True, exclude={"id"})
        # Convert date/datetime to string for SQL
        if "list_date" in data and isinstance(data["list_date"], date):
            data["list_date"] = data["list_date"].isoformat()
        if "delist_date" in data and isinstance(data["delist_date"], date):
            data["delist_date"] = data["delist_date"].isoformat()
        if "create_time" in data and isinstance(data["create_time"], datetime):
            data["create_time"] = data["create_time"].isoformat()
        if "update_time" in data and isinstance(data["update_time"], datetime):
            data["update_time"] = data["update_time"].isoformat()

        return self.save(data)

    def save_batch_models(self, stocks: List[StockBasic]) -> int:
        """
        Save multiple StockBasic models.

        Args:
            stocks: List of StockBasic model instances.

        Returns:
            Number of records saved.
        """
        data_list = []
        for stock in stocks:
            data = stock.model_dump(exclude_none=True, exclude={"id"})
            # Convert date/datetime to string
            if "list_date" in data and isinstance(data["list_date"], date):
                data["list_date"] = data["list_date"].isoformat()
            if "delist_date" in data and isinstance(data["delist_date"], date):
                data["delist_date"] = data["delist_date"].isoformat()
            if "create_time" in data and isinstance(data["create_time"], datetime):
                data["create_time"] = data["create_time"].isoformat()
            if "update_time" in data and isinstance(data["update_time"], datetime):
                data["update_time"] = data["update_time"].isoformat()
            data_list.append(data)

        return self.save_batch(data_list)

    def get_by_ts_code(self, ts_code: str) -> Optional[StockBasic]:
        """
        Get stock by ts_code.

        Args:
            ts_code: Stock code.

        Returns:
            StockBasic model if found, None otherwise.
        """
        engine = self._get_engine()
        table_name = self._get_full_table_name()

        with engine.connect() as conn:
            result = conn.execute(
                text(f'SELECT * FROM {table_name} WHERE ts_code = :ts_code'),
                {"ts_code": ts_code},
            )
            row = result.fetchone()
            if row:
                return StockBasic(**dict(row._mapping))
        return None

    def get_by_exchange(
        self, exchange: Optional[str] = None, list_status: Optional[str] = None
    ) -> List[StockBasic]:
        """
        Get stocks by exchange and list status.

        Args:
            exchange: Exchange code (SH/SE/SZ). If None or empty, returns all exchanges.
            list_status: List status (L=listed, D=delisted, P=pause, empty for all).
                Maps to is_listed field: L=True, D=False, P=False.

        Returns:
            List of StockBasic models.
        """
        engine = self._get_engine()
        table_name = self._get_full_table_name()

        conditions = []
        params = {}

        if exchange:
            conditions.append("exchange = :exchange")
            params["exchange"] = exchange

        if list_status:
            if list_status == "L":
                conditions.append("is_listed = :is_listed")
                params["is_listed"] = True
            elif list_status == "D":
                conditions.append("is_listed = :is_listed")
                params["is_listed"] = False
            # For "P" (pause), we might need additional logic if there's a pause field
            # For now, we'll treat it as delisted (is_listed=False)
            elif list_status == "P":
                conditions.append("is_listed = :is_listed")
                params["is_listed"] = False

        query = f'SELECT * FROM {table_name}'
        if conditions:
            query += ' WHERE ' + ' AND '.join(conditions)

        with engine.connect() as conn:
            result = conn.execute(text(query), params)
            return [StockBasic(**dict(row._mapping)) for row in result]

    def get_all(self) -> List[StockBasic]:
        """
        Get all stocks.

        Returns:
            List of all StockBasic models.
        """
        engine = self._get_engine()
        table_name = self._get_full_table_name()

        with engine.connect() as conn:
            result = conn.execute(text(f'SELECT * FROM {table_name}'))
            return [StockBasic(**dict(row._mapping)) for row in result]


# =============================================
# Stock Classify Repository
# =============================================


class StockClassifyRepo(DatabaseRepo):
    """Repository for stock classification."""

    def __init__(self, config: DatabaseConfig, schema: str = "quant"):
        """Initialize stock classify repository."""
        super().__init__(
            config=config,
            table_name="stock_classify",
            schema=schema,
            on_conflict="ignore",  # Use ignore for unique constraint
        )

    def save_model(self, classify: StockClassify) -> bool:
        """Save a StockClassify model."""
        data = classify.model_dump(exclude_none=True, exclude={"id"})
        if "update_time" in data and isinstance(data["update_time"], datetime):
            data["update_time"] = data["update_time"].isoformat()
        return self.save(data)

    def save_batch_models(self, classifies: List[StockClassify]) -> int:
        """Save multiple StockClassify models."""
        data_list = []
        for classify in classifies:
            data = classify.model_dump(exclude_none=True, exclude={"id"})
            if "update_time" in data and isinstance(data["update_time"], datetime):
                data["update_time"] = data["update_time"].isoformat()
            data_list.append(data)
        return self.save_batch(data_list)

    def get_by_ts_code(self, ts_code: str) -> List[StockClassify]:
        """Get classifications by ts_code."""
        engine = self._get_engine()
        table_name = self._get_full_table_name()

        with engine.connect() as conn:
            result = conn.execute(
                text(f'SELECT * FROM {table_name} WHERE ts_code = :ts_code'),
                {"ts_code": ts_code},
            )
            return [StockClassify(**dict(row._mapping)) for row in result]

    def get_by_classify(self, classify_type: str, classify_value: str) -> List[StockClassify]:
        """Get stocks by classification type and value."""
        engine = self._get_engine()
        table_name = self._get_full_table_name()

        with engine.connect() as conn:
            result = conn.execute(
                text(
                    f'SELECT * FROM {table_name} WHERE classify_type = :type AND classify_value = :value'
                ),
                {"type": classify_type, "value": classify_value},
            )
            return [StockClassify(**dict(row._mapping)) for row in result]


# =============================================
# Stock Trade Rule Repository
# =============================================


class StockTradeRuleRepo(DatabaseRepo):
    """Repository for stock trade rules."""

    def __init__(self, config: DatabaseConfig, schema: str = "quant"):
        """Initialize stock trade rule repository."""
        super().__init__(
            config=config,
            table_name="stock_trade_rule",
            schema=schema,
            on_conflict="update",
        )

    def save_model(self, rule: StockTradeRule) -> bool:
        """Save a StockTradeRule model."""
        data = rule.model_dump(exclude_none=True)
        if "suspend_start" in data and isinstance(data["suspend_start"], date):
            data["suspend_start"] = data["suspend_start"].isoformat()
        if "suspend_end" in data and isinstance(data["suspend_end"], date):
            data["suspend_end"] = data["suspend_end"].isoformat()
        if "update_time" in data and isinstance(data["update_time"], datetime):
            data["update_time"] = data["update_time"].isoformat()
        return self.save(data)

    def save_batch_models(self, rules: List[StockTradeRule]) -> int:
        """Save multiple StockTradeRule models."""
        data_list = []
        for rule in rules:
            data = rule.model_dump(exclude_none=True)
            if "suspend_start" in data and isinstance(data["suspend_start"], date):
                data["suspend_start"] = data["suspend_start"].isoformat()
            if "suspend_end" in data and isinstance(data["suspend_end"], date):
                data["suspend_end"] = data["suspend_end"].isoformat()
            if "update_time" in data and isinstance(data["update_time"], datetime):
                data["update_time"] = data["update_time"].isoformat()
            data_list.append(data)
        return self.save_batch(data_list)

    def get_by_ts_code(self, ts_code: str) -> Optional[StockTradeRule]:
        """Get trade rule by ts_code."""
        engine = self._get_engine()
        table_name = self._get_full_table_name()

        with engine.connect() as conn:
            result = conn.execute(
                text(f'SELECT * FROM {table_name} WHERE ts_code = :ts_code'),
                {"ts_code": ts_code},
            )
            row = result.fetchone()
            if row:
                return StockTradeRule(**dict(row._mapping))
        return None


# =============================================
# Stock Finance Basic Repository
# =============================================


class StockFinanceBasicRepo(DatabaseRepo):
    """Repository for stock finance basic information."""

    def __init__(self, config: DatabaseConfig, schema: str = "quant"):
        """Initialize stock finance basic repository."""
        super().__init__(
            config=config,
            table_name="stock_finance_basic",
            schema=schema,
            on_conflict="update",
        )

    def save_model(self, finance: StockFinanceBasic) -> bool:
        """Save a StockFinanceBasic model."""
        data = finance.model_dump(exclude_none=True, exclude={"id"})
        if "report_date" in data and isinstance(data["report_date"], date):
            data["report_date"] = data["report_date"].isoformat()
        if "update_time" in data and isinstance(data["update_time"], datetime):
            data["update_time"] = data["update_time"].isoformat()
        return self.save(data)

    def save_batch_models(self, finances: List[StockFinanceBasic]) -> int:
        """Save multiple StockFinanceBasic models."""
        data_list = []
        for finance in finances:
            data = finance.model_dump(exclude_none=True, exclude={"id"})
            if "report_date" in data and isinstance(data["report_date"], date):
                data["report_date"] = data["report_date"].isoformat()
            if "update_time" in data and isinstance(data["update_time"], datetime):
                data["update_time"] = data["update_time"].isoformat()
            data_list.append(data)
        return self.save_batch(data_list)

    def get_by_ts_code(self, ts_code: str, limit: Optional[int] = None) -> List[StockFinanceBasic]:
        """Get finance data by ts_code, ordered by report_date DESC."""
        engine = self._get_engine()
        table_name = self._get_full_table_name()

        query = f'SELECT * FROM {table_name} WHERE ts_code = :ts_code ORDER BY report_date DESC'
        if limit:
            query += f' LIMIT {limit}'

        with engine.connect() as conn:
            result = conn.execute(text(query), {"ts_code": ts_code})
            return [StockFinanceBasic(**dict(row._mapping)) for row in result]


# =============================================
# Stock Quote Snapshot Repository
# =============================================


class StockQuoteSnapshotRepo(DatabaseRepo):
    """Repository for stock quote snapshot."""

    def __init__(self, config: DatabaseConfig, schema: str = "quant"):
        """Initialize stock quote snapshot repository."""
        super().__init__(
            config=config,
            table_name="stock_quote_snapshot",
            schema=schema,
            on_conflict="update",
        )

    def save_model(self, quote: StockQuoteSnapshot) -> bool:
        """Save a StockQuoteSnapshot model."""
        data = quote.model_dump(exclude_none=True)
        if "update_time" in data and isinstance(data["update_time"], datetime):
            data["update_time"] = data["update_time"].isoformat()
        return self.save(data)

    def save_batch_models(self, quotes: List[StockQuoteSnapshot]) -> int:
        """Save multiple StockQuoteSnapshot models."""
        data_list = []
        for quote in quotes:
            data = quote.model_dump(exclude_none=True)
            if "update_time" in data and isinstance(data["update_time"], datetime):
                data["update_time"] = data["update_time"].isoformat()
            data_list.append(data)
        return self.save_batch(data_list)

    def get_by_ts_code(self, ts_code: str) -> Optional[StockQuoteSnapshot]:
        """Get quote snapshot by ts_code."""
        engine = self._get_engine()
        table_name = self._get_full_table_name()

        with engine.connect() as conn:
            result = conn.execute(
                text(f'SELECT * FROM {table_name} WHERE ts_code = :ts_code'),
                {"ts_code": ts_code},
            )
            row = result.fetchone()
            if row:
                return StockQuoteSnapshot(**dict(row._mapping))
        return None

    def get_top_by_pct_chg(self, limit: int = 10, desc: bool = True) -> List[StockQuoteSnapshot]:
        """Get top stocks by price change percentage."""
        engine = self._get_engine()
        table_name = self._get_full_table_name()

        order = "DESC" if desc else "ASC"
        query = f'SELECT * FROM {table_name} ORDER BY pct_chg {order} LIMIT :limit'

        with engine.connect() as conn:
            result = conn.execute(text(query), {"limit": limit})
            return [StockQuoteSnapshot(**dict(row._mapping)) for row in result]


# =============================================
# Stock K-line Sync State Repository
# =============================================


class StockKlineSyncStateRepo(DatabaseRepo):
    """Repository for stock K-line synchronization state."""

    def __init__(self, config: DatabaseConfig, schema: str = "quant"):
        """Initialize stock K-line sync state repository."""
        super().__init__(
            config=config,
            table_name="stock_kline_sync_state",
            schema=schema,
            on_conflict="update",
        )
        self._ensure_table()

    def _ensure_table(self) -> None:
        """Ensure sync state table exists."""
        engine = self._get_engine()
        table_name = self._get_full_table_name()

        create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            ts_code VARCHAR(20) NOT NULL,
            kline_type VARCHAR(20) NOT NULL,
            last_synced_date DATE,
            last_synced_time TIMESTAMP,
            total_records INTEGER DEFAULT 0,
            update_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (ts_code, kline_type)
        );
        CREATE INDEX IF NOT EXISTS idx_{table_name}_ts_code ON {table_name}(ts_code);
        CREATE INDEX IF NOT EXISTS idx_{table_name}_kline_type ON {table_name}(kline_type);
        """

        try:
            with engine.connect() as conn:
                conn.execute(text(create_table_sql))
                conn.commit()
        except Exception as e:
            logger.warning(f"Failed to create sync state table (may already exist): {e}")

    def save_model(self, state: StockKlineSyncState) -> bool:
        """
        Save or update sync state.

        Args:
            state: StockKlineSyncState model instance.

        Returns:
            True if save was successful.
        """
        data = state.model_dump(exclude_none=True)
        # Convert date/datetime to string
        if "last_synced_date" in data and isinstance(data["last_synced_date"], date):
            data["last_synced_date"] = data["last_synced_date"].isoformat()
        if "last_synced_time" in data and isinstance(data["last_synced_time"], datetime):
            data["last_synced_time"] = data["last_synced_time"].isoformat()
        if "update_time" in data and isinstance(data["update_time"], datetime):
            data["update_time"] = data["update_time"].isoformat()

        return self.save(data)

    def get_by_ts_code_and_type(
        self, ts_code: str, kline_type: str
    ) -> Optional[StockKlineSyncState]:
        """
        Get sync state by ts_code and kline_type.

        Args:
            ts_code: Stock code.
            kline_type: K-line type.

        Returns:
            StockKlineSyncState if found, None otherwise.
        """
        engine = self._get_engine()
        table_name = self._get_full_table_name()

        with engine.connect() as conn:
            result = conn.execute(
                text(
                    f'SELECT * FROM {table_name} WHERE ts_code = :ts_code AND kline_type = :kline_type'
                ),
                {"ts_code": ts_code, "kline_type": kline_type},
            )
            row = result.fetchone()
            if row:
                return StockKlineSyncState(**dict(row._mapping))
        return None



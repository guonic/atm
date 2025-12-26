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
    StockIndustryClassify,
    StockIndustryMember,
    StockKlineSyncState,
    StockPremarket,
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
        
        # Ensure is_listed is always included (even if False)
        # model_dump(exclude_none=True) will exclude False values, so we need to explicitly include it
        if "is_listed" not in data:
            data["is_listed"] = stock.is_listed

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
            
            # Ensure is_listed is always included (even if False)
            # model_dump(exclude_none=True) will exclude False values, so we need to explicitly include it
            if "is_listed" not in data:
                data["is_listed"] = stock.is_listed
            
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
                Note: If all stocks have is_listed=False in database, L filter will return all stocks.

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
                # Check if database has any stocks with is_listed=True
                # If not, return all stocks (assuming data import issue)
                try:
                    check_query = f"SELECT COUNT(*) FROM {table_name} WHERE is_listed = :is_listed"
                    with engine.connect() as check_conn:
                        check_result = check_conn.execute(text(check_query), {"is_listed": True})
                        count = check_result.fetchone()[0]
                    if count == 0:
                        # No stocks with is_listed=True, return all stocks (ignore is_listed filter)
                        logger.warning(
                            "No stocks with is_listed=True found in database. "
                            "Returning all stocks (assuming data import issue)."
                        )
                    else:
                        conditions.append("is_listed = :is_listed")
                        params["is_listed"] = True
                except Exception as e:
                    logger.warning(f"Failed to check is_listed distribution: {e}, applying filter anyway")
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
            stocks = [StockBasic(**dict(row._mapping)) for row in result]
            # Log query details for debugging
            if not stocks:
                logger.debug(
                    f"get_by_exchange returned 0 stocks. "
                    f"Query: {query}, Params: {params}, Conditions: {conditions}"
                )
            return stocks

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
# Stock Premarket Repository
# =============================================


class StockPremarketRepo(DatabaseRepo):
    """Repository for stock premarket information."""

    def __init__(self, config: DatabaseConfig, schema: str = "quant"):
        """Initialize stock premarket repository."""
        super().__init__(
            config=config,
            table_name="stock_premarket",
            schema=schema,
            on_conflict="update",
        )
        self._ensure_table()

    def _ensure_table(self) -> None:
        """Ensure premarket table exists."""
        engine = self._get_engine()
        table_name = self._get_full_table_name()

        create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            trade_date DATE NOT NULL,
            ts_code VARCHAR(20) NOT NULL,
            total_share DECIMAL(20, 4),
            float_share DECIMAL(20, 4),
            pre_close DECIMAL(10, 2),
            up_limit DECIMAL(10, 2),
            down_limit DECIMAL(10, 2),
            update_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (trade_date, ts_code)
        );
        """
        try:
            with engine.begin() as conn:
                conn.execute(text(create_table_sql))
            logger.info(f"Ensured table {table_name} exists.")
        except Exception as e:
            logger.warning(f"Failed to create premarket table {table_name} (may already exist): {e}")
            if not self.table_exists():
                raise RepoError(f"Table {table_name} does not exist and could not be created.") from e

        # Create indexes
        try:
            with engine.begin() as conn:
                conn.execute(text(f'CREATE INDEX IF NOT EXISTS idx_{self.table_name}_ts_code ON {table_name}(ts_code);'))
                conn.execute(text(f'CREATE INDEX IF NOT EXISTS idx_{self.table_name}_trade_date ON {table_name}(trade_date);'))
            logger.info(f"Ensured indexes for table {table_name} exist.")
        except Exception as e:
            logger.warning(f"Failed to create indexes for table {table_name}: {e}")

    def save_model(self, premarket: StockPremarket) -> bool:
        """Save a StockPremarket model."""
        data = premarket.model_dump(exclude_none=True)
        if "trade_date" in data and isinstance(data["trade_date"], date):
            data["trade_date"] = data["trade_date"].isoformat()
        if "update_time" in data and isinstance(data["update_time"], datetime):
            data["update_time"] = data["update_time"].isoformat()
        return self.save(data)

    def save_batch_models(self, premarkets: List[StockPremarket]) -> int:
        """Save multiple StockPremarket models."""
        data_list = []
        for premarket in premarkets:
            data = premarket.model_dump(exclude_none=True)
            if "trade_date" in data and isinstance(data["trade_date"], date):
                data["trade_date"] = data["trade_date"].isoformat()
            if "update_time" in data and isinstance(data["update_time"], datetime):
                data["update_time"] = data["update_time"].isoformat()
            data_list.append(data)
        return self.save_batch(data_list)

    def get_by_ts_code(
        self, ts_code: str, start_date: Optional[date] = None, end_date: Optional[date] = None
    ) -> List[StockPremarket]:
        """Get premarket data by ts_code."""
        engine = self._get_engine()
        table_name = self._get_full_table_name()

        conditions = ["ts_code = :ts_code"]
        params = {"ts_code": ts_code}

        if start_date:
            conditions.append("trade_date >= :start_date")
            params["start_date"] = start_date.isoformat()
        if end_date:
            conditions.append("trade_date <= :end_date")
            params["end_date"] = end_date.isoformat()

        query = f'SELECT * FROM {table_name} WHERE {" AND ".join(conditions)} ORDER BY trade_date DESC'

        with engine.connect() as conn:
            result = conn.execute(text(query), params)
            return [StockPremarket(**dict(row._mapping)) for row in result]

    def get_by_trade_date(self, trade_date: date) -> List[StockPremarket]:
        """Get premarket data by trade_date."""
        engine = self._get_engine()
        table_name = self._get_full_table_name()

        with engine.connect() as conn:
            result = conn.execute(
                text(f'SELECT * FROM {table_name} WHERE trade_date = :trade_date'),
                {"trade_date": trade_date.isoformat()},
            )
            return [StockPremarket(**dict(row._mapping)) for row in result]


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
        
        # Remove schema prefix for table name in index creation (PostgreSQL requires simple name)
        simple_table_name = self.table_name

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
        """

        create_index_sql1 = f"""
        CREATE INDEX IF NOT EXISTS idx_{simple_table_name}_ts_code ON {table_name}(ts_code);
        """

        create_index_sql2 = f"""
        CREATE INDEX IF NOT EXISTS idx_{simple_table_name}_kline_type ON {table_name}(kline_type);
        """

        try:
            # Use begin() for automatic transaction management and commit
            with engine.begin() as conn:
                conn.execute(text(create_table_sql))
                # Create indexes separately to handle errors gracefully
                try:
                    conn.execute(text(create_index_sql1))
                except Exception as e:
                    logger.debug(f"Index idx_{simple_table_name}_ts_code may already exist: {e}")
                try:
                    conn.execute(text(create_index_sql2))
                except Exception as e:
                    logger.debug(f"Index idx_{simple_table_name}_kline_type may already exist: {e}")
            logger.debug(f"Sync state table {table_name} ensured")
        except Exception as e:
            # Log error but don't raise - table might already exist from concurrent creation
            logger.warning(f"Failed to create sync state table (may already exist): {e}")
            # Try to verify table exists
            try:
                with engine.connect() as conn:
                    result = conn.execute(
                        text(f"SELECT COUNT(*) FROM {table_name} LIMIT 1")
                    )
                    result.fetchone()
                    logger.debug(f"Sync state table {table_name} exists")
            except Exception as verify_error:
                logger.error(f"Sync state table {table_name} does not exist and creation failed: {verify_error}")
                raise

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


# =============================================
# Stock Industry Classify Repository
# =============================================


class StockIndustryClassifyRepo(DatabaseRepo):
    """Repository for stock industry classification (申万行业分类)."""

    def __init__(self, config: DatabaseConfig, schema: str = "quant"):
        """Initialize stock industry classify repository."""
        super().__init__(
            config=config,
            table_name="stock_industry_classify",
            schema=schema,
            on_conflict="update",
        )
        self._ensure_table()

    def _ensure_table(self) -> None:
        """Ensure table exists."""
        engine = self._get_engine()
        table_name = self._get_full_table_name()
        simple_table_name = self.table_name

        create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            index_code VARCHAR(20) NOT NULL,
            industry_name VARCHAR(100) NOT NULL,
            parent_code VARCHAR(20) NOT NULL,
            level VARCHAR(10) NOT NULL,
            industry_code VARCHAR(20),
            is_pub VARCHAR(1),
            src VARCHAR(20) NOT NULL,
            update_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (index_code, src)
        );
        """

        create_index_sql1 = f"""
        CREATE INDEX IF NOT EXISTS idx_{simple_table_name}_parent ON {table_name}(parent_code, src);
        """

        create_index_sql2 = f"""
        CREATE INDEX IF NOT EXISTS idx_{simple_table_name}_level ON {table_name}(level, src);
        """

        create_index_sql3 = f"""
        CREATE INDEX IF NOT EXISTS idx_{simple_table_name}_src ON {table_name}(src);
        """

        try:
            with engine.begin() as conn:
                conn.execute(text(create_table_sql))
                conn.execute(text(create_index_sql1))
                conn.execute(text(create_index_sql2))
                conn.execute(text(create_index_sql3))
            logger.info(f"Ensured table {table_name} exists.")
        except Exception as e:
            logger.warning(f"Failed to create industry classify table {table_name} (may already exist): {e}")
            if not self.table_exists():
                raise RepoError(f"Table {table_name} does not exist and could not be created.") from e

    def save_model(self, classify: StockIndustryClassify) -> bool:
        """Save a StockIndustryClassify model."""
        data = classify.model_dump(exclude_none=True)
        if "update_time" in data and isinstance(data["update_time"], datetime):
            data["update_time"] = data["update_time"].isoformat()
        return self.save(data)

    def save_batch_models(self, classifies: List[StockIndustryClassify]) -> int:
        """Save multiple StockIndustryClassify models."""
        data_list = []
        for classify in classifies:
            data = classify.model_dump(exclude_none=True)
            if "update_time" in data and isinstance(data["update_time"], datetime):
                data["update_time"] = data["update_time"].isoformat()
            data_list.append(data)
        return self.save_batch(data_list)

    def get_by_level(self, level: str, src: str = "SW2021") -> List[StockIndustryClassify]:
        """Get classifications by level."""
        engine = self._get_engine()
        table_name = self._get_full_table_name()

        with engine.connect() as conn:
            result = conn.execute(
                text(f"SELECT * FROM {table_name} WHERE level = :level AND src = :src ORDER BY index_code"),
                {"level": level, "src": src},
            )
            return [StockIndustryClassify(**dict(row._mapping)) for row in result]

    def get_by_parent(self, parent_code: str, src: str = "SW2021") -> List[StockIndustryClassify]:
        """Get classifications by parent code."""
        engine = self._get_engine()
        table_name = self._get_full_table_name()

        with engine.connect() as conn:
            result = conn.execute(
                text(f"SELECT * FROM {table_name} WHERE parent_code = :parent_code AND src = :src ORDER BY index_code"),
                {"parent_code": parent_code, "src": src},
            )
            return [StockIndustryClassify(**dict(row._mapping)) for row in result]

    def delete_by_src(self, src: str) -> bool:
        """Delete all classifications for a specific source version."""
        engine = self._get_engine()
        table_name = self._get_full_table_name()

        try:
            with engine.begin() as conn:
                result = conn.execute(
                    text(f"DELETE FROM {table_name} WHERE src = :src"),
                    {"src": src},
                )
                logger.info(f"Deleted {result.rowcount} records for src={src}")
                return True
        except Exception as e:
            logger.error(f"Failed to delete classifications for src={src}: {e}")
            return False


# =============================================
# Stock Industry Member Repository
# =============================================


class StockIndustryMemberRepo(DatabaseRepo):
    """Repository for stock industry member (申万行业成分)."""

    def __init__(self, config: DatabaseConfig, schema: str = "quant"):
        """Initialize stock industry member repository."""
        super().__init__(
            config=config,
            table_name="stock_industry_member",
            schema=schema,
            on_conflict="update",
        )
        self._ensure_table()

    def _ensure_table(self) -> None:
        """Ensure table exists."""
        engine = self._get_engine()
        table_name = self._get_full_table_name()
        simple_table_name = self.table_name

        create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            ts_code VARCHAR(20) NOT NULL,
            l1_code VARCHAR(20) NOT NULL,
            l1_name VARCHAR(100) NOT NULL,
            l2_code VARCHAR(20) NOT NULL,
            l2_name VARCHAR(100) NOT NULL,
            l3_code VARCHAR(20) NOT NULL,
            l3_name VARCHAR(100) NOT NULL,
            stock_name VARCHAR(100),
            in_date DATE NOT NULL,
            out_date DATE,
            is_new VARCHAR(1) DEFAULT 'Y',
            update_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (ts_code, l3_code, in_date)
        );
        """

        create_index_sql1 = f"""
        CREATE INDEX IF NOT EXISTS idx_{simple_table_name}_ts_code ON {table_name}(ts_code);
        """

        create_index_sql2 = f"""
        CREATE INDEX IF NOT EXISTS idx_{simple_table_name}_l1_code ON {table_name}(l1_code);
        """

        create_index_sql3 = f"""
        CREATE INDEX IF NOT EXISTS idx_{simple_table_name}_l2_code ON {table_name}(l2_code);
        """

        create_index_sql4 = f"""
        CREATE INDEX IF NOT EXISTS idx_{simple_table_name}_l3_code ON {table_name}(l3_code);
        """

        create_index_sql5 = f"""
        CREATE INDEX IF NOT EXISTS idx_{simple_table_name}_in_date ON {table_name}(in_date);
        """

        create_index_sql6 = f"""
        CREATE INDEX IF NOT EXISTS idx_{simple_table_name}_out_date ON {table_name}(out_date);
        """

        create_index_sql7 = f"""
        CREATE INDEX IF NOT EXISTS idx_{simple_table_name}_is_new ON {table_name}(is_new);
        """

        create_index_sql8 = f"""
        CREATE INDEX IF NOT EXISTS idx_{simple_table_name}_current ON {table_name}(l3_code, out_date) 
        WHERE out_date IS NULL;
        """

        try:
            with engine.begin() as conn:
                conn.execute(text(create_table_sql))
                conn.execute(text(create_index_sql1))
                conn.execute(text(create_index_sql2))
                conn.execute(text(create_index_sql3))
                conn.execute(text(create_index_sql4))
                conn.execute(text(create_index_sql5))
                conn.execute(text(create_index_sql6))
                conn.execute(text(create_index_sql7))
                # Partial index may fail if not supported, ignore error
                try:
                    conn.execute(text(create_index_sql8))
                except Exception:
                    logger.debug("Partial index not supported, skipping")
            logger.info(f"Ensured table {table_name} exists.")
        except Exception as e:
            logger.warning(f"Failed to create industry member table {table_name} (may already exist): {e}")
            if not self.table_exists():
                raise RepoError(f"Table {table_name} does not exist and could not be created.") from e

    def save_model(self, member: StockIndustryMember) -> bool:
        """Save a StockIndustryMember model."""
        data = member.model_dump(exclude_none=True)
        if "update_time" in data and isinstance(data["update_time"], datetime):
            data["update_time"] = data["update_time"].isoformat()
        if "in_date" in data and isinstance(data["in_date"], date):
            data["in_date"] = data["in_date"].isoformat()
        if "out_date" in data and isinstance(data["out_date"], date):
            data["out_date"] = data["out_date"].isoformat()
        return self.save(data)

    def save_batch_models(self, members: List[StockIndustryMember]) -> int:
        """Save multiple StockIndustryMember models."""
        data_list = []
        for member in members:
            data = member.model_dump(exclude_none=True)
            if "update_time" in data and isinstance(data["update_time"], datetime):
                data["update_time"] = data["update_time"].isoformat()
            if "in_date" in data and isinstance(data["in_date"], date):
                data["in_date"] = data["in_date"].isoformat()
            if "out_date" in data and isinstance(data["out_date"], date):
                data["out_date"] = data["out_date"].isoformat()
            data_list.append(data)
        return self.save_batch(data_list)

    def get_by_ts_code(self, ts_code: str, current_only: bool = False) -> List[StockIndustryMember]:
        """Get industry members by stock code."""
        engine = self._get_engine()
        table_name = self._get_full_table_name()

        if current_only:
            sql = f"""
            SELECT * FROM {table_name} 
            WHERE ts_code = :ts_code AND (out_date IS NULL OR out_date > CURRENT_DATE)
            ORDER BY in_date DESC
            """
        else:
            sql = f"SELECT * FROM {table_name} WHERE ts_code = :ts_code ORDER BY in_date DESC"

        with engine.connect() as conn:
            result = conn.execute(text(sql), {"ts_code": ts_code})
            return [StockIndustryMember(**dict(row._mapping)) for row in result]

    def get_by_l3_code(self, l3_code: str, current_only: bool = False) -> List[StockIndustryMember]:
        """Get industry members by L3 industry code."""
        engine = self._get_engine()
        table_name = self._get_full_table_name()

        if current_only:
            sql = f"""
            SELECT * FROM {table_name} 
            WHERE l3_code = :l3_code AND (out_date IS NULL OR out_date > CURRENT_DATE)
            ORDER BY in_date DESC
            """
        else:
            sql = f"SELECT * FROM {table_name} WHERE l3_code = :l3_code ORDER BY in_date DESC"

        with engine.connect() as conn:
            result = conn.execute(text(sql), {"l3_code": l3_code})
            return [StockIndustryMember(**dict(row._mapping)) for row in result]



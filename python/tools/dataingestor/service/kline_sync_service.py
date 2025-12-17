"""
K-line synchronization service.

Synchronizes K-line data for all stocks, checking existing data and syncing from the appropriate start date.
"""

import logging
import sys
from datetime import date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Dict, List, Optional, Type

# Ensure project path is in sys.path before importing atm modules
_project_root = Path(__file__).parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from atm.config import DatabaseConfig
from atm.data.source import TushareSource, TushareSourceConfig
from atm.models.kline import (
    StockKline15Min,
    StockKline1Min,
    StockKline30Min,
    StockKline5Min,
    StockKlineDay,
    StockKlineHour,
    StockKlineMonth,
    StockKlineQuarter,
    StockKlineWeek,
)
from atm.models.stock import StockBasic
from atm.repo import (
    BaseStateRepo,
    BaseTaskLock,
    DatabaseStateRepo,
    DatabaseTaskLock,
    FileStateRepo,
    FileTaskLock,
    IngestionState,
    StockBasicRepo,
    StockKline15MinRepo,
    StockKline1MinRepo,
    StockKline30MinRepo,
    StockKline5MinRepo,
    StockKlineDayRepo,
    StockKlineHourRepo,
    StockKlineMonthRepo,
    StockKlineQuarterRepo,
    StockKlineWeekRepo,
    TaskLockError,
)

logger = logging.getLogger(__name__)

# Mapping from kline type to model and repo classes
KLINE_TYPE_MAP = {
    "day": {
        "model": StockKlineDay,
        "repo": StockKlineDayRepo,
        "time_column": "trade_date",
        "tushare_api": "daily",
        "tushare_freq": "D",
    },
    "week": {
        "model": StockKlineWeek,
        "repo": StockKlineWeekRepo,
        "time_column": "week_date",
        "tushare_api": "weekly",
        "tushare_freq": "W",
    },
    "month": {
        "model": StockKlineMonth,
        "repo": StockKlineMonthRepo,
        "time_column": "month_date",
        "tushare_api": "monthly",
        "tushare_freq": "M",
    },
    "quarter": {
        "model": StockKlineQuarter,
        "repo": StockKlineQuarterRepo,
        "time_column": "quarter_date",
        "tushare_api": "quarterly",
        "tushare_freq": "Q",
    },
    "hour": {
        "model": StockKlineHour,
        "repo": StockKlineHourRepo,
        "time_column": "trade_time",
        "tushare_api": "daily",
        "tushare_freq": "60",
    },
    "30min": {
        "model": StockKline30Min,
        "repo": StockKline30MinRepo,
        "time_column": "trade_time",
        "tushare_api": "daily",
        "tushare_freq": "30",
    },
    "15min": {
        "model": StockKline15Min,
        "repo": StockKline15MinRepo,
        "time_column": "trade_time",
        "tushare_freq": "15",
    },
    "5min": {
        "model": StockKline5Min,
        "repo": StockKline5MinRepo,
        "time_column": "trade_time",
        "tushare_api": "daily",
        "tushare_freq": "5",
    },
    "1min": {
        "model": StockKline1Min,
        "repo": StockKline1MinRepo,
        "time_column": "trade_time",
        "tushare_api": "daily",
        "tushare_freq": "1",
    },
}


class KlineSyncService:
    """
    Service for synchronizing K-line data for all stocks.

    This service:
    1. Iterates through all stocks in stock_basic table
    2. For each stock, checks existing K-line data
    3. Syncs from the last existing record (or list_date if no data exists)
    4. Syncs to the latest date
    """

    def __init__(
        self,
        db_config: DatabaseConfig,
        tushare_token: str,
        kline_type: str,
        state_repo: Optional[BaseStateRepo] = None,
        state_dir: Optional[str] = None,
        task_lock: Optional[BaseTaskLock] = None,
    ):
        """
        Initialize K-line sync service.

        Args:
            db_config: Database configuration.
            tushare_token: Tushare Pro API token.
            kline_type: K-line type (day, week, month, quarter, hour, 30min, 15min, 5min, 1min).
            state_repo: Optional state repository. If None, will use FileStateRepo.
            state_dir: Directory for file-based state storage (used if state_repo is None).
            task_lock: Optional task lock. If None, will create based on state_repo type.
        """
        if kline_type not in KLINE_TYPE_MAP:
            raise ValueError(f"Invalid kline_type: {kline_type}. Must be one of: {list(KLINE_TYPE_MAP.keys())}")

        self.db_config = db_config
        self.tushare_token = tushare_token
        self.kline_type = kline_type
        self.kline_config = KLINE_TYPE_MAP[kline_type]
        self._source: Optional[TushareSource] = None
        self._stock_repo: Optional[StockBasicRepo] = None
        self._kline_repo = None
        self._state_repo: Optional[BaseStateRepo] = state_repo
        self._state_dir = state_dir or "storage/state"
        self._task_lock: Optional[BaseTaskLock] = task_lock

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self._source:
            self._source.close()
        if self._stock_repo:
            self._stock_repo.close()
        if self._kline_repo:
            self._kline_repo.close()

    @property
    def source(self) -> TushareSource:
        """Get or create Tushare source."""
        if self._source is None:
            config = TushareSourceConfig(
                token=self.tushare_token,
                type="tushare",
            )
            self._source = TushareSource(config)
        return self._source

    @property
    def stock_repo(self) -> StockBasicRepo:
        """Get or create stock basic repository."""
        if self._stock_repo is None:
            self._stock_repo = StockBasicRepo(self.db_config)
        return self._stock_repo

    @property
    def kline_repo(self):
        """Get or create K-line repository."""
        if self._kline_repo is None:
            repo_class = self.kline_config["repo"]
            self._kline_repo = repo_class(self.db_config)
            # Use append mode to avoid overwriting historical data
            self._kline_repo.on_conflict = "ignore"
        return self._kline_repo

    @property
    def state_repo(self) -> BaseStateRepo:
        """Get or create state repository."""
        if self._state_repo is None:
            self._state_repo = FileStateRepo(state_dir=self._state_dir)
        return self._state_repo

    @property
    def task_lock(self) -> BaseTaskLock:
        """Get or create task lock."""
        if self._task_lock is None:
            if isinstance(self._state_repo, DatabaseStateRepo):
                self._task_lock = DatabaseTaskLock(self._state_repo)
            else:
                self._task_lock = FileTaskLock(lock_dir="storage/locks")
        return self._task_lock

    def _get_state(self, task_name: str) -> IngestionState:
        """Get or create ingestion state."""
        state = self.state_repo.get_state(task_name)
        if state is None:
            state = IngestionState(task_name=task_name)
        return state

    def _save_state(self, state: IngestionState) -> None:
        """Save ingestion state."""
        try:
            self.state_repo.save_state(state)
        except Exception as e:
            logger.warning(f"Failed to save state: {e}")

    def _get_last_synced_date(self, ts_code: str) -> Optional[date]:
        """
        Get the last synced date for a stock.

        Args:
            ts_code: Stock code.

        Returns:
            Last synced date if exists, None otherwise.
        """
        try:
            # Get the latest K-line record for this stock
            # Note: get_by_ts_code returns a list of model instances (not dicts)
            existing_data = self.kline_repo.get_by_ts_code(ts_code=ts_code, limit=1)
            if existing_data:
                time_column = self.kline_config["time_column"]
                last_record = existing_data[0]
                
                # Get time value from record (it's a model instance, not a dict)
                last_time = getattr(last_record, time_column, None)

                if last_time:
                    # Parse time value if it's a string
                    if isinstance(last_time, str):
                        try:
                            last_time = datetime.fromisoformat(last_time.replace("Z", "+00:00"))
                        except ValueError:
                            try:
                                last_time = datetime.strptime(last_time, "%Y-%m-%d")
                            except ValueError:
                                logger.warning(f"Failed to parse last_time: {last_time}")
                                return None

                    # For datetime fields, return the date
                    if isinstance(last_time, datetime):
                        return last_time.date()
                    elif isinstance(last_time, date):
                        return last_time
        except Exception as e:
            logger.warning(f"Failed to get last synced date for {ts_code}: {e}")
        return None

    def _convert_to_kline_model(self, record: Dict) -> Optional:
        """
        Convert Tushare record to K-line model.

        Args:
            record: Tushare API record.

        Returns:
            K-line model instance or None if conversion fails.
        """
        try:
            model_class = self.kline_config["model"]
            time_column = self.kline_config["time_column"]

            # Parse time field based on kline type
            if self.kline_type == "day":
                trade_date_str = record.get("trade_date", "")
                if not trade_date_str or trade_date_str == "00000000":
                    return None
                trade_date = datetime.strptime(trade_date_str, "%Y%m%d").date()
                time_value = datetime.combine(trade_date, datetime.min.time())
            elif self.kline_type in ["week", "month", "quarter"]:
                # For weekly/monthly/quarterly, use the first day of the period
                trade_date_str = record.get("trade_date", "")
                if not trade_date_str or trade_date_str == "00000000":
                    return None
                trade_date = datetime.strptime(trade_date_str, "%Y%m%d").date()
                time_value = datetime.combine(trade_date, datetime.min.time())
            else:
                # For intraday data (hour, 30min, etc.), use datetime
                trade_time_str = record.get("trade_time", "")
                if not trade_time_str:
                    return None
                # Try to parse different formats
                try:
                    time_value = datetime.fromisoformat(trade_time_str.replace(" ", "T"))
                except ValueError:
                    # Try YYYYMMDDHHMMSS format
                    try:
                        time_value = datetime.strptime(trade_time_str, "%Y%m%d%H%M%S")
                    except ValueError:
                        logger.warning(f"Failed to parse trade_time: {trade_time_str}")
                        return None

            # Validate required fields
            ts_code = record.get("ts_code", "").strip()
            if not ts_code:
                return None

            # Build model data
            model_data = {
                "ts_code": ts_code,
                time_column: time_value,
            }

            # Add price and volume fields
            if "open" in record:
                model_data["open"] = Decimal(str(record["open"]))
            if "high" in record:
                model_data["high"] = Decimal(str(record["high"]))
            if "low" in record:
                model_data["low"] = Decimal(str(record["low"]))
            if "close" in record:
                model_data["close"] = Decimal(str(record["close"]))
            if "vol" in record:
                model_data["volume"] = int(record["vol"])
            if "amount" in record:
                model_data["amount"] = Decimal(str(record["amount"]))

            return model_class(**model_data)
        except Exception as e:
            logger.error(f"Error converting to {self.kline_type} K-line model: {e}, record: {record}")
            return None

    def sync_all_stocks(
        self,
        exchange: str = "",
        list_status: str = "L",
        batch_size: int = 100,
        task_name: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Dict[str, Dict[str, int]]:
        """
        Sync K-line data for all stocks.

        Args:
            exchange: Exchange code filter (SSE/SZSE/BSE, empty for all).
            list_status: List status filter (L=listed, D=delisted, P=pause, empty for all).
            batch_size: Batch size for saving.
            task_name: Task name for state tracking.
            end_date: End date (YYYYMMDD, defaults to today).

        Returns:
            Dictionary mapping ts_code to statistics.
        """
        if end_date is None:
            end_date = datetime.now().strftime("%Y%m%d")

        task_name = task_name or f"kline_{self.kline_type}_sync_all"
        state = self._get_state(task_name)

        # Acquire task lock
        try:
            if not self.task_lock.acquire(task_name):
                raise TaskLockError(f"Task '{task_name}' is already running or locked.")
            logger.info(f"Task '{task_name}' lock acquired.")
        except TaskLockError as e:
            logger.error(f"Failed to acquire lock for task '{task_name}': {e}")
            raise

        results = {}
        failed_stocks = []

        try:
            state.status = "running"
            self._save_state(state)

            with self.source, self.stock_repo, self.kline_repo:
                # Test connection
                if not self.source.test_connection():
                    raise ConnectionError("Tushare connection test failed")

                # Get all stocks
                logger.info(f"Fetching stocks: exchange={exchange or 'ALL'}, list_status={list_status or 'ALL'}")
                if exchange:
                    stocks = self.stock_repo.get_by_exchange(exchange)
                else:
                    # Get all stocks by fetching from Tushare
                    stocks = []
                    for record in self.source.fetch_stock_basic(exchange="", list_status=list_status):
                        try:
                            stock = StockBasic(
                                ts_code=record.get("ts_code", ""),
                                symbol=record.get("symbol", ""),
                                full_name=record.get("name", ""),
                                exchange=record.get("exchange", ""),
                                market=record.get("market", ""),
                                list_date=datetime.strptime(record["list_date"], "%Y%m%d").date() if record.get("list_date") else None,
                            )
                            stocks.append(stock)
                        except Exception as e:
                            logger.warning(f"Failed to convert stock record: {e}")
                            continue

                total_stocks = len(stocks)
                logger.info(f"Found {total_stocks} stocks to sync")

                for idx, stock in enumerate(stocks, 1):
                    ts_code = stock.ts_code
                    logger.info(f"[{idx}/{total_stocks}] Processing {ts_code} ({stock.full_name})...")

                    try:
                        # Get last synced date
                        last_synced_date = self._get_last_synced_date(ts_code)

                        # Determine start date
                        if last_synced_date:
                            # Start from the day after last synced date
                            start_date = (last_synced_date + timedelta(days=1)).strftime("%Y%m%d")
                            logger.info(f"  Last synced date: {last_synced_date}, starting from: {start_date}")
                        else:
                            # Start from list_date
                            if stock.list_date:
                                start_date = stock.list_date.strftime("%Y%m%d")
                                logger.info(f"  No existing data, starting from list_date: {start_date}")
                            else:
                                logger.warning(f"  Skipping {ts_code}: no list_date")
                                continue

                        # Skip if start_date is after end_date
                        if start_date > end_date:
                            logger.info(f"  Skipping {ts_code}: start_date ({start_date}) > end_date ({end_date})")
                            continue

                        # Sync K-line data for this stock
                        stats = self._sync_stock_kline(
                            ts_code=ts_code,
                            start_date=start_date,
                            end_date=end_date,
                            batch_size=batch_size,
                        )
                        results[ts_code] = stats
                        logger.info(
                            f"  Completed {ts_code}: Fetched={stats['fetched']}, "
                            f"Saved={stats['saved']}, Errors={stats['errors']}"
                        )

                    except Exception as e:
                        logger.error(f"  Failed to sync {ts_code}: {e}", exc_info=True)
                        failed_stocks.append(ts_code)
                        results[ts_code] = {"fetched": 0, "saved": 0, "errors": 1}

                # Final state update
                state.status = "completed"
                state.last_processed_time = datetime.now()
                self._save_state(state)

                logger.info("=" * 80)
                logger.info("K-line Synchronization Completed")
                logger.info("=" * 80)
                logger.info(f"Total stocks processed: {total_stocks}")
                logger.info(f"Successfully synced: {len(results) - len(failed_stocks)}")
                logger.info(f"Failed: {len(failed_stocks)}")
                if failed_stocks:
                    logger.info(f"Failed stocks: {', '.join(failed_stocks)}")
                logger.info("=" * 80)

        except Exception as e:
            state.status = "failed"
            state.error_message = str(e)
            state.last_processed_time = datetime.now()
            self._save_state(state)
            logger.error(f"K-line synchronization failed: {e}", exc_info=True)
            raise
        finally:
            # Release task lock
            self.task_lock.release(task_name)
            logger.info(f"Task '{task_name}' lock released.")

        return results

    def _sync_stock_kline(
        self,
        ts_code: str,
        start_date: str,
        end_date: str,
        batch_size: int = 100,
    ) -> Dict[str, int]:
        """
        Sync K-line data for a single stock.

        Args:
            ts_code: Stock code.
            start_date: Start date (YYYYMMDD).
            end_date: End date (YYYYMMDD).
            batch_size: Batch size for saving.

        Returns:
            Dictionary with ingestion statistics.
        """
        stats = {"fetched": 0, "saved": 0, "errors": 0}
        batch = []

        try:
            # Fetch K-line data from Tushare
            if self.kline_type == "day":
                records = self.source.fetch_daily(
                    ts_code=ts_code,
                    start_date=start_date,
                    end_date=end_date,
                )
            else:
                # For other frequencies, use pro_bar
                # Note: pro_bar requires pro_api to be initialized
                if not self.source._initialized:
                    self.source._initialize()
                records = self.source.fetch_pro_bar(
                    ts_code=ts_code,
                    freq=self.kline_config["tushare_freq"],
                    start_date=start_date,
                    end_date=end_date,
                )

            for record in records:
                stats["fetched"] += 1

                try:
                    # Convert to model
                    kline = self._convert_to_kline_model(record)
                    if kline:
                        batch.append(kline)

                        # Save batch when it reaches batch_size
                        if len(batch) >= batch_size:
                            try:
                                saved = self.kline_repo.save_batch_models(batch)
                                stats["saved"] += saved
                                batch = []
                            except Exception as e:
                                stats["errors"] += len(batch)
                                logger.error(f"Error saving batch for {ts_code}: {e}", exc_info=True)
                                batch = []
                    else:
                        stats["errors"] += 1

                except Exception as e:
                    stats["errors"] += 1
                    logger.error(f"Error converting record for {ts_code}: {e}", exc_info=True)

            # Save remaining records
            if batch:
                try:
                    saved = self.kline_repo.save_batch_models(batch)
                    stats["saved"] += saved
                except Exception as e:
                    stats["errors"] += len(batch)
                    logger.error(f"Error saving final batch for {ts_code}: {e}", exc_info=True)

        except Exception as e:
            logger.error(f"Error syncing K-line for {ts_code}: {e}", exc_info=True)
            stats["errors"] += 1

        return stats


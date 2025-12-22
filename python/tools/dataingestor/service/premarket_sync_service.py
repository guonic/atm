"""
Stock premarket information synchronization service.

Synchronizes stock premarket data (股本情况盘前数据) from Tushare to database.
Reference: https://tushare.pro/document/2?doc_id=329
"""

import logging
from datetime import date, datetime
from decimal import Decimal
from typing import Dict, List, Optional

import sys
from pathlib import Path

# Add parent directory to path
_project_root = Path(__file__).parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from atm.config import DatabaseConfig
from atm.data.source import TushareSource, TushareSourceConfig
from atm.models.stock import StockPremarket
from atm.repo import (
    BaseStateRepo,
    BaseTaskLock,
    DatabaseStateRepo,
    DatabaseTaskLock,
    FileStateRepo,
    FileTaskLock,
    StockPremarketRepo,
    TaskLockError,
)

logger = logging.getLogger(__name__)


class PremarketSyncService:
    """
    Service for synchronizing stock premarket information from Tushare.

    This service fetches premarket data (股本情况盘前数据) including:
    - Total shares and float shares
    - Previous close price
    - Upper and lower limit prices
    """

    def __init__(
        self,
        db_config: DatabaseConfig,
        tushare_token: str,
        state_repo: Optional[BaseStateRepo] = None,
        state_dir: Optional[str] = None,
        task_lock: Optional[BaseTaskLock] = None,
    ):
        """
        Initialize premarket sync service.

        Args:
            db_config: Database configuration.
            tushare_token: Tushare Pro API token.
            state_repo: Optional state repository. If None, will use FileStateRepo.
            state_dir: Directory for file-based state storage (used if state_repo is None).
            task_lock: Optional task lock. If None, will create based on state_repo type.
        """
        self.db_config = db_config
        self.tushare_token = tushare_token
        self._source: Optional[TushareSource] = None
        self._premarket_repo: Optional[StockPremarketRepo] = None
        self._state_repo: Optional[BaseStateRepo] = state_repo
        self._state_dir = state_dir or "storage/state"
        self._task_lock: Optional[BaseTaskLock] = task_lock

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
    def premarket_repo(self) -> StockPremarketRepo:
        """Get or create premarket repository."""
        if self._premarket_repo is None:
            self._premarket_repo = StockPremarketRepo(self.db_config)
        return self._premarket_repo

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
            if isinstance(self.state_repo, DatabaseStateRepo):
                self._task_lock = DatabaseTaskLock(self.state_repo)
            else:
                self._task_lock = FileTaskLock(state_dir=self._state_dir)
        return self._task_lock

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        pass

    def sync_premarket(
        self,
        trade_date: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        ts_code: Optional[str] = None,
        batch_size: int = 100,
        task_name: str = "premarket_sync",
        resume: bool = True,
        mode: str = "upsert",
    ) -> Dict[str, int]:
        """
        Synchronize premarket data from Tushare.

        Args:
            trade_date: Specific trading date (YYYYMMDD). If provided, only sync this date.
            start_date: Start date (YYYYMMDD). If provided with end_date, sync date range.
            end_date: End date (YYYYMMDD). If provided with start_date, sync date range.
            ts_code: Stock code (e.g., '000001.SZ'). If provided, only sync this stock.
            batch_size: Batch size for saving records.
            task_name: Task name for state tracking.
            resume: Whether to resume from last checkpoint.
            mode: Sync mode ('upsert' or 'append').

        Returns:
            Dictionary with synchronization statistics.
        """
        stats = {"fetched": 0, "saved": 0, "errors": 0}

        # Acquire task lock
        try:
            if not self.task_lock.acquire(task_name):
                raise TaskLockError(f"Task {task_name} is already running")
        except TaskLockError as e:
            logger.error(f"Failed to acquire task lock: {e}")
            raise

        try:
            logger.info("=" * 80)
            logger.info("Stock Premarket Information Synchronization")
            logger.info("=" * 80)
            logger.info(f"Task Name: {task_name}")
            logger.info(f"Mode: {mode}")
            logger.info(f"Trade Date: {trade_date or 'N/A'}")
            logger.info(f"Date Range: {start_date or 'N/A'} to {end_date or 'N/A'}")
            logger.info(f"Stock Code: {ts_code or 'All'}")
            logger.info(f"Batch Size: {batch_size}")
            logger.info("=" * 80)

            with self.source, self.premarket_repo:
                # Fetch data from Tushare
                logger.info("Fetching premarket data from Tushare...")
                records = list(
                    self.source.fetch_premarket(
                        ts_code=ts_code or "",
                        trade_date=trade_date or "",
                        start_date=start_date or "",
                        end_date=end_date or "",
                    )
                )

                logger.info(f"Fetched {len(records)} records from Tushare")

                # Convert and save records
                batch = []
                for record in records:
                    stats["fetched"] += 1

                    try:
                        premarket = self._convert_to_premarket_model(record)
                        if premarket:
                            batch.append(premarket)

                            # Save batch when it reaches batch_size
                            if len(batch) >= batch_size:
                                saved = self.premarket_repo.save_batch_models(batch)
                                stats["saved"] += saved
                                logger.debug(f"Saved batch of {saved} records")
                                batch = []

                    except Exception as e:
                        stats["errors"] += 1
                        logger.error(f"Error converting record: {e}", exc_info=True)
                        logger.error(f"Record: {record}")

                # Save remaining records
                if batch:
                    try:
                        saved = self.premarket_repo.save_batch_models(batch)
                        stats["saved"] += saved
                        logger.debug(f"Saved final batch of {saved} records")
                    except Exception as e:
                        stats["errors"] += len(batch)
                        logger.error(f"Error saving final batch: {e}", exc_info=True)

            logger.info("=" * 80)
            logger.info("Synchronization Completed")
            logger.info("=" * 80)
            logger.info(f"Fetched: {stats['fetched']}")
            logger.info(f"Saved: {stats['saved']}")
            logger.info(f"Errors: {stats['errors']}")
            logger.info("=" * 80)

            return stats

        finally:
            # Release task lock
            try:
                self.task_lock.release(task_name)
            except Exception as e:
                logger.warning(f"Failed to release task lock: {e}")

    def _convert_to_premarket_model(self, record: Dict) -> Optional[StockPremarket]:
        """
        Convert Tushare record to StockPremarket model.

        Args:
            record: Raw record from Tushare API.

        Returns:
            StockPremarket model instance, or None if conversion fails.
        """
        try:
            # Parse trade_date
            trade_date_str = record.get("trade_date", "")
            if not trade_date_str:
                logger.warning("Missing trade_date in record")
                return None

            try:
                trade_date = datetime.strptime(trade_date_str, "%Y%m%d").date()
            except ValueError:
                logger.warning(f"Invalid trade_date format: {trade_date_str}")
                return None

            ts_code = record.get("ts_code", "").strip()
            if not ts_code:
                logger.warning("Missing ts_code in record")
                return None

            # Convert numeric fields
            def safe_decimal(value, field_name: str) -> Optional[Decimal]:
                if value is None or (isinstance(value, float) and value != value):  # Check for NaN
                    return None
                try:
                    return Decimal(str(value))
                except (ValueError, TypeError):
                    logger.debug(f"Invalid {field_name} value: {value} for {ts_code}")
                    return None

            total_share = safe_decimal(record.get("total_share"), "total_share")
            float_share = safe_decimal(record.get("float_share"), "float_share")
            pre_close = safe_decimal(record.get("pre_close"), "pre_close")
            up_limit = safe_decimal(record.get("up_limit"), "up_limit")
            down_limit = safe_decimal(record.get("down_limit"), "down_limit")

            return StockPremarket(
                trade_date=trade_date,
                ts_code=ts_code,
                total_share=total_share,
                float_share=float_share,
                pre_close=pre_close,
                up_limit=up_limit,
                down_limit=down_limit,
                update_time=datetime.now(),
            )

        except Exception as e:
            logger.error(f"Error converting to premarket model: {e}", exc_info=True)
            logger.error(f"Record: {record}")
            return None


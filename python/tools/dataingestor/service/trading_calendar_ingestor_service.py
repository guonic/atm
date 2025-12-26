"""
Trading calendar ingestor service.

Specialized service for ingesting trading calendar data from Tushare and storing to database.
"""

import logging
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional

# Ensure project path is in sys.path before importing atm modules
_project_root = Path(__file__).parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from nq.config import DatabaseConfig
from nq.data.source import TushareSource, TushareSourceConfig
from nq.models.trading_calendar import TradingCalendar
from nq.repo import (
    BaseStateRepo,
    BaseTaskLock,
    DatabaseStateRepo,
    DatabaseTaskLock,
    FileStateRepo,
    FileTaskLock,
    IngestionState,
    TaskLockError,
    TradingCalendarRepo,
)

logger = logging.getLogger(__name__)


class TradingCalendarIngestorService:
    """
    Service for ingesting trading calendar data from Tushare.

    This service uses TushareSource to fetch trading calendar data and
    TradingCalendarRepo to store it in the database.
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
        Initialize trading calendar ingestor service.

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
        self._calendar_repo: Optional[TradingCalendarRepo] = None
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
        if self._calendar_repo:
            self._calendar_repo.close()

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

    def ingest_trading_calendar(
        self,
        exchange: str = "",
        start_date: str = "",
        end_date: str = "",
        batch_size: int = 100,
        task_name: Optional[str] = None,
        mode: str = "upsert",
    ) -> Dict[str, int]:
        """
        Ingest trading calendar data with checkpoint/resume support.

        Args:
            exchange: Exchange code (SSE/SZSE/BSE, empty for all).
            start_date: Start date (YYYYMMDD).
            end_date: End date (YYYYMMDD).
            batch_size: Batch size for saving.
            task_name: Task name for state tracking (default: "trading_calendar_{exchange}_{start_date}_{end_date}").
            mode: Ingestion mode ('upsert' for 覆盖更新, 'append' for 追加).

        Returns:
            Dictionary with ingestion statistics.
        """
        task_name = task_name or f"trading_calendar_{exchange}_{start_date}_{end_date}"
        state = self._get_state(task_name)

        # Acquire task lock
        try:
            if not self.task_lock.acquire(task_name):
                raise TaskLockError(f"Task '{task_name}' is already running or locked.")
            logger.info(f"Task '{task_name}' lock acquired.")
        except TaskLockError as e:
            logger.error(f"Failed to acquire lock for task '{task_name}': {e}")
            raise

        # Reset stats for this run
        stats = {
            "fetched": 0,
            "saved": 0,
            "errors": 0,
        }
        batch: List[TradingCalendar] = []
        state_save_counter = 0
        failed_records: List[Dict] = []

        logger.info(
            f"Starting trading calendar ingestion: exchange={exchange or 'ALL'}, "
            f"start_date={start_date}, end_date={end_date}, mode={mode}"
        )

        try:
            state.status = "running"
            state.mode = mode
            self._save_state(state)

            # Create repository with appropriate mode
            calendar_repo = TradingCalendarRepo(self.db_config)
            if mode == "append":
                calendar_repo.on_conflict = "error"
            else:
                calendar_repo.on_conflict = "update"

            with self.source, calendar_repo:
                # Test connection
                if not self.source.test_connection():
                    raise ConnectionError("Tushare connection test failed")

                # Fetch trading calendar data
                record_count = 0
                for record in self.source.fetch_trade_cal(
                    exchange=exchange,
                    start_date=start_date,
                    end_date=end_date,
                ):
                    record_count += 1
                    stats["fetched"] += 1

                    try:
                        # Convert Tushare record to TradingCalendar model
                        calendar = self._convert_to_trading_calendar(record)
                        if calendar:
                            batch.append(calendar)

                            # Save batch when it reaches batch_size
                            if len(batch) >= batch_size:
                                try:
                                    saved = calendar_repo.save_batch_models(batch)
                                    stats["saved"] += saved
                                    logger.debug(f"Saved batch of {saved} calendar records")
                                    batch = []

                                    # Update state checkpoint
                                    state.last_processed_time = datetime.now()
                                    state.total_fetched = (state.total_fetched or 0) + stats["fetched"]
                                    state.total_saved = (state.total_saved or 0) + stats["saved"]
                                    state.total_errors = (state.total_errors or 0) + stats["errors"]
                                    state_save_counter += 1

                                    # Save state every 10 batches
                                    if state_save_counter >= 10:
                                        self._save_state(state)
                                        state_save_counter = 0
                                        stats = {"fetched": 0, "saved": 0, "errors": 0}
                                except Exception as e:
                                    stats["errors"] += len(batch)
                                    for cal in batch:
                                        failed_record = {
                                            "exchange": cal.exchange,
                                            "cal_date": str(cal.cal_date),
                                            "reason": f"batch_save_error: {str(e)}",
                                        }
                                        failed_records.append(failed_record)
                                    logger.error(f"Error saving batch (size={len(batch)}): {e}", exc_info=True)
                                    batch = []
                        else:
                            stats["errors"] += 1
                            failed_record = {
                                "exchange": record.get("exchange", "N/A"),
                                "cal_date": record.get("cal_date", "N/A"),
                                "reason": "conversion_failed",
                                "record": record.copy(),
                            }
                            failed_records.append(failed_record)
                            logger.info(
                                f"Skipped record (conversion failed): exchange={record.get('exchange', 'N/A')}, "
                                f"cal_date={record.get('cal_date', 'N/A')}, "
                                f"is_open={record.get('is_open', 'N/A')}, "
                                f"pretrade_date={record.get('pretrade_date', 'N/A')}"
                            )

                    except Exception as e:
                        stats["errors"] += 1
                        failed_record = {
                            "exchange": record.get("exchange", "N/A"),
                            "cal_date": record.get("cal_date", "N/A"),
                            "reason": f"exception: {str(e)}",
                            "record": record.copy(),
                        }
                        failed_records.append(failed_record)
                        logger.error(f"Error converting record: {e}, record={record}", exc_info=True)

                # Save remaining records
                if batch:
                    try:
                        saved = calendar_repo.save_batch_models(batch)
                        stats["saved"] += saved
                        logger.debug(f"Saved final batch of {saved} calendar records")
                    except Exception as e:
                        stats["errors"] += len(batch)
                        for cal in batch:
                            failed_record = {
                                "exchange": cal.exchange,
                                "cal_date": str(cal.cal_date),
                                "reason": f"final_batch_save_error: {str(e)}",
                            }
                            failed_records.append(failed_record)
                        logger.error(f"Error saving final batch: {e}", exc_info=True)

                # Final state update
                state.status = "completed"
                state.last_processed_key = None
                state.last_processed_time = datetime.now()
                state.total_fetched = (state.total_fetched or 0) + stats["fetched"]
                state.total_saved = (state.total_saved or 0) + stats["saved"]
                state.total_errors = (state.total_errors or 0) + stats["errors"]
                self._save_state(state)

                # Log final accumulated stats
                logger.info(
                    f"Trading calendar ingestion completed: "
                    f"Total (accumulated): Fetched={state.total_fetched}, Saved={state.total_saved}, Errors={state.total_errors} | "
                    f"This run: Fetched={stats['fetched']}, Saved={stats['saved']}, Errors={stats['errors']}"
                )
                logger.info(f"Total records fetched from Tushare API: {record_count}")
                logger.info(f"DEBUG: failed_records count = {len(failed_records)}, errors count = {stats['errors']}")

                # Print failed records if any
                if failed_records:
                    logger.info("=" * 80)
                    logger.info(f"Failed Records Summary (Total: {len(failed_records)})")
                    logger.info("=" * 80)
                    max_records_to_show = 100
                    for idx, failed_record in enumerate(failed_records[:max_records_to_show], 1):
                        logger.info(
                            f"[{idx}] exchange={failed_record['exchange']}, "
                            f"cal_date={failed_record['cal_date']}, reason={failed_record['reason']}"
                        )
                    if len(failed_records) > max_records_to_show:
                        logger.info(
                            f"... and {len(failed_records) - max_records_to_show} more failed records "
                            f"(total: {len(failed_records)})"
                        )
                    logger.info("=" * 80)
                elif stats['errors'] > 0:
                    logger.warning(f"WARNING: {stats['errors']} errors reported but no failed records collected. "
                                 f"This may indicate errors occurred during batch saves or other operations.")

        except Exception as e:
            state.status = "failed"
            state.error_message = str(e)
            state.last_processed_time = datetime.now()
            state.total_fetched = (state.total_fetched or 0) + stats["fetched"]
            state.total_saved = (state.total_saved or 0) + stats["saved"]
            state.total_errors = (state.total_errors or 0) + stats["errors"]
            self._save_state(state)
            logger.error(f"Trading calendar ingestion failed: {e}", exc_info=True)
            raise
        finally:
            # Release task lock
            self.task_lock.release(task_name)
            logger.info(f"Task '{task_name}' lock released.")

        return stats

    def _convert_to_trading_calendar(self, record: Dict) -> Optional[TradingCalendar]:
        """
        Convert Tushare record to TradingCalendar model.

        Args:
            record: Tushare API record.

        Returns:
            TradingCalendar model instance or None if conversion fails.
        """
        try:
            # Parse cal_date (required field)
            cal_date_str = record.get("cal_date", "")
            if not cal_date_str or cal_date_str == "00000000":
                logger.debug(f"Skipping record with invalid cal_date: {record.get('exchange', 'unknown')}")
                return None

            cal_date = datetime.strptime(cal_date_str, "%Y%m%d").date()

            # Parse pretrade_date if present
            pretrade_date = None
            pretrade_date_str = record.get("pretrade_date", "")
            if pretrade_date_str and pretrade_date_str != "00000000":
                pretrade_date = datetime.strptime(pretrade_date_str, "%Y%m%d").date()

            # Validate required fields
            exchange = record.get("exchange", "").strip()
            is_open_str = record.get("is_open", "")

            if not exchange:
                logger.debug(f"Skipping record with missing exchange: cal_date={cal_date_str}")
                return None

            # Convert is_open (Tushare returns "0" or "1" as string)
            is_open = False
            if is_open_str == "1" or is_open_str == 1 or is_open_str is True:
                is_open = True
            elif is_open_str == "0" or is_open_str == 0 or is_open_str is False:
                is_open = False
            else:
                logger.warning(f"Unexpected is_open value: {is_open_str}, defaulting to False")
                is_open = False

            return TradingCalendar(
                exchange=exchange,
                cal_date=cal_date,
                is_open=is_open,
                pretrade_date=pretrade_date,
            )
        except ValueError as e:
            logger.warning(f"Date parsing error for record {record.get('exchange', 'unknown')}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error converting to TradingCalendar: {e}, record: {record}")
            return None

    def get_task_state(self, task_name: str) -> Optional[IngestionState]:
        """
        Get ingestion state for a task.

        Args:
            task_name: Task name.

        Returns:
            IngestionState if found, None otherwise.
        """
        return self.state_repo.get_state(task_name)


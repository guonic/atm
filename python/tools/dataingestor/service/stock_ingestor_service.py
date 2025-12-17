"""
Stock data ingestor service.

Specialized service for ingesting stock data from Tushare and storing to database
using atm.repo.stock and atm.repo.kline repositories.

Note: This is a service module, not a standalone script.
Use sync_stock_basic.py or import this module in your code.
"""

import logging
import sys
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Dict, List, Optional

# Ensure project path is in sys.path before importing atm modules
_project_root = Path(__file__).parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from atm.config import DatabaseConfig
from atm.data.source import TushareSource, TushareSourceConfig
from atm.models.kline import StockKlineDay
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
    StockKlineDayRepo,
    TaskLockError,
)

logger = logging.getLogger(__name__)


class StockIngestorService:
    """
    Service for ingesting stock data from Tushare.

    This service uses TushareSource to fetch data and specialized repositories
    to store stock information and K-line data.
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
        Initialize stock ingestor service.

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
        self._stock_repo: Optional[StockBasicRepo] = None
        self._kline_repo: Optional[StockKlineDayRepo] = None
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
    def stock_repo(self) -> StockBasicRepo:
        """Get or create stock basic repository."""
        if self._stock_repo is None:
            self._stock_repo = StockBasicRepo(self.db_config)
        return self._stock_repo

    @property
    def kline_repo(self) -> StockKlineDayRepo:
        """Get or create K-line repository."""
        if self._kline_repo is None:
            self._kline_repo = StockKlineDayRepo(self.db_config)
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

    def _get_last_processed_key(self, task_name: str) -> Optional[str]:
        """Get last processed key from state."""
        state = self.state_repo.get_state(task_name)
        return state.last_processed_key if state else None

    def ingest_stock_basic(
        self,
        exchange: str = "",
        list_status: str = "L",
        batch_size: int = 100,
        task_name: Optional[str] = None,
        resume: bool = True,
        mode: str = "upsert",
    ) -> Dict[str, int]:
        """
        Ingest stock basic information with checkpoint/resume support.

        Args:
            exchange: Exchange code (SSE/SZSE/BSE, empty for all).
            list_status: List status (L=listed, D=delisted, P=pause).
            batch_size: Batch size for saving.
            task_name: Task name for state tracking (default: "stock_basic_{exchange}_{list_status}").
            resume: Whether to resume from last checkpoint.

        Returns:
            Dictionary with ingestion statistics.
        """
        task_name = task_name or f"stock_basic_{exchange}_{list_status}"
        state = self._get_state(task_name)

        # Acquire task lock
        try:
            if not self.task_lock.acquire(task_name):
                raise TaskLockError(f"Task '{task_name}' is already running or locked.")
            logger.info(f"Task '{task_name}' lock acquired.")
        except TaskLockError as e:
            logger.error(f"Failed to acquire lock for task '{task_name}': {e}")
            raise

        # Resume from checkpoint logic
        # Note: In upsert mode, we don't resume from checkpoint because we want to reprocess all data
        # to ensure everything is up-to-date. Resume is only useful for append mode.
        last_key = None
        if mode == "upsert":
            # Force start from beginning in upsert mode
            resume = False
            logger.info("Upsert mode: starting from beginning (no resume)")
        elif resume and state.last_processed_key:
            last_key = state.last_processed_key
            logger.info(f"Resuming from checkpoint: last_processed_key={last_key}")

        # Reset stats for this run (don't accumulate from previous runs)
        # We'll update state with final stats at the end
        stats = {
            "fetched": 0,  # Start fresh for this run
            "saved": 0,    # Start fresh for this run
            "errors": 0,    # Start fresh for this run
        }
        batch: List[StockBasic] = []
        skip_until_key = last_key
        state_save_counter = 0
        failed_records: List[Dict] = []  # Collect failed records for reporting

        logger.info(f"Starting stock basic ingestion: exchange={exchange}, status={list_status}, mode={mode}")

        try:
            state.status = "running"
            state.mode = mode
            self._save_state(state)

            # Create repository with appropriate mode
            stock_repo = StockBasicRepo(self.db_config)
            # Note: StockBasicRepo uses DatabaseRepo which supports on_conflict parameter
            # For append mode, we need to modify the repo's on_conflict behavior
            if mode == "append":
                stock_repo.on_conflict = "error"  # Will fail on conflict in append mode
            else:
                stock_repo.on_conflict = "update"  # Upsert mode

            with self.source, stock_repo:
                # Test connection
                if not self.source.test_connection():
                    raise ConnectionError("Tushare connection test failed")

                # Fetch stock basic data
                # Log fetch parameters for debugging
                logger.info(f"Fetching stock basic data with params: exchange={exchange or 'ALL'}, list_status={list_status or 'ALL'}")
                record_count = 0  # Track total records fetched from API
                for record in self.source.fetch_stock_basic(
                    exchange=exchange,
                    list_status=list_status,
                ):
                    record_count += 1
                    ts_code = record.get("ts_code", "")
                    
                    # Log progress every 1000 records
                    if record_count % 1000 == 0:
                        logger.info(f"Fetched {record_count} records from Tushare API so far...")

                    # Skip until we reach the last processed key
                    if skip_until_key and ts_code != skip_until_key:
                        continue
                    elif skip_until_key and ts_code == skip_until_key:
                        skip_until_key = None  # Found checkpoint, continue from next
                        logger.info(f"Resumed from checkpoint: {ts_code}")
                        continue

                    stats["fetched"] += 1

                    try:
                        # Convert Tushare record to StockBasic model
                        stock = self._convert_to_stock_basic(record)
                        if stock:
                            batch.append(stock)

                            # Save batch when it reaches batch_size
                            if len(batch) >= batch_size:
                                try:
                                    saved = stock_repo.save_batch_models(batch)
                                    stats["saved"] += saved
                                    logger.debug(f"Saved batch of {saved} stocks")
                                    batch = []

                                    # Update state checkpoint (only in append mode)
                                    # In upsert mode, we don't need checkpoint because we always start from beginning
                                    if mode != "upsert":
                                        state.last_processed_key = ts_code
                                        state.last_processed_time = datetime.now()
                                    # Accumulate stats: add current run stats to previous totals
                                    state.total_fetched = (state.total_fetched or 0) + stats["fetched"]
                                    state.total_saved = (state.total_saved or 0) + stats["saved"]
                                    state.total_errors = (state.total_errors or 0) + stats["errors"]
                                    state_save_counter += 1

                                    # Save state every 10 batches
                                    if state_save_counter >= 10:
                                        self._save_state(state)
                                        state_save_counter = 0
                                except Exception as e:
                                    stats["errors"] += len(batch)
                                    # Record all records in the failed batch
                                    for stock in batch:
                                        failed_record = {
                                            "ts_code": stock.ts_code,
                                            "reason": f"batch_save_error: {str(e)}",
                                            "record": {"ts_code": stock.ts_code, "symbol": stock.symbol, "name": stock.full_name},
                                        }
                                        failed_records.append(failed_record)
                                    logger.error(f"Error saving batch (size={len(batch)}): {e}", exc_info=True)
                                    batch = []  # Clear batch on error
                        else:
                            # Conversion returned None (missing required fields or conversion failed)
                            stats["errors"] += 1
                            failed_record = {
                                "ts_code": ts_code,
                                "reason": "conversion_failed",
                                "record": record.copy(),
                            }
                            failed_records.append(failed_record)
                            # Log at INFO level for better visibility
                            logger.info(f"Skipped record (conversion failed): ts_code={ts_code}, "
                                      f"list_date={record.get('list_date', 'N/A')}, "
                                      f"symbol={record.get('symbol', 'N/A')}, "
                                      f"name={record.get('name', 'N/A')}, "
                                      f"exchange={record.get('exchange', 'N/A')}, "
                                      f"market={record.get('market', 'N/A')}")

                    except Exception as e:
                        stats["errors"] += 1
                        failed_record = {
                            "ts_code": ts_code,
                            "reason": f"exception: {str(e)}",
                            "record": record.copy(),
                        }
                        failed_records.append(failed_record)
                        logger.error(f"Error converting record: {e}, ts_code={ts_code}, record={record}", exc_info=True)

                # Save remaining records
                if batch:
                    try:
                        saved = stock_repo.save_batch_models(batch)
                        stats["saved"] += saved
                        logger.debug(f"Saved final batch of {saved} stocks")
                    except Exception as e:
                        stats["errors"] += len(batch)
                        # Record all records in the failed final batch
                        for stock in batch:
                            failed_record = {
                                "ts_code": stock.ts_code,
                                "reason": f"final_batch_save_error: {str(e)}",
                                "record": {"ts_code": stock.ts_code, "symbol": stock.symbol, "name": stock.full_name},
                            }
                            failed_records.append(failed_record)
                        logger.error(f"Error saving final batch: {e}", exc_info=True)

                # Final state update
                state.status = "completed"
                state.last_processed_key = None  # Clear checkpoint on completion
                state.last_processed_time = datetime.now()
                # Accumulate stats: add current run stats to previous totals
                state.total_fetched = (state.total_fetched or 0) + stats["fetched"]
                state.total_saved = (state.total_saved or 0) + stats["saved"]
                state.total_errors = (state.total_errors or 0) + stats["errors"]
                self._save_state(state)

                # Log final accumulated stats
                logger.info(
                    f"Stock basic ingestion completed: "
                    f"Total (accumulated): Fetched={state.total_fetched}, Saved={state.total_saved}, Errors={state.total_errors} | "
                    f"This run: Fetched={stats['fetched']}, Saved={stats['saved']}, Errors={stats['errors']}"
                )
                
                # Print all failed records
                logger.info(f"DEBUG: failed_records count = {len(failed_records)}, errors count = {stats['errors']}")
                if failed_records:
                    logger.info("=" * 80)
                    logger.info(f"Failed Records Summary (Total: {len(failed_records)})")
                    logger.info("=" * 80)
                    # Limit output to first 100 records to avoid overwhelming logs
                    max_records_to_show = 100
                    for idx, failed_record in enumerate(failed_records[:max_records_to_show], 1):
                        logger.info(f"[{idx}] ts_code={failed_record['ts_code']}, reason={failed_record['reason']}")
                        record_info = failed_record.get('record', {})
                        if isinstance(record_info, dict):
                            logger.info(f"     Record details: ts_code={record_info.get('ts_code', 'N/A')}, "
                                      f"symbol={record_info.get('symbol', 'N/A')}, "
                                      f"name={record_info.get('name', record_info.get('full_name', 'N/A'))}, "
                                      f"list_date={record_info.get('list_date', 'N/A')}, "
                                      f"exchange={record_info.get('exchange', 'N/A')}, "
                                      f"market={record_info.get('market', 'N/A')}")
                        else:
                            logger.info(f"     Record: {record_info}")
                    if len(failed_records) > max_records_to_show:
                        logger.info(f"... and {len(failed_records) - max_records_to_show} more failed records (total: {len(failed_records)})")
                    logger.info("=" * 80)
                elif stats['errors'] > 0:
                    logger.warning(f"WARNING: {stats['errors']} errors reported but no failed records collected. "
                                 f"This may indicate errors occurred during batch saves or other operations.")

        except Exception as e:
            state.status = "failed"
            state.error_message = str(e)
            state.last_processed_time = datetime.now()
            # Accumulate stats: add current run stats to previous totals
            state.total_fetched = (state.total_fetched or 0) + stats["fetched"]
            state.total_saved = (state.total_saved or 0) + stats["saved"]
            state.total_errors = (state.total_errors or 0) + stats["errors"]
            self._save_state(state)
            logger.error(f"Stock basic ingestion failed: {e}", exc_info=True)
            
            # Print failed records even on exception
            if failed_records:
                logger.info("=" * 80)
                logger.info(f"Failed Records Summary (Total: {len(failed_records)})")
                logger.info("=" * 80)
                for idx, failed_record in enumerate(failed_records, 1):
                    logger.info(f"[{idx}] ts_code={failed_record['ts_code']}, reason={failed_record['reason']}")
                    record_info = failed_record.get('record', {})
                    if isinstance(record_info, dict):
                        logger.info(f"     Record details: ts_code={record_info.get('ts_code', 'N/A')}, "
                                  f"symbol={record_info.get('symbol', 'N/A')}, "
                                  f"name={record_info.get('name', record_info.get('full_name', 'N/A'))}, "
                                  f"list_date={record_info.get('list_date', 'N/A')}, "
                                  f"exchange={record_info.get('exchange', 'N/A')}, "
                                  f"market={record_info.get('market', 'N/A')}")
                    else:
                        logger.info(f"     Record: {record_info}")
                logger.info("=" * 80)
            raise
        finally:
            # Release task lock
            try:
                self.task_lock.release(task_name)
                logger.info(f"Task '{task_name}' lock released.")
            except Exception as e:
                logger.warning(f"Failed to release lock for task '{task_name}': {e}")

        return stats

    def ingest_daily_kline(
        self,
        ts_code: str,
        start_date: str,
        end_date: str,
        batch_size: int = 100,
        task_name: Optional[str] = None,
        resume: bool = True,
        mode: str = "upsert",
    ) -> Dict[str, int]:
        """
        Ingest daily K-line data for a stock with checkpoint/resume support.

        Args:
            ts_code: Stock code (e.g., '000001.SZ').
            start_date: Start date (YYYYMMDD).
            end_date: End date (YYYYMMDD).
            batch_size: Batch size for saving.
            task_name: Task name for state tracking (default: "kline_day_{ts_code}_{start_date}_{end_date}").
            resume: Whether to resume from last checkpoint.

        Returns:
            Dictionary with ingestion statistics.
        """
        task_name = task_name or f"kline_day_{ts_code}_{start_date}_{end_date}"
        state = self._get_state(task_name)

        # Acquire task lock
        try:
            if not self.task_lock.acquire(task_name):
                raise TaskLockError(f"Task '{task_name}' is already running or locked.")
            logger.info(f"Task '{task_name}' lock acquired.")
        except TaskLockError as e:
            logger.error(f"Failed to acquire lock for task '{task_name}': {e}")
            raise

        # Resume from checkpoint logic
        # Note: In upsert mode, we don't resume from checkpoint because we want to reprocess all data
        # to ensure everything is up-to-date. Resume is only useful for append mode.
        last_date = None
        if mode == "upsert":
            # Force start from beginning in upsert mode
            resume = False
            logger.info("Upsert mode: starting from beginning (no resume)")
        elif resume and state.last_processed_key:
            last_date = state.last_processed_key
            logger.info(f"Resuming from checkpoint: last_processed_date={last_date}")

        # Reset stats for this run (don't accumulate from previous runs)
        # We'll update state with final stats at the end
        stats = {
            "fetched": 0,  # Start fresh for this run
            "saved": 0,    # Start fresh for this run
            "errors": 0,    # Start fresh for this run
        }
        batch: List[StockKlineDay] = []
        skip_until_date = last_date
        state_save_counter = 0

        logger.info(f"Starting daily K-line ingestion: {ts_code}, {start_date} to {end_date}, mode={mode}")

        try:
            state.status = "running"
            state.mode = mode
            self._save_state(state)

            # Create repository with appropriate mode
            kline_repo = StockKlineDayRepo(self.db_config)
            if mode == "append":
                kline_repo.on_conflict = "error"  # Will fail on conflict in append mode
            else:
                kline_repo.on_conflict = "update"  # Upsert mode

            with self.source, kline_repo:
                # Test connection
                if not self.source.test_connection():
                    raise ConnectionError("Tushare connection test failed")

                # Fetch daily K-line data
                for record in self.source.fetch_daily(
                    ts_code=ts_code,
                    start_date=start_date,
                    end_date=end_date,
                ):
                    trade_date = record.get("trade_date", "")

                    # Skip until we reach the last processed date
                    if skip_until_date and trade_date != skip_until_date:
                        continue
                    elif skip_until_date and trade_date == skip_until_date:
                        skip_until_date = None  # Found checkpoint, continue from next
                        logger.info(f"Resumed from checkpoint: {trade_date}")
                        continue

                    stats["fetched"] += 1

                    try:
                        # Convert Tushare record to StockKlineDay model
                        kline = self._convert_to_kline_day(record)
                        if kline:
                            batch.append(kline)

                            # Save batch when it reaches batch_size
                            if len(batch) >= batch_size:
                                saved = kline_repo.save_batch_models(batch)
                                stats["saved"] += saved
                                logger.debug(f"Saved batch of {saved} K-lines")
                                batch = []

                                # Update state checkpoint (only in append mode)
                                # In upsert mode, we don't need checkpoint because we always start from beginning
                                if mode != "upsert":
                                    state.last_processed_key = trade_date
                                    state.last_processed_time = datetime.now()
                                # Accumulate stats: add current run stats to previous totals
                                state.total_fetched = (state.total_fetched or 0) + stats["fetched"]
                                state.total_saved = (state.total_saved or 0) + stats["saved"]
                                state.total_errors = (state.total_errors or 0) + stats["errors"]
                                state_save_counter += 1

                                # Save state every 10 batches
                                if state_save_counter >= 10:
                                    self._save_state(state)
                                    state_save_counter = 0
                                    # Reset stats for next batch cycle (already accumulated to state)
                                    stats = {"fetched": 0, "saved": 0, "errors": 0}

                    except Exception as e:
                        stats["errors"] += 1
                        logger.error(f"Error converting K-line record: {e}", exc_info=True)

                # Save remaining records
                if batch:
                    try:
                        saved = kline_repo.save_batch_models(batch)
                        stats["saved"] += saved
                        logger.debug(f"Saved final batch of {saved} K-lines")
                    except Exception as e:
                        stats["errors"] += len(batch)
                        logger.error(f"Error saving final K-line batch: {e}", exc_info=True)

                # Final state update
                state.status = "completed"
                state.last_processed_key = None  # Clear checkpoint on completion
                state.last_processed_time = datetime.now()
                # Accumulate stats: add current run stats to previous totals
                state.total_fetched = (state.total_fetched or 0) + stats["fetched"]
                state.total_saved = (state.total_saved or 0) + stats["saved"]
                state.total_errors = (state.total_errors or 0) + stats["errors"]
                self._save_state(state)

                # Log final accumulated stats
                logger.info(
                    f"Daily K-line ingestion completed for {ts_code}: "
                    f"Total (accumulated): Fetched={state.total_fetched}, Saved={state.total_saved}, Errors={state.total_errors} | "
                    f"This run: Fetched={stats['fetched']}, Saved={stats['saved']}, Errors={stats['errors']}"
                )

        except Exception as e:
            state.status = "failed"
            state.error_message = str(e)
            state.last_processed_time = datetime.now()
            # Accumulate stats: add current run stats to previous totals
            state.total_fetched = (state.total_fetched or 0) + stats["fetched"]
            state.total_saved = (state.total_saved or 0) + stats["saved"]
            state.total_errors = (state.total_errors or 0) + stats["errors"]
            self._save_state(state)
            logger.error(f"Daily K-line ingestion failed: {e}", exc_info=True)
            raise
        finally:
            # Release task lock
            try:
                self.task_lock.release(task_name)
                logger.info(f"Task '{task_name}' lock released.")
            except Exception as e:
                logger.warning(f"Failed to release lock for task '{task_name}': {e}")

        return stats

    def ingest_daily_kline_batch(
        self,
        ts_codes: List[str],
        start_date: str,
        end_date: str,
        batch_size: int = 100,
        resume: bool = True,
        mode: str = "upsert",
    ) -> Dict[str, Dict[str, int]]:
        """
        Ingest daily K-line data for multiple stocks with checkpoint/resume support.

        Args:
            ts_codes: List of stock codes.
            start_date: Start date (YYYYMMDD).
            end_date: End date (YYYYMMDD).
            batch_size: Batch size for saving.
            resume: Whether to resume from last checkpoint for each stock.

        Returns:
            Dictionary mapping ts_code to statistics.
        """
        results = {}

        for ts_code in ts_codes:
            logger.info(f"Processing {ts_code}...")
            try:
                stats = self.ingest_daily_kline(
                    ts_code=ts_code,
                    start_date=start_date,
                    end_date=end_date,
                    batch_size=batch_size,
                    resume=resume,
                    mode=mode,
                )
                results[ts_code] = stats
            except Exception as e:
                logger.error(f"Failed to ingest K-line for {ts_code}: {e}")
                results[ts_code] = {"fetched": 0, "saved": 0, "errors": 1}

        return results

    def get_task_state(self, task_name: str) -> Optional[IngestionState]:
        """
        Get ingestion state for a task.

        Args:
            task_name: Task name.

        Returns:
            IngestionState if found, None otherwise.
        """
        return self.state_repo.get_state(task_name)

    def list_task_states(self) -> list[IngestionState]:
        """
        List all task states.

        Returns:
            List of ingestion states.
        """
        return self.state_repo.list_states()

    def reset_task_state(self, task_name: str) -> bool:
        """
        Reset/delete task state.

        Args:
            task_name: Task name.

        Returns:
            True if reset was successful.
        """
        return self.state_repo.delete_state(task_name)

    def _convert_to_stock_basic(self, record: Dict) -> Optional[StockBasic]:
        """
        Convert Tushare record to StockBasic model.

        Args:
            record: Tushare API record.

        Returns:
            StockBasic model instance or None if conversion fails.
        """
        try:
            # Parse list_date (required field)
            list_date_str = record.get("list_date", "")
            if not list_date_str or list_date_str == "00000000":
                logger.debug(f"Skipping record with invalid list_date: {record.get('ts_code', 'unknown')}")
                return None

            list_date = datetime.strptime(list_date_str, "%Y%m%d").date()

            # Parse delist_date if present
            delist_date = None
            delist_date_str = record.get("delist_date", "")
            if delist_date_str and delist_date_str != "00000000":
                delist_date = datetime.strptime(delist_date_str, "%Y%m%d").date()

            # Validate required fields
            ts_code = record.get("ts_code", "").strip()
            symbol = record.get("symbol", "").strip()
            name = record.get("name", "").strip()
            exchange = record.get("exchange", "").strip()
            market = record.get("market", "").strip()

            if not ts_code or not symbol or not name:
                logger.debug(f"Skipping record with missing required fields: ts_code={ts_code}, symbol={symbol}, name={name}")
                return None

            # Infer exchange and market from ts_code if not provided by Tushare
            if not exchange:
                if ts_code.endswith(".SH"):
                    exchange = "SSE"
                elif ts_code.endswith(".SZ"):
                    exchange = "SZSE"
                elif ts_code.endswith(".BJ"):
                    exchange = "BSE"
                else:
                    logger.debug(f"Skipping record with empty exchange and cannot infer from ts_code: ts_code={ts_code}")
                    return None

            if not market:
                # Infer market from ts_code prefix
                code_prefix = ts_code.split(".")[0] if "." in ts_code else ts_code
                if code_prefix.startswith("688") or code_prefix.startswith("689"):
                    market = "科创板"
                elif code_prefix.startswith("300") or code_prefix.startswith("301"):
                    market = "创业板"
                elif code_prefix.startswith("8") or code_prefix.startswith("43") or code_prefix.startswith("83"):
                    market = "北交所"
                elif ts_code.endswith(".SH"):
                    market = "沪A"
                elif ts_code.endswith(".SZ"):
                    market = "深A"
                else:
                    logger.debug(f"Skipping record with empty market and cannot infer from ts_code: ts_code={ts_code}")
                    return None

            try:
                return StockBasic(
                    ts_code=ts_code,
                    symbol=symbol,
                    full_name=name,
                    exchange=exchange,
                    market=market,
                    list_date=list_date,
                    delist_date=delist_date,
                    is_listed=record.get("list_status", "") == "L",
                    currency=record.get("currency", "CNY").strip() or "CNY",
                )
            except Exception as e:
                # Pydantic validation error
                logger.warning(f"Pydantic validation error for record ts_code={ts_code}: {e}")
                return None
        except ValueError as e:
            logger.warning(f"Date parsing error for record ts_code={record.get('ts_code', 'unknown')}: {e}, list_date={record.get('list_date', 'N/A')}")
            return None
        except Exception as e:
            logger.error(f"Error converting to StockBasic: {e}, ts_code={record.get('ts_code', 'unknown')}, record={record}")
            return None

    def _convert_to_kline_day(self, record: Dict) -> Optional[StockKlineDay]:
        """
        Convert Tushare record to StockKlineDay model.

        Args:
            record: Tushare API record.

        Returns:
            StockKlineDay model instance or None if conversion fails.
        """
        try:
            # Parse trade_date
            trade_date_str = record.get("trade_date", "")
            if not trade_date_str:
                return None

            trade_date = datetime.strptime(trade_date_str, "%Y%m%d")

            # Convert numeric fields
            def to_decimal(value):
                if value is None:
                    return None
                try:
                    return Decimal(str(value))
                except (ValueError, TypeError):
                    return None

            def to_int(value):
                if value is None:
                    return None
                try:
                    return int(value)
                except (ValueError, TypeError):
                    return None

            return StockKlineDay(
                ts_code=record.get("ts_code", ""),
                trade_date=trade_date,
                open=to_decimal(record.get("open")),
                high=to_decimal(record.get("high")),
                low=to_decimal(record.get("low")),
                close=to_decimal(record.get("close")),
                pre_close=to_decimal(record.get("pre_close")),
                volume=to_int(record.get("vol")),
                amount=to_decimal(record.get("amount")),
                turnover=to_decimal(record.get("turnover_rate")),
                pct_chg=to_decimal(record.get("pct_chg")),
            )
        except Exception as e:
            logger.error(f"Error converting to StockKlineDay: {e}, record: {record}")
            return None

    def close(self) -> None:
        """Close all connections."""
        if self._source:
            self._source.close()
        if self._stock_repo:
            self._stock_repo.close()
        if self._kline_repo:
            self._kline_repo.close()
        if isinstance(self._state_repo, DatabaseStateRepo):
            self._state_repo.close()
        # Note: Task locks are released automatically via context manager or explicitly

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


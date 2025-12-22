"""
K-line synchronization service.

Synchronizes K-line data for all stocks, checking existing data and syncing from the appropriate start date.
"""

import logging
import sys
from datetime import date, datetime, timedelta
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

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
from atm.models.stock import StockKlineSyncState
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
    StockKlineSyncStateRepo,
    StockKlineWeekRepo,
    TaskLockError,
)

logger = logging.getLogger(__name__)

# Tushare API limits
TUSHARE_MAX_RECORDS = 6000  # Maximum records per API request

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
        self._sync_state_repo: Optional[StockKlineSyncStateRepo] = None
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
        if self._sync_state_repo:
            self._sync_state_repo.close()

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
    def sync_state_repo(self) -> StockKlineSyncStateRepo:
        """Get or create sync state repository."""
        if self._sync_state_repo is None:
            self._sync_state_repo = StockKlineSyncStateRepo(self.db_config)
        return self._sync_state_repo

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
        Get the last synced date for a stock from sync state table.

        Args:
            ts_code: Stock code.

        Returns:
            Last synced date if exists, None otherwise.
        """
        try:
            sync_state = self.sync_state_repo.get_by_ts_code_and_type(
                ts_code=ts_code, kline_type=self.kline_type
            )
            if sync_state and sync_state.last_synced_date:
                return sync_state.last_synced_date
        except Exception as e:
            logger.warning(f"Failed to get last synced date for {ts_code}: {e}")
        return None

    def _update_sync_state(
        self, ts_code: str, last_synced_date: date, total_records: int = 0
    ) -> None:
        """
        Update sync state for a stock.

        Args:
            ts_code: Stock code.
            last_synced_date: Last synced date.
            total_records: Total records synced (optional, for tracking).
        """
        try:
            sync_state = StockKlineSyncState(
                ts_code=ts_code,
                kline_type=self.kline_type,
                last_synced_date=last_synced_date,
                last_synced_time=datetime.now(),
                total_records=total_records,
                update_time=datetime.now(),
            )
            self.sync_state_repo.save_model(sync_state)
        except Exception as e:
            logger.warning(f"Failed to update sync state for {ts_code}: {e}")

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
            if self.kline_type in ["day", "week", "month", "quarter"]:
                # For daily/weekly/monthly/quarterly, use trade_date
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

            # Add price and volume fields (only if not None/NaN and valid)
            # Helper function to safely convert to Decimal
            def safe_decimal(value, field_name: str):
                """Safely convert value to Decimal, handling None, NaN, and invalid values."""
                if value is None:
                    return None
                # Check for pandas/numpy NaN
                if pd.isna(value) if hasattr(pd, 'isna') else (isinstance(value, float) and value != value):
                    return None
                try:
                    return Decimal(str(value))
                except (ValueError, TypeError, InvalidOperation, Exception):
                    logger.debug(f"Invalid {field_name} value: {value} for {ts_code}")
                    return None
            
            # Helper function to safely convert to int
            def safe_int(value, field_name: str):
                """Safely convert value to int, handling None, NaN, and invalid values."""
                if value is None:
                    return None
                # Check for pandas/numpy NaN
                if pd.isna(value) if hasattr(pd, 'isna') else (isinstance(value, float) and value != value):
                    return None
                try:
                    return int(float(value))
                except (ValueError, TypeError, OverflowError, Exception):
                    logger.debug(f"Invalid {field_name} value: {value} for {ts_code}")
                    return None
            
            # Add fields only if conversion succeeds
            if "open" in record:
                open_val = safe_decimal(record["open"], "open")
                if open_val is not None:
                    model_data["open"] = open_val
            if "high" in record:
                high_val = safe_decimal(record["high"], "high")
                if high_val is not None:
                    model_data["high"] = high_val
            if "low" in record:
                low_val = safe_decimal(record["low"], "low")
                if low_val is not None:
                    model_data["low"] = low_val
            if "close" in record:
                close_val = safe_decimal(record["close"], "close")
                if close_val is not None:
                    model_data["close"] = close_val
            if "vol" in record:
                vol_val = safe_int(record["vol"], "volume")
                if vol_val is not None:
                    model_data["volume"] = vol_val
            if "amount" in record:
                amount_val = safe_decimal(record["amount"], "amount")
                if amount_val is not None:
                    model_data["amount"] = amount_val

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

                # Get all stocks from database
                logger.info(f"Fetching stocks from database: exchange={exchange or 'ALL'}, list_status={list_status or 'ALL'}")
                stocks = self.stock_repo.get_by_exchange(exchange=exchange, list_status=list_status)

                total_stocks = len(stocks)
                logger.info(f"Found {total_stocks} stocks to sync")

                # First pass: collect sync requirements
                sync_queue = []
                for idx, stock in enumerate(stocks, 1):
                    ts_code = stock.ts_code
                    logger.info(f"[{idx}/{total_stocks}] Analyzing {ts_code} ({stock.full_name})...")

                    try:
                        # Get last synced date
                        last_synced_date = self._get_last_synced_date(ts_code)

                        # Determine start date
                        if last_synced_date:
                            # Start from the day after last synced date
                            start_date = (last_synced_date + timedelta(days=1)).strftime("%Y%m%d")
                        else:
                            # Start from list_date
                            if stock.list_date:
                                start_date = stock.list_date.strftime("%Y%m%d")
                            else:
                                logger.warning(f"  Skipping {ts_code}: no list_date")
                                continue

                        # Skip if start_date is after end_date
                        if start_date > end_date:
                            logger.info(f"  Skipping {ts_code}: start_date ({start_date}) > end_date ({end_date})")
                            continue

                        # Store sync info for batch processing
                        sync_queue.append({
                            "ts_code": ts_code,
                            "start_date": start_date,
                            "end_date": end_date,
                            "last_synced_date": last_synced_date,
                        })
                    except Exception as e:
                        logger.error(f"  Failed to analyze {ts_code}: {e}", exc_info=True)
                        failed_stocks.append(ts_code)
                        results[ts_code] = {"fetched": 0, "saved": 0, "errors": 1}

                # Second pass: batch sync by time range
                if sync_queue:
                    logger.info(f"Processing {len(sync_queue)} stocks with batch sync optimization...")
                    batch_results = self._batch_sync_kline(
                        sync_queue=sync_queue,
                        batch_size=batch_size,
                        max_batch_stocks=50,  # Maximum stocks per API call
                    )
                    results.update(batch_results)

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

    def _fetch_kline_records(
        self,
        ts_code: str,
        start_date: str,
        end_date: str,
    ) -> List[Dict]:
        """
        Fetch K-line records from Tushare for a single stock.

        Args:
            ts_code: Stock code.
            start_date: Start date (YYYYMMDD).
            end_date: End date (YYYYMMDD).

        Returns:
            List of records.
        """
        if self.kline_type == "day":
            return list(self.source.fetch_daily(
                ts_code=ts_code,
                start_date=start_date,
                end_date=end_date,
            ))
        else:
            # For other frequencies, use pro_bar
            if not self.source._initialized:
                self.source._initialize()
            records = list(self.source.fetch_pro_bar(
                ts_code=ts_code,
                freq=self.kline_config["tushare_freq"],
                start_date=start_date,
                end_date=end_date,
            ))
            # Add ts_code to each record for consistency
            for record in records:
                record["ts_code"] = ts_code
            return records

    def _process_and_save_records(
        self,
        ts_code: str,
        records: List[Dict],
        batch_size: int = 100,
        end_date: Optional[str] = None,
    ) -> Dict[str, int]:
        """
        Process records and save to database.

        Args:
            ts_code: Stock code.
            records: List of records to process.
            batch_size: Batch size for saving.
            end_date: End date (YYYYMMDD) for updating sync state.

        Returns:
            Dictionary with statistics (fetched, saved, errors).
        """
        stats = {"fetched": 0, "saved": 0, "errors": 0}
        batch = []
        last_synced_date = None

        for record in records:
            stats["fetched"] += 1

            try:
                # Convert to model
                kline = self._convert_to_kline_model(record)
                if kline:
                    batch.append(kline)
                    
                    # Track last synced date from records
                    time_column = self.kline_config["time_column"]
                    record_time = getattr(kline, time_column, None)
                    if record_time:
                        if isinstance(record_time, datetime):
                            record_date = record_time.date()
                        elif isinstance(record_time, date):
                            record_date = record_time
                        else:
                            try:
                                record_date = datetime.strptime(str(record_time), "%Y-%m-%d").date()
                            except:
                                record_date = None
                        
                        if record_date and (last_synced_date is None or record_date > last_synced_date):
                            last_synced_date = record_date

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

        # Update sync state if records were saved
        if stats["saved"] > 0 and last_synced_date:
            self._update_sync_state(ts_code, last_synced_date, stats["saved"])
        elif stats["saved"] > 0 and end_date:
            # Fallback to end_date if we couldn't extract from records
            try:
                last_synced_date = datetime.strptime(end_date, "%Y%m%d").date()
                self._update_sync_state(ts_code, last_synced_date, stats["saved"])
            except Exception as e:
                logger.warning(f"Failed to parse end_date for sync state update: {e}")

        return stats

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
        try:
            records = self._fetch_kline_records(ts_code, start_date, end_date)
            return self._process_and_save_records(ts_code, records, batch_size, end_date=end_date)
        except Exception as e:
            logger.error(f"Error syncing K-line for {ts_code}: {e}", exc_info=True)
            return {"fetched": 0, "saved": 0, "errors": 1}

    def _batch_sync_kline(
        self,
        sync_queue: List[Dict],
        batch_size: int = 100,
        max_batch_stocks: int = 50,
    ) -> Dict[str, Dict[str, int]]:
        """
        Batch sync K-line data for multiple stocks.

        Groups stocks by time range and uses batch API calls when possible.

        Args:
            sync_queue: List of sync requirements, each with ts_code, start_date, end_date.
            batch_size: Batch size for saving to database.
            max_batch_stocks: Maximum number of stocks per API call.

        Returns:
            Dictionary mapping ts_code to statistics.
        """
        results = {}

        # Group stocks by time range (start_date, end_date)
        time_range_groups = {}
        for item in sync_queue:
            time_key = (item["start_date"], item["end_date"])
            if time_key not in time_range_groups:
                time_range_groups[time_key] = []
            time_range_groups[time_key].append(item)

        logger.info(f"Grouped {len(sync_queue)} stocks into {len(time_range_groups)} time range groups")

        # Process each time range group
        for (start_date, end_date), group_items in time_range_groups.items():
            logger.info(
                f"Processing time range {start_date} to {end_date} with {len(group_items)} stocks"
            )

            # For all K-line types, use batch API if multiple stocks
            if len(group_items) > 1:
                # Use batch API call for multiple stocks
                ts_codes = [item["ts_code"] for item in group_items]
                
                # Estimate data volume and split if exceeds Tushare limit
                estimated_records = self._estimate_data_volume(
                    ts_codes=ts_codes,
                    start_date=start_date,
                    end_date=end_date,
                )
                
                if estimated_records > TUSHARE_MAX_RECORDS:
                    # Need to split by stocks or time range
                    logger.info(
                        f"  Estimated {estimated_records} records exceeds Tushare limit ({TUSHARE_MAX_RECORDS}), "
                        f"splitting into smaller batches"
                    )
                    
                    # Calculate max stocks per batch based on estimated records per stock
                    records_per_stock = estimated_records / len(ts_codes) if ts_codes else 0
                    if records_per_stock > 0:
                        max_stocks_per_batch = max(1, int(TUSHARE_MAX_RECORDS / records_per_stock))
                    else:
                        max_stocks_per_batch = max_batch_stocks
                    
                    # Split stocks into batches
                    for i in range(0, len(ts_codes), max_stocks_per_batch):
                        batch_ts_codes = ts_codes[i:i + max_stocks_per_batch]
                        batch_items = group_items[i:i + max_stocks_per_batch]
                        
                        logger.info(
                            f"  Batch API call for {len(batch_ts_codes)} stocks ({self.kline_type}): "
                            f"{batch_ts_codes[0]} ... {batch_ts_codes[-1]}"
                        )
                        
                        batch_results = self._batch_sync_kline_by_type(
                            ts_codes=batch_ts_codes,
                            start_date=start_date,
                            end_date=end_date,
                            batch_size=batch_size,
                        )
                        results.update(batch_results)
                else:
                    # Can process all stocks in one batch
                    logger.info(
                        f"  Batch API call for {len(ts_codes)} stocks ({self.kline_type}): "
                        f"{ts_codes[0]} ... {ts_codes[-1]}"
                    )
                    
                    batch_results = self._batch_sync_kline_by_type(
                        ts_codes=ts_codes,
                        start_date=start_date,
                        end_date=end_date,
                        batch_size=batch_size,
                    )
                    results.update(batch_results)
            else:
                # Process individually (single stock)
                for item in group_items:
                    stats = self._sync_stock_kline(
                        ts_code=item["ts_code"],
                        start_date=item["start_date"],
                        end_date=item["end_date"],
                        batch_size=batch_size,
                    )
                    results[item["ts_code"]] = stats

        return results

    def _estimate_data_volume(
        self,
        ts_codes: List[str],
        start_date: str,
        end_date: str,
    ) -> int:
        """
        Estimate data volume for a batch request.

        Args:
            ts_codes: List of stock codes.
            start_date: Start date (YYYYMMDD).
            end_date: End date (YYYYMMDD).

        Returns:
            Estimated number of records.
        """
        try:
            start_dt = datetime.strptime(start_date, "%Y%m%d")
            end_dt = datetime.strptime(end_date, "%Y%m%d")
            days = (end_dt - start_dt).days + 1
            
            # Estimate records per stock based on kline type
            # Trading days ratio: approximately 70% of calendar days
            trading_days_ratio = 0.7
            
            records_per_stock_map = {
                "day": days * trading_days_ratio,  # ~1 record per trading day
                "week": days / 7,  # ~1 record per week
                "month": days / 30,  # ~1 record per month
                "quarter": days / 90,  # ~1 record per quarter
                "hour": days * trading_days_ratio * 4,  # ~4 records per trading day
                "30min": days * trading_days_ratio * 8,  # ~8 records per trading day
                "15min": days * trading_days_ratio * 16,  # ~16 records per trading day
                "5min": days * trading_days_ratio * 48,  # ~48 records per trading day
                "1min": days * trading_days_ratio * 240,  # ~240 records per trading day
            }
            
            records_per_stock = records_per_stock_map.get(self.kline_type, days)
            
            total_records = int(records_per_stock * len(ts_codes))
            return total_records
        except Exception as e:
            logger.warning(f"Failed to estimate data volume: {e}, using conservative estimate")
            # Conservative estimate: assume 1 record per day per stock
            return len(ts_codes) * 365

    def _batch_sync_kline_by_type(
        self,
        ts_codes: List[str],
        start_date: str,
        end_date: str,
        batch_size: int = 100,
    ) -> Dict[str, Dict[str, int]]:
        """
        Batch sync K-line data for multiple stocks using single API call.
        Supports all K-line types (day, week, month, quarter, hour, minutes).

        Args:
            ts_codes: List of stock codes.
            start_date: Start date (YYYYMMDD).
            end_date: End date (YYYYMMDD).
            batch_size: Batch size for saving to database.

        Returns:
            Dictionary mapping ts_code to statistics.
        """
        results = {ts_code: {"fetched": 0, "saved": 0, "errors": 0} for ts_code in ts_codes}
        
        try:
            logger.info(
                f"  Fetching {self.kline_type} K-line for {len(ts_codes)} stocks "
                f"({start_date} to {end_date})"
            )
            
            # Fetch all stocks' data based on kline type
            all_records = []
            
            if self.kline_type in ["day", "week", "month"]:
                # Use batch API with comma-separated ts_code
                ts_codes_str = ",".join(ts_codes)
                
                if self.kline_type == "day":
                    all_records = list(self.source.fetch_daily(
                        ts_code=ts_codes_str,
                        start_date=start_date,
                        end_date=end_date,
                    ))
                else:
                    # week or month - use generic fetch with api_name
                    api_name = KLINE_TYPE_MAP[self.kline_type]["tushare_api"]
                    all_records = list(self.source.fetch(
                        api_name=api_name,
                        ts_code=ts_codes_str,
                        start_date=start_date,
                        end_date=end_date,
                    ))
            else:
                # For other types (quarter, hour, minutes), use pro_bar
                # Note: pro_bar doesn't support batch ts_code, so we need to loop
                for ts_code in ts_codes:
                    try:
                        records = self._fetch_kline_records(ts_code, start_date, end_date)
                        all_records.extend(records)
                    except Exception as e:
                        logger.error(f"Error fetching {self.kline_type} data for {ts_code}: {e}")
                        results[ts_code]["errors"] += 1
            
            logger.info(f"  Fetched {len(all_records)} records from API")
            
            # Check if exceeds Tushare limit (should not happen if estimation is correct)
            if len(all_records) > TUSHARE_MAX_RECORDS:
                logger.warning(
                    f"  WARNING: Fetched {len(all_records)} records exceeds Tushare limit ({TUSHARE_MAX_RECORDS}). "
                    f"Some data may be missing. Consider splitting the request."
                )
            
            # Group records by ts_code
            records_by_stock = {}
            for record in all_records:
                ts_code = record.get("ts_code", "").strip()
                if ts_code and ts_code in ts_codes:
                    if ts_code not in records_by_stock:
                        records_by_stock[ts_code] = []
                    records_by_stock[ts_code].append(record)
                    results[ts_code]["fetched"] += 1
            
            # Process and save each stock's data
            for ts_code, stock_records in records_by_stock.items():
                stock_stats = self._process_and_save_records(
                    ts_code, stock_records, batch_size, end_date=end_date
                )
                results[ts_code].update(stock_stats)
                
                logger.info(
                    f"  Completed {ts_code}: Fetched={results[ts_code]['fetched']}, "
                    f"Saved={results[ts_code]['saved']}, Errors={results[ts_code]['errors']}"
                )
            
            # Handle stocks with no data
            for ts_code in ts_codes:
                if ts_code not in records_by_stock:
                    logger.warning(f"  No data returned for {ts_code}")
                    
        except Exception as e:
            logger.error(f"Error in batch sync: {e}", exc_info=True)
            # Mark all stocks as having errors
            for ts_code in ts_codes:
                results[ts_code]["errors"] += 1
        
        return results


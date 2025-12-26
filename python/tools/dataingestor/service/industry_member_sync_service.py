"""
Industry member synchronization service.

Synchronizes Shenwan industry member data from Tushare API.
Reference: https://tushare.pro/document/2?doc_id=335
"""

import logging
import sys
import time
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

# Ensure project path is in sys.path before importing atm modules
_project_root = Path(__file__).parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from atm.config import DatabaseConfig
from atm.data.source import TushareSource, TushareSourceConfig
from atm.models.stock import StockIndustryMember
from atm.repo import (
    BaseStateRepo,
    BaseTaskLock,
    DatabaseStateRepo,
    DatabaseTaskLock,
    FileStateRepo,
    FileTaskLock,
    IngestionState,
    StockBasicRepo,
    StockIndustryMemberRepo,
    TaskLockError,
)

logger = logging.getLogger(__name__)

# Tushare API rate limits
TUSHARE_RATE_LIMIT_PER_MINUTE = 150  # Maximum requests per minute
TUSHARE_REQUEST_INTERVAL = 0.5  # Minimum interval between requests (seconds)
TUSHARE_RETRY_DELAY = 60  # Delay when rate limit hit (seconds)
TUSHARE_MAX_RETRIES = 3  # Maximum retries for rate limit errors


class IndustryMemberSyncService:
    """Service for synchronizing industry member data."""

    def __init__(
        self,
        db_config: DatabaseConfig,
        tushare_token: str,
        state_repo: Optional[BaseStateRepo] = None,
        state_dir: Optional[str] = None,
    ):
        """
        Initialize industry member sync service.

        Args:
            db_config: Database configuration.
            tushare_token: Tushare Pro API token.
            state_repo: State repository for tracking sync progress.
            state_dir: Directory for file-based state storage.
        """
        self.db_config = db_config
        self.tushare_token = tushare_token
        self.state_repo = state_repo
        self.state_dir = state_dir

        # Initialize repositories
        self.member_repo = StockIndustryMemberRepo(db_config)
        self.stock_repo = StockBasicRepo(db_config)

        # Initialize Tushare source
        self.tushare_source = TushareSource(
            TushareSourceConfig(
                token=tushare_token,
                api_name="index_member_all",
            )
        )

        # Initialize task lock
        if isinstance(state_repo, DatabaseStateRepo):
            self.task_lock: BaseTaskLock = DatabaseTaskLock(state_repo)
        else:
            self.task_lock: BaseTaskLock = FileTaskLock(lock_dir=state_dir or "storage/locks")

        self._initialized = False

    def __enter__(self):
        """Context manager entry."""
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def initialize(self) -> None:
        """Initialize the service."""
        if self._initialized:
            return

        try:
            # TushareSource initializes automatically on first fetch
            # Just test connection to ensure it's ready
            if not self.tushare_source.test_connection():
                raise ConnectionError("Failed to connect to Tushare API")
            self._initialized = True
            logger.info("Industry member sync service initialized")
        except Exception as e:
            logger.error(f"Failed to initialize service: {e}")
            raise

    def close(self) -> None:
        """Close the service."""
        if self._initialized:
            # TushareSource doesn't need explicit close
            self._initialized = False
            logger.info("Industry member sync service closed")

    def sync_all_stocks(
        self,
        batch_size: int = 100,
        is_new: str = "Y",
    ) -> Dict[str, int]:
        """
        Synchronize industry members for all stocks.

        Args:
            batch_size: Batch size for saving.
            is_new: Whether to sync only latest data (Y) or all data (N).

        Returns:
            Dictionary with sync results.
        """
        task_name = "industry_member_all"
        lock_key = f"sync_{task_name}"

        # Acquire task lock
        try:
            if not self.task_lock.acquire(lock_key, timeout=3600):
                raise TaskLockError(f"Failed to acquire lock for {lock_key}")
        except TaskLockError as e:
            logger.error(f"Task lock error: {e}")
            raise

        try:
            results = {
                "total_fetched": 0,
                "total_saved": 0,
                "errors": 0,
            }

            # Get all stocks
            stocks = self.stock_repo.get_all()
            logger.info(f"Syncing industry members for {len(stocks)} stocks...")

            # Sync by stock (to avoid API limits)
            for i, stock in enumerate(stocks, 1):
                try:
                    stock_results = self._sync_stock(stock.ts_code, is_new, batch_size)
                    results["total_fetched"] += stock_results["fetched"]
                    results["total_saved"] += stock_results["saved"]
                    results["errors"] += stock_results["errors"]

                    # Rate limiting: sleep between requests
                    if i < len(stocks):  # Don't sleep after last stock
                        time.sleep(TUSHARE_REQUEST_INTERVAL)

                    if i % 100 == 0:
                        logger.info(f"Progress: {i}/{len(stocks)} stocks processed")

                except Exception as e:
                    logger.error(f"Failed to sync stock {stock.ts_code}: {e}")
                    results["errors"] += 1
                    # Sleep longer on error to avoid hitting rate limit
                    time.sleep(TUSHARE_REQUEST_INTERVAL * 2)

            logger.info(f"Sync completed: {results}")
            return results

        finally:
            # Release task lock
            try:
                self.task_lock.release(lock_key)
            except Exception as e:
                logger.warning(f"Error releasing lock: {e}")

    def sync_by_industry(
        self,
        l1_code: Optional[str] = None,
        l2_code: Optional[str] = None,
        l3_code: Optional[str] = None,
        batch_size: int = 100,
        is_new: str = "Y",
    ) -> Dict[str, int]:
        """
        Synchronize industry members by industry code.

        Args:
            l1_code: L1 industry code.
            l2_code: L2 industry code.
            l3_code: L3 industry code.
            batch_size: Batch size for saving.
            is_new: Whether to sync only latest data (Y) or all data (N).

        Returns:
            Dictionary with sync results.
        """
        task_name = "industry_member_by_industry"
        lock_key = f"sync_{task_name}"

        # Acquire task lock
        try:
            if not self.task_lock.acquire(lock_key, timeout=3600):
                raise TaskLockError(f"Failed to acquire lock for {lock_key}")
        except TaskLockError as e:
            logger.error(f"Task lock error: {e}")
            raise

        try:
            results = {"fetched": 0, "saved": 0, "errors": 0}

            # Build params
            params = {"is_new": is_new}
            if l1_code:
                params["l1_code"] = l1_code
            if l2_code:
                params["l2_code"] = l2_code
            if l3_code:
                params["l3_code"] = l3_code

            logger.info(f"Fetching industry members with params: {params}...")
            records = list(self.tushare_source.fetch(api_name="index_member_all", **params))
            results["fetched"] = len(records)

            if not records:
                logger.warning("No data returned")
                return results

            # Convert to models and save
            models = []
            for record in records:
                try:
                    model = self._record_to_model(record)
                    if model:
                        models.append(model)
                except Exception as e:
                    logger.warning(f"Failed to create model for record {record}: {e}")
                    results["errors"] += 1

            # Save in batches
            if models:
                logger.info(f"Saving {len(models)} industry members...")
                total_saved = 0
                for i in range(0, len(models), batch_size):
                    batch = models[i : i + batch_size]
                    saved = self.member_repo.save_batch_models(batch)
                    total_saved += saved
                results["saved"] = total_saved
                logger.info(f"Saved {total_saved} industry members")

            return results

        finally:
            # Release task lock
            try:
                self.task_lock.release(lock_key)
            except Exception as e:
                logger.warning(f"Error releasing lock: {e}")

    def _sync_stock(self, ts_code: str, is_new: str, batch_size: int) -> Dict[str, int]:
        """
        Synchronize industry members for a single stock with retry mechanism.

        Args:
            ts_code: Stock code.
            is_new: Whether to sync only latest data (Y) or all data (N).
            batch_size: Batch size for saving.

        Returns:
            Dictionary with sync results.
        """
        results = {"fetched": 0, "saved": 0, "errors": 0}

        # Retry mechanism for rate limit errors
        for retry in range(TUSHARE_MAX_RETRIES):
            try:
                # Fetch data from Tushare
                params = {
                    "ts_code": ts_code,
                    "is_new": is_new,
                }

                records = list(self.tushare_source.fetch(api_name="index_member_all", **params))
                results["fetched"] = len(records)

                if not records:
                    return results

                # Convert to models
                models = []
                for record in records:
                    try:
                        model = self._record_to_model(record)
                        if model:
                            models.append(model)
                    except Exception as e:
                        logger.warning(f"Failed to create model for record {record}: {e}")
                        results["errors"] += 1

                # Save in batches
                if models:
                    total_saved = 0
                    for i in range(0, len(models), batch_size):
                        batch = models[i : i + batch_size]
                        saved = self.member_repo.save_batch_models(batch)
                        total_saved += saved
                    results["saved"] = total_saved

                # Success, return results
                return results

            except Exception as e:
                error_msg = str(e)
                # Check if it's a rate limit error
                if "每分钟最多访问" in error_msg or "rate limit" in error_msg.lower():
                    if retry < TUSHARE_MAX_RETRIES - 1:
                        wait_time = TUSHARE_RETRY_DELAY * (retry + 1)
                        logger.warning(
                            f"Rate limit hit for {ts_code}, retrying in {wait_time} seconds "
                            f"(attempt {retry + 1}/{TUSHARE_MAX_RETRIES})"
                        )
                        time.sleep(wait_time)
                        continue
                    else:
                        logger.error(f"Rate limit error for {ts_code} after {TUSHARE_MAX_RETRIES} retries")
                        results["errors"] += 1
                        return results
                else:
                    # Other errors, don't retry
                    logger.error(f"Failed to sync stock {ts_code}: {e}")
                    results["errors"] += 1
                    return results

        return results

    def _record_to_model(self, record: Dict) -> Optional[StockIndustryMember]:
        """
        Convert API record to model.

        Args:
            record: API record dictionary.

        Returns:
            StockIndustryMember model or None if conversion fails.
        """
        try:
            # Parse dates
            in_date_str = record.get("in_date", "")
            out_date_str = record.get("out_date", "")

            in_date = None
            if in_date_str:
                in_date = pd.to_datetime(in_date_str, format="%Y%m%d").date()

            out_date = None
            if out_date_str:
                out_date = pd.to_datetime(out_date_str, format="%Y%m%d").date()

            if not in_date:
                logger.warning(f"Invalid in_date in record: {record}")
                return None

            model = StockIndustryMember(
                ts_code=record.get("ts_code", ""),
                l1_code=record.get("l1_code", ""),
                l1_name=record.get("l1_name", ""),
                l2_code=record.get("l2_code", ""),
                l2_name=record.get("l2_name", ""),
                l3_code=record.get("l3_code", ""),
                l3_name=record.get("l3_name", ""),
                stock_name=record.get("name"),
                in_date=in_date,
                out_date=out_date,
                is_new=record.get("is_new", "Y"),
                update_time=datetime.now(),
            )
            return model

        except Exception as e:
            logger.warning(f"Failed to convert record to model: {e}")
            return None

    def cleanup_failed_sync(self, task_name: str = "industry_member_all") -> Dict[str, int]:
        """
        Clean up data from a failed sync operation.

        This method can be used to remove partial data if a sync operation fails.
        Note: This is a destructive operation. Use with caution.

        Args:
            task_name: Task name to identify which sync to clean up.

        Returns:
            Dictionary with cleanup results.
        """
        results = {"deleted": 0, "errors": 0}

        try:
            # Get all records that were updated recently (within last hour)
            # This is a heuristic to identify records from a failed sync
            engine = self.member_repo._get_engine()
            table_name = self.member_repo._get_full_table_name()

            from sqlalchemy import text

            # Delete records updated in the last hour (assuming failed sync was recent)
            delete_sql = f"""
            DELETE FROM {table_name}
            WHERE update_time > NOW() - INTERVAL '1 hour'
            """

            with engine.begin() as conn:
                result = conn.execute(text(delete_sql))
                results["deleted"] = result.rowcount

            logger.info(f"Cleaned up {results['deleted']} records from failed sync")
            return results

        except Exception as e:
            logger.error(f"Failed to cleanup failed sync: {e}")
            results["errors"] += 1
            return results

    def cleanup_by_date_range(
        self, start_date: Optional[date] = None, end_date: Optional[date] = None
    ) -> Dict[str, int]:
        """
        Clean up data within a date range.

        Args:
            start_date: Start date (inclusive).
            end_date: End date (inclusive).

        Returns:
            Dictionary with cleanup results.
        """
        results = {"deleted": 0, "errors": 0}

        try:
            engine = self.member_repo._get_engine()
            table_name = self.member_repo._get_full_table_name()

            from sqlalchemy import text

            conditions = []
            params = {}

            if start_date:
                conditions.append("in_date >= :start_date")
                params["start_date"] = start_date.isoformat()

            if end_date:
                conditions.append("in_date <= :end_date")
                params["end_date"] = end_date.isoformat()

            if not conditions:
                logger.warning("No date range specified, skipping cleanup")
                return results

            where_clause = " AND ".join(conditions)
            delete_sql = f"DELETE FROM {table_name} WHERE {where_clause}"

            with engine.begin() as conn:
                result = conn.execute(text(delete_sql), params)
                results["deleted"] = result.rowcount

            logger.info(f"Cleaned up {results['deleted']} records in date range")
            return results

        except Exception as e:
            logger.error(f"Failed to cleanup by date range: {e}")
            results["errors"] += 1
            return results


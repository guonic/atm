"""
Industry classification synchronization service.

Synchronizes Shenwan industry classification data from Tushare API.
Reference: https://tushare.pro/document/2?doc_id=181
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

# Ensure project path is in sys.path before importing atm modules
_project_root = Path(__file__).parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from atm.config import DatabaseConfig
from atm.data.source import TushareSource, TushareSourceConfig
from atm.models.stock import StockIndustryClassify
from atm.repo import (
    BaseStateRepo,
    BaseTaskLock,
    DatabaseStateRepo,
    DatabaseTaskLock,
    FileTaskLock,
    StockIndustryClassifyRepo,
    TaskLockError,
)

logger = logging.getLogger(__name__)


class IndustryClassifySyncService:
    """Service for synchronizing industry classification data."""

    def __init__(
        self,
        db_config: DatabaseConfig,
        tushare_token: str,
        src: str = "SW2021",
        state_repo: Optional[BaseStateRepo] = None,
        state_dir: Optional[str] = None,
    ):
        """
        Initialize industry classification sync service.

        Args:
            db_config: Database configuration.
            tushare_token: Tushare Pro API token.
            src: Industry classification source version (SW2014/SW2021).
            state_repo: State repository for tracking sync progress.
            state_dir: Directory for file-based state storage.
        """
        self.db_config = db_config
        self.src = src
        self.tushare_token = tushare_token
        self.state_repo = state_repo
        self.state_dir = state_dir

        # Initialize repositories
        self.classify_repo = StockIndustryClassifyRepo(db_config)

        # Initialize Tushare source
        self.tushare_source = TushareSource(
            TushareSourceConfig(
                token=tushare_token,
                api_name="index_classify",
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
            logger.info("Industry classification sync service initialized")
        except Exception as e:
            logger.error(f"Failed to initialize service: {e}")
            raise

    def close(self) -> None:
        """Close the service."""
        if self._initialized:
            # TushareSource doesn't need explicit close
            self._initialized = False
            logger.info("Industry classification sync service closed")

    def sync_all_levels(self, batch_size: int = 100) -> Dict[str, int]:
        """
        Synchronize all industry classification levels.

        Args:
            batch_size: Batch size for saving.

        Returns:
            Dictionary with sync results.
        """
        task_name = f"industry_classify_{self.src}"
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

            # Delete existing data for this source version (full sync)
            logger.info(f"Deleting existing classifications for src={self.src}...")
            self.classify_repo.delete_by_src(self.src)

            # Sync each level
            for level in ["L1", "L2", "L3"]:
                logger.info(f"Syncing {level} level classifications...")
                level_results = self._sync_level(level, batch_size)
                results["total_fetched"] += level_results["fetched"]
                results["total_saved"] += level_results["saved"]
                results["errors"] += level_results["errors"]

            logger.info(f"Sync completed: {results}")
            return results

        finally:
            # Release task lock
            try:
                self.task_lock.release(lock_key)
            except Exception as e:
                logger.warning(f"Error releasing lock: {e}")

    def _sync_level(self, level: str, batch_size: int) -> Dict[str, int]:
        """
        Synchronize a specific level of industry classification.

        Args:
            level: Industry level (L1/L2/L3).
            batch_size: Batch size for saving.

        Returns:
            Dictionary with sync results.
        """
        results = {"fetched": 0, "saved": 0, "errors": 0}

        try:
            # Fetch data from Tushare
            params = {
                "level": level,
                "src": self.src,
            }

            logger.info(f"Fetching {level} level data from Tushare...")
            records = list(self.tushare_source.fetch(api_name="index_classify", **params))
            results["fetched"] = len(records)

            if not records:
                logger.warning(f"No data returned for {level} level")
                return results

            # Convert to models
            models = []
            for record in records:
                try:
                    model = StockIndustryClassify(
                        index_code=record.get("index_code", ""),
                        industry_name=record.get("industry_name", ""),
                        parent_code=record.get("parent_code", "0"),
                        level=record.get("level", level),
                        industry_code=record.get("industry_code"),
                        is_pub=record.get("is_pub"),
                        src=self.src,
                        update_time=datetime.now(),
                    )
                    models.append(model)
                except Exception as e:
                    logger.warning(f"Failed to create model for record {record}: {e}")
                    results["errors"] += 1

            # Save in batches
            if models:
                logger.info(f"Saving {len(models)} {level} level classifications...")
                total_saved = 0
                for i in range(0, len(models), batch_size):
                    batch = models[i : i + batch_size]
                    saved = self.classify_repo.save_batch_models(batch)
                    total_saved += saved
                results["saved"] = total_saved
                logger.info(f"Saved {total_saved} {level} level classifications")

        except Exception as e:
            logger.error(f"Failed to sync {level} level: {e}", exc_info=True)
            results["errors"] += 1

        return results


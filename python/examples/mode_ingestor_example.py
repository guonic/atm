"""
Example demonstrating upsert and append modes for data ingestion.

Shows how to use different ingestion modes and task locking.
"""

import logging
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from atm.config import DatabaseConfig
from atm.repo import DatabaseStateRepo, DatabaseTaskLock, FileTaskLock, TaskLockError
from tools.dataingestor import StockIngestorService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


def example_upsert_mode():
    """Example: Upsert mode (覆盖更新) - updates existing records."""
    token = os.getenv("TUSHARE_TOKEN", "")
    if not token:
        logger.error("Please set TUSHARE_TOKEN environment variable")
        return

    db_config = DatabaseConfig(
        host="localhost",
        port=5432,
        user="quant",
        password="quant123",
        database="quant_db",
        schema="quant",
    )

    with StockIngestorService(db_config=db_config, tushare_token=token) as ingestor:
        # Upsert mode: if record exists, update it; otherwise insert
        stats = ingestor.ingest_stock_basic(
            exchange="",
            list_status="L",
            batch_size=100,
            task_name="stock_basic_upsert",
            mode="upsert",  # 覆盖更新模式
        )
        logger.info(f"Upsert mode completed: {stats}")


def example_append_mode():
    """Example: Append mode (追加) - only inserts new records, fails on conflict."""
    token = os.getenv("TUSHARE_TOKEN", "")
    if not token:
        logger.error("Please set TUSHARE_TOKEN environment variable")
        return

    db_config = DatabaseConfig(
        host="localhost",
        port=5432,
        user="quant",
        password="quant123",
        database="quant_db",
        schema="quant",
    )

    with StockIngestorService(db_config=db_config, tushare_token=token) as ingestor:
        # Append mode: only insert new records, will fail if record already exists
        try:
            stats = ingestor.ingest_stock_basic(
                exchange="",
                list_status="L",
                batch_size=100,
                task_name="stock_basic_append",
                mode="append",  # 追加模式
            )
            logger.info(f"Append mode completed: {stats}")
        except Exception as e:
            logger.error(f"Append mode failed (expected if records exist): {e}")


def example_concurrent_task_prevention():
    """Example: Preventing concurrent execution of the same task."""
    token = os.getenv("TUSHARE_TOKEN", "")
    if not token:
        logger.error("Please set TUSHARE_TOKEN environment variable")
        return

    db_config = DatabaseConfig(
        host="localhost",
        port=5432,
        user="quant",
        password="quant123",
        database="quant_db",
        schema="quant",
    )

    # Create state repo and task lock
    state_repo = DatabaseStateRepo(db_config=db_config, schema="quant")
    task_lock = DatabaseTaskLock(state_repo)

    # First instance
    ingestor1 = StockIngestorService(
        db_config=db_config,
        tushare_token=token,
        state_repo=state_repo,
        task_lock=task_lock,
    )

    # Try to start the same task from another instance
    ingestor2 = StockIngestorService(
        db_config=db_config,
        tushare_token=token,
        state_repo=state_repo,
        task_lock=task_lock,
    )

    task_name = "stock_basic_concurrent_test"

    try:
        # Start task in first instance
        logger.info("Starting task in instance 1...")
        # Note: In real scenario, this would be in a separate thread/process
        # For demo, we'll just check the lock mechanism
        if task_lock.acquire(task_name):
            logger.info("Instance 1 acquired lock")

            # Try to acquire lock in second instance (should fail)
            if not task_lock.acquire(task_name, timeout=0):
                logger.info("Instance 2 failed to acquire lock (expected)")

            # Release lock from instance 1
            task_lock.release(task_name)
            logger.info("Instance 1 released lock")

            # Now instance 2 can acquire lock
            if task_lock.acquire(task_name):
                logger.info("Instance 2 acquired lock after instance 1 released")
                task_lock.release(task_name)

    except TaskLockError as e:
        logger.error(f"Task lock error: {e}")
    finally:
        ingestor1.close()
        ingestor2.close()


def example_file_based_lock():
    """Example: Using file-based task lock."""
    token = os.getenv("TUSHARE_TOKEN", "")
    if not token:
        logger.error("Please set TUSHARE_TOKEN environment variable")
        return

    db_config = DatabaseConfig(
        host="localhost",
        port=5432,
        user="quant",
        password="quant123",
        database="quant_db",
        schema="quant",
    )

    # Use file-based lock
    task_lock = FileTaskLock(lock_dir="storage/locks")

    with StockIngestorService(
        db_config=db_config,
        tushare_token=token,
        task_lock=task_lock,
    ) as ingestor:
        try:
            stats = ingestor.ingest_daily_kline(
                ts_code="000001.SZ",
                start_date="20240101",
                end_date="20240131",
                task_name="kline_file_lock_test",
                mode="upsert",
            )
            logger.info(f"File lock test completed: {stats}")
        except TaskLockError as e:
            logger.error(f"Task already running: {e}")


if __name__ == "__main__":
    print("Ingestion Mode Examples")
    print("=" * 50)
    print("\nNote: Set TUSHARE_TOKEN environment variable with your Tushare Pro token\n")

    # Uncomment the example you want to run:
    # example_upsert_mode()
    # example_append_mode()
    # example_concurrent_task_prevention()
    # example_file_based_lock()

    print("\nExamples are ready. Uncomment the function calls above to run them.")


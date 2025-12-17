"""
Example usage of StockIngestorService with state management and resume support.

Demonstrates checkpoint/resume functionality for data ingestion.
"""

import logging
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from atm.config import DatabaseConfig
from atm.repo import (
    DatabaseStateRepo,
    DatabaseTaskLock,
    FileStateRepo,
    FileTaskLock,
    TaskLockError,
)
from tools.dataingestor import StockIngestorService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


def example_with_file_state():
    """Example: Using file-based state storage."""
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

    # Create file-based state repository
    state_repo = FileStateRepo(state_dir="storage/state")

    # Create ingestor with state repository
    with StockIngestorService(
        db_config=db_config,
        tushare_token=token,
        state_repo=state_repo,
    ) as ingestor:
        # Ingest stock basic information (with checkpoint support and upsert mode)
        stats = ingestor.ingest_stock_basic(
            exchange="",
            list_status="L",
            batch_size=100,
            task_name="stock_basic_all",
            resume=True,  # Enable resume from checkpoint
            mode="upsert",  # Upsert mode (覆盖更新)
        )

        logger.info(f"Ingestion completed: {stats}")

        # Check task state
        state = ingestor.get_task_state("stock_basic_all")
        if state:
            logger.info(f"Task state: {state.status}, Last key: {state.last_processed_key}")


def example_with_database_state():
    """Example: Using database-based state storage."""
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

    # Create database-based state repository
    state_repo = DatabaseStateRepo(db_config=db_config, schema="quant")

    # Create ingestor with database state repository
    with StockIngestorService(
        db_config=db_config,
        tushare_token=token,
        state_repo=state_repo,
    ) as ingestor:
        # Ingest daily K-line data
        stats = ingestor.ingest_daily_kline(
            ts_code="000001.SZ",
            start_date="20240101",
            end_date="20240131",
            batch_size=100,
            task_name="kline_000001_202401",
            resume=True,
        )

        logger.info(f"K-line ingestion completed: {stats}")


def example_resume_failed_task():
    """Example: Resume a failed task."""
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
        # Check existing task states
        states = ingestor.list_task_states()
        logger.info(f"Found {len(states)} task states")

        for state in states:
            logger.info(
                f"Task: {state.task_name}, "
                f"Status: {state.status}, "
                f"Last key: {state.last_processed_key}, "
                f"Progress: {state.total_saved}/{state.total_fetched}"
            )

            # Resume failed or incomplete tasks
            if state.status in ["failed", "running"]:
                logger.info(f"Resuming task: {state.task_name}")
                # Extract task parameters from task_name or metadata
                # For example: "kline_day_000001.SZ_20240101_20240131"
                if state.task_name.startswith("kline_day_"):
                    parts = state.task_name.split("_")
                    if len(parts) >= 5:
                        ts_code = parts[2]
                        start_date = parts[3]
                        end_date = parts[4]

                        stats = ingestor.ingest_daily_kline(
                            ts_code=ts_code,
                            start_date=start_date,
                            end_date=end_date,
                            task_name=state.task_name,
                            resume=True,
                        )
                        logger.info(f"Resumed task completed: {stats}")


def example_reset_task():
    """Example: Reset a task state to start fresh."""
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
        # Reset a specific task
        task_name = "stock_basic_all"
        if ingestor.reset_task_state(task_name):
            logger.info(f"Task state reset: {task_name}")
        else:
            logger.warning(f"Failed to reset task state: {task_name}")


if __name__ == "__main__":
    print("State Management Examples")
    print("=" * 50)
    print("\nNote: Set TUSHARE_TOKEN environment variable with your Tushare Pro token\n")

    # Uncomment the example you want to run:
    # example_with_file_state()
    # example_with_database_state()
    # example_resume_failed_task()
    # example_reset_task()

    print("\nExamples are ready. Uncomment the function calls above to run them.")


#!/usr/bin/env python3
"""
Trading calendar synchronization task.

Synchronizes trading calendar data from Tushare to database using upsert mode.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sqlalchemy import text

from nq.config import DatabaseConfig, load_config
from nq.repo import DatabaseStateRepo, FileStateRepo, StockBasicRepo, TradingCalendarRepo
from tools.dataingestor import TradingCalendarIngestorService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


def main():
    """Main entry point for trading calendar sync task."""
    parser = argparse.ArgumentParser(
        description="Synchronize trading calendar data from Tushare to database"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/data_ingestor.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--exchange",
        type=str,
        default="",
        help="Exchange code (SSE/SZSE/BSE, empty for all)",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default="",
        help="Start date (YYYYMMDD, empty for all available)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default="",
        help="End date (YYYYMMDD, empty for all available)",
    )
    parser.add_argument(
        "--task-name",
        type=str,
        default=None,
        help="Task name for state tracking (default: auto-generated)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Batch size for saving records",
    )
    parser.add_argument(
        "--state-dir",
        type=str,
        default="storage/state",
        help="Directory for file-based state storage",
    )
    parser.add_argument(
        "--use-db-state",
        action="store_true",
        help="Use database for state storage instead of files",
    )
    parser.add_argument(
        "--tushare-token",
        type=str,
        default=None,
        help="Tushare Pro API token (overrides config and env var)",
    )

    args = parser.parse_args()

    # Load configuration
    try:
        config = load_config(args.config)
        db_config = config.database
    except Exception as e:
        logger.warning(f"Failed to load config from {args.config}: {e}")
        logger.info("Using default database configuration")
        db_config = DatabaseConfig(
            host=os.getenv("DB_HOST", "localhost"),
            port=int(os.getenv("DB_PORT", "5432")),
            user=os.getenv("DB_USER", "quant"),
            password=os.getenv("DB_PASSWORD", "quant123"),
            database=os.getenv("DB_NAME", "quant_db"),
            schema=os.getenv("DB_SCHEMA", "quant"),
        )

    # Get Tushare token
    tushare_token = args.tushare_token or os.getenv("TUSHARE_TOKEN", "")
    if not tushare_token:
        logger.error("Tushare token is required. Set TUSHARE_TOKEN environment variable or use --tushare-token")
        sys.exit(1)

    # Create state repository
    if args.use_db_state:
        state_repo = DatabaseStateRepo(db_config=db_config, schema=db_config.schema)
        logger.info("Using database for state storage")
    else:
        state_repo = FileStateRepo(state_dir=args.state_dir)
        logger.info(f"Using file-based state storage: {args.state_dir}")

    # Generate task name if not provided
    task_name = args.task_name
    if not task_name:
        exchange_suffix = f"_{args.exchange}" if args.exchange else "_all"
        start_suffix = f"_{args.start_date}" if args.start_date else ""
        end_suffix = f"_{args.end_date}" if args.end_date else ""
        task_name = f"trading_calendar_sync{exchange_suffix}{start_suffix}{end_suffix}"

    logger.info("=" * 60)
    logger.info("Trading Calendar Synchronization Task")
    logger.info("=" * 60)
    logger.info(f"Task Name: {task_name}")
    logger.info(f"Exchange: {args.exchange or 'All'}")
    logger.info(f"Start Date: {args.start_date or 'All available'}")
    logger.info(f"End Date: {args.end_date or 'All available'}")
    logger.info(f"Mode: upsert (覆盖更新)")
    logger.info(f"Batch Size: {args.batch_size}")
    logger.info("=" * 60)

    # Create ingestor service
    try:
        with TradingCalendarIngestorService(
            db_config=db_config,
            tushare_token=tushare_token,
            state_repo=state_repo,
        ) as ingestor:
            # Check if task is already running
            state = ingestor.get_task_state(task_name)
            if state and state.status == "running":
                logger.warning(f"Task {task_name} appears to be running. Attempting to acquire lock...")

            # Start synchronization
            stats = ingestor.ingest_trading_calendar(
                exchange=args.exchange,
                start_date=args.start_date,
                end_date=args.end_date,
                batch_size=args.batch_size,
                task_name=task_name,
                mode="upsert",  # 覆盖更新模式
            )

            # Print results
            logger.info("=" * 60)
            logger.info("Synchronization Completed")
            logger.info("=" * 60)
            logger.info(f"This Run - Fetched: {stats['fetched']}")
            logger.info(f"This Run - Saved: {stats['saved']}")
            logger.info(f"This Run - Errors: {stats['errors']}")
            logger.info("=" * 60)

            # Show final state with accumulated stats
            final_state = ingestor.get_task_state(task_name)
            if final_state:
                logger.info("=" * 60)
                logger.info("Task State Summary")
                logger.info("=" * 60)
                logger.info(f"Task Status: {final_state.status}")
                logger.info(f"Accumulated Total Fetched: {final_state.total_fetched}")
                logger.info(f"Accumulated Total Saved: {final_state.total_saved}")
                logger.info(f"Accumulated Total Errors: {final_state.total_errors}")
                logger.info(f"Last Processed Time: {final_state.last_processed_time or 'N/A'}")
                logger.info("=" * 60)

            # Query actual database record count
            try:
                calendar_repo = TradingCalendarRepo(db_config)
                with calendar_repo:
                    # Build query conditions
                    conditions = []
                    params = {}
                    if args.exchange:
                        conditions.append("exchange = :exchange")
                        params["exchange"] = args.exchange
                    if args.start_date:
                        conditions.append("cal_date >= :start_date")
                        params["start_date"] = args.start_date
                    if args.end_date:
                        conditions.append("cal_date <= :end_date")
                        params["end_date"] = args.end_date

                    where_clause = ""
                    if conditions:
                        where_clause = "WHERE " + " AND ".join(conditions)

                    engine = calendar_repo._get_engine()
                    table_name = calendar_repo._get_full_table_name()

                    count_query = text(f"SELECT COUNT(*) as count FROM {table_name} {where_clause}")
                    with engine.connect() as conn:
                        result = conn.execute(count_query, params)
                        row = result.fetchone()
                        actual_count = row[0] if row else 0

                    logger.info("=" * 60)
                    logger.info("Database Record Count")
                    logger.info("=" * 60)
                    logger.info(f"Actual records in database: {actual_count}")
                    if args.exchange:
                        logger.info(f"Filter: exchange = {args.exchange}")
                    if args.start_date:
                        logger.info(f"Filter: start_date >= {args.start_date}")
                    if args.end_date:
                        logger.info(f"Filter: end_date <= {args.end_date}")
                    logger.info("=" * 60)
            except Exception as e:
                logger.warning(f"Failed to query database record count: {e}")

    except KeyboardInterrupt:
        logger.warning("Task interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Synchronization failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()


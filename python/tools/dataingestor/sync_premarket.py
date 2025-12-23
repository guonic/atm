#!/usr/bin/env python3
"""
Stock premarket information synchronization task.

Synchronizes stock premarket information (股本情况盘前数据) from Tushare or AkShare to database.
Reference:
- Tushare: https://tushare.pro/document/2?doc_id=329
- AkShare: https://akshare.akfamily.xyz/
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from atm.config import DatabaseConfig, load_config
from atm.repo import DatabaseStateRepo, FileStateRepo
from tools.dataingestor.service.premarket_sync_service import PremarketSyncService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


def main():
    """Main entry point for premarket sync task."""
    parser = argparse.ArgumentParser(
        description="Synchronize stock premarket information from Tushare to database"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/data_ingestor.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--trade-date",
        type=str,
        default=None,
        help="Specific trading date (YYYYMMDD). If provided, only sync this date.",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Start date (YYYYMMDD). If provided with --end-date, sync date range.",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="End date (YYYYMMDD). If provided with --start-date, sync date range.",
    )
    parser.add_argument(
        "--ts-code",
        type=str,
        default=None,
        help="Stock code (e.g., '000001.SZ'). If provided, only sync this stock.",
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
        "--resume",
        action="store_true",
        default=True,
        help="Resume from last checkpoint (default: True)",
    )
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Do not resume from checkpoint, start fresh",
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
        "--source",
        type=str,
        default="akshare",
        choices=["tushare", "akshare"],
        help="Data source type: 'tushare' or 'akshare' (default: akshare)",
    )
    parser.add_argument(
        "--tushare-token",
        type=str,
        default=None,
        help="Tushare Pro API token (required if --source is 'tushare', overrides config and env var)",
    )

    args = parser.parse_args()

    # Validate date arguments
    if args.trade_date and (args.start_date or args.end_date):
        logger.error("Cannot specify both --trade-date and --start-date/--end-date")
        sys.exit(1)

    if (args.start_date and not args.end_date) or (args.end_date and not args.start_date):
        logger.error("Both --start-date and --end-date must be provided together")
        sys.exit(1)

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

    # Get Tushare token (only required if source is tushare)
    tushare_token = None
    if args.source == "tushare":
        tushare_token = args.tushare_token or os.getenv("TUSHARE_TOKEN", "")
        if not tushare_token:
            logger.error("Tushare token is required when --source is 'tushare'. Set TUSHARE_TOKEN environment variable or use --tushare-token")
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
        if args.trade_date:
            task_name = f"premarket_sync_{args.trade_date}"
        elif args.start_date and args.end_date:
            task_name = f"premarket_sync_{args.start_date}_{args.end_date}"
        elif args.ts_code:
            task_name = f"premarket_sync_{args.ts_code}"
        else:
            task_name = "premarket_sync_all"

    logger.info("=" * 80)
    logger.info("Stock Premarket Information Synchronization Task")
    logger.info("=" * 80)
    logger.info(f"Task Name: {task_name}")
    logger.info(f"Data Source: {args.source.upper()}")
    logger.info(f"Trade Date: {args.trade_date or 'N/A'}")
    logger.info(f"Date Range: {args.start_date or 'N/A'} to {args.end_date or 'N/A'}")
    logger.info(f"Stock Code: {args.ts_code or 'All'}")
    logger.info(f"Mode: upsert (覆盖更新)")
    logger.info(f"Batch Size: {args.batch_size}")
    logger.info(f"Resume: {args.resume}")
    logger.info("=" * 80)

    # Create sync service
    try:
        with PremarketSyncService(
            db_config=db_config,
            tushare_token=tushare_token,
            source_type=args.source,
            state_repo=state_repo,
        ) as service:
            # Start synchronization
            stats = service.sync_premarket(
                trade_date=args.trade_date,
                start_date=args.start_date,
                end_date=args.end_date,
                ts_code=args.ts_code,
                batch_size=args.batch_size,
                task_name=task_name,
                resume=args.resume,
                mode="upsert",
            )

            # Print results
            logger.info("=" * 80)
            logger.info("Synchronization Completed")
            logger.info("=" * 80)
            logger.info(f"This Run - Fetched: {stats['fetched']}")
            logger.info(f"This Run - Saved: {stats['saved']}")
            logger.info(f"This Run - Errors: {stats['errors']}")
            logger.info("=" * 80)

    except KeyboardInterrupt:
        logger.warning("Task interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Synchronization failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()


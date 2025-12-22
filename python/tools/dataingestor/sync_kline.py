#!/usr/bin/env python3
"""
K-line data synchronization task.

Synchronizes K-line data for all stocks, checking existing data and syncing from the appropriate start date.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from atm.config import load_config
from atm.repo import DatabaseStateRepo, FileStateRepo
from tools.dataingestor.service.kline_sync_service import KlineSyncService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Synchronize K-line data for all stocks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Sync daily K-line data for all stocks
  python sync_kline.py --type day

  # Sync weekly K-line data for SSE stocks only
  python sync_kline.py --type week --exchange SSE

  # Sync daily K-line data up to a specific date
  python sync_kline.py --type day --end-date 20241231

Supported K-line types:
  day, week, month, quarter, hour, 30min, 15min, 5min, 1min
        """,
    )

    parser.add_argument(
        "--type",
        required=True,
        choices=["day", "week", "month", "quarter", "hour", "30min", "15min", "5min", "1min"],
        help="K-line type to sync",
    )
    parser.add_argument(
        "--exchange",
        default="",
        help="Exchange code filter (SSE/SZSE/BSE, empty for all)",
    )
    parser.add_argument(
        "--list-status",
        default="L",
        help="List status filter (L=listed, D=delisted, P=pause, empty for all)",
    )
    parser.add_argument(
        "--end-date",
        default=None,
        help="End date (YYYYMMDD, defaults to today)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Batch size for saving (default: 100)",
    )
    parser.add_argument(
        "--config",
        default="config/data_ingestor.yaml",
        help="Configuration file path (default: config/data_ingestor.yaml)",
    )
    parser.add_argument(
        "--state-dir",
        default="storage/state",
        help="Directory for file-based state storage (default: storage/state)",
    )
    parser.add_argument(
        "--use-db-state",
        action="store_true",
        help="Use database-based state storage instead of file-based",
    )

    args = parser.parse_args()

    # Load configuration
    try:
        config = load_config(args.config)
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        sys.exit(1)

    # Get Tushare token from environment or config
    # Note: config.source might not exist in the Config model, so we check it safely
    tushare_token = os.getenv("TUSHARE_TOKEN") or ""
    if not tushare_token:
        logger.error("TUSHARE_TOKEN environment variable is required")
        sys.exit(1)

    # Get database config (already a DatabaseConfig object)
    db_config = config.database

    # Create state repository
    if args.use_db_state:
        state_repo = DatabaseStateRepo(db_config)
    else:
        state_repo = FileStateRepo(state_dir=args.state_dir)

    # Create K-line sync service
    try:
        with KlineSyncService(
            db_config=db_config,
            tushare_token=tushare_token,
            kline_type=args.type,
            state_repo=state_repo,
            state_dir=args.state_dir,
        ) as service:
            # Sync all stocks
            results = service.sync_all_stocks(
                exchange=args.exchange,
                list_status=args.list_status,
                batch_size=args.batch_size,
                end_date=args.end_date,
            )

            # Print summary
            total_fetched = sum(r.get("fetched", 0) for r in results.values())
            total_saved = sum(r.get("saved", 0) for r in results.values())
            total_errors = sum(r.get("errors", 0) for r in results.values())

            logger.info("=" * 80)
            logger.info("Synchronization Summary")
            logger.info("=" * 80)
            logger.info(f"K-line type: {args.type}")
            logger.info(f"Total stocks processed: {len(results)}")
            logger.info(f"Total records fetched: {total_fetched}")
            logger.info(f"Total records saved: {total_saved}")
            logger.info(f"Total errors: {total_errors}")
            logger.info("=" * 80)

    except Exception as e:
        logger.error(f"K-line synchronization failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()


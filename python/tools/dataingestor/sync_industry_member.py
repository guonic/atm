#!/usr/bin/env python3
"""
Industry member synchronization task.

Synchronizes Shenwan industry member data from Tushare API.
Reference: https://tushare.pro/document/2?doc_id=335
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from nq.config import load_config
from nq.repo import DatabaseStateRepo, FileStateRepo
from tools.dataingestor.service.industry_member_sync_service import IndustryMemberSyncService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Synchronize Shenwan industry member data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Sync all stocks' industry members
  python sync_industry_member.py

  # Sync by L3 industry code
  python sync_industry_member.py --l3-code 850531.SI

  # Sync by L2 industry code
  python sync_industry_member.py --l2-code 801053.SI

  # Sync by L1 industry code
  python sync_industry_member.py --l1-code 801050.SI

  # Sync all historical data (not just latest)
  python sync_industry_member.py --is-new N

  # Use database state storage
  python sync_industry_member.py --use-db-state
        """,
    )

    parser.add_argument(
        "--l1-code",
        type=str,
        help="L1 industry code (e.g., 801050.SI)",
    )
    parser.add_argument(
        "--l2-code",
        type=str,
        help="L2 industry code (e.g., 801053.SI)",
    )
    parser.add_argument(
        "--l3-code",
        type=str,
        help="L3 industry code (e.g., 850531.SI)",
    )
    parser.add_argument(
        "--is-new",
        type=str,
        default="Y",
        choices=["Y", "N"],
        help="Whether to sync only latest data (Y) or all data (N, default: Y)",
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
    parser.add_argument(
        "--cleanup-failed",
        action="store_true",
        help="Clean up data from failed sync operations (removes records updated in last hour)",
    )
    parser.add_argument(
        "--cleanup-date-start",
        type=str,
        help="Start date for cleanup (YYYY-MM-DD, inclusive)",
    )
    parser.add_argument(
        "--cleanup-date-end",
        type=str,
        help="End date for cleanup (YYYY-MM-DD, inclusive)",
    )

    args = parser.parse_args()

    # Load configuration
    try:
        config = load_config(args.config)
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        sys.exit(1)

    # Get Tushare token from environment or config
    tushare_token = os.getenv("TUSHARE_TOKEN") or ""
    if not tushare_token:
        logger.error("TUSHARE_TOKEN environment variable is required")
        sys.exit(1)

    # Get database config
    db_config = config.database

    # Create state repository
    if args.use_db_state:
        state_repo = DatabaseStateRepo(db_config)
    else:
        state_repo = FileStateRepo(state_dir=args.state_dir)

    # Create sync service
    try:
        with IndustryMemberSyncService(
            db_config=db_config,
            tushare_token=tushare_token,
            state_repo=state_repo,
            state_dir=args.state_dir,
        ) as service:
            # Handle cleanup operations
            if args.cleanup_failed:
                logger.info("Cleaning up data from failed sync operations...")
                cleanup_results = service.cleanup_failed_sync()
                logger.info(f"Cleanup completed: {cleanup_results}")
                return

            if args.cleanup_date_start or args.cleanup_date_end:
                from datetime import datetime as dt

                start_date = None
                end_date = None

                if args.cleanup_date_start:
                    start_date = dt.strptime(args.cleanup_date_start, "%Y-%m-%d").date()
                if args.cleanup_date_end:
                    end_date = dt.strptime(args.cleanup_date_end, "%Y-%m-%d").date()

                logger.info(f"Cleaning up data in date range: {start_date} to {end_date}...")
                cleanup_results = service.cleanup_by_date_range(start_date, end_date)
                logger.info(f"Cleanup completed: {cleanup_results}")
                return
            # Sync based on parameters
            if args.l1_code or args.l2_code or args.l3_code:
                # Sync by industry code
                results = service.sync_by_industry(
                    l1_code=args.l1_code,
                    l2_code=args.l2_code,
                    l3_code=args.l3_code,
                    batch_size=args.batch_size,
                    is_new=args.is_new,
                )
            else:
                # Sync all stocks
                results = service.sync_all_stocks(
                    batch_size=args.batch_size,
                    is_new=args.is_new,
                )

            # Print summary
            logger.info("=" * 80)
            logger.info("Synchronization Summary")
            logger.info("=" * 80)
            if args.l1_code:
                logger.info(f"L1 industry code: {args.l1_code}")
            if args.l2_code:
                logger.info(f"L2 industry code: {args.l2_code}")
            if args.l3_code:
                logger.info(f"L3 industry code: {args.l3_code}")
            logger.info(f"Is new: {args.is_new}")
            logger.info(f"Total records fetched: {results.get('total_fetched', results.get('fetched', 0))}")
            logger.info(f"Total records saved: {results.get('total_saved', results.get('saved', 0))}")
            logger.info(f"Total errors: {results.get('errors', 0)}")
            logger.info("=" * 80)

    except Exception as e:
        logger.error(f"Industry member synchronization failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()


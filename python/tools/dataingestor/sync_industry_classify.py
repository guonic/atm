#!/usr/bin/env python3
"""
Industry classification synchronization task.

Synchronizes Shenwan industry classification data from Tushare API.
Reference: https://tushare.pro/document/2?doc_id=181
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
from tools.dataingestor.service.industry_classify_sync_service import IndustryClassifySyncService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Synchronize Shenwan industry classification data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Sync SW2021 industry classification
  python sync_industry_classify.py --src SW2021

  # Sync SW2014 industry classification
  python sync_industry_classify.py --src SW2014

  # Use database state storage
  python sync_industry_classify.py --src SW2021 --use-db-state
        """,
    )

    parser.add_argument(
        "--src",
        type=str,
        default="SW2021",
        choices=["SW2014", "SW2021"],
        help="Industry classification source version (SW2014/SW2021, default: SW2021)",
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
        with IndustryClassifySyncService(
            db_config=db_config,
            tushare_token=tushare_token,
            src=args.src,
            state_repo=state_repo,
            state_dir=args.state_dir,
        ) as service:
            # Sync all levels
            results = service.sync_all_levels(batch_size=args.batch_size)

            # Print summary
            logger.info("=" * 80)
            logger.info("Synchronization Summary")
            logger.info("=" * 80)
            logger.info(f"Source version: {args.src}")
            logger.info(f"Total records fetched: {results['total_fetched']}")
            logger.info(f"Total records saved: {results['total_saved']}")
            logger.info(f"Total errors: {results['errors']}")
            logger.info("=" * 80)

    except Exception as e:
        logger.error(f"Industry classification synchronization failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()


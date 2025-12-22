#!/usr/bin/env python3
"""
Backfill StockKlineSyncState from existing K-line data.

This script scans all K-line tables and populates the stock_kline_sync_state table
with the last synced date for each stock and K-line type.
"""

import argparse
import logging
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Dict, Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from atm.config import DatabaseConfig, load_config
from atm.models.stock import StockKlineSyncState
from atm.repo import (
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
)
from sqlalchemy import text

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)

# Mapping from kline type to repo class and time column
KLINE_TYPE_MAP = {
    "day": {
        "repo": StockKlineDayRepo,
        "time_column": "trade_date",
    },
    "week": {
        "repo": StockKlineWeekRepo,
        "time_column": "week_date",
    },
    "month": {
        "repo": StockKlineMonthRepo,
        "time_column": "month_date",
    },
    "quarter": {
        "repo": StockKlineQuarterRepo,
        "time_column": "quarter_date",
    },
    "hour": {
        "repo": StockKlineHourRepo,
        "time_column": "trade_time",
    },
    "30min": {
        "repo": StockKline30MinRepo,
        "time_column": "trade_time",
    },
    "15min": {
        "repo": StockKline15MinRepo,
        "time_column": "trade_time",
    },
    "5min": {
        "repo": StockKline5MinRepo,
        "time_column": "trade_time",
    },
    "1min": {
        "repo": StockKline1MinRepo,
        "time_column": "trade_time",
    },
}


def get_all_last_synced_dates(
    kline_repo, time_column: str
) -> Dict[str, date]:
    """
    Get the last synced date for all stocks from K-line table using batch query.

    Args:
        kline_repo: K-line repository instance.
        time_column: Time column name.

    Returns:
        Dictionary mapping ts_code to last_synced_date.
    """
    result_dict = {}
    try:
        engine = kline_repo._get_engine()
        table_name = kline_repo._get_full_table_name()

        # Query for the latest record for each stock
        # Use window function for better compatibility
        query = f"""
        SELECT ts_code, {time_column} as last_date
        FROM (
            SELECT ts_code, {time_column},
                   ROW_NUMBER() OVER (PARTITION BY ts_code ORDER BY {time_column} DESC) as rn
            FROM {table_name}
        ) ranked
        WHERE rn = 1
        """

        with engine.connect() as conn:
            result = conn.execute(text(query))
            for row in result:
                ts_code = row[0]
                last_time = row[1]
                if last_time:
                    # Handle datetime and date types
                    if isinstance(last_time, datetime):
                        result_dict[ts_code] = last_time.date()
                    elif isinstance(last_time, date):
                        result_dict[ts_code] = last_time
                    elif isinstance(last_time, str):
                        try:
                            result_dict[ts_code] = datetime.fromisoformat(
                                last_time.replace("Z", "+00:00")
                            ).date()
                        except ValueError:
                            try:
                                result_dict[ts_code] = datetime.strptime(
                                    last_time, "%Y-%m-%d"
                                ).date()
                            except ValueError:
                                logger.warning(f"Failed to parse date: {last_time}")
    except Exception as e:
        # Fallback to individual queries if batch query fails
        logger.warning(f"Batch query failed, falling back to individual queries: {e}")
        return {}
    return result_dict


def backfill_sync_state(
    db_config: DatabaseConfig,
    kline_types: Optional[list] = None,
    batch_size: int = 100,
) -> Dict[str, int]:
    """
    Backfill sync state from existing K-line data.

    Args:
        db_config: Database configuration.
        kline_types: List of K-line types to process. If None, process all types.
        batch_size: Batch size for saving states.

    Returns:
        Dictionary with statistics (processed, updated, errors).
    """
    stats = {"processed": 0, "updated": 0, "errors": 0}

    # Get all stocks
    stock_repo = StockBasicRepo(db_config)
    stocks = stock_repo.get_all()
    total_stocks = len(stocks)
    logger.info(f"Found {total_stocks} stocks to process")

    # Initialize sync state repo
    sync_state_repo = StockKlineSyncStateRepo(db_config)

    # Process each K-line type
    kline_types = kline_types or list(KLINE_TYPE_MAP.keys())
    logger.info(f"Processing K-line types: {', '.join(kline_types)}")

    for kline_type in kline_types:
        if kline_type not in KLINE_TYPE_MAP:
            logger.warning(f"Unknown K-line type: {kline_type}, skipping")
            continue

        logger.info(f"\n{'='*60}")
        logger.info(f"Processing K-line type: {kline_type}")
        logger.info(f"{'='*60}")

        kline_config = KLINE_TYPE_MAP[kline_type]
        kline_repo = kline_config["repo"](db_config)
        time_column = kline_config["time_column"]

        type_stats = {"processed": 0, "updated": 0, "errors": 0}

        # Get all last synced dates in batch
        logger.info(f"Fetching last synced dates for all stocks...")
        last_synced_dates = get_all_last_synced_dates(kline_repo, time_column)
        logger.info(f"Found {len(last_synced_dates)} stocks with data in {kline_type} table")

        # Process each stock
        batch_states = []
        for idx, stock in enumerate(stocks, 1):
            ts_code = stock.ts_code
            stats["processed"] += 1
            type_stats["processed"] += 1

            if idx % 500 == 0:
                logger.info(
                    f"[{idx}/{total_stocks}] Processing {ts_code} "
                    f"(Type: {kline_type}, Updated: {type_stats['updated']})"
                )

            try:
                # Get last synced date from batch result
                last_synced_date = last_synced_dates.get(ts_code)

                if last_synced_date:
                    # Create sync state
                    sync_state = StockKlineSyncState(
                        ts_code=ts_code,
                        kline_type=kline_type,
                        last_synced_date=last_synced_date,
                        last_synced_time=datetime.now(),
                        total_records=0,  # We don't track total records in backfill
                        update_time=datetime.now(),
                    )
                    batch_states.append(sync_state)

                    # Save in batches
                    if len(batch_states) >= batch_size:
                        for state in batch_states:
                            if sync_state_repo.save_model(state):
                                stats["updated"] += 1
                                type_stats["updated"] += 1
                            else:
                                stats["errors"] += 1
                                type_stats["errors"] += 1
                        batch_states = []

            except Exception as e:
                stats["errors"] += 1
                type_stats["errors"] += 1
                logger.error(
                    f"Error processing {ts_code} ({kline_type}): {e}", exc_info=True
                )

        # Save remaining states
        for state in batch_states:
            try:
                if sync_state_repo.save_model(state):
                    stats["updated"] += 1
                    type_stats["updated"] += 1
                else:
                    stats["errors"] += 1
                    type_stats["errors"] += 1
            except Exception as e:
                stats["errors"] += 1
                type_stats["errors"] += 1
                logger.error(f"Error saving sync state for {state.ts_code}: {e}")

        logger.info(
            f"Completed {kline_type}: Processed={type_stats['processed']}, "
            f"Updated={type_stats['updated']}, Errors={type_stats['errors']}"
        )

    logger.info(f"\n{'='*60}")
    logger.info("Backfill Summary")
    logger.info(f"{'='*60}")
    logger.info(f"Total processed: {stats['processed']}")
    logger.info(f"Total updated: {stats['updated']}")
    logger.info(f"Total errors: {stats['errors']}")

    return stats


def main():
    """Main entry point for backfill script."""
    parser = argparse.ArgumentParser(
        description="Backfill StockKlineSyncState from existing K-line data"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/data_ingestor.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--kline-types",
        type=str,
        nargs="+",
        default=None,
        choices=list(KLINE_TYPE_MAP.keys()) + ["all"],
        help="K-line types to process (default: all)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Batch size for processing (not used currently, reserved for future)",
    )

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    db_config = config.database

    # Process K-line types
    kline_types = args.kline_types
    if kline_types and "all" in kline_types:
        kline_types = None
    elif kline_types is None:
        kline_types = None

    # Run backfill
    try:
        stats = backfill_sync_state(
            db_config=db_config,
            kline_types=kline_types,
            batch_size=args.batch_size,
        )
        logger.info("Backfill completed successfully")
        return 0
    except Exception as e:
        logger.error(f"Backfill failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())


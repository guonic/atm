#!/usr/bin/env python3
"""
Get CSI index stock lists (CSI100, CSI300, CSI500).

Supports fetching stock lists from Tushare API or database.
Can export individual index lists or a combined list.
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set

# Add project root to path
_project_root = Path(__file__).parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# Add python directory to path (where nq package is located)
_python_dir = Path(__file__).parent.parent
if str(_python_dir) not in sys.path:
    sys.path.insert(0, str(_python_dir))

from nq.config import DatabaseConfig, load_config
from nq.data.source import TushareSource, TushareSourceConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Index code mapping
INDEX_CODES = {
    "csi100": "000903.SH",  # CSI100 (中证100)
    "csi300": "000300.SH",  # CSI300 (沪深300)
    "csi500": "000905.SH",  # CSI500 (中证500)
}


def get_index_stocks_from_tushare(
    tushare_token: str, index_name: str, trade_date: Optional[str] = None
) -> List[str]:
    """
    Get index component stocks from Tushare API.

    Args:
        tushare_token: Tushare Pro API token.
        index_name: Index name (csi100, csi300, or csi500).
        trade_date: Trade date (YYYYMMDD), if None then get latest data.

    Returns:
        List of stock codes (format: 000001.SZ).
    """
    if index_name not in INDEX_CODES:
        logger.error(f"Unknown index name: {index_name}. Supported: {list(INDEX_CODES.keys())}")
        return []

    index_code = INDEX_CODES[index_name]
    config = TushareSourceConfig(
        token=tushare_token,
        type="tushare",
    )
    source = TushareSource(config)

    try:
        params = {
            "index_code": index_code,
        }
        if trade_date:
            params["trade_date"] = trade_date

        logger.info(
            f"Fetching {index_name.upper()} stocks from Tushare "
            f"(index_code={index_code}, trade_date={trade_date or 'latest'})"
        )

        stocks = []

        # Method 1: Use index_weight API to get index component weights (recommended)
        try:
            logger.info("Trying index_weight API...")
            records = list(source.fetch(api_name="index_weight", **params))
            if records:
                stocks = []
                for record in records:
                    code = record.get("con_code") or record.get("code") or record.get("ts_code")
                    if code:
                        stocks.append(code)

                if stocks:
                    stocks = sorted(list(set(stocks)))
                    logger.info(f"✓ Fetched {len(stocks)} {index_name.upper()} stocks from index_weight API")
                    return stocks
        except Exception as e:
            logger.warning(f"index_weight API failed: {e}")

        logger.warning("index_weight not available, trying alternative methods...")

        # Method 2: If no date specified, try to get recent trade date
        if not trade_date:
            try:
                cal_records = list(
                    source.fetch(api_name="trade_cal", exchange="SSE", is_open="1", end_date="20241231")
                )
                if cal_records:
                    recent_dates = sorted(
                        [r.get("cal_date") for r in cal_records if r.get("cal_date")], reverse=True
                    )
                    if recent_dates:
                        trade_date = recent_dates[0]
                        logger.info(f"Using recent trade date: {trade_date}")
                        params["trade_date"] = trade_date

                        records = list(source.fetch(api_name="index_weight", **params))
                        if records:
                            stocks = []
                            for record in records:
                                code = record.get("con_code") or record.get("code") or record.get("ts_code")
                                if code:
                                    stocks.append(code)

                            if stocks:
                                stocks = sorted(list(set(stocks)))
                                logger.info(
                                    f"✓ Fetched {len(stocks)} {index_name.upper()} stocks using recent trade date"
                                )
                                return stocks
            except Exception as e:
                logger.warning(f"Failed to get recent trade date: {e}")

        if not stocks:
            logger.error(f"Failed to fetch {index_name.upper()} stocks from Tushare API")
            logger.info("Please check:")
            logger.info(f"1. Tushare token is valid and has access to index_weight API")
            logger.info(f"2. Index code is correct: {index_code}")
            logger.info("3. Trade date format is correct: YYYYMMDD")
            return []

        return stocks

    except Exception as e:
        logger.error(f"Error fetching {index_name.upper()} stocks from Tushare: {e}", exc_info=True)
        return []


def get_index_stocks_from_database(
    db_config: DatabaseConfig, index_name: str
) -> List[str]:
    """
    Get index component stocks from database.

    Note: This requires index component data to be already in the database.
    If not, you need to sync from Tushare API first.

    Args:
        db_config: Database configuration.
        index_name: Index name (csi100, csi300, or csi500).

    Returns:
        List of stock codes.
    """
    from nq.repo.stock_repo import StockBasicRepo

    repo = StockBasicRepo(db_config)

    # TODO: Implement database query based on actual schema
    # If there's an index component table, query from there
    # Otherwise, may need to get from other data sources

    logger.warning("Database method not fully implemented. Please use Tushare API method.")
    return []


def save_to_file(stocks: List[str], output_file: Path) -> None:
    """
    Save stock code list to file.

    Args:
        stocks: List of stock codes.
        output_file: Output file path.
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        for stock in sorted(stocks):
            f.write(f"{stock}\n")

    logger.info(f"Saved {len(stocks)} stocks to {output_file}")


def combine_stock_lists(stock_lists: Dict[str, List[str]]) -> List[str]:
    """
    Combine multiple stock lists into one unique list.

    Args:
        stock_lists: Dictionary mapping index names to stock lists.

    Returns:
        Combined unique stock list.
    """
    all_stocks: Set[str] = set()
    for index_name, stocks in stock_lists.items():
        all_stocks.update(stocks)
        logger.info(f"{index_name.upper()}: {len(stocks)} stocks")

    combined = sorted(list(all_stocks))
    logger.info(f"Combined total: {len(combined)} unique stocks")
    return combined


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Get CSI index stock lists (CSI100, CSI300, CSI500)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Get CSI300 stocks
  python python/tools/get_csi300_stocks.py --index csi300

  # Get all indices (CSI100, CSI300, CSI500) and combined list
  python python/tools/get_csi300_stocks.py --all --output-dir ./stock_lists

  # Get specific indices
  python python/tools/get_csi300_stocks.py --index csi100 csi500 --output-dir ./stock_lists
        """,
    )
    parser.add_argument(
        "--index",
        nargs="+",
        choices=["csi100", "csi300", "csi500"],
        default=["csi300"],
        help="Index names to fetch (default: csi300). Can specify multiple: --index csi100 csi300 csi500",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Fetch all indices (CSI100, CSI300, CSI500) and generate combined list",
    )
    parser.add_argument(
        "--source",
        choices=["tushare", "database"],
        default="tushare",
        help="Data source: tushare or database (default: tushare)",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=os.getenv("TUSHARE_TOKEN", ""),
        help="Tushare Pro API token (can also be set via TUSHARE_TOKEN environment variable)",
    )
    parser.add_argument(
        "--trade-date",
        type=str,
        default=None,
        help="Trade date (YYYYMMDD), if None then get latest data",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for stock list files (if not specified, print to console)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Config file path (for database method)",
    )

    args = parser.parse_args()

    # Determine which indices to fetch
    if args.all:
        indices_to_fetch = ["csi100", "csi300", "csi500"]
    else:
        indices_to_fetch = args.index

    # Fetch stocks for each index
    all_stock_lists: Dict[str, List[str]] = {}

    if args.source == "tushare":
        if not args.token:
            logger.error(
                "Tushare token is required. Set TUSHARE_TOKEN environment variable or use --token option."
            )
            sys.exit(1)

        for index_name in indices_to_fetch:
            logger.info(f"Fetching {index_name.upper()} stocks...")
            stocks = get_index_stocks_from_tushare(args.token, index_name, args.trade_date)
            if stocks:
                all_stock_lists[index_name] = stocks
            else:
                logger.warning(f"Failed to fetch {index_name.upper()} stocks")

    elif args.source == "database":
        config = load_config(args.config) if args.config else load_config()
        for index_name in indices_to_fetch:
            logger.info(f"Fetching {index_name.upper()} stocks from database...")
            stocks = get_index_stocks_from_database(config.database, index_name)
            if stocks:
                all_stock_lists[index_name] = stocks
            else:
                logger.warning(f"Failed to fetch {index_name.upper()} stocks from database")

    if not all_stock_lists:
        logger.error("No stocks found. Please check your data source and parameters.")
        sys.exit(1)

    # Output results
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save individual index lists
        for index_name, stocks in all_stock_lists.items():
            output_file = output_dir / f"{index_name}.txt"
            save_to_file(stocks, output_file)
            print(f"✓ Saved {len(stocks)} {index_name.upper()} stocks to {output_file}")

        # Generate combined list if multiple indices
        if len(all_stock_lists) > 1:
            combined_stocks = combine_stock_lists(all_stock_lists)
            combined_file = output_dir / "csi_all.txt"
            save_to_file(combined_stocks, combined_file)
            print(f"✓ Saved {len(combined_stocks)} combined stocks to {combined_file}")

        print(f"\n✓ Successfully exported {len(all_stock_lists)} index list(s) to {output_dir}")
    else:
        # Print to console
        for index_name, stocks in all_stock_lists.items():
            print(f"\n{index_name.upper()} Stock List ({len(stocks)} stocks):")
            print("=" * 60)
            for i, stock in enumerate(sorted(stocks), 1):
                print(f"{i:4d}. {stock}")
            print("=" * 60)

        # Print combined list if multiple indices
        if len(all_stock_lists) > 1:
            combined_stocks = combine_stock_lists(all_stock_lists)
            print(f"\nCombined Stock List ({len(combined_stocks)} unique stocks):")
            print("=" * 60)
            for i, stock in enumerate(combined_stocks, 1):
                print(f"{i:4d}. {stock}")
            print("=" * 60)


if __name__ == "__main__":
    main()


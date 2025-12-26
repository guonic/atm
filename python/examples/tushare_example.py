"""
Example usage of Tushare data source.

This example demonstrates how to use TushareSource to fetch financial data.
"""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nq.config import DatabaseConfig
from nq.data.source import TushareSource, TushareSourceConfig
from nq.models import StockBasic
from nq.repo import StockBasicRepo, StockKlineDayRepo

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


def example_fetch_stock_basic():
    """Example: Fetch stock basic information."""
    # Get token from environment variable or use placeholder
    token = os.getenv("TUSHARE_TOKEN", "your_tushare_token_here")

    # Create Tushare source configuration
    config = TushareSourceConfig(
        token=token,
        api_name="stock_basic",
        params={
            "exchange": "",  # Empty for all exchanges
            "list_status": "L",  # L=listed, D=delisted, P=pause
        },
        fields="ts_code,symbol,name,area,industry,list_date",
    )

    # Create source
    source = TushareSource(config)

    # Test connection
    if not source.test_connection():
        logger.error("Failed to connect to Tushare API")
        return

    logger.info("Connected to Tushare API successfully")

    # Fetch data
    count = 0
    for record in source.fetch_stock_basic():
        logger.info(f"Stock: {record.get('ts_code')} - {record.get('name')}")
        count += 1
        if count >= 10:  # Limit to 10 records for example
            break

    source.close()
    logger.info(f"Fetched {count} stock records")


def example_fetch_daily_kline():
    """Example: Fetch daily K-line data."""
    token = os.getenv("TUSHARE_TOKEN", "your_tushare_token_here")

    config = TushareSourceConfig(
        token=token,
        api_name="daily",
        params={
            "ts_code": "000001.SZ",  # Ping An Bank
            "start_date": "20240101",
            "end_date": "20240131",
        },
    )

    source = TushareSource(config)

    if not source.test_connection():
        logger.error("Failed to connect to Tushare API")
        return

    logger.info("Fetching daily K-line data...")

    count = 0
    for record in source.fetch_daily():
        logger.info(
            f"Date: {record.get('trade_date')}, "
            f"Close: {record.get('close')}, "
            f"Volume: {record.get('vol')}"
        )
        count += 1
        if count >= 5:  # Limit to 5 records
            break

    source.close()
    logger.info(f"Fetched {count} daily K-line records")


def example_fetch_pro_bar():
    """Example: Fetch K-line data using pro_bar API."""
    token = os.getenv("TUSHARE_TOKEN", "your_tushare_token_here")

    config = TushareSourceConfig(
        token=token,
        api_name="pro_bar",
    )

    source = TushareSource(config)

    if not source.test_connection():
        logger.error("Failed to connect to Tushare API")
        return

    logger.info("Fetching K-line data using pro_bar...")

    count = 0
    for record in source.fetch_pro_bar(
        ts_code="000001.SZ",
        freq="D",  # Daily
        start_date="20240101",
        end_date="20240131",
    ):
        logger.info(
            f"Date: {record.get('trade_date')}, "
            f"Open: {record.get('open')}, "
            f"High: {record.get('high')}, "
            f"Low: {record.get('low')}, "
            f"Close: {record.get('close')}"
        )
        count += 1
        if count >= 5:
            break

    source.close()
    logger.info(f"Fetched {count} K-line records")


def example_save_to_database():
    """Example: Fetch data and save to database."""
    token = os.getenv("TUSHARE_TOKEN", "your_tushare_token_here")

    # Database configuration
    db_config = DatabaseConfig(
        host="localhost",
        port=5432,
        user="quant",
        password="quant123",
        database="quant_db",
        schema="quant",
    )

    # Create Tushare source
    source_config = TushareSourceConfig(
        token=token,
        api_name="stock_basic",
        params={"exchange": "", "list_status": "L"},
    )
    source = TushareSource(source_config)

    if not source.test_connection():
        logger.error("Failed to connect to Tushare API")
        return

    # Create repository
    stock_repo = StockBasicRepo(db_config)

    # Fetch and save data
    logger.info("Fetching and saving stock basic information...")
    count = 0
    batch = []

    for record in source.fetch_stock_basic():
        # Convert Tushare format to our model format
        try:
            # Parse list_date
            list_date_str = record.get("list_date", "")
            if list_date_str:
                list_date = datetime.strptime(list_date_str, "%Y%m%d").date()
            else:
                continue  # Skip records without list_date

            stock = StockBasic(
                ts_code=record.get("ts_code", ""),
                symbol=record.get("symbol", ""),
                full_name=record.get("name", ""),
                exchange=record.get("exchange", ""),
                market=record.get("market", ""),
                list_date=list_date,
            )

            batch.append(stock)
            count += 1

            # Save in batches of 100
            if len(batch) >= 100:
                saved = stock_repo.save_batch_models(batch)
                logger.info(f"Saved batch of {saved} stocks")
                batch = []

            if count >= 500:  # Limit for example
                break

        except Exception as e:
            logger.error(f"Error processing record: {e}")
            continue

    # Save remaining records
    if batch:
        saved = stock_repo.save_batch_models(batch)
        logger.info(f"Saved final batch of {saved} stocks")

    source.close()
    stock_repo.close()
    logger.info(f"Total processed: {count} stocks")


if __name__ == "__main__":
    print("Tushare Data Source Examples")
    print("=" * 50)
    print("\nNote: Set TUSHARE_TOKEN environment variable with your Tushare Pro token")
    print("Get your token from: https://tushare.pro/user/register\n")

    # Uncomment the example you want to run:
    # example_fetch_stock_basic()
    # example_fetch_daily_kline()
    # example_fetch_pro_bar()
    # example_save_to_database()

    print("\nExamples are ready. Uncomment the function calls above to run them.")


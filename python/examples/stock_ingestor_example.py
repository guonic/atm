"""
Example usage of StockIngestorService.

Demonstrates how to use the stock ingestor to fetch and store stock data from Tushare.
"""

import logging
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from atm.config import DatabaseConfig
from tools.dataingestor import StockIngestorService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


def example_ingest_stock_basic():
    """Example: Ingest stock basic information."""
    # Get token from environment variable
    token = os.getenv("TUSHARE_TOKEN", "")
    if not token:
        logger.error("Please set TUSHARE_TOKEN environment variable")
        return

    # Database configuration
    db_config = DatabaseConfig(
        host="localhost",
        port=5432,
        user="quant",
        password="quant123",
        database="quant_db",
        schema="quant",
    )

    # Create ingestor service
    with StockIngestorService(db_config=db_config, tushare_token=token) as ingestor:
        # Ingest stock basic information
        stats = ingestor.ingest_stock_basic(
            exchange="",  # All exchanges
            list_status="L",  # Listed stocks only
            batch_size=100,
        )

        logger.info(f"Ingestion completed: {stats}")


def example_ingest_daily_kline():
    """Example: Ingest daily K-line data for a stock."""
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
        # Ingest daily K-line for a specific stock
        stats = ingestor.ingest_daily_kline(
            ts_code="000001.SZ",  # Ping An Bank
            start_date="20240101",
            end_date="20240131",
            batch_size=100,
        )

        logger.info(f"K-line ingestion completed: {stats}")


def example_ingest_multiple_stocks():
    """Example: Ingest daily K-line data for multiple stocks."""
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

    # List of stock codes to ingest
    stock_codes = [
        "000001.SZ",  # Ping An Bank
        "000002.SZ",  # Vanke A
        "600000.SH",  # Pudong Development Bank
    ]

    with StockIngestorService(db_config=db_config, tushare_token=token) as ingestor:
        # Ingest K-line data for multiple stocks
        results = ingestor.ingest_daily_kline_batch(
            ts_codes=stock_codes,
            start_date="20240101",
            end_date="20240131",
            batch_size=100,
        )

        # Print results
        for ts_code, stats in results.items():
            logger.info(f"{ts_code}: {stats}")


if __name__ == "__main__":
    print("Stock Ingestor Examples")
    print("=" * 50)
    print("\nNote: Set TUSHARE_TOKEN environment variable with your Tushare Pro token")
    print("Get your token from: https://tushare.pro/user/register\n")

    # Uncomment the example you want to run:
    # example_ingest_stock_basic()
    # example_ingest_daily_kline()
    # example_ingest_multiple_stocks()

    print("\nExamples are ready. Uncomment the function calls above to run them.")


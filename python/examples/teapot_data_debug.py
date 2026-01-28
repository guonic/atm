# -*- coding: utf-8 -*-
"""
teapot_data_debug.py

Debug script for Teapot data loading issues.
"""

import logging
from datetime import datetime

from nq.config import DatabaseConfig, load_config
from nq.data.processor.teapot import TeapotDataLoader
from nq.repo.kline_repo import StockKlineDayRepo
from nq.repo.stock_repo import StockBasicRepo

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def check_database_connection(db_config: DatabaseConfig):
    """Check database connection."""
    logger.info("Checking database connection...")
    try:
        from sqlalchemy import create_engine, text

        engine = create_engine(
            f"postgresql://{db_config.user}:{db_config.password}"
            f"@{db_config.host}:{db_config.port}/{db_config.database}"
        )
        with engine.connect() as conn:
            result = conn.execute(text("SELECT version()"))
            version = result.fetchone()[0]
            logger.info(f"✓ Database connection successful. PostgreSQL version: {version}")
            return True
    except Exception as e:
        logger.error(f"✗ Database connection failed: {e}")
        return False


def check_stock_basic_table(db_config: DatabaseConfig, schema: str = "quant"):
    """Check stock_basic table."""
    logger.info(f"Checking stock_basic table in schema '{schema}'...")
    try:
        repo = StockBasicRepo(db_config, schema)
        stocks = repo.get_all()
        logger.info(f"✓ Found {len(stocks)} stocks in stock_basic table")
        if stocks:
            logger.info(f"  Sample stocks: {[s.ts_code for s in stocks[:5]]}")
        return len(stocks) > 0
    except Exception as e:
        logger.error(f"✗ Failed to query stock_basic table: {e}")
        return False


def check_kline_table(db_config: DatabaseConfig, schema: str = "quant"):
    """Check stock_kline_day table."""
    logger.info(f"Checking stock_kline_day table in schema '{schema}'...")
    try:
        from sqlalchemy import create_engine, text

        engine = create_engine(
            f"postgresql://{db_config.user}:{db_config.password}"
            f"@{db_config.host}:{db_config.port}/{db_config.database}"
        )
        table_name = f'"{schema}"."stock_kline_day"'
        
        with engine.connect() as conn:
            # Check if table exists
            result = conn.execute(
                text(
                    f"""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_schema = :schema 
                        AND table_name = 'stock_kline_day'
                    )
                    """
                ),
                {"schema": schema},
            )
            exists = result.fetchone()[0]
            
            if not exists:
                logger.error(f"✗ Table {table_name} does not exist")
                return False
            
            logger.info(f"✓ Table {table_name} exists")
            
            # Check row count
            result = conn.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
            count = result.fetchone()[0]
            logger.info(f"✓ Total rows: {count}")
            
            # Check date range
            result = conn.execute(
                text(f"SELECT MIN(trade_date), MAX(trade_date) FROM {table_name}")
            )
            row = result.fetchone()
            if row[0] and row[1]:
                logger.info(f"✓ Date range: {row[0]} to {row[1]}")
            
            # Check sample data
            result = conn.execute(
                text(f"SELECT ts_code, trade_date, close FROM {table_name} LIMIT 5")
            )
            logger.info("✓ Sample data:")
            for row in result:
                logger.info(f"  {row.ts_code} - {row.trade_date} - {row.close}")
            
            return True
    except Exception as e:
        logger.error(f"✗ Failed to query stock_kline_day table: {e}")
        return False


def test_data_loading(
    db_config: DatabaseConfig,
    start_date: str,
    end_date: str,
    schema: str = "quant",
):
    """Test data loading."""
    logger.info(f"Testing data loading: {start_date} to {end_date}")
    try:
        loader = TeapotDataLoader(
            db_config=db_config,
            schema=schema,
            use_cache=False,
        )
        
        # Test loading specific symbols first
        test_symbols = ["000001.SZ", "600000.SH"]
        logger.info(f"Testing with specific symbols: {test_symbols}")
        df = loader.load_daily_data(
            start_date=start_date,
            end_date=end_date,
            symbols=test_symbols,
        )
        
        if df.is_empty():
            logger.warning("✗ No data loaded for test symbols")
            return False
        
        logger.info(f"✓ Loaded {len(df)} records for test symbols")
        logger.info(f"  Columns: {df.columns}")
        logger.info(f"  Date range: {df['trade_date'].min()} to {df['trade_date'].max()}")
        
        # Test loading all stocks (small date range)
        logger.info("Testing loading all stocks (this may take a while)...")
        df_all = loader.load_daily_data(
            start_date=start_date,
            end_date=end_date,
            symbols=None,
        )
        
        if df_all.is_empty():
            logger.warning("✗ No data loaded for all stocks")
            return False
        
        logger.info(f"✓ Loaded {len(df_all)} records for all stocks")
        logger.info(f"  Unique stocks: {df_all['ts_code'].n_unique()}")
        
        return True
    except Exception as e:
        logger.error(f"✗ Data loading failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def main():
    """Main function."""
    import argparse

    parser = argparse.ArgumentParser(description="Debug Teapot data loading")
    parser.add_argument(
        "--start-date",
        type=str,
        default="2023-01-01",
        help="Start date for testing (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default="2024-01-01",
        help="End date for testing (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--schema",
        type=str,
        default="quant",
        help="Database schema name",
    )

    args = parser.parse_args()

    # Load configuration
    try:
        config = load_config()
        db_config = config.database
    except Exception as e:
        logger.warning(f"Failed to load config: {e}, using defaults")
        db_config = DatabaseConfig()

    logger.info("=" * 80)
    logger.info("Teapot Data Loading Debug")
    logger.info("=" * 80)

    # Run checks
    checks = [
        ("Database Connection", check_database_connection(db_config)),
        ("Stock Basic Table", check_stock_basic_table(db_config, args.schema)),
        ("Kline Table", check_kline_table(db_config, args.schema)),
        (
            "Data Loading",
            test_data_loading(db_config, args.start_date, args.end_date, args.schema),
        ),
    ]

    logger.info("=" * 80)
    logger.info("Summary:")
    logger.info("=" * 80)
    for name, result in checks:
        status = "✓ PASS" if result else "✗ FAIL"
        logger.info(f"{status}: {name}")

    all_passed = all(result for _, result in checks)
    if all_passed:
        logger.info("\n✓ All checks passed! Data loading should work.")
    else:
        logger.warning("\n✗ Some checks failed. Please fix the issues above.")


if __name__ == "__main__":
    main()

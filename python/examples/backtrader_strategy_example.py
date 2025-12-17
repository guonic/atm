"""
Example: Using Backtrader Strategy with ATM Framework.

This example demonstrates how to:
1. Create a backtrader-based trading strategy
2. Load data from database
3. Run a backtest
4. Get backtest results
"""

import logging
from datetime import datetime

from atm.config import load_config
from atm.trading.strategy import SMACrossStrategy, StrategyConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


def main():
    """Run example backtrader strategy."""
    # Load configuration
    try:
        config = load_config()
    except FileNotFoundError as e:
        logger.error(f"Configuration file not found: {e}")
        logger.info("Please create a config.yaml file or set environment variables:")
        logger.info("  - ATM_DATABASE_HOST")
        logger.info("  - ATM_DATABASE_PORT")
        logger.info("  - ATM_DATABASE_USER")
        logger.info("  - ATM_DATABASE_PASSWORD")
        logger.info("  - ATM_DATABASE_NAME")
        logger.info("  - ATM_DATABASE_SCHEMA")
        return

    db_config = config.database

    # Check database configuration
    logger.info(f"Database config: host={db_config.host}, port={db_config.port}, "
                f"user={db_config.user}, database={db_config.database}, schema={db_config.schema}")
    
    if not db_config.password:
        logger.warning(
            "⚠️  Database password is empty!"
        )
        logger.info("Please set the database password using one of the following methods:")
        logger.info("  1. Set environment variable: export ATM_DATABASE_PASSWORD=your_password")
        logger.info("  2. Add password to config/config.yaml:")
        logger.info("     database:")
        logger.info("       password: your_password")
        logger.info("")
        logger.info("Attempting to connect without password (may fail if password is required)...")

    # Create strategy configuration
    strategy_config = StrategyConfig(
        name="SMA Cross Strategy Example",
        description="Simple moving average cross strategy",
        initial_cash=100000.0,
        commission=0.001,  # 0.1% commission
        slippage=0.0,
        params={
            "short_period": 5,  # Short SMA period
            "long_period": 20,  # Long SMA period
        },
    )

    # Create strategy instance
    strategy = SMACrossStrategy(config=strategy_config)

    # Add data feed (load from database)
    ts_code = "000001.SZ"  # Ping An Bank
    start_date = datetime(2025, 1, 1)
    end_date = datetime(2025, 12, 31)

    logger.info(f"Loading data for {ts_code} from {start_date} to {end_date}")
    try:
        strategy.add_data(
            db_config=db_config,
            ts_code=ts_code,
            start_date=start_date,
            end_date=end_date,
        )
    except Exception as e:
        logger.error(f"Failed to load data from database: {e}")
        logger.error("Please check:")
        logger.error("  1. Database is running and accessible")
        logger.error("  2. Database credentials are correct")
        logger.error("  3. Environment variables are set (ATM_DATABASE_PASSWORD, etc.)")
        logger.error("  4. Configuration file exists and is valid")
        return

    # Set broker parameters
    strategy.set_broker_params()

    # Run backtest
    logger.info("Running backtest...")
    results = strategy.run()

    # Get backtest results
    backtest_results = strategy.get_backtest_results()
    logger.info("=" * 60)
    logger.info("Backtest Results:")
    logger.info("=" * 60)
    for key, value in backtest_results.items():
        if isinstance(value, float):
            logger.info(f"{key}: {value:.2f}")
        else:
            logger.info(f"{key}: {value}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()


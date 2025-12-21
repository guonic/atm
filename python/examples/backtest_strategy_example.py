# -*- coding: utf-8 -*-
"""
backtest_strategy_example.py

Description:
    Backtest example for trading strategies using the backtest framework.
    This script demonstrates how to evaluate trading strategies by running
    them on historical data and collecting performance metrics.

Usage:
    python backtest_strategy_example.py --ts_code 000001.SZ --start_date 2024-01-01 --end_date 2024-06-30

Arguments:
    --ts_code      Stock code in Tushare format (e.g., 000001.SZ, 600000.SH)
    --start_date   Start date of backtest period (YYYY-MM-DD)
    --end_date     End date of backtest period (YYYY-MM-DD)
    --strategy     Strategy class name (default: SMACrossStrategy)
    --short-period Short SMA period for SMA Cross Strategy (default: 5)
    --long-period  Long SMA period for SMA Cross Strategy (default: 20)
    --initial-cash Initial cash amount (default: 100000.0)
    --commission   Commission rate (default: 0.001)
    --save-report  Save backtest report to file (default: False)

Output:
    - Prints backtest metrics and report to console
    - Optionally saves report to file

Example:
    python backtest_strategy_example.py --ts_code 000001.SZ --start_date 2024-01-01 --end_date 2024-06-30
    python backtest_strategy_example.py --ts_code 600000.SH --start_date 2024-01-01 --end_date 2024-06-30 --short-period 10 --long-period 30
"""

import argparse
import logging
import os
from datetime import datetime

from atm.analysis.backtest import BacktestReport, StrategyBacktester
from atm.config import load_config
from atm.trading.strategy import SMACrossStrategy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_date(date_str: str) -> datetime:
    """
    Parse date string to datetime object.

    Args:
        date_str: Date string in YYYY-MM-DD format.

    Returns:
        Datetime object.

    Raises:
        ValueError: If date format is invalid.
    """
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError as e:
        raise ValueError(f"Invalid date format: {date_str}. Expected YYYY-MM-DD format.") from e


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Backtest trading strategy using the backtest framework"
    )
    parser.add_argument(
        "--ts_code",
        type=str,
        default="000001.SZ",
        help="Stock code in Tushare format (e.g., 000001.SZ, 600000.SH)"
    )
    parser.add_argument(
        "--start_date",
        type=str,
        required=True,
        help="Start date of backtest period (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end_date",
        type=str,
        required=True,
        help="End date of backtest period (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="SMACrossStrategy",
        choices=["SMACrossStrategy"],
        help="Strategy class name (default: SMACrossStrategy)"
    )
    parser.add_argument(
        "--short-period",
        type=int,
        default=5,
        help="Short SMA period for SMA Cross Strategy (default: 5)"
    )
    parser.add_argument(
        "--long-period",
        type=int,
        default=20,
        help="Long SMA period for SMA Cross Strategy (default: 20)"
    )
    parser.add_argument(
        "--initial-cash",
        type=float,
        default=100000.0,
        help="Initial cash amount (default: 100000.0)"
    )
    parser.add_argument(
        "--commission",
        type=float,
        default=0.001,
        help="Commission rate (default: 0.001 = 0.1%)"
    )
    parser.add_argument(
        "--slippage",
        type=float,
        default=0.0,
        help="Slippage rate (default: 0.0)"
    )
    parser.add_argument(
        "--save-report",
        action="store_true",
        help="Save backtest report to file"
    )
    parser.add_argument(
        "--report-dir",
        type=str,
        default="./outputs",
        help="Directory to save report (default: ./outputs)"
    )

    args = parser.parse_args()

    # Parse dates
    try:
        start_date = parse_date(args.start_date)
        end_date = parse_date(args.end_date)
    except ValueError as e:
        logger.error(f"Date parsing error: {e}")
        return

    if start_date >= end_date:
        logger.error("start_date must be before end_date")
        return

    # Validate strategy parameters
    if args.strategy == "SMACrossStrategy":
        if args.short_period >= args.long_period:
            logger.error(f"short_period ({args.short_period}) must be less than long_period ({args.long_period})")
            return
        if args.short_period <= 0 or args.long_period <= 0:
            logger.error("short_period and long_period must be positive")
            return

    # Load configuration
    try:
        config = load_config()
        db_config = config.database
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        return

    # Select strategy class
    strategy_map = {
        "SMACrossStrategy": SMACrossStrategy,
    }
    strategy_class = strategy_map[args.strategy]

    # Prepare strategy parameters
    strategy_params = {}
    if args.strategy == "SMACrossStrategy":
        strategy_params = {
            "short_period": args.short_period,
            "long_period": args.long_period,
        }

    # Create backtester
    logger.info("Initializing strategy backtester...")
    backtester = StrategyBacktester(
        db_config=db_config,
        initial_cash=args.initial_cash,
        commission=args.commission,
        slippage=args.slippage,
    )

    # Run backtest
    logger.info(
        f"Running strategy backtest for {args.ts_code} from {start_date.date()} to {end_date.date()}"
    )

    try:
        result = backtester.run(
            ts_code=args.ts_code,
            start_date=start_date,
            end_date=end_date,
            strategy_class=strategy_class,
            strategy_params=strategy_params,
            add_analyzers=True,
        )

        # Print report
        BacktestReport.print_report(result)

        # Save report if requested
        if args.save_report:
            os.makedirs(args.report_dir, exist_ok=True)
            report_file = os.path.join(
                args.report_dir,
                f"strategy_backtest_{args.ts_code.replace('.', '_')}_{start_date.date()}_{end_date.date()}.txt"
            )
            BacktestReport.save_report(result, report_file)

        logger.info("Strategy backtest completed successfully")

    except Exception as e:
        logger.error(f"Strategy backtest failed: {e}", exc_info=True)
        return


if __name__ == "__main__":
    main()


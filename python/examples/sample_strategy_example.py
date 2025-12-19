"""
Example: Using Direct Backtrader Strategy (Standard Pattern).

This example demonstrates the standard backtrader pattern:
- Strategy directly inherits from backtrader.Strategy
- Indicators are initialized in __init__
- Trading logic in next() method based on indicator states
- Using StrategyRunner.run_strategy() for convenience

Usage:
    python backtrader_strategy_direct_example.py --ts-code 000001.SZ --start-date 2025-01-01 --end-date 2025-12-31
    python backtrader_strategy_direct_example.py --ts-code 000001.SZ --start-date 2025-01-01 --end-date 2025-12-31 --short-period 10 --long-period 30
"""

import argparse
import logging

from atm.config import load_config
from atm.trading.strategy import (
    SMACrossStrategy,
    StrategyRunner,
    create_strategy_parser,
    parse_date_args,
    parse_strategy_args,
    validate_dates,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


def add_sma_cross_args(parser: argparse.ArgumentParser) -> None:
    """Add SMA Cross Strategy specific arguments."""
    parser.add_argument(
        "--short-period",
        type=int,
        default=5,
        help="Short SMA period (default: 5)",
    )
    parser.add_argument(
        "--long-period",
        type=int,
        default=20,
        help="Long SMA period (default: 20)",
    )


def validate_sma_periods(args: argparse.Namespace) -> None:
    """Validate SMA period arguments."""
    if args.short_period >= args.long_period:
        raise ValueError(
            f"short_period ({args.short_period}) must be less than long_period ({args.long_period})"
        )
    if args.short_period <= 0:
        raise ValueError(f"short_period must be positive, got {args.short_period}")
    if args.long_period <= 0:
        raise ValueError(f"long_period must be positive, got {args.long_period}")


def validate_cash(args: argparse.Namespace) -> None:
    """Validate initial cash argument."""
    if args.initial_cash <= 0:
        raise ValueError(f"initial_cash must be positive, got {args.initial_cash}")


def main():
    """Run example using direct backtrader strategy with StrategyRunner."""
    # Create parser with custom arguments callback
    parser = create_strategy_parser(
        description="Run SMA Cross Strategy backtest",
        custom_args_callback=add_sma_cross_args,
    )

    # Parse arguments with validation callbacks (including date validation)
    args = parse_strategy_args(
        parser,
        validate_dates,
        validate_sma_periods,
        validate_cash,
    )
    if args is None:
        return  # Parsing or validation failed

    # Parse dates (already validated, so safe to parse)
    start_date, end_date = parse_date_args(args)

    # Load configuration
    try:
        config = load_config()
    except FileNotFoundError as e:
        logger.error(f"Configuration file not found: {e}")
        logger.info("Please create a config.yaml file or set environment variables")
        return

    db_config = config.database

    # Run strategy with all parameters in one call
    logger.info(
        f"Running backtest for {args.ts_code} from {start_date.date()} to {end_date.date()} "
        f"with short_period={args.short_period}, long_period={args.long_period}"
    )

    results = StrategyRunner.run_strategy(
        db_config=db_config,
        strategy_class=SMACrossStrategy,
        ts_code=args.ts_code,
        start_date=start_date,
        end_date=end_date,
        initial_cash=args.initial_cash,
        commission=args.commission,
        slippage=args.slippage,
        schema=args.schema,
        strategy_params={
            "short_period": args.short_period,
            "long_period": args.long_period,
        },
    )

    StrategyRunner.print_results(results)


if __name__ == "__main__":
    main()


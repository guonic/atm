"""
Command line argument parser for strategy backtesting.

Provides a common argument parser that can be reused across different strategy examples.
"""

import argparse
import logging
from datetime import datetime
from typing import Callable, Optional, Tuple

logger = logging.getLogger(__name__)


def parse_date(date_str: str) -> datetime:
    """
    Parse date string to datetime object.

    Args:
        date_str: Date string in YYYY-MM-DD format.

    Returns:
        Datetime object.

    Raises:
        ValueError: If date string format is invalid.
    """
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError as e:
        raise ValueError(f"Invalid date format: {date_str}. Expected YYYY-MM-DD format.") from e


def create_strategy_parser(
    description: str = "Run strategy backtest",
    custom_args_callback: Optional[Callable[[argparse.ArgumentParser], None]] = None,
) -> argparse.ArgumentParser:
    """
    Create a common argument parser for strategy backtesting.

    Args:
        description: Description for the argument parser.
        custom_args_callback: Optional callback function to register custom arguments.
                            The callback receives the parser as argument and can add
                            custom arguments to it.

    Returns:
        ArgumentParser instance with common strategy arguments and any custom arguments
        added by the callback.

    Example:
        def add_my_strategy_args(parser):
            parser.add_argument("--my-param", type=int, default=10)

        parser = create_strategy_parser(
            description="My strategy",
            custom_args_callback=add_my_strategy_args
        )
    """
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Data parameters
    parser.add_argument(
        "--ts-code",
        type=str,
        default="000001.SZ",
        help="Stock code (e.g., '000001.SZ', default: '000001.SZ')",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default="2025-01-01",
        help="Start date in YYYY-MM-DD format (default: 2025-01-01)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default="2025-12-31",
        help="End date in YYYY-MM-DD format (default: 2025-12-31)",
    )
    parser.add_argument(
        "--schema",
        type=str,
        default="quant",
        help="Database schema name (default: 'quant')",
    )

    # Broker parameters
    parser.add_argument(
        "--initial-cash",
        type=float,
        default=100000.0,
        help="Initial cash amount (default: 100000.0)",
    )
    parser.add_argument(
        "--commission",
        type=float,
        default=0.001,
        help="Commission rate (default: 0.001 = 0.1%%)",
    )
    parser.add_argument(
        "--slippage",
        type=float,
        default=0.0,
        help="Slippage rate (default: 0.0)",
    )

    # Call custom callback to add strategy-specific arguments
    if custom_args_callback is not None:
        custom_args_callback(parser)

    return parser


def parse_strategy_args(
    parser: Optional[argparse.ArgumentParser] = None,
    *validation_callbacks: Callable[[argparse.Namespace], None],
) -> Optional[argparse.Namespace]:
    """
    Parse command line arguments for strategy backtesting.

    Args:
        parser: Optional custom argument parser. If None, uses default parser.
        *validation_callbacks: Variable number of validation callback functions to check parsed arguments.
                              Each callback receives the parsed args namespace and should raise
                              ValueError or argparse.ArgumentError if validation fails.

    Returns:
        Parsed arguments namespace if successful, None if parsing or validation fails.
        Errors are logged internally, no exceptions are raised.

    Example:
        def validate_periods(args):
            if args.short_period >= args.long_period:
                raise ValueError("short_period must be less than long_period")

        def validate_cash(args):
            if args.initial_cash <= 0:
                raise ValueError("initial_cash must be positive")

        args = parse_strategy_args(
            my_parser,
            validate_periods,
            validate_cash,
        )
        if args is None:
            return  # Parsing or validation failed
    """
    if parser is None:
        parser = create_strategy_parser()

    try:
        args = parser.parse_args()
    except SystemExit:
        # argparse raises SystemExit on --help or invalid arguments
        return None
    except Exception as e:
        logger.error(f"Failed to parse arguments: {e}")
        return None

    # Run validation callbacks if provided
    if validation_callbacks:
        for callback in validation_callbacks:
            try:
                callback(args)
            except (ValueError, argparse.ArgumentError) as e:
                # Log validation errors and return None
                logger.error(f"Argument validation failed: {e}")
                return None
            except Exception as e:
                # Log unexpected errors and return None
                logger.error(f"Validation callback error: {e}")
                return None

    return args


def validate_dates(args: argparse.Namespace) -> None:
    """
    Validate date arguments (validation callback).

    This function can be used as a validation callback in parse_strategy_args.
    It validates that dates are in correct format and start_date < end_date.

    Args:
        args: Parsed arguments namespace.

    Raises:
        ValueError: If date format is invalid or date range is invalid.
    """
    try:
        start_date = parse_date(args.start_date)
        end_date = parse_date(args.end_date)
    except ValueError as e:
        raise ValueError(f"Date parsing error: {e}") from e

    if start_date >= end_date:
        raise ValueError("Start date must be before end date")


def parse_date_args(args: argparse.Namespace) -> Tuple[datetime, datetime]:
    """
    Parse date arguments and return datetime objects.

    This function should be called after validation to get parsed date objects.
    It assumes dates have already been validated.

    Args:
        args: Parsed arguments namespace.

    Returns:
        Tuple of (start_date, end_date) as datetime objects.

    Raises:
        ValueError: If date format is invalid.
    """
    start_date = parse_date(args.start_date)
    end_date = parse_date(args.end_date)
    return start_date, end_date


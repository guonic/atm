"""
Common argument parser for strategy evaluation tools.

Provides reusable argument parsing functionality for all strategy evaluation scripts.
"""

import argparse
from typing import Any, Dict, Optional


def create_base_parser(description: str) -> argparse.ArgumentParser:
    """
    Create a base argument parser with common arguments for strategy evaluation.

    Args:
        description: Description for the argument parser.

    Returns:
        ArgumentParser instance with common arguments added.
    """
    parser = argparse.ArgumentParser(description=description)

    # Configuration
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to configuration file (default: config/config.yaml)",
    )

    # Date range
    parser.add_argument(
        "--start-date",
        type=str,
        required=True,
        help="Backtest start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        required=True,
        help="Backtest end date (YYYY-MM-DD)",
    )

    # Stock selection
    parser.add_argument(
        "--min-market-cap",
        type=float,
        default=None,
        help="Minimum market cap (in 10K CNY)",
    )
    parser.add_argument(
        "--max-market-cap",
        type=float,
        default=None,
        help="Maximum market cap (in 10K CNY)",
    )
    parser.add_argument(
        "--num-stocks",
        type=int,
        default=100,
        help="Number of stocks to evaluate (default: 100)",
    )
    parser.add_argument(
        "--skip-market-cap-filter",
        action="store_true",
        help="Skip market cap filter and use all stocks",
    )
    parser.add_argument(
        "--exchange",
        type=str,
        default=None,
        choices=["SSE", "SZSE", "BSE"],
        help="Exchange filter (SSE/SZSE/BSE). If not specified, select from all exchanges.",
    )

    # Output and reporting
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV file path for results (optional, defaults to strategy-specific name)",
    )
    parser.add_argument(
        "--sort-by",
        type=str,
        default="total_return",
        choices=[
            "total_return",
            "sharpe_ratio",
            "max_drawdown",
            "win_rate",
            "final_value",
            "initial_value",
            "total_trades",
            "won_trades",
        ],
        help="Column to sort results by (default: total_return)",
    )
    parser.add_argument(
        "--ascending",
        action="store_true",
        help="Sort in ascending order (default: descending)",
    )

    # Backtest parameters
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
    parser.add_argument(
        "--schema",
        type=str,
        default="quant",
        help="Database schema name (default: quant)",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for stock selection (default: 42)",
    )
    parser.add_argument(
        "--kline-type",
        type=str,
        default="day",
        choices=["day", "hour", "30min", "15min", "5min", "1min"],
        help="K-line type for backtesting (default: day)",
    )

    return parser


def parse_common_args(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Parse common arguments and return as dictionary.

    Args:
        args: Parsed arguments namespace.

    Returns:
        Dictionary with common argument values.
    """
    return {
        "config": args.config,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "min_market_cap": args.min_market_cap,
        "max_market_cap": args.max_market_cap,
        "num_stocks": args.num_stocks,
        "skip_market_cap_filter": args.skip_market_cap_filter,
        "exchange": args.exchange,
        "output": args.output,
        "sort_by": args.sort_by,
        "ascending": args.ascending,
        "initial_cash": args.initial_cash,
        "commission": args.commission,
        "slippage": args.slippage,
        "schema": args.schema,
        "random_seed": args.random_seed,
        "kline_type": args.kline_type,
    }


def validate_dates(start_date_str: str, end_date_str: str) -> tuple:
    """
    Validate and parse date strings.

    Args:
        start_date_str: Start date string (YYYY-MM-DD).
        end_date_str: End date string (YYYY-MM-DD).

    Returns:
        Tuple of (start_date, end_date) as datetime objects.

    Raises:
        ValueError: If date format is invalid or start_date > end_date.
    """
    from datetime import datetime

    try:
        start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
        end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
    except ValueError as e:
        raise ValueError(f"Invalid date format. Use YYYY-MM-DD format. Error: {e}") from e

    if start_date > end_date:
        raise ValueError(f"Start date ({start_date_str}) must be before end date ({end_date_str})")

    return start_date, end_date


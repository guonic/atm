#!/usr/bin/env python3
"""
Fibonacci Moving Average Strategy Evaluation Tool.

Command-line tool for evaluating Fibonacci MA strategy on multiple stocks.
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from atm.config import DatabaseConfig, load_config
from atm.analysis.backtest.batch_evaluator import BatchStrategyEvaluator
from atm.trading.strategy.fibonacci_ma_strategy import FibonacciMAStrategy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


def main():
    """Main entry point for Fibonacci MA strategy evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate Fibonacci Moving Average strategy on selected stocks"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to configuration file",
    )
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
        help="Exchange code (SSE/SZSE/BSE). If not specified, select from all exchanges.",
    )
    # Fibonacci MA parameters
    parser.add_argument(
        "--ma13-period",
        type=int,
        default=13,
        help="13-day moving average period (default: 13)",
    )
    parser.add_argument(
        "--ma21-period",
        type=int,
        default=21,
        help="21-day moving average period (default: 21)",
    )
    parser.add_argument(
        "--ma34-period",
        type=int,
        default=34,
        help="34-day moving average period (default: 34)",
    )
    parser.add_argument(
        "--use-ma55",
        action="store_true",
        default=False,
        help="Enable 55-day MA for large-cap stocks (default: False)",
    )
    parser.add_argument(
        "--use-ma89",
        action="store_true",
        default=False,
        help="Enable 89-day MA for cyclical stocks (default: False)",
    )
    # Volume parameters
    parser.add_argument(
        "--volume-ma-period",
        type=int,
        default=20,
        help="Volume moving average period (default: 20)",
    )
    parser.add_argument(
        "--volume-threshold-buy",
        type=float,
        default=1.2,
        help="Volume threshold for buy signals (default: 1.2)",
    )
    parser.add_argument(
        "--volume-threshold-breakout",
        type=float,
        default=1.5,
        help="Volume threshold for breakout confirmation (default: 1.5)",
    )
    parser.add_argument(
        "--no-volume-confirmation",
        dest="use_volume_confirmation",
        action="store_false",
        default=True,
        help="Disable volume confirmation (default: enabled)",
    )
    # Strategy options
    parser.add_argument(
        "--no-multi-resonance",
        dest="use_multi_resonance",
        action="store_false",
        default=True,
        help="Disable multi-period resonance (default: enabled)",
    )
    parser.add_argument(
        "--stock-type",
        type=str,
        default="small_cap",
        choices=["small_cap", "large_cap", "cyclical"],
        help="Stock type: small_cap (13/21/34), large_cap (21/34/55), cyclical (13/34/89) (default: small_cap)",
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
        "--output",
        type=str,
        default=None,
        help="Output CSV file path for results (optional)",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for stock selection (default: 42)",
    )
    parser.add_argument(
        "--sort-by",
        type=str,
        default="total_return",
        choices=[
            "total_return",
            "sharpe_ratio",
            "max_drawdown",
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

    args = parser.parse_args()

    # Parse dates
    try:
        start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
        end_date = datetime.strptime(args.end_date, "%Y-%m-%d")
    except ValueError as e:
        logger.error(f"Invalid date format: {e}")
        return 1

    # Load configuration
    try:
        config = load_config(args.config)
        db_config = config.database
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        return 1

    # Initialize evaluator
    evaluator = BatchStrategyEvaluator(
        db_config=db_config,
        schema=args.schema,
        initial_cash=args.initial_cash,
        commission=args.commission,
        slippage=args.slippage,
    )

    # Select stocks
    selected_stocks = evaluator.select_stocks(
        min_market_cap=args.min_market_cap,
        max_market_cap=args.max_market_cap,
        num_stocks=args.num_stocks,
        exchange=args.exchange,
        random_seed=args.random_seed,
        skip_market_cap_filter=args.skip_market_cap_filter,
    )

    if not selected_stocks:
        logger.error("No stocks selected. Please check your market cap criteria.")
        return 1

    logger.info(f"Selected {len(selected_stocks)} stocks for evaluation")

    # Strategy parameters
    strategy_params = {
        "ma13_period": args.ma13_period,
        "ma21_period": args.ma21_period,
        "ma34_period": args.ma34_period,
        "use_ma55": args.use_ma55,
        "use_ma89": args.use_ma89,
        "volume_ma_period": args.volume_ma_period,
        "volume_threshold_buy": args.volume_threshold_buy,
        "volume_threshold_breakout": args.volume_threshold_breakout,
        "use_volume_confirmation": args.use_volume_confirmation,
        "use_multi_resonance": args.use_multi_resonance,
        "stock_type": args.stock_type,
    }

    # Run evaluation
    results = evaluator.evaluate_strategy(
        strategy_class=FibonacciMAStrategy,
        selected_stocks=selected_stocks,
        start_date=start_date,
        end_date=end_date,
        strategy_params=strategy_params,
        add_analyzers=True,
    )

    # Generate report
    summary_df = evaluator.generate_summary_report(
        results,
        output_file=args.output,
        sort_by=args.sort_by,
        ascending=args.ascending,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())


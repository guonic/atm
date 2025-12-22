#!/usr/bin/env python3
"""
CCI Strategy Evaluation Tool.

Command-line tool for evaluating CCI strategy on multiple stocks.
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
from atm.trading.strategy.cci_strategy import CCIStrategy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


def main():
    """Main entry point for CCI strategy evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate CCI strategy on selected stocks"
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
        default=5000000,  # 500B CNY in 10K CNY units (500亿 = 5000000万)
        help="Minimum market cap (in 10K CNY, default: 5000000 = 500B CNY)",
    )
    parser.add_argument(
        "--max-market-cap",
        type=float,
        default=10000000,  # 1000B CNY in 10K CNY units (1000亿 = 10000000万)
        help="Maximum market cap (in 10K CNY, default: 10000000 = 1000B CNY)",
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
        help="Skip market cap filter and use all stocks (useful when finance data is not available)",
    )
    parser.add_argument(
        "--exchange",
        type=str,
        default=None,
        choices=["SSE", "SZSE", "BSE"],
        help="Exchange code (SSE/SZSE/BSE). If not specified, select from all exchanges.",
    )
    parser.add_argument(
        "--cci-period",
        type=int,
        default=20,
        help="CCI period (default: 20)",
    )
    parser.add_argument(
        "--overbought",
        type=int,
        default=100,
        help="Overbought threshold (default: 100)",
    )
    parser.add_argument(
        "--oversold",
        type=int,
        default=-100,
        help="Oversold threshold (default: -100)",
    )
    parser.add_argument(
        "--divergence-lookback",
        type=int,
        default=20,
        help="Divergence lookback period (default: 20)",
    )
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
        choices=["total_return", "sharpe_ratio", "max_drawdown", "final_value", "initial_value", "total_trades", "won_trades"],
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
        logger.error("Expected format: YYYY-MM-DD")
        return 1

    if start_date >= end_date:
        logger.error("Start date must be before end date")
        return 1

    # Load configuration
    config = load_config(args.config)
    db_config = config.database

    # Create batch evaluator
    evaluator = BatchStrategyEvaluator(
        db_config=db_config,
        schema=args.schema,
        initial_cash=args.initial_cash,
        commission=args.commission,
        slippage=args.slippage,
    )

    # Select stocks
    if args.skip_market_cap_filter:
        selected_stocks = evaluator.select_stocks(
            num_stocks=args.num_stocks,
            exchange=args.exchange,
            random_seed=args.random_seed,
            skip_market_cap_filter=True,
        )
    else:
        selected_stocks = evaluator.select_stocks(
            min_market_cap=args.min_market_cap,
            max_market_cap=args.max_market_cap,
            num_stocks=args.num_stocks,
            exchange=args.exchange,
            random_seed=args.random_seed,
            skip_market_cap_filter=False,
        )

    if len(selected_stocks) == 0:
        logger.error("No stocks selected. Please check your market cap criteria.")
        return 1

    logger.info(f"Selected {len(selected_stocks)} stocks for evaluation")

    # Strategy parameters
    strategy_params = {
        "cci_period": args.cci_period,
        "overbought": args.overbought,
        "oversold": args.oversold,
        "divergence_lookback": args.divergence_lookback,
    }

    # Run evaluation
    results = evaluator.evaluate_strategy(
        strategy_class=CCIStrategy,
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


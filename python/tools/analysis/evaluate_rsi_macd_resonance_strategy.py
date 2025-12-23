#!/usr/bin/env python3
"""
RSI + MACD Resonance Strategy Evaluation Tool.

Command-line tool for evaluating RSI + MACD Resonance strategy on multiple stocks.
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
from atm.trading.strategy.rsi_macd_resonance_strategy import RSIMACDResonanceStrategy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


def main():
    """Main entry point for RSI + MACD Resonance strategy evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate RSI + MACD Resonance strategy on selected stocks"
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
        help="Exchange filter (SSE/SZSE/BSE)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="rsi_macd_resonance_results.csv",
        help="Output CSV file path (default: rsi_macd_resonance_results.csv)",
    )
    parser.add_argument(
        "--sort-by",
        type=str,
        default="win_rate",
        choices=["win_rate", "total_return", "sharpe_ratio", "max_drawdown"],
        help="Sort results by field (default: win_rate)",
    )
    parser.add_argument(
        "--ascending",
        action="store_true",
        help="Sort in ascending order (default: descending)",
    )
    # RSI parameters
    parser.add_argument(
        "--rsi-period",
        type=int,
        default=14,
        help="RSI calculation period (default: 14)",
    )
    parser.add_argument(
        "--rsi-oversold",
        type=float,
        default=30,
        help="RSI oversold threshold (default: 30)",
    )
    parser.add_argument(
        "--rsi-overbought",
        type=float,
        default=70,
        help="RSI overbought threshold (default: 70)",
    )
    parser.add_argument(
        "--rsi-midline",
        type=float,
        default=50,
        help="RSI strength/weakness dividing line (default: 50)",
    )
    # MACD parameters
    parser.add_argument(
        "--macd-fast",
        type=int,
        default=12,
        help="MACD fast period (default: 12)",
    )
    parser.add_argument(
        "--macd-slow",
        type=int,
        default=26,
        help="MACD slow period (default: 26)",
    )
    parser.add_argument(
        "--macd-signal",
        type=int,
        default=9,
        help="MACD signal period (default: 9)",
    )
    # Strategy parameters
    parser.add_argument(
        "--divergence-lookback",
        type=int,
        default=20,
        help="Divergence lookback period (default: 20)",
    )
    parser.add_argument(
        "--use-sensitive-entry",
        action="store_true",
        default=True,
        help="Use sensitive entry signal (MACD bar color change) (default: True)",
    )
    parser.add_argument(
        "--no-sensitive-entry",
        dest="use_sensitive_entry",
        action="store_false",
        help="Use classic entry signal instead of sensitive entry",
    )
    parser.add_argument(
        "--stop-loss-atr-multiplier",
        type=float,
        default=2.0,
        help="Stop loss ATR multiplier (default: 2.0)",
    )

    args = parser.parse_args()

    # Load configuration
    try:
        config = load_config(args.config)
        db_config = config.database
    except Exception as e:
        logger.warning(f"Failed to load config from {args.config}: {e}")
        logger.info("Using default database configuration")
        db_config = DatabaseConfig()

    # Create strategy parameters
    strategy_params = {
        "rsi_period": args.rsi_period,
        "rsi_oversold": args.rsi_oversold,
        "rsi_overbought": args.rsi_overbought,
        "rsi_midline": args.rsi_midline,
        "macd_fast": args.macd_fast,
        "macd_slow": args.macd_slow,
        "macd_signal": args.macd_signal,
        "divergence_lookback": args.divergence_lookback,
        "use_sensitive_entry": args.use_sensitive_entry,
        "stop_loss_atr_multiplier": args.stop_loss_atr_multiplier,
    }

    logger.info("=" * 80)
    logger.info("RSI + MACD Resonance Strategy Evaluation")
    logger.info("=" * 80)
    logger.info(f"Start Date: {args.start_date}")
    logger.info(f"End Date: {args.end_date}")
    logger.info(f"Number of Stocks: {args.num_stocks}")
    logger.info(f"Strategy Parameters: {strategy_params}")
    logger.info("=" * 80)

    # Parse dates
    try:
        start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
        end_date = datetime.strptime(args.end_date, "%Y-%m-%d")
    except ValueError as e:
        logger.error(f"Invalid date format: {e}")
        sys.exit(1)

    # Create evaluator
    evaluator = BatchStrategyEvaluator(db_config=db_config)

    # Select stocks
    selected_stocks = evaluator.select_stocks(
        min_market_cap=args.min_market_cap,
        max_market_cap=args.max_market_cap,
        num_stocks=args.num_stocks,
        exchange=args.exchange,
        skip_market_cap_filter=args.skip_market_cap_filter,
    )

    if not selected_stocks:
        logger.error("No stocks selected. Please check your market cap criteria.")
        sys.exit(1)

    logger.info(f"Selected {len(selected_stocks)} stocks for evaluation")

    # Run evaluation
    try:
        results = evaluator.evaluate_strategy(
            strategy_class=RSIMACDResonanceStrategy,
            selected_stocks=selected_stocks,
            start_date=start_date,
            end_date=end_date,
            strategy_params=strategy_params,
            add_analyzers=True,
            kline_type="day",
        )

        # Generate summary report
        evaluator.generate_summary_report(
            results=results,
            output_file=args.output,
            sort_by=args.sort_by,
            ascending=args.ascending,
        )

        logger.info(f"Evaluation completed. Results saved to {args.output}")

    except KeyboardInterrupt:
        logger.warning("Evaluation interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()


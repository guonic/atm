#!/usr/bin/env python3
"""
Price Entropy Strategy Evaluation Tool.

Command-line tool for evaluating Price Entropy (volatility compression breakout) strategy on multiple stocks.
"""

import logging
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from nq.analysis.backtest.common_args import create_base_parser
from nq.analysis.backtest.evaluation_runner import run_strategy_evaluation
from nq.trading.strategies.price_entropy_strategy import PriceEntropyStrategy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


def main():
    """Main entry point for Price Entropy strategy evaluation."""
    # Create base parser with common arguments
    parser = create_base_parser(
        description="Evaluate Price Entropy (volatility compression breakout) strategy on selected stocks"
    )

    # Override default output
    parser.set_defaults(output="price_entropy_results.csv")

    # Strategy parameters
    parser.add_argument(
        "--atr-period",
        type=int,
        default=14,
        help="ATR calculation period (default: 14)",
    )
    parser.add_argument(
        "--atr-long-period",
        type=int,
        default=252,
        help="Long ATR average period for compression baseline (default: 252 trading days)",
    )
    parser.add_argument(
        "--atr-compress-ratio",
        type=float,
        default=0.4,
        help="ATR compression ratio vs long ATR (default: 0.4)",
    )
    parser.add_argument(
        "--compress-days-atr",
        type=int,
        default=5,
        help="Consecutive days for ATR compression (default: 5)",
    )
    parser.add_argument(
        "--intraday-range-ratio",
        type=float,
        default=0.5,
        help="Daily range threshold vs ATR for compression (default: 0.5)",
    )
    parser.add_argument(
        "--volume-ratio-max",
        type=float,
        default=0.6,
        help="Volume threshold vs 120-day average for compression (default: 0.6)",
    )
    parser.add_argument(
        "--compress-days-intraday",
        type=int,
        default=3,
        help="Consecutive days for intraday compression (default: 3)",
    )
    parser.add_argument(
        "--bb-period",
        type=int,
        default=20,
        help="Bollinger Bands period (default: 20)",
    )
    parser.add_argument(
        "--bb-dev",
        type=float,
        default=2.0,
        help="Bollinger Bands deviation factor (default: 2.0)",
    )
    parser.add_argument(
        "--bb-width-min-pct",
        type=float,
        default=0.01,
        help="Minimum band width percentage of price to qualify as compression (default: 0.01 = 1%)",
    )
    parser.add_argument(
        "--range-lookback",
        type=int,
        default=20,
        help="Lookback for recent range high/low (default: 20)",
    )
    parser.add_argument(
        "--breakout-body-pct",
        type=float,
        default=0.02,
        help="Minimum candle body percentage for breakout (default: 0.02 = 2%)",
    )
    parser.add_argument(
        "--breakout-volume-mult",
        type=float,
        default=1.5,
        help="Breakout volume multiplier vs 20-day average (default: 1.5)",
    )
    parser.add_argument(
        "--initial-position-pct",
        type=float,
        default=0.3,
        help="Initial position size percentage (default: 0.3 = 30%)",
    )
    parser.add_argument(
        "--add-position-pct",
        type=float,
        default=0.3,
        help="Additional position size percentage on retest (default: 0.3 = 30%)",
    )
    parser.add_argument(
        "--stop-lookback",
        type=int,
        default=2,
        help="Bars to confirm breakdown below breakout level before exit (default: 2)",
    )
    parser.add_argument(
        "--max-drawdown-pct",
        type=float,
        default=0.08,
        help="Maximum drawdown from entry before stop (default: 0.08 = 8%)",
    )

    args = parser.parse_args()

    strategy_params = {
        "atr_period": args.atr_period,
        "atr_long_period": args.atr_long_period,
        "atr_compress_ratio": args.atr_compress_ratio,
        "compress_days_atr": args.compress_days_atr,
        "intraday_range_ratio": args.intraday_range_ratio,
        "volume_ratio_max": args.volume_ratio_max,
        "compress_days_intraday": args.compress_days_intraday,
        "bb_period": args.bb_period,
        "bb_dev": args.bb_dev,
        "bb_width_min_pct": args.bb_width_min_pct,
        "range_lookback": args.range_lookback,
        "breakout_body_pct": args.breakout_body_pct,
        "breakout_volume_mult": args.breakout_volume_mult,
        "initial_position_pct": args.initial_position_pct,
        "add_position_pct": args.add_position_pct,
        "stop_lookback": args.stop_lookback,
        "max_drawdown_pct": args.max_drawdown_pct,
    }

    return run_strategy_evaluation(
        strategy_class=PriceEntropyStrategy,
        strategy_name="Price Entropy",
        default_output="price_entropy_results.csv",
        strategy_params=strategy_params,
        args=args,
    )


if __name__ == "__main__":
    sys.exit(main())



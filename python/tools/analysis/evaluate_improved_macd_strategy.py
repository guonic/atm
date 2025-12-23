#!/usr/bin/env python3
"""
Improved MACD Strategy Evaluation Tool.

Command-line tool for evaluating Improved MACD strategy on multiple stocks.
"""

import logging
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from atm.analysis.backtest.common_args import create_base_parser
from atm.analysis.backtest.evaluation_runner import run_strategy_evaluation
from atm.trading.strategies.improved_macd_strategy import ImprovedMACDStrategy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


def main():
    """Main entry point for Improved MACD strategy evaluation."""
    # Create base parser with common arguments
    parser = create_base_parser(
        description="Evaluate Improved MACD strategy on selected stocks"
    )

    # Override default output
    parser.set_defaults(output="improved_macd_results.csv")
    # MACD parameters
    parser.add_argument(
        "--fast-period",
        type=int,
        default=8,
        help="Fast EMA period for MACD (default: 8)",
    )
    parser.add_argument(
        "--slow-period",
        type=int,
        default=17,
        help="Slow EMA period for MACD (default: 17)",
    )
    parser.add_argument(
        "--signal-period",
        type=int,
        default=5,
        help="Signal line period for MACD (default: 5)",
    )
    # Trend filter parameters
    parser.add_argument(
        "--ma-period",
        type=int,
        default=20,
        help="Moving average period for trend filter (default: 20)",
    )
    # Volume parameters
    parser.add_argument(
        "--volume-ma-period",
        type=int,
        default=5,
        help="Volume moving average period (default: 5)",
    )
    parser.add_argument(
        "--volume-threshold-buy",
        type=float,
        default=1.2,
        help="Volume threshold for buy signals (default: 1.2 = 1.2× average)",
    )
    parser.add_argument(
        "--volume-threshold-sell",
        type=float,
        default=0.7,
        help="Volume threshold for sell signals (default: 0.7 = 70%% of average)",
    )
    # Consolidation filter parameters
    parser.add_argument(
        "--no-consolidation-filter",
        dest="use_consolidation_filter",
        action="store_false",
        default=True,
        help="Disable consolidation box filtering (default: enabled)",
    )
    parser.add_argument(
        "--consolidation-lookback",
        type=int,
        default=20,
        help="Lookback period for consolidation detection (default: 20)",
    )
    parser.add_argument(
        "--consolidation-threshold",
        type=float,
        default=0.05,
        help="Price range threshold for consolidation (default: 0.05 = 5%%)",
    )
    # Failure tracking parameters
    parser.add_argument(
        "--max-consecutive-failures",
        type=int,
        default=2,
        help="Maximum consecutive golden cross failures allowed (default: 2)",
    )
    args = parser.parse_args()

    # Create strategy-specific parameters
    strategy_params = {
        "fast_period": args.fast_period,
        "slow_period": args.slow_period,
        "signal_period": args.signal_period,
        "ma_period": args.ma_period,
        "volume_ma_period": args.volume_ma_period,
        "volume_threshold_buy": args.volume_threshold_buy,
        "volume_threshold_sell": args.volume_threshold_sell,
        "use_consolidation_filter": args.use_consolidation_filter,
        "consolidation_lookback": args.consolidation_lookback,
        "consolidation_threshold": args.consolidation_threshold,
        "max_consecutive_failures": args.max_consecutive_failures,
    }

    # Run evaluation using common runner
    return run_strategy_evaluation(
        strategy_class=ImprovedMACDStrategy,
        strategy_name="Improved MACD",
        default_output="improved_macd_results.csv",
        strategy_params=strategy_params,
        args=args,
    )


if __name__ == "__main__":
    sys.exit(main())



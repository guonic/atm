#!/usr/bin/env python3
"""
Fibonacci Moving Average Strategy Evaluation Tool.

Command-line tool for evaluating Fibonacci MA strategy on multiple stocks.
"""

import logging
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from atm.analysis.backtest.common_args import create_base_parser
from atm.analysis.backtest.evaluation_runner import run_strategy_evaluation
from atm.trading.strategies.fibonacci_ma_strategy import FibonacciMAStrategy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


def main():
    """Main entry point for Fibonacci MA strategy evaluation."""
    # Create base parser with common arguments
    parser = create_base_parser(
        description="Evaluate Fibonacci Moving Average strategy on selected stocks"
    )

    # Override default output
    parser.set_defaults(output="fibonacci_ma_strategy_results.csv")
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
    args = parser.parse_args()

    # Create strategy-specific parameters
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

    # Run evaluation using common runner
    return run_strategy_evaluation(
        strategy_class=FibonacciMAStrategy,
        strategy_name="Fibonacci MA",
        default_output="fibonacci_ma_strategy_results.csv",
        strategy_params=strategy_params,
        args=args,
    )


if __name__ == "__main__":
    sys.exit(main())


#!/usr/bin/env python3
"""
Chip Peak Strategy Evaluation Tool.

Command-line tool for evaluating Chip Peak + 20-day Cost Line strategy on multiple stocks.
"""

import logging
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from atm.analysis.backtest.common_args import create_base_parser
from atm.analysis.backtest.evaluation_runner import run_strategy_evaluation
from atm.trading.strategies.chip_peak_strategy import ChipPeakStrategy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


def main():
    """Main entry point for Chip Peak strategy evaluation."""
    # Create base parser with common arguments
    parser = create_base_parser(
        description="Evaluate Chip Peak + 20-day Cost Line strategy on selected stocks"
    )

    # Override default output
    parser.set_defaults(output="chip_peak_strategy_results.csv")
    # Cost line parameters
    parser.add_argument(
        "--ma-period",
        type=int,
        default=20,
        help="Moving average period for cost line (default: 20)",
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
        default=0.5,
        help="Volume threshold for buy signals (default: 0.5 = 50%% of average)",
    )
    parser.add_argument(
        "--volume-threshold-sell",
        type=float,
        default=1.5,
        help="Volume threshold for sell signals (default: 1.5 = 150%% of average)",
    )
    # Chip peak parameters
    parser.add_argument(
        "--price-tolerance",
        type=float,
        default=0.02,
        help="Price tolerance for edge detection (default: 0.02 = 2%%)",
    )
    parser.add_argument(
        "--allow-multi-peak",
        dest="avoid_multi_peak",
        action="store_false",
        default=True,
        help="Allow trading multi-peak stocks (default: disabled)",
    )
    parser.add_argument(
        "--min-peak-sharpness",
        type=float,
        default=0.3,
        help="Minimum peak sharpness required (default: 0.3)",
    )
    args = parser.parse_args()

    # Create strategy-specific parameters
    # Note: chip_peak_indicator will be created per stock in the strategy
    strategy_params = {
        "chip_peak_indicator": None,  # Will use DummyChipPeakIndicator by default
        "ma_period": args.ma_period,
        "volume_ma_period": args.volume_ma_period,
        "volume_threshold_buy": args.volume_threshold_buy,
        "volume_threshold_sell": args.volume_threshold_sell,
        "price_tolerance": args.price_tolerance,
        "avoid_multi_peak": args.avoid_multi_peak,
        "min_peak_sharpness": args.min_peak_sharpness,
    }

    # Run evaluation using common runner (kline_type defaults to "day")
    return run_strategy_evaluation(
        strategy_class=ChipPeakStrategy,
        strategy_name="Chip Peak",
        default_output="chip_peak_strategy_results.csv",
        strategy_params=strategy_params,
        args=args,
    )


if __name__ == "__main__":
    sys.exit(main())


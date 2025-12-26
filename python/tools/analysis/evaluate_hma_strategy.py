#!/usr/bin/env python3
"""
Hull Moving Average (HMA) Strategy Evaluation Tool.

Command-line tool for evaluating HMA strategy on multiple stocks.
"""

import logging
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from nq.analysis.backtest.common_args import create_base_parser
from nq.analysis.backtest.evaluation_runner import run_strategy_evaluation
from nq.trading.strategies.hma_strategy import HMAStrategy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


def main():
    """Main entry point for HMA strategy evaluation."""
    # Create base parser with common arguments
    parser = create_base_parser(
        description="Evaluate Hull Moving Average (HMA) strategy on selected stocks"
    )

    # Override default output
    parser.set_defaults(output="hma_strategy_results.csv")
    # HMA parameters
    parser.add_argument(
        "--hma-period",
        type=int,
        default=20,
        help="HMA period (default: 20). Short-term: 10-15, Swing: 20-30, Long-term: 50-100",
    )
    # Volume parameters
    parser.add_argument(
        "--volume-ma-period",
        type=int,
        default=5,
        help="Volume moving average period (default: 5)",
    )
    parser.add_argument(
        "--volume-threshold",
        type=float,
        default=1.5,
        help="Volume threshold for buy signals (default: 1.5 = 150%% of average)",
    )
    # Slope parameters
    parser.add_argument(
        "--slope-threshold",
        type=float,
        default=0.52,
        help="Minimum slope for HMA upward turn in radians (default: 0.52 ≈ 30 degrees)",
    )
    # Filter parameters
    parser.add_argument(
        "--no-bollinger-filter",
        dest="use_bollinger_filter",
        action="store_false",
        default=True,
        help="Disable Bollinger Bands trend filter (default: enabled)",
    )
    parser.add_argument(
        "--no-macd-filter",
        dest="use_macd_filter",
        action="store_false",
        default=True,
        help="Disable MACD trend filter (default: enabled)",
    )
    # Stop loss parameters
    parser.add_argument(
        "--stop-loss-pct",
        type=float,
        default=0.02,
        help="Stop loss percentage below HMA (default: 0.02 = 2%%)",
    )
    args = parser.parse_args()

    # Create strategy-specific parameters
    strategy_params = {
        "hma_period": args.hma_period,
        "volume_ma_period": args.volume_ma_period,
        "volume_threshold": args.volume_threshold,
        "slope_threshold": args.slope_threshold,
        "use_bollinger_filter": args.use_bollinger_filter,
        "use_macd_filter": args.use_macd_filter,
        "stop_loss_pct": args.stop_loss_pct,
    }

    # Run evaluation using common runner (kline_type defaults to "day")
    return run_strategy_evaluation(
        strategy_class=HMAStrategy,
        strategy_name="HMA",
        default_output="hma_strategy_results.csv",
        strategy_params=strategy_params,
        args=args,
    )


if __name__ == "__main__":
    sys.exit(main())


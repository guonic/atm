#!/usr/bin/env python3
"""
DMI (Directional Movement Index) Strategy Evaluation Tool.

Command-line tool for evaluating DMI strategy on multiple stocks.
"""

import logging
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from atm.analysis.backtest.common_args import create_base_parser
from atm.analysis.backtest.evaluation_runner import run_strategy_evaluation
from atm.trading.strategies.dmi_strategy import DMIStrategy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


def main():
    """Main entry point for DMI strategy evaluation."""
    # Create base parser with common arguments
    parser = create_base_parser(
        description="Evaluate DMI (Directional Movement Index) strategy on selected stocks"
    )

    # Override default output
    parser.set_defaults(output="dmi_strategy_results.csv")

    # DMI parameters
    parser.add_argument(
        "--dmi-period",
        type=int,
        default=14,
        help="DMI calculation period (default: 14)",
    )
    parser.add_argument(
        "--adx-threshold-buy",
        type=float,
        default=20.0,
        help="Minimum ADX for buy signal (default: 20.0)",
    )
    parser.add_argument(
        "--adx-threshold-hold",
        type=float,
        default=25.0,
        help="ADX threshold for holding (default: 25.0)",
    )
    parser.add_argument(
        "--initial-position-pct",
        type=float,
        default=0.3,
        help="Initial position size percentage (default: 0.3 = 30%%)",
    )
    parser.add_argument(
        "--add-position-pct",
        type=float,
        default=0.3,
        help="Additional position size percentage when ADX > threshold (default: 0.3 = 30%%)",
    )

    args = parser.parse_args()

    # Create strategy-specific parameters
    strategy_params = {
        "dmi_period": args.dmi_period,
        "adx_threshold_buy": args.adx_threshold_buy,
        "adx_threshold_hold": args.adx_threshold_hold,
        "initial_position_pct": args.initial_position_pct,
        "add_position_pct": args.add_position_pct,
    }

    # Run evaluation using common runner (kline_type defaults to "day")
    return run_strategy_evaluation(
        strategy_class=DMIStrategy,
        strategy_name="DMI",
        default_output="dmi_strategy_results.csv",
        strategy_params=strategy_params,
        args=args,
    )


if __name__ == "__main__":
    sys.exit(main())


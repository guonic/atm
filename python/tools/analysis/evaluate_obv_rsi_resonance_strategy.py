#!/usr/bin/env python3
"""
OBV+RSI Dual Indicator Resonance Strategy Evaluation Tool.

Command-line tool for evaluating OBV+RSI Resonance strategy on multiple stocks.
"""

import logging
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from nq.analysis.backtest.common_args import create_base_parser
from nq.analysis.backtest.evaluation_runner import run_strategy_evaluation
from nq.trading.strategies.obv_rsi_resonance_strategy import OBVRSIResonanceStrategy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


def main():
    """Main entry point for OBV+RSI Resonance strategy evaluation."""
    # Create base parser with common arguments
    parser = create_base_parser(
        description="Evaluate OBV+RSI Dual Indicator Resonance strategy on selected stocks"
    )

    # Override default output
    parser.set_defaults(output="obv_rsi_resonance_results.csv")
    parser.set_defaults(sort_by="total_return")

    # OBV parameters
    parser.add_argument(
        "--obv-long-ma-period",
        type=int,
        default=30,
        help="OBV long-term moving average period (default: 30)",
    )
    parser.add_argument(
        "--obv-short-ma-period",
        type=int,
        default=5,
        help="OBV short-term support MA period (default: 5)",
    )
    parser.add_argument(
        "--obv-rise-periods",
        type=int,
        default=3,
        help="Minimum periods for OBV upward trend (default: 3)",
    )

    # RSI parameters
    parser.add_argument(
        "--rsi-period",
        type=int,
        default=14,
        help="RSI calculation period (default: 14)",
    )
    parser.add_argument(
        "--rsi-middle-low",
        type=float,
        default=40,
        help="RSI middle range lower boundary (default: 40)",
    )
    parser.add_argument(
        "--rsi-middle-high",
        type=float,
        default=60,
        help="RSI middle range upper boundary (default: 60)",
    )
    parser.add_argument(
        "--rsi-overbought",
        type=float,
        default=70,
        help="RSI overbought threshold (default: 70)",
    )
    parser.add_argument(
        "--rsi-oversold",
        type=float,
        default=30,
        help="RSI oversold threshold (default: 30)",
    )

    # Resonance parameters
    parser.add_argument(
        "--resonance-time-diff",
        type=int,
        default=2,
        help="Max time difference for synchronous breakout (default: 2)",
    )

    # Volume parameters
    parser.add_argument(
        "--volume-ma-period",
        type=int,
        default=20,
        help="Volume moving average period (default: 20)",
    )
    parser.add_argument(
        "--volume-increase-ratio",
        type=float,
        default=1.2,
        help="Minimum volume increase ratio (default: 1.2)",
    )

    # Parse arguments
    args = parser.parse_args()

    # Build strategy parameters
    strategy_params = {
        "obv_long_ma_period": args.obv_long_ma_period,
        "obv_short_ma_period": args.obv_short_ma_period,
        "obv_rise_periods": args.obv_rise_periods,
        "rsi_period": args.rsi_period,
        "rsi_middle_low": args.rsi_middle_low,
        "rsi_middle_high": args.rsi_middle_high,
        "rsi_overbought": args.rsi_overbought,
        "rsi_oversold": args.rsi_oversold,
        "resonance_time_diff": args.resonance_time_diff,
        "volume_ma_period": args.volume_ma_period,
        "volume_increase_ratio": args.volume_increase_ratio,
    }

    # Run evaluation
    exit_code = run_strategy_evaluation(
        strategy_class=OBVRSIResonanceStrategy,
        strategy_name="OBV+RSI Resonance",
        default_output="obv_rsi_resonance_results.csv",
        strategy_params=strategy_params,
        args=args,
    )

    sys.exit(exit_code)


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""
Optimized CCI Strategy Evaluation Tool.

Command-line tool for evaluating optimized CCI strategy on multiple stocks.
"""

import logging
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from nq.analysis.backtest.common_args import create_base_parser
from nq.analysis.backtest.evaluation_runner import run_strategy_evaluation
from nq.trading.strategies.cci_strategy_optimized import CCIStrategyOptimized

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


def main():
    """Main entry point for optimized CCI strategy evaluation."""
    # Create base parser with common arguments
    parser = create_base_parser(
        description="Evaluate Optimized CCI strategy on selected stocks"
    )

    # Override default output
    parser.set_defaults(output="cci_strategy_optimized_results.csv")
    # CCI parameters
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
    # Trend filter parameters
    parser.add_argument(
        "--ma-period",
        type=int,
        default=30,
        help="Moving average period for trend filter (default: 30)",
    )
    parser.add_argument(
        "--no-trend-filter",
        dest="use_trend_filter",
        action="store_false",
        default=True,
        help="Disable trend filter (default: enabled)",
    )
    # Risk management parameters
    parser.add_argument(
        "--stop-loss-pct",
        type=float,
        default=0.05,
        help="Stop loss percentage (default: 0.05 = 5%%)",
    )
    parser.add_argument(
        "--take-profit-pct",
        type=float,
        default=0.15,
        help="Take profit percentage (default: 0.15 = 15%%)",
    )
    # Volume confirmation parameters
    parser.add_argument(
        "--volume-ma-period",
        type=int,
        default=20,
        help="Volume moving average period (default: 20)",
    )
    parser.add_argument(
        "--volume-threshold",
        type=float,
        default=1.0,
        help="Volume must be above this multiple of average (default: 1.0)",
    )
    parser.add_argument(
        "--use-volume-filter",
        action="store_true",
        default=False,
        help="Enable volume filter (default: False)",
    )
    # Volatility filter parameters
    parser.add_argument(
        "--atr-period",
        type=int,
        default=14,
        help="ATR period for volatility filter (default: 14)",
    )
    parser.add_argument(
        "--atr-threshold",
        type=float,
        default=0.0,
        help="Minimum ATR multiplier for volatility filter (default: 0.0, disabled)",
    )
    parser.add_argument(
        "--use-atr-filter",
        action="store_true",
        default=False,
        help="Enable ATR volatility filter (default: False)",
    )
    # Divergence parameters
    parser.add_argument(
        "--divergence-lookback",
        type=int,
        default=20,
        help="Divergence lookback period (default: 20)",
    )
    args = parser.parse_args()

    # Create strategy-specific parameters
    strategy_params = {
        "cci_period": args.cci_period,
        "overbought": args.overbought,
        "oversold": args.oversold,
        "ma_period": args.ma_period,
        "stop_loss_pct": args.stop_loss_pct,
        "take_profit_pct": args.take_profit_pct,
        "volume_ma_period": args.volume_ma_period,
        "volume_threshold": args.volume_threshold,
        "atr_period": args.atr_period,
        "atr_threshold": args.atr_threshold,
        "divergence_lookback": args.divergence_lookback,
        "use_trend_filter": args.use_trend_filter,
        "use_volume_filter": args.use_volume_filter,
        "use_atr_filter": args.use_atr_filter,
    }

    # Run evaluation using common runner
    return run_strategy_evaluation(
        strategy_class=CCIStrategyOptimized,
        strategy_name="CCI Optimized",
        default_output="cci_strategy_optimized_results.csv",
        strategy_params=strategy_params,
        args=args,
    )


if __name__ == "__main__":
    sys.exit(main())


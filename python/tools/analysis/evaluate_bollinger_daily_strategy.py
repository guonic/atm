#!/usr/bin/env python3
"""
Daily Bollinger Bands Strategy Evaluation Tool.

Command-line tool for evaluating daily Bollinger Bands strategy on multiple stocks.
"""

import logging
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from atm.analysis.backtest.common_args import create_base_parser
from atm.analysis.backtest.evaluation_runner import run_strategy_evaluation
from atm.trading.strategy.bollinger_daily_strategy import BollingerDailyStrategy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


def main():
    """Main entry point for Daily Bollinger Bands strategy evaluation."""
    # Create base parser with common arguments
    parser = create_base_parser(
        description="Evaluate Daily Bollinger Bands strategy on selected stocks"
    )

    # Override default output (kline_type defaults to "day")
    parser.set_defaults(output="bollinger_daily_results.csv")
    # Bollinger Bands parameters
    parser.add_argument(
        "--period",
        type=int,
        default=20,
        help="Bollinger Bands period (default: 20)",
    )
    parser.add_argument(
        "--devfactor",
        type=float,
        default=2.0,
        help="Bollinger Bands deviation factor (default: 2.0)",
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
        default=0.6,
        help="Volume threshold for buy signals (default: 0.6 = 60%% of average)",
    )
    parser.add_argument(
        "--volume-threshold-sell",
        type=float,
        default=1.5,
        help="Volume threshold for sell signals (default: 1.5 = 150%% of average)",
    )
    # Holding period parameters
    parser.add_argument(
        "--max-holding-days",
        type=int,
        default=20,
        help="Maximum holding period in days (default: 20)",
    )
    parser.add_argument(
        "--band-tolerance",
        type=float,
        default=0.01,
        help="Price tolerance for band detection (default: 0.01 = 1%%)",
    )
    args = parser.parse_args()

    # Create strategy-specific parameters
    strategy_params = {
        "period": args.period,
        "devfactor": args.devfactor,
        "volume_ma_period": args.volume_ma_period,
        "volume_threshold_buy": args.volume_threshold_buy,
        "volume_threshold_sell": args.volume_threshold_sell,
        "max_holding_days": args.max_holding_days,
        "band_tolerance": args.band_tolerance,
    }

    # Run evaluation using common runner (kline_type defaults to "day")
    return run_strategy_evaluation(
        strategy_class=BollingerDailyStrategy,
        strategy_name="Bollinger Daily",
        default_output="bollinger_daily_results.csv",
        strategy_params=strategy_params,
        args=args,
    )


if __name__ == "__main__":
    sys.exit(main())


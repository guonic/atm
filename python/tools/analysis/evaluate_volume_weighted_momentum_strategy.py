#!/usr/bin/env python3
"""
Volume Weighted Momentum Strategy Evaluation Tool.

Command-line tool for evaluating Volume Weighted Momentum strategy on multiple stocks.
"""

import logging
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from nq.analysis.backtest.common_args import create_base_parser
from nq.analysis.backtest.evaluation_runner import run_strategy_evaluation
from nq.trading.strategies.volume_weighted_momentum_strategy import (
    VolumeWeightedMomentumStrategy,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


def main():
    """Main entry point for Volume Weighted Momentum strategy evaluation."""
    # Create base parser with common arguments
    parser = create_base_parser(
        description="Evaluate Volume Weighted Momentum strategy on selected stocks"
    )

    # Override default output
    parser.set_defaults(output="volume_weighted_momentum_results.csv")
    # Momentum parameters
    parser.add_argument(
        "--mom-len",
        type=int,
        default=10,
        help="Momentum period (default: 10)",
    )
    parser.add_argument(
        "--avg-len",
        type=int,
        default=20,
        help="EMA averaging period for VWM (default: 20)",
    )
    # ATR parameters
    parser.add_argument(
        "--atr-period",
        type=int,
        default=14,
        help="ATR period (default: 14)",
    )
    parser.add_argument(
        "--atr-multiplier",
        type=float,
        default=0.5,
        help="ATR multiplier for band calculation (default: 0.5)",
    )
    # Band parameters
    parser.add_argument(
        "--band-period",
        type=int,
        default=20,
        help="Moving average period for bands (default: 20)",
    )
    # Risk parameters
    parser.add_argument(
        "--risk-per-trade",
        type=float,
        default=0.02,
        help="Risk per trade as percentage of capital (default: 0.02 = 2%%)",
    )
    parser.add_argument(
        "--no-atr-filter",
        dest="use_atr_filter",
        action="store_false",
        default=True,
        help="Disable ATR volatility filter (default: enabled)",
    )
    parser.add_argument(
        "--allow-short",
        action="store_true",
        default=False,
        help="Allow short positions (default: False, suitable for A-share market)",
    )
    args = parser.parse_args()

    # Create strategy-specific parameters
    strategy_params = {
        "mom_len": args.mom_len,
        "avg_len": args.avg_len,
        "atr_period": args.atr_period,
        "atr_multiplier": args.atr_multiplier,
        "band_period": args.band_period,
        "risk_per_trade": args.risk_per_trade,
        "use_atr_filter": args.use_atr_filter,
        "allow_short": args.allow_short,
    }

    # Run evaluation using common runner (kline_type defaults to "day")
    return run_strategy_evaluation(
        strategy_class=VolumeWeightedMomentumStrategy,
        strategy_name="Volume Weighted Momentum",
        default_output="volume_weighted_momentum_results.csv",
        strategy_params=strategy_params,
        args=args,
    )


if __name__ == "__main__":
    sys.exit(main())


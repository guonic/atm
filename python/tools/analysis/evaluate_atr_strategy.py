#!/usr/bin/env python3
"""
ATR Strategy Evaluation Tool.

Command-line tool for evaluating ATR-based strategy on multiple stocks.
"""

import logging
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from atm.analysis.backtest.common_args import create_base_parser
from atm.analysis.backtest.evaluation_runner import run_strategy_evaluation
from atm.trading.strategies.atr_strategy import ATRStrategy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


def main():
    """Main entry point for ATR strategy evaluation."""
    # Create base parser with common arguments
    parser = create_base_parser(
        description="Evaluate ATR-based strategy on selected stocks"
    )

    # Override default output
    parser.set_defaults(output="atr_strategy_results.csv")
    # ATR parameters
    parser.add_argument(
        "--atr-period",
        type=int,
        default=14,
        help="ATR period (default: 14)",
    )
    parser.add_argument(
        "--stop-loss-multiplier",
        type=float,
        default=1.5,
        help="Stop loss multiplier (default: 1.5)",
    )
    parser.add_argument(
        "--take-profit-1-multiplier",
        type=float,
        default=2.0,
        help="First take profit multiplier (default: 2.0)",
    )
    parser.add_argument(
        "--take-profit-2-multiplier",
        type=float,
        default=3.0,
        help="Second take profit multiplier (default: 3.0)",
    )
    parser.add_argument(
        "--take-profit-1-size",
        type=float,
        default=0.3,
        help="Position size to reduce at first target (default: 0.3 = 30%%)",
    )
    parser.add_argument(
        "--take-profit-2-size",
        type=float,
        default=0.3,
        help="Position size to reduce at second target (default: 0.3 = 30%%)",
    )
    parser.add_argument(
        "--risk-per-trade",
        type=float,
        default=0.01,
        help="Risk percentage per trade (default: 0.01 = 1%%)",
    )
    # Trend filter parameters
    parser.add_argument(
        "--ma-period",
        type=int,
        default=20,
        help="Moving average period for trend filter (default: 20)",
    )
    parser.add_argument(
        "--no-trend-filter",
        dest="use_trend_filter",
        action="store_false",
        default=True,
        help="Disable trend filter (default: enabled)",
    )
    parser.add_argument(
        "--no-breakout-confirmation",
        dest="use_breakout_confirmation",
        action="store_false",
        default=True,
        help="Disable breakout confirmation (default: enabled)",
    )
    parser.add_argument(
        "--atr-expansion-threshold",
        type=float,
        default=1.1,
        help="ATR expansion threshold for trend identification (default: 1.1)",
    )
    args = parser.parse_args()

    # Create strategy-specific parameters
    strategy_params = {
        "atr_period": args.atr_period,
        "stop_loss_multiplier": args.stop_loss_multiplier,
        "take_profit_1_multiplier": args.take_profit_1_multiplier,
        "take_profit_2_multiplier": args.take_profit_2_multiplier,
        "take_profit_1_size": args.take_profit_1_size,
        "take_profit_2_size": args.take_profit_2_size,
        "risk_per_trade": args.risk_per_trade,
        "ma_period": args.ma_period,
        "use_trend_filter": args.use_trend_filter,
        "use_breakout_confirmation": args.use_breakout_confirmation,
        "atr_expansion_threshold": args.atr_expansion_threshold,
    }

    # Run evaluation using common runner
    return run_strategy_evaluation(
        strategy_class=ATRStrategy,
        strategy_name="ATR",
        default_output="atr_strategy_results.csv",
        strategy_params=strategy_params,
        args=args,
    )


if __name__ == "__main__":
    sys.exit(main())


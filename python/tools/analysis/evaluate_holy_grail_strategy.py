#!/usr/bin/env python3
"""Evaluation script for Holy Grail trend pullback strategy."""

import logging
import sys
from typing import Any, Dict

from nq.analysis.backtest.common_args import create_base_parser
from nq.analysis.backtest.evaluation_runner import run_strategy_evaluation
from nq.trading.strategies.holy_grail_strategy import HolyGrailStrategy

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> int:
    """Main entry point for Holy Grail strategy evaluation."""
    parser = create_base_parser(
        description="Evaluate Holy Grail trend pullback strategy on selected stocks"
    )
    parser.set_defaults(output="holy_grail_results.csv")
    parser.set_defaults(sort_by="total_return")

    parser.add_argument(
        "--adx-period",
        type=int,
        default=14,
        help="ADX period (default: 14)",
    )
    parser.add_argument(
        "--adx-threshold",
        type=float,
        default=30.0,
        help="ADX threshold for trend strength (default: 30)",
    )
    parser.add_argument(
        "--adx-confirm-bars",
        type=int,
        default=3,
        help="Bars required above threshold to confirm trend (default: 3)",
    )
    parser.add_argument(
        "--ema-period",
        type=int,
        default=20,
        help="EMA period for pullback reference (default: 20)",
    )
    parser.add_argument(
        "--swing-lookback",
        type=int,
        default=10,
        help="Lookback for swing-low trailing stop (default: 10)",
    )

    args = parser.parse_args()

    strategy_params: Dict[str, Any] = {
        "adx_period": args.adx_period,
        "adx_threshold": args.adx_threshold,
        "adx_confirm_bars": args.adx_confirm_bars,
        "ema_period": args.ema_period,
        "swing_lookback": args.swing_lookback,
    }

    return run_strategy_evaluation(
        strategy_class=HolyGrailStrategy,
        strategy_name="HolyGrail",
        default_output=args.output,
        strategy_params=strategy_params,
        args=args,
    )


if __name__ == "__main__":
    sys.exit(main())


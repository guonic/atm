#!/usr/bin/env python3
"""
ATR 动态止损趋势策略评估脚本。
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict

from atm.analysis.backtest.common_args import create_base_parser
from atm.analysis.backtest.evaluation_runner import run_strategy_evaluation
from atm.trading.strategies.atr_dynamic_stop_strategy import ATRDynamicStopStrategy

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> int:
    """Main entry for evaluating ATR 动态止损策略。"""
    parser = create_base_parser(
        description="Evaluate ATR dynamic stop strategy (trend + ATR trailing stop)"
    )
    parser.set_defaults(output="atr_dynamic_stop_results.csv")
    parser.set_defaults(sort_by="total_return")

    parser.add_argument(
        "--atr-period",
        type=int,
        default=14,
        help="ATR period (default: 14)",
    )
    parser.add_argument(
        "--atr-multiplier",
        type=float,
        default=1.8,
        help="ATR multiplier for stop distance (default: 1.8)",
    )
    parser.add_argument(
        "--ma-period",
        type=int,
        default=20,
        help="MA period for trend filter (default: 20)",
    )
    parser.add_argument(
        "--min-atr",
        type=float,
        default=0.0,
        help="Minimum ATR to avoid too-tight stops (default: 0)",
    )

    args = parser.parse_args()

    strategy_params: Dict[str, Any] = {
        "atr_period": args.atr_period,
        "atr_multiplier": args.atr_multiplier,
        "ma_period": args.ma_period,
        "min_atr": args.min_atr,
    }

    return run_strategy_evaluation(
        strategy_class=ATRDynamicStopStrategy,
        strategy_name="ATR Dynamic Stop",
        default_output=args.output,
        strategy_params=strategy_params,
        args=args,
    )


if __name__ == "__main__":
    sys.exit(main())


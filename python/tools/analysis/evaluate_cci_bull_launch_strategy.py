#!/usr/bin/env python3
"""
CCI 牛股启动策略评估脚本。
"""

import logging
import sys
from pathlib import Path
from typing import Dict, Any

from nq.analysis.backtest.common_args import create_base_parser
from nq.analysis.backtest.evaluation_runner import run_strategy_evaluation
from nq.trading.strategies.cci_bull_launch_strategy import CCIBullLaunchStrategy

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> int:
    """Main entry for evaluating CCI 牛股启动策略。"""
    parser = create_base_parser(
        description="Evaluate CCI Bull Launch strategy on selected stocks",
    )
    parser.set_defaults(output="cci_bull_launch_results.csv")

    parser.add_argument(
        "--cci-period",
        type=int,
        default=20,
        help="CCI period (default: 20)",
    )
    parser.add_argument(
        "--ma-period",
        type=int,
        default=5,
        help="MA period used for price confirmation (default: 5)",
    )
    parser.add_argument(
        "--cci-buy-level",
        type=float,
        default=100.0,
        help="CCI buy threshold (default: 100)",
    )
    parser.add_argument(
        "--cci-sell-level",
        type=float,
        default=100.0,
        help="CCI sell threshold (default: 100)",
    )

    args = parser.parse_args()

    strategy_params: Dict[str, Any] = {
        "cci_period": args.cci_period,
        "ma_period": args.ma_period,
        "cci_buy_level": args.cci_buy_level,
        "cci_sell_level": args.cci_sell_level,
    }

    return run_strategy_evaluation(
        strategy_class=CCIBullLaunchStrategy,
        strategy_name="CCI Bull Launch",
        default_output=args.output,
        strategy_params=strategy_params,
        args=args,
    )


if __name__ == "__main__":
    sys.exit(main())


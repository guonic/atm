#!/usr/bin/env python3
"""Evaluation script for CMF flow + trend resonance strategy."""

import logging
import sys
from typing import Any, Dict

from atm.analysis.backtest.common_args import create_base_parser
from atm.analysis.backtest.evaluation_runner import run_strategy_evaluation
from atm.trading.strategies.cmf_resonance_strategy import CMFResonanceStrategy

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> int:
    """Main entry for CMF resonance strategy evaluation."""
    parser = create_base_parser(
        description="Evaluate CMF + trend resonance strategy on selected stocks"
    )
    parser.set_defaults(output="cmf_resonance_results.csv")
    parser.set_defaults(sort_by="total_return")

    parser.add_argument(
        "--cmf-period",
        type=int,
        default=20,
        help="CMF period (default: 20)",
    )
    parser.add_argument(
        "--cmf-threshold",
        type=float,
        default=0.0,
        help="CMF threshold for net inflow confirmation (default: 0.0)",
    )
    parser.add_argument(
        "--ma-period",
        type=int,
        default=20,
        help="MA period for trend filter (default: 20)",
    )
    parser.add_argument(
        "--breakout-lookback",
        type=int,
        default=20,
        help="Lookback window for price breakout (default: 20)",
    )

    args = parser.parse_args()

    strategy_params: Dict[str, Any] = {
        "cmf_period": args.cmf_period,
        "cmf_threshold": args.cmf_threshold,
        "ma_period": args.ma_period,
        "breakout_lookback": args.breakout_lookback,
    }

    return run_strategy_evaluation(
        strategy_class=CMFResonanceStrategy,
        strategy_name="CMF Resonance",
        default_output=args.output,
        strategy_params=strategy_params,
        args=args,
    )


if __name__ == "__main__":
    sys.exit(main())


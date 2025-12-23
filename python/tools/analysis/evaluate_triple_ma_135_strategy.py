#!/usr/bin/env python3
"""Evaluation script for Triple MA 13/34/55 trend pullback strategy."""

import logging
import sys
from typing import Any, Dict

from atm.analysis.backtest.common_args import create_base_parser
from atm.analysis.backtest.evaluation_runner import run_strategy_evaluation
from atm.trading.strategies.triple_ma_135_strategy import TripleMA135Strategy

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> int:
    """Main entry for Triple MA 13/34/55 strategy evaluation."""
    parser = create_base_parser(
        description="Evaluate Triple MA 13/34/55 trend pullback strategy on selected stocks"
    )
    parser.set_defaults(output="triple_ma_135_results.csv")
    parser.set_defaults(sort_by="total_return")

    parser.add_argument(
        "--ema-short",
        type=int,
        default=13,
        help="Short EMA period (default: 13)",
    )
    parser.add_argument(
        "--ema-mid",
        type=int,
        default=34,
        help="Mid EMA period (default: 34)",
    )
    parser.add_argument(
        "--ema-long",
        type=int,
        default=55,
        help="Long EMA period (default: 55)",
    )
    parser.add_argument(
        "--cross-gap",
        type=int,
        default=5,
        help="Max bars between dual golden crosses (default: 5)",
    )
    parser.add_argument(
        "--pullback-tolerance",
        type=float,
        default=0.01,
        help="Tolerance for price near EMA(mid) when entering (default: 0.01)",
    )

    args = parser.parse_args()

    strategy_params: Dict[str, Any] = {
        "ema_short": args.ema_short,
        "ema_mid": args.ema_mid,
        "ema_long": args.ema_long,
        "cross_gap": args.cross_gap,
        "pullback_tolerance": args.pullback_tolerance,
    }

    return run_strategy_evaluation(
        strategy_class=TripleMA135Strategy,
        strategy_name="TripleMA135",
        default_output=args.output,
        strategy_params=strategy_params,
        args=args,
    )


if __name__ == "__main__":
    sys.exit(main())


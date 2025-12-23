#!/usr/bin/env python3
"""Evaluation script for RSI multi-signal strategy."""

import logging
import sys
from typing import Any, Dict

from atm.analysis.backtest.common_args import create_base_parser
from atm.analysis.backtest.evaluation_runner import run_strategy_evaluation
from atm.trading.strategies.rsi_multi_signal_strategy import RSIMultiSignalStrategy

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> int:
    """Main entry for RSI multi-signal evaluation."""
    parser = create_base_parser(
        description="Evaluate RSI multi-signal strategy (oversold rebound + low-zone golden cross; overbought drop + high-zone death cross)"
    )
    parser.set_defaults(output="rsi_multi_signal_results.csv")
    parser.set_defaults(sort_by="total_return")

    parser.add_argument("--rsi-fast", type=int, default=6, help="Fast RSI period (default: 6)")
    parser.add_argument("--rsi-slow", type=int, default=14, help="Slow RSI period (default: 14)")
    parser.add_argument(
        "--oversold",
        type=float,
        default=20.0,
        help="Oversold threshold for rebound buy (default: 20)",
    )
    parser.add_argument(
        "--overbought",
        type=float,
        default=80.0,
        help="Overbought threshold for profit-taking sell (default: 80)",
    )
    parser.add_argument(
        "--stop-loss-pct",
        type=float,
        default=0.05,
        help="Stop loss percent from entry price (default: 0.05)",
    )
    parser.add_argument(
        "--take-profit-pct",
        type=float,
        default=0.1,
        help="Take profit percent from entry price (default: 0.1)",
    )

    args = parser.parse_args()

    strategy_params: Dict[str, Any] = {
        "rsi_fast": args.rsi_fast,
        "rsi_slow": args.rsi_slow,
        "oversold": args.oversold,
        "overbought": args.overbought,
        "stop_loss_pct": args.stop_loss_pct,
        "take_profit_pct": args.take_profit_pct,
    }

    return run_strategy_evaluation(
        strategy_class=RSIMultiSignalStrategy,
        strategy_name="RSIMultiSignal",
        default_output=args.output,
        strategy_params=strategy_params,
        args=args,
    )


if __name__ == "__main__":
    sys.exit(main())


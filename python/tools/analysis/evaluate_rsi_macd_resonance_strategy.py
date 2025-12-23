#!/usr/bin/env python3
"""
RSI + MACD Resonance Strategy Evaluation Tool.

Command-line tool for evaluating RSI + MACD Resonance strategy on multiple stocks.
"""

import logging
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from atm.analysis.backtest.common_args import create_base_parser
from atm.analysis.backtest.evaluation_runner import run_strategy_evaluation
from atm.trading.strategy.rsi_macd_resonance_strategy import RSIMACDResonanceStrategy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


def main():
    """Main entry point for RSI + MACD Resonance strategy evaluation."""
    # Create base parser with common arguments
    parser = create_base_parser(
        description="Evaluate RSI + MACD Resonance strategy on selected stocks"
    )

    # Override default output
    parser.set_defaults(output="rsi_macd_resonance_results.csv")
    parser.set_defaults(sort_by="win_rate")
    # RSI parameters
    parser.add_argument(
        "--rsi-period",
        type=int,
        default=14,
        help="RSI calculation period (default: 14)",
    )
    parser.add_argument(
        "--rsi-oversold",
        type=float,
        default=30,
        help="RSI oversold threshold (default: 30)",
    )
    parser.add_argument(
        "--rsi-overbought",
        type=float,
        default=70,
        help="RSI overbought threshold (default: 70)",
    )
    parser.add_argument(
        "--rsi-midline",
        type=float,
        default=50,
        help="RSI strength/weakness dividing line (default: 50)",
    )
    # MACD parameters
    parser.add_argument(
        "--macd-fast",
        type=int,
        default=12,
        help="MACD fast period (default: 12)",
    )
    parser.add_argument(
        "--macd-slow",
        type=int,
        default=26,
        help="MACD slow period (default: 26)",
    )
    parser.add_argument(
        "--macd-signal",
        type=int,
        default=9,
        help="MACD signal period (default: 9)",
    )
    # Strategy parameters
    parser.add_argument(
        "--divergence-lookback",
        type=int,
        default=20,
        help="Divergence lookback period (default: 20)",
    )
    parser.add_argument(
        "--use-sensitive-entry",
        action="store_true",
        default=True,
        help="Use sensitive entry signal (MACD bar color change) (default: True)",
    )
    parser.add_argument(
        "--no-sensitive-entry",
        dest="use_sensitive_entry",
        action="store_false",
        help="Use classic entry signal instead of sensitive entry",
    )
    parser.add_argument(
        "--stop-loss-atr-multiplier",
        type=float,
        default=2.0,
        help="Stop loss ATR multiplier (default: 2.0)",
    )

    args = parser.parse_args()

    # Create strategy-specific parameters
    strategy_params = {
        "rsi_period": args.rsi_period,
        "rsi_oversold": args.rsi_oversold,
        "rsi_overbought": args.rsi_overbought,
        "rsi_midline": args.rsi_midline,
        "macd_fast": args.macd_fast,
        "macd_slow": args.macd_slow,
        "macd_signal": args.macd_signal,
        "divergence_lookback": args.divergence_lookback,
        "use_sensitive_entry": args.use_sensitive_entry,
        "stop_loss_atr_multiplier": args.stop_loss_atr_multiplier,
    }

    # Run evaluation using common runner
    return run_strategy_evaluation(
        strategy_class=RSIMACDResonanceStrategy,
        strategy_name="RSI + MACD Resonance",
        default_output="rsi_macd_resonance_results.csv",
        strategy_params=strategy_params,
        args=args,
    )


if __name__ == "__main__":
    sys.exit(main())


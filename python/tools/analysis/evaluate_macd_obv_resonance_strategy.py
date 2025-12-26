#!/usr/bin/env python3
"""
MACD + OBV Dual Indicator Resonance Strategy Evaluation Tool.

Command-line tool for evaluating MACD + OBV Resonance strategy on multiple stocks.
"""

import logging
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from nq.analysis.backtest.common_args import create_base_parser
from nq.analysis.backtest.evaluation_runner import run_strategy_evaluation
from nq.trading.strategies.macd_obv_resonance_strategy import MACDOBVResonanceStrategy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


def main():
    """Main entry point for MACD + OBV Resonance strategy evaluation."""
    # Create base parser with common arguments
    parser = create_base_parser(
        description="Evaluate MACD + OBV Dual Indicator Resonance strategy on selected stocks"
    )

    # Override default output
    parser.set_defaults(output="macd_obv_resonance_results.csv")
    parser.set_defaults(sort_by="total_return")

    # MACD parameters (short-term for faster signals)
    parser.add_argument(
        "--macd-fast",
        type=int,
        default=6,
        help="MACD fast period (default: 6)",
    )
    parser.add_argument(
        "--macd-slow",
        type=int,
        default=13,
        help="MACD slow period (default: 13)",
    )
    parser.add_argument(
        "--macd-signal",
        type=int,
        default=5,
        help="MACD signal period (default: 5)",
    )

    # OBV parameters
    parser.add_argument(
        "--obv-lookback",
        type=int,
        default=20,
        help="OBV lookback period for breakout detection (default: 20)",
    )

    # Divergence detection
    parser.add_argument(
        "--divergence-lookback",
        type=int,
        default=20,
        help="Divergence lookback period (default: 20)",
    )

    # Entry thresholds
    parser.add_argument(
        "--macd-hist-expansion-ratio",
        type=float,
        default=1.2,
        help="Minimum MACD histogram expansion ratio (default: 1.2)",
    )

    # Parse arguments
    args = parser.parse_args()

    # Build strategy parameters
    strategy_params = {
        "macd_fast": args.macd_fast,
        "macd_slow": args.macd_slow,
        "macd_signal": args.macd_signal,
        "obv_lookback": args.obv_lookback,
        "divergence_lookback": args.divergence_lookback,
        "macd_hist_expansion_ratio": args.macd_hist_expansion_ratio,
    }

    # Run evaluation
    exit_code = run_strategy_evaluation(
        strategy_class=MACDOBVResonanceStrategy,
        strategy_name="MACD+OBV Resonance",
        default_output="macd_obv_resonance_results.csv",
        strategy_params=strategy_params,
        args=args,
    )

    sys.exit(exit_code)


if __name__ == "__main__":
    main()


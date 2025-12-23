#!/usr/bin/env python3
"""
CCI Strategy Evaluation Tool.

Command-line tool for evaluating CCI strategy on multiple stocks.
"""

import logging
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from atm.analysis.backtest.common_args import create_base_parser
from atm.analysis.backtest.evaluation_runner import run_strategy_evaluation
from atm.trading.strategy.cci_strategy import CCIStrategy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


def main():
    """Main entry point for CCI strategy evaluation."""
    # Create base parser with common arguments
    parser = create_base_parser(
        description="Evaluate CCI strategy on selected stocks"
    )

    # Override default output and market cap defaults
    parser.set_defaults(output="cci_strategy_results.csv")
    parser.set_defaults(min_market_cap=5000000)  # 500B CNY
    parser.set_defaults(max_market_cap=10000000)  # 1000B CNY
    parser.add_argument(
        "--cci-period",
        type=int,
        default=20,
        help="CCI period (default: 20)",
    )
    parser.add_argument(
        "--overbought",
        type=int,
        default=100,
        help="Overbought threshold (default: 100)",
    )
    parser.add_argument(
        "--oversold",
        type=int,
        default=-100,
        help="Oversold threshold (default: -100)",
    )
    parser.add_argument(
        "--divergence-lookback",
        type=int,
        default=20,
        help="Divergence lookback period (default: 20)",
    )
    args = parser.parse_args()

    # Create strategy-specific parameters
    strategy_params = {
        "cci_period": args.cci_period,
        "overbought": args.overbought,
        "oversold": args.oversold,
        "divergence_lookback": args.divergence_lookback,
    }

    # Run evaluation using common runner
    return run_strategy_evaluation(
        strategy_class=CCIStrategy,
        strategy_name="CCI",
        default_output="cci_strategy_results.csv",
        strategy_params=strategy_params,
        args=args,
    )


if __name__ == "__main__":
    sys.exit(main())


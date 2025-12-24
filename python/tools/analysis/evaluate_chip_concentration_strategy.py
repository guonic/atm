#!/usr/bin/env python3
"""
Chip Concentration (12%) Strategy Evaluation Tool.

Command-line tool for evaluating 12% Chip Concentration Stock Selection strategy.
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from atm.analysis.backtest.common_args import create_base_parser
from atm.analysis.backtest.evaluation_runner import run_strategy_evaluation
from atm.trading.strategies.chip_concentration_strategy import ChipConcentrationStrategy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


def main() -> int:
    """Main entry point for Chip Concentration strategy evaluation."""
    parser = create_base_parser(
        description="Evaluate 12% Chip Concentration Stock Selection strategy"
    )

    # Strategy-specific arguments
    parser.add_argument(
        "--concentration-threshold",
        type=float,
        default=0.12,
        help="Chip concentration threshold (default: 0.12 = 12%%)",
    )
    parser.add_argument(
        "--pattern1-min-days",
        type=int,
        default=20,
        help="Minimum consolidation days for pattern 1 (default: 20)",
    )
    parser.add_argument(
        "--pattern2-min-days",
        type=int,
        default=30,
        help="Minimum consolidation days for pattern 2 (default: 30)",
    )
    parser.add_argument(
        "--volume-shrink-threshold",
        type=float,
        default=0.3,
        help="Volume shrink threshold during consolidation (default: 0.3 = 30%%)",
    )
    parser.add_argument(
        "--volume-breakout-threshold",
        type=float,
        default=1.5,
        help="Volume increase threshold for breakout (default: 1.5 = 50%% increase)",
    )
    parser.add_argument(
        "--initial-position-pct",
        type=float,
        default=0.3,
        help="Initial position percentage (default: 0.3 = 30%%)",
    )
    parser.add_argument(
        "--add-position-pct",
        type=float,
        default=0.4,
        help="Add-on position percentage (default: 0.4 = 40%%)",
    )
    parser.add_argument(
        "--add-position-days",
        type=int,
        default=3,
        help="Days to wait before add-on (default: 3)",
    )
    parser.add_argument(
        "--stop-loss-pct",
        type=float,
        default=0.05,
        help="Stop loss percentage (default: 0.05 = 5%%)",
    )
    parser.add_argument(
        "--take-profit-pct",
        type=float,
        default=0.15,
        help="Take profit percentage (default: 0.15 = 15%%)",
    )
    parser.set_defaults(output="chip_concentration_results.csv")

    args = parser.parse_args()

    strategy_params: Dict[str, Any] = {
        "chip_peak_indicator": None,  # Will use DummyChipPeakIndicator by default
        "concentration_threshold": args.concentration_threshold,
        "pattern1_min_days": args.pattern1_min_days,
        "pattern2_min_days": args.pattern2_min_days,
        "volume_shrink_threshold": args.volume_shrink_threshold,
        "volume_breakout_threshold": args.volume_breakout_threshold,
        "initial_position_pct": args.initial_position_pct,
        "add_position_pct": args.add_position_pct,
        "add_position_days": args.add_position_days,
        "stop_loss_pct": args.stop_loss_pct,
        "take_profit_pct": args.take_profit_pct,
    }

    return run_strategy_evaluation(
        strategy_class=ChipConcentrationStrategy,
        strategy_name="ChipConcentration",
        default_output=args.output,
        strategy_params=strategy_params,
        args=args,
    )


if __name__ == "__main__":
    sys.exit(main())


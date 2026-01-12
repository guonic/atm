"""
Extract exit model training data from backtest results.

This script extracts position snapshots from trading framework backtest results
and saves them in the format required for training the exit model.
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Extract exit model training data from backtest results"
    )
    parser.add_argument(
        "--backtest_results",
        type=str,
        required=True,
        help="Path to backtest results file (pickle or JSON) or use --from_storage",
    )
    parser.add_argument(
        "--from_storage",
        action="store_true",
        help="Extract from storage backend directly (requires storage instance)",
    )
    parser.add_argument(
        "--start_date",
        type=str,
        required=True,
        help="Start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end_date",
        type=str,
        required=True,
        help="End date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/exit_training_data.csv",
        help="Output CSV file path (default: outputs/exit_training_data.csv)",
    )
    parser.add_argument(
        "--future_days",
        type=int,
        default=3,
        help="Number of future days to look ahead for labels (default: 3)",
    )
    parser.add_argument(
        "--loss_threshold",
        type=float,
        default=-0.03,
        help="Loss threshold for labeling (default: -0.03, i.e., -3%)",
    )
    
    args = parser.parse_args()
    
    # Note: This is a template script
    # In practice, you would:
    # 1. Load backtest results (from file or storage)
    # 2. Extract training data
    # 3. Save to CSV
    
    logger.info("=" * 80)
    logger.info("EXIT MODEL TRAINING DATA EXTRACTION")
    logger.info("=" * 80)
    logger.info(f"Start Date: {args.start_date}")
    logger.info(f"End Date: {args.end_date}")
    logger.info(f"Output: {args.output}")
    logger.info("")
    logger.info("NOTE: This script requires backtest results from trading framework.")
    logger.info("To use this script:")
    logger.info("1. Run backtest using backtest_trading_framework.py")
    logger.info("2. Save backtest results (including storage backend)")
    logger.info("3. Run this script to extract training data")
    logger.info("")
    logger.info("Example workflow:")
    logger.info("  # Step 1: Run backtest")
    logger.info("  python python/examples/backtest_trading_framework.py \\")
    logger.info("      --model_path models/structure_expert_directional.pth \\")
    logger.info("      --exit_model_path models/exit_model.pkl \\")
    logger.info("      --start_date 2025-07-01 \\")
    logger.info("      --end_date 2025-08-01")
    logger.info("")
    logger.info("  # Step 2: Extract training data (after modifying script to save results)")
    logger.info("  python python/trading/backtest/extract_training_data.py \\")
    logger.info("      --backtest_results results.pkl \\")
    logger.info("      --start_date 2025-07-01 \\")
    logger.info("      --end_date 2025-08-01 \\")
    logger.info("      --output outputs/exit_training_data.csv")
    logger.info("=" * 80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

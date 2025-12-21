# -*- coding: utf-8 -*-
"""
backtest_kronos_predictor_example.py

Description:
    Backtest example for Kronos predictor using the backtest framework.
    This script demonstrates how to evaluate prediction models by comparing
    predictions with actual historical data.

Usage:
    python backtest_kronos_predictor_example.py --ts_code 000001.SZ --start_date 2024-01-01 --end_date 2024-06-30

Arguments:
    --ts_code      Stock code in Tushare format (e.g., 000001.SZ, 600000.SH)
    --start_date   Start date of backtest period (YYYY-MM-DD)
    --end_date     End date of backtest period (YYYY-MM-DD)
    --pred_len     Number of trading days to predict in each step (default: 20)
    --lookback     Number of historical days to use (default: 400)
    --step_size    Step size for walk-forward backtest (default: same as pred_len)
    --device       Device to use (default: cpu)
    --model_name   Pretrained model name (default: NeoQuasar/Kronos-base)
    --tokenizer_name Pretrained tokenizer name (default: NeoQuasar/Kronos-Tokenizer-base)
    --save_report  Save backtest report to file (default: False)

Output:
    - Prints backtest metrics and report to console
    - Optionally saves report to file

Example:
    python backtest_kronos_predictor_example.py --ts_code 000001.SZ --start_date 2024-01-01 --end_date 2024-06-30
    python backtest_kronos_predictor_example.py --ts_code 600000.SH --start_date 2024-01-01 --end_date 2024-06-30 --pred_len 10 --step_size 10
"""

import argparse
import logging
import os
from datetime import datetime

import pandas as pd

from atm.analysis.backtest import BacktestReport, PredictorBacktester
from atm.config import load_config
from atm.trading.predictor import KronosPredictor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Default settings
DEFAULT_PRED_LEN = 20
DEFAULT_LOOKBACK = 400
DEFAULT_DEVICE = "cpu"
DEFAULT_MODEL_NAME = "NeoQuasar/Kronos-base"
DEFAULT_TOKENIZER_NAME = "NeoQuasar/Kronos-Tokenizer-base"


def parse_date(date_str: str) -> datetime:
    """
    Parse date string to datetime object.

    Args:
        date_str: Date string in YYYY-MM-DD format.

    Returns:
        Datetime object.

    Raises:
        ValueError: If date format is invalid.
    """
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError as e:
        raise ValueError(f"Invalid date format: {date_str}. Expected YYYY-MM-DD format.") from e


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Backtest Kronos predictor using the backtest framework"
    )
    parser.add_argument(
        "--ts_code",
        type=str,
        default="000001.SZ",
        help="Stock code in Tushare format (e.g., 000001.SZ, 600000.SH)"
    )
    parser.add_argument(
        "--start_date",
        type=str,
        required=True,
        help="Start date of backtest period (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end_date",
        type=str,
        required=True,
        help="End date of backtest period (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--pred_len",
        type=int,
        default=DEFAULT_PRED_LEN,
        help=f"Number of trading days to predict in each step (default: {DEFAULT_PRED_LEN})"
    )
    parser.add_argument(
        "--lookback",
        type=int,
        default=DEFAULT_LOOKBACK,
        help=f"Number of historical days to use (default: {DEFAULT_LOOKBACK})"
    )
    parser.add_argument(
        "--step_size",
        type=int,
        default=None,
        help="Step size for walk-forward backtest (default: same as pred_len)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=DEFAULT_DEVICE,
        help=f"Device to use (default: {DEFAULT_DEVICE})"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help=f"Pretrained model name (default: {DEFAULT_MODEL_NAME})"
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=DEFAULT_TOKENIZER_NAME,
        help=f"Pretrained tokenizer name (default: {DEFAULT_TOKENIZER_NAME})"
    )
    parser.add_argument(
        "--save_report",
        action="store_true",
        help="Save backtest report to file"
    )
    parser.add_argument(
        "--report_dir",
        type=str,
        default="./outputs",
        help="Directory to save report (default: ./outputs)"
    )

    args = parser.parse_args()

    # Parse dates
    try:
        start_date = parse_date(args.start_date)
        end_date = parse_date(args.end_date)
    except ValueError as e:
        logger.error(f"Date parsing error: {e}")
        return

    if start_date >= end_date:
        logger.error("start_date must be before end_date")
        return

    # Load configuration
    try:
        config = load_config()
        db_config = config.database
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        return

    # Create predictor
    logger.info("Initializing Kronos predictor...")
    predictor = KronosPredictor(
        db_config=db_config,
        device=args.device,
        model_name=args.model_name,
        tokenizer_name=args.tokenizer_name,
    )

    # Load model
    try:
        predictor.load_model()
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return

    # Create backtester
    logger.info("Initializing backtester...")
    backtester = PredictorBacktester(
        predictor=predictor,
        db_config=db_config,
    )

    # Run backtest
    logger.info(
        f"Running backtest for {args.ts_code} from {start_date.date()} to {end_date.date()}"
    )

    try:
        result = backtester.run(
            ts_code=args.ts_code,
            start_date=start_date,
            end_date=end_date,
            pred_len=args.pred_len,
            lookback=args.lookback,
            step_size=args.step_size,
        )

        # Print report
        BacktestReport.print_report(result)

        # Save report if requested
        if args.save_report:
            os.makedirs(args.report_dir, exist_ok=True)
            report_file = os.path.join(
                args.report_dir,
                f"backtest_{args.ts_code.replace('.', '_')}_{start_date.date()}_{end_date.date()}.txt"
            )
            BacktestReport.save_report(result, report_file)

            # Also save predictions and actuals to CSV
            predictions_file = os.path.join(
                args.report_dir,
                f"backtest_{args.ts_code.replace('.', '_')}_{start_date.date()}_{end_date.date()}_predictions.csv"
            )
            # Merge predictions and actuals on date
            merged = pd.merge(
                result.predictions,
                result.actuals,
                on="date",
                suffixes=("_pred", "_actual"),
                how="inner"
            )
            merged.to_csv(predictions_file, index=False)
            logger.info(f"Predictions and actuals saved to: {predictions_file}")

        logger.info("Backtest completed successfully")

    except Exception as e:
        logger.error(f"Backtest failed: {e}", exc_info=True)
        return


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check_backtest_dates.py

Check if backtest dates exist in Qlib calendar file.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List
from datetime import datetime

import pandas as pd

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def check_backtest_dates(
    qlib_dir: Path,
    start_date: str,
    end_date: str,
    freq: str = "day",
) -> bool:
    """
    Check if backtest dates exist in calendar file.
    
    Args:
        qlib_dir: Qlib data directory.
        start_date: Start date (YYYY-MM-DD or YYYYMMDD).
        end_date: End date (YYYY-MM-DD or YYYYMMDD).
        freq: Data frequency.
    
    Returns:
        True if all dates exist, False otherwise.
    """
    calendar_file = qlib_dir / "calendars" / f"{freq}.txt"
    
    if not calendar_file.exists():
        logger.error(f"Calendar file not found: {calendar_file}")
        return False
    
    # Read calendar dates
    with open(calendar_file, "r", encoding="utf-8") as f:
        calendar_dates = [line.strip() for line in f if line.strip()]
    
    logger.info(f"Calendar file: {calendar_file}")
    logger.info(f"Total calendar dates: {len(calendar_dates)}")
    if calendar_dates:
        logger.info(f"Calendar range: {calendar_dates[0]} to {calendar_dates[-1]}")
    
    # Parse input dates
    try:
        if len(start_date) == 8 and start_date.isdigit():
            start_dt = pd.to_datetime(start_date, format="%Y%m%d")
        else:
            start_dt = pd.to_datetime(start_date)
        
        if len(end_date) == 8 and end_date.isdigit():
            end_dt = pd.to_datetime(end_date, format="%Y%m%d")
        else:
            end_dt = pd.to_datetime(end_date)
    except Exception as e:
        logger.error(f"Failed to parse dates: {e}")
        return False
    
    logger.info("")
    logger.info("=" * 80)
    logger.info("Checking Backtest Dates")
    logger.info("=" * 80)
    logger.info(f"Start date: {start_date} ({start_dt.strftime('%Y-%m-%d %A')})")
    logger.info(f"End date: {end_date} ({end_dt.strftime('%Y-%m-%d %A')})")
    logger.info("")
    
    # Generate all dates in range (including weekends/holidays for checking)
    date_range = pd.date_range(start=start_dt, end=end_dt, freq="D")
    
    # Check each date
    missing_dates = []
    found_dates = []
    weekend_dates = []
    
    for dt in date_range:
        date_str = dt.strftime("%Y%m%d")
        readable = dt.strftime("%Y-%m-%d (%A)")
        
        # Check if weekend
        if dt.weekday() >= 5:  # Saturday or Sunday
            weekend_dates.append((date_str, readable))
            continue
        
        # Check if in calendar
        if date_str in calendar_dates:
            found_dates.append((date_str, readable))
        else:
            missing_dates.append((date_str, readable))
    
    # Report results
    logger.info("Results:")
    logger.info(f"  ✓ Found in calendar: {len(found_dates)} trading days")
    logger.info(f"  ✗ Missing from calendar: {len(missing_dates)} dates")
    logger.info(f"  ⚠ Weekends (not trading days): {len(weekend_dates)} dates")
    
    if found_dates:
        logger.info("")
        logger.info("Found dates:")
        for date_str, readable in found_dates[:10]:
            logger.info(f"  ✓ {date_str} - {readable}")
        if len(found_dates) > 10:
            logger.info(f"  ... and {len(found_dates) - 10} more")
    
    if missing_dates:
        logger.warning("")
        logger.warning("Missing dates (should be in calendar but are not):")
        for date_str, readable in missing_dates:
            logger.warning(f"  ✗ {date_str} - {readable}")
        logger.warning("")
        logger.warning("These dates are weekdays but not in calendar.")
        logger.warning("Possible reasons:")
        logger.warning("  1. These are holidays (not trading days)")
        logger.warning("  2. Calendar file is incomplete")
        logger.warning("  3. Date format mismatch")
    
    if weekend_dates:
        logger.info("")
        logger.info("Weekend dates (expected to be missing):")
        for date_str, readable in weekend_dates[:5]:
            logger.info(f"  - {date_str} - {readable}")
        if len(weekend_dates) > 5:
            logger.info(f"  ... and {len(weekend_dates) - 5} more")
    
    # Summary
    logger.info("")
    logger.info("=" * 80)
    logger.info("Summary")
    logger.info("=" * 80)
    
    total_trading_days_expected = len(date_range) - len(weekend_dates)
    coverage = (len(found_dates) / total_trading_days_expected * 100) if total_trading_days_expected > 0 else 0
    
    logger.info(f"Expected trading days: {total_trading_days_expected}")
    logger.info(f"Found in calendar: {len(found_dates)}")
    logger.info(f"Coverage: {coverage:.1f}%")
    
    if len(missing_dates) == 0:
        logger.info("")
        logger.info("✓ All expected trading days are in calendar!")
        return True
    else:
        logger.warning("")
        logger.warning(f"⚠ {len(missing_dates)} trading days are missing from calendar")
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Check if backtest dates exist in Qlib calendar",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--qlib_dir",
        type=str,
        default="~/.qlib/qlib_data/cn_data",
        help="Qlib data directory (default: ~/.qlib/qlib_data/cn_data)",
    )
    parser.add_argument(
        "--start_date",
        type=str,
        required=True,
        help="Start date (YYYY-MM-DD or YYYYMMDD)",
    )
    parser.add_argument(
        "--end_date",
        type=str,
        required=True,
        help="End date (YYYY-MM-DD or YYYYMMDD)",
    )
    parser.add_argument(
        "--freq",
        type=str,
        default="day",
        help="Data frequency (default: day)",
    )
    
    args = parser.parse_args()
    
    qlib_dir = Path(args.qlib_dir).expanduser()
    
    if not qlib_dir.exists():
        logger.error(f"Qlib directory does not exist: {qlib_dir}")
        return 1
    
    success = check_backtest_dates(
        qlib_dir=qlib_dir,
        start_date=args.start_date,
        end_date=args.end_date,
        freq=args.freq,
    )
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())




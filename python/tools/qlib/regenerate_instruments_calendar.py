#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
regenerate_instruments_calendar.py

Regenerate Qlib instruments/all.txt and calendars/day.txt by scanning features directory.

This tool is useful when:
- Index data (e.g., 000300.SH) was added and overwrote instruments/all.txt
- Need to rebuild instruments list from actual features directory
- Need to rebuild calendar from bin files

Usage:
    # Regenerate from default Qlib directory
    python python/tools/qlib/regenerate_instruments_calendar.py

    # Specify Qlib directory
    python python/tools/qlib/regenerate_instruments_calendar.py \
        --qlib_dir ~/.qlib/qlib_data/cn_data

    # Exclude index codes (e.g., 000300.SH, 000905.SH)
    python python/tools/qlib/regenerate_instruments_calendar.py \
        --exclude_indices 000300.SH 000905.SH 399001.SZ

Arguments:
    --qlib_dir          Qlib data directory (default: ~/.qlib/qlib_data/cn_data)
    --freq              Data frequency (default: day)
    --exclude_indices   Index codes to exclude from instruments (e.g., 000300.SH)
    --backup            Backup existing files before regeneration
"""

import argparse
import logging
import shutil
import sys
from pathlib import Path
from typing import List, Set, Optional, Tuple
from datetime import datetime

import numpy as np
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def is_index_code(code: str) -> bool:
    """
    Check if a code is likely an index code.
    
    Common index codes:
    - 000300.SH (CSI300)
    - 000905.SH (CSI500)
    - 000001.SH (SSE Composite Index)
    - 399001.SZ (SZSE Component Index)
    - 399006.SZ (ChiNext Index)
    
    Args:
        code: Stock/index code (case-insensitive).
    
    Returns:
        True if likely an index code.
    """
    # Normalize to uppercase for comparison
    code_upper = code.upper()
    
    # Common index code patterns
    index_patterns = [
        "000300",  # CSI300
        "000905",  # CSI500
        "399001",  # SZSE Component
        "399006",  # ChiNext
        "399005",  # SZSE SME
    ]
    
    # Extract base code (before dot)
    code_base = code_upper.split(".")[0] if "." in code_upper else code_upper
    
    # Check if base code matches index pattern
    if code_base in index_patterns:
        return True
    
    # Special case: 000001.SH is SSE Composite Index (not a stock)
    if code_base == "000001" and (".SH" in code_upper or code_upper.endswith("SH")):
        return True
    
    return False


def scan_features_directory(
    features_dir: Path,
    freq: str = "day",
    exclude_codes: Optional[List[str]] = None,
) -> tuple:
    """
    Scan features directory to extract all stock codes and date ranges.
    
    Args:
        features_dir: Path to features directory.
        freq: Data frequency (default: day).
        exclude_codes: List of codes to exclude (e.g., index codes).
    
    Returns:
        Tuple of (stock_codes, all_dates_set)
            - stock_codes: Set of stock codes found
            - all_dates_set: Set of all dates found in bin files
    """
    if exclude_codes is None:
        exclude_codes = []
    
    exclude_set = set(exclude_codes)
    stock_codes = set()
    all_dates_set = set()
    
    if not features_dir.exists():
        logger.error(f"Features directory does not exist: {features_dir}")
        return stock_codes, all_dates_set
    
    logger.info(f"Scanning features directory: {features_dir}")
    
    # Scan all subdirectories in features
    total_dirs = 0
    skipped_exclude = 0
    skipped_index = 0
    skipped_no_bin = 0
    
    for stock_dir in features_dir.iterdir():
        if not stock_dir.is_dir():
            continue
        
        total_dirs += 1
        stock_code = stock_dir.name
        stock_code_upper = stock_code.upper()  # Normalize to uppercase for comparison
        
        # Skip if in exclude list (case-insensitive comparison)
        exclude_upper_set = {ex.upper() for ex in exclude_set}
        if stock_code_upper in exclude_upper_set:
            skipped_exclude += 1
            logger.debug(f"Skipping excluded code: {stock_code}")
            continue
        
        # Check if it's likely an index code
        if is_index_code(stock_code):
            skipped_index += 1
            logger.debug(f"Skipping index code: {stock_code}")
            continue
        
        # Check if bin file exists
        # Qlib bin files are named like: close.day.bin, open.day.bin, etc.
        # We check for any *.day.bin file (or *.{freq}.bin)
        bin_files = list(stock_dir.glob(f"*.{freq}.bin"))
        if not bin_files:
            # Also try the old format: day.bin
            bin_file = stock_dir / f"{freq}.bin"
            if not bin_file.exists():
                skipped_no_bin += 1
                logger.debug(f"No {freq}.bin or *.{freq}.bin file for {stock_code}, skipping")
                continue
        
        # Read bin file to extract date information
        try:
            # Bin file format: first float32 is date_index, followed by data
            # Date index points to calendar position
            # We need to read the calendar to get actual dates
            # For now, just collect the stock code
            stock_codes.add(stock_code)
            if len(stock_codes) <= 10:  # Log first 10
                logger.debug(f"Found stock: {stock_code}")
        except Exception as e:
            logger.warning(f"Failed to read bin file for {stock_code}: {e}")
            continue
    
    logger.info(f"Scan statistics: {total_dirs} directories, {skipped_exclude} excluded, {skipped_index} indices, {skipped_no_bin} no bin file, {len(stock_codes)} stocks found")
    
    logger.info(f"Found {len(stock_codes)} stocks in features directory")
    return stock_codes, all_dates_set


def extract_dates_from_calendar(calendar_file: Path) -> Set[str]:
    """
    Extract dates from existing calendar file.
    
    Args:
        calendar_file: Path to calendar file.
    
    Returns:
        Set of date strings in YYYYMMDD format (Qlib standard format).
    """
    if not calendar_file.exists():
        logger.warning(f"Calendar file does not exist: {calendar_file}")
        return set()
    
    try:
        dates = pd.read_csv(calendar_file, header=None).iloc[:, 0].tolist()
        # Convert to YYYYMMDD format (Qlib standard)
        date_set = set()
        for date in dates:
            if isinstance(date, str):
                date_str = date.strip()
                # Check if already in YYYYMMDD format
                if len(date_str) == 8 and date_str.isdigit():
                    date_set.add(date_str)
                else:
                    # Try to parse and convert to YYYYMMDD
                    try:
                        parsed_date = pd.to_datetime(date_str)
                        date_set.add(parsed_date.strftime("%Y%m%d"))
                    except:
                        logger.debug(f"Failed to parse date: {date_str}")
            else:
                # Convert from Timestamp or other format
                try:
                    date_str = pd.to_datetime(date).strftime("%Y%m%d")
                    date_set.add(date_str)
                except:
                    logger.debug(f"Failed to convert date: {date}")
        
        logger.info(f"Extracted {len(date_set)} dates from calendar file")
        return date_set
    except Exception as e:
        logger.error(f"Failed to read calendar file: {e}")
        return set()


def extract_dates_from_csv_files(
    csv_dir: Path,
    freq: str = "day",
) -> Set[str]:
    """
    Extract dates from CSV files.
    
    CSV file format: date,open,high,low,close,volume,factor
    First column is the date in YYYY-MM-DD format.
    
    Args:
        csv_dir: Path to directory containing CSV files.
        freq: Data frequency (for logging).
    
    Returns:
        Set of date strings in YYYYMMDD format.
    """
    date_set = set()
    
    if not csv_dir.exists():
        logger.debug(f"CSV directory does not exist: {csv_dir}")
        return date_set
    
    csv_files = list(csv_dir.glob("*.csv"))
    if not csv_files:
        logger.debug(f"No CSV files found in: {csv_dir}")
        return date_set
    
    logger.info(f"Extracting dates from {len(csv_files)} CSV files...")
    
    for csv_file in csv_files:
        try:
            # Read CSV file (no header, format: date,open,high,low,close,volume,factor)
            df = pd.read_csv(
                csv_file,
                header=None,
                names=["date", "open", "high", "low", "close", "volume", "factor"],
                low_memory=False,
            )
            
            if df.empty or "date" not in df.columns:
                continue
            
            # Extract dates and convert to YYYYMMDD format
            for date_str in df["date"].dropna().unique():
                try:
                    # Parse date (could be YYYY-MM-DD or YYYYMMDD)
                    if isinstance(date_str, str):
                        if len(date_str) == 8 and date_str.isdigit():
                            # Already in YYYYMMDD format
                            date_set.add(date_str)
                        else:
                            # Try to parse and convert
                            parsed_date = pd.to_datetime(date_str)
                            date_set.add(parsed_date.strftime("%Y%m%d"))
                    else:
                        # Already a Timestamp or other date type
                        parsed_date = pd.to_datetime(date_str)
                        date_set.add(parsed_date.strftime("%Y%m%d"))
                except Exception as e:
                    logger.debug(f"Failed to parse date from {csv_file.name}: {date_str}, error: {e}")
                    continue
        
        except Exception as e:
            logger.debug(f"Failed to read CSV file {csv_file.name}: {e}")
            continue
    
    logger.info(f"Extracted {len(date_set)} unique dates from CSV files")
    return date_set


def extract_dates_from_bin_files(
    features_dir: Path,
    stock_codes: Set[str],
    freq: str = "day",
    existing_calendar: Optional[List[str]] = None,
) -> Set[str]:
    """
    Extract dates from bin files by reading date indices.
    
    Bin file format:
    - First float32: date_index (points to position in calendar)
    - Following float32s: data values
    
    Args:
        features_dir: Path to features directory.
        stock_codes: Set of stock codes to process.
        freq: Data frequency.
        existing_calendar: Existing calendar list to map indices to dates.
    
    Returns:
        Set of date strings in YYYYMMDD format.
    """
    date_set = set()
    
    # If we have existing calendar with more than 1 date, use it
    if existing_calendar and len(existing_calendar) > 1:
        # Read date indices from bin files and map to dates
        # Sample more stocks to get better coverage
        sample_stocks = list(stock_codes)[:min(50, len(stock_codes))]
        date_indices_found = set()
        
        for stock_code in sample_stocks:
            # Try to find any bin file for this stock
            bin_files = list((features_dir / stock_code).glob(f"*.{freq}.bin"))
            if not bin_files:
                bin_file = features_dir / stock_code / f"{freq}.bin"
                if bin_file.exists():
                    bin_files = [bin_file]
            
            for bin_file in bin_files:
                try:
                    # Read first float32 (date_index) and calculate end index
                    data = np.fromfile(str(bin_file), dtype=np.float32)
                    if len(data) < 2:
                        continue
                    
                    start_date_index = int(data[0])
                    num_data_points = len(data) - 1
                    end_date_index = start_date_index + num_data_points - 1
                    
                    # Collect all date indices in range
                    for idx in range(start_date_index, min(end_date_index + 1, len(existing_calendar))):
                        if 0 <= idx < len(existing_calendar):
                            date_indices_found.add(idx)
                except Exception as e:
                    logger.debug(f"Failed to read date_index from {stock_code}: {e}")
        
        # Map indices to dates
        for idx in date_indices_found:
            if 0 <= idx < len(existing_calendar):
                date_str = existing_calendar[idx]
                # Ensure YYYYMMDD format
                if len(date_str) == 8 and date_str.isdigit():
                    date_set.add(date_str)
                else:
                    try:
                        date_set.add(pd.to_datetime(date_str).strftime("%Y%m%d"))
                    except:
                        pass
        
        # If we found dates, use all dates from existing calendar (they're all valid trading days)
        if date_set:
            # Convert all existing calendar dates to YYYYMMDD format
            all_dates = set()
            for date_str in existing_calendar:
                if len(date_str) == 8 and date_str.isdigit():
                    all_dates.add(date_str)
                else:
                    try:
                        all_dates.add(pd.to_datetime(date_str).strftime("%Y%m%d"))
                    except:
                        pass
            date_set = all_dates
            logger.info(f"Extracted {len(date_set)} dates from bin files using existing calendar")
    
    return date_set


def get_stock_date_range_from_bin(
    features_dir: Path,
    stock_code: str,
    freq: str,
    calendar_dates: Optional[List[str]] = None,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Get start and end date for a stock from its bin file.
    
    Bin file format:
    - First float32: date_index (points to position in calendar)
    - Following float32s: data values for each date
    
    Args:
        features_dir: Features directory.
        stock_code: Stock code.
        freq: Data frequency.
        calendar_dates: Calendar dates list (for mapping date_index to dates).
    
    Returns:
        Tuple of (start_date, end_date) in YYYYMMDD format, or (None, None) if cannot determine.
    """
    stock_dir = features_dir / stock_code
    if not stock_dir.exists():
        return None, None
    
    # Find any bin file (e.g., close.day.bin, open.day.bin)
    bin_files = list(stock_dir.glob(f"*.{freq}.bin"))
    if not bin_files:
        return None, None
    
    bin_file = bin_files[0]  # Use first bin file
    
    try:
        # Read bin file
        data = np.fromfile(str(bin_file), dtype=np.float32)
        if len(data) < 2:
            return None, None
        
        # First float32 is date_index (start position in calendar)
        start_date_index = int(data[0])
        
        if calendar_dates and 0 <= start_date_index < len(calendar_dates):
            start_date_str = calendar_dates[start_date_index]
            
            # Calculate end date_index
            # Bin file structure: [date_index, data_point_1, data_point_2, ...]
            # Number of data points = len(data) - 1
            # End index = start_index + number_of_data_points - 1
            num_data_points = len(data) - 1
            end_date_index = start_date_index + num_data_points - 1
            
            # Clamp to calendar range
            if end_date_index >= len(calendar_dates):
                end_date_index = len(calendar_dates) - 1
            
            if end_date_index >= start_date_index:
                end_date_str = calendar_dates[end_date_index]
                return start_date_str, end_date_str
            else:
                # Fallback: use start date as end date
                return start_date_str, start_date_str
        
        return None, None
    except Exception as e:
        logger.debug(f"Failed to read date range from {stock_code}: {e}")
        return None, None


def regenerate_instruments(
    qlib_dir: Path,
    stock_codes: Set[str],
    freq: str = "day",
    backup: bool = False,
) -> bool:
    """
    Regenerate instruments/all.txt file with proper Qlib format.
    
    Qlib requires format: <stock_code>\t<start_date>\t<end_date>
    This function reads date ranges from bin files or uses calendar range.
    
    Args:
        qlib_dir: Qlib data directory.
        stock_codes: Set of stock codes to include.
        freq: Data frequency.
        backup: Whether to backup existing file.
    
    Returns:
        True if successful.
    """
    instruments_dir = qlib_dir / "instruments"
    instruments_dir.mkdir(parents=True, exist_ok=True)
    
    instruments_file = instruments_dir / "all.txt"
    
    # Backup if requested
    if backup and instruments_file.exists():
        backup_file = instruments_file.with_suffix(".txt.backup")
        shutil.copy2(instruments_file, backup_file)
        logger.info(f"Backed up existing instruments file to: {backup_file}")
    
    # Load calendar to get date range
    calendar_file = qlib_dir / "calendars" / f"{freq}.txt"
    calendar_dates = None
    if calendar_file.exists():
        try:
            calendar_dates = extract_dates_from_calendar(calendar_file)
            if calendar_dates:
                calendar_dates = sorted(list(calendar_dates))
                logger.info(f"Loaded calendar with {len(calendar_dates)} dates for date range calculation")
        except Exception as e:
            logger.warning(f"Failed to load calendar: {e}, will try to read from bin files")
    
    # Get date ranges for each stock
    features_dir = qlib_dir / "features"
    sorted_codes = sorted(stock_codes)
    
    instruments_with_dates = []
    instruments_without_dates = []
    
    for code in sorted_codes:
        start_date, end_date = get_stock_date_range_from_bin(
            features_dir, code, freq, calendar_dates
        )
        
        if start_date and end_date:
            # Convert to YYYYMMDD format if needed
            try:
                # If calendar dates are in YYYYMMDD format, use directly
                if len(start_date) == 8 and start_date.isdigit():
                    start_fmt = start_date
                else:
                    # Convert from YYYY-MM-DD to YYYYMMDD
                    start_fmt = pd.to_datetime(start_date).strftime("%Y%m%d")
                
                if len(end_date) == 8 and end_date.isdigit():
                    end_fmt = end_date
                else:
                    end_fmt = pd.to_datetime(end_date).strftime("%Y%m%d")
                
                instruments_with_dates.append((code, start_fmt, end_fmt))
            except Exception as e:
                logger.debug(f"Failed to format dates for {code}: {e}")
                instruments_without_dates.append(code)
        else:
            # If cannot determine date range, use calendar range if available
            if calendar_dates and len(calendar_dates) > 0:
                start_fmt = calendar_dates[0]
                end_fmt = calendar_dates[-1]
                # Ensure YYYYMMDD format
                if len(start_fmt) != 8 or not start_fmt.isdigit():
                    try:
                        start_fmt = pd.to_datetime(start_fmt).strftime("%Y%m%d")
                    except:
                        start_fmt = calendar_dates[0]
                if len(end_fmt) != 8 or not end_fmt.isdigit():
                    try:
                        end_fmt = pd.to_datetime(end_fmt).strftime("%Y%m%d")
                    except:
                        end_fmt = calendar_dates[-1]
                instruments_with_dates.append((code, start_fmt, end_fmt))
            else:
                instruments_without_dates.append(code)
    
    # Write instruments file in Qlib format: <code>\t<start>\t<end>
    try:
        with open(instruments_file, "w", encoding="utf-8") as f:
            for code, start_date, end_date in instruments_with_dates:
                f.write(f"{code}\t{start_date}\t{end_date}\n")
            
            # If any stocks don't have dates, write them without dates (Qlib may still work)
            if instruments_without_dates:
                logger.warning(f"{len(instruments_without_dates)} stocks without date ranges, writing without dates")
                for code in instruments_without_dates:
                    f.write(f"{code}\n")
        
        logger.info(f"✓ Regenerated instruments/all.txt with {len(instruments_with_dates)} stocks (with date ranges)")
        if instruments_without_dates:
            logger.info(f"  And {len(instruments_without_dates)} stocks without date ranges")
        return True
    except Exception as e:
        logger.error(f"Failed to write instruments file: {e}")
        return False


def regenerate_calendar(
    qlib_dir: Path,
    dates: Set[str],
    freq: str = "day",
    backup: bool = False,
) -> bool:
    """
    Regenerate calendars/day.txt file.
    
    Qlib requires calendar file in YYYYMMDD format (e.g., 20231224).
    
    Args:
        qlib_dir: Qlib data directory.
        dates: Set of date strings (in YYYYMMDD format or will be converted).
        freq: Data frequency.
        backup: Whether to backup existing file.
    
    Returns:
        True if successful.
    """
    calendars_dir = qlib_dir / "calendars"
    calendars_dir.mkdir(parents=True, exist_ok=True)
    
    calendar_file = calendars_dir / f"{freq}.txt"
    
    # Backup if requested
    if backup and calendar_file.exists():
        backup_file = calendar_file.with_suffix(".txt.backup")
        shutil.copy2(calendar_file, backup_file)
        logger.info(f"Backed up existing calendar file to: {backup_file}")
    
    # Convert all dates to YYYYMMDD format and sort
    sorted_dates = []
    for date in dates:
        if isinstance(date, str):
            date_str = date.strip()
            # Check if already in YYYYMMDD format
            if len(date_str) == 8 and date_str.isdigit():
                sorted_dates.append(date_str)
            else:
                # Try to parse and convert to YYYYMMDD
                try:
                    parsed_date = pd.to_datetime(date_str)
                    sorted_dates.append(parsed_date.strftime("%Y%m%d"))
                except Exception as e:
                    logger.debug(f"Failed to parse date: {date_str}, error: {e}")
        else:
            # Convert from Timestamp or other format
            try:
                sorted_dates.append(pd.to_datetime(date).strftime("%Y%m%d"))
            except Exception as e:
                logger.debug(f"Failed to convert date: {date}, error: {e}")
    
    # Remove duplicates and sort
    sorted_dates = sorted(set(sorted_dates))
    
    if not sorted_dates:
        logger.error("No valid dates to write to calendar file")
        return False
    
    # Write calendar file in YYYYMMDD format (Qlib standard)
    try:
        with open(calendar_file, "w", encoding="utf-8") as f:
            for date in sorted_dates:
                f.write(f"{date}\n")
        
        logger.info(f"✓ Regenerated calendars/{freq}.txt with {len(sorted_dates)} dates (YYYYMMDD format)")
        logger.info(f"  Date range: {sorted_dates[0]} to {sorted_dates[-1]}")
        return True
    except Exception as e:
        logger.error(f"Failed to write calendar file: {e}")
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Regenerate Qlib instruments and calendar from features directory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--qlib_dir",
        type=str,
        default="~/.qlib/qlib_data/cn_data",
        help="Qlib data directory (default: ~/.qlib/qlib_data/cn_data)",
    )
    parser.add_argument(
        "--freq",
        type=str,
        default="day",
        help="Data frequency (default: day)",
    )
    parser.add_argument(
        "--exclude_indices",
        nargs="+",
        default=[],
        help="Index codes to exclude from instruments (e.g., 000300.SH 000905.SH)",
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Backup existing files before regeneration",
    )
    parser.add_argument(
        "--auto_exclude_indices",
        action="store_true",
        default=True,
        help="Automatically exclude common index codes (default: True)",
    )
    parser.add_argument(
        "--csv_dir",
        type=str,
        default=None,
        help="CSV directory to extract dates from (default: ~/.qlib/csv_data/cn_data). "
             "If calendar is missing, dates will be extracted from CSV files.",
    )
    
    args = parser.parse_args()
    
    # Expand paths
    qlib_dir = Path(args.qlib_dir).expanduser()
    features_dir = qlib_dir / "features"
    
    if not features_dir.exists():
        logger.error(f"Features directory does not exist: {features_dir}")
        return 1
    
    # Prepare exclude list
    exclude_codes = list(args.exclude_indices)
    if args.auto_exclude_indices:
        # Add common index codes
        common_indices = ["000300.SH", "000905.SH", "000001.SH", "399001.SZ", "399006.SZ"]
        exclude_codes.extend(common_indices)
        exclude_codes = list(set(exclude_codes))  # Remove duplicates
    
    logger.info(f"Excluding codes: {exclude_codes}")
    
    # Scan features directory
    stock_codes, _ = scan_features_directory(
        features_dir=features_dir,
        freq=args.freq,
        exclude_codes=exclude_codes,
    )
    
    if not stock_codes:
        logger.error("No stocks found in features directory. Cannot regenerate instruments.")
        return 1
    
    # Try to get dates from existing calendar
    calendar_file = qlib_dir / "calendars" / f"{args.freq}.txt"
    dates = extract_dates_from_calendar(calendar_file)
    existing_calendar_list = sorted(list(dates)) if dates else None
    
    # Check if calendar is corrupted (only 1 date or invalid format)
    calendar_corrupted = False
    if dates and len(dates) == 1:
        calendar_corrupted = True
        logger.error(
            f"⚠️  Calendar file appears to be corrupted (only 1 date found): {calendar_file}"
        )
        logger.error(
            "   This usually happens when calendar was overwritten incorrectly."
        )
        dates = set()  # Clear dates so we can try to recover from CSV
    elif not dates:
        logger.warning(
            f"No dates found in calendar file: {calendar_file}. "
            f"Will try to extract dates from CSV files if available."
        )
    
    # If calendar is missing or corrupted, try to extract dates from CSV files
    if not dates or calendar_corrupted:
        csv_dir = None
        if args.csv_dir:
            csv_dir = Path(args.csv_dir).expanduser()
        else:
            # Try default CSV directory
            default_csv_dir = Path.home() / ".qlib" / "csv_data" / "cn_data"
            if default_csv_dir.exists():
                csv_dir = default_csv_dir
        
        if csv_dir and csv_dir.exists():
            logger.info(f"Attempting to extract dates from CSV files in: {csv_dir}")
            dates_from_csv = extract_dates_from_csv_files(csv_dir, args.freq)
            if dates_from_csv:
                dates = dates_from_csv
                logger.info(f"✓ Successfully extracted {len(dates)} dates from CSV files")
            else:
                logger.warning(
                    f"No dates found in CSV files. "
                    f"Calendar cannot be regenerated from bin files alone (bin files only store date indices)."
                )
                logger.warning(
                    f"SOLUTION: Re-export data using export_qlib.py to regenerate calendar correctly."
                )
        else:
            logger.warning(
                f"CSV directory not found: {csv_dir if csv_dir else 'default location'}. "
                f"Calendar cannot be regenerated from bin files alone (bin files only store date indices)."
            )
            logger.warning(
                f"SOLUTION: Re-export data using export_qlib.py to regenerate calendar correctly."
            )
    else:
        # Calendar exists and is valid, try to extract from bin files to verify/update
        dates_from_bin = extract_dates_from_bin_files(
            features_dir, stock_codes, args.freq, existing_calendar_list
        )
        if dates_from_bin:
            dates = dates_from_bin
            logger.info(f"Using {len(dates)} dates extracted from bin files")
    
    # Regenerate instruments
    logger.info("\n" + "="*80)
    logger.info("Regenerating instruments/all.txt")
    logger.info("="*80)
    if not regenerate_instruments(qlib_dir, stock_codes, args.freq, args.backup):
        return 1
    
    # Regenerate calendar if we have dates
    if dates:
        logger.info("\n" + "="*80)
        logger.info("Regenerating calendars/day.txt")
        logger.info("="*80)
        if not regenerate_calendar(qlib_dir, dates, args.freq, args.backup):
            logger.warning("Failed to regenerate calendar, but instruments were updated")
    else:
        logger.warning(
            "\n⚠️  Could not regenerate calendar. "
            "You may need to use Qlib's dump_bin tool to regenerate it from CSV files."
        )
    
    logger.info("\n" + "="*80)
    logger.info("Summary")
    logger.info("="*80)
    logger.info(f"✓ Regenerated instruments/all.txt with {len(stock_codes)} stocks")
    if dates:
        logger.info(f"✓ Regenerated calendars/{args.freq}.txt with {len(dates)} dates")
    else:
        logger.info("⚠️  Calendar was not regenerated (no dates found)")
    
    logger.info(f"\nStock codes in instruments/all.txt:")
    sorted_codes = sorted(stock_codes)
    for i, code in enumerate(sorted_codes[:20], 1):
        logger.info(f"  {i}. {code}")
    if len(sorted_codes) > 20:
        logger.info(f"  ... and {len(sorted_codes) - 20} more")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())


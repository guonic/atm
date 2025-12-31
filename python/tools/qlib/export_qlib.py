#!/usr/bin/env python3
"""
Qlib data export tool for ATM project.

Supports:
- Full export of all stocks for all time frequencies (day, week, hour, etc.)
- Selective export of specific stocks
- Incremental export (tracks last export time, only exports new data)
- CSV cleanup (keeps only last row for next export time query)
"""

import argparse
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Dict, Tuple

import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from nq.config import DatabaseConfig, load_config
from nq.repo.kline_repo import (
    StockKlineDayRepo,
    StockKlineWeekRepo,
    StockKlineMonthRepo,
    StockKlineQuarterRepo,
    StockKlineHourRepo,
    StockKline30MinRepo,
    StockKline15MinRepo,
    StockKline5MinRepo,
    StockKline1MinRepo,
)
from nq.repo.stock_repo import StockBasicRepo

# Import dump_bin tool
tools_qlib_dir = Path(__file__).parent
if str(tools_qlib_dir) not in sys.path:
    sys.path.insert(0, str(tools_qlib_dir))

import dump_bin
DumpDataAll = dump_bin.DumpDataAll
DumpDataFix = dump_bin.DumpDataFix
DumpDataUpdate = dump_bin.DumpDataUpdate

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# Mapping from frequency to Qlib frequency and repository
FREQ_MAPPING = {
    "day": ("day", StockKlineDayRepo, "trade_date"),
    "week": ("week", StockKlineWeekRepo, "week_date"),
    "month": ("month", StockKlineMonthRepo, "month_date"),
    "quarter": ("quarter", StockKlineQuarterRepo, "quarter_date"),
    "hour": ("1h", StockKlineHourRepo, "trade_time"),
    "30min": ("30min", StockKline30MinRepo, "trade_time"),
    "15min": ("15min", StockKline15MinRepo, "trade_time"),
    "5min": ("5min", StockKline5MinRepo, "trade_time"),
    "1min": ("1min", StockKline1MinRepo, "trade_time"),
}


# Import unified data normalization
from nq.utils.data_normalize import normalize_stock_code as convert_ts_code_to_qlib_format


def get_last_export_time(csv_file: Path) -> Optional[datetime]:
    """
    Get the last export time from CSV file (last row's date).
    
    Args:
        csv_file: Path to CSV file.
        
    Returns:
        Last export date if file exists and has data, None otherwise.
    """
    if not csv_file.exists():
        return None
    
    try:
        # Read only the last line
        with open(csv_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            if not lines:
                return None
            
            # Last non-empty line
            last_line = None
            for line in reversed(lines):
                if line.strip():
                    last_line = line.strip()
                    break
            
            if not last_line:
                return None
            
            # Parse date from first column
            date_str = last_line.split(",")[0]
            parsed_date = pd.to_datetime(date_str, errors="coerce")
            if pd.isna(parsed_date):
                logger.warning(f"Failed to parse date from CSV file {csv_file}: {date_str}")
                return None
            return parsed_date
    except Exception as e:
        logger.warning(f"Failed to read last export time from {csv_file}: {e}")
        return None


def cleanup_csv_file(csv_file: Path) -> bool:
    """
    Cleanup CSV file, keeping only the last row.
    
    This is used to store the last export time for incremental export.
    The last row contains the most recent date, which will be used as
    the starting point for the next incremental export.
    
    Args:
        csv_file: Path to CSV file.
        
    Returns:
        True if cleanup successful, False otherwise.
    """
    if not csv_file.exists():
        return True
    
    try:
        # Read all data
        df = pd.read_csv(
            csv_file,
            header=None,
            names=["date", "open", "high", "low", "close", "volume", "factor"],
        )
        
        if len(df) == 0:
            # Empty file, delete it
            csv_file.unlink()
            logger.debug(f"Deleted empty CSV file: {csv_file.name}")
            return True
        
        if len(df) == 1:
            # Already cleaned up, no need to do anything
            logger.debug(f"CSV file {csv_file.name} already has only one row")
            return True
        
        # Keep only the last row
        last_row = df.iloc[[-1]]
        
        # Write back
        last_row.to_csv(csv_file, index=False, header=False)
        
        logger.debug(f"Cleaned up {csv_file.name}: kept last row (date: {last_row.iloc[0]['date']})")
        return True
    except Exception as e:
        logger.error(f"Failed to cleanup {csv_file}: {e}")
        return False


def export_stock_to_csv(
    repo,
    ts_code: str,
    start_date: Optional[datetime],
    end_date: Optional[datetime],
    output_dir: Path,
    time_column: str,
) -> Tuple[bool, Optional[datetime]]:
    """
    Export stock data to CSV format.
    
    Args:
        repo: K-line repository instance.
        ts_code: Stock code (e.g., 000001.SZ).
        start_date: Start date for export (None for all data).
        end_date: End date for export (None for all data).
        output_dir: Output directory for CSV files.
        time_column: Time column name (e.g., 'trade_date', 'trade_time').
        
    Returns:
        Tuple of (success, last_date) where last_date is the last exported date.
    """
    try:
        # Get data from repository
        klines = repo.get_by_ts_code(
            ts_code=ts_code,
            start_time=start_date,
            end_time=end_date,
        )
        
        if not klines:
            # Check if this is a same-day query (might not have data yet)
            if start_date and end_date and start_date.date() == end_date.date():
                logger.debug(f"No data found for {ts_code} on {start_date.date()} (data might not be available yet)")
            else:
                logger.warning(f"No data found for {ts_code} (start_date={start_date}, end_date={end_date})")
                logger.warning(f"  This may indicate:")
                logger.warning(f"  1. Stock {ts_code} is not in the database")
                logger.warning(f"  2. Stock {ts_code} has no data in the specified date range")
                logger.warning(f"  3. Stock {ts_code} may be delisted or suspended")
            return False, None
        
        # Convert to DataFrame
        data = []
        for kline in klines:
            row = kline.model_dump()
            data.append(row)
        
        df = pd.DataFrame(data)
        
        if len(df) == 0:
            logger.debug(f"No valid data for {ts_code}")
            return False, None
        
        # Check required columns
        required_cols = ["open", "high", "low", "close", "volume"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.warning(f"Missing columns for {ts_code}: {missing_cols}")
            return False, None
        
        # Prepare Qlib format DataFrame
        qlib_df = pd.DataFrame()
        
        # Convert time column to date string
        if time_column in df.columns:
            qlib_df["date"] = pd.to_datetime(df[time_column], errors="coerce").dt.strftime("%Y-%m-%d")
        else:
            logger.error(f"Time column '{time_column}' not found for {ts_code}")
            return False, None
        
        # Convert price and volume fields
        qlib_df["open"] = pd.to_numeric(df["open"], errors="coerce")
        qlib_df["high"] = pd.to_numeric(df["high"], errors="coerce")
        qlib_df["low"] = pd.to_numeric(df["low"], errors="coerce")
        qlib_df["close"] = pd.to_numeric(df["close"], errors="coerce")
        qlib_df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0).astype(int)
        
        # Add factor field (default to 1.0 if not available)
        if "adj_factor" in df.columns:
            qlib_df["factor"] = pd.to_numeric(df["adj_factor"], errors="coerce").fillna(1.0)
        elif "factor" in df.columns:
            qlib_df["factor"] = pd.to_numeric(df["factor"], errors="coerce").fillna(1.0)
        else:
            qlib_df["factor"] = 1.0
        
        # Remove rows with missing price data
        qlib_df = qlib_df.dropna(subset=["open", "high", "low", "close"])
        
        if len(qlib_df) == 0:
            logger.debug(f"No valid price data for {ts_code}")
            return False, None
        
        # Sort by date
        qlib_df = qlib_df.sort_values("date")
        
        # Get last date
        last_date = pd.to_datetime(qlib_df["date"].iloc[-1])
        
        # Convert ts_code to Qlib format (ensure consistency with data)
        # The code format must match what dump_bin will extract from filename
        qlib_code = convert_ts_code_to_qlib_format(ts_code)
        
        # Save to CSV (ensure column order: date, open, high, low, close, volume, factor)
        # CSV filename uses Qlib format code to ensure consistency with dump_bin extraction
        csv_file = output_dir / f"{qlib_code}.csv"
        column_order = ["date", "open", "high", "low", "close", "volume", "factor"]
        qlib_df = qlib_df[column_order]
        
        # Append to existing CSV if it exists (for incremental export)
        if csv_file.exists():
            existing_df = pd.read_csv(
                csv_file,
                header=None,
                names=column_order,
            )
            # Combine and remove duplicates
            combined_df = pd.concat([existing_df, qlib_df], ignore_index=True)
            combined_df = combined_df.drop_duplicates(subset=["date"], keep="last")
            combined_df = combined_df.sort_values("date")
            combined_df.to_csv(csv_file, index=False, header=False)
        else:
            qlib_df.to_csv(csv_file, index=False, header=False)
        
        logger.debug(f"Exported {len(qlib_df)} records for {ts_code} (last date: {last_date.strftime('%Y-%m-%d')})")
        return True, last_date
        
    except Exception as e:
        logger.error(f"Failed to export {ts_code}: {type(e).__name__}: {e}", exc_info=True)
        return False, None


def convert_csv_to_qlib_bin(csv_dir: Path, qlib_dir: Path, freq: str, ts_codes: Optional[List[str]] = None) -> bool:
    """
    Convert CSV files to Qlib bin format.
    
    Args:
        csv_dir: Directory containing CSV files.
        qlib_dir: Output directory for Qlib bin files.
        freq: Frequency string for Qlib (e.g., 'day', 'week').
        ts_codes: Optional list of stock codes to convert. If None, converts all CSV files.
        
    Returns:
        True if conversion successful, False otherwise.
    """
    logger.info("=" * 80)
    logger.info("Converting CSV files to Qlib bin format...")
    logger.info(f"CSV directory: {csv_dir}")
    logger.info(f"Qlib directory: {qlib_dir}")
    logger.info(f"Frequency: {freq}")
    if ts_codes:
        logger.info(f"Converting {len(ts_codes)} specified stocks")
    else:
        logger.info("Converting all CSV files in directory")
    logger.info("=" * 80)
    
    # Get CSV files to convert
    if ts_codes:
        # Only convert CSV files for specified stocks
        qlib_codes = [convert_ts_code_to_qlib_format(ts_code) for ts_code in ts_codes]
        csv_files = [csv_dir / f"{code}.csv" for code in qlib_codes]
        # Filter to only existing files
        csv_files = [f for f in csv_files if f.exists()]
        if not csv_files:
            logger.error(f"No CSV files found for specified stocks: {ts_codes}")
            return False
    else:
        # Convert all CSV files in directory
        csv_files = list(csv_dir.glob("*.csv"))
        if not csv_files:
            logger.error("No CSV files found in directory")
            return False
    
    logger.info(f"Validating {len(csv_files)} CSV files...")
    invalid_files = []
    for csv_file in csv_files[:min(5, len(csv_files))]:
        try:
            with open(csv_file, "r", encoding="utf-8") as f:
                first_line = f.readline().strip()
                col_count = len(first_line.split(","))
                if col_count != 7:
                    invalid_files.append((csv_file.name, col_count))
        except Exception as e:
            logger.warning(f"Failed to validate {csv_file.name}: {e}")
    
    if invalid_files:
        logger.error("✗ Some CSV files are missing factor column (must have 7 columns):")
        for filename, col_count in invalid_files:
            logger.error(f"  - {filename}: {col_count} columns (expected 7)")
        logger.error("")
        logger.error("Please regenerate CSV files with factor column.")
        return False
    
    logger.info("✓ All CSV files have correct format (7 columns including factor)")
    
    # If only converting specific stocks, create a temporary directory with only those CSV files
    temp_csv_dir = None
    try:
        if ts_codes and len(csv_files) < len(list(csv_dir.glob("*.csv"))):
            # Create temporary directory with only the CSV files we want to convert
            import tempfile
            import shutil
            temp_csv_dir = Path(tempfile.mkdtemp(prefix="qlib_export_"))
            logger.info(f"Creating temporary directory with {len(csv_files)} CSV files: {temp_csv_dir}")
            for csv_file in csv_files:
                shutil.copy2(csv_file, temp_csv_dir / csv_file.name)
            data_path = str(temp_csv_dir)
        else:
            # Use original directory (all files or already filtered)
            data_path = str(csv_dir)
        
        # Check if qlib data already exists for incremental update
        calendar_file = qlib_dir / "calendars" / f"{freq}.txt"
        instruments_file = qlib_dir / "instruments" / "all.txt"
        
        if calendar_file.exists() and instruments_file.exists():
            # Use DumpDataFix for incremental update (preserves existing calendar and instruments)
            logger.info("Existing Qlib data found, using incremental update mode...")
            logger.info("This will preserve existing calendar and instruments, only add/update new stock data")
            dumper = DumpDataFix(
                data_path=data_path,
                qlib_dir=str(qlib_dir),
                freq=freq,
                max_workers=4,
                date_field_name="date",
                file_suffix=".csv",
                include_fields="open,close,high,low,volume,factor",
            )
        else:
            # First time export, use DumpDataAll
            logger.info("No existing Qlib data found, using full export mode...")
            dumper = DumpDataAll(
                data_path=data_path,
                qlib_dir=str(qlib_dir),
                freq=freq,
                max_workers=4,
                date_field_name="date",
                file_suffix=".csv",
                include_fields="open,close,high,low,volume,factor",
            )
        
        logger.info("Starting conversion...")
        dumper.dump()
        
        # Post-process: Fix calendar file format (YYYY-MM-DD -> YYYYMMDD for Qlib standard)
        if freq == "day":
            calendar_file = qlib_dir / "calendars" / f"{freq}.txt"
            if calendar_file.exists():
                logger.info("Fixing calendar file format (YYYY-MM-DD -> YYYYMMDD)...")
                try:
                    with open(calendar_file, "r", encoding="utf-8") as f:
                        dates = [line.strip() for line in f if line.strip()]
                    
                    converted_dates = []
                    for date_str in dates:
                        try:
                            dt = pd.to_datetime(date_str)
                            converted_dates.append(dt.strftime("%Y%m%d"))
                        except (ValueError, TypeError):
                            if len(date_str) == 8 and date_str.isdigit():
                                converted_dates.append(date_str)
                    
                    if converted_dates:
                        with open(calendar_file, "w", encoding="utf-8") as f:
                            f.write("\n".join(converted_dates))
                            f.write("\n")
                        logger.info(f"  ✓ Converted {len(converted_dates)} dates to YYYYMMDD format")
                except Exception as e:
                    logger.warning(f"  ⚠ Failed to fix calendar format: {e}")
        
        logger.info("✓ Successfully converted to Qlib bin format")
        return True
        
    except Exception as e:
        logger.error(f"Failed to convert CSV to Qlib bin format: {e}", exc_info=True)
        return False
    finally:
        # Clean up temporary directory if created
        if temp_csv_dir and temp_csv_dir.exists():
            import shutil
            try:
                shutil.rmtree(temp_csv_dir)
                logger.debug(f"Cleaned up temporary directory: {temp_csv_dir}")
            except Exception as e:
                logger.warning(f"Failed to clean up temporary directory {temp_csv_dir}: {e}")


def export_qlib_data(
    freq: str,
    output_dir: Path,
    qlib_dir: Path,
    ts_codes: Optional[List[str]] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    incremental: bool = True,
    cleanup_csv: bool = True,
    config: Optional[DatabaseConfig] = None,
) -> bool:
    """
    Export ATM data to Qlib format.
    
    Args:
        freq: Frequency (day, week, month, quarter, hour, 30min, 15min, 5min, 1min).
        output_dir: Output directory for CSV files.
        qlib_dir: Output directory for Qlib bin files.
        ts_codes: List of stock codes to export (None for all stocks).
        start_date: Start date for export (None for all data or incremental from last export).
        end_date: End date for export (None for current date).
        incremental: Whether to use incremental export (read last export time from CSV).
        cleanup_csv: Whether to cleanup CSV files after conversion (keep only last row).
        config: Database configuration (None to load from default).
        
    Returns:
        True if export successful, False otherwise.
    """
    if freq not in FREQ_MAPPING:
        logger.error(f"Unsupported frequency: {freq}. Supported: {list(FREQ_MAPPING.keys())}")
        return False
    
    qlib_freq, repo_class, time_column = FREQ_MAPPING[freq]
    
    # Load config if not provided
    if config is None:
        full_config = load_config()
        db_config = full_config.database
    else:
        # If config is already DatabaseConfig, use it directly
        # Otherwise, extract database config from Config object
        if isinstance(config, DatabaseConfig):
            db_config = config
        else:
            db_config = config.database
    
    # Initialize repositories
    repo = repo_class(db_config)
    stock_repo = StockBasicRepo(db_config)
    
    # Get stock list
    if ts_codes is None:
        # Get all stocks
        stocks = stock_repo.get_all()
        ts_codes = [stock.ts_code for stock in stocks]
        logger.info(f"Exporting data for {len(ts_codes)} stocks")
    else:
        logger.info(f"Exporting data for {len(ts_codes)} specified stocks")
    
    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_dir = output_dir / "csv_data" / "cn_data"
    csv_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine date range
    if end_date is None:
        end_date = datetime.now()
    
    # Export each stock
    exported_count = 0
    failed_count = 0
    
    for ts_code in ts_codes:
        try:
            # Determine start date for this stock
            stock_start_date = start_date
            
            if incremental and stock_start_date is None:
                # Get last export time from CSV
                qlib_code = convert_ts_code_to_qlib_format(ts_code)
                csv_file = csv_dir / f"{qlib_code}.csv"
                last_export_time = get_last_export_time(csv_file)
                
                if last_export_time:
                    # Start from next day after last export
                    stock_start_date = last_export_time + timedelta(days=1)
                    logger.info(f"{ts_code}: Incremental export from {stock_start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
                    
                    # Check if start_date is after end_date (no new data)
                    if stock_start_date > end_date:
                        logger.info(f"{ts_code}: No new data to export (last export: {last_export_time.strftime('%Y-%m-%d')}, end_date: {end_date.strftime('%Y-%m-%d')})")
                        exported_count += 1  # Count as success (no new data is not a failure)
                        continue
                    
                    # Check if start_date equals end_date (same day, might not have data yet)
                    if stock_start_date.date() == end_date.date():
                        logger.info(f"{ts_code}: Exporting same day data (might not be available yet)")
                else:
                    logger.info(f"{ts_code}: Full export (no previous export found)")
            
            # Export stock data
            success, last_date = export_stock_to_csv(
                repo=repo,
                ts_code=ts_code,
                start_date=stock_start_date,
                end_date=end_date,
                output_dir=csv_dir,
                time_column=time_column,
            )
            
            if success:
                exported_count += 1
            else:
                # Check if this is a same-day incremental export with no data (not a real failure)
                if incremental and stock_start_date and end_date and stock_start_date.date() == end_date.date():
                    logger.info(f"{ts_code}: No new data available for today (this is normal, not a failure)")
                    exported_count += 1  # Count as success (no data for today is expected)
                else:
                    failed_count += 1
                
        except Exception as e:
            logger.error(f"Failed to export {ts_code}: {type(e).__name__}: {e}", exc_info=True)
            failed_count += 1
    
    logger.info(f"Export completed: {exported_count} stocks exported, {failed_count} failed")
    
    if exported_count == 0:
        logger.error("No stocks exported successfully")
        return False
    
    # Convert CSV to Qlib bin format (only for exported stocks)
    # IMPORTANT: Convert BEFORE cleaning up CSV files, otherwise only last row will be converted
    qlib_output_dir = qlib_dir / "qlib_data" / "cn_data"
    if not convert_csv_to_qlib_bin(csv_dir, qlib_output_dir, qlib_freq, ts_codes=ts_codes):
        logger.error("Failed to convert CSV to Qlib bin format")
        return False
    
    # Cleanup CSV files AFTER conversion (keep only last row for incremental export)
    if cleanup_csv:
        logger.info("Cleaning up CSV files (keeping only last row for incremental export)...")
        cleanup_count = 0
        for ts_code in ts_codes:
            try:
                qlib_code = convert_ts_code_to_qlib_format(ts_code)
                csv_file = csv_dir / f"{qlib_code}.csv"
                if csv_file.exists():
                    if cleanup_csv_file(csv_file):
                        cleanup_count += 1
            except Exception as e:
                logger.warning(f"Failed to cleanup CSV for {ts_code}: {e}")
        logger.info(f"Cleaned up {cleanup_count} CSV files")
    
    logger.info("=" * 80)
    logger.info("Export completed successfully!")
    logger.info(f"CSV files: {csv_dir}")
    logger.info(f"Qlib bin files: {qlib_output_dir}")
    logger.info("=" * 80)
    
    return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Export ATM data to Qlib format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full export of all stocks for daily data
  python export_qlib.py --freq day

  # Export specific stocks
  python export_qlib.py --freq day --stocks 000001.SZ 000002.SZ

  # Incremental export (default)
  python export_qlib.py --freq day --incremental

  # Full export (not incremental)
  python export_qlib.py --freq day --no-incremental

  # Export with date range
  python export_qlib.py --freq day --start-date 2024-01-01 --end-date 2024-12-31

  # Export weekly data
  python export_qlib.py --freq week

Supported frequencies:
  day, week, month, quarter, hour, 30min, 15min, 5min, 1min
        """,
    )
    
    parser.add_argument(
        "--freq",
        type=str,
        required=True,
        choices=list(FREQ_MAPPING.keys()),
        help="Data frequency (day, week, month, quarter, hour, 30min, 15min, 5min, 1min)",
    )
    parser.add_argument(
        "--stocks",
        type=str,
        nargs="+",
        help="List of stock codes to export (e.g., 000001.SZ 000002.SZ). If not specified, exports all stocks.",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        help="Start date for export (YYYY-MM-DD). If not specified and incremental is enabled, uses last export time.",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        help="End date for export (YYYY-MM-DD). Default: current date.",
    )
    parser.add_argument(
        "--incremental",
        action="store_true",
        default=True,
        help="Use incremental export (read last export time from CSV). Default: True.",
    )
    parser.add_argument(
        "--no-incremental",
        dest="incremental",
        action="store_false",
        help="Disable incremental export (full export).",
    )
    parser.add_argument(
        "--cleanup-csv",
        action="store_true",
        default=True,
        help="Cleanup CSV files after conversion (keep only last row). Default: True.",
    )
    parser.add_argument(
        "--no-cleanup-csv",
        dest="cleanup_csv",
        action="store_false",
        help="Disable CSV cleanup.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="~/.qlib",
        help="Output directory for CSV files. Default: ~/.qlib",
    )
    parser.add_argument(
        "--qlib-dir",
        type=str,
        default="~/.qlib",
        help="Output directory for Qlib bin files. Default: ~/.qlib",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to config file. Default: config/config.yaml",
    )
    
    args = parser.parse_args()
    
    # Parse dates
    start_date = None
    if args.start_date:
        start_date = pd.to_datetime(args.start_date)
    
    end_date = None
    if args.end_date:
        end_date = pd.to_datetime(args.end_date)
    
    # Load config
    full_config = None
    if args.config:
        full_config = load_config(args.config)
    else:
        full_config = load_config()
    
    # Extract database config
    db_config = full_config.database
    
    # Export data
    success = export_qlib_data(
        freq=args.freq,
        output_dir=Path(args.output_dir).expanduser(),
        qlib_dir=Path(args.qlib_dir).expanduser(),
        ts_codes=args.stocks,
        start_date=start_date,
        end_date=end_date,
        incremental=args.incremental,
        cleanup_csv=args.cleanup_csv,
        config=db_config,
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()


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

from atm.config import DatabaseConfig, load_config
from atm.repo.kline_repo import (
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
from atm.repo.stock_repo import StockBasicRepo

# Import dump_bin tool
tools_qlib_dir = Path(__file__).parent
if str(tools_qlib_dir) not in sys.path:
    sys.path.insert(0, str(tools_qlib_dir))

import dump_bin
DumpDataAll = dump_bin.DumpDataAll

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


def convert_ts_code_to_qlib_format(ts_code: str) -> str:
    """Convert ts_code (e.g., 000001.SZ) to Qlib format (e.g., 000001)."""
    return ts_code.split(".")[0]


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
            return pd.to_datetime(date_str, errors="coerce")
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
            logger.debug(f"No data found for {ts_code}")
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
        
        # Convert ts_code to Qlib format
        qlib_code = convert_ts_code_to_qlib_format(ts_code)
        
        # Save to CSV (ensure column order: date, open, high, low, close, volume, factor)
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


def convert_csv_to_qlib_bin(csv_dir: Path, qlib_dir: Path, freq: str) -> bool:
    """
    Convert CSV files to Qlib bin format.
    
    Args:
        csv_dir: Directory containing CSV files.
        qlib_dir: Output directory for Qlib bin files.
        freq: Frequency string for Qlib (e.g., 'day', 'week').
        
    Returns:
        True if conversion successful, False otherwise.
    """
    logger.info("=" * 80)
    logger.info("Converting CSV files to Qlib bin format...")
    logger.info(f"CSV directory: {csv_dir}")
    logger.info(f"Qlib directory: {qlib_dir}")
    logger.info(f"Frequency: {freq}")
    logger.info("=" * 80)
    
    # Validate CSV files before conversion
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
    
    try:
        dumper = DumpDataAll(
            data_path=str(csv_dir),
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
        config = load_config()
    
    # Initialize repositories
    repo = repo_class(config)
    stock_repo = StockBasicRepo(config)
    
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
                    logger.debug(f"{ts_code}: Incremental export from {stock_start_date.strftime('%Y-%m-%d')}")
                else:
                    logger.debug(f"{ts_code}: Full export (no previous export found)")
            
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
                
                # Cleanup CSV if requested (keep only last row)
                if cleanup_csv:
                    qlib_code = convert_ts_code_to_qlib_format(ts_code)
                    csv_file = csv_dir / f"{qlib_code}.csv"
                    cleanup_csv_file(csv_file)
            else:
                failed_count += 1
                
        except Exception as e:
            logger.error(f"Failed to export {ts_code}: {e}")
            failed_count += 1
    
    logger.info(f"Export completed: {exported_count} stocks exported, {failed_count} failed")
    
    if exported_count == 0:
        logger.error("No stocks exported successfully")
        return False
    
    # Convert CSV to Qlib bin format
    qlib_output_dir = qlib_dir / "qlib_data" / "cn_data"
    if not convert_csv_to_qlib_bin(csv_dir, qlib_output_dir, qlib_freq):
        logger.error("Failed to convert CSV to Qlib bin format")
        return False
    
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
    config = None
    if args.config:
        config = load_config(args.config)
    else:
        config = load_config()
    
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
        config=config,
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""
Export single stock data to Qlib format.

This tool exports daily K-line data for a single stock from the database
and converts it to Qlib's bin format.

Usage:
    python python/tools/qlib/export_single_stock.py sh.000001 [--start-date START_DATE] [--end-date END_DATE] [--output-dir OUTPUT_DIR]
"""

import argparse
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from nq.config import DatabaseConfig, load_config
from nq.repo.kline_repo import StockKlineDayRepo

# Import project's dump_bin module directly (same directory)
# Add the tools/qlib directory to path for direct import
tools_qlib_dir = Path(__file__).parent
if str(tools_qlib_dir) not in sys.path:
    sys.path.insert(0, str(tools_qlib_dir))

# Import dump_bin module directly (must be at top level for pickle to work)
import dump_bin
DumpDataAll = dump_bin.DumpDataAll

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def convert_ts_code_to_qlib_format(ts_code: str) -> str:
    """
    Convert ts_code to Qlib format, keeping exchange suffix.
    
    This function ensures the code format is consistent with the data.
    Qlib code format: '000001.SZ', '600000.SH' (with exchange suffix).
    The returned code must match what dump_bin will extract from the CSV filename.
    
    Args:
        ts_code: Stock code in format like '000001.SZ', '600000.SH', 'sh.000001', or 'sz.000001'

    Returns:
        Qlib format code (e.g., '000001.SZ', '600000.SH') - consistent with data format
    """
    # Handle different input formats and normalize to '000001.SZ' format
    if "." in ts_code:
        parts = ts_code.split(".")
        if len(parts) == 2:
            # If format is 'sh.000001' or 'sz.000001', convert to '000001.SH' or '000001.SZ'
            if parts[0].lower() in ["sh", "sz", "bj"]:
                exchange = parts[0].upper()
                if exchange == "SH":
                    return f"{parts[1]}.SH"
                elif exchange == "SZ":
                    return f"{parts[1]}.SZ"
                elif exchange == "BJ":
                    return f"{parts[1]}.BJ"
            # If format is '000001.SZ' or '600000.SH', keep as is
            return ts_code.upper()
    # If no dot, assume it's a pure code and try to determine exchange
    # For codes starting with 6, assume SH; for others starting with 0/3, assume SZ
    if ts_code.startswith("6"):
        return f"{ts_code}.SH"
    elif ts_code.startswith(("0", "3")):
        return f"{ts_code}.SZ"
    else:
        # Default to SZ if cannot determine
        return f"{ts_code}.SZ"


def export_stock_to_csv(
    kline_repo: StockKlineDayRepo,
    ts_code: str,
    start_date: datetime,
    end_date: datetime,
    output_dir: Path,
) -> bool:
    """
    Export single stock daily K-line data to CSV file in Qlib format.

    Args:
        kline_repo: Daily K-line repository.
        ts_code: Stock code (e.g., 'sh.000001' or '000001.SH').
        start_date: Start date for data export.
        end_date: End date for data export.
        output_dir: Output directory for CSV file.

    Returns:
        True if export successful, False otherwise.
    """
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Fetching data for {ts_code}...")
    logger.info(f"Date range: {start_date.date()} to {end_date.date()}")

    try:
        # Get daily K-line data
        klines = kline_repo.get_by_ts_code(
            ts_code=ts_code,
            start_time=start_date,
            end_time=end_date,
        )

        if not klines:
            logger.error(f"No data found for {ts_code} in the specified date range")
            return False

        logger.info(f"Found {len(klines)} records")

        # Convert models to dictionaries
        kline_dicts = []
        for kline in klines:
            if hasattr(kline, "model_dump"):
                # Pydantic model
                kline_dicts.append(kline.model_dump())
            elif hasattr(kline, "dict"):
                # Pydantic v1 model
                kline_dicts.append(kline.dict())
            elif isinstance(kline, dict):
                # Already a dict
                kline_dicts.append(kline)
            else:
                # Try to convert to dict
                kline_dicts.append(dict(kline))

        if not kline_dicts:
            logger.error(f"No valid data for {ts_code}")
            return False

        # Convert to DataFrame
        df = pd.DataFrame(kline_dicts)

        # Check for date column (try different possible names)
        date_col = None
        for col_name in ["trade_date", "trade_time", "date", "datetime"]:
            if col_name in df.columns:
                date_col = col_name
                break

        if date_col is None:
            logger.error(
                f"No date column found for {ts_code}. "
                f"Available columns: {df.columns.tolist()}"
            )
            return False

        # Check required columns
        required_cols = ["open", "high", "low", "close", "volume"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.error(
                f"Missing columns for {ts_code}: {missing_cols}. "
                f"Available columns: {df.columns.tolist()}"
            )
            return False

        # Convert date column to datetime if it's a string or other format
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        # Remove rows with invalid dates
        df = df.dropna(subset=[date_col])

        if len(df) == 0:
            logger.error(f"No valid data after date conversion for {ts_code}")
            return False

        # Prepare Qlib format DataFrame
        qlib_df = pd.DataFrame()
        qlib_df["date"] = df[date_col].dt.strftime("%Y-%m-%d")

        # Convert price and volume fields, handling None/NaN values
        qlib_df["open"] = pd.to_numeric(df["open"], errors="coerce")
        qlib_df["high"] = pd.to_numeric(df["high"], errors="coerce")
        qlib_df["low"] = pd.to_numeric(df["low"], errors="coerce")
        qlib_df["close"] = pd.to_numeric(df["close"], errors="coerce")
        qlib_df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0).astype(int)
        
        # Add factor field (复权因子) - default to 1.0 if not available
        # If adj_factor or factor exists in source data, use it; otherwise default to 1.0
        if "adj_factor" in df.columns:
            qlib_df["factor"] = pd.to_numeric(df["adj_factor"], errors="coerce").fillna(1.0)
        elif "factor" in df.columns:
            qlib_df["factor"] = pd.to_numeric(df["factor"], errors="coerce").fillna(1.0)
        else:
            # Default factor is 1.0 (no adjustment)
            qlib_df["factor"] = 1.0

        # Remove rows with missing price data
        qlib_df = qlib_df.dropna(subset=["open", "high", "low", "close"])

        if len(qlib_df) == 0:
            logger.error(f"No valid price data for {ts_code}")
            return False

        # Sort by date
        qlib_df = qlib_df.sort_values("date")

        # Convert ts_code to Qlib format (ensure consistency with data)
        # The code format must match what dump_bin will extract from filename
        qlib_code = convert_ts_code_to_qlib_format(ts_code)

        # Save to CSV (ensure column order: date, open, high, low, close, volume, factor)
        # CSV filename uses Qlib format code to ensure consistency with dump_bin extraction
        csv_file = output_dir / f"{qlib_code}.csv"
        # Reorder columns to ensure correct format
        column_order = ["date", "open", "high", "low", "close", "volume", "factor"]
        qlib_df = qlib_df[column_order]
        qlib_df.to_csv(csv_file, index=False, header=False)

        logger.info(f"✓ Exported {len(qlib_df)} records to {csv_file}")
        logger.info(f"  Date range: {qlib_df['date'].min()} to {qlib_df['date'].max()}")

        return True

    except Exception as e:
        logger.error(f"Failed to export {ts_code}: {type(e).__name__}: {e}", exc_info=True)
        return False


def convert_csv_to_qlib_bin(csv_dir: Path, qlib_dir: Path, freq: str = "day") -> bool:
    """
    Convert CSV files to Qlib bin format using project's dump_bin tool.

    Args:
        csv_dir: Directory containing CSV files.
        qlib_dir: Output directory for Qlib bin files.
        freq: Frequency ('day', '1min', etc.).

    Returns:
        True if conversion successful, False otherwise.
    """
    logger.info("=" * 80)
    logger.info("Converting CSV files to Qlib bin format using project dump_bin tool...")
    logger.info(f"CSV directory: {csv_dir}")
    logger.info(f"Qlib directory: {qlib_dir}")
    logger.info(f"Frequency: {freq}")
    logger.info("=" * 80)

    # Validate CSV files before conversion - must have factor column
    csv_files = list(csv_dir.glob("*.csv"))
    if not csv_files:
        logger.error("No CSV files found in directory")
        return False
    
    logger.info(f"Validating {len(csv_files)} CSV files...")
    invalid_files = []
    for csv_file in csv_files[:min(5, len(csv_files))]:  # Check first 5 files
        try:
            # Read first line to check column count
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
        logger.error("Please regenerate CSV files with factor column using the export tool.")
        logger.error("CSV format must be: date,open,high,low,close,volume,factor")
        return False
    
    logger.info("✓ All CSV files have correct format (7 columns including factor)")

    try:
        logger.info("Using project dump_bin tool")

        # Create instance and dump
        logger.info("Initializing dump_bin converter...")
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

                    # Convert YYYY-MM-DD to YYYYMMDD
                    converted_dates = []
                    for date_str in dates:
                        try:
                            # Try parsing as YYYY-MM-DD first
                            dt = pd.to_datetime(date_str)
                            converted_dates.append(dt.strftime("%Y%m%d"))
                        except (ValueError, TypeError):
                            # If already in YYYYMMDD format, keep it
                            if len(date_str) == 8 and date_str.isdigit():
                                converted_dates.append(date_str)
                            else:
                                logger.warning(f"  ⚠ Skipping invalid date format: {date_str}")

                    # Write back in correct format
                    with open(calendar_file, "w", encoding="utf-8") as f:
                        f.write("\n".join(converted_dates))
                        if converted_dates:
                            f.write("\n")

                    logger.info(f"  ✓ Converted {len(converted_dates)} dates to YYYYMMDD format")
                except Exception as e:
                    logger.warning(f"  ⚠ Failed to fix calendar format: {e}")

        logger.info("✓ Successfully converted using project dump_bin tool")
        logger.info(f"Qlib data directory: {qlib_dir}")
        return True

    except ImportError as e:
        logger.error(f"Failed to import project dump_bin module: {e}")
        logger.error("Please ensure dump_bin.py exists in python/tools/qlib/")
        return False
    except Exception as e:
        logger.error(f"Failed to convert CSV to Qlib bin format: {e}", exc_info=True)
        return False


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Export single stock data to Qlib format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export sh.000001 data for the last 2 years
  python python/tools/qlib/export_single_stock.py sh.000001

  # Export with custom date range
  python python/tools/qlib/export_single_stock.py sh.000001 \\
    --start-date 2023-01-01 --end-date 2024-12-31

  # Export to custom output directory
  python python/tools/qlib/export_single_stock.py sh.000001 \\
    --output-dir ~/.qlib/custom_data
        """,
    )

    parser.add_argument(
        "ts_code",
        type=str,
        help="Stock code (e.g., 'sh.000001', 'sz.000001', '000001.SH')",
    )

    parser.add_argument(
        "--start-date",
        type=str,
        help="Start date (YYYY-MM-DD). Default: 2 years ago",
    )

    parser.add_argument(
        "--end-date",
        type=str,
        help="End date (YYYY-MM-DD). Default: today",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="~/.qlib/qlib_data/cn_data",
        help="Output directory for Qlib data (default: ~/.qlib/qlib_data/cn_data)",
    )

    parser.add_argument(
        "--skip-bin",
        action="store_true",
        help="Skip bin conversion, only export CSV",
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to config file (default: config/config.yaml)",
    )

    parser.add_argument(
        "--schema",
        type=str,
        default="quant",
        help="Database schema (default: quant)",
    )

    args = parser.parse_args()

    # Calculate date range
    end_date = datetime.now()
    if args.end_date:
        end_date = datetime.strptime(args.end_date, "%Y-%m-%d")

    start_date = end_date - timedelta(days=365 * 2)  # Default: 2 years ago
    if args.start_date:
        start_date = datetime.strptime(args.start_date, "%Y-%m-%d")

    if start_date >= end_date:
        logger.error("Start date must be before end date")
        return 1

    # Expand output directory
    output_dir = Path(args.output_dir).expanduser()
    csv_dir = output_dir.parent / "csv_data" / output_dir.name
    qlib_dir = output_dir

    logger.info("=" * 80)
    logger.info("Export Single Stock to Qlib Format")
    logger.info("=" * 80)
    logger.info(f"Stock code: {args.ts_code}")
    logger.info(f"Date range: {start_date.date()} to {end_date.date()}")
    logger.info(f"CSV directory: {csv_dir}")
    logger.info(f"Qlib directory: {qlib_dir}")
    logger.info("=" * 80)

    # Load configuration
    try:
        config = load_config(args.config)
        db_config = config.database
    except Exception as e:
        logger.warning(f"Failed to load config from {args.config}: {e}")
        logger.info("Using default database configuration")
        db_config = DatabaseConfig()

    # Initialize repository
    kline_repo = StockKlineDayRepo(db_config, schema=args.schema)

    # Step 1: Export to CSV
    logger.info("Step 1: Exporting stock data to CSV format...")
    success = export_stock_to_csv(
        kline_repo=kline_repo,
        ts_code=args.ts_code,
        start_date=start_date,
        end_date=end_date,
        output_dir=csv_dir,
    )

    if not success:
        logger.error("Failed to export stock data to CSV")
        return 1

    # Step 2: Convert CSV to Qlib bin format
    if not args.skip_bin:
        logger.info("Step 2: Converting CSV to Qlib bin format...")
        success = convert_csv_to_qlib_bin(
            csv_dir=csv_dir,
            qlib_dir=qlib_dir,
            freq="day",
        )

        if not success:
            logger.error("Failed to convert CSV to Qlib bin format")
            return 1
    else:
        logger.info("Step 2: Skipping bin conversion (--skip-bin specified)")

    logger.info("=" * 80)
    logger.info("Export completed successfully!")
    logger.info(f"CSV file: {csv_dir}")
    logger.info(f"Qlib bin files: {qlib_dir}")
    logger.info("=" * 80)
    logger.info("To use the data in Qlib, initialize with:")
    logger.info(f'  qlib.init(provider_uri="{qlib_dir}")')
    logger.info("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())


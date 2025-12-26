#!/usr/bin/env python3
"""
Export daily K-line data to Qlib format.

This tool exports the last 2 years of daily K-line data from the database
and converts it to Qlib's bin format for use in Qlib workflows.

Usage:
    python python/tests/qlib/export_to_qlib.py [--output-dir OUTPUT_DIR] [--years YEARS]
"""

import argparse
import logging
import os
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from atm.config import DatabaseConfig, load_config
from atm.repo.kline_repo import StockKlineDayRepo
from atm.repo.stock_repo import StockBasicRepo

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def convert_ts_code_to_qlib_format(ts_code: str) -> str:
    """
    Convert ts_code to Qlib format.

    Args:
        ts_code: Stock code in format like '000001.SZ' or '600000.SH'

    Returns:
        Qlib format code like '000001' or '600000'
    """
    # Remove exchange suffix (.SZ, .SH, etc.)
    code = ts_code.split(".")[0]
    return code


def export_daily_data_to_csv(
    kline_repo: StockKlineDayRepo,
    stock_repo: StockBasicRepo,
    start_date: datetime,
    end_date: datetime,
    output_dir: Path,
    max_stocks: Optional[int] = None,
) -> int:
    """
    Export daily K-line data to CSV files in Qlib format.

    Args:
        kline_repo: Daily K-line repository.
        stock_repo: Stock basic repository.
        start_date: Start date for data export.
        end_date: End date for data export.
        output_dir: Output directory for CSV files.
        max_stocks: Maximum number of stocks to export (None for all).

    Returns:
        Number of stocks exported.
    """
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get all stocks
    logger.info("Fetching stock list...")
    all_stocks = stock_repo.get_all()
    if max_stocks:
        all_stocks = all_stocks[:max_stocks]
    logger.info(f"Found {len(all_stocks)} stocks to process")

    exported_count = 0
    failed_count = 0

    for idx, stock in enumerate(all_stocks):
        if (idx + 1) % 100 == 0:
            logger.info(f"Processing {idx + 1}/{len(all_stocks)} stocks...")

        try:
            # Get daily K-line data
            klines = kline_repo.get_by_ts_code(
                ts_code=stock.ts_code,
                start_time=start_date,
                end_time=end_date,
            )

            if not klines:
                logger.debug(f"No data for {stock.ts_code}")
                failed_count += 1
                continue

            # Convert models to dictionaries
            # Handle both model objects and dicts
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
                logger.debug(f"No valid data for {stock.ts_code}")
                failed_count += 1
                continue

            # Convert to DataFrame
            df = pd.DataFrame(kline_dicts)

            # Check for date column (try different possible names)
            date_col = None
            for col_name in ["trade_date", "trade_time", "date", "datetime"]:
                if col_name in df.columns:
                    date_col = col_name
                    break

            if date_col is None:
                logger.warning(
                    f"No date column found for {stock.ts_code}. "
                    f"Available columns: {df.columns.tolist()}"
                )
                failed_count += 1
                continue

            # Check required columns
            required_cols = ["open", "high", "low", "close", "volume"]
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                logger.warning(
                    f"Missing columns for {stock.ts_code}: {missing_cols}. "
                    f"Available columns: {df.columns.tolist()}"
                )
                failed_count += 1
                continue

            # Convert date column to datetime if it's a string or other format
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
            # Remove rows with invalid dates
            df = df.dropna(subset=[date_col])

            if len(df) == 0:
                logger.debug(f"No valid data after date conversion for {stock.ts_code}")
                failed_count += 1
                continue

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
                logger.debug(f"No valid price data for {stock.ts_code}")
                failed_count += 1
                continue

            # Sort by date
            qlib_df = qlib_df.sort_values("date")

            # Convert ts_code to Qlib format
            qlib_code = convert_ts_code_to_qlib_format(stock.ts_code)

            # Save to CSV (ensure column order: date, open, high, low, close, volume, factor)
            csv_file = output_dir / f"{qlib_code}.csv"
            # Reorder columns to ensure correct format
            column_order = ["date", "open", "high", "low", "close", "volume", "factor"]
            qlib_df = qlib_df[column_order]
            qlib_df.to_csv(csv_file, index=False, header=False)

            exported_count += 1

        except KeyError as e:
            logger.warning(
                f"Failed to export {stock.ts_code}: Missing key {e}. "
                f"This might indicate a data format issue."
            )
            failed_count += 1
            continue
        except Exception as e:
            logger.warning(
                f"Failed to export {stock.ts_code}: {type(e).__name__}: {e}"
            )
            if logger.isEnabledFor(logging.DEBUG):
                import traceback
                logger.debug(traceback.format_exc())
            failed_count += 1
            continue

    logger.info(
        f"Export completed: {exported_count} stocks exported, {failed_count} failed"
    )
    return exported_count


def fix_calendar_format(qlib_dir: Path, freq: str = "day") -> None:
    """
    Fix calendar file format from YYYY-MM-DD to YYYYMMDD (Qlib standard format).
    
    Args:
        qlib_dir: Qlib data directory.
        freq: Frequency ('day', '1min', etc.).
    """
    if freq != "day":
        return  # Only fix day frequency
    
    calendar_file = qlib_dir / "calendars" / f"{freq}.txt"
    if not calendar_file.exists():
        return
    
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
                # Otherwise skip invalid format

        # Write back in correct format
        if converted_dates:
            with open(calendar_file, "w", encoding="utf-8") as f:
                f.write("\n".join(converted_dates))
                f.write("\n")
            logger.debug(f"Fixed calendar format: {len(converted_dates)} dates converted to YYYYMMDD")
    except Exception as e:
        logger.warning(f"Failed to fix calendar format: {e}")


def convert_csv_to_qlib_bin(csv_dir: Path, qlib_dir: Path, freq: str = "day") -> bool:
    """
    Convert CSV files to Qlib bin format using qlib's official dump_bin tool.
    
    This function tries multiple methods to use qlib's official conversion tool:
    1. Direct import from qlib.tools.dump_bin
    2. Command-line interface via python -m qlib.tools.dump_bin
    3. Direct script execution if found in qlib installation
    4. Fallback to manual conversion (Method 4)

    Args:
        csv_dir: Directory containing CSV files.
        qlib_dir: Output directory for Qlib bin files.
        freq: Frequency ('day', '1min', etc.).

    Returns:
        True if conversion successful, False otherwise.
    """
    try:
        import qlib

        logger.info("=" * 80)
        logger.info("Converting CSV files to Qlib bin format using official tools...")
        logger.info(f"CSV directory: {csv_dir}")
        logger.info(f"Qlib directory: {qlib_dir}")
        logger.info(f"Frequency: {freq}")
        logger.info("=" * 80)

        # Method 1: Try to use qlib's dump_bin function directly (Official API)
        try:
            # Try different import paths
            dump_all_func = None
            import_paths = [
                "qlib.tools.dump_bin",
                "qlib.tools.data.dump_bin",
                "qlib.data.tools.dump_bin",
            ]
            
            for import_path in import_paths:
                try:
                    module = __import__(import_path, fromlist=["dump_all"])
                    dump_all_func = getattr(module, "dump_all", None)
                    if dump_all_func:
                        logger.info(f"✓ Found dump_all in {import_path}")
                        break
                except (ImportError, AttributeError):
                    continue
            
            if dump_all_func:
                logger.info("Using qlib official dump_all() function...")
                # Convert CSV to Qlib bin format
                # Note: Check function signature - some versions use different parameter names
                try:
                    # Try with keyword arguments (newer versions)
                    dump_all_func(
                        csv_path=str(csv_dir),
                        qlib_dir=str(qlib_dir),
                        include_fields="open,close,high,low,volume,factor",
                        freq=freq,
                        max_workers=4,
                    )
                except TypeError:
                    # Try with positional arguments or different parameter names
                    import inspect
                    sig = inspect.signature(dump_all_func)
                    params = list(sig.parameters.keys())
                    logger.debug(f"dump_all signature: {params}")
                    
                    # Try common parameter variations
                    kwargs = {}
                    if "csv_path" in params:
                        kwargs["csv_path"] = str(csv_dir)
                    elif "csv_dir" in params:
                        kwargs["csv_dir"] = str(csv_dir)
                    if "qlib_dir" in params:
                        kwargs["qlib_dir"] = str(qlib_dir)
                    if "include_fields" in params:
                        kwargs["include_fields"] = "open,close,high,low,volume,factor"
                    if "freq" in params:
                        kwargs["freq"] = freq
                    if "max_workers" in params:
                        kwargs["max_workers"] = 4
                    
                    dump_all_func(**kwargs)

                logger.info("✓ Successfully converted using qlib.tools.dump_bin.dump_all()")
                fix_calendar_format(qlib_dir, freq)
                logger.info(f"Qlib data directory: {qlib_dir}")
                return True
            else:
                raise ImportError("dump_all function not found in qlib.tools")

        except (ImportError, AttributeError, TypeError) as e:
            logger.debug(f"Direct import failed: {e}, trying CLI method...")

        # Method 2: Try using qlib CLI command (Official CLI)
        try:
            logger.info("Trying qlib official CLI command...")
            
            # Try different command formats
            cmd_variants = [
                # Format 1: python -m qlib.tools.dump_bin dump_all ...
                [
                    sys.executable,
                    "-m",
                    "qlib.tools.dump_bin",
                    "dump_all",
                    "--csv_path", str(csv_dir),
                    "--qlib_dir", str(qlib_dir),
                    "--include_fields", "open,close,high,low,volume,factor",
                    "--freq", freq,
                ],
                # Format 2: python -m qlib.tools.dump_bin --csv_path ... (without dump_all)
                [
                    sys.executable,
                    "-m",
                    "qlib.tools.dump_bin",
                    "--csv_path", str(csv_dir),
                    "--qlib_dir", str(qlib_dir),
                    "--include_fields", "open,close,high,low,volume,factor",
                    "--freq", freq,
                ],
            ]
            
            for cmd in cmd_variants:
                logger.info(f"Running command: {' '.join(cmd)}")
                result = subprocess.run(
                    cmd, 
                    capture_output=True, 
                    text=True, 
                    timeout=3600,
                    cwd=str(csv_dir.parent)  # Run from parent directory
                )

                if result.returncode == 0:
                    logger.info("✓ Successfully converted using qlib CLI")
                    fix_calendar_format(qlib_dir, freq)
                    logger.info(f"Qlib data directory: {qlib_dir}")
                    if result.stdout:
                        logger.debug(f"Output: {result.stdout[:500]}")  # First 500 chars
                    return True
                else:
                    logger.debug(f"Command variant failed with return code {result.returncode}")
                    if result.stderr:
                        logger.debug(f"Error: {result.stderr[:500]}")
                    continue  # Try next variant
            
            # If all variants failed, log the last error
            logger.warning("All CLI command variants failed")
            if result.stderr:
                logger.warning(f"Last error output: {result.stderr[:1000]}")

        except subprocess.TimeoutExpired:
            logger.error("Conversion command timed out after 1 hour")
            return False
        except Exception as e:
            logger.debug(f"CLI approach failed: {e}")

        # Method 3: Try to find and run dump_bin.py script directly (Official script)
        try:
            logger.info("Searching for qlib dump_bin.py script...")
            import site
            import importlib.util
            import glob

            # Search in common locations
            search_paths = []
            
            # Get qlib package path
            try:
                qlib_spec = importlib.util.find_spec("qlib")
                if qlib_spec and qlib_spec.origin:
                    qlib_path = Path(qlib_spec.origin).parent
                    search_paths.extend([
                        qlib_path / "tools" / "dump_bin.py",
                        qlib_path / "scripts" / "dump_bin.py",
                        qlib_path.parent / "scripts" / "dump_bin.py",
                        qlib_path / "data" / "tools" / "dump_bin.py",
                    ])
            except Exception as e:
                logger.debug(f"Could not get qlib spec: {e}")

            # Search in site-packages using glob
            for site_dir in site.getsitepackages():
                search_patterns = [
                    str(Path(site_dir) / "qlib" / "**" / "dump_bin.py"),
                    str(Path(site_dir) / "qlib" / "**" / "dump_bin*.py"),
                ]
                for pattern in search_patterns:
                    found = glob.glob(pattern, recursive=True)
                    search_paths.extend([Path(p) for p in found])

            dump_bin_script = None
            for path in search_paths:
                if path.exists() and path.is_file():
                    dump_bin_script = path
                    logger.info(f"✓ Found dump_bin.py at: {dump_bin_script}")
                    break

            if dump_bin_script:
                # Try different command formats
                cmd_variants = [
                    [
                        sys.executable,
                        str(dump_bin_script),
                        "dump_all",
                        "--csv_path", str(csv_dir),
                        "--qlib_dir", str(qlib_dir),
                        "--include_fields", "open,close,high,low,volume,factor",
                        "--freq", freq,
                    ],
                    [
                        sys.executable,
                        str(dump_bin_script),
                        "--csv_path", str(csv_dir),
                        "--qlib_dir", str(qlib_dir),
                        "--include_fields", "open,close,high,low,volume,factor",
                        "--freq", freq,
                    ],
                ]
                
                for cmd in cmd_variants:
                    logger.info(f"Running script: {' '.join(cmd)}")
                    result = subprocess.run(
                        cmd, 
                        capture_output=True, 
                        text=True, 
                        timeout=3600,
                        cwd=str(dump_bin_script.parent)
                    )

                    if result.returncode == 0:
                        logger.info("✓ Successfully converted using dump_bin.py script")
                        fix_calendar_format(qlib_dir, freq)
                        logger.info(f"Qlib data directory: {qlib_dir}")
                        return True
                    else:
                        logger.debug(f"Script variant failed: {result.returncode}")
                        if result.stderr:
                            logger.debug(f"Error: {result.stderr[:500]}")
                        continue
            else:
                logger.debug("dump_bin.py script not found in qlib installation")

        except Exception as e:
            logger.debug(f"Script search approach failed: {e}")

        # Method 4: Fallback - Manual conversion (when official tools are not available)
        try:
            logger.info("Official qlib tools not available, using manual conversion method...")
            logger.info("(This is a fallback method - official tools are preferred)")
            from qlib.data import D
            import pickle
            import struct
            from pathlib import Path as PathLib

            # Create qlib directory structure
            qlib_dir.mkdir(parents=True, exist_ok=True)
            calendars_dir = qlib_dir / "calendars"
            instruments_dir = qlib_dir / "instruments"
            features_dir = qlib_dir / "features"
            
            calendars_dir.mkdir(exist_ok=True)
            instruments_dir.mkdir(exist_ok=True)
            features_dir.mkdir(exist_ok=True)

            # Get all CSV files
            csv_files = list(csv_dir.glob("*.csv"))
            if not csv_files:
                logger.error(f"No CSV files found in {csv_dir}")
                return False

            logger.info(f"Processing {len(csv_files)} CSV files...")

            # Collect all dates and instruments
            all_dates = set()
            instruments = []

            for csv_file in csv_files:
                try:
                    # Read CSV
                    df = pd.read_csv(
                        csv_file,
                        header=None,
                        names=["date", "open", "high", "low", "close", "volume"],
                    )
                    if len(df) == 0:
                        continue

                    # Collect dates
                    df["date"] = pd.to_datetime(df["date"])
                    all_dates.update(df["date"].dt.strftime("%Y%m%d"))

                    # Get instrument code from filename
                    instrument = csv_file.stem
                    instruments.append(instrument)

                except Exception as e:
                    logger.warning(f"Failed to process {csv_file}: {e}")
                    continue

            if not instruments:
                logger.error("No valid instruments found")
                return False

            # Create calendar file
            calendar = sorted(all_dates)
            calendar_file = calendars_dir / f"{freq}.txt"
            with open(calendar_file, "w") as f:
                f.write("\n".join(calendar))

            # Create instruments file
            # Qlib requires UTF-8 encoding and proper line endings
            instruments_file = instruments_dir / "all.txt"
            with open(instruments_file, "w", encoding="utf-8") as f:
                # Write each instrument on a separate line, ensure no trailing newline issues
                sorted_instruments = sorted(instruments)
                f.write("\n".join(sorted_instruments))
                # Ensure file ends with newline (some qlib versions require this)
                if sorted_instruments:
                    f.write("\n")

            # Process each CSV file and create bin files
            # Important: bin files must be aligned with calendar
            logger.info("Creating bin files (aligned with calendar)...")
            processed = 0
            
            # Create a date index from calendar for alignment
            calendar_dates = pd.to_datetime(calendar, format="%Y%m%d")
            calendar_index = pd.Index(calendar_dates)
            
            for csv_file in csv_files:
                try:
                    instrument = csv_file.stem
                    df = pd.read_csv(
                        csv_file,
                        header=None,
                        names=["date", "open", "high", "low", "close", "volume"],
                    )
                    df["date"] = pd.to_datetime(df["date"])
                    df = df.sort_values("date")
                    df = df.set_index("date")

                    # Align with calendar (fill missing dates with NaN)
                    df_aligned = df.reindex(calendar_index)

                    # Create feature directory for this instrument
                    feature_dir = features_dir / instrument
                    feature_dir.mkdir(exist_ok=True)

                    # Save each field as a bin file
                    # Qlib expects float32 arrays aligned with calendar
                    for field in ["open", "close", "high", "low", "volume", "factor"]:
                        field_file = feature_dir / f"{field}.{freq}.bin"
                        # Convert to numpy array, handle NaN values
                        values = df_aligned[field].values.astype("float32")
                        # Replace NaN with 0 (or you can use np.nan)
                        import numpy as np
                        values = np.nan_to_num(values, nan=0.0)
                        
                        with open(field_file, "wb") as f:
                            f.write(values.tobytes())

                    processed += 1
                    if processed % 100 == 0:
                        logger.info(f"Processed {processed}/{len(csv_files)} files...")

                except Exception as e:
                    logger.warning(f"Failed to process {csv_file}: {e}")
                    import traceback
                    logger.debug(traceback.format_exc())
                    continue

            logger.info(f"Successfully converted {processed} CSV files to Qlib bin format")
            fix_calendar_format(qlib_dir, freq)
            logger.info(f"Qlib data directory: {qlib_dir}")
            return True

        except Exception as e:
            logger.debug(f"Direct API approach failed: {e}")

        # If all methods failed, provide helpful error message
        logger.error("=" * 80)
        logger.error("Failed to convert CSV to Qlib bin format using official tools.")
        logger.error("=" * 80)
        logger.error("Troubleshooting steps:")
        logger.error("1. Check qlib installation:")
        logger.error("   python -c 'import qlib; print(qlib.__file__)'")
        logger.error("2. Try upgrading qlib:")
        logger.error("   pip install --upgrade pyqlib")
        logger.error("3. Check if dump_bin is available:")
        logger.error("   python -c 'from qlib.tools.dump_bin import dump_all; print(\"OK\")'")
        logger.error("4. Manual conversion was attempted as fallback.")
        logger.error("=" * 80)
        return False

    except ImportError:
        logger.error(
            "Qlib is not installed. Please install it with: pip install pyqlib"
        )
        return False
    except Exception as e:
        logger.error(f"Failed to convert CSV to Qlib bin format: {e}", exc_info=True)
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Export daily K-line data to Qlib format"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="~/.qlib/csv_data/cn_data",
        help="Output directory for CSV files (default: ~/.qlib/csv_data/cn_data)",
    )
    parser.add_argument(
        "--qlib-dir",
        type=str,
        default="~/.qlib/qlib_data/cn_data",
        help="Output directory for Qlib bin files (default: ~/.qlib/qlib_data/cn_data)",
    )
    parser.add_argument(
        "--years",
        type=int,
        default=2,
        help="Number of years of data to export (default: 2)",
    )
    parser.add_argument(
        "--max-stocks",
        type=int,
        default=None,
        help="Maximum number of stocks to export (default: all)",
    )
    parser.add_argument(
        "--skip-csv",
        action="store_true",
        help="Skip CSV export if CSV directory already exists",
    )
    parser.add_argument(
        "--skip-bin",
        action="store_true",
        help="Skip bin conversion",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to configuration file (default: config/config.yaml)",
    )
    parser.add_argument(
        "--schema",
        type=str,
        default="quant",
        help="Database schema name (default: quant)",
    )

    args = parser.parse_args()

    # Expand user paths
    csv_dir = Path(args.output_dir).expanduser()
    qlib_dir = Path(args.qlib_dir).expanduser()

    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=args.years * 365)

    logger.info("=" * 80)
    logger.info("Qlib Data Export Tool")
    logger.info("=" * 80)
    logger.info(f"Date range: {start_date.date()} to {end_date.date()}")
    logger.info(f"CSV output directory: {csv_dir}")
    logger.info(f"Qlib output directory: {qlib_dir}")
    logger.info("=" * 80)

    # Load configuration
    try:
        config = load_config(args.config)
        db_config = config.database
    except Exception as e:
        logger.warning(f"Failed to load config from {args.config}: {e}")
        logger.info("Using default database configuration")
        db_config = DatabaseConfig()

    # Initialize repositories
    kline_repo = StockKlineDayRepo(db_config, schema=args.schema)
    stock_repo = StockBasicRepo(db_config, schema=args.schema)

    # Step 1: Export to CSV
    if not args.skip_csv or not csv_dir.exists():
        logger.info("Step 1: Exporting daily data to CSV format...")
        exported_count = export_daily_data_to_csv(
            kline_repo=kline_repo,
            stock_repo=stock_repo,
            start_date=start_date,
            end_date=end_date,
            output_dir=csv_dir,
            max_stocks=args.max_stocks,
        )

        if exported_count == 0:
            logger.error("No stocks exported. Exiting.")
            return 1
    else:
        logger.info(f"Step 1: Skipping CSV export (directory exists: {csv_dir})")
        exported_count = len(list(csv_dir.glob("*.csv")))
        logger.info(f"Found {exported_count} existing CSV files")

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
        logger.info("Step 2: Skipping bin conversion")

    logger.info("=" * 80)
    logger.info("Export completed successfully!")
    logger.info(f"CSV files: {csv_dir}")
    logger.info(f"Qlib bin files: {qlib_dir}")
    logger.info("=" * 80)
    logger.info("To use the data in Qlib, initialize with:")
    logger.info(f'  qlib.init(provider_uri="{qlib_dir}")')
    logger.info("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())


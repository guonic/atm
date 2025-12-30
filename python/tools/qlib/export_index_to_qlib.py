#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
export_index_to_qlib.py

Export index data (benchmark) to Qlib format.

This script collects index daily data from Tushare and exports it to Qlib format,
so that it can be used as benchmark in backtesting.

Usage:
    # Export CSI300 index data
    python python/tools/qlib/export_index_to_qlib.py \
        --index_code 000300.SH \
        --start_date 2020-01-01 \
        --end_date 2024-12-31 \
        --qlib_dir ~/.qlib/qlib_data/cn_data

    # Export multiple indices
    python python/tools/qlib/export_index_to_qlib.py \
        --index_code 000300.SH 000905.SH 399001.SZ \
        --start_date 2020-01-01 \
        --end_date 2024-12-31

Arguments:
    --index_code      Index code(s) in Qlib format (e.g., 000300.SH, 000905.SH)
    --start_date      Start date (YYYY-MM-DD)
    --end_date        End date (YYYY-MM-DD)
    --qlib_dir        Qlib data directory (default: ~/.qlib/qlib_data/cn_data)
    --token           Tushare token (or set TUSHARE_TOKEN env var)
    --config_path     Path to config file (for database config)
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from nq.data.source import TushareSource, TushareSourceConfig

# Import dump_bin tool
tools_qlib_dir = Path(__file__).parent
if str(tools_qlib_dir) not in sys.path:
    sys.path.insert(0, str(tools_qlib_dir))

import dump_bin
DumpDataAll = dump_bin.DumpDataAll
DumpDataFix = dump_bin.DumpDataFix
DumpDataUpdate = dump_bin.DumpDataUpdate

# Import functions from regenerate_instruments_calendar for preserving stock data
# Note: logger is defined after this, so we can't use it here
try:
    from regenerate_instruments_calendar import (
        scan_features_directory,
        regenerate_instruments,
        is_index_code,
    )
except ImportError:
    # Fallback: define minimal functions if import fails
    scan_features_directory = None
    regenerate_instruments = None
    is_index_code = None

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def convert_index_code_to_tushare_format(index_code: str) -> str:
    """
    Convert Qlib index code to Tushare format.
    
    Qlib format: 000300.SH -> Tushare format: 000300.SH (same)
    But Tushare index_daily API may need different format.
    
    Args:
        index_code: Index code in Qlib format (e.g., '000300.SH').
    
    Returns:
        Index code in Tushare format.
    """
    # For most cases, Qlib format is the same as Tushare format
    # But we may need to handle special cases
    return index_code


def fetch_index_data_from_tushare(
    index_code: str,
    start_date: str,
    end_date: str,
    tushare_token: str,
) -> pd.DataFrame:
    """
    Fetch index daily data from Tushare.
    
    Args:
        index_code: Index code in Qlib format (e.g., '000300.SH').
        start_date: Start date (YYYY-MM-DD).
        end_date: End date (YYYY-MM-DD).
        tushare_token: Tushare Pro API token.
    
    Returns:
        DataFrame with columns: date, open, high, low, close, volume, factor
    """
    logger.info(f"Fetching index data for {index_code} from {start_date} to {end_date}")
    
    # Convert date format
    start_date_ts = start_date.replace("-", "")
    end_date_ts = end_date.replace("-", "")
    
    # Convert index code to Tushare format
    ts_index_code = convert_index_code_to_tushare_format(index_code)
    
    # Initialize Tushare source
    config = TushareSourceConfig(
        token=tushare_token,
        type="tushare",
    )
    source = TushareSource(config)
    
    # Fetch index daily data
    try:
        records = list(source.fetch(
            api_name="index_daily",
            ts_code=ts_index_code,
            start_date=start_date_ts,
            end_date=end_date_ts,
        ))
        
        if not records:
            logger.warning(f"No data found for {index_code}")
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(records)
        
        # Rename columns to match Qlib format
        column_mapping = {
            "trade_date": "date",
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "vol": "volume",  # Tushare uses 'vol', Qlib uses 'volume'
        }
        
        # Select and rename columns
        df = df.rename(columns=column_mapping)
        
        # Convert date format from YYYYMMDD to YYYY-MM-DD
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], format="%Y%m%d").dt.strftime("%Y-%m-%d")
        
        # Handle factor column
        # Tushare index_daily API typically doesn't return factor/adj_factor for indices
        # For indices, factor is usually 1.0 (no adjustment needed)
        # Check if factor or adj_factor exists in original data
        if "factor" not in df.columns:
            # Check if adj_factor exists (some APIs might return this)
            if "adj_factor" in df.columns:
                df["factor"] = df["adj_factor"]
                df = df.drop(columns=["adj_factor"])
            else:
                # Default to 1.0 for index data (indices typically don't need adjustment)
                df["factor"] = 1.0
                logger.debug(f"Factor column not found in Tushare data, defaulting to 1.0 for index {index_code}")
        
        # Select required columns in correct order
        required_columns = ["date", "open", "high", "low", "close", "volume", "factor"]
        df = df[[col for col in required_columns if col in df.columns]]
        
        # Sort by date
        df = df.sort_values("date").reset_index(drop=True)
        
        logger.info(f"✓ Fetched {len(df)} records for {index_code}")
        return df
        
    except Exception as e:
        logger.error(f"Failed to fetch index data for {index_code}: {e}", exc_info=True)
        raise


def export_index_to_csv(
    index_code: str,
    df: pd.DataFrame,
    csv_dir: Path,
) -> Path:
    """
    Export index data to CSV file.
    
    Args:
        index_code: Index code (e.g., '000300.SH').
        df: DataFrame with index data.
        csv_dir: Directory to save CSV file.
    
    Returns:
        Path to created CSV file.
    """
    csv_dir.mkdir(parents=True, exist_ok=True)
    
    # CSV filename: use index code (e.g., 000300.SH.csv)
    csv_file = csv_dir / f"{index_code}.csv"
    
    # Save to CSV (no header, as required by Qlib)
    df.to_csv(csv_file, index=False, header=False)
    
    logger.info(f"✓ Exported CSV: {csv_file}")
    return csv_file


def convert_index_csv_to_qlib_bin(
    index_code: str,
    csv_dir: Path,
    qlib_dir: Path,
    freq: str = "day",
) -> bool:
    """
    Convert index CSV to Qlib bin format using incremental update.
    
    This function uses DumpDataFix to preserve existing calendar and instruments,
    and only adds the new index data.
    
    Args:
        index_code: Index code (e.g., '000300.SH').
        csv_dir: Directory containing CSV file.
        qlib_dir: Qlib data directory.
        freq: Data frequency (default: 'day').
    
    Returns:
        True if successful, False otherwise.
    """
    csv_file = csv_dir / f"{index_code}.csv"
    
    if not csv_file.exists():
        logger.error(f"CSV file not found: {csv_file}")
        return False
    
    logger.info(f"Converting {index_code} to Qlib bin format (incremental update)...")
    
    try:
        # Check if qlib data already exists
        calendar_file = qlib_dir / "calendars" / f"{freq}.txt"
        instruments_file = qlib_dir / "instruments" / "all.txt"
        
        if calendar_file.exists() and instruments_file.exists():
            # Use DumpDataFix for incremental update (preserves existing calendar and instruments)
            logger.info("Existing Qlib data found, using incremental update mode...")
            dumper = DumpDataFix(
                data_path=str(csv_dir),
                qlib_dir=str(qlib_dir),
                freq=freq,
                max_workers=1,
                date_field_name="date",
                file_suffix=".csv",
                include_fields="open,close,high,low,volume,factor",
            )
            
            logger.info("Starting incremental conversion...")
            dumper.dump()
            
            logger.info(f"✓ Successfully converted {index_code} to Qlib format (incremental update)")
            logger.info("✓ Calendar and instruments preserved, only index data added")
        else:
            # First time export, use DumpDataAll
            logger.info("No existing Qlib data found, using full export mode...")
            dumper = DumpDataAll(
                data_path=str(csv_dir),
                qlib_dir=str(qlib_dir),
                freq=freq,
                max_workers=1,
                date_field_name="date",
                file_suffix=".csv",
                include_fields="open,close,high,low,volume,factor",
            )
            
            logger.info("Starting full conversion...")
            dumper.dump()
            
            logger.info(f"✓ Successfully converted {index_code} to Qlib format (full export)")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to convert {index_code} to Qlib format: {e}", exc_info=True)
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Export index data to Qlib format for benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--index_code",
        nargs="+",
        required=True,
        help="Index code(s) in Qlib format (e.g., 000300.SH, 000905.SH, 399001.SZ)",
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
        "--qlib_dir",
        type=str,
        default="~/.qlib/qlib_data/cn_data",
        help="Qlib data directory (default: ~/.qlib/qlib_data/cn_data)",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=os.getenv("TUSHARE_TOKEN", ""),
        help="Tushare Pro API token (or set TUSHARE_TOKEN env var)",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default=None,
        help="Path to config file (optional)",
    )
    parser.add_argument(
        "--freq",
        type=str,
        default="day",
        help="Data frequency (default: day)",
    )
    
    args = parser.parse_args()
    
    # Validate token
    if not args.token:
        logger.error(
            "Tushare token is required. Set TUSHARE_TOKEN environment variable or use --token option."
        )
        return 1
    
    # Expand paths
    qlib_dir = Path(args.qlib_dir).expanduser()
    csv_dir = qlib_dir / "csv_index_data"
    csv_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each index
    success_count = 0
    for index_code in args.index_code:
        logger.info(f"\n{'='*80}")
        logger.info(f"Processing index: {index_code}")
        logger.info(f"{'='*80}")
        
        try:
            # Step 1: Fetch data from Tushare
            df = fetch_index_data_from_tushare(
                index_code=index_code,
                start_date=args.start_date,
                end_date=args.end_date,
                tushare_token=args.token,
            )
            
            if df.empty:
                logger.warning(f"No data for {index_code}, skipping")
                continue
            
            # Step 2: Export to CSV
            csv_file = export_index_to_csv(
                index_code=index_code,
                df=df,
                csv_dir=csv_dir,
            )
            
            # Step 3: Convert to Qlib bin format
            if convert_index_csv_to_qlib_bin(
                index_code=index_code,
                csv_dir=csv_dir,
                qlib_dir=qlib_dir,
                freq=args.freq,
            ):
                success_count += 1
                logger.info(f"✓ Successfully exported {index_code}")
            else:
                logger.error(f"✗ Failed to export {index_code}")
                
        except Exception as e:
            logger.error(f"✗ Error processing {index_code}: {e}", exc_info=True)
            continue
    
    # Summary
    logger.info(f"\n{'='*80}")
    logger.info(f"Export Summary: {success_count}/{len(args.index_code)} indices exported successfully")
    logger.info(f"{'='*80}")
    
    if success_count > 0:
        logger.info(f"\n✓ Index data exported to: {qlib_dir}")
        logger.info(f"You can now use these indices as benchmarks in backtesting:")
        for index_code in args.index_code:
            logger.info(f"  --benchmark {index_code}")
        
        # Final check: verify instruments/all.txt contains both stocks and indices
        instruments_file = qlib_dir / "instruments" / "all.txt"
        if instruments_file.exists():
            try:
                with open(instruments_file, "r", encoding="utf-8") as f:
                    instruments = [line.strip() for line in f if line.strip()]
                
                # Count stocks vs indices
                stock_count = 0
                index_count = 0
                for inst in instruments:
                    # Check if it's an index (simple heuristic)
                    inst_upper = inst.upper()
                    if is_index_code and is_index_code(inst):
                        index_count += 1
                    else:
                        stock_count += 1
                
                logger.info(f"\n✓ Instruments file contains: {stock_count} stocks, {index_count} indices (total: {len(instruments)})")
                
                if stock_count == 0:
                    logger.warning(
                        "⚠️  WARNING: instruments/all.txt contains no stocks, only indices! "
                        "This may indicate a problem. Please run regenerate_instruments_calendar.py to fix."
                    )
                elif stock_count < 100:
                    logger.warning(
                        f"⚠️  WARNING: instruments/all.txt contains only {stock_count} stocks, "
                        f"which seems low. Expected ~800+ stocks. "
                        "Please run regenerate_instruments_calendar.py to verify."
                    )
            except Exception as e:
                logger.warning(f"Could not verify instruments file: {e}")
    
    return 0 if success_count == len(args.index_code) else 1


if __name__ == "__main__":
    sys.exit(main())


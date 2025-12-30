#!/usr/bin/env python3
"""
Comprehensive tool to verify exported Qlib data correctness and check coverage.

This tool provides:
1. Data Verification:
   - File structure (calendars, instruments, features directories)
   - Calendar file format and content
   - Instruments file format and content
   - Feature bin files existence and size
   - Data loading via Qlib API
   - Data integrity (missing values, date alignment, etc.)

2. Data Coverage Analysis:
   - Total number of instruments
   - Date range (earliest and latest dates)
   - Date range for each instrument
   - Data completeness statistics
   - Missing data analysis
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

import numpy as np
import pandas as pd
import qlib
from qlib.data import D

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class QlibDataVerifier:
    """Comprehensive Qlib data verification tool."""

    def __init__(self, qlib_dir: str = "~/.qlib/qlib_data/cn_data", region: str = "cn", freq: str = "day"):
        """
        Initialize the verifier.

        Parameters
        ----------
        qlib_dir : str
            Path to Qlib data directory
        region : str
            Qlib region (default: "cn")
        freq : str
            Data frequency (default: "day")
        """
        self.qlib_dir = Path(qlib_dir.replace("~", str(Path.home()))).expanduser()
        self.region = region
        self.freq = freq
        self.errors = []
        self.warnings = []
        self.info = []

    def verify_all(self, check_coverage: bool = False, detailed_coverage: bool = False) -> bool:
        """
        Run all verification checks.
        
        Parameters
        ----------
        check_coverage : bool
            Whether to include coverage analysis
        detailed_coverage : bool
            Whether to show detailed instrument information in coverage report
        """
        logger.info("=" * 80)
        logger.info("Qlib Data Verification")
        logger.info("=" * 80)
        logger.info(f"Data directory: {self.qlib_dir}")
        logger.info("")

        # 1. Check directory structure
        logger.info("Step 1: Checking directory structure...")
        if not self._check_directory_structure():
            return False

        # 2. Check calendar file
        logger.info("\nStep 2: Checking calendar file...")
        calendar_dates = self._check_calendar_file()
        if calendar_dates is None:
            return False

        # 3. Check instruments file
        logger.info("\nStep 3: Checking instruments file...")
        instruments = self._check_instruments_file()
        if instruments is None:
            return False

        # 4. Check feature files
        logger.info("\nStep 4: Checking feature bin files...")
        if not self._check_feature_files(instruments):
            return False

        # 5. Coverage analysis (if requested)
        if check_coverage:
            logger.info("\n" + "=" * 80)
            logger.info("Coverage Analysis")
            logger.info("=" * 80)
            self._check_data_coverage(calendar_dates, instruments, detailed_coverage)

        # 6. Initialize Qlib and check API loading
        logger.info("\nStep 5: Testing Qlib API data loading...")
        if not self._check_qlib_api_loading(calendar_dates, instruments):
            return False

        # 7. Check data integrity
        logger.info("\nStep 6: Checking data integrity...")
        self._check_data_integrity(calendar_dates, instruments)

        # Print summary
        self._print_summary()

        return len(self.errors) == 0

    def _check_directory_structure(self) -> bool:
        """Check if required directories exist."""
        required_dirs = {
            "calendars": self.qlib_dir / "calendars",
            "instruments": self.qlib_dir / "instruments",
            "features": self.qlib_dir / "features",
        }

        all_exist = True
        for name, path in required_dirs.items():
            if path.exists() and path.is_dir():
                logger.info(f"  ✓ {name}/ directory exists")
            else:
                logger.error(f"  ✗ {name}/ directory not found: {path}")
                self.errors.append(f"Missing directory: {name}/")
                all_exist = False

        return all_exist

    def _check_calendar_file(self) -> Optional[List[str]]:
        """Check calendar file format and content."""
        calendar_file = self.qlib_dir / "calendars" / "day.txt"

        if not calendar_file.exists():
            logger.error(f"  ✗ Calendar file not found: {calendar_file}")
            self.errors.append("Calendar file missing")
            return None

        if calendar_file.stat().st_size == 0:
            logger.error(f"  ✗ Calendar file is empty: {calendar_file}")
            self.errors.append("Calendar file is empty")
            return None

        try:
            with open(calendar_file, "r", encoding="utf-8") as f:
                dates = [line.strip() for line in f if line.strip()]

            if len(dates) == 0:
                logger.error("  ✗ Calendar file contains no dates")
                self.errors.append("Calendar file has no dates")
                return None

            # Validate date format (support both YYYYMMDD and YYYY-MM-DD)
            invalid_dates = []
            for date in dates:
                # Support both YYYYMMDD and YYYY-MM-DD formats
                if len(date) == 8 and date.isdigit():
                    # YYYYMMDD format (Qlib standard)
                    pass
                elif len(date) == 10 and date.count("-") == 2:
                    # YYYY-MM-DD format (also acceptable, will convert)
                    try:
                        pd.to_datetime(date, format="%Y-%m-%d")
                    except ValueError:
                        invalid_dates.append(date)
                else:
                    invalid_dates.append(date)

            if invalid_dates:
                logger.warning(f"  ⚠ Found {len(invalid_dates)} invalid date formats")
                self.warnings.append(f"Invalid date formats: {invalid_dates[:5]}")

            logger.info(f"  ✓ Calendar file contains {len(dates)} trading days")
            logger.info(f"    First date: {dates[0]}")
            logger.info(f"    Last date: {dates[-1]}")

            # Check for duplicates
            unique_dates = set(dates)
            if len(unique_dates) != len(dates):
                duplicates = len(dates) - len(unique_dates)
                logger.warning(f"  ⚠ Found {duplicates} duplicate dates")
                self.warnings.append(f"Duplicate dates in calendar: {duplicates}")

            # Check if dates are sorted
            if dates != sorted(dates):
                logger.warning("  ⚠ Calendar dates are not sorted")
                self.warnings.append("Calendar dates not sorted")

            return dates

        except Exception as e:
            logger.error(f"  ✗ Failed to read calendar file: {e}")
            self.errors.append(f"Calendar file read error: {e}")
            return None

    def _check_instruments_file(self) -> Optional[List[str]]:
        """Check instruments file format and content."""
        instruments_file = self.qlib_dir / "instruments" / "all.txt"

        if not instruments_file.exists():
            logger.error(f"  ✗ Instruments file not found: {instruments_file}")
            self.errors.append("Instruments file missing")
            return None

        if instruments_file.stat().st_size == 0:
            logger.error(f"  ✗ Instruments file is empty: {instruments_file}")
            self.errors.append("Instruments file is empty")
            return None

        try:
            with open(instruments_file, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]

            if len(lines) == 0:
                logger.error("  ✗ Instruments file contains no stock codes")
                self.errors.append("Instruments file has no stock codes")
                return None

            # Parse Qlib instruments format: <stock_code>\t<start_date>\t<end_date>
            # or simple format: <stock_code>
            instruments = []
            invalid_lines = []
            for line in lines:
                parts = line.split("\t")
                if len(parts) == 1:
                    # Simple format: just stock code
                    instruments.append(parts[0].strip())
                elif len(parts) == 3:
                    # Qlib format: stock_code\tstart_date\tend_date
                    stock_code = parts[0].strip()
                    start_date = parts[1].strip()
                    end_date = parts[2].strip()
                    instruments.append(stock_code)
                    # Validate dates if present
                    try:
                        pd.to_datetime(start_date)
                        pd.to_datetime(end_date)
                    except ValueError:
                        invalid_lines.append(line)
                else:
                    invalid_lines.append(line)

            if len(instruments) == 0:
                logger.error("  ✗ No valid stock codes found in instruments file")
                self.errors.append("No valid stock codes in instruments file")
                return None

            if invalid_lines:
                logger.warning(f"  ⚠ Found {len(invalid_lines)} invalid lines in instruments file")
                self.warnings.append(f"Invalid instrument lines: {len(invalid_lines)}")

            # Check for duplicates
            unique_instruments = set(instruments)
            if len(unique_instruments) != len(instruments):
                duplicates = len(instruments) - len(unique_instruments)
                logger.warning(f"  ⚠ Found {duplicates} duplicate stock codes")
                self.warnings.append(f"Duplicate stock codes: {duplicates}")

            logger.info(f"  ✓ Instruments file contains {len(instruments)} stocks")
            logger.info(f"    Sample: {instruments[:5]}")

            # Check for invalid stock codes (basic validation)
            invalid_codes = []
            for code in instruments:
                if len(code) < 4 or len(code) > 10:
                    invalid_codes.append(code)

            if invalid_codes:
                logger.warning(f"  ⚠ Found {len(invalid_codes)} potentially invalid stock codes")
                self.warnings.append(f"Invalid stock codes: {invalid_codes[:5]}")

            return instruments

        except Exception as e:
            logger.error(f"  ✗ Failed to read instruments file: {e}")
            self.errors.append(f"Instruments file read error: {e}")
            return None

    def _check_feature_files(self, instruments: List[str]) -> bool:
        """Check feature bin files existence and size."""
        features_dir = self.qlib_dir / "features"
        if not features_dir.exists():
            logger.error(f"  ✗ Features directory not found: {features_dir}")
            logger.info("")
            logger.info("  💡 This usually means CSV files haven't been converted to bin format yet.")
            logger.info("  💡 To fix this, run the export tool to convert CSV to bin:")
            logger.info("")
            logger.info("     python python/tools/qlib/export_qlib.py --freq day --stocks sh.000001")
            logger.info("     # or")
            logger.info("     python python/tools/qlib/export_qlib.py --freq day")
            logger.info("")
            logger.info("  ⚠ IMPORTANT: CSV files MUST include 'factor' field (7 columns:")
            logger.info("     date,open,high,low,close,volume,factor")
            logger.info("     Old CSV files without factor column will be rejected.")
            logger.info("")
            self.errors.append("Features directory missing - CSV files need to be converted to bin format")
            return False

        # Sample check (first 10 instruments)
        sample_size = min(10, len(instruments))
        sample_instruments = instruments[:sample_size]

        missing_files = []
        empty_files = []
        valid_files = []

        for instrument in sample_instruments:
            instrument_dir = features_dir / instrument
            
            if not instrument_dir.exists():
                missing_files.append(instrument)
                continue
            
            # Check for any .bin files (Qlib uses field-specific bin files like close.day.bin, open.day.bin, etc.)
            bin_files = list(instrument_dir.glob(f"*.{self.freq}.bin"))
            if not bin_files:
                missing_files.append(instrument)
            elif any(f.stat().st_size == 0 for f in bin_files):
                empty_files.append(instrument)
            else:
                valid_files.append(instrument)

        logger.info(f"  ✓ Checked {sample_size} sample instruments:")
        logger.info(f"    Valid: {len(valid_files)}")
        if missing_files:
            logger.warning(f"    Missing: {len(missing_files)} ({missing_files[:3]})")
            self.warnings.append(f"Missing feature files: {len(missing_files)}/{sample_size}")
        if empty_files:
            logger.warning(f"    Empty: {len(empty_files)} ({empty_files[:3]})")
            self.warnings.append(f"Empty feature files: {len(empty_files)}/{sample_size}")

        # Check all instruments (quick check)
        total_missing = 0
        for instrument in instruments:
            instrument_dir = features_dir / instrument
            if not instrument_dir.exists():
                total_missing += 1
                continue
            # Check for any .bin files (Qlib uses field-specific bin files)
            bin_files = list(instrument_dir.glob(f"*.{self.freq}.bin"))
            if not bin_files or any(f.stat().st_size == 0 for f in bin_files):
                total_missing += 1

        if total_missing > 0:
            coverage = (len(instruments) - total_missing) / len(instruments) * 100
            logger.info(f"  Overall coverage: {coverage:.1f}% ({len(instruments) - total_missing}/{len(instruments)})")
            if coverage < 90:
                logger.warning(f"  ⚠ Low coverage: {total_missing} instruments missing feature files")
                self.warnings.append(f"Low feature file coverage: {coverage:.1f}%")

        return len(valid_files) > 0

    def _check_qlib_api_loading(self, calendar_dates: List[str], instruments: List[str]) -> bool:
        """Test Qlib API data loading."""
        try:
            qlib.init(provider_uri=str(self.qlib_dir), region=self.region)
            logger.info("  ✓ Qlib initialized successfully")

            # Test calendar loading
            try:
                # Support both YYYYMMDD and YYYY-MM-DD formats
                first_date_str = calendar_dates[0]
                last_date_str = calendar_dates[-1]
                if len(first_date_str) == 8 and first_date_str.isdigit():
                    first_date = pd.to_datetime(first_date_str, format="%Y%m%d")
                else:
                    first_date = pd.to_datetime(first_date_str)
                if len(last_date_str) == 8 and last_date_str.isdigit():
                    last_date = pd.to_datetime(last_date_str, format="%Y%m%d")
                else:
                    last_date = pd.to_datetime(last_date_str)
                qlib_calendar = D.calendar(start_time=first_date, end_time=last_date)
                logger.info(f"  ✓ Calendar loaded via API: {len(qlib_calendar)} days")
                if len(qlib_calendar) != len(calendar_dates):
                    logger.warning(f"  ⚠ Calendar count mismatch: file={len(calendar_dates)}, API={len(qlib_calendar)}")
                    self.warnings.append("Calendar count mismatch between file and API")
            except Exception as e:
                logger.error(f"  ✗ Failed to load calendar via API: {e}")
                self.errors.append(f"Calendar API loading failed: {e}")
                return False

            # Test instruments loading
            try:
                # Use file-based reading since D.instruments() may return filters
                qlib_instruments = instruments  # Already loaded from file
                logger.info(f"  ✓ Instruments available: {len(qlib_instruments)} stocks")
            except Exception as e:
                logger.error(f"  ✗ Failed to load instruments: {e}")
                self.errors.append(f"Instruments API loading failed: {e}")
                return False

            # Test data loading for sample instruments
            sample_instruments = instruments[:min(5, len(instruments))]
            test_start = first_date.strftime("%Y-%m-%d")
            test_end = min(
                first_date + pd.Timedelta(days=10),
                last_date
            ).strftime("%Y-%m-%d")

            try:
                test_data = D.features(
                    sample_instruments,
                    ["$close", "$open", "$high", "$low", "$volume", "$factor"],
                    start_time=test_start,
                    end_time=test_end,
                )

                if test_data.empty:
                    logger.error("  ✗ Test data loading returned empty DataFrame")
                    logger.error("")
                    logger.error("  Diagnosing the issue...")
                    self._diagnose_empty_dataframe(sample_instruments, test_start, test_end, calendar_dates)
                    self.errors.append("Data loading returned empty result")
                    return False

                logger.info(f"  ✓ Test data loaded: shape={test_data.shape}")
                logger.info(f"    Instruments tested: {sample_instruments}")
                logger.info(f"    Date range: {test_start} to {test_end}")
                logger.info(f"    Columns: {test_data.columns.tolist()}")

                # Check for all NaN columns
                for col in test_data.columns:
                    null_count = test_data[col].isna().sum()
                    total_count = len(test_data)
                    null_pct = null_count / total_count * 100
                    if null_pct == 100:
                        logger.warning(f"  ⚠ Column {col} is 100% NaN")
                        self.warnings.append(f"Column {col} is all NaN")
                    elif null_pct > 50:
                        logger.warning(f"  ⚠ Column {col} has {null_pct:.1f}% NaN values")
                        self.warnings.append(f"Column {col} has high NaN rate: {null_pct:.1f}%")
                
                # Check for factor field specifically
                if "$factor" in test_data.columns:
                    factor_data = test_data["$factor"]
                    if factor_data.isna().all():
                        logger.warning("  ⚠ $factor column exists but is all NaN")
                        self.warnings.append("$factor column is all NaN")
                    elif (factor_data <= 0).any():
                        negative_count = (factor_data <= 0).sum()
                        logger.warning(f"  ⚠ $factor has {negative_count} non-positive values (should be > 0)")
                        self.warnings.append(f"$factor has {negative_count} non-positive values")
                    else:
                        logger.info(f"  ✓ $factor field is valid (range: {factor_data.min():.4f} - {factor_data.max():.4f})")
                else:
                    logger.warning("  ⚠ $factor column is missing (Qlib standard format should include factor)")
                    self.warnings.append("$factor column is missing")

            except Exception as e:
                logger.error(f"  ✗ Failed to load test data: {e}")
                self.errors.append(f"Data loading failed: {e}")
                import traceback
                traceback.print_exc()
                return False

            return True

        except Exception as e:
            logger.error(f"  ✗ Qlib initialization failed: {e}")
            self.errors.append(f"Qlib initialization failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _diagnose_empty_dataframe(
        self, 
        instruments: List[str], 
        test_start: str, 
        test_end: str,
        calendar_dates: List[str]
    ):
        """Diagnose why data loading returns empty DataFrame."""
        logger.error("")
        logger.error("  === Diagnosis Report ===")
        
        # Check 1: Verify bin files exist and are readable
        logger.error("  1. Checking bin files...")
        features_dir = self.qlib_dir / "features"
        for instrument in instruments[:3]:  # Check first 3
            instrument_dir = features_dir / instrument
            if not instrument_dir.exists():
                logger.error(f"     ✗ {instrument}: Directory does not exist: {instrument_dir}")
                continue
            
            bin_files = list(instrument_dir.glob(f"*.{self.freq}.bin"))
            if not bin_files:
                logger.error(f"     ✗ {instrument}: No bin files found in {instrument_dir}")
                logger.error(f"       Expected files like: close.{self.freq}.bin, open.{self.freq}.bin, etc.")
                continue
            
            logger.info(f"     ✓ {instrument}: Found {len(bin_files)} bin files")
            
            # Check bin file content
            for bin_file in bin_files[:2]:  # Check first 2 bin files
                try:
                    file_size = bin_file.stat().st_size
                    if file_size == 0:
                        logger.error(f"       ✗ {bin_file.name}: File is empty (0 bytes)")
                        continue
                    
                    # Read first few bytes to check format
                    with open(bin_file, "rb") as f:
                        data = np.fromfile(f, dtype=np.float32, count=10)
                    
                    if len(data) == 0:
                        logger.error(f"       ✗ {bin_file.name}: Cannot read data (file may be corrupted)")
                        continue
                    
                    # First float32 should be date_index (should be >= 0 and < len(calendar))
                    date_index = int(data[0])
                    if date_index < 0 or date_index >= len(calendar_dates):
                        logger.error(f"       ✗ {bin_file.name}: Invalid date_index={date_index}")
                        logger.error(f"         Date index should be 0-{len(calendar_dates)-1} (calendar has {len(calendar_dates)} dates)")
                        logger.error(f"         This indicates calendar and bin file are MISMATCHED!")
                        logger.error(f"         The bin file was created with a different calendar.")
                    else:
                        logger.info(f"       ✓ {bin_file.name}: date_index={date_index}, file_size={file_size} bytes")
                        # Show what date this index points to
                        if date_index < len(calendar_dates):
                            date_str = calendar_dates[date_index]
                            logger.info(f"         Points to calendar date: {date_str}")
                
                except Exception as e:
                    logger.error(f"       ✗ {bin_file.name}: Error reading file: {e}")
        
        # Check 2: Verify instruments file format
        logger.error("")
        logger.error("  2. Checking instruments file format...")
        instruments_file = self.qlib_dir / "instruments" / "all.txt"
        if instruments_file.exists():
            try:
                with open(instruments_file, "r", encoding="utf-8") as f:
                    lines = [line.strip() for line in f if line.strip()]
                
                logger.info(f"     Total lines in file: {len(lines)}")
                
                # Check first 10 lines
                sample_lines = lines[:10]
                simple_format_count = 0
                qlib_format_count = 0
                invalid_format_count = 0
                
                for i, line in enumerate(sample_lines):
                    parts = line.split("\t")
                    if len(parts) == 1:
                        simple_format_count += 1
                        if i < 3:  # Show first 3 examples
                            logger.warning(f"     ⚠ Line {i+1}: Simple format (no date range): {line}")
                            logger.warning(f"       Qlib REQUIRES format: <code>\\t<start_date>\\t<end_date>")
                            logger.warning(f"       Without date ranges, Qlib cannot filter instruments properly!")
                    elif len(parts) == 3:
                        qlib_format_count += 1
                        code, start_date, end_date = parts
                        if i < 3:  # Show first 3 examples
                            logger.info(f"     ✓ Line {i+1}: {code} ({start_date} to {end_date})")
                    else:
                        invalid_format_count += 1
                        if i < 3:  # Show first 3 examples
                            logger.error(f"     ✗ Line {i+1}: Invalid format: {line}")
                            logger.error(f"       Expected: <code>\\t<start_date>\\t<end_date> or just <code>")
                
                logger.error("")
                logger.error(f"     Format summary (first 10 lines):")
                logger.error(f"       Qlib format (with dates): {qlib_format_count}")
                logger.error(f"       Simple format (no dates): {simple_format_count}")
                logger.error(f"       Invalid format: {invalid_format_count}")
                
                if simple_format_count > 0:
                    logger.error("")
                    logger.error(f"     ✗ PROBLEM FOUND: Instruments file uses simple format (no date ranges)")
                    logger.error(f"       Qlib requires tab-separated format with date ranges:")
                    logger.error(f"       Format: <stock_code>\\t<start_date>\\t<end_date>")
                    logger.error(f"       Example: 000001.SZ\\t20000104\\t20251226")
                    logger.error(f"")
                    logger.error(f"       SOLUTION: Regenerate instruments file using dump_bin tool")
                    logger.error(f"       Or use regenerate_instruments_calendar.py with proper date ranges")
                    
            except Exception as e:
                logger.error(f"     ✗ Failed to read instruments file: {e}")
        
        # Check 3: Verify calendar file format
        logger.error("")
        logger.error("  3. Checking calendar file format...")
        calendar_file = self.qlib_dir / "calendars" / f"{self.freq}.txt"
        if calendar_file.exists():
            try:
                with open(calendar_file, "r", encoding="utf-8") as f:
                    dates = [line.strip() for line in f if line.strip()]
                
                logger.info(f"     Calendar has {len(dates)} dates")
                logger.info(f"     First date: {dates[0]}")
                logger.info(f"     Last date: {dates[-1]}")
                
                # Check date format
                first_date = dates[0]
                if len(first_date) == 8 and first_date.isdigit():
                    logger.info(f"     Date format: YYYYMMDD (Qlib standard)")
                elif len(first_date) == 10 and first_date.count("-") == 2:
                    logger.warning(f"     ⚠ Date format: YYYY-MM-DD (should be YYYYMMDD for Qlib)")
                    logger.warning(f"       This may cause data loading issues!")
                else:
                    logger.error(f"     ✗ Invalid date format: {first_date}")
            except Exception as e:
                logger.error(f"     ✗ Failed to read calendar file: {e}")
        
        # Check 4: Test with Qlib's internal methods
        logger.error("")
        logger.error("  4. Testing Qlib internal data access...")
        try:
            from qlib.data import D
            # Try to get raw data
            test_instrument = instruments[0]
            logger.info(f"     Testing instrument: {test_instrument}")
            
            # Try to get calendar
            qlib_cal = D.calendar()
            logger.info(f"     Qlib calendar length: {len(qlib_cal)}")
            
            # Try to get instruments
            # Qlib's D.instruments() may return a filter or list
            # IMPORTANT: D.instruments() without arguments may return a filter object, not the actual list
            # We need to read the file directly or use D.instruments() with proper parameters
            qlib_inst = D.instruments()
            qlib_inst_list = []
            
            # Check what D.instruments() actually returns
            logger.info(f"     D.instruments() type: {type(qlib_inst)}")
            
            if hasattr(qlib_inst, "__iter__"):
                try:
                    qlib_inst_list = list(qlib_inst)
                except Exception as e:
                    logger.warning(f"     ⚠ Failed to convert D.instruments() to list: {e}")
            
            # If we got filter objects like 'market' or 'filter_pipe', Qlib is not reading the file correctly
            if qlib_inst_list:
                # Check if these are actual stock codes or filter objects
                is_filter_objects = any(inst in ['market', 'filter_pipe', 'all', 'cn'] for inst in qlib_inst_list)
                
                if is_filter_objects:
                    logger.error(f"     ✗ D.instruments() returned filter objects, not stock codes!")
                    logger.error(f"       Returned: {qlib_inst_list}")
                    logger.error(f"       This means Qlib cannot read instruments/all.txt file!")
                    logger.error(f"")
                    logger.error(f"       ROOT CAUSE: Instruments file format is incorrect or missing date ranges")
                    logger.error(f"")
                    logger.error(f"       Qlib requires format: <code>\\t<start_date>\\t<end_date>")
                    logger.error(f"       Current file may have: <code> (simple format without dates)")
                    logger.error(f"")
                    logger.error(f"       SOLUTION:")
                    logger.error(f"       1. Check instruments file format (see diagnosis section 2 above)")
                    logger.error(f"       2. Regenerate instruments file with date ranges:")
                    logger.error(f"          python python/tools/qlib/regenerate_instruments_calendar.py")
                    logger.error(f"       3. Or re-export data using export_qlib.py (will generate correct format)")
                else:
                    logger.info(f"     Qlib instruments count: {len(qlib_inst_list)}")
                    logger.info(f"     Sample Qlib instruments: {qlib_inst_list[:5]}")
                    
                    if test_instrument not in qlib_inst_list:
                        logger.error(f"     ✗ {test_instrument} not found in Qlib instruments!")
                        logger.error(f"       This indicates instruments file format issue")
                        logger.error(f"")
                        logger.error(f"       DIAGNOSIS:")
                        logger.error(f"       - File has {len(instruments)} instruments")
                        logger.error(f"       - Qlib only sees {len(qlib_inst_list)} instruments")
                        logger.error(f"       - Qlib instruments: {qlib_inst_list}")
                        logger.error(f"")
                        logger.error(f"       This usually means:")
                        logger.error(f"       1. Instruments file format is incorrect (should be tab-separated: code\\tstart\\tend)")
                        logger.error(f"       2. Qlib is filtering instruments based on date ranges")
                        logger.error(f"       3. Instruments file was regenerated incorrectly (missing date ranges)")
            else:
                logger.warning(f"     ⚠ Qlib instruments is not iterable or empty: {type(qlib_inst)}")
                logger.warning(f"       This may indicate Qlib cannot read instruments file properly")
                logger.warning(f"       Try reading file directly (see diagnosis section 2)")
            
        except Exception as e:
            logger.error(f"     ✗ Failed to test Qlib internal access: {e}")
            import traceback
            traceback.print_exc()
        
        logger.error("")
        logger.error("  === End Diagnosis ===")
        logger.error("")
        logger.error("  Possible causes:")
        logger.error("    1. Calendar and bin files were created with different date ranges")
        logger.error("       → Solution: Re-export all data using incremental update mode")
        logger.error("    2. Bin files have invalid date_index values")
        logger.error("       → Solution: Check if bin files were created with correct calendar")
        logger.error("    3. Instruments file format is incorrect")
        logger.error("       → Solution: Regenerate instruments file using regenerate_instruments_calendar.py")
        logger.error("    4. Calendar file format is incorrect (should be YYYYMMDD)")
        logger.error("       → Solution: Fix calendar file format")

    def _check_data_integrity(self, calendar_dates: List[str], instruments: List[str]) -> bool:
        """Check data integrity (date alignment, missing values, etc.)."""
        try:
            # Sample a few instruments for detailed check
            sample_instruments = instruments[:min(3, len(instruments))]
            # Support both YYYYMMDD and YYYY-MM-DD formats
            first_date_str = calendar_dates[0]
            last_date_str = calendar_dates[-1]
            if len(first_date_str) == 8 and first_date_str.isdigit():
                first_date = pd.to_datetime(first_date_str, format="%Y%m%d")
            else:
                first_date = pd.to_datetime(first_date_str)
            if len(last_date_str) == 8 and last_date_str.isdigit():
                last_date = pd.to_datetime(last_date_str, format="%Y%m%d")
            else:
                last_date = pd.to_datetime(last_date_str)

            logger.info(f"  Checking data integrity for {len(sample_instruments)} sample instruments...")

            for instrument in sample_instruments:
                try:
                    data = D.features(
                        [instrument],
                        ["$close", "$open", "$high", "$low", "$volume", "$factor"],
                        start_time=first_date.strftime("%Y-%m-%d"),
                        end_time=last_date.strftime("%Y-%m-%d"),
                    )

                    if data.empty:
                        logger.warning(f"    ⚠ {instrument}: No data available")
                        self.warnings.append(f"{instrument}: Empty data")
                        continue

                    # Check date alignment
                    data_dates = set(data.index.get_level_values("datetime").strftime("%Y%m%d"))
                    calendar_set = set(calendar_dates)
                    missing_dates = calendar_set - data_dates
                    extra_dates = data_dates - calendar_set

                    if missing_dates:
                        missing_pct = len(missing_dates) / len(calendar_set) * 100
                        logger.info(f"    {instrument}: Missing {len(missing_dates)} dates ({missing_pct:.1f}%)")
                        if missing_pct > 20:
                            self.warnings.append(f"{instrument}: High missing date rate: {missing_pct:.1f}%")

                    if extra_dates:
                        logger.warning(f"    ⚠ {instrument}: {len(extra_dates)} dates not in calendar")
                        self.warnings.append(f"{instrument}: Dates not in calendar")

                    # Check for negative prices
                    price_cols = ["$close", "$open", "$high", "$low"]
                    for col in price_cols:
                        if col in data.columns:
                            negative_count = (data[col] < 0).sum()
                            if negative_count > 0:
                                logger.warning(f"    ⚠ {instrument}: {negative_count} negative values in {col}")
                                self.warnings.append(f"{instrument}: Negative values in {col}")

                    # Check for zero volume
                    if "$volume" in data.columns:
                        zero_volume = (data["$volume"] == 0).sum()
                        zero_pct = zero_volume / len(data) * 100
                        if zero_pct > 50:
                            logger.warning(f"    ⚠ {instrument}: {zero_pct:.1f}% zero volume")
                            self.warnings.append(f"{instrument}: High zero volume rate: {zero_pct:.1f}%")

                except Exception as e:
                    logger.warning(f"    ⚠ {instrument}: Integrity check failed: {e}")
                    self.warnings.append(f"{instrument}: Integrity check error: {e}")

            return True

        except Exception as e:
            logger.error(f"  ✗ Data integrity check failed: {e}")
            self.errors.append(f"Data integrity check failed: {e}")
            return False

    def _check_data_coverage(
        self,
        calendar_dates: List[str],
        instruments: List[str],
        detailed: bool = False,
    ) -> None:
        """
        Check Qlib data coverage.
        
        Args:
            calendar_dates: List of calendar dates.
            instruments: List of instrument codes.
            detailed: Whether to show detailed information for each instrument.
        """
        # 1. Calendar coverage
        logger.info("1. Calendar Coverage")
        logger.info(f"  Total trading days: {len(calendar_dates)}")
        if calendar_dates:
            logger.info(f"  Earliest date: {calendar_dates[0]}")
            logger.info(f"  Latest date: {calendar_dates[-1]}")
            
            # Calculate date range in years
            try:
                first_date = pd.to_datetime(calendar_dates[0], format="%Y%m%d" if len(calendar_dates[0]) == 8 else None)
                last_date = pd.to_datetime(calendar_dates[-1], format="%Y%m%d" if len(calendar_dates[-1]) == 8 else None)
                years = (last_date - first_date).days / 365.25
                logger.info(f"  Date range: {years:.2f} years")
            except:
                pass
        
        # 2. Instruments coverage
        logger.info("")
        logger.info("2. Instruments Coverage")
        logger.info(f"  Total instruments: {len(instruments)}")
        
        # Get instruments with date ranges
        instruments_file = self.qlib_dir / "instruments" / "all.txt"
        instruments_with_dates = 0
        instruments_without_dates = 0
        start_dates = []
        end_dates = []
        instruments_date_ranges = {}
        
        if instruments_file.exists():
            try:
                with open(instruments_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        
                        parts = line.split("\t")
                        if len(parts) >= 3:
                            code = parts[0]
                            start_date = parts[1]
                            end_date = parts[2]
                            instruments_date_ranges[code] = (start_date, end_date)
                            instruments_with_dates += 1
                            
                            try:
                                if len(start_date) == 8 and start_date.isdigit():
                                    start_dates.append(pd.to_datetime(start_date, format="%Y%m%d"))
                                else:
                                    start_dates.append(pd.to_datetime(start_date))
                                
                                if len(end_date) == 8 and end_date.isdigit():
                                    end_dates.append(pd.to_datetime(end_date, format="%Y%m%d"))
                                else:
                                    end_dates.append(pd.to_datetime(end_date))
                            except:
                                pass
                        elif len(parts) >= 1:
                            instruments_without_dates += 1
            except Exception as e:
                logger.debug(f"Failed to read instruments date ranges: {e}")
        
        logger.info(f"  Instruments with date ranges: {instruments_with_dates}")
        logger.info(f"  Instruments without date ranges: {instruments_without_dates}")
        
        if start_dates and end_dates:
            earliest_start = min(start_dates)
            latest_start = max(start_dates)
            earliest_end = min(end_dates)
            latest_end = max(end_dates)
            
            logger.info("")
            logger.info("  Date Range Statistics:")
            logger.info(f"    Earliest start date: {earliest_start.strftime('%Y-%m-%d')}")
            logger.info(f"    Latest start date: {latest_start.strftime('%Y-%m-%d')}")
            logger.info(f"    Earliest end date: {earliest_end.strftime('%Y-%m-%d')}")
            logger.info(f"    Latest end date: {latest_end.strftime('%Y-%m-%d')}")
        
        # 3. Features coverage
        logger.info("")
        logger.info("3. Features Coverage")
        features_dir = self.qlib_dir / "features"
        
        if not features_dir.exists():
            logger.error(f"  Features directory not found: {features_dir}")
            return
        
        # Count instruments with bin files
        instrument_dirs = [d for d in features_dir.iterdir() if d.is_dir()]
        logger.info(f"  Instruments with feature directories: {len(instrument_dirs)}")
        
        # Create a mapping from lowercase to actual case for case-insensitive matching
        instrument_dir_map = {d.name.lower(): d.name for d in instrument_dirs}
        
        # Check bin files for each instrument
        instruments_with_bin_files = 0
        instruments_without_bin_files = []
        date_ranges_from_bin = {}
        
        required_fields = ["open", "close", "high", "low", "volume"]
        field_coverage = defaultdict(int)
        
        # Check instruments from file
        for stock_code in instruments:
            # Try to find matching directory (case-insensitive)
            stock_code_lower = stock_code.lower()
            if stock_code_lower in instrument_dir_map:
                actual_dir_name = instrument_dir_map[stock_code_lower]
                instrument_dir = features_dir / actual_dir_name
            else:
                instrument_dir = features_dir / stock_code
            
            bin_files = list(instrument_dir.glob(f"*.{self.freq}.bin"))
            
            if bin_files:
                instruments_with_bin_files += 1
                
                # Check which fields are present
                for field in required_fields:
                    field_bin = instrument_dir / f"{field}.{self.freq}.bin"
                    if field_bin.exists():
                        field_coverage[field] += 1
                
                # Get date range from bin file
                if calendar_dates:
                    actual_stock_code = instrument_dir_map.get(stock_code_lower, stock_code)
                    start_date, end_date = self._get_stock_date_range_from_bin(
                        features_dir, actual_stock_code, calendar_dates
                    )
                    if start_date and end_date:
                        date_ranges_from_bin[stock_code] = (start_date, end_date)
            else:
                instruments_without_bin_files.append(stock_code)
        
        logger.info(f"  Instruments with bin files: {instruments_with_bin_files}")
        logger.info(f"  Instruments without bin files: {len(instruments_without_bin_files)}")
        
        if instruments_without_bin_files and len(instruments_without_bin_files) <= 10:
            logger.warning(f"  Missing bin files for: {', '.join(instruments_without_bin_files)}")
        elif instruments_without_bin_files:
            logger.warning(f"  Missing bin files for {len(instruments_without_bin_files)} instruments (showing first 10):")
            logger.warning(f"    {', '.join(instruments_without_bin_files[:10])}...")
        
        logger.info("")
        logger.info("  Field Coverage:")
        for field in required_fields:
            coverage_pct = (field_coverage[field] / len(instruments)) * 100 if instruments else 0
            logger.info(f"    {field}: {field_coverage[field]}/{len(instruments)} ({coverage_pct:.1f}%)")
        
        # 4. Data consistency
        logger.info("")
        logger.info("4. Data Consistency")
        
        # Case-insensitive comparison
        instruments_in_file_lower = {code.lower() for code in instruments}
        instruments_in_features_lower = {d.name.lower() for d in instrument_dirs}
        
        only_in_file = {code for code in instruments if code.lower() not in instruments_in_features_lower}
        only_in_features = {d.name for d in instrument_dirs if d.name.lower() not in instruments_in_file_lower}
        
        if only_in_file:
            logger.warning(f"  Instruments in file but not in features: {len(only_in_file)}")
            if len(only_in_file) <= 10:
                logger.warning(f"    {', '.join(sorted(only_in_file))}")
        
        if only_in_features:
            logger.warning(f"  Instruments in features but not in file: {len(only_in_features)}")
            if len(only_in_features) <= 10:
                logger.warning(f"    {', '.join(sorted(only_in_features))}")
        
        if not only_in_file and not only_in_features:
            logger.info("  ✓ Instruments file and features directory are consistent")
        
        # 5. Detailed information (if requested)
        if detailed:
            logger.info("")
            logger.info("5. Detailed Instrument Information")
            
            sorted_codes = sorted(instruments)
            for i, code in enumerate(sorted_codes[:50], 1):  # Show first 50
                start_date, end_date = instruments_date_ranges.get(code, (None, None))
                bin_start, bin_end = date_ranges_from_bin.get(code, (None, None))
                
                logger.info(f"  {i}. {code}")
                if start_date and end_date:
                    logger.info(f"     Instruments file: {start_date} to {end_date}")
                else:
                    logger.info(f"     Instruments file: No date range")
                
                if bin_start and bin_end:
                    logger.info(f"     Bin file: {bin_start} to {bin_end}")
                else:
                    logger.info(f"     Bin file: No date range or cannot read")
            
            if len(sorted_codes) > 50:
                logger.info(f"  ... and {len(sorted_codes) - 50} more instruments")
        
        # Summary
        logger.info("")
        logger.info("Coverage Summary")
        logger.info(f"  Calendar: {len(calendar_dates)} trading days")
        logger.info(f"  Instruments: {len(instruments)} total")
        logger.info(f"  Features: {instruments_with_bin_files} with bin files")
        if instruments:
            coverage_pct = (instruments_with_bin_files / len(instruments)) * 100
            logger.info(f"  Coverage: {coverage_pct:.1f}%")

    def _get_stock_date_range_from_bin(
        self,
        features_dir: Path,
        stock_code: str,
        calendar_dates: List[str],
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Get start and end date for a stock from its bin file.
        
        Returns:
            Tuple of (start_date, end_date) in YYYYMMDD format, or (None, None) if cannot determine.
        """
        stock_dir = features_dir / stock_code
        if not stock_dir.exists():
            return None, None
        
        bin_files = list(stock_dir.glob(f"*.{self.freq}.bin"))
        if not bin_files:
            return None, None
        
        bin_file = bin_files[0]
        
        try:
            data = np.fromfile(str(bin_file), dtype=np.float32)
            if len(data) < 2:
                return None, None
            
            start_date_index = int(data[0])
            
            if calendar_dates and 0 <= start_date_index < len(calendar_dates):
                start_date_str = calendar_dates[start_date_index]
                
                num_data_points = len(data) - 1
                end_date_index = start_date_index + num_data_points - 1
                
                if end_date_index >= len(calendar_dates):
                    end_date_index = len(calendar_dates) - 1
                
                if end_date_index >= start_date_index:
                    end_date_str = calendar_dates[end_date_index]
                    return start_date_str, end_date_str
            
            return None, None
        except Exception as e:
            logger.debug(f"Failed to read date range from {stock_code}: {e}")
            return None, None

    def _print_summary(self):
        """Print verification summary."""
        logger.info("")
        logger.info("=" * 80)
        logger.info("Verification Summary")
        logger.info("=" * 80)

        if len(self.errors) == 0:
            logger.info("✓ All critical checks passed!")
        else:
            logger.error(f"✗ Found {len(self.errors)} critical error(s):")
            for error in self.errors:
                logger.error(f"  - {error}")

        if len(self.warnings) > 0:
            logger.warning(f"⚠ Found {len(self.warnings)} warning(s):")
            for warning in self.warnings[:10]:  # Show first 10 warnings
                logger.warning(f"  - {warning}")
            if len(self.warnings) > 10:
                logger.warning(f"  ... and {len(self.warnings) - 10} more warnings")

        logger.info("=" * 80)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Verify exported Qlib data correctness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--qlib-dir",
        type=str,
        default="~/.qlib/qlib_data/cn_data",
        help="Path to Qlib data directory (default: ~/.qlib/qlib_data/cn_data)",
    )
    parser.add_argument(
        "--region",
        type=str,
        default="cn",
        help="Qlib region (default: cn)",
    )
    parser.add_argument(
        "--freq",
        type=str,
        default="day",
        help="Data frequency (default: day)",
    )
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="Include data coverage analysis",
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Show detailed instrument information in coverage report",
    )

    args = parser.parse_args()

    verifier = QlibDataVerifier(qlib_dir=args.qlib_dir, region=args.region, freq=args.freq)
    success = verifier.verify_all(check_coverage=args.coverage, detailed_coverage=args.detailed)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()


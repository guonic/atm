#!/usr/bin/env python3
"""
Comprehensive tool to verify exported Qlib data correctness.

This tool checks:
1. File structure (calendars, instruments, features directories)
2. Calendar file format and content
3. Instruments file format and content
4. Feature bin files existence and size
5. Data loading via Qlib API
6. Data integrity (missing values, date alignment, etc.)
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional

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

    def verify_all(self) -> bool:
        """Run all verification checks."""
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

        # 5. Initialize Qlib and check API loading
        logger.info("\nStep 5: Testing Qlib API data loading...")
        if not self._check_qlib_api_loading(calendar_dates, instruments):
            return False

        # 6. Check data integrity
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

    args = parser.parse_args()

    verifier = QlibDataVerifier(qlib_dir=args.qlib_dir, region=args.region, freq=args.freq)
    success = verifier.verify_all()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()


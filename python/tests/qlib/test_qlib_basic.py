#!/usr/bin/env python3
"""
Basic Qlib test workflow.

This script demonstrates basic Qlib operations including:
1. Initializing Qlib with exported data
2. Loading stock data
3. Extracting features
4. Basic data analysis
5. Simple strategy example

Usage:
    python python/tests/qlib/test_qlib_basic.py [--qlib-dir QLIB_DIR] [--region REGION]
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import qlib
from qlib.data import D

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def test_qlib_init(qlib_dir: str, region: str = "cn") -> bool:
    """
    Test Qlib initialization.

    Args:
        qlib_dir: Path to Qlib data directory.
        region: Qlib region (default: "cn").

    Returns:
        True if initialization successful, False otherwise.
    """
    logger.info("=" * 80)
    logger.info("Test 1: Qlib Initialization")
    logger.info("=" * 80)

    try:
        # Check if data directory exists
        qlib_path = Path(qlib_dir)
        if not qlib_path.exists():
            logger.error(f"Qlib data directory not found: {qlib_dir}")
            return False

        # Check for required subdirectories
        required_dirs = ["calendars", "instruments", "features"]
        missing_dirs = [d for d in required_dirs if not (qlib_path / d).exists()]
        if missing_dirs:
            logger.error(f"Missing required directories: {missing_dirs}")
            logger.error(f"Please run data export script to generate Qlib data")
            return False

        # Initialize Qlib
        qlib.init(provider_uri=qlib_dir, region=region)
        logger.info(f"✓ Qlib initialized successfully")
        logger.info(f"  Data directory: {qlib_dir}")
        logger.info(f"  Region: {region}")
        return True

    except Exception as e:
        logger.error(f"✗ Qlib initialization failed: {e}", exc_info=True)
        return False


def test_calendar_loading() -> bool:
    """
    Test calendar loading.

    Returns:
        True if calendar loaded successfully, False otherwise.
    """
    logger.info("")
    logger.info("=" * 80)
    logger.info("Test 2: Calendar Loading")
    logger.info("=" * 80)

    try:
        # Get calendar
        calendar = D.calendar()
        logger.info(f"✓ Calendar loaded successfully")
        logger.info(f"  Total trading days: {len(calendar)}")
        logger.info(f"  First date: {calendar[0]}")
        logger.info(f"  Last date: {calendar[-1]}")

        # Get calendar for specific date range
        start_date = calendar[0]
        end_date = calendar[min(10, len(calendar) - 1)]
        calendar_range = D.calendar(start_time=start_date, end_time=end_date)
        logger.info(f"  Calendar range ({start_date} to {end_date}): {len(calendar_range)} days")

        return True

    except Exception as e:
        logger.error(f"✗ Calendar loading failed: {e}", exc_info=True)
        return False


def test_instruments_loading() -> bool:
    """
    Test instruments loading.

    Returns:
        True if instruments loaded successfully, False otherwise.
    """
    logger.info("")
    logger.info("=" * 80)
    logger.info("Test 3: Instruments Loading")
    logger.info("=" * 80)

    try:
        # Get all instruments
        instruments = D.instruments()
        # Convert to list if it's not already a list
        if not isinstance(instruments, list):
            instruments = list(instruments)
        
        logger.info(f"✓ Instruments loaded successfully")
        logger.info(f"  Total instruments: {len(instruments)}")
        logger.info(f"  Sample instruments: {instruments[:5]}")

        return True

    except Exception as e:
        logger.error(f"✗ Instruments loading failed: {e}", exc_info=True)
        return False


def test_data_loading(instruments: Optional[List[str]] = None, sample_size: int = 5) -> bool:
    """
    Test data loading for sample instruments.

    Args:
        instruments: List of instrument codes. If None, uses sample from D.instruments().
        sample_size: Number of instruments to test.

    Returns:
        True if data loaded successfully, False otherwise.
    """
    logger.info("")
    logger.info("=" * 80)
    logger.info("Test 4: Data Loading")
    logger.info("=" * 80)

    try:
        # Get sample instruments
        if instruments is None:
            all_instruments = D.instruments()
            # Convert to list if it's not already a list
            if not isinstance(all_instruments, list):
                all_instruments = list(all_instruments)
            instruments = all_instruments[:sample_size]

        logger.info(f"Testing data loading for {len(instruments)} instruments: {instruments}")

        # Get calendar for date range
        calendar = D.calendar()
        start_date = calendar[0]
        end_date = calendar[min(30, len(calendar) - 1)]

        # Load basic features
        fields = ["$close", "$open", "$high", "$low", "$volume", "$factor"]
        data = D.features(
            instruments,
            fields,
            start_time=start_date,
            end_time=end_date,
        )

        logger.info(f"✓ Data loaded successfully")
        logger.info(f"  Data shape: {data.shape}")
        logger.info(f"  Date range: {start_date} to {end_date}")
        logger.info(f"  Columns: {data.columns.tolist()}")

        # Check for missing values
        missing_pct = data.isna().sum() / len(data) * 100
        logger.info(f"  Missing values percentage:")
        for col in data.columns:
            logger.info(f"    {col}: {missing_pct[col]:.2f}%")

        # Display sample data
        logger.info(f"  Sample data (first 5 rows):")
        logger.info(f"\n{data.head()}")

        return True

    except Exception as e:
        logger.error(f"✗ Data loading failed: {e}", exc_info=True)
        return False


def test_feature_extraction(instruments: Optional[List[str]] = None) -> bool:
    """
    Test feature extraction with technical indicators.

    Args:
        instruments: List of instrument codes. If None, uses sample from D.instruments().

    Returns:
        True if feature extraction successful, False otherwise.
    """
    logger.info("")
    logger.info("=" * 80)
    logger.info("Test 5: Feature Extraction")
    logger.info("=" * 80)

    try:
        # Get sample instruments
        if instruments is None:
            all_instruments = D.instruments()
            # Convert to list if it's not already a list
            if not isinstance(all_instruments, list):
                all_instruments = list(all_instruments)
            instruments = all_instruments[:3]

        logger.info(f"Testing feature extraction for {len(instruments)} instruments: {instruments}")

        # Get calendar for date range
        calendar = D.calendar()
        start_date = calendar[0]
        end_date = calendar[min(60, len(calendar) - 1)]

        # Define features (using Qlib expression syntax)
        features = [
            "$close",  # Close price
            "$open",   # Open price
            "$volume", # Volume
            "Mean($close, 5)",  # 5-day moving average
            "Mean($close, 20)", # 20-day moving average
            "Std($close, 5)",   # 5-day standard deviation
            "Ref($close, -1)",  # Previous day close
            "($close - Ref($close, -1)) / Ref($close, -1)",  # Daily return
        ]

        data = D.features(
            instruments,
            features,
            start_time=start_date,
            end_time=end_date,
        )

        logger.info(f"✓ Feature extraction successful")
        logger.info(f"  Data shape: {data.shape}")
        logger.info(f"  Features: {features}")

        # Display sample data
        logger.info(f"  Sample data (first 5 rows):")
        logger.info(f"\n{data.head()}")

        # Calculate basic statistics
        logger.info(f"  Feature statistics:")
        logger.info(f"\n{data.describe()}")

        return True

    except Exception as e:
        logger.error(f"✗ Feature extraction failed: {e}", exc_info=True)
        return False


def test_simple_strategy(instruments: Optional[List[str]] = None) -> bool:
    """
    Test a simple moving average crossover strategy.

    Args:
        instruments: List of instrument codes. If None, uses sample from D.instruments().

    Returns:
        True if strategy test successful, False otherwise.
    """
    logger.info("")
    logger.info("=" * 80)
    logger.info("Test 6: Simple Strategy (Moving Average Crossover)")
    logger.info("=" * 80)

    try:
        # Get sample instruments
        if instruments is None:
            all_instruments = D.instruments()
            # Convert to list if it's not already a list
            if not isinstance(all_instruments, list):
                all_instruments = list(all_instruments)
            instruments = all_instruments[:3]

        logger.info(f"Testing strategy for {len(instruments)} instruments: {instruments}")

        # Get calendar for date range
        calendar = D.calendar()
        start_date = calendar[0]
        end_date = calendar[min(60, len(calendar) - 1)]

        # Calculate moving averages
        ma_short = "Mean($close, 5)"
        ma_long = "Mean($close, 20)"

        # Load data
        data = D.features(
            instruments,
            ["$close", ma_short, ma_long],
            start_time=start_date,
            end_time=end_date,
        )

        logger.info(f"✓ Strategy data loaded")
        logger.info(f"  Data shape: {data.shape}")

        # Generate signals
        # Buy signal: short MA crosses above long MA
        # Sell signal: short MA crosses below long MA
        # Qlib returns DataFrame with MultiIndex (instrument, datetime)
        signals_dict = {}
        
        for instrument in instruments:
            try:
                # Qlib data has MultiIndex, need to use xs to select by instrument
                if isinstance(data.index, pd.MultiIndex):
                    instrument_data = data.xs(instrument, level=0)
                else:
                    # If not MultiIndex, try direct access
                    instrument_data = data.loc[instrument]
                
                if len(instrument_data) == 0:
                    logger.warning(f"No data for {instrument}")
                    continue

                # Calculate signal
                signal = pd.Series(0, index=instrument_data.index)
                ma_short_col = f"{ma_short}"
                ma_long_col = f"{ma_long}"

                if ma_short_col in instrument_data.columns and ma_long_col in instrument_data.columns:
                    # Buy when short MA crosses above long MA
                    signal[instrument_data[ma_short_col] > instrument_data[ma_long_col]] = 1
                    # Sell when short MA crosses below long MA
                    signal[instrument_data[ma_short_col] < instrument_data[ma_long_col]] = -1

                signals_dict[instrument] = signal
                
            except KeyError as e:
                logger.warning(f"Could not access data for {instrument}: {e}")
                continue
        
        if not signals_dict:
            logger.warning("No signals generated for any instrument")
            return False

        logger.info(f"✓ Signals generated")
        logger.info(f"  Signal statistics:")
        for instrument, signal in signals_dict.items():
            buy_signals = (signal == 1).sum()
            sell_signals = (signal == -1).sum()
            logger.info(f"    {instrument}: Buy={buy_signals}, Sell={sell_signals}")

        return True

    except Exception as e:
        logger.error(f"✗ Strategy test failed: {e}", exc_info=True)
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Basic Qlib test workflow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test with default Qlib data directory
  python python/tests/qlib/test_qlib_basic.py

  # Test with custom Qlib data directory
  python python/tests/qlib/test_qlib_basic.py --qlib-dir ~/.qlib/qlib_data/cn_data

  # Test with specific instruments
  python python/tests/qlib/test_qlib_basic.py --instruments 000001.SZ 000002.SZ
        """,
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
        "--instruments",
        type=str,
        nargs="+",
        help="List of instrument codes to test (e.g., 000001.SZ 000002.SZ). If not specified, uses sample instruments.",
    )

    parser.add_argument(
        "--skip-strategy",
        action="store_true",
        help="Skip strategy test",
    )

    args = parser.parse_args()

    # Expand user path
    qlib_dir = str(Path(args.qlib_dir).expanduser())

    # Run tests
    results = []

    # Test 1: Initialize Qlib
    results.append(("Qlib Initialization", test_qlib_init(qlib_dir, args.region)))
    if not results[-1][1]:
        logger.error("Qlib initialization failed. Cannot continue tests.")
        return 1

    # Test 2: Load calendar
    results.append(("Calendar Loading", test_calendar_loading()))

    # Test 3: Load instruments
    results.append(("Instruments Loading", test_instruments_loading()))

    # Test 4: Load data
    results.append(("Data Loading", test_data_loading(args.instruments)))

    # Test 5: Extract features
    results.append(("Feature Extraction", test_feature_extraction(args.instruments)))

    # Test 6: Simple strategy
    if not args.skip_strategy:
        results.append(("Simple Strategy", test_simple_strategy(args.instruments)))

    # Print summary
    logger.info("")
    logger.info("=" * 80)
    logger.info("Test Summary")
    logger.info("=" * 80)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        logger.info(f"  {test_name}: {status}")

    logger.info("")
    logger.info(f"Total: {passed}/{total} tests passed")

    if passed == total:
        logger.info("=" * 80)
        logger.info("✓ All tests passed!")
        logger.info("=" * 80)
        return 0
    else:
        logger.error("=" * 80)
        logger.error(f"✗ {total - passed} test(s) failed")
        logger.error("=" * 80)
        return 1


if __name__ == "__main__":
    sys.exit(main())


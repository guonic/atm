#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Complete example of correlation algorithm optimization framework.

This script demonstrates the full workflow:
1. Generate return matrix from StructureExpert rankings
2. Optimize correlation algorithms with different parameters
3. Generate sensitivity analysis reports

Usage:
    python python/examples/test_correlation_optimization.py \
        --start-date 2024-01-01 \
        --end-date 2024-12-31 \
        --model-path models/structure_expert_directional.pth \
        --top-k 20
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import qlib

from qlib.data import D

from nq.analysis.correlation import (
    CorrelationOptimizer,
    ReturnMatrixGenerator,
    SensitivityAnalyzer,
)
from nq.config import load_config
from nq.trading.strategies.buy_models.structure_expert import StructureExpertBuyModel
from nq.trading.utils.feature_loader import get_qlib_data_range
from tools.qlib.train.structure_expert import GraphDataBuilder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_daily_ranks(
        model: StructureExpertBuyModel,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        instruments: List[str],
) -> Dict[pd.Timestamp, pd.DataFrame]:
    """
    Load daily StructureExpert rankings for date range.

    Args:
        model: StructureExpertBuyModel instance.
        start_date: Start date.
        end_date: End date.
        instruments: List of stock symbols.

    Returns:
        Dictionary mapping dates to ranking DataFrames.
    """
    # Check available data range first
    full_calendar = D.calendar()
    if len(full_calendar) == 0:
        logger.error("No trading days found in Qlib data. Please check data export.")
        return {}

    available_start = pd.Timestamp(full_calendar[0])
    available_end = pd.Timestamp(full_calendar[-1])
    logger.info(f"Available Qlib data range: {available_start.date()} to {available_end.date()}")

    # Adjust date range to match available data
    if start_date < available_start:
        logger.warning(
            f"Start date {start_date.date()} is before available data, adjusting to {available_start.date()}")
        start_date = available_start
    if end_date > available_end:
        logger.warning(f"End date {end_date.date()} is after available data, adjusting to {available_end.date()}")
        end_date = available_end

    # Get trading calendar
    calendar = D.calendar(start_time=start_date.strftime("%Y-%m-%d"), end_time=end_date.strftime("%Y-%m-%d"))

    if len(calendar) == 0:
        logger.error(f"No trading days found in date range {start_date.date()} to {end_date.date()}")
        logger.error(f"Available data range: {available_start.date()} to {available_end.date()}")
        return {}

    logger.info(f"Found {len(calendar)} trading days in date range")

    if len(instruments) < 2:
        logger.warning(
            f"Only {len(instruments)} instruments provided. StructureExpert requires at least 2 stocks for ranking.")
        logger.warning(
            "Consider providing more instruments or using a different date range with more available stocks.")

    daily_ranks = {}
    failed_count = 0

    for date_str in calendar:
        date = pd.Timestamp(date_str)

        try:
            # Load market data for this date
            market_data = D.features(
                instruments=instruments,
                fields=["$close", "$open", "$high", "$low", "$volume"],
                start_time=date_str,
                end_time=date_str,
            )

            if market_data.empty:
                logger.debug(f"No market data for {date.strftime('%Y-%m-%d')}")
                failed_count += 1
                continue

            # Generate rankings
            ranks_df = model.generate_ranks(date=date, market_data=market_data)

            if not ranks_df.empty:
                daily_ranks[date] = ranks_df
                logger.debug(f"Generated rankings for {date.strftime('%Y-%m-%d')}: {len(ranks_df)} stocks")
            else:
                logger.debug(f"Empty rankings for {date.strftime('%Y-%m-%d')}")
                failed_count += 1

        except Exception as e:
            logger.warning(f"Failed to generate rankings for {date}: {e}")
            failed_count += 1
            continue

    logger.info(f"Loaded daily ranks for {len(daily_ranks)} dates (failed: {failed_count})")

    if len(daily_ranks) == 0:
        logger.error("No daily ranks generated. Possible reasons:")
        logger.error("1. Date range has no trading days in Qlib data")
        logger.error("2. Insufficient instruments (need at least 2 stocks)")
        logger.error("3. Market data loading failed")
        logger.error("4. Model failed to generate rankings")

    return daily_ranks


def load_returns_data(
        instruments: List[str],
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        lookback_days: int = 120,
) -> pd.DataFrame:
    """
    Load returns data for correlation calculation.

    Args:
        instruments: List of stock symbols.
        start_date: Start date for testing.
        end_date: End date for testing.
        lookback_days: Number of days to look back before start_date for historical data
                      (default: 120 days to ensure enough data for correlation calculation).

    Returns:
        DataFrame with returns (columns: symbols, index: dates).
    """
    from qlib.data import D
    from nq.trading.utils.data_normalizer import normalize_qlib_features_result

    # Calculate actual start date with lookback window
    # Need historical data before start_date for correlation calculation
    data_start_date = start_date - pd.Timedelta(days=lookback_days * 2)  # Buffer for trading days

    logger.info(
        f"Loading returns data: {len(instruments)} instruments, "
        f"{data_start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')} "
        f"(lookback: {lookback_days} days)"
    )

    # Load close prices (with lookback window)
    price_data = D.features(
        instruments=instruments,
        fields=["$close"],
        start_time=data_start_date.strftime("%Y-%m-%d"),
        end_time=end_date.strftime("%Y-%m-%d"),
    )

    if price_data.empty:
        logger.warning("No price data loaded from Qlib")
        return pd.DataFrame()

    # Normalize data format to MultiIndex (instrument, datetime)
    normalized_data = normalize_qlib_features_result(price_data)

    if normalized_data.empty:
        logger.warning("Normalized price data is empty")
        return pd.DataFrame()

    # Calculate returns for each instrument
    # After normalization, data format is: MultiIndex (instrument, datetime) index, single-level columns
    returns_data = {}

    for instrument in instruments:
        try:
            # Get data for this instrument (normalized format: MultiIndex index)
            if not isinstance(normalized_data.index, pd.MultiIndex):
                logger.warning(f"Expected MultiIndex index, got {type(normalized_data.index)}")
                continue

            instrument_data = normalized_data.loc[instrument, :]

            if instrument_data.empty:
                continue

            # Get close prices
            if isinstance(instrument_data, pd.Series):
                # Single column case
                close_prices = instrument_data
            elif '$close' in instrument_data.columns:
                close_prices = instrument_data['$close']
            elif len(instrument_data.columns) == 1:
                # Only one column, use it
                close_prices = instrument_data.iloc[:, 0]
            else:
                logger.debug(f"Unexpected columns for {instrument}: {instrument_data.columns.tolist()}")
                continue

            # Calculate returns (pct_change)
            returns = close_prices.pct_change(fill_method=None).dropna()

            if not returns.empty:
                # After normalization, index should be datetime (single level after loc[instrument])
                # Store returns with dates as index
                returns_data[instrument] = returns

        except KeyError:
            # Instrument not in data
            continue
        except Exception as e:
            logger.debug(f"Failed to calculate returns for {instrument}: {e}")
            continue

    if not returns_data:
        logger.warning("No returns data calculated")
        return pd.DataFrame()

    # Convert to DataFrame (columns: instruments, index: dates)
    returns_df = pd.DataFrame(returns_data)

    # Sort by date
    returns_df = returns_df.sort_index()

    logger.info(f"Loaded returns data: {len(returns_df)} days, {len(returns_df.columns)} stocks")

    return returns_df


def load_highs_lows_data(
    instruments: List[str],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    lookback_days: int = 120,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load highs and lows data for VolatilitySync correlation calculation.
    
    Args:
        instruments: List of stock symbols.
        start_date: Start date for testing.
        end_date: End date for testing.
        lookback_days: Number of days to look back before start_date for historical data.
    
    Returns:
        Tuple of (highs_df, lows_df) DataFrames.
    """
    from qlib.data import D
    from nq.trading.utils.data_normalizer import normalize_qlib_features_result
    
    # Calculate actual start date with lookback window
    data_start_date = start_date - pd.Timedelta(days=lookback_days * 2)  # Buffer for trading days
    
    logger.info(
        f"Loading highs/lows data: {len(instruments)} instruments, "
        f"{data_start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
    )
    
    # Load high and low prices
    price_data = D.features(
        instruments=instruments,
        fields=["$high", "$low"],
        start_time=data_start_date.strftime("%Y-%m-%d"),
        end_time=end_date.strftime("%Y-%m-%d"),
    )
    
    if price_data.empty:
        logger.warning("No highs/lows data loaded from Qlib")
        return pd.DataFrame(), pd.DataFrame()
    
    # Normalize data format
    normalized_data = normalize_qlib_features_result(price_data)
    
    if normalized_data.empty:
        logger.warning("Normalized highs/lows data is empty")
        return pd.DataFrame(), pd.DataFrame()
    
    # Extract highs and lows for each instrument
    highs_data = {}
    lows_data = {}
    
    for instrument in instruments:
        try:
            if not isinstance(normalized_data.index, pd.MultiIndex):
                continue
            
            instrument_data = normalized_data.loc[instrument, :]
            
            if instrument_data.empty:
                continue
            
            # Get high and low prices
            if '$high' in instrument_data.columns:
                highs = instrument_data['$high']
            elif len(instrument_data.columns) >= 2:
                highs = instrument_data.iloc[:, 0]  # First column
            else:
                continue
            
            if '$low' in instrument_data.columns:
                lows = instrument_data['$low']
            elif len(instrument_data.columns) >= 2:
                lows = instrument_data.iloc[:, 1]  # Second column
            else:
                continue
            
            if not highs.empty and not lows.empty:
                highs_data[instrument] = highs
                lows_data[instrument] = lows
        
        except KeyError:
            continue
        except Exception as e:
            logger.debug(f"Failed to extract highs/lows for {instrument}: {e}")
            continue
    
    if not highs_data or not lows_data:
        logger.warning("No highs/lows data extracted")
        return pd.DataFrame(), pd.DataFrame()
    
    # Convert to DataFrames
    highs_df = pd.DataFrame(highs_data)
    lows_df = pd.DataFrame(lows_data)
    
    # Sort by date
    highs_df = highs_df.sort_index()
    lows_df = lows_df.sort_index()
    
    logger.info(f"Loaded highs/lows data: {len(highs_df)} days, {len(highs_df.columns)} stocks")
    
    return highs_df, lows_df


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Correlation algorithm optimization test")
    parser.add_argument(
        "--start-date",
        type=str,
        required=True,
        help="Start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        required=True,
        help="End date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to StructureExpert model file",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Number of top stocks to track (default: 20)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Output directory for results (default: outputs)",
    )
    parser.add_argument(
        "--instruments",
        type=str,
        nargs="+",
        help="List of stock symbols (if not provided, uses all from Qlib)",
    )
    parser.add_argument(
        "--qlib-dir",
        type=str,
        default=None,
        help="Qlib data directory (default: ~/.qlib/qlib_data/cn_data)",
    )
    parser.add_argument(
        "--region",
        type=str,
        default="cn",
        help="Qlib region (default: cn)",
    )

    args = parser.parse_args()

    # Initialize Qlib first
    if args.qlib_dir is None:
        qlib_dir = str(Path.home() / ".qlib" / "qlib_data" / "cn_data")
    else:
        qlib_dir = str(Path(args.qlib_dir).expanduser())

    logger.info(f"Initializing Qlib with data directory: {qlib_dir}")
    qlib.init(provider_uri=qlib_dir, region=args.region)

    # Parse dates
    start_date = pd.Timestamp(args.start_date)
    end_date = pd.Timestamp(args.end_date)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Starting correlation optimization test")
    logger.info(f"Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    logger.info(f"Model: {args.model_path}")
    logger.info(f"Top K: {args.top_k}")

    # Load instruments
    if args.instruments:
        instruments = args.instruments
    else:
        # Use all instruments from Qlib as default
        from qlib.data import D

        # Get instruments filter
        instruments_filter = D.instruments(market="all")

        # Convert filter to list using D.list_instruments
        try:
            instruments = D.list_instruments(
                instruments=instruments_filter,
                as_list=True,
                freq="day"
            )
        except Exception as e:
            logger.warning(f"Failed to use D.list_instruments: {e}, trying direct conversion")
            # Fallback: try direct conversion
            if hasattr(instruments_filter, "__iter__"):
                try:
                    instruments = list(instruments_filter)
                except Exception as e2:
                    logger.error(f"Failed to convert instruments to list: {e2}")
                    # Last resort: read from file
                    logger.info("Trying to read instruments from file...")
                    qlib_dir_path = Path(qlib_dir)
                    instruments_file = qlib_dir_path / 'instruments' / 'all.txt'
                    if instruments_file.exists():
                        instruments = []
                        with open(instruments_file, 'r') as f:
                            for line in f:
                                line = line.strip()
                                if not line:
                                    continue
                                parts = line.split('\t')
                                if len(parts) >= 1:
                                    stock_code = parts[0].strip()
                                    if stock_code:
                                        instruments.append(stock_code)
                        logger.info(f"Loaded {len(instruments)} instruments from file")
                    else:
                        logger.error(f"Instruments file not found: {instruments_file}")
                        instruments = []
            else:
                instruments = []

        if not instruments:
            logger.error("No instruments found. Please check Qlib data.")
            return

        logger.info(f"Loaded {len(instruments)} instruments from Qlib")

        # Limit to first 100 stocks for faster testing (can be removed)
        if len(instruments) > 100:
            logger.info(f"Limiting to first 100 instruments for faster testing")
            instruments = instruments[:100]

    logger.info(f"Using {len(instruments)} instruments: {instruments[:5]}..." if len(
        instruments) > 5 else f"Using {len(instruments)} instruments: {instruments}")

    # Load model
    logger.info("Loading StructureExpert model...")
    data_start_date, _ = get_qlib_data_range()

    # For DirectionalStockGNN, we need edge attributes
    # Option 1: Use correlation-based edge attributes (recommended)
    # Option 2: Use industry-based edges (requires industry_map)
    # Option 3: Use simple edge attributes (fallback)

    # Try to load industry map if config available
    industry_map = {}
    try:
        config = load_config()
        if hasattr(config, 'database'):
            from datetime import datetime
            from nq.utils.industry import load_industry_map
            end_dt = datetime.strptime(args.end_date, "%Y-%m-%d")
            industry_map = load_industry_map(config.database, target_date=end_dt)
            logger.info(f"Loaded industry mapping: {len(industry_map)} stocks")
    except Exception as e:
        logger.debug(f"Could not load industry map: {e}, using empty map")

    # Enable correlation for edge attributes if DirectionalStockGNN
    # Note: This requires historical returns data, which we'll load on-demand
    builder = GraphDataBuilder(
        industry_map=industry_map,
        use_correlation=True,  # Enable correlation for edge attributes
    )

    model = StructureExpertBuyModel(
        model_path=args.model_path,
        builder=builder,
        device="cpu",  # Use CPU for compatibility
    )

    # Stage 1: Generate return matrix
    logger.info("=" * 80)
    logger.info("Stage 1: Generating return matrix")
    logger.info("=" * 80)

    logger.info("Loading daily rankings...")
    daily_ranks = load_daily_ranks(model, start_date, end_date, instruments)

    if not daily_ranks:
        logger.error("No daily ranks generated. Exiting.")
        return

    logger.info("Generating return matrix...")
    return_matrix_gen = ReturnMatrixGenerator(
        holding_periods=[3, 5, 8, 10, 15, 20, 30, 60],
        top_k=args.top_k,
    )

    return_matrix = return_matrix_gen.generate(
        daily_ranks=daily_ranks,
        instruments=instruments,
    )

    if return_matrix.empty:
        logger.error("Failed to generate return matrix. Exiting.")
        return

    logger.info(f"Generated return matrix: {len(return_matrix)} entries")

    # Save return matrix
    return_matrix_path = output_dir / "return_matrix.csv"
    return_matrix.to_csv(return_matrix_path)
    logger.info(f"Saved return matrix to {return_matrix_path}")

    # Stage 2: Optimize correlation algorithms
    logger.info("=" * 80)
    logger.info("Stage 2: Optimizing correlation algorithms")
    logger.info("=" * 80)

    logger.info("Loading returns data...")
    returns_data = load_returns_data(instruments, start_date, end_date)
    
    if returns_data.empty:
        logger.error("Failed to load returns data. Exiting.")
        return
    
    # Load highs/lows data for VolatilitySync
    logger.info("Loading highs/lows data for VolatilitySync...")
    highs_data, lows_data = load_highs_lows_data(instruments, start_date, end_date)
    
    if highs_data.empty or lows_data.empty:
        logger.warning("Failed to load highs/lows data. VolatilitySync will be skipped.")
        highs_data = None
        lows_data = None
    
    logger.info("Running optimization...")
    optimizer = CorrelationOptimizer(
        window_params=[3, 5, 8, 13],
        thresholds=[0.5, 0.6, 0.7, 0.8, 0.9],
        top_k=args.top_k,
    )
    
    optimization_results = optimizer.optimize(
        daily_ranks=daily_ranks,
        return_matrix=return_matrix,
        returns_data=returns_data,
        highs_data=highs_data,
        lows_data=lows_data,
    )

    if optimization_results.empty:
        logger.error("Optimization produced no results. Exiting.")
        return

    logger.info(f"Optimization completed: {len(optimization_results)} combinations evaluated")

    # Save optimization results
    optimization_path = output_dir / "optimization_results.csv"
    optimization_results.to_csv(optimization_path, index=False)
    logger.info(f"Saved optimization results to {optimization_path}")

    # Stage 3: Generate sensitivity analysis
    logger.info("=" * 80)
    logger.info("Stage 3: Generating sensitivity analysis")
    logger.info("=" * 80)

    analyzer = SensitivityAnalyzer(output_dir=str(output_dir))

    # Generate optimal combination matrix
    logger.info("Generating optimal combination matrix...")
    optimal_matrix = analyzer.generate_optimal_combination_matrix(optimization_results)

    optimal_path = output_dir / "optimal_combinations.csv"
    optimal_matrix.to_csv(optimal_path, index=False)
    logger.info(f"Saved optimal combinations to {optimal_path}")

    # Print summary
    print("\n" + "=" * 80)
    print("OPTIMAL COMBINATIONS SUMMARY")
    print("=" * 80)
    print(optimal_matrix.to_string(index=False))

    # Generate plots
    logger.info("Generating ranking drift plot...")
    analyzer.plot_ranking_drift(return_matrix)

    logger.info("Generating correlation threshold heatmap...")
    analyzer.plot_correlation_threshold_heatmap(
        optimization_results,
        target_period=5,
    )

    # Generate summary report
    logger.info("Generating summary report...")
    summary_report = analyzer.generate_summary_report(
        optimization_results,
        return_matrix,
    )

    logger.info("=" * 80)
    logger.info("Analysis completed successfully!")
    logger.info(f"Results saved to: {output_dir}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

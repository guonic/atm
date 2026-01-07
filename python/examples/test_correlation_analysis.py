#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for EnhancedGraphDataBuilder and correlation analysis methods.

This script demonstrates how to use the correlation analysis module to:
1. Calculate various correlation matrices
2. Build enhanced graphs with multi-dimensional edge features
3. Visualize correlation relationships

Usage:
    # Use mock data (default)
    python python/examples/test_correlation_analysis.py

    # Use real data from database
    python python/examples/test_correlation_analysis.py --use-db

    # Use real data with specific stocks and date range
    python python/examples/test_correlation_analysis.py --use-db \
        --stocks 000001.SZ 000002.SZ 600000.SH \
        --start-date 2024-01-01 --end-date 2024-06-30

    # Use Qlib data
    python python/examples/test_correlation_analysis.py --use-qlib \
        --qlib-dir ~/.qlib/qlib_data/cn_data
"""

import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from nq.analysis.correlation import (
    CrossLaggedCorrelation,
    DynamicCrossSectionalCorrelation,
    EnhancedGraphDataBuilder,
    GrangerCausality,
    TransferEntropy,
    VolatilitySync,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_data_from_database(
    stocks: List[str],
    start_date: datetime,
    end_date: datetime,
    config_path: Optional[str] = None,
    schema: str = "quant",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """
    Load stock data from database.
    
    Args:
        stocks: List of stock codes (e.g., ['000001.SZ', '000002.SZ']).
        start_date: Start date.
        end_date: End date.
        config_path: Path to config file (default: config/config.yaml).
        schema: Database schema name.
    
    Returns:
        Tuple of (returns_df, highs_df, lows_df, industry_map).
    """
    from nq.config import load_config
    from nq.repo.kline_repo import StockKlineDayRepo
    from nq.utils.industry import load_industry_map
    
    # Load config
    if config_path is None:
        config_path = "config/config.yaml"
    
    config = load_config(config_path)
    db_config = config.database
    
    # Load industry map
    logger.info("Loading industry mapping...")
    industry_map = load_industry_map(db_config, target_date=end_date, schema=schema)
    
    # Load kline data for each stock
    repo = StockKlineDayRepo(db_config, schema)
    
    returns_data = {}
    highs_data = {}
    lows_data = {}
    
    for stock in stocks:
        logger.info(f"Loading data for {stock}...")
        klines = repo.get_by_ts_code(
            ts_code=stock,
            start_time=start_date,
            end_time=end_date,
        )
        
        if not klines:
            logger.warning(f"No data found for {stock}")
            continue
        
        # Convert to DataFrame
        data_list = []
        for kline in klines:
            data_list.append({
                "date": kline.trade_date,
                "open": float(kline.open) if kline.open else None,
                "high": float(kline.high) if kline.high else None,
                "low": float(kline.low) if kline.low else None,
                "close": float(kline.close) if kline.close else None,
            })
        
        df = pd.DataFrame(data_list)
        df = df.sort_values("date").reset_index(drop=True)
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)
        
        # Calculate returns
        returns = df["close"].pct_change().dropna()
        
        # Store data
        returns_data[stock] = returns
        highs_data[stock] = df["high"]
        lows_data[stock] = df["low"]
    
    # Align all dataframes
    returns_df = pd.DataFrame(returns_data)
    highs_df = pd.DataFrame(highs_data)
    lows_df = pd.DataFrame(lows_data)
    
    # Drop rows with all NaN
    returns_df = returns_df.dropna(how="all")
    highs_df = highs_df.dropna(how="all")
    lows_df = lows_df.dropna(how="all")
    
    logger.info(f"Loaded data: {len(returns_df)} days, {len(returns_df.columns)} stocks")
    
    return returns_df, highs_df, lows_df, industry_map


def load_data_from_qlib(
    stocks: List[str],
    start_date: str,
    end_date: str,
    qlib_dir: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """
    Load stock data from Qlib.
    
    Args:
        stocks: List of stock codes (e.g., ['000001.SZ', '000002.SZ']).
        start_date: Start date (YYYY-MM-DD).
        end_date: End date (YYYY-MM-DD).
        qlib_dir: Qlib data directory.
    
    Returns:
        Tuple of (returns_df, highs_df, lows_df, industry_map).
    """
    import importlib.util
    
    # Check if qlib is available
    _qlib_spec = importlib.util.find_spec("qlib")
    if _qlib_spec is None:
        raise ImportError("Qlib is not installed. Please install it first: pip install pyqlib")
    
    import qlib
    from qlib.data import D
    
    # Initialize Qlib
    if qlib_dir is None:
        qlib_dir = str(Path.home() / ".qlib" / "qlib_data" / "cn_data")
    
    qlib.init(provider_uri=qlib_dir, region="cn")
    
    # Load data
    logger.info(f"Loading data from Qlib: {qlib_dir}")
    
    returns_data = {}
    highs_data = {}
    lows_data = {}
    
    for stock in stocks:
        logger.info(f"Loading data for {stock}...")
        try:
            # Load OHLCV data
            data = D.features(
                instruments=[stock],
                fields=["$close", "$high", "$low"],
                start_time=start_date,
                end_time=end_date,
            )
            
            if data.empty:
                logger.warning(f"No data found for {stock}")
                continue
            
            # Calculate returns
            returns = data[f"{stock}"]["$close"].pct_change().dropna()
            
            # Store data
            returns_data[stock] = returns
            highs_data[stock] = data[f"{stock}"]["$high"]
            lows_data[stock] = data[f"{stock}"]["$low"]
        except Exception as e:
            logger.warning(f"Failed to load {stock}: {e}")
            continue
    
    # Align all dataframes
    returns_df = pd.DataFrame(returns_data)
    highs_df = pd.DataFrame(highs_data)
    lows_df = pd.DataFrame(lows_data)
    
    # Drop rows with all NaN
    returns_df = returns_df.dropna(how="all")
    highs_df = highs_df.dropna(how="all")
    lows_df = lows_df.dropna(how="all")
    
    # Load industry map (simplified - would need proper Qlib industry data)
    industry_map = {}
    
    logger.info(f"Loaded data: {len(returns_df)} days, {len(returns_df.columns)} stocks")
    
    return returns_df, highs_df, lows_df, industry_map


def generate_mock_data(n_stocks: int = 10, n_days: int = 100) -> tuple:
    """
    Generate mock stock data for testing.
    
    Args:
        n_stocks: Number of stocks.
        n_days: Number of trading days.
    
    Returns:
        Tuple of (returns_df, highs_df, lows_df, features_df, industry_map).
    """
    # Generate dates
    start_date = datetime(2024, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(n_days)]
    
    # Generate stock symbols
    symbols = [f"{i:06d}.SZ" for i in range(1, n_stocks + 1)]
    
    # Generate returns (with some correlation structure)
    np.random.seed(42)
    
    # Create correlated returns
    base_returns = np.random.randn(n_days, n_stocks) * 0.02
    
    # Add some correlation: stocks 0-2 are correlated, stocks 3-5 are correlated
    base_returns[:, 1] = base_returns[:, 0] * 0.7 + np.random.randn(n_days) * 0.01
    base_returns[:, 2] = base_returns[:, 0] * 0.5 + np.random.randn(n_days) * 0.015
    base_returns[:, 4] = base_returns[:, 3] * 0.8 + np.random.randn(n_days) * 0.01
    base_returns[:, 5] = base_returns[:, 3] * 0.6 + np.random.randn(n_days) * 0.012
    
    returns_df = pd.DataFrame(base_returns, index=dates, columns=symbols)
    
    # Generate high/low prices (for volatility sync)
    # Start with base price of 10
    prices = 10 * (1 + returns_df).cumprod()
    highs_df = prices * (1 + np.abs(np.random.randn(n_days, n_stocks)) * 0.01)
    lows_df = prices * (1 - np.abs(np.random.randn(n_days, n_stocks)) * 0.01)
    
    # Generate node features (Alpha158-like features)
    n_features = 158
    features_data = np.random.randn(n_days * n_stocks, n_features)
    
    # Create MultiIndex
    dates_repeated = np.repeat(dates, n_stocks)
    symbols_repeated = symbols * n_days
    multi_index = pd.MultiIndex.from_arrays(
        [dates_repeated, symbols_repeated],
        names=["datetime", "instrument"]
    )
    
    features_df = pd.DataFrame(features_data, index=multi_index)
    
    # Generate industry map (group stocks into industries)
    industry_map = {}
    for i, symbol in enumerate(symbols):
        industry_code = f"IND{i // 3}"  # Every 3 stocks in same industry
        industry_map[symbol] = industry_code
    
    return returns_df, highs_df, lows_df, features_df, industry_map


def test_dynamic_cross_sectional_correlation(returns_df: pd.DataFrame):
    """Test dynamic cross-sectional correlation."""
    print("\n" + "=" * 80)
    print("1. Dynamic Cross-Sectional Correlation")
    print("=" * 80)
    
    calc = DynamicCrossSectionalCorrelation(window=60, threshold=0.3)
    corr_matrix = calc.calculate(returns_df)
    
    print(f"\nCorrelation Matrix (threshold=0.3):")
    print(corr_matrix.round(3))
    
    # Test pairwise
    symbols = returns_df.columns.tolist()
    if len(symbols) >= 2:
        corr_pair = calc.calculate_pairwise(returns_df[symbols[0]], returns_df[symbols[1]])
        print(f"\nPairwise correlation ({symbols[0]} vs {symbols[1]}): {corr_pair:.4f}")


def test_cross_lagged_correlation(returns_df: pd.DataFrame):
    """Test cross-lagged correlation."""
    print("\n" + "=" * 80)
    print("2. Cross-Lagged Correlation")
    print("=" * 80)
    
    calc = CrossLaggedCorrelation(lag=1)
    
    # Test pairwise
    symbols = returns_df.columns.tolist()
    if len(symbols) >= 2:
        corr_i_to_j, corr_j_to_i, direction = calc.calculate_directed(
            returns_df[symbols[0]], returns_df[symbols[1]]
        )
        print(f"\nPairwise lagged correlation ({symbols[0]} vs {symbols[1]}):")
        print(f"  {symbols[0]} -> {symbols[1]}: {corr_i_to_j:.4f}")
        print(f"  {symbols[1]} -> {symbols[0]}: {corr_j_to_i:.4f}")
        print(f"  Direction: {direction}")
    
    # Calculate matrix
    corr_matrix, direction_matrix = calc.calculate_matrix(returns_df)
    
    print(f"\nLagged Correlation Matrix:")
    print(corr_matrix.round(3))
    
    print(f"\nDirection Matrix:")
    print(direction_matrix)


def test_volatility_sync(highs_df: pd.DataFrame, lows_df: pd.DataFrame):
    """Test volatility sync."""
    print("\n" + "=" * 80)
    print("3. Volatility Sync")
    print("=" * 80)
    
    calc = VolatilitySync(bandwidth=0.1)
    
    # Test pairwise
    symbols = highs_df.columns.tolist()
    if len(symbols) >= 2:
        range_i = calc.calculate_range(highs_df[symbols[0]], lows_df[symbols[0]])
        range_j = calc.calculate_range(highs_df[symbols[1]], lows_df[symbols[1]])
        sync_rate = calc.calculate_sync_rate(range_i, range_j)
        print(f"\nPairwise sync rate ({symbols[0]} vs {symbols[1]}): {sync_rate:.4f}")
    
    # Calculate matrix
    sync_matrix = calc.calculate_matrix(highs_df, lows_df)
    
    print(f"\nVolatility Sync Matrix:")
    print(sync_matrix.round(3))


def test_granger_causality(returns_df: pd.DataFrame):
    """Test Granger causality."""
    print("\n" + "=" * 80)
    print("4. Granger Causality Test")
    print("=" * 80)
    
    calc = GrangerCausality(maxlag=2, significance_level=0.05)
    
    # Test pairwise
    symbols = returns_df.columns.tolist()
    if len(symbols) >= 2:
        is_causal, p_value, direction = calc.test(
            returns_df[symbols[0]], returns_df[symbols[1]]
        )
        print(f"\nPairwise Granger causality ({symbols[0]} vs {symbols[1]}):")
        print(f"  Is causal: {is_causal}")
        print(f"  P-value: {p_value:.4f}")
        print(f"  Direction: {direction}")
    
    # Calculate matrix (may take time for many stocks)
    print("\nCalculating Granger causality matrix (this may take a while)...")
    causality_matrix, p_value_matrix = calc.calculate_matrix(returns_df)
    
    print(f"\nGranger Causality Matrix (True = causal):")
    print(causality_matrix)
    
    print(f"\nP-value Matrix:")
    print(p_value_matrix.round(4))


def test_transfer_entropy(returns_df: pd.DataFrame):
    """Test transfer entropy."""
    print("\n" + "=" * 80)
    print("5. Transfer Entropy")
    print("=" * 80)
    
    calc = TransferEntropy(n_bins=3, threshold_percentile=(33.3, 66.7))
    
    # Test pairwise
    symbols = returns_df.columns.tolist()
    if len(symbols) >= 2:
        te_a_to_b, te_b_to_a, direction = calc.calculate(
            returns_df[symbols[0]], returns_df[symbols[1]]
        )
        print(f"\nPairwise transfer entropy ({symbols[0]} vs {symbols[1]}):")
        print(f"  {symbols[0]} -> {symbols[1]}: {te_a_to_b:.4f}")
        print(f"  {symbols[1]} -> {symbols[0]}: {te_b_to_a:.4f}")
        print(f"  Direction: {direction}")
    
    # Calculate matrix
    print("\nCalculating transfer entropy matrix...")
    te_matrix, direction_matrix = calc.calculate_matrix(returns_df)
    
    print(f"\nTransfer Entropy Matrix:")
    print(te_matrix.round(4))
    
    print(f"\nDirection Matrix:")
    print(direction_matrix)


def test_enhanced_graph_builder(
    returns_df: pd.DataFrame,
    highs_df: pd.DataFrame,
    lows_df: pd.DataFrame,
    features_df: pd.DataFrame,
    industry_map: dict,
):
    """Test enhanced graph builder."""
    print("\n" + "=" * 80)
    print("6. Enhanced Graph Builder")
    print("=" * 80)
    
    builder = EnhancedGraphDataBuilder(
        industry_map=industry_map,
        correlation_window=60,
        correlation_threshold=0.3,
        lag=1,
        use_granger=False,  # Set to False for faster testing
        use_transfer_entropy=True,
    )
    
    # Build edge features
    print("\nBuilding edge features...")
    edge_features = builder.build_edge_features(
        returns=returns_df,
        highs=highs_df,
        lows=lows_df,
    )
    
    print(f"\nEdge Features Summary:")
    print(f"  Total edges: {len(edge_features)}")
    
    # Show sample edge features
    print(f"\nSample Edge Features (first 5):")
    for i, ((sym_i, sym_j), features) in enumerate(list(edge_features.items())[:5]):
        print(f"  {sym_i} -> {sym_j}:")
        print(f"    [correlation={features[0]:.4f}, lagged={features[1]:.4f}, "
              f"volatility_sync={features[2]:.4f}, transfer_entropy={features[3]:.4f}]")
    
    # Build graph
    print("\nBuilding graph...")
    graph = builder.build_graph(
        node_features=features_df,
        returns=returns_df,
        highs=highs_df,
        lows=lows_df,
        use_industry_edges=True,
        use_correlation_edges=True,
        correlation_threshold=0.2,
    )
    
    print(f"\nGraph Summary:")
    print(f"  Nodes: {graph.x.shape[0]}")
    print(f"  Node features: {graph.x.shape[1]}")
    print(f"  Edges: {graph.edge_index.shape[1]}")
    print(f"  Edge features: {graph.edge_attr.shape[1]} (dimensions)")
    
    # Analyze edge features
    if graph.edge_attr.shape[0] > 0:
        edge_attr_df = pd.DataFrame(
            graph.edge_attr.numpy(),
            columns=["correlation", "lagged_weight", "volatility_sync", "transfer_entropy"]
        )
        
        print(f"\nEdge Features Statistics:")
        print(edge_attr_df.describe().round(4))
        
        # Count edge types
        industry_edges = (edge_attr_df["correlation"] == 1.0).sum()
        correlation_edges = (edge_attr_df["correlation"] != 1.0).sum()
        
        print(f"\nEdge Type Distribution:")
        print(f"  Industry edges: {industry_edges}")
        print(f"  Correlation edges: {correlation_edges}")


def main():
    """Main test function."""
    parser = argparse.ArgumentParser(
        description="Test EnhancedGraphDataBuilder and correlation analysis methods"
    )
    parser.add_argument(
        "--use-db",
        action="store_true",
        help="Use real data from database instead of mock data",
    )
    parser.add_argument(
        "--use-qlib",
        action="store_true",
        help="Use real data from Qlib instead of mock data",
    )
    parser.add_argument(
        "--stocks",
        nargs="+",
        default=["000001.SZ", "000002.SZ", "600000.SH", "600036.SH", "000858.SZ"],
        help="List of stock codes (default: ['000001.SZ', '000002.SZ', ...])",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default="2024-01-01",
        help="Start date (YYYY-MM-DD, default: 2024-01-01)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default="2024-06-30",
        help="End date (YYYY-MM-DD, default: 2024-06-30)",
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default="config/config.yaml",
        help="Path to config file (default: config/config.yaml)",
    )
    parser.add_argument(
        "--qlib-dir",
        type=str,
        default=None,
        help="Qlib data directory (default: ~/.qlib/qlib_data/cn_data)",
    )
    parser.add_argument(
        "--test-granger",
        action="store_true",
        help="Run Granger causality test (may be slow)",
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Enhanced Graph Data Builder Test")
    print("=" * 80)
    
    # Load data
    if args.use_db:
        print("\nLoading data from database...")
        start_dt = datetime.strptime(args.start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(args.end_date, "%Y-%m-%d")
        
        returns_df, highs_df, lows_df, industry_map = load_data_from_database(
            stocks=args.stocks,
            start_date=start_dt,
            end_date=end_dt,
            config_path=args.config_path,
        )
        
        # Generate mock features (since we don't have Alpha158 features in DB)
        print("Generating mock features...")
        n_days = len(returns_df)
        n_stocks = len(returns_df.columns)
        n_features = 158
        
        features_data = np.random.randn(n_days * n_stocks, n_features)
        dates_repeated = np.repeat(returns_df.index, n_stocks)
        symbols_repeated = list(returns_df.columns) * n_days
        multi_index = pd.MultiIndex.from_arrays(
            [dates_repeated, symbols_repeated],
            names=["datetime", "instrument"]
        )
        features_df = pd.DataFrame(features_data, index=multi_index)
        
    elif args.use_qlib:
        print("\nLoading data from Qlib...")
        returns_df, highs_df, lows_df, industry_map = load_data_from_qlib(
            stocks=args.stocks,
            start_date=args.start_date,
            end_date=args.end_date,
            qlib_dir=args.qlib_dir,
        )
        
        # Generate mock features
        print("Generating mock features...")
        n_days = len(returns_df)
        n_stocks = len(returns_df.columns)
        n_features = 158
        
        features_data = np.random.randn(n_days * n_stocks, n_features)
        dates_repeated = np.repeat(returns_df.index, n_stocks)
        symbols_repeated = list(returns_df.columns) * n_days
        multi_index = pd.MultiIndex.from_arrays(
            [dates_repeated, symbols_repeated],
            names=["datetime", "instrument"]
        )
        features_df = pd.DataFrame(features_data, index=multi_index)
        
    else:
        # Generate mock data
        print("\nGenerating mock data...")
        returns_df, highs_df, lows_df, features_df, industry_map = generate_mock_data(
            n_stocks=len(args.stocks), n_days=100
        )
    
    print(f"\nData summary:")
    print(f"  Stocks: {len(returns_df.columns)}")
    print(f"  Trading days: {len(returns_df)}")
    print(f"  Features per stock: {features_df.shape[1]}")
    print(f"  Industries: {len(set(industry_map.values())) if industry_map else 0}")
    
    # Test individual correlation methods
    test_dynamic_cross_sectional_correlation(returns_df)
    test_cross_lagged_correlation(returns_df)
    test_volatility_sync(highs_df, lows_df)
    
    # Granger causality test (optional, may be slow)
    if args.test_granger:
        test_granger_causality(returns_df)
    
    test_transfer_entropy(returns_df)
    
    # Test enhanced graph builder
    test_enhanced_graph_builder(
        returns_df, highs_df, lows_df, features_df, industry_map
    )
    
    print("\n" + "=" * 80)
    print("Test completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for EnhancedGraphDataBuilder and correlation analysis methods.

This script demonstrates how to use the correlation analysis module to:
1. Calculate various correlation matrices
2. Build enhanced graphs with multi-dimensional edge features
3. Visualize correlation relationships

Usage:
    cd /Users/guonic/Workspace/OpenSource/atm
    source .venv/bin/activate
    export PYTHONPATH=/Users/guonic/Workspace/OpenSource/atm/python:$PYTHONPATH
    python python/examples/test_correlation_analysis.py
"""

import logging
from datetime import datetime, timedelta

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
    print("=" * 80)
    print("Enhanced Graph Data Builder Test")
    print("=" * 80)
    
    # Generate mock data
    print("\nGenerating mock data...")
    returns_df, highs_df, lows_df, features_df, industry_map = generate_mock_data(
        n_stocks=10, n_days=100
    )
    
    print(f"Generated data:")
    print(f"  Stocks: {len(returns_df.columns)}")
    print(f"  Trading days: {len(returns_df)}")
    print(f"  Features per stock: {features_df.shape[1]}")
    print(f"  Industries: {len(set(industry_map.values()))}")
    
    # Test individual correlation methods
    test_dynamic_cross_sectional_correlation(returns_df)
    test_cross_lagged_correlation(returns_df)
    test_volatility_sync(highs_df, lows_df)
    
    # Note: Granger causality test may be slow, skip for quick test
    # Uncomment to test:
    # test_granger_causality(returns_df)
    
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


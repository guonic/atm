"""
Enhanced graph builder using correlation analysis.

Combines multiple correlation methods to build dynamic, directed, multi-center graph topologies.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from .correlation import (
    CrossLaggedCorrelation,
    DynamicCrossSectionalCorrelation,
    GrangerCausality,
    TransferEntropy,
    VolatilitySync,
)

logger = logging.getLogger(__name__)


class EnhancedGraphDataBuilder:
    """
    Enhanced graph data builder using correlation analysis.
    
    Combines multiple correlation methods to build graph edges with multi-dimensional
    edge features for GNN models.
    
    Edge feature vector:
        Edge_Feature_ij = [
            correlation_value,      # Cross-sectional correlation
            lagged_weight,          # Lagged correlation weight
            volatility_sync_score, # Risk synchronization
            transfer_entropy       # Information flow strength
        ]
    """
    
    def __init__(
        self,
        industry_map: Optional[Dict[str, str]] = None,
        correlation_window: int = 60,
        correlation_threshold: Optional[float] = 0.5,
        lag: int = 1,
        use_granger: bool = True,
        use_transfer_entropy: bool = True,
    ):
        """
        Initialize enhanced graph data builder.
        
        Args:
            industry_map: Dictionary mapping stock codes to industry codes (optional).
            correlation_window: Window size for correlation calculation (default: 60).
            correlation_threshold: Threshold for correlation sparsification (default: 0.5).
            lag: Lag period for cross-lagged correlation (default: 1).
            use_granger: Whether to use Granger causality test (default: True).
            use_transfer_entropy: Whether to use transfer entropy (default: True).
        """
        self.industry_map = industry_map or {}
        
        # Initialize correlation calculators
        self.cross_sectional = DynamicCrossSectionalCorrelation(
            window=correlation_window,
            threshold=correlation_threshold,
        )
        self.lagged = CrossLaggedCorrelation(lag=lag)
        self.volatility_sync = VolatilitySync()
        self.granger = GrangerCausality() if use_granger else None
        self.transfer_entropy = TransferEntropy() if use_transfer_entropy else None
    
    def build_edge_features(
        self,
        returns: pd.DataFrame,
        highs: Optional[pd.DataFrame] = None,
        lows: Optional[pd.DataFrame] = None,
        symbols: Optional[List[str]] = None,
    ) -> Dict[Tuple[str, str], np.ndarray]:
        """
        Build edge features for all stock pairs.
        
        Args:
            returns: DataFrame with returns data (columns: symbols, index: dates).
            highs: Optional DataFrame with high prices.
            lows: Optional DataFrame with low prices.
            symbols: Optional list of symbols to process.
        
        Returns:
            Dictionary mapping (symbol_i, symbol_j) to edge feature vector.
        """
        if symbols is None:
            symbols = returns.columns.tolist()
        
        edge_features = {}
        
        # Calculate cross-sectional correlation matrix
        corr_matrix = self.cross_sectional.calculate_matrix(returns, symbols)
        
        # Calculate lagged correlation matrix
        lagged_corr, lagged_dir = self.lagged.calculate_matrix(returns, symbols)
        
        # Calculate volatility sync matrix (if high/low data available)
        if highs is not None and lows is not None:
            sync_matrix = self.volatility_sync.calculate_matrix(
                returns, symbols, highs=highs, lows=lows
            )
        else:
            sync_matrix = pd.DataFrame(0.0, index=symbols, columns=symbols)
        
        # Calculate Granger causality (if enabled)
        if self.granger:
            granger_causality, granger_p = self.granger.calculate_matrix(returns, symbols)
        else:
            granger_causality = pd.DataFrame(False, index=symbols, columns=symbols)
            granger_p = pd.DataFrame(1.0, index=symbols, columns=symbols)
        
        # Calculate transfer entropy (if enabled)
        if self.transfer_entropy:
            te_matrix, te_dir = self.transfer_entropy.calculate_matrix(returns, symbols)
        else:
            te_matrix = pd.DataFrame(0.0, index=symbols, columns=symbols)
        
        # Build edge features for each pair
        for i, sym_i in enumerate(symbols):
            for j, sym_j in enumerate(symbols):
                if i == j:
                    continue
                
                # Extract features
                correlation = corr_matrix.loc[sym_i, sym_j]
                lagged_weight = lagged_corr.loc[sym_i, sym_j]
                volatility_sync = sync_matrix.loc[sym_i, sym_j]
                
                # Use Granger causality as binary feature
                granger_feature = 1.0 if granger_causality.loc[sym_i, sym_j] else 0.0
                
                # Transfer entropy
                transfer_entropy = te_matrix.loc[sym_i, sym_j]
                
                # Build edge feature vector
                edge_feature = np.array([
                    correlation,
                    lagged_weight,
                    volatility_sync,
                    transfer_entropy,
                ])
                
                edge_features[(sym_i, sym_j)] = edge_feature
        
        return edge_features
    
    def build_graph(
        self,
        node_features: pd.DataFrame,
        returns: pd.DataFrame,
        highs: Optional[pd.DataFrame] = None,
        lows: Optional[pd.DataFrame] = None,
        use_industry_edges: bool = True,
        use_correlation_edges: bool = True,
        correlation_threshold: float = 0.3,
    ) -> Data:
        """
        Build PyTorch Geometric graph with correlation-based edges.
        
        Args:
            node_features: DataFrame with node features (MultiIndex: datetime, instrument).
            returns: DataFrame with returns data (columns: symbols, index: dates).
            highs: Optional DataFrame with high prices.
            lows: Optional DataFrame with low prices.
            use_industry_edges: Whether to include industry-based edges (default: True).
            use_correlation_edges: Whether to include correlation-based edges (default: True).
            correlation_threshold: Threshold for correlation edges (default: 0.3).
        
        Returns:
            PyTorch Geometric Data object.
        """
        # Get stock symbols
        if isinstance(node_features.index, pd.MultiIndex):
            symbols = node_features.index.get_level_values("instrument").unique().tolist()
        else:
            symbols = node_features.index.tolist()
        
        # Convert node features to tensor
        x = torch.tensor(node_features.values, dtype=torch.float)
        
        # Build edge list
        edge_index = []
        edge_attr = []
        
        # Add industry-based edges
        if use_industry_edges and self.industry_map:
            ind_to_nodes: Dict[str, List[int]] = {}
            for idx, symbol in enumerate(symbols):
                industry = self.industry_map.get(symbol)
                if industry:
                    if industry not in ind_to_nodes:
                        ind_to_nodes[industry] = []
                    ind_to_nodes[industry].append(idx)
            
            for nodes in ind_to_nodes.values():
                for i in nodes:
                    for j in nodes:
                        if i != j:
                            edge_index.append([i, j])
                            # Default edge feature for industry edges
                            edge_attr.append([1.0, 0.0, 0.0, 0.0])
        
        # Add correlation-based edges
        if use_correlation_edges:
            edge_features = self.build_edge_features(returns, highs, lows, symbols)
            
            # Create symbol to index mapping
            symbol_to_idx = {sym: idx for idx, sym in enumerate(symbols)}
            
            for (sym_i, sym_j), features in edge_features.items():
                if sym_i not in symbol_to_idx or sym_j not in symbol_to_idx:
                    continue
                
                idx_i = symbol_to_idx[sym_i]
                idx_j = symbol_to_idx[sym_j]
                
                # Apply threshold
                if abs(features[0]) >= correlation_threshold:  # correlation threshold
                    edge_index.append([idx_i, idx_j])
                    edge_attr.append(features.tolist())
        
        # Convert to tensors
        if len(edge_index) == 0:
            edge_index_tensor = torch.zeros((2, 0), dtype=torch.long)
            edge_attr_tensor = torch.zeros((0, 4), dtype=torch.float)
        else:
            edge_index_tensor = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            edge_attr_tensor = torch.tensor(edge_attr, dtype=torch.float)
        
        logger.info(
            f"Built graph with {len(symbols)} nodes, {len(edge_index)} edges "
            f"(industry: {len([e for e in edge_attr if e[0] == 1.0])}, "
            f"correlation: {len([e for e in edge_attr if e[0] != 1.0])})"
        )
        
        return Data(
            x=x,
            edge_index=edge_index_tensor,
            edge_attr=edge_attr_tensor,
            symbols=symbols,
        )


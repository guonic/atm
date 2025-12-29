#!/usr/bin/env python3
"""
Structure Expert GNN model for stock prediction.

This module implements a Graph Attention Network (GAT) based model for stock
prediction using industry structure information. The model captures interactions
between stocks within the same industry or concept through graph neural networks.

Classes:
    StructureExpertGNN: GAT-based neural network model for stock prediction.
    GraphDataBuilder: Utility class for converting Qlib data to graph format.
    StructureTrainer: Training manager for the structure expert model.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv

logger = logging.getLogger(__name__)


class StructureExpertGNN(nn.Module):
    """
    Graph Attention Network (GAT) based model for stock prediction.

    This model uses GAT layers to capture interactions between stocks within
    the same industry or concept. It processes stock features through two GAT
    layers and outputs prediction scores.

    Attributes:
        conv1: First GAT layer for capturing stock-industry interactions.
        conv2: Second GAT layer for feature fusion.
        predict: Prediction layer that outputs stock scores.
    """

    def __init__(self, n_feat: int, n_hidden: int = 64, n_heads: int = 4) -> None:
        """
        Initialize the Structure Expert GNN model.

        Args:
            n_feat: Number of input features per stock.
            n_hidden: Hidden dimension size (default: 64).
            n_heads: Number of attention heads in first GAT layer (default: 4).
        """
        super(StructureExpertGNN, self).__init__()
        # First GAT layer: captures interactions between stocks and industry/concept neighbors
        self.conv1 = GATv2Conv(n_feat, n_hidden, heads=n_heads, concat=True)
        # Second GAT layer: feature fusion
        self.conv2 = GATv2Conv(n_hidden * n_heads, n_hidden, heads=1, concat=False)
        # Prediction layer: outputs stock scores
        self.predict = nn.Sequential(
            nn.Linear(n_hidden, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 1),
        )

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the model.

        Args:
            x: Node features tensor of shape [num_nodes, num_features].
            edge_index: Edge index tensor of shape [2, num_edges].

        Returns:
            Tuple of (logits, embedding):
                - logits: Prediction scores of shape [num_nodes, 1].
                - embedding: High-dimensional structure embedding of shape [num_nodes, n_hidden].
        """
        # x: [num_nodes, num_features], edge_index: [2, num_edges]
        h = F.leaky_relu(self.conv1(x, edge_index))
        h = self.conv2(h, edge_index)  # h is high-dimensional structure embedding
        logits = self.predict(h)
        return logits, h


class GraphDataBuilder:
    """
    Utility class for converting Qlib format data to PyTorch Geometric graph format.

    This class builds graph structures where stocks in the same industry are
    connected by edges, enabling the GNN to capture industry relationships.

    Attributes:
        industry_map: Dictionary mapping stock codes to industry IDs.
    """

    def __init__(self, industry_map: Dict[str, int]) -> None:
        """
        Initialize the GraphDataBuilder.

        Args:
            industry_map: Dictionary mapping stock codes to industry IDs.
                Format: {stock_code: industry_id}
        """
        self.industry_map = industry_map

    def get_daily_graph(
        self, df_x: pd.DataFrame, df_y: Optional[pd.DataFrame] = None
    ) -> Data:
        """
        Convert Qlib format daily cross-sectional DataFrame to PyG Data object.

        Args:
            df_x: Input features DataFrame with MultiIndex (datetime, instrument).
            df_y: Optional target values DataFrame with same index as df_x.

        Returns:
            PyTorch Geometric Data object containing:
                - x: Node features tensor.
                - y: Optional target values tensor.
                - edge_index: Edge index tensor connecting stocks in same industry.
                - symbols: List of stock symbols.
        """
        # Clean NaN and Inf values before converting to tensor
        if df_x.isna().any().any():
            logger.warning(f"Found NaN in df_x, filling with 0")
            df_x = df_x.fillna(0.0)
        if (df_x == np.inf).any().any() or (df_x == -np.inf).any().any():
            logger.warning(f"Found Inf in df_x, replacing with 0")
            df_x = df_x.replace([np.inf, -np.inf], 0.0)
        
        stock_list = df_x.index.get_level_values("instrument").tolist()
        x = torch.tensor(df_x.values, dtype=torch.float)
        
        # Check for NaN/Inf in tensor (should not happen after cleaning, but double check)
        if torch.isnan(x).any() or torch.isinf(x).any():
            logger.warning("Found NaN/Inf in tensor after conversion, replacing with 0")
            x = torch.where(torch.isnan(x) | torch.isinf(x), torch.zeros_like(x), x)
        
        y = None
        if df_y is not None:
            # Clean labels
            if df_y.isna().any().any():
                logger.warning(f"Found NaN in df_y, filling with 0")
                df_y = df_y.fillna(0.0)
            if (df_y == np.inf).any().any() or (df_y == -np.inf).any().any():
                logger.warning(f"Found Inf in df_y, replacing with 0")
                df_y = df_y.replace([np.inf, -np.inf], 0.0)
            
            y = torch.tensor(df_y.values, dtype=torch.float)
            
            # Check for NaN/Inf in y tensor
            if torch.isnan(y).any() or torch.isinf(y).any():
                logger.warning("Found NaN/Inf in y tensor after conversion, replacing with 0")
                y = torch.where(torch.isnan(y) | torch.isinf(y), torch.zeros_like(y), y)

        # Build edges: connect stocks in the same industry pairwise
        edge_index = []
        # Pre-optimization: group nodes by industry
        ind_to_nodes: Dict[int, List[int]] = {}
        for idx, symbol in enumerate(stock_list):
            ind = self.industry_map.get(symbol, -1)
            if ind != -1:
                if ind not in ind_to_nodes:
                    ind_to_nodes[ind] = []
                ind_to_nodes[ind].append(idx)

        for nodes in ind_to_nodes.values():
            for i in nodes:
                for j in nodes:
                    if i != j:
                        edge_index.append([i, j])

        if len(edge_index) == 0:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
        else:
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

        return Data(x=x, y=y, edge_index=edge_index, symbols=stock_list)


class StructureTrainer:
    """
    Training manager for the Structure Expert GNN model.

    This class handles model training, optimization, and embedding extraction
    for visualization purposes.

    Attributes:
        device: PyTorch device (CPU or CUDA).
        model: Structure Expert GNN model.
        optimizer: Adam optimizer for training.
        criterion: MSE loss function.
    """

    def __init__(
        self, model: StructureExpertGNN, lr: float = 1e-3, device: str = "cuda"
    ) -> None:
        """
        Initialize the StructureTrainer.

        Args:
            model: Structure Expert GNN model instance.
            lr: Learning rate for optimizer (default: 1e-3).
            device: Device to use ('cuda' or 'cpu', default: 'cuda').
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=1e-5
        )
        self.criterion = nn.MSELoss()

    def train_step(self, daily_graph: Data) -> float:
        """
        Perform one training step.

        Args:
            daily_graph: PyTorch Geometric Data object for one day.

        Returns:
            Training loss value.
        """
        self.model.train()
        data = daily_graph.to(self.device)
        
        # Check for NaN in input data
        if torch.isnan(data.x).any():
            logger.warning("Input features contain NaN, skipping this step")
            return float('nan')
        
        self.optimizer.zero_grad()
        pred, embedding = self.model(data.x, data.edge_index)
        
        # Check for NaN in predictions
        if torch.isnan(pred).any() or torch.isinf(pred).any():
            logger.warning("Model predictions contain NaN/Inf, skipping this step")
            return float('nan')
        
        # Check if labels are available and have correct size
        if data.y is not None:
            y = data.y
            
            # Check for NaN in labels
            if torch.isnan(y).any() or torch.isinf(y).any():
                logger.warning("Labels contain NaN/Inf, skipping this step")
                return float('nan')
            
            # Ensure y has the same size as pred
            if y.shape[0] != pred.shape[0]:
                # Size mismatch, skip this step
                logger.warning(
                    f"Label size ({y.shape[0]}) doesn't match prediction size ({pred.shape[0]}), skipping"
                )
                return 0.0
            
            # Ensure y has the same shape as pred
            if y.shape != pred.shape:
                # Reshape y to match pred if needed
                if y.dim() == 1:
                    y = y.unsqueeze(1)
                elif y.shape[1] != pred.shape[1]:
                    y = y[:, :pred.shape[1]] if y.shape[1] > pred.shape[1] else y
            
            loss = self.criterion(pred, y)
        else:
            # No labels, use a dummy loss (e.g., L2 regularization on outputs)
            loss = torch.mean(pred**2)
        
        # Check for NaN/Inf in loss
        if torch.isnan(loss) or torch.isinf(loss):
            logger.warning("Loss is NaN/Inf, skipping this step")
            return float('nan')
        
        loss.backward()
        
        # Gradient clipping to prevent gradient explosion
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        # Check for NaN in gradients
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    logger.warning(f"Gradient for {name} contains NaN/Inf, skipping this step")
                    self.optimizer.zero_grad()  # Clear gradients
                    return float('nan')
        
        self.optimizer.step()
        return loss.item()

    def get_embeddings(self, daily_graph: Data) -> np.ndarray:
        """
        Extract embeddings for visualization.

        Args:
            daily_graph: PyTorch Geometric Data object for one day.

        Returns:
            Numpy array of embeddings with shape [num_nodes, embedding_dim].
        """
        self.model.eval()
        with torch.no_grad():
            data = daily_graph.to(self.device)
            _, embedding = self.model(data.x, data.edge_index)
        return embedding.cpu().numpy()


# Constants for demo
N_STOCKS = 100
N_FEATURES = 158  # Qlib Alpha158 dimension


def main() -> None:
    """
    Demo function to demonstrate model usage.

    This function creates mock data and runs a training example.
    """
    # 1. Mock industry mapping (in practice, get from AkShare or other data source)
    mock_ind_map = {f"SH600{i:03d}": i // 10 for i in range(N_STOCKS)}

    # 2. Mock daily Qlib data
    idx = pd.MultiIndex.from_product(
        [["2023-10-27"], mock_ind_map.keys()], names=["datetime", "instrument"]
    )
    mock_x = pd.DataFrame(np.random.randn(N_STOCKS, N_FEATURES), index=idx)
    mock_y = pd.DataFrame(np.random.randn(N_STOCKS, 1), index=idx)

    # 3. Initialize and convert
    builder = GraphDataBuilder(mock_ind_map)
    daily_data = builder.get_daily_graph(mock_x, mock_y)
    model = StructureExpertGNN(n_feat=N_FEATURES)
    trainer = StructureTrainer(model)

    # 4. Simulate training process
    print("Starting Training...")
    for epoch in range(10):
        loss = trainer.train_step(daily_data)
        if epoch % 2 == 0:
            print(f"Epoch {epoch} | Loss: {loss:.6f}")

    # 5. Extract embeddings for visualization
    emb = trainer.get_embeddings(daily_data)
    print(f"Extraction successful. Embedding shape: {emb.shape}")
    print("Next step: Use t-SNE to plot these embeddings.")


if __name__ == "__main__":
    main()

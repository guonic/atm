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
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv

# Add parent directory to path for utils import
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from tools.qlib.utils import clean_dataframe

logger = logging.getLogger(__name__)



class DirectionalStockGNN(torch.nn.Module):
    """
    Directional Stock Graph Neural Network model for stock prediction.

    This model uses Graph Attention Network (GAT) layers with edge attributes to
    capture directional relationships between stocks. It processes stock features
    through two GAT layers that incorporate edge attributes (correlation metrics)
    and outputs prediction scores for next period returns.

    The model is designed to handle directional relationships between stocks by
    incorporating edge attributes that represent correlation metrics between
    connected stocks.

    Attributes:
        conv1: First GAT layer that fuses node features with edge attributes.
        conv2: Second GAT layer for deep structural feature extraction.
        fc: Fully connected layer that outputs prediction scores.
    """

    def __init__(
        self, node_in_channels: int, edge_in_channels: int, hidden_channels: int
    ) -> None:
        """
        Initialize the DirectionalStockGNN model.

        Args:
            node_in_channels: Number of input features per stock node.
            edge_in_channels: Number of edge attributes (correlation metrics).
                Typically 4 for four correlation metrics.
            hidden_channels: Hidden dimension size for GAT layers.
        """
        super(DirectionalStockGNN, self).__init__()

        # edge_dim = 4 (four correlation metrics we calculated)
        self.conv1 = GATv2Conv(
            node_in_channels, hidden_channels, edge_dim=edge_in_channels
        )
        self.conv2 = GATv2Conv(
            hidden_channels, hidden_channels, edge_dim=edge_in_channels
        )

        if node_in_channels != hidden_channels:
            self.fc_shortcut = torch.nn.Linear(node_in_channels, hidden_channels)
        else:
            self.fc_shortcut = torch.nn.Identity()

        self.fc = torch.nn.Linear(hidden_channels, 1)  # Predict next period return

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            x: Node features tensor of shape [N, D] where N is the number of nodes
                (stocks) and D is the number of features per node (e.g., open, high,
                low, close prices and other technical indicators).
            edge_index: Edge index tensor of shape [2, E] where E is the number of
                edges. Each column represents a directed edge from source to target.
            edge_attr: Edge attributes tensor of shape [E, edge_in_channels]
                containing correlation metrics between connected stocks.

        Returns:
            Prediction tensor of shape [N, 1] containing predicted next period
            returns for each stock.
        """
        # Reserve same dimension
        identity = self.fc_shortcut(x)

        # First layer convolution: fuse node features with edge attributes
        x = self.conv1(x, edge_index, edge_attr=edge_attr)
        x = F.elu(x)

        # Second layer convolution: extract structural features deeply
        x = self.conv2(x, edge_index, edge_attr=edge_attr)
        x = F.elu(x)

        # Give more weight with
        return self.fc(x + identity)


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

        if n_feat != n_hidden:
            self.shortcut = nn.Linear(n_feat, n_hidden)
        else:
            self.shortcut = nn.Identity()

        # Prediction layer: outputs stock scores
        self.predict = nn.Sequential(
            nn.Linear(n_hidden, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 1),
        )

    # def forward(
    #     self, x: torch.Tensor, edge_index: torch.Tensor
    # ) -> Tuple[torch.Tensor, torch.Tensor]:
    #     """
    #     Forward pass through the model.
    #
    #     Args:
    #         x: Node features tensor of shape [num_nodes, num_features].
    #         edge_index: Edge index tensor of shape [2, num_edges].
    #
    #     Returns:
    #         Tuple of (logits, embedding):
    #             - logits: Prediction scores of shape [num_nodes, 1].
    #             - embedding: High-dimensional structure embedding of shape [num_nodes, n_hidden].
    #     """
    #     # x: [num_nodes, num_features], edge_index: [2, num_edges]
    #     h = F.leaky_relu(self.conv1(x, edge_index))
    #     h = self.conv2(h, edge_index)  # h is high-dimensional structure embedding
    #     logits = self.predict(h)
    #     return logits, h

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
        # 1. Preserve the original features for residual connection
        identity = self.shortcut(x)

        # 2. First layer of GAT convolution
        h = self.conv1(x, edge_index)
        h = F.leaky_relu(h)
        h = F.dropout(h, p=0.2, training=self.training)

        # 3. Second layer of GAT convolution
        h = self.conv2(h, edge_index)

        # 4. 【Core Change】Introduce residual connection
        # Add the structured feature h learned by GNN to the projected original feature identity
        # This way, even if GNN "muddles" the feature calculation,
        # the model still retains the characteristic individuality of the original individual stocks
        h_final = h + identity

        # 5. Output prediction
        logits = self.predict(h_final)

        return logits, h_final


class GraphDataBuilder:
    """
    Utility class for converting Qlib format data to PyTorch Geometric graph format.

    This class builds graph structures where stocks in the same industry are
    connected by edges, enabling the GNN to capture industry relationships.

    Attributes:
        industry_map: Dictionary mapping stock codes to industry codes (L3 level).
    """

    def __init__(self, industry_map: Dict[str, str]) -> None:
        """
        Initialize the GraphDataBuilder.

        Args:
            industry_map: Dictionary mapping stock codes to industry codes (L3 level).
                Format: {stock_code: l3_code}
        """
        self.industry_map = industry_map

    def get_daily_graph(
        self, df_x: pd.DataFrame, df_y: Optional[pd.DataFrame] = None, 
        include_edge_attr: bool = False
    ) -> Data:
        """
        Convert Qlib format daily cross-sectional DataFrame to PyG Data object.

        Args:
            df_x: Input features DataFrame with MultiIndex (datetime, instrument).
            df_y: Optional target values DataFrame with same index as df_x.
            include_edge_attr: If True, generate edge attributes for DirectionalStockGNN.
                Edge attributes are simple 4D vectors based on industry relationships.

        Returns:
            PyTorch Geometric Data object containing:
                - x: Node features tensor.
                - y: Optional target values tensor.
                - edge_index: Edge index tensor connecting stocks in same industry.
                - edge_attr: Optional edge attributes tensor (if include_edge_attr=True).
                - symbols: List of stock symbols.
        """
        # Clean NaN and Inf values before converting to tensor using unified function
        df_x = clean_dataframe(df_x, fill_value=0.0, log_stats=True, context="df_x in get_daily_graph")
        
        stock_list = df_x.index.get_level_values("instrument").tolist()
        x = torch.tensor(df_x.values, dtype=torch.float)
        
        # Check for NaN/Inf in tensor (should not happen after cleaning, but double check)
        if torch.isnan(x).any() or torch.isinf(x).any():
            logger.warning("Found NaN/Inf in tensor after conversion, replacing with 0")
            x = torch.where(torch.isnan(x) | torch.isinf(x), torch.zeros_like(x), x)
        
        y = None
        if df_y is not None:
            # Clean labels using unified function
            df_y = clean_dataframe(df_y, fill_value=0.0, log_stats=True, context="df_y in get_daily_graph")
            
            y = torch.tensor(df_y.values, dtype=torch.float)
            
            # Check for NaN/Inf in y tensor
            if torch.isnan(y).any() or torch.isinf(y).any():
                logger.warning("Found NaN/Inf in y tensor after conversion, replacing with 0")
                y = torch.where(torch.isnan(y) | torch.isinf(y), torch.zeros_like(y), y)

        # Build edges: connect stocks in the same industry pairwise
        edge_index = []
        # Pre-optimization: group nodes by industry code (L3 level)
        ind_to_nodes: Dict[str, List[int]] = {}
        for idx, symbol in enumerate(stock_list):
            ind_code = self.industry_map.get(symbol)
            # If stock is not in industry_map, skip it (won't have edges)
            # This allows the model to process all stocks, even if they don't have industry mapping
            if ind_code:
                if ind_code not in ind_to_nodes:
                    ind_to_nodes[ind_code] = []
                ind_to_nodes[ind_code].append(idx)
            # Note: Stocks without industry mapping are not added to ind_to_nodes,
            # so they won't have edges, but they will still be included in the graph

        edge_attr_list = []
        for nodes in ind_to_nodes.values():
            for i in nodes:
                for j in nodes:
                    if i != j:
                        edge_index.append([i, j])
                        # Generate simple edge attributes if requested
                        # For DirectionalStockGNN, we use 4D edge attributes
                        # [same_industry_flag, node_i_idx_normalized, node_j_idx_normalized, edge_count]
                        if include_edge_attr:
                            edge_attr = torch.tensor([
                                1.0,  # Same industry flag
                                float(i) / len(stock_list),  # Normalized source node index
                                float(j) / len(stock_list),  # Normalized target node index
                                1.0 / len(nodes),  # Inverse of industry size (edge density)
                            ], dtype=torch.float)
                            edge_attr_list.append(edge_attr)

        if len(edge_index) == 0:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = None
        else:
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            if include_edge_attr and len(edge_attr_list) > 0:
                edge_attr = torch.stack(edge_attr_list)
            else:
                edge_attr = None

        return Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr, symbols=stock_list)


class StructureTrainer:
    """
    Training manager for the Structure Expert GNN model.

    This class handles model training, optimization, and embedding extraction
    for visualization purposes. Supports both StructureExpertGNN and DirectionalStockGNN models.

    Attributes:
        device: PyTorch device (CPU or CUDA).
        model: Structure Expert GNN model (StructureExpertGNN or DirectionalStockGNN).
        optimizer: Adam optimizer for training.
        criterion: MSE loss function.
        use_edge_attr: Whether the model requires edge attributes.
    """

    def __init__(
        self, model: torch.nn.Module, lr: float = 1e-3, device: str = "cuda"
    ) -> None:
        """
        Initialize the StructureTrainer.

        Args:
            model: GNN model instance (StructureExpertGNN or DirectionalStockGNN).
            lr: Learning rate for optimizer (default: 1e-3).
            device: Device to use ('cuda' or 'cpu', default: 'cuda').
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=1e-5
        )
        self.criterion = nn.MSELoss()
        # Check if model requires edge attributes
        self.use_edge_attr = isinstance(model, DirectionalStockGNN)

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
        
        # Call model forward based on model type
        if self.use_edge_attr:
            # DirectionalStockGNN requires edge_attr
            if not hasattr(data, 'edge_attr') or data.edge_attr is None:
                logger.warning("Model requires edge_attr but data doesn't have it, skipping")
                return float('nan')
            pred = self.model(data.x, data.edge_index, data.edge_attr)
            # DirectionalStockGNN only returns predictions, not embeddings
            embedding = None
        else:
            # StructureExpertGNN doesn't require edge_attr
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
            For DirectionalStockGNN, returns predictions as embeddings.
        """
        self.model.eval()
        with torch.no_grad():
            data = daily_graph.to(self.device)
            if self.use_edge_attr:
                # DirectionalStockGNN only returns predictions
                if not hasattr(data, 'edge_attr') or data.edge_attr is None:
                    logger.warning("Model requires edge_attr but data doesn't have it")
                    return np.array([])
                pred = self.model(data.x, data.edge_index, data.edge_attr)
                return pred.cpu().numpy()
            else:
                # StructureExpertGNN returns (pred, embedding)
                _, embedding = self.model(data.x, data.edge_index)
                return embedding.cpu().numpy()


# Constants for demo
N_STOCKS = 100
N_FEATURES = 158  # Qlib Alpha158 dimension


def load_structure_expert_model(
    model_path: str,
    n_feat: int = 158,
    n_hidden: int = 128,
    n_heads: int = 8,
    edge_in_channels: int = 4,
    device: str = "cuda",
) -> torch.nn.Module:
    """
    Load trained GNN model from checkpoint file.

    This function automatically detects the model type (StructureExpertGNN or DirectionalStockGNN)
    from the state_dict and loads the appropriate model. It automatically detects model parameters
    from the state_dict if not provided.

    Args:
        model_path: Path to model checkpoint file (.pth).
        n_feat: Number of input features (default: 158 for Alpha158).
        n_hidden: Hidden layer size (default: 128). Will be auto-detected if None.
        n_heads: Number of attention heads for StructureExpertGNN (default: 8). Will be auto-detected if None.
        edge_in_channels: Number of edge attribute channels for DirectionalStockGNN (default: 4).
        device: Device to load model on ('cuda' or 'cpu', default: 'cuda').

    Returns:
        Loaded model (StructureExpertGNN or DirectionalStockGNN) in evaluation mode.

    Raises:
        FileNotFoundError: If model file does not exist.
        RuntimeError: If model parameters cannot be inferred or model loading fails.

    Examples:
        >>> model = load_structure_expert_model("models/structure_expert.pth")
        >>> model.eval()  # Already in eval mode
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    logger.info(f"Loading model from {model_path}")

    # Load state_dict first to infer parameters and model type
    device_obj = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
    state_dict = torch.load(model_path, map_location=device_obj)

    # Detect model type based on state_dict keys
    is_directional = "fc.weight" in state_dict and "conv1.lin_edge.weight" in state_dict
    is_structure_expert = "shortcut.weight" in state_dict or "predict.0.weight" in state_dict

    if is_directional:
        logger.info("Detected DirectionalStockGNN model")
        # Detect parameters for DirectionalStockGNN
        detected_n_feat = n_feat
        detected_n_hidden = n_hidden
        detected_edge_in_channels = edge_in_channels

        # Infer from conv1.lin_l (node features input)
        if "conv1.lin_l.weight" in state_dict:
            detected_n_feat = state_dict["conv1.lin_l.weight"].shape[1]
            # conv1.lin_l output is hidden_channels
            detected_n_hidden = state_dict["conv1.lin_l.weight"].shape[0]
            logger.info(f"Detected from conv1.lin_l: n_feat={detected_n_feat}, n_hidden={detected_n_hidden}")

        # Infer edge_in_channels from conv1.lin_edge
        if "conv1.lin_edge.weight" in state_dict:
            detected_edge_in_channels = state_dict["conv1.lin_edge.weight"].shape[1]
            logger.info(f"Detected edge_in_channels={detected_edge_in_channels} from conv1.lin_edge")

        # Use detected values
        n_feat = detected_n_feat
        n_hidden = detected_n_hidden
        edge_in_channels = detected_edge_in_channels

        # Initialize DirectionalStockGNN model
        model = DirectionalStockGNN(
            node_in_channels=n_feat,
            edge_in_channels=edge_in_channels,
            hidden_channels=n_hidden,
        )
        logger.info(
            f"Initialized DirectionalStockGNN: "
            f"node_in={n_feat}, edge_in={edge_in_channels}, hidden={n_hidden}"
        )

    elif is_structure_expert:
        logger.info("Detected StructureExpertGNN model")
        # Auto-detect parameters from state_dict
        detected_n_feat = n_feat
        detected_n_hidden = n_hidden
        detected_n_heads = n_heads

        # Try to infer n_feat from shortcut layer (if exists)
        if "shortcut.weight" in state_dict:
            detected_n_feat = state_dict["shortcut.weight"].shape[1]
            detected_n_hidden = state_dict["shortcut.weight"].shape[0]
            logger.info(f"Detected shortcut layer: n_feat={detected_n_feat}, n_hidden={detected_n_hidden}")
        # Try to infer from conv1 layer
        elif "conv1.lin_l.weight" in state_dict:
            # GATv2Conv: lin_l is [n_heads * n_hidden, n_feat]
            conv1_out_dim = state_dict["conv1.lin_l.weight"].shape[0]
            detected_n_feat = state_dict["conv1.lin_l.weight"].shape[1]
            # Try to infer n_heads from conv1.att shape
            if "conv1.att.weight" in state_dict:
                # conv1.att shape is [1, n_heads, n_hidden]
                att_shape = state_dict["conv1.att.weight"].shape
                if len(att_shape) == 3:
                    detected_n_heads = att_shape[1]
                    detected_n_hidden = att_shape[2]
                else:
                    # Fallback: assume single head
                    detected_n_heads = 1
                    detected_n_hidden = conv1_out_dim
            else:
                detected_n_heads = n_heads  # Keep provided value or default
                if detected_n_heads > 0:
                    detected_n_hidden = conv1_out_dim // detected_n_heads
            logger.info(
                f"Detected from conv1: n_feat={detected_n_feat}, "
                f"n_hidden={detected_n_hidden}, n_heads={detected_n_heads}"
            )
        # Try to infer from conv2 layer
        elif "conv2.lin_l.weight" in state_dict:
            # conv2 input is n_hidden * n_heads from conv1 output
            conv2_in_dim = state_dict["conv2.lin_l.weight"].shape[1]
            conv2_out_dim = state_dict["conv2.lin_l.weight"].shape[0]
            detected_n_hidden = conv2_out_dim
            if detected_n_heads > 0:
                # conv2 input should be n_hidden * n_heads
                detected_n_hidden = (
                    conv2_in_dim // detected_n_heads if detected_n_heads > 0 else conv2_out_dim
                )
            logger.info(f"Detected from conv2: n_hidden={detected_n_hidden}")

        # Use detected values
        n_feat = detected_n_feat
        n_hidden = detected_n_hidden
        n_heads = detected_n_heads

        # Initialize StructureExpertGNN model
        model = StructureExpertGNN(n_feat=n_feat, n_hidden=n_hidden, n_heads=n_heads)
        logger.info(
            f"Initialized StructureExpertGNN: "
            f"n_feat={n_feat}, n_hidden={n_hidden}, n_heads={n_heads}"
        )
    else:
        raise RuntimeError(
            "Could not determine model type from state_dict. "
            "Expected either DirectionalStockGNN (with fc.weight, conv1.lin_edge.weight) "
            "or StructureExpertGNN (with shortcut.weight or predict.0.weight)."
        )

    # Load weights
    try:
        model.load_state_dict(state_dict, strict=True)
        logger.info("Model loaded with strict=True (all keys matched)")
    except RuntimeError as e:
        if "Missing key(s)" in str(e) or "Unexpected key(s)" in str(e) or "size mismatch" in str(e):
            logger.warning(f"Model loading with strict=True failed: {e}")
            logger.info("Attempting to load with strict=False (allowing missing/extra keys)")
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            if missing_keys:
                logger.warning(f"Missing keys (not loaded): {missing_keys}")
            if unexpected_keys:
                logger.warning(f"Unexpected keys (ignored): {unexpected_keys}")
            logger.info("Model loaded with strict=False (some keys may be missing)")
        else:
            raise

    # Move to device and set to eval mode (inference only, no training)
    model = model.to(device_obj)
    model.eval()  # Set to evaluation mode - no gradient computation, no training

    model_type = "DirectionalStockGNN" if is_directional else "StructureExpertGNN"
    logger.info(f"{model_type} loaded successfully on {device_obj} (evaluation mode - inference only)")
    return model


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

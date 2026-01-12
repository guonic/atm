"""
Structure Expert buy model implementation.
"""

import logging

import pandas as pd
import torch

from tools.qlib.train.structure_expert import GraphDataBuilder, load_structure_expert_model
from ...interfaces.models import IBuyModel

logger = logging.getLogger(__name__)


class StructureExpertBuyModel(IBuyModel):
    """Structure Expert model as buy model.
    
    Uses Structure Expert GNN to generate stock rankings.
    """
    
    def __init__(
        self,
        model_path: str,
        builder: GraphDataBuilder,
        device: str = "cuda",
        n_feat: int = 158,
        n_hidden: int = 128,
        n_heads: int = 8,
    ):
        """
        Initialize Structure Expert buy model.
        
        Args:
            model_path: Path to trained model file.
            builder: GraphDataBuilder instance.
            device: Device to run inference on.
            n_feat: Number of input features.
            n_hidden: Hidden layer size.
            n_heads: Number of attention heads.
        """
        self.model_path = model_path
        self.builder = builder
        self.device = device
        self.n_feat = n_feat
        self.n_hidden = n_hidden
        self.n_heads = n_heads
        
        # Load model
        self.model = load_structure_expert_model(
            model_path=model_path,
            n_feat=n_feat,
            n_hidden=n_hidden,
            n_heads=n_heads,
            device=device,
        )
        self.device_obj = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        self.model.eval()  # Set to evaluation mode
        
        logger.info(f"Structure Expert buy model loaded from {model_path}")
    
    def generate_ranks(
        self,
        date: pd.Timestamp,
        market_data: pd.DataFrame,
        **kwargs
    ) -> pd.DataFrame:
        """
        Generate stock rankings for a specific date.
        
        Args:
            date: Current trading date.
            market_data: Market data DataFrame (from Qlib).
            **kwargs: Additional parameters.
        
        Returns:
            DataFrame with columns: ['symbol', 'score', 'rank']
            Sorted by score (descending).
        """
        try:
            from qlib.data import D
            from ...utils.feature_loader import load_features_for_date, get_qlib_data_range
            
            # Detect model type
            from tools.qlib.train.structure_expert import DirectionalStockGNN
            is_directional = isinstance(self.model, DirectionalStockGNN)
            
            # Get instruments from market_data
            if isinstance(market_data.index, pd.MultiIndex):
                instruments = market_data.index.get_level_values(0).unique().tolist()
            else:
                instruments = market_data.index.tolist()
            
            # Get data range
            data_start_date, _ = get_qlib_data_range()
            LOOKBACK_DAYS = 60
            
            # Load features for this date
            df_x = load_features_for_date(
                trade_date=date,
                lookback_days=LOOKBACK_DAYS,
                instruments=instruments,
                data_start_date=data_start_date,
            )
            
            if df_x is None or df_x.empty:
                logger.warning(f"No features loaded for {date.strftime('%Y-%m-%d')}")
                return pd.DataFrame(columns=['symbol', 'score', 'rank'])
            
            # Build graph
            daily_graph = self.builder.get_daily_graph(
                df_x,
                None,
                include_edge_attr=is_directional
            )
            
            if daily_graph.x.shape[0] == 0:
                logger.warning(f"No stocks in graph for {date.strftime('%Y-%m-%d')}")
                return pd.DataFrame(columns=['symbol', 'score', 'rank'])
            
            # Run inference
            with torch.no_grad():
                data = daily_graph.to(self.device_obj)
                if is_directional:
                    if not hasattr(data, 'edge_attr') or data.edge_attr is None:
                        logger.warning(f"No edge_attr for {date.strftime('%Y-%m-%d')}")
                        return pd.DataFrame(columns=['symbol', 'score', 'rank'])
                    pred = self.model(data.x, data.edge_index, edge_attr=data.edge_attr)
                    pred = pred.cpu().numpy().flatten()
                else:
                    pred, _ = self.model(data.x, data.edge_index)
                    pred = pred.cpu().numpy().flatten()
            
            # Get stock symbols
            if hasattr(daily_graph, "symbols"):
                symbols = daily_graph.symbols
            else:
                symbols = df_x.index.get_level_values("instrument").unique().tolist()
            
            # Normalize symbols to match Qlib format
            from nq.utils.data_normalize import normalize_stock_code
            symbols_normalized = [normalize_stock_code(s) for s in symbols]
            
            # Validate symbol count
            if len(symbols_normalized) != len(pred):
                logger.warning(
                    f"Symbol count ({len(symbols_normalized)}) != prediction count ({len(pred)}) "
                    f"for {date.strftime('%Y-%m-%d')}"
                )
                return pd.DataFrame(columns=['symbol', 'score', 'rank'])
            
            # Create rankings DataFrame
            ranks_df = pd.DataFrame({
                'symbol': symbols_normalized,
                'score': pred,
            })
            ranks_df = ranks_df.sort_values('score', ascending=False)
            ranks_df['rank'] = range(1, len(ranks_df) + 1)
            
            logger.debug(
                f"Generated rankings for {date.strftime('%Y-%m-%d')}: "
                f"{len(ranks_df)} stocks"
            )
            
            return ranks_df
            
        except Exception as e:
            logger.error(f"Failed to generate ranks for {date}: {e}", exc_info=True)
            return pd.DataFrame(columns=['symbol', 'score', 'rank'])

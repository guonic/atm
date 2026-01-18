"""
Structure Expert buy model implementation.
"""

import logging

import pandas as pd
import torch

from tools.qlib.train.structure_expert import GraphDataBuilder, load_structure_expert_model
from ...interfaces.models import IBuyModel
from ...utils.feature_loader import load_features_for_date, get_qlib_data_range

from ...utils.market_data import MarketDataFrame

# Detect model type
from tools.qlib.train.structure_expert import DirectionalStockGNN

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
            is_directional = isinstance(self.model, DirectionalStockGNN)
            
            # Get instruments from market_data using MarketDataFrame for unified access
            # CRITICAL: market_data should be normalized to MultiIndex(instrument, datetime) format
            # If it's not, we need to handle it properly

            mdf = MarketDataFrame(market_data)
            instruments = mdf.get_all_symbols()
            
            # Validate: Structure Expert model requires multiple stocks for ranking
            if len(instruments) == 0:
                logger.warning(f"No instruments found in market_data for {date.strftime('%Y-%m-%d')}")
                return pd.DataFrame(columns=['symbol', 'score', 'rank'])
            
            if len(instruments) == 1:
                logger.warning(
                    f"Only 1 instrument available: {instruments[0]}. "
                    f"Structure Expert model requires multiple stocks for meaningful ranking. "
                    f"Returning single stock with score=0."
                )
                # Return single stock ranking (with score=0 as placeholder)
                return pd.DataFrame({
                    'symbol': instruments,
                    'score': [0.0],
                    'rank': [1]
                })
            
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
            
            # Load historical data for edge attributes if needed
            returns_df = None
            highs_df = None
            lows_df = None
            if is_directional:
                try:
                    from tools.qlib.train.train_structure_expert import load_historical_data_for_correlation
                    from datetime import datetime as dt
                    
                    returns_df, highs_df, lows_df = load_historical_data_for_correlation(
                        stock_list=instruments,
                        current_date=dt.combine(date.date(), dt.min.time()),
                        window=self.builder.correlation_window if hasattr(self.builder, 'correlation_window') else 60,
                    )
                    
                    if returns_df is None:
                        logger.debug(f"Could not load historical data for {date.strftime('%Y-%m-%d')}, using simple edge attributes")
                except Exception as e:
                    logger.debug(f"Failed to load historical data for correlation: {e}")
            
            # Build graph
            daily_graph = self.builder.get_daily_graph(
                df_x,
                None,
                include_edge_attr=is_directional,
                returns_df=returns_df,
                highs_df=highs_df,
                lows_df=lows_df,
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

"""
ML Exit sell model implementation.
"""

from typing import TYPE_CHECKING
import pandas as pd
import logging

from qlib.data import D

from ...interfaces.models import ISellModel
from ...utils.data_validation import validate_single_instrument_data
from nq.analysis.exit import ExitModel
from ...utils.market_data import MarketDataFrame
from ...utils.data_normalizer import normalize_qlib_features_result

if TYPE_CHECKING:
    from ...state import Position

logger = logging.getLogger(__name__)


class MLExitSellModel(ISellModel):
    """ML Exit model as sell model.
    
    Uses trained exit model to predict exit probability.
    """
    
    def __init__(
        self,
        exit_model: ExitModel,
        threshold: float = 0.65,
    ):
        """
        Initialize ML Exit sell model.
        
        Args:
            exit_model: ExitModel instance.
            threshold: Risk probability threshold (default: 0.65).
        """
        self.exit_model = exit_model
        self._threshold = threshold
    
    @property
    def threshold(self) -> float:
        """Risk probability threshold for exit signal."""
        return self._threshold
    
    def predict_exit(
        self,
        position: 'Position',
        market_data: pd.DataFrame,
        date: pd.Timestamp,
        **kwargs
    ) -> float:
        """
        Predict exit probability for a position.
        
        Args:
            position: Position object.
            market_data: Market data DataFrame (from Qlib).
            date: Current trading date.
            **kwargs: Additional parameters.
        
        Returns:
            Risk probability (0-1). If > threshold, should exit.
        """
        try:
            # Get historical data (from Qlib, but not dependent on Qlib state)
            end_date = date.strftime("%Y-%m-%d")
            start_date = (date - pd.Timedelta(days=15)).strftime("%Y-%m-%d")

            raw_hist_data = D.features(
                instruments=[position.symbol],
                fields=["$close", "$high", "$low", "$volume"],
                start_time=start_date,
                end_time=end_date,
            )
            
            if raw_hist_data.empty:
                logger.warning(f"No historical data for {position.symbol} on {date} (hist_data is empty)")
                return 0.0
            
            # CRITICAL: Normalize Qlib's inconsistent format to unified format
            # Unified format: Index=MultiIndex(instrument, datetime), Columns=single-level fields
            hist_data = normalize_qlib_features_result(raw_hist_data, instruments=[position.symbol])
            
            # CRITICAL: Validate data for NaN before using it
            required_fields = ['$close', '$high', '$low', '$volume']
            if not validate_single_instrument_data(
                data=hist_data,
                required_fields=required_fields,
                symbol=position.symbol,
                date=date
            ):
                logger.warning(
                    f"Invalid data (NaN or missing fields) for {position.symbol} on {date}. "
                    f"Skipping exit prediction."
                )
                return 0.0
            
            # Extract OHLCV data using MarketDataFrame for unified access

            mdf = MarketDataFrame(hist_data)
            symbol_data = mdf.get_symbol_data(position.symbol)
            
            if symbol_data.empty:
                logger.warning(f"Empty symbol_data for {position.symbol} on {date}")
                return 0.0
            
            # Extract OHLCV data (now guaranteed to be single-level columns after normalization)
            try:
                daily_df = pd.DataFrame({
                    'close': symbol_data['$close'],
                    'high': symbol_data['$high'],
                    'low': symbol_data['$low'],
                    'volume': symbol_data['$volume'],
                })
                
                if daily_df.empty:
                    logger.warning(f"Empty daily_df for {position.symbol} on {date}")
                    return 0.0
                    
            except (KeyError, IndexError) as e:
                logger.warning(
                    f"Failed to extract data for {position.symbol} on {date}: {e}. "
                    f"Data shape: {hist_data.shape}, columns: {hist_data.columns.tolist()}"
                )
                return 0.0
            
            # Calculate days held
            days_held = (date - position.entry_date).days
            
            # Get current price
            current_price = float(daily_df['close'].iloc[-1])
            
            # Predict exit probability
            proba = self.exit_model.predict_proba(
                daily_df=daily_df,
                entry_price=position.entry_price,
                highest_price_since_entry=position.high_price_since_entry,
                days_held=days_held,
            )
            
            if len(proba) == 0:
                return 0.0
            
            latest_proba = proba[-1]
            
            return latest_proba
            
        except Exception as e:
            logger.warning(f"Failed to predict exit for {position.symbol} on {date}: {e}", exc_info=True)
            return 0.0

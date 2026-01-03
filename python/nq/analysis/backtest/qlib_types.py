"""
Standard data structures for Qlib backtest results.

This module defines standard Python dataclasses for all Qlib return types.
All functions have explicit type annotations. Invalid inputs will cause program to crash.
"""

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Optional, Union, Tuple, TYPE_CHECKING

import pandas as pd

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    # Forward reference to avoid circular import
    # PositionWrapper refers to Position class from backtest_structure_expert.py
    # which has _position_dict: Dict[str, Union[float, Dict[str, float]]]
    # and get_holdings() -> Dict[str, Tuple[float, float]] method
    PositionWrapper = None  # Type stub - actual type is defined in backtest_structure_expert.py


# Type alias for position data dict structure
# Keys can be "cash", "market_value" (float) or stock symbols (Dict[str, float] with "price" and "amount")
PositionDataDict = Dict[str, Union[float, Dict[str, float]]]

# Type alias for position data
PositionData = PositionDataDict


@dataclass
class PositionHolding:
    """Standard structure for a single stock holding in a position."""
    symbol: str
    price: float
    amount: float


class Position:
    """Standard structure for portfolio position at a specific date.
    
    Can wrap either:
    1. Our own Position dataclass (from dict)
    2. Qlib Position object (direct use)
    """
    def __init__(self, date_obj: date, pos_data):
        """
        Initialize Position from date and position data.
        
        Args:
            date_obj: Date for the position.
            pos_data: Either PositionDataDict (dict) or Qlib Position object.
        """
        self.date = date_obj
        # If pos_data is dict, extract fields
        # If pos_data is Qlib Position object, use it directly
        if isinstance(pos_data, dict):
            self._is_dict = True
            self._cash = float(pos_data["cash"])
            self._market_value = float(pos_data["market_value"])
            self._holdings = []
            for symbol, pos_info in pos_data.items():
                if symbol in ["cash", "market_value", "now_account_value"]:
                    continue
                self._holdings.append(PositionHolding(
                    symbol=symbol,
                    price=float(pos_info["price"]),
                    amount=float(pos_info["amount"]),
                ))
        else:
            # pos_data is Qlib Position object - use directly
            self._is_dict = False
            self._qlib_position = pos_data
    
    @property
    def cash(self) -> float:
        """Get cash amount."""
        if self._is_dict:
            return self._cash
        else:
            return float(self._qlib_position.get_cash())
    
    @property
    def market_value(self) -> float:
        """Get market value."""
        if self._is_dict:
            return self._market_value
        else:
            # Qlib Position doesn't have get_market_value() - calculate from cash + holdings
            cash = float(self._qlib_position.get_cash())
            holdings_value = 0.0
            qlib_holdings = self._qlib_position.position
            if isinstance(qlib_holdings, dict):
                for symbol, holding_info in qlib_holdings.items():
                    if isinstance(holding_info, dict):
                        price = float(holding_info.get("price", 0.0))
                        amount = float(holding_info.get("amount", 0.0))
                        holdings_value += price * amount
            return cash + holdings_value
    
    def get_holdings_dict(self) -> Dict[str, tuple]:
        """Get holdings as dict mapping symbol to (price, amount) tuple."""
        if self._is_dict:
            return {h.symbol: (h.price, h.amount) for h in self._holdings}
        else:
            # Extract from Qlib Position object
            holdings = {}
            qlib_holdings = self._qlib_position.position
            if isinstance(qlib_holdings, dict):
                for symbol, holding_info in qlib_holdings.items():
                    if isinstance(holding_info, dict):
                        price = float(holding_info.get("price", 0.0))
                        amount = float(holding_info.get("amount", 0.0))
                        if amount > 0:
                            holdings[symbol] = (price, amount)
            return holdings
    
    def get_stock_price(self, symbol: str) -> Optional[float]:
        """Get price for a specific stock."""
        holdings = self.get_holdings_dict()
        if symbol in holdings:
            return holdings[symbol][0]
        return None


@dataclass
class PositionDetails:
    """Standard structure for position details over time."""
    positions: Dict[date, Position] = field(default_factory=dict)
    
    def get_dates(self) -> List[date]:
        """Get sorted list of dates."""
        return sorted(self.positions.keys())
    
    def get_position(self, date_val: date) -> Optional[Position]:
        """Get Position for a specific date."""
        return self.positions.get(date_val)


@dataclass
class PortfolioMetrics:
    """Standard structure for portfolio metrics from Qlib backtest."""
    metric_df: pd.DataFrame
    position_details: Optional[PositionDetails] = None
    
    def has_data(self) -> bool:
        """Check if metrics data is available."""
        return not self.metric_df.empty


@dataclass
class IndicatorEntry:
    """Standard structure for a single indicator entry."""
    value: Union[int, float, str, pd.DataFrame, pd.Series]
    
    def get_value(self) -> Union[int, float, str, pd.DataFrame, pd.Series]:
        """Get the indicator value."""
        return self.value


@dataclass
class Indicator:
    """Standard structure for backtest indicators."""
    entries: Dict[str, IndicatorEntry] = field(default_factory=dict)
    
    def items(self):
        """Get items from indicator data."""
        return self.entries.items()
    
    def get(self, key: str) -> Optional[IndicatorEntry]:
        """Get indicator entry by key."""
        return self.entries.get(key)


@dataclass
class PortfolioMetricEntry:
    """Standard structure for a single portfolio metric entry."""
    metric_df: pd.DataFrame
    position_dict: Dict[str, PositionDataDict] = field(default_factory=dict)
    
    def has_position_data(self) -> bool:
        """Check if position data is available."""
        return len(self.position_dict) > 0


@dataclass
class QlibBacktestResult:
    """Standard structure for complete Qlib backtest result."""
    portfolio_metrics: PortfolioMetrics
    indicator: Indicator
    
    @classmethod
    def from_dataframe_and_dict(
        cls,
        metric_df: pd.DataFrame,
        position_details: Optional[PositionDetails],
        indicator_data: Dict[str, Union[int, float, str, pd.DataFrame, pd.Series]],
    ) -> "QlibBacktestResult":
        """
        Create QlibBacktestResult from explicit inputs.
        
        Args:
            metric_df: Portfolio metrics DataFrame.
            position_details: Position details (optional).
            indicator_data: Indicator data dict.
            
        Returns:
            QlibBacktestResult instance.
        """
        portfolio_metrics = PortfolioMetrics(
            metric_df=metric_df,
            position_details=position_details,
        )
        
        indicator_entries = {}
        for key, value in indicator_data.items():
            indicator_entries[key] = IndicatorEntry(value=value)
        indicator_obj = Indicator(entries=indicator_entries)
        
        return cls(
            portfolio_metrics=portfolio_metrics,
            indicator=indicator_obj,
        )
    
    @classmethod
    def _parse_qlib_portfolio_metric(
        cls,
        portfolio_metric_dict: Dict,
    ) -> Tuple[pd.DataFrame, Optional[Dict[str, PositionDataDict]]]:
        """
        Parse Qlib portfolio_metric dict to extract DataFrame and position dict.
        
        Args:
            portfolio_metric_dict: Raw dict from Qlib backtest().
            
        Returns:
            Tuple of (metric_df, position_dict).
        """
        logger.debug(f"Parsing Qlib portfolio_metric_dict: type={type(portfolio_metric_dict)}, keys={list(portfolio_metric_dict.keys()) if isinstance(portfolio_metric_dict, dict) else 'N/A'}")
        
        # Log structure of each key
        for key, value in portfolio_metric_dict.items():
            logger.debug(f"  portfolio_metric_dict['{key}']: type={type(value)}")
            if isinstance(value, tuple):
                logger.debug(f"    tuple length={len(value)}")
                for i, item in enumerate(value):
                    logger.debug(f"      [{i}]: type={type(item)}")
                    if isinstance(item, dict):
                        logger.debug(f"        dict keys={list(item.keys())[:5]}..." if len(item) > 5 else f"        dict keys={list(item.keys())}")
                        # Check first position data if available
                        if len(item) > 0:
                            first_key = list(item.keys())[0]
                            first_value = item[first_key]
                            logger.debug(f"        first key='{first_key}', value type={type(first_value)}")
                            if isinstance(first_value, dict):
                                logger.debug(f"          value keys={list(first_value.keys())}")
                            elif hasattr(first_value, '__dict__'):
                                logger.debug(f"          value has __dict__: {list(first_value.__dict__.keys())[:5]}")
            elif isinstance(value, pd.DataFrame):
                logger.debug(f"    DataFrame shape={value.shape}, columns={list(value.columns)}")
            elif isinstance(value, pd.Series):
                logger.debug(f"    Series length={len(value)}, dtype={value.dtype}")
        
        # Extract DataFrame and position dict from first entry
        metric_df = None
        position_dict = None
        
        for key, value in portfolio_metric_dict.items():
            if isinstance(value, tuple) and len(value) >= 1:
                # value is Tuple[pd.DataFrame, Dict[str, PositionDataDict]]
                metric_df = value[0]
                if len(value) >= 2:
                    position_dict = value[1]
                logger.debug(f"Extracted from '{key}': metric_df shape={metric_df.shape if metric_df is not None else None}, position_dict type={type(position_dict)}, position_dict keys count={len(position_dict) if isinstance(position_dict, dict) else 0}")
                break
        
        # If no tuple found, construct from individual Series entries
        if metric_df is None:
            logger.debug("No tuple found, constructing from individual Series entries")
            if "return" in portfolio_metric_dict:
                returns = portfolio_metric_dict["return"]
                if isinstance(returns, tuple):
                    returns = returns[0]
                metric_df = pd.DataFrame({"return": returns})
                for col in ["cash", "market_value", "deal_amount", "turnover_rate", "pos_count", "account"]:
                    if col in portfolio_metric_dict:
                        col_data = portfolio_metric_dict[col]
                        if isinstance(col_data, tuple):
                            col_data = col_data[0]
                        metric_df[col] = col_data
                logger.debug(f"Constructed metric_df: shape={metric_df.shape}, columns={list(metric_df.columns)}")
        
        # Validate position_dict structure and convert Qlib Position objects to dicts
        if position_dict is not None:
            logger.info(f"position_dict type={type(position_dict)}, length={len(position_dict) if isinstance(position_dict, dict) else 'N/A'}")
            if isinstance(position_dict, dict):
                # Check first few entries
                sample_keys = list(position_dict.keys())[:3]
                for sample_key in sample_keys:
                    sample_value = position_dict[sample_key]
                    logger.info(f"  position_dict['{sample_key}']: type={type(sample_value)}, class={sample_value.__class__.__name__ if hasattr(sample_value, '__class__') else 'N/A'}")
                    if isinstance(sample_value, dict):
                        logger.info(f"    dict keys={list(sample_value.keys())[:10]}")
                    elif hasattr(sample_value, '__dict__'):
                        logger.warning(f"    object has __dict__: {list(sample_value.__dict__.keys())[:10]}")
                        logger.warning(f"    This is NOT a dict! It's a {type(sample_value).__name__} object")
                        # Check if it's Qlib Position object
                        if hasattr(sample_value, 'position') and hasattr(sample_value, 'init_cash'):
                            logger.info(f"    Detected Qlib Position object - will convert to dict")
                
                # Keep Qlib Position objects as-is - Position class will wrap them directly
                # No conversion needed - Position class handles both dict and Qlib Position object
                logger.info(f"Keeping position data as-is (Position class will handle both dict and Qlib Position object)")
        
        return metric_df, position_dict
    
    @classmethod
    def _parse_qlib_indicator(
        cls,
        indicator_dict: Dict,
    ) -> Dict[str, IndicatorEntry]:
        """
        Parse Qlib indicator dict to extract indicator entries.
        
        Args:
            indicator_dict: Raw dict from Qlib backtest().
            
        Returns:
            Dict of indicator entries.
        """
        logger.debug(f"Parsing Qlib indicator_dict: type={type(indicator_dict)}, keys={list(indicator_dict.keys()) if isinstance(indicator_dict, dict) else 'N/A'}")
        
        # Log structure of each key
        for key, value in indicator_dict.items():
            logger.debug(f"  indicator_dict['{key}']: type={type(value)}")
            if isinstance(value, pd.DataFrame):
                logger.debug(f"    DataFrame shape={value.shape}, columns={list(value.columns)}")
            elif isinstance(value, pd.Series):
                logger.debug(f"    Series length={len(value)}, dtype={value.dtype}")
            elif isinstance(value, (int, float, str)):
                logger.debug(f"    value={value}")
        
        indicator_entries = {}
        for key, value in indicator_dict.items():
            indicator_entries[key] = IndicatorEntry(value=value)
        
        return indicator_entries
    
    @classmethod
    def from_qlib_dict_output(
        cls,
        portfolio_metric_dict: Dict,
        indicator_dict: Dict,
    ) -> "QlibBacktestResult":
        """
        Create QlibBacktestResult from Qlib dict output format.
        
        This method parses the raw Qlib output and converts it to standard structures.
        
        Args:
            portfolio_metric_dict: Raw dict from Qlib portfolio_metric.
            indicator_dict: Raw dict from Qlib indicator.
            
        Returns:
            QlibBacktestResult instance.
        """
        logger.info("Parsing Qlib backtest output...")
        
        # Parse portfolio_metric to extract DataFrame and position dict
        metric_df, position_dict = cls._parse_qlib_portfolio_metric(portfolio_metric_dict)
        
        # Extract position details from position_dict
        position_details = None
        if position_dict is not None:
            # position_dict values can be either PositionDataDict (dict) or Qlib Position object
            # Position class will handle both cases - no conversion needed
            positions = {}
            for date_str, pos_data in position_dict.items():
                date_obj = pd.to_datetime(date_str).date()
                # Position class handles both dict and Qlib Position object
                position = Position(date_obj, pos_data)
                positions[date_obj] = position
            position_details = PositionDetails(positions=positions)
            logger.info(f"Extracted {len(positions)} position records")
        
        # Parse indicator dict
        indicator_entries = cls._parse_qlib_indicator(indicator_dict)
        indicator_obj = Indicator(entries=indicator_entries)
        logger.info(f"Extracted {len(indicator_entries)} indicator entries")
        
        portfolio_metrics = PortfolioMetrics(
            metric_df=metric_df,
            position_details=position_details,
        )
        
        return cls(
            portfolio_metrics=portfolio_metrics,
            indicator=indicator_obj,
        )
    

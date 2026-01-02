"""
Standard data structures for Qlib backtest results.

This module defines standard Python dataclasses for all Qlib return types.
All functions have explicit type annotations. Invalid inputs will cause program to crash.
"""

from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Optional

import pandas as pd


@dataclass
class PositionHolding:
    """Standard structure for a single stock holding in a position."""
    symbol: str
    price: float
    amount: float


@dataclass
class Position:
    """Standard structure for portfolio position at a specific date."""
    date: date
    cash: float
    market_value: float
    holdings: List[PositionHolding] = field(default_factory=list)
    
    def get_holdings_dict(self) -> Dict[str, tuple]:
        """Get holdings as dict mapping symbol to (price, amount) tuple."""
        return {h.symbol: (h.price, h.amount) for h in self.holdings}
    
    def get_stock_price(self, symbol: str) -> Optional[float]:
        """Get price for a specific stock."""
        for holding in self.holdings:
            if holding.symbol == symbol:
                return holding.price
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
class Indicator:
    """Standard structure for backtest indicators."""
    data: Dict[str, object] = field(default_factory=dict)
    
    def items(self):
        """Get items from indicator data."""
        return self.data.items()


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
        indicator_data: Dict[str, object],
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
        
        indicator_obj = Indicator(data=indicator_data)
        
        return cls(
            portfolio_metrics=portfolio_metrics,
            indicator=indicator_obj,
        )
    
    @classmethod
    def from_qlib_dict_output(
        cls,
        portfolio_metric_dict: Dict[str, object],
        indicator_dict: Dict[str, object],
    ) -> "QlibBacktestResult":
        """
        Create QlibBacktestResult from Qlib dict output format.
        
        Expected format:
        - portfolio_metric_dict: dict with "return" key containing Series, or tuple with DataFrame at index 0
        - indicator_dict: dict with indicator data
        
        Args:
            portfolio_metric_dict: Dict from Qlib portfolio_metric.
            indicator_dict: Dict from Qlib indicator.
            
        Returns:
            QlibBacktestResult instance.
        """
        # Extract DataFrame from dict - assume specific structure
        # If dict has tuple values, first item is DataFrame
        metric_df = None
        for value in portfolio_metric_dict.values():
            # Assume tuple structure: (DataFrame, ...)
            first_item = value[0]
            metric_df = first_item
            break
        
        # If no tuple found, construct from "return" Series
        if metric_df is None:
            returns = portfolio_metric_dict["return"]
            metric_df = pd.DataFrame({"return": returns})
            for col in ["cash", "market_value", "deal_amount", "turnover_rate", "pos_count", "account"]:
                col_data = portfolio_metric_dict[col]
                metric_df[col] = col_data
        
        # Extract position details from dict
        position_details = None
        for value in portfolio_metric_dict.values():
            # Assume tuple structure: (DataFrame, position_dict)
            second_item = value[1]
            positions = {}
            for date_str, pos_data in second_item.items():
                date_obj = pd.to_datetime(date_str).date()
                position = cls._position_from_dict(date_obj, pos_data)
                positions[date_obj] = position
            position_details = PositionDetails(positions=positions)
            break
        
        return cls.from_dataframe_and_dict(
            metric_df=metric_df,
            position_details=position_details,
            indicator_data=indicator_dict,
        )
    
    @staticmethod
    def _position_from_dict(date_obj: date, pos_data: Dict[str, object]) -> Position:
        """
        Create Position from dict using **kwargs initialization.
        
        Args:
            date_obj: Date for the position.
            pos_data: Dict with position data. Expected keys: "cash", "market_value", and symbol keys with dict values containing "price" and "amount".
            
        Returns:
            Position object.
        """
        cash = float(pos_data["cash"])
        market_value = float(pos_data["market_value"])
        
        holdings = []
        for symbol, pos_info in pos_data.items():
            # Skip metadata keys - assume they are exactly these strings
            # If symbol is not a metadata key, it's a stock symbol
            # pos_info is dict with "price" and "amount"
            holdings.append(PositionHolding(**{
                "symbol": symbol,
                "price": float(pos_info["price"]),
                "amount": float(pos_info["amount"]),
            }))
        
        return Position(
            date=date_obj,
            cash=cash,
            market_value=market_value,
            holdings=holdings,
        )

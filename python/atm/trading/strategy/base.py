"""
Base strategy class for trading strategies.

Provides abstract interface for all trading strategies.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class StrategyConfig(BaseModel):
    """Base configuration for trading strategies."""

    name: str = Field(..., description="Strategy name")
    description: Optional[str] = Field(None, description="Strategy description")
    initial_cash: float = Field(100000.0, description="Initial cash amount")
    commission: float = Field(0.001, description="Commission rate (0.001 = 0.1%)")
    slippage: float = Field(0.0, description="Slippage rate")
    params: Dict[str, Any] = Field(default_factory=dict, description="Strategy parameters")


class BaseStrategy(ABC):
    """
    Base class for all trading strategies.

    All trading strategies should inherit from this class and implement
    the required abstract methods.
    """

    def __init__(self, config: StrategyConfig):
        """
        Initialize base strategy.

        Args:
            config: Strategy configuration.
        """
        self.config = config
        self.name = config.name
        self.description = config.description
        self.initial_cash = config.initial_cash
        self.commission = config.commission
        self.slippage = config.slippage
        self.params = config.params

    @abstractmethod
    def next(self, data: Any) -> None:
        """
        Called for each bar in the data feed.

        This is the main strategy logic that should be implemented by subclasses.

        Args:
            data: Current bar data.
        """
        pass

    @abstractmethod
    def start(self) -> None:
        """Called when the strategy starts running."""
        pass

    @abstractmethod
    def stop(self) -> None:
        """Called when the strategy stops running."""
        pass

    def notify_order(self, order: Any) -> None:
        """
        Called when an order status changes.

        Args:
            order: Order object.
        """
        pass

    def notify_trade(self, trade: Any) -> None:
        """
        Called when a trade is closed.

        Args:
            trade: Trade object.
        """
        pass

    def get_info(self) -> Dict[str, Any]:
        """
        Get strategy information.

        Returns:
            Dictionary containing strategy information.
        """
        return {
            "name": self.name,
            "description": self.description,
            "initial_cash": self.initial_cash,
            "commission": self.commission,
            "slippage": self.slippage,
            "params": self.params,
        }


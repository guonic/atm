"""
Strategy base classes and configuration.

Provides base class for backtrader strategies and configuration model.
"""

import logging
from datetime import date, datetime, time
from typing import Any, Dict, List, Optional

import backtrader as bt
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class StrategyConfig(BaseModel):
    """Base configuration for trading strategies."""

    name: str = Field(..., description="Strategy name")
    description: Optional[str] = Field(None, description="Strategy description")
    initial_cash: float = Field(100000.0, description="Initial cash amount")
    commission: float = Field(0.001, description="Commission rate (0.001 = 0.1%)")
    slippage: float = Field(0.0, description="Slippage rate")
    params: Dict[str, Any] = Field(default_factory=dict, description="Strategy parameters")


class BaseStrategy(bt.Strategy):
    """
    Base class for backtrader strategies.

    This class directly inherits from backtrader.Strategy and follows
    the standard backtrader pattern:
    - Indicators are initialized in __init__
    - Trading logic is in next() method
    - Indicators trigger buy/sell operations

    Subclasses should:
    1. Initialize indicators in __init__
    2. Implement trading logic in next() based on indicator states
    3. Optionally override notify_order() and notify_trade() for logging
    """

    params = (
        ("strategy_config", None),  # StrategyConfig instance
        ("run_id", None),  # Backtest run ID for signal tracking
        ("ts_code", None),  # Stock code for signal tracking
    )

    def __init__(self):
        """Initialize strategy."""
        super().__init__()
        self.config: Optional[StrategyConfig] = self.params.strategy_config

        # Store strategy info
        if self.config:
            self.strategy_name = self.config.name
            self.strategy_description = self.config.description
        else:
            self.strategy_name = self.__class__.__name__
            self.strategy_description = ""

        # Signal collection for backtest tracking
        # These are set via params by StrategyRunner
        self.run_id: Optional[str] = self.params.run_id
        self.ts_code: Optional[str] = self.params.ts_code
        self.signals: List[Dict[str, Any]] = []

        logger.info(f"Strategy {self.strategy_name} initialized")

    def start(self):
        """Called when strategy starts."""
        logger.info(f"Strategy {self.strategy_name} started")

    def next(self):
        """
        Called for each bar.

        Subclasses should override this method to implement trading logic
        based on indicator states.

        Example:
            if self.crossover[0] > 0 and not self.position:
                self.buy()
        """
        pass

    def stop(self):
        """Called when strategy stops."""
        logger.info(f"Strategy {self.strategy_name} stopped")

    def notify_order(self, order):
        """
        Called when an order status changes.

        Subclasses can override this for custom logging and signal collection.

        Args:
            order: Order object.
        """
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status in [order.Completed]:
            # Collect signal for backtest tracking
            if self.run_id and self.ts_code:
                signal_type = "buy" if order.isbuy() else "sell"
                
                # Get signal time from current bar
                try:
                    # Backtrader provides datetime through datetime.datetime(0) or datetime.date(0)
                    if hasattr(self.datas[0].datetime, 'datetime'):
                        dt = self.datas[0].datetime.datetime(0)
                        # Convert to Python datetime if needed
                        if isinstance(dt, datetime):
                            signal_time = dt
                        else:
                            # If it's a date object, combine with time
                            if isinstance(dt, date):
                                # Try to get time component
                                if hasattr(self.datas[0].datetime, 'time'):
                                    t = self.datas[0].datetime.time(0)
                                    signal_time = datetime.combine(dt, t)
                                else:
                                    signal_time = datetime.combine(dt, time.min)
                            else:
                                signal_time = datetime.now()
                    else:
                        signal_time = datetime.now()
                except Exception as e:
                    logger.debug(f"Failed to get signal time from bar: {e}, using current time")
                    signal_time = datetime.now()
                
                signal = {
                    "signal_time": signal_time,
                    "signal_type": signal_type,
                    "price": float(order.executed.price) if order.executed.price else None,
                    "size": float(order.executed.size) if order.executed.size else None,
                    "extra": {
                        "order_ref": order.ref,
                        "executed_value": float(order.executed.value) if order.executed.value else None,
                        "commission": float(order.executed.comm) if order.executed.comm else None,
                    }
                }
                self.signals.append(signal)

            if order.isbuy():
                logger.info(
                    f"BUY EXECUTED: Price={order.executed.price:.2f}, "
                    f"Size={order.executed.size}, Cost={order.executed.value:.2f}, "
                    f"Comm={order.executed.comm:.2f}"
                )
            else:
                logger.info(
                    f"SELL EXECUTED: Price={order.executed.price:.2f}, "
                    f"Size={order.executed.size}, Cost={order.executed.value:.2f}, "
                    f"Comm={order.executed.comm:.2f}"
                )
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            logger.warning(f"Order {order.ref} {order.getstatusname()}")

    def get_signals(self) -> List[Dict[str, Any]]:
        """
        Get collected signals for this strategy instance.

        Returns:
            List of signal dictionaries.
        """
        return self.signals.copy()

    def notify_trade(self, trade):
        """
        Called when a trade is closed.

        Subclasses can override this for custom logging.

        Args:
            trade: Trade object.
        """
        if not trade.isclosed:
            return

        # Calculate return percentage, avoiding division by zero
        if trade.value != 0:
            return_pct = trade.pnlcomm / trade.value * 100
        else:
            return_pct = 0.0

        logger.info(
            f"TRADE PROFIT: Gross={trade.pnl:.2f}, Net={trade.pnlcomm:.2f}, "
            f"Return={return_pct:.2f}%"
        )

    def get_info(self) -> Dict[str, Any]:
        """
        Get strategy information.

        Returns:
            Dictionary containing strategy information.
        """
        info = {
            "name": self.strategy_name,
            "description": self.strategy_description,
        }
        if self.config:
            info.update(
                {
                    "initial_cash": self.config.initial_cash,
                    "commission": self.config.commission,
                    "slippage": self.config.slippage,
                    "params": self.config.params,
                }
            )
        return info


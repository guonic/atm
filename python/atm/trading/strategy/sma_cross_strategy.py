"""
Simple Moving Average (SMA) Cross Strategy.

A simple example strategy that buys when short SMA crosses above long SMA
and sells when short SMA crosses below long SMA.
"""

import logging
from typing import Any, Dict, Optional

import backtrader as bt

from atm.trading.strategy.backtrader_strategy import BacktraderStrategy
from atm.trading.strategy.base import StrategyConfig

logger = logging.getLogger(__name__)


class SMACrossStrategy(BacktraderStrategy):
    """
    Simple Moving Average Cross Strategy.

    This strategy uses two moving averages:
    - Short SMA: Fast moving average
    - Long SMA: Slow moving average

    Buy signal: When short SMA crosses above long SMA (golden cross)
    Sell signal: When short SMA crosses below long SMA (death cross)
    """

    def __init__(self, config: StrategyConfig):
        """
        Initialize SMA Cross Strategy.

        Args:
            config: Strategy configuration. Should include:
                - params['short_period']: Short SMA period (default: 5)
                - params['long_period']: Long SMA period (default: 20)
        """
        super().__init__(config)

        # Get parameters
        self.short_period = self.params.get("short_period", 5)
        self.long_period = self.params.get("long_period", 20)

        # Indicators will be initialized in start()
        self.short_sma: Optional[bt.indicators.SMA] = None
        self.long_sma: Optional[bt.indicators.SMA] = None
        self.crossover: Optional[bt.indicators.CrossOver] = None

    def start(self) -> None:
        """Called when the strategy starts running."""
        super().start()
        logger.info(
            f"SMA Cross Strategy started with short_period={self.short_period}, "
            f"long_period={self.long_period}"
        )

        # Initialize indicators
        # Note: _datas is set by BacktraderStrategyWrapper in start() method
        if hasattr(self, "_datas") and self._datas and len(self._datas) > 0:
            data = self._datas[0]
            self.short_sma = bt.indicators.SMA(data.close, period=self.short_period)
            self.long_sma = bt.indicators.SMA(data.close, period=self.long_period)
            self.crossover = bt.indicators.CrossOver(self.short_sma, self.long_sma)

    def next(self, data: Any) -> None:
        """
        Called for each bar in the data feed.

        Args:
            data: Current bar data from backtrader.
        """
        if self.crossover is None:
            return

        # Check if we have a position
        # Note: _broker is set by BacktraderStrategyWrapper in start() method
        if not hasattr(self, "_broker") or self._broker is None:
            return
        position = self._broker.getposition(data).size

        # Golden cross: short SMA crosses above long SMA -> Buy
        if self.crossover[0] > 0 and position == 0:
            logger.debug(f"Golden cross detected at {data.datetime.date(0)}, buying")
            self.buy(data=data)

        # Death cross: short SMA crosses below long SMA -> Sell
        elif self.crossover[0] < 0 and position > 0:
            logger.debug(f"Death cross detected at {data.datetime.date(0)}, selling")
            self.sell(data=data)

    def notify_order(self, order: Any) -> None:
        """
        Called when an order status changes.

        Args:
            order: Order object.
        """
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status in [order.Completed]:
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

    def notify_trade(self, trade: Any) -> None:
        """
        Called when a trade is closed.

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
        info = super().get_info()
        info.update(
            {
                "short_period": self.short_period,
                "long_period": self.long_period,
            }
        )
        return info


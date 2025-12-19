"""
Simple Moving Average (SMA) Cross Strategy.

A simple example strategy that buys when short SMA crosses above long SMA
and sells when short SMA crosses below long SMA.

This strategy follows the standard backtrader pattern:
- Indicators are initialized in __init__
- Trading logic is in next() method
- Indicators trigger buy/sell operations
"""

import logging
from typing import Dict

import backtrader as bt

from atm.trading.strategy.base import BaseStrategy

logger = logging.getLogger(__name__)


class SMACrossStrategy(BaseStrategy):
    """
    Simple Moving Average Cross Strategy.

    This strategy uses two moving averages:
    - Short SMA: Fast moving average
    - Long SMA: Slow moving average

    Buy signal: When short SMA crosses above long SMA (golden cross)
    Sell signal: When short SMA crosses below long SMA (death cross)

    This strategy follows backtrader's standard pattern:
    1. Indicators are initialized in __init__
    2. Trading logic is in next() method
    3. Indicators trigger buy/sell operations
    """
    alias = ('SMA_CrossOver',)

    params = (
        ("short_period", 5),  # Short SMA period
        ("long_period", 20),  # Long SMA period
        ("_movav", bt.indicators.MovAv.SMA) # moving average to use
    )

    def __init__(self):
        """
        Initialize SMA Cross Strategy.

        Indicators are initialized here following backtrader's standard pattern.
        """
        super().__init__()

        # Initialize indicators in __init__ (backtrader standard pattern)
        # self.datas[0] is the main data feed
        short_sma = self.p._movav(period=self.p.short_period)
        long_sma = self.p._movav(period=self.p.long_period)
        self.crossover = bt.indicators.CrossOver(short_sma, long_sma)

        logger.info(
            f"SMA Cross Strategy initialized with short_period={self.p.short_period}, "
            f"long_period={self.p.long_period}"
        )

    def next(self):
        """
        Called for each bar.

        Trading logic based on indicator states:
        - Golden cross (short SMA crosses above long SMA) -> Buy
        - Death cross (short SMA crosses below long SMA) -> Sell
        """
        # Check if we have a position
        # In backtrader, self.position gives the current position size
        if not self.position:
            # No position: look for buy signal
            # crossover[0] > 0 means short SMA just crossed above long SMA
            if self.crossover > 0:
                logger.debug(f"Golden cross detected at {self.datas[0].datetime.date(0)}, buying")
                self.buy()  # Buy at market price
        else:
            # Have position: look for sale signal
            # crossover[0] < 0 means short SMA just crossed below long SMA
            if self.crossover < 0:
                logger.debug(f"Death cross detected at {self.datas[0].datetime.date(0)}, selling")
                self.sell()  # Sell at market price

    def get_info(self) -> Dict:
        """
        Get strategy information.

        Returns:
            Dictionary containing strategy information.
        """
        info = super().get_info()
        info.update(
            {
                "short_period": self.p.short_period,
                "long_period": self.p.long_period,
            }
        )
        return info


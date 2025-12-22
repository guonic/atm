"""
CCI (Commodity Channel Index) Strategy.

A trading strategy based on CCI indicator that:
- Buys when CCI crosses above oversold level or crosses above zero
- Sells when CCI crosses below overbought level or crosses below zero
- Detects top and bottom divergences for additional signals

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


class CCIStrategy(BaseStrategy):
    """
    CCI (Commodity Channel Index) Strategy.

    This strategy uses CCI indicator to generate trading signals:
    - Buy signals:
      1. CCI crosses above oversold level (-100)
      2. CCI crosses above zero line
      3. Bottom divergence detected (price lower but CCI higher)
    - Sell signals:
      1. CCI crosses below overbought level (100)
      2. CCI crosses below zero line
      3. Top divergence detected (price higher but CCI lower)

    This strategy follows backtrader's standard pattern:
    1. Indicators are initialized in __init__
    2. Trading logic is in next() method
    3. Indicators trigger buy/sell operations

    Args:
        cci_period: CCI calculation period (default: 20).
        overbought: Overbought threshold (default: 100).
        oversold: Oversold threshold (default: -100).
        divergence_lookback: Lookback period for divergence detection (default: 20).
    """

    params = (
        ("cci_period", 20),  # CCI period
        ("overbought", 100),  # Overbought threshold
        ("oversold", -100),  # Oversold threshold
        ("divergence_lookback", 20),  # Divergence lookback period
    )

    def __init__(self):
        """
        Initialize CCI Strategy.

        Indicators are initialized here following backtrader's standard pattern.
        """
        super().__init__()

        # Initialize CCI indicator
        self.cci = bt.indicators.CCI(self.data, period=self.p.cci_period)

        # Track position state
        self.in_position = False
        self.sell_partial = False

        logger.info(
            f"CCI Strategy initialized with cci_period={self.p.cci_period}, "
            f"overbought={self.p.overbought}, oversold={self.p.oversold}, "
            f"divergence_lookback={self.p.divergence_lookback}"
        )

    def next(self):
        """
        Called for each bar.

        Trading logic based on CCI indicator states:
        - Buy when CCI crosses above oversold or zero
        - Sell when CCI crosses below overbought or zero
        - Check for divergences for additional signals
        """
        current_cci = self.cci[0]
        prev_cci = self.cci[-1]
        current_close = self.data.close[0]
        prev_close = self.data.close[-1]

        # Buy signals
        # Signal 1: CCI crosses above oversold level
        if (prev_cci <= self.p.oversold) and (current_cci > self.p.oversold):
            if not self.in_position:
                logger.debug(
                    f"CCI crossed above oversold at {self.datas[0].datetime.date(0)}, "
                    f"buying 50% position"
                )
                self.buy(size=0.5)

        # Signal 2: CCI crosses above zero line
        if (prev_cci <= 0) and (current_cci > 0):
            if not self.in_position:
                logger.debug(
                    f"CCI crossed above zero at {self.datas[0].datetime.date(0)}, "
                    f"buying full position"
                )
                self.buy(size=1)
                self.in_position = True
            elif not self.sell_partial:
                logger.debug(
                    f"CCI crossed above zero at {self.datas[0].datetime.date(0)}, "
                    f"adding 50% position"
                )
                self.buy(size=0.5)

        # Signal 3: Bottom divergence detected
        if self.check_bottom_divergence():
            if not self.in_position:
                logger.debug(
                    f"Bottom divergence detected at {self.datas[0].datetime.date(0)}, "
                    f"buying full position"
                )
                self.buy(size=1)
                self.in_position = True

        # Sell signals
        # Signal 1: CCI crosses below overbought level
        if (prev_cci >= self.p.overbought) and (current_cci < self.p.overbought):
            if self.in_position and not self.sell_partial:
                logger.debug(
                    f"CCI crossed below overbought at {self.datas[0].datetime.date(0)}, "
                    f"selling 50% position"
                )
                self.sell(size=0.5)
                self.sell_partial = True

        # Signal 2: CCI crosses below zero line
        if (prev_cci >= 0) and (current_cci < 0):
            if self.in_position:
                logger.debug(
                    f"CCI crossed below zero at {self.datas[0].datetime.date(0)}, "
                    f"selling full position"
                )
                self.sell(size=self.position.size)
                self.in_position = False
                self.sell_partial = False

        # Signal 3: Top divergence detected
        if self.check_top_divergence():
            if self.in_position:
                logger.debug(
                    f"Top divergence detected at {self.datas[0].datetime.date(0)}, "
                    f"selling full position"
                )
                self.sell(size=self.position.size)
                self.in_position = False
                self.sell_partial = False

    def check_top_divergence(self) -> bool:
        """
        Check for top divergence (bearish signal).

        Top divergence occurs when:
        - Price makes a higher high
        - CCI makes a lower high

        Returns:
            True if top divergence is detected, False otherwise.
        """
        if len(self.data) < self.p.divergence_lookback:
            return False

        # Get price high over lookback period
        # Use get() with ago parameter to get historical values
        price_values = self.data.close.get(
            ago=self.p.divergence_lookback - 1, size=self.p.divergence_lookback
        )
        if not price_values:
            return False
        price_high = max(price_values)
        current_price_high = self.data.close[0]
        is_price_higher = current_price_high >= price_high

        # Get CCI high over lookback period
        cci_values = self.cci.get(
            ago=self.p.divergence_lookback - 1, size=self.p.divergence_lookback
        )
        if not cci_values:
            return False
        cci_high = max(cci_values)
        current_cci_high = self.cci[0]
        is_cci_lower = current_cci_high < cci_high

        return is_price_higher and is_cci_lower

    def check_bottom_divergence(self) -> bool:
        """
        Check for bottom divergence (bullish signal).

        Bottom divergence occurs when:
        - Price makes a lower low
        - CCI makes a higher low

        Returns:
            True if bottom divergence is detected, False otherwise.
        """
        if len(self.data) < self.p.divergence_lookback:
            return False

        # Get price low over lookback period
        # Use get() with ago parameter to get historical values
        price_values = self.data.close.get(
            ago=self.p.divergence_lookback - 1, size=self.p.divergence_lookback
        )
        if not price_values:
            return False
        price_low = min(price_values)
        current_price_low = self.data.close[0]
        is_price_lower = current_price_low <= price_low

        # Get CCI low over lookback period
        cci_values = self.cci.get(
            ago=self.p.divergence_lookback - 1, size=self.p.divergence_lookback
        )
        if not cci_values:
            return False
        cci_low = min(cci_values)
        current_cci_low = self.cci[0]
        is_cci_higher = current_cci_low > cci_low

        return is_price_lower and is_cci_higher

    def get_info(self) -> Dict:
        """
        Get strategy information.

        Returns:
            Dictionary containing strategy information.
        """
        info = super().get_info()
        info.update(
            {
                "cci_period": self.p.cci_period,
                "overbought": self.p.overbought,
                "oversold": self.p.oversold,
                "divergence_lookback": self.p.divergence_lookback,
            }
        )
        return info

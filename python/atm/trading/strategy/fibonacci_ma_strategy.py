"""
Fibonacci Moving Average Trading Strategy.

This strategy uses Fibonacci sequence numbers (13, 21, 34, 55, 89) as moving average periods
instead of traditional periods (5, 10, 20). The strategy implements:

1. Core Fibonacci MAs: 13, 21, 34-day moving averages
2. Golden Cross and Death Cross detection
3. Multi-period resonance (all MAs aligned)
4. Volume confirmation
5. Slope-based trend identification
6. Different parameter sets for different stock types

This strategy follows the standard backtrader pattern:
- Indicators are initialized in __init__
- Trading logic is in next() method
- Indicators trigger buy/sell operations
"""

import logging
from typing import Dict, Optional, Tuple

import backtrader as bt

from atm.trading.strategy.base import BaseStrategy

logger = logging.getLogger(__name__)


class FibonacciMAStrategy(BaseStrategy):
    """
    Fibonacci Moving Average Trading Strategy.

    This strategy uses Fibonacci sequence numbers for moving average periods:
    - 13-day MA: Short-term trend lifeline
    - 21-day MA: Swing trading navigator
    - 34-day MA: Mid-term trend watershed

    Buy signals:
    1. Golden cross: 13-day MA crosses above 21-day MA (both MAs sloping up)
    2. Multi-period resonance: 13 > 21 > 34, all MAs sloping up
    3. Price above 13-day MA with upward slope and volume confirmation

    Sell signals:
    1. Death cross: 13-day MA crosses below 21-day MA
    2. Price below 13-day MA with downward slope
    3. Multi-period death cross: 13 < 21 < 34, all MAs sloping down

    Args:
        ma13_period: 13-day moving average period (default: 13).
        ma21_period: 21-day moving average period (default: 21).
        ma34_period: 34-day moving average period (default: 34).
        use_ma55: Enable 55-day MA for large-cap stocks (default: False).
        use_ma89: Enable 89-day MA for cyclical stocks (default: False).
        volume_ma_period: Volume moving average period (default: 20).
        volume_threshold: Volume must be above this multiple of average (default: 1.2 for buy, 1.5 for breakout).
        use_volume_confirmation: Enable volume confirmation (default: True).
        use_multi_resonance: Enable multi-period resonance (default: True).
        stock_type: Stock type - 'small_cap', 'large_cap', or 'cyclical' (default: 'small_cap').
    """

    params = (
        ("ma13_period", 13),  # 13-day MA period
        ("ma21_period", 21),  # 21-day MA period
        ("ma34_period", 34),  # 34-day MA period
        ("use_ma55", False),  # Enable 55-day MA
        ("use_ma89", False),  # Enable 89-day MA
        ("volume_ma_period", 20),  # Volume MA period
        ("volume_threshold_buy", 1.2),  # Volume threshold for buy signals
        ("volume_threshold_breakout", 1.5),  # Volume threshold for breakout
        ("use_volume_confirmation", True),  # Enable volume confirmation
        ("use_multi_resonance", True),  # Enable multi-period resonance
        ("stock_type", "small_cap"),  # Stock type: small_cap, large_cap, cyclical
    )

    def __init__(self):
        """
        Initialize Fibonacci MA Strategy.

        Indicators are initialized here following backtrader's standard pattern.
        """
        super().__init__()

        # Adjust parameters based on stock type
        # For large_cap: use 21, 34, 55
        # For cyclical: use 13, 34, 89
        # For small_cap: use 13, 21, 34 (default)
        ma13_period = self.p.ma13_period
        ma21_period = self.p.ma21_period
        ma34_period = self.p.ma34_period
        
        if self.p.stock_type == "large_cap":
            ma13_period = 21
            ma21_period = 34
            ma34_period = 55
        elif self.p.stock_type == "cyclical":
            ma13_period = 13
            ma21_period = 34
            ma34_period = 89

        # Store actual periods used
        self.actual_ma13_period = ma13_period
        self.actual_ma21_period = ma21_period
        self.actual_ma34_period = ma34_period

        # Core Fibonacci moving averages
        # Note: Indicators will be created even if data is insufficient
        # We'll check data availability in next() method
        self.ma13 = bt.indicators.SMA(self.data.close, period=ma13_period)
        self.ma21 = bt.indicators.SMA(self.data.close, period=ma21_period)
        self.ma34 = bt.indicators.SMA(self.data.close, period=ma34_period)
        
        # Optional longer-term MAs
        self.ma55 = None
        if self.p.use_ma55:
            # Only create if we have enough data
            try:
                self.ma55 = bt.indicators.SMA(self.data.close, period=55)
            except Exception as e:
                logger.warning(f"Failed to create MA55 indicator: {e}")
                self.ma55 = None

        self.ma89 = None
        if self.p.use_ma89:
            # Only create if we have enough data
            try:
                self.ma89 = bt.indicators.SMA(self.data.close, period=89)
            except Exception as e:
                logger.warning(f"Failed to create MA89 indicator: {e}")
                self.ma89 = None

        # Crossovers for golden cross and death cross detection
        self.cross_13_21 = bt.indicators.CrossOver(self.ma13, self.ma21)
        self.cross_21_34 = bt.indicators.CrossOver(self.ma21, self.ma34)

        # Volume confirmation
        if self.p.use_volume_confirmation:
            self.volume_ma = bt.indicators.SMA(self.data.volume, period=self.p.volume_ma_period)
            self.volume_ratio = self.data.volume / self.volume_ma

        # Track position state
        self.entry_price = None
        self.position_reduced = False  # Track if position was reduced

        logger.info(
            f"Fibonacci MA Strategy initialized with "
            f"MA13={self.actual_ma13_period}, MA21={self.actual_ma21_period}, MA34={self.actual_ma34_period}, "
            f"stock_type={self.p.stock_type}, "
            f"multi_resonance={self.p.use_multi_resonance}"
        )

    def next(self):
        """
        Called for each bar.

        Trading logic based on Fibonacci moving averages:
        - Golden cross and death cross detection
        - Multi-period resonance
        - Volume confirmation
        - Slope-based trend identification
        """
        # Skip if not enough data
        # Need at least max_period + some buffer for indicators to stabilize
        max_period = max(self.actual_ma13_period, self.actual_ma21_period, self.actual_ma34_period)
        if self.ma55 is not None:
            max_period = max(max_period, 55)
        if self.ma89 is not None:
            max_period = max(max_period, 89)
        
        # Add buffer for indicator calculation
        min_required = max_period + 5
        
        if len(self.data) < min_required:
            return
        
        # Check if indicators have valid values
        try:
            # Try to access indicator values to ensure they're ready
            _ = self.ma13[0]
            _ = self.ma21[0]
            _ = self.ma34[0]
        except (IndexError, TypeError):
            # Indicators not ready yet
            return

        # Safely access indicator values
        try:
            current_price = self.data.close[0]
            ma13_val = self.ma13[0]
            ma21_val = self.ma21[0]
            ma34_val = self.ma34[0]
        except (IndexError, TypeError) as e:
            logger.debug(f"Indicators not ready yet: {e}")
            return

        # Calculate MA slopes (current - previous)
        try:
            ma13_slope = self._calculate_slope(self.ma13)
            ma21_slope = self._calculate_slope(self.ma21)
            ma34_slope = self._calculate_slope(self.ma34)
        except (IndexError, TypeError) as e:
            logger.debug(f"Failed to calculate slopes: {e}")
            return

        # Check volume confirmation
        volume_confirmed = True
        if self.p.use_volume_confirmation:
            try:
                if hasattr(self, "volume_ratio"):
                    volume_ratio = self.volume_ratio[0]
                    volume_confirmed = volume_ratio >= self.p.volume_threshold_buy
            except (IndexError, TypeError):
                # Volume indicator not ready yet
                volume_confirmed = False

        # Buy signals
        if not self.position:
            # Signal 1: Golden cross (13 crosses above 21)
            try:
                golden_cross = self.cross_13_21[0] > 0
            except (IndexError, TypeError):
                golden_cross = False
            golden_cross_valid = (
                golden_cross
                and ma13_slope > 0
                and ma21_slope > 0
                and current_price > ma13_val
            )

            # Signal 2: Multi-period resonance (bullish)
            multi_resonance_bullish = False
            if self.p.use_multi_resonance:
                multi_resonance_bullish = (
                    ma13_val > ma21_val
                    and ma21_val > ma34_val
                    and ma13_slope > 0
                    and ma21_slope > 0
                    and ma34_slope > 0
                    and current_price > ma13_val
                )

            # Signal 3: Price above 13-day MA with upward slope
            price_above_ma13 = (
                current_price > ma13_val
                and ma13_slope > 0
                and current_price > self.data.close[-1]  # Price trending up
            )

            # Entry conditions
            if (golden_cross_valid or multi_resonance_bullish or price_above_ma13) and volume_confirmed:
                logger.debug(
                    f"Buy signal at {self.datas[0].datetime.date(0)}, "
                    f"price={current_price:.2f}, "
                    f"MA13={ma13_val:.2f}, MA21={ma21_val:.2f}, MA34={ma34_val:.2f}, "
                    f"golden_cross={golden_cross_valid}, "
                    f"multi_resonance={multi_resonance_bullish}"
                )
                self.buy()
                self.entry_price = current_price
                self.position_reduced = False

        # Sell signals
        if self.position:
            # Signal 1: Death cross (13 crosses below 21)
            try:
                death_cross = self.cross_13_21[0] < 0
            except (IndexError, TypeError):
                death_cross = False
            death_cross_valid = death_cross or (
                ma13_val < ma21_val
                and ma13_slope < 0
                and ma21_slope < 0
            )

            # Signal 2: Multi-period death cross
            multi_resonance_bearish = False
            if self.p.use_multi_resonance:
                multi_resonance_bearish = (
                    ma13_val < ma21_val
                    and ma21_val < ma34_val
                    and ma13_slope < 0
                    and ma21_slope < 0
                    and ma34_slope < 0
                )

            # Signal 3: Price below 13-day MA with downward slope
            price_below_ma13 = (
                current_price < ma13_val
                and ma13_slope < 0
            )

            # Signal 4: Price below 21-day MA (reduce position by 50%)
            price_below_ma21 = (
                current_price < ma21_val
                and ma21_slope < 0
            )

            # Exit conditions
            if death_cross_valid or multi_resonance_bearish:
                logger.debug(
                    f"Sell signal (death cross/multi-resonance) at "
                    f"{self.datas[0].datetime.date(0)}, price={current_price:.2f}"
                )
                self.close()
                self.entry_price = None
                self.position_reduced = False
            elif price_below_ma21 and not self.position_reduced:
                # Reduce position by 50% when price breaks below 21-day MA
                logger.debug(
                    f"Reduce position signal (price below MA21) at "
                    f"{self.datas[0].datetime.date(0)}, price={current_price:.2f}"
                )
                self.sell(size=self.position.size * 0.5)
                self.position_reduced = True
            elif price_below_ma13:
                # Full exit when price breaks below 13-day MA
                logger.debug(
                    f"Sell signal (price below MA13) at "
                    f"{self.datas[0].datetime.date(0)}, price={current_price:.2f}"
                )
                self.close()
                self.entry_price = None
                self.position_reduced = False

    def _calculate_slope(self, ma_indicator) -> float:
        """
        Calculate moving average slope.

        Args:
            ma_indicator: Moving average indicator.

        Returns:
            Slope value (positive = upward, negative = downward).
        """
        if len(ma_indicator) < 2:
            return 0.0
        return ma_indicator[0] - ma_indicator[-1]

    def get_info(self) -> Dict:
        """
        Get strategy information.

        Returns:
            Dictionary containing strategy information.
        """
        info = super().get_info()
        info.update(
            {
                "ma13_period": self.actual_ma13_period,
                "ma21_period": self.actual_ma21_period,
                "ma34_period": self.actual_ma34_period,
                "use_ma55": self.p.use_ma55,
                "use_ma89": self.p.use_ma89,
                "volume_ma_period": self.p.volume_ma_period,
                "volume_threshold_buy": self.p.volume_threshold_buy,
                "volume_threshold_breakout": self.p.volume_threshold_breakout,
                "use_volume_confirmation": self.p.use_volume_confirmation,
                "use_multi_resonance": self.p.use_multi_resonance,
                "stock_type": self.p.stock_type,
            }
        )
        return info


"""
Hull Moving Average (HMA) Trading Strategy.

This strategy implements a trading system based on Hull Moving Average (HMA)
to identify trend changes with minimal lag compared to traditional moving averages.

Buy signals (all must be true):
1. HMA turns from flat/downward to clearly upward (slope > threshold)
2. Price is above HMA, and doesn't break HMA during pullbacks
3. Volume moderately increases (volume ratio ≥ 1.5)

Sell signals (any of):
1. Warning: HMA turns from rising to flat or slope decreases significantly (reduce 30%-50%)
2. Exit: Price breaks below HMA and HMA starts to turn downward (full exit)

Pitfalls to avoid:
- Don't use HMA alone in choppy markets (combine with trend filters)
- Don't over-pursue sensitivity by shortening periods too much
- Don't ignore fundamentals and blindly follow signals

This strategy follows the standard backtrader pattern:
- Indicators are initialized in __init__
- Trading logic is in next() method
- Indicators trigger buy/sell operations
"""

import logging
import math
from datetime import datetime
from typing import Dict, Optional

import backtrader as bt

from atm.trading.indicators.hma import HullMovingAverage
from atm.trading.strategy.base import BaseStrategy

logger = logging.getLogger(__name__)


class HMAStrategy(BaseStrategy):
    """
    Hull Moving Average (HMA) Trading Strategy.

    This strategy uses HMA to identify trend changes with minimal lag.
    HMA reduces lag compared to traditional moving averages by using
    weighted moving averages and mathematical formulas.

    Buy signals (all must be true):
    1. HMA turns from flat/downward to clearly upward (slope > threshold)
    2. Price is above HMA, and doesn't break HMA during pullbacks
    3. Volume moderately increases (volume ratio ≥ 1.5)

    Sell signals:
    - Warning: HMA flattens or slope decreases (reduce position 30%-50%)
    - Exit: Price breaks below HMA and HMA turns downward (full exit)

    Args:
        hma_period: HMA period (default: 20).
        volume_ma_period: Volume moving average period (default: 5).
        volume_threshold: Volume threshold for buy signals (default: 1.5).
        slope_threshold: Minimum slope for HMA upward turn (default: 30 degrees in radians).
        use_bollinger_filter: Use Bollinger Bands as trend filter (default: True).
        use_macd_filter: Use MACD as trend filter (default: True).
        stop_loss_pct: Stop loss percentage below HMA (default: 0.02 = 2%).
    """

    params = (
        ("hma_period", 20),  # HMA period
        ("volume_ma_period", 5),  # Volume moving average period
        ("volume_threshold", 1.5),  # Volume threshold for buy signals
        ("slope_threshold", 0.52),  # Minimum slope for HMA upward turn (~30 degrees in radians)
        ("use_bollinger_filter", True),  # Use Bollinger Bands as trend filter
        ("use_macd_filter", True),  # Use MACD as trend filter
        ("stop_loss_pct", 0.02),  # Stop loss percentage below HMA (2%)
    )

    def __init__(self):
        """
        Initialize HMA Strategy.

        Indicators are initialized here following backtrader's standard pattern.
        """
        super().__init__()

        # Hull Moving Average
        self.hma = HullMovingAverage(self.data.close, period=self.p.hma_period)

        # Volume confirmation
        self.volume_ma = bt.indicators.SMA(self.data.volume, period=self.p.volume_ma_period)
        self.volume_ratio = self.data.volume / self.volume_ma

        # Trend filters
        if self.p.use_bollinger_filter:
            self.bollinger = bt.indicators.BollingerBands(
                self.data.close, period=20, devfactor=2.0
            )

        if self.p.use_macd_filter:
            self.macd = bt.indicators.MACD(self.data.close)

        # Track position state
        self.entry_price = None
        self.position_reduced = False  # Track if position was reduced

        logger.info(
            f"HMA Strategy initialized with "
            f"HMA({self.p.hma_period}), "
            f"bollinger_filter={self.p.use_bollinger_filter}, "
            f"macd_filter={self.p.use_macd_filter}"
        )

    def next(self):
        """
        Called for each bar.

        Trading logic based on HMA:
        - HMA upward turn with volume confirmation
        - HMA flattening or downward turn for exits
        - Trend filters for signal validation
        """
        # Skip if not enough data
        max_period = max(self.p.hma_period, self.p.volume_ma_period, 20)
        if len(self.data) < max_period + 5:
            return

        # Safely access indicator values
        try:
            current_price = self.data.close[0]
            current_hma = self.hma[0]
            current_volume_ratio = self.volume_ratio[0]
        except (IndexError, TypeError):
            return

        # Calculate HMA slope (rate of change)
        hma_slope = self._calculate_hma_slope()

        # Buy signals (all must be true)
        if not self.position:
            # Signal 1: HMA turns from flat/downward to clearly upward
            hma_turning_up = self._check_hma_turning_up(hma_slope)

            # Signal 2: Price is above HMA
            price_above_hma = current_price > current_hma

            # Signal 3: Volume moderately increases
            volume_confirmed = current_volume_ratio >= self.p.volume_threshold

            # Trend filters
            trend_confirmed = True
            if self.p.use_bollinger_filter:
                try:
                    # Price should be above Bollinger middle band and bands opening upward
                    price_above_bb_mid = current_price > self.bollinger.lines.mid[0]
                    bb_expanding = (
                        self.bollinger.lines.top[0] - self.bollinger.lines.bot[0]
                    ) > (
                        self.bollinger.lines.top[-1] - self.bollinger.lines.bot[-1]
                    )
                    trend_confirmed = trend_confirmed and (
                        price_above_bb_mid or bb_expanding
                    )
                except (IndexError, TypeError):
                    pass

            if self.p.use_macd_filter:
                try:
                    # MACD should be above zero or showing golden cross
                    macd_above_zero = self.macd.lines.macd[0] > 0
                    macd_golden_cross = (
                        self.macd.lines.macd[0] > self.macd.lines.signal[0]
                        and self.macd.lines.macd[-1] <= self.macd.lines.signal[-1]
                    )
                    trend_confirmed = trend_confirmed and (
                        macd_above_zero or macd_golden_cross
                    )
                except (IndexError, TypeError):
                    pass

            if hma_turning_up and price_above_hma and volume_confirmed and trend_confirmed:
                logger.debug(
                    f"Buy signal (HMA turning up + price above HMA + volume) at "
                    f"{self.datas[0].datetime.date(0)}, "
                    f"price={current_price:.2f}, "
                    f"HMA={current_hma:.2f}, "
                    f"slope={hma_slope:.4f}, "
                    f"volume_ratio={current_volume_ratio:.2f}"
                )
                self.buy()
                self.entry_price = current_price
                self.position_reduced = False

        # Sell signals
        if self.position:
            # Check stop loss
            if self.entry_price is not None:
                stop_loss_price = current_hma * (1 - self.p.stop_loss_pct)
                if current_price < stop_loss_price:
                    logger.debug(
                        f"Stop loss triggered at {self.datas[0].datetime.date(0)}, "
                        f"price={current_price:.2f}, stop_loss={stop_loss_price:.2f}"
                    )
                    self.close()
                    self.entry_price = None
                    self.position_reduced = False
                    return

            # Warning signal: HMA flattens or slope decreases significantly
            hma_flattening = self._check_hma_flattening(hma_slope)
            if hma_flattening and not self.position_reduced:
                # Reduce position by 30%-50%
                position_size = self.position.size
                reduce_size = int(position_size * 0.4)  # Reduce by 40%
                if reduce_size > 0:
                    logger.debug(
                        f"Warning signal (HMA flattening), reducing position at "
                        f"{self.datas[0].datetime.date(0)}, "
                        f"reduce_size={reduce_size}"
                    )
                    self.sell(size=reduce_size)
                    self.position_reduced = True

            # Exit signal: Price breaks below HMA and HMA turns downward
            price_below_hma = current_price < current_hma
            hma_turning_down = self._check_hma_turning_down(hma_slope)

            if price_below_hma and hma_turning_down:
                logger.debug(
                    f"Exit signal (price below HMA + HMA turning down) at "
                    f"{self.datas[0].datetime.date(0)}, "
                    f"price={current_price:.2f}, "
                    f"HMA={current_hma:.2f}, "
                    f"slope={hma_slope:.4f}"
                )
                self.close()
                self.entry_price = None
                self.position_reduced = False

    def _calculate_hma_slope(self) -> float:
        """
        Calculate HMA slope (rate of change).

        Returns:
            HMA slope value (positive = upward, negative = downward).
        """
        if len(self.hma) < 2:
            return 0.0

        try:
            current_hma = self.hma[0]
            prev_hma = self.hma[-1]
            # Calculate slope as percentage change
            if prev_hma > 0:
                slope = (current_hma - prev_hma) / prev_hma
            else:
                slope = 0.0
            return slope
        except (IndexError, TypeError):
            return 0.0

    def _check_hma_turning_up(self, current_slope: float) -> bool:
        """
        Check if HMA is turning from flat/downward to clearly upward.

        Args:
            current_slope: Current HMA slope.

        Returns:
            True if HMA is turning upward, False otherwise.
        """
        if len(self.hma) < 3:
            return False

        try:
            # Current slope should be positive
            slope_positive = current_slope > 0

            # Calculate angle in degrees (approximate)
            # For small angles, tan(angle) ≈ angle in radians
            # 30 degrees ≈ 0.52 radians, but we use percentage change
            # A 1% change per period is roughly equivalent to a significant upward move
            slope_above_threshold = current_slope >= 0.01  # At least 1% increase

            # Previous slope should be flat or negative
            prev_slope = (self.hma[-1] - self.hma[-2]) / self.hma[-2] if self.hma[-2] > 0 else 0.0
            prev_flat_or_down = prev_slope <= 0.005  # Allow very small positive slope

            # HMA value should be increasing
            hma_increasing = self.hma[0] > self.hma[-1]

            return slope_positive and slope_above_threshold and prev_flat_or_down and hma_increasing
        except (IndexError, TypeError, ZeroDivisionError):
            return False

    def _check_hma_flattening(self, current_slope: float) -> bool:
        """
        Check if HMA is flattening (slope decreasing significantly).

        Args:
            current_slope: Current HMA slope.

        Returns:
            True if HMA is flattening, False otherwise.
        """
        if len(self.hma) < 3:
            return False

        try:
            # Current slope should be decreasing compared to previous
            prev_slope = (self.hma[-1] - self.hma[-2]) / self.hma[-2] if self.hma[-2] > 0 else 0.0
            slope_decreasing = current_slope < prev_slope * 0.5  # Slope reduced by 50%

            # Current slope should be small (flattening)
            slope_small = abs(current_slope) < 0.005  # Less than 0.5% change

            # HMA should be flattening (not increasing much)
            hma_flat = abs(self.hma[0] - self.hma[-1]) / self.hma[-1] < 0.01 if self.hma[-1] > 0 else False

            return (slope_decreasing and prev_slope > 0) or slope_small or hma_flat
        except (IndexError, TypeError, ZeroDivisionError):
            return False

    def _check_hma_turning_down(self, current_slope: float) -> bool:
        """
        Check if HMA is turning downward.

        Args:
            current_slope: Current HMA slope.

        Returns:
            True if HMA is turning downward, False otherwise.
        """
        if len(self.hma) < 2:
            return False

        try:
            # Current slope should be negative
            slope_negative = current_slope < 0

            # Previous slope should be positive or flat
            prev_slope = (self.hma[-1] - self.hma[-2]) / self.hma[-2] if self.hma[-2] > 0 else 0.0
            prev_positive_or_flat = prev_slope >= -0.01

            return slope_negative and prev_positive_or_flat
        except (IndexError, TypeError, ZeroDivisionError):
            return False

    def get_info(self) -> Dict:
        """
        Get strategy information.

        Returns:
            Dictionary containing strategy information.
        """
        info = super().get_info()
        info.update(
            {
                "hma_period": self.p.hma_period,
                "volume_ma_period": self.p.volume_ma_period,
                "volume_threshold": self.p.volume_threshold,
                "slope_threshold": self.p.slope_threshold,
                "use_bollinger_filter": self.p.use_bollinger_filter,
                "use_macd_filter": self.p.use_macd_filter,
                "stop_loss_pct": self.p.stop_loss_pct,
            }
        )
        return info


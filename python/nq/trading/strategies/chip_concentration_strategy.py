"""
Chip Concentration (12%) Stock Selection Strategy.

This strategy implements the "12% Chip Concentration Stock Selection Method"
to identify stocks where institutional investors (main force) have completed
chip accumulation, avoiding junk stocks dominated by retail investors.

Core Logic:
1. Chip concentration ≤ 12% (90% chips in narrow price range)
2. Two accumulation patterns:
   - Pattern 1: Low-level gentle accumulation (低位温和吸筹)
   - Pattern 2: Suppressed to low platform accumulation (打压至低平台吸筹)
3. Consolidation duration validation (≥20 days for pattern 1, ≥30 days for pattern 2)
4. Volume validation (shrink >30% during consolidation)
5. Breakout signal (price breaks above chip concentration area + volume >50% increase)
6. Position sizing: 30% initial, 40% add-on, 30% reserved

This strategy follows the standard backtrader pattern:
- Indicators are initialized in __init__
- Trading logic is in next() method
- Indicators trigger buy/sell operations
"""

import logging
from typing import Any, Dict, Optional, Tuple

import backtrader.indicators as btind

from nq.trading.indicators.chip_peak import (
    ChipPeakIndicator,
    ChipPeakPattern,
    DummyChipPeakIndicator,
)
from nq.trading.strategies.base import BaseStrategy

logger = logging.getLogger(__name__)


class ChipConcentrationStrategy(BaseStrategy):
    """
    12% Chip Concentration Stock Selection Strategy.

    This strategy identifies stocks where the main force has completed
    chip accumulation by analyzing chip concentration and accumulation patterns.

    Buy Signals (all must be true):
    1. Chip concentration ≤ 12% (90% chips in narrow price range)
    2. Chips highly concentrated at low price levels
    3. One of two accumulation patterns:
       - Pattern 1: Low-level gentle accumulation (≥20 days consolidation)
       - Pattern 2: Suppressed to low platform accumulation (≥30 days total)
    4. Volume shrinks >30% during consolidation vs. previous decline
    5. Price breaks above chip concentration area
    6. Volume increases >50% vs. 5-day average

    Sell Signals (any of):
    1. Volume shrinks significantly (量能萎缩)
    2. Price falls back below breakout level
    3. Chip concentration > 12% (chips become dispersed)

    Position Sizing:
    - Initial: 30% of funds
    - Add-on: 40% of funds (if no pullback within 3 days after breakout)
    - Reserved: 30% of funds

    Parameters:
    - chip_peak_indicator: Chip peak indicator instance (default: DummyChipPeakIndicator)
    - concentration_threshold: Chip concentration threshold (default: 0.12 = 12%)
    - pattern1_min_days: Minimum consolidation days for pattern 1 (default: 20)
    - pattern2_min_days: Minimum consolidation days for pattern 2 (default: 30)
    - volume_shrink_threshold: Volume shrink threshold during consolidation (default: 0.3 = 30%)
    - volume_breakout_threshold: Volume increase threshold for breakout (default: 1.5 = 50% increase)
    - initial_position_pct: Initial position percentage (default: 0.3 = 30%)
    - add_position_pct: Add-on position percentage (default: 0.4 = 40%)
    - add_position_days: Days to wait before add-on (default: 3)
    - stop_loss_pct: Stop loss percentage (default: 0.05 = 5%)
    - take_profit_pct: Take profit percentage (default: 0.15 = 15%)
    """

    params = (
        ("chip_peak_indicator", None),  # Chip peak indicator instance
        ("concentration_threshold", 0.12),  # 12% concentration threshold
        ("pattern1_min_days", 20),  # Minimum days for pattern 1
        ("pattern2_min_days", 30),  # Minimum days for pattern 2
        ("volume_shrink_threshold", 0.3),  # Volume shrink threshold (30%)
        ("volume_breakout_threshold", 1.5),  # Volume increase for breakout (50% increase = 1.5x)
        ("initial_position_pct", 0.3),  # Initial position (30%)
        ("add_position_pct", 0.4),  # Add-on position (40%)
        ("add_position_days", 3),  # Days to wait before add-on
        ("stop_loss_pct", 0.05),  # Stop loss (5%)
        ("take_profit_pct", 0.15),  # Take profit (15%)
        ("volume_ma_period", 5),  # Volume MA period for breakout check
        ("consolidation_lookback", 60),  # Lookback period for consolidation detection
    )

    def __init__(self):
        """Initialize Chip Concentration Strategy."""
        super().__init__()

        # Chip peak indicator
        if self.p.chip_peak_indicator is None:
            self.chip_peak = DummyChipPeakIndicator(self.data)
        elif isinstance(self.p.chip_peak_indicator, ChipPeakIndicator):
            self.chip_peak = self.p.chip_peak_indicator
        elif callable(self.p.chip_peak_indicator):
            self.chip_peak = self.p.chip_peak_indicator(self.data)
        else:
            logger.warning(
                f"Invalid chip_peak_indicator type: {type(self.p.chip_peak_indicator)}. "
                "Using DummyChipPeakIndicator."
            )
            self.chip_peak = DummyChipPeakIndicator(self.data)

        # Volume indicators
        self.volume_ma = btind.SMA(self.data.volume, period=self.p.volume_ma_period)
        self.volume_ma_long = btind.SMA(
            self.data.volume, period=self.p.consolidation_lookback
        )

        # Price indicators for consolidation detection
        self.price_ma = btind.SMA(self.data.close, period=20)
        self.atr = btind.ATR(self.data, period=14)

        # Track state
        self.entry_price = None
        self.breakout_price = None
        self.breakout_bar = None
        self.position_stage = 0  # 0: no position, 1: initial (30%), 2: full (70%)
        self.consolidation_start = None
        self.consolidation_days = 0
        self.pattern_type = None  # 1 or 2

        # Ensure enough data
        max_period = max(
            self.p.volume_ma_period,
            self.p.consolidation_lookback,
            20,
            14,
        )
        self.addminperiod(max_period + 10)

    def _calculate_concentration(self) -> Optional[float]:
        """
        Calculate chip concentration.

        Formula: (High_90pct - Low_90pct) / (High_90pct + Low_90pct)
        Where High_90pct and Low_90pct are the upper and lower bounds
        of the 90% holding cost range.

        Returns:
            Concentration value (0-1), or None if not available.
        """
        try:
            upper_edge = self.chip_peak.get_upper_edge()
            lower_edge = self.chip_peak.get_lower_edge()

            if upper_edge is None or lower_edge is None:
                return None

            if upper_edge <= 0 or lower_edge <= 0:
                return None

            # Calculate concentration
            concentration = (upper_edge - lower_edge) / (upper_edge + lower_edge)
            return concentration
        except (TypeError, ValueError, ZeroDivisionError):
            return None

    def _detect_consolidation(self) -> Tuple[bool, int, Optional[int]]:
        """
        Detect consolidation pattern.

        Returns:
            (is_consolidating, consolidation_days, pattern_type)
            - is_consolidating: Whether currently in consolidation
            - consolidation_days: Number of days in consolidation
            - pattern_type: 1 (gentle) or 2 (suppressed), or None
        """
        try:
            lookback = min(self.p.consolidation_lookback, len(self.data))
            if lookback < 20:
                return False, 0, None

            # Calculate price range and volatility
            prices = [self.data.close[-i] for i in range(lookback)]
            price_high = max(prices)
            price_low = min(prices)
            price_range = price_high - price_low
            current_price = self.data.close[0]

            # Check if price is in a narrow range (consolidation)
            price_range_pct = price_range / current_price if current_price > 0 else 1.0
            atr_value = self.atr[0] if len(self.atr) > 0 else price_range

            # Consolidation: price range < 10% and low volatility
            is_consolidating = price_range_pct < 0.10 and atr_value < current_price * 0.05

            if not is_consolidating:
                return False, 0, None

            # Count consolidation days
            consolidation_days = 0
            for i in range(1, lookback):
                if abs(self.data.close[-i] - self.data.close[-i - 1]) / self.data.close[-i] < 0.02:
                    consolidation_days += 1
                else:
                    break

            # Determine pattern type based on price level
            # Pattern 1: Low-level gentle (price near recent low)
            # Pattern 2: Suppressed (price dropped then consolidated)
            recent_low = min([self.data.low[-i] for i in range(min(30, lookback))])
            price_vs_low = (current_price - recent_low) / recent_low if recent_low > 0 else 0

            if price_vs_low < 0.05:  # Within 5% of recent low
                pattern_type = 1  # Low-level gentle
            else:
                pattern_type = 2  # Suppressed to low platform

            return True, consolidation_days, pattern_type
        except (IndexError, TypeError, ValueError, ZeroDivisionError):
            return False, 0, None

    def _check_volume_shrink(self) -> bool:
        """
        Check if volume has shrunk during consolidation.

        Volume during consolidation should shrink >30% vs. previous decline phase.
        """
        try:
            lookback = min(self.p.consolidation_lookback, len(self.data))
            if lookback < 20:
                return False

            # Average volume during recent consolidation (last 10 days)
            recent_volume = sum([self.data.volume[-i] for i in range(1, min(11, lookback))]) / min(10, lookback - 1)

            # Average volume during previous decline phase (20-40 days ago)
            if lookback >= 40:
                decline_volume = sum([self.data.volume[-i] for i in range(20, 40)]) / 20
            else:
                decline_volume = sum([self.data.volume[-i] for i in range(20, lookback)]) / (lookback - 20)

            if decline_volume <= 0:
                return False

            volume_shrink = (decline_volume - recent_volume) / decline_volume
            return volume_shrink >= self.p.volume_shrink_threshold
        except (IndexError, TypeError, ValueError, ZeroDivisionError):
            return False

    def _check_breakout(self) -> bool:
        """
        Check if price breaks above chip concentration area with volume confirmation.

        Breakout conditions:
        1. Price breaks above upper edge of chip concentration
        2. Volume increases >50% vs. 5-day average
        """
        try:
            upper_edge = self.chip_peak.get_upper_edge()
            if upper_edge is None:
                return False

            current_price = self.data.close[0]
            current_volume = self.data.volume[0]

            # Check price breakout
            price_breakout = current_price > upper_edge * 1.01  # 1% above upper edge

            # Check volume increase
            volume_ma_value = self.volume_ma[0] if len(self.volume_ma) > 0 else current_volume
            if volume_ma_value <= 0:
                return False

            volume_increase = current_volume / volume_ma_value
            volume_confirmed = volume_increase >= self.p.volume_breakout_threshold

            return price_breakout and volume_confirmed
        except (IndexError, TypeError, ValueError, ZeroDivisionError):
            return False

    def next(self):
        """Main strategy logic."""
        # Skip if not enough data
        if len(self.data) < self.p.consolidation_lookback + 10:
            return

        try:
            current_price = self.data.close[0]
            current_volume = self.data.volume[0]
        except (IndexError, TypeError):
            return

        # Calculate chip concentration
        concentration = self._calculate_concentration()
        if concentration is None:
            return

        # Risk management for existing position
        if self.position:
            # Stop loss
            if self.entry_price is not None:
                if current_price <= self.entry_price * (1 - self.p.stop_loss_pct):
                    logger.debug(
                        f"Stop loss triggered at {self.data.datetime.date(0)}, "
                        f"price={current_price:.2f}, entry={self.entry_price:.2f}"
                    )
                    self.close()
                    self._reset_position()
                    return

            # Take profit
            if self.entry_price is not None:
                if current_price >= self.entry_price * (1 + self.p.take_profit_pct):
                    logger.debug(
                        f"Take profit triggered at {self.data.datetime.date(0)}, "
                        f"price={current_price:.2f}, entry={self.entry_price:.2f}"
                    )
                    self.close()
                    self._reset_position()
                    return

            # Sell signal: Concentration > threshold (chips dispersed)
            if concentration > self.p.concentration_threshold:
                logger.debug(
                    f"Sell signal (concentration > threshold) at {self.data.datetime.date(0)}, "
                    f"concentration={concentration:.4f}, price={current_price:.2f}"
                )
                self.close()
                self._reset_position()
                return

            # Sell signal: Volume shrinks significantly
            volume_ma_value = self.volume_ma[0] if len(self.volume_ma) > 0 else current_volume
            if volume_ma_value > 0:
                volume_ratio = current_volume / volume_ma_value
                if volume_ratio < 0.5:  # Volume shrinks to <50% of average
                    logger.debug(
                        f"Sell signal (volume shrinks) at {self.data.datetime.date(0)}, "
                        f"volume_ratio={volume_ratio:.2f}, price={current_price:.2f}"
                    )
                    self.close()
                    self._reset_position()
                    return

            # Sell signal: Price falls back below breakout level
            if self.breakout_price is not None:
                if current_price < self.breakout_price * 0.98:  # 2% below breakout
                    logger.debug(
                        f"Sell signal (price falls back) at {self.data.datetime.date(0)}, "
                        f"price={current_price:.2f}, breakout={self.breakout_price:.2f}"
                    )
                    self.close()
                    self._reset_position()
                    return

            # Add-on position logic
            if self.position_stage == 1 and self.breakout_bar is not None:
                bars_since_breakout = len(self.data) - self.breakout_bar
                if bars_since_breakout >= self.p.add_position_days:
                    # Check if price hasn't pulled back
                    if current_price >= self.breakout_price * 0.98:
                        # Add 40% position
                        cash_to_use = self.broker.getcash() * self.p.add_position_pct
                        size = int(cash_to_use / current_price)
                        if size > 0:
                            self.buy(size=size)
                            self.position_stage = 2
                            logger.debug(
                                f"Add position at {self.data.datetime.date(0)}, "
                                f"price={current_price:.2f}, size={size}"
                            )

        # Entry logic
        if not self.position:
            # Check chip concentration
            if concentration > self.p.concentration_threshold:
                return  # Concentration too high, skip

            # Check chip pattern (should be single peak at low level)
            chip_pattern = self.chip_peak.get_pattern()
            if chip_pattern != ChipPeakPattern.SINGLE_PEAK:
                return  # Not single peak, skip

            # Detect consolidation
            is_consolidating, consolidation_days, pattern_type = self._detect_consolidation()

            if not is_consolidating:
                return  # Not in consolidation, skip

            # Check consolidation duration
            min_days = (
                self.p.pattern1_min_days
                if pattern_type == 1
                else self.p.pattern2_min_days
            )
            if consolidation_days < min_days:
                return  # Consolidation duration insufficient, skip

            # Check volume shrink
            if not self._check_volume_shrink():
                return  # Volume hasn't shrunk enough, skip

            # Check breakout signal
            if not self._check_breakout():
                return  # No breakout signal, skip

            # All conditions met, enter initial position (30%)
            cash_to_use = self.broker.getcash() * self.p.initial_position_pct
            size = int(cash_to_use / current_price)
            if size > 0:
                self.buy(size=size)
                self.entry_price = current_price
                self.breakout_price = current_price
                self.breakout_bar = len(self.data)
                self.position_stage = 1
                self.pattern_type = pattern_type
                logger.debug(
                    f"Initial buy at {self.data.datetime.date(0)}, "
                    f"price={current_price:.2f}, size={size}, "
                    f"concentration={concentration:.4f}, pattern={pattern_type}, "
                    f"consolidation_days={consolidation_days}"
                )

    def _reset_position(self):
        """Reset position tracking variables."""
        self.entry_price = None
        self.breakout_price = None
        self.breakout_bar = None
        self.position_stage = 0
        self.consolidation_start = None
        self.consolidation_days = 0
        self.pattern_type = None

    def get_info(self) -> Dict[str, Any]:
        """Get strategy information."""
        info = super().get_info()
        info.update(
            {
                "concentration_threshold": self.p.concentration_threshold,
                "pattern1_min_days": self.p.pattern1_min_days,
                "pattern2_min_days": self.p.pattern2_min_days,
                "volume_shrink_threshold": self.p.volume_shrink_threshold,
                "volume_breakout_threshold": self.p.volume_breakout_threshold,
                "initial_position_pct": self.p.initial_position_pct,
                "add_position_pct": self.p.add_position_pct,
                "add_position_days": self.p.add_position_days,
                "stop_loss_pct": self.p.stop_loss_pct,
                "take_profit_pct": self.p.take_profit_pct,
            }
        )
        return info


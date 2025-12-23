"""
Chip Peak + 20-day Cost Line Trading Strategy.

This strategy implements a trading system based on chip peak analysis
and the 20-day cost line to identify accumulation and distribution
activities of institutional investors (main force).

Buy signals (all must be true):
1. Chip peak is "single peak concentrated" (单峰密集)
2. Price at upper edge of chip peak with 20-day cost line support
3. Trading volume < 50% of average volume of previous 5 days

Sell signals (any of):
1. Chip peak changes from single peak to multi-peak
2. Price at lower edge of chip peak with 20-day cost line pressure
3. Volume increases significantly but price stagnates (对倒出货)

Pitfalls to avoid:
- Avoid "multi-peak concentrated" stocks
- Peak tip should be as sharp as possible

This strategy follows the standard backtrader pattern:
- Indicators are initialized in __init__
- Trading logic is in next() method
- Indicators trigger buy/sell operations
"""

import logging
from typing import Dict, Optional

import backtrader as bt

from atm.trading.indicators.chip_peak import (
    ChipPeakIndicator,
    ChipPeakPattern,
    DummyChipPeakIndicator,
)
from atm.trading.strategies.base import BaseStrategy

logger = logging.getLogger(__name__)


class ChipPeakStrategy(BaseStrategy):
    """
    Chip Peak + 20-day Cost Line Trading Strategy.

    This strategy uses chip peak analysis to identify where institutional
    investors (main force) have accumulated shares, combined with the
    20-day cost line to determine entry and exit points.

    Buy signals (all must be true):
    1. Chip peak is "single peak concentrated" (单峰密集)
    2. Price at upper edge of chip peak with 20-day cost line support
    3. Trading volume < 50% of average volume of previous 5 days

    Sell signals (any of):
    1. Chip peak changes from single peak to multi-peak
    2. Price at lower edge of chip peak with 20-day cost line pressure
    3. Volume increases significantly but price stagnates

    Args:
        chip_peak_indicator: Chip peak indicator instance (default: DummyChipPeakIndicator).
        ma_period: Moving average period for cost line (default: 20).
        volume_ma_period: Volume moving average period (default: 5).
        volume_threshold_buy: Volume threshold for buy signals (default: 0.5 = 50%).
        volume_threshold_sell: Volume threshold for sell signals (default: 1.5 = 150%).
        price_tolerance: Price tolerance for edge detection (default: 0.02 = 2%).
        avoid_multi_peak: Avoid trading multi-peak stocks (default: True).
        min_peak_sharpness: Minimum peak sharpness required (default: 0.3).
    """

    params = (
        ("chip_peak_indicator", None),  # Chip peak indicator instance
        ("ma_period", 20),  # Moving average period for cost line
        ("volume_ma_period", 5),  # Volume moving average period
        ("volume_threshold_buy", 0.5),  # Volume threshold for buy (50% of average)
        ("volume_threshold_sell", 1.5),  # Volume threshold for sell (150% of average)
        ("price_tolerance", 0.02),  # Price tolerance for edge detection (2%)
        ("avoid_multi_peak", True),  # Avoid trading multi-peak stocks
        ("min_peak_sharpness", 0.3),  # Minimum peak sharpness required
    )

    def __init__(self):
        """
        Initialize Chip Peak Strategy.

        Indicators are initialized here following backtrader's standard pattern.
        """
        super().__init__()

        # Chip peak indicator
        if self.p.chip_peak_indicator is None:
            # Use dummy implementation if not provided
            self.chip_peak = DummyChipPeakIndicator(self.data)
        elif isinstance(self.p.chip_peak_indicator, ChipPeakIndicator):
            # Already an instance, use it directly
            self.chip_peak = self.p.chip_peak_indicator
        elif callable(self.p.chip_peak_indicator):
            # It's a class or factory function, instantiate it
            self.chip_peak = self.p.chip_peak_indicator(self.data)
        else:
            # Fallback to dummy implementation
            logger.warning(
                f"Invalid chip_peak_indicator type: {type(self.p.chip_peak_indicator)}. "
                "Using DummyChipPeakIndicator."
            )
            self.chip_peak = DummyChipPeakIndicator(self.data)

        # 20-day cost line (moving average)
        self.cost_line = bt.indicators.SMA(self.data.close, period=self.p.ma_period)

        # Volume confirmation
        self.volume_ma = bt.indicators.SMA(self.data.volume, period=self.p.volume_ma_period)
        self.volume_ratio = self.data.volume / self.volume_ma

        # Track position state
        self.entry_price = None
        self.entry_pattern = None  # Track pattern at entry

        logger.info(
            f"Chip Peak Strategy initialized with "
            f"MA{self.p.ma_period}, "
            f"volume_threshold_buy={self.p.volume_threshold_buy}, "
            f"avoid_multi_peak={self.p.avoid_multi_peak}"
        )

    def next(self):
        """
        Called for each bar.

        Trading logic based on chip peak and cost line:
        - Single peak concentrated buy signals
        - Multi-peak sell signals
        - Volume and price action confirmation
        """
        # Skip if not enough data
        max_period = max(self.p.ma_period, self.p.volume_ma_period)
        if len(self.data) < max_period + 5:
            return

        # Safely access indicator values
        try:
            current_price = self.data.close[0]
            current_cost_line = self.cost_line[0]
            current_volume_ratio = self.volume_ratio[0]
        except (IndexError, TypeError):
            return

        # Get chip peak values
        chip_pattern = self.chip_peak.get_pattern()
        upper_edge = self.chip_peak.get_upper_edge()
        lower_edge = self.chip_peak.get_lower_edge()
        peak_sharpness = self.chip_peak.get_peak_sharpness()

        # Skip if chip peak data is not available
        if chip_pattern == ChipPeakPattern.UNKNOWN:
            return

        # Avoid multi-peak stocks if configured
        if self.p.avoid_multi_peak and chip_pattern == ChipPeakPattern.MULTI_PEAK:
            if not self.position:
                # Skip buy signals for multi-peak stocks
                return

        # Check peak sharpness
        if peak_sharpness < self.p.min_peak_sharpness:
            if not self.position:
                # Skip buy signals if peak is not sharp enough
                return

        # Buy signals (all must be true)
        if not self.position:
            # Signal 1: Single peak concentrated
            is_single_peak = chip_pattern == ChipPeakPattern.SINGLE_PEAK

            # Signal 2: Price at upper edge with cost line support
            price_at_upper_edge = False
            cost_line_support = False
            if upper_edge is not None:
                # Check if price is near upper edge (within tolerance)
                price_diff_upper = abs(current_price - upper_edge) / upper_edge
                price_at_upper_edge = price_diff_upper <= self.p.price_tolerance

                # Check if cost line is at upper edge (support)
                if price_at_upper_edge:
                    cost_line_diff = abs(current_cost_line - upper_edge) / upper_edge
                    cost_line_support = cost_line_diff <= self.p.price_tolerance

            # Signal 3: Low trading volume (< 50% of average)
            volume_low = current_volume_ratio < self.p.volume_threshold_buy

            if is_single_peak and price_at_upper_edge and cost_line_support and volume_low:
                logger.debug(
                    f"Buy signal (single peak + cost line support + low volume) at "
                    f"{self.datas[0].datetime.date(0)}, "
                    f"price={current_price:.2f}, "
                    f"upper_edge={upper_edge:.2f}, "
                    f"cost_line={current_cost_line:.2f}, "
                    f"volume_ratio={current_volume_ratio:.2f}"
                )
                self.buy()
                self.entry_price = current_price
                self.entry_pattern = chip_pattern

        # Sell signals (any of)
        if self.position:
            # Signal 1: Chip peak changes from single peak to multi-peak
            if self.entry_pattern == ChipPeakPattern.SINGLE_PEAK:
                if chip_pattern == ChipPeakPattern.MULTI_PEAK:
                    logger.debug(
                        f"Sell signal (pattern changed to multi-peak) at "
                        f"{self.datas[0].datetime.date(0)}, "
                        f"price={current_price:.2f}"
                    )
                    self.close()
                    self.entry_price = None
                    self.entry_pattern = None
                    return

            # Signal 2: Price at lower edge with cost line pressure
            if lower_edge is not None:
                # Check if price is near lower edge (within tolerance)
                price_diff_lower = abs(current_price - lower_edge) / lower_edge
                price_at_lower_edge = price_diff_lower <= self.p.price_tolerance

                # Check if cost line is at lower edge (pressure)
                if price_at_lower_edge:
                    cost_line_diff = abs(current_cost_line - lower_edge) / lower_edge
                    cost_line_pressure = cost_line_diff <= self.p.price_tolerance

                    if cost_line_pressure:
                        logger.debug(
                            f"Sell signal (price at lower edge + cost line pressure) at "
                            f"{self.datas[0].datetime.date(0)}, "
                            f"price={current_price:.2f}, "
                            f"lower_edge={lower_edge:.2f}, "
                            f"cost_line={current_cost_line:.2f}"
                        )
                        self.close()
                        self.entry_price = None
                        self.entry_pattern = None
                        return

            # Signal 3: Volume increases but price stagnates (对倒出货)
            if current_volume_ratio >= self.p.volume_threshold_sell:
                # Check if price has not risen significantly
                if self.entry_price is not None:
                    price_change = (current_price - self.entry_price) / self.entry_price
                    # Price change < 1% while volume > 150% indicates dumping
                    if price_change < 0.01:
                        logger.debug(
                            f"Sell signal (volume increase + price stagnation) at "
                            f"{self.datas[0].datetime.date(0)}, "
                            f"price={current_price:.2f}, "
                            f"volume_ratio={current_volume_ratio:.2f}, "
                            f"price_change={price_change*100:.2f}%"
                        )
                        self.close()
                        self.entry_price = None
                        self.entry_pattern = None
                        return

    def get_info(self) -> Dict:
        """
        Get strategy information.

        Returns:
            Dictionary containing strategy information.
        """
        info = super().get_info()
        info.update(
            {
                "ma_period": self.p.ma_period,
                "volume_ma_period": self.p.volume_ma_period,
                "volume_threshold_buy": self.p.volume_threshold_buy,
                "volume_threshold_sell": self.p.volume_threshold_sell,
                "price_tolerance": self.p.price_tolerance,
                "avoid_multi_peak": self.p.avoid_multi_peak,
                "min_peak_sharpness": self.p.min_peak_sharpness,
                "chip_peak_indicator": (
                    self.chip_peak.__class__.__name__
                    if self.chip_peak
                    else "None"
                ),
            }
        )
        return info


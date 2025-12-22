"""
Chip Peak (筹码峰) Indicator Interface.

This module defines the interface for chip peak analysis indicators.
Chip peak analysis is used to identify the distribution of shares held
at different price levels, which helps identify accumulation and distribution
activities of institutional investors (main force).

The actual implementation of chip peak calculation can be done separately,
as it requires complex data processing and may vary based on data sources.
"""

import logging
from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional, Tuple

import backtrader as bt

logger = logging.getLogger(__name__)


class ChipPeakPattern(Enum):
    """Chip peak pattern types."""

    SINGLE_PEAK = "single_peak"  # 单峰密集
    MULTI_PEAK = "multi_peak"  # 多峰
    UNKNOWN = "unknown"  # 未知/未计算


class ChipPeakIndicator(bt.Indicator):
    """
    Abstract base class for Chip Peak indicators.

    This class defines the interface that chip peak indicators should implement.
    Subclasses should implement the actual chip peak calculation logic.

    The chip peak indicator provides:
    - Pattern type (single peak, multi-peak)
    - Upper edge price (筹码峰上沿)
    - Lower edge price (筹码峰下沿)
    - Peak sharpness (峰尖尖锐度)
    - Main cost price (主力成本价)

    Attributes:
        pattern: Current chip peak pattern (single peak, multi-peak, etc.)
        upper_edge: Upper edge price of the chip peak
        lower_edge: Lower edge price of the chip peak
        main_cost: Main force's cost price
        peak_sharpness: Sharpness of the peak (higher = sharper)
    """

    lines = (
        "pattern",  # Pattern type (encoded as float: 0=unknown, 1=single_peak, 2=multi_peak)
        "upper_edge",  # Upper edge price
        "lower_edge",  # Lower edge price
        "main_cost",  # Main force cost price
        "peak_sharpness",  # Peak sharpness (0-1, higher = sharper)
    )

    params = (
        ("lookback_period", 60),  # Lookback period for chip calculation
    )

    def __init__(self):
        """Initialize chip peak indicator."""
        super().__init__()
        # Initialize with default values
        self.lines.pattern[0] = ChipPeakPattern.UNKNOWN.value
        self.lines.upper_edge[0] = 0.0
        self.lines.lower_edge[0] = 0.0
        self.lines.main_cost[0] = 0.0
        self.lines.peak_sharpness[0] = 0.0

    def next(self):
        """
        Calculate chip peak values for current bar.

        This method should be overridden by subclasses to implement
        the actual chip peak calculation logic.
        """
        # Default implementation: set to unknown
        self.lines.pattern[0] = ChipPeakPattern.UNKNOWN.value
        self.lines.upper_edge[0] = self.data.close[0]
        self.lines.lower_edge[0] = self.data.close[0]
        self.lines.main_cost[0] = self.data.close[0]
        self.lines.peak_sharpness[0] = 0.0

    def get_pattern(self) -> ChipPeakPattern:
        """
        Get current chip peak pattern.

        Returns:
            ChipPeakPattern enum value.
        """
        try:
            pattern_value = self.lines.pattern[0]
            if isinstance(pattern_value, str):
                return ChipPeakPattern(pattern_value)
            elif isinstance(pattern_value, (int, float)):
                # Handle encoded values
                if pattern_value == 1:
                    return ChipPeakPattern.SINGLE_PEAK
                elif pattern_value == 2:
                    return ChipPeakPattern.MULTI_PEAK
            return ChipPeakPattern.UNKNOWN
        except (IndexError, TypeError, ValueError):
            return ChipPeakPattern.UNKNOWN

    def is_single_peak(self) -> bool:
        """
        Check if current pattern is single peak concentrated.

        Returns:
            True if single peak, False otherwise.
        """
        return self.get_pattern() == ChipPeakPattern.SINGLE_PEAK

    def is_multi_peak(self) -> bool:
        """
        Check if current pattern is multi-peak.

        Returns:
            True if multi-peak, False otherwise.
        """
        return self.get_pattern() == ChipPeakPattern.MULTI_PEAK

    def get_upper_edge(self) -> Optional[float]:
        """
        Get upper edge price of chip peak.

        Returns:
            Upper edge price, or None if not available.
        """
        try:
            value = self.lines.upper_edge[0]
            return float(value) if value > 0 else None
        except (IndexError, TypeError, ValueError):
            return None

    def get_lower_edge(self) -> Optional[float]:
        """
        Get lower edge price of chip peak.

        Returns:
            Lower edge price, or None if not available.
        """
        try:
            value = self.lines.lower_edge[0]
            return float(value) if value > 0 else None
        except (IndexError, TypeError, ValueError):
            return None

    def get_main_cost(self) -> Optional[float]:
        """
        Get main force's cost price.

        Returns:
            Main cost price, or None if not available.
        """
        try:
            value = self.lines.main_cost[0]
            return float(value) if value > 0 else None
        except (IndexError, TypeError, ValueError):
            return None

    def get_peak_sharpness(self) -> float:
        """
        Get peak sharpness (0-1, higher = sharper).

        Returns:
            Peak sharpness value.
        """
        try:
            value = self.lines.peak_sharpness[0]
            return float(value) if 0 <= value <= 1 else 0.0
        except (IndexError, TypeError, ValueError):
            return 0.0


class DummyChipPeakIndicator(ChipPeakIndicator):
    """
    Dummy implementation of chip peak indicator.

    This is a placeholder implementation that returns default values.
    It can be used for testing or when actual chip peak calculation
    is not yet implemented.

    In a real implementation, this would calculate chip distribution
    based on historical volume and price data.
    """

    def __init__(self):
        """Initialize dummy chip peak indicator."""
        super().__init__()
        logger.warning(
            "Using DummyChipPeakIndicator. "
            "This is a placeholder - actual chip peak calculation is not implemented."
        )

    def next(self):
        """
        Return dummy values for chip peak.

        For testing purposes, this implementation:
        - Always returns SINGLE_PEAK pattern
        - Sets upper/lower edges based on recent price range
        - Sets main cost to current close price
        """
        try:
            # Use recent price range as dummy chip peak edges
            lookback = min(self.p.lookback_period, len(self.data))
            if lookback > 0:
                prices = [self.data.close[-i] for i in range(lookback)]
                price_high = max(prices)
                price_low = min(prices)
                price_range = price_high - price_low

                # Set pattern to single peak (for testing)
                self.lines.pattern[0] = ChipPeakPattern.SINGLE_PEAK.value

                # Set edges based on price range
                current_price = self.data.close[0]
                self.lines.upper_edge[0] = current_price + price_range * 0.1
                self.lines.lower_edge[0] = current_price - price_range * 0.1
                self.lines.main_cost[0] = current_price
                self.lines.peak_sharpness[0] = 0.5  # Medium sharpness
            else:
                # Not enough data
                current_price = self.data.close[0]
                self.lines.pattern[0] = ChipPeakPattern.UNKNOWN.value
                self.lines.upper_edge[0] = current_price
                self.lines.lower_edge[0] = current_price
                self.lines.main_cost[0] = current_price
                self.lines.peak_sharpness[0] = 0.0
        except (IndexError, TypeError, ValueError) as e:
            logger.debug(f"Error in DummyChipPeakIndicator.next(): {e}")
            # Fallback to current price
            current_price = self.data.close[0]
            self.lines.pattern[0] = ChipPeakPattern.UNKNOWN.value
            self.lines.upper_edge[0] = current_price
            self.lines.lower_edge[0] = current_price
            self.lines.main_cost[0] = current_price
            self.lines.peak_sharpness[0] = 0.0


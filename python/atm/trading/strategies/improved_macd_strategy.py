"""
Improved MACD Trading Strategy.

This strategy implements an improved MACD system with:
1. Adjusted MACD parameters: (8, 17, 5) instead of default (12, 26, 9)
2. Volume confirmation for buy signals
3. Divergence detection for sell signals
4. Consolidation box filtering
5. Consecutive failure tracking

Buy signals:
- DIFF line crosses above DEA line (golden cross)
- Volume on golden cross day > 1.2× average volume of previous 5 days
- Stock price above 20-day moving average

Sell signals:
- DIFF line crosses below DEA line (death cross)
- Top divergence: Price still rising but DIFF line declining
- Volume < 70% of average volume of previous 5 days (main force reducing volume)

Pitfalls to avoid:
- Do not use in consolidation box (price oscillating in a range)
- If golden cross fails twice consecutively, do not trade that stock

This strategy follows the standard backtrader pattern:
- Indicators are initialized in __init__
- Trading logic is in next() method
- Indicators trigger buy/sell operations
"""

import logging
from typing import Dict, Optional, Tuple

import backtrader as bt

from atm.trading.strategies.base import BaseStrategy

logger = logging.getLogger(__name__)


class ImprovedMACDStrategy(BaseStrategy):
    """
    Improved MACD Trading Strategy.

    This strategy uses adjusted MACD parameters (8, 17, 5) instead of default (12, 26, 9)
    to reduce lag and improve signal quality.

    Buy signals (all must be true):
    1. DIFF line crosses above DEA line (golden cross)
    2. Volume on golden cross day > 1.2× average volume of previous 5 days
    3. Stock price above 20-day moving average

    Sell signals (any of):
    1. DIFF line crosses below DEA line (death cross)
    2. Top divergence: Price rising but DIFF declining
    3. Volume < 70% of average volume of previous 5 days

    Pitfalls:
    - Avoid trading in consolidation boxes
    - Skip stocks with 2 consecutive golden cross failures

    Args:
        fast_period: Fast EMA period for MACD (default: 8).
        slow_period: Slow EMA period for MACD (default: 17).
        signal_period: Signal line period for MACD (default: 5).
        ma_period: Moving average period for trend filter (default: 20).
        volume_ma_period: Volume moving average period (default: 5).
        volume_threshold_buy: Volume threshold for buy signals (default: 1.2).
        volume_threshold_sell: Volume threshold for sell signals (default: 0.7).
        use_consolidation_filter: Enable consolidation box filtering (default: True).
        consolidation_lookback: Lookback period for consolidation detection (default: 20).
        consolidation_threshold: Price range threshold for consolidation (default: 0.05 = 5%).
        max_consecutive_failures: Maximum consecutive golden cross failures allowed (default: 2).
    """

    params = (
        ("fast_period", 8),  # Fast EMA period
        ("slow_period", 17),  # Slow EMA period
        ("signal_period", 5),  # Signal line period
        ("ma_period", 20),  # Moving average period for trend filter
        ("volume_ma_period", 5),  # Volume moving average period
        ("volume_threshold_buy", 1.2),  # Volume threshold for buy (1.2× average)
        ("volume_threshold_sell", 0.7),  # Volume threshold for sell (70% of average)
        ("use_consolidation_filter", True),  # Enable consolidation box filtering
        ("consolidation_lookback", 20),  # Lookback period for consolidation detection
        ("consolidation_threshold", 0.05),  # Price range threshold (5%)
        ("max_consecutive_failures", 2),  # Max consecutive golden cross failures
    )

    def __init__(self):
        """
        Initialize Improved MACD Strategy.

        Indicators are initialized here following backtrader's standard pattern.
        """
        super().__init__()

        # Improved MACD with adjusted parameters (8, 17, 5)
        self.macd = bt.indicators.MACD(
            self.data.close,
            period_me1=self.p.fast_period,
            period_me2=self.p.slow_period,
            period_signal=self.p.signal_period,
        )

        # MACD components
        self.macd_diff = self.macd.macd  # DIFF line
        self.macd_dea = self.macd.signal  # DEA line (signal line)
        # Note: Histogram can be calculated as macd_diff - macd_dea if needed
        # self.macd_hist = self.macd_diff - self.macd_dea

        # Golden cross and death cross detection
        self.macd_cross = bt.indicators.CrossOver(self.macd_diff, self.macd_dea)

        # Trend filter: 20-day moving average
        self.ma20 = bt.indicators.SMA(self.data.close, period=self.p.ma_period)

        # Volume confirmation
        self.volume_ma = bt.indicators.SMA(self.data.volume, period=self.p.volume_ma_period)
        self.volume_ratio = self.data.volume / self.volume_ma

        # Track position state
        self.entry_price = None
        self.golden_cross_failures = 0  # Track consecutive golden cross failures
        self.last_golden_cross_price = None  # Track last golden cross entry price
        self.ts_code = None  # Track current stock code for failure tracking

        logger.info(
            f"Improved MACD Strategy initialized with "
            f"MACD({self.p.fast_period}, {self.p.slow_period}, {self.p.signal_period}), "
            f"MA{self.p.ma_period}, "
            f"consolidation_filter={self.p.use_consolidation_filter}"
        )

    def next(self):
        """
        Called for each bar.

        Trading logic based on improved MACD:
        - Golden cross with volume confirmation
        - Death cross and divergence detection
        - Consolidation box filtering
        - Consecutive failure tracking
        """
        # Skip if not enough data
        max_period = max(
            self.p.slow_period,
            self.p.ma_period,
            self.p.volume_ma_period,
            self.p.consolidation_lookback,
        )
        if len(self.data) < max_period + 5:
            return

        # Safely access indicator values
        try:
            current_price = self.data.close[0]
            current_diff = self.macd_diff[0]
            current_dea = self.macd_dea[0]
            current_ma20 = self.ma20[0]
            current_volume_ratio = self.volume_ratio[0]
        except (IndexError, TypeError):
            return

        # Check consolidation box (pitfall avoidance)
        if self.p.use_consolidation_filter:
            if self._is_in_consolidation_box():
                # Skip trading in consolidation box
                return

        # Buy signals
        if not self.position:
            # Check if we should skip this stock due to consecutive failures
            if self.golden_cross_failures >= self.p.max_consecutive_failures:
                return

            # Signal 1: Golden cross (DIFF crosses above DEA)
            try:
                golden_cross = self.macd_cross[0] > 0
            except (IndexError, TypeError):
                golden_cross = False

            if golden_cross:
                # Signal 2: Volume confirmation (> 1.2× average volume)
                volume_confirmed = current_volume_ratio >= self.p.volume_threshold_buy

                # Signal 3: Price above 20-day MA (uptrend)
                price_above_ma = current_price > current_ma20

                if volume_confirmed and price_above_ma:
                    logger.debug(
                        f"Buy signal (golden cross + volume + trend) at "
                        f"{self.datas[0].datetime.date(0)}, "
                        f"price={current_price:.2f}, "
                        f"DIFF={current_diff:.4f}, DEA={current_dea:.4f}, "
                        f"volume_ratio={current_volume_ratio:.2f}"
                    )
                    self.buy()
                    self.entry_price = current_price
                    self.last_golden_cross_price = current_price
                    self.golden_cross_failures = 0  # Reset failure count on successful entry

        # Sell signals
        if self.position:
            # Signal 1: Death cross (DIFF crosses below DEA)
            try:
                death_cross = self.macd_cross[0] < 0
            except (IndexError, TypeError):
                death_cross = False

            if death_cross:
                logger.debug(
                    f"Sell signal (death cross) at {self.datas[0].datetime.date(0)}, "
                    f"price={current_price:.2f}"
                )
                self.close()
                self._check_golden_cross_failure(current_price)
                self.entry_price = None
                return

            # Signal 2: Top divergence (price rising but DIFF declining)
            if self._check_top_divergence():
                logger.debug(
                    f"Sell signal (top divergence) at {self.datas[0].datetime.date(0)}, "
                    f"price={current_price:.2f}"
                )
                self.close()
                self._check_golden_cross_failure(current_price)
                self.entry_price = None
                return

            # Signal 3: Volume reduction (main force reducing volume)
            if current_volume_ratio < self.p.volume_threshold_sell:
                logger.debug(
                    f"Sell signal (volume reduction) at {self.datas[0].datetime.date(0)}, "
                    f"price={current_price:.2f}, volume_ratio={current_volume_ratio:.2f}"
                )
                self.close()
                self._check_golden_cross_failure(current_price)
                self.entry_price = None
                return

    def _is_in_consolidation_box(self) -> bool:
        """
        Check if price is in a consolidation box (震荡箱体).

        Consolidation box: Price oscillating within a narrow range.

        Returns:
            True if price is in consolidation box, False otherwise.
        """
        if len(self.data) < self.p.consolidation_lookback:
            return False

        try:
            # Get price range over lookback period
            price_values = self.data.close.get(
                ago=0, size=self.p.consolidation_lookback
            )
            if not price_values:
                return False

            price_high = max(price_values)
            price_low = min(price_values)
            price_range = price_high - price_low
            avg_price = sum(price_values) / len(price_values)

            # Check if price range is within threshold (e.g., 5% of average price)
            range_ratio = price_range / avg_price if avg_price > 0 else 1.0

            return range_ratio < self.p.consolidation_threshold
        except (IndexError, TypeError):
            return False

    def _check_top_divergence(self) -> bool:
        """
        Check for top divergence (顶背离).

        Top divergence: Price is still rising, but DIFF line is declining.

        Returns:
            True if top divergence is detected, False otherwise.
        """
        if len(self.data) < 5 or self.entry_price is None:
            return False

        try:
            # Check if price is rising (compared to entry price or recent price)
            current_price = self.data.close[0]
            prev_price = self.data.close[-1] if len(self.data) > 1 else current_price
            
            # Price should be higher than entry or recent price
            price_rising = current_price > self.entry_price or current_price > prev_price

            # Check if DIFF is declining
            if len(self.macd_diff) < 2:
                return False
            
            current_diff = self.macd_diff[0]
            prev_diff = self.macd_diff[-1]
            diff_declining = current_diff < prev_diff

            # Top divergence: price rising but DIFF declining
            return price_rising and diff_declining
        except (IndexError, TypeError):
            return False

    def _check_golden_cross_failure(self, exit_price: float) -> None:
        """
        Check if the last golden cross was a failure.

        A golden cross is considered a failure if:
        - We exited at a loss (exit_price < entry_price)

        Args:
            exit_price: Price at which we exited the position.
        """
        if self.last_golden_cross_price is not None:
            if exit_price < self.last_golden_cross_price:
                # Golden cross failed (exited at loss)
                self.golden_cross_failures += 1
                logger.debug(
                    f"Golden cross failure #{self.golden_cross_failures} detected. "
                    f"Entry: {self.last_golden_cross_price:.2f}, Exit: {exit_price:.2f}"
                )
            else:
                # Successful trade, reset failure count
                self.golden_cross_failures = 0

    def get_info(self) -> Dict:
        """
        Get strategy information.

        Returns:
            Dictionary containing strategy information.
        """
        info = super().get_info()
        info.update(
            {
                "fast_period": self.p.fast_period,
                "slow_period": self.p.slow_period,
                "signal_period": self.p.signal_period,
                "ma_period": self.p.ma_period,
                "volume_ma_period": self.p.volume_ma_period,
                "volume_threshold_buy": self.p.volume_threshold_buy,
                "volume_threshold_sell": self.p.volume_threshold_sell,
                "use_consolidation_filter": self.p.use_consolidation_filter,
                "consolidation_lookback": self.p.consolidation_lookback,
                "consolidation_threshold": self.p.consolidation_threshold,
                "max_consecutive_failures": self.p.max_consecutive_failures,
            }
        )
        return info


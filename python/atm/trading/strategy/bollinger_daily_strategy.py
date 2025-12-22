"""
Daily Bollinger Bands Trading Strategy.

This strategy implements a trading system based on daily Bollinger Bands
to identify medium-term trading opportunities.

Buy signals (all must be true):
1. K-line at lower band (oversold)
2. Volume contraction (< 60% of average volume of previous 5 days)
3. Lower band turning upwards (crucial: not a lower band in continuous downtrend)

Sell signals (all must be true):
1. K-line at upper band (overbought)
2. Volume expansion (> 150% of average volume of previous 5 days)
3. Upper band turning downwards (crucial: not an upper band in continuous uptrend)

Pitfalls to avoid:
- Avoid during sudden negative news (lower band can be broken through)
- Medium-term holding limit: Do not hold for more than 20 days

This strategy follows the standard backtrader pattern:
- Indicators are initialized in __init__
- Trading logic is in next() method
- Indicators trigger buy/sell operations
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Optional

import backtrader as bt

from atm.trading.strategy.base import BaseStrategy

logger = logging.getLogger(__name__)


class BollingerDailyStrategy(BaseStrategy):
    """
    Daily Bollinger Bands Trading Strategy.

    This strategy uses daily Bollinger Bands to identify medium-term
    trading opportunities. Suitable for swing trading.

    Buy signals (all must be true):
    1. K-line at lower band (oversold)
    2. Volume contraction (< 60% of average volume)
    3. Lower band turning upwards

    Sell signals (all must be true):
    1. K-line at upper band (overbought)
    2. Volume expansion (> 150% of average volume)
    3. Upper band turning downwards

    Args:
        period: Bollinger Bands period (default: 20).
        devfactor: Bollinger Bands deviation factor (default: 2.0).
        volume_ma_period: Volume moving average period (default: 5).
        volume_threshold_buy: Volume threshold for buy signals (default: 0.6 = 60%).
        volume_threshold_sell: Volume threshold for sell signals (default: 1.5 = 150%).
        max_holding_days: Maximum holding period in days (default: 20).
        band_tolerance: Price tolerance for band detection (default: 0.01 = 1%).
    """

    params = (
        ("period", 20),  # Bollinger Bands period
        ("devfactor", 2.0),  # Bollinger Bands deviation factor
        ("volume_ma_period", 5),  # Volume moving average period
        ("volume_threshold_buy", 0.6),  # Volume threshold for buy (60% of average)
        ("volume_threshold_sell", 1.5),  # Volume threshold for sell (150% of average)
        ("max_holding_days", 20),  # Maximum holding period in days
        ("band_tolerance", 0.01),  # Price tolerance for band detection (1%)
    )

    def __init__(self):
        """
        Initialize Daily Bollinger Bands Strategy.

        Indicators are initialized here following backtrader's standard pattern.
        """
        super().__init__()

        # Bollinger Bands indicator
        self.bollinger = bt.indicators.BollingerBands(
            self.data.close,
            period=self.p.period,
            devfactor=self.p.devfactor,
        )

        # Bollinger Bands components
        self.bb_upper = self.bollinger.lines.top  # Upper band
        self.bb_middle = self.bollinger.lines.mid  # Middle band (SMA)
        self.bb_lower = self.bollinger.lines.bot  # Lower band

        # Volume confirmation
        self.volume_ma = bt.indicators.SMA(self.data.volume, period=self.p.volume_ma_period)
        self.volume_ratio = self.data.volume / self.volume_ma

        # Track position state
        self.entry_price = None
        self.entry_time = None  # Track entry time for holding period limit

        logger.info(
            f"Daily Bollinger Bands Strategy initialized with "
            f"period={self.p.period}, devfactor={self.p.devfactor}, "
            f"max_holding_days={self.p.max_holding_days}"
        )

    def next(self):
        """
        Called for each bar.

        Trading logic based on daily Bollinger Bands:
        - Lower band buy signals with volume contraction
        - Upper band sell signals with volume expansion
        - Band turning direction detection
        - Holding period limit
        """
        # Skip if not enough data
        max_period = max(self.p.period, self.p.volume_ma_period)
        if len(self.data) < max_period + 5:
            return

        # Safely access indicator values
        try:
            current_price = self.data.close[0]
            current_upper = self.bb_upper[0]
            current_lower = self.bb_lower[0]
            current_volume_ratio = self.volume_ratio[0]
        except (IndexError, TypeError):
            return

        # Check holding period limit
        if self.position and self.entry_time is not None:
            # Calculate holding period
            current_time = self.datas[0].datetime.datetime(0)
            if isinstance(current_time, datetime):
                holding_days = (current_time - self.entry_time).days
                if holding_days >= self.p.max_holding_days:
                    logger.debug(
                        f"Force sell due to max holding period at "
                        f"{self.datas[0].datetime.date(0)}, "
                        f"holding_days={holding_days}"
                    )
                    self.close()
                    self.entry_price = None
                    self.entry_time = None
                    return

        # Buy signals (all must be true)
        if not self.position:
            # Signal 1: K-line at lower band
            price_diff_lower = abs(current_price - current_lower) / current_lower
            price_at_lower_band = price_diff_lower <= self.p.band_tolerance

            # Signal 2: Volume contraction (< 60% of average)
            volume_contracted = current_volume_ratio < self.p.volume_threshold_buy

            # Signal 3: Lower band turning upwards
            lower_band_turning_up = self._check_lower_band_turning_up()

            if price_at_lower_band and volume_contracted and lower_band_turning_up:
                logger.debug(
                    f"Buy signal (lower band + volume contraction + band turning up) at "
                    f"{self.datas[0].datetime.date(0)}, "
                    f"price={current_price:.2f}, "
                    f"lower_band={current_lower:.2f}, "
                    f"volume_ratio={current_volume_ratio:.2f}"
                )
                self.buy()
                self.entry_price = current_price
                # Record entry time
                current_time = self.datas[0].datetime.datetime(0)
                if isinstance(current_time, datetime):
                    self.entry_time = current_time
                else:
                    # Fallback: use current bar index
                    self.entry_time = datetime.now()

        # Sell signals (all must be true)
        if self.position:
            # Signal 1: K-line at upper band
            price_diff_upper = abs(current_price - current_upper) / current_upper
            price_at_upper_band = price_diff_upper <= self.p.band_tolerance

            # Signal 2: Volume expansion (> 150% of average)
            volume_expanded = current_volume_ratio >= self.p.volume_threshold_sell

            # Signal 3: Upper band turning downwards
            upper_band_turning_down = self._check_upper_band_turning_down()

            if price_at_upper_band and volume_expanded and upper_band_turning_down:
                logger.debug(
                    f"Sell signal (upper band + volume expansion + band turning down) at "
                    f"{self.datas[0].datetime.date(0)}, "
                    f"price={current_price:.2f}, "
                    f"upper_band={current_upper:.2f}, "
                    f"volume_ratio={current_volume_ratio:.2f}"
                )
                self.close()
                self.entry_price = None
                self.entry_time = None

    def _check_lower_band_turning_up(self) -> bool:
        """
        Check if lower band is turning upwards.

        This is crucial: it should not be a lower band in continuous downtrend.

        Returns:
            True if lower band is turning upwards, False otherwise.
        """
        if len(self.bb_lower) < 3:
            return False

        try:
            # Get recent lower band values
            current_lower = self.bb_lower[0]
            prev_lower = self.bb_lower[-1]
            prev2_lower = self.bb_lower[-2] if len(self.bb_lower) > 2 else prev_lower

            # Lower band turning up: current > previous, and previous was declining
            # This ensures it's a turning point, not just a temporary bounce
            turning_up = (
                current_lower > prev_lower
                and prev_lower <= prev2_lower  # Previous was declining or flat
            )

            return turning_up
        except (IndexError, TypeError):
            return False

    def _check_upper_band_turning_down(self) -> bool:
        """
        Check if upper band is turning downwards.

        This is crucial: it should not be an upper band in continuous uptrend.

        Returns:
            True if upper band is turning downwards, False otherwise.
        """
        if len(self.bb_upper) < 3:
            return False

        try:
            # Get recent upper band values
            current_upper = self.bb_upper[0]
            prev_upper = self.bb_upper[-1]
            prev2_upper = self.bb_upper[-2] if len(self.bb_upper) > 2 else prev_upper

            # Upper band turning down: current < previous, and previous was rising
            # This ensures it's a turning point, not just a temporary pullback
            turning_down = (
                current_upper < prev_upper
                and prev_upper >= prev2_upper  # Previous was rising or flat
            )

            return turning_down
        except (IndexError, TypeError):
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
                "period": self.p.period,
                "devfactor": self.p.devfactor,
                "volume_ma_period": self.p.volume_ma_period,
                "volume_threshold_buy": self.p.volume_threshold_buy,
                "volume_threshold_sell": self.p.volume_threshold_sell,
                "max_holding_days": self.p.max_holding_days,
                "band_tolerance": self.p.band_tolerance,
            }
        )
        return info


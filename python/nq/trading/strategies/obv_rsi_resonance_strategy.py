"""
OBV+RSI Dual Indicator Resonance Trading Strategy.

This strategy implements a "capital + trend" dual-verification bull-catching logic
by combining OBV (On-Balance Volume) for capital tracking and RSI (Relative Strength Index)
for trend calibration.

Core Logic:
1. Step 1: OBV Capital Validation - Filter targets with "real capital support"
2. Step 2: RSI Trend Calibration - Filter targets with "sustainable trends"
3. Step 3: Dual Indicator Resonance Confirmation - Lock in "capital + trend synchronization" opportunities
4. Step 4: Resonance Continuation Tracking - Hold onto big bulls without premature exit

Entry Signals:
- OBV breaks above long-term MA and moves upward synchronously
- RSI operates steadily in middle range and synchronizes with price
- Both indicators break through simultaneously (time difference <= 2 periods)
- Enter on synchronous pullback confirmation

Exit Signals:
- OBV shows capital divergence (price up but OBV down)
- RSI shows trend divergence (price up but RSI down) and breaks below middle range
"""

"""
OBV + RSI Resonance Strategy.

Captures “capital + trend” synchronized signals:
- OBV validates capital inflow; RSI validates trend sustainability.
- Entry when both breakouts occur within a short window and hold supports.
- Exit on capital/trend divergence or support break.
"""

import logging
from typing import Optional

import backtrader as bt
import backtrader.indicators as btind

from nq.trading.indicators.indicators import OnBalanceVolume
from nq.trading.strategies.base import BaseStrategy

logger = logging.getLogger(__name__)


class OBVRSIResonanceStrategy(BaseStrategy):
    """
    OBV+RSI dual-indicator resonance strategy.

    - OBV: capital tracking (breakout above long MA + rising with volume)
    - RSI: trend calibration (middle range, synchronized with price)
    - Resonance: both signals occur within a short time window; pullback holds support
    """

    params = (
        # OBV parameters
        ("obv_long_ma_period", 30),  # Long-term MA for OBV
        ("obv_short_ma_period", 5),  # Short-term support MA for OBV
        ("obv_rise_periods", 3),  # Minimum periods for OBV upward trend
        # RSI parameters
        ("rsi_period", 14),  # RSI period
        ("rsi_middle_low", 40),  # Lower boundary of middle range
        ("rsi_middle_high", 60),  # Upper boundary of middle range
        ("rsi_overbought", 70),  # Overbought threshold
        ("rsi_oversold", 30),  # Oversold threshold
        # Resonance parameters
        ("resonance_time_diff", 2),  # Max time difference for synchronous breakout
        # Volume confirmation
        ("volume_ma_period", 20),  # Period for volume moving average
        ("volume_increase_ratio", 1.2),  # Minimum volume increase ratio
    )

    def __init__(self):
        """Initialize OBV+RSI Resonance Strategy."""
        super().__init__()

        # OBV indicator (custom)
        self.obv = OnBalanceVolume(self.data)
        self.obv_long_ma = btind.SMA(self.obv, period=self.p.obv_long_ma_period)
        self.obv_short_ma = btind.SMA(self.obv, period=self.p.obv_short_ma_period)

        # RSI indicator
        self.rsi = btind.RSI(self.data.close, period=self.p.rsi_period)

        # Volume confirmation
        self.volume_ma = btind.SMA(self.data.volume, period=self.p.volume_ma_period)

        # Price moving average for trend confirmation
        self.price_ma = btind.SMA(self.data.close, period=5)

        # State tracking
        self.obv_breakout_bar = None  # Bar index when OBV broke above long MA
        self.rsi_breakout_bar = None  # Bar index when RSI broke above middle range
        self.entry_bar = None  # Bar index when position was entered
        self.obv_rising_count = 0  # Count of consecutive OBV rises
        self.last_obv_value = None

        # Ensure enough data for indicators
        max_period = max(
            self.p.obv_long_ma_period,
            self.p.rsi_period,
            self.p.volume_ma_period,
            5,  # price_ma
        )
        self.addminperiod(max_period + 10)

    def _check_obv_capital_validation(self) -> bool:
        """
        Step 1: OBV Capital Validation.

        Check if OBV shows real capital support:
        - OBV line continuously rises for at least obv_rise_periods periods
        - OBV breaks above long-term MA
        - Volume shows coordinated increase
        """
        if len(self.obv) < self.p.obv_rise_periods + 1:
            return False

        # Check OBV upward trend (current period not lower than previous)
        if self.last_obv_value is not None:
            if self.obv[0] >= self.last_obv_value:
                self.obv_rising_count += 1
            else:
                self.obv_rising_count = 0

        self.last_obv_value = self.obv[0]

        # OBV must have risen for at least obv_rise_periods periods
        if self.obv_rising_count < self.p.obv_rise_periods:
            return False

        # OBV must break above long-term MA
        if len(self.obv_long_ma) < 2:
            return False

        # Check if OBV broke above long MA recently
        obv_above_ma = self.obv[0] > self.obv_long_ma[0]
        
        # Check if this is a new breakout (was below MA before)
        if len(self.obv) > 1 and len(self.obv_long_ma) > 1:
            obv_was_below_ma = self.obv[-1] <= self.obv_long_ma[-1]
            if obv_above_ma and obv_was_below_ma and self.obv_breakout_bar is None:
                self.obv_breakout_bar = len(self.data) - 1

        # Volume coordination: current volume should be higher than average
        if len(self.volume_ma) < 1:
            return False

        volume_ok = self.data.volume[0] >= self.volume_ma[0] * self.p.volume_increase_ratio

        return obv_above_ma and volume_ok

    def _check_rsi_trend_calibration(self) -> bool:
        """
        Step 2: RSI Trend Calibration.

        Check if RSI shows sustainable trend:
        - RSI operates in middle range (not overbought/oversold)
        - RSI moves synchronously with price
        - RSI does not show extreme fluctuations
        """
        if len(self.rsi) < 2:
            return False

        current_rsi = self.rsi[0]
        prev_rsi = self.rsi[-1]

        # RSI must be in middle range (stable, not extreme)
        rsi_in_middle = (
            self.p.rsi_middle_low <= current_rsi <= self.p.rsi_middle_high
        )

        # RSI should not frequently rush into overbought zone
        rsi_not_overbought = current_rsi < self.p.rsi_overbought

        # RSI should move synchronously with price
        # When price rises, RSI should also rise or remain stable
        price_rising = self.data.close[0] > self.data.close[-1]
        rsi_sync = (price_rising and current_rsi >= prev_rsi) or (
            not price_rising and current_rsi <= prev_rsi
        )

        # Check if RSI broke above middle range recently
        if rsi_in_middle and prev_rsi < self.p.rsi_middle_low and self.rsi_breakout_bar is None:
            self.rsi_breakout_bar = len(self.data) - 1

        return rsi_in_middle and rsi_not_overbought and rsi_sync

    def _check_resonance_confirmation(self) -> bool:
        """
        Step 3: Dual Indicator Resonance Confirmation.

        Check if both indicators show synchronous signals:
        - Time synchronization: OBV and RSI breakouts occur within resonance_time_diff periods
        - Pullback synchronization: During pullback, both indicators hold above support
        """
        if self.obv_breakout_bar is None or self.rsi_breakout_bar is None:
            return False

        # Time synchronization: breakouts should occur close together
        time_diff = abs(self.obv_breakout_bar - self.rsi_breakout_bar)
        if time_diff > self.p.resonance_time_diff:
            return False

        # Both indicators must currently be above their support levels
        obv_above_support = self.obv[0] > self.obv_short_ma[0]
        rsi_above_support = self.rsi[0] >= self.p.rsi_middle_low

        return obv_above_support and rsi_above_support

    def _check_resonance_continuation(self) -> bool:
        """
        Step 4: Resonance Continuation Tracking.

        Check if resonance continues (no divergence):
        - OBV non-divergence: price up, OBV also up
        - RSI non-divergence: price up, RSI stable in middle range
        """
        if len(self.obv) < 2 or len(self.rsi) < 2:
            return True  # Not enough data, assume continuation

        price_rising = self.data.close[0] > self.data.close[-1]

        # OBV non-divergence: if price rises, OBV should also rise
        obv_non_divergence = True
        if price_rising:
            obv_non_divergence = self.obv[0] >= self.obv[-1]

        # RSI non-divergence: if price rises, RSI should stay in middle range
        rsi_non_divergence = True
        if price_rising:
            current_rsi = self.rsi[0]
            rsi_non_divergence = (
                self.p.rsi_middle_low <= current_rsi <= self.p.rsi_middle_high
            ) and (current_rsi >= self.rsi[-1] or current_rsi >= self.p.rsi_middle_low)

        return obv_non_divergence and rsi_non_divergence

    def _check_exit_signals(self) -> bool:
        """
        Check for exit signals:
        - OBV capital divergence (price up but OBV down)
        - RSI trend divergence (price up but RSI down) and RSI breaks below middle range
        """
        if len(self.obv) < 2 or len(self.rsi) < 2:
            return False

        price_rising = self.data.close[0] > self.data.close[-1]

        # OBV capital divergence
        obv_divergence = False
        if price_rising:
            obv_divergence = self.obv[0] < self.obv[-1]

        # RSI trend divergence
        rsi_divergence = False
        if price_rising:
            current_rsi = self.rsi[0]
            prev_rsi = self.rsi[-1]
            rsi_divergence = current_rsi < prev_rsi and current_rsi < self.p.rsi_middle_low

        return obv_divergence or rsi_divergence

    def next(self):
        """Called for each bar."""
        # Skip if not enough data
        if len(self.data) < self.p.obv_long_ma_period + self.p.rsi_period:
            return

        # Step 1: OBV Capital Validation
        obv_valid = self._check_obv_capital_validation()

        # Step 2: RSI Trend Calibration
        rsi_valid = self._check_rsi_trend_calibration()

        # Step 3: Resonance Confirmation
        resonance_confirmed = self._check_resonance_confirmation()

        # Entry logic: All three steps must pass
        if not self.position:
            if obv_valid and rsi_valid and resonance_confirmed:
                # Additional confirmation: price should be above short-term MA
                if len(self.price_ma) > 0 and self.data.close[0] > self.price_ma[0]:
                    logger.debug(
                        f"Entry signal at {self.data.datetime.date(0)}: "
                        f"OBV={self.obv[0]:.2f}, RSI={self.rsi[0]:.2f}, "
                        f"Price={self.data.close[0]:.2f}"
                    )
                    self.buy()
                    self.entry_bar = len(self.data) - 1

        # Exit logic: Check for divergence or resonance breakdown
        else:
            # Step 4: Check resonance continuation
            if not self._check_resonance_continuation():
                logger.debug(
                    f"Exit signal (resonance breakdown) at {self.data.datetime.date(0)}: "
                    f"OBV={self.obv[0]:.2f}, RSI={self.rsi[0]:.2f}"
                )
                self.sell()
                self._reset_state()

            # Check explicit exit signals
            elif self._check_exit_signals():
                logger.debug(
                    f"Exit signal (divergence) at {self.data.datetime.date(0)}: "
                    f"OBV={self.obv[0]:.2f}, RSI={self.rsi[0]:.2f}"
                )
                self.sell()
                self._reset_state()

    def _reset_state(self):
        """Reset state after exit."""
        self.obv_breakout_bar = None
        self.rsi_breakout_bar = None
        self.entry_bar = None
        self.obv_rising_count = 0


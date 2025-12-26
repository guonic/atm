"""
MACD + OBV Dual Indicator Resonance Strategy.

This strategy combines MACD with short-term parameters (6,13,5) for faster signals
and OBV for capital flow validation. It focuses on capturing intraday opportunities
through dual-indicator resonance and divergence validation.

Core Logic:
1. Buy Signal (Dual-Indicator Resonance):
   - MACD golden cross above zero axis (short-term strong trend)
   - MACD histogram rapidly expanding (red bars)
   - OBV breaks above previous high (capital inflow confirmation)
   - Volume-price resonance formed

2. Sell Signal (Top Divergence Validation):
   - Price makes new high
   - MACD histogram shrinks (divergence)
   - OBV does not make new high (divergence)
   - Trigger sell if 2 out of 3 conditions met

3. Bottom Divergence Buy:
   - Price makes new low
   - MACD histogram shrinks (green bars shrinking)
   - OBV does not make new low (divergence)
"""

import logging
from typing import Optional, Tuple

import backtrader.indicators as btind

from nq.trading.indicators.indicators import OnBalanceVolume
from nq.trading.strategies.base import BaseStrategy

logger = logging.getLogger(__name__)


class MACDOBVResonanceStrategy(BaseStrategy):
    """
    MACD + OBV Dual Indicator Resonance Strategy.

    Uses MACD with short-term parameters (6,13,5) for faster signals and OBV
    for capital flow validation. Captures intraday opportunities through
    dual-indicator resonance and divergence validation.
    """

    params = (
        # MACD parameters (short-term for faster signals)
        ("macd_fast", 6),  # Fast EMA period
        ("macd_slow", 13),  # Slow EMA period
        ("macd_signal", 5),  # Signal line period
        # OBV parameters
        ("obv_lookback", 20),  # Lookback period for OBV high/low
        # Divergence detection
        ("divergence_lookback", 20),  # Lookback period for divergence
        # Entry/exit thresholds
        ("macd_hist_expansion_ratio", 1.2),  # Minimum histogram expansion ratio
    )

    def __init__(self):
        """Initialize MACD + OBV Resonance Strategy."""
        super().__init__()

        # MACD indicator with short-term parameters
        self.macd = btind.MACD(
            self.data.close,
            period_me1=self.p.macd_fast,
            period_me2=self.p.macd_slow,
            period_signal=self.p.macd_signal,
        )

        # MACD components
        self.macd_diff = self.macd.macd  # DIF line
        self.macd_dea = self.macd.signal  # DEA line
        # Calculate histogram (MACD - Signal)
        self.macd_hist = self.macd.macd - self.macd.signal

        # OBV indicator
        self.obv = OnBalanceVolume(self.data)

        # Track price and indicator history for divergence detection
        self.price_history = []
        self.macd_hist_history = []
        self.obv_history = []

        # Track previous OBV high for breakout detection
        self.prev_obv_high: Optional[float] = None

        # Ensure enough data for indicators
        max_period = max(
            self.p.macd_slow,
            self.p.macd_signal,
            self.p.obv_lookback,
            self.p.divergence_lookback,
        )
        self.addminperiod(max_period + 10)

    def _update_history(self):
        """Update price and indicator history for divergence detection."""
        if len(self.data) >= 2:
            self.price_history.append(self.data.close[0])
            self.macd_hist_history.append(self.macd_hist[0])
            self.obv_history.append(self.obv[0])

            # Keep only recent history
            if len(self.price_history) > self.p.divergence_lookback:
                self.price_history.pop(0)
                self.macd_hist_history.pop(0)
                self.obv_history.pop(0)

    def _check_macd_golden_cross_above_zero(self) -> bool:
        """Check if MACD has golden cross above zero axis."""
        if len(self.macd_diff) < 2:
            return False

        # Current MACD line and signal line
        curr_diff = self.macd_diff[0]
        curr_dea = self.macd_dea[0]
        prev_diff = self.macd_diff[-1]
        prev_dea = self.macd_dea[-1]

        # Golden cross: DIF crosses above DEA
        golden_cross = curr_diff > curr_dea and prev_diff <= prev_dea

        # Above zero axis
        above_zero = curr_diff > 0 and curr_dea > 0

        return golden_cross and above_zero

    def _check_macd_histogram_expansion(self) -> bool:
        """Check if MACD histogram is rapidly expanding."""
        if len(self.macd_hist) < 2:
            return False

        curr_hist = self.macd_hist[0]
        prev_hist = self.macd_hist[-1]

        # Histogram should be positive (red bars in bullish context)
        if curr_hist <= 0:
            return False

        # Rapid expansion: current > previous * expansion_ratio
        expansion = curr_hist >= prev_hist * self.p.macd_hist_expansion_ratio

        return expansion

    def _check_obv_breakout(self) -> bool:
        """Check if OBV breaks above previous high."""
        if len(self.obv) < self.p.obv_lookback + 1:
            return False

        current_obv = self.obv[0]

        # Find previous high in lookback period
        prev_high = max(self.obv[-i] for i in range(1, min(self.p.obv_lookback + 1, len(self.obv))))

        # Breakout: current OBV > previous high
        breakout = current_obv > prev_high

        if breakout:
            self.prev_obv_high = prev_high

        return breakout

    def _check_top_divergence(self) -> Tuple[bool, int]:
        """
        Check for top divergence.

        Returns:
            (has_divergence, divergence_count) where divergence_count is
            the number of divergence conditions met (0-3)
        """
        if len(self.price_history) < 2:
            return False, 0

        # Find recent price high
        recent_high_idx = len(self.price_history) - 1
        for i in range(len(self.price_history) - 2, -1, -1):
            if self.price_history[i] > self.price_history[recent_high_idx]:
                recent_high_idx = i

        if recent_high_idx == len(self.price_history) - 1:
            # Current price is new high
            current_price = self.price_history[-1]
            prev_high_price = max(self.price_history[:-1]) if len(self.price_history) > 1 else current_price

            if current_price > prev_high_price:
                # Price makes new high, check for divergence
                divergence_count = 0

                # Check MACD histogram divergence
                if len(self.macd_hist_history) >= 2:
                    current_hist = self.macd_hist_history[-1]
                    prev_high_hist = max(self.macd_hist_history[:-1])
                    if current_hist < prev_high_hist:
                        divergence_count += 1

                # Check OBV divergence
                if len(self.obv_history) >= 2:
                    current_obv = self.obv_history[-1]
                    prev_high_obv = max(self.obv_history[:-1])
                    if current_obv < prev_high_obv:
                        divergence_count += 1

                # Trigger if 2 or more divergence conditions met
                return divergence_count >= 2, divergence_count

        return False, 0

    def _check_bottom_divergence(self) -> bool:
        """Check for bottom divergence."""
        if len(self.price_history) < 2:
            return False

        # Find recent price low
        recent_low_idx = len(self.price_history) - 1
        for i in range(len(self.price_history) - 2, -1, -1):
            if self.price_history[i] < self.price_history[recent_low_idx]:
                recent_low_idx = i

        if recent_low_idx == len(self.price_history) - 1:
            # Current price is new low
            current_price = self.price_history[-1]
            prev_low_price = min(self.price_history[:-1]) if len(self.price_history) > 1 else current_price

            if current_price < prev_low_price:
                # Price makes new low, check for divergence
                divergence_count = 0

                # Check MACD histogram divergence (green bars shrinking)
                if len(self.macd_hist_history) >= 2:
                    current_hist = self.macd_hist_history[-1]
                    prev_low_hist = min(self.macd_hist_history[:-1])
                    if current_hist > prev_low_hist:
                        divergence_count += 1

                # Check OBV divergence
                if len(self.obv_history) >= 2:
                    current_obv = self.obv_history[-1]
                    prev_low_obv = min(self.obv_history[:-1])
                    if current_obv > prev_low_obv:
                        divergence_count += 1

                # Trigger if both divergence conditions met
                return divergence_count >= 2

        return False

    def next(self):
        """Called for each bar."""
        # Skip if not enough data
        if len(self.data) < self.p.macd_slow + self.p.divergence_lookback:
            return

        # Update history for divergence detection
        self._update_history()

        # Buy signals
        if not self.position:
            # Signal 1: Dual-indicator resonance (MACD golden cross + OBV breakout)
            macd_golden_cross = self._check_macd_golden_cross_above_zero()
            macd_expansion = self._check_macd_histogram_expansion()
            obv_breakout = self._check_obv_breakout()

            if macd_golden_cross and macd_expansion and obv_breakout:
                logger.debug(
                    f"Buy signal (resonance) at {self.data.datetime.date(0)}: "
                    f"MACD golden cross, histogram expansion, OBV breakout"
                )
                self.buy()

            # Signal 2: Bottom divergence
            elif self._check_bottom_divergence():
                logger.debug(
                    f"Buy signal (bottom divergence) at {self.data.datetime.date(0)}: "
                    f"Price new low but MACD/OBV not following"
                )
                self.buy()

        # Sell signals
        else:
            # Top divergence validation
            has_divergence, divergence_count = self._check_top_divergence()
            if has_divergence:
                logger.debug(
                    f"Sell signal (top divergence) at {self.data.datetime.date(0)}: "
                    f"Divergence count: {divergence_count}"
                )
                self.sell()

    def get_info(self) -> dict:
        """Get strategy information."""
        info = super().get_info()
        info.update(
            {
                "macd_fast": self.p.macd_fast,
                "macd_slow": self.p.macd_slow,
                "macd_signal": self.p.macd_signal,
                "obv_lookback": self.p.obv_lookback,
                "divergence_lookback": self.p.divergence_lookback,
            }
        )
        return info


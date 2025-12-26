"""
RSI + MACD Resonance Strategy.

A trading strategy that combines RSI and MACD indicators to generate high-probability trading signals.
The strategy follows a three-step approach:
1. RSI provides positioning and divergence warnings
2. MACD provides trend qualification and momentum confirmation
3. Resonance signals are generated when both indicators agree

This strategy follows the standard backtrader pattern:
- Indicators are initialized in __init__
- Trading logic is in next() method
- Indicators trigger buy/sell operations
"""

import logging
from typing import List

import backtrader as bt

from nq.trading.strategies.base import BaseStrategy

logger = logging.getLogger(__name__)


class RSIMACDResonanceStrategy(BaseStrategy):
    """
    RSI + MACD Resonance Strategy.

    This strategy combines RSI and MACD indicators:
    - RSI: Responsible for "positioning the battlefield" and "warning of divergence"
    - MACD: Responsible for "qualitative trend" and "confirming momentum"
    - Resonance: Trading signals are generated when both indicators agree

    Buy signals (bottom rebound):
    1. RSI enters oversold zone (<30) and forms bottom divergence
    2. MACD histogram shows bottom divergence and lines are below zero axis
    3. Resonance confirmation:
       - Classic: RSI crosses above 50 (or stabilizes above 30) + MACD golden cross at low level
       - Sensitive: RSI bottom divergence low point turns up + MACD bar changes from green to red

    Sell signals (top reversal):
    1. RSI top divergence
    2. MACD death cross
    3. MACD red bars shrinking

    Args:
        rsi_period: RSI calculation period (default: 14).
        rsi_oversold: RSI oversold threshold (default: 30).
        rsi_overbought: RSI overbought threshold (default: 70).
        rsi_midline: RSI strength/weakness dividing line (default: 50).
        macd_fast: MACD fast period (default: 12).
        macd_slow: MACD slow period (default: 26).
        macd_signal: MACD signal period (default: 9).
        divergence_lookback: Lookback period for divergence detection (default: 20).
        use_sensitive_entry: Use sensitive entry signal (MACD bar color change) (default: True).
        stop_loss_atr_multiplier: Stop loss multiplier based on ATR (default: 2.0).
    """

    params = (
        ("rsi_period", 14),  # RSI period
        ("rsi_oversold", 30),  # RSI oversold threshold
        ("rsi_overbought", 70),  # RSI overbought threshold
        ("rsi_midline", 50),  # RSI strength/weakness dividing line
        ("macd_fast", 12),  # MACD fast period
        ("macd_slow", 26),  # MACD slow period
        ("macd_signal", 9),  # MACD signal period
        ("divergence_lookback", 20),  # Divergence lookback period
        ("use_sensitive_entry", True),  # Use sensitive entry signal
        ("stop_loss_atr_multiplier", 2.0),  # Stop loss ATR multiplier
    )

    def __init__(self):
        """
        Initialize RSI + MACD Resonance Strategy.

        Indicators are initialized here following backtrader's standard pattern.
        """
        super().__init__()

        # Initialize RSI indicator
        self.rsi = bt.indicators.RSI(
            self.data.close,
            period=self.p.rsi_period,
        )

        # Initialize MACD indicator
        self.macd = bt.indicators.MACD(
            self.data.close,
            period_me1=self.p.macd_fast,
            period_me2=self.p.macd_slow,
            period_signal=self.p.macd_signal,
        )

        # MACD components
        # In backtrader, MACD indicator has:
        # - macd: the MACD line (DIF)
        # - signal: the signal line (DEA)
        # - histo: the histogram (BAR) = macd - signal
        self.macd_diff = self.macd.macd  # DIF line
        self.macd_dea = self.macd.signal  # DEA line
        # Calculate histogram manually if not available
        try:
            self.macd_hist = self.macd.histo  # Histogram (BAR)
        except AttributeError:
            # If histo is not available, calculate it
            self.macd_hist = self.macd.macd - self.macd.signal

        # ATR for stop loss
        self.atr = bt.indicators.ATR(self.data, period=14)

        # Track position state
        self.entry_price = None
        self.entry_bar_index = None
        self.stop_loss_price = None

        # Track RSI divergence
        self.rsi_bottom_divergence = False
        self.rsi_top_divergence = False
        self.rsi_oversold_zone = False
        self.rsi_overbought_zone = False

        # Track MACD state
        self.macd_bottom_divergence = False
        self.macd_top_divergence = False
        self.macd_below_zero = False
        self.macd_above_zero = False
        self.macd_bar_color_changed = False  # Green to red (bullish) or red to green (bearish)

        # Price and indicator history for divergence detection
        self.price_history: List[float] = []
        self.rsi_history: List[float] = []
        self.macd_hist_history: List[float] = []

        logger.info(
            f"RSI + MACD Resonance Strategy initialized with "
            f"RSI({self.p.rsi_period}), "
            f"MACD({self.p.macd_fast}, {self.p.macd_slow}, {self.p.macd_signal})"
        )

    def next(self):
        """
        Called for each bar.

        Trading logic based on RSI + MACD resonance:
        1. RSI positioning and divergence detection
        2. MACD trend qualification and momentum confirmation
        3. Resonance signal generation
        """
        # Skip if not enough data
        # MACD needs: slow_period (26) + signal_period (9) + buffer
        # RSI needs: rsi_period (14) + buffer
        # ATR needs: 14 + buffer
        max_period = max(
            self.p.rsi_period,
            self.p.macd_slow + self.p.macd_signal,  # MACD needs both periods
            self.p.divergence_lookback,
            14,  # ATR period
        )
        # Add larger buffer for indicator stability
        min_required = max_period + 10
        if len(self.data) < min_required:
            return

        # Check if indicators have valid values before accessing
        try:
            # Try to access indicator values to ensure they're ready
            _ = self.rsi[0]
            _ = self.macd_diff[0]
            _ = self.macd_dea[0]
            _ = self.macd_hist[0]
            _ = self.atr[0]
        except (IndexError, TypeError):
            # Indicators not ready yet
            return

        # Safely access indicator values
        try:
            current_price = self.data.close[0]
            current_rsi = self.rsi[0]
            current_macd_diff = self.macd_diff[0]
            current_macd_dea = self.macd_dea[0]
            current_macd_hist = self.macd_hist[0]
            current_atr = self.atr[0]
        except (IndexError, TypeError) as e:
            logger.debug(f"Failed to access indicator values: {e}")
            return

        # Update history for divergence detection
        self._update_history(current_price, current_rsi, current_macd_hist)

        # Step 1: RSI positioning and divergence detection
        self._detect_rsi_signals(current_price, current_rsi)

        # Step 2: MACD trend qualification and momentum confirmation
        self._detect_macd_signals(current_price, current_macd_diff, current_macd_dea, current_macd_hist)

        # Step 3: Generate resonance signals
        if not self.position:
            # Check for buy signals (bottom rebound)
            if self._check_buy_resonance(current_rsi, current_macd_diff, current_macd_dea, current_macd_hist):
                self._enter_long(current_price, current_atr)
        else:
            # Check for sell signals (top reversal)
            if self._check_sell_resonance(current_rsi, current_macd_diff, current_macd_dea, current_macd_hist):
                self._exit_long(current_price)

            # Check stop loss
            if self.stop_loss_price and current_price < self.stop_loss_price:
                logger.info(f"Stop loss triggered at {current_price:.2f}")
                self._exit_long(current_price)

    def _update_history(self, price: float, rsi: float, macd_hist: float):
        """Update price and indicator history for divergence detection."""
        self.price_history.append(price)
        self.rsi_history.append(rsi)
        self.macd_hist_history.append(macd_hist)

        # Keep only recent history
        max_history = self.p.divergence_lookback + 10
        if len(self.price_history) > max_history:
            self.price_history.pop(0)
            self.rsi_history.pop(0)
            self.macd_hist_history.pop(0)

    def _detect_rsi_signals(self, price: float, rsi: float):
        """Detect RSI positioning and divergence signals."""
        # Reset divergence flags at the start
        self.rsi_bottom_divergence = False
        self.rsi_top_divergence = False

        # Check oversold/overbought zones
        self.rsi_oversold_zone = rsi < self.p.rsi_oversold
        self.rsi_overbought_zone = rsi > self.p.rsi_overbought

        # Detect bottom divergence (price makes new low, RSI doesn't)
        if len(self.price_history) >= self.p.divergence_lookback:
            recent_prices = self.price_history[-self.p.divergence_lookback:]
            recent_rsi = self.rsi_history[-self.p.divergence_lookback:]

            # Find the lowest price point in the lookback period
            min_price_idx = recent_prices.index(min(recent_prices))
            min_price = recent_prices[min_price_idx]
            min_rsi_at_low = recent_rsi[min_price_idx]

            # Find the RSI value at the previous low point (before the current low)
            if min_price_idx > 0:
                prev_prices = recent_prices[:min_price_idx]
                prev_rsi = recent_rsi[:min_price_idx]
                if prev_prices:
                    prev_min_price_idx = prev_prices.index(min(prev_prices))
                    prev_min_rsi = prev_rsi[prev_min_price_idx]

                    # Bottom divergence: current price is lower, but RSI at current low is higher
                    if min_price < min(prev_prices) and min_rsi_at_low > prev_min_rsi:
                        self.rsi_bottom_divergence = True
                        logger.debug(f"RSI bottom divergence detected: price {min_price:.2f} < {min(prev_prices):.2f}, RSI {min_rsi_at_low:.2f} > {prev_min_rsi:.2f}")

        # Detect top divergence (price makes new high, RSI doesn't)
        if len(self.price_history) >= self.p.divergence_lookback:
            recent_prices = self.price_history[-self.p.divergence_lookback:]
            recent_rsi = self.rsi_history[-self.p.divergence_lookback:]

            # Find the highest price point in the lookback period
            max_price_idx = recent_prices.index(max(recent_prices))
            max_price = recent_prices[max_price_idx]
            max_rsi_at_high = recent_rsi[max_price_idx]

            # Find the RSI value at the previous high point (before the current high)
            if max_price_idx > 0:
                prev_prices = recent_prices[:max_price_idx]
                prev_rsi = recent_rsi[:max_price_idx]
                if prev_prices:
                    prev_max_price_idx = prev_prices.index(max(prev_prices))
                    prev_max_rsi = prev_rsi[prev_max_price_idx]

                    # Top divergence: current price is higher, but RSI at current high is lower
                    if max_price > max(prev_prices) and max_rsi_at_high < prev_max_rsi:
                        self.rsi_top_divergence = True
                        logger.debug(f"RSI top divergence detected: price {max_price:.2f} > {max(prev_prices):.2f}, RSI {max_rsi_at_high:.2f} < {prev_max_rsi:.2f}")

    def _detect_macd_signals(self, price: float, macd_diff: float, macd_dea: float, macd_hist: float):
        """Detect MACD trend qualification and momentum signals."""
        # Reset divergence flags at the start
        self.macd_bottom_divergence = False
        self.macd_top_divergence = False
        self.macd_bar_color_changed = False

        # Check MACD position relative to zero axis
        self.macd_below_zero = macd_diff < 0 and macd_dea < 0
        self.macd_above_zero = macd_diff > 0 and macd_dea > 0

        # Detect MACD histogram divergence
        if len(self.price_history) >= self.p.divergence_lookback:
            recent_prices = self.price_history[-self.p.divergence_lookback:]
            recent_macd_hist = self.macd_hist_history[-self.p.divergence_lookback:]

            # Bottom divergence: price makes new low, but MACD histogram area shrinks
            min_price_idx = recent_prices.index(min(recent_prices))
            if min_price_idx > 0 and min_price_idx < len(recent_prices) - 3:
                # Compare MACD histogram area (sum of absolute values for negative bars)
                # Early period: before the low
                early_hist_area = sum([abs(h) for h in recent_macd_hist[:min_price_idx] if h < 0])
                # Late period: after the low
                late_hist_area = sum([abs(h) for h in recent_macd_hist[min_price_idx:] if h < 0])
                # Bottom divergence: late period histogram area is significantly smaller
                if early_hist_area > 0 and late_hist_area < early_hist_area * 0.6:
                    self.macd_bottom_divergence = True
                    logger.debug("MACD bottom divergence detected (histogram area shrinks)")

            # Top divergence: price makes new high, but MACD histogram area shrinks
            max_price_idx = recent_prices.index(max(recent_prices))
            if max_price_idx > 0 and max_price_idx < len(recent_prices) - 3:
                # Compare MACD histogram area (sum of positive values)
                # Early period: before the high
                early_hist_area = sum([h for h in recent_macd_hist[:max_price_idx] if h > 0])
                # Late period: after the high
                late_hist_area = sum([h for h in recent_macd_hist[max_price_idx:] if h > 0])
                # Top divergence: late period histogram area is significantly smaller
                if early_hist_area > 0 and late_hist_area < early_hist_area * 0.6:
                    self.macd_top_divergence = True
                    logger.debug("MACD top divergence detected (histogram area shrinks)")

        # Detect MACD bar color change (green to red for bullish, red to green for bearish)
        if len(self.macd_hist_history) >= 2:
            prev_hist = self.macd_hist_history[-2]
            curr_hist = self.macd_hist_history[-1]
            # Green (negative) to red (positive) = bullish momentum change
            if prev_hist < 0 and curr_hist > 0:
                self.macd_bar_color_changed = True
                logger.debug("MACD bar changed from green to red (bullish)")
            # Red (positive) to green (negative) = bearish momentum change
            elif prev_hist > 0 and curr_hist < 0:
                self.macd_bar_color_changed = True
                logger.debug("MACD bar changed from red to green (bearish)")

    def _check_buy_resonance(
        self, rsi: float, macd_diff: float, macd_dea: float, macd_hist: float
    ) -> bool:
        """
        Check for buy resonance signals (bottom rebound).

        Returns:
            True if buy signal is generated.
        """
        # Step 1: RSI positioning and warning
        if not (self.rsi_oversold_zone or self.rsi_bottom_divergence):
            return False

        # Step 2: MACD trend qualification
        if not (self.macd_bottom_divergence and self.macd_below_zero):
            return False

        # Step 3: Resonance confirmation
        if self.p.use_sensitive_entry:
            # Sensitive entry: RSI bottom divergence low point turns up + MACD bar color change
            if self.rsi_bottom_divergence and self.macd_bar_color_changed and macd_hist > 0:
                # Check if RSI is turning up from oversold zone
                if len(self.rsi_history) >= 3:
                    if self.rsi_history[-1] > self.rsi_history[-2] > self.rsi_history[-3]:
                        logger.info("Buy signal: Sensitive resonance (RSI bottom divergence + MACD bar color change)")
                        return True
        else:
            # Classic entry: RSI crosses above 50 (or stabilizes above 30) + MACD golden cross
            rsi_above_midline = rsi > self.p.rsi_midline
            rsi_stable_above_oversold = False
            if len(self.rsi_history) >= 3:
                rsi_stable_above_oversold = all(
                    self.rsi_history[-i] > self.p.rsi_oversold 
                    for i in range(1, min(4, len(self.rsi_history) + 1))
                )

            # Check for MACD golden cross
            macd_golden_cross = False
            try:
                if len(self.data) >= 2:
                    curr_diff = self.macd_diff[0]
                    curr_dea = self.macd_dea[0]
                    prev_diff = self.macd_diff[-1]
                    prev_dea = self.macd_dea[-1]
                    macd_golden_cross = curr_diff > curr_dea and prev_diff <= prev_dea
            except (IndexError, TypeError):
                pass

            if (rsi_above_midline or rsi_stable_above_oversold) and macd_golden_cross:
                logger.info("Buy signal: Classic resonance (RSI above midline + MACD golden cross)")
                return True

        return False

    def _check_sell_resonance(
        self, rsi: float, macd_diff: float, macd_dea: float, macd_hist: float
    ) -> bool:
        """
        Check for sell resonance signals (top reversal).

        Returns:
            True if sell signal is generated.
        """
        # RSI top divergence
        if self.rsi_top_divergence:
            logger.info("Sell signal: RSI top divergence")
            return True

        # MACD death cross
        try:
            if len(self.data) >= 2:
                curr_diff = self.macd_diff[0]
                curr_dea = self.macd_dea[0]
                prev_diff = self.macd_diff[-1]
                prev_dea = self.macd_dea[-1]
                if curr_diff < curr_dea and prev_diff >= prev_dea:
                    logger.info("Sell signal: MACD death cross")
                    return True
        except (IndexError, TypeError):
            pass

        # MACD red bars shrinking (upward momentum weakening)
        if len(self.macd_hist_history) >= 3:
            if all(h > 0 for h in self.macd_hist_history[-3:]):  # All red bars
                if self.macd_hist_history[-1] < self.macd_hist_history[-2] < self.macd_hist_history[-3]:
                    logger.info("Sell signal: MACD red bars shrinking")
                    return True

        return False

    def _enter_long(self, price: float, atr: float):
        """Enter long position."""
        size = self.broker.getcash() / price * 0.95  # Use 95% of cash
        self.buy(size=size)
        self.entry_price = price
        self.entry_bar_index = len(self.data)
        # Set stop loss below the lowest point of the entry bar
        self.stop_loss_price = self.data.low[0] - atr * self.p.stop_loss_atr_multiplier
        logger.info(
            f"Entered long: Price={price:.2f}, Size={size:.2f}, "
            f"Stop Loss={self.stop_loss_price:.2f}"
        )

    def _exit_long(self, price: float):
        """Exit long position."""
        self.close()
        logger.info(f"Exited long: Price={price:.2f}, Entry={self.entry_price:.2f}")
        # Reset state
        self.entry_price = None
        self.entry_bar_index = None
        self.stop_loss_price = None
        # Reset divergence flags
        self.rsi_bottom_divergence = False
        self.rsi_top_divergence = False
        self.macd_bottom_divergence = False
        self.macd_top_divergence = False
        self.macd_bar_color_changed = False


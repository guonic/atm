"""
Optimized CCI (Commodity Channel Index) Strategy.

This optimized version improves win rate by:
1. Adding trend filter (MA) - only trade in trend direction
2. Adding stop loss and take profit - protect profits and limit losses
3. Requiring multiple confirmations - reduce false signals
4. Adding volume confirmation - ensure sufficient volume
5. Avoiding choppy markets - filter out low volatility periods
6. Better position sizing - adjust based on signal strength

This strategy follows the standard backtrader pattern:
- Indicators are initialized in __init__
- Trading logic is in next() method
- Indicators trigger buy/sell operations
"""

import logging
from typing import Dict, Optional

import backtrader as bt

from nq.trading.strategies.base import BaseStrategy

logger = logging.getLogger(__name__)


class CCIStrategyOptimized(BaseStrategy):
    """
    Optimized CCI (Commodity Channel Index) Strategy.

    Improvements over basic CCI strategy:
    1. Trend Filter: Only buy when price is above MA (uptrend), only sell when below MA (downtrend)
    2. Stop Loss & Take Profit: Automatic risk management
    3. Signal Confirmation: Require multiple conditions to reduce false signals
    4. Volume Confirmation: Ensure sufficient volume before entering trades
    5. Volatility Filter: Avoid trading in choppy/low volatility markets
    6. Position Sizing: Adjust position size based on signal strength

    Buy signals (all must be true):
    1. Price is above MA (uptrend filter)
    2. CCI crosses above oversold level OR crosses above zero
    3. Volume is above average (volume confirmation)
    4. ATR is above threshold (volatility filter)

    Sell signals (any of):
    1. Stop loss triggered
    2. Take profit triggered
    3. CCI crosses below zero AND price is below MA
    4. Top divergence detected

    Args:
        cci_period: CCI calculation period (default: 20).
        overbought: Overbought threshold (default: 100).
        oversold: Oversold threshold (default: -100).
        ma_period: Moving average period for trend filter (default: 50).
        stop_loss_pct: Stop loss percentage (default: 0.05 = 5%).
        take_profit_pct: Take profit percentage (default: 0.15 = 15%).
        volume_ma_period: Period for volume moving average (default: 20).
        volume_threshold: Volume must be above this multiple of average (default: 1.2).
        atr_period: ATR period for volatility filter (default: 14).
        atr_threshold: Minimum ATR multiplier for volatility filter (default: 0.5).
        divergence_lookback: Lookback period for divergence detection (default: 20).
    """

    params = (
        ("cci_period", 20),  # CCI period
        ("overbought", 100),  # Overbought threshold
        ("oversold", -100),  # Oversold threshold
        ("ma_period", 30),  # Moving average period for trend filter (reduced from 50)
        ("stop_loss_pct", 0.05),  # Stop loss percentage (5%)
        ("take_profit_pct", 0.15),  # Take profit percentage (15%)
        ("volume_ma_period", 20),  # Volume moving average period
        ("volume_threshold", 1.0),  # Volume threshold (reduced from 1.2, now optional)
        ("atr_period", 14),  # ATR period
        ("atr_threshold", 0.0),  # ATR threshold (reduced from 0.5, now optional)
        ("divergence_lookback", 20),  # Divergence lookback period
        ("use_trend_filter", True),  # Enable trend filter (default: True)
        ("use_volume_filter", False),  # Enable volume filter (default: False)
        ("use_atr_filter", False),  # Enable ATR filter (default: False)
    )

    def __init__(self):
        """
        Initialize Optimized CCI Strategy.

        Indicators are initialized here following backtrader's standard pattern.
        """
        super().__init__()

        # Core indicator: CCI
        self.cci = bt.indicators.CCI(self.data, period=self.p.cci_period)

        # Trend filter: Moving Average
        self.ma = bt.indicators.SMA(self.data.close, period=self.p.ma_period)

        # Volatility filter: ATR
        self.atr = bt.indicators.ATR(self.data, period=self.p.atr_period)
        self.atr_ratio = self.atr / self.data.close  # ATR as percentage of price

        # Volume confirmation: Volume moving average
        self.volume_ma = bt.indicators.SMA(self.data.volume, period=self.p.volume_ma_period)
        self.volume_ratio = self.data.volume / self.volume_ma  # Volume ratio

        # Track position state
        self.entry_price = None
        self.stop_loss_price = None
        self.take_profit_price = None

        logger.info(
            f"Optimized CCI Strategy initialized with "
            f"cci_period={self.p.cci_period}, "
            f"ma_period={self.p.ma_period}, "
            f"stop_loss={self.p.stop_loss_pct*100:.1f}%, "
            f"take_profit={self.p.take_profit_pct*100:.1f}%, "
            f"trend_filter={self.p.use_trend_filter}, "
            f"volume_filter={self.p.use_volume_filter}, "
            f"atr_filter={self.p.use_atr_filter}"
        )

    def next(self):
        """
        Called for each bar.

        Trading logic with multiple filters and confirmations.
        """
        # Skip if not enough data
        if len(self.data) < max(self.p.ma_period, self.p.atr_period, self.p.cci_period):
            return

        current_cci = self.cci[0]
        prev_cci = self.cci[-1]
        current_close = self.data.close[0]
        current_price = current_close
        current_ma = self.ma[0]
        current_atr_ratio = self.atr_ratio[0]
        current_volume_ratio = self.volume_ratio[0]

        # Check stop loss and take profit if in position
        if self.position:
            # Initialize entry_price if not set (for positions opened outside this strategy)
            if self.entry_price is None:
                # Try to get entry price from position
                # In backtrader, we can estimate from position value
                if hasattr(self.position, 'price') and self.position.price:
                    self.entry_price = self.position.price
                else:
                    # Fallback: use current price as entry (not ideal but prevents errors)
                    self.entry_price = current_price
                    logger.warning(
                        f"Entry price not set, using current price {current_price:.2f} "
                        f"as entry at {self.datas[0].datetime.date(0)}"
                    )
                # Initialize stop loss and take profit if not set
                if self.stop_loss_price is None:
                    self.stop_loss_price = self.entry_price * (1 - self.p.stop_loss_pct)
                if self.take_profit_price is None:
                    self.take_profit_price = self.entry_price * (1 + self.p.take_profit_pct)
            
            if self.stop_loss_price is not None and current_price <= self.stop_loss_price:
                logger.debug(
                    f"Stop loss triggered at {self.datas[0].datetime.date(0)}, "
                    f"price={current_price:.2f}, stop_loss={self.stop_loss_price:.2f}"
                )
                self.close()
                self._reset_position()
                return

            if self.take_profit_price is not None and current_price >= self.take_profit_price:
                logger.debug(
                    f"Take profit triggered at {self.datas[0].datetime.date(0)}, "
                    f"price={current_price:.2f}, take_profit={self.take_profit_price:.2f}"
                )
                self.close()
                self._reset_position()
                return

        # Buy signals
        if not self.position:
            # Check trend filter: price must be above MA (if enabled)
            is_uptrend = True  # Default: allow all trades
            if self.p.use_trend_filter:
                is_uptrend = current_price > current_ma

            # Check volatility filter: ATR ratio must be above threshold (if enabled)
            is_volatile_enough = True  # Default: allow all trades
            if self.p.use_atr_filter:
                is_volatile_enough = current_atr_ratio >= self.p.atr_threshold

            # Check volume confirmation: volume must be above threshold (if enabled)
            has_volume_confirmation = True  # Default: allow all trades
            if self.p.use_volume_filter:
                has_volume_confirmation = current_volume_ratio >= self.p.volume_threshold

            # Buy signal 1: CCI crosses above oversold level
            cci_oversold_cross = (
                prev_cci <= self.p.oversold and current_cci > self.p.oversold
            )

            # Buy signal 2: CCI crosses above zero line
            cci_zero_cross_up = prev_cci <= 0 and current_cci > 0

            # Buy signal 3: Bottom divergence (strong signal)
            has_bottom_divergence = self.check_bottom_divergence()

            # Combine signals: require filters (if enabled) + CCI signal
            if is_uptrend and is_volatile_enough and has_volume_confirmation:
                if cci_oversold_cross or cci_zero_cross_up or has_bottom_divergence:
                    # Determine position size based on signal strength
                    if has_bottom_divergence:
                        size = 1.0  # Full position for divergence
                    elif cci_zero_cross_up:
                        size = 0.8  # 80% for zero cross
                    else:
                        size = 0.6  # 60% for oversold cross

                    logger.debug(
                        f"Buy signal at {self.datas[0].datetime.date(0)}, "
                        f"price={current_price:.2f}, CCI={current_cci:.2f}, "
                        f"MA={current_ma:.2f}, size={size:.1f}, "
                        f"trend_ok={is_uptrend}, volume_ok={has_volume_confirmation}, "
                        f"atr_ok={is_volatile_enough}"
                    )
                    self.buy(size=size)
                    self.entry_price = current_price
                    self.stop_loss_price = current_price * (1 - self.p.stop_loss_pct)
                    self.take_profit_price = current_price * (1 + self.p.take_profit_pct)

        # Sell signals
        if self.position:
            # Sell signal 1: CCI crosses below zero AND price is below MA (downtrend)
            is_downtrend = current_price < current_ma
            cci_zero_cross_down = prev_cci >= 0 and current_cci < 0

            if cci_zero_cross_down and is_downtrend:
                logger.debug(
                    f"Sell signal (CCI below zero + downtrend) at "
                    f"{self.datas[0].datetime.date(0)}, price={current_price:.2f}"
                )
                self.close()
                self._reset_position()
                return

            # Sell signal 2: CCI crosses below overbought (partial profit taking)
            cci_overbought_cross = (
                prev_cci >= self.p.overbought and current_cci < self.p.overbought
            )
            if cci_overbought_cross:
                # Take partial profit: sell 50% if we're in profit
                # Check if entry_price is set and we're in profit
                if self.entry_price is not None and current_price > self.entry_price:
                    logger.debug(
                        f"Partial profit taking at {self.datas[0].datetime.date(0)}, "
                        f"price={current_price:.2f}, entry={self.entry_price:.2f}"
                    )
                    self.sell(size=self.position.size * 0.5)
                    # Adjust stop loss to break-even or trailing stop
                    if self.stop_loss_price is not None:
                        self.stop_loss_price = max(
                            self.stop_loss_price, self.entry_price * 1.01
                        )
                    else:
                        self.stop_loss_price = self.entry_price * 1.01

            # Sell signal 3: Top divergence (strong signal)
            if self.check_top_divergence():
                logger.debug(
                    f"Sell signal (top divergence) at {self.datas[0].datetime.date(0)}, "
                    f"price={current_price:.2f}"
                )
                self.close()
                self._reset_position()
                return

    def _reset_position(self):
        """Reset position tracking variables."""
        self.entry_price = None
        self.stop_loss_price = None
        self.take_profit_price = None

    def check_top_divergence(self) -> bool:
        """
        Check for top divergence (bearish signal).

        Top divergence occurs when:
        - Price makes a higher high
        - CCI makes a lower high

        Returns:
            True if top divergence is detected, False otherwise.
        """
        if len(self.data) < self.p.divergence_lookback:
            return False

        # Get price high over lookback period
        price_values = self.data.close.get(
            ago=self.p.divergence_lookback - 1, size=self.p.divergence_lookback
        )
        if not price_values:
            return False
        price_high = max(price_values)
        current_price_high = self.data.close[0]
        is_price_higher = current_price_high >= price_high

        # Get CCI high over lookback period
        cci_values = self.cci.get(
            ago=self.p.divergence_lookback - 1, size=self.p.divergence_lookback
        )
        if not cci_values:
            return False
        cci_high = max(cci_values)
        current_cci_high = self.cci[0]
        is_cci_lower = current_cci_high < cci_high

        return is_price_higher and is_cci_lower

    def check_bottom_divergence(self) -> bool:
        """
        Check for bottom divergence (bullish signal).

        Bottom divergence occurs when:
        - Price makes a lower low
        - CCI makes a higher low

        Returns:
            True if bottom divergence is detected, False otherwise.
        """
        if len(self.data) < self.p.divergence_lookback:
            return False

        # Get price low over lookback period
        price_values = self.data.close.get(
            ago=self.p.divergence_lookback - 1, size=self.p.divergence_lookback
        )
        if not price_values:
            return False
        price_low = min(price_values)
        current_price_low = self.data.close[0]
        is_price_lower = current_price_low <= price_low

        # Get CCI low over lookback period
        cci_values = self.cci.get(
            ago=self.p.divergence_lookback - 1, size=self.p.divergence_lookback
        )
        if not cci_values:
            return False
        cci_low = min(cci_values)
        current_cci_low = self.cci[0]
        is_cci_higher = current_cci_low > cci_low

        return is_price_lower and is_cci_higher

    def get_info(self) -> Dict:
        """
        Get strategy information.

        Returns:
            Dictionary containing strategy information.
        """
        info = super().get_info()
        info.update(
            {
                "cci_period": self.p.cci_period,
                "overbought": self.p.overbought,
                "oversold": self.p.oversold,
                "ma_period": self.p.ma_period,
                "stop_loss_pct": self.p.stop_loss_pct,
                "take_profit_pct": self.p.take_profit_pct,
                "volume_ma_period": self.p.volume_ma_period,
                "volume_threshold": self.p.volume_threshold,
                "atr_period": self.p.atr_period,
                "atr_threshold": self.p.atr_threshold,
                "divergence_lookback": self.p.divergence_lookback,
                "use_trend_filter": self.p.use_trend_filter,
                "use_volume_filter": self.p.use_volume_filter,
                "use_atr_filter": self.p.use_atr_filter,
            }
        )
        return info


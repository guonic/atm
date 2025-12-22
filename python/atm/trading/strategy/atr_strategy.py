"""
ATR (Average True Range) Based Trading Strategy.

This strategy implements a comprehensive ATR-based trading system including:
1. ATR Stop Loss: Entry price ± 1.5×ATR
2. ATR Stepped Take Profit:
   - First target: Entry price + 2×ATR (reduce position by 30%)
   - Second target: Entry price + 3×ATR (reduce position by another 30%)
   - Final stop loss: Move to cost price + 1×ATR
3. ATR Position Sizing: Calculate position size based on risk percentage
4. ATR Trend Identification: Use ATR expansion/contraction to identify trends
5. ATR Breakout Confirmation: Confirm breakouts with ATR expansion

This strategy follows the standard backtrader pattern:
- Indicators are initialized in __init__
- Trading logic is in next() method
- Indicators trigger buy/sell operations
"""

import logging
from typing import Dict, Optional

import backtrader as bt

from atm.trading.strategy.base import BaseStrategy

logger = logging.getLogger(__name__)


class ATRStrategy(BaseStrategy):
    """
    ATR (Average True Range) Based Trading Strategy.

    This strategy uses ATR for:
    1. Dynamic stop loss and take profit
    2. Position sizing based on risk
    3. Trend identification
    4. Breakout confirmation

    Args:
        atr_period: ATR calculation period (default: 14).
        stop_loss_multiplier: Stop loss multiplier (default: 1.5).
        take_profit_1_multiplier: First take profit multiplier (default: 2.0).
        take_profit_2_multiplier: Second take profit multiplier (default: 3.0).
        take_profit_1_size: Position size to reduce at first target (default: 0.3 = 30%).
        take_profit_2_size: Position size to reduce at second target (default: 0.3 = 30%).
        risk_per_trade: Risk percentage per trade (default: 0.01 = 1%).
        ma_period: Moving average period for trend filter (default: 20).
        use_trend_filter: Enable trend filter (default: True).
        use_breakout_confirmation: Enable breakout confirmation with ATR (default: True).
        atr_expansion_threshold: ATR expansion threshold for trend identification (default: 1.1).
    """

    params = (
        ("atr_period", 14),  # ATR period
        ("stop_loss_multiplier", 1.5),  # Stop loss multiplier (1.5×ATR)
        ("take_profit_1_multiplier", 2.0),  # First take profit multiplier (2×ATR)
        ("take_profit_2_multiplier", 3.0),  # Second take profit multiplier (3×ATR)
        ("take_profit_1_size", 0.3),  # Reduce 30% at first target
        ("take_profit_2_size", 0.3),  # Reduce another 30% at second target
        ("risk_per_trade", 0.01),  # Risk 1% per trade
        ("ma_period", 20),  # Moving average period for trend filter
        ("use_trend_filter", True),  # Enable trend filter
        ("use_breakout_confirmation", True),  # Enable breakout confirmation
        ("atr_expansion_threshold", 1.1),  # ATR expansion threshold (10% increase)
    )

    def __init__(self):
        """
        Initialize ATR Strategy.

        Indicators are initialized here following backtrader's standard pattern.
        """
        super().__init__()

        # Core indicator: ATR
        self.atr = bt.indicators.ATR(self.data, period=self.p.atr_period)

        # Trend filter: Moving Average
        if self.p.use_trend_filter:
            self.ma = bt.indicators.SMA(self.data.close, period=self.p.ma_period)

        # Track ATR for trend identification
        self.atr_ma = bt.indicators.SMA(self.atr, period=self.p.atr_period)
        self.prev_atr = None

        # Track position state
        self.entry_price = None
        self.stop_loss_price = None
        self.take_profit_1_price = None
        self.take_profit_2_price = None
        self.take_profit_1_hit = False
        self.take_profit_2_hit = False
        self.initial_position_size = None

        logger.info(
            f"ATR Strategy initialized with "
            f"atr_period={self.p.atr_period}, "
            f"stop_loss_multiplier={self.p.stop_loss_multiplier}, "
            f"take_profit_1={self.p.take_profit_1_multiplier}, "
            f"take_profit_2={self.p.take_profit_2_multiplier}, "
            f"risk_per_trade={self.p.risk_per_trade*100:.1f}%"
        )

    def next(self):
        """
        Called for each bar.

        Trading logic based on ATR:
        - Entry signals with ATR confirmation
        - Dynamic stop loss and take profit
        - Position sizing based on risk
        """
        # Skip if not enough data
        if len(self.data) < max(self.p.atr_period, self.p.ma_period if self.p.use_trend_filter else 0):
            return

        current_price = self.data.close[0]
        current_atr = self.atr[0]
        current_atr_ma = self.atr_ma[0]

        # Track ATR for trend identification
        if self.prev_atr is not None:
            atr_expansion = current_atr / self.prev_atr if self.prev_atr > 0 else 1.0
        else:
            atr_expansion = 1.0
        self.prev_atr = current_atr

        # Check stop loss and take profit if in position
        if self.position:
            # Initialize entry_price if not set
            if self.entry_price is None:
                self._initialize_position_prices(current_price, current_atr)

            # Check stop loss
            if self.stop_loss_price is not None and current_price <= self.stop_loss_price:
                logger.debug(
                    f"Stop loss triggered at {self.datas[0].datetime.date(0)}, "
                    f"price={current_price:.2f}, stop_loss={self.stop_loss_price:.2f}"
                )
                self.close()
                self._reset_position()
                return

            # Check first take profit
            if (
                not self.take_profit_1_hit
                and self.take_profit_1_price is not None
                and current_price >= self.take_profit_1_price
            ):
                logger.debug(
                    f"First take profit hit at {self.datas[0].datetime.date(0)}, "
                    f"price={current_price:.2f}, target={self.take_profit_1_price:.2f}"
                )
                # Reduce position by take_profit_1_size
                reduce_size = self.position.size * self.p.take_profit_1_size
                self.sell(size=reduce_size)
                self.take_profit_1_hit = True
                # Move stop loss to break-even + 1×ATR
                self.stop_loss_price = self.entry_price + current_atr
                logger.debug(f"Stop loss moved to break-even + ATR: {self.stop_loss_price:.2f}")

            # Check second take profit
            if (
                self.take_profit_1_hit
                and not self.take_profit_2_hit
                and self.take_profit_2_price is not None
                and current_price >= self.take_profit_2_price
            ):
                logger.debug(
                    f"Second take profit hit at {self.datas[0].datetime.date(0)}, "
                    f"price={current_price:.2f}, target={self.take_profit_2_price:.2f}"
                )
                # Reduce position by take_profit_2_size
                reduce_size = self.position.size * self.p.take_profit_2_size
                self.sell(size=reduce_size)
                self.take_profit_2_hit = True
                # Move stop loss to entry + 1×ATR (trailing stop)
                self.stop_loss_price = self.entry_price + current_atr
                logger.debug(f"Stop loss moved to entry + ATR: {self.stop_loss_price:.2f}")

            # Trailing stop: Update stop loss if price moves favorably
            if self.take_profit_1_hit and self.stop_loss_price is not None:
                new_stop_loss = current_price - self.p.stop_loss_multiplier * current_atr
                if new_stop_loss > self.stop_loss_price:
                    self.stop_loss_price = new_stop_loss
                    logger.debug(f"Trailing stop updated to: {self.stop_loss_price:.2f}")

        # Buy signals
        if not self.position:
            # Check trend filter
            is_uptrend = True
            if self.p.use_trend_filter:
                is_uptrend = current_price > self.ma[0]

            # Check ATR expansion for trend identification
            atr_expanding = atr_expansion >= self.p.atr_expansion_threshold

            # Breakout confirmation: Price breaks above MA and ATR expands
            breakout_confirmed = True
            if self.p.use_breakout_confirmation and self.p.use_trend_filter:
                price_above_ma = current_price > self.ma[0]
                prev_price_below_ma = self.data.close[-1] <= self.ma[-1] if len(self.data) > 1 else False
                breakout_confirmed = price_above_ma and (not prev_price_below_ma or atr_expanding)

            # Entry signal: Golden cross or ATR expansion in uptrend
            if is_uptrend and (breakout_confirmed or atr_expanding):
                # Calculate position size based on risk
                position_size = self._calculate_position_size(current_price, current_atr)

                if position_size > 0:
                    logger.debug(
                        f"Buy signal at {self.datas[0].datetime.date(0)}, "
                        f"price={current_price:.2f}, ATR={current_atr:.2f}, "
                        f"size={position_size:.2f}"
                    )
                    self.buy(size=position_size)
                    self.entry_price = current_price
                    self.initial_position_size = position_size
                    self._initialize_position_prices(current_price, current_atr)

        # Sell signals (exit conditions)
        if self.position:
            # Sell signal: Price breaks below MA in downtrend
            if self.p.use_trend_filter:
                is_downtrend = current_price < self.ma[0]
                prev_price_above_ma = self.data.close[-1] >= self.ma[-1] if len(self.data) > 1 else False

                if is_downtrend and prev_price_above_ma:
                    logger.debug(
                        f"Sell signal (downtrend) at {self.datas[0].datetime.date(0)}, "
                        f"price={current_price:.2f}"
                    )
                    self.close()
                    self._reset_position()
                    return

            # Sell signal: ATR contraction (potential trend change)
            if atr_expansion < 1.0 / self.p.atr_expansion_threshold:
                logger.debug(
                    f"Sell signal (ATR contraction) at {self.datas[0].datetime.date(0)}, "
                    f"price={current_price:.2f}, ATR_expansion={atr_expansion:.2f}"
                )
                self.close()
                self._reset_position()
                return

    def _initialize_position_prices(self, entry_price: float, atr_value: float) -> None:
        """
        Initialize stop loss and take profit prices based on ATR.

        Args:
            entry_price: Entry price.
            atr_value: Current ATR value.
        """
        # Stop loss: Entry price - stop_loss_multiplier × ATR
        self.stop_loss_price = entry_price - self.p.stop_loss_multiplier * atr_value

        # First take profit: Entry price + take_profit_1_multiplier × ATR
        self.take_profit_1_price = entry_price + self.p.take_profit_1_multiplier * atr_value

        # Second take profit: Entry price + take_profit_2_multiplier × ATR
        self.take_profit_2_price = entry_price + self.p.take_profit_2_multiplier * atr_value

        logger.debug(
            f"Position prices initialized: entry={entry_price:.2f}, "
            f"stop_loss={self.stop_loss_price:.2f}, "
            f"tp1={self.take_profit_1_price:.2f}, "
            f"tp2={self.take_profit_2_price:.2f}"
        )

    def _calculate_position_size(self, entry_price: float, atr_value: float) -> float:
        """
        Calculate position size based on risk percentage.

        Formula: Position Size = Account Risk ÷ (Stop Loss Distance × Price per Share)
        Where Stop Loss Distance = stop_loss_multiplier × ATR

        In backtrader, size can be:
        - A float (0.0 to 1.0): Percentage of available cash
        - An integer: Number of shares

        Args:
            entry_price: Entry price.
            atr_value: Current ATR value.

        Returns:
            Position size as percentage of available cash (0.0 to 1.0).
        """
        # Calculate account value
        account_value = self.broker.getcash() + self.broker.getvalue()
        available_cash = self.broker.getcash()

        # Calculate risk amount (risk_per_trade × account_value)
        risk_amount = account_value * self.p.risk_per_trade

        # Calculate stop loss distance
        stop_loss_distance = self.p.stop_loss_multiplier * atr_value

        # Avoid division by zero
        if stop_loss_distance <= 0 or entry_price <= 0:
            logger.warning(f"Invalid parameters: stop_loss_distance={stop_loss_distance}, entry_price={entry_price}")
            return 0.0

        # Calculate position size in shares
        # Position size = Risk Amount ÷ Stop Loss Distance
        position_shares = risk_amount / stop_loss_distance

        # Calculate position value
        position_value = position_shares * entry_price

        # Convert to percentage of available cash
        if available_cash <= 0:
            return 0.0

        position_size_pct = min(position_value / available_cash, 1.0)

        # Ensure we have enough cash
        if position_size_pct * available_cash < entry_price:
            return 0.0

        return max(0.0, min(position_size_pct, 1.0))

    def _reset_position(self) -> None:
        """Reset position tracking variables."""
        self.entry_price = None
        self.stop_loss_price = None
        self.take_profit_1_price = None
        self.take_profit_2_price = None
        self.take_profit_1_hit = False
        self.take_profit_2_hit = False
        self.initial_position_size = None

    def get_info(self) -> Dict:
        """
        Get strategy information.

        Returns:
            Dictionary containing strategy information.
        """
        info = super().get_info()
        info.update(
            {
                "atr_period": self.p.atr_period,
                "stop_loss_multiplier": self.p.stop_loss_multiplier,
                "take_profit_1_multiplier": self.p.take_profit_1_multiplier,
                "take_profit_2_multiplier": self.p.take_profit_2_multiplier,
                "take_profit_1_size": self.p.take_profit_1_size,
                "take_profit_2_size": self.p.take_profit_2_size,
                "risk_per_trade": self.p.risk_per_trade,
                "ma_period": self.p.ma_period,
                "use_trend_filter": self.p.use_trend_filter,
                "use_breakout_confirmation": self.p.use_breakout_confirmation,
                "atr_expansion_threshold": self.p.atr_expansion_threshold,
            }
        )
        return info


"""
Volume Weighted Momentum Trading Strategy.

This strategy implements a trading system based on volume-weighted momentum
to identify trend direction and entry opportunities, combined with ATR
volatility filtering to avoid false breakouts.

Core Principles:
- Trend Following: Determine trend direction based on volume-weighted momentum
- Volume Confirmation: Only price changes with volume are valid signals
- Volatility Filter: Use ATR to avoid false signals during low volatility
- Trend Reversal Exit: Close positions immediately when trend changes
- Strict Risk Control: Fixed-ratio risk control and capital management

Entry Signals:
- Long: K>=1 (volume-weighted momentum positive for >=1 bars) AND
        price breaks above upper band + 0.5×ATR
- Short: K>=1 (volume-weighted momentum negative for >=1 bars) AND
         price breaks below lower band - 0.5×ATR

Exit Signals:
- Long Exit: Previous bar shows bearish trend (volume-weighted momentum negative)
- Short Exit: Previous bar shows bullish trend (volume-weighted momentum positive)

This strategy follows the standard backtrader pattern:
- Indicators are initialized in __init__
- Trading logic is in next() method
- Indicators trigger buy/sell operations
"""

import logging
from typing import Dict, Optional

import backtrader as bt

from atm.trading.indicators.volume_weighted_momentum import VolumeWeightedMomentum
from atm.trading.strategy.base import BaseStrategy

logger = logging.getLogger(__name__)


class VolumeWeightedMomentumStrategy(BaseStrategy):
    """
    Volume Weighted Momentum Trading Strategy.

    This strategy uses volume-weighted momentum to determine trend direction
    and combines it with ATR volatility filtering to select entry opportunities.

    Entry Signals:
    - Long: Volume-weighted momentum positive for >=1 bars AND
            price breaks above upper band + 0.5×ATR
    - Short: Volume-weighted momentum negative for >=1 bars AND
             price breaks below lower band - 0.5×ATR

    Exit Signals:
    - Long Exit: Previous bar shows bearish trend
    - Short Exit: Previous bar shows bullish trend

    Args:
        mom_len: Momentum period (default: 10).
        avg_len: EMA averaging period for VWM (default: 20).
        atr_period: ATR period (default: 14).
        atr_multiplier: ATR multiplier for band calculation (default: 0.5).
        band_period: Moving average period for bands (default: 20).
        risk_per_trade: Risk per trade as percentage of capital (default: 0.02 = 2%).
        use_atr_filter: Use ATR volatility filter (default: True).
    """

    params = (
        ("mom_len", 10),  # Momentum period
        ("avg_len", 20),  # EMA averaging period for VWM
        ("atr_period", 14),  # ATR period
        ("atr_multiplier", 0.5),  # ATR multiplier for band calculation
        ("band_period", 20),  # Moving average period for bands
        ("risk_per_trade", 0.02),  # Risk per trade (2% of capital)
        ("use_atr_filter", True),  # Use ATR volatility filter
        ("allow_short", False),  # Allow short positions (default: False for A-share market)
    )

    def __init__(self):
        """
        Initialize Volume Weighted Momentum Strategy.

        Indicators are initialized here following backtrader's standard pattern.
        """
        super().__init__()

        # Volume Weighted Momentum
        self.vwm = VolumeWeightedMomentum(
            self.data, mom_len=self.p.mom_len, avg_len=self.p.avg_len
        )

        # ATR for volatility filtering
        self.atr = bt.indicators.ATR(self.data, period=self.p.atr_period)

        # Moving average for bands (used as reference for entry thresholds)
        self.ma = bt.indicators.SMA(self.data.close, period=self.p.band_period)

        # Upper and lower bands (using MA as reference)
        # Entry thresholds will be: upper_band + 0.5×ATR or lower_band - 0.5×ATR
        self.upper_band = self.ma  # Upper band reference
        self.lower_band = self.ma  # Lower band reference

        # Track trend state
        self.trend_count = 0  # Count of consecutive bars in current trend
        self.current_trend = None  # 'long', 'short', or None
        self.prev_trend = None  # Previous trend state

        logger.info(
            f"Volume Weighted Momentum Strategy initialized with "
            f"mom_len={self.p.mom_len}, avg_len={self.p.avg_len}, "
            f"atr_period={self.p.atr_period}, allow_short={self.p.allow_short}"
        )

    def next(self):
        """
        Called for each bar.

        Trading logic based on volume-weighted momentum:
        - Trend identification
        - Entry signals with ATR filtering
        - Trend reversal exit
        """
        # Skip if not enough data
        max_period = max(self.p.mom_len, self.p.avg_len, self.p.atr_period, self.p.band_period)
        if len(self.data) < max_period + 5:
            return

        # Safely access indicator values
        try:
            current_price = self.data.close[0]
            current_vwm = self.vwm.lines.vwm[0]
            current_atr = self.atr[0]
            current_upper_band = self.upper_band[0]
            current_lower_band = self.lower_band[0]
        except (IndexError, TypeError):
            return

        # Determine current trend based on volume-weighted momentum
        if current_vwm > 0:
            current_trend = "long"
        elif current_vwm < 0:
            current_trend = "short"
        else:
            current_trend = None

        # Update trend count
        if current_trend == self.current_trend:
            self.trend_count += 1
        else:
            self.trend_count = 1
            self.prev_trend = self.current_trend
            self.current_trend = current_trend

        # Exit signals (trend reversal)
        if self.position:
            if self.position.size > 0:  # Long position
                # Exit if previous bar showed bearish trend
                if self.prev_trend == "short" or current_trend == "short":
                    logger.debug(
                        f"Long exit (trend reversal to bearish) at "
                        f"{self.datas[0].datetime.date(0)}, "
                        f"price={current_price:.2f}, VWM={current_vwm:.2f}"
                    )
                    self.close()
                    return

            elif self.position.size < 0:  # Short position
                # Exit if previous bar showed bullish trend
                if self.prev_trend == "long" or current_trend == "long":
                    logger.debug(
                        f"Short exit (trend reversal to bullish) at "
                        f"{self.datas[0].datetime.date(0)}, "
                        f"price={current_price:.2f}, VWM={current_vwm:.2f}"
                    )
                    self.close()
                    return

        # Entry signals
        if not self.position:
            # Long entry: K>=1 AND price breaks above upper band + 0.5×ATR
            if current_trend == "long" and self.trend_count >= 1:
                # Calculate entry threshold
                entry_threshold = current_upper_band + self.p.atr_multiplier * current_atr

                # Check if price breaks above threshold
                price_above_threshold = current_price > entry_threshold

                # ATR filter (optional)
                atr_filter_passed = True
                if self.p.use_atr_filter:
                    # Only enter if ATR is above a minimum threshold
                    # Use average ATR over recent period as reference
                    try:
                        avg_atr = sum([self.atr[-i] for i in range(5)]) / 5
                        atr_filter_passed = current_atr >= avg_atr * 0.8  # ATR not too low
                    except (IndexError, TypeError):
                        atr_filter_passed = True

                if price_above_threshold and atr_filter_passed:
                    # Calculate position size based on risk
                    position_size = self._calculate_position_size(current_price, current_atr)

                    if position_size > 0:
                        logger.debug(
                            f"Long entry at {self.datas[0].datetime.date(0)}, "
                            f"price={current_price:.2f}, "
                            f"threshold={entry_threshold:.2f}, "
                            f"VWM={current_vwm:.2f}, "
                            f"trend_count={self.trend_count}, "
                            f"size={position_size:.2f}"
                        )
                        self.buy(size=position_size)

            # Short entry: K>=1 AND price breaks below lower band - 0.5×ATR
            elif (
                self.p.allow_short
                and current_trend == "short"
                and self.trend_count >= 1
            ):
                # Calculate entry threshold
                entry_threshold = current_lower_band - self.p.atr_multiplier * current_atr

                # Check if price breaks below threshold
                price_below_threshold = current_price < entry_threshold

                # ATR filter (optional)
                atr_filter_passed = True
                if self.p.use_atr_filter:
                    try:
                        avg_atr = sum([self.atr[-i] for i in range(5)]) / 5
                        atr_filter_passed = current_atr >= avg_atr * 0.8
                    except (IndexError, TypeError):
                        atr_filter_passed = True

                if price_below_threshold and atr_filter_passed:
                    # Calculate position size based on risk
                    position_size = self._calculate_position_size(current_price, current_atr)

                    if position_size > 0:
                        logger.debug(
                            f"Short entry at {self.datas[0].datetime.date(0)}, "
                            f"price={current_price:.2f}, "
                            f"threshold={entry_threshold:.2f}, "
                            f"VWM={current_vwm:.2f}, "
                            f"trend_count={self.trend_count}, "
                            f"size={position_size:.2f}"
                        )
                        self.sell(size=position_size)

    def _calculate_position_size(self, entry_price: float, atr_value: float) -> float:
        """
        Calculate position size based on risk per trade.

        Args:
            entry_price: Entry price.
            atr_value: Current ATR value.

        Returns:
            Position size.
        """
        if entry_price <= 0 or atr_value <= 0:
            return 0.0

        # Calculate risk amount
        account_value = self.broker.getvalue()
        risk_amount = account_value * self.p.risk_per_trade

        # Use ATR as stop loss distance
        stop_loss_distance = atr_value * 1.5  # 1.5×ATR stop loss

        # Calculate position size
        position_size = risk_amount / stop_loss_distance

        # Round down to avoid over-leveraging
        return int(position_size)

    def get_info(self) -> Dict:
        """
        Get strategy information.

        Returns:
            Dictionary containing strategy information.
        """
        info = super().get_info()
        info.update(
            {
                "mom_len": self.p.mom_len,
                "avg_len": self.p.avg_len,
                "atr_period": self.p.atr_period,
                "atr_multiplier": self.p.atr_multiplier,
                "band_period": self.p.band_period,
                "risk_per_trade": self.p.risk_per_trade,
                "use_atr_filter": self.p.use_atr_filter,
                "allow_short": self.p.allow_short,
            }
        )
        return info


"""
RSI multi-signal long-only strategy.

Rules (RSI1 = fast, RSI2 = slow):
- Buy 1: RSI1 rebounded from oversold (below oversold then back above).
- Buy 2: RSI1 crosses above RSI2 while both are below 50 (low-zone golden cross).
- Sell 1: RSI1 drops back below overbought (profit take).
- Sell 2: RSI1 crosses below RSI2 while both are above 50 (high-zone death cross).

Risk:
- Optional stop loss / take profit based on entry price.
"""

import logging
from typing import Any, Dict, Optional

import backtrader as bt

from nq.trading.strategies.base import BaseStrategy

logger = logging.getLogger(__name__)


class RSIMultiSignalStrategy(BaseStrategy):
    """RSI multi-signal strategy with dual RSI lines."""

    params = (
        ("rsi_fast", 6),
        ("rsi_slow", 14),
        ("oversold", 20.0),
        ("overbought", 80.0),
        ("stop_loss_pct", 0.05),
        ("take_profit_pct", 0.1),
        ("strategy_config", None),
        ("run_id", None),
        ("ts_code", None),
    )

    def __init__(self):
        """Initialize indicators and state."""
        super().__init__()
        self.rsi_fast = bt.ind.RSI(period=self.p.rsi_fast)
        self.rsi_slow = bt.ind.RSI(period=self.p.rsi_slow)

        self.entry_price: Optional[float] = None
        self.addminperiod(max(self.p.rsi_fast, self.p.rsi_slow) + 5)

    def _buy_signal_rebound(self) -> bool:
        """RSI1 rebounded from oversold region."""
        try:
            return self.rsi_fast[-1] < self.p.oversold <= self.rsi_fast[0]
        except Exception:
            return False

    def _buy_signal_low_zone_cross(self) -> bool:
        """RSI1 crosses above RSI2 while both below 50."""
        try:
            cross_up = self.rsi_fast[-1] <= self.rsi_slow[-1] and self.rsi_fast[0] > self.rsi_slow[0]
            low_zone = max(self.rsi_fast[0], self.rsi_slow[0]) < 50
            return cross_up and low_zone
        except Exception:
            return False

    def _sell_signal_overbought_fall(self) -> bool:
        """RSI1 falls back from overbought."""
        try:
            return self.rsi_fast[-1] > self.p.overbought >= self.rsi_fast[0]
        except Exception:
            return False

    def _sell_signal_high_zone_cross(self) -> bool:
        """RSI1 crosses below RSI2 while both above 50."""
        try:
            cross_down = self.rsi_fast[-1] >= self.rsi_slow[-1] and self.rsi_fast[0] < self.rsi_slow[0]
            high_zone = min(self.rsi_fast[0], self.rsi_slow[0]) > 50
            return cross_down and high_zone
        except Exception:
            return False

    def _risk_hit(self) -> bool:
        """Check stop loss / take profit."""
        if self.entry_price is None:
            return False
        price = float(self.data.close[0])
        if price <= self.entry_price * (1 - self.p.stop_loss_pct):
            return True
        if price >= self.entry_price * (1 + self.p.take_profit_pct):
            return True
        return False

    def next(self):
        price = float(self.data.close[0])

        if not self.position:
            if self._buy_signal_rebound() or self._buy_signal_low_zone_cross():
                self.buy()
                self.entry_price = price
                logger.debug(
                    "Buy at %s price=%.2f rsi_fast=%.2f rsi_slow=%.2f",
                    self.data.datetime.date(0),
                    price,
                    float(self.rsi_fast[0]),
                    float(self.rsi_slow[0]),
                )
            return

        # Exit conditions
        if self._sell_signal_overbought_fall() or self._sell_signal_high_zone_cross() or self._risk_hit():
            logger.debug(
                "Sell at %s price=%.2f rsi_fast=%.2f rsi_slow=%.2f",
                self.data.datetime.date(0),
                price,
                float(self.rsi_fast[0]),
                float(self.rsi_slow[0]),
            )
            self.close()
            self.entry_price = None

    def get_info(self) -> Dict[str, Any]:
        """Return strategy parameters."""
        info = super().get_info()
        info.update(
            {
                "rsi_fast": self.p.rsi_fast,
                "rsi_slow": self.p.rsi_slow,
                "oversold": self.p.oversold,
                "overbought": self.p.overbought,
                "stop_loss_pct": self.p.stop_loss_pct,
                "take_profit_pct": self.p.take_profit_pct,
            }
        )
        return info


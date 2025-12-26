"""
Holy Grail Trend Pullback Strategy (Linda Raschke style).

Rules (long side):
- Trend strength: ADX(14) must be above a threshold (default 30) for several bars.
- Trend filter: price above EMA(20) and EMA is rising.
- Setup: wait for the first pullback to EMA(20) after trend strength is confirmed.
- Entry: on a bar that closes back above EMA(20) while ADX stays above threshold.
- Exit: price closes below EMA(20) or below the trailing stop (recent swing low).

Notes:
- This implementation only covers the long side. Short side can be mirrored if needed.
- Trailing stop is anchored to the lowest close over a lookback window after entry.
"""

import logging
from typing import Any, Dict

import backtrader.indicators as btind

from nq.trading.strategies.base import BaseStrategy

logger = logging.getLogger(__name__)


class HolyGrailStrategy(BaseStrategy):
    """Holy Grail trend pullback strategy."""

    params = (
        ("adx_period", 14),
        ("adx_threshold", 30.0),
        ("adx_confirm_bars", 3),
        ("ema_period", 20),
        ("swing_lookback", 10),
        ("strategy_config", None),
        ("run_id", None),
        ("ts_code", None),
    )

    def __init__(self):
        """Initialize indicators and state."""
        super().__init__()
        self.ema = btind.EMA(self.data.close, period=self.p.ema_period)
        self.adx = btind.ADX(self.data, period=self.p.adx_period)
        self.lowest_since_entry = None

        # State flags
        self.trend_confirmed = False
        self.pullback_ready = False

        max_period = max(self.p.adx_period, self.p.ema_period, self.p.swing_lookback)
        self.addminperiod(max_period + 5)

    def _trend_ok(self) -> bool:
        """Check if trend strength and direction are OK."""
        try:
            adx_ok = all(self.adx[-i] >= self.p.adx_threshold for i in range(self.p.adx_confirm_bars))
        except Exception:
            adx_ok = False
        try:
            ema_up = self.ema[0] > self.ema[-1]
            price_above = self.data.close[0] > self.ema[0]
        except Exception:
            ema_up = False
            price_above = False
        return adx_ok and ema_up and price_above

    def _update_trailing_stop(self):
        """Update trailing stop based on recent swing lows."""
        try:
            swing_low = btind.Lowest(self.data.close, period=self.p.swing_lookback)[0]
        except Exception:
            return
        if self.lowest_since_entry is None or swing_low > self.lowest_since_entry:
            self.lowest_since_entry = swing_low

    def next(self):
        # Detect trend confirmation
        if self._trend_ok():
            self.trend_confirmed = True

        # Wait for first pullback to EMA after confirmation
        if self.trend_confirmed and not self.pullback_ready:
            try:
                if self.data.close[0] <= self.ema[0]:
                    self.pullback_ready = True
            except Exception:
                pass

        # Entry
        if not self.position:
            try:
                entry_condition = (
                    self.trend_confirmed
                    and self.pullback_ready
                    and self.data.close[0] > self.ema[0]
                    and self.adx[0] >= self.p.adx_threshold
                )
            except Exception:
                entry_condition = False

            if entry_condition:
                self.buy()
                self.lowest_since_entry = float(self.data.close[0])
                self._update_trailing_stop()
                logger.debug(
                    "Entry: price=%.2f ema=%.2f adx=%.2f",
                    float(self.data.close[0]),
                    float(self.ema[0]),
                    float(self.adx[0]),
                )
                # Reset pullback flag for next setups only after exit
            return

        # In position: update trailing stop
        self._update_trailing_stop()

        price = float(self.data.close[0])
        ema_now = float(self.ema[0])

        stop_hit = self.lowest_since_entry is not None and price < self.lowest_since_entry
        ema_break = price < ema_now

        if stop_hit or ema_break:
            logger.debug(
                "Exit: price=%.2f stop=%.2f ema=%.2f adx=%.2f stop_hit=%s ema_break=%s",
                price,
                float(self.lowest_since_entry) if self.lowest_since_entry is not None else float("nan"),
                ema_now,
                float(self.adx[0]),
                stop_hit,
                ema_break,
            )
            self.close()
            # Reset state for next cycle
            self.trend_confirmed = False
            self.pullback_ready = False
            self.lowest_since_entry = None

    def get_info(self) -> Dict[str, Any]:
        info = super().get_info()
        info.update(
            {
                "adx_period": self.p.adx_period,
                "adx_threshold": self.p.adx_threshold,
                "adx_confirm_bars": self.p.adx_confirm_bars,
                "ema_period": self.p.ema_period,
                "swing_lookback": self.p.swing_lookback,
            }
        )
        return info


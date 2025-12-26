"""
Triple MA 13/34/55 trend pullback strategy.

Idea:
- Use fast EMA(13), mid EMA(34), long EMA(55) to align trend and spot first pullback.
- Wait for dual golden crosses (13>34 and 34>55) within a short gap, confirming trend start.
- Enter on the first pullback to EMA(34) while alignment holds and price stays above EMA(55).
- Exit when trend weakens: price closes below EMA(55) or EMA(13) drops below EMA(34).

Notes:
- Only long side is implemented (A-share style). Short side can be mirrored if needed.
"""

import logging
from typing import Any, Dict, Optional

import backtrader.indicators as btind

from nq.trading.strategies.base import BaseStrategy

logger = logging.getLogger(__name__)


class TripleMA135Strategy(BaseStrategy):
    """13/34/55 EMA trend pullback strategy."""

    params = (
        ("ema_short", 13),
        ("ema_mid", 34),
        ("ema_long", 55),
        ("cross_gap", 5),  # max bars between the two golden crosses
        ("pullback_tolerance", 0.01),  # price proximity to EMA(34)
        ("strategy_config", None),
        ("run_id", None),
        ("ts_code", None),
    )

    def __init__(self):
        """Initialize indicators and state."""
        super().__init__()
        self.ema_s = btind.EMA(self.data.close, period=self.p.ema_short)
        self.ema_m = btind.EMA(self.data.close, period=self.p.ema_mid)
        self.ema_l = btind.EMA(self.data.close, period=self.p.ema_long)

        # Track cross bars
        self.cross_s_m_bar: Optional[int] = None
        self.cross_m_l_bar: Optional[int] = None
        self.waiting_pullback = False

        max_period = max(self.p.ema_short, self.p.ema_mid, self.p.ema_long)
        self.addminperiod(max_period + 5)

    def _update_cross_states(self):
        """Detect golden crosses and track bar indices."""
        try:
            # 13 crosses above 34
            if self.ema_s[-1] <= self.ema_m[-1] and self.ema_s[0] > self.ema_m[0]:
                self.cross_s_m_bar = len(self.data)

            # 34 crosses above 55
            if self.ema_m[-1] <= self.ema_l[-1] and self.ema_m[0] > self.ema_l[0]:
                self.cross_m_l_bar = len(self.data)
        except Exception:
            return

        if (
            self.cross_s_m_bar is not None
            and self.cross_m_l_bar is not None
            and abs(self.cross_s_m_bar - self.cross_m_l_bar) <= self.p.cross_gap
        ):
            self.waiting_pullback = True

    def _alignment_ok(self) -> bool:
        """Check EMA alignment 13 > 34 > 55."""
        try:
            return self.ema_s[0] > self.ema_m[0] > self.ema_l[0]
        except Exception:
            return False

    def _pullback_ok(self) -> bool:
        """
        Check pullback condition: price near EMA(34) but above EMA(55).

        Allow a small tolerance to consider a touch.
        """
        try:
            price = self.data.close[0]
            ema_m = self.ema_m[0]
            ema_l = self.ema_l[0]
        except Exception:
            return False

        near_mid = price >= ema_m * (1 - self.p.pullback_tolerance) and price <= ema_m * (
            1 + self.p.pullback_tolerance
        )
        above_long = price > ema_l
        return near_mid and above_long

    def next(self):
        # Update cross states
        self._update_cross_states()

        # Entry
        if not self.position:
            if self.waiting_pullback and self._alignment_ok() and self._pullback_ok():
                self.buy()
                logger.debug(
                    "Entry at %s: price=%.2f ema13=%.2f ema34=%.2f ema55=%.2f",
                    self.data.datetime.date(0),
                    float(self.data.close[0]),
                    float(self.ema_s[0]),
                    float(self.ema_m[0]),
                    float(self.ema_l[0]),
                )
                # After entry, wait for new crosses for next setup
                self.waiting_pullback = False
            return

        # Exit
        price = float(self.data.close[0])
        try:
            trend_break = price < self.ema_l[0] or self.ema_s[0] < self.ema_m[0]
            mid_break = price < self.ema_m[0] * (1 - self.p.pullback_tolerance)
        except Exception:
            trend_break = False
            mid_break = False

        if trend_break or mid_break:
            logger.debug(
                "Exit at %s: price=%.2f ema13=%.2f ema34=%.2f ema55=%.2f trend_break=%s mid_break=%s",
                self.data.datetime.date(0),
                price,
                float(self.ema_s[0]),
                float(self.ema_m[0]),
                float(self.ema_l[0]),
                trend_break,
                mid_break,
            )
            self.close()
            # reset to wait new crosses
            self.cross_s_m_bar = None
            self.cross_m_l_bar = None
            self.waiting_pullback = False

    def get_info(self) -> Dict[str, Any]:
        """Return strategy parameters."""
        info = super().get_info()
        info.update(
            {
                "ema_short": self.p.ema_short,
                "ema_mid": self.p.ema_mid,
                "ema_long": self.p.ema_long,
                "cross_gap": self.p.cross_gap,
                "pullback_tolerance": self.p.pullback_tolerance,
            }
        )
        return info


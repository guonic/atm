"""
Price Entropy Trading Strategy.

This strategy detects prolonged volatility compression (low "price entropy")
and trades the breakout when entropy expands. It uses three compression signals:

1) ATR channel squeeze: ATR-based channel width shrinks to < 40% of 1-year avg
   and persists for N days.
2) Intraday range + volume contraction: daily range < 50% ATR and volume
   < 60% of 120-day average for M consecutive days.
3) Bollinger Band extreme adhesion: band width reaches multi-month lows and
   prices are glued inside the band.

Breakout entry:
- After compression detected (recent), a wide candle (body >= 2%) breaks the
  recent range high with volume > 1.5x 20-day avg and ATR expanding.
- Enter 30% position on breakout bar (close).
- Add 30% on successful retest/hold of breakout level while ATR keeps rising.

Exit:
- If price closes back below breakout level for 2 bars, exit all.
- Safety stop: if price falls 8% below breakout close, exit.

Parameters are configurable to adapt to different symbols/timeframes.
"""

import logging
from typing import Optional

import backtrader as bt
import backtrader.indicators as btind

from nq.trading.strategies.base import BaseStrategy

logger = logging.getLogger(__name__)


class PriceEntropyStrategy(BaseStrategy):
    """Volatility-compression breakout strategy (price entropy)."""

    params = (
        # Compression detection
        ("atr_period", 14),
        ("atr_long_period", 252),
        ("atr_compress_ratio", 0.4),
        ("compress_days_atr", 5),
        ("intraday_range_ratio", 0.5),
        ("volume_ratio_max", 0.6),
        ("compress_days_intraday", 3),
        ("bb_period", 20),
        ("bb_dev", 2.0),
        ("bb_width_min_pct", 0.01),  # 1%
        # Breakout detection
        ("range_lookback", 20),
        ("breakout_body_pct", 0.02),
        ("breakout_volume_mult", 1.5),
        ("retest_lookback", 3),
        # Position sizing
        ("initial_position_pct", 0.3),
        ("add_position_pct", 0.3),
        # Risk management
        ("stop_lookback", 2),
        ("max_drawdown_pct", 0.08),
    )

    def __init__(self):
        super().__init__()

        # Indicators
        self.atr = btind.ATR(self.data, period=self.p.atr_period)
        self.atr_long = btind.SMA(self.atr, period=self.p.atr_long_period)

        self.volume_avg20 = btind.SMA(self.data.volume, period=20)
        self.volume_avg120 = btind.SMA(self.data.volume, period=120)

        self.bb = btind.BollingerBands(
            self.data.close, period=self.p.bb_period, devfactor=self.p.bb_dev
        )

        # State
        self.breakout_level: Optional[float] = None
        self.entry_price: Optional[float] = None
        self.initial_position_added = False
        self.add_position_added = False

        # Ensure enough history before indicators operate to avoid IndexError
        max_period = max(
            self.p.atr_long_period,
            self.p.range_lookback,
            120,  # volume_avg120
            self.p.bb_period,
            self.p.compress_days_atr + self.p.atr_period,
            self.p.compress_days_intraday + 5,
        )
        self.addminperiod(max_period + 5)

    def _is_atr_compressed(self) -> bool:
        if len(self.atr_long) < max(self.p.atr_long_period, self.p.compress_days_atr):
            return False
        ratio = self.atr[0] / (self.atr_long[0] + 1e-9)
        # check consecutive days
        count = 0
        for i in range(self.p.compress_days_atr):
            try:
                r = self.atr[-i] / (self.atr_long[-i] + 1e-9)
            except (IndexError, TypeError):
                return False
            if r < self.p.atr_compress_ratio:
                count += 1
        return count >= self.p.compress_days_atr

    def _is_intraday_compressed(self) -> bool:
        if len(self.data) < self.p.compress_days_intraday:
            return False
        if len(self.volume_avg120) < 120:
            return False
        count = 0
        for i in range(self.p.compress_days_intraday):
            try:
                rng = (self.data.high[-i] - self.data.low[-i]) / (
                    self.atr[-i] + 1e-9
                )
                vol_cond = self.data.volume[-i] < self.volume_avg120[-i] * self.p.volume_ratio_max
            except (IndexError, TypeError):
                return False
            if rng < self.p.intraday_range_ratio and vol_cond:
                count += 1
        return count >= self.p.compress_days_intraday

    def _is_bb_compressed(self) -> bool:
        if len(self.data) < self.p.bb_period:
            return False
        try:
            width = (self.bb.top[0] - self.bb.bot[0]) / (self.data.close[0] + 1e-9)
            return width < self.p.bb_width_min_pct
        except (IndexError, TypeError):
            return False

    def _recent_compression(self) -> bool:
        # at least two of three compression signals within last few bars
        signals = []
        signals.append(self._is_atr_compressed())
        signals.append(self._is_intraday_compressed())
        signals.append(self._is_bb_compressed())
        return sum(signals) >= 2

    def _is_breakout(self) -> bool:
        if len(self.data) < self.p.range_lookback + 2:
            return False
        try:
            recent_high = max(self.data.high.get(size=self.p.range_lookback))
            body_pct = abs(self.data.close[0] - self.data.open[0]) / (
                self.data.open[0] + 1e-9
            )
            volume_cond = self.data.volume[0] > self.volume_avg20[0] * self.p.breakout_volume_mult
            atr_expanding = self.atr[0] > self.atr[-1]
        except (IndexError, TypeError):
            return False

        if (
            self.data.close[0] > recent_high
            and body_pct >= self.p.breakout_body_pct
            and volume_cond
            and atr_expanding
        ):
            self.breakout_level = recent_high
            return True
        return False

    def _should_add(self) -> bool:
        if self.breakout_level is None:
            return False
        # Add on successful retest/hold of breakout level with rising ATR
        try:
            retest_ok = self.data.close[0] >= self.breakout_level
            atr_rising = self.atr[0] >= self.atr[-1]
        except (IndexError, TypeError):
            return False
        return retest_ok and atr_rising

    def _should_stop(self) -> bool:
        if self.breakout_level is None:
            return False
        # Stop if price closes below breakout level for stop_lookback bars
        try:
            below = [
                self.data.close[-i] < self.breakout_level
                for i in range(self.p.stop_lookback)
            ]
        except (IndexError, TypeError):
            return False
        if all(below):
            return True
        if self.entry_price:
            dd = (self.entry_price - self.data.close[0]) / (self.entry_price + 1e-9)
            if dd >= self.p.max_drawdown_pct:
                return True
        return False

    def next(self):
        # Ensure data length
        if len(self.data) < max(60, self.p.range_lookback + 5):
            return

        compression = self._recent_compression()
        breakout = self._is_breakout() if compression else False

        # Entry
        if breakout and not self.position:
            cash = self.broker.getcash()
            size = int(cash * self.p.initial_position_pct / self.data.close[0])
            if size > 0:
                self.buy(size=size)
                self.initial_position_added = True
                self.entry_price = self.data.close[0]
                self.add_position_added = False
                logger.info(
                    f"Breakout entry: close={self.data.close[0]:.2f}, "
                    f"level={self.breakout_level:.2f}, size={size}"
                )

        # Add position on retest/hold
        if (
            self.position
            and self.initial_position_added
            and not self.add_position_added
            and self._should_add()
        ):
            cash = self.broker.getcash()
            size = int(cash * self.p.add_position_pct / self.data.close[0])
            if size > 0:
                self.buy(size=size)
                self.add_position_added = True
                logger.info(
                    f"Add on hold: close={self.data.close[0]:.2f}, "
                    f"level={self.breakout_level:.2f}, size={size}"
                )

        # Exit rules
        if self.position and self._should_stop():
            self.close()
            logger.info(
                f"Exit: close={self.data.close[0]:.2f}, level={self.breakout_level:.2f}"
            )
            self.breakout_level = None
            self.initial_position_added = False
            self.add_position_added = False
            self.entry_price = None



"""
ATR Dynamic Stop Trend Strategy.

Idea:
- Use ATR to define “normal volatility band”; stop distance = k × ATR (dynamic).
- Initial stop: entry_price - k × ATR.
- Trailing stop: highest_close - k × ATR after price makes new highs.
- Trend filter with MA to avoid whipsaws.

Params:
- atr_period: ATR period (default 14)
- atr_multiplier: ATR multiplier for stops (default 1.8, typical 1.5~2.5)
- ma_period: MA period for trend filter (default 20)
- min_atr: minimum ATR safeguard to avoid over-tight stops (default 0)
"""

import logging
from typing import Any, Dict, Optional

import backtrader as bt

from atm.trading.strategies.base import BaseStrategy

logger = logging.getLogger(__name__)


class ATRDynamicStopStrategy(BaseStrategy):
    """
    基于 ATR 的动态止损/止盈趋势策略。
    """

    params = (
        ("atr_period", 14),
        ("atr_multiplier", 1.8),
        ("ma_period", 20),
        ("min_atr", 0.0),
        ("strategy_config", None),
        ("run_id", None),
        ("ts_code", None),
    )

    def __init__(self):
        super().__init__()
        self.atr = bt.indicators.ATR(self.data, period=self.p.atr_period)
        self.ma = bt.indicators.SMA(self.data.close, period=self.p.ma_period)

        # 状态
        self.stop_price: Optional[float] = None
        self.entry_price: Optional[float] = None
        self.highest_close: Optional[float] = None

        max_period = max(self.p.atr_period, self.p.ma_period)
        self.addminperiod(max_period + 5)

    def _current_atr(self) -> float:
        try:
            val = float(self.atr[0])
        except Exception:
            val = 0.0
        return max(val, self.p.min_atr)

    def _update_trailing_stop(self):
        if self.highest_close is None:
            return
        atr_val = self._current_atr()
        new_stop = self.highest_close - self.p.atr_multiplier * atr_val
        if self.stop_price is None or new_stop > self.stop_price:
            self.stop_price = new_stop

    def _reset_state(self):
        self.stop_price = None
        self.entry_price = None
        self.highest_close = None

    def next(self):
        # 趋势过滤：价格高于均线且均线向上
        try:
            trend_up = self.data.close[0] > self.ma[0] and self.ma[0] > self.ma[-1]
        except Exception:
            trend_up = False

        atr_val = self._current_atr()
        price = float(self.data.close[0])

        # 入场逻辑
        if not self.position:
            if trend_up and atr_val > 0:
                self.buy()
                self.entry_price = price
                self.highest_close = price
                self.stop_price = price - self.p.atr_multiplier * atr_val
                logger.debug(
                    "Entry: price=%.2f atr=%.4f stop=%.2f",
                    price,
                    atr_val,
                    self.stop_price,
                )
            return

        # 持仓中：更新最高收盘和移动止损
        if self.highest_close is None or price > self.highest_close:
            self.highest_close = price
        self._update_trailing_stop()

        # 离场：价格跌破止损
        if self.stop_price is not None and price < self.stop_price:
            logger.debug(
                "Exit: price=%.2f stop=%.2f atr=%.4f",
                price,
                self.stop_price,
                atr_val,
            )
            self.sell()
            self._reset_state()

    def get_info(self) -> Dict[str, Any]:
        info = super().get_info()
        info.update(
            {
                "atr_period": self.p.atr_period,
                "atr_multiplier": self.p.atr_multiplier,
                "ma_period": self.p.ma_period,
                "min_atr": self.p.min_atr,
            }
        )
        return info


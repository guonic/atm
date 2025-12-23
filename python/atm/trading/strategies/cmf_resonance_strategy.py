"""
CMF flow + trend resonance strategy.

Core:
- Use CMF to validate real capital inflow (separate genuine buying vs. distribution).
- Trend filter: price above MA; CMF > 0 indicates net inflow.
- Entry: price breaks recent high and CMF crosses above threshold (default 0).
- Exit: CMF falls below threshold or price breaks the MA.
"""

import logging
from typing import Any, Dict, Optional

import backtrader.indicators as btind

from atm.trading.indicators.indicators import ChaikinMoneyFlow
from atm.trading.strategies.base import BaseStrategy

logger = logging.getLogger(__name__)


class CMFResonanceStrategy(BaseStrategy):
    """CMF plus trend resonance strategy."""

    params = (
        ("cmf_period", 20),
        ("cmf_threshold", 0.0),  # CMF 上穿此阈值视为资金净流入确认
        ("ma_period", 20),  # 趋势过滤均线
        ("breakout_lookback", 20),  # 价格突破的高点回看窗口
        ("strategy_config", None),
        ("run_id", None),
        ("ts_code", None),
    )

    def __init__(self):
        """Initialize indicators and state."""
        super().__init__()
        self.cmf = ChaikinMoneyFlow(period=self.p.cmf_period)
        self.ma = btind.SMA(self.data.close, period=self.p.ma_period)

        # 近高点参考
        self.recent_high = btind.Highest(self.data.close, period=self.p.breakout_lookback)

        # 状态
        self.entry_price: Optional[float] = None

        max_period = max(self.p.cmf_period, self.p.ma_period, self.p.breakout_lookback)
        self.addminperiod(max_period + 5)

    def _trend_ok(self) -> bool:
        try:
            return self.data.close[0] > self.ma[0] and self.ma[0] >= self.ma[-1]
        except Exception:
            return False

    def _cmf_ok(self) -> bool:
        try:
            return self.cmf[0] > self.p.cmf_threshold and self.cmf[0] >= self.cmf[-1]
        except Exception:
            return False

    def _price_breakout(self) -> bool:
        try:
            return self.data.close[0] > self.recent_high[-1] and self.data.close[0] > self.recent_high[0]
        except Exception:
            return False

    def next(self):
        # 入场
        if not self.position:
            if self._trend_ok() and self._cmf_ok() and self._price_breakout():
                self.buy()
                self.entry_price = float(self.data.close[0])
                logger.debug(
                    "Entry: price=%.2f cmf=%.4f ma=%.2f high=%.2f",
                    self.entry_price,
                    float(self.cmf[0]),
                    float(self.ma[0]),
                    float(self.recent_high[0]),
                )
            return

        # 持仓：退出条件
        price = float(self.data.close[0])

        cmf_drop = False
        try:
            cmf_drop = self.cmf[0] < self.p.cmf_threshold
        except Exception:
            cmf_drop = False

        trend_break = False
        try:
            trend_break = price < self.ma[0]
        except Exception:
            trend_break = False

        if cmf_drop or trend_break:
            logger.debug(
                "Exit: price=%.2f cmf=%.4f ma=%.2f drop=%s trend_break=%s",
                price,
                float(self.cmf[0]),
                float(self.ma[0]),
                cmf_drop,
                trend_break,
            )
            self.close()
            self.entry_price = None

    def get_info(self) -> Dict[str, Any]:
        info = super().get_info()
        info.update(
            {
                "cmf_period": self.p.cmf_period,
                "cmf_threshold": self.p.cmf_threshold,
                "ma_period": self.p.ma_period,
                "breakout_lookback": self.p.breakout_lookback,
            }
        )
        return info


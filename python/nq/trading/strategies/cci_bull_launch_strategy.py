"""
CCI 牛股启动策略

基于文章描述的简化规则：
- 使用 CCI(默认 20) 作为“牛股启动探测器”
- 当 CCI 从低于买入阈值（默认 +100）上穿该阈值，且价格站上短期均线并向上，买入
- 持有条件：CCI 不跌破卖出阈值（默认 +100）且短期均线不拐头向下
- 卖出条件：CCI 跌破卖出阈值，或短期均线拐头向下
"""

import logging
from typing import Any, Dict, Optional

import backtrader as bt

from nq.trading.strategies.base import BaseStrategy

logger = logging.getLogger(__name__)


class CCIBullLaunchStrategy(BaseStrategy):
    """
    CCI 牛股启动策略实现。

    逻辑概要：
    - 指标：CCI（默认周期 20），价格均线（默认 5 日）
    - 买入：CCI 从下方首次上穿买入阈值（默认 +100），且价格高于均线，均线向上
    - 卖出：CCI 跌破卖出阈值（默认 +100）或均线向下拐头
    """

    params = (
        ("cci_period", 20),
        ("ma_period", 5),
        ("cci_buy_level", 100),
        ("cci_sell_level", 100),
        ("strategy_config", None),
        ("run_id", None),
        ("ts_code", None),
    )

    def __init__(self):
        super().__init__()

        self.cci = bt.indicators.CommodityChannelIndex(period=self.p.cci_period)
        self.ma = bt.indicators.SMA(self.data.close, period=self.p.ma_period)

        max_period = max(self.p.cci_period, self.p.ma_period)
        self.addminperiod(max_period + 5)

    def _ma_trending_up(self) -> bool:
        try:
            return self.ma[0] > self.ma[-1]
        except (IndexError, TypeError):
            return False

    def _ma_turning_down(self) -> bool:
        try:
            return self.ma[0] < self.ma[-1]
        except (IndexError, TypeError):
            return False

    def next(self):
        try:
            cci_now = float(self.cci[0])
            cci_prev = float(self.cci[-1])
            price = float(self.data.close[0])
        except (IndexError, TypeError, ValueError):
            return

        # 买入：CCI 首次上穿 buy_level，价格高于均线，均线向上
        if not self.position:
            if cci_prev < self.p.cci_buy_level <= cci_now and price > self.ma[0] and self._ma_trending_up():
                logger.debug(
                    "CCI 上穿买入阈值，买入: cci_prev=%.2f cci=%.2f price=%.2f ma=%.2f",
                    cci_prev,
                    cci_now,
                    price,
                    self.ma[0],
                )
                self.buy(size=1.0)
                return

        # 卖出：CCI 跌破卖出阈值 或 均线拐头向下
        if self.position:
            sell_due_to_cci = cci_now < self.p.cci_sell_level
            sell_due_to_ma = self._ma_turning_down()

            if sell_due_to_cci or sell_due_to_ma:
                logger.debug(
                    "卖出: cci=%.2f(阈值 %.2f) price=%.2f ma=%.2f ma_down=%s",
                    cci_now,
                    self.p.cci_sell_level,
                    price,
                    self.ma[0],
                    sell_due_to_ma,
                )
                self.close()

    def get_info(self) -> Dict[str, Any]:
        info = super().get_info()
        info.update(
            {
                "cci_period": self.p.cci_period,
                "ma_period": self.p.ma_period,
                "cci_buy_level": self.p.cci_buy_level,
                "cci_sell_level": self.p.cci_sell_level,
            }
        )
        return info


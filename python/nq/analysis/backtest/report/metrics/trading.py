"""
Trading statistics metrics calculators.
"""

import logging

from .base import BaseMetricCalculator
from .registry import MetricRegistry
from ..models import BacktestData, MetricResult

logger = logging.getLogger(__name__)


@MetricRegistry.register(category="trading", name="total_trades", description="总交易次数")
class TotalTradesCalculator(BaseMetricCalculator):
    """Calculate total number of trades."""
    
    def calculate(self, data: BacktestData) -> MetricResult:
        """Calculate total number of trades."""
        if data.trades.empty:
            return MetricResult(
                name="total_trades",
                category="trading",
                value=0.0,
                description="总交易次数",
            )
        
        total_trades = len(data.trades)
        
        return MetricResult(
            name="total_trades",
            category="trading",
            value=int(total_trades),
            description="总交易次数",
        )


@MetricRegistry.register(category="trading", name="buy_count", description="买入次数")
class BuyCountCalculator(BaseMetricCalculator):
    """Calculate number of buy trades."""
    
    def calculate(self, data: BacktestData) -> MetricResult:
        """Calculate number of buy trades."""
        if data.trades.empty:
            return MetricResult(
                name="buy_count",
                category="trading",
                value=0.0,
                description="买入次数",
            )
        
        # Handle both 'side' and 'direction' field names
        if 'direction' in data.trades.columns:
            buy_count = len(data.trades[data.trades['direction'] == 1])
        elif 'side' in data.trades.columns:
            buy_count = len(data.trades[data.trades['side'] == 1])
        else:
            buy_count = 0
        
        return MetricResult(
            name="buy_count",
            category="trading",
            value=int(buy_count),
            description="买入次数",
        )


@MetricRegistry.register(category="trading", name="sell_count", description="卖出次数")
class SellCountCalculator(BaseMetricCalculator):
    """Calculate number of sell trades."""
    
    def calculate(self, data: BacktestData) -> MetricResult:
        """Calculate number of sell trades."""
        if data.trades.empty:
            return MetricResult(
                name="sell_count",
                category="trading",
                value=0.0,
                description="卖出次数",
            )
        
        # Handle both 'side' and 'direction' field names
        if 'direction' in data.trades.columns:
            sell_count = len(data.trades[data.trades['direction'] == -1])
        elif 'side' in data.trades.columns:
            sell_count = len(data.trades[data.trades['side'] == -1])
        else:
            sell_count = 0
        
        return MetricResult(
            name="sell_count",
            category="trading",
            value=int(sell_count),
            description="卖出次数",
        )


@MetricRegistry.register(category="trading", name="win_rate", description="胜率")
class WinRateCalculator(BaseMetricCalculator):
    """Calculate win rate."""
    
    def calculate(self, data: BacktestData) -> MetricResult:
        """Calculate win rate."""
        if data.trades.empty:
            return MetricResult(
                name="win_rate",
                category="trading",
                value=0.0,
                description="胜率（无交易数据）",
            )
        
        # Check if pnl_ratio column exists
        if 'pnl_ratio' not in data.trades.columns:
            return MetricResult(
                name="win_rate",
                category="trading",
                value=None,
                description="胜率（缺少盈亏数据）",
            )
        
        # Filter sell trades (only sell trades have pnl_ratio)
        sell_trades = data.trades[data.trades['pnl_ratio'].notna()]
        
        if len(sell_trades) == 0:
            return MetricResult(
                name="win_rate",
                category="trading",
                value=0.0,
                description="胜率（无卖出交易）",
            )
        
        winning_trades = len(sell_trades[sell_trades['pnl_ratio'] > 0])
        total_sell_trades = len(sell_trades)
        win_rate = winning_trades / total_sell_trades if total_sell_trades > 0 else 0.0
        
        return MetricResult(
            name="win_rate",
            category="trading",
            value=win_rate,
            unit="%",
            format="{:.2f}%",
            description="胜率",
        )


@MetricRegistry.register(category="trading", name="avg_hold_days", description="平均持仓天数")
class AvgHoldDaysCalculator(BaseMetricCalculator):
    """Calculate average holding period in days."""
    
    def calculate(self, data: BacktestData) -> MetricResult:
        """Calculate average holding period."""
        if data.trades.empty:
            return MetricResult(
                name="avg_hold_days",
                category="trading",
                value=0.0,
                description="平均持仓天数（无交易数据）",
            )
        
        # Check if hold_days column exists
        if 'hold_days' not in data.trades.columns:
            return MetricResult(
                name="avg_hold_days",
                category="trading",
                value=None,
                description="平均持仓天数（缺少持仓天数数据）",
            )
        
        hold_days = data.trades['hold_days'].dropna()
        
        if len(hold_days) == 0:
            return MetricResult(
                name="avg_hold_days",
                category="trading",
                value=0.0,
                description="平均持仓天数（无有效数据）",
            )
        
        avg_hold_days = hold_days.mean()
        
        return MetricResult(
            name="avg_hold_days",
            category="trading",
            value=float(avg_hold_days),
            format="{:.1f}",
            description="平均持仓天数",
        )


@MetricRegistry.register(category="trading", name="profit_factor", description="盈亏比")
class ProfitFactorCalculator(BaseMetricCalculator):
    """Calculate profit factor."""
    
    def calculate(self, data: BacktestData) -> MetricResult:
        """Calculate profit factor."""
        if data.trades.empty:
            return MetricResult(
                name="profit_factor",
                category="trading",
                value=0.0,
                description="盈亏比（无交易数据）",
            )
        
        # Check if pnl_ratio column exists
        if 'pnl_ratio' not in data.trades.columns:
            return MetricResult(
                name="profit_factor",
                category="trading",
                value=None,
                description="盈亏比（缺少盈亏数据）",
            )
        
        # Filter sell trades with pnl_ratio
        sell_trades = data.trades[data.trades['pnl_ratio'].notna()].copy()
        
        if len(sell_trades) == 0:
            return MetricResult(
                name="profit_factor",
                category="trading",
                value=0.0,
                description="盈亏比（无卖出交易）",
            )
        
        # Calculate total profit and total loss
        # Note: pnl_ratio might be in percentage or absolute value
        # We'll assume it's a ratio (e.g., 0.1 means 10% profit)
        total_profit = sell_trades[sell_trades['pnl_ratio'] > 0]['pnl_ratio'].sum()
        total_loss = abs(sell_trades[sell_trades['pnl_ratio'] < 0]['pnl_ratio'].sum())
        
        if total_loss > 0:
            profit_factor = total_profit / total_loss
        elif total_profit > 0:
            profit_factor = float('inf')
        else:
            profit_factor = 0.0
        
        return MetricResult(
            name="profit_factor",
            category="trading",
            value=profit_factor if profit_factor != float('inf') else None,
            format="{:.2f}",
            description="盈亏比（总盈利/总亏损）",
            metadata={"is_inf": profit_factor == float('inf')},
        )


@MetricRegistry.register(category="trading", name="winning_trades", description="盈利交易数")
class WinningTradesCalculator(BaseMetricCalculator):
    """Calculate number of winning trades."""
    
    def calculate(self, data: BacktestData) -> MetricResult:
        """Calculate number of winning trades."""
        if data.trades.empty:
            return MetricResult(
                name="winning_trades",
                category="trading",
                value=0.0,
                description="盈利交易数",
            )
        
        if 'pnl_ratio' not in data.trades.columns:
            return MetricResult(
                name="winning_trades",
                category="trading",
                value=None,
                description="盈利交易数（缺少盈亏数据）",
            )
        
        sell_trades = data.trades[data.trades['pnl_ratio'].notna()]
        winning_trades = len(sell_trades[sell_trades['pnl_ratio'] > 0])
        
        return MetricResult(
            name="winning_trades",
            category="trading",
            value=int(winning_trades),
            description="盈利交易数",
        )


@MetricRegistry.register(category="trading", name="losing_trades", description="亏损交易数")
class LosingTradesCalculator(BaseMetricCalculator):
    """Calculate number of losing trades."""
    
    def calculate(self, data: BacktestData) -> MetricResult:
        """Calculate number of losing trades."""
        if data.trades.empty:
            return MetricResult(
                name="losing_trades",
                category="trading",
                value=0.0,
                description="亏损交易数",
            )
        
        if 'pnl_ratio' not in data.trades.columns:
            return MetricResult(
                name="losing_trades",
                category="trading",
                value=None,
                description="亏损交易数（缺少盈亏数据）",
            )
        
        sell_trades = data.trades[data.trades['pnl_ratio'].notna()]
        losing_trades = len(sell_trades[sell_trades['pnl_ratio'] < 0])
        
        return MetricResult(
            name="losing_trades",
            category="trading",
            value=int(losing_trades),
            description="亏损交易数",
        )


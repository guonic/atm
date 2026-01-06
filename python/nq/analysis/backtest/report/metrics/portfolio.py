"""
Portfolio metrics calculators.
"""

import logging

import pandas as pd

try:
    import empyrical as ep
    HAS_EMPYRICAL = True
except ImportError:
    HAS_EMPYRICAL = False
    ep = None

from ..models import BacktestData, MetricResult
from .base import BaseMetricCalculator
from .registry import MetricRegistry

logger = logging.getLogger(__name__)


@MetricRegistry.register(category="portfolio", name="total_return", description="总收益率")
class TotalReturnCalculator(BaseMetricCalculator):
    """Calculate total return."""
    
    def calculate(self, data: BacktestData) -> MetricResult:
        """Calculate total return."""
        if data.ledger.empty:
            return MetricResult(
                name="total_return",
                category="portfolio",
                value=None,
                description="总收益率（数据不足）",
            )
        
        initial_nav = float(data.ledger.iloc[0]['nav'])
        final_nav = float(data.ledger.iloc[-1]['nav'])
        
        if initial_nav <= 0:
            return MetricResult(
                name="total_return",
                category="portfolio",
                value=None,
                description="总收益率（初始净值无效）",
            )
        
        total_return = (final_nav - initial_nav) / initial_nav
        
        return MetricResult(
            name="total_return",
            category="portfolio",
            value=total_return,
            unit="%",
            format="{:.2f}%",
            description="总收益率",
        )


@MetricRegistry.register(category="portfolio", name="annualized_return", description="年化收益率")
class AnnualizedReturnCalculator(BaseMetricCalculator):
    """Calculate annualized return."""
    
    def calculate(self, data: BacktestData) -> MetricResult:
        """Calculate annualized return."""
        if data.ledger.empty or len(data.ledger) < 2:
            return MetricResult(
                name="annualized_return",
                category="portfolio",
                value=None,
                description="年化收益率（数据不足）",
            )
        
        initial_nav = float(data.ledger.iloc[0]['nav'])
        final_nav = float(data.ledger.iloc[-1]['nav'])
        
        if initial_nav <= 0:
            return MetricResult(
                name="annualized_return",
                category="portfolio",
                value=None,
                description="年化收益率（初始净值无效）",
            )
        
        # Calculate number of trading days
        num_days = len(data.ledger)
        if num_days == 0:
            return MetricResult(
                name="annualized_return",
                category="portfolio",
                value=None,
                description="年化收益率（数据不足）",
            )
        
        total_return = (final_nav - initial_nav) / initial_nav
        years = num_days / 252.0  # Assume 252 trading days per year
        
        if years > 0:
            annualized_return = (1 + total_return) ** (1.0 / years) - 1
        else:
            annualized_return = None
        
        return MetricResult(
            name="annualized_return",
            category="portfolio",
            value=annualized_return,
            unit="%",
            format="{:.2f}%",
            description="年化收益率",
        )


@MetricRegistry.register(category="portfolio", name="volatility", description="波动率")
class VolatilityCalculator(BaseMetricCalculator):
    """Calculate annualized volatility."""
    
    def calculate(self, data: BacktestData) -> MetricResult:
        """Calculate annualized volatility."""
        if data.ledger.empty or len(data.ledger) < 2:
            return MetricResult(
                name="volatility",
                category="portfolio",
                value=None,
                description="波动率（数据不足）",
            )
        
        # Calculate daily returns
        navs = data.ledger['nav'].values
        returns = pd.Series(navs).pct_change().dropna()
        
        if len(returns) <= 1:
            return MetricResult(
                name="volatility",
                category="portfolio",
                value=None,
                description="波动率（数据不足）",
            )
        
        if HAS_EMPYRICAL:
            # Use empyrical for accurate calculation
            volatility = ep.annual_volatility(returns)
        else:
            # Manual calculation
            volatility = returns.std() * (252 ** 0.5)
        
        return MetricResult(
            name="volatility",
            category="portfolio",
            value=volatility,
            unit="%",
            format="{:.2f}%",
            description="年化波动率",
        )


@MetricRegistry.register(category="portfolio", name="sharpe_ratio", description="夏普比率")
class SharpeRatioCalculator(BaseMetricCalculator):
    """Calculate Sharpe ratio."""
    
    def calculate(self, data: BacktestData) -> MetricResult:
        """Calculate Sharpe ratio."""
        if data.ledger.empty or len(data.ledger) < 2:
            return MetricResult(
                name="sharpe_ratio",
                category="portfolio",
                value=None,
                description="夏普比率（数据不足）",
            )
        
        # Calculate daily returns
        navs = data.ledger['nav'].values
        returns = pd.Series(navs).pct_change().dropna()
        
        if len(returns) <= 1:
            return MetricResult(
                name="sharpe_ratio",
                category="portfolio",
                value=None,
                description="夏普比率（数据不足）",
            )
        
        if HAS_EMPYRICAL:
            # Use empyrical (assumes risk-free rate = 0)
            sharpe = ep.sharpe_ratio(returns, annualization=252)
        else:
            # Manual calculation (assumes risk-free rate = 0)
            if returns.std() == 0:
                sharpe = 0.0
            else:
                sharpe = returns.mean() / returns.std() * (252 ** 0.5)
        
        return MetricResult(
            name="sharpe_ratio",
            category="portfolio",
            value=sharpe,
            format="{:.4f}",
            description="夏普比率（年化，无风险利率=0）",
        )


@MetricRegistry.register(category="portfolio", name="max_drawdown", description="最大回撤")
class MaxDrawdownCalculator(BaseMetricCalculator):
    """Calculate maximum drawdown."""
    
    def calculate(self, data: BacktestData) -> MetricResult:
        """Calculate maximum drawdown."""
        if data.ledger.empty:
            return MetricResult(
                name="max_drawdown",
                category="portfolio",
                value=None,
                description="最大回撤（数据不足）",
            )
        
        navs = data.ledger['nav'].values
        account_values = pd.Series(navs).dropna()
        
        if len(account_values) == 0:
            return MetricResult(
                name="max_drawdown",
                category="portfolio",
                value=None,
                description="最大回撤（数据不足）",
            )
        
        if HAS_EMPYRICAL:
            # Use empyrical
            returns = account_values.pct_change().dropna()
            max_dd = ep.max_drawdown(returns)
        else:
            # Manual calculation
            running_max = account_values.expanding().max()
            drawdown = (account_values - running_max) / running_max
            max_dd = drawdown.min()
        
        return MetricResult(
            name="max_drawdown",
            category="portfolio",
            value=max_dd,
            unit="%",
            format="{:.2f}%",
            description="最大回撤",
        )


@MetricRegistry.register(category="portfolio", name="initial_account", description="初始账户价值")
class InitialAccountCalculator(BaseMetricCalculator):
    """Get initial account value."""
    
    def calculate(self, data: BacktestData) -> MetricResult:
        """Get initial account value."""
        if data.ledger.empty:
            return MetricResult(
                name="initial_account",
                category="portfolio",
                value=None,
                description="初始账户价值（数据不足）",
            )
        
        initial_account = float(data.ledger.iloc[0]['nav'])
        
        return MetricResult(
            name="initial_account",
            category="portfolio",
            value=initial_account,
            format="{:,.2f}",
            description="初始账户价值",
        )


@MetricRegistry.register(category="portfolio", name="final_account", description="最终账户价值")
class FinalAccountCalculator(BaseMetricCalculator):
    """Get final account value."""
    
    def calculate(self, data: BacktestData) -> MetricResult:
        """Get final account value."""
        if data.ledger.empty:
            return MetricResult(
                name="final_account",
                category="portfolio",
                value=None,
                description="最终账户价值（数据不足）",
            )
        
        final_account = float(data.ledger.iloc[-1]['nav'])
        
        return MetricResult(
            name="final_account",
            category="portfolio",
            value=final_account,
            format="{:,.2f}",
            description="最终账户价值",
        )


@MetricRegistry.register(category="portfolio", name="net_pnl", description="净盈亏")
class NetPnLCalculator(BaseMetricCalculator):
    """Calculate net P&L."""
    
    def get_dependencies(self) -> list:
        return ["portfolio.initial_account", "portfolio.final_account"]
    
    def calculate(self, data: BacktestData) -> MetricResult:
        """Calculate net P&L."""
        if data.ledger.empty:
            return MetricResult(
                name="net_pnl",
                category="portfolio",
                value=None,
                description="净盈亏（数据不足）",
            )
        
        initial_account = float(data.ledger.iloc[0]['nav'])
        final_account = float(data.ledger.iloc[-1]['nav'])
        net_pnl = final_account - initial_account
        
        return MetricResult(
            name="net_pnl",
            category="portfolio",
            value=net_pnl,
            format="{:,.2f}",
            description="净盈亏",
        )


@MetricRegistry.register(category="portfolio", name="trading_days", description="交易天数")
class TradingDaysCalculator(BaseMetricCalculator):
    """Get number of trading days."""
    
    def calculate(self, data: BacktestData) -> MetricResult:
        """Get number of trading days."""
        if data.ledger.empty:
            return MetricResult(
                name="trading_days",
                category="portfolio",
                value=0,
                description="交易天数",
            )
        
        trading_days = len(data.ledger)
        
        return MetricResult(
            name="trading_days",
            category="portfolio",
            value=int(trading_days),
            description="交易天数",
        )


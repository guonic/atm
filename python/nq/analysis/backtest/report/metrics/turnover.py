"""
Turnover statistics metrics calculators.
"""

import logging

from .base import BaseMetricCalculator
from .registry import MetricRegistry
from ..models import BacktestData, MetricResult

logger = logging.getLogger(__name__)


@MetricRegistry.register(category="turnover", name="total_turnover", description="总换手")
class TotalTurnoverCalculator(BaseMetricCalculator):
    """Calculate total turnover."""
    
    def calculate(self, data: BacktestData) -> MetricResult:
        """Calculate total turnover."""
        if data.ledger.empty:
            return MetricResult(
                name="total_turnover",
                category="turnover",
                value=None,
                description="总换手（数据不足）",
            )
        
        # Check if turnover_rate or deal_amount column exists
        if 'turnover_rate' in data.ledger.columns:
            total_turnover = data.ledger['turnover_rate'].sum()
        elif 'deal_amount' in data.ledger.columns:
            # Use deal_amount as proxy for turnover
            total_turnover = data.ledger['deal_amount'].sum()
        else:
            return MetricResult(
                name="total_turnover",
                category="turnover",
                value=None,
                description="总换手（缺少换手数据）",
            )
        
        return MetricResult(
            name="total_turnover",
            category="turnover",
            value=float(total_turnover),
            format="{:,.2f}",
            description="总换手",
        )


@MetricRegistry.register(category="turnover", name="avg_daily_turnover", description="平均日换手率")
class AvgDailyTurnoverCalculator(BaseMetricCalculator):
    """Calculate average daily turnover rate."""
    
    def calculate(self, data: BacktestData) -> MetricResult:
        """Calculate average daily turnover rate."""
        if data.ledger.empty:
            return MetricResult(
                name="avg_daily_turnover",
                category="turnover",
                value=None,
                description="平均日换手率（数据不足）",
            )
        
        # Check if turnover_rate column exists
        if 'turnover_rate' not in data.ledger.columns:
            return MetricResult(
                name="avg_daily_turnover",
                category="turnover",
                value=None,
                description="平均日换手率（缺少换手率数据）",
            )
        
        turnover_rates = data.ledger['turnover_rate'].dropna()
        
        if len(turnover_rates) == 0:
            return MetricResult(
                name="avg_daily_turnover",
                category="turnover",
                value=None,
                description="平均日换手率（无有效数据）",
            )
        
        avg_daily_turnover = turnover_rates.mean()
        
        return MetricResult(
            name="avg_daily_turnover",
            category="turnover",
            value=avg_daily_turnover,
            unit="%",
            format="{:.2f}%",
            description="平均日换手率",
        )


@MetricRegistry.register(category="turnover", name="pos_count", description="持仓数量")
class PosCountCalculator(BaseMetricCalculator):
    """Get average position count."""
    
    def calculate(self, data: BacktestData) -> MetricResult:
        """Get average position count."""
        if data.ledger.empty:
            return MetricResult(
                name="pos_count",
                category="turnover",
                value=None,
                description="持仓数量（数据不足）",
            )
        
        # Check if pos_count column exists
        if 'pos_count' not in data.ledger.columns:
            return MetricResult(
                name="pos_count",
                category="turnover",
                value=None,
                description="持仓数量（缺少持仓数量数据）",
            )
        
        pos_counts = data.ledger['pos_count'].dropna()
        
        if len(pos_counts) == 0:
            return MetricResult(
                name="pos_count",
                category="turnover",
                value=None,
                description="持仓数量（无有效数据）",
            )
        
        avg_pos_count = pos_counts.mean()
        
        return MetricResult(
            name="pos_count",
            category="turnover",
            value=float(avg_pos_count),
            format="{:.1f}",
            description="平均持仓数量",
        )


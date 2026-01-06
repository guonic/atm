"""
Pydantic schemas for Eidos API requests and responses.
"""

from __future__ import annotations

from datetime import date as Date, datetime as DateTime
from typing import Any, Dict, List, Optional

from pydantic import Field, BaseModel

from nq.models.eidos import (
    Experiment,
    LedgerEntry,
    Trade,
    ModelOutput,
)


# Request schemas
class ExperimentCreateRequest(BaseModel):
    """Request schema for creating a new experiment."""

    name: str = Field(..., description="Experiment name")
    model_type: Optional[str] = Field(None, description="Model type")
    engine_type: Optional[str] = Field(None, description="Engine type")
    start_date: Date = Field(..., description="Start date")
    end_date: Date = Field(..., description="End date")
    config: Dict[str, Any] = Field(default_factory=dict, description="Configuration")
    metrics_summary: Optional[Dict[str, Any]] = Field(None, description="Metrics summary")


# Response schemas
class ExperimentResponse(Experiment):
    """Response schema for experiment data."""

    created_at: Optional[DateTime] = Field(None, description="Creation timestamp")
    updated_at: Optional[DateTime] = Field(None, description="Update timestamp")


class LedgerEntryResponse(LedgerEntry):
    """Response schema for ledger entry."""

    pass


class TradeResponse(Trade):
    """Response schema for trade data."""

    pass


class ModelOutputResponse(ModelOutput):
    """Response schema for model output."""

    pass


class PerformanceMetricsResponse(BaseModel):
    """Response schema for performance metrics."""

    total_return: float = Field(..., description="Total return ratio")
    max_drawdown: float = Field(..., description="Maximum drawdown ratio")
    final_nav: float = Field(..., description="Final NAV")
    trading_days: int = Field(..., description="Number of trading days")
    sharpe_ratio: Optional[float] = Field(None, description="Sharpe ratio")
    annual_return: Optional[float] = Field(None, description="Annual return ratio")


class TradeStatsResponse(BaseModel):
    """Response schema for trade statistics."""

    total_trades: int = Field(..., description="Total number of trades")
    buy_count: int = Field(..., description="Number of buy trades")
    sell_count: int = Field(..., description="Number of sell trades")
    win_rate: float = Field(..., description="Win rate (0-1)")
    avg_hold_days: float = Field(..., description="Average holding days")


class ErrorResponse(BaseModel):
    """Error response schema."""

    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Error details")


# Report schemas
class MetricResultResponse(BaseModel):
    """Response schema for metric result."""
    
    name: str = Field(..., description="Metric name")
    category: str = Field(..., description="Metric category")
    value: Optional[float] = Field(None, description="Metric value")
    unit: Optional[str] = Field(None, description="Unit")
    format: Optional[str] = Field(None, description="Format string")
    description: Optional[str] = Field(None, description="Description")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class BacktestReportResponse(BaseModel):
    """Response schema for backtest report."""
    
    exp_id: str = Field(..., description="Experiment ID")
    experiment_name: str = Field(..., description="Experiment name")
    start_date: str = Field(..., description="Start date")
    end_date: str = Field(..., description="End date")
    generated_at: str = Field(..., description="Report generation timestamp")
    metrics: List[MetricResultResponse] = Field(..., description="List of all metrics")
    metrics_by_category: Dict[str, List[MetricResultResponse]] = Field(
        ..., description="Metrics organized by category"
    )


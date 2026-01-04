"""
Pydantic schemas for Eidos API requests and responses.
"""

from datetime import date as Date, datetime as DateTime
from typing import Any, Dict, Optional

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


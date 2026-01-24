"""
Eidos (Universal Backtest Attribution System) data models.

Defines Pydantic models for all Eidos database tables.
"""

from datetime import date as Date, datetime as DateTime
from decimal import Decimal
from typing import Any, Dict, List, Optional

from pydantic import Field

from nq.models.base import BaseModel


class Experiment(BaseModel):
    """Experiment metadata model (bt_experiment table)."""

    exp_id: str = Field(..., description="Experiment ID (8-character hex string)")
    name: str = Field(..., description="Experiment name")
    model_type: Optional[str] = Field(None, description="Model type (e.g., 'GNN', 'GRU', 'Linear')")
    engine_type: Optional[str] = Field(None, description="Engine type (e.g., 'Qlib', 'Backtrader')")
    start_date: Date = Field(..., description="Backtest start date")
    end_date: Date = Field(..., description="Backtest end date")
    config: Dict[str, Any] = Field(default_factory=dict, description="Experiment configuration (JSONB)")
    metrics_summary: Optional[Dict[str, Any]] = Field(None, description="Summary metrics (JSONB)")
    version: int = Field(1, description="Experiment version")
    status: str = Field("running", description="Experiment status (running, completed, failed)")

    class Config:
        """Pydantic configuration."""

        json_encoders = {
            Date: lambda v: v.isoformat() if v else None,
            DateTime: lambda v: v.isoformat() if v else None,
            Decimal: lambda v: float(v) if v else None,
        }


class LedgerEntry(BaseModel):
    """Account daily ledger model (bt_ledger table)."""

    exp_id: str = Field(..., description="Experiment ID")
    date: Date = Field(..., description="Trading date")
    nav: Decimal = Field(..., description="Net Asset Value (NAV)")
    cash: Optional[Decimal] = Field(None, description="Cash amount")
    market_value: Optional[Decimal] = Field(None, description="Market value of positions")
    deal_amount: Optional[Decimal] = Field(None, description="Total deal amount for the day")
    turnover_rate: Optional[float] = Field(None, description="Turnover rate")
    pos_count: Optional[int] = Field(None, description="Number of positions")

    class Config:
        """Pydantic configuration."""

        json_encoders = {
            Date: lambda v: v.isoformat() if v else None,
            DateTime: lambda v: v.isoformat() if v else None,
            Decimal: lambda v: float(v) if v else None,
        }


class Trade(BaseModel):
    """Trade order model (bt_trades table)."""

    trade_id: Optional[int] = Field(None, description="Trade ID (auto-generated)")
    exp_id: str = Field(..., description="Experiment ID")
    symbol: str = Field(..., description="Stock symbol")
    deal_time: DateTime = Field(..., description="Trade execution time")
    direction: int = Field(..., alias="side", description="Trade direction (1=Buy, -1=Sell)")
    price: Decimal = Field(..., description="Trade price")
    amount: int = Field(..., description="Trade amount (shares)")
    rank_at_deal: Optional[int] = Field(None, description="Model rank at deal time")
    score_at_deal: Optional[float] = Field(None, description="Model score at deal time")
    reason: Optional[str] = Field(None, description="Trade reason (e.g., 'rank_out', 'stop_loss')")
    pnl_ratio: Optional[float] = Field(None, description="P&L ratio (for sell orders)")
    hold_days: Optional[int] = Field(None, description="Holding period in days")

    class Config:
        """Pydantic configuration."""

        populate_by_name = True  # Allow both 'side' and 'direction' as input
        json_encoders = {
            Date: lambda v: v.isoformat() if v else None,
            DateTime: lambda v: v.isoformat() if v else None,
            Decimal: lambda v: float(v) if v else None,
        }


class ModelOutput(BaseModel):
    """Model dense output model (bt_model_outputs table)."""

    exp_id: str = Field(..., description="Experiment ID")
    date: Date = Field(..., description="Prediction date")
    symbol: str = Field(..., description="Stock symbol")
    score: float = Field(..., description="Model prediction score")
    rank: int = Field(..., description="Cross-sectional rank")
    extra_scores: Optional[Dict[str, Any]] = Field(None, description="Additional scores (JSONB)")

    class Config:
        """Pydantic configuration."""

        json_encoders = {
            Date: lambda v: v.isoformat() if v else None,
            DateTime: lambda v: v.isoformat() if v else None,
            Decimal: lambda v: float(v) if v else None,
        }


class ModelLink(BaseModel):
    """Model link model (bt_model_links table, for GNNs)."""

    exp_id: str = Field(..., description="Experiment ID")
    date: Date = Field(..., description="Link date")
    source: str = Field(..., description="Source node symbol")
    target: str = Field(..., description="Target node symbol")
    weight: float = Field(..., description="Link weight")
    link_type: str = Field("attention", description="Link type (e.g., 'attention', 'correlation')")

    class Config:
        """Pydantic configuration."""

        json_encoders = {
            Date: lambda v: v.isoformat() if v else None,
            DateTime: lambda v: v.isoformat() if v else None,
            Decimal: lambda v: float(v) if v else None,
        }


class Embedding(BaseModel):
    """Embedding model (bt_embeddings table)."""

    exp_id: str = Field(..., description="Experiment ID")
    date: Date = Field(..., description="Embedding date")
    symbol: str = Field(..., description="Stock symbol")
    vec: List[float] = Field(..., description="Embedding vector")
    vec_dim: int = Field(..., description="Vector dimension")

    class Config:
        """Pydantic configuration."""

        json_encoders = {
            Date: lambda v: v.isoformat() if v else None,
            DateTime: lambda v: v.isoformat() if v else None,
            Decimal: lambda v: float(v) if v else None,
        }


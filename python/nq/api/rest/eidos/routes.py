"""
FastAPI routes for Eidos API.
"""

from datetime import date, datetime
from typing import List, Optional

from fastapi import APIRouter, Query

from nq.api.rest.eidos import handlers
from nq.api.rest.eidos.schemas import (
    ExperimentResponse,
    LedgerEntryResponse,
    TradeResponse,
    PerformanceMetricsResponse,
    TradeStatsResponse,
)

router = APIRouter(prefix="/api/v1", tags=["eidos"])


@router.get("/experiments", response_model=List[ExperimentResponse])
async def get_experiments():
    """
    Get all experiments.
    
    Returns:
        List of all experiments.
    """
    return await handlers.get_experiments_handler()


@router.get("/experiments/{exp_id}", response_model=ExperimentResponse)
async def get_experiment(exp_id: str):
    """
    Get a single experiment by ID.
    
    Args:
        exp_id: Experiment ID (8-character hex string).
    
    Returns:
        Experiment data.
    """
    return await handlers.get_experiment_handler(exp_id)


@router.get("/experiments/{exp_id}/ledger", response_model=List[LedgerEntryResponse])
async def get_ledger(
    exp_id: str,
    start_date: Optional[date] = Query(None, description="Start date filter"),
    end_date: Optional[date] = Query(None, description="End date filter"),
):
    """
    Get ledger entries (daily NAV) for an experiment.
    
    Args:
        exp_id: Experiment ID.
        start_date: Optional start date filter.
        end_date: Optional end date filter.
    
    Returns:
        List of ledger entries.
    """
    return await handlers.get_ledger_handler(exp_id, start_date=start_date, end_date=end_date)


@router.get("/experiments/{exp_id}/trades", response_model=List[TradeResponse])
async def get_trades(
    exp_id: str,
    symbol: Optional[str] = Query(None, description="Symbol filter"),
    start_date: Optional[datetime] = Query(None, description="Start date/time filter"),
    end_date: Optional[datetime] = Query(None, description="End date/time filter"),
):
    """
    Get trades for an experiment.
    
    Args:
        exp_id: Experiment ID.
        symbol: Optional symbol filter.
        start_date: Optional start date/time filter.
        end_date: Optional end date/time filter.
    
    Returns:
        List of trades.
    """
    return await handlers.get_trades_handler(
        exp_id, symbol=symbol, start_date=start_date, end_date=end_date
    )


@router.get("/experiments/{exp_id}/metrics", response_model=PerformanceMetricsResponse)
async def get_performance_metrics(exp_id: str):
    """
    Get performance metrics for an experiment.
    
    Calculates:
    - Total return
    - Maximum drawdown
    - Final NAV
    - Trading days
    - Sharpe ratio (if available)
    - Annual return (if available)
    
    Args:
        exp_id: Experiment ID.
    
    Returns:
        Performance metrics.
    """
    return await handlers.get_performance_metrics_handler(exp_id)


@router.get("/experiments/{exp_id}/trade-stats", response_model=TradeStatsResponse)
async def get_trade_stats(exp_id: str):
    """
    Get trade statistics for an experiment.
    
    Calculates:
    - Total trades
    - Buy/Sell counts
    - Win rate
    - Average holding days
    
    Args:
        exp_id: Experiment ID.
    
    Returns:
        Trade statistics.
    """
    return await handlers.get_trade_stats_handler(exp_id)


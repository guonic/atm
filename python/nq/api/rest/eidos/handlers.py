"""
Request handlers for Eidos API endpoints.
"""

import logging
from datetime import date, datetime
from decimal import Decimal
from typing import List, Optional

from nq.api.rest.eidos.dependencies import get_eidos_repo
from nq.api.rest.eidos.schemas import (
    ExperimentResponse,
    LedgerEntryResponse,
    TradeResponse,
    PerformanceMetricsResponse,
    TradeStatsResponse,
    ErrorResponse,
)
from nq.repo.eidos_repo import EidosRepo

logger = logging.getLogger(__name__)


async def get_experiments_handler() -> List[ExperimentResponse]:
    """
    Get all experiments.
    
    Returns:
        List of experiments.
    """
    repo = get_eidos_repo()
    # Get all experiments (with pagination, but we'll fetch all)
    experiments = repo.experiment.list_experiments(limit=1000, offset=0)
    
    result = []
    for exp in experiments:
        try:
            # Convert date strings to date objects if needed
            if isinstance(exp.get("start_date"), str):
                exp["start_date"] = date.fromisoformat(exp["start_date"])
            if isinstance(exp.get("end_date"), str):
                exp["end_date"] = date.fromisoformat(exp["end_date"])
            result.append(ExperimentResponse(**exp))
        except Exception as e:
            logger.warning(f"Failed to parse experiment {exp.get('exp_id')}: {e}")
            continue
    
    return result


async def get_experiment_handler(exp_id: str) -> ExperimentResponse:
    """
    Get a single experiment by ID.
    
    Args:
        exp_id: Experiment ID.
    
    Returns:
        Experiment data.
    
    Raises:
        HTTPException: If experiment not found.
    """
    from fastapi import HTTPException
    
    repo = get_eidos_repo()
    exp = repo.experiment.get_experiment(exp_id)
    
    if not exp:
        raise HTTPException(status_code=404, detail=f"Experiment {exp_id} not found")
    
    # Convert date strings to date objects if needed
    if isinstance(exp.get("start_date"), str):
        exp["start_date"] = date.fromisoformat(exp["start_date"])
    if isinstance(exp.get("end_date"), str):
        exp["end_date"] = date.fromisoformat(exp["end_date"])
    
    return ExperimentResponse(**exp)


async def get_ledger_handler(
    exp_id: str,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
) -> List[LedgerEntryResponse]:
    """
    Get ledger entries for an experiment.
    
    Args:
        exp_id: Experiment ID.
        start_date: Optional start date filter.
        end_date: Optional end date filter.
    
    Returns:
        List of ledger entries.
    """
    repo = get_eidos_repo()
    entries = repo.ledger.get_ledger(exp_id, start_date=start_date, end_date=end_date)
    
    result = []
    for entry in entries:
        try:
            # Convert date strings to date objects if needed
            if isinstance(entry.get("date"), str):
                entry["date"] = date.fromisoformat(entry["date"])
            result.append(LedgerEntryResponse(**entry))
        except Exception as e:
            logger.warning(f"Failed to parse ledger entry: {e}")
            continue
    
    return result


async def get_trades_handler(
    exp_id: str,
    symbol: Optional[str] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
) -> List[TradeResponse]:
    """
    Get trades for an experiment.
    
    Args:
        exp_id: Experiment ID.
        symbol: Optional symbol filter.
        start_date: Optional start date filter.
        end_date: Optional end date filter.
    
    Returns:
        List of trades.
    """
    logger.info(f"Getting trades for exp_id: {exp_id}, symbol: {symbol}, start_date: {start_date}, end_date: {end_date}")
    
    repo = get_eidos_repo()
    trades = repo.trades.get_trades(
        exp_id, start_time=start_date, end_time=end_date, symbol=symbol
    )
    
    logger.info(f"Retrieved {len(trades)} trades from repository")
    
    result = []
    for idx, trade in enumerate(trades):
        try:
            # Ensure we have 'side' field for Pydantic (it uses alias="side")
            # The repository already converts 'side' to 'direction', but Pydantic needs 'side' due to alias
            if "direction" in trade and "side" not in trade:
                trade["side"] = trade["direction"]
            elif "side" in trade and "direction" not in trade:
                trade["direction"] = trade["side"]
            
            # Convert datetime strings if needed
            if isinstance(trade.get("deal_time"), str):
                trade["deal_time"] = datetime.fromisoformat(trade["deal_time"].replace("Z", "+00:00"))
            # Ensure exp_id is present
            if "exp_id" not in trade:
                trade["exp_id"] = exp_id
            result.append(TradeResponse(**trade))
        except Exception as e:
            logger.warning(f"Failed to parse trade {idx}: {e}, trade data: {trade}", exc_info=True)
            continue
    
    logger.info(f"Successfully parsed {len(result)} trades for exp_id: {exp_id}")
    return result


async def get_performance_metrics_handler(exp_id: str) -> PerformanceMetricsResponse:
    """
    Calculate performance metrics for an experiment.
    
    Args:
        exp_id: Experiment ID.
    
    Returns:
        Performance metrics.
    
    Raises:
        HTTPException: If experiment not found or insufficient data.
    """
    from fastapi import HTTPException
    
    repo = get_eidos_repo()
    
    # Get experiment to verify it exists
    exp = repo.experiment.get_experiment(exp_id)
    if not exp:
        raise HTTPException(status_code=404, detail=f"Experiment {exp_id} not found")
    
    # Get ledger entries
    entries = repo.ledger.get_ledger(exp_id)
    if not entries:
        raise HTTPException(status_code=404, detail=f"No ledger data found for experiment {exp_id}")
    
    # Sort by date
    entries.sort(key=lambda x: x.get("date", ""))
    
    # Calculate metrics
    initial_nav = entries[0].get("nav", 0)
    final_nav = entries[-1].get("nav", 0)
    
    if isinstance(initial_nav, (str, Decimal)):
        initial_nav = float(initial_nav)
    if isinstance(final_nav, (str, Decimal)):
        final_nav = float(final_nav)
    
    total_return = (final_nav - initial_nav) / initial_nav if initial_nav > 0 else 0.0
    
    # Calculate max drawdown
    max_nav = initial_nav
    max_drawdown = 0.0
    for entry in entries:
        nav = entry.get("nav", 0)
        if isinstance(nav, (str, Decimal)):
            nav = float(nav)
        if nav > max_nav:
            max_nav = nav
        drawdown = (max_nav - nav) / max_nav if max_nav > 0 else 0.0
        if drawdown > max_drawdown:
            max_drawdown = drawdown
    
    trading_days = len(entries)
    
    # Get metrics from experiment if available
    metrics_summary = exp.get("metrics_summary", {}) or {}
    sharpe_ratio = metrics_summary.get("sharpe_ratio")
    annual_return = metrics_summary.get("annual_return")
    
    return PerformanceMetricsResponse(
        total_return=total_return,
        max_drawdown=max_drawdown,
        final_nav=float(final_nav),  # Convert to float for JSON serialization
        trading_days=trading_days,
        sharpe_ratio=sharpe_ratio,
        annual_return=annual_return,
    )


async def get_trade_stats_handler(exp_id: str) -> TradeStatsResponse:
    """
    Calculate trade statistics for an experiment.
    
    Args:
        exp_id: Experiment ID.
    
    Returns:
        Trade statistics.
    
    Raises:
        HTTPException: If experiment not found.
    """
    from fastapi import HTTPException
    
    repo = get_eidos_repo()
    
    # Get experiment to verify it exists
    exp = repo.experiment.get_experiment(exp_id)
    if not exp:
        raise HTTPException(status_code=404, detail=f"Experiment {exp_id} not found")
    
    # Get all trades
    trades = repo.trades.get_trades(exp_id)
    
    logger.info(f"Retrieved {len(trades)} trades for stats calculation")
    
    if not trades:
        return TradeStatsResponse(
            total_trades=0,
            buy_count=0,
            sell_count=0,
            win_rate=0.0,
            avg_hold_days=0.0,
        )
    
    # Get side/direction value - repository converts 'side' to 'direction', but may have both
    buy_count = 0
    sell_count = 0
    for t in trades:
        # Check both 'side' and 'direction' fields
        side_value = t.get("side") or t.get("direction")
        if side_value == 1:
            buy_count += 1
        elif side_value == -1:
            sell_count += 1
        else:
            logger.warning(f"Trade {t.get('trade_id')} has invalid side/direction: {side_value}")
    
    logger.info(f"Buy count: {buy_count}, Sell count: {sell_count}, Total: {len(trades)}")
    
    # Calculate win rate from trades with pnl_ratio
    winning_trades = [t for t in trades if t.get("pnl_ratio") and t.get("pnl_ratio", 0) > 0]
    win_rate = len(winning_trades) / len(trades) if trades else 0.0
    
    # Calculate average hold days
    hold_days_list = [t.get("hold_days") for t in trades if t.get("hold_days") is not None]
    avg_hold_days = sum(hold_days_list) / len(hold_days_list) if hold_days_list else 0.0
    
    return TradeStatsResponse(
        total_trades=len(trades),
        buy_count=buy_count,
        sell_count=sell_count,
        win_rate=win_rate,
        avg_hold_days=avg_hold_days,
    )


"""
Converter for custom backtest results to Eidos format.

Converts custom backtest results (from run_custom_backtest) to QlibBacktestResult format
for Eidos integration.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import date

import pandas as pd
import numpy as np

from nq.analysis.backtest.qlib_types import (
    QlibBacktestResult,
    PortfolioMetrics,
    PositionDetails,
    Position,
    Indicator,
    IndicatorEntry,
)
from nq.analysis.backtest.date_utils import to_date
from nq.trading.state import Order, OrderStatus, OrderSide

logger = logging.getLogger(__name__)


def convert_custom_results_to_qlib_format(
    results: Dict[str, Any],
    initial_cash: float,
) -> QlibBacktestResult:
    """
    Convert custom backtest results to QlibBacktestResult format.
    
    Args:
        results: Results dict from run_custom_backtest with keys:
            - 'account': Account object
            - 'positions': Dict of Position objects
            - 'orders': List of Order objects
            - 'snapshots': List of snapshot dicts
        initial_cash: Initial cash amount.
    
    Returns:
        QlibBacktestResult instance.
    """
    snapshots = results.get('snapshots', [])
    orders = results.get('orders', [])
    account = results.get('account')
    positions = results.get('positions', {})
    
    if not snapshots:
        raise ValueError("No snapshots found in results")
    
    # Build metric_df from snapshots
    metric_data = []
    prev_total_value = initial_cash
    
    for snapshot in snapshots:
        date_str = snapshot['date']
        date_ts = pd.Timestamp(date_str)
        total_value = snapshot['total_value']
        cash = snapshot['cash']
        holdings_value = snapshot.get('holdings_value', total_value - cash)
        position_count = snapshot.get('position_count', 0)
        
        # Calculate daily return
        daily_return = (total_value - prev_total_value) / prev_total_value if prev_total_value > 0 else 0.0
        prev_total_value = total_value
        
        # Calculate deal_amount from orders executed on this date
        deal_amount = 0.0
        for order in orders:
            if order.status in [OrderStatus.FILLED, OrderStatus.PARTIAL_FILLED]:
                if order.date.date() == date_ts.date() and order.filled_price is not None:
                    deal_amount += abs(order.filled_amount * order.filled_price)
        
        # Calculate turnover_rate (deal_amount / total_value)
        turnover_rate = deal_amount / total_value if total_value > 0 else 0.0
        
        metric_data.append({
            'return': daily_return,
            'cash': cash,
            'market_value': holdings_value,
            'deal_amount': deal_amount,
            'turnover_rate': turnover_rate,
            'pos_count': position_count,
        })
    
    # Create metric_df
    dates = [pd.Timestamp(s['date']) for s in snapshots]
    metric_df = pd.DataFrame(metric_data, index=dates)
    
    # Build position_details from snapshots and positions
    position_details_dict = {}
    for snapshot in snapshots:
        date_str = snapshot['date']
        date_obj = pd.to_datetime(date_str).date()
        cash = snapshot['cash']
        holdings_value = snapshot.get('holdings_value', 0.0)
        
        # Build position dict for this date
        pos_dict: Dict[str, Any] = {
            'cash': cash,
            'market_value': holdings_value,
        }
        
        # Add holdings from current positions (if available)
        # Note: We use positions from the final state, as we don't track historical positions
        # This is a limitation - ideally we should track positions at each date
        if date_obj == dates[-1].date():  # Only for last date
            for symbol, position in positions.items():
                pos_dict[symbol] = {
                    'price': position.entry_price,
                    'amount': position.amount,
                }
        
        position_details_dict[date_str] = pos_dict
    
    # Create PositionDetails
    position_details = None
    if position_details_dict:
        positions_obj = {}
        for date_str, pos_data in position_details_dict.items():
            date_obj = pd.to_datetime(date_str).date()
            positions_obj[date_obj] = Position(date_obj, pos_data)
        position_details = PositionDetails(positions=positions_obj)
    
    # Create PortfolioMetrics
    portfolio_metrics = PortfolioMetrics(
        metric_df=metric_df,
        position_details=position_details,
    )
    
    # Create empty Indicator (no indicators in custom framework yet)
    indicator = Indicator(entries={})
    
    return QlibBacktestResult(
        portfolio_metrics=portfolio_metrics,
        indicator=indicator,
    )


def extract_trades_from_custom_orders(
    orders: List[Order],
) -> List[Dict[str, Any]]:
    """
    Extract trade data from custom Order objects.
    
    Args:
        orders: List of Order objects from custom backtest.
    
    Returns:
        List of trade records in Eidos format.
    """
    trades_data = []
    
    for order in orders:
        # Only include filled orders
        if order.status not in [OrderStatus.FILLED, OrderStatus.PARTIAL_FILLED]:
            continue
        
        # Skip orders without filled price/amount
        if order.filled_price is None or order.filled_amount <= 0:
            continue
        
        # Convert OrderSide to Eidos direction (1=Buy, -1=Sell)
        if order.side == OrderSide.BUY:
            direction = 1
        elif order.side == OrderSide.SELL:
            direction = -1
        else:
            logger.warning(f"Unknown order side: {order.side}, skipping")
            continue
        
        # Convert date to datetime
        deal_time = pd.to_datetime(order.date)
        
        trades_data.append({
            "symbol": order.symbol,
            "deal_time": deal_time,
            "side": direction,  # Use 'side' instead of 'direction' for Eidos (1=Buy, -1=Sell)
            "price": float(order.filled_price),
            "amount": int(order.filled_amount),
            "rank_at_deal": None,  # Not available in custom framework
            "score_at_deal": None,  # Not available in custom framework
            "reason": None,  # Not available in custom framework
            "pnl_ratio": None,  # Not available in custom framework
            "hold_days": None,  # Not available in custom framework
        })
    
    return trades_data


def save_custom_backtest_to_eidos(
    exp_id: str,
    writer: "EidosBacktestWriter",
    results: Dict[str, Any],
    initial_cash: float,
    predictions: Optional[pd.DataFrame] = None,
) -> Dict[str, int]:
    """
    Save custom backtest results to Eidos.
    
    Args:
        exp_id: Experiment ID.
        writer: EidosBacktestWriter instance.
        results: Results dict from run_custom_backtest.
        initial_cash: Initial cash amount.
        predictions: Optional predictions DataFrame (for model outputs).
    
    Returns:
        Dictionary with counts of inserted records.
    """
    from nq.analysis.backtest.eidos_structure_expert import (
        extract_model_outputs_from_predictions,
        extract_model_links_from_graph,
    )
    
    logger.info(f"Saving custom backtest results to Eidos (exp_id: {exp_id})")
    
    # Convert custom results to QlibBacktestResult format
    qlib_result = convert_custom_results_to_qlib_format(results, initial_cash)
    logger.info("Converted custom results to QlibBacktestResult format")
    
    # Extract ledger data
    ledger_data = extract_ledger_from_custom_results(qlib_result, initial_cash)
    logger.info(f"Extracted {len(ledger_data)} ledger records")
    
    # Extract trades data from orders
    orders = results.get('orders', [])
    trades_data = extract_trades_from_custom_orders(orders)
    logger.info(f"Extracted {len(trades_data)} trade records")
    
    # Extract model outputs (if predictions provided)
    model_outputs_data = []
    if predictions is not None and not predictions.empty:
        model_outputs_data = extract_model_outputs_from_predictions(predictions)
        logger.info(f"Extracted {len(model_outputs_data)} model output records")
    
    # Save to Eidos using writer's methods
    counts = {}
    
    # Save ledger
    if ledger_data:
        ledger_counts = writer.repo.ledger.batch_upsert(exp_id, ledger_data)
        counts['ledger'] = ledger_counts
        logger.info(f"Saved {ledger_counts} ledger records")
    
    # Save trades
    if trades_data:
        trade_counts = writer.repo.trades.batch_insert(exp_id, trades_data)
        counts['trades'] = trade_counts
        logger.info(f"Saved {trade_counts} trade records")
    
    # Save model outputs
    if model_outputs_data:
        model_output_counts = writer.repo.model_outputs.batch_upsert(exp_id, model_outputs_data)
        counts['model_outputs'] = model_output_counts
        logger.info(f"Saved {model_output_counts} model output records")
    
    return counts


def extract_ledger_from_custom_results(
    qlib_result: QlibBacktestResult,
    initial_cash: float,
) -> List[Dict[str, Any]]:
    """
    Extract ledger data from converted QlibBacktestResult.
    
    Args:
        qlib_result: QlibBacktestResult from convert_custom_results_to_qlib_format.
        initial_cash: Initial cash amount.
    
    Returns:
        List of ledger records.
    """
    ledger_data = []
    
    portfolio_metrics = qlib_result.portfolio_metrics
    metric_df = portfolio_metrics.metric_df
    
    if metric_df is None or metric_df.empty:
        return ledger_data
    
    # Calculate cumulative NAV from returns
    if "return" not in metric_df.columns:
        return ledger_data
    
    returns = metric_df["return"]
    nav = initial_cash * (1 + returns).cumprod()
    
    for date_val, nav_val in nav.items():
        date_obj = to_date(date_val)
        
        # Extract metrics from DataFrame
        cash = metric_df.loc[date_val, "cash"] if "cash" in metric_df.columns else 0.0
        market_value = metric_df.loc[date_val, "market_value"] if "market_value" in metric_df.columns else 0.0
        deal_amount = metric_df.loc[date_val, "deal_amount"] if "deal_amount" in metric_df.columns else 0.0
        turnover_rate = metric_df.loc[date_val, "turnover_rate"] if "turnover_rate" in metric_df.columns else 0.0
        pos_count = int(metric_df.loc[date_val, "pos_count"]) if "pos_count" in metric_df.columns else 0
        
        ledger_data.append({
            "date": date_obj,
            "nav": float(nav_val),
            "cash": float(cash),
            "market_value": float(market_value),
            "deal_amount": float(deal_amount),
            "turnover_rate": float(turnover_rate),
            "pos_count": int(pos_count),
        })
    
    return ledger_data

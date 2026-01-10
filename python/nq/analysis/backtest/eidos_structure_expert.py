"""
Eidos Integration for Structure Expert Backtest.

This module provides functions to extract and save Structure Expert backtest data to Eidos.
All functions have explicit type annotations. Invalid inputs will cause program to crash.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from nq.analysis.backtest.date_utils import to_date
from nq.analysis.backtest.eidos_integration import EidosBacktestWriter
from nq.analysis.backtest.qlib_types import QlibBacktestResult
from tools.qlib.train.structure_expert import GraphDataBuilder, StructureExpertGNN

logger = logging.getLogger(__name__)


def extract_ledger_from_backtest_results(
    qlib_result: QlibBacktestResult,
    initial_cash: float,
) -> List[Dict[str, Any]]:
    """
    Extract ledger data from Qlib backtest results.

    Args:
        qlib_result: Standard QlibBacktestResult structure.
        initial_cash: Initial cash amount.

    Returns:
        List of ledger records.
    """
    ledger_data = []

    portfolio_metrics = qlib_result.portfolio_metrics
    metric_df = portfolio_metrics.metric_df
    position_details = portfolio_metrics.position_details
    
    # Calculate cumulative NAV from returns
    returns = metric_df["return"]
    nav = initial_cash * (1 + returns).cumprod()

    for date_val, nav_val in nav.items():
        date_obj = to_date(date_val)

        # Extract metrics from DataFrame if available, otherwise from position_details
        cash = metric_df.loc[date_val, "cash"] if "cash" in metric_df.columns else None
        market_value = metric_df.loc[date_val, "market_value"] if "market_value" in metric_df.columns else None
        deal_amount = metric_df.loc[date_val, "deal_amount"] if "deal_amount" in metric_df.columns else 0.0
        turnover_rate = metric_df.loc[date_val, "turnover_rate"] if "turnover_rate" in metric_df.columns else 0.0
        pos_count = metric_df.loc[date_val, "pos_count"] if "pos_count" in metric_df.columns else 0
        
        # If missing, get from position_details
        if position_details is not None:
            pos = position_details.get_position(date_obj)
            if pos is not None:
                if cash is None:
                    cash = pos.cash
                if market_value is None:
                    market_value = pos.market_value
                if pos_count == 0:
                    pos_count = len(pos.get_holdings_dict())

        ledger_data.append({
            "date": date_obj,
            "nav": float(nav_val),
            "cash": float(cash) if cash is not None else 0.0,
            "market_value": float(market_value) if market_value is not None else 0.0,
            "deal_amount": float(deal_amount),
            "turnover_rate": float(turnover_rate),
            "pos_count": int(pos_count),
        })

    return ledger_data


def extract_trades_from_backtest_results(
        strategy_instance: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """
    Extract trade data from strategy instance.
    
    Only extracts trades from strategy instance with get_executed_orders() method.
    Does not calculate from position changes or use executor fallback.

    Args:
        strategy_instance: Strategy instance with get_executed_orders() method (required).

    Returns:
        List of trade records. Empty list if no strategy instance or no orders available.
    """
    if strategy_instance is None:
        logger.warning("No strategy instance provided, cannot extract trades")
        return []
    
    if not hasattr(strategy_instance, "get_executed_orders"):
        logger.warning("Strategy instance does not have get_executed_orders() method, cannot extract trades")
        return []
    
    try:
        executed_orders = strategy_instance.get_executed_orders()
        if not executed_orders or len(executed_orders) == 0:
            logger.info("No executed orders found in strategy instance")
            return []
        
        logger.info(f"Extracted {len(executed_orders)} orders from strategy instance")
        # Convert strategy order format to Eidos trade format
        trades_data = []
        for order_info in executed_orders:
            # Skip orders with invalid direction (0 means cancelled/invalid order)
            # Database constraint requires side to be 1 (Buy) or -1 (Sell)
            direction = int(order_info.get("direction", 0))
            if direction == 0:
                logger.warning(
                    f"Skipping order with direction=0: {order_info.get('instrument', 'unknown')}, "
                    f"amount={order_info.get('amount', 0)}, price={order_info.get('trade_price', 0.0)}"
                )
                continue
            
            # Skip orders with invalid amount (must be > 0 per database constraint)
            amount = int(order_info.get("amount", 0))
            if amount <= 0:
                logger.warning(
                    f"Skipping order with invalid amount (must be > 0): "
                    f"symbol={order_info.get('instrument', 'unknown')}, "
                    f"amount={amount}, direction={direction}, price={order_info.get('trade_price', 0.0)}"
                )
                continue
            
            # Extract deal_time from order if available
            deal_time = order_info.get("deal_time")
            if deal_time is not None:
                # Convert to datetime if it's a date or string
                if isinstance(deal_time, str):
                    deal_time = pd.to_datetime(deal_time)
                elif hasattr(deal_time, "date"):  # datetime object
                    deal_time = pd.to_datetime(deal_time)
                else:
                    deal_time = pd.to_datetime(deal_time)
            
            trades_data.append({
                "symbol": str(order_info.get("instrument", "")),
                "deal_time": deal_time,
                "direction": direction,  # Must be 1 (Buy) or -1 (Sell), not 0
                "price": float(order_info.get("trade_price", 0.0)),
                "amount": amount,  # Already validated to be > 0 above
                "rank_at_deal": order_info.get("rank_at_deal"),
                "score_at_deal": order_info.get("score_at_deal"),
                "reason": order_info.get("reason"),
                "pnl_ratio": order_info.get("pnl_ratio"),
                "hold_days": order_info.get("hold_days"),
            })
        return trades_data
    except Exception as e:
        logger.error(f"Failed to get orders from strategy instance: {e}")
        return []


def extract_model_outputs_from_predictions(
    predictions: pd.DataFrame,
) -> List[Dict[str, Any]]:
    """
    Extract model outputs from predictions DataFrame.

    Args:
        predictions: Predictions DataFrame with MultiIndex (datetime, instrument) and 'score' column.

    Returns:
        List of model output records.
    """
    outputs_data = []

    # Calculate ranks for each date
    for date_val in predictions.index.get_level_values(0).unique():
        date_predictions = predictions.loc[date_val]

        # Convert to DataFrame if Series, otherwise use directly
        if isinstance(date_predictions, pd.Series):
            date_predictions_df = date_predictions.to_frame("score")
        else:
            # Already a DataFrame - ensure it has 'score' column
            date_predictions_df = date_predictions.copy()
            if "score" not in date_predictions_df.columns:
                # If no 'score' column, use first column
                date_predictions_df = date_predictions_df.iloc[:, [0]].rename(columns={date_predictions_df.columns[0]: "score"})

        # Sort by score descending and assign ranks
        date_predictions_sorted = date_predictions_df.sort_values("score", ascending=False)
        date_predictions_sorted["rank"] = range(1, len(date_predictions_sorted) + 1)

        date_obj = to_date(date_val)

        # Extract records
        for symbol, row in date_predictions_sorted.iterrows():
            outputs_data.append({
                "date": date_obj,
                "symbol": str(symbol),
                "score": float(row["score"]),
                "rank": int(row["rank"]),
                "extra_scores": None,
            })

    return outputs_data


def extract_model_links_from_graph(
    builder: GraphDataBuilder,
    predictions: pd.DataFrame,
    model: StructureExpertGNN,
    device: torch.device,
) -> List[Dict[str, Any]]:
    """
    Extract model links (graph edges) from Structure Expert model.

    Args:
        builder: GraphDataBuilder instance.
        predictions: Predictions DataFrame.
        model: Structure Expert model.
        device: PyTorch device.

    Returns:
        List of model link records.
    """
    # TODO: Implement actual link extraction from model attention weights
    return []


def extract_embeddings_from_predictions(
    predictions: pd.DataFrame,
    embeddings_dict: Optional[Dict[Tuple[str, str], np.ndarray]],
) -> List[Dict[str, Any]]:
    """
    Extract embeddings from predictions and embeddings dictionary.

    Args:
        predictions: Predictions DataFrame.
        embeddings_dict: Dictionary mapping (date, symbol) to embedding vector (optional, can be None).

    Returns:
        List of embedding records.
    """
    embeddings_data = []

    if embeddings_dict is None:
        return embeddings_data

    for (date_val, symbol), embedding in embeddings_dict.items():
        date_obj = to_date(date_val)

        vec = embedding.tolist()
        
        embeddings_data.append({
            "date": date_obj,
            "symbol": str(symbol),
            "vec": vec,
            "vec_dim": len(vec),
        })

    return embeddings_data


def save_structure_expert_backtest_to_eidos(
    exp_id: str,
    writer: EidosBacktestWriter,
    qlib_result: QlibBacktestResult,
    predictions: pd.DataFrame,
    initial_cash: float,
    builder: GraphDataBuilder,
    model: StructureExpertGNN,
    device: torch.device,
    embeddings_dict: Optional[Dict[Tuple[str, str], np.ndarray]],
    strategy_instance: Optional[Any] = None,
) -> Dict[str, int]:
    """
    Save Structure Expert backtest results to Eidos.

    Args:
        exp_id: Experiment ID.
        writer: Eidos writer instance.
        qlib_result: Standard QlibBacktestResult structure.
        predictions: Predictions DataFrame.
        initial_cash: Initial cash amount.
        builder: GraphDataBuilder instance (optional, for extracting links).
        model: Structure Expert model (optional, for extracting links).
        device: PyTorch device (optional, for extracting links).
        embeddings_dict: Dictionary of embeddings (optional).
        strategy_instance: Strategy instance with get_executed_orders() method (required for trade extraction).

    Returns:
        Dictionary with counts of inserted records.
    """
    logger.info(f"Saving Structure Expert backtest results to Eidos (exp_id: {exp_id})")

    # Extract ledger data
    ledger_data = extract_ledger_from_backtest_results(qlib_result, initial_cash)
    logger.info(f"Extracted {len(ledger_data)} ledger records")

    # Extract trades data from strategy instance only
    trades_data = extract_trades_from_backtest_results(
        strategy_instance=strategy_instance,
    )
    logger.info(f"Extracted {len(trades_data)} trade records")

    # Extract model outputs
    model_outputs_data = extract_model_outputs_from_predictions(predictions)
    logger.info(f"Extracted {len(model_outputs_data)} model output records")

    # Extract model links
    links_data = extract_model_links_from_graph(builder, predictions, model, device)
    logger.info(f"Extracted {len(links_data)} model link records")

    # Extract embeddings
    embeddings_data = extract_embeddings_from_predictions(predictions, embeddings_dict)
    logger.info(f"Extracted {len(embeddings_data)} embedding records")

    # Save all data using repo
    counts = writer.repo.save_backtest_results(
        exp_id=exp_id,
        ledger_data=ledger_data,
        trades_data=trades_data,
        model_outputs_data=model_outputs_data,
        model_links_data=links_data,
        embeddings_data=embeddings_data,
    )

    logger.info(f"Saved to Eidos: {counts}")
    return counts

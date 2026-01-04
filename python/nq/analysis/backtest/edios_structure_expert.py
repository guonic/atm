"""
EDiOS Integration for Structure Expert Backtest.

This module provides functions to extract and save Structure Expert backtest data to EDiOS.
All functions have explicit type annotations. Invalid inputs will cause program to crash.
"""

import logging
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from nq.analysis.backtest.date_utils import to_date
from nq.analysis.backtest.edios_integration import EdiosBacktestWriter
from nq.analysis.backtest.qlib_types import QlibBacktestResult
from nq.config import DatabaseConfig
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
    qlib_result: QlibBacktestResult,
) -> List[Dict[str, Any]]:
    """
    Extract trade data from Qlib backtest results by analyzing position changes.

    Args:
        qlib_result: Standard QlibBacktestResult structure.

    Returns:
        List of trade records.
    """
    trades_data = []

    portfolio_metrics = qlib_result.portfolio_metrics
    position_details = portfolio_metrics.position_details
    
    dates = position_details.get_dates()
    
    # Track holdings: {symbol: (buy_date, buy_price, amount)}
    holdings: Dict[str, Tuple[date, float, float]] = {}
    trade_id = 0
    
    for i, current_date in enumerate(dates):
        current_pos = position_details.get_position(current_date)
        current_holdings = current_pos.get_holdings_dict()
        
        if i == 0:
            # First day - all are new buys
            for symbol, (price, amount) in current_holdings.items():
                holdings[symbol] = (current_date, price, amount)
        else:
            prev_date = dates[i - 1]
            prev_pos = position_details.get_position(prev_date)
            prev_holdings = prev_pos.get_holdings_dict()
            
            # Find new buys (in current but not in prev)
            for symbol, (price, amount) in current_holdings.items():
                if symbol not in prev_holdings:
                    # New buy
                    holdings[symbol] = (current_date, price, amount)
                    trade_id += 1
                    trades_data.append({
                        "symbol": symbol,
                        "deal_time": datetime.combine(current_date, datetime.min.time()),
                        "side": 1,  # Buy
                        "price": price,
                        "amount": int(amount),
                        "rank_at_deal": None,
                        "score_at_deal": None,
                        "reason": "rank_in",
                        "pnl_ratio": None,
                        "hold_days": 0,
                    })
            
            # Find sells (in prev but not in current, or reduced amount)
            for symbol, (prev_price, prev_amount) in prev_holdings.items():
                if symbol not in current_holdings:
                    # Complete sell
                    buy_date, buy_price, buy_amount = holdings[symbol]
                    pnl_ratio = (prev_price - buy_price) / buy_price
                    hold_days = (current_date - buy_date).days
                    
                    trade_id += 1
                    trades_data.append({
                        "symbol": symbol,
                        "deal_time": datetime.combine(current_date, datetime.min.time()),
                        "side": -1,  # Sell
                        "price": prev_price,
                        "amount": int(prev_amount),
                        "rank_at_deal": None,
                        "score_at_deal": None,
                        "reason": "rank_out",
                        "pnl_ratio": float(pnl_ratio),
                        "hold_days": int(hold_days),
                    })
                    del holdings[symbol]
                elif current_holdings[symbol][1] < prev_amount:
                    # Partial sell
                    buy_date, buy_price, buy_amount = holdings[symbol]
                    sell_price = current_holdings[symbol][0]
                    sold_amount = prev_amount - current_holdings[symbol][1]
                    pnl_ratio = (sell_price - buy_price) / buy_price
                    hold_days = (current_date - buy_date).days
                    
                    trade_id += 1
                    trades_data.append({
                        "symbol": symbol,
                        "deal_time": datetime.combine(current_date, datetime.min.time()),
                        "side": -1,  # Sell
                        "price": sell_price,
                        "amount": int(sold_amount),
                        "rank_at_deal": None,
                        "score_at_deal": None,
                        "reason": "partial_sell",
                        "pnl_ratio": float(pnl_ratio),
                        "hold_days": int(hold_days),
                    })
                    # Update holdings
                    remaining_amount = current_holdings[symbol][1]
                    if remaining_amount > 0:
                        holdings[symbol] = (buy_date, buy_price, remaining_amount)
                    else:
                        del holdings[symbol]
    
    logger.info(f"Extracted {len(trades_data)} trades from position changes")
    
    return trades_data


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


def save_structure_expert_backtest_to_edios(
    exp_id: str,
    writer: EdiosBacktestWriter,
    qlib_result: QlibBacktestResult,
    predictions: pd.DataFrame,
    initial_cash: float,
    builder: GraphDataBuilder,
    model: StructureExpertGNN,
    device: torch.device,
    embeddings_dict: Dict[Tuple[str, str], np.ndarray],
) -> Dict[str, int]:
    """
    Save Structure Expert backtest results to EDiOS.

    Args:
        exp_id: Experiment ID.
        writer: EDiOS writer instance.
        qlib_result: Standard QlibBacktestResult structure.
        predictions: Predictions DataFrame.
        initial_cash: Initial cash amount.
        builder: GraphDataBuilder instance (optional, for extracting links).
        model: Structure Expert model (optional, for extracting links).
        device: PyTorch device (optional, for extracting links).
        embeddings_dict: Dictionary of embeddings (optional).

    Returns:
        Dictionary with counts of inserted records.
    """
    logger.info(f"Saving Structure Expert backtest results to EDiOS (exp_id: {exp_id})")

    # Extract ledger data
    ledger_data = extract_ledger_from_backtest_results(qlib_result, initial_cash)
    logger.info(f"Extracted {len(ledger_data)} ledger records")

    # Extract trades data
    trades_data = extract_trades_from_backtest_results(qlib_result)
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

    logger.info(f"Saved to EDiOS: {counts}")
    return counts

"""
Eidos Integration for Backtest System.

Provides integration layer to automatically save backtest results to Eidos system.
"""

import logging
from datetime import date, datetime
from typing import Any, Dict, List, Optional

import pandas as pd

from nq.analysis.backtest.base import BacktestResult, BaseBacktester
from nq.config import DatabaseConfig
from nq.repo.eidos_repo import EidosRepo

logger = logging.getLogger(__name__)


class EidosBacktestWriter:
    """
    Writer for saving backtest results to Eidos system.
    
    This class provides methods to convert backtest results into Eidos format
    and save them to the database.
    """

    def __init__(self, db_config: DatabaseConfig, schema: str = "eidos"):
        """
        Initialize Eidos backtest writer.

        Args:
            db_config: Database configuration.
            schema: Database schema name (default: 'eidos').
        """
        self.repo = EidosRepo(db_config, schema)
        self.schema = schema

    def create_experiment_from_backtest(
        self,
        name: str,
        start_date: date,
        end_date: date,
        config: Dict[str, Any],
        model_type: Optional[str] = None,
        engine_type: str = "Backtrader",
        strategy_name: Optional[str] = None,
    ) -> str:
        """
        Create an experiment from backtest configuration.

        Args:
            name: Experiment name.
            model_type: Model type (e.g., 'GNN', 'GRU', 'Linear').
            engine_type: Engine type (default: 'Backtrader').
            start_date: Start date.
            end_date: End date.
            config: Configuration parameters.
            strategy_name: Strategy name (optional).

        Returns:
            Experiment ID.
        """
        if strategy_name:
            name = f"{strategy_name} - {name}"

        exp_id = self.repo.experiment.create_experiment(
            name=name,
            model_type=model_type,
            engine_type=engine_type,
            start_date=start_date,
            end_date=end_date,
            config=config,
            status="running",
        )
        logger.info(f"Created experiment: {exp_id}")
        return exp_id

    def save_backtest_result(
        self,
        exp_id: str,
        result: BacktestResult,
        broker_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, int]:
        """
        Save a single backtest result to Eidos.

        Args:
            exp_id: Experiment ID.
            result: Backtest result.
            broker_data: Broker data containing daily NAV, positions, etc. (optional).

        Returns:
            Dictionary with counts of inserted records.
        """
        counts = {}

        # Extract ledger data from broker_data or result metadata
        if broker_data and "ledger" in broker_data:
            ledger_data = broker_data["ledger"]
            counts["ledger"] = self.repo.ledger.batch_upsert(exp_id, ledger_data)

        # Extract trades from result metadata
        trades_data = self._extract_trades_from_result(exp_id, result)
        if trades_data:
            counts["trades"] = self.repo.trades.batch_insert(exp_id, trades_data)

        # Extract model outputs from result predictions
        model_outputs_data = self._extract_model_outputs_from_result(exp_id, result)
        if model_outputs_data:
            counts["model_outputs"] = self.repo.model_outputs.batch_upsert(
                exp_id, model_outputs_data
            )

        return counts

    def _extract_trades_from_result(
        self, exp_id: str, result: BacktestResult
    ) -> List[Dict[str, Any]]:
        """
        Extract trades from backtest result.

        Args:
            exp_id: Experiment ID.
            result: Backtest result.

        Returns:
            List of trade records.
        """
        trades_data = []

        # Check if trades are in metadata
        if "trades" in result.metadata:
            for trade in result.metadata["trades"]:
                trades_data.append({
                    "symbol": result.ts_code,
                    "deal_time": trade.get("deal_time", result.backtest_date),
                    "direction": trade.get("direction", trade.get("side", 1)),  # 1=Buy, -1=Sell
                    "price": trade.get("price", 0.0),
                    "amount": trade.get("amount", 0),
                    "rank_at_deal": trade.get("rank_at_deal"),
                    "score_at_deal": trade.get("score_at_deal"),
                    "reason": trade.get("reason"),
                    "pnl_ratio": trade.get("pnl_ratio"),
                    "hold_days": trade.get("hold_days"),
                })

        # Check if signals are in metadata (convert signals to trades)
        elif "signals" in result.metadata:
            for signal in result.metadata["signals"]:
                direction = 1 if signal.get("signal_type") in ["buy", "add"] else -1
                trades_data.append({
                    "symbol": result.ts_code,
                    "deal_time": signal.get("signal_time", result.backtest_date),
                    "direction": direction,
                    "price": signal.get("price", 0.0),
                    "amount": signal.get("size", 0),
                    "reason": signal.get("signal_type"),
                })

        return trades_data

    def _extract_model_outputs_from_result(
        self, exp_id: str, result: BacktestResult
    ) -> List[Dict[str, Any]]:
        """
        Extract model outputs from backtest result.

        Args:
            exp_id: Experiment ID.
            result: Backtest result.

        Returns:
            List of model output records.
        """
        outputs_data = []

        # If predictions DataFrame has date and score/rank columns
        if not result.predictions.empty:
            df = result.predictions.copy()

            # Try to identify date column
            date_col = None
            for col in ["date", "trade_date", "time", "datetime"]:
                if col in df.columns:
                    date_col = col
                    break

            # Try to identify score column
            score_col = None
            for col in ["score", "prediction", "pred", "value"]:
                if col in df.columns:
                    score_col = col
                    break

            # Try to identify rank column
            rank_col = None
            for col in ["rank", "ranking", "position"]:
                if col in df.columns:
                    rank_col = col
                    break

            if date_col and score_col:
                for _, row in df.iterrows():
                    date_val = row[date_col]
                    if isinstance(date_val, datetime):
                        date_val = date_val.date()
                    elif isinstance(date_val, pd.Timestamp):
                        date_val = date_val.date()
                    elif isinstance(date_val, str):
                        date_val = pd.to_datetime(date_val).date()

                    score = float(row[score_col]) if pd.notna(row[score_col]) else 0.0
                    rank = (
                        int(row[rank_col])
                        if rank_col and pd.notna(row.get(rank_col))
                        else 0
                    )

                    extra_scores = {}
                    # Collect other numeric columns as extra_scores
                    for col in df.columns:
                        if col not in [date_col, score_col, rank_col] and pd.notna(
                            row[col]
                        ):
                            try:
                                extra_scores[col] = float(row[col])
                            except (ValueError, TypeError):
                                pass

                    outputs_data.append({
                        "date": date_val,
                        "symbol": result.ts_code,
                        "score": score,
                        "rank": rank if rank > 0 else 1,
                        "extra_scores": extra_scores if extra_scores else None,
                    })

        return outputs_data

    def finalize_experiment(
        self,
        exp_id: str,
        metrics_summary: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Finalize experiment by updating status and metrics.

        Args:
            exp_id: Experiment ID.
            metrics_summary: Summary metrics (optional).
        """
        self.repo.experiment.update_experiment(
            exp_id, status="completed", metrics_summary=metrics_summary
        )
        logger.info(f"Finalized experiment: {exp_id}")


class EidosBacktesterMixin:
    """
    Mixin class to add Eidos integration to existing backtesters.
    
    Usage:
        class MyBacktester(BaseBacktester, EidosBacktesterMixin):
            ...
    """

    def __init__(self, *args, **kwargs):
        """Initialize with Eidos support."""
        super().__init__(*args, **kwargs)
        self.eidos_writer: Optional[EidosBacktestWriter] = None
        self.eidos_exp_id: Optional[str] = None

    def enable_eidos(
        self,
        db_config: DatabaseConfig,
        exp_id: Optional[str] = None,
        experiment_name: Optional[str] = None,
        model_type: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Enable Eidos integration for this backtester.

        Args:
            db_config: Database configuration.
            exp_id: Existing experiment ID (optional).
            experiment_name: Experiment name (if creating new).
            model_type: Model type.
            config: Experiment configuration.

        Returns:
            Experiment ID.
        """
        self.eidos_writer = EidosBacktestWriter(db_config)

        if exp_id:
            self.eidos_exp_id = exp_id
        elif experiment_name:
            # Create new experiment
            from datetime import date

            self.eidos_exp_id = self.eidos_writer.create_experiment_from_backtest(
                name=experiment_name,
                start_date=date.today(),  # Will be updated from actual backtest
                end_date=date.today(),
                config=config or {},
                model_type=model_type,
            )
        else:
            raise ValueError("Either exp_id or experiment_name must be provided")

        return self.eidos_exp_id

    def save_to_eidos(
        self,
        result: BacktestResult,
        broker_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, int]:
        """
        Save backtest result to Eidos.

        Args:
            result: Backtest result.
            broker_data: Broker data (optional).

        Returns:
            Dictionary with counts of inserted records.
        """
        if not self.eidos_writer or not self.eidos_exp_id:
            raise ValueError("Eidos not enabled. Call enable_eidos() first.")

        return self.eidos_writer.save_backtest_result(
            self.eidos_exp_id, result, broker_data
        )


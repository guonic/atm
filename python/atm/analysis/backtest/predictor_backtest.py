"""
Predictor backtest implementation.

Provides backtest functionality for prediction models.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Optional

import pandas as pd

from atm.repo.kline_repo import StockKlineDayRepo
from atm.trading.predictor.base import BasePredictor, PredictionResult

from .base import BaseBacktester, BacktestResult
from .metrics import BacktestMetrics

logger = logging.getLogger(__name__)


class PredictorBacktester(BaseBacktester):
    """
    Backtester for prediction models.

    Evaluates prediction models by comparing predictions with actual data.
    """

    def __init__(
        self,
        predictor: BasePredictor,
        db_config,
        schema: str = "quant",
    ):
        """
        Initialize predictor backtester.

        Args:
            predictor: Predictor instance to backtest.
            db_config: Database configuration.
            schema: Database schema name.
        """
        super().__init__(db_config, schema)
        self.predictor = predictor

    def _load_actual_data(
        self,
        ts_code: str,
        start_date: datetime,
        end_date: datetime,
    ) -> pd.DataFrame:
        """
        Load actual data for comparison.

        Args:
            ts_code: Stock code.
            start_date: Start date (inclusive).
            end_date: End date (inclusive).

        Returns:
            DataFrame with actual OHLCV data.
        """
        repo = StockKlineDayRepo(self.db_config, self.schema)

        klines = repo.get_by_ts_code(
            ts_code=ts_code,
            start_time=start_date,
            end_time=end_date,
        )

        if not klines:
            raise ValueError(f"No actual data found for {ts_code} in date range")

        # Convert to DataFrame
        data_list = []
        for kline in klines:
            data_list.append({
                "date": kline.trade_date,
                "open": float(kline.open) if kline.open else None,
                "high": float(kline.high) if kline.high else None,
                "low": float(kline.low) if kline.low else None,
                "close": float(kline.close) if kline.close else None,
                "volume": int(kline.volume) if kline.volume else 0,
                "amount": float(kline.amount) if kline.amount else None,
            })

        df = pd.DataFrame(data_list)
        df = df.sort_values("date").reset_index(drop=True)

        # Ensure date is datetime
        if not isinstance(df["date"].dtype, pd.DatetimeIndex):
            df["date"] = pd.to_datetime(df["date"])

        return df

    def run(
        self,
        ts_code: str,
        start_date: datetime,
        end_date: datetime,
        pred_len: int,
        lookback: Optional[int] = None,
        step_size: Optional[int] = None,
        **kwargs,
    ) -> BacktestResult:
        """
        Run backtest for predictor.

        Args:
            ts_code: Stock code in Tushare format (e.g., 000001.SZ).
            start_date: Start date of backtest period (inclusive).
            end_date: End date of backtest period (inclusive).
            pred_len: Number of days to predict in each step.
            lookback: Number of historical days to use for each prediction.
                     If None, uses predictor default.
            step_size: Step size for walk-forward backtest. If None, uses pred_len.
                      If step_size < pred_len, predictions will overlap.
            **kwargs: Additional parameters passed to predictor.predict().

        Returns:
            BacktestResult containing predictions, actuals, and metrics.
        """
        self.validate_inputs(ts_code, start_date, end_date)

        if step_size is None:
            step_size = pred_len

        logger.info(
            f"Running backtest for {ts_code} from {start_date.date()} to {end_date.date()}, "
            f"pred_len={pred_len}, step_size={step_size}"
        )

        all_predictions = []
        all_actuals = []

        # Walk-forward backtest
        # Start from start_date + lookback to ensure we have enough historical data
        if lookback is None:
            lookback = 400  # Use default lookback if not specified
        
        # Calculate the first prediction date (need lookback days of history first)
        first_pred_date = start_date + timedelta(days=lookback)
        if first_pred_date >= end_date:
            raise ValueError(
                f"Backtest period too short. Need at least {lookback} days of history. "
                f"start_date={start_date.date()}, end_date={end_date.date()}"
            )
        
        current_date = first_pred_date
        step = 0

        while current_date < end_date:
            step += 1
            # Calculate prediction end date
            pred_end_date = min(current_date + timedelta(days=pred_len), end_date)

            logger.info(
                f"Step {step}: Predicting from {current_date.date()} "
                f"(cutoff: {current_date.date()}, predicting to {pred_end_date.date()})"
            )

            try:
                # Make prediction using data up to current_date
                # Use start_date to ensure consistent historical data range
                prediction_result = self.predictor.predict(
                    ts_code=ts_code,
                    pred_len=pred_len,
                    lookback=lookback,
                    start_date=start_date,  # Always start from the same start_date
                    end_date=current_date,  # But cutoff at current_date (no future data)
                    **kwargs,
                )

                # Get predicted data for the period we want to compare
                pred_df = prediction_result.predicted_data.copy()
                pred_df = pred_df[pred_df["date"] <= pred_end_date]

                # Load actual data for comparison
                actual_df = self._load_actual_data(
                    ts_code=ts_code,
                    start_date=current_date + timedelta(days=1),
                    end_date=pred_end_date,
                )

                # Align dates for comparison
                if not pred_df.empty and not actual_df.empty:
                    # Merge on date
                    merged = pd.merge(
                        pred_df[["date", "close"]],
                        actual_df[["date", "close"]],
                        on="date",
                        suffixes=("_pred", "_actual"),
                    )

                    if not merged.empty:
                        all_predictions.append(merged[["date", "close_pred"]].rename(columns={"close_pred": "close"}))
                        all_actuals.append(merged[["date", "close_actual"]].rename(columns={"close_actual": "close"}))

            except Exception as e:
                logger.warning(f"Step {step} failed: {e}")
                # Continue with next step

            # Move to next step
            current_date = current_date + timedelta(days=step_size)

        if not all_predictions:
            raise ValueError("No valid predictions generated during backtest")

        # Combine all predictions and actuals
        predictions_df = pd.concat(all_predictions, ignore_index=True).sort_values("date")
        actuals_df = pd.concat(all_actuals, ignore_index=True).sort_values("date")

        # Remove duplicates (in case of overlapping predictions)
        predictions_df = predictions_df.drop_duplicates(subset=["date"], keep="last")
        actuals_df = actuals_df.drop_duplicates(subset=["date"], keep="last")

        # Merge to ensure alignment
        merged = pd.merge(
            predictions_df,
            actuals_df,
            on="date",
            suffixes=("_pred", "_actual"),
        )

        if merged.empty:
            raise ValueError("No aligned predictions and actuals found")

        # Calculate metrics
        metrics = BacktestMetrics.calculate_all_metrics(
            predicted=merged[["date", "close_pred"]].rename(columns={"close_pred": "close"}),
            actual=merged[["date", "close_actual"]].rename(columns={"close_actual": "close"}),
        )

        # Create result
        result = BacktestResult(
            ts_code=ts_code,
            backtest_date=datetime.now(),
            start_date=start_date,
            end_date=end_date,
            predictions=merged[["date", "close_pred"]].rename(columns={"close_pred": "close"}),
            actuals=merged[["date", "close_actual"]].rename(columns={"close_actual": "close"}),
            metrics=metrics,
            metadata={
                "pred_len": pred_len,
                "step_size": step_size,
                "lookback": lookback,
                "num_steps": step,
                "num_predictions": len(merged),
            },
        )

        logger.info(f"Backtest completed: {len(merged)} predictions evaluated")
        logger.info(f"Metrics: MAE={metrics.get('mae', 0):.2f}, "
                   f"RMSE={metrics.get('rmse', 0):.2f}, "
                   f"Direction Accuracy={metrics.get('direction_accuracy', 0):.2f}%")

        return result

    def get_info(self) -> dict:
        """
        Get backtester information.

        Returns:
            Dictionary containing backtester information.
        """
        info = super().get_info()
        info.update({
            "predictor": self.predictor.get_info() if hasattr(self.predictor, 'get_info') else str(type(self.predictor)),
        })
        return info


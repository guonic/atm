"""
Evaluator for Teapot pattern recognition signals.

Computes forward returns and holding returns.
"""

import logging
from typing import List, Optional

import polars as pl

logger = logging.getLogger(__name__)


class TeapotEvaluator:
    """
    Evaluator for Teapot signals.

    Computes forward returns, maximum drawdown, and holding returns.
    """

    def __init__(
        self,
        forward_horizons: List[int] = [5, 20],
        min_holding_days: int = 1,
        max_holding_days: int = 60,
    ):
        """
        Initialize evaluator.

        Args:
            forward_horizons: List of forward horizons (e.g., [5, 20] for T+5, T+20).
            min_holding_days: Minimum holding days.
            max_holding_days: Maximum holding days.
        """
        self.forward_horizons = forward_horizons
        self.min_holding_days = min_holding_days
        self.max_holding_days = max_holding_days

    def compute_forward_returns(
        self,
        signals: pl.DataFrame,
        market_data: pl.DataFrame,
    ) -> pl.DataFrame:
        """
        Compute forward returns.

        Args:
            signals: Signals DataFrame with columns: ts_code, signal_date.
            market_data: Market data DataFrame with columns: ts_code, trade_date, close.

        Returns:
            DataFrame with added columns:
            - return_t5: T+5 return
            - return_t20: T+20 return
            - max_drawdown_t5: T+5 max drawdown
            - max_drawdown_t20: T+20 max drawdown
            - peak_date: Peak date
            - peak_return: Peak return
        """
        results = []

        for signal in signals.iter_rows(named=True):
            ts_code = signal["ts_code"]
            signal_date = signal["signal_date"]

            # Get stock data
            stock_data = market_data.filter(pl.col("ts_code") == ts_code).sort(
                "trade_date"
            )

            if stock_data.is_empty():
                continue

            # Find signal date index
            signal_idx = None
            for idx, row in enumerate(stock_data.iter_rows(named=True)):
                if str(row["trade_date"]) == signal_date:
                    signal_idx = idx
                    break

            if signal_idx is None:
                continue

            # Get signal price
            signal_price = stock_data[signal_idx]["close"]

            # Compute forward returns for each horizon
            eval_result = {"ts_code": ts_code, "signal_date": signal_date}

            for horizon in self.forward_horizons:
                forward_idx = signal_idx + horizon
                if forward_idx < len(stock_data):
                    forward_price = stock_data[forward_idx]["close"]
                    forward_return = (forward_price - signal_price) / signal_price

                    # Compute max drawdown in this period
                    period_data = stock_data[signal_idx : forward_idx + 1]
                    prices = period_data["close"].to_list()
                    if prices:
                        cummax = []
                        max_val = prices[0]
                        for p in prices:
                            max_val = max(max_val, p)
                            cummax.append(max_val)

                        drawdowns = [
                            (p - cm) / cm for p, cm in zip(prices, cummax)
                        ]
                        max_drawdown = min(drawdowns) if drawdowns else 0

                        eval_result[f"return_t{horizon}"] = forward_return
                        eval_result[f"max_drawdown_t{horizon}"] = max_drawdown

            # Find peak return
            if signal_idx + self.max_holding_days < len(stock_data):
                period_data = stock_data[
                    signal_idx : signal_idx + self.max_holding_days + 1
                ]
                prices = period_data["close"].to_list()
                dates = period_data["trade_date"].to_list()

                if prices:
                    returns = [(p - signal_price) / signal_price for p in prices]
                    peak_idx = returns.index(max(returns))
                    eval_result["peak_date"] = str(dates[peak_idx])
                    eval_result["peak_return"] = returns[peak_idx]

            results.append(eval_result)

        if not results:
            return pl.DataFrame()

        return pl.DataFrame(results)

    def compute_holding_returns(
        self,
        signals: pl.DataFrame,
        market_data: pl.DataFrame,
        exit_strategy: str = "fixed_days",
        holding_days: int = 20,
        target_return: float = 0.15,
        stop_loss: float = -0.10,
    ) -> pl.DataFrame:
        """
        Compute holding returns with exit strategy.

        Args:
            signals: Signals DataFrame.
            market_data: Market data DataFrame.
            exit_strategy: Exit strategy ("fixed_days", "target_return", "stop_loss", "combined").
            holding_days: Fixed holding days.
            target_return: Target return for exit.
            stop_loss: Stop loss threshold.

        Returns:
            DataFrame with holding return information.
        """
        results = []

        for signal in signals.iter_rows(named=True):
            ts_code = signal["ts_code"]
            signal_date = signal["signal_date"]

            # Get stock data
            stock_data = market_data.filter(pl.col("ts_code") == ts_code).sort(
                "trade_date"
            )

            if stock_data.is_empty():
                continue

            # Find signal date index
            signal_idx = None
            for idx, row in enumerate(stock_data.iter_rows(named=True)):
                if str(row["trade_date"]) == signal_date:
                    signal_idx = idx
                    break

            if signal_idx is None:
                continue

            signal_price = stock_data[signal_idx]["close"]
            exit_idx = None
            exit_reason = None

            # Apply exit strategy
            if exit_strategy == "fixed_days":
                exit_idx = signal_idx + holding_days
                exit_reason = "fixed_days"
            elif exit_strategy == "target_return":
                for idx in range(signal_idx + 1, min(signal_idx + self.max_holding_days + 1, len(stock_data))):
                    price = stock_data[idx]["close"]
                    ret = (price - signal_price) / signal_price
                    if ret >= target_return:
                        exit_idx = idx
                        exit_reason = "target_return"
                        break
            elif exit_strategy == "stop_loss":
                for idx in range(signal_idx + 1, min(signal_idx + self.max_holding_days + 1, len(stock_data))):
                    price = stock_data[idx]["close"]
                    ret = (price - signal_price) / signal_price
                    if ret <= stop_loss:
                        exit_idx = idx
                        exit_reason = "stop_loss"
                        break
            elif exit_strategy == "combined":
                for idx in range(signal_idx + 1, min(signal_idx + self.max_holding_days + 1, len(stock_data))):
                    price = stock_data[idx]["close"]
                    ret = (price - signal_price) / signal_price
                    if ret >= target_return:
                        exit_idx = idx
                        exit_reason = "target_return"
                        break
                    elif ret <= stop_loss:
                        exit_idx = idx
                        exit_reason = "stop_loss"
                        break
                if exit_idx is None:
                    exit_idx = signal_idx + holding_days
                    exit_reason = "fixed_days"

            if exit_idx is None or exit_idx >= len(stock_data):
                continue

            exit_price = stock_data[exit_idx]["close"]
            exit_date = stock_data[exit_idx]["trade_date"]
            holding_return = (exit_price - signal_price) / signal_price
            actual_holding_days = exit_idx - signal_idx

            results.append(
                {
                    "ts_code": ts_code,
                    "signal_date": signal_date,
                    "exit_date": str(exit_date),
                    "entry_price": signal_price,
                    "exit_price": exit_price,
                    "holding_return": holding_return,
                    "holding_days": actual_holding_days,
                    "exit_reason": exit_reason,
                }
            )

        if not results:
            return pl.DataFrame()

        return pl.DataFrame(results)

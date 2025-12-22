"""
Strategy backtest implementation.

Provides backtest functionality for backtrader-based trading strategies.
"""

import logging
from datetime import datetime
from typing import Any, Dict, Optional, Type

import pandas as pd

from atm.trading.strategy.base import BaseStrategy
from atm.trading.strategy.strategy_runner import StrategyRunner
from .base import BaseBacktester, BacktestResult

logger = logging.getLogger(__name__)


class StrategyBacktester(BaseBacktester):
    """
    Backtester for backtrader-based trading strategies.

    This class implements backtesting for strategies that inherit from BaseStrategy.
    It uses StrategyRunner to execute the strategy and collects results.
    """

    def __init__(
        self,
        db_config,
        schema: str = "quant",
        initial_cash: float = 100000.0,
        commission: float = 0.001,
        slippage: float = 0.0,
    ):
        """
        Initialize StrategyBacktester.

        Args:
            db_config: Database configuration.
            schema: Database schema name.
            initial_cash: Initial cash amount for backtesting.
            commission: Commission rate (default: 0.001 = 0.1%).
            slippage: Slippage rate (default: 0.0).
        """
        super().__init__(db_config, schema)
        self.initial_cash = initial_cash
        self.commission = commission
        self.slippage = slippage
        self.logger = logging.getLogger(self.__class__.__name__)

    def run(
        self,
        ts_code: str,
        start_date: datetime,
        end_date: datetime,
        strategy_class: Type[BaseStrategy],
        strategy_params: Optional[Dict[str, Any]] = None,
        add_analyzers: bool = True,
        **kwargs,
    ) -> BacktestResult:
        """
        Run backtest for a trading strategy.

        Args:
            ts_code: Stock code in Tushare format (e.g., 000001.SZ).
            start_date: Start date of backtest period (inclusive).
            end_date: End date of backtest period (inclusive).
            strategy_class: Strategy class (must inherit from BaseStrategy).
            strategy_params: Strategy parameters dictionary (default: None).
            add_analyzers: Whether to add backtrader analyzers (default: True).
            **kwargs: Additional parameters passed to StrategyRunner.

        Returns:
            BacktestResult containing strategy performance metrics and results.
        """
        self.validate_inputs(ts_code, start_date, end_date)

        logger.info(
            f"Running strategy backtest for {ts_code} from {start_date.date()} to {end_date.date()}"
        )

        try:
            # Extract kline_type from kwargs or use default
            kline_type = kwargs.get("kline_type", "day")

            # Run strategy using StrategyRunner
            runner = StrategyRunner.run_strategy(
                db_config=self.db_config,
                strategy_class=strategy_class,
                ts_code=ts_code,
                start_date=start_date,
                end_date=end_date,
                initial_cash=self.initial_cash,
                commission=self.commission,
                slippage=self.slippage,
                schema=self.schema,
                strategy_params=strategy_params,
                add_analyzers=add_analyzers,
                kline_type=kline_type,
            )

            # Get basic results
            basic_results = runner.get_results()
            analyzer_results = runner.get_analyzer_results() if add_analyzers else {}

            # Extract metrics from analyzer results
            # BacktestResult.metrics expects Dict[str, Any], so we can include nested structures
            # but we'll also extract key numeric values for easier access
            # All float values are rounded to 2 decimal places
            
            def round_float(value, decimals: int = 2):
                """Round float value to specified decimal places."""
                if value is None:
                    return None
                try:
                    return round(float(value), decimals)
                except (ValueError, TypeError):
                    return value
            
            all_metrics = {
                "initial_value": round_float(basic_results.get("initial_value")),
                "final_value": round_float(basic_results.get("final_value")),
                "total_return": round_float(basic_results.get("total_return")),
            }
            
            # Add analyzer results as nested structures
            if analyzer_results:
                all_metrics["analyzer_results"] = analyzer_results
                
                # Extract key numeric metrics for easy access (rounded to 2 decimals)
                if "trades" in analyzer_results:
                    trades = analyzer_results["trades"]
                    if isinstance(trades, dict):
                        if "total" in trades and "total" in trades["total"]:
                            all_metrics["total_trades"] = round_float(trades["total"]["total"], decimals=0)
                        if "won" in trades and "total" in trades["won"]:
                            all_metrics["won_trades"] = round_float(trades["won"]["total"], decimals=0)
                        if "lost" in trades and "total" in trades["lost"]:
                            all_metrics["lost_trades"] = round_float(trades["lost"]["total"], decimals=0)
                
                if "sharpe" in analyzer_results:
                    sharpe = analyzer_results["sharpe"]
                    if isinstance(sharpe, dict) and "sharperatio" in sharpe:
                        all_metrics["sharpe_ratio"] = round_float(sharpe["sharperatio"], decimals=2) if sharpe["sharperatio"] is not None else None
                
                if "drawdown" in analyzer_results:
                    dd = analyzer_results["drawdown"]
                    if isinstance(dd, dict) and "max" in dd:
                        max_dd = dd["max"]
                        if isinstance(max_dd, dict):
                            if "drawdown" in max_dd:
                                all_metrics["max_drawdown"] = round_float(max_dd["drawdown"], decimals=2) if max_dd["drawdown"] is not None else None
                            if "len" in max_dd:
                                all_metrics["max_drawdown_len"] = round_float(max_dd["len"], decimals=0) if max_dd["len"] is not None else None
                
                if "returns" in analyzer_results:
                    returns = analyzer_results["returns"]
                    if isinstance(returns, dict):
                        if "rnorm100" in returns:
                            all_metrics["normalized_return"] = round_float(returns["rnorm100"], decimals=2) if returns["rnorm100"] is not None else None
                        if "rtot" in returns:
                            all_metrics["total_return_pct"] = round_float(returns["rtot"], decimals=2) if returns["rtot"] is not None else None

            # Create predictions and actuals DataFrames for compatibility
            # For strategy backtest, we use the portfolio value over time
            # Load actual price data for comparison
            actual_df = self._load_actual_data(
                ts_code=ts_code,
                start_date=start_date,
                end_date=end_date,
            )

            # For strategy backtest, predictions represent portfolio value
            # We'll create a simple DataFrame with dates and portfolio values
            # Note: This is a simplified representation
            # In a real scenario, you might want to track portfolio value over time
            pred_df = pd.DataFrame({
                "date": [start_date, end_date],
                "close": [
                    self.initial_cash,  # Start with initial cash
                    basic_results.get("final_value", self.initial_cash),  # End with final value
                ],
            })

            logger.info(f"Strategy backtest completed. Final value: {basic_results.get('final_value', 0):.2f}")

            return BacktestResult(
                ts_code=ts_code,
                backtest_date=datetime.now(),
                start_date=start_date,
                end_date=end_date,
                predictions=pred_df,
                actuals=actual_df,
                metrics=all_metrics,
                metadata={
                    "strategy_class": strategy_class.__name__,
                    "strategy_params": strategy_params or {},
                    "initial_cash": self.initial_cash,
                    "commission": self.commission,
                    "slippage": self.slippage,
                },
            )

        except Exception as e:
            logger.error(f"Strategy backtest failed: {e}", exc_info=True)
            raise

    def _load_actual_data(
        self,
        ts_code: str,
        start_date: datetime,
        end_date: datetime,
    ) -> pd.DataFrame:
        """
        Load actual price data for comparison.

        Args:
            ts_code: Stock code.
            start_date: Start date.
            end_date: End date.

        Returns:
            DataFrame with actual price data.
        """
        from atm.repo.kline_repo import StockKlineDayRepo

        repo = StockKlineDayRepo(self.db_config, self.schema)

        klines = repo.get_by_ts_code(
            ts_code=ts_code,
            start_time=start_date,
            end_time=end_date,
        )

        if not klines:
            logger.warning(f"No actual data found for {ts_code} in date range")
            return pd.DataFrame()

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

        if not isinstance(df["date"].dtype, pd.DatetimeIndex):
            df["date"] = pd.to_datetime(df["date"])

        return df


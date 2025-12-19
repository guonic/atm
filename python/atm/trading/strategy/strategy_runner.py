"""
Strategy runner for backtrader strategies.

Provides a convenient way to run backtrader strategies with data loading,
broker configuration, and result reporting.
"""

import logging
from datetime import datetime
from typing import Any, Dict, Optional, Type

import backtrader as bt

from atm.config import DatabaseConfig
from atm.trading.strategy.data_feed import create_data_feed
from atm.trading.strategy.base import BaseStrategy

logger = logging.getLogger(__name__)


class StrategyRunner:
    """
    Runner for backtrader strategies.

    This class provides a convenient interface to:
    - Load data from database
    - Configure broker parameters
    - Add and run strategies
    - Get backtest results
    """

    def __init__(
        self,
        db_config: DatabaseConfig,
        initial_cash: float = 100000.0,
        commission: float = 0.001,
        slippage: float = 0.0,
    ):
        """
        Initialize strategy runner.

        Args:
            db_config: Database configuration for loading data.
            initial_cash: Initial cash amount (default: 100000.0).
            commission: Commission rate (default: 0.001 = 0.1%).
            slippage: Slippage rate (default: 0.0).
        """
        self.db_config = db_config
        self.initial_cash = initial_cash
        self.commission = commission
        self.slippage = slippage
        self.cerebro = bt.Cerebro()
        self._setup_broker()

    def _setup_broker(self):
        """Set up broker parameters."""
        self.cerebro.broker.setcash(self.initial_cash)
        self.cerebro.broker.setcommission(commission=self.commission)
        if self.slippage > 0:
            self.cerebro.broker.set_slippage_perc(self.slippage)

    def add_data(
        self,
        ts_code: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        schema: str = "quant",
    ) -> None:
        """
        Add data feed to cerebro.

        Args:
            ts_code: Stock code (e.g., '000001.SZ').
            start_date: Start date for data (inclusive).
            end_date: End date for data (inclusive).
            schema: Database schema name.
        """
        logger.info(f"Loading data for {ts_code} from {start_date} to {end_date}")
        try:
            data_feed = create_data_feed(
                db_config=self.db_config,
                ts_code=ts_code,
                start_date=start_date,
                end_date=end_date,
                schema=schema,
            )
            self.cerebro.adddata(data_feed)
        except Exception as e:
            logger.error(f"Failed to load data from database: {e}")
            raise

    def add_strategy(
        self,
        strategy_class: Type[BaseStrategy],
        **strategy_params,
    ) -> None:
        """
        Add strategy to cerebro.

        Args:
            strategy_class: Strategy class (must inherit from BaseStrategy).
            **strategy_params: Strategy parameters to pass to strategy.
        """
        self.cerebro.addstrategy(strategy_class, **strategy_params)

    def run(self) -> Any:
        """
        Run backtest.

        Returns:
            Backtest results from cerebro.run().
        """
        logger.info("Running backtest...")
        logger.info(f"Starting Portfolio Value: {self.cerebro.broker.getvalue():.2f}")

        results = self.cerebro.run()

        logger.info(f"Final Portfolio Value: {self.cerebro.broker.getvalue():.2f}")

        return results

    def get_results(self) -> Dict[str, Any]:
        """
        Get backtest results summary.

        Returns:
            Dictionary containing backtest statistics.
        """
        broker = self.cerebro.broker
        initial_value = broker.startingcash
        final_value = broker.getvalue()
        total_return = (
            (final_value - initial_value) / initial_value * 100
            if initial_value > 0
            else 0.0
        )

        return {
            "initial_value": initial_value,
            "final_value": final_value,
            "total_return": total_return,
        }

    @classmethod
    def print_results(self, results) -> None:
        """Print backtest results to logger."""
        logger.info("=" * 60)
        logger.info("Backtest Results:")
        logger.info("=" * 60)
        for key, value in results.items():
            if isinstance(value, float):
                logger.info(f"{key}: {value:.2f}")
            else:
                logger.info(f"{key}: {value}")
        logger.info("=" * 60)

    @classmethod
    def run_strategy(
        cls,
        db_config: DatabaseConfig,
        strategy_class: Type[BaseStrategy],
        ts_code: str,
        start_date: datetime,
        end_date: datetime,
        initial_cash: float = 100000.0,
        commission: float = 0.001,
        slippage: float = 0.0,
        schema: str = "quant",
        strategy_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Run a strategy with all necessary parameters in one call.

        This is a convenience method that combines all steps into a single call.

        Args:
            db_config: Database configuration for loading data.
            strategy_class: Strategy class (must inherit from BaseStrategy).
            ts_code: Stock code (e.g., '000001.SZ').
            start_date: Start date for data (inclusive).
            end_date: End date for data (inclusive).
            initial_cash: Initial cash amount (default: 100000.0).
            commission: Commission rate (default: 0.001 = 0.1%).
            slippage: Slippage rate (default: 0.0).
            schema: Database schema name (default: 'quant').
            strategy_params: Strategy parameters dictionary (default: None).

        Returns:
            Dictionary containing backtest results.
        """
        runner = cls(
            db_config=db_config,
            initial_cash=initial_cash,
            commission=commission,
            slippage=slippage,
        )

        runner.add_data(
            ts_code=ts_code,
            start_date=start_date,
            end_date=end_date,
            schema=schema,
        )

        runner.add_strategy(
            strategy_class,
            **(strategy_params or {}),
        )

        runner.run()

        return runner.get_results()


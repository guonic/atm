"""
Strategy runner for backtrader strategies.

Provides a convenient way to run backtrader strategies with data loading,
broker configuration, and result reporting.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Type

import backtrader as bt

from atm.config import DatabaseConfig
from atm.trading.strategies.data_feed import create_data_feed
from atm.trading.strategies.base import BaseStrategy

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
        run_id: Optional[str] = None,
        ts_code: Optional[str] = None,
    ):
        """
        Initialize strategy runner.

        Args:
            db_config: Database configuration for loading data.
            initial_cash: Initial cash amount (default: 100000.0).
            commission: Commission rate (default: 0.001 = 0.1%).
            slippage: Slippage rate (default: 0.0).
            run_id: Backtest run ID for signal tracking (default: None).
            ts_code: Stock code for signal tracking (default: None).
        """
        self.db_config = db_config
        self.initial_cash = initial_cash
        self.commission = commission
        self.slippage = slippage
        self.run_id = run_id
        self.ts_code = ts_code
        self.cerebro = bt.Cerebro()
        self._last_results: Optional[Any] = None
        self._strategy_instance: Optional[BaseStrategy] = None
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
        kline_type: str = "day",
    ) -> None:
        """
        Add data feed to cerebro.

        Args:
            ts_code: Stock code (e.g., '000001.SZ').
            start_date: Start date for data (inclusive).
            end_date: End date for data (inclusive).
            schema: Database schema name.
            kline_type: K-line type (day, hour, 30min, 15min, 5min, 1min).
        """
        logger.info(
            f"Loading {kline_type} data for {ts_code} from {start_date} to {end_date}"
        )
        try:
            data_feed = create_data_feed(
                db_config=self.db_config,
                ts_code=ts_code,
                start_date=start_date,
                end_date=end_date,
                schema=schema,
                kline_type=kline_type,
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
        # Pass run_id and ts_code to strategy for signal collection
        if self.run_id:
            strategy_params["run_id"] = self.run_id
        if self.ts_code:
            strategy_params["ts_code"] = self.ts_code
        
        self.cerebro.addstrategy(strategy_class, **strategy_params)

    def add_analyzers(self) -> None:
        """
        Add common analyzers to cerebro.

        Adds analyzers for:
        - Trade analysis (win/loss, profit/loss, etc.)
        - Sharpe ratio
        - Drawdown analysis
        - Returns analysis
        """
        # Trade analyzer
        self.cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
        
        # Sharpe ratio
        self.cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe")
        
        # Drawdown analyzer
        self.cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
        
        # Returns analyzer
        self.cerebro.addanalyzer(bt.analyzers.Returns, _name="returns")
        
        # Time return analyzer
        self.cerebro.addanalyzer(bt.analyzers.TimeReturn, _name="timereturn")
        
        logger.info("Added analyzers: trades, sharpe, drawdown, returns, timereturn")

    def plot(self, style: str = "bar", plotter: Optional[Any] = None, **kwargs) -> None:
        """
        Plot backtest results.

        Args:
            style: Plot style ('bar', 'candle', 'line', etc.).
            plotter: Optional custom plotter instance.
            **kwargs: Additional plotting options.
        """
        try:
            if plotter is None:
                self.cerebro.plot(style=style, **kwargs)
            else:
                plotter.plot(self.cerebro, style=style, **kwargs)
            logger.info("Plotting completed")
        except Exception as e:
            logger.error(f"Failed to plot: {e}")
            logger.info("Plotting requires matplotlib. Install with: pip install matplotlib")

    def run(self, optreturn: bool = False) -> Any:
        """
        Run backtest.

        Args:
            optreturn: If True, use optimized run mode (faster but may have issues with insufficient data).
                      If False, use standard run mode (default: False).

        Returns:
            Backtest results from cerebro.run().
        """
        logger.info("Running backtest...")
        logger.info(f"Starting Portfolio Value: {self.cerebro.broker.getvalue():.2f}")

        # Use standard run mode to avoid array index issues with insufficient data
        results = self.cerebro.run(optreturn=optreturn)

        # Store strategy instance for signal retrieval
        # Try multiple ways to get the strategy instance
        try:
            # Method 1: From cerebro.runstrats (most reliable)
            if hasattr(self.cerebro, 'runstrats') and self.cerebro.runstrats:
                if len(self.cerebro.runstrats) > 0 and len(self.cerebro.runstrats[0]) > 0:
                    self._strategy_instance = self.cerebro.runstrats[0][0]
            # Method 2: From results (fallback)
            elif results and len(results) > 0:
                if isinstance(results[0], list) and len(results[0]) > 0:
                    self._strategy_instance = results[0][0]
                elif isinstance(results[0], BaseStrategy):
                    self._strategy_instance = results[0]
        except Exception as e:
            logger.warning(f"Failed to get strategy instance for signal retrieval: {e}")

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

    def print_results(self) -> None:
        """Print backtest results to logger."""
        results = self.get_results()
        logger.info("=" * 60)
        logger.info("Backtest Results:")
        logger.info("=" * 60)
        for key, value in results.items():
            if isinstance(value, float):
                logger.info(f"{key}: {value:.2f}")
            else:
                logger.info(f"{key}: {value}")
        logger.info("=" * 60)

    def print_analyzer_results(self) -> None:
        """Print analyzer results to logger."""
        analyzer_results = self.get_analyzer_results()
        if not analyzer_results:
            logger.info("No analyzer results available")
            return

        logger.info("=" * 60)
        logger.info("Analyzer Results:")
        logger.info("=" * 60)

        # Trade analysis
        if "trades" in analyzer_results:
            trades = analyzer_results["trades"]
            try:
                if "total" in trades and "total" in trades["total"]:
                    total_trades = trades["total"]["total"]
                    logger.info(f"Total Trades: {total_trades}")
                if "won" in trades and "total" in trades["won"]:
                    won_trades = trades["won"]["total"]
                    logger.info(f"Won Trades: {won_trades}")
                if "lost" in trades and "total" in trades["lost"]:
                    lost_trades = trades["lost"]["total"]
                    logger.info(f"Lost Trades: {lost_trades}")
            except (KeyError, TypeError) as e:
                logger.warning(f"Failed to parse trade analysis: {e}")

        # Sharpe ratio
        if "sharpe" in analyzer_results:
            sharpe = analyzer_results["sharpe"]
            try:
                sharpe_ratio = sharpe.get("sharperatio")
                if sharpe_ratio is not None:
                    logger.info(f"Sharpe Ratio: {sharpe_ratio:.4f}")
                else:
                    logger.info("Sharpe Ratio: N/A (insufficient data)")
            except (KeyError, TypeError, ValueError) as e:
                logger.warning(f"Failed to parse Sharpe ratio: {e}")

        # Drawdown
        if "drawdown" in analyzer_results:
            dd = analyzer_results["drawdown"]
            try:
                if "max" in dd and dd["max"]:
                    max_dd = dd["max"].get("drawdown")
                    max_dd_len = dd["max"].get("len")
                    if max_dd is not None:
                        logger.info(f"Max Drawdown: {max_dd:.2f}%")
                    if max_dd_len is not None:
                        logger.info(f"Max Drawdown Period: {max_dd_len}")
            except (KeyError, TypeError, ValueError) as e:
                logger.warning(f"Failed to parse drawdown analysis: {e}")

        # Returns
        if "returns" in analyzer_results:
            returns = analyzer_results["returns"]
            try:
                rnorm100 = returns.get("rnorm100")
                if rnorm100 is not None:
                    logger.info(f"Normalized Return: {rnorm100:.2f}%")
            except (KeyError, TypeError, ValueError) as e:
                logger.warning(f"Failed to parse returns analysis: {e}")

        logger.info("=" * 60)

    def get_analyzer_results(self) -> Dict[str, Any]:
        """
        Get analyzer results from the last run.

        Returns:
            Dictionary containing analyzer results.
        """
        if not hasattr(self, "_last_results") or self._last_results is None:
            return {}

        results = {}
        strategy_result = self._last_results[0] if self._last_results else None

        if strategy_result is None:
            return results

        # Extract analyzer results
        analyzers = getattr(strategy_result, "analyzers", None)
        if analyzers is None:
            return results

        # Trade analyzer
        if hasattr(analyzers, "trades") and analyzers.trades:
            try:
                trade_analysis = analyzers.trades.get_analysis()
                results["trades"] = trade_analysis
            except Exception as e:
                logger.warning(f"Failed to get trade analysis: {e}")

        # Sharpe ratio
        if hasattr(analyzers, "sharpe") and analyzers.sharpe:
            try:
                sharpe_analysis = analyzers.sharpe.get_analysis()
                results["sharpe"] = sharpe_analysis
            except Exception as e:
                logger.warning(f"Failed to get Sharpe ratio: {e}")

        # Drawdown
        if hasattr(analyzers, "drawdown") and analyzers.drawdown:
            try:
                drawdown_analysis = analyzers.drawdown.get_analysis()
                results["drawdown"] = drawdown_analysis
            except Exception as e:
                logger.warning(f"Failed to get drawdown analysis: {e}")

        # Returns
        if hasattr(analyzers, "returns") and analyzers.returns:
            try:
                returns_analysis = analyzers.returns.get_analysis()
                results["returns"] = returns_analysis
            except Exception as e:
                logger.warning(f"Failed to get returns analysis: {e}")

        return results

    def get_signals(self) -> List[Dict[str, Any]]:
        """
        Get signals collected by the strategy instance.

        Returns:
            List of signal dictionaries, or empty list if no strategy instance or signals.
        """
        if self._strategy_instance is not None and hasattr(self._strategy_instance, "get_signals"):
            return self._strategy_instance.get_signals()
        return []

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
        add_analyzers: bool = True,
        kline_type: str = "day",
        run_id: Optional[str] = None,
    ) -> "StrategyRunner":
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
            add_analyzers: Whether to add analyzers (default: True).
            kline_type: K-line type (day, hour, 30min, 15min, 5min, 1min) (default: 'day').
            run_id: Backtest run ID for signal tracking (default: None).

        Returns:
            StrategyRunner instance with results available via get_results() and get_analyzer_results().
        """
        runner = cls(
            db_config=db_config,
            initial_cash=initial_cash,
            commission=commission,
            slippage=slippage,
            run_id=run_id,
            ts_code=ts_code,
        )

        runner.add_data(
            ts_code=ts_code,
            start_date=start_date,
            end_date=end_date,
            schema=schema,
            kline_type=kline_type,
        )

        runner.add_strategy(
            strategy_class,
            **(strategy_params or {}),
        )

        if add_analyzers:
            runner.add_analyzers()

        runner._last_results = runner.run()

        return runner


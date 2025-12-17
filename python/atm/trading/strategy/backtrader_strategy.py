"""
Backtrader-based trading strategy implementation.

Provides a wrapper around backtrader strategies to integrate with ATM framework.
"""

import logging
from datetime import datetime
from typing import Any, Dict, Optional

import backtrader as bt

from atm.config import DatabaseConfig
from atm.trading.strategy.base import BaseStrategy, StrategyConfig
from atm.trading.strategy.data_feed import create_data_feed

logger = logging.getLogger(__name__)


class BacktraderStrategyWrapper(bt.Strategy):
    """
    Wrapper class that adapts BaseStrategy to backtrader.Strategy.

    This allows using ATM's BaseStrategy interface with backtrader engine.
    """

    params = (
        ("strategy_instance", None),  # The actual strategy instance
    )

    def __init__(self):
        """Initialize wrapper strategy."""
        self.strategy = self.params.strategy_instance
        if self.strategy:
            # Store wrapper instance so strategy can call buy/sell/close
            # Note: _cerebro, _broker, _datas will be set in start() method
            self.strategy._wrapper_instance = self

    def start(self):
        """Called when strategy starts."""
        if self.strategy:
            # Set references after backtrader has initialized them
            # In backtrader, these are accessed as self.cerebro, self.broker, self.datas
            self.strategy._cerebro = self.cerebro
            self.strategy._broker = self.broker
            self.strategy._datas = self.datas
            # Now call the strategy's start method
            self.strategy.start()

    def next(self):
        """Called for each bar."""
        if self.strategy:
            self.strategy.next(self.datas[0])

    def stop(self):
        """Called when strategy stops."""
        if self.strategy:
            self.strategy.stop()

    def notify_order(self, order):
        """Called when order status changes."""
        if self.strategy:
            self.strategy.notify_order(order)

    def notify_trade(self, trade):
        """Called when trade is closed."""
        if self.strategy:
            self.strategy.notify_trade(trade)


class BacktraderStrategy(BaseStrategy):
    """
    Base class for backtrader-based trading strategies.

    This class provides integration between ATM's strategy framework
    and backtrader's backtesting engine.
    """

    def __init__(self, config: StrategyConfig):
        """
        Initialize backtrader strategy.

        Args:
            config: Strategy configuration.
        """
        super().__init__(config)
        self.cerebro: Optional[bt.Cerebro] = None
        self._cerebro: Optional[bt.Cerebro] = None
        self._broker: Optional[bt.Broker] = None
        self._datas: Optional[list] = None
        self.results: Optional[Any] = None

    def add_data(
        self,
        db_config: DatabaseConfig,
        ts_code: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        schema: str = "quant",
    ) -> None:
        """
        Add data feed to strategy.

        Args:
            db_config: Database configuration.
            ts_code: Stock code (e.g., '000001.SZ').
            start_date: Start date for data (inclusive).
            end_date: End date for data (inclusive).
            schema: Database schema name.
        """
        if self.cerebro is None:
            self.cerebro = bt.Cerebro()

        data_feed = create_data_feed(
            db_config=db_config,
            ts_code=ts_code,
            start_date=start_date,
            end_date=end_date,
            schema=schema,
        )

        self.cerebro.adddata(data_feed)

    def set_broker_params(
        self,
        cash: Optional[float] = None,
        commission: Optional[float] = None,
        slippage: Optional[float] = None,
    ) -> None:
        """
        Set broker parameters.

        Args:
            cash: Initial cash amount.
            commission: Commission rate (e.g., 0.001 = 0.1%).
            slippage: Slippage rate.
        """
        if self.cerebro is None:
            self.cerebro = bt.Cerebro()

        broker = self.cerebro.broker
        if cash is not None:
            broker.setcash(cash)
        else:
            broker.setcash(self.initial_cash)

        if commission is not None:
            broker.setcommission(commission=commission)
        else:
            broker.setcommission(commission=self.commission)

        if slippage is not None:
            broker.set_slippage_perc(slippage)
        else:
            if self.slippage > 0:
                broker.set_slippage_perc(self.slippage)

    def run(self) -> Any:
        """
        Run the backtest.

        Returns:
            Backtest results.
        """
        if self.cerebro is None:
            raise ValueError("Cerebro not initialized. Add data feed first.")

        # Create wrapper strategy class with strategy instance
        # We need to create a class with the strategy instance as a parameter
        class Wrapper(BacktraderStrategyWrapper):
            params = (("strategy_instance", self),)

        # Add strategy to cerebro
        self.cerebro.addstrategy(Wrapper)

        # Run backtest
        logger.info(f"Running backtest for strategy: {self.name}")
        self.results = self.cerebro.run()

        # Store wrapper instance for buy/sell/close methods
        if self.results and len(self.results) > 0:
            self._wrapper_instance = self.results[0]

        return self.results

    def get_backtest_results(self) -> Dict[str, Any]:
        """
        Get backtest results summary.

        Returns:
            Dictionary containing backtest statistics.
        """
        if self.results is None or len(self.results) == 0:
            return {}

        result = self.results[0]
        broker = self.cerebro.broker

        # Calculate total return
        initial_value = broker.startingcash
        final_value = broker.getvalue()
        total_return = (final_value - initial_value) / initial_value if initial_value > 0 else 0.0

        # Try to get total trades from analyzers if available
        total_trades = 0
        if hasattr(result, "analyzers") and result.analyzers:
            try:
                if hasattr(result.analyzers, "trades"):
                    trades_analysis = result.analyzers.trades.get_analysis()
                    if isinstance(trades_analysis, dict) and "total" in trades_analysis:
                        total_info = trades_analysis["total"]
                        if isinstance(total_info, dict) and "total" in total_info:
                            total_trades = total_info["total"]
                        elif isinstance(total_info, (int, float)):
                            total_trades = int(total_info)
            except (AttributeError, KeyError, TypeError):
                # Analyzers not available or different structure, use default
                pass

        return {
            "strategy_name": self.name,
            "initial_value": initial_value,
            "final_value": final_value,
            "total_return": total_return,
            "total_trades": total_trades,
        }

    def next(self, data: Any) -> None:
        """
        Called for each bar in the data feed.

        This method should be overridden by subclasses to implement
        the actual trading logic.

        Args:
            data: Current bar data from backtrader.
        """
        # Default implementation: do nothing
        pass

    def start(self) -> None:
        """Called when the strategy starts running."""
        logger.info(f"Strategy {self.name} started")

    def stop(self) -> None:
        """Called when the strategy stops running."""
        logger.info(f"Strategy {self.name} stopped")

    def buy(self, data: Optional[Any] = None, size: Optional[float] = None, price: Optional[float] = None, **kwargs) -> Any:
        """
        Place a buy order.

        This method should be called from within the strategy's next() method.
        It delegates to the wrapper strategy's buy() method.

        Args:
            data: Data feed (default: first data feed).
            size: Order size (default: use position sizing logic).
            price: Limit price (None for market order).
            **kwargs: Additional order parameters.

        Returns:
            Order object.
        """
        # This method will be called from within the wrapper strategy
        # The wrapper strategy has access to self.buy() from backtrader
        # We need to get the wrapper instance to call its buy method
        if not hasattr(self, "_wrapper_instance"):
            raise RuntimeError("Strategy not running. Call run() first.")

        wrapper = self._wrapper_instance
        if data is None:
            if self._datas and len(self._datas) > 0:
                data = self._datas[0]
            else:
                raise ValueError("No data feed available")

        return wrapper.buy(data=data, size=size, price=price, **kwargs)

    def sell(self, data: Optional[Any] = None, size: Optional[float] = None, price: Optional[float] = None, **kwargs) -> Any:
        """
        Place a sell order.

        This method should be called from within the strategy's next() method.
        It delegates to the wrapper strategy's sell() method.

        Args:
            data: Data feed (default: first data feed).
            size: Order size (default: use position sizing logic).
            price: Limit price (None for market order).
            **kwargs: Additional order parameters.

        Returns:
            Order object.
        """
        if not hasattr(self, "_wrapper_instance"):
            raise RuntimeError("Strategy not running. Call run() first.")

        wrapper = self._wrapper_instance
        if data is None:
            if self._datas and len(self._datas) > 0:
                data = self._datas[0]
            else:
                raise ValueError("No data feed available")

        return wrapper.sell(data=data, size=size, price=price, **kwargs)

    def close(self, data: Optional[Any] = None, **kwargs) -> Any:
        """
        Close current position.

        This method should be called from within the strategy's next() method.
        It delegates to the wrapper strategy's close() method.

        Args:
            data: Data feed (default: first data feed).
            **kwargs: Additional order parameters.

        Returns:
            Order object.
        """
        if not hasattr(self, "_wrapper_instance"):
            raise RuntimeError("Strategy not running. Call run() first.")

        wrapper = self._wrapper_instance
        if data is None:
            if self._datas and len(self._datas) > 0:
                data = self._datas[0]
            else:
                raise ValueError("No data feed available")

        return wrapper.close(data=data, **kwargs)


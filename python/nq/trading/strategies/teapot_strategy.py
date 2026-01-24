"""
Backtrader strategy for Teapot pattern recognition.

Implements Teapot signal-based trading strategy.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import backtrader as bt
import pandas as pd

logger = logging.getLogger(__name__)


class TeapotStrategy(bt.Strategy):
    """
    Teapot pattern recognition strategy for Backtrader.

    Strategy logic:
    1. Buy when Teapot signal is detected
    2. Exit based on exit strategy (fixed days, target return, stop loss)
    3. Manage position and risk
    """

    params = (
        ("signals_file", None),  # Signal file path
        ("exit_strategy", "fixed_days"),  # Exit strategy
        ("holding_days", 20),  # Fixed holding days
        ("target_return", 0.15),  # Target return (15%)
        ("stop_loss", -0.10),  # Stop loss (-10%)
        ("position_size", 0.1),  # Position size (10%)
        ("max_positions", 10),  # Maximum positions
        ("use_ml_score", False),  # Use ML score
        ("ml_score_threshold", 0.7),  # ML score threshold
    )

    def __init__(self):
        """Initialize strategy."""
        # Load signals
        self.signals = self._load_signals()

        # Track positions
        self.positions_data = {}  # {data: {entry_date, entry_price, ...}}

        # Track signals used
        self.used_signals = set()

    def _load_signals(self) -> pd.DataFrame:
        """Load signals from file."""
        if self.params.signals_file is None:
            return pd.DataFrame()

        signals_path = Path(self.params.signals_file)
        if not signals_path.exists():
            logger.warning(f"Signals file not found: {signals_path}")
            return pd.DataFrame()

        signals = pd.read_csv(signals_path)
        signals["signal_date"] = pd.to_datetime(signals["signal_date"])

        # Filter by ML score if enabled
        if self.params.use_ml_score and "ml_score" in signals.columns:
            signals = signals[
                signals["ml_score"] >= self.params.ml_score_threshold
            ]

        return signals

    def next(self):
        """Execute on each bar."""
        current_date = self.data.datetime.date(0)

        # Check for new signals
        date_signals = self.signals[
            self.signals["signal_date"].dt.date == current_date
        ]

        # Buy logic
        for _, signal in date_signals.iterrows():
            ts_code = signal["ts_code"]
            signal_key = f"{ts_code}_{current_date}"

            # Check if signal already used
            if signal_key in self.used_signals:
                continue

            # Check if we can open new position
            if len(self.positions_data) >= self.params.max_positions:
                continue

            # Check if we already have position in this stock
            has_position = False
            for data in self.datas:
                if hasattr(data, "_name") and data._name == ts_code:
                    if self.getposition(data).size > 0:
                        has_position = True
                        break

            if not has_position:
                # Find data feed for this stock
                for data in self.datas:
                    if hasattr(data, "_name") and data._name == ts_code:
                        # Buy
                        size = (
                            self.broker.getcash()
                            * self.params.position_size
                            / data.close[0]
                        )
                        self.buy(data=data, size=size)
                        self.used_signals.add(signal_key)

                        # Track position
                        self.positions_data[data] = {
                            "entry_date": current_date,
                            "entry_price": data.close[0],
                            "signal": signal,
                        }
                        logger.info(
                            f"Buy signal: {ts_code} at {data.close[0]} on {current_date}"
                        )
                        break

        # Exit logic
        for data, pos_info in list(self.positions_data.items()):
            if self.getposition(data).size == 0:
                continue

            entry_date = pos_info["entry_date"]
            entry_price = pos_info["entry_price"]
            current_price = data.close[0]
            current_return = (current_price - entry_price) / entry_price
            holding_days = (current_date - entry_date).days

            should_exit = False
            exit_reason = None

            if self.params.exit_strategy == "fixed_days":
                if holding_days >= self.params.holding_days:
                    should_exit = True
                    exit_reason = "fixed_days"
            elif self.params.exit_strategy == "target_return":
                if current_return >= self.params.target_return:
                    should_exit = True
                    exit_reason = "target_return"
            elif self.params.exit_strategy == "stop_loss":
                if current_return <= self.params.stop_loss:
                    should_exit = True
                    exit_reason = "stop_loss"
            elif self.params.exit_strategy == "combined":
                if current_return >= self.params.target_return:
                    should_exit = True
                    exit_reason = "target_return"
                elif current_return <= self.params.stop_loss:
                    should_exit = True
                    exit_reason = "stop_loss"
                elif holding_days >= self.params.holding_days:
                    should_exit = True
                    exit_reason = "fixed_days"

            if should_exit:
                self.close(data=data)
                logger.info(
                    f"Sell signal: {data._name} at {current_price} on {current_date}, "
                    f"return: {current_return:.2%}, reason: {exit_reason}"
                )
                del self.positions_data[data]

    def notify_order(self, order):
        """Notify order status."""
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status in [order.Completed]:
            if order.isbuy():
                logger.debug(
                    f"BUY EXECUTED: {order.data._name}, "
                    f"Price: {order.executed.price:.2f}, "
                    f"Size: {order.executed.size}"
                )
            elif order.issell():
                logger.debug(
                    f"SELL EXECUTED: {order.data._name}, "
                    f"Price: {order.executed.price:.2f}, "
                    f"Size: {order.executed.size}"
                )

        elif order.status in [
            order.Canceled,
            order.Margin,
            order.Rejected,
        ]:
            logger.warning(f"Order {order.status}: {order.data._name}")

    def notify_trade(self, trade):
        """Notify trade status."""
        if not trade.isclosed:
            return

        logger.info(
            f"TRADE PROFIT: {trade.data._name}, "
            f"PnL: {trade.pnl:.2f}, "
            f"Return: {trade.pnlcomm / trade.value * 100:.2f}%"
        )

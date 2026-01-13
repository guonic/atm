#!/usr/bin/env python3
"""
ML Exit Strategy with separate buy and sell logic.

This strategy:
- Buy: Uses Structure Expert model predictions to select top K stocks
- Sell: Uses ML Exit model to determine when to exit positions (not based on top K ranking)
"""

import logging
from typing import Dict, List, Optional

import pandas as pd
from qlib.backtest.decision import Order, OrderDir
from qlib.data import D

from examples.backtest_structure_expert import (
    BaseCustomStrategy,
    get_refined_top_k,
)
from nq.analysis.exit import ExitModel

logger = logging.getLogger(__name__)


class MLExitStrategy(BaseCustomStrategy):
    """
    Strategy with ML-based exit model.
    
    Buy logic: Uses Structure Expert model predictions to select top K stocks
    Sell logic: Uses ML Exit model to determine when to exit (not based on ranking)
    """
    
    def __init__(
        self,
        signal: pd.DataFrame,
        topk: int = 30,
        buffer_ratio: float = 0.15,
        exit_model_path: Optional[str] = None,
        exit_scaler_path: Optional[str] = None,
        exit_threshold: float = 0.65,
        use_ml_exit: bool = True,
        n_drop: int = 5,
    ):
        """
        Initialize MLExitStrategy.
        
        Args:
            signal: Signal DataFrame with MultiIndex (datetime, instrument) and 'score' column.
            topk: Target number of stocks to hold.
            buffer_ratio: Buffer ratio for reducing turnover (default: 0.15).
            exit_model_path: Path to trained exit model (if None, ML exit is disabled).
            exit_scaler_path: Path to feature scaler (if None, auto-generated).
            exit_threshold: Risk probability threshold for exit signal (default: 0.65).
            use_ml_exit: Whether to use ML exit model (default: True).
            n_drop: Number of bottom stocks to drop (default: 5).
        """
        # Initialize base TopkDropoutStrategy with required parameters
        super().__init__(signal=signal, topk=topk, n_drop=n_drop)
        
        self.topk = topk
        self.buffer_ratio = buffer_ratio
        self.use_ml_exit = use_ml_exit
        self.exit_model: Optional[ExitModel] = None
        self.exit_threshold = exit_threshold
        
        # Track positions for exit model
        self.position_tracker: Dict[str, Dict] = {}
        
        # Track ML exit decisions for debugging
        self.ml_exit_decisions: List[Dict] = []
        
        # Store original signal for buy logic
        self.original_signal = signal
        
        # Load exit model if provided
        if use_ml_exit and exit_model_path:
            try:
                self.exit_model = ExitModel.load(
                    model_path=exit_model_path,
                    scaler_path=exit_scaler_path,
                    threshold=exit_threshold,
                )
                logger.info(f"✓ Exit model loaded from {exit_model_path} (threshold={exit_threshold})")
            except Exception as e:
                logger.warning(f"Failed to load exit model: {e}. ML exit will be disabled.")
                self.use_ml_exit = False
                self.exit_model = None
        elif use_ml_exit:
            logger.warning("use_ml_exit=True but no exit_model_path provided. ML exit disabled.")
            self.use_ml_exit = False
    
    def generate_trade_decision(self, trade_exchange: Optional["Exchange"] = None) -> List[Order]:
        """
        Generate trade decisions for current trading day.
        
        This method is called by Qlib for each trading day to generate orders.
        
        Args:
            trade_exchange: Qlib Exchange object (maybe None in some Qlib versions).
        
        Returns:
            List of Order objects.
        """
        # Handle None trade_exchange (some Qlib versions pass None)
        if trade_exchange is None:
            logger.warning("trade_exchange is None, falling back to parent class generate_trade_decision")
            # Fall back to parent class implementation (TopkDropoutStrategy)
            return super().generate_trade_decision(trade_exchange)
        
        # Get current date from trade_exchange
        try:
            current_date = pd.Timestamp(trade_exchange.get_current_time())
        except AttributeError:
            logger.error("trade_exchange.get_current_time() failed, falling back to parent class")
            return super().generate_trade_decision(trade_exchange)
        
        # Get current holdings
        current_holdings = self._get_current_holdings(trade_exchange)
        
        orders = []
        
        # 1. SELL LOGIC: Check ML Exit model for existing positions
        if self.use_ml_exit and self.exit_model:
            sell_symbols = self._check_ml_exit_signals(current_date, current_holdings)
            for symbol in sell_symbols:
                if symbol in current_holdings:
                    amount = current_holdings[symbol]
                    if amount > 0:
                        order = Order(
                            stock_id=symbol,
                            amount=amount,
                            direction=OrderDir.SELL,
                            factor=1.0,
                        )
                        orders.append(order)
                        logger.info(
                            f"ML Exit: Generate SELL order for {symbol} on {current_date.strftime('%Y-%m-%d')}, "
                            f"amount={amount:.0f}"
                        )
        
        # 2. BUY LOGIC: Select top K stocks from Structure Expert predictions
        # Get signal for current date
        if current_date in self.original_signal.index.get_level_values(0):
            date_signal = self.original_signal.loc[current_date]
            
            # Get prediction scores
            if isinstance(date_signal, pd.Series):
                pred_scores = {date_signal.name: date_signal.get('score', 0.0)}
            else:
                pred_scores = date_signal['score'].to_dict()
            
            # Apply refined top K selection (with buffer to reduce turnover)
            selected_stocks = get_refined_top_k(
                current_holdings=list(current_holdings.keys()),
                pred_scores=pred_scores,
                top_k=self.topk,
                buffer_ratio=self.buffer_ratio,
            )
            
            # Generate buy orders for stocks not in current holdings
            # Calculate target position value per stock
            try:
                account = trade_exchange.get_account()
                total_value = account.get_cash() + account.get_stock_value()
                target_position_value = total_value / self.topk if self.topk > 0 else total_value
                
                for symbol in selected_stocks:
                    if symbol not in current_holdings:
                        # Get current price to calculate shares
                        try:
                            price_data = D.features(
                                instruments=[symbol],
                                fields=["$close"],
                                start_time=current_date.strftime("%Y-%m-%d"),
                                end_time=current_date.strftime("%Y-%m-%d"),
                            )
                            if not price_data.empty and symbol in price_data.columns.get_level_values(0):
                                current_price = float(price_data[symbol]['$close'].iloc[0])
                                if current_price > 0:
                                    # Calculate amount based on target position value
                                    amount = int(target_position_value / current_price / 100) * 100  # Round to 100 shares
                                    if amount > 0:
                                        order = Order(
                                            stock_id=symbol,
                                            amount=amount,
                                            direction=OrderDir.BUY,
                                            factor=1.0,
                                        )
                                        orders.append(order)
                                        logger.debug(
                                            f"Generate BUY order for {symbol} on {current_date.strftime('%Y-%m-%d')} "
                                            f"(score={pred_scores.get(symbol, 0.0):.4f}, amount={amount:.0f}, price={current_price:.2f})"
                                        )
                        except Exception as e:
                            logger.warning(f"Failed to generate buy order for {symbol}: {e}")
            except Exception as e:
                logger.warning(f"Failed to calculate target position value: {e}")
        
        return orders
    
    def _get_current_holdings(self, trade_exchange: "Exchange") -> Dict[str, float]:
        """Get current holdings from trade_exchange."""
        holdings = {}
        try:
            position = trade_exchange.get_account().get_position()
            if hasattr(position, 'position') and isinstance(position.position, dict):
                for symbol, pos_info in position.position.items():
                    if isinstance(pos_info, dict):
                        amount = float(pos_info.get('amount', 0.0))
                        if amount > 0:
                            holdings[symbol] = amount
        except Exception as e:
            logger.warning(f"Failed to get current holdings: {e}")
        return holdings
    
    def _check_ml_exit_signals(
        self, current_date: pd.Timestamp, current_holdings: Dict[str, float]
    ) -> List[str]:
        """
        Check ML exit signals for all current holdings.
        
        Args:
            current_date: Current trading date.
            current_holdings: Dict of current holdings {symbol: amount}.
        
        Returns:
            List of symbols that should be sold.
        """
        exit_symbols = []
        
        for symbol, amount in current_holdings.items():
            if amount <= 0:
                continue
            
            try:
                # Get current price
                current_price_data = D.features(
                    instruments=[symbol],
                    fields=["$close"],
                    start_time=current_date.strftime("%Y-%m-%d"),
                    end_time=current_date.strftime("%Y-%m-%d"),
                )
                
                if current_price_data.empty or symbol not in current_price_data.columns.get_level_values(0):
                    continue
                
                current_price = float(current_price_data[symbol]['$close'].iloc[0])
                
                # Check ML exit signal
                if self._check_ml_exit_signal(symbol, current_date, current_price):
                    exit_symbols.append(symbol)
            except Exception as e:
                logger.debug(f"Failed to check exit for {symbol}: {e}")
        
        return exit_symbols
    
    def _check_ml_exit_signal(
        self, symbol: str, date: pd.Timestamp, current_price: float
    ) -> bool:
        """
        Check if ML model suggests exiting position.
        
        Args:
            symbol: Stock symbol.
            date: Current date.
            current_price: Current price.
        
        Returns:
            True if should exit, False otherwise.
        """
        if not self.use_ml_exit or self.exit_model is None:
            return False
        
        # Get or create position info
        if symbol not in self.position_tracker:
            # New position - shouldn't exit immediately
            return False
        
        pos = self.position_tracker[symbol]
        
        # Get historical data for feature construction
        try:
            # Get last 10 days of data for this symbol
            end_date = date.strftime("%Y-%m-%d")
            start_date = (date - pd.Timedelta(days=15)).strftime("%Y-%m-%d")
            
            hist_data = D.features(
                instruments=[symbol],
                fields=["$close", "$high", "$low", "$volume"],
                start_time=start_date,
                end_time=end_date,
            )
            
            if hist_data.empty or symbol not in hist_data.columns.get_level_values(0):
                logger.debug(f"No historical data for {symbol} on {date}")
                return False
            
            # Extract OHLCV data
            symbol_data = hist_data[symbol]
            daily_df = pd.DataFrame({
                'close': symbol_data['$close'],
                'high': symbol_data['$high'],
                'low': symbol_data['$low'],
                'volume': symbol_data['$volume'],
            })
            
            if daily_df.empty:
                return False
            
            # Calculate days held
            days_held = (date - pos['entry_date']).days
            
            # Predict exit probability
            proba = self.exit_model.predict_proba(
                daily_df=daily_df,
                entry_price=pos['entry_price'],
                highest_price_since_entry=pos['highest_price_since_entry'],
                days_held=days_held,
            )
            
            if len(proba) == 0:
                return False
            
            latest_proba = proba[-1]
            
            # Log prediction
            if latest_proba > self.exit_threshold:
                logger.info(
                    f"ML Exit signal: {symbol} on {date.strftime('%Y-%m-%d')}, "
                    f"risk_prob={latest_proba:.3f}, "
                    f"curr_ret={(current_price - pos['entry_price']) / pos['entry_price']:.2%}, "
                    f"drawdown={(pos['highest_price_since_entry'] - current_price) / pos['highest_price_since_entry']:.2%}, "
                    f"days_held={days_held}"
                )
                return True
            
            return False
            
        except Exception as e:
            logger.debug(f"Failed to check ML exit signal for {symbol}: {e}")
            return False
    
    def post_exe_step(self, execute_result: Optional[list]) -> None:
        """
        Update position tracker after orders are executed.
        
        This method is called by Qlib after each execution step.
        """
        if not execute_result:
            return
        
        # Get current date from executed orders
        current_date = None
        if len(execute_result) > 0:
            try:
                first_order = execute_result[0][0]
                if hasattr(first_order, 'start_time'):
                    current_date = pd.Timestamp(first_order.start_time)
            except Exception:
                pass
        
        if current_date is None:
            return
        
        # Update position tracker from executed orders
        self._update_position_tracker_from_orders(execute_result, current_date)
    
    def _update_position_tracker_from_orders(
        self, execute_result: Optional[list], date: pd.Timestamp
    ) -> None:
        """
        Update position tracker from executed orders.
        
        Args:
            execute_result: List of executed order tuples: (order, trade_val, trade_cost, trade_price).
            date: Current date.
        """
        if not execute_result:
            return
        
        logger.debug(f"ML Exit: Processing {len(execute_result)} executed orders to update position tracker")
        
        try:
            for order_tuple in execute_result:
                if not isinstance(order_tuple, (tuple, list)) or len(order_tuple) != 4:
                    continue
                
                order, trade_val, trade_cost, trade_price = order_tuple
                symbol = order.stock_id
                amount = float(order.amount)
                price = float(trade_price)
                
                if amount <= 0:
                    continue
                
                # Handle buy orders (add to position)
                if order.direction == OrderDir.BUY:
                    if symbol not in self.position_tracker:
                        self.position_tracker[symbol] = {
                            'entry_date': date,
                            'entry_price': price,
                            'highest_price_since_entry': price,
                            'highest_date': date,
                            'amount': amount,
                        }
                    else:
                        pos = self.position_tracker[symbol]
                        old_value = pos['entry_price'] * pos['amount']
                        new_value = price * amount
                        pos['amount'] += amount
                        pos['entry_price'] = (old_value + new_value) / pos['amount'] if pos['amount'] > 0 else price
                        if price > pos['highest_price_since_entry']:
                            pos['highest_price_since_entry'] = price
                            pos['highest_date'] = date
                
                # Handle sell orders (reduce or remove position)
                elif order.direction == OrderDir.SELL:
                    if symbol in self.position_tracker:
                        pos = self.position_tracker[symbol]
                        pos['amount'] -= amount
                        if pos['amount'] <= 0:
                            del self.position_tracker[symbol]
                            
        except Exception as e:
            logger.error(f"ML Exit: Failed to update position tracker from orders: {e}", exc_info=True)

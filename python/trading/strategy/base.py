"""
Base strategy classes.

Defines DualModelStrategy that coordinates buy and sell models.
"""

from typing import TYPE_CHECKING, List
import pandas as pd
import logging

from ..interfaces.models import IBuyModel, ISellModel
from ..state import Order, OrderBook, OrderSide, OrderType
from ..logic import RiskManager, PositionAllocator

if TYPE_CHECKING:
    from ..state import PositionManager

logger = logging.getLogger(__name__)


class DualModelStrategy:
    """
    Dual-model strategy: Coordinates buy and sell logic.
    
    Workflow:
    1. Scan positions: Sell Model runs to generate sell signals
    2. Scan market: Buy Model runs to generate buy signals (only if free slots)
    3. Submit orders through OrderBus
    """
    
    def __init__(
        self,
        buy_model: IBuyModel,
        sell_model: ISellModel,
        position_manager: 'PositionManager',
        order_book: OrderBook,
        risk_manager: RiskManager,
        position_allocator: PositionAllocator,
    ):
        """
        Initialize dual-model strategy.
        
        Args:
            buy_model: Buy model instance.
            sell_model: Sell model instance.
            position_manager: PositionManager instance.
            order_book: OrderBook instance.
            risk_manager: RiskManager instance.
            position_allocator: PositionAllocator instance.
        """
        self.buy_model = buy_model
        self.sell_model = sell_model
        self.position_manager = position_manager
        self.order_book = order_book
        self.risk_manager = risk_manager
        self.position_allocator = position_allocator
    
    def on_bar(
        self,
        date: pd.Timestamp,
        market_data: pd.DataFrame
    ) -> None:
        """
        Daily trading logic (completely independent from Qlib callbacks).
        
        Args:
            date: Current trading date.
            market_data: Market data DataFrame (from Qlib, but not dependent on Qlib state).
        """
        # Step 1: Scan positions, Sell Model runs
        sell_orders = []
        sell_check_count = 0
        sell_prob_sum = 0.0
        sell_prob_max = 0.0
        sell_prob_min = 1.0
        
        for symbol, position in self.position_manager.all_positions.items():
            try:
                sell_check_count += 1
                risk_prob = self.sell_model.predict_exit(
                    position=position,
                    market_data=market_data,
                    date=date
                )
                
                sell_prob_sum += risk_prob
                sell_prob_max = max(sell_prob_max, risk_prob)
                sell_prob_min = min(sell_prob_min, risk_prob)
                
                # Log first few checks at INFO level for visibility
                if sell_check_count <= 3:
                    logger.info(
                        f"Exit check: {symbol} on {date.strftime('%Y-%m-%d')}, "
                        f"risk_prob={risk_prob:.3f}, threshold={self.sell_model.threshold:.3f}, "
                        f"days_held={(date - position.entry_date).days}, "
                        f"entry_price={position.entry_price:.2f}, high={position.high_price_since_entry:.2f}"
                    )
                else:
                    logger.debug(
                        f"Exit check: {symbol} on {date.strftime('%Y-%m-%d')}, "
                        f"risk_prob={risk_prob:.3f}, threshold={self.sell_model.threshold:.3f}, "
                        f"days_held={(date - position.entry_date).days}"
                    )
                
                if risk_prob > self.sell_model.threshold:
                    # Generate sell order
                    order = Order(
                        symbol=symbol,
                        side=OrderSide.SELL,
                        amount=position.amount,
                        order_type=OrderType.MARKET,
                        date=date,
                    )
                    
                    # Risk check
                    if self.risk_manager.check_order(order, market_data):
                        sell_orders.append(order)
                        logger.info(
                            f"Sell signal: {symbol} on {date.strftime('%Y-%m-%d')}, "
                            f"risk_prob={risk_prob:.3f}"
                        )
                    else:
                        logger.debug(f"Sell order rejected by risk manager: {symbol}")
            except Exception as e:
                logger.warning(f"Failed to check exit for {symbol}: {e}", exc_info=True)
        
        if sell_check_count > 0:
            avg_prob = sell_prob_sum / sell_check_count
            logger.info(
                f"Exit model stats on {date.strftime('%Y-%m-%d')}: "
                f"checked={sell_check_count} positions, avg_prob={avg_prob:.3f}, "
                f"min={sell_prob_min:.3f}, max={sell_prob_max:.3f}, threshold={self.sell_model.threshold:.3f}, "
                f"sell_signals={len(sell_orders)}"
            )
        
        # Step 2: Scan market, Buy Model runs (only if free slots)
        # Note: Consider pending sell orders when checking free slots
        # If we have sell orders, they will free up slots after execution
        current_positions = len(self.position_manager.all_positions)
        pending_sell_count = len(sell_orders)
        effective_positions = current_positions - pending_sell_count
        has_free_slot = effective_positions < self.position_allocator.target_positions
        
        buy_orders = []
        if has_free_slot:
            try:
                # Generate rankings
                ranks = self.buy_model.generate_ranks(
                    date=date,
                    market_data=market_data
                )
                
                # Select top K (exclude existing positions)
                current_symbols = set(self.position_manager.all_positions.keys())
                # Also exclude symbols that are being sold
                sell_symbols = {order.symbol for order in sell_orders}
                exclude_symbols = current_symbols | sell_symbols
                available_ranks = ranks[~ranks['symbol'].isin(exclude_symbols)]
                
                # Calculate how many positions we can buy
                # We can buy up to (target_positions - effective_positions)
                max_buy_count = self.position_allocator.target_positions - effective_positions
                top_k = available_ranks.head(max_buy_count)
                
                # Generate buy orders
                for _, row in top_k.iterrows():
                    symbol = row['symbol']
                    try:
                        target_value = self.position_allocator.calculate_position_size(
                            symbol=symbol,
                            account=self.position_manager.account,
                            position_manager=self.position_manager,
                            market_data=market_data
                        )
                        
                        if target_value > 0:
                            order = Order(
                                symbol=symbol,
                                side=OrderSide.BUY,
                                amount=0,  # Will be calculated by executor based on target_value
                                target_value=target_value,
                                order_type=OrderType.MARKET,
                                date=date,
                            )
                            
                            # Risk check
                            if self.risk_manager.check_order(order, market_data):
                                buy_orders.append(order)
                                logger.debug(
                                    f"Buy signal: {symbol} on {date.strftime('%Y-%m-%d')}, "
                                    f"score={row.get('score', 0.0):.4f}"
                                )
                            else:
                                logger.debug(f"Buy order rejected by risk manager: {symbol}")
                    except Exception as e:
                        logger.warning(f"Failed to generate buy order for {symbol}: {e}")
            except Exception as e:
                logger.warning(f"Failed to generate buy signals: {e}")
        
        # Step 3: Submit orders to OrderBook
        submitted_count = 0
        for order in sell_orders + buy_orders:
            try:
                self.order_book.submit(order)
                submitted_count += 1
                logger.debug(
                    f"Submitted order: {order.symbol} {order.side.value} "
                    f"(target_value={order.target_value}, amount={order.amount})"
                )
            except Exception as e:
                logger.warning(f"Failed to submit order for {order.symbol}: {e}", exc_info=True)
        
        logger.info(
            f"Generated {len(buy_orders)} buy orders, {len(sell_orders)} sell orders, "
            f"submitted {submitted_count} orders on {date.strftime('%Y-%m-%d')} "
            f"(positions: {current_positions}, effective: {effective_positions}, target: {self.position_allocator.target_positions})"
        )

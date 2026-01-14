"""
Asymmetric Strategy (非对称策略).

Coordinates buy and sell models with asymmetric logic:
- Buy model: Generates stock rankings and buy signals
- Sell model: Generates exit signals based on position state

This strategy implements the dual-model approach where buy and sell decisions
are made by separate models, allowing for asymmetric trading logic.
"""

from typing import TYPE_CHECKING
import pandas as pd
import logging

from ..interfaces.models import IBuyModel, ISellModel
from .base_custom import BaseCustomStrategy
from ..state import Order, OrderSide, OrderType

if TYPE_CHECKING:
    from ..state import PositionManager
    from ..logic import RiskManager, PositionAllocator
    from ..state import Account, OrderBook

logger = logging.getLogger(__name__)


class AsymmetricStrategy(BaseCustomStrategy):
    """
    Asymmetric Strategy (非对称策略): Coordinates buy and sell models.
    
    This strategy uses separate models for buy and sell decisions:
    - Buy Model: Analyzes market data to generate stock rankings
    - Sell Model: Analyzes position state to generate exit signals
    
    Workflow:
    1. Scan positions: Sell Model runs to generate sell signals
    2. Scan market: Buy Model runs to generate buy signals (only if free slots)
    3. Submit orders through OrderBook
    
    The "asymmetric" name reflects that buy and sell logic are independent
    and can have different characteristics (e.g., different models, thresholds).
    """
    
    def __init__(
        self,
        buy_model: IBuyModel,
        sell_model: ISellModel,
        position_manager: 'PositionManager',
        order_book: 'OrderBook',
        risk_manager: 'RiskManager',
        position_allocator: 'PositionAllocator',
        account: 'Account',
    ):
        """
        Initialize asymmetric strategy.
        
        Args:
            buy_model: Buy model instance (IBuyModel).
            sell_model: Sell model instance (ISellModel).
            position_manager: PositionManager instance.
            order_book: OrderBook instance.
            risk_manager: RiskManager instance.
            position_allocator: PositionAllocator instance.
            account: Account instance.
        """
        super().__init__(
            position_manager=position_manager,
            order_book=order_book,
            risk_manager=risk_manager,
            position_allocator=position_allocator,
            account=account,
        )
        self.buy_model = buy_model
        self.sell_model = sell_model
    
    @property
    def name(self) -> str:
        """Strategy name."""
        return "AsymmetricStrategy"
    
    def on_bar(
        self,
        date: pd.Timestamp,
        market_data: pd.DataFrame,
    ) -> None:
        """
        Daily trading logic.
        
        Args:
            date: Current trading date.
            market_data: Market data DataFrame in normalized format.
        """
        super().on_bar(date, market_data)
        
        # Step 1: Scan positions, Sell Model runs
        sell_orders = self._generate_sell_signals(date, market_data)
        
        # Step 2: Scan market, Buy Model runs (only if free slots)
        buy_orders = self._generate_buy_signals(date, market_data, sell_orders)
        
        # Step 3: Submit orders
        self._submit_orders(sell_orders + buy_orders, date)
        
        # Capture daily stats
        self._capture_daily_stats(date, len(sell_orders), len(buy_orders))
    
    def _generate_sell_signals(
        self,
        date: pd.Timestamp,
        market_data: pd.DataFrame,
    ) -> list[Order]:
        """
        Generate sell signals by checking all positions.
        
        Args:
            date: Current trading date.
            market_data: Market data DataFrame.
        
        Returns:
            List of sell orders.
        """
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
                        f"[{self.name}] Exit check: {symbol} on {date.strftime('%Y-%m-%d')}, "
                        f"risk_prob={risk_prob:.3f}, threshold={self.sell_model.threshold:.3f}, "
                        f"days_held={(date - position.entry_date).days}, "
                        f"entry_price={position.entry_price:.2f}, high={position.high_price_since_entry:.2f}"
                    )
                else:
                    logger.debug(
                        f"[{self.name}] Exit check: {symbol} on {date.strftime('%Y-%m-%d')}, "
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
                            f"[{self.name}] Sell signal: {symbol} on {date.strftime('%Y-%m-%d')}, "
                            f"risk_prob={risk_prob:.3f}"
                        )
                    else:
                        logger.debug(f"[{self.name}] Sell order rejected by risk manager: {symbol}")
            except Exception as e:
                logger.warning(f"[{self.name}] Failed to check exit for {symbol}: {e}", exc_info=True)
        
        if sell_check_count > 0:
            avg_prob = sell_prob_sum / sell_check_count
            logger.info(
                f"[{self.name}] Exit model stats on {date.strftime('%Y-%m-%d')}: "
                f"checked={sell_check_count} positions, avg_prob={avg_prob:.3f}, "
                f"min={sell_prob_min:.3f}, max={sell_prob_max:.3f}, threshold={self.sell_model.threshold:.3f}, "
                f"sell_signals={len(sell_orders)}"
            )
        
        return sell_orders
    
    def _generate_buy_signals(
        self,
        date: pd.Timestamp,
        market_data: pd.DataFrame,
        sell_orders: list[Order],
    ) -> list[Order]:
        """
        Generate buy signals by scanning market.
        
        Args:
            date: Current trading date.
            market_data: Market data DataFrame.
            sell_orders: List of pending sell orders (to consider when calculating free slots).
        
        Returns:
            List of buy orders.
        """
        # Calculate free slots (consider pending sell orders)
        current_positions = len(self.position_manager.all_positions)
        pending_sell_count = len(sell_orders)
        effective_positions = current_positions - pending_sell_count
        has_free_slot = effective_positions < self.position_allocator.target_positions
        
        buy_orders = []
        if not has_free_slot:
            logger.debug(
                f"[{self.name}] No free slots: positions={current_positions}, "
                f"target={self.position_allocator.target_positions}"
            )
            return buy_orders
        
        try:
            # Generate rankings
            ranks = self.buy_model.generate_ranks(
                date=date,
                market_data=market_data
            )
            
            # Select top K (exclude existing positions and symbols being sold)
            current_symbols = set(self.position_manager.all_positions.keys())
            sell_symbols = {order.symbol for order in sell_orders}
            exclude_symbols = current_symbols | sell_symbols
            available_ranks = ranks[~ranks['symbol'].isin(exclude_symbols)]
            
            # Calculate how many positions we can buy
            max_buy_count = self.position_allocator.target_positions - effective_positions
            top_k = available_ranks.head(max_buy_count)
            
            # Generate buy orders
            for _, row in top_k.iterrows():
                symbol = row['symbol']
                try:
                    target_value = self.position_allocator.calculate_position_size(
                        symbol=symbol,
                        account=self.account,
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
                                f"[{self.name}] Buy signal: {symbol} on {date.strftime('%Y-%m-%d')}, "
                                f"score={row.get('score', 0.0):.4f}"
                            )
                        else:
                            logger.debug(f"[{self.name}] Buy order rejected by risk manager: {symbol}")
                except Exception as e:
                    logger.warning(f"[{self.name}] Failed to generate buy order for {symbol}: {e}")
        except Exception as e:
            logger.warning(f"[{self.name}] Failed to generate buy signals: {e}")
        
        return buy_orders
    
    def _submit_orders(
        self,
        orders: list[Order],
        date: pd.Timestamp,
    ) -> None:
        """
        Submit orders to order book.
        
        Args:
            orders: List of orders to submit.
            date: Current trading date.
        """
        submitted_count = 0
        for order in orders:
            try:
                self.order_book.submit(order)
                submitted_count += 1
                logger.debug(
                    f"[{self.name}] Submitted order: {order.symbol} {order.side.value} "
                    f"(target_value={order.target_value}, amount={order.amount})"
                )
            except Exception as e:
                logger.warning(f"[{self.name}] Failed to submit order for {order.symbol}: {e}", exc_info=True)
        
        buy_count = sum(1 for o in orders if o.side == OrderSide.BUY)
        sell_count = sum(1 for o in orders if o.side == OrderSide.SELL)
        current_positions = len(self.position_manager.all_positions)
        
        logger.info(
            f"[{self.name}] Generated {buy_count} buy orders, {sell_count} sell orders, "
            f"submitted {submitted_count} orders on {date.strftime('%Y-%m-%d')} "
            f"(positions: {current_positions}, target: {self.position_allocator.target_positions})"
        )
    
    def _capture_daily_stats(
        self,
        date: pd.Timestamp,
        sell_count: int,
        buy_count: int,
    ) -> None:
        """
        Capture daily statistics.
        
        Args:
            date: Current trading date.
            sell_count: Number of sell signals generated.
            buy_count: Number of buy signals generated.
        """
        stats = {
            'sell_signals': sell_count,
            'buy_signals': buy_count,
            'position_count': len(self.position_manager.all_positions),
            'total_value': self.account.get_total_value(),
            'cash': self.account.available_cash,
        }
        self.capture_daily_stats(stats)

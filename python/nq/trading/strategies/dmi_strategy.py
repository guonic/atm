"""
DMI (Directional Movement Index) Trading Strategy.

This strategy implements a trading system based on DMI (Directional Movement Index)
following the "old stock investor's stable profit system" with 3 clear signals.

DMI Components:
- +DI (plusDI): Power of rising prices (white line)
- -DI (minusDI): Power of falling prices (yellow line)
- ADX: Strength of the trend (purple line)

Buy Signal 1: "+DI crosses -DI" - Green light, you can buy
- Condition: +DI crosses above -DI from bottom to top
- Validation: ADX must simultaneously move upwards (trend strengthening)
- Additional: ADX must be above 20 (strong trend required)
- Operation: On cross day, buy 30% position. If ADX rises past 20, add another 30%

Hold Signal 2: "ADX exceeds 25" - Trend is strong, hold and don't sell
- Condition: ADX exceeds 25 (trend enters acceleration phase)
- Operation: As long as ADX doesn't fall below 25, keep holding

Sell Signal 3: "+DI crosses below -DI" - Red light, sell quickly
- Condition: +DI crosses below -DI from above
- Validation: ADX simultaneously moves downwards (trend weakening)
- Operation: Sell everything on cross day, don't hesitate

Pitfalls to avoid:
- Don't just look at daily DMI, should combine with weekly DMI
- Don't buy when ADX is below 20 (weak trend)

This strategy follows the standard backtrader pattern:
- Indicators are initialized in __init__
- Trading logic is in next() method
- Indicators trigger buy/sell operations
"""

import logging
from typing import Optional

import backtrader as bt
import backtrader.indicators as btind

from nq.trading.strategies.base import BaseStrategy

logger = logging.getLogger(__name__)


class DMIStrategy(BaseStrategy):
    """
    DMI (Directional Movement Index) Trading Strategy.

    This strategy uses DMI to identify trend strength and direction changes.
    Following the "old stock investor's stable profit system" with 3 clear signals.

    Buy Signal 1: "+DI crosses -DI" with ADX confirmation
    - +DI crosses above -DI
    - ADX moving upwards
    - ADX > 20 (strong trend required)
    - Buy 30% on cross, add 30% when ADX > 20

    Hold Signal 2: "ADX exceeds 25" - Keep holding
    - ADX > 25 indicates strong trend
    - Hold as long as ADX >= 25

    Sell Signal 3: "+DI crosses below -DI" with ADX confirmation
    - +DI crosses below -DI
    - ADX moving downwards
    - Sell everything on cross day

    Args:
        dmi_period: DMI calculation period (default: 14).
        adx_threshold_buy: Minimum ADX for buy signal (default: 20).
        adx_threshold_hold: ADX threshold for holding (default: 25).
        initial_position_pct: Initial position size percentage (default: 0.3 = 30%).
        add_position_pct: Additional position size percentage when ADX > 20 (default: 0.3 = 30%).
    """

    params = (
        ("dmi_period", 14),  # DMI calculation period
        ("adx_threshold_buy", 20),  # Minimum ADX for buy signal
        ("adx_threshold_hold", 25),  # ADX threshold for holding
        ("initial_position_pct", 0.3),  # Initial position size (30%)
        ("add_position_pct", 0.3),  # Additional position size when ADX > 20 (30%)
    )

    def __init__(self):
        """
        Initialize DMI Strategy.

        Indicators are initialized here following backtrader's standard pattern.
        """
        super().__init__()

        # DMI Indicator
        # Backtrader's DirectionalMovementIndex provides:
        # - plusDI: +DI line (power of rising prices)
        # - minusDI: -DI line (power of falling prices)
        # - adx: ADX line (trend strength)
        self.dmi = btind.DirectionalMovementIndex(
            self.data, period=self.p.dmi_period
        )

        # Track position state
        self.entry_price: Optional[float] = None
        self.initial_position_added = False  # Track if initial 30% position added
        self.add_position_added = False  # Track if additional 30% position added
        self.last_plus_di: Optional[float] = None
        self.last_minus_di: Optional[float] = None
        self.last_adx: Optional[float] = None  # Track last ADX for add position logic

        logger.info(
            f"DMI Strategy initialized with "
            f"DMI({self.p.dmi_period}), "
            f"ADX_buy_threshold={self.p.adx_threshold_buy}, "
            f"ADX_hold_threshold={self.p.adx_threshold_hold}"
        )

    def next(self):
        """
        Called for each bar.

        Trading logic based on DMI:
        1. Buy Signal: +DI crosses -DI with ADX confirmation
        2. Hold Signal: ADX > 25, keep holding
        3. Sell Signal: +DI crosses below -DI with ADX confirmation
        """
        # Skip if not enough data
        if len(self.data) < self.p.dmi_period + 5:
            return

        # Safely access indicator values
        try:
            current_plus_di = self.dmi.plusDI[0]
            current_minus_di = self.dmi.minusDI[0]
            current_adx = self.dmi.adx[0]
            prev_plus_di = self.dmi.plusDI[-1]
            prev_minus_di = self.dmi.minusDI[-1]
            prev_adx = self.dmi.adx[-1]
        except (IndexError, TypeError):
            return

        # Check for crossovers
        plus_cross_above = (
            prev_plus_di <= prev_minus_di and current_plus_di > current_minus_di
        )
        plus_cross_below = (
            prev_plus_di >= prev_minus_di and current_plus_di < current_minus_di
        )

        # Check ADX direction
        adx_rising = current_adx > prev_adx
        adx_falling = current_adx < prev_adx

        # Buy Signal 1: "+DI crosses -DI" - Green light, you can buy
        if not self.position:
            if plus_cross_above and adx_rising:
                # Additional validation: ADX must be above threshold for buy signal
                if current_adx >= self.p.adx_threshold_buy:
                    # Buy initial 30% position on cross day
                    cash = self.broker.getcash()
                    size = int(cash * self.p.initial_position_pct / self.data.close[0])
                    if size > 0:
                        self.buy(size=size)
                        self.initial_position_added = True
                        self.entry_price = self.data.close[0]
                        logger.info(
                            f"Buy Signal 1: +DI crossed -DI, ADX={current_adx:.2f}, "
                            f"Bought {size} shares (30% position)"
                        )
                else:
                    logger.debug(
                        f"Buy signal ignored: ADX={current_adx:.2f} < "
                        f"{self.p.adx_threshold_buy} (weak trend)"
                    )

        # Add position when ADX rises past threshold (can happen after initial buy)
        # Logic: If ADX was below threshold and now rises above, or continues rising above threshold
        if (
            self.position
            and self.initial_position_added
            and not self.add_position_added
            and current_adx >= self.p.adx_threshold_buy
        ):
            # Add additional 30% position when ADX rises above threshold
            # Check if ADX just crossed above threshold or is rising
            should_add = False
            if self.last_adx is not None:
                # ADX was below threshold and now above, or ADX is rising above threshold
                if (
                    self.last_adx < self.p.adx_threshold_buy
                    and current_adx >= self.p.adx_threshold_buy
                ) or (current_adx > self.last_adx and current_adx >= self.p.adx_threshold_buy):
                    should_add = True
            else:
                # First time checking, if ADX is above threshold, add position
                should_add = True

            if should_add:
                cash = self.broker.getcash()
                size = int(cash * self.p.add_position_pct / self.data.close[0])
                if size > 0:
                    self.buy(size=size)
                    self.add_position_added = True
                    logger.info(
                        f"Added position: ADX={current_adx:.2f} >= "
                        f"{self.p.adx_threshold_buy}, Bought {size} shares (30% position)"
                    )

        # Hold Signal 2: "ADX exceeds 25" - Trend is strong, hold and don't sell
        if self.position:
            # As long as ADX >= 25, keep holding
            if current_adx >= self.p.adx_threshold_hold:
                # Do nothing, just hold
                pass
            # If ADX falls below 25 but still above buy threshold, continue holding
            elif current_adx >= self.p.adx_threshold_buy:
                # Still in strong trend, continue holding
                pass

        # Sell Signal 3: "+DI crosses below -DI" - Red light, sell quickly
        if self.position:
            if plus_cross_below and adx_falling:
                # Sell everything on cross day
                self.close()
                self.initial_position_added = False
                self.add_position_added = False
                self.entry_price = None
                logger.info(
                    f"Sell Signal 3: +DI crossed below -DI, ADX={current_adx:.2f}, "
                    f"Sold all positions"
                )

        # Update last values for next iteration
        self.last_plus_di = current_plus_di
        self.last_minus_di = current_minus_di
        self.last_adx = current_adx


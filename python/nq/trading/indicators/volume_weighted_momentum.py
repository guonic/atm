"""
Volume Weighted Momentum Indicator.

The Volume Weighted Momentum (VWM) indicator reflects the strength of price
changes supported by specific trading volume. It emphasizes trend-following
trades in a clear trend direction.

Calculation:
1. Price Momentum: C - REF(C, MOMLEN)
2. Volume Weighted Momentum: EMA(VOL * MOMVALUE, AVGLEN)

This implementation follows backtrader's indicator pattern.
"""

import backtrader as bt
import backtrader.indicators as btind


class VolumeWeightedMomentum(bt.Indicator):
    """
    Volume Weighted Momentum (VWM) Indicator.

    This indicator calculates momentum weighted by volume to reflect the
    strength of price changes supported by trading volume.

    Parameters:
        mom_len: Momentum period (default: 10).
        avg_len: EMA averaging period (default: 20).

    Lines:
        vwm: Volume Weighted Momentum value.
        momentum: Raw price momentum value.

    Example:
        vwm = VolumeWeightedMomentum(data, mom_len=10, avg_len=20)
    """

    lines = ("vwm", "momentum")
    params = (("mom_len", 10), ("avg_len", 20))

    def __init__(self):
        """Initialize Volume Weighted Momentum indicator."""
        super().__init__()

        # Step 1: Calculate price momentum
        # Momentum = Current Close - Close N periods ago
        self.lines.momentum = self.data.close - self.data.close(-self.p.mom_len)

        # Step 2: Calculate volume weighted momentum
        # VWM = EMA(Volume * Momentum, AVGLEN)
        volume_momentum = self.data.volume * self.lines.momentum
        self.lines.vwm = btind.ExponentialSmoothing(
            volume_momentum, period=self.p.avg_len
        )


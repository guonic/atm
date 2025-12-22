"""
Hull Moving Average (HMA) Indicator.

The Hull Moving Average (HMA) is a technical indicator developed by Alan Hull
that reduces lag compared to traditional moving averages by using weighted
moving averages and mathematical formulas.

HMA Calculation Steps:
1. Calculate WMA for half-period (n/2) and full-period (n)
2. Create de-lagged series: 2×WMA(n/2) - WMA(n)
3. Apply WMA smoothing with square root period: WMA(√n)

This implementation follows backtrader's indicator pattern.
"""

import math

import backtrader as bt
import backtrader.indicators as btind


class HullMovingAverage(bt.Indicator):
    """
    Hull Moving Average (HMA) Indicator.

    The HMA reduces lag compared to traditional moving averages by using
    weighted moving averages and mathematical formulas to create a "de-lagged"
    price series.

    Parameters:
        period (int): Period for HMA calculation (default: 20).

    Lines:
        hma: Hull Moving Average value.

    Example:
        hma = HullMovingAverage(data, period=20)
    """

    lines = ("hma",)
    params = (("period", 20),)

    def __init__(self):
        """Initialize Hull Moving Average indicator."""
        super().__init__()

        # Step 1: Calculate WMA for half-period and full-period
        half_period = max(1, int(self.p.period / 2))
        wma_half = btind.WeightedMovingAverage(self.data, period=half_period)
        wma_full = btind.WeightedMovingAverage(self.data, period=self.p.period)

        # Step 2: Create de-lagged series: 2×WMA(n/2) - WMA(n)
        delagged = 2 * wma_half - wma_full

        # Step 3: Apply WMA smoothing with square root period
        sqrt_period = max(1, int(math.sqrt(self.p.period)))
        self.lines.hma = btind.WeightedMovingAverage(delagged, period=sqrt_period)


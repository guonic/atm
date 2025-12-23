"""
Custom technical indicators for backtrader.

This module contains various custom technical indicators that can be used
in backtrader strategies. All indicators inherit from backtrader.Indicator
and follow the standard backtrader indicator pattern.
"""

import backtrader as bt
import backtrader.indicators as btind
import backtrader.functions as btfunc


class SentimentZoneOscillator(bt.Indicator):
    """
    Sentiment Zone Oscillator (SZO) indicator.

    Measures market sentiment by analyzing price changes and applying
    a triple exponential moving average.

    Parameters:
        length (int): Period for calculation (default: 14).

    Lines:
        szo: Sentiment Zone Oscillator value.

    Example:
        szo = SentimentZoneOscillator(data, length=14)
    """
    lines = ('szo',)
    params = (('length',14),)

    def __init__(self):
        price_diff = self.data - self.data(-1)
        sign = (price_diff>0)*2-1
        sign = 100 * sign/self.params.length
        self.lines.szo = btind.TripleExponentialMovingAverage(sign, period=self.params.length)

class VolumeMovingAverage(bt.Indicator):
    """
    Volume Weighted Moving Average indicator.

    Calculates a moving average weighted by volume, using the typical price
    (high + low + close) / 3.

    Parameters:
        period (int): Period for calculation (default: 21).

    Lines:
        movav: Volume weighted moving average value.

    Example:
        vma = VolumeMovingAverage(data, period=21)
    """
    lines = ('movav',)
    params = (('period', 21),)

    plotinfo = dict(
        subplot=False
    )

    def __init__(self):
        price = (self.data.close+self.data.high+self.data.low) / 3
        vol_sum = btind.SumN(self.data.volume, period=self.p.period)
        self.lines.movav = btind.SumN(price*self.data.volume, period=self.p.period) / vol_sum

class VolSZO(bt.Indicator):
    """
    Volume-weighted Sentiment Zone Oscillator with buy/sell signals.

    Combines volume-weighted price with Sentiment Zone Oscillator to generate
    trading signals.

    Parameters:
        szolength (int): SZO calculation period (default: 2).
        sumlength (int): Volume sum period (default: 12).
        signal_percent (float): Signal threshold percentage (default: 95).

    Lines:
        szo: Sentiment Zone Oscillator value.
        buy: Buy signal line (positive values indicate buy).
        sell: Sell signal line (negative values indicate sell).

    Example:
        volszo = VolSZO(data, szolength=2, sumlength=12, signal_percent=95)
    """
    lines = ('szo', 'buy', 'sell',)
    params = (('szolength',2), ('sumlength',12), ('signal_percent',95))

    def __init__(self):
        price = (self.data.close+self.data.high+self.data.low) / 3
        vol_sum = btind.SumN(self.data.volume, period=self.params.sumlength)
        vol_weighted_price = btind.SumN(price*self.data.volume, period=self.params.sumlength)

        self.lines.szo = SentimentZoneOscillator(vol_weighted_price/vol_sum, length=self.params.szolength)
        self.lines.buy = btfunc.Max(self.p.signal_percent/self.params.szolength,0)
        self.lines.sell = btfunc.Min(-self.p.signal_percent/self.params.szolength,0)

class ZackPivotPoints(bt.Indicator):
    """
    Zack Pivot Points indicator.

    Calculates dynamic pivot points based on volume-weighted price and
    support/resistance levels.

    Parameters:
        avglength (int): Moving average period for slope calculation (default: 5).
        sumlength (int): Sum period for volume-weighted price (default: 5).

    Lines:
        final: Final pivot point value.
        zero: Zero reference line.

    Example:
        zpp = ZackPivotPoints(data, avglength=5, sumlength=5)
    """
    lines = ('final', 'zero')
    params = (('avglength',5), ('sumlength',5))

    def __init__(self):
        price = (self.data.close+self.data.high+self.data.low) / 3
        vol_price = btind.SumN(price*self.data.volume, period=self.p.sumlength) / btind.SumN(self.data.volume+1, period=self.p.sumlength)
        r2 = price + self.data.high - self.data.low
        s2 = price - self.data.high + self.data.low

        rslope = r2(0) - r2(-1)
        sslope = s2(0) - s2(-1)
        diff = vol_price(0) - vol_price(-1)

        resistance = btind.MovingAverageWilder(rslope, period=self.p.avglength)
        support = btind.MovingAverageWilder(sslope, period=self.p.avglength)
        difference = btind.MovingAverageWilder(diff, period=self.p.avglength)

        self.lines.final = (resistance+support+difference*5) / 3
        # self.lines.final = difference
        self.lines.zero = bt.LineNum(0)

class ZPP(bt.Indicator):
    """
    Zack Pivot Points (ZPP) buy/sell signal indicator.

    Generates buy and sell signals based on price crossing above/below
    dynamic support and resistance levels.

    Parameters:
        avglength (int): Moving average period (default: 12).

    Lines:
        buy: Buy signal (1 when price crosses above support, 0 otherwise).
        sell: Sell signal (1 when price crosses below resistance, 0 otherwise).

    Example:
        zpp = ZPP(data, avglength=12)
    """
    lines = ('buy', 'sell')
    params = (('avglength',12),)

    def __init__(self):
        self.price = (self.data.close+self.data.high+self.data.low) / 3
        self.r2 = btind.MovingAverageSimple(self.price*2-self.data.low, period=self.p.avglength)
        self.s2 = btind.MovingAverageSimple(self.price*2-self.data.high, period=self.p.avglength)

    def next(self):
        self.lines.buy[0] = (self.price > self.s2 and self.price[-1] <= self.s2[-1])
        self.lines.sell[0] = (self.price < self.r2 and self.price[-1] >= self.r2[-1])

class ZackOverMA(bt.Indicator):
    """
    Zack Over Moving Average momentum indicator.

    Measures momentum by counting how many periods price is above/below
    a moving average.

    Parameters:
        avglength (int): Moving average period (default: 21).
        sumlength (int): Sum period for counting (default: 20).
        movav: Moving average type (default: Simple).

    Lines:
        momentum: Momentum value (positive when above MA, negative when below).
        slope: Rate of change of momentum.
        zero: Zero reference line.

    Example:
        zoma = ZackOverMA(data, avglength=21, sumlength=20)
    """
    lines = ('momentum', 'slope','zero')
    params = (
        ('avglength', 21),
        ('sumlength', 20),
        ('movav', btind.MovAv.Simple)
    )

    def __init__(self):
        ma = self.p.movav(self.data, period=self.p.avglength)
        over = btind.SumN(self.data > ma, period=self.p.sumlength)
        under = btind.SumN(self.data < ma, period=self.p.sumlength)

        self.lines.momentum = (over - under) / self.p.sumlength
        self.lines.slope = self.lines.momentum(0) - self.lines.momentum(-1)
        self.lines.zero = bt.LineNum(0)

class ZackOverMA2(bt.Indicator):
    """
    Zack Over Moving Average 2 indicator (percentage-based).

    Similar to ZackOverMA but uses percentage deviation from moving average
    instead of simple count.

    Parameters:
        avglength (int): Moving average period (default: 21).
        sumlength (int): Sum period for accumulation (default: 20).
        movav: Moving average type (default: Simple).

    Lines:
        momentum: Percentage-based momentum value.
        slope: Rate of change of momentum.
        zero: Zero reference line.

    Example:
        zoma2 = ZackOverMA2(data, avglength=21, sumlength=20)
    """
    lines = ('momentum', 'slope','zero')
    params = (
        ('avglength', 21),
        ('sumlength', 20),
        ('movav', btind.MovAv.Simple)
    )

    def __init__(self):
        ma = self.p.movav(self.data, period=self.p.avglength)
        over = btind.SumN(self.data > ma, period=self.p.sumlength)
        under = btind.SumN(self.data < ma, period=self.p.sumlength)

        self.lines.momentum = btind.SumN((self.data - ma) / self.data, period=self.p.sumlength)
        self.lines.slope = self.lines.momentum(0) - self.lines.momentum(-1)
        self.lines.zero = bt.LineNum(0)

class ZackVolumeSignalOld(bt.Indicator):
    """
    Zack Volume Signal (old version) indicator.

    Generates volume-based buy/sell signals by analyzing price movements
    when volume exceeds average volume.

    Note: This is the old version. Consider using ZackVolumeSignal instead.

    Parameters:
        volperiod (int): Period for average volume calculation (default: 12).
        period (int): Smoothing period (default: 12).
        movav: Moving average type (default: Exponential).

    Lines:
        up: Upward volume signal.
        down: Downward volume signal.

    Example:
        zvs_old = ZackVolumeSignalOld(data, volperiod=12, period=12)
    """
    lines = ('up', 'down')
    params = (
        ('volperiod', 12),
        ('period', 12),
        ('movav', btind.MovAv.Exponential)
    )

    def __init__(self):
        avgvol = btind.ExponentialMovingAverage(self.data.volume, period=self.p.volperiod)
        priceUp = btind.If(self.data.volume(0)>avgvol(0), btind.If(self.data(0)>self.data(-1), (self.data(0)-self.data(-1))/self.data(0), 0), 0)
        priceDown = btind.If(self.data.volume(0)>avgvol(0), btind.If(self.data(0)<self.data(-1), (self.data(-1)-self.data(0))/self.data(0), 0), 0)
        self.lines.up = self.p.movav(priceUp, period=self.p.period)
        self.lines.down = self.p.movav(priceDown, period=self.p.period)

class ZackVolumeSignal(bt.Indicator):
    """
    Zack Volume Signal indicator.

    Generates volume-based buy/sell signals using standard deviation
    of volume to filter significant volume spikes.

    Parameters:
        volperiod (int): Period for average volume calculation (default: 12).
        period (int): Period for standard deviation calculation (default: 12).
        movav: Moving average type (default: Exponential).

    Lines:
        up: Upward volume signal.
        down: Downward volume signal.

    Example:
        zvs = ZackVolumeSignal(data, volperiod=12, period=12)
    """
    lines = ('up', 'down')
    params = (
        ('volperiod', 12),
        ('period', 12),
        ('movav', btind.MovAv.Exponential)
    )

    def __init__(self):
        avgvol = btind.ExponentialMovingAverage(self.data.volume, period=self.p.volperiod)
        stdev = btind.StdDev(self.data.volume, period=self.p.period-1)
        priceUp = btind.If(self.data.volume(0)>avgvol(0)+stdev(-1), btind.If(self.data(0)>self.data(-1), (self.data(0)-self.data(-1))/self.data(0), 0), 0)
        priceDown = btind.If(self.data.volume(0)>avgvol(0)+stdev(-1), btind.If(self.data(0)<self.data(-1), (self.data(-1)-self.data(0))/self.data(0), 0), 0)
        self.lines.up = self.p.movav(priceUp, period=self.p.period)
        self.lines.down = self.p.movav(priceDown, period=self.p.period)

class ZackVolumeSignalStdDev(bt.Indicator):
    """
    Zack Volume Signal based on Standard Deviation.

    Uses volume standard deviation changes to generate trading signals,
    with additional smoothing.

    Parameters:
        period (int): Period for standard deviation calculation (default: 12).
        movav: Moving average type (default: Exponential).
        smoothPeriod (int): Additional smoothing period (default: 3).

    Lines:
        up: Upward volume signal.
        down: Downward volume signal.

    Example:
        zvs_std = ZackVolumeSignalStdDev(data, period=12, smoothPeriod=3)
    """
    lines = ('up', 'down')
    params = (
        ('period', 12),
        ('movav', btind.MovAv.Exponential),
        ('smoothPeriod', 3)
    )

    def __init__(self):
        stdev = btind.StdDev(self.data.volume, period=self.p.period)
        price = self.data.close

        priceUp = btind.If(stdev(0)>stdev(-1), btind.If(price(0)>price(-1), (price(0)-price(-1))/price(0), 0), 0)
        priceDown = btind.If(stdev(0)>stdev(-1), btind.If(price(0)<price(-1), (price(-1)-price(0))/price(0), 0), 0)
        upSmoothed = self.p.movav(priceUp, period=self.p.period)
        downSmoothed = self.p.movav(priceDown, period=self.p.period)
        self.lines.up = self.p.movav(upSmoothed, period=self.p.smoothPeriod)
        self.lines.down = self.p.movav(downSmoothed, period=self.p.smoothPeriod)

class MT5Accelerator(bt.Indicator):
    """
    MT5 Accelerator indicator.

    Measures the acceleration of price movement by comparing fast and slow
    moving averages and their difference.

    Parameters:
        fastPeriod (int): Fast moving average period (default: 5).
        slowPeriod (int): Slow moving average period (default: 34).
        lookback (int): Lookback period for difference calculation (default: 5).
        movav: Moving average type (default: Exponential).

    Lines:
        acc: Accelerator value.

    Example:
        acc = MT5Accelerator(data, fastPeriod=5, slowPeriod=34, lookback=5)
    """
    lines = ('acc',)
    params = (
        ('fastPeriod', 5),
        ('slowPeriod', 34),
        ('lookback', 5),
        ('movav', btind.MovAv.Exponential)
    )

    def __init__(self):
        fastMA = self.p.movav(self.data, period=self.p.fastPeriod)
        slowMA = self.p.movav(self.data, period=self.p.slowPeriod)
        diff = fastMA - slowMA
        diffSum = btind.SumN(diff, period=self.p.lookback)
        self.lines.acc = diff - diffSum/self.p.lookback

class ZackMinMax(bt.Indicator):
    """
    Zack Min/Max indicator.

    Calculates highest, lowest, and midpoint values over a period.

    Parameters:
        period (int): Period for min/max calculation (default: 20).

    Lines:
        mid: Midpoint between highest and lowest.
        top: Highest value over period.
        bot: Lowest value over period.

    Example:
        zmm = ZackMinMax(data, period=20)
    """
    lines = ('mid', 'top', 'bot')
    params = (
        ('period', 20),
    )
    plotinfo=dict(subplot=False)

    def __init__(self):
        self.lines.top = btind.Highest(self.data.high, period=self.p.period)
        self.lines.bot = btind.Lowest(self.data.low, period=self.p.period)
        self.lines.mid = (self.lines.top(0) + self.lines.bot(0)) / 2

class MinMaxPercentage(bt.Indicator):
    """
    Min/Max Percentage indicator.

    Calculates the percentage position of current price within the
    min-max range over a period.

    Parameters:
        period (int): Period for min/max calculation (default: 20).

    Lines:
        percent: Percentage value (0-100) indicating position in range.

    Example:
        mmp = MinMaxPercentage(data, period=20)
    """
    lines = ('percent',)
    params = (
        ('period', 20),
    )

    def __init__(self):
        minmax = ZackMinMax(self.data, period=self.p.period)
        self.lines.percent = (self.data - minmax.bot) / (minmax.top - minmax.bot) * 100

class AboveMAAccum(bt.Indicator):
    """
    Above Moving Average Accumulation indicator.

    Accumulates the percentage deviation from moving average over a period.

    Parameters:
        avglength (int): Moving average period (default: 21).
        sumlength (int): Accumulation period (default: 50).

    Lines:
        accum: Accumulated percentage deviation.
        slope: Rate of change of accumulation.

    Example:
        ama = AboveMAAccum(data, avglength=21, sumlength=50)
    """
    lines = ('accum', 'slope')
    params = (
        ('avglength', 21),
        ('sumlength', 50)
    )

    def __init__(self):
        ma = btind.MovingAverageSimple(self.data, period=self.p.avglength)
        # self.lines.accum = btind.Accum((self.data - ma) / ma)
        self.lines.accum = btind.SumN((self.data-ma) / ma, period=self.p.sumlength)
        self.lines.slope = self.accum(0) - self.accum(-1)

class BHErgodic(bt.Indicator):
    """
    Bill's Ergodic indicator.

    A momentum oscillator that uses multiple exponential moving averages
    to smooth price changes.

    Parameters:
        rPeriod (int): First smoothing period (default: 2).
        sPeriod (int): Second smoothing period (default: 10).
        uPeriod (int): Third smoothing period (default: 5).
        triggerPeriod (int): Signal line period (default: 3).
        movav: Moving average type (default: Exponential).

    Lines:
        erg: Ergodic oscillator value.
        signal: Signal line.

    Example:
        erg = BHErgodic(data, rPeriod=2, sPeriod=10, uPeriod=5, triggerPeriod=3)
    """
    lines = ('erg', 'signal')
    params = (
        ('rPeriod', 2),
        ('sPeriod', 10),
        ('uPeriod', 5),
        ('triggerPeriod', 3),
        ('movav', btind.MovAv.Exponential)
    )

    def __init__(self):
        delta = self.data-self.data(-1)
        delta2 = abs(self.data-self.data(-1))

        rma = self.p.movav(delta, period=self.p.rPeriod)
        rma2 = self.p.movav(delta2, period=self.p.rPeriod)
        sma = self.p.movav(rma, period=self.p.sPeriod)
        sma2 = self.p.movav(rma2, period=self.p.sPeriod)
        uma = self.p.movav(sma, period=self.p.uPeriod)
        uma2 = self.p.movav(sma2, period=self.p.uPeriod)

        self.lines.erg = btind.If(uma2 > 0, 100*uma / uma2, 0)
        self.lines.signal = self.p.movav(self.lines.erg, period=self.p.triggerPeriod)

class ZackMADiff(bt.Indicator):
    """
    Zack Moving Average Difference indicator.

    Calculates the difference between exponential and simple moving averages,
    normalized by price.

    Parameters:
        period (int): Period for moving averages (default: 12).

    Lines:
        res: Normalized difference value.

    Example:
        zmad = ZackMADiff(data, period=12)
    """
    lines = ('res',)
    params = (
        ('period', 12),
    )

    def __init__(self):
        ema = btind.ExponentialMovingAverage(self.data, period=self.p.period)
        sma = btind.MovingAverageSimple(self.data, period=self.p.period)

        self.lines.res = (ema*ema-sma*sma) / (self.data)

class ZackAverageVelocity(bt.Indicator):
    """
    Zack Average Velocity indicator.

    Measures the average rate of price change over a period.

    Parameters:
        period (int): Period for velocity calculation (default: 50).

    Lines:
        vel: Average velocity value.

    Example:
        zav = ZackAverageVelocity(data, period=50)
    """
    lines = ('vel',)
    params = (
        ('period', 50),
    )

    def __init__(self):
        vel = self.data(0) - self.data(-1)
        self.lines.vel = btind.SumN(vel/self.p.period, period=self.p.period)

class RexOscillator(bt.Indicator):
    """
    Rex Oscillator indicator.

    A momentum oscillator based on typical value calculation.

    Parameters:
        period (int): Oscillator period (default: 14).
        signalPeriod (int): Signal line period (default: 14).
        movav: Moving average type (default: Exponential).

    Lines:
        rex: Rex oscillator value.
        signal: Signal line.

    Example:
        rex = RexOscillator(data, period=14, signalPeriod=14)
    """
    lines = ('rex', 'signal')
    params = (
        ('period', 14),
        ('signalPeriod', 14),
        ('movav', btind.MovAv.Exponential)
    )

    def __init__(self):
        tvb = 3*self.data.close - (self.data.low+self.data.open+self.data.high)
        self.lines.rex = self.p.movav(tvb, period=self.p.period)
        self.lines.signal = self.p.movav(self.lines.rex, period=self.p.signalPeriod)

class AbsoluteStrengthOscillator(bt.Indicator):
    """
    Absolute Strength Oscillator indicator.

    Measures bullish and bearish strength relative to recent high/low range.

    Parameters:
        lookback (int): Lookback period for high/low (default: 6).
        period (int): Initial smoothing period (default: 2).
        smoothPeriod (int): Final smoothing period (default: 9).
        movav: Moving average type (default: Exponential).

    Lines:
        bulls: Bullish strength value.
        bears: Bearish strength value.

    Example:
        aso = AbsoluteStrengthOscillator(data, lookback=6, period=2, smoothPeriod=9)
    """
    lines = ('bulls', 'bears')
    params = (
        ('lookback', 6),
        ('period', 2),
        ('smoothPeriod', 9),
        ('movav', btind.MovAv.Exponential)
    )

    def __init__(self):
        smallest = btind.Lowest(self.data, period=self.p.lookback)
        highest = btind.Highest(self.data, period=self.p.lookback)
        bulls = (self.data(0) - smallest) / self.data(0)
        bears = (highest - self.data(0)) / self.data(0)

        avgBulls = self.p.movav(bulls, period=self.p.period)
        avgBears = self.p.movav(bears, period=self.p.period)

        self.lines.bulls = self.p.movav(avgBulls, period=self.p.smoothPeriod)
        self.lines.bears = self.p.movav(avgBears, period=self.p.smoothPeriod)

class Vortex(bt.Indicator):
    """
    Vortex Indicator.

    Measures trend direction and strength using high-low relationships
    and true range.

    Parameters:
        period (int): Calculation period (default: 14).

    Lines:
        VIU: Vortex Indicator Up (positive trend).
        VID: Vortex Indicator Down (negative trend).

    Example:
        vortex = Vortex(data, period=14)
    """
    lines = ('VIU', 'VID')
    params = (
        ('period', 14),
    )

    def __init__(self):
        viu = btind.SumN(abs(self.data.high - self.data.low(-1)), period=self.p.period)
        vid = btind.SumN(abs(self.data.low - self.data.high(-1)), period=self.p.period)
        str = btind.SumN(btind.AverageTrueRange(self.data, period=1), period=self.p.period)
        self.lines.VIU = viu / str
        self.lines.VID = vid / str

class SWMA(bt.Indicator):
    """
    Sine Weighted Moving Average indicator.

    A fixed-length moving average with sine-weighted coefficients.

    Lines:
        ma: Sine weighted moving average value.

    Example:
        swma = SWMA(data)
    """
    lines = ('ma',)

    def __init__(self):
        self.lines.ma = self.data(0)*1/6 + self.data(-1)*2/6 + self.data(-2)*2/6 + self.data(-3)*1/6

class RVI(bt.Indicator):
    """
    Relative Vigor Index indicator.

    Measures the strength of a trend by comparing closing price to opening price
    relative to the trading range.

    Parameters:
        period (int): Calculation period (default: 10).

    Lines:
        rvi: Relative Vigor Index value.
        sig: Signal line (smoothed RVI).

    Example:
        rvi = RVI(data, period=10)
    """
    lines = ('rvi', 'sig')
    params = (
        ('period', 10),
    )

    def __init__(self):
        self.lines.rvi = btind.SumN(SWMA(self.data.close-self.data.open), period=self.p.period) / btind.SumN(SWMA(self.data.high-self.data.low), period=self.p.period)
        self.lines.sig = SWMA(self.lines.rvi)

class WIMA(bt.Indicator):
    """
    Wilder's Initial Moving Average indicator.

    A type of exponential moving average that uses Wilder's smoothing method.

    Parameters:
        period (int): Calculation period (default: 14).

    Lines:
        ma: Wilder's moving average value.

    Example:
        wima = WIMA(data, period=14)
    """
    lines = ('ma',)
    params = (
        ('period', 14),
    )

    # def __init__(self):
    #     self.lines.ma = (self.data + self.lines.ma(-1) * (self.p.period-1)) / self.p.period

    def nextstart(self):
        self.lines.ma[0] = sum(self.data.get(size=self.p.period)) / self.p.period

    def next(self):
        self.lines.ma[0] = (self.data + self.lines.ma[-1] * (self.p.period-1)) / self.p.period

class QQE(bt.Indicator):
    """
    Quantitative Qualitative Estimation indicator.

    A modified RSI indicator with dynamic bands based on ATR.

    Note: This implementation is incomplete.

    Parameters:
        rsi_period (int): RSI calculation period (default: 8).
        rsi_smoothing_period (int): RSI smoothing period (default: 1).
        atr_period (int): ATR period for bands (default: 14).
        fast_mult (float): Fast multiplier for bands (default: 3).
        slow_mult (float): Slow multiplier for bands (default: 4.236).

    Lines:
        up: Upper band.
        down: Lower band.

    Example:
        qqe = QQE(data, rsi_period=8, atr_period=14)
    """
    lines = ('up', 'down')
    params = (
        ('rsi_period', 8),
        ('rsi_smoothing_period', 1),
        ('atr_period', 14),
        ('fast_mult', 3),
        ('slow_mult', 4.236)
    )

    def __init__(self):
        rsi_ema = btind.MovAv.Exponential(btind.RSI(self.data, period=self.p.rsi_period), period=self.p.rsi_smoothing_period)

        TH = btind.If(rsi_ema(-1) > rsi_ema, rsi_ema(-1), rsi_ema)
        TL = btind.IF(rsi_ema(-1) < rsi_ema, rsi_ema(-1), rsi_ema)
        TR = TH - TL

        atr_rsi = WIMA(TR, period=self.p.atr_period) # should be WIMA
        smoothed_atr_rsi = WIMA(atr_rsi, period=self.p.atr_period) # should be WIMA

        delta_fast_atr_rsi = smoothed_atr_rsi * self.p.fast_mult

        newshortband = rsi_ema + delta_fast_atr_rsi
        newlongband = rsi_ema - delta_fast_atr_rsi

        # longband = btind.If(rsi_ema(-1) > longband) # can't reasonably continue...

class HLCTrend(bt.Indicator):
    """
    High-Low-Close Trend indicator.

    Measures trend strength using separate moving averages for high, low, and close.

    Parameters:
        close_period (int): Close moving average period (default: 5).
        low_period (int): Low moving average period (default: 13).
        high_period (int): High moving average period (default: 34).
        movav: Moving average type (default: Exponential).

    Lines:
        up: Upward trend strength (close - high MA).
        down: Downward trend strength (low MA - close).

    Example:
        hlc = HLCTrend(data, close_period=5, low_period=13, high_period=34)
    """
    lines = ('up', 'down')
    params = (
        ('close_period', 5),
        ('low_period', 13),
        ('high_period', 34),
        ('movav', btind.MovAv.Exponential),
    )

    def __init__(self):
        emac = self.p.movav(self.data.close, period=self.p.close_period)
        emal = self.p.movav(self.data.low, period=self.p.low_period)
        emah = self.p.movav(self.data.high, period=self.p.high_period)

        self.lines.up = emac - emah
        self.lines.down = emal - emac

class ZackLargestCandle(bt.Indicator):
    """
    Zack Largest Candle indicator.

    Analyzes the ratio of up/down candles relative to the largest candle
    in a lookback period.

    Parameters:
        lookback (int): Lookback period for largest candle (default: 10).
        period (int): Averaging period (default: 10).
        smoothingPeriod (int): Smoothing period (default: 2).
        movav: Moving average type (default: Exponential).

    Lines:
        up: Upward candle strength.
        down: Downward candle strength.

    Example:
        zlc = ZackLargestCandle(data, lookback=10, period=10, smoothingPeriod=2)
    """
    lines = ('up', 'down')
    params = (
        ('lookback', 10),
        ('period', 10),
        ('smoothingPeriod', 2),
        ('movav', btind.MovAv.Exponential),
    )

    def __init__(self):
        upCandles = btind.If(self.data.close > self.data.open, (self.data.close - self.data.open) / self.data.open, 0)
        downCandles = btind.If(self.data.close < self.data.open, (self.data.open - self.data.close) / self.data.open, 0)

        hUpCandle = btind.Highest(upCandles, period=self.p.lookback)
        hDownCandle = btind.Highest(downCandles, period=self.p.lookback)

        up1 = self.p.movav(upCandles, period=self.p.period) / btind.If(hDownCandle == 0, 0.01, hDownCandle)
        down1 = self.p.movav(downCandles, period=self.p.period) / btind.If(hUpCandle == 0, 0.01, hUpCandle)

        self.lines.up = self.p.movav(up1, period=self.p.smoothingPeriod)
        self.lines.down = self.p.movav(down1, period=self.p.smoothingPeriod)

class DidiIndex(bt.Indicator):
    """
    Didi Index indicator.

    Measures trend strength using three moving averages of different periods.

    Parameters:
        short (int): Short moving average period (default: 3).
        mid (int): Medium moving average period (default: 8).
        long (int): Long moving average period (default: 30).
        movav: Moving average type (default: Exponential).

    Lines:
        up: Upward trend (short MA - mid MA).
        down: Downward trend (long MA - mid MA).

    Example:
        didi = DidiIndex(data, short=3, mid=8, long=30)
    """
    lines = ('up', 'down')
    params = (
        ('short', 3),
        ('mid', 8),
        ('long', 30),
        ('movav', btind.MovAv.Exponential),
    )

    def __init__(self):
        self.lines.up = self.p.movav(self.data.close, period=self.p.short) - self.p.movav(self.data.close, period=self.p.mid)
        self.lines.down = self.p.movav(self.data.close, period=self.p.long) - self.p.movav(self.data.close, period=self.p.mid)

class MADiff(bt.Indicator):
    """
    Moving Average Difference indicator.

    Calculates the difference between short and long moving averages.

    Parameters:
        short (int): Short moving average period (default: 3).
        long (int): Long moving average period (default: 8).
        movav: Moving average type (default: Exponential).

    Lines:
        sig: Signal value (short MA - long MA).

    Example:
        mad = MADiff(data, short=3, long=8)
    """
    lines = ('sig',)
    params = (
        ('short', 3),
        ('long', 8),
        ('movav', btind.MovAv.Exponential),
    )

    def __init__(self):
        self.lines.sig = self.p.movav(self.data.close, period=self.p.short) - self.p.movav(self.data.close, period=self.p.long)

class SchaffTrend(bt.Indicator):
    """
    Schaff Trend Cycle indicator.

    A momentum oscillator that combines MACD with stochastic calculation.

    Parameters:
        fastPeriod (int): Fast MACD period (default: 23).
        slowPeriod (int): Slow MACD period (default: 50).
        kPeriod (int): Stochastic K period (default: 10).
        dPeriod (int): Stochastic D period (default: 3).
        movav: Moving average type (default: Exponential).

    Lines:
        trend: Schaff Trend Cycle value.

    Example:
        stc = SchaffTrend(data, fastPeriod=23, slowPeriod=50, kPeriod=10, dPeriod=3)
    """
    lines = ('trend',)
    params = (
        ('fastPeriod', 23),
        ('slowPeriod', 50),
        ('kPeriod', 10),
        ('dPeriod', 3),
        ('movav', btind.MovAv.Exponential)
    )

    def __init__(self):
        macd = self.p.movav(self.data, period=self.p.fastPeriod) - self.p.movav(self.data, period=self.p.slowPeriod)
        high = btind.Highest(self.data, period=self.p.kPeriod)
        low = btind.Lowest(self.data, period=self.p.kPeriod)
        fastk1= btind.If(high-low > 0, (self.data(0)-low) / (high-low) * 100, 0)
        fastd1 = self.p.movav(fastk1, period=self.p.dPeriod)

        high2 = btind.Highest(fastd1, period=self.p.kPeriod)
        low2 = btind.Lowest(fastd1, period=self.p.kPeriod)
        fastk2 = btind.If(high2-low2 > 0, (fastd1(0)-low2) / (high2-low2) * 100, 0)
        self.lines.trend = self.p.movav(fastk2, period=self.p.dPeriod)

class Effort(bt.Indicator):
    """
    Effort indicator.

    Measures the efficiency of price movement relative to volume.

    Parameters:
        period (int): Calculation period (default: 14).

    Lines:
        effort: Effort value (price change / max volume).

    Example:
        effort = Effort(data, period=14)
    """
    lines = ('effort',)
    params = (
        ('period', 14),
    )

    def __init__(self):
        price = btind.MovingAverageSimple(self.data, period=self.p.period)
        roc = 100 * (price(0) / price(-self.p.period)-1)
        maxvol = btind.Highest(self.data.volume, period=self.p.period)
        self.lines.effort = roc / maxvol

class VolumeWeightedAveragePrice(bt.Indicator):
    """
    Volume Weighted Average Price (VWAP) indicator.

    Calculates the average price weighted by volume over a period.

    Parameters:
        period (int): Calculation period (default: 30).

    Lines:
        VWAP: Volume Weighted Average Price value.

    Example:
        vwap = VolumeWeightedAveragePrice(data, period=30)
        # or
        vwap = VWAP(data, period=30)
    """
    plotinfo = dict(subplot=False)
    params = (('period', 30), )

    alias = ('VWAP', 'VolumeWeightedAveragePrice',)
    lines = ('VWAP',)
    plotlines = dict(VWAP=dict(alpha=0.50, linestyle='-.', linewidth=2.0))

    def __init__(self):
        # Before super to ensure mixins (right-hand side in subclassing)
        # can see the assignment operation and operate on the line
        cumvol = bt.ind.SumN(self.data.volume, period = self.p.period)
        typprice = ((self.data.close + self.data.high + self.data.low)/3) * self.data.volume
        cumtypprice = bt.ind.SumN(typprice, period=self.p.period)
        self.lines[0] = cumtypprice / cumvol

        super(VolumeWeightedAveragePrice, self).__init__()


class VolatilitySwitch(bt.Indicator):
    """
    Volatility Switch indicator.

    Measures volatility regime by comparing current volatility to historical
    volatility over a period.

    Parameters:
        period (int): Period for volatility calculation (default: 21).
        smoothPeriod (int): Smoothing period for price (default: 2).

    Lines:
        switch: Volatility switch value (0-1, higher means higher volatility regime).
        vol: Volatility value.

    Example:
        vs = VolatilitySwitch(data, period=21, smoothPeriod=2)
    """
    lines = ('switch', 'vol')
    params = (
        ('period', 21),
        ('smoothPeriod', 2)
    )

    def __init__(self):
        smoothFactor = btind.MovingAverageSimple(self.data, period=self.p.smoothPeriod)
        dailyReturn = (self.data(0)-self.data(-1)) / smoothFactor
        self.lines.vol = btind.StandardDeviation(dailyReturn, period=self.p.period)

    def next(self):
        res = 0
        for i in range(self.p.period):
            if self.lines.vol[0] >= self.lines.vol[-i]:
                res += 1

        self.lines.switch[0] = res / self.p.period

class VolatilitySwitchMod(bt.Indicator):
    """
    Volatility Switch Modified indicator.

    Modified version of VolatilitySwitch using smoothed daily returns.

    Parameters:
        period (int): Period for volatility calculation (default: 21).
        dailyperiod (int): Period for daily return smoothing (default: 10).
        movav: Moving average type (default: Smoothed).

    Lines:
        switch: Volatility switch value (0-1).
        vol: Volatility value.

    Example:
        vsm = VolatilitySwitchMod(data, period=21, dailyperiod=10)
    """
    lines = ('switch', 'vol')
    params = (
        ('period', 21),
        ('dailyperiod', 10),
        ('movav', btind.MovAv.Smoothed)
    )

    def __init__(self):
        dailyReturn = (self.data(0)-self.data(-1)) / ((self.data(0)+self.data(-1)) / 2)
        ma = self.p.movav(dailyReturn, period=self.p.dailyperiod)
        self.lines.vol = btind.StandardDeviation(ma, period=self.p.period)

    def next(self):
        res = 0
        for i in range(self.p.period):
            if self.lines.vol[0] >= self.lines.vol[-i]:
                res += 1

        self.lines.switch[0] = res / self.p.period

class PercentChange(bt.Indicator):
    """
    Percent Change indicator.

    Calculates the percentage change in price over a period.

    Parameters:
        period (int): Period for change calculation (default: 14).

    Lines:
        change: Percentage change value.

    Example:
        pc = PercentChange(data, period=14)
    """
    lines = ('change',)
    params = (
        ('period', 14),
    )

    def __init__(self):
        self.lines.change = 100 * (self.data(0) / self.data(-self.p.period) - 1)

class ChaikinVolatility(bt.Indicator):
    """
    Chaikin Volatility indicator.

    Measures volatility using the rate of change of the exponential moving average
    of the difference between high and low prices.

    Parameters:
        rocperiod (int): Rate of change period (default: 10).
        period (int): Moving average period (default: 10).

    Lines:
        cv: Chaikin Volatility value.

    Example:
        cv = ChaikinVolatility(data, rocperiod=10, period=10)
    """
    lines = ('cv',)
    params = (
        ('rocperiod', 10),
        ('period', 10)
    )

    def __init__(self):
        diff = self.data.high(0) - self.data.low(0)
        avg = btind.MovingAverageSimple(diff, period=self.p.period)

        self.lines.cv = btind.If(avg(-self.p.rocperiod) == 0, 100, (avg(0) - avg(-self.p.rocperiod)) / avg(-self.p.rocperiod) * 100)

class HeikenAshiDiff(bt.Indicator):
    """
    Heiken Ashi Difference indicator.

    Calculates the difference between Heiken Ashi close and open prices,
    smoothed with a moving average.

    Parameters:
        period (int): Smoothing period (default: 5).

    Lines:
        diff: Smoothed difference between HA close and open.
        ha_open: Heiken Ashi open price.
        ha_close: Heiken Ashi close price.

    Example:
        had = HeikenAshiDiff(data, period=5)
    """
    lines = ('diff','ha_open','ha_close')
    params = (
        ('period', 5),
    )
    _nextforce = True

    def __init__(self):
        self.l.ha_close = ha_close = (self.data.open + self.data.close + self.data.high + self.data.low) / 4.0
        self.l.ha_open = ha_open = (self.l.ha_open(-1) + ha_close(-1)) / 2.0
        diff = ha_close(0) - ha_open(0)
        self.lines.diff = btind.MovingAverageSimple(diff, period=5)

    def prenext(self):
        # seed recursive value
        self.lines.ha_open[0] = (self.data.open[0] + self.data.close[0]) / 2.0

class DMIStoch(bt.Indicator):
    """
    DMI Stochastic indicator.

    Combines Directional Movement Index (DMI) with stochastic calculation.

    Parameters:
        period (int): DMI period (default: 10).
        sumperiod (int): Stochastic sum period (default: 3).

    Lines:
        stoch: DMI Stochastic value.

    Example:
        dmi_stoch = DMIStoch(data, period=10, sumperiod=3)
    """
    lines = ('stoch',)
    params = (
        ('period', 10),
        ('sumperiod', 3)
    )

    def __init__(self):
        dmi = btind.DirectionalMovementIndex(self.data, period=self.p.period)
        osc = dmi.plusDI - dmi.minusDI
        hh = btind.Highest(osc, period=self.p.sumperiod)
        ll = btind.Lowest(osc, period=self.p.sumperiod)

        self.lines.stoch = btind.SumN(osc - ll, period=self.p.sumperiod) / btind.SumN(hh - ll, period=self.p.sumperiod) * 100


class DMIIndicator(bt.Indicator):
    """
    DMI (Directional Movement Index) composite indicator.

    Wraps Backtrader's DirectionalMovementIndex and exposes +DI, -DI, ADX, and ADXR.
    ADXR is computed as a simple average of the current and previous ADX values.

    Parameters:
        period (int): DMI calculation period (default: 14).

    Lines:
        plus_di: +DI line (rising strength)
        minus_di: -DI line (falling strength)
        adx: Trend strength line
        adxr: Smoothed trend strength (average of current and previous ADX)
    """

    lines = ("plus_di", "minus_di", "adx", "adxr")
    params = (("period", 14),)

    def __init__(self):
        super().__init__()

        base = btind.DirectionalMovementIndex(self.data, period=self.p.period)

        # Map underlying lines
        self.lines.plus_di = base.plusDI
        self.lines.minus_di = base.minusDI
        self.lines.adx = base.adx

        # ADXR: average of current and previous ADX (simple smoothing)
        self.lines.adxr = btind.MovAv.Simple(base.adx, period=2)


class ZackVolatility(bt.Indicator):
    """
    Zack Volatility indicator.

    Calculates volatility using standard deviation and its moving average.

    Parameters:
        period (int): Standard deviation period (default: 21).
        movav: Moving average type for smoothing (default: Exponential).

    Lines:
        vol: Volatility (standard deviation) value.
        ma: Moving average of volatility.

    Example:
        zv = ZackVolatility(data, period=21)
    """
    lines = ('vol', 'ma')
    params = (
        ('period', 21),
        ('movav', btind.MovAv.Exponential)
    )

    def __init__(self):
        self.lines.vol = btind.StandardDeviation(self.data, period=self.p.period)
        self.lines.ma = self.p.movav(self.lines.vol, period=self.p.period*2)

class MoneyFlowIndex(bt.Indicator):
    """
    Money Flow Index (MFI) indicator.

    A momentum indicator that uses both price and volume to identify
    overbought or oversold conditions.

    Parameters:
        period (int): Calculation period (default: 14).
        movav: Moving average type (default: Smoothed).

    Lines:
        mfi: Money Flow Index value (0-100).

    Example:
        mfi = MoneyFlowIndex(data, period=14)
    """
    lines = ('mfi',)
    params = (
        ('period', 14),
        ('movav', btind.MovAv.Smoothed)
    )

    def __init__(self):
        price = (self.data.high + self.data.low + self.data.close) / 3
        pricema = self.p.movav(price, period=self.p.period)
        posmf = btind.If(price(0) > price(-1), price*self.data.volume, 0)
        negmf = btind.If(price(0) < price(-1), price*self.data.volume, 0)
        mfi = bt.DivByZero(btind.SumN(posmf, period=self.p.period), btind.SumN(negmf, period=self.p.period))
        self.lines.mfi = 100 - 100 / (1 + mfi)

class RelativeStrengthIndex(bt.Indicator):
    """
    Relative Strength Index (RSI) indicator.

    A momentum oscillator that measures the speed and magnitude of price changes.
    Uses typical price (high + low + close) / 3.

    Parameters:
        period (int): Calculation period (default: 14).
        movav: Moving average type (default: Smoothed).

    Lines:
        rsi: RSI value (0-100).

    Example:
        rsi = RelativeStrengthIndex(data, period=14)
    """
    lines = ('rsi',)
    params = (
        ('period', 14),
        ('movav', btind.MovAv.Smoothed)
    )

    def __init__(self):
        price = (self.data.high + self.data.low + self.data.close) / 3
        pricema = self.p.movav(price, period=self.p.period)
        posrs = btind.If(price(0) > price(-1), price, 0)
        negrs = btind.If(price(0) < price(-1), price, 0)
        rsi = bt.DivByZero(btind.SumN(posrs, period=self.p.period), btind.SumN(negrs, period=self.p.period))
        self.lines.rsi = 100 - 100 / (1 + rsi)

class PolarizedFractalEfficiency(bt.Indicator):
    """
    Polarized Fractal Efficiency (PFE) indicator.

    Measures the efficiency of price movement using fractal geometry concepts.

    Parameters:
        slowPeriod (int): Slow rate of change period (default: 10).
        fastPeriod (int): Fast rate of change period (default: 1).
        scaleFactor (float): Scale factor for calculation (default: 1).
        period (int): Smoothing period (default: 5).
        movav: Moving average type (default: Smoothed).

    Lines:
        pfe: Polarized Fractal Efficiency value.

    Example:
        pfe = PolarizedFractalEfficiency(data, slowPeriod=10, fastPeriod=1, period=5)
    """
    lines = ('pfe',)
    params = (
        ('slowPeriod', 10),
        ('fastPeriod', 1),
        ('scaleFactor', 1),
        ('period', 5),
        ('movav', btind.MovAv.Smoothed)
    )

    def __init__(self):
        longRoc = 100 * (self.data(0) / self.data(-self.p.slowPeriod)-1)
        shortRoc = 100 * (self.data(0) / self.data(-self.p.fastPeriod)-1)
        Z = btind.DivByZero(pow(longRoc(0) * longRoc(0) + 100, 0.5), pow(shortRoc(0) * shortRoc(0), 0.5) + self.p.scaleFactor)
        B = btind.If(self.data(0) > self.data(-self.p.slowPeriod), 100*Z(0), -100*Z(0))
        self.lines.pfe = self.p.movav(B, period=self.p.period)

class Juice(bt.Indicator):
    """
    Juice indicator.

    Measures the strength of price movement relative to a volatility threshold.

    Parameters:
        fastPeriod (int): Fast moving average period (default: 6).
        slowPeriod (int): Slow moving average period (default: 14).
        period (int): Smoothing period (default: 5).
        movav: Moving average type (default: Smoothed).
        volatility (float): Volatility threshold (default: 0.02).

    Lines:
        juice: Juice value (absolute momentum minus volatility threshold).

    Example:
        juice = Juice(data, fastPeriod=6, slowPeriod=14, period=5, volatility=0.02)
    """
    lines = ('juice',)
    params = (
        ('fastPeriod', 6),
        ('slowPeriod', 14),
        ('period', 5),
        ('movav', btind.MovAv.Smoothed),
        ('volatility', 0.02)
    )

    def __init__(self):
        val = (self.p.movav(self.data, period=self.p.fastPeriod) - self.p.movav(self.data, period=self.p.slowPeriod)) / self.data(0)
        avg = self.p.movav(val, period=self.p.period)
        self.lines.juice = abs(avg) - self.p.volatility

class VolumeOsc(bt.Indicator):
    """
    Volume Oscillator indicator.

    Measures the difference between fast and slow volume moving averages
    as a percentage of slow volume.

    Parameters:
        fastPeriod (int): Fast volume moving average period (default: 14).
        slowPeriod (int): Slow volume moving average period (default: 28).
        movav: Moving average type (default: Simple).

    Lines:
        osc: Volume oscillator value.

    Example:
        vo = VolumeOsc(data, fastPeriod=14, slowPeriod=28)
    """
    lines = ('osc',)
    params = (
        ('fastPeriod', 14),
        ('slowPeriod', 28),
        ('movav', btind.MovAv.Simple),
    )

    def __init__(self):
        fast = self.p.movav(self.data.volume, period=self.p.fastPeriod)
        slow = self.p.movav(self.data.volume, period=self.p.slowPeriod)
        self.lines.osc = (fast - slow) / slow

class AroonDown(bt.Indicator):
    """
    Aroon Down indicator.

    Measures how long it has been since the lowest low within a period.

    Parameters:
        period (int): Calculation period (default: 14).

    Lines:
        down: Aroon Down value (0-100).

    Example:
        aroon_down = AroonDown(data, period=14)
    """
    lines = ('down',)
    params = (
        ('period', 14),
    )

    def __init__(self):
        lidx = btind.FindFirstIndexLowest(self.data.low, period=self.p.period)
        self.lines.down = (self.p.period - 1 - lidx) * 100 / (self.p.period - 1)#(100.0 / self.p.period) * (self.p.period - llidx)

class MarketMeannessIndicator(bt.Indicator):
    """
    Market Meanness Indicator (MMI).

    Measures whether the market is trending or mean-reverting by analyzing
    price position relative to the median.

    Parameters:
        period (int): Calculation period (default: 200).
        movav: Moving average type for smoothing (default: Hull).

    Lines:
        mmi: Market Meanness Indicator value.

    Example:
        mmi = MarketMeannessIndicator(data, period=200)
    """
    lines = ('mmi',)
    params = (
        ('period', 200),
        ('movav', btind.MovAv.Hull)
    )

    def __init__(self):
        m = (btind.Highest(self.data.high, period=self.p.period)+btind.Lowest(self.data.low, period=self.p.period)) / 2
        nhh = btind.If(self.data>m, btind.If(self.data>self.data(-1), 1, 0), 0)
        nll = btind.If(self.data<m, btind.If(self.data<self.data(-1), 1, 0), 0)
        nh = btind.SumN(nhh, period=self.p.period)
        nl = btind.SumN(nll, period=self.p.period)
        mmi = 100*(nl+nh)/(self.p.period-1)

        self.lines.mmi = self.p.movav(mmi, period=self.p.period)

class VPN(bt.Indicator):
    """
    Volume Price Net indicator.

    Measures volume flow in up and down directions based on price movement
    relative to ATR.

    Parameters:
        period (int): Calculation period (default: 20).
        movav: Moving average type (default: Exponential).

    Lines:
        up: Upward volume flow.
        down: Downward volume flow.

    Example:
        vpn = VPN(data, period=20)
    """
    lines = ('up','down')
    params = (
        ('period', 20),
        ('movav', btind.MovAv.Exponential),
    )

    def __init__(self):
        atr = btind.AverageTrueRange(self.data, period=self.p.period)
        price = (self.data.high+self.data.low+self.data.close) / 3
        up = btind.If(price(0) > price(-1)+atr*0.1, self.data.volume(0), 0)
        down = btind.If(price(0) < price(-1)-atr*0.1, self.data.volume(0), 0)

        totalVol = btind.SumN(self.data.volume, period=self.p.period)

        self.lines.up = self.p.movav(up, period=self.p.period) / totalVol
        self.lines.down = self.p.movav(down, period=self.p.period) / totalVol

class ATRP(bt.Indicator):
    """
    Average True Range Percentage difference indicator.

    Calculates the difference between fast and slow ATR percentages.

    Parameters:
        slowperiod (int): Slow ATR percentage period (default: 20).
        fastperiod (int): Fast ATR percentage period (default: 5).
        movav: Moving average type (default: Smoothed).

    Lines:
        diff: Difference between fast and slow ATR percentages.

    Example:
        atrp = ATRP(data, slowperiod=20, fastperiod=5)
    """
    lines = ('diff',)
    params = (
        ('slowperiod', 20),
        ('fastperiod', 5),
        ('movav', btind.MovAv.Smoothed),
    )

    def __init__(self):
        slow = self.p.movav(btind.TrueRange(self.data)/self.data.close, period=self.p.slowperiod)
        fast = self.p.movav(btind.TrueRange(self.data)/self.data.close, period=self.p.fastperiod)
        self.lines.diff = fast - slow
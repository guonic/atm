"""
Technical indicator calculation utilities.

This module provides functions to calculate various technical indicators
using pandas for stable and consistent calculations.
"""

import logging
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def _to_list_with_none(series: pd.Series) -> List[Optional[float]]:
    """Convert pandas Series to list, replacing NaN with None for JSON serialization."""
    return [None if pd.isna(x) else float(x) for x in series.tolist()]


def calculate_macd(
    closes: pd.Series,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> Dict[str, List[Optional[float]]]:
    """
    Calculate MACD (Moving Average Convergence Divergence).
    
    Args:
        closes: Series of closing prices.
        fast_period: Fast EMA period (default: 12).
        slow_period: Slow EMA period (default: 26).
        signal_period: Signal line EMA period (default: 9).
    
    Returns:
        Dictionary with 'macd', 'signal', 'histogram' lists.
    """
    if len(closes) < slow_period:
        return {
            "macd": [None] * len(closes),
            "signal": [None] * len(closes),
            "histogram": [None] * len(closes),
        }
    
    # Calculate EMAs
    fast_ema = closes.ewm(span=fast_period, adjust=False).mean()
    slow_ema = closes.ewm(span=slow_period, adjust=False).mean()
    
    # MACD line
    macd_line = fast_ema - slow_ema
    
    # Signal line (EMA of MACD)
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    
    # Histogram
    histogram = macd_line - signal_line
    
    return {
        "macd": _to_list_with_none(macd_line),
        "signal": _to_list_with_none(signal_line),
        "histogram": _to_list_with_none(histogram),
    }


def calculate_rsi(closes: pd.Series, period: int = 14) -> List[Optional[float]]:
    """
    Calculate RSI (Relative Strength Index).
    
    Args:
        closes: Series of closing prices.
        period: RSI period (default: 14).
    
    Returns:
        List of RSI values (0-100).
    """
    if len(closes) < period + 1:
        return [None] * len(closes)
    
    delta = closes.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    
    return _to_list_with_none(rsi.fillna(50))


def calculate_bollinger_bands(
    closes: pd.Series, period: int = 20, num_std: float = 2.0
) -> Dict[str, List[Optional[float]]]:
    """
    Calculate Bollinger Bands.
    
    Args:
        closes: Series of closing prices.
        period: Moving average period (default: 20).
        num_std: Number of standard deviations (default: 2.0).
    
    Returns:
        Dictionary with 'upper', 'middle', 'lower' lists.
    """
    if len(closes) < period:
        return {
            "upper": [None] * len(closes),
            "middle": [None] * len(closes),
            "lower": [None] * len(closes),
        }
    
    sma = closes.rolling(window=period).mean()
    std = closes.rolling(window=period).std()
    
    upper = sma + (std * num_std)
    lower = sma - (std * num_std)
    
    return {
        "upper": _to_list_with_none(upper),
        "middle": _to_list_with_none(sma),
        "lower": _to_list_with_none(lower),
    }


def calculate_atr(
    highs: pd.Series,
    lows: pd.Series,
    closes: pd.Series,
    period: int = 14,
) -> List[Optional[float]]:
    """
    Calculate ATR (Average True Range).
    
    Args:
        highs: Series of high prices.
        lows: Series of low prices.
        closes: Series of closing prices.
        period: ATR period (default: 14).
    
    Returns:
        List of ATR values.
    """
    if len(highs) < period + 1:
        return [None] * len(highs)
    
    high_low = highs - lows
    high_close = np.abs(highs - closes.shift())
    low_close = np.abs(lows - closes.shift())
    
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(window=period).mean()
    
    return _to_list_with_none(atr)


def calculate_sma(closes: pd.Series, period: int) -> List[Optional[float]]:
    """
    Calculate SMA (Simple Moving Average).
    
    Args:
        closes: Series of closing prices.
        period: SMA period.
    
    Returns:
        List of SMA values.
    """
    if len(closes) < period:
        return [None] * len(closes)
    
    sma = closes.rolling(window=period).mean()
    return _to_list_with_none(sma)


def calculate_ema(closes: pd.Series, period: int = 12) -> List[Optional[float]]:
    """
    Calculate EMA (Exponential Moving Average).
    
    Args:
        closes: Series of closing prices.
        period: EMA period (default: 12).
    
    Returns:
        List of EMA values.
    """
    if len(closes) < period:
        return [None] * len(closes)
    
    ema = closes.ewm(span=period, adjust=False).mean()
    return _to_list_with_none(ema)


def calculate_wma(closes: pd.Series, period: int = 14) -> List[Optional[float]]:
    """
    Calculate WMA (Weighted Moving Average).
    
    Args:
        closes: Series of closing prices.
        period: WMA period (default: 14).
    
    Returns:
        List of WMA values.
    """
    if len(closes) < period:
        return [None] * len(closes)
    
    weights = np.arange(1, period + 1)
    wma = closes.rolling(window=period).apply(
        lambda x: np.dot(x, weights) / weights.sum(), raw=True
    )
    return _to_list_with_none(wma)


def calculate_kdj(
    highs: pd.Series,
    lows: pd.Series,
    closes: pd.Series,
    period: int = 9,
    k_period: int = 3,
    d_period: int = 3,
) -> Dict[str, List[Optional[float]]]:
    """
    Calculate KDJ (Stochastic Oscillator).
    
    Args:
        highs: Series of high prices.
        lows: Series of low prices.
        closes: Series of closing prices.
        period: RSV period (default: 9).
        k_period: K smoothing period (default: 3).
        d_period: D smoothing period (default: 3).
    
    Returns:
        Dictionary with 'k', 'd', 'j' lists.
    """
    if len(highs) < period + d_period:
        return {
            "k": [None] * len(highs),
            "d": [None] * len(highs),
            "j": [None] * len(highs),
        }
    
    # Calculate RSV (Raw Stochastic Value)
    lowest_low = lows.rolling(window=period).min()
    highest_high = highs.rolling(window=period).max()
    # Avoid division by zero
    denominator = highest_high - lowest_low
    rsv = 100 * (closes - lowest_low) / denominator.replace(0, np.nan)
    rsv = rsv.fillna(50)  # Fill NaN with 50 (neutral value)
    
    # Calculate K (smoothed RSV)
    k = rsv.ewm(alpha=1/k_period, adjust=False).mean()
    
    # Calculate D (smoothed K)
    d = k.ewm(alpha=1/d_period, adjust=False).mean()
    
    # Calculate J = 3K - 2D
    j = 3 * k - 2 * d
    
    return {
        "k": _to_list_with_none(k),
        "d": _to_list_with_none(d),
        "j": _to_list_with_none(j),
    }


def calculate_cci(
    highs: pd.Series,
    lows: pd.Series,
    closes: pd.Series,
    period: int = 14,
) -> List[Optional[float]]:
    """
    Calculate CCI (Commodity Channel Index).
    
    Args:
        highs: Series of high prices.
        lows: Series of low prices.
        closes: Series of closing prices.
        period: CCI period (default: 14).
    
    Returns:
        List of CCI values.
    """
    if len(highs) < period:
        return [None] * len(highs)
    
    # Typical Price
    tp = (highs + lows + closes) / 3
    
    # Simple Moving Average of TP
    sma_tp = tp.rolling(window=period).mean()
    
    # Mean Deviation
    md = tp.rolling(window=period).apply(
        lambda x: np.mean(np.abs(x - x.mean())), raw=True
    )
    
    # CCI = (TP - SMA_TP) / (0.015 * MD)
    # Avoid division by zero
    denominator = 0.015 * md
    cci = (tp - sma_tp) / denominator.replace(0, np.nan)
    cci = cci.fillna(0)  # Fill NaN with neutral value (0)
    
    return _to_list_with_none(cci)


def calculate_wr(
    highs: pd.Series,
    lows: pd.Series,
    closes: pd.Series,
    period: int = 14,
) -> List[Optional[float]]:
    """
    Calculate WR (Williams %R).
    
    Args:
        highs: Series of high prices.
        lows: Series of low prices.
        closes: Series of closing prices.
        period: WR period (default: 14).
    
    Returns:
        List of WR values (ranging from -100 to 0).
    """
    if len(highs) < period:
        return [None] * len(highs)
    
    highest_high = highs.rolling(window=period).max()
    lowest_low = lows.rolling(window=period).min()
    
    # Avoid division by zero
    denominator = highest_high - lowest_low
    wr = -100 * (highest_high - closes) / denominator.replace(0, np.nan)
    wr = wr.fillna(-50)  # Fill NaN with neutral value (-50)
    
    return _to_list_with_none(wr)


def calculate_obv(closes: pd.Series, volumes: pd.Series) -> List[Optional[float]]:
    """
    Calculate OBV (On-Balance Volume).
    
    Args:
        closes: Series of closing prices.
        volumes: Series of volumes.
    
    Returns:
        List of OBV values.
    """
    if len(closes) < 2:
        return [None] * len(closes)
    
    price_change = closes.diff()
    obv = (volumes * np.sign(price_change)).fillna(0).cumsum()
    
    return _to_list_with_none(obv)


def calculate_dmi(
    highs: pd.Series,
    lows: pd.Series,
    closes: pd.Series,
    period: int = 14,
) -> Dict[str, List[Optional[float]]]:
    """
    Calculate DMI (Directional Movement Index).
    
    Args:
        highs: Series of high prices.
        lows: Series of low prices.
        closes: Series of closing prices.
        period: DMI period (default: 14).
    
    Returns:
        Dictionary with 'pdi', 'mdi', 'adx', 'adxr' lists.
    """
    if len(highs) < period + 1:
        return {
            "pdi": [None] * len(highs),
            "mdi": [None] * len(highs),
            "adx": [None] * len(highs),
            "adxr": [None] * len(highs),
        }
    
    # Calculate True Range
    high_low = highs - lows
    high_close = np.abs(highs - closes.shift())
    low_close = np.abs(lows - closes.shift())
    
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    
    # Calculate Directional Movement
    up_move = highs - highs.shift()
    down_move = lows.shift() - lows
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    plus_dm = pd.Series(plus_dm, index=highs.index)
    minus_dm = pd.Series(minus_dm, index=highs.index)
    
    # Smooth TR and DM
    atr = tr.rolling(window=period).mean()
    # Avoid division by zero in ATR
    atr = atr.replace(0, np.nan)
    plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
    plus_di = plus_di.fillna(0)
    minus_di = minus_di.fillna(0)
    
    # Calculate DX
    # Avoid division by zero
    denominator = plus_di + minus_di
    dx = 100 * np.abs(plus_di - minus_di) / denominator.replace(0, np.nan)
    dx = dx.fillna(0)  # Fill NaN with 0 (neutral value)
    
    # Calculate ADX (smoothed DX)
    adx = dx.rolling(window=period).mean()
    
    # Calculate ADXR (ADX smoothed)
    adxr = adx.rolling(window=period).mean()
    
    return {
        "pdi": _to_list_with_none(plus_di),
        "mdi": _to_list_with_none(minus_di),
        "adx": _to_list_with_none(adx),
        "adxr": _to_list_with_none(adxr),
    }


def calculate_envelope(
    closes: pd.Series, period: int = 20, percent: float = 0.05
) -> Dict[str, List[Optional[float]]]:
    """
    Calculate Envelope (Price Channel).
    
    Args:
        closes: Series of closing prices.
        period: Moving average period (default: 20).
        percent: Percentage offset (default: 0.05 = 5%).
    
    Returns:
        Dictionary with 'upper', 'lower' lists.
    """
    if len(closes) < period:
        return {
            "upper": [None] * len(closes),
            "lower": [None] * len(closes),
        }
    
    sma = closes.rolling(window=period).mean()
    upper = sma * (1 + percent)
    lower = sma * (1 - percent)
    
    return {
        "upper": _to_list_with_none(upper),
        "lower": _to_list_with_none(lower),
    }


def calculate_bbw(closes: pd.Series, period: int = 20) -> List[Optional[float]]:
    """
    Calculate BBW (Bollinger Band Width).
    
    Args:
        closes: Series of closing prices.
        period: Moving average period (default: 20).
    
    Returns:
        List of BBW values.
    """
    if len(closes) < period:
        return [None] * len(closes)
    
    sma = closes.rolling(window=period).mean()
    std = closes.rolling(window=period).std()
    
    # BBW = (Upper Band - Lower Band) / Middle Band
    # Upper = SMA + 2*STD, Lower = SMA - 2*STD
    upper = sma + (std * 2)
    lower = sma - (std * 2)
    
    # Avoid division by zero
    bbw = (upper - lower) / sma.replace(0, np.nan) * 100
    bbw = bbw.fillna(0)
    
    return _to_list_with_none(bbw)


def calculate_vwap(
    highs: pd.Series,
    lows: pd.Series,
    closes: pd.Series,
    volumes: pd.Series,
) -> List[Optional[float]]:
    """
    Calculate VWAP (Volume Weighted Average Price).
    
    Args:
        highs: Series of high prices.
        lows: Series of low prices.
        closes: Series of closing prices.
        volumes: Series of volumes.
    
    Returns:
        List of VWAP values.
    """
    if len(closes) < 1:
        return [None] * len(closes)
    
    # Typical Price
    tp = (highs + lows + closes) / 3
    
    # Cumulative (TP * Volume) / Cumulative Volume
    cumulative_tpv = (tp * volumes).cumsum()
    cumulative_volume = volumes.cumsum()
    
    # Avoid division by zero
    vwap = cumulative_tpv / cumulative_volume.replace(0, np.nan)
    vwap = vwap.fillna(tp)  # Fill NaN with typical price
    
    return _to_list_with_none(vwap)


def calculate_indicators(
    kline_data: List[Dict],
    indicators: Optional[Dict[str, bool]] = None,
) -> Dict[str, any]:
    """
    Calculate all requested technical indicators from K-line data.
    
    Args:
        kline_data: List of K-line dictionaries with 'date', 'open', 'high', 'low', 'close', 'volume'.
        indicators: Dictionary indicating which indicators to calculate.
                    Supports: 'macd', 'rsi', 'bollinger', 'atr', 'ma5', 'ma10', 'ma20', 'ma30',
                    'ma60', 'ma120', 'ema', 'wma', 'kdj', 'cci', 'wr', 'obv', 'dmi', 'envelope',
                    'bbw', 'vwap'.
    
    Returns:
        Dictionary with calculated indicator data.
    """
    if not kline_data:
        return {}
    
    if indicators is None:
        indicators = {}
    
    # Convert to DataFrame
    df = pd.DataFrame(kline_data)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    
    result = {}
    
    # Trend indicators (overlay on K-line)
    if indicators.get("ma5", False):
        result["ma5"] = calculate_sma(df["close"], 5)
    
    if indicators.get("ma10", False):
        result["ma10"] = calculate_sma(df["close"], 10)
    
    if indicators.get("ma20", False):
        result["ma20"] = calculate_sma(df["close"], 20)
    
    if indicators.get("ma30", False):
        result["ma30"] = calculate_sma(df["close"], 30)
    
    if indicators.get("ma60", False):
        result["ma60"] = calculate_sma(df["close"], 60)
    
    if indicators.get("ma120", False):
        result["ma120"] = calculate_sma(df["close"], 120)
    
    if indicators.get("ema", False):
        result["ema"] = calculate_ema(df["close"], 12)
    
    if indicators.get("wma", False):
        result["wma"] = calculate_wma(df["close"], 14)
    
    # Oscillator indicators (separate subplots)
    if indicators.get("rsi", False):
        result["rsi"] = calculate_rsi(df["close"])
    
    if indicators.get("kdj", False):
        result["kdj"] = calculate_kdj(df["high"], df["low"], df["close"])
    
    if indicators.get("cci", False):
        result["cci"] = calculate_cci(df["high"], df["low"], df["close"])
    
    if indicators.get("wr", False):
        result["wr"] = calculate_wr(df["high"], df["low"], df["close"])
    
    if indicators.get("obv", False):
        result["obv"] = calculate_obv(df["close"], df["volume"])
    
    # Trend + Oscillator indicators
    if indicators.get("macd", False):
        result["macd"] = calculate_macd(df["close"])
    
    if indicators.get("dmi", False):
        result["dmi"] = calculate_dmi(df["high"], df["low"], df["close"])
    
    # Channel indicators (overlay on K-line)
    if indicators.get("bollinger", False):
        result["bollinger"] = calculate_bollinger_bands(df["close"])
    
    if indicators.get("envelope", False):
        result["envelope"] = calculate_envelope(df["close"])
    
    # Volatility indicators
    if indicators.get("atr", False):
        result["atr"] = calculate_atr(df["high"], df["low"], df["close"])
    
    if indicators.get("bbw", False):
        result["bbw"] = calculate_bbw(df["close"])
    
    # Volume indicators
    if indicators.get("vwap", False):
        result["vwap"] = calculate_vwap(df["high"], df["low"], df["close"], df["volume"])
    
    return result

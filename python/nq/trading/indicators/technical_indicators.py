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
    
    # Signal line (EMA of MACD line)
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
        List of RSI values.
    """
    if len(closes) < period + 1:
        return [None] * len(closes)
    
    delta = closes.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    
    return _to_list_with_none(rsi)


def calculate_bollinger_bands(
    closes: pd.Series,
    period: int = 20,
    std_dev: float = 2.0,
) -> Dict[str, List[Optional[float]]]:
    """
    Calculate Bollinger Bands.
    
    Args:
        closes: Series of closing prices.
        period: Moving average period (default: 20).
        std_dev: Standard deviation multiplier (default: 2.0).
    
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
    
    upper = sma + (std * std_dev)
    lower = sma - (std * std_dev)
    
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
    if len(closes) < period + 1:
        return [None] * len(closes)
    
    high_low = highs - lows
    high_close = (highs - closes.shift()).abs()
    low_close = (lows - closes.shift()).abs()
    
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(window=period).mean()
    
    return _to_list_with_none(atr)


def calculate_sma(closes: pd.Series, period: int) -> List[Optional[float]]:
    """
    Calculate SMA (Simple Moving Average).
    
    Args:
        closes: Series of closing prices.
        period: Moving average period.
    
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


def calculate_wma(closes: pd.Series, period: int = 12) -> List[Optional[float]]:
    """
    Calculate WMA (Weighted Moving Average).
    
    Args:
        closes: Series of closing prices.
        period: WMA period (default: 12).
    
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
        k_period: K line smoothing period (default: 3).
        d_period: D line smoothing period (default: 3).
    
    Returns:
        Dictionary with 'k', 'd', 'j' lists.
    """
    if len(closes) < period:
        return {
            "k": [None] * len(closes),
            "d": [None] * len(closes),
            "j": [None] * len(closes),
        }
    
    lowest_low = lows.rolling(window=period).min()
    highest_high = highs.rolling(window=period).max()
    
    rsv = 100 * (closes - lowest_low) / (highest_high - lowest_low).replace(0, np.nan)
    rsv = rsv.fillna(50)  # Fill initial NaN with 50
    
    k = rsv.ewm(alpha=1/k_period, adjust=False).mean()
    d = k.ewm(alpha=1/d_period, adjust=False).mean()
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
    period: int = 20,
) -> List[Optional[float]]:
    """
    Calculate CCI (Commodity Channel Index).
    
    Args:
        highs: Series of high prices.
        lows: Series of low prices.
        closes: Series of closing prices.
        period: CCI period (default: 20).
    
    Returns:
        List of CCI values.
    """
    if len(closes) < period:
        return [None] * len(closes)
    
    tp = (highs + lows + closes) / 3
    sma_tp = tp.rolling(window=period).mean()
    md = tp.rolling(window=period).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    
    denominator = 0.015 * md
    cci = (tp - sma_tp) / denominator.replace(0, np.nan)
    
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
        List of WR values.
    """
    if len(closes) < period:
        return [None] * len(closes)
    
    highest_high = highs.rolling(window=period).max()
    lowest_low = lows.rolling(window=period).min()
    
    denominator = highest_high - lowest_low
    wr = -100 * (highest_high - closes) / denominator.replace(0, np.nan)
    
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
    if len(closes) != len(volumes):
        return [None] * len(closes)
    
    obv = (volumes * np.sign(closes.diff())).fillna(0).cumsum()
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
    if len(closes) < period * 2:
        return {
            "pdi": [None] * len(closes),
            "mdi": [None] * len(closes),
            "adx": [None] * len(closes),
            "adxr": [None] * len(closes),
        }
    
    # Calculate True Range
    high_low = highs - lows
    high_close = (highs - closes.shift()).abs()
    low_close = (lows - closes.shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    
    # Calculate ATR
    atr = tr.rolling(window=period).mean()
    atr = atr.replace(0, np.nan)
    
    # Calculate Directional Movement
    plus_dm = highs.diff()
    minus_dm = -lows.diff()
    
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    
    plus_dm[(plus_dm < minus_dm)] = 0
    minus_dm[(minus_dm < plus_dm)] = 0
    
    # Calculate DI
    plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
    
    plus_di = plus_di.fillna(0)
    minus_di = minus_di.fillna(0)
    
    # Calculate DX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)
    
    # Calculate ADX
    adx = dx.rolling(window=period).mean()
    
    # Calculate ADXR
    adxr = (adx + adx.shift(period)) / 2
    
    return {
        "pdi": _to_list_with_none(plus_di),
        "mdi": _to_list_with_none(minus_di),
        "adx": _to_list_with_none(adx),
        "adxr": _to_list_with_none(adxr),
    }


def calculate_envelope(
    closes: pd.Series,
    period: int = 20,
    percent: float = 2.5,
) -> Dict[str, List[Optional[float]]]:
    """
    Calculate Envelope (Price Channel).
    
    Args:
        closes: Series of closing prices.
        period: Moving average period (default: 20).
        percent: Envelope percentage (default: 2.5).
    
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
    upper = sma * (1 + percent / 100)
    lower = sma * (1 - percent / 100)
    
    return {
        "upper": _to_list_with_none(upper),
        "middle": _to_list_with_none(sma),
        "lower": _to_list_with_none(lower),
    }


def calculate_bbw(
    closes: pd.Series,
    period: int = 20,
    std_dev: float = 2.0,
) -> List[Optional[float]]:
    """
    Calculate BBW (Bollinger Band Width).
    
    Args:
        closes: Series of closing prices.
        period: Moving average period (default: 20).
        std_dev: Standard deviation multiplier (default: 2.0).
    
    Returns:
        List of BBW values.
    """
    if len(closes) < period:
        return [None] * len(closes)
    
    sma = closes.rolling(window=period).mean()
    std = closes.rolling(window=period).std()
    
    upper = sma + (std * std_dev)
    lower = sma - (std * std_dev)
    
    bbw = (upper - lower) / sma.replace(0, np.nan)
    
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
    if len(closes) != len(volumes):
        return [None] * len(closes)
    
    typical_price = (highs + lows + closes) / 3
    tpv = typical_price * volumes
    
    cumulative_tpv = tpv.cumsum()
    cumulative_volume = volumes.cumsum()
    
    vwap = cumulative_tpv / cumulative_volume.replace(0, np.nan)
    
    return _to_list_with_none(vwap)


def calculate_indicators(
    kline_data: List[Dict[str, any]],
    indicators: Dict[str, bool],
) -> Dict[str, any]:
    """
    Calculate multiple technical indicators from K-line data.
    
    Args:
        kline_data: List of K-line dictionaries with 'date', 'open', 'high', 'low', 'close', 'volume'.
        indicators: Dictionary of indicator names to boolean flags.
    
    Returns:
        Dictionary of calculated indicator data.
    """
    if not kline_data:
        return {}
    
    # Convert to pandas DataFrame
    df = pd.DataFrame(kline_data)
    
    # Extract series
    opens = pd.Series(df['open'].values)
    highs = pd.Series(df['high'].values)
    lows = pd.Series(df['low'].values)
    closes = pd.Series(df['close'].values)
    volumes = pd.Series(df['volume'].values)
    
    result = {}
    
    # Moving Averages
    if indicators.get('ma5'):
        result['ma5'] = calculate_sma(closes, 5)
    if indicators.get('ma10'):
        result['ma10'] = calculate_sma(closes, 10)
    if indicators.get('ma20'):
        result['ma20'] = calculate_sma(closes, 20)
    if indicators.get('ma30'):
        result['ma30'] = calculate_sma(closes, 30)
    if indicators.get('ma60'):
        result['ma60'] = calculate_sma(closes, 60)
    if indicators.get('ma120'):
        result['ma120'] = calculate_sma(closes, 120)
    
    # EMA
    if indicators.get('ema'):
        result['ema'] = calculate_ema(closes)
    
    # WMA
    if indicators.get('wma'):
        result['wma'] = calculate_wma(closes)
    
    # RSI
    if indicators.get('rsi'):
        result['rsi'] = calculate_rsi(closes)
    
    # KDJ
    if indicators.get('kdj'):
        result['kdj'] = calculate_kdj(highs, lows, closes)
    
    # CCI
    if indicators.get('cci'):
        result['cci'] = calculate_cci(highs, lows, closes)
    
    # WR
    if indicators.get('wr'):
        result['wr'] = calculate_wr(highs, lows, closes)
    
    # OBV
    if indicators.get('obv'):
        result['obv'] = calculate_obv(closes, volumes)
    
    # MACD
    if indicators.get('macd'):
        result['macd'] = calculate_macd(closes)
    
    # DMI
    if indicators.get('dmi'):
        result['dmi'] = calculate_dmi(highs, lows, closes)
    
    # Bollinger Bands
    if indicators.get('bollinger'):
        result['bollinger'] = calculate_bollinger_bands(closes)
    
    # Envelope
    if indicators.get('envelope'):
        result['envelope'] = calculate_envelope(closes)
    
    # ATR
    if indicators.get('atr'):
        result['atr'] = calculate_atr(highs, lows, closes)
    
    # BBW
    if indicators.get('bbw'):
        result['bbw'] = calculate_bbw(closes)
    
    # VWAP
    if indicators.get('vwap'):
        result['vwap'] = calculate_vwap(highs, lows, closes, volumes)
    
    return result


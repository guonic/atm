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
        "macd": macd_line.tolist(),
        "signal": signal_line.tolist(),
        "histogram": histogram.tolist(),
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
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi.tolist()


def calculate_bollinger_bands(
    closes: pd.Series,
    period: int = 20,
    std_dev: int = 2,
) -> Dict[str, List[Optional[float]]]:
    """
    Calculate Bollinger Bands.
    
    Args:
        closes: Series of closing prices.
        period: Moving average period (default: 20).
        std_dev: Standard deviation multiplier (default: 2).
    
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
        "upper": upper.tolist(),
        "middle": sma.tolist(),
        "lower": lower.tolist(),
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
    
    return atr.tolist()


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
    return sma.tolist()


def calculate_indicators(
    kline_data: List[Dict],
    indicators: Optional[Dict[str, bool]] = None,
) -> Dict[str, any]:
    """
    Calculate all requested technical indicators from K-line data.
    
    Args:
        kline_data: List of K-line dictionaries with 'date', 'open', 'high', 'low', 'close', 'volume'.
        indicators: Dictionary indicating which indicators to calculate.
                    Keys: 'macd', 'rsi', 'bollinger', 'atr', 'ma5', 'ma10', 'ma20', 'ma30'.
    
    Returns:
        Dictionary with calculated indicator data.
    """
    if not kline_data:
        return {}
    
    if indicators is None:
        indicators = {
            "macd": True,
            "rsi": True,
            "bollinger": True,
            "atr": True,
            "ma5": True,
            "ma10": True,
            "ma20": True,
            "ma30": True,
        }
    
    # Convert to DataFrame
    df = pd.DataFrame(kline_data)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    
    result = {}
    
    # Calculate indicators
    if indicators.get("macd", False):
        result["macd"] = calculate_macd(df["close"])
    
    if indicators.get("rsi", False):
        result["rsi"] = calculate_rsi(df["close"])
    
    if indicators.get("bollinger", False):
        result["bollinger"] = calculate_bollinger_bands(df["close"])
    
    if indicators.get("atr", False):
        result["atr"] = calculate_atr(df["high"], df["low"], df["close"])
    
    if indicators.get("ma5", False):
        result["ma5"] = calculate_sma(df["close"], 5)
    
    if indicators.get("ma10", False):
        result["ma10"] = calculate_sma(df["close"], 10)
    
    if indicators.get("ma20", False):
        result["ma20"] = calculate_sma(df["close"], 20)
    
    if indicators.get("ma30", False):
        result["ma30"] = calculate_sma(df["close"], 30)
    
    return result


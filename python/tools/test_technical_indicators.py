"""
Test script for technical indicators calculation.

This script tests all technical indicator functions to ensure they work correctly.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from nq.utils.technical_indicators import (
    calculate_macd,
    calculate_rsi,
    calculate_bollinger_bands,
    calculate_atr,
    calculate_sma,
    calculate_ema,
    calculate_wma,
    calculate_kdj,
    calculate_cci,
    calculate_wr,
    calculate_obv,
    calculate_dmi,
    calculate_envelope,
    calculate_bbw,
    calculate_vwap,
    calculate_indicators,
)

def generate_sample_data(n=100):
    """Generate sample OHLCV data for testing."""
    np.random.seed(42)
    
    # Generate price data with trend
    base_price = 100
    dates = pd.date_range('2024-01-01', periods=n, freq='D')
    
    closes = []
    opens = []
    highs = []
    lows = []
    volumes = []
    
    current_price = base_price
    for i in range(n):
        # Random walk with slight upward trend
        change = np.random.normal(0.1, 2)
        current_price = max(10, current_price + change)
        
        open_price = current_price + np.random.normal(0, 0.5)
        high_price = max(open_price, current_price) + abs(np.random.normal(0, 1))
        low_price = min(open_price, current_price) - abs(np.random.normal(0, 1))
        close_price = current_price
        
        opens.append(open_price)
        highs.append(high_price)
        lows.append(low_price)
        closes.append(close_price)
        volumes.append(int(np.random.uniform(1000000, 10000000)))
    
    return {
        'dates': dates,
        'open': pd.Series(opens),
        'high': pd.Series(highs),
        'low': pd.Series(lows),
        'close': pd.Series(closes),
        'volume': pd.Series(volumes),
    }


def test_trend_indicators(data):
    """Test trend indicators."""
    print("\n=== Testing Trend Indicators ===")
    
    # Test SMA
    ma5 = calculate_sma(data['close'], 5)
    ma10 = calculate_sma(data['close'], 10)
    ma20 = calculate_sma(data['close'], 20)
    ma30 = calculate_sma(data['close'], 30)
    ma60 = calculate_sma(data['close'], 60)
    ma120 = calculate_sma(data['close'], 120)
    
    print(f"✓ SMA: MA5={len([x for x in ma5 if x is not None])} valid, "
          f"MA10={len([x for x in ma10 if x is not None])} valid, "
          f"MA20={len([x for x in ma20 if x is not None])} valid, "
          f"MA30={len([x for x in ma30 if x is not None])} valid, "
          f"MA60={len([x for x in ma60 if x is not None])} valid, "
          f"MA120={len([x for x in ma120 if x is not None])} valid")
    
    # Test EMA
    ema = calculate_ema(data['close'], 12)
    print(f"✓ EMA: {len([x for x in ema if x is not None])} valid values")
    
    # Test WMA
    wma = calculate_wma(data['close'], 14)
    print(f"✓ WMA: {len([x for x in wma if x is not None])} valid values")
    
    return True


def test_oscillator_indicators(data):
    """Test oscillator indicators."""
    print("\n=== Testing Oscillator Indicators ===")
    
    # Test RSI
    rsi = calculate_rsi(data['close'])
    valid_rsi = [x for x in rsi if x is not None and not (isinstance(x, float) and np.isnan(x))]
    print(f"✓ RSI: {len(valid_rsi)} valid values")
    if valid_rsi:
        print(f"  Range: {min(valid_rsi):.2f} - {max(valid_rsi):.2f} (should be 0-100)")
        assert all(0 <= x <= 100 for x in valid_rsi), "RSI values should be between 0 and 100"
    
    # Test KDJ
    kdj = calculate_kdj(data['high'], data['low'], data['close'])
    valid_k = [x for x in kdj['k'] if x is not None and not (isinstance(x, float) and np.isnan(x))]
    valid_d = [x for x in kdj['d'] if x is not None and not (isinstance(x, float) and np.isnan(x))]
    valid_j = [x for x in kdj['j'] if x is not None and not (isinstance(x, float) and np.isnan(x))]
    print(f"✓ KDJ: K={len(valid_k)} valid, D={len(valid_d)} valid, J={len(valid_j)} valid")
    if valid_k:
        print(f"  K range: {min(valid_k):.2f} - {max(valid_k):.2f} (should be 0-100)")
        assert all(0 <= x <= 100 for x in valid_k), "KDJ K values should be between 0 and 100"
    
    # Test CCI
    cci = calculate_cci(data['high'], data['low'], data['close'])
    valid_cci = [x for x in cci if x is not None and not (isinstance(x, float) and np.isnan(x))]
    print(f"✓ CCI: {len(valid_cci)} valid values")
    if valid_cci:
        print(f"  Range: {min(valid_cci):.2f} - {max(valid_cci):.2f}")
    
    # Test WR
    wr = calculate_wr(data['high'], data['low'], data['close'])
    valid_wr = [x for x in wr if x is not None and not (isinstance(x, float) and np.isnan(x))]
    print(f"✓ WR: {len(valid_wr)} valid values")
    if valid_wr:
        print(f"  Range: {min(valid_wr):.2f} - {max(valid_wr):.2f} (should be -100 to 0)")
        assert all(-100 <= x <= 0 for x in valid_wr), "WR values should be between -100 and 0"
    
    # Test OBV
    obv = calculate_obv(data['close'], data['volume'])
    print(f"✓ OBV: {len([x for x in obv if x is not None])} valid values")
    
    return True


def test_trend_oscillator_indicators(data):
    """Test trend + oscillator indicators."""
    print("\n=== Testing Trend + Oscillator Indicators ===")
    
    # Test MACD
    macd = calculate_macd(data['close'])
    print(f"✓ MACD: MACD={len([x for x in macd['macd'] if x is not None])} valid, "
          f"Signal={len([x for x in macd['signal'] if x is not None])} valid, "
          f"Histogram={len([x for x in macd['histogram'] if x is not None])} valid")
    
    # Test DMI
    dmi = calculate_dmi(data['high'], data['low'], data['close'])
    print(f"✓ DMI: PDI={len([x for x in dmi['pdi'] if x is not None])} valid, "
          f"MDI={len([x for x in dmi['mdi'] if x is not None])} valid, "
          f"ADX={len([x for x in dmi['adx'] if x is not None])} valid, "
          f"ADXR={len([x for x in dmi['adxr'] if x is not None])} valid")
    
    return True


def test_channel_indicators(data):
    """Test channel indicators."""
    print("\n=== Testing Channel Indicators ===")
    
    # Test Bollinger Bands
    bb = calculate_bollinger_bands(data['close'])
    print(f"✓ Bollinger Bands: Upper={len([x for x in bb['upper'] if x is not None])} valid, "
          f"Middle={len([x for x in bb['middle'] if x is not None])} valid, "
          f"Lower={len([x for x in bb['lower'] if x is not None])} valid")
    
    # Test Envelope
    envelope = calculate_envelope(data['close'])
    print(f"✓ Envelope: Upper={len([x for x in envelope['upper'] if x is not None])} valid, "
          f"Lower={len([x for x in envelope['lower'] if x is not None])} valid")
    
    return True


def test_volatility_indicators(data):
    """Test volatility indicators."""
    print("\n=== Testing Volatility Indicators ===")
    
    # Test ATR
    atr = calculate_atr(data['high'], data['low'], data['close'])
    print(f"✓ ATR: {len([x for x in atr if x is not None])} valid values")
    
    # Test BBW
    bbw = calculate_bbw(data['close'])
    print(f"✓ BBW: {len([x for x in bbw if x is not None])} valid values")
    
    return True


def test_volume_indicators(data):
    """Test volume indicators."""
    print("\n=== Testing Volume Indicators ===")
    
    # Test VWAP
    vwap = calculate_vwap(data['high'], data['low'], data['close'], data['volume'])
    print(f"✓ VWAP: {len([x for x in vwap if x is not None])} valid values")
    
    return True


def test_calculate_indicators(data):
    """Test the main calculate_indicators function."""
    print("\n=== Testing calculate_indicators Function ===")
    
    # Prepare kline data
    kline_data = []
    for i in range(len(data['dates'])):
        kline_data.append({
            'date': data['dates'][i].strftime('%Y-%m-%d'),
            'open': float(data['open'].iloc[i]),
            'high': float(data['high'].iloc[i]),
            'low': float(data['low'].iloc[i]),
            'close': float(data['close'].iloc[i]),
            'volume': float(data['volume'].iloc[i]),
        })
    
    # Test with all indicators enabled
    all_indicators = {
        'ma5': True, 'ma10': True, 'ma20': True, 'ma30': True,
        'ma60': True, 'ma120': True, 'ema': True, 'wma': True,
        'rsi': True, 'kdj': True, 'cci': True, 'wr': True, 'obv': True,
        'macd': True, 'dmi': True,
        'bollinger': True, 'envelope': True,
        'atr': True, 'bbw': True,
        'vwap': True,
    }
    
    result = calculate_indicators(kline_data, all_indicators)
    
    print(f"✓ calculate_indicators returned {len(result)} indicator groups:")
    for key in sorted(result.keys()):
        if isinstance(result[key], dict):
            print(f"  - {key}: {list(result[key].keys())}")
        else:
            count = len([x for x in result[key] if x is not None]) if isinstance(result[key], list) else 0
            print(f"  - {key}: {count} valid values")
    
    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("Technical Indicators Test Suite")
    print("=" * 60)
    
    # Generate sample data
    print("\nGenerating sample data (100 days)...")
    data = generate_sample_data(100)
    print(f"✓ Generated {len(data['dates'])} days of OHLCV data")
    print(f"  Price range: {data['close'].min():.2f} - {data['close'].max():.2f}")
    
    # Run tests
    try:
        test_trend_indicators(data)
        test_oscillator_indicators(data)
        test_trend_oscillator_indicators(data)
        test_channel_indicators(data)
        test_volatility_indicators(data)
        test_volume_indicators(data)
        test_calculate_indicators(data)
        
        print("\n" + "=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)
        return 0
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())


/**
 * Configuration for StockChart indicators
 */

import type { IndicatorConfig } from './StockChart.types'

export const DEFAULT_INDICATORS: IndicatorConfig = {
  // 趋势类
  ma5: false,
  ma10: false,
  ma20: false,
  ma30: false,
  ma60: false,
  ma120: false,
  ema: false,
  wma: false,
  
  // 震荡类
  rsi: false,
  kdj: false,
  cci: false,
  wr: false,
  obv: false,
  
  // 趋势+震荡
  macd: false,
  dmi: false,
  
  // 通道类
  bollinger: false,
  envelope: false,
  
  // 波动类
  atr: false,
  bbw: false,
  
  // 成交量类
  vol: false,
  vwap: false,
}

export const INDICATOR_CATEGORIES = {
  trend: {
    label: '趋势类',
    indicators: ['ma5', 'ma10', 'ma20', 'ma30', 'ma60', 'ma120', 'ema', 'wma'] as const,
  },
  oscillator: {
    label: '震荡类',
    indicators: ['rsi', 'kdj', 'cci', 'wr', 'obv'] as const,
  },
  trendOscillator: {
    label: '趋势+震荡',
    indicators: ['macd', 'dmi'] as const,
  },
  channel: {
    label: '通道类',
    indicators: ['bollinger', 'envelope'] as const,
  },
  volatility: {
    label: '波动类',
    indicators: ['atr', 'bbw'] as const,
  },
  volume: {
    label: '成交量类',
    indicators: ['vol', 'vwap'] as const,
  },
}

export const INDICATOR_LABELS: Record<keyof IndicatorConfig, string> = {
  ma5: 'MA5',
  ma10: 'MA10',
  ma20: 'MA20',
  ma30: 'MA30',
  ma60: 'MA60',
  ma120: 'MA120',
  ema: 'EMA',
  wma: 'WMA',
  rsi: 'RSI',
  kdj: 'KDJ',
  cci: 'CCI',
  wr: 'WR',
  obv: 'OBV',
  macd: 'MACD',
  dmi: 'DMI',
  bollinger: '布林带',
  envelope: '通道',
  atr: 'ATR',
  bbw: 'BBW',
  vol: '成交量',
  vwap: 'VWAP',
}


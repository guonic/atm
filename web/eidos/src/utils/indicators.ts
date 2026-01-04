/**
 * Technical indicators calculation utilities
 */

/**
 * Calculate EMA (Exponential Moving Average)
 */
export function calculateEMA(data: number[], period: number): number[] {
  const ema: (number | undefined)[] = new Array(data.length).fill(undefined)
  const multiplier = 2 / (period + 1)
  
  // First EMA value is SMA
  let sum = 0
  for (let i = 0; i < period && i < data.length; i++) {
    sum += data[i]
  }
  if (period - 1 < data.length) {
    ema[period - 1] = sum / period
  }
  
  // Calculate subsequent EMA values
  for (let i = period; i < data.length; i++) {
    if (ema[i - 1] !== undefined) {
      ema[i] = (data[i] - ema[i - 1]!) * multiplier + ema[i - 1]!
    }
  }
  
  return ema as number[]
}

/**
 * Calculate SMA (Simple Moving Average)
 */
export function calculateSMA(data: number[], period: number): (number | undefined)[] {
  const sma: (number | undefined)[] = new Array(data.length).fill(undefined)
  
  for (let i = period - 1; i < data.length; i++) {
    let sum = 0
    for (let j = i - period + 1; j <= i; j++) {
      sum += data[j]
    }
    sma[i] = sum / period
  }
  
  return sma
}

/**
 * Calculate MACD (Moving Average Convergence Divergence)
 */
export function calculateMACD(closes: number[], fastPeriod: number = 12, slowPeriod: number = 26, signalPeriod: number = 9) {
  const fastEMA = calculateEMA(closes, fastPeriod)
  const slowEMA = calculateEMA(closes, slowPeriod)
  
  const macdLine: (number | undefined)[] = new Array(closes.length).fill(undefined)
  for (let i = slowPeriod - 1; i < closes.length; i++) {
    if (fastEMA[i] !== undefined && slowEMA[i] !== undefined) {
      macdLine[i] = fastEMA[i]! - slowEMA[i]!
    }
  }
  
  // Calculate signal line (EMA of MACD line values that are defined)
  const macdValues: number[] = []
  const macdIndices: number[] = []
  for (let i = slowPeriod - 1; i < macdLine.length; i++) {
    if (macdLine[i] !== undefined) {
      macdValues.push(macdLine[i]!)
      macdIndices.push(i)
    }
  }
  
  if (macdValues.length < signalPeriod) {
    // Not enough data for signal line
    return {
      macd: macdLine as number[],
      signal: [],
      histogram: new Array(closes.length).fill(undefined) as number[],
    }
  }
  
  const signalLine = calculateEMA(macdValues, signalPeriod)
  
  // Calculate histogram - align with original dates array
  const histogram: (number | undefined)[] = new Array(closes.length).fill(undefined)
  
  // Signal line starts after signalPeriod-1 values in the macdValues array
  // The corresponding original index is macdIndices[signalPeriod - 1 + i]
  for (let i = 0; i < signalLine.length; i++) {
    if (signalLine[i] !== undefined) {
      const signalStartInMacdValues = signalPeriod - 1
      const macdValueIndex = signalStartInMacdValues + i
      if (macdValueIndex < macdIndices.length) {
        const originalIndex = macdIndices[macdValueIndex]
        if (originalIndex < closes.length && macdLine[originalIndex] !== undefined) {
          histogram[originalIndex] = macdLine[originalIndex]! - signalLine[i]!
        }
      }
    }
  }
  
  // Build signal array with indices
  const signalWithIndices: Array<{ value: number; index: number }> = []
  for (let i = 0; i < signalLine.length; i++) {
    if (signalLine[i] !== undefined) {
      const signalStartInMacdValues = signalPeriod - 1
      const macdValueIndex = signalStartInMacdValues + i
      if (macdValueIndex < macdIndices.length) {
        const originalIndex = macdIndices[macdValueIndex]
        signalWithIndices.push({ value: signalLine[i]!, index: originalIndex })
      }
    }
  }
  
  return {
    macd: macdLine as number[],
    signal: signalWithIndices,
    histogram: histogram as number[],
  }
}

/**
 * Calculate RSI (Relative Strength Index)
 */
export function calculateRSI(closes: number[], period: number = 14): (number | undefined)[] {
  const rsi: (number | undefined)[] = new Array(closes.length).fill(undefined)
  const gains: number[] = new Array(closes.length).fill(0)
  const losses: number[] = new Array(closes.length).fill(0)
  
  // Calculate price changes
  for (let i = 1; i < closes.length; i++) {
    const change = closes[i] - closes[i - 1]
    gains[i] = change > 0 ? change : 0
    losses[i] = change < 0 ? -change : 0
  }
  
  // Calculate initial average gain and loss
  let avgGain = 0
  let avgLoss = 0
  for (let i = 1; i <= period && i < closes.length; i++) {
    avgGain += gains[i]
    avgLoss += losses[i]
  }
  avgGain /= period
  avgLoss /= period
  
  // Calculate RSI
  for (let i = period; i < closes.length; i++) {
    if (i === period) {
      const rs = avgLoss === 0 ? 100 : avgGain / avgLoss
      rsi[i] = 100 - (100 / (1 + rs))
    } else {
      // Use Wilder's smoothing method
      avgGain = (avgGain * (period - 1) + gains[i]) / period
      avgLoss = (avgLoss * (period - 1) + losses[i]) / period
      const rs = avgLoss === 0 ? 100 : avgGain / avgLoss
      rsi[i] = 100 - (100 / (1 + rs))
    }
  }
  
  return rsi
}

/**
 * Calculate Bollinger Bands
 */
export function calculateBollingerBands(closes: number[], period: number = 20, stdDev: number = 2) {
  const sma = calculateSMA(closes, period)
  const upper: (number | undefined)[] = new Array(closes.length).fill(undefined)
  const lower: (number | undefined)[] = new Array(closes.length).fill(undefined)
  
  for (let i = period - 1; i < closes.length; i++) {
    if (sma[i] === undefined) continue
    
    // Calculate standard deviation
    let sumSquaredDiff = 0
    for (let j = i - period + 1; j <= i; j++) {
      const diff = closes[j] - sma[i]!
      sumSquaredDiff += diff * diff
    }
    const variance = sumSquaredDiff / period
    const standardDev = Math.sqrt(variance)
    
    upper[i] = sma[i]! + (standardDev * stdDev)
    lower[i] = sma[i]! - (standardDev * stdDev)
  }
  
  return {
    middle: sma,
    upper: upper as number[],
    lower: lower as number[],
  }
}

/**
 * Calculate ATR (Average True Range)
 */
export function calculateATR(highs: number[], lows: number[], closes: number[], period: number = 14): (number | undefined)[] {
  const trueRanges: number[] = new Array(highs.length).fill(0)
  const atr: (number | undefined)[] = new Array(highs.length).fill(undefined)
  
  // Calculate True Range
  for (let i = 1; i < highs.length; i++) {
    const tr1 = highs[i] - lows[i]
    const tr2 = Math.abs(highs[i] - closes[i - 1])
    const tr3 = Math.abs(lows[i] - closes[i - 1])
    trueRanges[i] = Math.max(tr1, tr2, tr3)
  }
  
  // Calculate initial ATR (SMA of first period TRs)
  let sum = 0
  for (let i = 1; i <= period && i < trueRanges.length; i++) {
    sum += trueRanges[i]
  }
  if (period < trueRanges.length) {
    atr[period] = sum / period
  }
  
  // Calculate subsequent ATR using Wilder's smoothing
  for (let i = period + 1; i < trueRanges.length; i++) {
    if (atr[i - 1] !== undefined) {
      atr[i] = (atr[i - 1]! * (period - 1) + trueRanges[i]) / period
    }
  }
  
  return atr
}


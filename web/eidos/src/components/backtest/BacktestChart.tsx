/**
 * Backtest Chart Component
 * 
 * Combines StockChart with backtest-specific overlays:
 * - Buy/sell trade markers
 * - Backtest period highlight
 * - Trade statistics
 */

import React, { useState, useEffect } from 'react'
import { StockChart } from '../stock/StockChart'
import type { IndicatorConfig, IndicatorData, KlineData, TimeInterval } from '../stock/StockChart.types'
import type { Trade } from '@/types/eidos'
import { getStockKline, getTrades } from '@/services/api'

export interface BacktestChartProps {
  expId: string
  symbol: string
  onClose?: () => void
  embedded?: boolean
  className?: string
  style?: React.CSSProperties
}

export function BacktestChart({
  expId,
  symbol,
  onClose,
  embedded = false,
  className = '',
  style,
}: BacktestChartProps) {
  const [klineData, setKlineData] = useState<KlineData[]>([])
  const [indicatorData, setIndicatorData] = useState<IndicatorData>({})
  const [trades, setTrades] = useState<Trade[]>([])
  const [backtestStart, setBacktestStart] = useState<string | null>(null)
  const [backtestEnd, setBacktestEnd] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)
  const [indicators, setIndicators] = useState<IndicatorConfig>({
    ma5: false,
    ma10: false,
    ma20: false,
    ma30: false,
    ma60: false,
    ma120: false,
    ema: false,
    wma: false,
    rsi: false,
    kdj: false,
    cci: false,
    wr: false,
    obv: false,
    macd: false,
    dmi: false,
    bollinger: false,
    envelope: false,
    atr: false,
    bbw: false,
    vol: false,
    vwap: false,
  })
  
  // Load data
  useEffect(() => {
    loadData()
  }, [expId, symbol])
  
  // Reload when indicators change
  useEffect(() => {
    if (klineData.length > 0) {
      loadData()
    }
  }, [indicators])
  
  const loadData = async () => {
    try {
      setLoading(true)
      
      // Determine which indicators to request
      const indicatorList: string[] = []
      Object.entries(indicators).forEach(([key, enabled]) => {
        if (enabled) {
          indicatorList.push(key)
        }
      })
      
      // Load K-line data with indicators
      const klineResponse = await getStockKline(expId, symbol, undefined, undefined, indicatorList)
      setKlineData(klineResponse.kline_data)
      setIndicatorData(klineResponse.indicators || {})
      setBacktestStart(klineResponse.backtest_start || null)
      setBacktestEnd(klineResponse.backtest_end || null)
      
      // Load trades
      const symbolTrades = await getTrades(expId, { symbol })
      setTrades(symbolTrades)
    } catch (error) {
      console.error('Failed to load backtest data:', error)
    } finally {
      setLoading(false)
    }
  }
  
  // Prepare trade markers for overlay
  const tradeMarkers = trades.map((trade) => {
    const direction = (trade.direction ?? trade.side ?? 1) as 1 | -1
    const dealTime = new Date(trade.deal_time)
    
    // 使用本地时区获取日期（避免 UTC 时区问题）
    // 例如：中国时间 2025-01-06 23:00 在 UTC 是 2025-01-06 15:00，但应该匹配 2025-01-06 的 K 线
    const year = dealTime.getFullYear()
    const month = String(dealTime.getMonth() + 1).padStart(2, '0')
    const day = String(dealTime.getDate()).padStart(2, '0')
    const tradeDate = `${year}-${month}-${day}` // YYYY-MM-DD in local timezone
    
    // 尝试匹配 K 线数据（支持多种日期格式）
    const kline = klineData.find((k) => {
      const klineDate = k.date.split('T')[0] // 移除时间部分
      return klineDate === tradeDate
    })
    
    return {
      date: tradeDate,
      price: kline?.close || trade.price,
      direction: direction === 1 ? 1 : -1,
    }
  }).filter(m => m.date && m.price)
  
  if (loading) {
    return (
      <div className={`bg-eidos-surface rounded-lg border border-eidos-muted/20 flex items-center justify-center ${className}`} style={style}>
        <div className="text-eidos-muted text-sm">加载中...</div>
      </div>
    )
  }
  
  return (
    <div className={`flex flex-col ${className}`} style={style}>
      <StockChart
        symbol={symbol}
        klineData={klineData}
        indicatorData={indicatorData}
        indicators={indicators}
        onIndicatorsChange={setIndicators}
        timeInterval="1d" // Backtest is fixed to daily (not changeable)
        embedded={embedded}
        className="flex-1"
        backtestOverlay={{
          backtestStart,
          backtestEnd,
          tradeMarkers,
        }}
      />
      
      {/* Trade Statistics */}
      <div className="p-2 border-t border-eidos-muted/20 bg-eidos-surface/30">
        <div className="grid grid-cols-3 gap-2 text-[10px]">
          <div>
            <div className="text-eidos-muted">买入次数</div>
            <div className="text-eidos-accent font-bold">
              {trades.filter((t) => (t.direction ?? t.side ?? 1) === 1).length}
            </div>
          </div>
          <div>
            <div className="text-eidos-muted">卖出次数</div>
            <div className="text-eidos-danger font-bold">
              {trades.filter((t) => (t.direction ?? t.side ?? -1) === -1).length}
            </div>
          </div>
          <div>
            <div className="text-eidos-muted">总交易次数</div>
            <div className="text-white font-bold">{trades.length}</div>
          </div>
        </div>
      </div>
    </div>
  )
}


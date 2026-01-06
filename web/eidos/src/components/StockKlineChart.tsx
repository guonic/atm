/**
 * @deprecated This component is deprecated. Use BacktestChart from './backtest/BacktestChart' instead.
 * 
 * This component has been replaced by:
 * - StockChart: Standard stock analysis panel (./stock/StockChart)
 * - BacktestChart: Backtest-specific chart with trade markers (./backtest/BacktestChart)
 * 
 * Migration guide:
 * - Replace `StockKlineChart` with `BacktestChart` for backtest scenarios
 * - Use `StockChart` for general stock analysis
 */

import React, { useState, useEffect, useRef } from 'react'
import * as echarts from 'echarts'
import { getStockKline, getTrades } from '@/services/api'
import type { Trade } from '@/types/eidos'

interface StockKlineChartProps {
  expId: string
  symbol: string
  onClose: () => void
  embedded?: boolean  // If true, render as embedded component instead of modal
}

interface IndicatorConfig {
  macd: boolean
  rsi: boolean
  bollinger: boolean
  atr: boolean
  ma5: boolean
  ma10: boolean
  ma20: boolean
  ma30: boolean
}

interface KlineData {
  date: string
  open: number
  high: number
  low: number
  close: number
  volume: number
}

interface IndicatorData {
  macd?: { macd: (number | null)[]; signal: (number | null)[]; histogram: (number | null)[] }
  rsi?: (number | null)[]
  bollinger?: { upper: (number | null)[]; middle: (number | null)[]; lower: (number | null)[] }
  atr?: (number | null)[]
  ma5?: (number | null)[]
  ma10?: (number | null)[]
  ma20?: (number | null)[]
  ma30?: (number | null)[]
}

function StockKlineChart({ expId, symbol, onClose, embedded = false }: StockKlineChartProps) {
  const [klineData, setKlineData] = useState<KlineData[]>([])
  const [indicatorData, setIndicatorData] = useState<IndicatorData>({})
  const [backtestStart, setBacktestStart] = useState<string | null>(null)
  const [backtestEnd, setBacktestEnd] = useState<string | null>(null)
  const [trades, setTrades] = useState<Trade[]>([])
  const [loading, setLoading] = useState(true)
  const [loadingMore, setLoadingMore] = useState(false)
  const [dataRange, setDataRange] = useState<{ startDate: string | null; endDate: string | null }>({
    startDate: null,
    endDate: null,
  })
  const [visibleWindow, setVisibleWindow] = useState<{ start: number; end: number }>({ start: 0, end: 100 })
  const [sliderButtonPosition, setSliderButtonPosition] = useState<number>(50) // Center position (0-100)
  const isDraggingSliderRef = useRef(false)
  const dragStartRef = useRef<{ sliderStart: number; windowStart: number; windowEnd: number } | null>(null)
  const sliderContainerRef = useRef<HTMLDivElement>(null)
  const [indicators, setIndicators] = useState<IndicatorConfig>({
    macd: false,
    rsi: false,
    bollinger: false,
    atr: false,
    ma5: false,
    ma10: false,
    ma20: false,
    ma30: false,
  })
  const chartRef = useRef<HTMLDivElement>(null)
  const chartInstanceRef = useRef<echarts.ECharts | null>(null)
  const resizeHandlerRef = useRef<(() => void) | null>(null)
  const dataZoomHandlerRef = useRef<((params: any) => void) | null>(null)

  useEffect(() => {
    loadData()
  }, [expId, symbol])

  useEffect(() => {
    // Add a small delay to ensure DOM is ready
    if (klineData.length > 0 && chartRef.current) {
      // Use setTimeout to ensure the container has proper dimensions
      const timer = setTimeout(() => {
        if (chartRef.current) {
          // Force resize after a brief delay to ensure container has dimensions
          renderChart()
          // Resize chart after render to ensure it fits the container
          setTimeout(() => {
            if (chartInstanceRef.current && !chartInstanceRef.current.isDisposed()) {
              chartInstanceRef.current.resize()
            }
          }, 200)
        }
      }, 100)
      return () => clearTimeout(timer)
    }
    
    // Cleanup function
    return () => {
      // Remove resize event listener if it exists
      if (resizeHandlerRef.current) {
        window.removeEventListener('resize', resizeHandlerRef.current)
        resizeHandlerRef.current = null
      }
      
      // Remove dataZoom event listener if it exists
      if (dataZoomHandlerRef.current && chartInstanceRef.current) {
        try {
          if (!chartInstanceRef.current.isDisposed()) {
            chartInstanceRef.current.off('dataZoom', dataZoomHandlerRef.current)
          }
        } catch (e) {
          // Ignore errors during disposal
        }
        dataZoomHandlerRef.current = null
      }
      
      // Dispose chart instance
      if (chartInstanceRef.current) {
        try {
          if (!chartInstanceRef.current.isDisposed()) {
            chartInstanceRef.current.dispose()
          }
        } catch (e) {
          // Ignore errors during disposal
        }
        chartInstanceRef.current = null
      }
    }
  }, [klineData, trades, indicators, visibleWindow, sliderButtonPosition])  // Re-render when indicators, window, or slider change

  const loadData = async (startDate?: string, endDate?: string, append: boolean = false) => {
    try {
      if (append) {
        setLoadingMore(true)
      } else {
        setLoading(true)
      }
      
      // Determine which indicators to request
      const indicatorList: string[] = []
      if (indicators.macd) indicatorList.push('macd')
      if (indicators.rsi) indicatorList.push('rsi')
      if (indicators.bollinger) indicatorList.push('bollinger')
      if (indicators.atr) indicatorList.push('atr')
      if (indicators.ma5) indicatorList.push('ma5')
      if (indicators.ma10) indicatorList.push('ma10')
      if (indicators.ma20) indicatorList.push('ma20')
      if (indicators.ma30) indicatorList.push('ma30')
      
      // Load kline data with indicators and trades for this symbol
      const [klineResponse, symbolTrades] = await Promise.all([
        getStockKline(expId, symbol, startDate, endDate, indicatorList.length > 0 ? indicatorList : undefined),
        getTrades(expId, { symbol }),
      ])
      
      if (append) {
        // Merge new data with existing data
        const existingDates = new Set(klineData.map(d => d.date))
        const newKlineData = klineResponse.kline_data.filter(d => !existingDates.has(d.date))
        
        // Sort by date and merge
        const mergedData = [...klineData, ...newKlineData].sort((a, b) => 
          a.date.localeCompare(b.date)
        )
        setKlineData(mergedData)
        
        // Update data range
        if (mergedData.length > 0) {
          setDataRange({
            startDate: mergedData[0].date,
            endDate: mergedData[mergedData.length - 1].date,
          })
        }
        
        // Note: For indicators, we would need to recalculate with all data
        // For now, we'll keep the original indicators and recalculate on next full load
        console.log(`Loaded ${newKlineData.length} new data points (total: ${mergedData.length})`)
      } else {
        setKlineData(klineResponse.kline_data)
        setIndicatorData(klineResponse.indicators || {})
        setBacktestStart(klineResponse.backtest_start || null)
        setBacktestEnd(klineResponse.backtest_end || null)
        
        // Update data range
        if (klineResponse.kline_data.length > 0) {
          setDataRange({
            startDate: klineResponse.kline_data[0].date,
            endDate: klineResponse.kline_data[klineResponse.kline_data.length - 1].date,
          })
          // Reset visible window and slider to initial state
          if (!append) {
            setVisibleWindow({ start: 0, end: 100 })
            setSliderButtonPosition(50) // Center
            dragStartRef.current = null
          }
        }
      }
      
      // Always update trades (they don't change when loading more data)
      setTrades(symbolTrades)
    } catch (error) {
      console.error('Failed to load kline data:', error)
    } finally {
      setLoading(false)
      setLoadingMore(false)
    }
  }
  
  // Handle slider drag to move window
  const handleSliderDrag = (sliderPercentage: number) => {
    if (!chartInstanceRef.current || chartInstanceRef.current.isDisposed()) return
    
    // Initialize drag start if not already set
    if (!dragStartRef.current) {
      dragStartRef.current = {
        sliderStart: sliderButtonPosition,
        windowStart: visibleWindow.start,
        windowEnd: visibleWindow.end,
      }
    }
    
    // Calculate drag distance (slider movement from center)
    const dragDistance = sliderPercentage - 50 // -50 to +50
    
    // Calculate window size
    const windowSize = dragStartRef.current.windowEnd - dragStartRef.current.windowStart
    
    // Calculate new window position based on drag distance
    // Drag left (negative) = move window left (earlier data)
    // Drag right (positive) = move window right (later data)
    const windowMoveDistance = dragDistance * 2 // Scale: slider moves 50% range, window moves proportionally
    
    const newWindowStart = Math.max(0, Math.min(100 - windowSize, 
      dragStartRef.current.windowStart - windowMoveDistance))
    const newWindowEnd = newWindowStart + windowSize
    
    // Update visible window state
    setVisibleWindow({ start: newWindowStart, end: newWindowEnd })
    setSliderButtonPosition(sliderPercentage)
    
    // Directly update chart dataZoom to make it responsive
    // Get all x-axis indices by checking the current option
    try {
      const currentOption = chartInstanceRef.current.getOption() as any
      const xAxisCount = currentOption.xAxis?.length || 1
      const xAxisIndices = Array.from({ length: xAxisCount }, (_, i) => i)
      
      chartInstanceRef.current.setOption({
        dataZoom: [
          {
            type: 'inside',
            xAxisIndex: xAxisIndices, // Update all x-axes
            start: newWindowStart,
            end: newWindowEnd,
          },
        ],
      }, { notMerge: false, lazyUpdate: false })
    } catch (error) {
      console.error('Error updating chart dataZoom:', error)
    }
  }
  
  // Load more data when scrolling to boundaries
  const loadMoreData = async (direction: 'left' | 'right') => {
    if (loadingMore || !dataRange.startDate || !dataRange.endDate) return
    
    try {
      setLoadingMore(true)
      
      if (direction === 'left') {
        // Load earlier data (before startDate)
        const endDate = new Date(dataRange.startDate)
        endDate.setDate(endDate.getDate() - 1) // One day before current start
        const startDate = new Date(endDate)
        startDate.setDate(startDate.getDate() - 60) // Load 60 days earlier
        
        await loadData(startDate.toISOString().split('T')[0], endDate.toISOString().split('T')[0], true)
      } else {
        // Load later data (after endDate)
        const startDate = new Date(dataRange.endDate)
        startDate.setDate(startDate.getDate() + 1) // One day after current end
        const endDate = new Date(startDate)
        endDate.setDate(endDate.getDate() + 60) // Load 60 days later
        
        await loadData(startDate.toISOString().split('T')[0], endDate.toISOString().split('T')[0], true)
      }
    } catch (error) {
      console.error('Failed to load more data:', error)
    } finally {
      setLoadingMore(false)
    }
  }
  
  // Reload data when indicators change
  useEffect(() => {
    if (klineData.length > 0) {
      loadData()
    }
  }, [indicators])

  const renderChart = () => {
    if (!chartRef.current || klineData.length === 0) {
      console.warn('Cannot render chart: chartRef or klineData is missing', {
        hasChartRef: !!chartRef.current,
        klineDataLength: klineData.length
      })
      return
    }

    // Clean up existing chart and resize handler
    if (resizeHandlerRef.current) {
      window.removeEventListener('resize', resizeHandlerRef.current)
      resizeHandlerRef.current = null
    }
    
    // Dispose existing chart
    if (chartInstanceRef.current) {
      try {
        if (!chartInstanceRef.current.isDisposed()) {
          chartInstanceRef.current.dispose()
        }
      } catch (e) {
        // Ignore errors during disposal
      }
      chartInstanceRef.current = null
    }

    // Create new chart instance
    let chart: echarts.ECharts
    try {
      chart = echarts.init(chartRef.current, 'dark')
      if (!chart) {
        console.error('Failed to initialize ECharts instance')
        return
      }
      chartInstanceRef.current = chart
      console.log('ECharts instance created successfully')
    } catch (error) {
      console.error('Error initializing ECharts:', error)
      return
    }

    // Prepare data
    const dates = klineData.map((d) => d.date)
    const klineValues = klineData.map((d) => [d.open, d.close, d.low, d.high])
    const volumes = klineData.map((d) => d.volume)
    const closes = klineData.map((d) => d.close)
    const highs = klineData.map((d) => d.high)
    const lows = klineData.map((d) => d.low)
    
    // Use backend-calculated indicators
    const macdData = indicators.macd && indicatorData.macd ? indicatorData.macd : null
    const rsiData = indicators.rsi && indicatorData.rsi ? indicatorData.rsi : null
    const bollingerData = indicators.bollinger && indicatorData.bollinger ? indicatorData.bollinger : null
    const atrData = indicators.atr && indicatorData.atr ? indicatorData.atr : null
    const ma5Data = indicators.ma5 && indicatorData.ma5 ? indicatorData.ma5 : null
    const ma10Data = indicators.ma10 && indicatorData.ma10 ? indicatorData.ma10 : null
    const ma20Data = indicators.ma20 && indicatorData.ma20 ? indicatorData.ma20 : null
    const ma30Data = indicators.ma30 && indicatorData.ma30 ? indicatorData.ma30 : null
    
    // Find backtest period indices for visual distinction
    // Normalize dates for comparison (both should be YYYY-MM-DD format)
    const normalizeDate = (dateStr: string | null | undefined): string | null => {
      if (!dateStr) return null
      // Extract YYYY-MM-DD from ISO string if needed
      return dateStr.split('T')[0]
    }
    
    const normalizedBacktestStart = normalizeDate(backtestStart)
    const normalizedBacktestEnd = normalizeDate(backtestEnd)
    
    const backtestStartIdx = normalizedBacktestStart 
      ? dates.findIndex(d => {
          const normalizedD = normalizeDate(d)
          return normalizedD && normalizedD >= normalizedBacktestStart
        })
      : -1
    
    // For end index, find the last date that is <= normalizedBacktestEnd
    // If not found, use the last index of dates array
    let backtestEndIdx = dates.length
    if (normalizedBacktestEnd) {
      // Find the last index where date <= normalizedBacktestEnd
      for (let i = dates.length - 1; i >= 0; i--) {
        const normalizedD = normalizeDate(dates[i])
        if (normalizedD && normalizedD <= normalizedBacktestEnd) {
          backtestEndIdx = i + 1  // +1 to include this date in the range
          break
        }
      }
      // If no date found <= end date, but we have a start index, use dates.length
      if (backtestEndIdx === dates.length && backtestStartIdx >= 0) {
        backtestEndIdx = dates.length
      }
    }
    
    const hasBacktestPeriod = backtestStartIdx >= 0 && backtestEndIdx > backtestStartIdx
    
    console.log('=== Chart Rendering (Backend Indicators) ===')
    console.log('Kline data length:', klineData.length)
    console.log('Backtest period:', { 
      start: backtestStart, 
      normalizedStart: normalizedBacktestStart,
      end: backtestEnd, 
      normalizedEnd: normalizedBacktestEnd,
      startIdx: backtestStartIdx, 
      endIdx: backtestEndIdx,
      hasBacktestPeriod,
      dateSamples: dates.slice(0, 5),
    })
    console.log('Indicators available:', Object.keys(indicatorData))

    // Prepare trade markers
    const buyPoints: Array<{ date: string; price: number; originalDate: string; utcDate?: string }> = []
    const sellPoints: Array<{ date: string; price: number; originalDate: string; utcDate?: string }> = []

    trades.forEach((trade) => {
      // Extract date from trade deal_time
      // Handle timezone: deal_time might be in UTC, but we need local date
      const dealTime = new Date(trade.deal_time)
      
      // Get date in local timezone (China timezone UTC+8)
      // Use getFullYear, getMonth, getDate to get local date components
      const year = dealTime.getFullYear()
      const month = String(dealTime.getMonth() + 1).padStart(2, '0')
      const day = String(dealTime.getDate()).padStart(2, '0')
      const tradeDateLocal = `${year}-${month}-${day}`  // YYYY-MM-DD in local timezone
      
      // Also get UTC date for comparison
      const tradeDateUTC = dealTime.toISOString().split('T')[0]
      
      const direction = trade.direction ?? trade.side ?? 1
      if (direction === 1) {
        buyPoints.push({ 
          date: tradeDateLocal,  // Use local date
          price: trade.price,
          originalDate: trade.deal_time,
          utcDate: tradeDateUTC  // Keep UTC date for debugging
        })
      } else if (direction === -1) {
        sellPoints.push({ 
          date: tradeDateLocal,  // Use local date
          price: trade.price,
          originalDate: trade.deal_time,
          utcDate: tradeDateUTC  // Keep UTC date for debugging
        })
      }
    })

    console.log('=== Date Matching Debug ===')
    console.log('Trade deal_time samples:', trades.slice(0, 3).map(t => {
      const dt = new Date(t.deal_time)
      return {
        deal_time: t.deal_time,
        utc_date: dt.toISOString().split('T')[0],
        local_date: `${dt.getFullYear()}-${String(dt.getMonth() + 1).padStart(2, '0')}-${String(dt.getDate()).padStart(2, '0')}`,
        timezone: Intl.DateTimeFormat().resolvedOptions().timeZone
      }
    }))
    console.log('Kline date samples:', dates.slice(0, 10))
    console.log('Buy points:', buyPoints.map(p => ({ date: p.date, utcDate: p.utcDate, original: p.originalDate })))
    console.log('Sell points:', sellPoints.map(p => ({ date: p.date, utcDate: p.utcDate, original: p.originalDate })))

    // Create buy/sell markers data - only use exact matches, no approximation
    const buyMarkers = buyPoints.map((point) => {
      // Try exact match first
      let index = dates.findIndex((d) => d === point.date)
      
      // If no exact match, try without time component (in case kline date has time)
      if (index < 0) {
        index = dates.findIndex((d) => {
          const dStr = d.split(' ')[0]  // Remove time if present
          return dStr === point.date
        })
      }
      
      // If still no match, try date string comparison (handles different formats)
      if (index < 0) {
        const pointDate = new Date(point.date)
        if (!isNaN(pointDate.getTime())) {
          index = dates.findIndex((d) => {
            try {
              const dDate = new Date(d)
              if (!isNaN(dDate.getTime())) {
                // Compare year, month, day only
                return dDate.getFullYear() === pointDate.getFullYear() &&
                       dDate.getMonth() === pointDate.getMonth() &&
                       dDate.getDate() === pointDate.getDate()
              }
            } catch {
              return false
            }
            return false
          })
        }
      }
      
      if (index >= 0) {
        console.log(`✓ Buy point matched: ${point.date} (original: ${point.originalDate}) -> index ${index} (${dates[index]})`)
        return [index, point.price]
      }
      
      // Log detailed mismatch information
      console.error(`✗ Buy point date NOT FOUND: ${point.date} (original: ${point.originalDate})`)
      console.error(`  Available dates: ${dates.slice(0, 10).join(', ')}${dates.length > 10 ? '...' : ''}`)
      console.error(`  Total available dates: ${dates.length}`)
      return null
    }).filter((item): item is number[] => item !== null)

    const sellMarkers = sellPoints.map((point) => {
      // Try exact match first
      let index = dates.findIndex((d) => d === point.date)
      
      // If no exact match, try without time component (in case kline date has time)
      if (index < 0) {
        index = dates.findIndex((d) => {
          const dStr = d.split(' ')[0]  // Remove time if present
          return dStr === point.date
        })
      }
      
      // If still no match, try date string comparison (handles different formats)
      if (index < 0) {
        const pointDate = new Date(point.date)
        if (!isNaN(pointDate.getTime())) {
          index = dates.findIndex((d) => {
            try {
              const dDate = new Date(d)
              if (!isNaN(dDate.getTime())) {
                // Compare year, month, day only
                return dDate.getFullYear() === pointDate.getFullYear() &&
                       dDate.getMonth() === pointDate.getMonth() &&
                       dDate.getDate() === pointDate.getDate()
              }
            } catch {
              return false
            }
            return false
          })
        }
      }
      
      if (index >= 0) {
        console.log(`✓ Sell point matched: ${point.date} (original: ${point.originalDate}) -> index ${index} (${dates[index]})`)
        return [index, point.price]
      }
      
      // Log detailed mismatch information
      console.error(`✗ Sell point date NOT FOUND: ${point.date} (original: ${point.originalDate})`)
      console.error(`  Available dates: ${dates.slice(0, 10).join(', ')}${dates.length > 10 ? '...' : ''}`)
      console.error(`  Total available dates: ${dates.length}`)
      return null
    }).filter((item): item is number[] => item !== null)
    
    console.log('Buy markers:', buyMarkers)
    console.log('Sell markers:', sellMarkers)

    // Build legend data dynamically based on enabled indicators
    const legendData: string[] = ['K线', '成交量', '买入', '卖出']
    if (indicators.ma5) legendData.push('MA5')
    if (indicators.ma10) legendData.push('MA10')
    if (indicators.ma20) legendData.push('MA20')
    if (indicators.ma30) legendData.push('MA30')
    if (indicators.macd) legendData.push('MACD', 'Signal', 'Histogram')
    if (indicators.rsi) legendData.push('RSI')
    if (indicators.bollinger) legendData.push('BB Upper', 'BB Middle', 'BB Lower')
    if (indicators.atr) legendData.push('ATR')

    // Calculate grid layout based on enabled indicators
    // Grid 0: K线图
    // Grid 1: 成交量
    // Grid 2+: 技术指标（MACD, RSI等）
    const grids: any[] = [
      {
        left: '10%',
        right: '8%',
        top: '15%',
        height: '40%',
      },
      {
        left: '10%',
        right: '8%',
        top: '58%',
        height: '12%',
      },
    ]
    
    const xAxes: any[] = [
      {
        type: 'category',
        data: dates,
        scale: true,
        boundaryGap: false,
        axisLine: { onZero: false },
        splitLine: { show: false },
        min: 'dataMin',
        max: 'dataMax',
        axisLabel: {
          color: '#8B949E',
          fontSize: 10,
          rotate: -45,
        },
      },
      {
        type: 'category',
        gridIndex: 1,
        data: dates,
        scale: true,
        boundaryGap: false,
        axisLine: { onZero: false },
        axisTick: { show: false },
        splitLine: { show: false },
        axisLabel: { show: false },
        min: 'dataMin',
        max: 'dataMax',
      },
    ]
    
    const yAxes: any[] = [
      {
        scale: true,
        splitArea: {
          show: true,
          areaStyle: {
            color: ['rgba(139, 148, 158, 0.05)', 'rgba(139, 148, 158, 0.02)'],
          },
        },
        axisLabel: {
          color: '#8B949E',
          fontSize: 10,
        },
      },
      {
        scale: true,
        gridIndex: 1,
        splitNumber: 2,
        axisLabel: { show: false },
        axisLine: { show: false },
        axisTick: { show: false },
        splitLine: { show: false },
      },
    ]
    
    let currentTop = 73
    let gridIndex = 2
    
    // Add MACD subplot if enabled
    if (indicators.macd && macdData) {
      grids.push({
        left: '10%',
        right: '8%',
        top: `${currentTop}%`,
        height: '12%',
      })
      xAxes.push({
        type: 'category',
        gridIndex: gridIndex,
        data: dates,
        scale: true,
        boundaryGap: false,
        axisLine: { onZero: false },
        axisTick: { show: false },
        splitLine: { show: false },
        axisLabel: { show: false },
        min: 'dataMin',
        max: 'dataMax',
      })
      yAxes.push({
        scale: true,
        gridIndex: gridIndex,
        splitNumber: 2,
        axisLabel: { 
          color: '#8B949E',
          fontSize: 9,
        },
        axisLine: { show: false },
        axisTick: { show: false },
        splitLine: { show: false },
      })
      currentTop += 15
      gridIndex++
    }
    
    // Add RSI subplot if enabled
    if (indicators.rsi && rsiData) {
      grids.push({
        left: '10%',
        right: '8%',
        top: `${currentTop}%`,
        height: '12%',
      })
      xAxes.push({
        type: 'category',
        gridIndex: gridIndex,
        data: dates,
        scale: true,
        boundaryGap: false,
        axisLine: { onZero: false },
        axisTick: { show: false },
        splitLine: { show: false },
        axisLabel: { show: false },
        min: 'dataMin',
        max: 'dataMax',
      })
      yAxes.push({
        scale: true,
        gridIndex: gridIndex,
        min: 0,
        max: 100,
        splitNumber: 4,
        axisLabel: { 
          color: '#8B949E',
          fontSize: 9,
        },
        axisLine: { show: false },
        axisTick: { show: false },
        splitLine: { 
          show: true,
          lineStyle: { color: '#8B949E', opacity: 0.2 },
        },
      })
      currentTop += 15
      gridIndex++
    }
    
    // Add ATR subplot if enabled
    if (indicators.atr && atrData) {
      grids.push({
        left: '10%',
        right: '8%',
        top: `${currentTop}%`,
        height: '12%',
      })
      xAxes.push({
        type: 'category',
        gridIndex: gridIndex,
        data: dates,
        scale: true,
        boundaryGap: false,
        axisLine: { onZero: false },
        axisTick: { show: false },
        splitLine: { show: false },
        axisLabel: { show: false },
        min: 'dataMin',
        max: 'dataMax',
      })
      yAxes.push({
        scale: true,
        gridIndex: gridIndex,
        splitNumber: 2,
        axisLabel: { 
          color: '#8B949E',
          fontSize: 9,
        },
        axisLine: { show: false },
        axisTick: { show: false },
        splitLine: { show: false },
      })
      currentTop += 15
      gridIndex++
    }
    
    // Adjust dataZoom top position based on number of grids
    // Ensure slider is always visible at the bottom
    const dataZoomTop = Math.min(currentTop + 5, 85) // Cap at 85% to ensure visibility

    const option: echarts.EChartsOption = {
      backgroundColor: 'transparent',
      animation: false,
      legend: {
        data: legendData,
        textStyle: { color: '#8B949E' },
        top: 10,
        type: 'scroll',
      },
      tooltip: {
        trigger: 'axis',
        axisPointer: {
          type: 'cross',
        },
        backgroundColor: '#161B22',
        borderColor: '#C5A059',
        textStyle: { color: '#8B949E' },
      },
      grid: grids,
      xAxis: xAxes,
      yAxis: yAxes,
      dataZoom: [
        {
          type: 'inside',
          xAxisIndex: xAxes.map((_, i) => i),
          start: visibleWindow.start,
          end: visibleWindow.end,
          zoomOnMouseWheel: true, // Enable mouse wheel zoom
          moveOnMouseMove: true, // Enable drag to pan
          moveOnMouseWheel: false, // Disable wheel to pan (use wheel for zoom)
          preventDefaultMouseMove: true,
        },
      ],
      series: [
        {
          name: 'K线',
          type: 'candlestick',
          xAxisIndex: 0,
          yAxisIndex: 0,
          data: klineValues,
          itemStyle: {
            color: '#00F2FF', // 上涨颜色（青色）
            color0: '#FF3D00', // 下跌颜色（红色）
            borderColor: null,
            borderColor0: null,
          },
          // Add markArea to highlight backtest period
          // For category xAxis, use index; for time xAxis, use date string
          markArea: hasBacktestPeriod && backtestStartIdx >= 0 && backtestEndIdx > backtestStartIdx ? {
            silent: false,
            itemStyle: {
              color: 'rgba(197, 160, 89, 0.2)', // Eidos Gold background for backtest period
              borderColor: 'rgba(197, 160, 89, 0.5)',
              borderWidth: 1,
            },
            data: [
              [
                {
                  name: '回测开始',
                  xAxis: backtestStartIdx,
                },
                {
                  name: '回测结束',
                  xAxis: Math.min(backtestEndIdx - 1, dates.length - 1),
                },
              ],
            ],
            label: {
              show: true,
              position: 'insideTop',
              formatter: '回测区间',
              color: '#C5A059',
              fontSize: 12,
              fontWeight: 'bold',
              backgroundColor: 'rgba(197, 160, 89, 0.4)',
              padding: [4, 8],
              borderRadius: 4,
            },
          } : undefined,
        },
        // 均线（显示在K线图上）
        ...(indicators.ma5 && ma5Data ? [{
          name: 'MA5',
          type: 'line',
          xAxisIndex: 0,
          yAxisIndex: 0,
          data: ma5Data.map(v => v !== undefined && !isNaN(v) ? v : null),
          lineStyle: { color: '#FFD700', width: 1.5 }, // 金色
          symbol: 'none',
          smooth: false,
        }] : []),
        ...(indicators.ma10 && ma10Data ? [{
          name: 'MA10',
          type: 'line',
          xAxisIndex: 0,
          yAxisIndex: 0,
          data: ma10Data.map(v => v !== undefined && !isNaN(v) ? v : null),
          lineStyle: { color: '#00F2FF', width: 1.5 }, // 青色
          symbol: 'none',
          smooth: false,
        }] : []),
        ...(indicators.ma20 && ma20Data ? [{
          name: 'MA20',
          type: 'line',
          xAxisIndex: 0,
          yAxisIndex: 0,
          data: ma20Data.map(v => v !== undefined && !isNaN(v) ? v : null),
          lineStyle: { color: '#FF6B9D', width: 1.5 }, // 粉红色
          symbol: 'none',
          smooth: false,
        }] : []),
        ...(indicators.ma30 && ma30Data ? [{
          name: 'MA30',
          type: 'line',
          xAxisIndex: 0,
          yAxisIndex: 0,
          data: ma30Data.map(v => v !== undefined && !isNaN(v) ? v : null),
          lineStyle: { color: '#C5A059', width: 1.5 }, // 金色
          symbol: 'none',
          smooth: false,
        }] : []),
        // 布林带（显示在K线图上，使用后端计算的数据）
        ...(indicators.bollinger && bollingerData ? (() => {
          // Convert null/undefined to null for ECharts compatibility
          const upperData = bollingerData.upper.map(v => v !== null && v !== undefined && !isNaN(v) ? v : null)
          const middleData = bollingerData.middle.map(v => v !== null && v !== undefined && !isNaN(v) ? v : null)
          const lowerData = bollingerData.lower.map(v => v !== null && v !== undefined && !isNaN(v) ? v : null)
          
          const upperValidCount = upperData.filter(v => v !== null).length
          const middleValidCount = middleData.filter(v => v !== null).length
          const lowerValidCount = lowerData.filter(v => v !== null).length
          console.log('Bollinger Series Data:', {
            upperValidCount,
            middleValidCount,
            lowerValidCount,
            upperSample: upperData.filter(v => v !== null).slice(0, 3),
            middleSample: middleData.filter(v => v !== null).slice(0, 3),
            lowerSample: lowerData.filter(v => v !== null).slice(0, 3),
          })
          
          return [
            {
              name: 'BB Upper',
              type: 'line',
              xAxisIndex: 0,
              yAxisIndex: 0,
              data: upperData,
              lineStyle: { color: '#C5A059', width: 1, type: 'dashed' },
              symbol: 'none',
              silent: true,
            },
            {
              name: 'BB Middle',
              type: 'line',
              xAxisIndex: 0,
              yAxisIndex: 0,
              data: middleData,
              lineStyle: { color: '#8B949E', width: 1 },
              symbol: 'none',
              silent: true,
            },
            {
              name: 'BB Lower',
              type: 'line',
              xAxisIndex: 0,
              yAxisIndex: 0,
              data: lowerData,
              lineStyle: { color: '#C5A059', width: 1, type: 'dashed' },
              symbol: 'none',
              silent: true,
              areaStyle: {
                color: {
                  type: 'linear',
                  x: 0,
                  y: 0,
                  x2: 0,
                  y2: 1,
                  colorStops: [
                    { offset: 0, color: 'rgba(197, 160, 89, 0.1)' },
                    { offset: 1, color: 'rgba(197, 160, 89, 0.05)' },
                  ],
                },
              },
            },
          ]
        })() : []),
        {
          name: '买入',
          type: 'scatter',
          xAxisIndex: 0,
          yAxisIndex: 0,
          data: buyMarkers,
          symbol: 'triangle',
          symbolSize: 16,
          symbolOffset: [0, -8],  // Offset upward to make it more visible
          itemStyle: {
            color: '#00F2FF', // 青色
            borderColor: '#00F2FF',
            borderWidth: 2,
          },
          label: {
            show: true,
            position: 'bottom',
            formatter: '买',
            color: '#00F2FF',
            fontSize: 12,
            fontWeight: 'bold',
            backgroundColor: 'rgba(0, 242, 255, 0.2)',
            padding: [2, 4],
            borderRadius: 2,
          },
          emphasis: {
            itemStyle: {
              color: '#00F2FF',
              borderColor: '#FFFFFF',
              borderWidth: 3,
              shadowBlur: 10,
              shadowColor: '#00F2FF',
            },
            label: {
              fontSize: 14,
            },
          },
        },
        {
          name: '卖出',
          type: 'scatter',
          xAxisIndex: 0,
          yAxisIndex: 0,
          data: sellMarkers,
          symbol: 'triangle',
          symbolRotate: 180,
          symbolSize: 16,
          symbolOffset: [0, 8],  // Offset downward to make it more visible
          itemStyle: {
            color: '#FF3D00', // 红色
            borderColor: '#FF3D00',
            borderWidth: 2,
          },
          label: {
            show: true,
            position: 'top',
            formatter: '卖',
            color: '#FF3D00',
            fontSize: 12,
            fontWeight: 'bold',
            backgroundColor: 'rgba(255, 61, 0, 0.2)',
            padding: [2, 4],
            borderRadius: 2,
          },
          emphasis: {
            itemStyle: {
              color: '#FF3D00',
              borderColor: '#FFFFFF',
              borderWidth: 3,
              shadowBlur: 10,
              shadowColor: '#FF3D00',
            },
            label: {
              fontSize: 14,
            },
          },
        },
        {
          name: '成交量',
          type: 'bar',
          xAxisIndex: 1,
          yAxisIndex: 1,
          data: volumes,
          itemStyle: {
            color: (params: any) => {
              const index = params.dataIndex
              if (index > 0) {
                const current = klineData[index]
                const prev = klineData[index - 1]
                return current.close >= prev.close ? '#00F2FF' : '#FF3D00'
              }
              return '#8B949E'
            },
          },
        },
        // MACD指标
        ...(indicators.macd && macdData ? (() => {
          const macdGridIndex = 2
          const macdYAxisIndex = 2
          const macdXAxisIndex = 2
          
          // Prepare MACD data arrays aligned with dates
          const macdLineData: (number | null)[] = new Array(dates.length).fill(null)
          const signalLineData: (number | null)[] = new Array(dates.length).fill(null)
          const histogramData: (number | null)[] = new Array(dates.length).fill(null)
          
          // Fill MACD line data (backend returns arrays directly)
          macdData.macd.forEach((val, idx) => {
            if (val !== null && val !== undefined && !isNaN(val) && idx < dates.length) {
              macdLineData[idx] = val
            }
          })
          
          // Fill Signal line data (backend returns arrays directly)
          macdData.signal.forEach((val, idx) => {
            if (val !== null && val !== undefined && !isNaN(val) && idx < dates.length) {
              signalLineData[idx] = val
            }
          })
          
          // Fill Histogram data (backend returns arrays directly)
          macdData.histogram.forEach((val, idx) => {
            if (val !== null && val !== undefined && !isNaN(val) && idx < dates.length) {
              histogramData[idx] = val
            }
          })
          
          const macdValidCount = macdLineData.filter(v => v !== null).length
          const signalValidCount = signalLineData.filter(v => v !== null).length
          const histogramValidCount = histogramData.filter(v => v !== null).length
          
          console.log('MACD Series Data (aligned with dates):', {
            datesLength: dates.length,
            macdValidCount,
            signalValidCount,
            histogramValidCount,
            macdSample: macdLineData.filter(v => v !== null).slice(0, 5),
            signalSample: signalLineData.filter(v => v !== null).slice(0, 5),
            histogramSample: histogramData.filter(v => v !== null).slice(0, 5),
            macdRawSample: macdData.macd.filter(v => v !== null).slice(0, 5),
            signalRawSample: macdData.signal.filter(v => v !== null).slice(0, 5),
            histogramRawSample: macdData.histogram.filter(v => v !== null).slice(0, 5),
          })
          
          // Only add series if there's valid data
          const series: any[] = []
          
          if (macdValidCount > 0) {
            series.push({
              name: 'MACD',
              type: 'line',
              xAxisIndex: macdXAxisIndex,
              yAxisIndex: macdYAxisIndex,
              data: macdLineData,
              lineStyle: { color: '#00F2FF', width: 1.5 },
              symbol: 'none',
            })
          } else {
            console.warn(`⚠️ MACD line has no valid data (need at least 26 data points, have ${dates.length})`)
          }
          
          if (signalValidCount > 0) {
            series.push({
              name: 'Signal',
              type: 'line',
              xAxisIndex: macdXAxisIndex,
              yAxisIndex: macdYAxisIndex,
              data: signalLineData,
              lineStyle: { color: '#FF3D00', width: 1.5 },
              symbol: 'none',
            })
          } else {
            console.warn(`⚠️ MACD Signal line has no valid data (need at least 34 data points, have ${dates.length})`)
          }
          
          if (histogramValidCount > 0) {
            series.push({
              name: 'Histogram',
              type: 'bar',
              xAxisIndex: macdXAxisIndex,
              yAxisIndex: macdYAxisIndex,
              data: histogramData,
              itemStyle: {
                color: (params: any) => {
                  const val = params.value
                  return val !== null && val >= 0 ? '#00F2FF' : '#FF3D00'
                },
              },
            })
          } else {
            console.warn(`⚠️ MACD Histogram has no valid data (need at least 34 data points, have ${dates.length})`)
          }
          
          return series
        })() : []),
        // RSI指标
        ...(indicators.rsi && rsiData ? (() => {
          const rsiGridIndex = indicators.macd ? 3 : 2
          const rsiYAxisIndex = indicators.macd ? 3 : 2
          const rsiXAxisIndex = indicators.macd ? 3 : 2
          
          const rsiLineData: (number | null)[] = new Array(dates.length).fill(null)
          rsiData.forEach((val, idx) => {
            if (val !== undefined && !isNaN(val) && idx < dates.length) {
              rsiLineData[idx] = val
            }
          })
          
          return [
            {
              name: 'RSI',
              type: 'line',
              xAxisIndex: rsiXAxisIndex,
              yAxisIndex: rsiYAxisIndex,
              data: rsiLineData,
              lineStyle: { color: '#C5A059', width: 1.5 },
              symbol: 'none',
              markLine: {
                data: [
                  { yAxis: 70, name: '超买', lineStyle: { color: '#FF3D00', type: 'dashed' } },
                  { yAxis: 30, name: '超卖', lineStyle: { color: '#00F2FF', type: 'dashed' } },
                  { yAxis: 50, name: '中线', lineStyle: { color: '#8B949E', type: 'dashed' } },
                ],
                label: { show: false },
              },
            },
          ]
        })() : []),
        // ATR指标
        ...(indicators.atr && atrData ? (() => {
          // Calculate grid index based on enabled indicators
          let atrGridIndex = 2
          if (indicators.macd) atrGridIndex++
          if (indicators.rsi) atrGridIndex++
          const atrYAxisIndex = atrGridIndex
          const atrXAxisIndex = atrGridIndex
          
          const atrLineData: (number | null)[] = new Array(dates.length).fill(null)
          atrData.forEach((val, idx) => {
            if (val !== undefined && !isNaN(val) && idx < dates.length) {
              atrLineData[idx] = val
            }
          })
          
          const atrValidCount = atrLineData.filter(v => v !== null).length
          console.log('ATR Series Data:', {
            atrValidCount,
            atrGridIndex,
            atrSample: atrLineData.filter(v => v !== null).slice(0, 3),
          })
          
          return [
            {
              name: 'ATR',
              type: 'line',
              xAxisIndex: atrXAxisIndex,
              yAxisIndex: atrYAxisIndex,
              data: atrLineData,
              lineStyle: { color: '#8B949E', width: 1.5 },
              symbol: 'none',
              areaStyle: {
                color: {
                  type: 'linear',
                  x: 0,
                  y: 0,
                  x2: 0,
                  y2: 1,
                  colorStops: [
                    { offset: 0, color: 'rgba(139, 148, 158, 0.2)' },
                    { offset: 1, color: 'rgba(139, 148, 158, 0.05)' },
                  ],
                },
              },
            },
          ]
        })() : []),
      ],
    }

    if (!chartInstanceRef.current) {
      console.error('Chart instance is null, cannot set option')
      return
    }
    
    try {
      chartInstanceRef.current.setOption(option)
      const seriesArray = Array.isArray(option.series) ? option.series : (option.series ? [option.series] : [])
      console.log('Chart option set successfully', { 
        dataPoints: klineData.length,
        buyMarkers: buyMarkers.length,
        sellMarkers: sellMarkers.length,
        totalSeries: seriesArray.length,
        grids: grids.length,
        xAxes: xAxes.length,
        yAxes: yAxes.length,
        legendData: legendData.length,
      })
      console.log('Series names:', seriesArray.map((s: any) => s.name))
    } catch (error) {
      console.error('Error setting chart option:', error)
      return
    }

    // Handle window resize
    const handleResize = () => {
      if (chartInstanceRef.current && !chartInstanceRef.current.isDisposed()) {
        try {
          chartInstanceRef.current.resize()
        } catch (e) {
          // Ignore errors if chart is disposed during resize
          console.warn('Chart resize error:', e)
        }
      }
    }
    
    // Handle dataZoom events for window-based navigation
    const handleDataZoom = (params: any) => {
      if (!chartInstanceRef.current) return
      
      const { start, end, batch } = params
      
      // Update visible window
      setVisibleWindow({ start, end })
      
      
      // Load more data when scrolling near boundaries (within 10% of edge)
      if (!loadingMore) {
        const threshold = 0.1 // 10% threshold
        
        if (start < threshold * 100 && dataRange.startDate) {
          // Scrolling to the left edge, load earlier data
          console.log('Loading earlier data...', { start, totalDataPoints: klineData.length })
          loadMoreData('left')
        } else if (end > (1 - threshold) * 100 && dataRange.endDate) {
          // Scrolling to the right edge, load later data
          console.log('Loading later data...', { end, totalDataPoints: klineData.length })
          loadMoreData('right')
        }
      }
    }
    
    // Remove previous handlers if exist
    if (resizeHandlerRef.current) {
      window.removeEventListener('resize', resizeHandlerRef.current)
    }
    if (dataZoomHandlerRef.current && chartInstanceRef.current) {
      chartInstanceRef.current.off('dataZoom', dataZoomHandlerRef.current)
    }
    
    // Add new resize handler
    window.addEventListener('resize', handleResize)
    resizeHandlerRef.current = handleResize
    
    // Add dataZoom event listener
    chart.on('dataZoom', handleDataZoom)
    dataZoomHandlerRef.current = handleDataZoom
  }

  if (loading) {
    if (embedded) {
      return (
        <div className="flex items-center justify-center p-8">
          <div className="text-eidos-muted">加载中...</div>
        </div>
      )
    }
    return (
      <div className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center">
        <div className="bg-eidos-surface/90 rounded-xl p-6">
          <div className="text-eidos-muted">加载中...</div>
        </div>
      </div>
    )
  }

  // Embedded mode - render inline
  if (embedded) {
    return (
      <div className="w-full">
        {/* Indicator Selection */}
        <div className="p-2 border-b border-eidos-muted/20 bg-eidos-surface/30">
          <div className="flex items-center gap-3 flex-wrap">
            <span className="text-xs text-eidos-muted">技术指标:</span>
            <label className="flex items-center gap-1 cursor-pointer">
              <input
                type="checkbox"
                checked={indicators.ma5}
                onChange={(e) => setIndicators({ ...indicators, ma5: e.target.checked })}
                className="w-3 h-3 rounded border-eidos-muted/30 bg-eidos-surface text-eidos-accent focus:ring-eidos-accent"
              />
              <span className="text-xs text-eidos-muted">MA5</span>
            </label>
            <label className="flex items-center gap-1 cursor-pointer">
              <input
                type="checkbox"
                checked={indicators.ma10}
                onChange={(e) => setIndicators({ ...indicators, ma10: e.target.checked })}
                className="w-3 h-3 rounded border-eidos-muted/30 bg-eidos-surface text-eidos-accent focus:ring-eidos-accent"
              />
              <span className="text-xs text-eidos-muted">MA10</span>
            </label>
            <label className="flex items-center gap-1 cursor-pointer">
              <input
                type="checkbox"
                checked={indicators.ma20}
                onChange={(e) => setIndicators({ ...indicators, ma20: e.target.checked })}
                className="w-3 h-3 rounded border-eidos-muted/30 bg-eidos-surface text-eidos-accent focus:ring-eidos-accent"
              />
              <span className="text-xs text-eidos-muted">MA20</span>
            </label>
            <label className="flex items-center gap-1 cursor-pointer">
              <input
                type="checkbox"
                checked={indicators.ma30}
                onChange={(e) => setIndicators({ ...indicators, ma30: e.target.checked })}
                className="w-3 h-3 rounded border-eidos-muted/30 bg-eidos-surface text-eidos-accent focus:ring-eidos-accent"
              />
              <span className="text-xs text-eidos-muted">MA30</span>
            </label>
            <label className="flex items-center gap-1 cursor-pointer">
              <input
                type="checkbox"
                checked={indicators.macd}
                onChange={(e) => setIndicators({ ...indicators, macd: e.target.checked })}
                className="w-3 h-3 rounded border-eidos-muted/30 bg-eidos-surface text-eidos-accent focus:ring-eidos-accent"
              />
              <span className="text-xs text-eidos-muted">MACD</span>
            </label>
            <label className="flex items-center gap-1 cursor-pointer">
              <input
                type="checkbox"
                checked={indicators.rsi}
                onChange={(e) => setIndicators({ ...indicators, rsi: e.target.checked })}
                className="w-3 h-3 rounded border-eidos-muted/30 bg-eidos-surface text-eidos-accent focus:ring-eidos-accent"
              />
              <span className="text-xs text-eidos-muted">RSI</span>
            </label>
            <label className="flex items-center gap-1 cursor-pointer">
              <input
                type="checkbox"
                checked={indicators.bollinger}
                onChange={(e) => setIndicators({ ...indicators, bollinger: e.target.checked })}
                className="w-3 h-3 rounded border-eidos-muted/30 bg-eidos-surface text-eidos-accent focus:ring-eidos-accent"
              />
              <span className="text-xs text-eidos-muted">布林带</span>
            </label>
            <label className="flex items-center gap-1 cursor-pointer">
              <input
                type="checkbox"
                checked={indicators.atr}
                onChange={(e) => setIndicators({ ...indicators, atr: e.target.checked })}
                className="w-3 h-3 rounded border-eidos-muted/30 bg-eidos-surface text-eidos-accent focus:ring-eidos-accent"
              />
              <span className="text-xs text-eidos-muted">ATR</span>
            </label>
          </div>
        </div>
        
        {/* Custom Slider Controller */}
        <div className="px-4 py-2 border-b border-eidos-muted/20">
          <div 
            ref={sliderContainerRef}
            className="relative w-full h-8 cursor-pointer"
            onMouseDown={(e) => {
              e.preventDefault()
              e.stopPropagation()
              if (!sliderContainerRef.current) return
              
              isDraggingSliderRef.current = true
              dragStartRef.current = null // Reset drag start
              
              const rect = sliderContainerRef.current.getBoundingClientRect()
              const x = e.clientX - rect.left
              const percentage = Math.max(0, Math.min(100, (x / rect.width) * 100))
              handleSliderDrag(percentage)
              
              const handleMouseMove = (moveEvent: MouseEvent) => {
                if (!sliderContainerRef.current || !isDraggingSliderRef.current) return
                moveEvent.preventDefault()
                moveEvent.stopPropagation()
                
                const moveRect = sliderContainerRef.current.getBoundingClientRect()
                const moveX = moveEvent.clientX - moveRect.left
                const movePercentage = Math.max(0, Math.min(100, (moveX / moveRect.width) * 100))
                handleSliderDrag(movePercentage)
              }
              
              const handleMouseUp = (upEvent: MouseEvent) => {
                upEvent.preventDefault()
                upEvent.stopPropagation()
                
                isDraggingSliderRef.current = false
                dragStartRef.current = null
                
                // Reset slider button to center after drag
                setTimeout(() => {
                  setSliderButtonPosition(50)
                }, 100)
                
                document.removeEventListener('mousemove', handleMouseMove)
                document.removeEventListener('mouseup', handleMouseUp)
              }
              
              document.addEventListener('mousemove', handleMouseMove, { passive: false })
              document.addEventListener('mouseup', handleMouseUp, { passive: false })
            }}
          >
            {/* Slider Track */}
            <div className="absolute inset-0 flex items-center">
              <div className="w-full h-1 bg-eidos-muted/20 rounded-full"></div>
            </div>
            
            {/* Slider Button */}
            <div
              className="absolute top-1/2 -translate-y-1/2 w-4 h-4 bg-eidos-gold rounded-full shadow-lg cursor-grab active:cursor-grabbing transition-all hover:scale-110"
              style={{ left: `calc(${sliderButtonPosition}% - 8px)` }}
            >
              <div className="absolute inset-0 bg-eidos-gold rounded-full opacity-50 animate-pulse"></div>
            </div>
          </div>
        </div>
        
        {/* Chart */}
        <div className="p-4" style={{ minHeight: '500px' }}>
          <div 
            ref={chartRef} 
            className="w-full"
            style={{ minHeight: '500px', height: '500px', width: '100%' }}
          />
          {klineData.length === 0 && !loading && (
            <div className="flex items-center justify-center h-[400px] text-eidos-muted text-sm">
              暂无K线数据
            </div>
          )}
        </div>

        {/* Trade Summary */}
        <div className="p-4 border-t border-eidos-muted/20 bg-eidos-surface/30">
          <div className="grid grid-cols-3 gap-4 text-xs">
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

  // Modal mode - render as popup
  return (
    <div className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-4">
      <div className="bg-eidos-surface/90 rounded-xl shadow-2xl w-full max-w-6xl max-h-[90vh] flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-eidos-muted/20">
          <div>
            <h2 className="text-lg font-semibold text-eidos-gold">{symbol}</h2>
            <p className="text-xs text-eidos-muted mt-1">K线图与交易点位</p>
          </div>
          <button
            onClick={onClose}
            className="text-eidos-muted hover:text-white transition-colors px-3 py-1 rounded"
          >
            关闭
          </button>
        </div>
        
        {/* Indicator Selection */}
        <div className="p-2 border-b border-eidos-muted/20 bg-eidos-surface/30">
          <div className="flex items-center gap-3 flex-wrap">
            <span className="text-xs text-eidos-muted">技术指标:</span>
            <label className="flex items-center gap-1 cursor-pointer">
              <input
                type="checkbox"
                checked={indicators.ma5}
                onChange={(e) => setIndicators({ ...indicators, ma5: e.target.checked })}
                className="w-3 h-3 rounded border-eidos-muted/30 bg-eidos-surface text-eidos-accent focus:ring-eidos-accent"
              />
              <span className="text-xs text-eidos-muted">MA5</span>
            </label>
            <label className="flex items-center gap-1 cursor-pointer">
              <input
                type="checkbox"
                checked={indicators.ma10}
                onChange={(e) => setIndicators({ ...indicators, ma10: e.target.checked })}
                className="w-3 h-3 rounded border-eidos-muted/30 bg-eidos-surface text-eidos-accent focus:ring-eidos-accent"
              />
              <span className="text-xs text-eidos-muted">MA10</span>
            </label>
            <label className="flex items-center gap-1 cursor-pointer">
              <input
                type="checkbox"
                checked={indicators.ma20}
                onChange={(e) => setIndicators({ ...indicators, ma20: e.target.checked })}
                className="w-3 h-3 rounded border-eidos-muted/30 bg-eidos-surface text-eidos-accent focus:ring-eidos-accent"
              />
              <span className="text-xs text-eidos-muted">MA20</span>
            </label>
            <label className="flex items-center gap-1 cursor-pointer">
              <input
                type="checkbox"
                checked={indicators.ma30}
                onChange={(e) => setIndicators({ ...indicators, ma30: e.target.checked })}
                className="w-3 h-3 rounded border-eidos-muted/30 bg-eidos-surface text-eidos-accent focus:ring-eidos-accent"
              />
              <span className="text-xs text-eidos-muted">MA30</span>
            </label>
            <label className="flex items-center gap-1 cursor-pointer">
              <input
                type="checkbox"
                checked={indicators.macd}
                onChange={(e) => setIndicators({ ...indicators, macd: e.target.checked })}
                className="w-3 h-3 rounded border-eidos-muted/30 bg-eidos-surface text-eidos-accent focus:ring-eidos-accent"
              />
              <span className="text-xs text-eidos-muted">MACD</span>
            </label>
            <label className="flex items-center gap-1 cursor-pointer">
              <input
                type="checkbox"
                checked={indicators.rsi}
                onChange={(e) => setIndicators({ ...indicators, rsi: e.target.checked })}
                className="w-3 h-3 rounded border-eidos-muted/30 bg-eidos-surface text-eidos-accent focus:ring-eidos-accent"
              />
              <span className="text-xs text-eidos-muted">RSI</span>
            </label>
            <label className="flex items-center gap-1 cursor-pointer">
              <input
                type="checkbox"
                checked={indicators.bollinger}
                onChange={(e) => setIndicators({ ...indicators, bollinger: e.target.checked })}
                className="w-3 h-3 rounded border-eidos-muted/30 bg-eidos-surface text-eidos-accent focus:ring-eidos-accent"
              />
              <span className="text-xs text-eidos-muted">布林带</span>
            </label>
            <label className="flex items-center gap-1 cursor-pointer">
              <input
                type="checkbox"
                checked={indicators.atr}
                onChange={(e) => setIndicators({ ...indicators, atr: e.target.checked })}
                className="w-3 h-3 rounded border-eidos-muted/30 bg-eidos-surface text-eidos-accent focus:ring-eidos-accent"
              />
              <span className="text-xs text-eidos-muted">ATR</span>
            </label>
          </div>
        </div>

        {/* Chart */}
        <div className="flex-1 p-4 overflow-hidden">
          <div 
            ref={chartRef} 
            className="w-full h-full min-h-[500px]"
            style={{ minHeight: '500px', width: '100%' }}
          />
          {klineData.length === 0 && !loading && (
            <div className="absolute inset-0 flex items-center justify-center text-eidos-muted text-sm">
              暂无K线数据
            </div>
          )}
        </div>

        {/* Trade Summary */}
        <div className="p-4 border-t border-eidos-muted/20">
          <div className="grid grid-cols-3 gap-4 text-xs">
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
    </div>
  )
}

export default StockKlineChart


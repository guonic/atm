import React, { useState, useEffect, useRef } from 'react'
import * as echarts from 'echarts'
import { getStockKline, getTrades } from '@/services/api'
import type { Trade } from '@/types/eidos'
import { calculateMACD, calculateRSI, calculateBollingerBands, calculateATR, calculateSMA } from '@/utils/indicators'

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

function StockKlineChart({ expId, symbol, onClose, embedded = false }: StockKlineChartProps) {
  const [klineData, setKlineData] = useState<KlineData[]>([])
  const [trades, setTrades] = useState<Trade[]>([])
  const [loading, setLoading] = useState(true)
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
  }, [klineData, trades, indicators])  // Re-render when indicators change

  const loadData = async () => {
    try {
      setLoading(true)
      // Load kline data and trades for this symbol
      const [kline, symbolTrades] = await Promise.all([
        getStockKline(expId, symbol),
        getTrades(expId, { symbol }),
      ])
      setKlineData(kline)
      setTrades(symbolTrades)
    } catch (error) {
      console.error('Failed to load kline data:', error)
    } finally {
      setLoading(false)
    }
  }

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
    
    // Calculate indicators if enabled
    const macdData = indicators.macd ? calculateMACD(closes) : null
    const rsiData = indicators.rsi ? calculateRSI(closes) : null
    const bollingerData = indicators.bollinger ? calculateBollingerBands(closes) : null
    const atrData = indicators.atr ? calculateATR(highs, lows, closes) : null
    
    // Calculate moving averages
    const ma5Data = indicators.ma5 ? calculateSMA(closes, 5) : null
    const ma10Data = indicators.ma10 ? calculateSMA(closes, 10) : null
    const ma20Data = indicators.ma20 ? calculateSMA(closes, 20) : null
    const ma30Data = indicators.ma30 ? calculateSMA(closes, 30) : null
    
    // Debug: Log indicator data
    console.log('=== Indicator Calculation Debug ===')
    console.log('Indicators enabled:', indicators)
    console.log('Kline data length:', klineData.length)
    console.log('Closes length:', closes.length)
    
    if (indicators.macd) {
      console.log('MACD enabled, calculating...')
      if (macdData) {
        console.log('MACD Data calculated:', {
          macdLength: macdData.macd.length,
          signalLength: macdData.signal.length,
          histogramLength: macdData.histogram.length,
          macdSample: macdData.macd.filter((v, i) => v !== undefined && i >= 25).slice(0, 5),
          signalSample: macdData.signal.slice(0, 3),
          histogramSample: macdData.histogram.filter((v, i) => v !== undefined && i >= 34).slice(0, 5),
        })
      } else {
        console.warn('MACD data is null!')
      }
    }
    
    if (indicators.rsi) {
      console.log('RSI enabled, calculating...')
      if (rsiData) {
        const rsiValues = rsiData.filter(v => v !== undefined)
        console.log('RSI Data calculated:', {
          rsiLength: rsiData.length,
          rsiValidCount: rsiValues.length,
          rsiSample: rsiValues.slice(0, 5),
        })
      } else {
        console.warn('RSI data is null!')
      }
    }
    
    if (indicators.bollinger) {
      console.log('Bollinger enabled, calculating...')
      if (bollingerData) {
        const upperValues = bollingerData.upper.filter(v => v !== undefined)
        const middleValues = bollingerData.middle.filter(v => v !== undefined)
        const lowerValues = bollingerData.lower.filter(v => v !== undefined)
        console.log('Bollinger Data calculated:', {
          upperLength: bollingerData.upper.length,
          upperValidCount: upperValues.length,
          middleLength: bollingerData.middle.length,
          middleValidCount: middleValues.length,
          lowerLength: bollingerData.lower.length,
          lowerValidCount: lowerValues.length,
          sample: {
            upper: upperValues.slice(0, 3),
            middle: middleValues.slice(0, 3),
            lower: lowerValues.slice(0, 3),
          },
        })
      } else {
        console.warn('Bollinger data is null!')
      }
    }
    
    if (indicators.atr) {
      console.log('ATR enabled, calculating...')
      if (atrData) {
        const atrValues = atrData.filter(v => v !== undefined)
        console.log('ATR Data calculated:', {
          atrLength: atrData.length,
          atrValidCount: atrValues.length,
          atrSample: atrValues.slice(0, 5),
        })
      } else {
        console.warn('ATR data is null!')
      }
    }

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
    const dataZoomTop = currentTop + 5

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
          start: 0,
          end: 100,
        },
        {
          show: true,
          xAxisIndex: xAxes.map((_, i) => i),
          type: 'slider',
          top: `${dataZoomTop}%`,
          start: 0,
          end: 100,
          textStyle: { color: '#8B949E' },
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
        // 布林带（显示在K线图上）
        ...(indicators.bollinger && bollingerData ? (() => {
          // Convert undefined to null for ECharts compatibility
          const upperData = bollingerData.upper.map(v => v !== undefined && !isNaN(v) ? v : null)
          const middleData = bollingerData.middle.map(v => v !== undefined && !isNaN(v) ? v : null)
          const lowerData = bollingerData.lower.map(v => v !== undefined && !isNaN(v) ? v : null)
          
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
          
          macdData.macd.forEach((val, idx) => {
            if (val !== undefined && !isNaN(val) && idx < dates.length) {
              macdLineData[idx] = val
            }
          })
          
          macdData.signal.forEach(({ value, index }) => {
            if (value !== undefined && !isNaN(value) && index < dates.length) {
              signalLineData[index] = value
            }
          })
          
          macdData.histogram.forEach((val, idx) => {
            if (val !== undefined && !isNaN(val) && idx < dates.length) {
              histogramData[idx] = val
            }
          })
          
          const macdValidCount = macdLineData.filter(v => v !== null).length
          const signalValidCount = signalLineData.filter(v => v !== null).length
          const histogramValidCount = histogramData.filter(v => v !== null).length
          console.log('MACD Series Data:', {
            macdValidCount,
            signalValidCount,
            histogramValidCount,
            macdSample: macdLineData.filter(v => v !== null).slice(0, 3),
            signalSample: signalLineData.filter(v => v !== null).slice(0, 3),
          })
          
          return [
            {
              name: 'MACD',
              type: 'line',
              xAxisIndex: macdXAxisIndex,
              yAxisIndex: macdYAxisIndex,
              data: macdLineData,
              lineStyle: { color: '#00F2FF', width: 1.5 },
              symbol: 'none',
            },
            {
              name: 'Signal',
              type: 'line',
              xAxisIndex: macdXAxisIndex,
              yAxisIndex: macdYAxisIndex,
              data: signalLineData,
              lineStyle: { color: '#FF3D00', width: 1.5 },
              symbol: 'none',
            },
            {
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
            },
          ]
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
      console.log('Chart option set successfully', { 
        dataPoints: klineData.length,
        buyMarkers: buyMarkers.length,
        sellMarkers: sellMarkers.length,
        totalSeries: option.series?.length || 0,
        grids: grids.length,
        xAxes: xAxes.length,
        yAxes: yAxes.length,
        legendData: legendData.length,
      })
      console.log('Series names:', option.series?.map((s: any) => s.name))
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
    
    // Remove previous resize handler if exists
    if (resizeHandlerRef.current) {
      window.removeEventListener('resize', resizeHandlerRef.current)
    }
    
    // Add new resize handler and store reference
    window.addEventListener('resize', handleResize)
    resizeHandlerRef.current = handleResize
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
        
        {/* Chart */}
        <div className="p-4" style={{ minHeight: '400px' }}>
          <div 
            ref={chartRef} 
            className="w-full"
            style={{ minHeight: '400px', height: '400px', width: '100%' }}
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


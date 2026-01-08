import { useState, useEffect, useMemo } from 'react'
import { getTrades } from '@/services/api'
import type { Trade } from '@/types/eidos'
import { format } from 'date-fns'
import { BacktestChart } from './backtest/BacktestChart'

interface TradesTableProps {
  expId: string
  symbolFilter?: string
  startDate?: string
  endDate?: string
}

type TabType = 'symbols' | 'pnl'

interface PnlRange {
  label: string
  min: number
  max: number
  symbols: string[]
}

function TradesTable({ expId, symbolFilter, startDate, endDate }: TradesTableProps) {
  const [allTrades, setAllTrades] = useState<Trade[]>([])
  const [loading, setLoading] = useState(true)
  const [page, setPage] = useState(1)
  const [pageSize] = useState(50)
  const [selectedSymbol, setSelectedSymbol] = useState<string | null>(null)
  const [activeTab, setActiveTab] = useState<TabType>('symbols')
  const [selectedPnlRange, setSelectedPnlRange] = useState<PnlRange | null>(null)
  
  // Filter trades by selected symbol or pnl range
  const trades = useMemo(() => {
    if (selectedSymbol) {
      return allTrades.filter(t => t.symbol === selectedSymbol)
    }
    if (selectedPnlRange) {
      return allTrades.filter(t => selectedPnlRange.symbols.includes(t.symbol))
    }
    return allTrades
  }, [allTrades, selectedSymbol, selectedPnlRange])

  useEffect(() => {
    loadTrades()
  }, [expId, symbolFilter, startDate, endDate])

  const loadTrades = async () => {
    try {
      setLoading(true)
      const data = await getTrades(expId, {
        symbol: symbolFilter || undefined,
        start_date: startDate || undefined,
        end_date: endDate || undefined,
      })
      setAllTrades(data)
    } catch (error) {
      console.error('Failed to load trades:', error)
    } finally {
      setLoading(false)
    }
  }
  
  // Reset page when symbol or pnl range changes
  useEffect(() => {
    setPage(1)
  }, [selectedSymbol, selectedPnlRange])

  // Extract unique symbols and calculate statistics
  const symbolStats = useMemo(() => {
    const symbolMap = new Map<string, { trades: Trade[], avgPnl: number | null }>()

    allTrades.forEach(trade => {
      if (!symbolMap.has(trade.symbol)) {
        symbolMap.set(trade.symbol, { trades: [], avgPnl: null })
      }
      symbolMap.get(trade.symbol)!.trades.push(trade)
    })

    // Calculate average PnL for each symbol
    symbolMap.forEach((stats, symbol) => {
      const pnlValues = stats.trades
        .map(t => t.pnl_ratio)
        .filter((pnl): pnl is number => pnl !== undefined && pnl !== null && !isNaN(pnl))

      if (pnlValues.length > 0) {
        stats.avgPnl = pnlValues.reduce((sum, pnl) => sum + pnl, 0) / pnlValues.length
      }
    })

    return Array.from(symbolMap.entries())
      .map(([symbol, stats]) => ({ symbol, ...stats }))
      .sort((a, b) => a.symbol.localeCompare(b.symbol))
  }, [allTrades])

  // Group symbols by PnL ranges
  const pnlRanges = useMemo((): PnlRange[] => {
    const ranges: PnlRange[] = [
      { label: '> 10%', min: 0.10, max: Infinity, symbols: [] },
      { label: '5% - 10%', min: 0.05, max: 0.10, symbols: [] },
      { label: '0% - 5%', min: 0, max: 0.05, symbols: [] },
      { label: '-5% - 0%', min: -0.05, max: 0, symbols: [] },
      { label: '-10% - -5%', min: -0.10, max: -0.05, symbols: [] },
      { label: '< -10%', min: -Infinity, max: -0.10, symbols: [] },
      { label: '无收益率数据', min: NaN, max: NaN, symbols: [] },
    ]

    symbolStats.forEach(({ symbol, avgPnl }) => {
      if (avgPnl === null || isNaN(avgPnl)) {
        ranges[6].symbols.push(symbol)
      } else {
        for (const range of ranges.slice(0, 6)) {
          const minCheck = isFinite(range.min) ? avgPnl > range.min : true
          const maxCheck = isFinite(range.max) ? avgPnl <= range.max : true
          if (minCheck && maxCheck) {
            range.symbols.push(symbol)
            break
          }
        }
      }
    })

    // Sort ranges by PnL (descending)
    return ranges.filter(r => r.symbols.length > 0)
      .sort((a, b) => {
        if (isNaN(a.min) && isNaN(b.min)) return 0
        if (isNaN(a.min)) return 1
        if (isNaN(b.min)) return -1
        return b.min - a.min
      })
  }, [symbolStats])

  // Handle symbol selection
  const handleSymbolClick = (symbol: string) => {
    if (selectedSymbol === symbol) {
      setSelectedSymbol(null)
      setSelectedPnlRange(null)
    } else {
      setSelectedSymbol(symbol)
      setSelectedPnlRange(null)
    }
    setPage(1)
  }

  // Handle PnL range selection
  const handlePnlRangeClick = (range: PnlRange) => {
    if (selectedPnlRange?.label === range.label) {
      setSelectedPnlRange(null)
      setSelectedSymbol(null)
    } else {
      setSelectedPnlRange(range)
      setSelectedSymbol(null)
    }
    setPage(1)
  }

  if (loading) {
    return (
      <div className="flex justify-center items-center h-64">
        <div className="text-eidos-muted">加载中...</div>
      </div>
    )
  }

  if (trades.length === 0) {
    return (
      <div className="flex justify-center items-center h-64">
        <div className="text-eidos-muted">暂无交易数据</div>
      </div>
    )
  }

  // 分页计算
  const totalPages = Math.ceil(trades.length / pageSize)
  const startIndex = (page - 1) * pageSize
  const endIndex = startIndex + pageSize
  const paginatedTrades = trades.slice(startIndex, endIndex)

  return (
    <div className="flex gap-3 h-full">
      {/* 左侧标签页列表 */}
      <div className="w-48 flex-shrink-0 border-r border-eidos-muted/20">
        <div className="space-y-2">
          {/* 标签页按钮 */}
          <div className="flex flex-col gap-1">
            <button
              onClick={() => {
                setActiveTab('symbols')
                setSelectedPnlRange(null)
              }}
              className={`px-3 py-2 text-xs rounded transition-all text-left ${
                activeTab === 'symbols'
                  ? 'bg-eidos-surface/60 text-eidos-gold font-semibold'
                  : 'text-eidos-muted hover:text-white hover:bg-eidos-surface/30'
              }`}
            >
              标的
            </button>
            <button
              onClick={() => {
                setActiveTab('pnl')
                setSelectedSymbol(null)
              }}
              className={`px-3 py-2 text-xs rounded transition-all text-left ${
                activeTab === 'pnl'
                  ? 'bg-eidos-surface/60 text-eidos-gold font-semibold'
                  : 'text-eidos-muted hover:text-white hover:bg-eidos-surface/30'
              }`}
            >
              收益率
            </button>
          </div>

          {/* 标签页内容 */}
          <div className="border-t border-eidos-muted/20 pt-2">
            {activeTab === 'symbols' && (
              <div className="max-h-[calc(100vh-300px)] overflow-y-auto space-y-1">
                {symbolStats.length === 0 ? (
                  <div className="text-xs text-eidos-muted px-3 py-2">暂无标的</div>
                ) : (
                  symbolStats.map(({ symbol, avgPnl }) => (
                    <button
                      key={symbol}
                      onClick={() => handleSymbolClick(symbol)}
                      className={`w-full px-3 py-1.5 text-xs rounded transition-all text-left ${
                        selectedSymbol === symbol
                          ? 'bg-eidos-accent/20 text-eidos-gold font-semibold'
                          : 'text-white hover:bg-eidos-surface/30'
                      }`}
                    >
                      <div className="flex items-center justify-between">
                        <span className="font-mono">{symbol}</span>
                        {avgPnl !== null && (
                          <span className={`text-[10px] ml-2 ${
                            avgPnl >= 0 ? 'text-eidos-accent' : 'text-eidos-danger'
                          }`}>
                            {avgPnl >= 0 ? '+' : ''}{(avgPnl * 100).toFixed(1)}%
                          </span>
                        )}
                      </div>
                    </button>
                  ))
                )}
              </div>
            )}

            {activeTab === 'pnl' && (
              <div className="max-h-[calc(100vh-300px)] overflow-y-auto space-y-1">
                {pnlRanges.length === 0 ? (
                  <div className="text-xs text-eidos-muted px-3 py-2">暂无收益率数据</div>
                ) : (
                  pnlRanges.map((range) => (
                    <button
                      key={range.label}
                      onClick={() => handlePnlRangeClick(range)}
                      className={`w-full px-3 py-2 text-xs rounded transition-all text-left ${
                        selectedPnlRange?.label === range.label
                          ? 'bg-eidos-accent/20 text-eidos-gold font-semibold'
                          : 'text-white hover:bg-eidos-surface/30'
                      }`}
                    >
                      <div className="flex items-center justify-between">
                        <span>{range.label}</span>
                        <span className="text-[10px] text-eidos-muted ml-2">
                          ({range.symbols.length})
                        </span>
                      </div>
                    </button>
                  ))
                )}
              </div>
            )}
          </div>

          {/* 显示选中分段下的标的（仅收益率标签页） */}
          {activeTab === 'pnl' && selectedPnlRange && (
            <div className="border-t border-eidos-muted/20 pt-2 mt-2">
              <div className="px-3 py-1 text-xs text-eidos-muted mb-2">
                该分段下的标的：
              </div>
              <div className="max-h-48 overflow-y-auto space-y-1">
                {selectedPnlRange.symbols.map((symbol) => {
                  const stats = symbolStats.find(s => s.symbol === symbol)
                  return (
                    <button
                      key={symbol}
                      onClick={() => handleSymbolClick(symbol)}
                      className={`w-full px-3 py-1.5 text-xs rounded transition-all text-left ${
                        selectedSymbol === symbol
                          ? 'bg-eidos-accent/20 text-eidos-gold font-semibold'
                          : 'text-white hover:bg-eidos-surface/30'
                      }`}
                    >
                      <div className="flex items-center justify-between">
                        <span className="font-mono">{symbol}</span>
                        {stats?.avgPnl !== null && stats?.avgPnl !== undefined && (
                          <span className={`text-[10px] ml-2 ${
                            stats.avgPnl >= 0 ? 'text-eidos-accent' : 'text-eidos-danger'
                          }`}>
                            {stats.avgPnl >= 0 ? '+' : ''}{(stats.avgPnl * 100).toFixed(1)}%
                          </span>
                        )}
                      </div>
                    </button>
                  )
                })}
              </div>
            </div>
          )}
        </div>
      </div>

      {/* 右侧交易表格 */}
      <div className="flex-1 space-y-2 min-w-0">
        {/* K线图 - 显示在列表上方 */}
        {selectedSymbol && (
          <div className="mb-4">
            <BacktestChart
              expId={expId}
              symbol={selectedSymbol}
              onClose={() => {
                setSelectedSymbol(null)
                setSelectedPnlRange(null)
              }}
              embedded={true}
            />
          </div>
        )}

        {/* 统计信息 */}
        <div className="flex items-center justify-between text-[10px] text-eidos-muted">
          <div>
            共 {trades.length} 笔交易
            {selectedSymbol && <span className="ml-1">（已筛选: {selectedSymbol}）</span>}
            {selectedPnlRange && <span className="ml-1">（已筛选: {selectedPnlRange.label}）</span>}
            {symbolFilter && !selectedSymbol && !selectedPnlRange && <span className="ml-1">（已筛选: {symbolFilter}）</span>}
          </div>
          <div>
            第 {page} / {totalPages} 页
          </div>
        </div>

      {/* 交易表格 */}
      <div className="overflow-x-auto">
        <table className="w-full text-[10px]">
          <thead>
            <tr className="border-b border-eidos-muted/20">
              <th className="text-left py-1 px-1.5 text-[10px] font-semibold text-eidos-muted uppercase tracking-wider">
                时间
              </th>
              <th className="text-left py-1 px-1.5 text-[10px] font-semibold text-eidos-muted uppercase tracking-wider">
                股票代码
              </th>
              <th className="text-right py-1 px-1.5 text-[10px] font-semibold text-eidos-muted uppercase tracking-wider">
                方向
              </th>
              <th className="text-right py-1 px-1.5 text-[10px] font-semibold text-eidos-muted uppercase tracking-wider">
                价格
              </th>
              <th className="text-right py-1 px-1.5 text-[10px] font-semibold text-eidos-muted uppercase tracking-wider">
                数量
              </th>
              <th className="text-right py-1 px-1.5 text-[10px] font-semibold text-eidos-muted uppercase tracking-wider">
                金额
              </th>
              <th className="text-right py-1 px-1.5 text-[10px] font-semibold text-eidos-muted uppercase tracking-wider">
                排名
              </th>
              <th className="text-right py-1 px-1.5 text-[10px] font-semibold text-eidos-muted uppercase tracking-wider">
                分数
              </th>
              <th className="text-left py-1 px-1.5 text-[10px] font-semibold text-eidos-muted uppercase tracking-wider">
                原因
              </th>
            </tr>
          </thead>
          <tbody>
            {paginatedTrades.map((trade) => {
              const directionValue = trade.direction ?? trade.side ?? 1
              const direction = directionValue === 1 ? '买入' : '卖出'
              const directionColor = directionValue === 1 ? 'text-eidos-accent' : 'text-eidos-danger'
              const amount = trade.price * trade.amount

              return (
                <tr
                  key={trade.trade_id}
                  className="border-b border-eidos-muted/10 hover:bg-eidos-surface/30 transition-colors"
                >
                  <td className="py-1 px-1.5 text-[10px] text-white font-mono">
                    {format(new Date(trade.deal_time), 'yyyy-MM-dd HH:mm:ss')}
                  </td>
                  <td 
                    className="py-1 px-1.5 text-[10px] text-white font-mono cursor-pointer hover:text-eidos-accent transition-colors"
                    onClick={() => handleSymbolClick(trade.symbol)}
                    title={selectedSymbol === trade.symbol ? "点击取消筛选" : "点击查看K线图并筛选"}
                  >
                    <span className={selectedSymbol === trade.symbol ? "text-eidos-gold font-bold" : ""}>
                      {trade.symbol}
                    </span>
                  </td>
                  <td className={`py-1 px-1.5 text-[10px] font-medium text-right ${directionColor}`}>
                    {direction}
                  </td>
                  <td className="py-1 px-1.5 text-[10px] text-white font-mono text-right">
                    {trade.price.toFixed(4)}
                  </td>
                  <td className="py-1 px-1.5 text-[10px] text-white font-mono text-right">
                    {trade.amount.toLocaleString()}
                  </td>
                  <td className="py-1 px-1.5 text-[10px] text-white font-mono text-right">
                    {amount.toLocaleString('zh-CN', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                  </td>
                  <td className="py-1 px-1.5 text-[10px] text-eidos-muted font-mono text-right">
                    {trade.rank_at_deal || '-'}
                  </td>
                  <td className="py-1 px-1.5 text-[10px] text-eidos-muted font-mono text-right">
                    {trade.score_at_deal ? trade.score_at_deal.toFixed(4) : '-'}
                  </td>
                  <td className="py-1 px-1.5 text-[10px] text-eidos-muted">
                    {trade.reason || '-'}
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>

      {/* 分页控件 */}
      {totalPages > 1 && (
        <div className="flex items-center justify-between pt-1">
          <button
            onClick={() => setPage(Math.max(1, page - 1))}
            disabled={page === 1}
            className="px-2 py-0.5 rounded text-[10px] text-eidos-muted hover:text-white hover:bg-eidos-surface/50 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
          >
            上一页
          </button>
          <div className="text-[10px] text-eidos-muted">
            第 {page} / {totalPages} 页
          </div>
          <button
            onClick={() => setPage(Math.min(totalPages, page + 1))}
            disabled={page === totalPages}
            className="px-2 py-0.5 rounded text-[10px] text-eidos-muted hover:text-white hover:bg-eidos-surface/50 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
          >
            下一页
          </button>
        </div>
      )}
      </div>
    </div>
  )
}

export default TradesTable


import { useState, useEffect } from 'react'
import { getTrades } from '@/services/api'
import type { Trade } from '@/types/eidos'
import { format } from 'date-fns'
import StockKlineChart from './StockKlineChart'

interface TradesTableProps {
  expId: string
  symbolFilter?: string
  startDate?: string
  endDate?: string
}

function TradesTable({ expId, symbolFilter, startDate, endDate }: TradesTableProps) {
  const [allTrades, setAllTrades] = useState<Trade[]>([])
  const [loading, setLoading] = useState(true)
  const [page, setPage] = useState(1)
  const [pageSize] = useState(50)
  const [selectedSymbol, setSelectedSymbol] = useState<string | null>(null)
  
  // Filter trades by selected symbol
  const trades = selectedSymbol 
    ? allTrades.filter(t => t.symbol === selectedSymbol)
    : allTrades

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
  
  // Reset page when symbol changes
  useEffect(() => {
    setPage(1)
  }, [selectedSymbol])

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
    <div className="space-y-2">
      {/* K线图 - 显示在列表上方 */}
      {selectedSymbol && (
        <div className="mb-4 border border-eidos-muted/20 rounded-lg overflow-hidden">
          <div className="p-2 bg-eidos-surface/50 border-b border-eidos-muted/20 flex items-center justify-between">
            <div>
              <span className="text-sm font-semibold text-eidos-gold">{selectedSymbol}</span>
              <span className="text-xs text-eidos-muted ml-2">K线图与交易点位</span>
            </div>
            <button
              onClick={() => setSelectedSymbol(null)}
              className="text-xs text-eidos-muted hover:text-white transition-colors px-2 py-1 rounded"
            >
              取消筛选
            </button>
          </div>
          <div className="bg-eidos-surface/30">
            <StockKlineChart
              expId={expId}
              symbol={selectedSymbol}
              onClose={() => setSelectedSymbol(null)}
              embedded={true}
            />
          </div>
        </div>
      )}
      
      {/* 统计信息 */}
      <div className="flex items-center justify-between text-[10px] text-eidos-muted">
        <div>
          共 {trades.length} 笔交易
          {selectedSymbol && <span className="ml-1">（已筛选: {selectedSymbol}）</span>}
          {symbolFilter && !selectedSymbol && <span className="ml-1">（已筛选: {symbolFilter}）</span>}
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
                    onClick={() => {
                      if (selectedSymbol === trade.symbol) {
                        setSelectedSymbol(null)
                      } else {
                        setSelectedSymbol(trade.symbol)
                      }
                    }}
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
  )
}

export default TradesTable


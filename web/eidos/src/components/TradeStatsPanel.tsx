import { useState, useEffect } from 'react'
import { getTradeStats } from '@/services/api'
import type { TradeStats } from '@/types/eidos'

interface TradeStatsPanelProps {
  expId: string
}

function TradeStatsPanel({ expId }: TradeStatsPanelProps) {
  const [stats, setStats] = useState<TradeStats | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    loadStats()
  }, [expId])

  const loadStats = async () => {
    try {
      setLoading(true)
      const data = await getTradeStats(expId)
      setStats(data)
    } catch (error) {
      console.error('Failed to load trade stats:', error)
    } finally {
      setLoading(false)
    }
  }

  if (loading) {
    return (
      <div className="bg-eidos-surface/40 glass-effect rounded-xl shadow-lg backdrop-blur-sm p-6">
        <h2 className="text-lg font-semibold mb-4 text-eidos-gold">交易统计</h2>
        <div className="text-eidos-muted">加载中...</div>
      </div>
    )
  }

  if (!stats) {
    return (
      <div className="bg-eidos-surface/40 glass-effect rounded-xl shadow-lg backdrop-blur-sm p-6">
        <h2 className="text-lg font-semibold mb-4 text-eidos-gold">交易统计</h2>
        <div className="text-eidos-muted">暂无数据</div>
      </div>
    )
  }

  return (
    <div className="bg-eidos-surface/40 glass-effect rounded-xl shadow-lg backdrop-blur-sm p-6">
      <h2 className="text-lg font-semibold mb-4 text-eidos-gold">交易统计</h2>
      <div className="grid grid-cols-2 gap-4">
        <div>
          <div className="text-sm text-eidos-muted mb-1">总交易次数</div>
          <div className="text-2xl font-bold font-mono text-white">
            {stats.total_trades}
          </div>
        </div>
        <div>
          <div className="text-sm text-eidos-muted mb-1">买入次数</div>
          <div className="text-2xl font-bold font-mono text-eidos-accent">
            {stats.buy_count}
          </div>
        </div>
        <div>
          <div className="text-sm text-eidos-muted mb-1">卖出次数</div>
          <div className="text-2xl font-bold font-mono text-eidos-danger">
            {stats.sell_count}
          </div>
        </div>
        <div>
          <div className="text-sm text-eidos-muted mb-1">胜率</div>
          <div className="text-2xl font-bold font-mono text-eidos-gold">
            {(stats.win_rate * 100).toFixed(2)}%
          </div>
        </div>
        <div className="col-span-2">
          <div className="text-sm text-eidos-muted mb-1">平均持仓天数</div>
          <div className="text-2xl font-bold font-mono text-white">
            {stats.avg_hold_days.toFixed(1)} 天
          </div>
        </div>
      </div>
    </div>
  )
}

export default TradeStatsPanel


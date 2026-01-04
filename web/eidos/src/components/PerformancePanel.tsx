import { useState, useEffect } from 'react'
import { getPerformanceMetrics } from '@/services/api'
import type { PerformanceMetrics } from '@/types/eidos'

interface PerformancePanelProps {
  expId: string
}

function PerformancePanel({ expId }: PerformancePanelProps) {
  const [metrics, setMetrics] = useState<PerformanceMetrics | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    loadMetrics()
  }, [expId])

  const loadMetrics = async () => {
    try {
      setLoading(true)
      const data = await getPerformanceMetrics(expId)
      setMetrics(data)
    } catch (error) {
      console.error('Failed to load performance metrics:', error)
    } finally {
      setLoading(false)
    }
  }

  if (loading) {
    return (
      <div className="bg-eidos-surface/40 glass-effect rounded-xl shadow-lg backdrop-blur-sm p-3">
        <h2 className="text-sm font-semibold mb-3 text-eidos-gold">性能指标</h2>
        <div className="text-eidos-muted text-xs">加载中...</div>
      </div>
    )
  }

  if (!metrics) {
    return (
      <div className="bg-eidos-surface/40 glass-effect rounded-xl shadow-lg backdrop-blur-sm p-3">
        <h2 className="text-sm font-semibold mb-3 text-eidos-gold">性能指标</h2>
        <div className="text-eidos-muted text-xs">暂无数据</div>
      </div>
    )
  }

  const totalReturnColor = metrics.total_return >= 0 ? 'text-eidos-accent' : 'text-eidos-danger'
  const annualReturnColor = metrics.annual_return && metrics.annual_return >= 0 ? 'text-eidos-accent' : 'text-eidos-danger'

  return (
    <div className="bg-eidos-surface/40 glass-effect rounded-xl shadow-lg backdrop-blur-sm p-3">
      <h2 className="text-sm font-semibold mb-3 text-eidos-gold">性能指标</h2>
      <div className="grid grid-cols-2 gap-3">
        <div>
          <div className="text-xs text-eidos-muted mb-0.5">总收益率</div>
          <div className={`text-lg font-bold font-mono ${totalReturnColor}`}>
            {(metrics.total_return * 100).toFixed(2)}%
          </div>
        </div>
        <div>
          <div className="text-xs text-eidos-muted mb-0.5">最大回撤</div>
          <div className="text-lg font-bold font-mono text-eidos-danger">
            {(metrics.max_drawdown * 100).toFixed(2)}%
          </div>
        </div>
        <div>
          <div className="text-xs text-eidos-muted mb-0.5">最终净值</div>
          <div className="text-lg font-bold font-mono text-white">
            {Number(metrics.final_nav).toFixed(4)}
          </div>
        </div>
        <div>
          <div className="text-xs text-eidos-muted mb-0.5">交易天数</div>
          <div className="text-lg font-bold font-mono text-white">
            {metrics.trading_days}
          </div>
        </div>
        {metrics.sharpe_ratio !== undefined && metrics.sharpe_ratio !== null && (
          <div>
            <div className="text-xs text-eidos-muted mb-0.5">夏普比率</div>
            <div className="text-lg font-bold font-mono text-eidos-gold gold-glow">
              {Number(metrics.sharpe_ratio).toFixed(2)}
            </div>
          </div>
        )}
        {metrics.annual_return !== undefined && metrics.annual_return !== null && (
          <div>
            <div className="text-xs text-eidos-muted mb-0.5">年化收益率</div>
            <div className={`text-lg font-bold font-mono ${annualReturnColor}`}>
              {(Number(metrics.annual_return) * 100).toFixed(2)}%
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

export default PerformancePanel


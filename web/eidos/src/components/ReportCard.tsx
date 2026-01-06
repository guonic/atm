import { useState, useEffect } from 'react'
import { getBacktestReport } from '@/services/api'
import type { BacktestReport } from '@/types/eidos'

interface ReportCardProps {
  expId: string
}

export default function ReportCard({ expId }: ReportCardProps) {
  const [report, setReport] = useState<BacktestReport | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    loadReport()
  }, [expId])

  const loadReport = async () => {
    try {
      setLoading(true)
      const data = await getBacktestReport(expId)
      setReport(data)
    } catch (error) {
      console.error('Failed to load report:', error)
    } finally {
      setLoading(false)
    }
  }

  if (loading) {
    return (
      <div className="bg-eidos-surface/40 glass-effect rounded-xl shadow-lg backdrop-blur-sm p-3">
        <h2 className="text-sm font-semibold mb-2 text-eidos-gold">回测报告</h2>
        <div className="text-eidos-muted text-xs">加载中...</div>
      </div>
    )
  }

  if (!report) {
    return (
      <div className="bg-eidos-surface/40 glass-effect rounded-xl shadow-lg backdrop-blur-sm p-3">
        <h2 className="text-sm font-semibold mb-2 text-eidos-gold">回测报告</h2>
        <div className="text-eidos-muted text-xs">暂无数据</div>
      </div>
    )
  }

  const formatValue = (value: number | null, format?: string, unit?: string): string => {
    if (value === null || value === undefined) return 'N/A'
    if (format) {
      if (format.includes('{:.2f}')) return format.replace('{:.2f}', value.toFixed(2))
      if (format.includes('{:.4f}')) return format.replace('{:.4f}', value.toFixed(4))
      if (format.includes('{:.1f}')) return format.replace('{:.1f}', value.toFixed(1))
      if (format.includes('{:,.2f}')) {
        return format.replace('{:,.2f}', value.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 }))
      }
      if (format.includes('{:.0f}')) return format.replace('{:.0f}', Math.round(value).toString())
    }
    if (unit === '%') return `${(value * 100).toFixed(2)}%`
    return value.toFixed(4)
  }

  const getValueColor = (value: number | null, name: string): string => {
    if (value === null) return 'text-eidos-muted'
    if (name.includes('return') || name.includes('profit')) {
      return value >= 0 ? 'text-eidos-accent' : 'text-eidos-danger'
    }
    if (name.includes('drawdown') || name.includes('loss')) {
      return value <= 0 ? 'text-eidos-accent' : 'text-eidos-danger'
    }
    if (name.includes('sharpe')) {
      return value >= 1 ? 'text-eidos-accent' : value >= 0 ? 'text-eidos-gold' : 'text-eidos-danger'
    }
    return 'text-white'
  }

  // 按分类组织指标
  const portfolioMetrics = report.metrics_by_category.portfolio || []
  const tradingMetrics = report.metrics_by_category.trading || []
  const turnoverMetrics = report.metrics_by_category.turnover || []

  // 提取关键指标
  const getMetric = (name: string) => {
    return report.metrics.find((m) => m.name === name)
  }

  const totalReturn = getMetric('total_return')
  const annualizedReturn = getMetric('annualized_return')
  const sharpeRatio = getMetric('sharpe_ratio')
  const maxDrawdown = getMetric('max_drawdown')
  const winRate = getMetric('win_rate')
  const profitFactor = getMetric('profit_factor')
  const totalTrades = getMetric('total_trades')
  const avgHoldDays = getMetric('avg_hold_days')

  return (
    <div className="bg-eidos-surface/40 glass-effect rounded-xl shadow-lg backdrop-blur-sm p-2">
      <h2 className="text-xs font-semibold mb-1.5 text-eidos-gold">回测报告</h2>
      
      {/* 组合指标组 */}
      <div className="mb-1.5">
        <div className="text-[10px] text-eidos-muted/80 mb-0.5 font-medium">组合指标</div>
        <div className="grid grid-cols-4 gap-0.5 text-[10px]">
          {totalReturn && (
            <div>
              <div className="text-eidos-muted/60 text-[9px]">总收益</div>
              <div className={`font-mono font-bold text-[11px] leading-tight ${getValueColor(totalReturn.value, totalReturn.name)}`}>
                {formatValue(totalReturn.value, totalReturn.format, totalReturn.unit)}
              </div>
            </div>
          )}
          {annualizedReturn && (
            <div>
              <div className="text-eidos-muted/60 text-[9px]">年化</div>
              <div className={`font-mono font-bold text-[11px] leading-tight ${getValueColor(annualizedReturn.value, annualizedReturn.name)}`}>
                {formatValue(annualizedReturn.value, annualizedReturn.format, annualizedReturn.unit)}
              </div>
            </div>
          )}
          {sharpeRatio && (
            <div>
              <div className="text-eidos-muted/60 text-[9px]">夏普</div>
              <div className={`font-mono font-bold text-[11px] leading-tight ${getValueColor(sharpeRatio.value, sharpeRatio.name)}`}>
                {formatValue(sharpeRatio.value, sharpeRatio.format)}
              </div>
            </div>
          )}
          {maxDrawdown && (
            <div>
              <div className="text-eidos-muted/60 text-[9px]">回撤</div>
              <div className={`font-mono font-bold text-[11px] leading-tight ${getValueColor(maxDrawdown.value, maxDrawdown.name)}`}>
                {formatValue(maxDrawdown.value, maxDrawdown.format, maxDrawdown.unit)}
              </div>
            </div>
          )}
        </div>
      </div>

      {/* 交易统计组 */}
      <div>
        <div className="text-[10px] text-eidos-muted/80 mb-0.5 font-medium">交易统计</div>
        <div className="grid grid-cols-4 gap-0.5 text-[10px]">
          {totalTrades && (
            <div>
              <div className="text-eidos-muted/60 text-[9px]">总交易</div>
              <div className="font-mono font-bold text-[11px] leading-tight text-white">
                {formatValue(totalTrades.value, '{:.0f}')}
              </div>
            </div>
          )}
          {winRate && (
            <div>
              <div className="text-eidos-muted/60 text-[9px]">胜率</div>
              <div className="font-mono font-bold text-[11px] leading-tight text-eidos-gold">
                {formatValue(winRate.value, winRate.format, winRate.unit)}
              </div>
            </div>
          )}
          {profitFactor && (
            <div>
              <div className="text-eidos-muted/60 text-[9px]">盈亏比</div>
              <div className="font-mono font-bold text-[11px] leading-tight text-white">
                {formatValue(profitFactor.value, profitFactor.format)}
              </div>
            </div>
          )}
          {avgHoldDays && (
            <div>
              <div className="text-eidos-muted/60 text-[9px]">持仓天数</div>
              <div className="font-mono font-bold text-[11px] leading-tight text-white">
                {formatValue(avgHoldDays.value, avgHoldDays.format)}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}


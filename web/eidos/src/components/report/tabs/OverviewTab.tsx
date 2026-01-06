import type { BacktestReport } from '@/types/eidos'
import MetricCard from '../MetricCard'

interface OverviewTabProps {
  report: BacktestReport
}

export default function OverviewTab({ report }: OverviewTabProps) {
  // Filter key metrics from all metrics
  const keyMetrics = [
    'total_return',
    'annualized_return',
    'sharpe_ratio',
    'max_drawdown',
    'win_rate',
    'profit_factor',
  ]

  const metrics = report.metrics.filter((m) => keyMetrics.includes(m.name))

  return (
    <div>
      <h2 className="text-xl font-bold text-eidos-gold mb-4">关键指标概览</h2>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {metrics.map((metric) => (
          <MetricCard key={metric.name} metric={metric} />
        ))}
      </div>
    </div>
  )
}


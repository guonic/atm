import type { MetricResult } from '@/types/eidos'
import MetricCard from '../MetricCard'

interface TurnoverTabProps {
  metrics: MetricResult[]
}

export default function TurnoverTab({ metrics }: TurnoverTabProps) {
  return (
    <div>
      <h2 className="text-xl font-bold text-eidos-gold mb-4">💰 换手统计</h2>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {metrics.map((metric) => (
          <MetricCard key={metric.name} metric={metric} />
        ))}
      </div>
    </div>
  )
}


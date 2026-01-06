import type { MetricResult } from '@/types/eidos'

interface MetricCardProps {
  metric: MetricResult
}

export default function MetricCard({ metric }: MetricCardProps) {
  const formatValue = (value: number | null, format?: string, unit?: string): string => {
    if (value === null || value === undefined) {
      return 'N/A'
    }

    if (format) {
      // Support simple format strings like "{:.2f}%"
      if (format.includes('{:.2f}')) {
        return format.replace('{:.2f}', value.toFixed(2))
      }
      if (format.includes('{:.4f}')) {
        return format.replace('{:.4f}', value.toFixed(4))
      }
      if (format.includes('{:.1f}')) {
        return format.replace('{:.1f}', value.toFixed(1))
      }
      if (format.includes('{:,.2f}')) {
        return format.replace('{:,.2f}', value.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 }))
      }
    }

    if (unit === '%') {
      return `${(value * 100).toFixed(2)}%`
    }

    return value.toFixed(4)
  }

  const getValueColor = (value: number | null, name: string): string => {
    if (value === null) return 'text-eidos-muted'

    // Set color based on metric type and value
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

  return (
    <div className="bg-eidos-surface/40 glass-effect rounded-xl shadow-lg backdrop-blur-sm p-4">
      <div className="text-xs text-eidos-muted mb-1">{metric.description || metric.name}</div>
      <div className={`text-2xl font-bold font-mono ${getValueColor(metric.value, metric.name)}`}>
        {formatValue(metric.value, metric.format, metric.unit)}
      </div>
      {metric.unit && metric.unit !== '%' && (
        <div className="text-xs text-eidos-muted mt-1">{metric.unit}</div>
      )}
    </div>
  )
}


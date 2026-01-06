import type { BacktestReport } from '@/types/eidos'

interface ReportHeaderProps {
  report: BacktestReport
}

export default function ReportHeader({ report }: ReportHeaderProps) {
  return (
    <div className="bg-eidos-surface/40 glass-effect rounded-xl shadow-lg backdrop-blur-sm p-4 mb-4">
      <h1 className="text-2xl font-bold text-eidos-gold mb-2">{report.experiment_name}</h1>
      <div className="flex gap-4 text-sm text-eidos-muted">
        <div>
          <span className="font-semibold">实验 ID:</span> {report.exp_id}
        </div>
        <div>
          <span className="font-semibold">回测期间:</span> {report.start_date} ~ {report.end_date}
        </div>
        <div>
          <span className="font-semibold">生成时间:</span>{' '}
          {new Date(report.generated_at).toLocaleString()}
        </div>
      </div>
    </div>
  )
}


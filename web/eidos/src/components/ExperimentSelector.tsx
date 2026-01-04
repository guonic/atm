import { Experiment } from '@/types/eidos'

interface ExperimentSelectorProps {
  experiments: Experiment[]
  selectedExpId: string | null
  onSelect: (expId: string) => void
}

function ExperimentSelector({ experiments, selectedExpId, onSelect }: ExperimentSelectorProps) {
  return (
    <div className="bg-eidos-surface glass-effect rounded-lg border border-eidos-muted/20 p-4">
      <label className="block text-sm font-medium text-eidos-muted mb-2">
        选择实验
      </label>
      <select
        value={selectedExpId || ''}
        onChange={(e) => onSelect(e.target.value)}
        className="w-full px-3 py-2 bg-eidos-bg border border-eidos-muted/30 rounded-md text-white focus:outline-none focus:ring-2 focus:ring-eidos-gold focus:border-eidos-gold transition-all"
      >
        <option value="" className="bg-eidos-bg">请选择实验</option>
        {experiments.map((exp) => (
          <option key={exp.exp_id} value={exp.exp_id} className="bg-eidos-bg">
            {exp.name} ({exp.exp_id}) - {exp.start_date} 至 {exp.end_date}
          </option>
        ))}
      </select>
    </div>
  )
}

export default ExperimentSelector


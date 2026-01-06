import { exportReport } from '@/services/api'

interface ReportActionsProps {
  expId: string
}

export default function ReportActions({ expId }: ReportActionsProps) {
  const handleExport = async (format: 'html' | 'markdown') => {
    try {
      const blob = await exportReport(expId, format)
      const url = window.URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `backtest_report_${expId}.${format === 'html' ? 'html' : 'md'}`
      document.body.appendChild(a)
      a.click()
      document.body.removeChild(a)
      window.URL.revokeObjectURL(url)
    } catch (error) {
      console.error('Failed to export report:', error)
      alert('导出失败，请稍后重试')
    }
  }

  return (
    <div className="flex gap-2 mb-4">
      <button
        onClick={() => handleExport('html')}
        className="px-4 py-2 bg-eidos-accent text-white rounded-lg hover:bg-eidos-accent/80 transition-colors"
      >
        📄 导出 HTML
      </button>
      <button
        onClick={() => handleExport('markdown')}
        className="px-4 py-2 bg-eidos-surface text-eidos-gold rounded-lg hover:bg-eidos-surface/80 transition-colors"
      >
        📝 导出 Markdown
      </button>
    </div>
  )
}


import { useState, useEffect } from 'react'
import { getBacktestReport } from '@/services/api'
import type { BacktestReport } from '@/types/eidos'
import ReportHeader from './ReportHeader'
import ReportTabs from './ReportTabs'
import ReportActions from './ReportActions'

interface ReportViewProps {
  expId: string
}

export default function ReportView({ expId }: ReportViewProps) {
  const [report, setReport] = useState<BacktestReport | null>(null)
  const [loading, setLoading] = useState(true)
  const [activeTab, setActiveTab] = useState<string>('overview')

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
      <div className="flex justify-center items-center h-full">
        <div className="text-eidos-muted">加载中...</div>
      </div>
    )
  }

  if (!report) {
    return (
      <div className="flex justify-center items-center h-full">
        <div className="text-eidos-muted">暂无报告数据</div>
      </div>
    )
  }

  return (
    <div className="h-full overflow-y-auto bg-eidos-bg p-4">
      <ReportHeader report={report} />
      <ReportActions expId={expId} />
      <ReportTabs report={report} activeTab={activeTab} onTabChange={setActiveTab} />
    </div>
  )
}


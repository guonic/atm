import { useState, useEffect } from 'react'
import Sidebar from '@/components/Sidebar'
import ConfigPanel from '@/components/ConfigPanel'
import TraceView from '@/components/TraceView'
import ReportView from '@/components/report/ReportView'
import { Experiment } from '@/types/eidos'
import { getExperiments } from '@/services/api'

function Dashboard() {
  const [experiments, setExperiments] = useState<Experiment[]>([])
  const [selectedExpId, setSelectedExpId] = useState<string | null>(null)
  const [selectedSubsystem, setSelectedSubsystem] = useState<string>('trace')
  const [currentModule, setCurrentModule] = useState<string>('overview')
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    loadExperiments()
  }, [])

  const loadExperiments = async () => {
    try {
      setLoading(true)
      const exps = await getExperiments()
      setExperiments(exps)
      if (exps.length > 0 && !selectedExpId) {
        setSelectedExpId(exps[0].exp_id)
      }
    } catch (error) {
      console.error('Failed to load experiments:', error)
    } finally {
      setLoading(false)
    }
  }

  const renderSubsystemContent = () => {
    if (!selectedExpId) {
      return (
        <div className="flex justify-center items-center h-full">
          <div className="text-eidos-muted">请选择一个实验</div>
        </div>
      )
    }

    switch (selectedSubsystem) {
      case 'trace':
        return <TraceView expId={selectedExpId} loading={loading} onModuleChange={setCurrentModule} />
      case 'report':
        return <ReportView expId={selectedExpId} />
      default:
        return (
          <div className="flex justify-center items-center h-full">
            <div className="text-eidos-muted">子系统开发中...</div>
          </div>
        )
    }
  }

  return (
    <div className="flex h-screen overflow-hidden">
      {/* 左侧子系统导航栏 */}
      <Sidebar
        selectedSubsystem={selectedSubsystem}
        onSelectSubsystem={setSelectedSubsystem}
      />

      {/* 中间内容区 */}
      <div className="flex-1 overflow-y-auto bg-eidos-bg">
        {renderSubsystemContent()}
      </div>

      {/* 右侧配置面板 */}
      <ConfigPanel
        experiments={experiments}
        selectedExpId={selectedExpId}
        onSelectExp={setSelectedExpId}
        currentSubsystem={selectedSubsystem}
        currentModule={currentModule}
      />
    </div>
  )
}

export default Dashboard


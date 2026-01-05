import { useState } from 'react'
import PerformancePanel from './PerformancePanel'
import TradeStatsPanel from './TradeStatsPanel'
import NavChart from './NavChart'
import TradesTable from './TradesTable'

interface TraceViewProps {
  expId: string | null
  loading: boolean
}

function TraceView({ expId, loading, onModuleChange }: TraceViewProps) {
  const [selectedModule, setSelectedModule] = useState<string>('overview')

  const handleModuleChange = (module: string) => {
    setSelectedModule(module)
    onModuleChange?.(module)
  }

  // Trace 子系统内的模块
  const modules = [
    { id: 'overview', name: '概览' },
    { id: 'performance', name: '性能分析' },
    { id: 'trades', name: '交易明细' },
    { id: 'attribution', name: '归因分析' },
    { id: 'risk', name: '风险指标' },
  ]

  if (loading) {
    return (
      <div className="flex justify-center items-center h-full">
        <div className="text-eidos-muted">加载中...</div>
      </div>
    )
  }

  if (!expId) {
    return (
      <div className="flex justify-center items-center h-full">
        <div className="text-center">
          <p className="text-eidos-muted text-lg mb-2">请选择一个实验</p>
          <p className="text-eidos-muted text-sm">在右侧配置面板中选择实验开始分析</p>
        </div>
      </div>
    )
  }

  return (
    <div className="h-full flex flex-col">
      {/* Trace 子系统标题和模块导航 */}
      <div className="bg-eidos-surface/30 glass-effect backdrop-blur-sm px-6 py-3 mb-4">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-lg font-bold font-display text-eidos-gold">Trace</h1>
            <p className="text-xs text-eidos-muted mt-0.5">回测追踪分析</p>
          </div>
          <nav className="flex space-x-1">
            {modules.map((module) => (
              <button
                key={module.id}
                onClick={() => handleModuleChange(module.id)}
                className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${
                  selectedModule === module.id
                    ? 'bg-eidos-gold/15 text-eidos-gold shadow-lg shadow-eidos-gold/10'
                    : 'text-eidos-muted hover:bg-eidos-surface/50 hover:text-white'
                }`}
              >
                {module.name}
              </button>
            ))}
          </nav>
        </div>
      </div>

      {/* 内容区域 */}
      <div className="flex-1 overflow-y-auto">
        {selectedModule === 'overview' && (
          <div className="space-y-3 p-3">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              <PerformancePanel expId={expId} />
              <TradeStatsPanel expId={expId} />
            </div>
            <div className="bg-eidos-surface/40 glass-effect rounded-xl shadow-lg backdrop-blur-sm p-3">
              <NavChart expId={expId} />
            </div>
          </div>
        )}

        {selectedModule === 'performance' && (
          <div className="space-y-3 p-3">
            <div className="bg-eidos-surface/40 glass-effect rounded-xl shadow-lg backdrop-blur-sm p-3">
              <NavChart expId={expId} />
            </div>
            <PerformancePanel expId={expId} />
          </div>
        )}

        {selectedModule === 'trades' && (
          <div className="p-3">
            <div className="bg-eidos-surface/40 glass-effect rounded-xl shadow-lg backdrop-blur-sm p-3">
              <TradesTable expId={expId} />
            </div>
          </div>
        )}

        {selectedModule === 'attribution' && (
          <div className="p-6">
            <div className="bg-eidos-surface/40 glass-effect rounded-xl shadow-lg backdrop-blur-sm p-6">
              <h2 className="text-lg font-semibold mb-4 text-eidos-gold">归因分析</h2>
              <p className="text-eidos-muted">归因分析图表（待实现）</p>
            </div>
          </div>
        )}

        {selectedModule === 'risk' && (
          <div className="p-6">
            <div className="bg-eidos-surface/40 glass-effect rounded-xl shadow-lg backdrop-blur-sm p-6">
              <h2 className="text-lg font-semibold mb-4 text-eidos-gold">风险指标</h2>
              <p className="text-eidos-muted">风险指标分析（待实现）</p>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

export default TraceView


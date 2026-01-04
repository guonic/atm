import { useState } from 'react'
import { Experiment } from '@/types/eidos'

interface ConfigPanelProps {
  experiments: Experiment[]
  selectedExpId: string | null
  onSelectExp: (expId: string) => void
  currentSubsystem: string
  currentModule?: string
}

function ConfigPanel({ experiments, selectedExpId, onSelectExp, currentSubsystem, currentModule }: ConfigPanelProps) {
  const [isExpanded, setIsExpanded] = useState(false) // 默认折叠

  return (
    <div className={`bg-eidos-surface/50 glass-effect backdrop-blur-xl h-full flex flex-col transition-all duration-300 ${
      isExpanded ? 'w-80' : 'w-12'
    }`}>
      {/* Header */}
      <div className="p-3 mb-2">
        <div className="flex items-center justify-between">
          {isExpanded && (
            <h2 className="text-sm font-semibold text-eidos-gold">配置</h2>
          )}
          <button
            onClick={() => setIsExpanded(!isExpanded)}
            className={`text-eidos-muted hover:text-eidos-gold transition-colors ${!isExpanded ? 'w-full flex justify-center' : ''}`}
            title={isExpanded ? '折叠' : '展开'}
          >
            {isExpanded ? '◀' : '▶'}
          </button>
        </div>
      </div>

      {/* Content */}
      {isExpanded && (
        <div className="flex-1 overflow-y-auto p-4 space-y-6">
        {/* Experiment Selector */}
        <div>
          <label className="block text-sm font-medium text-eidos-muted mb-2">
            选择实验
          </label>
          <select
            value={selectedExpId || ''}
            onChange={(e) => onSelectExp(e.target.value)}
            className="w-full px-3 py-2 bg-eidos-bg border border-eidos-muted/30 rounded-md text-white text-sm focus:outline-none focus:ring-2 focus:ring-eidos-gold focus:border-eidos-gold transition-all"
          >
            <option value="" className="bg-eidos-bg">请选择实验</option>
            {experiments.map((exp) => (
              <option key={exp.exp_id} value={exp.exp_id} className="bg-eidos-bg">
                {exp.name}
              </option>
            ))}
          </select>
          {selectedExpId && (
            <div className="mt-2 text-xs text-eidos-muted">
              <div>ID: {selectedExpId}</div>
              {experiments.find((e) => e.exp_id === selectedExpId) && (
                <div>
                  {experiments.find((e) => e.exp_id === selectedExpId)?.start_date} -{' '}
                  {experiments.find((e) => e.exp_id === selectedExpId)?.end_date}
                </div>
              )}
            </div>
          )}
        </div>

        {/* Trace 子系统配置 - 根据当前模块显示不同配置 */}
        {currentSubsystem === 'trace' && (
          <>
            {currentModule === 'overview' && (
              <div>
                <h3 className="text-sm font-medium text-eidos-muted mb-3">概览设置</h3>
                <div className="space-y-3">
                  <label className="flex items-center text-sm text-eidos-muted">
                    <input type="checkbox" className="mr-2" defaultChecked />
                    显示性能指标
                  </label>
                  <label className="flex items-center text-sm text-eidos-muted">
                    <input type="checkbox" className="mr-2" defaultChecked />
                    显示交易统计
                  </label>
                  <label className="flex items-center text-sm text-eidos-muted">
                    <input type="checkbox" className="mr-2" />
                    显示风险指标
                  </label>
                </div>
              </div>
            )}

            {currentModule === 'performance' && (
              <div>
                <h3 className="text-sm font-medium text-eidos-muted mb-3">性能分析</h3>
                <div className="space-y-3">
                  <div>
                    <label className="block text-xs text-eidos-muted mb-1">时间范围</label>
                    <select className="w-full px-2 py-1 bg-eidos-bg border border-eidos-muted/30 rounded text-white text-sm">
                      <option>全部</option>
                      <option>最近1个月</option>
                      <option>最近3个月</option>
                      <option>最近6个月</option>
                    </select>
                  </div>
                  <div>
                    <label className="block text-xs text-eidos-muted mb-1">基准对比</label>
                    <select className="w-full px-2 py-1 bg-eidos-bg border border-eidos-muted/30 rounded text-white text-sm">
                      <option>无基准</option>
                      <option>沪深300</option>
                      <option>中证500</option>
                    </select>
                  </div>
                </div>
              </div>
            )}

            {currentModule === 'trades' && (
              <div>
                <h3 className="text-sm font-medium text-eidos-muted mb-3">交易筛选</h3>
                <div className="space-y-3">
                  <div>
                    <label className="block text-xs text-eidos-muted mb-1">股票代码</label>
                    <input
                      type="text"
                      placeholder="如: 000001.SZ"
                      className="w-full px-2 py-1 bg-eidos-bg border border-eidos-muted/30 rounded text-white text-sm placeholder-eidos-muted focus:outline-none focus:ring-2 focus:ring-eidos-gold focus:border-eidos-gold"
                    />
                  </div>
                  <div>
                    <label className="block text-xs text-eidos-muted mb-1">开始日期</label>
                    <input
                      type="date"
                      className="w-full px-2 py-1 bg-eidos-bg border border-eidos-muted/30 rounded text-white text-sm focus:outline-none focus:ring-2 focus:ring-eidos-gold focus:border-eidos-gold"
                    />
                  </div>
                  <div>
                    <label className="block text-xs text-eidos-muted mb-1">结束日期</label>
                    <input
                      type="date"
                      className="w-full px-2 py-1 bg-eidos-bg border border-eidos-muted/30 rounded text-white text-sm focus:outline-none focus:ring-2 focus:ring-eidos-gold focus:border-eidos-gold"
                    />
                  </div>
                </div>
              </div>
            )}

            {currentModule === 'attribution' && (
              <div>
                <h3 className="text-sm font-medium text-eidos-muted mb-3">归因分析</h3>
                <div className="space-y-3">
                  <label className="flex items-center text-sm text-eidos-muted">
                    <input type="checkbox" className="mr-2" defaultChecked />
                    因子暴露度
                  </label>
                  <label className="flex items-center text-sm text-eidos-muted">
                    <input type="checkbox" className="mr-2" defaultChecked />
                    行业归因
                  </label>
                  <label className="flex items-center text-sm text-eidos-muted">
                    <input type="checkbox" className="mr-2" />
                    风格归因
                  </label>
                </div>
              </div>
            )}

            {currentModule === 'risk' && (
              <div>
                <h3 className="text-sm font-medium text-eidos-muted mb-3">风险指标</h3>
                <div className="space-y-3">
                  <label className="flex items-center text-sm text-eidos-muted">
                    <input type="checkbox" className="mr-2" defaultChecked />
                    最大回撤
                  </label>
                  <label className="flex items-center text-sm text-eidos-muted">
                    <input type="checkbox" className="mr-2" defaultChecked />
                    波动率
                  </label>
                  <label className="flex items-center text-sm text-eidos-muted">
                    <input type="checkbox" className="mr-2" />
                    VaR
                  </label>
                </div>
              </div>
            )}
          </>
        )}
        </div>
      )}
    </div>
  )
}

export default ConfigPanel


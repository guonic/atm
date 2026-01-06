interface SidebarProps {
  selectedSubsystem: string
  onSelectSubsystem: (subsystem: string) => void
}

// 子系统列表
const subsystems = [
  { id: 'trace', name: 'Trace', description: '回测追踪分析' },
  { id: 'report', name: 'Report', description: '回测报告' },
  // 未来可以添加更多子系统，如：
  // { id: 'analyze', name: 'Analyze', description: '深度分析' },
  // { id: 'compare', name: 'Compare', description: '对比分析' },
]

function Sidebar({ selectedSubsystem, onSelectSubsystem }: SidebarProps) {
  return (
    <div className="w-20 bg-eidos-surface/50 glass-effect h-full flex flex-col backdrop-blur-xl">
      {/* Logo/Title */}
      <div className="p-2 mb-1">
        <h1 className="text-sm font-bold font-display text-eidos-gold gold-glow">
          EIDOS
        </h1>
      </div>

      {/* Subsystem Navigation */}
      <nav className="flex-1 p-1 space-y-1">
        {subsystems.map((subsystem) => (
          <button
            key={subsystem.id}
            onClick={() => onSelectSubsystem(subsystem.id)}
            className={`w-full px-1 py-1.5 rounded transition-all ${
              selectedSubsystem === subsystem.id
                ? 'bg-eidos-gold/15 text-eidos-gold shadow-lg shadow-eidos-gold/10'
                : 'text-eidos-muted hover:bg-eidos-bg/50 hover:text-white'
            }`}
          >
            <div className="flex flex-col items-center justify-center">
              <div className="text-[10px] font-medium leading-tight text-center">{subsystem.name}</div>
            </div>
          </button>
        ))}
      </nav>

    </div>
  )
}

export default Sidebar


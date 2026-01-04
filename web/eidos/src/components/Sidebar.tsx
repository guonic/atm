interface SidebarProps {
  selectedSubsystem: string
  onSelectSubsystem: (subsystem: string) => void
}

// 子系统列表
const subsystems = [
  { id: 'trace', name: 'Trace', icon: '🔍', description: '回测追踪分析' },
  // 未来可以添加更多子系统，如：
  // { id: 'analyze', name: 'Analyze', icon: '📊', description: '深度分析' },
  // { id: 'compare', name: 'Compare', icon: '⚖️', description: '对比分析' },
]

function Sidebar({ selectedSubsystem, onSelectSubsystem }: SidebarProps) {
  return (
    <div className="w-48 bg-eidos-surface/50 glass-effect h-full flex flex-col backdrop-blur-xl">
      {/* Logo/Title */}
      <div className="p-4 mb-2">
        <h1 className="text-xl font-bold font-display text-eidos-gold gold-glow">
          EIDOS
        </h1>
      </div>

      {/* Subsystem Navigation */}
      <nav className="flex-1 p-3 space-y-1.5">
        {subsystems.map((subsystem) => (
          <button
            key={subsystem.id}
            onClick={() => onSelectSubsystem(subsystem.id)}
            className={`w-full text-left px-3 py-2 rounded-lg transition-all ${
              selectedSubsystem === subsystem.id
                ? 'bg-eidos-gold/15 text-eidos-gold shadow-lg shadow-eidos-gold/10'
                : 'text-eidos-muted hover:bg-eidos-bg/50 hover:text-white'
            }`}
          >
            <div className="flex items-center">
              <span className="mr-2 text-base">{subsystem.icon}</span>
              <div className="flex-1 min-w-0">
                <div className="font-medium text-sm truncate">{subsystem.name}</div>
                <div className="text-xs opacity-70 mt-0.5 truncate">{subsystem.description}</div>
              </div>
            </div>
          </button>
        ))}
      </nav>

    </div>
  )
}

export default Sidebar


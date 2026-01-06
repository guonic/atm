import type { BacktestReport } from '@/types/eidos'
import OverviewTab from './tabs/OverviewTab'
import PortfolioTab from './tabs/PortfolioTab'
import TradingTab from './tabs/TradingTab'
import TurnoverTab from './tabs/TurnoverTab'

interface ReportTabsProps {
  report: BacktestReport
  activeTab: string
  onTabChange: (tab: string) => void
}

const tabs = [
  { id: 'overview', label: '概览', icon: '📊' },
  { id: 'portfolio', label: '组合指标', icon: '💼' },
  { id: 'trading', label: '交易统计', icon: '📈' },
  { id: 'turnover', label: '换手统计', icon: '💰' },
]

export default function ReportTabs({ report, activeTab, onTabChange }: ReportTabsProps) {
  const renderTabContent = () => {
    switch (activeTab) {
      case 'overview':
        return <OverviewTab report={report} />
      case 'portfolio':
        return <PortfolioTab metrics={report.metrics_by_category.portfolio || []} />
      case 'trading':
        return <TradingTab metrics={report.metrics_by_category.trading || []} />
      case 'turnover':
        return <TurnoverTab metrics={report.metrics_by_category.turnover || []} />
      default:
        return null
    }
  }

  return (
    <div>
      {/* Tab navigation */}
      <div className="flex gap-2 mb-4 border-b border-eidos-surface">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            onClick={() => onTabChange(tab.id)}
            className={`px-4 py-2 font-semibold transition-colors ${
              activeTab === tab.id
                ? 'text-eidos-gold border-b-2 border-eidos-gold'
                : 'text-eidos-muted hover:text-eidos-gold'
            }`}
          >
            <span className="mr-2">{tab.icon}</span>
            {tab.label}
          </button>
        ))}
      </div>

      {/* Tab content */}
      <div className="mt-4">{renderTabContent()}</div>
    </div>
  )
}


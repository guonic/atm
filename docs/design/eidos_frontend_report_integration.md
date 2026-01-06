# Eidos 前端回测报告集成设计方案

## 1. 概述

设计如何在前端展示完整的回测报告，整合现有的 PerformancePanel、TradeStatsPanel 等组件，并添加新的报告视图。

## 2. 架构设计

### 2.1 组件层次结构

```
Dashboard
├── Sidebar (子系统导航)
├── TraceView (现有)
│   ├── PerformancePanel (现有 - 性能指标)
│   ├── TradeStatsPanel (现有 - 交易统计)
│   ├── NavChart (现有 - 净值图表)
│   └── TradesTable (现有 - 交易表格)
└── ReportView (新增 - 完整报告视图)
    ├── ReportHeader (报告头部 - 实验信息)
    ├── ReportTabs (报告标签页)
    │   ├── OverviewTab (概览)
    │   ├── PortfolioTab (组合指标)
    │   ├── TradingTab (交易统计)
    │   ├── TurnoverTab (换手统计)
    │   ├── RiskTab (风险指标)
    │   └── ModelTab (模型表现)
    └── ReportActions (报告操作 - 导出等)
```

### 2.2 数据流

```
前端组件
    ↓
API 调用 (services/api.ts)
    ↓
REST API (Python FastAPI)
    ↓
报告生成器 (BacktestReportGenerator)
    ↓
EidosRepo (数据加载)
    ↓
PostgreSQL (Eidos 数据库)
```

## 3. API 接口设计

### 3.1 后端 API 接口

在 `python/nq/api/rest/eidos/routes.py` 中添加：

```python
@router.get("/experiments/{exp_id}/report", response_model=BacktestReportResponse)
async def get_backtest_report(
    exp_id: str,
    format: str = "json",  # json, console, html, markdown
    categories: Optional[str] = None,  # 逗号分隔的指标分类: portfolio,trading,turnover
    metrics: Optional[str] = None,  # 逗号分隔的指标名称: total_return,sharpe_ratio
) -> BacktestReportResponse:
    """
    获取完整的回测报告。
    
    Args:
        exp_id: 实验 ID
        format: 输出格式 (json, console, html, markdown)
        categories: 指标分类筛选
        metrics: 指标名称筛选
    
    Returns:
        回测报告数据
    """
    from nq.analysis.backtest.report import BacktestReportGenerator
    from nq.config import load_config
    
    # 加载配置
    config = load_config("config/config.yaml")
    db_config = config.database
    
    # 创建报告生成器
    generator = BacktestReportGenerator(db_config)
    
    # 生成报告
    report = generator.generate_report(
        exp_id=exp_id,
        config=ReportConfig(
            metric_categories=categories.split(",") if categories else None,
            metric_names=metrics.split(",") if metrics else None,
            output_format=format,
        )
    )
    
    return BacktestReportResponse(**report.to_dict())
```

### 3.2 响应模型

在 `python/nq/api/rest/eidos/schemas.py` 中添加：

```python
class MetricResultResponse(BaseModel):
    """指标结果响应"""
    name: str
    category: str
    value: Optional[float] = None
    unit: Optional[str] = None
    format: Optional[str] = None
    description: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

class BacktestReportResponse(BaseModel):
    """回测报告响应"""
    exp_id: str
    experiment_name: str
    start_date: str
    end_date: str
    generated_at: str
    metrics: List[MetricResultResponse]
    # 按分类组织的指标（便于前端展示）
    metrics_by_category: Dict[str, List[MetricResultResponse]]
```

## 4. 前端类型定义

### 4.1 更新 `web/eidos/src/types/eidos.ts`

```typescript
// 指标结果
export interface MetricResult {
  name: string
  category: string
  value: number | null
  unit?: string
  format?: string
  description?: string
  metadata?: Record<string, any>
}

// 回测报告
export interface BacktestReport {
  exp_id: string
  experiment_name: string
  start_date: string
  end_date: string
  generated_at: string
  metrics: MetricResult[]
  metrics_by_category: Record<string, MetricResult[]>
}

// 报告配置
export interface ReportConfig {
  format?: 'json' | 'console' | 'html' | 'markdown'
  categories?: string[]
  metrics?: string[]
}
```

## 5. 前端 API 服务

### 5.1 更新 `web/eidos/src/services/api.ts`

```typescript
/**
 * 获取完整的回测报告
 */
export async function getBacktestReport(
  expId: string,
  config?: ReportConfig
): Promise<BacktestReport> {
  const params: Record<string, string> = {}
  
  if (config?.format) {
    params.format = config.format
  }
  if (config?.categories) {
    params.categories = config.categories.join(',')
  }
  if (config?.metrics) {
    params.metrics = config.metrics.join(',')
  }
  
  const response = await api.get<BacktestReport>(`/experiments/${expId}/report`, { params })
  return response.data
}

/**
 * 导出报告（HTML/Markdown）
 */
export async function exportReport(
  expId: string,
  format: 'html' | 'markdown'
): Promise<Blob> {
  const response = await api.get(`/experiments/${expId}/report`, {
    params: { format },
    responseType: 'blob',
  })
  return response.data
}
```

## 6. 前端组件设计

### 6.1 ReportView 主组件

```typescript
// web/eidos/src/components/report/ReportView.tsx

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
    return <div className="text-eidos-muted">加载中...</div>
  }

  if (!report) {
    return <div className="text-eidos-muted">暂无报告数据</div>
  }

  return (
    <div className="h-full overflow-y-auto bg-eidos-bg p-4">
      <ReportHeader report={report} />
      <ReportActions expId={expId} />
      <ReportTabs
        report={report}
        activeTab={activeTab}
        onTabChange={setActiveTab}
      />
    </div>
  )
}
```

### 6.2 ReportHeader 组件

```typescript
// web/eidos/src/components/report/ReportHeader.tsx

import type { BacktestReport } from '@/types/eidos'

interface ReportHeaderProps {
  report: BacktestReport
}

export default function ReportHeader({ report }: ReportHeaderProps) {
  return (
    <div className="bg-eidos-surface/40 glass-effect rounded-xl shadow-lg backdrop-blur-sm p-4 mb-4">
      <h1 className="text-2xl font-bold text-eidos-gold mb-2">
        {report.experiment_name}
      </h1>
      <div className="flex gap-4 text-sm text-eidos-muted">
        <div>
          <span className="font-semibold">实验 ID:</span> {report.exp_id}
        </div>
        <div>
          <span className="font-semibold">回测期间:</span>{' '}
          {report.start_date} ~ {report.end_date}
        </div>
        <div>
          <span className="font-semibold">生成时间:</span>{' '}
          {new Date(report.generated_at).toLocaleString()}
        </div>
      </div>
    </div>
  )
}
```

### 6.3 ReportTabs 组件

```typescript
// web/eidos/src/components/report/ReportTabs.tsx

import { useState } from 'react'
import type { BacktestReport } from '@/types/eidos'
import OverviewTab from './tabs/OverviewTab'
import PortfolioTab from './tabs/PortfolioTab'
import TradingTab from './tabs/TradingTab'
import TurnoverTab from './tabs/TurnoverTab'
import RiskTab from './tabs/RiskTab'
import ModelTab from './tabs/ModelTab'

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
  { id: 'risk', label: '风险指标', icon: '⚠️' },
  { id: 'model', label: '模型表现', icon: '🤖' },
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
      case 'risk':
        return <RiskTab metrics={report.metrics_by_category.risk || []} />
      case 'model':
        return <ModelTab metrics={report.metrics_by_category.model || []} />
      default:
        return null
    }
  }

  return (
    <div>
      {/* 标签页导航 */}
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

      {/* 标签页内容 */}
      <div className="mt-4">{renderTabContent()}</div>
    </div>
  )
}
```

### 6.4 OverviewTab 组件（概览）

```typescript
// web/eidos/src/components/report/tabs/OverviewTab.tsx

import type { BacktestReport } from '@/types/eidos'
import MetricCard from '../MetricCard'

interface OverviewTabProps {
  report: BacktestReport
}

export default function OverviewTab({ report }: OverviewTabProps) {
  // 从所有指标中筛选关键指标
  const keyMetrics = [
    'total_return',
    'annualized_return',
    'sharpe_ratio',
    'max_drawdown',
    'win_rate',
    'profit_factor',
  ]

  const metrics = report.metrics.filter((m) => keyMetrics.includes(m.name))

  return (
    <div>
      <h2 className="text-xl font-bold text-eidos-gold mb-4">关键指标概览</h2>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {metrics.map((metric) => (
          <MetricCard key={metric.name} metric={metric} />
        ))}
      </div>
    </div>
  )
}
```

### 6.5 MetricCard 组件（指标卡片）

```typescript
// web/eidos/src/components/report/MetricCard.tsx

import type { MetricResult } from '@/types/eidos'

interface MetricCardProps {
  metric: MetricResult
}

export default function MetricCard({ metric }: MetricCardProps) {
  const formatValue = (value: number | null, format?: string, unit?: string): string => {
    if (value === null || value === undefined) {
      return 'N/A'
    }

    if (format) {
      // 支持简单的格式化字符串，如 "{:.2f}%"
      const formatted = format.replace('{:.2f}', value.toFixed(2))
      return formatted
    }

    if (unit === '%') {
      return `${(value * 100).toFixed(2)}%`
    }

    return value.toFixed(4)
  }

  const getValueColor = (value: number | null, name: string): string => {
    if (value === null) return 'text-eidos-muted'
    
    // 根据指标类型和值设置颜色
    if (name.includes('return') || name.includes('profit')) {
      return value >= 0 ? 'text-eidos-accent' : 'text-eidos-danger'
    }
    if (name.includes('drawdown') || name.includes('loss')) {
      return value <= 0 ? 'text-eidos-accent' : 'text-eidos-danger'
    }
    if (name.includes('sharpe')) {
      return value >= 1 ? 'text-eidos-accent' : value >= 0 ? 'text-eidos-gold' : 'text-eidos-danger'
    }
    
    return 'text-white'
  }

  return (
    <div className="bg-eidos-surface/40 glass-effect rounded-xl shadow-lg backdrop-blur-sm p-4">
      <div className="text-xs text-eidos-muted mb-1">
        {metric.description || metric.name}
      </div>
      <div className={`text-2xl font-bold font-mono ${getValueColor(metric.value, metric.name)}`}>
        {formatValue(metric.value, metric.format, metric.unit)}
      </div>
      {metric.unit && metric.unit !== '%' && (
        <div className="text-xs text-eidos-muted mt-1">{metric.unit}</div>
      )}
    </div>
  )
}
```

### 6.6 PortfolioTab 组件（组合指标）

```typescript
// web/eidos/src/components/report/tabs/PortfolioTab.tsx

import type { MetricResult } from '@/types/eidos'
import MetricCard from '../MetricCard'

interface PortfolioTabProps {
  metrics: MetricResult[]
}

export default function PortfolioTab({ metrics }: PortfolioTabProps) {
  return (
    <div>
      <h2 className="text-xl font-bold text-eidos-gold mb-4">📊 组合指标</h2>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {metrics.map((metric) => (
          <MetricCard key={metric.name} metric={metric} />
        ))}
      </div>
    </div>
  )
}
```

### 6.7 ReportActions 组件（报告操作）

```typescript
// web/eidos/src/components/report/ReportActions.tsx

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
```

## 7. 集成到 Dashboard

### 7.1 更新 Dashboard.tsx

```typescript
// web/eidos/src/pages/Dashboard.tsx

import ReportView from '@/components/report/ReportView'

function Dashboard() {
  // ... 现有代码 ...

  const renderSubsystemContent = () => {
    switch (selectedSubsystem) {
      case 'trace':
        return <TraceView expId={selectedExpId} loading={loading} onModuleChange={setCurrentModule} />
      case 'report':  // 新增报告子系统
        return <ReportView expId={selectedExpId} />
      default:
        return (
          <div className="flex justify-center items-center h-full">
            <div className="text-eidos-muted">子系统开发中...</div>
          </div>
        )
    }
  }

  // ... 其余代码 ...
}
```

### 7.2 更新 Sidebar.tsx

```typescript
// 在 Sidebar 中添加报告选项
const subsystems = [
  { id: 'trace', label: '追踪视图', icon: '🔍' },
  { id: 'report', label: '回测报告', icon: '📊' },  // 新增
]
```

## 8. 样式设计

### 8.1 使用现有的 Eidos 设计系统

- **颜色**：使用 `eidos-gold`、`eidos-accent`、`eidos-danger` 等
- **效果**：使用 `glass-effect`、`backdrop-blur-sm` 等
- **布局**：使用 Tailwind CSS 的 grid 和 flex 布局

### 8.2 响应式设计

- 移动端：单列布局
- 平板：2 列布局
- 桌面：3 列布局

## 9. 使用流程

1. **用户选择实验**：在 ConfigPanel 中选择实验
2. **切换到报告视图**：在 Sidebar 中选择"回测报告"
3. **查看报告**：ReportView 自动加载并展示报告
4. **切换标签页**：查看不同分类的指标
5. **导出报告**：点击导出按钮，下载 HTML 或 Markdown 格式

## 10. 文件结构

```
web/eidos/src/
├── components/
│   ├── report/                    # 新增：报告组件目录
│   │   ├── ReportView.tsx        # 主报告视图
│   │   ├── ReportHeader.tsx      # 报告头部
│   │   ├── ReportTabs.tsx        # 标签页导航
│   │   ├── ReportActions.tsx     # 报告操作
│   │   ├── MetricCard.tsx        # 指标卡片
│   │   └── tabs/                 # 标签页内容
│   │       ├── OverviewTab.tsx   # 概览
│   │       ├── PortfolioTab.tsx  # 组合指标
│   │       ├── TradingTab.tsx    # 交易统计
│   │       ├── TurnoverTab.tsx   # 换手统计
│   │       ├── RiskTab.tsx       # 风险指标
│   │       └── ModelTab.tsx      # 模型表现
│   └── ... (现有组件)
├── services/
│   └── api.ts                    # 更新：添加报告 API
└── types/
    └── eidos.ts                  # 更新：添加报告类型
```

## 11. 实现优先级

### Phase 1: 基础功能
1. 后端 API 接口（`/experiments/{exp_id}/report`）
2. 前端类型定义
3. ReportView 主组件
4. OverviewTab 概览标签页
5. MetricCard 指标卡片

### Phase 2: 完整功能
1. 所有标签页组件（Portfolio, Trading, Turnover, Risk, Model）
2. 报告导出功能
3. 集成到 Dashboard

### Phase 3: 增强功能
1. 指标对比（多实验对比）
2. 图表可视化（使用现有 NavChart）
3. 报告缓存和性能优化

## 12. 注意事项

1. **向后兼容**：保持与现有 PerformancePanel、TradeStatsPanel 的兼容
2. **数据一致性**：确保报告数据与现有组件数据一致
3. **错误处理**：处理数据缺失、API 错误等情况
4. **性能优化**：对于大量指标，考虑虚拟滚动或分页
5. **用户体验**：提供加载状态、错误提示、空状态等


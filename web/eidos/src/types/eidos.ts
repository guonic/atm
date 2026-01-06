/**
 * Eidos 归因系统类型定义
 */

export interface Experiment {
  exp_id: string
  name: string
  model_type?: string
  engine_type?: string
  start_date: string
  end_date: string
  config: Record<string, any>
  metrics_summary?: Record<string, any>
  status: 'running' | 'completed' | 'failed'
  created_at: string
}

export interface LedgerEntry {
  exp_id: string
  date: string
  nav: number
  cash?: number
  market_value?: number
  deal_amount?: number
  turnover_rate?: number
  pos_count?: number
}

export interface Trade {
  trade_id: number
  exp_id: string
  symbol: string
  deal_time: string
  side?: 1 | -1  // 1=Buy, -1=Sell (legacy field name)
  direction?: 1 | -1  // 1=Buy, -1=Sell (preferred field name)
  price: number
  amount: number
  rank_at_deal?: number
  score_at_deal?: number
  reason?: string
  pnl_ratio?: number
  hold_days?: number
}

export interface ModelOutput {
  exp_id: string
  date: string
  symbol: string
  score: number
  rank: number
  extra_scores?: Record<string, any>
}

export interface PerformanceMetrics {
  total_return: number
  max_drawdown: number
  final_nav: number
  trading_days: number
  sharpe_ratio?: number
  annual_return?: number
}

export interface TradeStats {
  total_trades: number
  buy_count: number
  sell_count: number
  win_rate: number
  avg_hold_days: number
}

// Report types
export interface MetricResult {
  name: string
  category: string
  value: number | null  // Can be int or float
  unit?: string
  format?: string
  description?: string
  metadata?: Record<string, any>
}

export interface BacktestReport {
  exp_id: string
  experiment_name: string
  start_date: string
  end_date: string
  generated_at: string
  metrics: MetricResult[]
  metrics_by_category: Record<string, MetricResult[]>
}

export interface ReportConfig {
  format?: 'json' | 'console' | 'html' | 'markdown'
  categories?: string[]
  metrics?: string[]
}


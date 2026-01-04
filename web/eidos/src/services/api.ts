import axios from 'axios'
import type { Experiment, LedgerEntry, Trade, PerformanceMetrics, TradeStats } from '@/types/eidos'

const api = axios.create({
  baseURL: '/api/v1',
  timeout: 10000,
  headers: {
    'Content-Type': 'application/json',
  },
})

/**
 * 获取所有实验列表
 */
export async function getExperiments(): Promise<Experiment[]> {
  const response = await api.get<Experiment[]>('/experiments')
  return response.data
}

/**
 * 获取单个实验信息
 */
export async function getExperiment(expId: string): Promise<Experiment> {
  const response = await api.get<Experiment>(`/experiments/${expId}`)
  return response.data
}

/**
 * 获取实验的账户流水
 */
export async function getLedger(expId: string): Promise<LedgerEntry[]> {
  const response = await api.get<LedgerEntry[]>(`/experiments/${expId}/ledger`)
  return response.data
}

/**
 * 获取实验的交易记录
 */
export async function getTrades(
  expId: string,
  params?: { symbol?: string; start_date?: string; end_date?: string }
): Promise<Trade[]> {
  const response = await api.get<Trade[]>(`/experiments/${expId}/trades`, { params })
  return response.data
}

/**
 * 获取实验的性能指标
 */
export async function getPerformanceMetrics(expId: string): Promise<PerformanceMetrics> {
  const response = await api.get<PerformanceMetrics>(`/experiments/${expId}/metrics`)
  return response.data
}

/**
 * 获取实验的交易统计
 */
export async function getTradeStats(expId: string): Promise<TradeStats> {
  const response = await api.get<TradeStats>(`/experiments/${expId}/trade-stats`)
  return response.data
}

export default api


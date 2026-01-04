import { useState, useEffect } from 'react'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'
import { getLedger } from '@/services/api'
import type { LedgerEntry } from '@/types/eidos'
import { format } from 'date-fns'

interface NavChartProps {
  expId: string
}

function NavChart({ expId }: NavChartProps) {
  const [data, setData] = useState<Array<{ date: string; nav: number }>>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    loadData()
  }, [expId])

  const loadData = async () => {
    try {
      setLoading(true)
      const ledger = await getLedger(expId)
      const chartData = ledger
        .sort((a, b) => new Date(a.date).getTime() - new Date(b.date).getTime())
        .map((entry) => ({
          date: format(new Date(entry.date), 'yyyy-MM-dd'),
          nav: entry.nav,
        }))
      setData(chartData)
    } catch (error) {
      console.error('Failed to load ledger data:', error)
    } finally {
      setLoading(false)
    }
  }

  if (loading) {
    return (
      <div>
        <h2 className="text-lg font-semibold mb-4 text-eidos-gold">净值曲线</h2>
        <div className="text-eidos-muted">加载中...</div>
      </div>
    )
  }

  if (data.length === 0) {
    return (
      <div>
        <h2 className="text-lg font-semibold mb-4 text-eidos-gold">净值曲线</h2>
        <div className="text-eidos-muted">暂无数据</div>
      </div>
    )
  }

  return (
    <div>
      <h2 className="text-lg font-semibold mb-4 text-eidos-gold">净值曲线</h2>
      <ResponsiveContainer width="100%" height={400}>
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="#8B949E" opacity={0.2} />
          <XAxis 
            dataKey="date" 
            tick={{ fontSize: 12, fill: '#8B949E' }}
            angle={-45}
            textAnchor="end"
            height={80}
            stroke="#8B949E"
          />
          <YAxis 
            tick={{ fontSize: 12, fill: '#8B949E' }}
            label={{ value: '净值', angle: -90, position: 'insideLeft', fill: '#8B949E' }}
            stroke="#8B949E"
          />
          <Tooltip 
            contentStyle={{
              backgroundColor: '#161B22',
              border: '1px solid #8B949E',
              borderRadius: '8px',
              color: '#fff',
            }}
            formatter={(value: number) => value.toFixed(4)}
            labelFormatter={(label) => `日期: ${label}`}
          />
          <Legend 
            wrapperStyle={{ color: '#8B949E' }}
          />
          <Line 
            type="monotone" 
            dataKey="nav" 
            stroke="#00F2FF" 
            strokeWidth={2}
            name="净值"
            dot={false}
            activeDot={{ r: 4, fill: '#00F2FF' }}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}

export default NavChart


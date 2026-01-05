# Stock Chart 组件迁移指南

## 概述

原有的 `StockKlineChart` 组件已被重构为两个独立的组件：
1. **StockChart** - 标准股票分析面板（可复用）
2. **BacktestChart** - 回测专用图表（组合 StockChart + 回测功能）

## 组件架构

```
StockChart (标准股票分析面板)
├── K线图 + 成交量
├── 技术指标系统（20+ 指标）
├── 时间区间选择（1m, 5m, 15m, 30m, 60m, 1d, 1w, 1M）
├── 面板放大缩小
└── 窗口拖动控制器

BacktestChart (回测图表)
├── 使用 StockChart 作为基础
├── 回测区间高亮
├── 买卖点标记
└── 交易统计信息
```

## 迁移步骤

### 1. 更新导入

**旧代码：**
```typescript
import StockKlineChart from './StockKlineChart'
```

**新代码：**
```typescript
import { BacktestChart } from './backtest/BacktestChart'
// 或
import { StockChart } from './stock/StockChart'
```

### 2. 更新组件使用

**旧代码：**
```tsx
<StockKlineChart
  expId={expId}
  symbol={symbol}
  onClose={onClose}
  embedded={embedded}
/>
```

**新代码（回测场景）：**
```tsx
<BacktestChart
  expId={expId}
  symbol={symbol}
  onClose={onClose}
  embedded={embedded}
/>
```

**新代码（通用股票分析）：**
```tsx
<StockChart
  symbol={symbol}
  klineData={klineData}
  indicatorData={indicatorData}
  indicators={indicators}
  onIndicatorsChange={setIndicators}
  timeInterval="1d"
  onTimeIntervalChange={setTimeInterval}
  embedded={embedded}
/>
```

## 已完成的迁移

### ✅ TradesTable.tsx
- 已从 `StockKlineChart` 迁移到 `BacktestChart`
- 位置：`web/eidos/src/components/TradesTable.tsx`

## 功能对比

| 功能 | StockKlineChart (旧) | StockChart (新) | BacktestChart (新) |
|------|---------------------|----------------|-------------------|
| K线图 | ✅ | ✅ | ✅ |
| 成交量 | ✅ | ✅ | ✅ |
| 技术指标 | 部分（8个） | 全部（20+个） | 全部（20+个） |
| 时间区间选择 | ❌ | ✅ | ❌（固定日线） |
| 面板放大缩小 | ❌ | ✅ | ✅ |
| 窗口拖动 | ✅ | ✅ | ✅ |
| 回测区间高亮 | ✅ | ✅（可选） | ✅ |
| 买卖点标记 | ✅ | ✅（可选） | ✅ |
| 交易统计 | ✅ | ❌ | ✅ |

## 新功能

### StockChart 新增功能

1. **完整的技术指标支持**
   - 趋势类：MA5/10/20/30/60/120, EMA, WMA
   - 震荡类：RSI, KDJ, CCI, WR, OBV
   - 趋势+震荡：MACD, DMI
   - 通道类：Bollinger Bands, Envelope
   - 波动类：ATR, BBW
   - 成交量类：VWAP

2. **时间区间选择**
   - 支持：1m, 5m, 15m, 30m, 60m, 1d, 1w, 1M
   - 可通过 `onTimeIntervalChange` 回调处理切换

3. **面板放大缩小**
   - 点击最小化按钮可折叠面板
   - 再次点击可展开

4. **可选的回测叠加**
   - 通过 `backtestOverlay` prop 传入回测相关数据
   - 支持回测区间高亮和买卖点标记

### BacktestChart 特性

1. **自动数据加载**
   - 自动从 API 加载 K 线数据和交易记录
   - 自动计算技术指标

2. **回测专用**
   - 时间区间固定为日线（不可更改）
   - 自动显示回测区间和买卖点

3. **交易统计**
   - 显示买入次数、卖出次数、总交易次数

## 组件位置

- **StockChart**: `web/eidos/src/components/stock/StockChart.tsx`
- **BacktestChart**: `web/eidos/src/components/backtest/BacktestChart.tsx`
- **类型定义**: `web/eidos/src/components/stock/StockChart.types.ts`
- **配置**: `web/eidos/src/components/stock/StockChart.config.ts`

## 废弃组件

- **StockKlineChart**: `web/eidos/src/components/StockKlineChart.tsx`
  - 已标记为 `@deprecated`
  - 保留用于向后兼容
  - 建议尽快迁移到新组件

## 注意事项

1. **时间区间选择**：`BacktestChart` 固定为日线，不支持切换（回测场景需求）
2. **指标计算**：所有指标计算在后端完成，前端仅负责渲染
3. **数据格式**：新组件使用统一的类型定义，确保类型安全

## 后续计划

- [ ] 移除废弃的 `StockKlineChart` 组件（在确认所有引用已迁移后）
- [ ] 添加更多时间区间支持（需要后端 API 支持）
- [ ] 优化指标渲染性能


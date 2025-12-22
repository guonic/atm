# 日线布林带策略使用指南

## 策略概述

日线布林带策略是一个中期交易系统，通过分析日线级别的布林带和成交量，识别中期交易机会。适合波段交易。

## 核心优势

1. **中期持仓**：适合波段交易，持仓时间1-20天
2. **信号稳定**：日线级别信号相对稳定，减少假信号
3. **过滤假信号**：通过成交量验证和布林带拐头检测，减少无效交易
4. **自动止盈止损**：自动限制持仓时间，避免被套

## 策略原理

### 为什么选择日线级别？

- **60分钟级别**：信号频繁，适合短期交易，但需要频繁操作
- **日线级别**：信号稳定，适合中期波段交易，减少操作频率
- **周线级别**：信号太慢，可能错过最佳入场时机

### 布林带分析

布林带由三条线组成：
- **上轨**：价格上限，通常表示超买区域
- **中轨**：移动平均线，表示价格趋势
- **下轨**：价格下限，通常表示超卖区域

## 买入信号（下轨+缩量）

**所有条件必须同时满足**：

1. **K线到达下轨**
   - 日线K线跌到布林带下轨
   - 表明股票超卖

2. **成交量萎缩**
   - 当日成交量 < 前5日均量的60%
   - 表明卖压大幅减少，没人卖了

3. **下轨向上拐头**
   - 下轨"向上拐头"（关键！）
   - 不能是连续下跌的下轨
   - 确保是真正的反弹信号，而不是继续下跌

## 卖出信号（上轨+放量）

**所有条件必须同时满足**：

1. **K线到达上轨**
   - 日线K线涨到布林带上轨
   - 表明股票超买

2. **成交量放大**
   - 当日成交量 > 前5日均量的150%
   - 表明主力开始砸盘

3. **上轨向下拐头**
   - 上轨"向下拐头"（关键！）
   - 不能是连续上涨的上轨
   - 确保是真正的反转信号，而不是继续上涨

## 避坑点

### 1. 避免突发利空

**问题**：如果有突发利空（如公司暴雷），下轨可能被直接击穿，信号失效。

**解决方案**：
- 策略会自动检测下轨拐头，避免在连续下跌中买入
- 建议结合基本面分析，避免有重大负面消息的股票

### 2. 中期持仓限制

**问题**：日线级别行情一般持续1-20个交易日，持仓太久可能错过更好的机会或被套。

**解决方案**：
- 策略默认最大持仓时间为20天
- 超过20天自动强制卖出
- 可通过 `--max-holding-days` 调整

## 使用示例

### 基本使用

```bash
python python/tools/analysis/evaluate_bollinger_daily_strategy.py \
    --start-date 2023-01-01 \
    --end-date 2024-01-01 \
    --num-stocks 100 \
    --skip-market-cap-filter
```

**注意**：使用此策略前，需要先同步日线K线数据（通常已经同步）：

```bash
python python/tools/dataingestor/sync_kline.py --type day
```

### 自定义布林带参数

```bash
python python/tools/analysis/evaluate_bollinger_daily_strategy.py \
    --start-date 2023-01-01 \
    --end-date 2024-01-01 \
    --num-stocks 100 \
    --skip-market-cap-filter \
    --period 20 \
    --devfactor 2.0
```

### 调整成交量阈值

```bash
python python/tools/analysis/evaluate_bollinger_daily_strategy.py \
    --start-date 2023-01-01 \
    --end-date 2024-01-01 \
    --num-stocks 100 \
    --skip-market-cap-filter \
    --volume-threshold-buy 0.5 \
    --volume-threshold-sell 1.8
```

### 调整持仓时间限制

```bash
python python/tools/analysis/evaluate_bollinger_daily_strategy.py \
    --start-date 2023-01-01 \
    --end-date 2024-01-01 \
    --num-stocks 100 \
    --skip-market-cap-filter \
    --max-holding-days 15
```

### 调整价格容差

```bash
python python/tools/analysis/evaluate_bollinger_daily_strategy.py \
    --start-date 2023-01-01 \
    --end-date 2024-01-01 \
    --num-stocks 100 \
    --skip-market-cap-filter \
    --band-tolerance 0.015
```

## 策略逻辑

### 买入流程

1. 检测K线是否在下轨附近（容差范围内）
2. 验证成交量是否萎缩（< 60%均量）
3. 检测下轨是否向上拐头（关键！）
4. 全部通过后买入

### 卖出流程

1. 检测K线是否在上轨附近（容差范围内）
2. 验证成交量是否放大（> 150%均量）
3. 检测上轨是否向下拐头（关键！）
4. 全部通过后卖出

### 持仓时间限制

- 每次买入时记录入场时间
- 每个bar检查持仓时间
- 如果持仓时间 >= 最大持仓天数，强制卖出

### 布林带拐头检测

**下轨向上拐头**：
- 当前下轨值 > 前一个下轨值
- 前一个下轨值 <= 前两个下轨值（之前在下行或持平）

**上轨向下拐头**：
- 当前上轨值 < 前一个上轨值
- 前一个上轨值 >= 前两个上轨值（之前在上升或持平）

## 参数说明

### 布林带参数

- `period`: 布林带周期（默认：20）
- `devfactor`: 布林带标准差倍数（默认：2.0）

### 成交量参数

- `volume_ma_period`: 成交量均线周期（默认：5）
- `volume_threshold_buy`: 买入成交量阈值（默认：0.6 = 60%）
- `volume_threshold_sell`: 卖出成交量阈值（默认：1.5 = 150%）

### 持仓参数

- `max_holding_days`: 最大持仓天数（默认：20）
- `band_tolerance`: 价格容差（默认：0.01 = 1%）

## 策略优势

1. **中期持仓**：适合波段交易，减少操作频率
2. **信号稳定**：日线级别信号相对稳定
3. **过滤假信号**：成交量验证和拐头检测
4. **避免被套**：自动限制持仓时间（20天）
5. **及时止盈**：上轨放量拐头及时卖出

## 策略局限性

1. **信号滞后**：相比60分钟级别，信号可能滞后
2. **市场环境**：在极端市场环境下可能失效
3. **突发利空**：无法完全避免突发利空的影响
4. **持仓限制**：可能错过一些长期上涨的机会

## 与60分钟策略对比

| 特性 | 日线策略 | 60分钟策略 |
|------|---------|-----------|
| 持仓时间 | 1-20天 | 1-3天 |
| 信号频率 | 较低 | 较高 |
| 操作频率 | 较低 | 较高 |
| 适合人群 | 波段交易者 | 短线交易者 |
| 数据需求 | 日线数据（通常已有） | 60分钟数据（需同步） |

## 注意事项

1. **数据准备**：确保日线K线数据已同步
2. **突发利空**：避免在有重大负面消息的股票上使用
3. **持仓时间**：严格遵守20天持仓限制，不要贪心
4. **拐头检测**：确保布林带真正拐头，不是假信号

## 数据同步

在使用策略前，确保日线K线数据已同步：

```bash
# 同步日线K线数据
python python/tools/dataingestor/sync_kline.py --type day

# 同步指定日期范围的数据
python python/tools/dataingestor/sync_kline.py \
    --type day \
    --start-date 20230101 \
    --end-date 20241231
```

## 对比测试

建议同时运行日线布林带和60分钟布林带进行对比：

```bash
# 日线布林带（默认参数）
python python/tools/analysis/evaluate_bollinger_daily_strategy.py \
    --start-date 2023-01-01 \
    --end-date 2024-01-01 \
    --num-stocks 100 \
    --skip-market-cap-filter \
    --output results_daily_bb.csv

# 60分钟布林带（默认参数）
python python/tools/analysis/evaluate_bollinger_60min_strategy.py \
    --start-date 2023-01-01 \
    --end-date 2024-01-01 \
    --num-stocks 100 \
    --skip-market-cap-filter \
    --output results_60min_bb.csv
```

## 风险提示

本文内容仅为技术分析分享，不构成任何投资建议。市场有风险，投资需谨慎。

实际操作中，建议先模拟演练1-2个月，熟悉策略特性，再逐步实盘应用。同时记住，日线级别行情一般持续1-20个交易日，不要贪心，严格遵守持仓时间限制。


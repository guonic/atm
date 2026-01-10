# ATR 策略使用指南

## 策略概述

基于 ATR (Average True Range) 的交易策略，实现了文档中提到的所有核心功能：

1. **ATR 止损**：动态止损，根据市场波动调整
2. **ATR 阶梯止盈**：分阶段锁定利润
3. **ATR 仓位管理**：根据风险百分比计算仓位
4. **ATR 趋势识别**：识别趋势加速和变盘信号
5. **ATR 突破确认**：过滤假突破

## 核心功能

### 1. ATR 止损

**公式**：
- 做多：止损位 = 入场价格 - 1.5×ATR
- 做空：止损位 = 入场价格 + 1.5×ATR

**特点**：
- 根据市场波动动态调整
- 避免固定百分比止损的局限性
- 适应不同品种的波动特性

### 2. ATR 阶梯止盈

**策略**：
- **第一目标**：入场价 + 2×ATR (减仓30%)
- **第二目标**：入场价 + 3×ATR (再减仓30%)
- **最终止损**：移动至成本价 + 1×ATR

**优势**：
- 分阶段锁定利润
- 让剩余仓位充分奔跑
- 保护已实现利润

### 3. ATR 仓位管理

**公式**：
```
仓位 = 账户总风险的1% ÷ (1.5×ATR × 每点价值)
```

**示例**：
- 10万元账户，单笔愿意亏损1000元(1%)
- 该股ATR为0.5元
- 1.5×ATR = 0.75元
- 可买入股数 = 1000 ÷ 0.75 ≈ 1300股

**效果**：
- 无论个股波动大小，单笔亏损都严格可控
- 自动适应不同波动率的股票

### 4. ATR 趋势识别

**规则**：
- **ATR 值持续扩大**：表明趋势加速
- **ATR 值持续收缩**：预示变盘在即

**应用**：
- 趋势加速时加大仓位
- ATR 收缩时准备退出

### 5. ATR 突破确认

**规则**：
- **真实突破**：价格突破关键位且 ATR 同步放大
- **假突破**：价格突破但 ATR 无明显变化

**效果**：
- 过滤假突破信号
- 提高交易质量

## 参数说明

### 核心参数

- `atr_period`: ATR 计算周期（默认：14）
- `stop_loss_multiplier`: 止损倍数（默认：1.5）
- `take_profit_1_multiplier`: 第一止盈倍数（默认：2.0）
- `take_profit_2_multiplier`: 第二止盈倍数（默认：3.0）
- `take_profit_1_size`: 第一目标减仓比例（默认：0.3 = 30%）
- `take_profit_2_size`: 第二目标减仓比例（默认：0.3 = 30%）
- `risk_per_trade`: 单笔风险百分比（默认：0.01 = 1%）

### 趋势过滤参数

- `ma_period`: 移动平均周期（默认：20）
- `use_trend_filter`: 启用趋势过滤（默认：True）
- `use_breakout_confirmation`: 启用突破确认（默认：True）
- `atr_expansion_threshold`: ATR 扩张阈值（默认：1.1）

## 使用示例

### 基本使用

```bash
python python/tools/analysis/evaluate_atr_strategy.py \
    --start-date 2023-01-01 \
    --end-date 2024-01-01 \
    --num-stocks 100 \
    --skip-market-cap-filter
```

### 自定义参数

```bash
python python/tools/analysis/evaluate_atr_strategy.py \
    --start-date 2023-01-01 \
    --end-date 2024-01-01 \
    --num-stocks 100 \
    --skip-market-cap-filter \
    --atr-period 20 \
    --stop-loss-multiplier 1.5 \
    --take-profit-1-multiplier 2.0 \
    --take-profit-2-multiplier 3.0 \
    --risk-per-trade 0.01 \
    --ma-period 30
```

### 高风险配置（适合高波动股票）

```bash
python python/tools/analysis/evaluate_atr_strategy.py \
    --start-date 2023-01-01 \
    --end-date 2024-01-01 \
    --num-stocks 100 \
    --skip-market-cap-filter \
    --stop-loss-multiplier 1.2 \
    --take-profit-1-multiplier 2.5 \
    --take-profit-2-multiplier 4.0 \
    --risk-per-trade 0.02
```

### 低风险配置（适合低波动股票）

```bash
python python/tools/analysis/evaluate_atr_strategy.py \
    --start-date 2023-01-01 \
    --end-date 2024-01-01 \
    --num-stocks 100 \
    --skip-market-cap-filter \
    --stop-loss-multiplier 2.0 \
    --take-profit-1-multiplier 1.5 \
    --take-profit-2-multiplier 2.5 \
    --risk-per-trade 0.005
```

### 禁用趋势过滤（更多交易机会）

```bash
python python/tools/analysis/evaluate_atr_strategy.py \
    --start-date 2023-01-01 \
    --end-date 2024-01-01 \
    --num-stocks 100 \
    --skip-market-cap-filter \
    --no-trend-filter
```

## 策略逻辑

### 买入信号

1. **趋势过滤**：价格在移动平均线之上（如果启用）
2. **ATR 扩张**：ATR 值持续扩大（趋势加速）
3. **突破确认**：价格突破 MA 且 ATR 同步放大（如果启用）

### 卖出信号

1. **止损触发**：价格触及止损位
2. **止盈触发**：价格触及止盈目标
3. **趋势反转**：价格跌破移动平均线
4. **ATR 收缩**：ATR 值持续收缩（变盘信号）

### 仓位管理

- 根据风险百分比自动计算仓位
- 确保单笔亏损不超过账户的指定百分比
- 自动适应不同波动率的股票

## 进阶应用

### 初级阶段

- 统一使用 1.5×ATR 止损
- 固定参数，简单易用

### 中级阶段

- 个性化参数调整
- 高波动品种用 1.2×ATR
- 低波动品种用 2×ATR

### 高级阶段

- 动态 ATR 系统
- 根据市场波动率动态调整系数
- 结合其他指标综合判断

## 注意事项

1. **技术指标不是盈利保证**：ATR 同样存在局限性
2. **极端单边市**：在极端单边市中，ATR 可能失效
3. **综合判断**：应结合其他指标综合判断
4. **市场有风险**：投资需谨慎

## 对比测试

建议同时运行多个配置进行对比：

```bash
# 默认配置
python python/tools/analysis/evaluate_atr_strategy.py \
    --start-date 2023-01-01 \
    --end-date 2024-01-01 \
    --num-stocks 100 \
    --skip-market-cap-filter \
    --output results_atr_default.csv

# 高风险配置
python python/tools/analysis/evaluate_atr_strategy.py \
    --start-date 2023-01-01 \
    --end-date 2024-01-01 \
    --num-stocks 100 \
    --skip-market-cap-filter \
    --stop-loss-multiplier 1.2 \
    --risk-per-trade 0.02 \
    --output results_atr_high_risk.csv

# 低风险配置
python python/tools/analysis/evaluate_atr_strategy.py \
    --start-date 2023-01-01 \
    --end-date 2024-01-01 \
    --num-stocks 100 \
    --skip-market-cap-filter \
    --stop-loss-multiplier 2.0 \
    --risk-per-trade 0.005 \
    --output results_atr_low_risk.csv
```

## 策略优势

1. **动态止损**：根据市场波动自动调整，避免被震荡出局
2. **阶梯止盈**：分阶段锁定利润，让利润充分奔跑
3. **风险控制**：严格的仓位管理，单笔亏损可控
4. **趋势识别**：利用 ATR 识别趋势加速和变盘信号
5. **突破确认**：过滤假突破，提高交易质量

## 策略局限性

1. **震荡市场**：在震荡市场中可能频繁止损
2. **极端行情**：在极端单边市中可能失效
3. **参数敏感**：需要根据不同市场环境调整参数
4. **滞后性**：ATR 基于历史数据，存在滞后性


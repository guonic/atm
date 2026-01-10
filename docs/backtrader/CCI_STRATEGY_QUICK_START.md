# CCI 优化策略快速使用指南

## 问题：优化后没有交易？

如果优化后的策略没有产生交易，说明过滤条件太严格。以下是解决方案：

## 解决方案

### 方案 1：禁用部分过滤器（推荐）

默认情况下，只有**趋势过滤器**是启用的，其他过滤器都已禁用。如果仍然没有交易，可以禁用趋势过滤器：

```bash
python python/tools/analysis/evaluate_cci_strategy_optimized.py \
    --start-date 2023-01-01 \
    --end-date 2024-01-01 \
    --num-stocks 100 \
    --skip-market-cap-filter \
    --no-trend-filter  # 禁用趋势过滤器
```

### 方案 2：调整 MA 周期

如果趋势过滤器太严格，可以减小 MA 周期：

```bash
python python/tools/analysis/evaluate_cci_strategy_optimized.py \
    --start-date 2023-01-01 \
    --end-date 2024-01-01 \
    --num-stocks 100 \
    --skip-market-cap-filter \
    --ma-period 20  # 从默认的 30 减小到 20
```

### 方案 3：完全禁用所有过滤器（最宽松）

如果仍然没有交易，可以完全禁用所有过滤器，只保留止损止盈：

```bash
python python/tools/analysis/evaluate_cci_strategy_optimized.py \
    --start-date 2023-01-01 \
    --end-date 2024-01-01 \
    --num-stocks 100 \
    --skip-market-cap-filter \
    --no-trend-filter \
    # 注意：volume 和 atr 过滤器默认已禁用
```

## 默认配置说明

优化策略的默认配置：

- ✅ **趋势过滤器**：启用（MA 周期 = 30）
- ❌ **成交量过滤器**：禁用
- ❌ **ATR 波动率过滤器**：禁用
- ✅ **止损止盈**：启用（止损 5%，止盈 15%）

这意味着默认情况下，策略只需要：
1. 价格在 MA 之上（趋势过滤）
2. CCI 信号（超卖反弹、零轴穿越或背离）

## 参数调整建议

### 如果交易太少：

1. **禁用趋势过滤器**：`--no-trend-filter`
2. **减小 MA 周期**：`--ma-period 20` 或更小
3. **降低成交量阈值**：`--volume-threshold 0.8`（如果启用了成交量过滤）

### 如果交易太多（胜率低）：

1. **启用趋势过滤器**：默认已启用
2. **增加 MA 周期**：`--ma-period 50` 或更大
3. **启用成交量过滤器**：`--use-volume-filter --volume-threshold 1.2`
4. **启用 ATR 过滤器**：`--use-atr-filter --atr-threshold 0.5`

## 推荐配置

### 保守配置（高质量信号，交易较少）

```bash
python python/tools/analysis/evaluate_cci_strategy_optimized.py \
    --start-date 2023-01-01 \
    --end-date 2024-01-01 \
    --num-stocks 100 \
    --skip-market-cap-filter \
    --ma-period 50 \
    --use-volume-filter \
    --volume-threshold 1.2 \
    --use-atr-filter \
    --atr-threshold 0.5
```

### 平衡配置（默认，推荐）

```bash
python python/tools/analysis/evaluate_cci_strategy_optimized.py \
    --start-date 2023-01-01 \
    --end-date 2024-01-01 \
    --num-stocks 100 \
    --skip-market-cap-filter
```

### 激进配置（更多交易，可能胜率较低）

```bash
python python/tools/analysis/evaluate_cci_strategy_optimized.py \
    --start-date 2023-01-01 \
    --end-date 2024-01-01 \
    --num-stocks 100 \
    --skip-market-cap-filter \
    --no-trend-filter \
    --ma-period 20
```

## 调试技巧

如果仍然没有交易，可以：

1. **检查日志**：查看是否有 "Buy signal" 的调试日志
2. **检查数据**：确保数据时间范围足够长
3. **检查 CCI 值**：确认 CCI 指标是否正常计算
4. **逐步放宽条件**：从最严格的配置开始，逐步放宽

## 对比测试

建议同时运行原始版本和优化版本进行对比：

```bash
# 原始版本（无过滤器）
python python/tools/analysis/evaluate_cci_strategy.py \
    --start-date 2023-01-01 \
    --end-date 2024-01-01 \
    --num-stocks 100 \
    --skip-market-cap-filter \
    --output results_original.csv

# 优化版本（默认配置）
python python/tools/analysis/evaluate_cci_strategy_optimized.py \
    --start-date 2023-01-01 \
    --end-date 2024-01-01 \
    --num-stocks 100 \
    --skip-market-cap-filter \
    --output results_optimized.csv

# 优化版本（无趋势过滤器）
python python/tools/analysis/evaluate_cci_strategy_optimized.py \
    --start-date 2023-01-01 \
    --end-date 2024-01-01 \
    --num-stocks 100 \
    --skip-market-cap-filter \
    --no-trend-filter \
    --output results_optimized_no_filter.csv
```


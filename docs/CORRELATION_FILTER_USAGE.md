# 相关性过滤策略使用指南

## 概述

相关性过滤器（`CorrelationFilter`）是一个可选的股票选择过滤器，可以在 `StructureExpert` 模型生成的排序基础上进行二次过滤。它使用相关性算法来识别具有高相关性的股票，从而提升买入标的的质量。

## 功能特点

1. **支持多种相关性算法**：
   - `DynamicCrossSectional`: 动态截面相关性
   - `CrossLagged`: 滞后相关性
   - `VolatilitySync`: 波动率同步
   - `GrangerCausality`: 格兰杰因果检验
   - `TransferEntropy`: 转移熵

2. **灵活的过滤策略**：
   - 支持平均相关性或最小相关性阈值
   - 可配置窗口参数和阈值
   - 自动处理数据加载和计算

3. **无缝集成**：
   - 作为可选参数集成到 `AsymmetricStrategy`
   - 不影响原有策略逻辑
   - 失败时自动回退到原始排序

## 使用方法

### 1. 创建相关性过滤器

```python
from nq.analysis.correlation import CorrelationFilter

# 创建过滤器
correlation_filter = CorrelationFilter(
    algorithm='DynamicCrossSectional',  # 相关性算法
    window=5,                           # 窗口参数
    threshold=0.7,                      # 相关性阈值
    use_average_correlation=True,      # 使用平均相关性
    min_stocks=2,                      # 最少股票数量
)
```

### 2. 集成到策略中

```python
from nq.trading.strategies import AsymmetricStrategy
from nq.trading.strategies.buy_models import StructureExpertBuyModel
from nq.trading.strategies.sell_models import MLExitSellModel

# 创建买入模型
buy_model = StructureExpertBuyModel(
    model_path="models/structure_expert_directional.pth",
    builder=graph_builder,
)

# 创建卖出模型
sell_model = MLExitSellModel(...)

# 创建策略（带相关性过滤器）
strategy = AsymmetricStrategy(
    buy_model=buy_model,
    sell_model=sell_model,
    position_manager=position_manager,
    order_book=order_book,
    risk_manager=risk_manager,
    position_allocator=position_allocator,
    account=account,
    correlation_filter=correlation_filter,  # 添加相关性过滤器
)
```

### 3. 配置参数说明

#### `algorithm` (str)
相关性算法类型，可选值：
- `'DynamicCrossSectional'`: 动态截面相关性（默认）
- `'CrossLagged'`: 滞后相关性
- `'VolatilitySync'`: 波动率同步（需要 highs/lows 数据）
- `'GrangerCausality'`: 格兰杰因果检验
- `'TransferEntropy'`: 转移熵

#### `window` (int)
相关性计算的窗口参数：
- `DynamicCrossSectional`: 回看窗口天数（默认：5）
- `CrossLagged`: 滞后天数（默认：5）
- `VolatilitySync`: 带宽参数（window / 100.0）
- `GrangerCausality`: 最大滞后阶数（默认：5）
- `TransferEntropy`: 分箱数量（默认：5）

#### `threshold` (float)
相关性阈值，只有相关性 >= threshold 的股票会被保留（默认：0.7）

#### `use_average_correlation` (bool)
- `True`: 使用平均相关性（股票与其他股票的平均相关性）
- `False`: 使用最小相关性（更严格的过滤）

#### `min_stocks` (int)
进行相关性计算所需的最少股票数量（默认：2）

## 工作原理

1. **生成排序**：`StructureExpert` 模型生成股票排序
2. **加载历史数据**：自动加载历史收益率数据（默认回看 60 天）
3. **计算相关性矩阵**：使用指定的相关性算法计算股票间的相关性
4. **应用阈值过滤**：保留相关性 >= threshold 的股票
5. **生成买入信号**：基于过滤后的股票列表生成买入订单

## 示例：完整回测流程

```python
from nq.trading.backtest import run_custom_backtest
from nq.analysis.correlation import CorrelationFilter

# 创建相关性过滤器
correlation_filter = CorrelationFilter(
    algorithm='DynamicCrossSectional',
    window=5,
    threshold=0.7,
)

# 创建策略
strategy = AsymmetricStrategy(
    buy_model=buy_model,
    sell_model=sell_model,
    position_manager=position_manager,
    order_book=order_book,
    risk_manager=risk_manager,
    position_allocator=position_allocator,
    account=account,
    correlation_filter=correlation_filter,
)

# 运行回测
results = run_custom_backtest(
    strategy=strategy,
    start_date="2025-01-01",
    end_date="2025-12-31",
    initial_cash=1000000.0,
    instruments=None,  # 使用所有可用股票
)
```

## 性能考虑

1. **数据加载**：相关性过滤器需要加载历史收益率数据，可能增加计算时间
2. **缓存优化**：可以考虑缓存历史数据以减少重复加载
3. **并行计算**：对于大量股票，可以考虑并行计算相关性矩阵

## 故障处理

- **数据不足**：如果历史数据不足，过滤器会自动跳过，使用原始排序
- **计算失败**：如果相关性计算失败，策略会回退到原始排序，不会中断交易
- **日志记录**：所有过滤操作都会记录到日志中，便于调试

## 最佳实践

1. **参数调优**：根据回测结果调整窗口参数和阈值
2. **算法选择**：不同算法适用于不同市场环境，可以通过 `CorrelationOptimizer` 进行评测
3. **组合使用**：可以结合 `CorrelationOptimizer` 的评测结果选择最优参数组合
4. **监控效果**：定期检查过滤器的效果，确保提升策略表现

## 相关文档

- [相关性算法文档](CORRELATION_ANALYSIS.md)
- [相关性测试设计](docs/algo/correlation_test_design.md)
- [相关性优化器使用](docs/algo/correlation_test_design_feasibility.md)

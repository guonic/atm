# 相关性分析模块使用说明

## 目录结构

```
python/nq/analysis/correlation/
├── __init__.py              # 模块导出
├── correlation.py           # 五种相关性计算方法
└── graph_builder.py         # 增强的图构建器
```

## 快速开始

### 1. 动态截面相关性

```python
from nq.analysis.correlation import DynamicCrossSectionalCorrelation

# 初始化计算器
correlation_calc = DynamicCrossSectionalCorrelation(
    window=60,           # 回看窗口 60 天
    threshold=0.5,      # 相关性阈值（可选）
)

# 计算相关性矩阵
returns_df = ...  # DataFrame: columns=symbols, index=dates, values=returns
corr_matrix = correlation_calc.calculate(returns_df)

# 计算特定股票对的相关性
corr_value = correlation_calc.calculate_pairwise(
    returns_df['000001.SZ'],
    returns_df['000002.SZ']
)
```

### 2. 滞后相关性

```python
from nq.analysis.correlation import CrossLaggedCorrelation

# 初始化计算器
lagged_calc = CrossLaggedCorrelation(lag=1)

# 计算有向相关性
corr_i_to_j, corr_j_to_i, direction = lagged_calc.calculate_directed(
    returns_df['000001.SZ'],
    returns_df['000002.SZ']
)
# direction: 'i->j', 'j->i', or None

# 计算所有股票对的滞后相关性矩阵
corr_matrix, direction_matrix = lagged_calc.calculate_matrix(returns_df)
```

### 3. 协同波动率

```python
from nq.analysis.correlation import VolatilitySync

# 初始化计算器
volatility_calc = VolatilitySync(bandwidth=0.1)

# 计算波动率同步矩阵
highs_df = ...  # DataFrame with high prices
lows_df = ...   # DataFrame with low prices
sync_matrix = volatility_calc.calculate_matrix(highs_df, lows_df)

# 计算特定股票对的同步率
range_i = volatility_calc.calculate_range(highs_df['000001.SZ'], lows_df['000001.SZ'])
range_j = volatility_calc.calculate_range(highs_df['000002.SZ'], lows_df['000002.SZ'])
sync_rate = volatility_calc.calculate_sync_rate(range_i, range_j)
```

### 4. 格兰杰因果检验

```python
from nq.analysis.correlation import GrangerCausality

# 初始化计算器（需要 statsmodels）
granger_calc = GrangerCausality(
    maxlag=2,
    significance_level=0.05
)

# 检验 A 是否导致 B
is_causal, p_value, direction = granger_calc.test(
    returns_df['000001.SZ'],
    returns_df['000002.SZ']
)

# 计算所有股票对的因果矩阵
causality_matrix, p_value_matrix = granger_calc.calculate_matrix(returns_df)
```

### 5. 传递熵

```python
from nq.analysis.correlation import TransferEntropy

# 初始化计算器
te_calc = TransferEntropy(
    n_bins=3,
    threshold_percentile=(33.3, 66.7)
)

# 计算传递熵
te_a_to_b, te_b_to_a, direction = te_calc.calculate(
    returns_df['000001.SZ'],
    returns_df['000002.SZ']
)

# 计算所有股票对的传递熵矩阵
te_matrix, direction_matrix = te_calc.calculate_matrix(returns_df)
```

### 6. 增强的图构建器

```python
from nq.analysis.correlation import EnhancedGraphDataBuilder

# 初始化图构建器
builder = EnhancedGraphDataBuilder(
    industry_map=industry_map,      # 行业映射（可选）
    correlation_window=60,
    correlation_threshold=0.5,
    lag=1,
    use_granger=True,
    use_transfer_entropy=True,
)

# 构建带有多维边特征的图
graph = builder.build_graph(
    node_features=features_df,      # 节点特征
    returns=returns_df,              # 收益率数据
    highs=highs_df,                  # 最高价（可选）
    lows=lows_df,                    # 最低价（可选）
    use_industry_edges=True,
    use_correlation_edges=True,
    correlation_threshold=0.3,
)

# graph.edge_attr 包含 4 维边特征：
# [correlation, lagged_weight, volatility_sync, transfer_entropy]
```

## 边特征向量

增强图构建器生成的边特征向量包含 4 个维度：

```python
Edge_Feature_ij = [
    correlation_value,      # 截面联动（动态截面相关性）
    lagged_weight,         # 滞后权重（滞后相关性）
    volatility_sync_score, # 风险同步（协同波动率）
    transfer_entropy       # 信息流向强度（传递熵）
]
```

## 性能优化建议

1. **稀疏化图**：使用 `correlation_threshold` 过滤弱相关边
2. **KNN 剪枝**：对每个节点只保留 top-K 最相关的边
3. **批量计算**：使用 `calculate_matrix()` 方法批量计算，避免循环调用
4. **缓存结果**：相关性矩阵可以缓存，避免重复计算

## 依赖要求

- **必需**：`numpy`, `pandas`
- **可选**：`statsmodels`（用于格兰杰因果检验）

安装可选依赖：
```bash
pip install statsmodels
```

## 注意事项

1. **数据要求**：确保收益率数据对齐，缺失值已处理
2. **计算复杂度**：O(n²) 复杂度，对于大量股票可能需要较长时间
3. **统计显著性**：格兰杰因果检验需要足够的数据点（建议 > 50）
4. **传递熵**：符号化过程可能丢失部分信息，适合捕捉非线性关系

## 集成到 GNN 模型

```python
from nq.analysis.correlation import EnhancedGraphDataBuilder
from tools.qlib.train.structure_expert import StructureExpertGNN

# 构建增强图
builder = EnhancedGraphDataBuilder(industry_map=industry_map)
graph = builder.build_graph(node_features, returns, highs, lows)

# 使用图进行模型推理
model = StructureExpertGNN(n_feat=158)
logits, embeddings = model(graph.x, graph.edge_index)
```

## 示例：完整工作流

```python
import pandas as pd
from nq.analysis.correlation import EnhancedGraphDataBuilder

# 1. 准备数据
returns_df = ...  # 收益率数据
highs_df = ...    # 最高价数据
lows_df = ...     # 最低价数据
features_df = ... # 节点特征数据
industry_map = ... # 行业映射

# 2. 构建增强图
builder = EnhancedGraphDataBuilder(
    industry_map=industry_map,
    correlation_window=60,
    correlation_threshold=0.3,
    use_granger=True,
    use_transfer_entropy=True,
)

graph = builder.build_graph(
    node_features=features_df,
    returns=returns_df,
    highs=highs_df,
    lows=lows_df,
    use_industry_edges=True,
    use_correlation_edges=True,
    correlation_threshold=0.3,
)

# 3. 使用图进行模型训练或推理
# graph.x: 节点特征
# graph.edge_index: 边索引
# graph.edge_attr: 边特征（4 维向量）
```


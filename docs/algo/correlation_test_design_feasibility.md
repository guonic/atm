# 相关性算法测试设计可行性评估报告

## 执行摘要

**总体评估：高度可行，但需要补充部分基础设施**

该测试设计文档思路清晰，目标明确，与现有代码架构兼容。核心框架可以直接实现，但需要补充一些辅助工具类。

---

## 0. 模型说明 (Model Overview)

### 0.1 StructureExpert 模型架构

**模型类型：** 图神经网络 (Graph Neural Network, GNN)

**模型实现：**
- **类名：** `StructureExpertBuyModel` (位于 `python/nq/trading/strategies/buy_models/structure_expert.py`)
- **底层模型：** 
  - `StructureExpertGNN` - 基础版本（无边属性）
  - `DirectionalStockGNN` - 方向性版本（支持边属性，包含相关性特征）

**模型输入：**
1. **节点特征 (Node Features):** 
   - 形状：`[N, D]`，其中 N 是股票数量，D 是特征维度（默认 158 维）
   - 内容：技术指标、价格特征等（通过 Qlib 加载）
2. **边索引 (Edge Index):** 
   - 形状：`[2, E]`，表示股票之间的连接关系
   - 来源：基于行业分类或相关性分析构建
3. **边属性 (Edge Attributes，仅 DirectionalStockGNN):**
   - 形状：`[E, 4]`，包含 4 种相关性指标：
     - `correlation_value` - 截面相关性
     - `lagged_weight` - 滞后相关性权重
     - `volatility_sync_score` - 波动率同步分数
     - `transfer_entropy` - 转移熵

**模型输出：**
- **预测分数 (Prediction Scores):** 
  - 形状：`[N, 1]`，每个股票的预测收益率
  - 用途：用于股票排序，分数越高表示预期收益越好

**模型工作流程：**
```python
# 1. 加载特征数据
df_x = load_features_for_date(date, instruments, lookback_days=60)

# 2. 构建图结构
daily_graph = builder.get_daily_graph(
    df_x,
    include_edge_attr=True  # 如果使用 DirectionalStockGNN
)

# 3. 模型推理
with torch.no_grad():
    pred = model(daily_graph.x, daily_graph.edge_index, edge_attr=daily_graph.edge_attr)
    # pred: [N, 1] 预测分数

# 4. 生成排序
ranks_df = pd.DataFrame({
    'symbol': symbols,
    'score': pred.flatten(),
    'rank': range(1, len(symbols) + 1)
}).sort_values('score', ascending=False)
```

---

### 0.2 相关性算法在模型中的角色

**当前集成方式（训练时）：**
- 相关性算法用于构建**边属性 (Edge Attributes)**
- 在 `GraphDataBuilder.get_daily_graph()` 中计算：
  ```python
  # 计算相关性矩阵
  corr_matrix = cross_sectional.calculate_matrix(returns_df, symbols)
  lagged_corr, _ = lagged.calculate_matrix(returns_df, symbols)
  sync_matrix = volatility_sync.calculate_matrix(highs_df, lows_df, symbols)
  te_matrix, _ = transfer_entropy.calculate_matrix(returns_df, symbols)
  
  # 构建边特征向量
  edge_feature = [
      correlation_value,      # 截面相关性
      lagged_weight,          # 滞后相关性
      volatility_sync_score,  # 波动率同步
      transfer_entropy        # 转移熵
  ]
  ```

**测试框架中的新角色（推理时）：**
- 相关性算法作为**二次过滤器 (Secondary Filter)**
- 在 `StructureExpertBuyModel.generate_ranks()` 生成的排序基础上：
  1. 获取 Top K 标的（原始排序）
  2. 计算这些标的之间的相关性分值
  3. 应用相关性阈值过滤（例如：只保留 `ρ > 0.7` 的标的）
  4. 形成最终的买入列表

**两种角色的区别：**

| 维度 | 训练时（边属性） | 测试时（过滤器） |
|------|-----------------|-----------------|
| **用途** | 模型学习股票间关系 | 过滤买入标的 |
| **计算时机** | 图构建时 | 排序后 |
| **输入范围** | 所有股票 | Top K 标的 |
| **输出** | 边特征向量 | 过滤后的标的列表 |
| **影响** | 影响模型预测 | 影响最终买入决策 |

---

### 0.3 模型在测试框架中的使用

**测试框架的核心逻辑：**

```
┌─────────────────────────────────────────────────────────────┐
│  1. StructureExpert 模型生成每日截面排序                    │
│     Input: 市场数据 (market_data)                           │
│     Output: DataFrame['symbol', 'score', 'rank']           │
└─────────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────┐
│  2. 相关性算法计算相关性分值                                  │
│     Input: Top K 标的的历史收益率                            │
│     Output: 相关性矩阵或相关性分值                           │
└─────────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────┐
│  3. 应用相关性阈值过滤                                        │
│     Input: Top K 标的 + 相关性分值                           │
│     Output: 过滤后的标的列表（实验组）                        │
└─────────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────┐
│  4. 计算未来收益率并统计                                      │
│     - 对比组：原始 Top K 标的的收益率                         │
│     - 实验组：过滤后标的的收益率                             │
│     - 计算胜率提升、Rank IC 等指标                           │
└─────────────────────────────────────────────────────────────┘
```

**关键代码位置：**
- 模型调用：`python/nq/trading/strategies/buy_models/structure_expert.py::StructureExpertBuyModel.generate_ranks()`
- 策略集成：`python/nq/trading/strategies/asymmetric.py::AsymmetricStrategy._generate_buy_signals()`
- 相关性计算：`python/nq/analysis/correlation/correlation.py`（5种算法）

---

## 1. 可行性评估维度

### ✅ 1.1 数据基础 - **完全具备**

**现有能力：**
- ✅ `StructureExpertBuyModel.generate_ranks()` 可以获取每日截面排序（返回 `DataFrame` 包含 `symbol`, `score`, `rank`）
- ✅ 5种相关性算法已实现 `ICorrelationCalculator` 接口，统一调用方式
- ✅ 历史价格数据可通过 Qlib 或数据库获取
- ✅ 回测框架 `run_custom_backtest()` 可运行策略回测

**需要补充：**
- ⚠️ 多周期远期收益率计算工具（T+3, T+5, T+10等）
- ⚠️ 排名演变追踪机制（买入后排名变化）

**实现难度：** ⭐⭐ (简单)

---

### ✅ 1.2 算法集成 - **完全具备**

**现有能力：**
- ✅ 5种相关性算法：`DynamicCrossSectionalCorrelation`, `CrossLaggedCorrelation`, `VolatilitySync`, `GrangerCausality`, `TransferEntropy`
- ✅ 统一接口 `ICorrelationCalculator`，支持 `calculate_matrix()` 和 `calculate_pairwise()`
- ✅ 支持不同窗口参数配置
- ✅ 支持阈值过滤

**需要补充：**
- ⚠️ 相关性过滤器的策略集成（在 `AsymmetricStrategy` 中应用相关性过滤）

**实现难度：** ⭐⭐⭐ (中等)

---

### ⚠️ 1.3 评测指标 - **部分具备**

**现有能力：**
- ✅ 胜率计算（可通过回测结果统计）
- ✅ 收益率计算（已有 `calculate_metrics_from_results()`）
- ✅ 盈亏比计算（可通过订单数据计算）

**需要补充：**
- ❌ **Rank IC 计算**：相关性分值 vs 未来收益率的秩相关系数
- ⚠️ 排名半衰期计算（排名跌出前20%的平均天数）
- ⚠️ 回撤保护统计（相关性分值低时，回撤>5%的频率）

**实现难度：** ⭐⭐ (简单，主要是统计计算)

---

### ✅ 1.4 回测框架 - **完全具备**

**现有能力：**
- ✅ `run_custom_backtest()` 支持自定义策略回测
- ✅ `AsymmetricStrategy` 已集成 `StructureExpertBuyModel`
- ✅ 支持数据捕获和日志记录（`get_executed_orders()`, `get_signals()`, `get_daily_stats()`）
- ✅ 支持多周期回测（通过调整日期范围）

**需要补充：**
- ⚠️ 对比组 vs 实验组的并行回测框架
- ⚠️ 参数网格搜索工具

**实现难度：** ⭐⭐⭐ (中等)

---

## 2. 分阶段实施可行性

### 第一阶段：多周期收益率矩阵生成

**可行性：** ✅ **高度可行**

**所需组件：**
1. `ReturnMatrixGenerator` 类
   - 输入：每日截面排序数据、历史价格数据
   - 输出：多周期远期收益率矩阵、排名演变数据

**实现要点：**
```python
class ReturnMatrixGenerator:
    def __init__(self, holding_periods: List[int] = [3, 5, 8, 10, 15, 20, 30, 60]):
        self.holding_periods = holding_periods
    
    def generate(
        self,
        daily_ranks: Dict[pd.Timestamp, pd.DataFrame],  # 每日排序
        price_data: pd.DataFrame,  # 历史价格
    ) -> pd.DataFrame:
        # 计算每个标的在未来各周期的收益率
        # 追踪排名演变
        pass
```

**依赖：**
- ✅ Qlib 数据加载（已有）
- ✅ 价格数据获取（已有）
- ⚠️ 交易日历（需要确认）

**预计工作量：** 2-3天

---

### 第二阶段：相关性算法集成评测

**可行性：** ✅ **高度可行**

**所需组件：**
1. `CorrelationOptimizer` 类
   - 遍历5种算法 × 多个窗口参数
   - 应用相关性阈值过滤
   - 与收益率矩阵对齐
   - 统计性能指标

**实现要点：**
```python
class CorrelationOptimizer:
    def __init__(
        self,
        correlation_calculators: List[ICorrelationCalculator],
        window_params: List[int] = [3, 5, 8, 13],
        thresholds: List[float] = [0.5, 0.6, 0.7, 0.8, 0.9],
    ):
        pass
    
    def optimize(
        self,
        daily_ranks: Dict[pd.Timestamp, pd.DataFrame],
        returns_data: pd.DataFrame,
        return_matrix: pd.DataFrame,
    ) -> pd.DataFrame:
        # 网格搜索最优参数组合
        # 统计胜率、盈亏比、最大跌幅
        pass
```

**依赖：**
- ✅ 相关性算法接口（已有）
- ✅ 收益率矩阵（第一阶段）
- ⚠️ 相关性过滤策略（需要实现）

**预计工作量：** 3-5天

---

### 第三阶段：参数寻优与敏感度分析

**可行性：** ✅ **可行**

**所需组件：**
1. 分析报告脚本
   - 最优组合矩阵
   - 排名漂移图
   - 热力图

**依赖：**
- ✅ 第二阶段结果数据
- ⚠️ 可视化库（matplotlib/seaborn）

**预计工作量：** 2-3天

---

## 3. 潜在挑战与风险

### ⚠️ 3.1 计算复杂度

**风险：** 高
- 5种算法 × 4个窗口参数 × 5个阈值 = 100种组合
- 每日Top K标的 × 多个持有期 = 大量计算

**缓解措施：**
- ✅ 使用 Pandas Vectorization（文档已建议）
- ✅ 使用 Numba 加速（文档已建议）
- ⚠️ 考虑并行计算（多进程/多线程）
- ⚠️ 缓存中间结果

---

### ⚠️ 3.2 过拟合风险

**风险：** 中高
- 在多个算法和参数组合中寻找最优，可能存在过拟合
- 样本外表现可能不佳

**缓解措施：**
- ⚠️ 使用 Walk-Forward 验证（时间序列交叉验证）
- ⚠️ 保留样本外测试集
- ⚠️ 统计显著性检验（t-test, bootstrap）

---

### ⚠️ 3.3 数据质量

**风险：** 中
- 历史数据可能存在缺失、异常值
- 停牌、退市股票的处理

**缓解措施：**
- ✅ 已有数据清洗机制（`clean_dataframe`, `validate_and_filter_nan`）
- ⚠️ 需要加强异常值处理

---

### ⚠️ 3.4 统计显著性

**风险：** 中
- 胜率提升可能是随机波动
- 需要确保结果具有统计显著性

**缓解措施：**
- ❌ 需要实现统计检验（t-test, Mann-Whitney U test）
- ❌ 需要计算置信区间

---

## 4. 建议的改进点

### 4.1 增加统计显著性检验

**建议：** 在文档中明确要求统计显著性检验

```python
def calculate_statistical_significance(
    baseline_win_rate: float,
    filtered_win_rate: float,
    baseline_n: int,
    filtered_n: int,
) -> Tuple[float, bool]:  # (p-value, is_significant)
    # 使用 t-test 或 Mann-Whitney U test
    pass
```

---

### 4.2 增加回撤保护统计

**建议：** 采纳文档末尾的建议，增加回撤保护统计

```python
def calculate_drawdown_protection(
    correlation_scores: pd.Series,
    returns: pd.Series,
    drawdown_threshold: float = 0.05,
) -> Dict[str, float]:
    # 统计相关性分值低时，回撤>5%的频率
    pass
```

---

### 4.3 增加 Walk-Forward 验证

**建议：** 使用时间序列交叉验证，避免过拟合

```python
def walk_forward_validation(
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    train_window: int = 252,  # 1年
    test_window: int = 63,     # 1季度
    step_size: int = 21,      # 1个月
):
    # 滚动窗口验证
    pass
```

---

### 4.4 增加分组统计

**建议：** 按 `StructExperts` 原始分值区间分组（文档已提及）

```python
def group_by_score_interval(
    ranks: pd.DataFrame,
    intervals: List[Tuple[int, int]] = [(1, 10), (11, 20), (21, 50)],
) -> Dict[str, pd.DataFrame]:
    # 按排名区间分组
    pass
```

---

## 5. 实施优先级建议

### 高优先级（核心功能）
1. ✅ `ReturnMatrixGenerator` - 多周期收益率计算
2. ✅ `CorrelationOptimizer` - 算法集成评测
3. ✅ Rank IC 计算
4. ✅ 胜率提升统计

### 中优先级（增强功能）
5. ⚠️ 排名演变追踪
6. ⚠️ 统计显著性检验
7. ⚠️ 回撤保护统计

### 低优先级（优化功能）
8. ⚠️ 可视化报告
9. ⚠️ 参数自动寻优
10. ⚠️ 并行计算优化

---

## 6. 总体结论

### ✅ 可行性评分：**8.5/10**

**优势：**
- ✅ 思路清晰，目标明确
- ✅ 与现有代码架构高度兼容
- ✅ 核心功能实现难度适中
- ✅ 有明确的数据基础和工具支持

**需要补充：**
- ⚠️ Rank IC 计算工具
- ⚠️ 多周期收益率计算工具
- ⚠️ 统计显著性检验
- ⚠️ 相关性过滤策略集成

**建议：**
1. **立即开始实施** - 框架设计合理，可以开始开发
2. **分阶段推进** - 按照文档的三个阶段逐步实施
3. **加强统计验证** - 增加统计显著性检验，避免过拟合
4. **性能优化** - 考虑并行计算和缓存机制

---

## 7. 实施时间估算

| 阶段 | 任务 | 预计时间 |
|------|------|----------|
| 第一阶段 | ReturnMatrixGenerator | 2-3天 |
| 第二阶段 | CorrelationOptimizer | 3-5天 |
| 第三阶段 | 分析报告和可视化 | 2-3天 |
| **总计** | | **7-11天** |

**加上测试和优化：** 预计 **10-15个工作日**

---

## 8. 下一步行动建议

1. **立即开始：** 创建 `ReturnMatrixGenerator` 类
2. **并行开发：** 实现 Rank IC 计算工具
3. **集成测试：** 在现有回测框架中测试相关性过滤
4. **文档完善：** 补充统计显著性检验要求

---

**评估日期：** 2026-01-14  
**文档版本：** v1.0

# 灵活回测报告系统设计方案

## 1. 设计目标

设计一个灵活、可扩展的回测报告系统，能够：
- **与 Eidos 系统框架深度集成**：直接使用 `EidosRepo` 加载数据，复用现有基础设施
- 从 Eidos 数据库读取已存储的回测数据
- 支持多种指标类型的计算和展示
- 支持自定义指标扩展
- 支持多种输出格式（控制台、JSON、HTML、Markdown）
- 支持指标分组和筛选
- 支持多实验对比分析
- **与现有 REST API 协调**：复用或扩展现有的 handlers

## 2. 架构设计

### 2.1 核心组件（与 Eidos 系统集成）

```
┌─────────────────────────────────────────────────────────┐
│              BacktestReportGenerator                     │
│  (主入口，协调各个组件)                                    │
└─────────────────────────────────────────────────────────┘
                        │
        ┌───────────────┼───────────────┐
        │               │               │
┌───────▼──────┐  ┌─────▼──────┐  ┌─────▼──────┐
│ EidosRepo    │  │ Calculator │  │ Formatter  │
│ (数据加载)   │  │ (指标计算)  │  │ (格式化输出)│
│ 直接使用现有  │  │             │  │            │
│ EidosRepo    │  │             │  │            │
└──────────────┘  └────────────┘  └────────────┘
        │               │               │
        └───────┬───────┴───────┬───────┘
                │               │
        ┌───────▼───────┐  ┌───▼──────┐
        │ MetricRegistry │  │ Config   │
        │ (指标注册表)    │  │ (配置)   │
        └───────────────┘  └──────────┘
                │
        ┌───────▼──────────────┐
        │ MetricCalculator      │
        │ (指标计算实现层)        │
        │ - 标准库 (empyrical)   │
        │ - 自行实现 (现有逻辑)   │
        └───────────────────────┘
```

### 2.2 与 Eidos 系统集成点

1. **数据加载**：直接使用 `EidosRepo`，无需重新实现数据访问层
2. **计算逻辑复用**：可以复用现有的 `PerformanceMetrics` 类
3. **API 协调**：可以扩展现有的 REST API handlers
4. **Go 服务集成**：可以调用 Go 端的 attribution 服务（可选）

### 2.3 组件职责

#### 2.2.1 DataLoader (数据加载器) - 使用 EidosRepo
- **直接使用 `EidosRepo`**：复用现有的数据访问层
- 从 Eidos 数据库读取回测数据
- 支持按 exp_id 加载单个实验
- 支持批量加载多个实验
- 数据缓存和预加载优化
- **与现有 REST API 协调**：可以复用 `handlers.py` 中的数据加载逻辑

#### 2.2.2 Calculator (指标计算器) - 混合实现策略
- **标准库优先**：使用 `empyrical` 等成熟库计算标准指标（Sharpe, MaxDD, Volatility 等）
- **自行实现补充**：对于标准库不支持的指标，使用现有逻辑或自行实现
- **复用现有代码**：可以复用 `backtest_structure_expert.py` 中的 `PerformanceMetrics` 类
- 交易统计指标（胜率、盈亏比等）- 自行实现（已有逻辑）
- 自定义指标扩展接口
- 指标依赖关系管理
- **Go 服务调用**（可选）：对于复杂计算，可以调用 Go 端的 attribution 服务

#### 2.2.3 Formatter (格式化器)
- 控制台输出（ConsoleFormatter）
- JSON 输出（JSONFormatter）
- HTML 报告（HTMLFormatter）
- Markdown 报告（MarkdownFormatter）

#### 2.2.4 MetricRegistry (指标注册表)
- 指标注册和发现机制
- 指标分类（Portfolio、Trading、Risk、Custom）
- 指标元数据管理（名称、描述、单位、格式）

#### 2.2.5 Config (配置管理)
- 指标选择配置
- 输出格式配置
- 显示选项配置（精度、单位等）

## 3. 数据模型设计

### 3.1 BacktestData (回测数据模型)

```python
@dataclass
class BacktestData:
    """回测数据容器"""
    exp_id: str
    experiment: Dict[str, Any]  # bt_experiment 表数据
    ledger: pd.DataFrame        # bt_ledger 表数据 (date, nav, cash, ...)
    trades: pd.DataFrame        # bt_trades 表数据
    model_outputs: pd.DataFrame  # bt_model_outputs 表数据 (可选)
    model_links: pd.DataFrame   # bt_model_links 表数据 (可选)
    embeddings: pd.DataFrame    # bt_embeddings 表数据 (可选)
```

### 3.2 MetricResult (指标结果模型)

```python
@dataclass
class MetricResult:
    """指标计算结果"""
    name: str                    # 指标名称
    category: str               # 指标分类
    value: Any                   # 指标值
    unit: Optional[str] = None   # 单位
    format: Optional[str] = None # 格式化字符串
    description: Optional[str] = None  # 描述
    metadata: Dict[str, Any] = field(default_factory=dict)  # 额外元数据
```

### 3.3 ReportConfig (报告配置)

```python
@dataclass
class ReportConfig:
    """报告配置"""
    # 指标选择
    metric_categories: List[str] = field(default_factory=lambda: ["portfolio", "trading", "turnover"])
    metric_names: Optional[List[str]] = None  # None 表示选择所有
    
    # 输出格式
    output_format: str = "console"  # console, json, html, markdown
    output_path: Optional[str] = None
    
    # 显示选项
    precision: int = 2
    show_details: bool = True
    show_trading_stats: bool = True
    show_turnover_stats: bool = True
    
    # 对比分析
    compare_experiments: Optional[List[str]] = None  # 多个 exp_id 对比
```

## 4. 指标系统设计

### 4.1 指标分类

#### 4.1.1 Portfolio Metrics (组合指标)
- Total Return (总收益)
- Annualized Return (年化收益)
- Volatility (波动率)
- Sharpe Ratio (夏普比率)
- Sortino Ratio (索提诺比率)
- Max Drawdown (最大回撤)
- Calmar Ratio (卡玛比率)
- Information Ratio (信息比率)

#### 4.1.2 Trading Statistics (交易统计)
- Total Trades (总交易数)
- Winning Trades (盈利交易数)
- Losing Trades (亏损交易数)
- Win Rate (胜率)
- Average Holding Period (平均持仓天数)
- Average Return per Trade (平均每笔收益)
- Profit Factor (盈亏比)
- Avg Win / Avg Loss (平均盈亏比)
- Total Profit (总盈利)
- Total Loss (总亏损)
- Largest Win (最大盈利)
- Largest Loss (最大亏损)

#### 4.1.3 Turnover Statistics (换手统计)
- Total Turnover (总换手)
- Average Daily Turnover (平均日换手率)
- Turnover Frequency (换手频率)
- Position Holding Days Distribution (持仓天数分布)

#### 4.1.4 Risk Metrics (风险指标)
- Value at Risk (VaR)
- Conditional VaR (CVaR)
- Downside Deviation (下行波动率)
- Beta (相对于基准)
- Alpha (超额收益)

#### 4.1.5 Model Performance (模型表现)
- Score Distribution (分数分布)
- Rank Stability (排名稳定性)
- Signal Quality (信号质量)
- Prediction Accuracy (预测准确度)

### 4.2 指标注册机制

```python
class MetricRegistry:
    """指标注册表"""
    
    _metrics: Dict[str, MetricDefinition] = {}
    
    @classmethod
    def register(cls, category: str, name: str, calculator: Callable):
        """注册指标"""
        cls._metrics[f"{category}.{name}"] = MetricDefinition(
            category=category,
            name=name,
            calculator=calculator
        )
    
    @classmethod
    def get_metric(cls, category: str, name: str) -> Optional[MetricDefinition]:
        """获取指标定义"""
        return cls._metrics.get(f"{category}.{name}")
    
    @classmethod
    def list_metrics(cls, category: Optional[str] = None) -> List[MetricDefinition]:
        """列出指标"""
        if category:
            return [m for m in cls._metrics.values() if m.category == category]
        return list(cls._metrics.values())
```

### 4.3 指标计算接口（混合实现策略）

#### 4.3.1 标准库实现（推荐）

```python
# 使用 empyrical 库计算标准指标
try:
    import empyrical as ep
    HAS_EMPYRICAL = True
except ImportError:
    HAS_EMPYRICAL = False
    # 回退到自行实现

class BaseMetricCalculator:
    """指标计算器基类"""
    
    def calculate(self, data: BacktestData) -> MetricResult:
        """计算指标"""
        raise NotImplementedError
    
    def get_dependencies(self) -> List[str]:
        """获取依赖的其他指标"""
        return []

# 使用标准库的实现
class SharpeRatioCalculator(BaseMetricCalculator):
    """使用 empyrical 计算夏普比率"""
    
    def calculate(self, data: BacktestData) -> MetricResult:
        if data.ledger.empty:
            return MetricResult(
                name="sharpe_ratio",
                category="portfolio",
                value=None,
                description="夏普比率（数据不足）"
            )
        
        # 计算日收益率
        navs = data.ledger['nav'].values
        returns = pd.Series(navs).pct_change().dropna()
        
        if HAS_EMPYRICAL:
            # 使用标准库计算（更准确、经过验证）
            sharpe = ep.sharpe_ratio(returns, annualization=252)
        else:
            # 回退到自行实现（复用现有逻辑）
            sharpe = self._calculate_sharpe_manual(returns)
        
        return MetricResult(
            name="sharpe_ratio",
            category="portfolio",
            value=sharpe,
            format="{:.4f}",
            description="夏普比率（年化）"
        )
    
    def _calculate_sharpe_manual(self, returns: pd.Series) -> float:
        """手动计算夏普比率（复用现有逻辑）"""
        if len(returns) == 0 or returns.std() == 0:
            return 0.0
        return returns.mean() / returns.std() * (252 ** 0.5)
```

#### 4.3.2 自行实现（复用现有代码）

```python
# 复用 backtest_structure_expert.py 中的 PerformanceMetrics 类
from nq.analysis.backtest.performance_metrics import PerformanceMetrics

class TotalReturnCalculator(BaseMetricCalculator):
    """总收益计算器 - 复用现有逻辑"""
    
    def calculate(self, data: BacktestData) -> MetricResult:
        if data.ledger.empty:
            return MetricResult(
                name="total_return",
                category="portfolio",
                value=None,
                description="总收益率（数据不足）"
            )
        
        # 复用现有的 PerformanceMetrics 逻辑
        # 将 ledger DataFrame 转换为 PerformanceMetrics 期望的格式
        metric_df = data.ledger.set_index('date')[['nav']].rename(columns={'nav': 'account'})
        perf_metrics = PerformanceMetrics(metric_df)
        
        total_return = perf_metrics.total_return
        
        return MetricResult(
            name="total_return",
            category="portfolio",
            value=total_return,
            unit="%",
            format="{:.2f}%",
            description="总收益率"
        )
```

#### 4.3.3 Go 服务调用（可选）

```python
class AdvancedAttributionCalculator(BaseMetricCalculator):
    """复杂归因分析 - 调用 Go 服务"""
    
    def __init__(self, go_service_url: Optional[str] = None):
        self.go_service_url = go_service_url
    
    def calculate(self, data: BacktestData) -> MetricResult:
        if not self.go_service_url:
            # 如果没有 Go 服务，回退到 Python 实现
            return self._calculate_python(data)
        
        # 调用 Go 端的 attribution 服务
        response = requests.post(
            f"{self.go_service_url}/attribution/basic-stats",
            json={"exp_id": data.exp_id}
        )
        stats = response.json()
        
        return MetricResult(
            name="advanced_attribution",
            category="attribution",
            value=stats,
            description="高级归因分析（来自 Go 服务）"
        )
```

### 4.4 指标计算策略总结

| 指标类型 | 实现方式 | 理由 |
|---------|---------|------|
| **标准金融指标**（Sharpe, MaxDD, Volatility, Sortino 等） | **优先使用 empyrical** | 经过验证、准确、维护良好 |
| **交易统计指标**（胜率、盈亏比、持仓天数等） | **自行实现** | 业务逻辑特定，已有实现 |
| **换手统计指标** | **自行实现** | 业务逻辑特定 |
| **模型表现指标**（排名稳定性、信号质量等） | **自行实现** | 业务逻辑特定 |
| **复杂归因分析** | **可选调用 Go 服务** | 性能考虑，Go 端已有实现 |
| **自定义指标** | **用户自行实现** | 灵活性需求 |

### 4.5 依赖管理

```python
# 在 pyproject.toml 中添加可选依赖
[project.optional-dependencies]
backtest-report = [
    "empyrical>=0.5.5",  # 标准金融指标计算库
    # 可选：其他标准库
    # "quantstats>=0.0.59",  # 更全面的量化统计库
]
```

## 5. 数据加载设计（与 Eidos 系统集成）

### 5.1 EidosDataLoader - 直接使用 EidosRepo

```python
from nq.repo.eidos_repo import EidosRepo
from nq.config import DatabaseConfig

class EidosDataLoader:
    """
    从 Eidos 数据库加载回测数据。
    
    直接使用 EidosRepo，复用现有的数据访问层，无需重新实现。
    """
    
    def __init__(self, db_config: DatabaseConfig, schema: str = "eidos"):
        """
        初始化数据加载器。
        
        Args:
            db_config: 数据库配置（与 EidosRepo 使用相同的配置）
            schema: 数据库 schema（默认 'eidos'）
        """
        # 直接使用 EidosRepo，复用现有基础设施
        self.repo = EidosRepo(db_config, schema)
    
    def load_experiment(self, exp_id: str) -> BacktestData:
        """
        加载单个实验数据。
        
        使用 EidosRepo 的各个子仓库加载数据，与现有 REST API handlers 逻辑一致。
        """
        # 加载实验元数据（使用 experiment 仓库）
        experiment = self.repo.experiment.get_experiment(exp_id)
        if not experiment:
            raise ValueError(f"Experiment {exp_id} not found")
        
        # 加载账户流水（使用 ledger 仓库）
        ledger_data = self.repo.ledger.get_ledger(exp_id)
        ledger = pd.DataFrame(ledger_data) if ledger_data else pd.DataFrame()
        
        # 加载交易记录（使用 trades 仓库）
        trades_data = self.repo.trades.get_trades(exp_id)
        trades = pd.DataFrame(trades_data) if trades_data else pd.DataFrame()
        
        # 加载模型输出（可选，使用 model_outputs 仓库）
        model_outputs_data = self.repo.model_outputs.get_model_outputs(exp_id)
        model_outputs = pd.DataFrame(model_outputs_data) if model_outputs_data else pd.DataFrame()
        
        # 加载模型链接（可选，使用 model_links 仓库）
        model_links_data = self.repo.model_links.get_model_links(exp_id)
        model_links = pd.DataFrame(model_links_data) if model_links_data else pd.DataFrame()
        
        # 加载嵌入向量（可选，使用 embeddings 仓库）
        embeddings_data = self.repo.embeddings.get_embeddings(exp_id)
        embeddings = pd.DataFrame(embeddings_data) if embeddings_data else pd.DataFrame()
        
        return BacktestData(
            exp_id=exp_id,
            experiment=experiment,
            ledger=ledger,
            trades=trades,
            model_outputs=model_outputs,
            model_links=model_links,
            embeddings=embeddings
        )
    
    def load_experiments(self, exp_ids: List[str]) -> List[BacktestData]:
        """批量加载多个实验数据"""
        return [self.load_experiment(exp_id) for exp_id in exp_ids]
```

### 5.2 与现有 REST API 协调

现有的 `handlers.py` 中已经有类似的数据加载逻辑，报告系统可以：
1. **复用逻辑**：提取公共的数据加载函数
2. **扩展 handlers**：在现有 handlers 基础上添加报告生成功能
3. **独立服务**：作为独立模块，但使用相同的 EidosRepo

## 6. 报告生成流程

### 6.1 基本流程

```
1. 加载配置 (ReportConfig)
   ↓
2. 加载数据 (EidosDataLoader.load_experiment)
   ↓
3. 计算指标 (Calculator.calculate_all)
   ↓
4. 格式化输出 (Formatter.format)
   ↓
5. 保存/显示报告
```

### 6.2 多实验对比流程

```
1. 加载多个实验数据
   ↓
2. 分别计算每个实验的指标
   ↓
3. 生成对比表格
   ↓
4. 格式化输出对比报告
```

## 7. 输出格式设计

### 7.1 ConsoleFormatter (控制台输出)

```python
class ConsoleFormatter:
    """控制台格式化器"""
    
    def format(self, results: List[MetricResult], config: ReportConfig) -> str:
        """格式化输出"""
        output = []
        
        # 按分类分组
        grouped = self._group_by_category(results)
        
        for category, metrics in grouped.items():
            output.append(f"\n{self._get_category_emoji(category)} {category.upper()} Metrics:")
            for metric in metrics:
                output.append(f"  {metric.name}: {self._format_value(metric)}")
        
        return "\n".join(output)
```

### 7.2 JSONFormatter (JSON 输出)

```python
class JSONFormatter:
    """JSON 格式化器"""
    
    def format(self, results: List[MetricResult], config: ReportConfig) -> str:
        """格式化为 JSON"""
        data = {
            "experiment_id": config.exp_id,
            "generated_at": datetime.now().isoformat(),
            "metrics": [
                {
                    "category": m.category,
                    "name": m.name,
                    "value": m.value,
                    "unit": m.unit,
                    "description": m.description
                }
                for m in results
            ]
        }
        return json.dumps(data, indent=2, ensure_ascii=False)
```

### 7.3 HTMLFormatter (HTML 报告)

```python
class HTMLFormatter:
    """HTML 格式化器"""
    
    def format(self, results: List[MetricResult], config: ReportConfig) -> str:
        """格式化为 HTML"""
        # 使用模板引擎生成 HTML 报告
        # 包含图表、表格等
        pass
```

### 7.4 MarkdownFormatter (Markdown 报告)

```python
class MarkdownFormatter:
    """Markdown 格式化器"""
    
    def format(self, results: List[MetricResult], config: ReportConfig) -> str:
        """格式化为 Markdown"""
        output = []
        output.append("# Backtest Report\n")
        
        grouped = self._group_by_category(results)
        for category, metrics in grouped.items():
            output.append(f"## {category.upper()} Metrics\n")
            output.append("| Metric | Value |\n")
            output.append("|--------|-------|\n")
            for metric in metrics:
                output.append(f"| {metric.name} | {self._format_value(metric)} |\n")
        
        return "".join(output)
```

## 8. 扩展性设计

### 8.1 自定义指标扩展

```python
# 用户自定义指标示例
@MetricRegistry.register(category="custom", name="my_custom_metric")
def calculate_my_metric(data: BacktestData) -> MetricResult:
    # 自定义计算逻辑
    value = ...
    return MetricResult(
        name="my_custom_metric",
        category="custom",
        value=value,
        description="我的自定义指标"
    )
```

### 8.2 自定义格式化器扩展

```python
class MyCustomFormatter(BaseFormatter):
    """自定义格式化器"""
    
    def format(self, results: List[MetricResult], config: ReportConfig) -> str:
        # 自定义格式化逻辑
        pass
```

## 9. 使用示例

### 9.1 基本使用

```python
from nq.analysis.backtest.report import BacktestReportGenerator

# 创建报告生成器
generator = BacktestReportGenerator(db_config)

# 生成报告
report = generator.generate_report(
    exp_id="a1b2c3d4",
    config=ReportConfig(
        metric_categories=["portfolio", "trading", "turnover"],
        output_format="console"
    )
)

# 打印报告
print(report)
```

### 9.2 多实验对比

```python
# 对比多个实验
report = generator.generate_comparison_report(
    exp_ids=["exp1", "exp2", "exp3"],
    config=ReportConfig(
        output_format="html",
        output_path="comparison_report.html"
    )
)
```

### 9.3 自定义指标

```python
# 注册自定义指标
@MetricRegistry.register(category="custom", name="custom_sharpe")
def calculate_custom_sharpe(data: BacktestData) -> MetricResult:
    # 自定义夏普比率计算
    pass

# 使用自定义指标
report = generator.generate_report(
    exp_id="a1b2c3d4",
    config=ReportConfig(
        metric_names=["portfolio.total_return", "custom.custom_sharpe"]
    )
)
```

## 10. 文件结构（与 Eidos 系统集成）

```
python/nq/analysis/backtest/
├── report/                   # 新增：报告系统模块
│   ├── __init__.py
│   ├── generator.py          # BacktestReportGenerator 主类
│   ├── loader.py             # EidosDataLoader（使用 EidosRepo）
│   ├── calculator.py         # 指标计算器协调类
│   ├── formatter/
│   │   ├── __init__.py
│   │   ├── base.py          # BaseFormatter 基类
│   │   ├── console.py        # ConsoleFormatter
│   │   ├── json.py           # JSONFormatter
│   │   ├── html.py           # HTMLFormatter
│   │   └── markdown.py       # MarkdownFormatter
│   ├── metrics/
│   │   ├── __init__.py
│   │   ├── registry.py      # MetricRegistry 注册表
│   │   ├── base.py          # BaseMetricCalculator 基类
│   │   ├── portfolio.py     # 组合指标（使用 empyrical）
│   │   ├── trading.py       # 交易统计指标（自行实现）
│   │   ├── turnover.py      # 换手统计指标（自行实现）
│   │   ├── risk.py          # 风险指标（使用 empyrical）
│   │   └── model.py          # 模型表现指标（自行实现）
│   ├── models.py            # 数据模型
│   └── utils.py              # 工具函数
├── performance_metrics.py     # 复用：现有的 PerformanceMetrics 类
├── eidos_integration.py      # 现有：Eidos 集成
└── eidos_structure_expert.py # 现有：Structure Expert 集成

# 与现有代码的关系：
# - report/loader.py 使用 nq.repo.eidos_repo.EidosRepo
# - report/metrics/portfolio.py 可以复用 performance_metrics.py 的逻辑
# - report/metrics/portfolio.py 优先使用 empyrical 库
# - 可以扩展 nq.api.rest.eidos.handlers 添加报告生成接口
```

## 11. 实现优先级

### Phase 1: 核心功能
1. 数据加载器 (EidosDataLoader)
2. 基础指标计算器 (Portfolio, Trading, Turnover)
3. 控制台格式化器 (ConsoleFormatter)
4. 报告生成器主类 (BacktestReportGenerator)

### Phase 2: 扩展功能
1. JSON/HTML/Markdown 格式化器
2. 风险指标计算器
3. 模型表现指标计算器
4. 多实验对比功能

### Phase 3: 高级功能
1. 自定义指标扩展机制
2. 指标依赖关系管理
3. 缓存和性能优化
4. 可视化图表生成

## 12. 与 Eidos 系统集成的注意事项

### 12.1 数据访问层
1. **复用 EidosRepo**：直接使用现有的 `EidosRepo`，不重新实现数据访问
2. **保持一致性**：与现有 REST API handlers 使用相同的数据加载逻辑
3. **错误处理**：遵循 EidosRepo 的错误处理模式

### 12.2 指标计算策略
1. **标准库优先**：使用 `empyrical` 计算标准金融指标（Sharpe, MaxDD, Volatility 等）
2. **复用现有逻辑**：对于业务特定指标，复用 `PerformanceMetrics` 类
3. **可选 Go 服务**：对于复杂计算，可以调用 Go 端的 attribution 服务
4. **依赖管理**：将 `empyrical` 作为可选依赖，支持回退到自行实现

### 12.3 API 协调
1. **扩展现有 handlers**：可以在 `handlers.py` 中添加报告生成接口
2. **独立模块**：也可以作为独立模块，但使用相同的 EidosRepo
3. **向后兼容**：保持与现有 `print_results` 函数的兼容性

### 12.4 性能优化
1. **缓存机制**：对于重复计算的指标，考虑缓存
2. **批量加载**：支持批量加载多个实验数据
3. **增量计算**：对于大型实验，考虑增量计算

### 12.5 数据验证
1. **完整性检查**：在计算指标前验证数据的完整性
2. **默认值处理**：对于缺失的数据字段，提供合理的默认值或错误提示
3. **类型转换**：处理数据库返回的 Decimal 等类型

### 12.6 文档完善
1. **指标文档**：为每个指标提供清晰的文档说明
2. **使用示例**：提供完整的使用示例
3. **集成指南**：说明如何与 Eidos 系统集成


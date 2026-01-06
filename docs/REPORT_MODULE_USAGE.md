# 回测报告模块使用说明

## 概述

回测报告模块已完整实现，可以从 Eidos 数据库读取回测数据并生成完整的报告。

## 后端使用

### 1. 基本使用

```python
from nq.analysis.backtest.report import BacktestReportGenerator, ReportConfig
from nq.config import load_config

# 加载配置
config = load_config("config/config.yaml")
db_config = config.database

# 创建报告生成器
generator = BacktestReportGenerator(db_config)

# 生成报告
report = generator.generate_report(
    exp_id="a1b2c3d4",
    config=ReportConfig(
        metric_categories=["portfolio", "trading", "turnover"],
        output_format="json"
    )
)

print(report)
```

### 2. API 接口

#### 获取报告
```
GET /api/v1/experiments/{exp_id}/report
```

**查询参数：**
- `format`: 输出格式（json, console, html, markdown）- 默认 json
- `categories`: 指标分类（逗号分隔）- 可选
- `metrics`: 指标名称（逗号分隔）- 可选

**示例：**
```bash
# 获取完整报告
curl http://localhost:8000/api/v1/experiments/a1b2c3d4/report

# 只获取组合指标
curl http://localhost:8000/api/v1/experiments/a1b2c3d4/report?categories=portfolio

# 获取特定指标
curl http://localhost:8000/api/v1/experiments/a1b2c3d4/report?metrics=total_return,sharpe_ratio
```

## 前端使用

### 1. 访问报告视图

1. 打开 Eidos Dashboard
2. 在左侧 Sidebar 中选择 "Report" 子系统
3. 在右侧 ConfigPanel 中选择一个实验
4. 报告会自动加载并显示

### 2. 报告标签页

- **概览**: 显示关键指标（总收益、年化收益、夏普比率、最大回撤、胜率、盈亏比）
- **组合指标**: 显示所有组合相关指标
- **交易统计**: 显示交易相关统计
- **换手统计**: 显示换手相关统计

### 3. 导出报告

点击报告上方的"导出 HTML"或"导出 Markdown"按钮即可下载报告。

## 已实现的指标

### 组合指标 (Portfolio)
- `total_return`: 总收益率
- `annualized_return`: 年化收益率
- `volatility`: 年化波动率
- `sharpe_ratio`: 夏普比率
- `max_drawdown`: 最大回撤
- `initial_account`: 初始账户价值
- `final_account`: 最终账户价值
- `net_pnl`: 净盈亏
- `trading_days`: 交易天数

### 交易统计 (Trading)
- `total_trades`: 总交易次数
- `buy_count`: 买入次数
- `sell_count`: 卖出次数
- `win_rate`: 胜率
- `avg_hold_days`: 平均持仓天数
- `profit_factor`: 盈亏比
- `winning_trades`: 盈利交易数
- `losing_trades`: 亏损交易数

### 换手统计 (Turnover)
- `total_turnover`: 总换手
- `avg_daily_turnover`: 平均日换手率
- `pos_count`: 平均持仓数量

## 扩展指标

### 添加自定义指标

1. 创建指标计算器类：

```python
from nq.analysis.backtest.report.metrics.base import BaseMetricCalculator
from nq.analysis.backtest.report.metrics.registry import MetricRegistry
from nq.analysis.backtest.report.models import BacktestData, MetricResult

@MetricRegistry.register(category="custom", name="my_metric", description="我的自定义指标")
class MyMetricCalculator(BaseMetricCalculator):
    def calculate(self, data: BacktestData) -> MetricResult:
        # 计算逻辑
        value = ...
        return MetricResult(
            name="my_metric",
            category="custom",
            value=value,
            description="我的自定义指标"
        )
```

2. 在 `metrics/__init__.py` 中导入：

```python
from . import custom  # noqa: F401
```

## 注意事项

1. **数据要求**: 报告生成需要实验有 ledger 和 trades 数据
2. **指标计算**: 标准金融指标优先使用 `empyrical` 库（如果安装），否则使用自行实现
3. **性能**: 对于大型实验，报告生成可能需要几秒钟
4. **错误处理**: 如果某个指标计算失败，会记录警告但不会中断整个报告生成

## 故障排查

### 报告为空
- 检查实验是否有 ledger 数据
- 检查实验是否有 trades 数据
- 查看后端日志了解具体错误

### 某些指标显示 N/A
- 检查数据是否包含该指标所需的字段（如 `pnl_ratio`、`hold_days` 等）
- 某些指标需要特定数据才能计算

### API 返回 404
- 确认实验 ID 正确
- 确认实验存在于数据库中


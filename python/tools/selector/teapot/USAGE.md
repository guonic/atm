# Teapot Pattern Recognition - 使用指南

## 完整工作流程

### 步骤 1: 生成缓存（首次运行前）

```bash
# 生成 5 年数据的缓存
python python/tools/selector/teapot/build_cache.py \
    --start-date 2020-01-01 \
    --end-date 2024-12-31 \
    --cache-dir storage/teapot_cache \
    --schema quant \
    --config config/config.yaml

# 验证缓存完整性
python python/tools/selector/teapot/build_cache.py \
    --validate \
    --cache-dir storage/teapot_cache \
    --start-date 2020-01-01 \
    --end-date 2024-12-31
```

### 步骤 2: 扫描市场信号

#### 生产环境（直接使用 PostgreSQL）
```bash
python python/tools/selector/teapot/scan_market.py \
    --start-date 2022-01-01 \
    --end-date 2024-12-31 \
    --output outputs/teapot/signals/signals.csv \
    --config config/config.yam
    --schema quant
```

#### 回测环境（强制使用缓存）
```bash
python python/tools/selector/teapot/scan_market.py \
    --start-date 2022-01-01 \
    --end-date 2024-12-31 \
    --strict-cache \
    --cache-dir storage/teapot_cache \
    --output outputs/teapot/signals/signals.csv \
    --config config/config.yaml
```

#### 指定股票列表
```bash
python python/tools/selector/teapot/scan_market.py \
    --start-date 2022-01-01 \
    --end-date 2024-12-31 \
    --symbols 000001.SZ,000002.SZ,600000.SH \
    --output outputs/teapot/signals/signals.csv
```

### 步骤 3: 评估信号质量

```bash
python python/tools/selector/teapot/evaluate_signals.py \
    --signals-file outputs/teapot/signals/signals.csv \
    --start-date 2022-01-01 \
    --end-date 2024-12-31 \
    --forward-horizons 5 20 \
    --generate-plots \
    --plot-output-dir outputs/teapot/visualizations \
    --report-output outputs/teapot/reports/evaluation_report.csv \
    --use-cache \
    --cache-dir storage/teapot_cache
```

输出文件：
- `outputs/teapot/reports/evaluation_report.csv` - 评估结果
- `outputs/teapot/reports/statistics_summary.json` - 统计摘要
- `outputs/teapot/visualizations/success/*.png` - 成功信号图表
- `outputs/teapot/visualizations/failure/*.png` - 失败信号图表

### 步骤 4: 运行回测

```bash
python python/tools/selector/teapot/run_backtest.py \
    --signals-file outputs/teapot/signals/signals.csv \
    --start-date 2022-01-01 \
    --end-date 2024-12-31 \
    --initial-cash 1000000 \
    --output outputs/teapot/backtest/backtest_results.csv \
    --generate-report
```

输出文件：
- `outputs/teapot/backtest/portfolio_value.csv` - 组合价值曲线
- `outputs/teapot/backtest/trades.csv` - 交易记录
- `outputs/teapot/backtest/performance_metrics.json` - 性能指标

## 配置文件

### Teapot 配置 (`config/teapot/config.yaml`)

```yaml
# 数据加载
data:
  use_cache: false
  strict_cache: false
  cache_dir: "storage/teapot_cache"

# 特征计算
features:
  box_window: 40

# 状态机参数
state_machine:
  box_window: 40
  box_volatility_threshold: 0.15
  box_r2_threshold: 0.7
  trap_max_depth: 0.20
  trap_max_days: 5
  reverse_max_days: 10
  reverse_recover_ratio: 0.8
  breakout_vol_ratio: 1.5

# 过滤器
filters:
  min_turnover: 0.01
  min_amount: 10000000.0
  max_gap: 0.10
  max_trap_depth: 0.20
```

### 策略配置（回测时使用）

创建 `config/teapot/strategy_config.yaml`:

```yaml
exit_strategy: "combined"  # fixed_days, target_return, stop_loss, combined
holding_days: 20
target_return: 0.15  # 15%
stop_loss: -0.10  # -10%
position_size: 0.1  # 10%
max_positions: 10
use_ml_score: false
ml_score_threshold: 0.7
```

## Python API 使用

### 基本使用

```python
from nq.config import load_config
from nq.trading.selector.teapot import TeapotSelector

# 加载配置
config = load_config("config/config.yaml")
db_config = config.database

# 初始化选择器
selector = TeapotSelector(
    db_config=db_config,
    schema="quant",
    config={
        "use_cache": False,
        "box_window": 40,
    }
)

# 扫描市场
signals = selector.scan_market(
    start_date="2022-01-01",
    end_date="2024-12-31",
    symbols=None  # None 表示所有股票
)

# 保存信号
signals.write_csv("outputs/teapot/signals/signals.csv")
```

### 评估信号

```python
from nq.analysis.pattern.teapot import TeapotEvaluator, TeapotStatistics
from nq.data.processor.teapot import TeapotDataLoader

# 加载数据
data_loader = TeapotDataLoader(db_config, schema="quant")
market_data = data_loader.load_daily_data(
    start_date="2022-01-01",
    end_date="2024-12-31",
)

# 加载信号
signals = pl.read_csv("outputs/teapot/signals/signals.csv")

# 评估
evaluator = TeapotEvaluator(forward_horizons=[5, 20])
evaluation_results = evaluator.compute_forward_returns(signals, market_data)

# 统计
statistics = TeapotStatistics()
stats = statistics.compute_basic_stats(evaluation_results)
print(f"Win Rate T+20: {stats['win_rate_t20']:.2%}")
```

### 生成可视化

```python
from tools.visualization.teapot import BatchPlotter

plotter = BatchPlotter(
    output_dir=Path("outputs/teapot/visualizations"),
    n_workers=4
)

plot_stats = plotter.generate_all_plots(
    signals=signals,
    market_data=market_data,
    evaluation_results=evaluation_results
)
```

## 输出文件说明

### 信号文件 (`signals.csv`)
```csv
ts_code,signal_date,box_h,box_l,box_width,trap_depth,breakout_vol_ratio,signal_score
000001.SZ,2024-01-15,12.50,10.00,0.25,0.08,1.8,0.85
```

### 评估结果 (`evaluation_report.csv`)
```csv
ts_code,signal_date,return_t5,return_t20,max_drawdown_t5,max_drawdown_t20,peak_date,peak_return
000001.SZ,2024-01-15,0.05,0.12,-0.02,-0.03,2024-01-25,0.15
```

### 统计摘要 (`statistics_summary.json`)
```json
{
  "basic_stats": {
    "total_signals": 1234,
    "win_rate_t5": 0.65,
    "win_rate_t20": 0.72,
    "avg_return_t5": 0.03,
    "avg_return_t20": 0.08,
    "sharpe_ratio_t20": 1.2
  }
}
```

## 常见问题

### 1. 缓存不存在错误
```
CacheNotFoundError: Cache directory not found
```
**解决**：运行 `build_cache.py` 生成缓存

### 2. 数据加载慢
**解决**：使用缓存模式（`--use-cache`）或指定股票列表（`--symbols`）

### 3. 可视化中文显示为方块
**解决**：安装中文字体或修改 `plotter.py` 中的字体配置

### 4. Backtrader 数据加载失败
**解决**：实现 `TeapotBacktester._load_data_feed` 方法，根据实际数据源加载数据

## 性能优化建议

1. **使用缓存**：开发/测试环境使用 Parquet 缓存，提升加载速度
2. **批量处理**：扫描时指定股票列表，避免加载全市场数据
3. **并行处理**：可视化生成使用多进程（`n_workers` 参数）
4. **数据分片**：大数据集按年份或股票分组处理

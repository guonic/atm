这份文档旨在指导大语言模型（如 Cursor/LLM）构建一套**通用量化回测归因系统**。该系统支持任意模型（GNN、时序模型、线性模型等）的结果存储与深度归因分析，采用 **Python (回测/展示) + Go (计算) + PostgreSQL (存储)** 的架构。

---

# 通用量化回测归因系统 (Universal Backtest Attribution System) 设计文档

## 1. 系统目标
- **解耦性**：支持任何量化模型的输出存储与归因。
- **高性能**：利用 Go 处理亿级截面数据的重算，利用 PG 索引加速检索。
- **深度归因**：支持从“组合表现 -> 交易行为 -> 模型分数波动 -> 样本空间结构”的全链路回溯。

---

## 2. 数据库设计 (PostgreSQL + pgvector)

### 2.1 实验元数据表 `bt_experiment`
记录一次回测的全局背景。
```sql
CREATE TABLE bt_experiment (
    exp_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL,
    model_type TEXT,            -- e.g., 'GNN', 'GRU', 'Linear'
    engine_type TEXT,           -- e.g., 'Qlib', 'Backtrader'
    start_date DATE NOT NULL,
    end_date DATE NOT NULL,
    config JSONB NOT NULL,      -- 存储超参数: {topk: 5, fee: 0.0005, initial_cash: 1000000}
    metrics_summary JSONB,      -- 冗余存储最终汇总指标 (Sharpe, MaxDD等)
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### 2.2 账户流水表 `bt_ledger`
记录每日净值，用于生成 Performance Indicators。
```sql
CREATE TABLE bt_ledger (
    exp_id UUID REFERENCES bt_experiment(exp_id),
    date DATE NOT NULL,
    nav NUMERIC(18, 4) NOT NULL,        -- 净值 (Net Asset Value)
    cash NUMERIC(18, 4),                -- 现金
    market_value NUMERIC(18, 4),        -- 持仓市值
    deal_amount NUMERIC(18, 2),         -- 当日成交笔数总额
    turnover_rate FLOAT,                -- 当日换手率
    pos_count INTEGER,                  -- 当日持仓数量
    PRIMARY KEY (exp_id, date)
);
```

### 2.3 交易订单表 `bt_trades`
记录回测产生的具体交易行为，对应 Trading Statistics。
```sql
CREATE TABLE bt_trades (
    trade_id SERIAL PRIMARY KEY,
    exp_id UUID REFERENCES bt_experiment(exp_id),
    symbol TEXT NOT NULL,
    deal_time TIMESTAMP NOT NULL,
    side INTEGER NOT NULL,              -- 1(Buy), -1(Sell)
    price NUMERIC(18, 4) NOT NULL,
    amount INTEGER NOT NULL,
    rank_at_deal INTEGER,               -- 成交时的模型排名
    score_at_deal FLOAT,                -- 成交时的模型分数
    reason TEXT,                        -- 信号来源: e.g., 'rank_out', 'stop_loss'
    pnl_ratio FLOAT,                    -- 该笔订单最终结项盈亏比 (仅针对卖出单)
    hold_days INTEGER                   -- 持仓天数
);
```

### 2.4 模型稠密输出表 `bt_model_outputs`
**最为核心。** 存储模型对全市场的预测结果，用于回溯信号稳定性。
```sql
CREATE TABLE bt_model_outputs (
    exp_id UUID REFERENCES bt_experiment(exp_id),
    date DATE NOT NULL,
    symbol TEXT NOT NULL,
    score FLOAT NOT NULL,               -- 预测分
    rank INTEGER NOT NULL,              -- 截面排名
    extra_scores JSONB,                 -- 扩展: {exit_prob: 0.1, volatility: 0.02}
    PRIMARY KEY (exp_id, date, symbol)
);
```

### 2.5 拓扑与关联表 `bt_model_links` (针对 GNN 等结构模型)
记录节点间的动态关系。
```sql
CREATE TABLE bt_model_links (
    exp_id UUID REFERENCES bt_experiment(exp_id),
    date DATE NOT NULL,
    source TEXT NOT NULL,
    target TEXT NOT NULL,
    weight FLOAT NOT NULL,
    link_type TEXT DEFAULT 'attention', -- 链接类型: attention, correlation
    PRIMARY KEY (exp_id, date, source, target, link_type)
);
```

### 2.6 隐空间特征表 `bt_embeddings`
记录节点 Embedding，用于 T-SNE 聚类归因。
```sql
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE bt_embeddings (
    exp_id UUID REFERENCES bt_experiment(exp_id),
    date DATE NOT NULL,
    symbol TEXT NOT NULL,
    vec vector(128),                    -- 降维后的隐向量
    PRIMARY KEY (exp_id, date, symbol)
);
```

---

## 3. 计算端设计 (Go)

Go 后端作为服务引擎，需实现以下计算逻辑：

### 3.1 基础统计算子 (Basic Stats)
- **输入**：`bt_ledger` 与 `bt_trades`。
- **输出**：
    - `Win Rate`：`count(pnl_ratio > 0) / count(total_trades)`。
    - `Profit Factor`：`sum(positive_pnl) / abs(sum(negative_pnl))`。
    - `Sharp Ratio`：基于 `nav` 序列的日对数收益率。

### 3.2 深度分析算子 (Advanced Attribution)
- **换手归因 (Turnover Attribution)**：对比 `bt_trades` 中的 `reason` 与 `bt_model_outputs`。分析有多少卖出是因为排名从 5 掉到 6（极其接近），从而量化信号噪声。
- **结构敏感度 (Structural Analysis)**：通过 `bt_model_links` 寻找亏损单的“传染源”。

---

## 4. 展示端设计 (Python/Streamlit)

### 4.1 核心视图：Qlib 样式经典报告
- **Top Panel**：总收益、年化、回撤、夏普。
- **Trade Stat Panel**：胜率、获利因子、盈亏比（15.52）。
- **Data Table**：每日净值与成交细节的滚动清单。

### 4.2 诊断视图：GNN 特色
- **Rank Jump 轨迹**：在 K 线图中叠加 `bt_model_outputs` 中的 Rank 变化线。
- **邻居权重图**：点击任意买卖点，动态展示当时 `bt_model_links` 中权重最高的邻居，解释模型为何“看多”或“突然看空”。
- **T-SNE 空间分布**：从 `bt_embeddings` 提取数据，展示不同盈利能力的订单在隐空间的簇分布。

---

## 5. 工作流数据转换指南 (Cursor 集成用)

**步骤 1：存储 (Python 端)**
回测循环中，每 $T$ 天将 `DataFrame` 或 `Pytorch Tensor` 解构成上述 SQL 表结构并执行 `upsert`。

**步骤 2：聚合 (Go 端)**
定义统一的 `AnalysisRequest` 结构，允许用户传入 `backtest_id` 和任意 `time_range`。Go 从数据库取数后，在内存中完成截面指标聚合。

**步骤 3：显示 (Python 端)**
使用 `plotly.subplots` 创建多图联动：
- 鼠标滑过收益曲线 -> 显示当日买卖订单。
- 鼠标点开订单 -> 查看模型在该节点附近的拓扑权重。

---

## 6. 归因逻辑矩阵 (常见分析建议)

| 分析目标 | 考察表 | 考察指标 |
| :--- | :--- | :--- |
| **为什么换手率高？** | `bt_model_outputs` | 统计 Rank 在 TopK 边缘波动的频率 |
| **为什么回撤大？** | `bt_ledger` vs 基准数据 | 寻找 NAV 与基准偏离的方向性 |
| **模型是否捕捉了共振？**| `bt_model_links` | 分析高盈利单产生时邻居权重的集中度 |
| **决策是否提前？** | `bt_model_outputs` | 观察分数上涨是否领先于股价实际上涨 |
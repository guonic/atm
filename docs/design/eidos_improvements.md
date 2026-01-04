# Eidos 系统设计改进建议

本文档针对 `eidos.md` 设计文档中发现的问题，提供具体的改进建议。

## 1. 数据库设计改进

### 1.1 添加索引设计

```sql
-- bt_experiment 表索引
CREATE INDEX idx_bt_experiment_created_at ON bt_experiment(created_at DESC);
CREATE INDEX idx_bt_experiment_dates ON bt_experiment(start_date, end_date);
CREATE INDEX idx_bt_experiment_model_type ON bt_experiment(model_type) WHERE model_type IS NOT NULL;

-- bt_ledger 表索引（时间序列优化）
CREATE INDEX idx_bt_ledger_exp_date ON bt_ledger(exp_id, date DESC);
-- 如果使用 TimescaleDB，转换为 Hypertable
SELECT create_hypertable('bt_ledger', 'date', 
    chunk_time_interval => INTERVAL '1 month',
    if_not_exists => TRUE);

-- bt_trades 表索引（查询优化）
CREATE INDEX idx_bt_trades_exp_time ON bt_trades(exp_id, deal_time DESC);
CREATE INDEX idx_bt_trades_symbol ON bt_trades(symbol, deal_time DESC);
CREATE INDEX idx_bt_trades_exp_symbol ON bt_trades(exp_id, symbol, deal_time);
CREATE INDEX idx_bt_trades_reason ON bt_trades(reason) WHERE reason IS NOT NULL;
CREATE INDEX idx_bt_trades_pnl ON bt_trades(exp_id, pnl_ratio) WHERE pnl_ratio IS NOT NULL;

-- bt_model_outputs 表索引（核心表，需要重点优化）
CREATE INDEX idx_bt_model_outputs_exp_date ON bt_model_outputs(exp_id, date DESC);
CREATE INDEX idx_bt_model_outputs_exp_rank ON bt_model_outputs(exp_id, date, rank);
CREATE INDEX idx_bt_model_outputs_exp_score ON bt_model_outputs(exp_id, date, score DESC);
CREATE INDEX idx_bt_model_outputs_symbol_date ON bt_model_outputs(symbol, date DESC);
-- JSONB 字段 GIN 索引
CREATE INDEX idx_bt_model_outputs_extra_scores ON bt_model_outputs USING GIN (extra_scores);
-- 转换为 TimescaleDB Hypertable
SELECT create_hypertable('bt_model_outputs', 'date', 
    chunk_time_interval => INTERVAL '1 month',
    if_not_exists => TRUE);

-- bt_model_links 表索引
CREATE INDEX idx_bt_model_links_exp_date ON bt_model_links(exp_id, date DESC);
CREATE INDEX idx_bt_model_links_source ON bt_model_links(exp_id, date, source, weight DESC);
CREATE INDEX idx_bt_model_links_target ON bt_model_links(exp_id, date, target, weight DESC);
CREATE INDEX idx_bt_model_links_type ON bt_model_links(link_type) WHERE link_type IS NOT NULL;
-- 转换为 TimescaleDB Hypertable
SELECT create_hypertable('bt_model_links', 'date', 
    chunk_time_interval => INTERVAL '1 month',
    if_not_exists => TRUE);

-- bt_embeddings 表索引（向量相似度搜索）
CREATE INDEX idx_bt_embeddings_exp_date ON bt_embeddings(exp_id, date DESC);
-- 向量索引（使用 ivfflat 或 hnsw）
CREATE INDEX idx_bt_embeddings_vec ON bt_embeddings 
    USING ivfflat (vec vector_cosine_ops) WITH (lists = 100);
-- 转换为 TimescaleDB Hypertable
SELECT create_hypertable('bt_embeddings', 'date', 
    chunk_time_interval => INTERVAL '1 month',
    if_not_exists => TRUE);
```

### 1.2 改进表结构设计

```sql
-- 改进后的 bt_experiment 表
CREATE TABLE bt_experiment (
    exp_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL,
    model_type TEXT,
    engine_type TEXT,
    start_date DATE NOT NULL,
    end_date DATE NOT NULL,
    config JSONB NOT NULL,
    metrics_summary JSONB,
    version INTEGER DEFAULT 1,  -- 版本控制
    status TEXT DEFAULT 'running',  -- running / completed / failed
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT chk_dates CHECK (end_date >= start_date)
);

-- 改进后的 bt_ledger 表（添加时区）
CREATE TABLE bt_ledger (
    exp_id UUID NOT NULL REFERENCES bt_experiment(exp_id) ON DELETE CASCADE,
    date DATE NOT NULL,
    nav NUMERIC(18, 4) NOT NULL,
    cash NUMERIC(18, 4),
    market_value NUMERIC(18, 4),
    deal_amount NUMERIC(18, 2),
    turnover_rate FLOAT,
    pos_count INTEGER,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (exp_id, date)
);

-- 改进后的 bt_trades 表（优化主键策略）
CREATE TABLE bt_trades (
    trade_id BIGSERIAL,  -- 保留作为唯一标识，但不作为主键
    exp_id UUID NOT NULL REFERENCES bt_experiment(exp_id) ON DELETE CASCADE,
    symbol TEXT NOT NULL,
    deal_time TIMESTAMPTZ NOT NULL,  -- 使用 TIMESTAMPTZ
    side INTEGER NOT NULL CHECK (side IN (1, -1)),
    price NUMERIC(18, 4) NOT NULL CHECK (price > 0),
    amount INTEGER NOT NULL CHECK (amount > 0),
    rank_at_deal INTEGER,
    score_at_deal FLOAT,
    reason TEXT,
    pnl_ratio FLOAT,
    hold_days INTEGER,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (exp_id, deal_time, symbol, side),  -- 复合主键
    CONSTRAINT chk_side CHECK (side IN (1, -1))
);

-- 改进后的 bt_model_outputs 表
CREATE TABLE bt_model_outputs (
    exp_id UUID NOT NULL REFERENCES bt_experiment(exp_id) ON DELETE CASCADE,
    date DATE NOT NULL,
    symbol TEXT NOT NULL,
    score FLOAT NOT NULL,
    rank INTEGER NOT NULL CHECK (rank > 0),
    extra_scores JSONB,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (exp_id, date, symbol)
);

-- 改进后的 bt_embeddings 表（支持可变维度）
CREATE TABLE bt_embeddings (
    exp_id UUID NOT NULL REFERENCES bt_experiment(exp_id) ON DELETE CASCADE,
    date DATE NOT NULL,
    symbol TEXT NOT NULL,
    vec vector,  -- 不固定维度，但需要指定维度约束
    vec_dim INTEGER NOT NULL,  -- 记录向量维度
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (exp_id, date, symbol),
    CONSTRAINT chk_vec_dim CHECK (vec_dim > 0 AND vec_dim <= 2048)
);
```

### 1.3 添加数据压缩和保留策略

```sql
-- 对时间序列表启用压缩策略（30天前的数据自动压缩）
SELECT add_compression_policy('bt_ledger', INTERVAL '30 days');
SELECT add_compression_policy('bt_model_outputs', INTERVAL '30 days');
SELECT add_compression_policy('bt_model_links', INTERVAL '30 days');
SELECT add_compression_policy('bt_embeddings', INTERVAL '30 days');

-- 对交易表启用压缩策略（90天前的数据自动压缩）
SELECT add_compression_policy('bt_trades', INTERVAL '90 days');

-- 可选：对历史数据设置保留策略（如保留2年）
-- SELECT add_retention_policy('bt_ledger', INTERVAL '2 years');
```

## 2. 架构设计改进

### 2.1 定义 Go 与 Python 接口

#### REST API 设计（推荐）

```go
// Go 端 API 定义
type AnalysisRequest struct {
    ExpID     string    `json:"exp_id"`
    StartDate time.Time `json:"start_date,omitempty"`
    EndDate   time.Time `json:"end_date,omitempty"`
    Symbols   []string  `json:"symbols,omitempty"`
    Metrics   []string  `json:"metrics,omitempty"`  // ["win_rate", "sharpe", "turnover"]
}

type AnalysisResponse struct {
    ExpID      string                 `json:"exp_id"`
    Metrics    map[string]interface{} `json:"metrics"`
    TimeSeries []TimeSeriesPoint      `json:"time_series,omitempty"`
    Error      string                 `json:"error,omitempty"`
}

// API 端点
POST   /api/v1/analysis/basic-stats      // 基础统计
POST   /api/v1/analysis/turnover         // 换手归因
POST   /api/v1/analysis/structural       // 结构敏感度
GET    /api/v1/experiments/{exp_id}      // 获取实验信息
```

#### gRPC 接口设计（高性能场景）

```protobuf
// pkg/proto/backtest_attribution.proto
syntax = "proto3";

package backtest.attribution;

service AttributionService {
    rpc GetBasicStats(AnalysisRequest) returns (BasicStatsResponse);
    rpc GetTurnoverAttribution(AnalysisRequest) returns (TurnoverAttributionResponse);
    rpc GetStructuralAnalysis(StructuralRequest) returns (StructuralAnalysisResponse);
}

message AnalysisRequest {
    string exp_id = 1;
    string start_date = 2;  // ISO 8601 format
    string end_date = 3;
    repeated string symbols = 4;
    repeated string metrics = 5;
}
```

### 2.2 批量操作策略

```python
# Python 端批量插入示例
from sqlalchemy import create_engine
from sqlalchemy.dialects.postgresql import insert

def batch_upsert_model_outputs(engine, exp_id, data_list, batch_size=1000):
    """
    批量插入模型输出数据
    
    Args:
        engine: SQLAlchemy engine
        exp_id: 实验ID
        data_list: 数据列表，每个元素为 (date, symbol, score, rank, extra_scores)
        batch_size: 批次大小
    """
    with engine.begin() as conn:
        for i in range(0, len(data_list), batch_size):
            batch = data_list[i:i+batch_size]
            stmt = insert(bt_model_outputs).values([
                {
                    'exp_id': exp_id,
                    'date': date,
                    'symbol': symbol,
                    'score': score,
                    'rank': rank,
                    'extra_scores': extra_scores
                }
                for date, symbol, score, rank, extra_scores in batch
            ])
            stmt = stmt.on_conflict_do_update(
                index_elements=['exp_id', 'date', 'symbol'],
                set_=dict(
                    score=stmt.excluded.score,
                    rank=stmt.excluded.rank,
                    extra_scores=stmt.excluded.extra_scores
                )
            )
            conn.execute(stmt)
```

### 2.3 缓存策略

```go
// Go 端缓存设计（使用 Redis）
type CacheManager struct {
    redis *redis.Client
    ttl   time.Duration
}

func (c *CacheManager) GetBasicStats(expID string) (*BasicStats, error) {
    key := fmt.Sprintf("attribution:basic_stats:%s", expID)
    
    // 尝试从缓存获取
    cached, err := c.redis.Get(ctx, key).Result()
    if err == nil {
        var stats BasicStats
        json.Unmarshal([]byte(cached), &stats)
        return &stats, nil
    }
    
    // 缓存未命中，从数据库计算
    stats := calculateBasicStats(expID)
    
    // 写入缓存（TTL: 1小时）
    data, _ := json.Marshal(stats)
    c.redis.Set(ctx, key, data, c.ttl)
    
    return stats, nil
}
```

## 3. 数据一致性改进

### 3.1 事务处理

```python
# Python 端事务示例
def save_backtest_results(engine, exp_id, ledger_data, trades_data, model_outputs):
    """
    原子性保存回测结果
    """
    with engine.begin() as conn:
        try:
            # 1. 保存账户流水
            conn.execute(bt_ledger.insert(), ledger_data)
            
            # 2. 保存交易订单
            conn.execute(bt_trades.insert(), trades_data)
            
            # 3. 保存模型输出
            conn.execute(bt_model_outputs.insert(), model_outputs)
            
            # 4. 更新实验状态
            conn.execute(
                bt_experiment.update()
                .where(bt_experiment.c.exp_id == exp_id)
                .values(status='completed', updated_at=func.now())
            )
        except Exception as e:
            # 回滚所有操作
            raise
```

### 3.2 数据校验

```sql
-- 添加数据校验约束
ALTER TABLE bt_ledger ADD CONSTRAINT chk_nav_positive CHECK (nav > 0);
ALTER TABLE bt_ledger ADD CONSTRAINT chk_turnover_rate CHECK (turnover_rate >= 0 AND turnover_rate <= 1);

ALTER TABLE bt_trades ADD CONSTRAINT chk_amount_positive CHECK (amount > 0);
ALTER TABLE bt_trades ADD CONSTRAINT chk_hold_days_non_negative CHECK (hold_days >= 0);

ALTER TABLE bt_model_outputs ADD CONSTRAINT chk_rank_positive CHECK (rank > 0);
```

## 4. 查询优化

### 4.1 物化视图

```sql
-- 创建物化视图用于快速查询实验汇总指标
CREATE MATERIALIZED VIEW mv_experiment_summary AS
SELECT 
    exp_id,
    COUNT(DISTINCT date) as trading_days,
    MIN(nav) as min_nav,
    MAX(nav) as max_nav,
    AVG(turnover_rate) as avg_turnover_rate,
    SUM(deal_amount) as total_deal_amount
FROM bt_ledger
GROUP BY exp_id;

CREATE UNIQUE INDEX ON mv_experiment_summary(exp_id);

-- 定期刷新物化视图（可通过定时任务）
REFRESH MATERIALIZED VIEW CONCURRENTLY mv_experiment_summary;
```

### 4.2 查询优化建议

```sql
-- 使用覆盖索引优化常见查询
CREATE INDEX idx_bt_trades_covering ON bt_trades(exp_id, deal_time, symbol, side, price, amount, pnl_ratio);

-- 使用部分索引优化特定查询
CREATE INDEX idx_bt_trades_profitable ON bt_trades(exp_id, pnl_ratio) 
    WHERE pnl_ratio > 0;  -- 只索引盈利订单
```

## 5. 监控和运维

### 5.1 添加监控指标

```sql
-- 创建监控表
CREATE TABLE bt_system_metrics (
    metric_name TEXT NOT NULL,
    metric_value NUMERIC,
    recorded_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (metric_name, recorded_at)
);

-- 定期记录表大小、行数等指标
INSERT INTO bt_system_metrics (metric_name, metric_value)
SELECT 
    'bt_model_outputs_row_count',
    COUNT(*)
FROM bt_model_outputs;
```

### 5.2 数据归档策略

```sql
-- 创建归档表（用于归档历史数据）
CREATE TABLE bt_model_outputs_archive (
    LIKE bt_model_outputs INCLUDING ALL
) PARTITION BY RANGE (date);

-- 定期将旧数据移动到归档表
INSERT INTO bt_model_outputs_archive
SELECT * FROM bt_model_outputs
WHERE date < CURRENT_DATE - INTERVAL '2 years';

DELETE FROM bt_model_outputs
WHERE date < CURRENT_DATE - INTERVAL '2 years';
```

## 6. 总结

### 主要改进点

1. ✅ **添加完整的索引设计**：包括时间序列索引、JSONB GIN 索引、向量索引
2. ✅ **使用 TimescaleDB Hypertable**：优化时间序列数据存储和查询
3. ✅ **改进主键设计**：`bt_trades` 使用复合主键
4. ✅ **添加外键级联删除**：保证数据一致性
5. ✅ **支持时区**：使用 `TIMESTAMPTZ`
6. ✅ **定义 API 接口**：明确 Go 与 Python 的交互方式
7. ✅ **批量操作策略**：优化大数据量插入
8. ✅ **缓存策略**：提升查询性能
9. ✅ **数据校验**：添加 CHECK 约束
10. ✅ **物化视图**：优化汇总查询

### 实施优先级

**P0（必须）**：
- 索引设计
- 外键级联删除
- 时区处理
- API 接口定义

**P1（重要）**：
- TimescaleDB Hypertable
- 批量操作策略
- 数据校验约束

**P2（优化）**：
- 缓存策略
- 物化视图
- 数据归档策略


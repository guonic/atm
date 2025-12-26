# Tushare 数据同步工具存储结构设计

## 背景

参考 Tushare API 文档设计同步工具，需要确定合适的数据存储结构。本文档讨论不同数据类型的存储方案。

## Tushare API 数据类型分析

### 1. 基础数据（低频更新）
- **股票基本信息** (`stock_basic`) - 已实现
- **交易日历** (`trade_cal`) - 已实现
- **股票分类** (`stock_classify`) - 部分实现
- **行业分类** (`index_classify`)
- **概念分类** (`concept`)
- **板块分类** (`stock_board`)

### 2. 财务数据（季度/年度更新）
- **财务指标** (`fina_indicator`)
- **利润表** (`income`)
- **资产负债表** (`balancesheet`)
- **现金流量表** (`cashflow`)
- **业绩预告** (`forecast`)
- **业绩快报** (`express`)

### 3. 行情数据（高频更新）
- **K线数据** (`daily`, `weekly`, `monthly`) - 已实现
- **分钟线数据** (`minute`) - 已实现
- **复权因子** (`adj_factor`)
- **涨跌停数据** (`limit_list`)
- **停复牌信息** (`suspend`)

### 4. 资金流向数据（日频更新）
- **资金流向** (`moneyflow`)
- **大单交易** (`stk_moneyflow`)
- **主力资金** (`fund_flow`)
- **北向资金** (`moneyflow_hsgt`)

### 5. 股东数据（季度/年度更新）
- **股东人数** (`stk_holdernumber`)
- **十大股东** (`top10_holders`)
- **十大流通股东** (`top10_floatholders`)
- **股东变动** (`stk_holdertrade`)

### 6. 公司公告（事件驱动）
- **公司公告** (`ann`)
- **业绩预告** (`forecast`)
- **分红送股** (`dividend`)

## 存储结构设计方案

### 方案一：按数据更新频率分类存储（推荐）

#### 1. 关系型数据表（PostgreSQL/TimescaleDB）

**优点：**
- 结构化存储，支持复杂查询
- 事务支持，数据一致性
- 适合关系型数据（股票-财务、股票-股东等）

**适用场景：**
- 股票基本信息
- 财务数据（季度/年度）
- 股东数据
- 公司公告
- 分类信息

**表结构设计：**

```sql
-- 财务数据表（按报告期）
CREATE TABLE stock_finance_indicator (
    ts_code VARCHAR(20) NOT NULL,
    report_date DATE NOT NULL,
    end_date DATE NOT NULL,
    -- 财务指标字段
    total_revenue DECIMAL(20, 2),
    net_profit DECIMAL(20, 2),
    total_assets DECIMAL(20, 2),
    -- ... 其他字段
    update_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (ts_code, report_date, end_date)
);

-- 股东数据表
CREATE TABLE stock_holder (
    ts_code VARCHAR(20) NOT NULL,
    report_date DATE NOT NULL,
    holder_name VARCHAR(255),
    hold_amount DECIMAL(20, 2),
    hold_ratio DECIMAL(10, 4),
    -- ... 其他字段
    update_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (ts_code, report_date, holder_name)
);

-- 公司公告表
CREATE TABLE stock_announcement (
    id BIGSERIAL PRIMARY KEY,
    ts_code VARCHAR(20) NOT NULL,
    ann_date DATE NOT NULL,
    ann_type VARCHAR(50),
    title TEXT,
    content TEXT,
    -- ... 其他字段
    update_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX idx_ann_ts_code_date ON stock_announcement(ts_code, ann_date);
```

#### 2. 时序数据表（TimescaleDB Hypertable）

**优点：**
- 专为时间序列数据优化
- 高效的写入和查询性能
- 自动分区管理
- 支持时间范围查询优化

**适用场景：**
- K线数据（日线、周线、月线、分钟线）- **已实现**
- 资金流向数据（日频）
- 涨跌停数据（日频）
- 复权因子（日频）

**表结构设计：**

```sql
-- 资金流向数据（转换为时序表）
SELECT create_hypertable('stock_moneyflow', 'trade_date');

CREATE TABLE stock_moneyflow (
    ts_code VARCHAR(20) NOT NULL,
    trade_date DATE NOT NULL,
    buy_sm_amount DECIMAL(20, 2),  -- 小单买入
    sell_sm_amount DECIMAL(20, 2), -- 小单卖出
    buy_md_amount DECIMAL(20, 2),  -- 中单买入
    sell_md_amount DECIMAL(20, 2), -- 中单卖出
    buy_lg_amount DECIMAL(20, 2),  -- 大单买入
    sell_lg_amount DECIMAL(20, 2), -- 大单卖出
    buy_elg_amount DECIMAL(20, 2), -- 特大单买入
    sell_elg_amount DECIMAL(20, 2),-- 特大单卖出
    net_mf_amount DECIMAL(20, 2),  -- 净流入
    update_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (ts_code, trade_date)
);
CREATE INDEX idx_mf_ts_code ON stock_moneyflow(ts_code);
CREATE INDEX idx_mf_date ON stock_moneyflow(trade_date);
```

#### 3. JSONB 字段存储（PostgreSQL）

**优点：**
- 灵活的字段结构
- 适合非结构化或半结构化数据
- 支持 JSON 查询和索引

**适用场景：**
- 公司公告内容（富文本）
- 动态字段的财务数据
- API 返回的原始 JSON 数据（备份）

**表结构设计：**

```sql
-- 使用 JSONB 存储原始数据
CREATE TABLE stock_finance_raw (
    ts_code VARCHAR(20) NOT NULL,
    report_date DATE NOT NULL,
    api_name VARCHAR(50) NOT NULL,
    raw_data JSONB NOT NULL,  -- 存储完整的 API 返回数据
    update_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (ts_code, report_date, api_name)
);

-- 创建 GIN 索引支持 JSONB 查询
CREATE INDEX idx_finance_raw_data ON stock_finance_raw USING GIN (raw_data);
```

### 方案二：按数据访问模式分类存储

#### 1. 热数据（频繁查询）
- **存储位置：** PostgreSQL 主表 + 索引优化
- **数据：** K线数据、资金流向、涨跌停
- **优化：** 分区表、索引、物化视图

#### 2. 温数据（偶尔查询）
- **存储位置：** PostgreSQL 普通表
- **数据：** 财务数据、股东数据
- **优化：** 索引、归档策略

#### 3. 冷数据（很少查询）
- **存储位置：** PostgreSQL 归档表 或 对象存储（S3/MinIO）
- **数据：** 历史公告、历史财务数据（5年以上）
- **优化：** 压缩、归档

## 推荐方案：混合存储结构

### 核心原则

1. **时序数据 → TimescaleDB Hypertable**
   - K线数据（已实现）
   - 资金流向数据
   - 涨跌停数据
   - 复权因子

2. **关系型数据 → PostgreSQL 普通表**
   - 股票基本信息（已实现）
   - 财务数据（按报告期）
   - 股东数据
   - 分类信息

3. **文档数据 → PostgreSQL JSONB**
   - 公司公告内容
   - API 原始数据备份
   - 动态字段数据

4. **状态管理 → 独立状态表**
   - 同步状态（已实现：`ingestion_state`）
   - 任务锁（已实现：`task_lock`）

### 具体表设计建议

#### 1. 资金流向数据（时序表）

```sql
-- 转换为时序表
SELECT create_hypertable('stock_moneyflow', 'trade_date', 
    chunk_time_interval => INTERVAL '1 year');

CREATE TABLE stock_moneyflow (
    ts_code VARCHAR(20) NOT NULL,
    trade_date DATE NOT NULL,
    -- 资金流向字段
    buy_sm_amount DECIMAL(20, 2),
    sell_sm_amount DECIMAL(20, 2),
    -- ... 其他字段
    update_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (ts_code, trade_date)
);
```

#### 2. 财务数据（关系表）

```sql
-- 财务指标表（按报告期）
CREATE TABLE stock_finance_indicator (
    ts_code VARCHAR(20) NOT NULL,
    end_date DATE NOT NULL,  -- 报告期结束日期
    report_type VARCHAR(10) NOT NULL,  -- 报告类型：Q1/Q2/Q3/Q4/年报
    -- 财务指标字段（根据 Tushare API 字段定义）
    total_revenue DECIMAL(20, 2),
    net_profit DECIMAL(20, 2),
    -- ... 其他字段
    update_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (ts_code, end_date, report_type)
);
CREATE INDEX idx_finance_ts_code ON stock_finance_indicator(ts_code);
CREATE INDEX idx_finance_end_date ON stock_finance_indicator(end_date);
```

#### 3. 股东数据（关系表）

```sql
-- 十大股东表
CREATE TABLE stock_top10_holders (
    ts_code VARCHAR(20) NOT NULL,
    end_date DATE NOT NULL,
    holder_rank INTEGER NOT NULL,  -- 排名 1-10
    holder_name VARCHAR(255),
    hold_amount DECIMAL(20, 2),
    hold_ratio DECIMAL(10, 4),
    -- ... 其他字段
    update_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (ts_code, end_date, holder_rank)
);
```

#### 4. 公司公告（文档表 + JSONB）

```sql
-- 公告表（结构化字段 + JSONB 原始数据）
CREATE TABLE stock_announcement (
    id BIGSERIAL PRIMARY KEY,
    ts_code VARCHAR(20) NOT NULL,
    ann_date DATE NOT NULL,
    ann_type VARCHAR(50),
    title TEXT,
    -- 结构化字段用于快速查询
    -- JSONB 存储完整内容
    raw_data JSONB,  -- 存储完整的公告内容、附件链接等
    update_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX idx_ann_ts_code_date ON stock_announcement(ts_code, ann_date);
CREATE INDEX idx_ann_type ON stock_announcement(ann_type);
CREATE INDEX idx_ann_raw_data ON stock_announcement USING GIN (raw_data);
```

## 同步状态管理

### 现有实现
- `ingestion_state` 表：通用同步状态
- `stock_kline_sync_state` 表：K线同步状态

### 建议扩展

```sql
-- 通用同步状态表（已存在，可复用）
-- 为不同数据类型创建不同的 task_name
-- 例如：
-- - 'tushare_moneyflow_day'
-- - 'tushare_finance_indicator'
-- - 'tushare_top10_holders'
-- - 'tushare_announcement'
```

## 数据更新策略

### 1. 全量同步（首次）
- 获取所有历史数据
- 批量插入数据库

### 2. 增量同步（日常）
- 基于 `last_synced_date` 或 `last_synced_time`
- 只同步新数据或更新的数据

### 3. 覆盖更新 vs 追加
- **覆盖更新（upsert）：** 财务数据、股东数据（同一报告期可能更新）
- **追加（append）：** K线数据、资金流向（历史数据不变）

## 索引策略

### 时序数据索引
```sql
-- 主键索引（自动创建）
PRIMARY KEY (ts_code, trade_date)

-- 单字段索引（用于单股票查询）
CREATE INDEX idx_ts_code ON table_name(ts_code);

-- 时间范围索引（用于时间范围查询）
CREATE INDEX idx_trade_date ON table_name(trade_date);
```

### 关系数据索引
```sql
-- 主键索引
PRIMARY KEY (ts_code, report_date)

-- 外键索引（如果有关联）
CREATE INDEX idx_ts_code ON table_name(ts_code);

-- 查询字段索引
CREATE INDEX idx_report_date ON table_name(report_date);
CREATE INDEX idx_ann_type ON table_name(ann_type);
```

## 总结

### 推荐存储结构

1. **时序数据（TimescaleDB Hypertable）**
   - K线数据 ✅（已实现）
   - 资金流向数据
   - 涨跌停数据
   - 复权因子

2. **关系数据（PostgreSQL 普通表）**
   - 股票基本信息 ✅（已实现）
   - 财务数据（按报告期）
   - 股东数据
   - 分类信息

3. **文档数据（PostgreSQL JSONB）**
   - 公司公告
   - API 原始数据备份

4. **状态管理（独立表）**
   - `ingestion_state` ✅（已实现）
   - `stock_kline_sync_state` ✅（已实现）

### 下一步

1. 确定具体要同步的 Tushare API（doc_id 181, 335）
2. 根据 API 返回的数据结构设计具体的表结构
3. 实现同步工具（参考现有的 `sync_kline.py`）
4. 实现数据模型和 Repository（参考现有的 `StockKlineDayRepo`）

## Tushare API 181: 申万行业分类 (index_classify)

### API 说明
- **接口名称**: `index_classify`
- **数据特点**: 层级结构（L1/L2/L3），树形关系
- **更新频率**: 低频（行业分类相对稳定，但会有版本更新：SW2014, SW2021）

### 数据字段
- `index_code`: 指数代码（如 801020.SI）
- `industry_name`: 行业名称（如 "采掘"）
- `parent_code`: 父级代码（一级为0）
- `level`: 行业层级（L1/L2/L3）
- `industry_code`: 行业代码
- `is_pub`: 是否发布了指数
- `src`: 行业分类版本（SW2014/SW2021）

### 存储结构设计

#### 方案 A: 单表存储（推荐）

**优点：**
- 结构简单，查询方便
- 支持层级查询（通过 parent_code）
- 易于维护

**表结构：**

```sql
CREATE TABLE stock_industry_classify (
    index_code VARCHAR(20) NOT NULL,
    industry_name VARCHAR(100) NOT NULL,
    parent_code VARCHAR(20) NOT NULL,  -- 父级代码，一级为 '0'
    level VARCHAR(10) NOT NULL,  -- L1/L2/L3
    industry_code VARCHAR(20),
    is_pub VARCHAR(1),  -- 是否发布了指数
    src VARCHAR(20) NOT NULL,  -- SW2014/SW2021
    update_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (index_code, src)
);

-- 索引设计
CREATE INDEX idx_industry_parent ON stock_industry_classify(parent_code, src);
CREATE INDEX idx_industry_level ON stock_industry_classify(level, src);
CREATE INDEX idx_industry_src ON stock_industry_classify(src);
```

**查询示例：**
```sql
-- 查询一级行业
SELECT * FROM stock_industry_classify WHERE level = 'L1' AND src = 'SW2021';

-- 查询某个行业的子行业
SELECT * FROM stock_industry_classify WHERE parent_code = '801020.SI' AND src = 'SW2021';

-- 查询行业树（递归查询）
WITH RECURSIVE industry_tree AS (
    SELECT * FROM stock_industry_classify WHERE index_code = '801020.SI' AND src = 'SW2021'
    UNION ALL
    SELECT i.* FROM stock_industry_classify i
    INNER JOIN industry_tree it ON i.parent_code = it.index_code
    WHERE i.src = 'SW2021'
)
SELECT * FROM industry_tree;
```

#### 方案 B: 分层表存储

**优点：**
- 查询性能更好（不需要过滤 level）
- 表结构更清晰

**缺点：**
- 需要维护多个表
- 跨层级查询需要 UNION

**表结构：**

```sql
-- 一级行业表
CREATE TABLE stock_industry_l1 (
    index_code VARCHAR(20) PRIMARY KEY,
    industry_name VARCHAR(100) NOT NULL,
    industry_code VARCHAR(20),
    is_pub VARCHAR(1),
    src VARCHAR(20) NOT NULL,
    update_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (index_code, src)
);

-- 二级行业表
CREATE TABLE stock_industry_l2 (
    index_code VARCHAR(20) PRIMARY KEY,
    industry_name VARCHAR(100) NOT NULL,
    parent_code VARCHAR(20) NOT NULL,  -- 引用 L1 的 index_code
    industry_code VARCHAR(20),
    is_pub VARCHAR(1),
    src VARCHAR(20) NOT NULL,
    update_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (parent_code, src) REFERENCES stock_industry_l1(index_code, src),
    UNIQUE (index_code, src)
);

-- 三级行业表
CREATE TABLE stock_industry_l3 (
    index_code VARCHAR(20) PRIMARY KEY,
    industry_name VARCHAR(100) NOT NULL,
    parent_code VARCHAR(20) NOT NULL,  -- 引用 L2 的 index_code
    industry_code VARCHAR(20),
    is_pub VARCHAR(1),
    src VARCHAR(20) NOT NULL,
    update_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (parent_code, src) REFERENCES stock_industry_l2(index_code, src),
    UNIQUE (index_code, src)
);
```

### 推荐方案：方案 A（单表存储）

**理由：**
1. 结构简单，易于维护
2. 支持灵活的层级查询（递归查询）
3. 版本管理方便（通过 src 字段区分）
4. 符合现有代码风格（参考 `StockClassify` 模型）

### 数据模型设计

```python
class StockIndustryClassify(BaseModel):
    """申万行业分类模型"""
    
    index_code: str = Field(..., description="指数代码 (e.g., 801020.SI)")
    industry_name: str = Field(..., description="行业名称 (e.g., 采掘)")
    parent_code: str = Field(..., description="父级代码，一级为 '0'")
    level: str = Field(..., description="行业层级 (L1/L2/L3)")
    industry_code: Optional[str] = Field(None, description="行业代码")
    is_pub: Optional[str] = Field(None, description="是否发布了指数")
    src: str = Field(..., description="行业分类版本 (SW2014/SW2021)")
    update_time: Optional[datetime] = Field(None, description="更新时间")
```

### 同步策略

1. **全量同步**：每次同步时，先删除指定版本（src）的所有数据，再插入新数据
2. **版本管理**：支持多个版本并存（SW2014, SW2021）
3. **更新频率**：低频更新（行业分类相对稳定）

## Tushare API 335: 申万行业成分构成(分级) (index_member_all)

### API 说明
- **接口名称**: `index_member_all`
- **数据特点**: 股票与行业的关联关系，支持历史数据（纳入/剔除日期）
- **更新频率**: 中频（股票行业分类会变化，需要跟踪历史）

### 数据字段
- `l1_code`: 一级行业代码
- `l1_name`: 一级行业名称
- `l2_code`: 二级行业代码
- `l2_name`: 二级行业名称
- `l3_code`: 三级行业代码
- `l3_name`: 三级行业名称
- `ts_code`: 成分股票代码
- `name`: 成分股票名称
- `in_date`: 纳入日期
- `out_date`: 剔除日期（NULL 表示当前仍在）
- `is_new`: 是否最新（Y/N）

### 存储结构设计

#### 方案 A: 单表存储（推荐）

**优点：**
- 结构简单，查询方便
- 支持历史数据查询（通过 in_date/out_date）
- 支持多级行业查询

**表结构：**

```sql
CREATE TABLE stock_industry_member (
    ts_code VARCHAR(20) NOT NULL,
    l1_code VARCHAR(20) NOT NULL,
    l1_name VARCHAR(100) NOT NULL,
    l2_code VARCHAR(20) NOT NULL,
    l2_name VARCHAR(100) NOT NULL,
    l3_code VARCHAR(20) NOT NULL,
    l3_name VARCHAR(100) NOT NULL,
    stock_name VARCHAR(100),  -- 股票名称（冗余字段，便于查询）
    in_date DATE NOT NULL,  -- 纳入日期
    out_date DATE,  -- 剔除日期（NULL 表示当前仍在）
    is_new VARCHAR(1) DEFAULT 'Y',  -- 是否最新
    update_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (ts_code, l3_code, in_date)  -- 同一股票可能多次纳入同一行业
);

-- 索引设计
CREATE INDEX idx_member_ts_code ON stock_industry_member(ts_code);
CREATE INDEX idx_member_l1_code ON stock_industry_member(l1_code);
CREATE INDEX idx_member_l2_code ON stock_industry_member(l2_code);
CREATE INDEX idx_member_l3_code ON stock_industry_member(l3_code);
CREATE INDEX idx_member_in_date ON stock_industry_member(in_date);
CREATE INDEX idx_member_out_date ON stock_industry_member(out_date);
CREATE INDEX idx_member_is_new ON stock_industry_member(is_new);
```

**查询示例：**
```sql
-- 查询某个股票的当前行业分类
SELECT * FROM stock_industry_member 
WHERE ts_code = '000001.SZ' AND (out_date IS NULL OR out_date > CURRENT_DATE)
ORDER BY in_date DESC;

-- 查询某个行业的所有成分股（当前）
SELECT * FROM stock_industry_member 
WHERE l3_code = '850531.SI' AND (out_date IS NULL OR out_date > CURRENT_DATE);

-- 查询某个行业的历史成分股
SELECT * FROM stock_industry_member 
WHERE l3_code = '850531.SI'
ORDER BY in_date DESC;

-- 查询某个股票的历史行业分类
SELECT * FROM stock_industry_member 
WHERE ts_code = '000001.SZ'
ORDER BY in_date DESC;
```

#### 方案 B: 分离当前和历史数据

**优点：**
- 当前数据查询更快（不需要过滤 out_date）
- 历史数据可以归档

**缺点：**
- 需要维护两个表
- 跨表查询需要 UNION

**表结构：**

```sql
-- 当前行业成分表（out_date IS NULL）
CREATE TABLE stock_industry_member_current (
    ts_code VARCHAR(20) NOT NULL,
    l1_code VARCHAR(20) NOT NULL,
    l1_name VARCHAR(100) NOT NULL,
    l2_code VARCHAR(20) NOT NULL,
    l2_name VARCHAR(100) NOT NULL,
    l3_code VARCHAR(20) NOT NULL,
    l3_name VARCHAR(100) NOT NULL,
    stock_name VARCHAR(100),
    in_date DATE NOT NULL,
    update_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (ts_code, l3_code)
);

-- 历史行业成分表（包含所有历史记录）
CREATE TABLE stock_industry_member_history (
    ts_code VARCHAR(20) NOT NULL,
    l1_code VARCHAR(20) NOT NULL,
    l1_name VARCHAR(100) NOT NULL,
    l2_code VARCHAR(20) NOT NULL,
    l2_name VARCHAR(100) NOT NULL,
    l3_code VARCHAR(20) NOT NULL,
    l3_name VARCHAR(100) NOT NULL,
    stock_name VARCHAR(100),
    in_date DATE NOT NULL,
    out_date DATE,
    update_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (ts_code, l3_code, in_date)
);
```

### 推荐方案：方案 A（单表存储）

**理由：**
1. 结构简单，易于维护
2. 支持完整的历史数据查询
3. 通过索引优化查询性能
4. 符合现有代码风格

### 数据模型设计

```python
class StockIndustryMember(BaseModel):
    """申万行业成分模型"""
    
    ts_code: str = Field(..., description="股票代码 (e.g., 000001.SZ)")
    l1_code: str = Field(..., description="一级行业代码")
    l1_name: str = Field(..., description="一级行业名称")
    l2_code: str = Field(..., description="二级行业代码")
    l2_name: str = Field(..., description="二级行业名称")
    l3_code: str = Field(..., description="三级行业代码")
    l3_name: str = Field(..., description="三级行业名称")
    stock_name: Optional[str] = Field(None, description="股票名称")
    in_date: date = Field(..., description="纳入日期")
    out_date: Optional[date] = Field(None, description="剔除日期 (NULL 表示当前仍在)")
    is_new: str = Field("Y", description="是否最新 (Y/N)")
    update_time: Optional[datetime] = Field(None, description="更新时间")
```

### 同步策略

1. **全量同步**：每次同步时，获取最新数据（is_new='Y'），更新当前记录
2. **历史数据**：保留所有历史记录（in_date, out_date），支持历史查询
3. **更新逻辑**：
   - 如果股票从某个行业剔除，更新 out_date
   - 如果股票纳入新行业，插入新记录
   - 如果股票在同一行业多次纳入/剔除，保留所有历史记录

## 两个 API 的关联关系

### 数据关系
- **API 181 (index_classify)**: 提供行业分类的层级结构
- **API 335 (index_member_all)**: 提供股票与行业的关联关系

### 关联查询示例

```sql
-- 查询某个行业的所有成分股（关联行业分类表）
SELECT 
    m.ts_code,
    m.stock_name,
    m.in_date,
    m.out_date,
    c.industry_code,
    c.is_pub
FROM stock_industry_member m
JOIN stock_industry_classify c ON m.l3_code = c.index_code
WHERE m.l3_code = '850531.SI' 
  AND (m.out_date IS NULL OR m.out_date > CURRENT_DATE)
  AND c.src = 'SW2021';
```

## 综合存储结构设计

### 表关系图

```
stock_industry_classify (行业分类表)
    ├── index_code (PK)
    ├── parent_code (FK -> index_code)
    └── level (L1/L2/L3)

stock_industry_member (行业成分表)
    ├── ts_code (FK -> stock_basic.ts_code)
    ├── l1_code (FK -> stock_industry_classify.index_code)
    ├── l2_code (FK -> stock_industry_classify.index_code)
    ├── l3_code (FK -> stock_industry_classify.index_code)
    ├── in_date
    └── out_date
```

### 索引优化建议

```sql
-- 行业分类表索引
CREATE INDEX idx_classify_parent ON stock_industry_classify(parent_code, src);
CREATE INDEX idx_classify_level ON stock_industry_classify(level, src);

-- 行业成分表索引
CREATE INDEX idx_member_ts_code ON stock_industry_member(ts_code);
CREATE INDEX idx_member_l3_code ON stock_industry_member(l3_code);
CREATE INDEX idx_member_dates ON stock_industry_member(in_date, out_date);
CREATE INDEX idx_member_current ON stock_industry_member(l3_code, out_date) 
    WHERE out_date IS NULL;  -- 部分索引，只索引当前成分
```

## 同步工具设计建议

### 同步顺序
1. **先同步行业分类** (API 181)
   - 确保行业分类数据完整
2. **再同步行业成分** (API 335)
   - 依赖行业分类数据

### 同步策略
- **行业分类**：全量覆盖（删除指定版本，再插入）
- **行业成分**：增量更新（基于 in_date/out_date 判断）


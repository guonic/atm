# Tushare 数据同步工具实现总结

## 已实现的功能

### 1. 数据模型 (Models)

#### `StockIndustryClassify` - 申万行业分类模型
- **位置**: `python/atm/models/stock.py`
- **字段**:
  - `index_code`: 指数代码
  - `industry_name`: 行业名称
  - `parent_code`: 父级代码
  - `level`: 行业层级 (L1/L2/L3)
  - `industry_code`: 行业代码
  - `is_pub`: 是否发布了指数
  - `src`: 行业分类版本 (SW2014/SW2021)

#### `StockIndustryMember` - 申万行业成分模型
- **位置**: `python/atm/models/stock.py`
- **字段**:
  - `ts_code`: 股票代码
  - `l1_code`, `l1_name`: 一级行业代码和名称
  - `l2_code`, `l2_name`: 二级行业代码和名称
  - `l3_code`, `l3_name`: 三级行业代码和名称
  - `stock_name`: 股票名称
  - `in_date`: 纳入日期
  - `out_date`: 剔除日期
  - `is_new`: 是否最新

### 2. 数据仓库 (Repositories)

#### `StockIndustryClassifyRepo` - 行业分类仓库
- **位置**: `python/atm/repo/stock_repo.py`
- **功能**:
  - 自动创建表结构
  - 批量保存行业分类数据
  - 按层级查询 (`get_by_level`)
  - 按父级查询 (`get_by_parent`)
  - 按版本删除 (`delete_by_src`)

#### `StockIndustryMemberRepo` - 行业成分仓库
- **位置**: `python/atm/repo/stock_repo.py`
- **功能**:
  - 自动创建表结构（包含索引优化）
  - 批量保存行业成分数据
  - 按股票代码查询 (`get_by_ts_code`)
  - 按行业代码查询 (`get_by_l3_code`)
  - 支持当前成分和历史成分查询

### 3. 同步服务 (Services)

#### `IndustryClassifySyncService` - 行业分类同步服务
- **位置**: `python/tools/dataingestor/service/industry_classify_sync_service.py`
- **功能**:
  - 同步所有层级 (L1/L2/L3)
  - 全量覆盖同步（删除指定版本，再插入新数据）
  - 支持多版本 (SW2014, SW2021)
  - 任务锁机制
  - 批量保存优化

#### `IndustryMemberSyncService` - 行业成分同步服务
- **位置**: `python/tools/dataingestor/service/industry_member_sync_service.py`
- **功能**:
  - 同步所有股票的行业成分
  - 按行业代码同步（L1/L2/L3）
  - 支持历史数据同步
  - 任务锁机制
  - 批量保存优化

### 4. 同步工具脚本

#### `sync_industry_classify.py` - 行业分类同步脚本
- **位置**: `python/tools/dataingestor/sync_industry_classify.py`
- **用法**:
  ```bash
  python python/tools/dataingestor/sync_industry_classify.py --src SW2021
  ```

#### `sync_industry_member.py` - 行业成分同步脚本
- **位置**: `python/tools/dataingestor/sync_industry_member.py`
- **用法**:
  ```bash
  # 同步所有股票
  python python/tools/dataingestor/sync_industry_member.py
  
  # 按行业代码同步
  python python/tools/dataingestor/sync_industry_member.py --l3-code 850531.SI
  ```

### 5. 命令框架集成

#### 已注册的命令
- `atm sync industry-classify` - 同步申万行业分类
- `atm sync industry-member` - 同步申万行业成分

#### 使用示例
```bash
# 同步申万行业分类 (SW2021)
atm sync industry-classify --src SW2021

# 同步申万行业成分（所有股票）
atm sync industry-member

# 按 L3 行业代码同步
atm sync industry-member --l3-code 850531.SI

# 同步历史数据
atm sync industry-member --is-new N
```

## 数据库表结构

### `stock_industry_classify` 表

```sql
CREATE TABLE stock_industry_classify (
    index_code VARCHAR(20) NOT NULL,
    industry_name VARCHAR(100) NOT NULL,
    parent_code VARCHAR(20) NOT NULL,
    level VARCHAR(10) NOT NULL,
    industry_code VARCHAR(20),
    is_pub VARCHAR(1),
    src VARCHAR(20) NOT NULL,
    update_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (index_code, src)
);

-- 索引
CREATE INDEX idx_stock_industry_classify_parent ON stock_industry_classify(parent_code, src);
CREATE INDEX idx_stock_industry_classify_level ON stock_industry_classify(level, src);
CREATE INDEX idx_stock_industry_classify_src ON stock_industry_classify(src);
```

### `stock_industry_member` 表

```sql
CREATE TABLE stock_industry_member (
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
    is_new VARCHAR(1) DEFAULT 'Y',
    update_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (ts_code, l3_code, in_date)
);

-- 索引
CREATE INDEX idx_stock_industry_member_ts_code ON stock_industry_member(ts_code);
CREATE INDEX idx_stock_industry_member_l1_code ON stock_industry_member(l1_code);
CREATE INDEX idx_stock_industry_member_l2_code ON stock_industry_member(l2_code);
CREATE INDEX idx_stock_industry_member_l3_code ON stock_industry_member(l3_code);
CREATE INDEX idx_stock_industry_member_in_date ON stock_industry_member(in_date);
CREATE INDEX idx_stock_industry_member_out_date ON stock_industry_member(out_date);
CREATE INDEX idx_stock_industry_member_is_new ON stock_industry_member(is_new);
```

## 同步策略

### 行业分类同步策略
- **模式**: 全量覆盖
- **流程**: 
  1. 删除指定版本 (src) 的所有数据
  2. 按层级 (L1/L2/L3) 依次同步
  3. 批量保存到数据库

### 行业成分同步策略
- **模式**: 增量追加
- **流程**:
  1. 按股票或行业代码获取数据
  2. 保留历史记录（in_date, out_date）
  3. 批量保存到数据库

## 使用流程

### 首次同步

```bash
# 1. 先同步行业分类（建立行业层级结构）
atm sync industry-classify --src SW2021

# 2. 再同步行业成分（建立股票-行业关联）
atm sync industry-member
```

### 日常同步

```bash
# 行业分类（低频更新，按需同步）
atm sync industry-classify --src SW2021

# 行业成分（中频更新，定期同步）
atm sync industry-member
```

## 数据查询示例

### 查询行业分类

```python
from atm.repo import StockIndustryClassifyRepo
from atm.config import load_config

config = load_config()
repo = StockIndustryClassifyRepo(config.database)

# 查询一级行业
l1_industries = repo.get_by_level("L1", src="SW2021")

# 查询某个行业的子行业
sub_industries = repo.get_by_parent("801050.SI", src="SW2021")
```

### 查询行业成分

```python
from atm.repo import StockIndustryMemberRepo
from atm.config import load_config

config = load_config()
repo = StockIndustryMemberRepo(config.database)

# 查询某个股票的当前行业分类
members = repo.get_by_ts_code("000001.SZ", current_only=True)

# 查询某个行业的所有成分股（当前）
members = repo.get_by_l3_code("850531.SI", current_only=True)
```

## 参考文档

- [Tushare API 181: 申万行业分类](https://tushare.pro/document/2?doc_id=181)
- [Tushare API 335: 申万行业成分构成](https://tushare.pro/document/2?doc_id=335)
- [存储结构设计文档](./TUSHARE_SYNC_DESIGN.md)


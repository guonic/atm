# 数据同步任务文档

## 概述

数据同步任务用于将外部数据源（如 Tushare）的数据同步到本地数据库。支持断点续传、任务锁和两种同步模式。

## 股票基础信息同步任务

### 功能说明

同步股票基础信息（股票代码、名称、上市日期等）从 Tushare 到数据库。

**特点**：
- ✅ 覆盖更新模式（upsert）- 自动更新已存在的记录
- ✅ 断点续传支持 - 任务中断后可继续
- ✅ 任务锁机制 - 防止并发执行
- ✅ 批量处理 - 高效的数据写入

### 使用方法

#### 方法 1: 使用控制器脚本（推荐）

```bash
# 同步所有已上市股票
./scripts/controller.sh sync stock-basic

# 同步指定交易所
./scripts/controller.sh sync stock-basic --exchange SSE

# 同步所有状态（包括已退市）
./scripts/controller.sh sync stock-basic --list-status ""

# 自定义批次大小
./scripts/controller.sh sync stock-basic --batch-size 200

# 不使用断点续传（重新开始）
./scripts/controller.sh sync stock-basic --no-resume

# 使用数据库存储状态（多实例部署）
./scripts/controller.sh sync stock-basic --use-db-state
```

#### 方法 2: 直接使用 Python 脚本

```bash
# 设置环境变量
export TUSHARE_TOKEN=your_tushare_token

# 运行同步任务
python3 python/tools/dataingestor/sync_stock_basic.py

# 带参数运行
python3 python/tools/dataingestor/sync_stock_basic.py \
    --exchange SSE \
    --list-status L \
    --batch-size 100 \
    --task-name stock_basic_sse_listed
```

#### 方法 3: 使用 Shell 脚本

```bash
# 设置环境变量
export TUSHARE_TOKEN=your_tushare_token

# 运行同步脚本
./scripts/sync_stock_basic.sh

# 带参数运行
./scripts/sync_stock_basic.sh \
    --exchange SSE \
    --list-status L \
    --batch-size 100
```

### 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--exchange` | 交易所代码（SSE/SZSE/BSE，空表示全部） | `""` (全部) |
| `--list-status` | 上市状态（L=已上市, D=已退市, P=暂停, 空=全部） | `L` |
| `--batch-size` | 批次大小（每次保存的记录数） | `100` |
| `--task-name` | 任务名称（用于状态跟踪） | 自动生成 |
| `--resume` | 启用断点续传 | `True` |
| `--no-resume` | 禁用断点续传，重新开始 | - |
| `--state-dir` | 文件状态存储目录 | `storage/state` |
| `--use-db-state` | 使用数据库存储状态 | `False` |
| `--config` | 配置文件路径 | `config/data_ingestor.yaml` |
| `--tushare-token` | Tushare Pro API Token | 环境变量 `TUSHARE_TOKEN` |

### 环境变量

```bash
# Tushare Token（必需）
export TUSHARE_TOKEN=your_tushare_token

# 数据库配置（可选，有默认值）
export DB_HOST=localhost
export DB_PORT=5432
export DB_USER=quant
export DB_PASSWORD=quant123
export DB_NAME=quant_db
export DB_SCHEMA=quant
```

### 任务状态管理

#### 查看任务状态

任务状态保存在 `storage/state/` 目录（文件模式）或数据库 `quant.ingestion_state` 表（数据库模式）。

```python
from atm.config import DatabaseConfig
from atm.repo import FileStateRepo
from tools.dataingestor import StockIngestorService

db_config = DatabaseConfig(...)
state_repo = FileStateRepo(state_dir="storage/state")

with StockIngestorService(db_config=db_config, state_repo=state_repo, ...) as ingestor:
    # 查看特定任务状态
    state = ingestor.get_task_state("stock_basic_sync_all_L")
    if state:
        print(f"Status: {state.status}")
        print(f"Progress: {state.total_saved}/{state.total_fetched}")
        print(f"Last Key: {state.last_processed_key}")
    
    # 列出所有任务状态
    states = ingestor.list_task_states()
    for s in states:
        print(f"{s.task_name}: {s.status}")
```

#### 重置任务状态

如果需要重新开始某个任务：

```python
ingestor.reset_task_state("stock_basic_sync_all_L")
```

### 同步模式

#### Upsert 模式（覆盖更新）- 默认

- **行为**: 如果记录存在则更新，不存在则插入
- **适用场景**: 数据更新、全量同步
- **特点**: 幂等性，可重复执行

```bash
# 默认就是 upsert 模式
./scripts/controller.sh sync stock-basic
```

#### Append 模式（追加）

- **行为**: 只插入新记录，如果记录已存在则失败
- **适用场景**: 顺序写入、增量同步
- **特点**: 保证数据顺序，不允许重复

```python
# 在代码中使用 append 模式
stats = ingestor.ingest_stock_basic(
    exchange="",
    list_status="L",
    mode="append",  # 追加模式
)
```

### 任务锁机制

#### 防止并发执行

相同任务名不能同时运行。如果尝试启动已运行的任务，会抛出 `TaskLockError`。

```bash
# 第一个实例
./scripts/controller.sh sync stock-basic --task-name test_task &
# 进程 ID: 12345

# 第二个实例（会失败）
./scripts/controller.sh sync stock-basic --task-name test_task
# Error: Task test_task is already running
```

#### 锁类型

1. **文件锁**（默认）
   - 锁文件保存在 `storage/locks/` 目录
   - 适合单机部署

2. **数据库锁**
   - 锁信息保存在数据库
   - 适合多实例部署
   - 支持锁超时（30分钟）

### 断点续传

#### 工作原理

1. 任务开始时检查是否有保存的状态
2. 如果有，从 `last_processed_key` 继续处理
3. 每处理 10 个批次保存一次检查点
4. 任务完成时清除检查点

#### 使用示例

```bash
# 第一次运行（处理到一半中断）
./scripts/controller.sh sync stock-basic
# ... 处理了 5000 条记录后中断

# 第二次运行（从第 5000 条继续）
./scripts/controller.sh sync stock-basic --resume
# 从上次停止的地方继续处理
```

### 常见问题

#### 1. 任务一直显示 "running" 状态

**原因**: 任务异常退出，锁未释放

**解决**:
```python
# 手动释放锁
state = ingestor.get_task_state("task_name")
if state and state.status == "running":
    ingestor.reset_task_state("task_name")
```

#### 2. 数据库连接失败

**检查**:
- 数据库服务是否运行: `./scripts/controller.sh storage status`
- 连接参数是否正确
- 防火墙设置

#### 3. Tushare API 限流

**建议**:
- 减小批次大小: `--batch-size 50`
- 添加延迟处理
- 使用 Tushare Pro 高级权限

### 日志

任务运行日志会输出到标准输出，包含：
- 任务开始/结束时间
- 处理进度
- 错误信息
- 最终统计

示例输出：
```
2024-12-17 20:30:00 - INFO - Stock Basic Information Synchronization Task
2024-12-17 20:30:00 - INFO - Task Name: stock_basic_sync_all_L
2024-12-17 20:30:00 - INFO - Exchange: All
2024-12-17 20:30:00 - INFO - List Status: L
2024-12-17 20:30:00 - INFO - Mode: upsert (覆盖更新)
2024-12-17 20:30:05 - INFO - Saved batch of 100 stocks
...
2024-12-17 20:35:00 - INFO - Synchronization Completed
2024-12-17 20:35:00 - INFO - Total Fetched: 5000
2024-12-17 20:35:00 - INFO - Total Saved: 5000
2024-12-17 20:35:00 - INFO - Total Errors: 0
```

### 定时任务

#### 使用 cron

```bash
# 编辑 crontab
crontab -e

# 每天凌晨 2 点同步股票基础信息
0 2 * * * cd /path/to/atm && ./scripts/controller.sh sync stock-basic >> /var/log/atm/sync_stock_basic.log 2>&1
```

#### 使用 systemd timer

创建 `/etc/systemd/system/atm-sync-stock-basic.service`:
```ini
[Unit]
Description=ATM Stock Basic Sync Task
After=network.target

[Service]
Type=oneshot
User=quant
WorkingDirectory=/path/to/atm
Environment="TUSHARE_TOKEN=your_token"
ExecStart=/path/to/atm/scripts/controller.sh sync stock-basic
```

创建 `/etc/systemd/system/atm-sync-stock-basic.timer`:
```ini
[Unit]
Description=Run ATM Stock Basic Sync Daily
Requires=atm-sync-stock-basic.service

[Timer]
OnCalendar=daily
OnCalendar=02:00

[Install]
WantedBy=timers.target
```

启用定时器:
```bash
sudo systemctl enable atm-sync-stock-basic.timer
sudo systemctl start atm-sync-stock-basic.timer
```

### 性能优化

1. **批次大小**: 根据内存和网络情况调整（建议 100-500）
2. **并发控制**: 避免同时运行多个相同任务
3. **数据库连接**: 使用连接池
4. **状态存储**: 多实例部署使用数据库状态存储

### 相关文档

- [数据摄取工具文档](python/tools/dataingestor/README.md)
- [状态管理文档](docs/SYNC_TASKS.md)
- [数据库配置文档](docs/PROJECT_STRUCTURE.md)


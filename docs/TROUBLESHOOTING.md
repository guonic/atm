# 故障排查指南

## 数据同步错误分析

### 错误统计说明

同步任务会统计以下指标：
- **Total Fetched**: 从数据源获取的总记录数
- **Total Saved**: 成功保存到数据库的记录数
- **Total Errors**: 处理失败的记录数

### 常见错误原因

#### 1. 数据转换失败

**原因**：
- 缺少必需字段（如 `list_date`、`ts_code`、`symbol`、`name`）
- 日期格式无效（如 `00000000`）
- 字段值为空或格式不正确

**解决方案**：
- 检查 Tushare API 返回的数据格式
- 查看日志中的详细错误信息
- 确认数据源的数据质量

**日志示例**：
```
DEBUG - Skipping record with invalid list_date: 000001.SZ
DEBUG - Skipping record with missing required fields: ts_code=, symbol=, name=
```

#### 2. 数据库保存失败

**原因**：
- 数据库连接问题
- 违反唯一约束（主键冲突）
- 数据类型不匹配
- 外键约束失败

**解决方案**：
- 检查数据库连接：`./scripts/atm storage status`
- 查看数据库日志
- 检查表结构和约束
- 确认使用正确的同步模式（upsert/append）

**日志示例**：
```
ERROR - Error saving batch: duplicate key value violates unique constraint
ERROR - Error saving batch: foreign key constraint failed
```

#### 3. 数据验证失败

**原因**：
- Pydantic 模型验证失败
- 字段类型不匹配
- 必需字段缺失

**解决方案**：
- 检查模型定义和实际数据格式
- 查看 Pydantic 验证错误详情
- 调整数据转换逻辑

### 错误分析工具

#### 查看详细错误日志

启用 DEBUG 日志级别查看详细信息：

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

或在脚本中添加：
```bash
python python/tools/dataingestor/sync_stock_basic.py \
    --tushare-token your_token \
    2>&1 | grep -E "(ERROR|WARNING|DEBUG)" | head -50
```

#### 检查数据库中的数据

```sql
-- 连接到数据库
psql -h localhost -p 5432 -U quant -d quant_db

-- 查看已保存的记录数
SELECT COUNT(*) FROM quant.stock_basic;

-- 查看最近保存的记录
SELECT ts_code, symbol, full_name, list_date 
FROM quant.stock_basic 
ORDER BY create_time DESC 
LIMIT 10;

-- 检查是否有重复记录
SELECT ts_code, COUNT(*) 
FROM quant.stock_basic 
GROUP BY ts_code 
HAVING COUNT(*) > 1;
```

#### 分析错误模式

```python
from atm.repo import FileStateRepo, IngestionState

# 查看任务状态
state_repo = FileStateRepo(state_dir="storage/state")
state = state_repo.get_state("stock_basic_sync_all_L")

if state:
    print(f"Total Fetched: {state.total_fetched}")
    print(f"Total Saved: {state.total_saved}")
    print(f"Total Errors: {state.total_errors}")
    print(f"Error Rate: {state.total_errors / state.total_fetched * 100:.2f}%")
    if state.error_message:
        print(f"Last Error: {state.error_message}")
```

### 错误处理改进

代码已改进错误处理：

1. **转换失败统计**：`_convert_to_stock_basic` 返回 `None` 时计入错误
2. **批量保存错误**：批量保存失败时，整个批次计入错误
3. **详细日志**：记录更多调试信息，便于排查

### 常见问题

#### Q: 为什么错误数接近 (Fetched - Saved)？

**A**: 这通常表示：
- 大部分错误是数据转换失败（缺少必需字段）
- 少量错误可能是数据库保存失败

#### Q: 如何减少错误？

**A**:
1. 检查数据源质量
2. 调整数据过滤条件（如只同步已上市股票）
3. 使用 `--list-status L` 过滤已退市股票
4. 检查数据库约束和索引

#### Q: 错误会影响已保存的数据吗？

**A**: 不会。错误只影响失败的记录，已成功保存的数据不受影响。

### 性能优化建议

1. **调整批次大小**：根据错误率调整 `--batch-size`
2. **过滤数据**：使用 `--exchange` 和 `--list-status` 过滤
3. **检查网络**：确保 Tushare API 连接稳定
4. **数据库优化**：检查数据库性能和索引

### 获取帮助

如果错误持续存在：
1. 查看完整错误日志
2. 检查数据库状态
3. 验证 Tushare API 返回的数据格式
4. 参考 [SYNC_TASKS.md](SYNC_TASKS.md) 文档


# Eidos REST API

FastAPI 实现的 Eidos 回测归因系统 REST API。

## 功能

- 实验管理：获取实验列表和详情
- 账户流水：查询每日净值数据
- 交易记录：查询交易订单数据
- 性能指标：计算回测性能指标
- 交易统计：计算交易统计数据

## API 端点

### 实验管理

- `GET /api/v1/experiments` - 获取所有实验列表
- `GET /api/v1/experiments/{exp_id}` - 获取单个实验详情

### 账户流水

- `GET /api/v1/experiments/{exp_id}/ledger` - 获取账户流水
  - 查询参数：
    - `start_date` (可选): 开始日期
    - `end_date` (可选): 结束日期

### 交易记录

- `GET /api/v1/experiments/{exp_id}/trades` - 获取交易记录
  - 查询参数：
    - `symbol` (可选): 股票代码过滤
    - `start_date` (可选): 开始日期/时间
    - `end_date` (可选): 结束日期/时间

### 性能指标

- `GET /api/v1/experiments/{exp_id}/metrics` - 获取性能指标
  - 返回：总收益率、最大回撤、最终净值、交易天数、夏普比率、年化收益率

### 交易统计

- `GET /api/v1/experiments/{exp_id}/trade-stats` - 获取交易统计
  - 返回：总交易次数、买入/卖出次数、胜率、平均持仓天数

## 启动

### 方法 1: 使用启动脚本

```bash
./scripts/start_eidos_api.sh
```

### 方法 2: 使用 Python 模块运行

```bash
# 需要设置 PYTHONPATH 或从 python 目录运行
export PYTHONPATH=python:$PYTHONPATH
python -m nq.api.rest.eidos

# 或从 python 目录运行
cd python
python -m nq.api.rest.eidos
```

### 方法 3: 使用 uvicorn

```bash
uvicorn nq.api.rest.eidos.main:app --host 0.0.0.0 --port 8000 --reload
```

## 访问

- API 服务: http://localhost:8000
- API 文档: http://localhost:8000/docs (Swagger UI)
- ReDoc 文档: http://localhost:8000/redoc
- 健康检查: http://localhost:8000/health

## 配置

确保 `config/config.yaml` 中包含正确的数据库配置：

```yaml
database:
  host: localhost
  port: 5432
  database: quant_db
  user: quant
  password: your_password
```

## CORS

API 默认允许来自以下源的前端请求：
- http://localhost:3000
- http://127.0.0.1:3000

如需修改，请编辑 `python/nq/api/rest/eidos/main.py` 中的 CORS 配置。

## 依赖

- FastAPI
- Uvicorn
- SQLAlchemy
- Pydantic

安装依赖：

```bash
pip install fastapi uvicorn[standard]
```


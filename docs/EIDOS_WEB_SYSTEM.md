# Eidos 前后端分离归因系统

## 概述

Eidos 归因系统采用前后端分离架构，提供现代化的 Web 界面用于展示和分析回测结果。

## 架构

```
┌─────────────┐         HTTP/REST API         ┌─────────────┐
│   Frontend  │ ────────────────────────────> │   Backend   │
│  (React)    │ <──────────────────────────── │  (FastAPI)  │
│             │                                │             │
│ Port: 3000  │                                │ Port: 8000  │
└─────────────┘                                └─────────────┘
                                                       │
                                                       ▼
                                              ┌─────────────┐
                                              │ PostgreSQL  │
                                              │  (Eidos DB) │
                                              └─────────────┘
```

## 目录结构

### 前端 (`web/eidos/`)

```
web/eidos/
├── src/
│   ├── components/      # React 组件
│   │   ├── ExperimentSelector.tsx
│   │   ├── PerformancePanel.tsx
│   │   ├── TradeStatsPanel.tsx
│   │   └── NavChart.tsx
│   ├── pages/           # 页面组件
│   │   └── Dashboard.tsx
│   ├── services/        # API 客户端
│   │   └── api.ts
│   ├── types/           # TypeScript 类型
│   │   └── eidos.ts
│   ├── App.tsx
│   └── main.tsx
├── package.json
├── vite.config.ts
└── tailwind.config.js
```

### 后端 (`python/nq/api/rest/eidos/`)

```
python/nq/api/rest/eidos/
├── __init__.py
├── main.py              # FastAPI 应用入口
├── routes.py            # API 路由定义
├── handlers.py          # 请求处理函数
├── schemas.py           # Pydantic 请求/响应模型
├── dependencies.py      # 依赖注入
└── README.md
```

## 技术栈

### 前端
- **React 18** - UI 框架
- **TypeScript** - 类型安全
- **Tailwind CSS** - 样式框架
- **Vite** - 构建工具
- **React Router** - 路由管理
- **Axios** - HTTP 客户端
- **Recharts** - 图表库

### 后端
- **FastAPI** - Web 框架
- **Uvicorn** - ASGI 服务器
- **Pydantic** - 数据验证
- **SQLAlchemy** - 数据库 ORM
- **PostgreSQL** - 数据库

## 快速开始

### 1. 启动后端 API

```bash
# 方法 1: 使用启动脚本
./scripts/start_eidos_api.sh

# 方法 2: 使用启动脚本（推荐）
./scripts/start_eidos_api.sh

# 方法 3: 使用 Python 模块运行（需要设置 PYTHONPATH）
export PYTHONPATH=python:$PYTHONPATH
python -m nq.api.rest.eidos

# 方法 3: 使用 uvicorn
uvicorn nq.api.rest.eidos.main:app --host 0.0.0.0 --port 8000 --reload
```

后端将在 http://localhost:8000 启动。

### 2. 启动前端

```bash
# 进入前端目录
cd web/eidos

# 安装依赖（首次运行）
npm install

# 启动开发服务器
npm run dev

# 或使用启动脚本
../scripts/start_eidos_web.sh
```

前端将在 http://localhost:3000 启动。

### 3. 访问应用

- 前端界面: http://localhost:3000
- API 文档: http://localhost:8000/docs
- API 健康检查: http://localhost:8000/health

## API 端点

### 实验管理

- `GET /api/v1/experiments` - 获取所有实验
- `GET /api/v1/experiments/{exp_id}` - 获取实验详情

### 账户流水

- `GET /api/v1/experiments/{exp_id}/ledger` - 获取账户流水
  - 查询参数: `start_date`, `end_date`

### 交易记录

- `GET /api/v1/experiments/{exp_id}/trades` - 获取交易记录
  - 查询参数: `symbol`, `start_date`, `end_date`

### 性能指标

- `GET /api/v1/experiments/{exp_id}/metrics` - 获取性能指标
  - 返回: 总收益率、最大回撤、最终净值、交易天数、夏普比率、年化收益率

### 交易统计

- `GET /api/v1/experiments/{exp_id}/trade-stats` - 获取交易统计
  - 返回: 总交易次数、买入/卖出次数、胜率、平均持仓天数

## 开发

### 前端开发

```bash
cd web/eidos
npm run dev          # 启动开发服务器
npm run build        # 构建生产版本
npm run preview      # 预览生产构建
npm run lint         # 代码检查
```

### 后端开发

```bash
# 启动开发服务器（自动重载）
python python/examples/eidos_api_server.py --reload

# 或使用 uvicorn
uvicorn nq.api.rest.eidos.main:app --reload
```

## 配置

### 数据库配置

确保 `config/config.yaml` 中包含正确的数据库配置：

```yaml
database:
  host: localhost
  port: 5432
  database: quant_db
  user: quant
  password: your_password
```

### CORS 配置

后端默认允许来自以下源的前端请求：
- http://localhost:3000
- http://127.0.0.1:3000

如需修改，请编辑 `python/nq/api/rest/eidos/main.py`。

### API 代理配置

前端通过 Vite 代理连接到后端 API。配置在 `web/eidos/vite.config.ts`：

```typescript
server: {
  proxy: {
    '/api': {
      target: 'http://localhost:8000',
      changeOrigin: true,
    },
  },
}
```

## 部署

### 前端部署

```bash
cd web/eidos
npm run build
```

构建产物在 `dist/` 目录，可以部署到 Nginx、CDN 或静态托管服务。

### 后端部署

```bash
# 使用 uvicorn 生产模式
uvicorn nq.api.rest.eidos.main:app --host 0.0.0.0 --port 8000 --workers 4

# 或使用 Gunicorn + Uvicorn workers
gunicorn nq.api.rest.eidos.main:app -w 4 -k uvicorn.workers.UvicornWorker
```

## 功能特性

### 已实现

- ✅ 实验列表和选择
- ✅ 性能指标展示
- ✅ 交易统计展示
- ✅ 净值曲线图表
- ✅ REST API 完整实现
- ✅ CORS 支持
- ✅ API 文档（Swagger UI）

### 待实现

- ⏳ 排名轨迹分析
- ⏳ 交易明细表格
- ⏳ 模型输出可视化
- ⏳ 归因分析图表
- ⏳ 数据导出功能

## 故障排除

### 前端无法连接后端

1. 检查后端是否正在运行（http://localhost:8000/health）
2. 检查 Vite 代理配置
3. 检查 CORS 设置

### API 返回 404

1. 检查实验 ID 是否正确
2. 检查数据库中是否存在该实验
3. 查看后端日志

### 数据库连接错误

1. 检查 `config/config.yaml` 中的数据库配置
2. 确保 PostgreSQL 服务正在运行
3. 确保数据库和 schema 已创建

## 相关文档

- [Eidos 设计文档](design/eidos.md)
- [Eidos Dashboard 指南](EIDOS_DASHBOARD_GUIDE.md)
- [API README](python/nq/api/rest/eidos/README.md)


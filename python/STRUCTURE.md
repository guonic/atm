# Python 项目结构说明

## 目录组织

### `python/atm/` - 核心业务模块

核心业务逻辑和共享组件：

- **`models/`** - 数据模型
  - `base.py` - 基础模型类
  - 定义通用的数据模型和 Pydantic 模型

- **`repo/`** - 数据仓库层（核心）
  - `base.py` - 仓库基类接口
  - `database_repo.py` - 数据库仓库实现
  - 供所有服务使用的通用仓库实现

- **`service/`** - 服务层（核心）
  - `dataingestor_service.py` - 数据摄取服务
  - 核心业务服务，供其他模块调用

- **`config/`** - 配置管理
- **`data/`** - 数据处理模块
  - `source/` - 数据源抽象和实现（BaseSource, HttpSource 等）
- **`trading/`** - 交易模块
- **`analysis/`** - 分析模块
- **`api/`** - API 服务
- **`utils/`** - 工具函数

### `python/tools/` - 工具和独立应用

独立的工具和应用程序：

- **`dataingestor/`** - 数据摄取工具
  - `source/` - 数据源实现（工具专用）
    - `base.py` - 数据源基类
    - `http_source.py` - HTTP 数据源
  - `service/` - 服务实现（工具版本）
  - `repo/` - 仓库接口定义（引用 atm.repo）

## 模块职责

### Models (`atm/models/`)
- 定义数据模型和数据结构
- 使用 Pydantic 进行数据验证
- 通用模型基类

### Repo (`atm/repo/`)
- **核心仓库实现**，供所有服务使用
- 数据库访问抽象层
- 支持多种存储后端

### Service (`atm/service/`)
- **核心服务实现**，供其他模块调用
- 业务逻辑协调层
- 使用 repo 和 models

### Tools (`tools/dataingestor/`)
- **独立工具**，可单独运行
- 包含完整的数据摄取流程
- Source 层实现（工具专用）

## 使用方式

### 使用核心服务

```python
from atm.service import DataIngestorService
from atm.repo import DatabaseRepo
from atm.models import BaseModel
```

### 使用工具

```python
from atm.tools.dataingestor import DataIngestorService
from atm.data.source import HttpSource
```

## 设计原则

1. **核心模块** (`atm/`) - 可被其他模块依赖
2. **工具模块** (`tools/`) - 独立工具，不依赖核心业务逻辑
3. **分层清晰** - models, repo, service 各司其职
4. **可扩展** - 易于添加新的模型、仓库和服务



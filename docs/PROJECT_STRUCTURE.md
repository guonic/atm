# 项目目录结构规划

## 概述

本项目采用 Python、Golang、C++ 混合开发架构，支持量化交易系统的不同性能需求。

## 目录结构

```
atm/
├── README.md                    # 项目说明
├── LICENSE                       # 许可证
├── .gitignore                   # Git 忽略文件
├── .github/                      # GitHub 配置
│   ├── workflows/               # CI/CD 工作流
│   ├── ISSUE_TEMPLATE/          # Issue 模板
│   └── PULL_REQUEST_TEMPLATE.md # PR 模板
│
├── config/                       # 配置文件目录
│   ├── config.yaml              # 主配置文件
│   ├── config.dev.yaml          # 开发环境配置
│   ├── config.prod.yaml         # 生产环境配置
│   └── config.test.yaml         # 测试环境配置
│
├── scripts/                      # 脚本工具目录
│   ├── controller.sh            # 主控制器脚本
│   ├── storage_controller.sh    # 存储控制器
│   ├── build.sh                 # 构建脚本
│   └── deploy.sh                 # 部署脚本
│
├── docker/                       # Docker 配置
│   ├── storage-docker-compose.yaml
│   ├── Dockerfile.python
│   ├── Dockerfile.go
│   └── docker-compose.yaml
│
├── storage/                      # 数据存储
│   └── database/                # 数据库相关
│       ├── data/                # 数据库数据目录
│       ├── *.sql                # SQL 脚本
│       └── postgresql.conf       # PostgreSQL 配置
│
├── pkg/                          # 共享包/库（跨语言接口）
│   ├── proto/                    # Protocol Buffers 定义
│   │   ├── trading.proto
│   │   ├── data.proto
│   │   └── common.proto
│   ├── api/                      # API 接口定义
│   │   ├── rest/                 # REST API 定义
│   │   └── grpc/                 # gRPC 接口定义
│   └── schema/                   # 数据模型定义
│       ├── json/                 # JSON Schema
│       └── sql/                  # SQL Schema
│
├── python/                       # Python 代码
│   ├── pyproject.toml            # Python 项目配置
│   ├── requirements.txt          # Python 依赖
│   ├── requirements-dev.txt      # 开发依赖
│   ├── setup.py                  # 安装脚本
│   │
│   ├── atm/                      # Python 主包
│   │   ├── __init__.py
│   │   ├── config/               # 配置管理
│   │   │   ├── __init__.py
│   │   │   └── loader.py
│   │   │
│   │   ├── data/                 # 数据处理
│   │   │   ├── __init__.py
│   │   │   ├── dataingestor/   # 数据摄取
│   │   │   ├── processor/       # 数据处理
│   │   │   └── storage/         # 数据存储
│   │   │
│   │   ├── trading/              # 交易模块
│   │   │   ├── __init__.py
│   │   │   ├── strategy/        # 策略模块
│   │   │   │   ├── __init__.py
│   │   │   │   ├── base.py     # 策略基类
│   │   │   │   └── index/      # 指数策略
│   │   │   ├── execution/       # 执行模块
│   │   │   └── risk/            # 风险管理
│   │   │
│   │   ├── analysis/            # 分析模块
│   │   │   ├── __init__.py
│   │   │   ├── indicators/     # 技术指标
│   │   │   └── backtest/       # 回测
│   │   │
│   │   ├── api/                 # API 服务
│   │   │   ├── __init__.py
│   │   │   ├── rest/            # REST API
│   │   │   └── grpc/            # gRPC 服务
│   │   │
│   │   └── utils/               # 工具函数
│   │       ├── __init__.py
│   │       ├── logger.py
│   │       └── helpers.py
│   │
│   ├── tests/                   # Python 测试
│   │   ├── __init__.py
│   │   ├── unit/                # 单元测试
│   │   ├── integration/         # 集成测试
│   │   └── fixtures/             # 测试数据
│   │
│   └── examples/                # Python 示例
│       ├── strategy_example.py
│       └── dataingestor_example.py
│
├── go/                           # Golang 代码
│   ├── go.mod                    # Go 模块定义
│   ├── go.sum                    # Go 依赖校验
│   ├── Makefile                  # Make 构建文件
│   │
│   ├── cmd/                      # 可执行程序入口
│   │   ├── trader/              # 交易服务
│   │   │   └── main.go
│   │   ├── dataingestor/         # 数据摄取服务
│   │   │   └── main.go
│   │   └── api/                 # API 服务
│   │       └── main.go
│   │
│   ├── internal/                 # 内部包（不对外暴露）
│   │   ├── config/               # 配置管理
│   │   ├── data/                # 数据处理
│   │   │   ├── dataingestor/   # 数据摄取
│   │   │   ├── processor/       # 数据处理
│   │   │   └── storage/         # 数据存储
│   │   ├── trading/             # 交易模块
│   │   │   ├── strategy/        # 策略
│   │   │   ├── execution/       # 执行
│   │   │   └── risk/            # 风险管理
│   │   ├── analysis/            # 分析模块
│   │   └── api/                 # API 服务
│   │
│   ├── pkg/                      # 公共包（可被外部引用）
│   │   ├── client/              # 客户端库
│   │   ├── models/              # 数据模型
│   │   └── utils/                # 工具函数
│   │
│   ├── api/                      # API 定义（生成代码）
│   │   ├── proto/               # 生成的 proto 代码
│   │   └── openapi/             # OpenAPI 定义
│   │
│   ├── tests/                    # Go 测试
│   │   ├── unit/                # 单元测试
│   │   ├── integration/         # 集成测试
│   │   └── testdata/            # 测试数据
│   │
│   └── examples/                # Go 示例
│       └── simple_trader.go
│
├── cpp/                          # C++ 代码
│   ├── CMakeLists.txt            # CMake 构建配置
│   ├── conanfile.txt             # Conan 依赖管理（可选）
│   │
│   ├── include/                  # 头文件
│   │   └── atm/                  # 命名空间
│   │       ├── core/             # 核心模块
│   │       │   ├── types.hpp
│   │       │   └── config.hpp
│   │       ├── data/             # 数据处理
│   │       │   ├── dataingestor.hpp
│   │       │   └── processor.hpp
│   │       ├── trading/           # 交易模块
│   │       │   ├── strategy.hpp
│   │       │   └── execution.hpp
│   │       └── analysis/         # 分析模块
│   │           └── indicators.hpp
│   │
│   ├── src/                      # 源文件
│   │   ├── core/                 # 核心实现
│   │   ├── data/                 # 数据处理实现
│   │   ├── trading/              # 交易实现
│   │   └── analysis/             # 分析实现
│   │
│   ├── bindings/                 # 语言绑定
│   │   ├── python/               # Python 绑定（pybind11）
│   │   │   ├── CMakeLists.txt
│   │   │   └── bindings.cpp
│   │   └── go/                   # Go 绑定（cgo）
│   │       └── cgo.go
│   │
│   ├── tests/                    # C++ 测试
│   │   ├── unit/                 # 单元测试（Google Test）
│   │   ├── integration/          # 集成测试
│   │   └── benchmarks/           # 性能测试
│   │
│   ├── examples/                 # C++ 示例
│   │   └── simple_example.cpp
│   │
│   └── third_party/              # 第三方库
│       └── CMakeLists.txt
│
├── trading/                      # 交易相关（业务逻辑）
│   ├── strategy/                 # 策略定义
│   │   ├── index/                # 指数策略
│   │   │   ├── doc/             # 文档
│   │   │   └── README.md
│   │   └── readme.md
│   │
│   └── config/                   # 交易配置
│       └── strategies.yaml
│
├── docs/                         # 文档目录
│   ├── PROJECT_STRUCTURE.md      # 项目结构说明（本文件）
│   ├── ARCHITECTURE.md           # 架构设计文档
│   ├── API.md                    # API 文档
│   ├── DEVELOPMENT.md            # 开发指南
│   └── DEPLOYMENT.md             # 部署指南
│
├── tools/                        # 开发工具
│   ├── codegen/                 # 代码生成工具
│   │   ├── proto_gen.sh         # Proto 代码生成
│   │   └── api_gen.sh            # API 代码生成
│   ├── scripts/                 # 工具脚本
│   └── migrations/              # 数据库迁移脚本
│
├── build/                        # 构建输出目录（gitignore）
│   ├── python/                   # Python 构建产物
│   ├── go/                       # Go 构建产物
│   └── cpp/                      # C++ 构建产物
│
├── dist/                         # 发布包目录（gitignore）
│   ├── python/                   # Python 包
│   ├── go/                       # Go 二进制
│   └── cpp/                      # C++ 库
│
└── .env.example                  # 环境变量示例
```

## 语言职责划分

### Python
- **用途**: 快速开发、数据分析、策略研究、API 服务
- **主要模块**:
  - 数据采集和处理
  - 策略研究和回测
  - REST API 服务
  - 数据分析工具

### Golang
- **用途**: 高性能服务、并发处理、微服务
- **主要模块**:
  - 实时数据采集服务
  - 交易执行服务
  - gRPC 服务
  - 高性能数据处理

### C++
- **用途**: 超高性能计算、核心算法、底层优化
- **主要模块**:
  - 高频交易核心算法
  - 高性能指标计算
  - 底层数据处理
  - 提供 Python/Go 绑定

## 跨语言通信

### 1. Protocol Buffers (gRPC)
- 定义在 `pkg/proto/` 目录
- 用于服务间通信
- 自动生成各语言代码

### 2. REST API
- Python 提供 REST API
- Go 服务调用 REST API
- 定义在 `pkg/api/rest/`

### 3. 共享库
- C++ 编译为共享库（.so/.dylib/.dll）
- Python 通过 pybind11 调用
- Go 通过 cgo 调用

### 4. 消息队列（可选）
- Redis/RabbitMQ 用于异步通信
- 定义在 `pkg/mq/`

## 构建系统

### Python
```bash
cd python
pip install -e .
```

### Golang
```bash
cd go
go build ./cmd/trader
```

### C++
```bash
cd cpp
mkdir build && cd build
cmake ..
make
```

## 测试策略

- **单元测试**: 各语言独立测试
- **集成测试**: 跨语言服务测试
- **性能测试**: C++ 核心算法基准测试

## 开发工作流

1. **配置管理**: 统一在 `config/` 目录
2. **代码生成**: 使用 `tools/codegen/` 生成跨语言代码
3. **构建**: 使用 `scripts/build.sh` 统一构建
4. **测试**: 各语言独立测试 + 集成测试
5. **部署**: 使用 Docker 容器化部署

## 注意事项

1. **依赖管理**: 
   - Python: `requirements.txt` / `pyproject.toml`
   - Go: `go.mod` / `go.sum`
   - C++: `CMakeLists.txt` / `conanfile.txt`

2. **版本控制**:
   - 所有生成代码不提交到 Git
   - 只提交源文件（.proto, .yaml 等）

3. **文档同步**:
   - API 变更需同步更新所有语言文档
   - 使用 OpenAPI/Swagger 统一管理

4. **性能考虑**:
   - 高频操作使用 C++
   - 并发处理使用 Go
   - 快速迭代使用 Python


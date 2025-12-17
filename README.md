# ATM - Automated Trading Machine

量化交易系统，支持 Python、Golang、C++ 混合开发。

## 项目结构

详细的项目结构说明请参考 [docs/PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md)

## 快速开始

### 前置要求

- Python 3.9+
- Go 1.21+
- C++17 编译器 (GCC/Clang)
- CMake 3.15+
- Docker & Docker Compose
- PostgreSQL/TimescaleDB

### 环境设置

1. **克隆项目**
```bash
git clone <repository-url>
cd atm
```

2. **设置开发环境**
```bash
make dev-setup
```

3. **安装 Git Hooks（推荐）**
```bash
make install-hooks
```
这将安装提交信息格式检查和代码质量检查的 Git hooks。

3. **启动数据库**
```bash
make storage-start
# 或者
./scripts/controller.sh storage start
```

4. **构建项目**
```bash
make build
# 或者构建特定语言
make build-python
make build-go
make build-cpp
```

### 常用命令

查看所有可用命令：
```bash
make help
```

主要命令：
- `make build` - 构建所有语言
- `make test` - 运行所有测试
- `make clean` - 清理构建产物
- `make storage-start` - 启动数据库
- `make storage-stop` - 停止数据库
- `make storage-status` - 查看数据库状态
- `make format` - 格式化代码
- `make lint` - 代码检查

### 开发指南

#### Python 开发
```bash
cd python
pip install -e ".[dev]"
pytest
```

#### Go 开发
```bash
cd go
go mod download
go test ./...
go build ./cmd/trader
```

#### C++ 开发
```bash
cd cpp
mkdir build && cd build
cmake ..
make
```

## 项目架构

- **Python**: 快速开发、数据分析、策略研究
- **Golang**: 高性能服务、并发处理、微服务
- **C++**: 超高性能计算、核心算法、底层优化

## 文档

- [项目结构](docs/PROJECT_STRUCTURE.md) - 详细的目录结构说明
- [提交规范](docs/COMMIT_CONVENTION.md) - Git 提交信息规范
- [分支命名规范](docs/BRANCH_NAMING.md) - Git 分支命名规范
- [Git Hooks](docs/GIT_HOOKS.md) - Git hooks 使用指南
- [贡献指南](docs/CONTRIBUTING.md) - 如何为项目做贡献
- [架构设计](docs/ARCHITECTURE.md) (待完善)
- [API 文档](docs/API.md) (待完善)
- [开发指南](docs/DEVELOPMENT.md) (待完善)

## 许可证

MIT License

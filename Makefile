.PHONY: help build clean test proto docker-up docker-down install dev-setup

# 默认目标
.DEFAULT_GOAL := help

# 变量定义
PYTHON_DIR := python
GO_DIR := go
CPP_DIR := cpp
SCRIPTS_DIR := scripts
BUILD_DIR := build
DIST_DIR := dist

# 颜色定义
GREEN := \033[0;32m
YELLOW := \033[0;33m
NC := \033[0m # No Color

## help: 显示帮助信息
help:
	@echo "$(GREEN)ATM - Automated Trading Machine$(NC)"
	@echo ""
	@echo "可用命令:"
	@echo "  $(YELLOW)make build$(NC)          - 构建所有语言（Python、Go、C++）"
	@echo "  $(YELLOW)make build-python$(NC)   - 构建 Python 包"
	@echo "  $(YELLOW)make build-go$(NC)       - 构建 Go 程序"
	@echo "  $(YELLOW)make build-cpp$(NC)      - 构建 C++ 库"
	@echo "  $(YELLOW)make proto$(NC)           - 生成 Protocol Buffers 代码"
	@echo "  $(YELLOW)make test$(NC)           - 运行所有测试"
	@echo "  $(YELLOW)make test-python$(NC)    - 运行 Python 测试"
	@echo "  $(YELLOW)make test-go$(NC)        - 运行 Go 测试"
	@echo "  $(YELLOW)make test-cpp$(NC)       - 运行 C++ 测试"
	@echo "  $(YELLOW)make clean$(NC)          - 清理所有构建产物"
	@echo "  $(YELLOW)make install$(NC)        - 安装所有依赖"
	@echo "  $(YELLOW)make dev-setup$(NC)      - 设置开发环境"
	@echo "  $(YELLOW)make docker-up$(NC)       - 启动 Docker 服务"
	@echo "  $(YELLOW)make docker-down$(NC)    - 停止 Docker 服务"
	@echo "  $(YELLOW)make storage-start$(NC)  - 启动数据库服务"
	@echo "  $(YELLOW)make storage-stop$(NC)   - 停止数据库服务"
	@echo "  $(YELLOW)make storage-status$(NC) - 查看数据库状态"
	@echo "  $(YELLOW)make format$(NC)         - 格式化代码"
	@echo "  $(YELLOW)make lint$(NC)           - 代码检查"
	@echo "  $(YELLOW)make install-hooks$(NC)  - 安装 Git hooks
  $(YELLOW)make venv$(NC)           - 创建 Python 虚拟环境"

## build: 构建所有语言
build: proto build-python build-go build-cpp
	@echo "$(GREEN)✓ 所有构建完成$(NC)"

## build-python: 构建 Python 包
build-python:
	@echo "$(YELLOW)构建 Python 包...$(NC)"
	@cd $(PYTHON_DIR) && pip install -e . || echo "警告: Python 构建失败"

## build-go: 构建 Go 程序
build-go:
	@echo "$(YELLOW)构建 Go 程序...$(NC)"
	@mkdir -p $(BUILD_DIR)/go
	@cd $(GO_DIR) && go build -o ../$(BUILD_DIR)/go/trader ./cmd/trader || echo "警告: Go 构建失败"
	@cd $(GO_DIR) && go build -o ../$(BUILD_DIR)/go/collector ./cmd/collector 2>/dev/null || true
	@cd $(GO_DIR) && go build -o ../$(BUILD_DIR)/go/api ./cmd/api 2>/dev/null || true

## build-cpp: 构建 C++ 库
build-cpp:
	@echo "$(YELLOW)构建 C++ 库...$(NC)"
	@mkdir -p $(CPP_DIR)/build
	@cd $(CPP_DIR)/build && cmake .. && cmake --build . -j$$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4) || echo "警告: C++ 构建失败"

## proto: 生成 Protocol Buffers 代码
proto:
	@echo "$(YELLOW)生成 Protocol Buffers 代码...$(NC)"
	@$(SCRIPTS_DIR)/build.sh proto || $(SCRIPTS_DIR)/../tools/codegen/proto_gen.sh || echo "警告: Proto 代码生成失败"

## test: 运行所有测试
test: test-python test-go test-cpp
	@echo "$(GREEN)✓ 所有测试完成$(NC)"

## test-python: 运行 Python 测试
test-python:
	@echo "$(YELLOW)运行 Python 测试...$(NC)"
	@cd $(PYTHON_DIR) && python -m pytest tests/ -v || echo "警告: Python 测试失败"

## test-go: 运行 Go 测试
test-go:
	@echo "$(YELLOW)运行 Go 测试...$(NC)"
	@cd $(GO_DIR) && go test ./... -v || echo "警告: Go 测试失败"

## test-cpp: 运行 C++ 测试
test-cpp:
	@echo "$(YELLOW)运行 C++ 测试...$(NC)"
	@cd $(CPP_DIR)/build && ctest -V || echo "警告: C++ 测试失败（需要先构建）"

## clean: 清理所有构建产物
clean:
	@echo "$(YELLOW)清理构建产物...$(NC)"
	@rm -rf $(BUILD_DIR) $(DIST_DIR)
	@find $(PYTHON_DIR) -type d -name "__pycache__" -exec rm -r {} + 2>/dev/null || true
	@find $(PYTHON_DIR) -type d -name "*.egg-info" -exec rm -r {} + 2>/dev/null || true
	@find $(PYTHON_DIR) -type f -name "*.pyc" -delete 2>/dev/null || true
	@cd $(GO_DIR) && go clean -cache 2>/dev/null || true
	@rm -rf $(CPP_DIR)/build
	@echo "$(GREEN)✓ 清理完成$(NC)"

## install: 安装所有依赖
install: install-python install-go
	@echo "$(GREEN)✓ 依赖安装完成$(NC)"

## venv: 创建 Python 虚拟环境
venv:
	@echo "$(YELLOW)创建 Python 虚拟环境...$(NC)"
	@python3 -m venv .venv || echo "警告: 虚拟环境创建失败"
	@echo "$(GREEN)✓ 虚拟环境已创建$(NC)"
	@echo "激活虚拟环境: source .venv/bin/activate"

## install-python: 安装 Python 依赖
install-python:
	@echo "$(YELLOW)安装 Python 依赖...$(NC)"
	@if [ -d ".venv" ]; then \
		. .venv/bin/activate && cd $(PYTHON_DIR) && pip install -e ".[dev]" || echo "警告: Python 依赖安装失败"; \
	else \
		echo "警告: 虚拟环境不存在，请先运行 'make venv'"; \
	fi

## install-go: 安装 Go 依赖
install-go:
	@echo "$(YELLOW)安装 Go 依赖...$(NC)"
	@cd $(GO_DIR) && go mod download || echo "警告: Go 依赖安装失败"

## dev-setup: 设置开发环境
dev-setup: install proto
	@echo "$(GREEN)✓ 开发环境设置完成$(NC)"

## docker-up: 启动 Docker 服务
docker-up:
	@echo "$(YELLOW)启动 Docker 服务...$(NC)"
	@docker-compose -f docker/storage-docker-compose.yaml up -d || echo "警告: Docker 启动失败"

## docker-down: 停止 Docker 服务
docker-down:
	@echo "$(YELLOW)停止 Docker 服务...$(NC)"
	@docker-compose -f docker/storage-docker-compose.yaml down || echo "警告: Docker 停止失败"

## storage-start: 启动数据库服务
storage-start:
	@$(SCRIPTS_DIR)/atm storage start

## storage-stop: 停止数据库服务
storage-stop:
	@$(SCRIPTS_DIR)/atm storage stop

## storage-status: 查看数据库状态
storage-status:
	@$(SCRIPTS_DIR)/atm storage status

## storage-restart: 重启数据库服务
storage-restart:
	@$(SCRIPTS_DIR)/atm storage restart

## storage-login: 登录数据库
storage-login:
	@$(SCRIPTS_DIR)/atm storage login

## storage-psql: 显示 psql 连接命令
storage-psql:
	@$(SCRIPTS_DIR)/atm storage psql

## format: 格式化代码
format: format-python format-go format-cpp
	@echo "$(GREEN)✓ 代码格式化完成$(NC)"

## format-python: 格式化 Python 代码
format-python:
	@echo "$(YELLOW)格式化 Python 代码...$(NC)"
	@cd $(PYTHON_DIR) && black . 2>/dev/null || echo "警告: 需要安装 black"

## format-go: 格式化 Go 代码
format-go:
	@echo "$(YELLOW)格式化 Go 代码...$(NC)"
	@cd $(GO_DIR) && gofmt -w . || echo "警告: Go 格式化失败"

## format-cpp: 格式化 C++ 代码
format-cpp:
	@echo "$(YELLOW)格式化 C++ 代码...$(NC)"
	@which clang-format > /dev/null && find $(CPP_DIR)/src $(CPP_DIR)/include -name "*.cpp" -o -name "*.hpp" | xargs clang-format -i || echo "警告: 需要安装 clang-format"

## lint: 代码检查
lint: lint-python lint-go lint-cpp
	@echo "$(GREEN)✓ 代码检查完成$(NC)"

## lint-python: Python 代码检查
lint-python:
	@echo "$(YELLOW)检查 Python 代码...$(NC)"
	@cd $(PYTHON_DIR) && flake8 . 2>/dev/null || echo "警告: 需要安装 flake8"

## lint-go: Go 代码检查
lint-go:
	@echo "$(YELLOW)检查 Go 代码...$(NC)"
	@cd $(GO_DIR) && golangci-lint run 2>/dev/null || echo "警告: 需要安装 golangci-lint"

## lint-cpp: C++ 代码检查
lint-cpp:
	@echo "$(YELLOW)检查 C++ 代码...$(NC)"
	@which cppcheck > /dev/null && cppcheck $(CPP_DIR)/src $(CPP_DIR)/include 2>/dev/null || echo "警告: 需要安装 cppcheck"

## install-hooks: 安装 Git hooks
install-hooks:
	@echo "$(YELLOW)安装 Git hooks...$(NC)"
	@$(SCRIPTS_DIR)/install-git-hooks.sh


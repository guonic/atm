# Commit 提交规范

本文档定义了 ATM 项目的 Git 提交信息规范，遵循 [Conventional Commits](https://www.conventionalcommits.org/) 标准。

## 提交信息格式

```
<type>(<scope>): <subject>

<body>

<footer>
```

### 格式说明

- **type**（必需）: 提交类型
- **scope**（可选）: 影响范围
- **subject**（必需）: 简短描述，不超过 50 个字符
- **body**（可选）: 详细描述，每行不超过 72 个字符
- **footer**（可选）: 关闭的 Issue 或破坏性变更说明

## 提交类型 (Type)

### 主要类型

- **feat**: 新功能
  ```
  feat(trading): add order execution engine
  ```

- **fix**: 修复 Bug
  ```
  fix(data): correct timestamp parsing error
  ```

- **docs**: 文档变更
  ```
  docs: update API documentation
  ```

- **style**: 代码格式变更（不影响代码运行）
  ```
  style(python): format code with black
  ```

- **refactor**: 代码重构（既不是新功能也不是修复 Bug）
  ```
  refactor(go): simplify error handling
  ```

- **perf**: 性能优化
  ```
  perf(cpp): optimize matrix multiplication
  ```

- **test**: 测试相关变更
  ```
  test(python): add unit tests for data collector
  ```

- **build**: 构建系统或外部依赖变更
  ```
  build: update CMakeLists.txt for C++17
  ```

- **ci**: CI/CD 配置变更
  ```
  ci: add GitHub Actions workflow
  ```

- **chore**: 其他变更（不修改源代码或测试）
  ```
  chore: update dependencies
  ```

- **revert**: 回滚之前的提交
  ```
  revert: revert "feat: add new feature"
  ```

## 影响范围 (Scope)

Scope 用于说明提交影响的范围，可以是：

### 按语言分类
- `python` - Python 代码
- `go` - Golang 代码
- `cpp` - C++ 代码

### 按模块分类
- `trading` - 交易模块
- `data` - 数据处理模块
- `api` - API 服务
- `config` - 配置管理
- `storage` - 存储模块
- `analysis` - 分析模块

### 按功能分类
- `strategy` - 策略相关
- `execution` - 执行相关
- `risk` - 风险管理
- `collector` - 数据采集
- `processor` - 数据处理

### 其他
- `docker` - Docker 配置
- `scripts` - 脚本工具
- `docs` - 文档
- `proto` - Protocol Buffers

## 提交信息示例

### 简单提交（只有 type 和 subject）

```
feat: add data collector module
```

```
fix: resolve memory leak in C++ code
```

### 带 scope 的提交

```
feat(trading): implement order execution engine

Add a new order execution engine that supports market and limit orders.
The engine includes order validation, risk checks, and execution tracking.
```

```
fix(data): correct timestamp parsing for different timezones

Previously, timestamps were incorrectly parsed when the data source
used a different timezone. This fix ensures proper timezone conversion.
```

### 多行提交（带 body）

```
feat(api): add REST API for order management

- Add POST /api/v1/orders endpoint for placing orders
- Add GET /api/v1/orders endpoint for querying orders
- Add DELETE /api/v1/orders/:id endpoint for canceling orders
- Include request validation and error handling

Closes #123
```

### 破坏性变更（BREAKING CHANGE）

```
feat(api)!: change order response format

BREAKING CHANGE: The order response format has been changed from
a flat structure to a nested structure. The 'price' field is now
under 'order.price' instead of at the root level.

Migration guide:
- Update clients to access price via order.price
- Update all API consumers before deploying this version
```

### 修复 Issue

```
fix(storage): resolve database connection timeout

The database connection was timing out after 30 seconds of inactivity.
This fix implements connection pooling and automatic reconnection.

Fixes #456
```

### 多范围提交

```
feat(python,go): add shared configuration loader

Add a unified configuration loader that works across Python and Go.
This allows both languages to use the same configuration format
and ensures consistency.
```

## 提交信息最佳实践

### ✅ 好的提交信息

```
feat(trading): add stop-loss order support
fix(data): handle missing data gracefully
docs: update installation guide
refactor(go): extract common error handling
perf(cpp): optimize vector operations
test(python): add integration tests for API
```

### ❌ 不好的提交信息

```
update code                    # 太模糊，没有类型
fix bug                        # 没有说明修复了什么
WIP                           # 不完整
fixes                         # 不完整
feat: add stuff               # subject 太模糊
feat: add a lot of new features and fix some bugs  # 太长，应该分开提交
```

## 提交频率

- **频繁提交**: 每个逻辑变更都应该单独提交
- **原子性**: 每个提交应该是一个完整的、可工作的变更
- **避免大提交**: 如果一次变更涉及多个功能，应该拆分成多个提交

## 提交前检查清单

在提交前，请确保：

- [ ] 代码已通过所有测试
- [ ] 代码已格式化（`make format`）
- [ ] 代码已通过 lint 检查（`make lint`）
- [ ] 提交信息遵循规范
- [ ] 没有提交临时文件或调试代码
- [ ] 没有提交敏感信息（密码、密钥等）

## 使用 Commitizen（可选）

项目可以使用 [Commitizen](https://github.com/commitizen/cz-cli) 来帮助生成规范的提交信息：

```bash
# 安装 Commitizen
npm install -g commitizen
npm install -g cz-conventional-changelog

# 使用
git cz
```

## 自动化检查

可以使用 Git hooks 来检查提交信息格式：

### 使用 commitlint

```bash
# 安装 commitlint
npm install --save-dev @commitlint/cli @commitlint/config-conventional

# 创建 commitlint.config.js
echo "module.exports = {extends: ['@commitlint/config-conventional']}" > commitlint.config.js

# 安装 husky
npm install --save-dev husky
npx husky install
npx husky add .husky/commit-msg 'npx --no -- commitlint --edit $1'
```

## 变更日志生成

遵循此规范后，可以使用工具自动生成 CHANGELOG：

- [conventional-changelog](https://github.com/conventional-changelog/conventional-changelog)
- [standard-version](https://github.com/conventional-changelog/standard-version)

## 相关链接

- [Conventional Commits 规范](https://www.conventionalcommits.org/)
- [Angular 提交规范](https://github.com/angular/angular/blob/main/CONTRIBUTING.md#commit)
- [项目贡献指南](CONTRIBUTING.md)


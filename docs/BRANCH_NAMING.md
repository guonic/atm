# 分支命名规范

本文档定义了 ATM 项目的 Git 分支命名规范。

## 分支类型

### 主分支

- **`main`** / **`master`** - 主分支，用于生产环境
  - 只接受来自 `develop` 或 `release/*` 分支的合并
  - 所有提交必须经过代码审查
  - 必须通过所有测试和检查
  - 受保护分支，不能直接推送

- **`develop`** - 开发分支，用于集成所有功能
  - 从 `main` 分支创建
  - 接受来自 `feature/*` 和 `fix/*` 分支的合并
  - 持续集成和测试

### 功能分支

- **`feature/<name>`** - 新功能开发
  - 从 `develop` 分支创建
  - 命名规则：`feature/` + 简短描述（小写，用连字符分隔）
  - 示例：
    - `feature/order-execution`
    - `feature/data-ingestor`
    - `feature/api-authentication`

### 修复分支

- **`fix/<name>`** - Bug 修复
  - 从 `develop` 分支创建（如果是生产环境 bug，从 `main` 创建）
  - 命名规则：`fix/` + 简短描述（小写，用连字符分隔）
  - 示例：
    - `fix/timestamp-parsing`
    - `fix/memory-leak`
    - `fix/api-timeout`

### 发布分支

- **`release/<version>`** - 发布准备
  - 从 `develop` 分支创建
  - 命名规则：`release/` + 版本号（遵循语义化版本）
  - 示例：
    - `release/1.0.0`
    - `release/1.2.0-beta`
    - `release/2.0.0-rc.1`

### 热修复分支

- **`hotfix/<name>`** - 紧急修复（生产环境）
  - 从 `main` 分支创建
  - 修复后合并回 `main` 和 `develop`
  - 命名规则：`hotfix/` + 简短描述
  - 示例：
    - `hotfix/security-patch`
    - `hotfix/critical-bug`

### 其他分支

- **`chore/<name>`** - 杂项任务（文档、配置等）
  - 从 `develop` 分支创建
  - 示例：`chore/update-docs`, `chore/refactor-config`

- **`docs/<name>`** - 文档相关
  - 从 `develop` 分支创建
  - 示例：`docs/api-documentation`, `docs/architecture`

- **`test/<name>`** - 测试相关
  - 从 `develop` 分支创建
  - 示例：`test/integration-tests`, `test/performance`

## 命名规则

### 基本规则

1. **使用小写字母**
   - ✅ `feature/order-execution`
   - ❌ `feature/OrderExecution`
   - ❌ `feature/ORDER_EXECUTION`

2. **使用连字符分隔单词**
   - ✅ `feature/data-ingestor`
   - ❌ `feature/data_collector`
   - ❌ `feature/dataCollector`

3. **简短且描述性**
   - ✅ `fix/memory-leak`
   - ❌ `fix/fix-the-memory-leak-that-happens-when`
   - ❌ `fix/bug`

4. **避免特殊字符**
   - ✅ `feature/api-v2`
   - ❌ `feature/api@v2`
   - ❌ `feature/api#2`

5. **使用英文**
   - ✅ `feature/order-execution`
   - ❌ `feature/订单执行`

### 命名示例

#### 功能分支
```
feature/trading-strategy
feature/data-ingestor-api
feature/risk-management
feature/python-api-client
feature/go-trading-service
feature/cpp-optimization
```

#### 修复分支
```
fix/timestamp-parsing
fix/memory-leak-cpp
fix/api-authentication
fix/database-connection
fix/go-concurrency-issue
```

#### 发布分支
```
release/1.0.0
release/1.1.0
release/2.0.0-beta
release/2.0.0-rc.1
```

#### 热修复分支
```
hotfix/security-patch
hotfix/critical-bug
hotfix/data-loss-fix
```

## 分支工作流

### 功能开发流程

```
main → develop → feature/new-feature
                      ↓
                 (开发完成)
                      ↓
                 develop (合并)
                      ↓
                 (测试通过)
                      ↓
                 release/1.0.0
                      ↓
                 main (发布)
```

### Bug 修复流程

```
develop → fix/bug-description
              ↓
         (修复完成)
              ↓
         develop (合并)
```

### 热修复流程

```
main → hotfix/critical-issue
         ↓
    (修复完成)
         ↓
    main (合并)
         ↓
    develop (合并)
```

## 分支保护规则

### main 分支

- ✅ 必须通过代码审查
- ✅ 必须通过所有测试
- ✅ 必须通过 CI/CD 检查
- ✅ 提交信息必须符合规范
- ❌ 禁止直接推送
- ❌ 禁止强制推送

### develop 分支

- ✅ 建议代码审查
- ✅ 必须通过基本测试
- ✅ 提交信息建议符合规范
- ⚠️ 允许直接推送（开发阶段）

## 分支清理

### 合并后删除

功能分支、修复分支在合并后应该删除：

```bash
# 合并后删除本地分支
git branch -d feature/order-execution

# 删除远程分支
git push origin --delete feature/order-execution
```

### 定期清理

定期清理已合并的分支：

```bash
# 查看已合并的分支
git branch --merged

# 删除已合并的本地分支（除了 main 和 develop）
git branch --merged | grep -v "\*\|main\|develop" | xargs -n 1 git branch -d
```

## 分支命名检查

项目提供了 Git hook 来自动检查分支命名：

- 创建分支时检查命名是否符合规范
- 推送时检查分支名称
- 合并到 main 时强制检查

## 相关文档

- [提交规范](COMMIT_CONVENTION.md)
- [Git Hooks](GIT_HOOKS.md)
- [贡献指南](CONTRIBUTING.md)


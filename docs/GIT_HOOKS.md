# Git Hooks 使用指南

本项目使用 Git hooks 来自动检查提交信息格式和代码质量。

## 安装 Git Hooks

### 方法 1: 使用 Make 命令（推荐）

```bash
make install-hooks
```

### 方法 2: 使用安装脚本

```bash
./scripts/install-git-hooks.sh
```

### 方法 3: 手动安装

```bash
# 复制 hooks 到 .git/hooks 目录
cp .githooks/commit-msg .git/hooks/commit-msg
cp .githooks/pre-commit .git/hooks/pre-commit
chmod +x .git/hooks/commit-msg .git/hooks/pre-commit
```

## 可用的 Hooks

### commit-msg Hook

**功能**: 检查提交信息格式是否符合 [Conventional Commits](https://www.conventionalcommits.org/) 规范

**重要**: 此 hook **只对 main 分支的合并提交进行严格检查**，开发分支（feature/*, fix/*, develop 等）可以自由提交，不进行严格检查。

**检查项**:
- 提交信息格式：`<type>(<scope>): <subject>`
- 提交类型是否在允许列表中
- Scope 是否在推荐列表中（警告，不阻止）
- Subject 长度检查（警告超过 72 字符）
- Body 行长度检查（警告超过 72 字符）
- BREAKING CHANGE 检查

**示例**:

✅ **通过**:
```bash
git commit -m "feat(trading): add order execution engine"
git commit -m "fix(data): correct timestamp parsing"
git commit -m "docs: update API documentation"
```

❌ **失败**:
```bash
git commit -m "update code"  # 格式不正确
git commit -m "feat: "       # subject 为空
git commit -m "unknown: test" # 未知类型
```

### pre-commit Hook

**功能**: 提交前检查代码质量和安全性

**注意**: 此 hook 对所有分支生效，用于防止提交敏感信息和格式问题。

### pre-push Hook

**功能**: 推送前检查分支命名规范和提交信息

**检查项**:
- 分支命名是否符合规范（见 [分支命名规范](BRANCH_NAMING.md)）
- 推送到 main 分支时的提交信息格式检查
- 推送到 main 分支时的警告提示

**示例**:

✅ **通过**:
```bash
git push origin feature/order-execution  # 分支名符合规范
```

❌ **失败**:
```bash
git push origin my-branch  # 分支名不符合规范
# ❌ 错误: 分支名称不符合规范
```

**检查项**:
- 敏感文件检查（密码、密钥、.env 文件等）
- 大文件检查（>10MB，警告）
- Python 代码格式检查（如果安装了 black）
- Go 代码格式检查（如果安装了 gofmt）

**示例**:

如果尝试提交包含敏感信息的文件：
```bash
git add .env
git commit -m "feat: add config"
# ❌ 错误: 检测到敏感文件: .env
```

## 使用 Commitlint（可选）

如果您安装了 Node.js 和 commitlint，可以使用更强大的检查：

### 安装 Commitlint

```bash
npm install --save-dev @commitlint/cli @commitlint/config-conventional
```

### 配置

项目已包含 `.commitlintrc.json` 配置文件。

### 集成到 Git Hook

安装 hooks 时会自动检测 commitlint 并集成：

```bash
make install-hooks
```

## 跳过 Hooks（不推荐）

在某些特殊情况下，您可能需要跳过 hooks：

### 跳过 commit-msg hook

```bash
git commit -m "message" --no-verify
```

### 跳过 pre-commit hook

```bash
git commit --no-verify
```

**注意**: 只有在紧急情况下才应该跳过 hooks，并且需要确保提交符合规范。

## 故障排除

### Hook 不执行

1. 检查 hook 文件是否有执行权限：
   ```bash
   ls -l .git/hooks/commit-msg
   ```

2. 如果没有执行权限，添加权限：
   ```bash
   chmod +x .git/hooks/commit-msg
   ```

3. 重新安装 hooks：
   ```bash
   make install-hooks
   ```

### Hook 执行失败

如果 hook 执行失败，检查：

1. **commit-msg hook**: 检查提交信息格式是否正确
2. **pre-commit hook**: 检查是否有敏感文件或格式问题

查看详细错误信息，根据提示修复问题。

### 更新 Hooks

如果 hooks 有更新，重新安装：

```bash
make install-hooks
```

## 自定义 Hooks

如果需要自定义 hooks，可以修改 `.githooks/` 目录中的文件，然后重新安装：

```bash
# 修改 .githooks/commit-msg 或 .githooks/pre-commit
# 然后重新安装
make install-hooks
```

## 分支检查策略

### 开发分支（宽松检查）

以下分支允许自由提交，不进行严格的提交信息格式检查：
- `develop`
- `feature/*`
- `fix/*`
- `hotfix/*`
- `release/*`
- `chore/*`
- `docs/*`
- `test/*`

### 主分支（严格检查）

以下分支的合并提交必须符合规范：
- `main` / `master`

当合并到 main 分支时，所有提交信息都会被严格检查。

## 相关文档

- [提交规范](COMMIT_CONVENTION.md)
- [分支命名规范](BRANCH_NAMING.md)
- [贡献指南](CONTRIBUTING.md)


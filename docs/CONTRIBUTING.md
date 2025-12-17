# 贡献指南

感谢您对 ATM 项目的兴趣！我们欢迎所有形式的贡献。

## 如何贡献

### 报告 Bug

如果您发现了 bug，请：

1. 检查是否已经有相关的 Issue
2. 如果没有，请创建一个新的 Issue，使用 Bug 报告模板
3. 提供尽可能详细的信息，包括：
   - 复现步骤
   - 预期行为
   - 实际行为
   - 环境信息（操作系统、版本等）
   - 错误日志或截图

### 提出新功能

如果您有功能建议：

1. 检查是否已经有相关的 Issue 或讨论
2. 如果没有，请创建一个新的 Issue，使用功能请求模板
3. 详细描述功能的使用场景和预期效果
4. 如果可能，提供实现思路或示例

### 提交代码

1. **Fork 本仓库**

2. **创建您的特性分支**
   ```bash
   git checkout -b feature/your-feature-name
   # 或
   git checkout -b fix/your-bug-fix
   ```

3. **进行开发**
   - 遵循项目的代码规范（见 `.cursorrules`）
   - 确保代码通过所有测试
   - 添加必要的测试用例
   - 更新相关文档

4. **提交您的更改**
   ```bash
   git add .
   git commit -m "feat(scope): your commit message"
   ```
   
   请遵循 [提交规范](COMMIT_CONVENTION.md)

5. **推送到分支**
   ```bash
   git push origin feature/your-feature-name
   ```

6. **开启一个 Pull Request**
   - 填写 PR 模板
   - 描述您的更改
   - 关联相关的 Issue（如果有）

## 代码规范

### 通用规范

- 遵循项目的 `.cursorrules` 文件中的规范
- 代码必须通过格式化和 lint 检查
- 所有公共 API 必须有文档
- 添加适当的注释，特别是复杂逻辑

### 语言特定规范

#### Python
- 遵循 PEP 8
- 使用类型提示
- 所有公共函数和类必须有 docstring
- 运行 `make format-python` 格式化代码
- 运行 `make lint-python` 检查代码

#### Golang
- 遵循 Go 官方代码规范
- 运行 `gofmt` 格式化代码
- 所有公开函数必须有注释
- 运行 `make format-go` 格式化代码
- 运行 `make lint-go` 检查代码

#### C++
- 遵循 Google C++ Style Guide
- 使用 C++17 标准
- 优先使用智能指针
- 运行 `make format-cpp` 格式化代码
- 运行 `make lint-cpp` 检查代码

## 测试要求

- 所有新功能必须包含测试
- 测试覆盖率不应降低
- 运行 `make test` 确保所有测试通过

## 提交前检查清单

在提交 PR 前，请确保：

- [ ] 代码已通过所有测试（`make test`）
- [ ] 代码已格式化（`make format`）
- [ ] 代码已通过 lint 检查（`make lint`）
- [ ] 提交信息遵循规范（见 [COMMIT_CONVENTION.md](COMMIT_CONVENTION.md)）
- [ ] 没有提交临时文件或调试代码
- [ ] 没有提交敏感信息（密码、密钥等）
- [ ] 已更新相关文档
- [ ] 已添加必要的测试用例

## Pull Request 流程

1. **创建 PR**
   - 使用清晰的标题和描述
   - 关联相关的 Issue
   - 填写 PR 模板

2. **代码审查**
   - 等待维护者审查
   - 根据反馈进行修改
   - 保持 PR 更新（rebase 或 merge）

3. **合并**
   - 维护者会在审查通过后合并
   - PR 会自动关闭关联的 Issue

## 开发环境设置

1. **克隆仓库**
   ```bash
   git clone <repository-url>
   cd atm
   ```

2. **设置开发环境**
   ```bash
   make dev-setup
   ```

3. **启动数据库**
   ```bash
   make storage-start
   ```

4. **运行测试**
   ```bash
   make test
   ```

## 获取帮助

如果您在贡献过程中遇到问题：

- 查看 [项目结构文档](PROJECT_STRUCTURE.md)
- 查看 [提交规范](COMMIT_CONVENTION.md)
- 创建 Issue 询问
- 联系维护者

## 行为准则

- 尊重所有贡献者
- 接受建设性的批评
- 专注于对项目最有利的事情
- 对其他社区成员表示同理心

感谢您的贡献！


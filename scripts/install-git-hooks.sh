#!/bin/bash

# install-git-hooks.sh - 安装 Git hooks

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
HOOKS_DIR="$PROJECT_ROOT/.githooks"
GIT_HOOKS_DIR="$PROJECT_ROOT/.git/hooks"

# 检查是否在 Git 仓库中
if [ ! -d "$PROJECT_ROOT/.git" ]; then
    echo "错误: 当前目录不是 Git 仓库"
    exit 1
fi

echo "安装 Git hooks..."

# 创建 .git/hooks 目录（如果不存在）
mkdir -p "$GIT_HOOKS_DIR"

# 安装 commit-msg hook
if [ -f "$HOOKS_DIR/commit-msg" ]; then
    cp "$HOOKS_DIR/commit-msg" "$GIT_HOOKS_DIR/commit-msg"
    chmod +x "$GIT_HOOKS_DIR/commit-msg"
    echo "✓ 已安装 commit-msg hook"
else
    echo "⚠️  警告: 未找到 commit-msg hook"
fi

# 安装 pre-commit hook
if [ -f "$HOOKS_DIR/pre-commit" ]; then
    cp "$HOOKS_DIR/pre-commit" "$GIT_HOOKS_DIR/pre-commit"
    chmod +x "$GIT_HOOKS_DIR/pre-commit"
    echo "✓ 已安装 pre-commit hook"
else
    echo "⚠️  警告: 未找到 pre-commit hook"
fi

# 安装 pre-push hook
if [ -f "$HOOKS_DIR/pre-push" ]; then
    cp "$HOOKS_DIR/pre-push" "$GIT_HOOKS_DIR/pre-push"
    chmod +x "$GIT_HOOKS_DIR/pre-push"
    echo "✓ 已安装 pre-push hook"
else
    echo "⚠️  警告: 未找到 pre-push hook"
fi

# 如果使用 commitlint，尝试安装 commitlint hook
if command -v commitlint &> /dev/null && [ -f "$PROJECT_ROOT/.commitlintrc.json" ]; then
    echo "检测到 commitlint，创建 commitlint hook..."
    
    cat > "$GIT_HOOKS_DIR/commit-msg-commitlint" << 'EOF'
#!/bin/bash
npx --no -- commitlint --edit "$1"
EOF
    
    chmod +x "$GIT_HOOKS_DIR/commit-msg-commitlint"
    
    # 将 commitlint 集成到 commit-msg hook 中
    if [ -f "$GIT_HOOKS_DIR/commit-msg" ]; then
        # 在现有 commit-msg hook 末尾添加 commitlint 调用
        if ! grep -q "commitlint" "$GIT_HOOKS_DIR/commit-msg"; then
            echo "" >> "$GIT_HOOKS_DIR/commit-msg"
            echo "# 运行 commitlint（如果可用）" >> "$GIT_HOOKS_DIR/commit-msg"
            echo "if command -v commitlint &> /dev/null; then" >> "$GIT_HOOKS_DIR/commit-msg"
            echo "    npx --no -- commitlint --edit \"\$1\" || exit 1" >> "$GIT_HOOKS_DIR/commit-msg"
            echo "fi" >> "$GIT_HOOKS_DIR/commit-msg"
        fi
    fi
    
    echo "✓ 已集成 commitlint"
fi

echo ""
echo "Git hooks 安装完成！"
echo ""
echo "现在您的提交将自动检查："
echo "  - 提交信息格式（commit-msg hook，仅 main 分支合并时）"
echo "  - 敏感文件检查（pre-commit hook）"
echo "  - 代码格式检查（pre-commit hook）"
echo "  - 分支命名规范（pre-push hook）"
echo ""
echo "注意："
echo "  - 开发分支（feature/*, fix/* 等）可以自由提交，不进行严格检查"
echo "  - 合并到 main 分支时会进行严格的提交信息格式检查"
echo "  - 推送时会检查分支命名是否符合规范"


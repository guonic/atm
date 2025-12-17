#!/bin/bash

# activate_venv.sh - 激活 Python 虚拟环境的便捷脚本

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_DIR="$PROJECT_ROOT/venv"

if [ ! -d "$VENV_DIR" ]; then
    echo "错误: 虚拟环境不存在，请先创建："
    echo "  python3 -m venv venv"
    exit 1
fi

if [ ! -f "$VENV_DIR/bin/activate" ]; then
    echo "错误: 虚拟环境激活脚本不存在"
    exit 1
fi

echo "激活 Python 虚拟环境..."
source "$VENV_DIR/bin/activate"

echo "虚拟环境已激活！"
echo "Python 版本: $(python --version)"
echo "Python 路径: $(which python)"
echo ""
echo "提示: 使用 'deactivate' 退出虚拟环境"



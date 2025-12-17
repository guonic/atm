#!/bin/bash

# sync_trading_calendar.sh - Synchronize trading calendar data

set -e

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_SCRIPT="$PROJECT_ROOT/python/tools/dataingestor/sync_trading_calendar.py"

# 查找并激活 Python 虚拟环境
PYTHON_EXEC=""
if [ -f "$PROJECT_ROOT/.venv/bin/python" ]; then
    PYTHON_EXEC="$PROJECT_ROOT/.venv/bin/python"
elif [ -f "$PROJECT_ROOT/venv/bin/python" ]; then
    PYTHON_EXEC="$PROJECT_ROOT/venv/bin/python"
else
    echo "错误: Python 虚拟环境未找到。请先运行 'make venv'。"
    exit 1
fi

# 设置 Python 路径
export PYTHONPATH="$PROJECT_ROOT/python:$PYTHONPATH"

# 执行 Python 脚本
"$PYTHON_EXEC" "$PYTHON_SCRIPT" "$@"


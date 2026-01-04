#!/bin/bash
# Eidos REST API 服务器启动脚本

set -e

# 获取脚本所在目录的父目录（项目根目录）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_DIR="$PROJECT_ROOT/python"

# 激活虚拟环境（如果存在）
if [ -d "$PROJECT_ROOT/.venv" ]; then
    source "$PROJECT_ROOT/.venv/bin/activate"
fi

# 设置 PYTHONPATH 指向 python 目录
export PYTHONPATH="$PYTHON_DIR:$PYTHONPATH"

# 启动 API 服务器
echo "启动 Eidos REST API 服务器..."
echo "访问地址: http://localhost:8000"
echo "API 文档: http://localhost:8000/docs"
echo "健康检查: http://localhost:8000/health"

cd "$PROJECT_ROOT"
python -m nq.api.rest.eidos.main --host 0.0.0.0 --port 8000 --reload


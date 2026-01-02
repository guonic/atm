#!/bin/bash
# EDiOS Dashboard 启动脚本
# 
# 使用方法:
#   ./scripts/start_edios_dashboard.sh [port]
#
# 示例:
#   ./scripts/start_edios_dashboard.sh 8502

set -e

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# 切换到项目根目录
cd "$PROJECT_ROOT"

# 检查虚拟环境
if [ ! -d ".venv" ]; then
    echo "错误: 未找到虚拟环境 .venv"
    echo "请先创建虚拟环境: python -m venv .venv"
    exit 1
fi

# 激活虚拟环境
source .venv/bin/activate

# 获取端口号（默认为 8502）
PORT=${1:-8502}

# 设置 Python 路径
export PYTHONPATH="$PROJECT_ROOT/python:$PROJECT_ROOT:$PYTHONPATH"

# 启动 Streamlit
echo "启动 EDiOS Dashboard..."
echo "访问地址: http://localhost:$PORT"
echo "按 Ctrl+C 停止服务"
echo ""

streamlit run python/nq/analysis/edios/visualization.py --server.port "$PORT"


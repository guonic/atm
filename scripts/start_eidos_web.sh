#!/bin/bash
# Eidos 前端开发服务器启动脚本

set -e

# 获取脚本所在目录的父目录（项目根目录）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WEB_DIR="$PROJECT_ROOT/web/eidos"

# 检查 web/eidos 目录是否存在
if [ ! -d "$WEB_DIR" ]; then
    echo "错误: web/eidos 目录不存在"
    exit 1
fi

# 进入 web/eidos 目录
cd "$WEB_DIR"

# 检查 node_modules 是否存在，如果不存在则安装依赖
if [ ! -d "node_modules" ]; then
    echo "检测到未安装依赖，正在安装..."
    npm install
fi

# 启动开发服务器
echo "启动 Eidos 前端开发服务器..."
echo "访问地址: http://localhost:3000"
npm run dev


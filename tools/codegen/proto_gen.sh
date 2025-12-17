#!/bin/bash

# proto_gen.sh - 生成 Protocol Buffers 代码

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PROTO_DIR="$PROJECT_ROOT/pkg/proto"

if [ ! -d "$PROTO_DIR" ]; then
    echo "错误: 未找到 proto 目录: $PROTO_DIR"
    exit 1
fi

if ! command -v protoc &> /dev/null; then
    echo "错误: 未找到 protoc 命令，请安装 Protocol Buffers 编译器"
    exit 1
fi

echo "生成 Protocol Buffers 代码..."

# 生成 Python 代码
if [ -d "$PROJECT_ROOT/python" ]; then
    echo "  生成 Python proto 代码..."
    PYTHON_PROTO_DIR="$PROJECT_ROOT/python/atm/api/proto"
    mkdir -p "$PYTHON_PROTO_DIR"
    
    find "$PROTO_DIR" -name "*.proto" | while read proto_file; do
        protoc \
            --python_out="$PYTHON_PROTO_DIR" \
            --grpc_python_out="$PYTHON_PROTO_DIR" \
            --proto_path="$PROTO_DIR" \
            "$proto_file"
    done
fi

# 生成 Go 代码
if [ -d "$PROJECT_ROOT/go" ]; then
    echo "  生成 Go proto 代码..."
    GO_PROTO_DIR="$PROJECT_ROOT/go/api/proto"
    mkdir -p "$GO_PROTO_DIR"
    
    find "$PROTO_DIR" -name "*.proto" | while read proto_file; do
        protoc \
            --go_out="$GO_PROTO_DIR" \
            --go-grpc_out="$GO_PROTO_DIR" \
            --proto_path="$PROTO_DIR" \
            "$proto_file"
    done
fi

echo "Protocol Buffers 代码生成完成！"


#!/bin/bash

# build.sh - 统一构建脚本，支持 Python、Go、C++ 混合构建

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# 显示帮助信息
show_help() {
    cat << EOF
用法: build.sh [options] [targets...]

目标:
  all       构建所有语言（默认）
  python    构建 Python 包
  go        构建 Go 程序
  cpp       构建 C++ 库
  proto     生成 Protocol Buffers 代码
  clean     清理构建产物

选项:
  -h, --help     显示此帮助信息
  -r, --release  发布模式构建（优化）
  -d, --debug    调试模式构建

示例:
  build.sh                    # 构建所有
  build.sh python go          # 只构建 Python 和 Go
  build.sh --release cpp       # 发布模式构建 C++
  build.sh clean               # 清理所有构建产物

EOF
}

# 构建 Python
build_python() {
    echo "构建 Python 包..."
    cd "$PROJECT_ROOT/python"
    if [ -f "setup.py" ]; then
        pip install -e .
    elif [ -f "pyproject.toml" ]; then
        pip install -e .
    else
        echo "警告: 未找到 Python 项目配置文件"
    fi
}

# 构建 Go
build_go() {
    echo "构建 Go 程序..."
    cd "$PROJECT_ROOT/go"
    
    if [ ! -f "go.mod" ]; then
        echo "警告: 未找到 go.mod 文件"
        return
    fi
    
    # 构建所有 cmd 下的程序
    for cmd_dir in cmd/*/; do
        if [ -d "$cmd_dir" ]; then
            cmd_name=$(basename "$cmd_dir")
            echo "  构建 $cmd_name..."
            go build -o "$PROJECT_ROOT/build/go/$cmd_name" "./cmd/$cmd_name"
        fi
    done
}

# 构建 C++
build_cpp() {
    echo "构建 C++ 库..."
    cd "$PROJECT_ROOT/cpp"
    
    BUILD_TYPE=${BUILD_TYPE:-Release}
    BUILD_DIR="build"
    
    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR"
    
    cmake .. -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
    cmake --build . -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
}

# 生成 Proto 代码
build_proto() {
    echo "生成 Protocol Buffers 代码..."
    
    if ! command -v protoc &> /dev/null; then
        echo "错误: 未找到 protoc 命令，请安装 Protocol Buffers 编译器"
        return 1
    fi
    
    PROTO_DIR="$PROJECT_ROOT/pkg/proto"
    if [ ! -d "$PROTO_DIR" ]; then
        echo "警告: 未找到 proto 目录: $PROTO_DIR"
        return
    fi
    
    # 生成 Python 代码
    if [ -d "$PROJECT_ROOT/python" ]; then
        echo "  生成 Python proto 代码..."
        find "$PROTO_DIR" -name "*.proto" -exec protoc \
            --python_out="$PROJECT_ROOT/python" \
            --grpc_python_out="$PROJECT_ROOT/python" \
            --proto_path="$PROTO_DIR" {} \;
    fi
    
    # 生成 Go 代码
    if [ -d "$PROJECT_ROOT/go" ]; then
        echo "  生成 Go proto 代码..."
        find "$PROTO_DIR" -name "*.proto" -exec protoc \
            --go_out="$PROJECT_ROOT/go/api/proto" \
            --go-grpc_out="$PROJECT_ROOT/go/api/proto" \
            --proto_path="$PROTO_DIR" {} \;
    fi
}

# 清理构建产物
clean_build() {
    echo "清理构建产物..."
    rm -rf "$PROJECT_ROOT/build"
    rm -rf "$PROJECT_ROOT/dist"
    
    # Python
    find "$PROJECT_ROOT/python" -type d -name "__pycache__" -exec rm -r {} + 2>/dev/null || true
    find "$PROJECT_ROOT/python" -type d -name "*.egg-info" -exec rm -r {} + 2>/dev/null || true
    
    # Go
    find "$PROJECT_ROOT/go" -name "*.test" -delete 2>/dev/null || true
    
    # C++
    rm -rf "$PROJECT_ROOT/cpp/build"
    
    echo "清理完成"
}

# 主逻辑
main() {
    BUILD_TYPE="Release"
    TARGETS=()
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_help
                exit 0
                ;;
            -r|--release)
                BUILD_TYPE="Release"
                shift
                ;;
            -d|--debug)
                BUILD_TYPE="Debug"
                shift
                ;;
            clean)
                clean_build
                exit 0
                ;;
            all|python|go|cpp|proto)
                TARGETS+=("$1")
                shift
                ;;
            *)
                echo "错误: 未知选项 '$1'"
                show_help
                exit 1
                ;;
        esac
    done
    
    # 如果没有指定目标，默认构建所有
    if [ ${#TARGETS[@]} -eq 0 ]; then
        TARGETS=("all")
    fi
    
    # 创建构建目录
    mkdir -p "$PROJECT_ROOT/build/go"
    mkdir -p "$PROJECT_ROOT/dist"
    
    # 执行构建
    for target in "${TARGETS[@]}"; do
        case $target in
            all)
                build_proto
                build_python
                build_go
                build_cpp
                ;;
            python)
                build_python
                ;;
            go)
                build_go
                ;;
            cpp)
                build_cpp
                ;;
            proto)
                build_proto
                ;;
        esac
    done
    
    echo "构建完成！"
}

export BUILD_TYPE
main "$@"


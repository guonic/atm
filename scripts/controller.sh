#!/bin/bash

# controller.sh - 主入口脚本，支持子命令插件机制
# 类似于 kubectl 的命令结构

set -e

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# 显示帮助信息
show_help() {
    cat << EOF
用法: controller.sh <command> [options]

命令:
  storage   管理数据库服务（TimescaleDB）
            - start: 启动数据库服务
            - stop: 停止数据库服务
            - restart: 重启数据库服务
            - status: 查看服务状态
            - login: 登录数据库
            - psql: 打印 psql 连接命令

  help      显示此帮助信息

示例:
  controller.sh storage start
  controller.sh storage stop
  controller.sh storage login
  controller.sh storage psql
  controller.sh help

EOF
}

# 执行子命令
execute_subcommand() {
    local command=$1
    shift
    
    case "$command" in
        storage)
            if [ -f "$SCRIPT_DIR/storage_controller.sh" ]; then
                "$SCRIPT_DIR/storage_controller.sh" "$@"
            else
                echo "错误: 找不到 storage_controller.sh 脚本"
                exit 1
            fi
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            echo "错误: 未知命令 '$command'"
            echo ""
            show_help
            exit 1
            ;;
    esac
}

# 主逻辑
main() {
    if [ $# -eq 0 ]; then
        show_help
        exit 0
    fi
    
    execute_subcommand "$@"
}

main "$@"


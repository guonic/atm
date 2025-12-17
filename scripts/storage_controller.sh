#!/bin/bash

# storage_controller.sh - 数据库存储控制器
# 提供 TimescaleDB 的启动、停止、登录等操作

set -e

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
COMPOSE_FILE="$PROJECT_ROOT/docker/storage-docker-compose.yaml"

# 检查 docker-compose 文件是否存在
check_compose_file() {
    if [ ! -f "$COMPOSE_FILE" ]; then
        echo "错误: 找不到 storage-docker-compose.yaml 文件: $COMPOSE_FILE"
        exit 1
    fi
}

# 显示帮助信息
show_help() {
    cat << EOF
用法: storage_controller.sh <command>

命令:
  start     启动 TimescaleDB 服务
  stop      停止 TimescaleDB 服务
  restart   重启 TimescaleDB 服务
  status    查看服务状态
  logs      查看服务日志
  login     登录到数据库（psql）
  psql      打印 psql 连接命令
  help      显示此帮助信息

示例:
  storage_controller.sh start
  storage_controller.sh stop
  storage_controller.sh login
  storage_controller.sh psql

EOF
}

# 启动服务
start_service() {
    check_compose_file
    echo "正在启动 TimescaleDB 服务..."
    cd "$PROJECT_ROOT"
    docker-compose -f "$COMPOSE_FILE" up -d
    echo "TimescaleDB 服务已启动"
    echo "连接信息:"
    echo "  Host: localhost"
    echo "  Port: 5432"
    echo "  User: quant"
    echo "  Database: quant_db"
}

# 停止服务
stop_service() {
    check_compose_file
    echo "正在停止 TimescaleDB 服务..."
    cd "$PROJECT_ROOT"
    docker-compose -f "$COMPOSE_FILE" stop
    echo "TimescaleDB 服务已停止"
}

# 重启服务
restart_service() {
    check_compose_file
    echo "正在重启 TimescaleDB 服务..."
    cd "$PROJECT_ROOT"
    docker-compose -f "$COMPOSE_FILE" restart
    echo "TimescaleDB 服务已重启"
}

# 查看状态
show_status() {
    check_compose_file
    echo "TimescaleDB 服务状态:"
    cd "$PROJECT_ROOT"
    docker-compose -f "$COMPOSE_FILE" ps
}

# 查看日志
show_logs() {
    check_compose_file
    cd "$PROJECT_ROOT"
    docker-compose -f "$COMPOSE_FILE" logs -f timescaledb
}

# 登录数据库
login_database() {
    check_compose_file
    echo "正在连接到 TimescaleDB..."
    cd "$PROJECT_ROOT"
    
    # 检查容器是否运行
    if ! docker-compose -f "$COMPOSE_FILE" ps | grep -q "timescaledb-quant.*Up"; then
        echo "错误: TimescaleDB 容器未运行，请先执行 'start' 命令"
        exit 1
    fi
    
    # 使用 docker exec 登录到数据库
    docker exec -it timescaledb-quant psql -U quant -d quant_db
}

# 打印 psql 连接命令
print_psql_command() {
    check_compose_file
    echo "psql 连接命令："
    echo ""
    echo "方式1: 使用 Docker 容器内连接（推荐）"
    echo "  docker exec -it timescaledb-quant psql -U quant -d quant_db"
    echo ""
    echo "方式2: 使用本地 psql 客户端连接"
    echo "  psql -h localhost -p 5432 -U quant -d quant_db"
    echo ""
    echo "连接信息:"
    echo "  Host: localhost"
    echo "  Port: 5432"
    echo "  User: quant"
    echo "  Password: quant123"
    echo "  Database: quant_db"
    echo ""
    echo "提示: 使用方式2时，系统会提示输入密码，密码为: quant123"
}

# 主逻辑
main() {
    if [ $# -eq 0 ]; then
        show_help
        exit 0
    fi
    
    case "$1" in
        start)
            start_service
            ;;
        stop)
            stop_service
            ;;
        restart)
            restart_service
            ;;
        status)
            show_status
            ;;
        logs)
            show_logs
            ;;
        login)
            login_database
            ;;
        psql)
            print_psql_command
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            echo "错误: 未知命令 '$1'"
            echo ""
            show_help
            exit 1
            ;;
    esac
}

main "$@"


#!/bin/bash
#
# Data Synchronization Controller
#
# Manages data synchronization tasks (stock basic, kline, etc.)
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Show usage
show_usage() {
    cat <<EOF
Data Synchronization Controller

Usage: $0 <command> [options]

Commands:
  stock-basic          Synchronize stock basic information (upsert mode)
  trading-calendar    Synchronize trading calendar data (upsert mode)
  kline               Synchronize K-line data for all stocks (append mode)
  premarket           Synchronize stock premarket information (股本情况盘前数据)
  industry-classify   Synchronize Shenwan industry classification (申万行业分类)
  industry-member     Synchronize Shenwan industry member (申万行业成分)
  help                 Show this help message

Examples:
  # Sync all listed stocks
  $0 stock-basic

  # Sync SSE stocks only
  $0 stock-basic --exchange SSE

  # Sync trading calendar for all exchanges
  $0 trading-calendar

  # Sync trading calendar for SSE with date range
  $0 trading-calendar --exchange SSE --start-date 20240101 --end-date 20241231

  # Sync daily K-line data for all stocks
  $0 kline --type day

  # Sync weekly K-line data for SSE stocks only
  $0 kline --type week --exchange SSE

  # Sync premarket data for a specific date
  $0 premarket --trade-date 20241222

  # Sync premarket data for a date range
  $0 premarket --start-date 20240101 --end-date 20241231

  # Sync premarket data for a specific stock
  $0 premarket --ts-code 000001.SZ

  # Sync Shenwan industry classification (SW2021)
  $0 industry-classify --src SW2021

  # Sync Shenwan industry members for all stocks
  $0 industry-member

  # Sync industry members by L3 code
  $0 industry-member --l3-code 850531.SI

  # Sync with custom batch size
  $0 stock-basic --batch-size 200

  # Use database for state storage
  $0 stock-basic --use-db-state

For detailed options, use:
  $0 stock-basic --help
  $0 trading-calendar --help
  $0 kline --help
  $0 premarket --help
  $0 industry-classify --help
  $0 industry-member --help
EOF
}

# Main command dispatch
case "${1:-}" in
    stock-basic)
        shift
        exec "$SCRIPT_DIR/sync_stock_basic.sh" "$@"
        ;;
    trading-calendar)
        shift
        exec "$SCRIPT_DIR/sync_trading_calendar.sh" "$@"
        ;;
    kline)
        shift
        exec "$SCRIPT_DIR/sync_kline.sh" "$@"
        ;;
    premarket)
        shift
        exec "$SCRIPT_DIR/sync_premarket.sh" "$@"
        ;;
    industry-classify)
        shift
        exec "$SCRIPT_DIR/sync_industry_classify.sh" "$@"
        ;;
    industry-member)
        shift
        exec "$SCRIPT_DIR/sync_industry_member.sh" "$@"
        ;;
    help|--help|-h|"")
        show_usage
        exit 0
        ;;
    *)
        echo "Unknown command: $1"
        echo ""
        show_usage
        exit 1
        ;;
esac


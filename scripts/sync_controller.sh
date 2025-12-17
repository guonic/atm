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

  # Sync with custom batch size
  $0 stock-basic --batch-size 200

  # Use database for state storage
  $0 stock-basic --use-db-state

For detailed options, use:
  $0 stock-basic --help
  $0 trading-calendar --help
  $0 kline --help
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


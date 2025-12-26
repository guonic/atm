#!/bin/bash
#
# Industry Member Synchronization Script
#
# Synchronizes Shenwan industry member data from Tushare API
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Show usage
show_usage() {
    cat <<EOF
Industry Member Synchronization

Usage: $0 [options]

Options:
  --l1-code <code>      L1 industry code (e.g., 801050.SI)
  --l2-code <code>      L2 industry code (e.g., 801053.SI)
  --l3-code <code>      L3 industry code (e.g., 850531.SI)
  --is-new <Y|N>        Whether to sync only latest data (Y) or all data (N, default: Y)
  --batch-size <size>   Batch size for saving (default: 100)
  --config <path>       Configuration file path (default: config/data_ingestor.yaml)
  --state-dir <dir>     Directory for file-based state storage (default: storage/state)
  --use-db-state        Use database-based state storage instead of file-based
  --help                Show this help message

Examples:
  # Sync all stocks' industry members
  $0

  # Sync by L3 industry code
  $0 --l3-code 850531.SI

  # Sync by L2 industry code
  $0 --l2-code 801053.SI

  # Sync by L1 industry code
  $0 --l1-code 801050.SI

  # Sync all historical data (not just latest)
  $0 --is-new N

For detailed options, use:
  python python/tools/dataingestor/sync_industry_member.py --help
EOF
}

# Main command dispatch
if [ "$1" = "--help" ] || [ "$1" = "-h" ] || [ "$1" = "" ]; then
    show_usage
    exit 0
fi

# Execute sync
cd "$PROJECT_ROOT"
exec python python/tools/dataingestor/sync_industry_member.py "$@"


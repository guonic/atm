#!/bin/bash
#
# Industry Classification Synchronization Script
#
# Synchronizes Shenwan industry classification data from Tushare API
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Show usage
show_usage() {
    cat <<EOF
Industry Classification Synchronization

Usage: $0 [options]

Options:
  --src <version>        Industry classification source version (SW2014/SW2021, default: SW2021)
  --batch-size <size>    Batch size for saving (default: 100)
  --config <path>        Configuration file path (default: config/data_ingestor.yaml)
  --state-dir <dir>      Directory for file-based state storage (default: storage/state)
  --use-db-state         Use database-based state storage instead of file-based
  --help                 Show this help message

Examples:
  # Sync SW2021 industry classification
  $0 --src SW2021

  # Sync SW2014 industry classification
  $0 --src SW2014

  # Use database state storage
  $0 --src SW2021 --use-db-state

For detailed options, use:
  python python/tools/dataingestor/sync_industry_classify.py --help
EOF
}

# Main command dispatch
if [ "$1" = "--help" ] || [ "$1" = "-h" ] || [ "$1" = "" ]; then
    show_usage
    exit 0
fi

# Execute sync
cd "$PROJECT_ROOT"
exec python python/tools/dataingestor/sync_industry_classify.py "$@"


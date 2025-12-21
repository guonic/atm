#!/bin/bash
#
# K-line Data Synchronization Script
#
# Wrapper script for syncing K-line data for all stocks
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Activate virtual environment if it exists
if [ -f "$PROJECT_ROOT/.venv/bin/activate" ]; then
    source "$PROJECT_ROOT/.venv/bin/activate"
fi

# Run the Python script
exec python3 "$PROJECT_ROOT/python/tools/dataingestor/sync_kline.py" "$@"


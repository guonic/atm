#!/bin/bash
#
# Stock Basic Information Synchronization Script
#
# Synchronizes stock basic information from Tushare to database using upsert mode.
#

set -e

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Activate virtual environment if it exists
if [ -f "$PROJECT_ROOT/.venv/bin/activate" ]; then
    source "$PROJECT_ROOT/.venv/bin/activate"
fi

# Set Python path
export PYTHONPATH="$PROJECT_ROOT/python:$PYTHONPATH"

# Default values
EXCHANGE=""
LIST_STATUS="L"
BATCH_SIZE=100
RESUME=true
USE_DB_STATE=false
CONFIG_FILE="config/data_ingestor.yaml"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --exchange)
            EXCHANGE="$2"
            shift 2
            ;;
        --list-status)
            LIST_STATUS="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --no-resume)
            RESUME=false
            shift
            ;;
        --use-db-state)
            USE_DB_STATE=true
            shift
            ;;
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --tushare-token)
            export TUSHARE_TOKEN="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --exchange EXCHANGE      Exchange code (SSE/SZSE/BSE, empty for all)"
            echo "  --list-status STATUS     List status (L=listed, D=delisted, P=pause, empty=all)"
            echo "  --batch-size SIZE        Batch size for saving records (default: 100)"
            echo "  --no-resume              Do not resume from checkpoint"
            echo "  --use-db-state           Use database for state storage"
            echo "  --config FILE            Configuration file path"
            echo "  --tushare-token TOKEN    Tushare Pro API token"
            echo "  --help                   Show this help message"
            echo ""
            echo "Environment Variables:"
            echo "  TUSHARE_TOKEN            Tushare Pro API token"
            echo "  DB_HOST                  Database host (default: localhost)"
            echo "  DB_PORT                  Database port (default: 5432)"
            echo "  DB_USER                  Database user (default: quant)"
            echo "  DB_PASSWORD              Database password (default: quant123)"
            echo "  DB_NAME                  Database name (default: quant_db)"
            echo "  DB_SCHEMA                Database schema (default: quant)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Check Tushare token
if [ -z "$TUSHARE_TOKEN" ]; then
    echo "Error: TUSHARE_TOKEN environment variable is required"
    echo "Set it with: export TUSHARE_TOKEN=your_token"
    exit 1
fi

# Build command arguments
ARGS=(
    --config "$CONFIG_FILE"
    --exchange "$EXCHANGE"
    --list-status "$LIST_STATUS"
    --batch-size "$BATCH_SIZE"
)

if [ "$RESUME" = true ]; then
    ARGS+=(--resume)
else
    ARGS+=(--no-resume)
fi

if [ "$USE_DB_STATE" = true ]; then
    ARGS+=(--use-db-state)
fi

# Run synchronization
cd "$PROJECT_ROOT"
python3 python/tools/dataingestor/sync_stock_basic.py "${ARGS[@]}"


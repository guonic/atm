#!/bin/bash
#
# Qlib Export Controller
#
# Manages Qlib data export tasks
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Show usage
show_usage() {
    cat <<EOF
Qlib Export Controller

Usage: $0 <frequency> [options]

Frequencies:
  day        Daily K-line data
  week       Weekly K-line data
  month      Monthly K-line data
  quarter    Quarterly K-line data
  hour       Hourly K-line data
  30min      30-minute K-line data
  15min      15-minute K-line data
  5min       5-minute K-line data
  1min       1-minute K-line data

Options:
  --stocks <codes>          List of stock codes to export (e.g., 000001.SZ 000002.SZ)
  --start-date <date>       Start date for export (YYYY-MM-DD)
  --end-date <date>         End date for export (YYYY-MM-DD)
  --incremental             Use incremental export (default)
  --no-incremental          Disable incremental export (full export)
  --cleanup-csv             Cleanup CSV files after conversion (default)
  --no-cleanup-csv          Disable CSV cleanup
  --output-dir <dir>        Output directory for CSV files (default: ~/.qlib)
  --qlib-dir <dir>          Output directory for Qlib bin files (default: ~/.qlib)
  --config <path>           Path to config file

Examples:
  # Full export of all stocks for daily data
  $0 day

  # Export specific stocks
  $0 day --stocks 000001.SZ 000002.SZ

  # Incremental export (default)
  $0 day --incremental

  # Full export (not incremental)
  $0 day --no-incremental

  # Export with date range
  $0 day --start-date 2024-01-01 --end-date 2024-12-31

  # Export weekly data
  $0 week

  # Export hourly data for specific stocks
  $0 hour --stocks 000001.SZ 000002.SZ

For detailed options, use:
  python python/tools/qlib/export_qlib.py --help
EOF
}

# Main command dispatch
if [ $# -eq 0 ]; then
    show_usage
    exit 0
fi

FREQ=$1
shift

# Validate frequency
VALID_FREQS="day week month quarter hour 30min 15min 5min 1min"
if ! echo "$VALID_FREQS" | grep -q "\b$FREQ\b"; then
    echo "Error: Invalid frequency: $FREQ"
    echo ""
    show_usage
    exit 1
fi

# Execute export
cd "$PROJECT_ROOT"
exec python python/tools/qlib/export_qlib.py --freq "$FREQ" "$@"


#!/usr/bin/env python3
"""
Entry point for stock_ingestor_service module.

This allows running the module with: python -m tools.dataingestor.service
"""

import sys
from pathlib import Path

# Add project root to path
_project_root = Path(__file__).parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

if __name__ == "__main__":
    print("This is a service module, not a standalone script.")
    print("Please use one of the following:")
    print("  1. python python/tools/dataingestor/sync_stock_basic.py --help")
    print("  2. ./scripts/controller.sh sync stock-basic --help")
    print("  3. Import and use in your code:")
    print("     from tools.dataingestor import StockIngestorService")
    sys.exit(1)

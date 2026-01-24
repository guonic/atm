#!/usr/bin/env python3
"""
Market scanner for Teapot pattern recognition.

Main script to scan market and generate signals.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from nq.config import DatabaseConfig, load_config
from nq.trading.selector.teapot import TeapotSelector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Scan market for Teapot pattern signals"
    )
    parser.add_argument(
        "--start-date",
        type=str,
        required=True,
        help="Start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        required=True,
        help="End date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Config file path",
    )
    parser.add_argument(
        "--teapot-config",
        type=str,
        help="Teapot config file path (optional)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/teapot/signals/signals.csv",
        help="Output CSV file path",
    )
    parser.add_argument(
        "--symbols",
        type=str,
        help="Comma-separated list of stock codes (optional)",
    )
    parser.add_argument(
        "--use-cache",
        action="store_true",
        help="Use Parquet cache",
    )
    parser.add_argument(
        "--strict-cache",
        action="store_true",
        help="Strict cache mode (for backtesting)",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="storage/teapot_cache",
        help="Cache directory",
    )
    parser.add_argument(
        "--schema",
        type=str,
        default="quant",
        help="Database schema",
    )

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    db_config = config.database

    # Load Teapot config if provided
    teapot_config = {}
    if args.teapot_config:
        import yaml

        with open(args.teapot_config, "r", encoding="utf-8") as f:
            teapot_config = yaml.safe_load(f)

    # Add cache settings
    teapot_config["use_cache"] = args.use_cache
    teapot_config["strict_cache"] = args.strict_cache
    teapot_config["cache_dir"] = Path(args.cache_dir)

    # Parse symbols
    symbols = None
    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(",")]

    # Initialize selector
    selector = TeapotSelector(
        db_config=db_config,
        schema=args.schema,
        config=teapot_config,
    )

    # Scan market
    logger.info(f"Scanning market: {args.start_date} to {args.end_date}")
    signals = selector.scan_market(
        start_date=args.start_date,
        end_date=args.end_date,
        symbols=symbols,
    )

    if signals.is_empty():
        logger.warning("No signals detected")
        sys.exit(0)

    # Save signals
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    signals.write_csv(output_path)

    logger.info(f"Signals saved to {output_path}")

    # Generate summary
    summary = {
        "scan_period": {
            "start_date": args.start_date,
            "end_date": args.end_date,
        },
        "total_signals": len(signals),
        "signals_by_stock": (
            signals.group_by("ts_code")
            .agg(pl.count().alias("count"))
            .sort("count", descending=True)
            .head(10)
            .to_dicts()
        ),
    }

    summary_path = output_path.parent / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info(f"Summary saved to {summary_path}")
    print(f"\nScan complete: {len(signals)} signals detected")
    print(f"Signals saved to: {output_path}")


if __name__ == "__main__":
    main()

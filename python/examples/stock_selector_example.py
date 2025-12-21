# -*- coding: utf-8 -*-
"""
stock_selector_example.py

Description:
    Example demonstrating how to use stock selectors to filter stocks based on
    technical indicators, fundamental analysis, or composite criteria.

Usage:
    python stock_selector_example.py --type technical --exchange SSE
    python stock_selector_example.py --type fundamental --min-pe 5 --max-pe 20
    python stock_selector_example.py --type composite --operation AND

Arguments:
    --type          Selector type (technical/fundamental/composite)
    --exchange      Exchange code (SSE/SZSE/BSE)
    --min-price     Minimum price (for technical selector)
    --max-price     Maximum price (for technical selector)
    --min-volume    Minimum daily volume (for technical selector)
    --min-pe        Minimum PE ratio (for fundamental selector)
    --max-pe        Maximum PE ratio (for fundamental selector)
    --min-pb        Minimum PB ratio (for fundamental selector)
    --max-pb        Maximum PB ratio (for fundamental selector)
    --operation     Logical operation for composite selector (AND/OR/NOT)
    --limit         Maximum number of stocks to return (default: 100)

Output:
    - Prints selected stock codes and selection criteria
    - Optionally saves results to CSV file

Example:
    python stock_selector_example.py --type technical --exchange SSE --min-price 10 --max-price 100
    python stock_selector_example.py --type fundamental --min-pe 5 --max-pe 20 --min-pb 1 --max-pb 5
"""

import argparse
import logging
import os
from datetime import datetime
from typing import List

import pandas as pd

from atm.config import load_config
from atm.trading.selector import (
    CompositeSelector,
    FundamentalSelector,
    TechnicalSelector,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Stock selector example"
    )
    parser.add_argument(
        "--type",
        type=str,
        required=True,
        choices=["technical", "fundamental", "composite"],
        help="Selector type"
    )
    parser.add_argument(
        "--exchange",
        type=str,
        default=None,
        choices=["SSE", "SZSE", "BSE"],
        help="Exchange code (SSE/SZSE/BSE)"
    )
    parser.add_argument(
        "--market",
        type=str,
        default=None,
        help="Market type (沪A/深A/北交所)"
    )

    # Technical selector arguments
    parser.add_argument(
        "--min-price",
        type=float,
        default=None,
        help="Minimum price (for technical selector)"
    )
    parser.add_argument(
        "--max-price",
        type=float,
        default=None,
        help="Maximum price (for technical selector)"
    )
    parser.add_argument(
        "--min-volume",
        type=int,
        default=None,
        help="Minimum daily volume (for technical selector)"
    )
    parser.add_argument(
        "--min-price-change-pct",
        type=float,
        default=None,
        help="Minimum price change percentage (for technical selector)"
    )
    parser.add_argument(
        "--max-price-change-pct",
        type=float,
        default=None,
        help="Maximum price change percentage (for technical selector)"
    )

    # Fundamental selector arguments
    parser.add_argument(
        "--min-market-cap",
        type=float,
        default=None,
        help="Minimum market capitalization (in 10K CNY, for fundamental selector)"
    )
    parser.add_argument(
        "--max-market-cap",
        type=float,
        default=None,
        help="Maximum market capitalization (in 10K CNY, for fundamental selector)"
    )
    parser.add_argument(
        "--min-pe",
        type=float,
        default=None,
        help="Minimum PE ratio (for fundamental selector)"
    )
    parser.add_argument(
        "--max-pe",
        type=float,
        default=None,
        help="Maximum PE ratio (for fundamental selector)"
    )
    parser.add_argument(
        "--min-pb",
        type=float,
        default=None,
        help="Minimum PB ratio (for fundamental selector)"
    )
    parser.add_argument(
        "--max-pb",
        type=float,
        default=None,
        help="Maximum PB ratio (for fundamental selector)"
    )

    # Composite selector arguments
    parser.add_argument(
        "--operation",
        type=str,
        default="AND",
        choices=["AND", "OR", "NOT"],
        help="Logical operation for composite selector (default: AND)"
    )

    # Output arguments
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Maximum number of stocks to return (default: 100)"
    )
    parser.add_argument(
        "--save-csv",
        action="store_true",
        help="Save results to CSV file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs",
        help="Directory to save output (default: ./outputs)"
    )

    args = parser.parse_args()

    # Load configuration
    try:
        config = load_config()
        db_config = config.database
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        return

    # Create selector based on type
    if args.type == "technical":
        selector = TechnicalSelector(
            db_config=db_config,
            min_price=args.min_price,
            max_price=args.max_price,
            min_volume=args.min_volume,
            min_price_change_pct=args.min_price_change_pct,
            max_price_change_pct=args.max_price_change_pct,
        )
    elif args.type == "fundamental":
        selector = FundamentalSelector(
            db_config=db_config,
            min_market_cap=args.min_market_cap,
            max_market_cap=args.max_market_cap,
            min_pe=args.min_pe,
            max_pe=args.max_pe,
            min_pb=args.min_pb,
            max_pb=args.max_pb,
        )
    elif args.type == "composite":
        # Create multiple selectors for composite
        selectors = []
        if args.min_price or args.max_price or args.min_volume:
            selectors.append(
                TechnicalSelector(
                    db_config=db_config,
                    min_price=args.min_price,
                    max_price=args.max_price,
                    min_volume=args.min_volume,
                )
            )
        if args.min_pe or args.max_pe or args.min_pb or args.max_pb:
            selectors.append(
                FundamentalSelector(
                    db_config=db_config,
                    min_pe=args.min_pe,
                    max_pe=args.max_pe,
                    min_pb=args.min_pb,
                    max_pb=args.max_pb,
                )
            )

        if not selectors:
            logger.error("Composite selector requires at least one sub-selector")
            return

        selector = CompositeSelector(
            selectors=selectors,
            operation=args.operation,
            db_config=db_config,
        )

    # Run selection
    logger.info(f"Running {args.type} selector...")
    try:
        result = selector.select(
            exchange=args.exchange,
            market=args.market,
        )

        # Limit results
        selected_stocks = result.selected_stocks[:args.limit]

        # Print results
        logger.info("=" * 80)
        logger.info(f"Selection Results ({args.type} selector)")
        logger.info("=" * 80)
        logger.info(f"Total candidates: {result.total_candidates}")
        logger.info(f"Selected stocks: {len(selected_stocks)}")
        logger.info(f"Selection criteria: {result.selection_criteria}")
        logger.info("-" * 80)
        logger.info("Selected stock codes:")
        for i, ts_code in enumerate(selected_stocks, 1):
            logger.info(f"  {i}. {ts_code}")
        logger.info("=" * 80)

        # Save to CSV if requested
        if args.save_csv:
            os.makedirs(args.output_dir, exist_ok=True)
            df = pd.DataFrame({
                "ts_code": selected_stocks,
                "selection_date": result.selection_date,
            })
            output_file = os.path.join(
                args.output_dir,
                f"selected_stocks_{args.type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            )
            df.to_csv(output_file, index=False)
            logger.info(f"Results saved to: {output_file}")

    except Exception as e:
        logger.error(f"Selection failed: {e}", exc_info=True)
        return


if __name__ == "__main__":
    main()


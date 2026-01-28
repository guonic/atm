#!/usr/bin/env python3
"""
Cache builder for Teapot pattern recognition.

Offline tool to generate Parquet cache from PostgreSQL.
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import polars as pl

from nq.config import DatabaseConfig, load_config
from nq.repo.kline_repo import StockKlineDayRepo
from nq.repo.stock_repo import StockBasicRepo

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class CacheBuilder:
    """Cache builder for Teapot pattern recognition."""

    def __init__(
        self,
        db_config: DatabaseConfig,
        cache_dir: Path,
        schema: str = "quant",
    ):
        """
        Initialize cache builder.

        Args:
            db_config: Database configuration.
            cache_dir: Cache directory path.
            schema: Database schema name.
        """
        self.db_config = db_config
        self.cache_dir = Path(cache_dir)
        self.schema = schema
        self.daily_dir = self.cache_dir / "daily"
        self.metadata_file = self.cache_dir / "metadata.json"

        # Create directories
        self.daily_dir.mkdir(parents=True, exist_ok=True)

        self.kline_repo = StockKlineDayRepo(db_config, schema)
        self.stock_repo = StockBasicRepo(db_config, schema)

    def build_cache(
        self,
        start_date: str,
        end_date: str,
        symbols: Optional[List[str]] = None,
        overwrite: bool = False,
    ) -> dict:
        """
        Build cache.

        Args:
            start_date: Start date (YYYY-MM-DD).
            end_date: End date (YYYY-MM-DD).
            symbols: Optional list of stock codes (None for all stocks).
            overwrite: Whether to overwrite existing cache.

        Returns:
            Cache metadata dictionary.
        """
        logger.info(f"Building cache: {start_date} to {end_date}")

        if overwrite:
            self._clear_existing_cache()

        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")

        # Determine years to export
        start_year = start_dt.year
        end_year = end_dt.year

        total_records = 0
        total_stocks = 0
        files = {}

        # Export by year
        for year in range(start_year, end_year + 1):
            year_start = max(
                start_dt, datetime(year, 1, 1)
            ).strftime("%Y-%m-%d")
            year_end = min(
                end_dt, datetime(year, 12, 31)
            ).strftime("%Y-%m-%d")

            logger.info(f"Exporting year {year}: {year_start} to {year_end}")
            year_data = self._export_year_data(year, year_start, year_end, symbols)

            if not year_data.is_empty():
                parquet_file = self.daily_dir / f"{year}.parquet"
                year_data.write_parquet(parquet_file)
                files[str(year)] = f"daily/{year}.parquet"

                total_records += len(year_data)
                if total_stocks == 0:
                    total_stocks = len(year_data["ts_code"].unique())

                logger.info(
                    f"Exported {len(year_data)} records for year {year}"
                )

        # Create metadata
        metadata = {
            "cache_version": "1.0",
            "created_at": datetime.now().isoformat(),
            "start_date": start_date,
            "end_date": end_date,
            "total_stocks": total_stocks,
            "total_records": total_records,
            "years": list(range(start_year, end_year + 1)),
            "files": files,
        }

        # Save metadata
        with open(self.metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        logger.info(
            f"Cache built successfully: {total_records} records, {total_stocks} stocks"
        )

        return metadata

    def _export_year_data(
        self,
        year: int,
        start_date: str,
        end_date: str,
        symbols: Optional[List[str]] = None,
    ) -> pl.DataFrame:
        """Export data for a specific year."""
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")

        all_data = []

        if symbols:
            stock_list = symbols
        else:
            # Get all stocks from stock_basic table
            logger.warning(
                "Exporting all stocks - this may take a long time. Consider specifying symbols."
            )
            try:
                stocks = self.stock_repo.get_all()
                stock_list = [stock.ts_code for stock in stocks]
                logger.info(f"Found {len(stock_list)} stocks to export")
            except Exception as e:
                logger.error(f"Failed to get stock list: {e}")
                return pl.DataFrame()

        # Process stocks in batches for better performance
        batch_size = 100
        total_stocks = len(stock_list)
        
        for batch_idx in range(0, total_stocks, batch_size):
            batch = stock_list[batch_idx:batch_idx + batch_size]
            logger.info(
                f"Processing batch {batch_idx // batch_size + 1}/"
                f"{(total_stocks + batch_size - 1) // batch_size}: "
                f"{len(batch)} stocks"
            )
            
            for ts_code in batch:
                try:
                    klines = self.kline_repo.get_by_ts_code(
                        ts_code=ts_code,
                        start_time=start_dt,
                        end_time=end_dt,
                    )
                    for kline in klines:
                        all_data.append(
                            {
                                "ts_code": kline.ts_code,
                                "trade_date": kline.trade_date.strftime("%Y-%m-%d")
                                if isinstance(kline.trade_date, datetime)
                                else str(kline.trade_date),
                                "open": float(kline.open) if kline.open else None,
                                "high": float(kline.high) if kline.high else None,
                                "low": float(kline.low) if kline.low else None,
                                "close": float(kline.close) if kline.close else None,
                                "volume": int(kline.volume) if kline.volume else 0,
                                "amount": float(kline.amount) if kline.amount else None,
                            }
                        )
                except Exception as e:
                    logger.warning(f"Failed to export {ts_code}: {e}")

        if not all_data:
            return pl.DataFrame()

        df = pl.DataFrame(all_data)
        return df.sort(["ts_code", "trade_date"])

    def _clear_existing_cache(self) -> None:
        """Clear existing cache files."""
        if self.daily_dir.exists():
            for file in self.daily_dir.glob("*.parquet"):
                file.unlink()
                logger.info(f"Deleted {file}")

        if self.metadata_file.exists():
            self.metadata_file.unlink()
            logger.info(f"Deleted {self.metadata_file}")

    def validate_cache(
        self,
        start_date: str,
        end_date: str,
        symbols: Optional[List[str]] = None,
    ) -> dict:
        """
        Validate cache integrity.

        Returns:
            Validation result dictionary.
        """
        logger.info(f"Validating cache: {start_date} to {end_date}")

        # Load metadata
        if not self.metadata_file.exists():
            return {"valid": False, "error": "Metadata file not found"}

        with open(self.metadata_file, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        # Check date range
        cache_start = metadata["start_date"]
        cache_end = metadata["end_date"]

        if start_date < cache_start or end_date > cache_end:
            return {
                "valid": False,
                "error": f"Cache range ({cache_start} to {cache_end}) does not cover requested range ({start_date} to {end_date})",
            }

        # Check files exist
        for year_file in metadata["files"].values():
            file_path = self.cache_dir / year_file
            if not file_path.exists():
                return {
                    "valid": False,
                    "error": f"Cache file not found: {file_path}",
                }

        return {"valid": True, "metadata": metadata}


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Build Parquet cache for Teapot pattern recognition"
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
        "--cache-dir",
        type=str,
        default="storage/teapot_cache",
        help="Cache directory",
    )
    parser.add_argument(
        "--symbols",
        type=str,
        help="Comma-separated list of stock codes (e.g., 000001.SZ,000002.SZ)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing cache",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate cache instead of building",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Config file path",
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

    # Initialize builder
    builder = CacheBuilder(
        db_config=db_config,
        cache_dir=Path(args.cache_dir),
        schema=args.schema,
    )

    if args.validate:
        # Validate cache
        symbols = args.symbols.split(",") if args.symbols else None
        result = builder.validate_cache(
            start_date=args.start_date,
            end_date=args.end_date,
            symbols=symbols,
        )
        if result["valid"]:
            logger.info("Cache validation passed")
            print(json.dumps(result["metadata"], indent=2))
        else:
            logger.error(f"Cache validation failed: {result['error']}")
            sys.exit(1)
    else:
        # Build cache
        symbols = args.symbols.split(",") if args.symbols else None
        metadata = builder.build_cache(
            start_date=args.start_date,
            end_date=args.end_date,
            symbols=symbols,
            overwrite=args.overwrite,
        )
        logger.info("Cache built successfully")
        print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

"""
Cache manager for Teapot pattern recognition.

Provides Parquet cache management with strict mode support for backtesting.
"""

import json
import logging
from datetime import date, datetime
from pathlib import Path
from typing import List, Optional

import polars as pl

from nq.data.processor.teapot.exceptions import CacheIncompleteError, CacheNotFoundError

logger = logging.getLogger(__name__)


class CacheMetadata:
    """Cache metadata model."""

    def __init__(
        self,
        cache_version: str,
        created_at: str,
        start_date: str,
        end_date: str,
        total_stocks: int,
        total_records: int,
        years: List[int],
        files: dict,
    ):
        self.cache_version = cache_version
        self.created_at = created_at
        self.start_date = start_date
        self.end_date = end_date
        self.total_stocks = total_stocks
        self.total_records = total_records
        self.years = years
        self.files = files

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "cache_version": self.cache_version,
            "created_at": self.created_at,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "total_stocks": self.total_stocks,
            "total_records": self.total_records,
            "years": self.years,
            "files": self.files,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CacheMetadata":
        """Create from dictionary."""
        return cls(
            cache_version=data.get("cache_version", "1.0"),
            created_at=data.get("created_at", ""),
            start_date=data.get("start_date", ""),
            end_date=data.get("end_date", ""),
            total_stocks=data.get("total_stocks", 0),
            total_records=data.get("total_records", 0),
            years=data.get("years", []),
            files=data.get("files", {}),
        )


class CacheManager:
    """
    Cache manager for Teapot pattern recognition.

    Provides Parquet cache loading with strict mode support for backtesting.
    """

    def __init__(
        self,
        cache_dir: str = "storage/teapot_cache",
        strict_mode: bool = False,
    ):
        """
        Initialize cache manager.

        Args:
            cache_dir: Cache directory path.
            strict_mode: Strict mode, if True, raises exceptions when cache is missing or incomplete.
        """
        self.cache_dir = Path(cache_dir)
        self.strict_mode = strict_mode
        self.metadata_file = self.cache_dir / "metadata.json"
        self.daily_dir = self.cache_dir / "daily"

    def get_cache_metadata(self) -> Optional[CacheMetadata]:
        """Get cache metadata."""
        if not self.metadata_file.exists():
            return None

        try:
            with open(self.metadata_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            return CacheMetadata.from_dict(data)
        except Exception as e:
            logger.warning(f"Failed to load cache metadata: {e}")
            return None

    def validate_cache_range(
        self,
        start_date: str,
        end_date: str,
    ) -> bool:
        """
        Validate cache date range.

        Args:
            start_date: Requested start date.
            end_date: Requested end date.

        Returns:
            True if cache covers the date range.

        Raises:
            CacheIncompleteError: If cache is incomplete (strict_mode=True).
        """
        metadata = self.get_cache_metadata()
        if metadata is None:
            if self.strict_mode:
                raise CacheNotFoundError(
                    f"Cache metadata not found in {self.cache_dir}"
                )
            return False

        cache_start = datetime.strptime(metadata.start_date, "%Y-%m-%d").date()
        cache_end = datetime.strptime(metadata.end_date, "%Y-%m-%d").date()
        req_start = datetime.strptime(start_date, "%Y-%m-%d").date()
        req_end = datetime.strptime(end_date, "%Y-%m-%d").date()

        if req_start < cache_start or req_end > cache_end:
            if self.strict_mode:
                raise CacheIncompleteError(
                    f"Cache does not cover date range: {start_date} to {end_date}. "
                    f"Cache range: {metadata.start_date} to {metadata.end_date}"
                )
            return False

        return True

    def load_from_cache(
        self,
        start_date: str,
        end_date: str,
        symbols: Optional[List[str]] = None,
    ) -> pl.DataFrame:
        """
        Load data from cache.

        Args:
            start_date: Start date (YYYY-MM-DD).
            end_date: End date (YYYY-MM-DD).
            symbols: Optional list of stock codes to filter.

        Returns:
            Polars DataFrame with columns: ts_code, trade_date, open, high, low, close, volume, amount.

        Raises:
            CacheNotFoundError: If cache not found (strict_mode=True).
            CacheIncompleteError: If cache incomplete (strict_mode=True).
        """
        # 1. Check cache directory exists
        if not self.cache_dir.exists():
            if self.strict_mode:
                raise CacheNotFoundError(
                    f"Cache directory not found: {self.cache_dir}"
                )
            logger.warning(f"Cache directory not found: {self.cache_dir}")
            return pl.DataFrame()

        # 2. Check cache metadata
        metadata = self.get_cache_metadata()
        if metadata is None:
            if self.strict_mode:
                raise CacheNotFoundError(
                    f"Cache metadata not found in {self.cache_dir}"
                )
            logger.warning(f"Cache metadata not found")
            return pl.DataFrame()

        # 3. Validate date range
        if not self.validate_cache_range(start_date, end_date):
            return pl.DataFrame()

        # 4. Load parquet files
        data = self._load_parquet_files(start_date, end_date)

        if data.is_empty():
            if self.strict_mode:
                raise CacheIncompleteError(
                    f"No data found in cache for date range: {start_date} to {end_date}"
                )
            return pl.DataFrame()

        # 5. Filter by symbols if provided
        if symbols:
            data = data.filter(pl.col("ts_code").is_in(symbols))
            missing_symbols = set(symbols) - set(data["ts_code"].unique())
            if missing_symbols and self.strict_mode:
                raise CacheIncompleteError(
                    f"Missing symbols in cache: {missing_symbols}"
                )

        logger.info(
            f"Loaded {len(data)} records from cache ({start_date} to {end_date})"
        )
        return data

    def _load_parquet_files(
        self,
        start_date: str,
        end_date: str,
    ) -> pl.DataFrame:
        """Load parquet files for date range."""
        start_dt = datetime.strptime(start_date, "%Y-%m-%d").date()
        end_dt = datetime.strptime(end_date, "%Y-%m-%d").date()

        # Determine which year files to load
        start_year = start_dt.year
        end_year = end_dt.year

        data_frames = []
        for year in range(start_year, end_year + 1):
            parquet_file = self.daily_dir / f"{year}.parquet"
            if not parquet_file.exists():
                if self.strict_mode:
                    raise CacheIncompleteError(
                        f"Cache file not found: {parquet_file}"
                    )
                logger.warning(f"Cache file not found: {parquet_file}")
                continue

            try:
                df = pl.read_parquet(parquet_file)
                # Filter by date range
                df = df.filter(
                    (pl.col("trade_date") >= start_date)
                    & (pl.col("trade_date") <= end_date)
                )
                if not df.is_empty():
                    data_frames.append(df)
            except Exception as e:
                logger.warning(f"Failed to load {parquet_file}: {e}")
                if self.strict_mode:
                    raise CacheIncompleteError(
                        f"Failed to load cache file {parquet_file}: {e}"
                    )

        if not data_frames:
            return pl.DataFrame()

        # Concatenate all dataframes
        result = pl.concat(data_frames)
        return result.sort(["ts_code", "trade_date"])

    def clear_cache(self) -> None:
        """Clear cache (delete all parquet files and metadata)."""
        if self.daily_dir.exists():
            for file in self.daily_dir.glob("*.parquet"):
                file.unlink()
            logger.info(f"Cleared cache files in {self.daily_dir}")

        if self.metadata_file.exists():
            self.metadata_file.unlink()
            logger.info(f"Cleared cache metadata: {self.metadata_file}")

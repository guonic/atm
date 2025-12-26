"""
Data ingestor service implementation.

Orchestrates data ingestion from sources and storage in repositories.
"""

import logging
from typing import Optional

from nq.config import DataIngestorTaskConfig, DatabaseConfig
from nq.repo.base import BaseRepo
from nq.repo.database_repo import DatabaseRepo
from nq.data.source.base import BaseSource
from nq.data.source.http_source import HttpSource, HttpSourceConfig

logger = logging.getLogger(__name__)


class DataIngestorService:
    """
    Service for ingesting data from sources and storing in repositories.

    This service coordinates between data sources and repositories,
    handling batching, error recovery, and logging.
    """

    def __init__(
        self,
        task_config: DataIngestorTaskConfig,
        db_config: DatabaseConfig,
        source: Optional[BaseSource] = None,
        repo: Optional[BaseRepo] = None,
    ):
        """
        Initialize data ingestor service.

        Args:
            task_config: Task configuration.
            db_config: Database configuration.
            source: Optional pre-configured source (if None, will be created from config).
            repo: Optional pre-configured repository (if None, will be created from config).
        """
        self.task_config = task_config
        self.db_config = db_config
        self._source = source
        self._repo = repo

    @property
    def source(self) -> BaseSource:
        """
        Get or create data source.

        Returns:
            Data source instance.
        """
        if self._source is None:
            self._source = self._create_source()
        return self._source

    @property
    def repo(self) -> BaseRepo:
        """
        Get or create data repository.

        Returns:
            Data repository instance.
        """
        if self._repo is None:
            self._repo = self._create_repo()
        return self._repo

    def _create_source(self) -> BaseSource:
        """
        Create data source from configuration.

        Returns:
            Data source instance.

        Raises:
            ValueError: If source type is not supported.
        """
        source_config = self.task_config.source

        if source_config.type == "http":
            http_config = HttpSourceConfig(
                url=source_config.url or "",
                method=source_config.params.get("method", "GET"),
                headers=source_config.headers,
                params=source_config.params.get("params", {}),
                timeout=source_config.timeout,
                retry_count=source_config.retry_count,
                retry_delay=source_config.retry_delay,
                data_path=source_config.params.get("data_path"),
                pagination=source_config.params.get("pagination"),
            )
            return HttpSource(http_config)
        else:
            raise ValueError(f"Unsupported source type: {source_config.type}")

    def _create_repo(self) -> BaseRepo:
        """
        Create data repository from configuration.

        Returns:
            Data repository instance.
        """
        return DatabaseRepo(
            config=self.db_config,
            table_name=self.task_config.table,
            schema=self.db_config.schema,
            on_conflict=self.task_config.on_conflict,
        )

    def ingest(self, **kwargs) -> dict:
        """
        Execute data ingestion task.

        Args:
            **kwargs: Additional parameters to pass to source (e.g., date range).

        Returns:
            Dictionary with collection statistics:
            - fetched: Number of records fetched
            - saved: Number of records saved
            - errors: Number of errors encountered
        """
        stats = {"fetched": 0, "saved": 0, "errors": 0}
        batch = []
        batch_size = self.task_config.batch_size

        logger.info(f"Starting data ingestion task: {self.task_config.name}")

        try:
            with self.source, self.repo:
                # Test source connection
                if not self.source.test_connection():
                    raise ConnectionError(f"Source connection test failed for {self.task_config.name}")

                # Fetch data from source
                for record in self.source.fetch(**kwargs):
                    stats["fetched"] += 1
                    batch.append(record)

                    # Save batch when it reaches batch_size
                    if len(batch) >= batch_size:
                        try:
                            saved_count = self.repo.save_batch(batch)
                            stats["saved"] += saved_count
                            logger.debug(f"Saved batch of {saved_count} records")
                        except Exception as e:
                            stats["errors"] += len(batch)
                            logger.error(f"Error saving batch: {e}")
                        finally:
                            batch = []

                # Save remaining records
                if batch:
                    try:
                        saved_count = self.repo.save_batch(batch)
                        stats["saved"] += saved_count
                        logger.debug(f"Saved final batch of {saved_count} records")
                    except Exception as e:
                        stats["errors"] += len(batch)
                        logger.error(f"Error saving final batch: {e}")

                logger.info(
                    f"Data ingestion task completed: {self.task_config.name} - "
                    f"Fetched: {stats['fetched']}, Saved: {stats['saved']}, Errors: {stats['errors']}"
                )

        except Exception as e:
            logger.error(f"Data ingestion task failed: {self.task_config.name} - {e}")
            raise

        return stats

    def close(self) -> None:
        """Close source and repository connections."""
        if self._source:
            self._source.close()
        if self._repo:
            self._repo.close()


class ConnectionError(Exception):
    """Exception raised when connection fails."""

    pass


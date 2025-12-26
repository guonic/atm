"""
Base data source interface.

Defines the contract for all data sources in the ATM project.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, Optional

from pydantic import BaseModel


class SourceConfig(BaseModel):
    """Base source configuration."""

    type: str
    params: Dict[str, Any] = {}


class BaseSource(ABC):
    """
    Abstract base class for data sources.

    All data sources must implement this interface to be used by services in the ATM project.
    """

    def __init__(self, config: SourceConfig):
        """
        Initialize the data source.

        Args:
            config: Source configuration.
        """
        self.config = config

    @abstractmethod
    def fetch(self, **kwargs) -> Iterator[Dict[str, Any]]:
        """
        Fetch data from the source.

        This method should yield data records one at a time or in batches.
        Each record should be a dictionary with field names as keys.

        Args:
            **kwargs: Additional parameters for fetching (e.g., date range, filters).

        Yields:
            Dictionary representing a single data record.

        Raises:
            SourceError: If data fetching fails.
        """
        pass

    @abstractmethod
    def test_connection(self) -> bool:
        """
        Test the connection to the data source.

        Returns:
            True if connection is successful, False otherwise.
        """
        pass

    def close(self) -> None:
        """
        Close the data source and release resources.

        This method can be overridden to clean up resources.
        """
        pass

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


class SourceError(Exception):
    """Base exception for source-related errors."""

    pass


class ConnectionError(SourceError):
    """Exception raised when connection to source fails."""

    pass


class DataFetchError(SourceError):
    """Exception raised when data fetching fails."""

    pass


"""
Base data repository interface.

Defines the contract for all data repositories in the ATM project.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class BaseRepo(ABC):
    """
    Abstract base class for data repositories.

    All repositories must implement this interface to be used by services in the ATM project.
    """

    @abstractmethod
    def save(self, data: Dict[str, Any]) -> bool:
        """
        Save a single data record.

        Args:
            data: Data record to save.

        Returns:
            True if save was successful, False otherwise.

        Raises:
            RepoError: If save operation fails.
        """
        pass

    @abstractmethod
    def save_batch(self, data: List[Dict[str, Any]]) -> int:
        """
        Save multiple data records in a batch.

        Args:
            data: List of data records to save.

        Returns:
            Number of records successfully saved.

        Raises:
            RepoError: If batch save operation fails.
        """
        pass

    @abstractmethod
    def exists(self, data: Dict[str, Any]) -> bool:
        """
        Check if a record already exists.

        Args:
            data: Data record to check.

        Returns:
            True if record exists, False otherwise.
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """
        Close the repository and release resources.

        This method should be called to clean up connections, etc.
        """
        pass

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


class RepoError(Exception):
    """Base exception for repository-related errors."""

    pass


class ConnectionError(RepoError):
    """Exception raised when repository connection fails."""

    pass


class SaveError(RepoError):
    """Exception raised when save operation fails."""

    pass


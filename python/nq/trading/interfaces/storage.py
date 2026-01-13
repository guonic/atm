"""
Storage backend interface.

Defines abstract interface for storage backends (Memory/Redis/SQL).
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any


class IStorageBackend(ABC):
    """Storage backend interface.
    
    Supports multiple storage backends:
    - MemoryStorage: For backtesting (fast, in-memory)
    - RedisStorage: For live trading / distributed systems
    - SQLStorage: For Eidos database integration
    """
    
    @abstractmethod
    def save(self, key: str, data: Dict[str, Any]) -> None:
        """
        Save data to storage.
        
        Args:
            key: Storage key.
            data: Data dictionary to save.
        """
        pass
    
    @abstractmethod
    def load(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Load data from storage.
        
        Args:
            key: Storage key.
        
        Returns:
            Data dictionary if found, None otherwise.
        """
        pass
    
    @abstractmethod
    def delete(self, key: str) -> None:
        """
        Delete data from storage.
        
        Args:
            key: Storage key.
        """
        pass
    
    @abstractmethod
    def exists(self, key: str) -> bool:
        """
        Check if key exists in storage.
        
        Args:
            key: Storage key.
        
        Returns:
            True if exists, False otherwise.
        """
        pass

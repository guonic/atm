"""
Memory storage backend.

Fast in-memory storage for backtesting.
"""

from typing import Dict, Optional, Any
import logging

from ..interfaces.storage import IStorageBackend

logger = logging.getLogger(__name__)


class MemoryStorage(IStorageBackend):
    """In-memory storage backend.
    
    Fast storage for backtesting. Data is stored in memory and lost after process ends.
    """
    
    def __init__(self):
        """Initialize memory storage."""
        self.data: Dict[str, Dict[str, Any]] = {}
    
    def save(self, key: str, data: Dict[str, Any]) -> None:
        """
        Save data to memory.
        
        Args:
            key: Storage key.
            data: Data dictionary to save.
        """
        self.data[key] = data.copy()
    
    def load(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Load data from memory.
        
        Args:
            key: Storage key.
        
        Returns:
            Data dictionary if found, None otherwise.
        """
        return self.data.get(key)
    
    def delete(self, key: str) -> None:
        """
        Delete data from memory.
        
        Args:
            key: Storage key.
        """
        if key in self.data:
            del self.data[key]
    
    def exists(self, key: str) -> bool:
        """
        Check if key exists in memory.
        
        Args:
            key: Storage key.
        
        Returns:
            True if exists, False otherwise.
        """
        return key in self.data
    
    def clear(self) -> None:
        """Clear all data."""
        self.data.clear()
    
    def get_all_keys(self, prefix: Optional[str] = None) -> list:
        """
        Get all keys (optionally filtered by prefix).
        
        Args:
            prefix: Optional prefix to filter keys.
        
        Returns:
            List of keys.
        """
        if prefix:
            return [k for k in self.data.keys() if k.startswith(prefix)]
        return list(self.data.keys())

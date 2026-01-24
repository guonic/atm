"""
Exceptions for Teapot cache management.
"""


class CacheNotFoundError(Exception):
    """Raised when cache directory or metadata is not found."""

    pass


class CacheIncompleteError(Exception):
    """Raised when cache data is incomplete (missing dates, symbols, etc.)."""

    pass

"""
Logic layer.

This module implements:
- RiskManager: Risk control and order validation
- PositionAllocator: Position sizing and allocation
"""

from .risk import RiskManager
from .allocation import PositionAllocator

__all__ = [
    "RiskManager",
    "PositionAllocator",
]

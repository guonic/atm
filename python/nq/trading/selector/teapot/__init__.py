"""
Teapot pattern recognition selector module.

Provides pattern recognition and signal generation for Teapot strategy.
"""

from nq.trading.selector.teapot.base import TeapotSelector
from nq.trading.selector.teapot.features import TeapotFeatures
from nq.trading.selector.teapot.filters import TeapotFilters
from nq.trading.selector.teapot.state_machine import TeapotStateMachine

__all__ = [
    "TeapotSelector",
    "TeapotFeatures",
    "TeapotStateMachine",
    "TeapotFilters",
]

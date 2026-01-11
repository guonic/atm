"""
Exit strategy analysis module.

This module provides tools for building and training exit models that predict
when to sell positions based on momentum exhaustion and position management features.
"""

from .exit_model import ExitModel
from .feature_builder import ExitFeatureBuilder

__all__ = ["ExitModel", "ExitFeatureBuilder"]

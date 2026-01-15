"""
Correlation analysis module for stock structure features.

Provides five core correlation calculation methods for building dynamic, directed,
multi-center graph topologies to quantify stock interactions, causality, and information flow.
"""

from .correlation import (
    ICorrelationCalculator,
    DynamicCrossSectionalCorrelation,
    CrossLaggedCorrelation,
    VolatilitySync,
    GrangerCausality,
    TransferEntropy,
    IndustryCorrelation,
)
from .graph_builder import EnhancedGraphDataBuilder

__all__ = [
    "ICorrelationCalculator",
    "DynamicCrossSectionalCorrelation",
    "CrossLaggedCorrelation",
    "VolatilitySync",
    "GrangerCausality",
    "TransferEntropy",
    "IndustryCorrelation",
    "EnhancedGraphDataBuilder",
]


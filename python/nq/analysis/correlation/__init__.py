"""
Correlation analysis module for stock structure features.

Provides five core correlation calculation methods for building dynamic, directed,
multi-center graph topologies to quantify stock interactions, causality, and information flow.

Also includes evaluation framework for testing correlation algorithm impact on StructureExpert model.
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
from .return_matrix_generator import ReturnMatrixGenerator
from .correlation_optimizer import CorrelationOptimizer
from .rank_ic import (
    calculate_rank_ic,
    calculate_rank_ic_series,
    calculate_ic_statistics,
)
from .sensitivity_analyzer import SensitivityAnalyzer
from .correlation_filter import CorrelationFilter

__all__ = [
    "ICorrelationCalculator",
    "DynamicCrossSectionalCorrelation",
    "CrossLaggedCorrelation",
    "VolatilitySync",
    "GrangerCausality",
    "TransferEntropy",
    "IndustryCorrelation",
    "EnhancedGraphDataBuilder",
    "ReturnMatrixGenerator",
    "CorrelationOptimizer",
    "calculate_rank_ic",
    "calculate_rank_ic_series",
    "calculate_ic_statistics",
    "SensitivityAnalyzer",
    "CorrelationFilter",
]

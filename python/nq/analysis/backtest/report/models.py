"""
Data models for backtest report system.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import pandas as pd


@dataclass
class BacktestData:
    """Backtest data container."""
    
    exp_id: str
    experiment: Dict[str, Any]  # bt_experiment table data
    ledger: pd.DataFrame  # bt_ledger table data (date, nav, cash, ...)
    trades: pd.DataFrame  # bt_trades table data
    model_outputs: pd.DataFrame = field(default_factory=pd.DataFrame)  # bt_model_outputs (optional)
    model_links: pd.DataFrame = field(default_factory=pd.DataFrame)  # bt_model_links (optional)
    embeddings: pd.DataFrame = field(default_factory=pd.DataFrame)  # bt_embeddings (optional)


@dataclass
class MetricResult:
    """Metric calculation result."""
    
    name: str  # Metric name
    category: str  # Metric category (portfolio, trading, turnover, risk, model)
    value: Optional[Union[int, float]] = None  # Metric value (can be int or float)
    unit: Optional[str] = None  # Unit (%, days, etc.)
    format: Optional[str] = None  # Format string (e.g., "{:.2f}%")
    description: Optional[str] = None  # Description
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional metadata
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        # Preserve integer values as integers, not floats
        value = self.value
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        
        return {
            "name": self.name,
            "category": self.category,
            "value": value,
            "unit": self.unit,
            "format": self.format,
            "description": self.description,
            "metadata": self.metadata,
        }


@dataclass
class ReportConfig:
    """Report configuration."""
    
    # Metric selection
    metric_categories: Optional[List[str]] = None  # None means all categories
    metric_names: Optional[List[str]] = None  # None means all metrics
    
    # Output format
    output_format: str = "json"  # json, console, html, markdown
    output_path: Optional[str] = None
    
    # Display options
    precision: int = 2
    show_details: bool = True
    show_trading_stats: bool = True
    show_turnover_stats: bool = True
    
    # Comparison
    compare_experiments: Optional[List[str]] = None  # Multiple exp_ids for comparison

